# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Chemin tok in-process : texte → split → B → E_tok → pooling fixe → têtes → CIR.

Références : ``plan-remodelage-tout-appris.md`` (T3 flux découplé, T6 tout
appris dans E_tok, R1 pooling figé + copies, R2 fenêtre k=2, R4 format,
R5 OOV, R7 pas de all_pairs), ``plan-remodelage-inference.md`` §[A]
(``analyze → split → forward_ids``).

Périmètre STRICT — inférence seule : aucun ``backward``, aucun entraînement
(câblage ultérieur avec ``refacto_loss`` + masques R8). Lecture seule du
pipeline legacy : ``forward()`` / ``vectorize_*`` intacts, zéro modification.

Séparation imposée par la garde du runtime : le runtime appris (``pipeline/``,
``training/``, ``engine.py``) n'importe JAMAIS le socle B — la segmentation
vit dans ``nlu/tok.py`` (zéro lexique, T6 : pas un frontend). Ce module
reçoit du PRÉ-SEGMENTÉ (spans + textes + connecteurs) et ne connaît que des
ids stables.

Honnêteté (doctrine Ph5 NO-GO → 6b) : les têtes (E_tok + MLP + R-GCN) sont
NON ENTRAÎNÉES par défaut (aléatoires seedées) — les CIR produits sont
STRUCTURELLEMENT valides (schéma serde Rust) mais NON CALIBRÉS. Aucun
chiffre issu de ce chemin ne doit être présenté comme mesure.

Différences assumées vs chemin legacy (documentées, pas silencieuses) :
- segmentation = socle B (clauses ``{id, texte, span}`` + ``source``), jamais
  de lattice subprocess ;
- pas de features UD : base d'arête = ``[src|dst|gap]`` explicite
  (2·d_model + 1), pas de ``vectorize_edge`` ;
- ``negated=False`` partout (aucun signal de négation sur ce chemin v0) ;
- ``scope="unknown"``, ``origin="explicit"`` (B n'infère ni l'un ni l'autre) ;
- paires dans la fenêtre k=2 (R7 — ``all_pairs`` interdit) ;
- R-GCN sans teacher forcing (types no-edge) : pas de gold à l'inférence,
  two-pass exclu (œuf-et-poule) ;
- pas de masque de classes (têtes non entraînées — R8 : aucun seuil avant
  N≥30/classe) ;
- marqueurs = positions B observées (span → token), jamais devinés.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..constants import (
    NODE_ORIGIN_VALUES,
    NODE_TYPES,
    RELATION_TYPES,
    THETA_AMBIGUITY_DEFAULT,
    rgcn_n_relations,
    rgcn_no_edge_idx,
)
from ..layer0.tokenizer import split as tok_split
from ..layer1.l1_forward import align_spans, forward_ids, pool_clauses
from ..layer1.tok_embedding import TokEmbedding
from ..layer2.reference import MLPEncoder
from ..layer3.reference import RGCNLayer
from .ir_emitter import emit

#: Budget langues B — MIROIR du budget du socle (duplication forcée : ce
#: module n'a pas le droit d'importer le socle — garde moteur-B du runtime).
#: Toute divergence = bug (détecté par les tests frontend qui comparent).
LANGS_BUDGET = ("fr", "en", "es", "de", "pt", "it")

#: Fenêtre de paires (R7 — all_pairs interdit) : 1 ≤ j-i ≤ WINDOW_K.
WINDOW_K = 2
#: Dimension de référence (défaut TokEmbedding + arch gcn-train ; §9 : figée).
D_MODEL_DEFAULT = 128
#: Seuil d'alerte OOV prod (R5 : raise en train/val, warn en prod).
OOV_WARN_RATE = 0.05


def _edge_base_dim(d_model: int) -> int:
    """Base d'arête tok : [src(d) | dst(d) | gap(1)] — explicite, fixe."""
    return 2 * d_model + 1


def _edge_full_dim(d_model: int, n_node_types: int = len(NODE_TYPES)) -> int:
    """Arête enrichie : [base | enriched_src | enriched_dst | p_src | p_dst]."""
    return _edge_base_dim(d_model) + d_model + d_model + 2 * n_node_types


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Softmax stable par ligne (copie — jamais d'alias sur les logits)."""
    x = np.asarray(logits, dtype=np.float32).copy()
    x -= x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class TokForwardEngine:
    """Moteur d'inférence tok (E_tok + MLP + R-GCN), indépendant du legacy.

    Dérivation de seeds documentée : E_tok ``seed``, MLP ``seed+1``,
    R-GCN ``seed+2`` (même convention que les dérivations fixes du train).
    ``edge_dropout=0.0`` partout (inférence — le dropout d'entraînement
    n'a pas de sens ici et casserait le déterminisme).
    """

    def __init__(
        self,
        d_model: int = D_MODEL_DEFAULT,
        seed: int = 42,
        temperature: float = 1.0,
        edge_threshold: float = 0.0,
        theta_ambiguity: float = THETA_AMBIGUITY_DEFAULT,
        bidirectional: bool = False,
    ) -> None:
        if d_model < 1:
            raise ValueError(f"d_model doit être ≥ 1 (reçu {d_model!r}).")
        if float(temperature) <= 0:
            raise ValueError(f"temperature doit être > 0 (reçue {temperature!r}).")
        if not (0.0 <= float(edge_threshold) < 1.0):
            raise ValueError(f"edge_threshold doit être dans [0, 1[ (reçu {edge_threshold!r}).")
        self.d_model = int(d_model)
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.edge_threshold = float(edge_threshold)
        self.theta_ambiguity = float(theta_ambiguity)
        self.bidirectional = bool(bidirectional)
        self.emb = TokEmbedding(d_model=self.d_model, seed=self.seed)
        self.encoder = MLPEncoder(
            d_clause=self.d_model,
            d_edge=_edge_full_dim(self.d_model),
            seed=self.seed + 1,
            edge_dropout=0.0,
        )
        self.rgcn = RGCNLayer(
            d_in=self.d_model,
            d_out=self.d_model,
            n_relations=rgcn_n_relations(len(RELATION_TYPES), self.bidirectional),
            seed=self.seed + 2,
            dropout=0.0,
        )
        self._no_edge = rgcn_no_edge_idx(len(RELATION_TYPES))
        self.last_oov_rate: float = 0.0

    # ── vocabulaire ──────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Lignes de E_tok (0 = moteur inutilisable — gate, voir analyze_tok)."""
        return self.emb.vocab_size

    def build_vocab(self, texts, lang: str = "fr") -> int:
        """Ajoute les ids stables des textes (ordre indifférent — triés).

        ``lang`` validé contre le budget B (miroir — voir LANGS_BUDGET) pour
        refuser tôt les langues hors support au lieu de produire des CIR
        vides silencieux. Le tokenizer lui-même est agnostique (T6).
        """
        if lang not in LANGS_BUDGET:
            raise ValueError(f"langue hors budget fr/en/es/de/pt/it : {lang!r}")
        for _t in texts:
            if not isinstance(_t, str):
                raise TypeError(f"texte doit être str, reçu {type(_t).__name__}")
            ids, _, _ = tok_split(_t)
            self.emb.build_vocab(ids)
        return self.emb.vocab_size

    # ── inférence ────────────────────────────────────────────────

    def _marker_for_pair(self, conn_span, offsets: list) -> int | None:
        """Span B [début, fin] → premier indice token (observé, jamais deviné)."""
        try:
            debut = float(conn_span[0])
        except (TypeError, ValueError, IndexError):
            return None
        for i, (a, _b) in enumerate(offsets):
            if a >= debut:
                return i
        return None

    def forward_from_spans(
        self,
        text: str,
        ids: list,
        offsets: list,
        clause_spans: list,
        clause_textes: list,
        connecteurs: list | None = None,
        source_b: str = "tok",
    ) -> dict:
        """Pré-segmenté → CIR dict (runtime pur — aucune entrée B/texte ici).

        ``clause_spans`` : spans caractères (socle B) ; ``clause_textes`` :
        libellés ; ``connecteurs`` : dicts B avec ``span`` (marqueurs) ;
        ``source_b`` : ``silver-B|fallback`` tracé en ``doc_ref``.
        La segmentation (socle + split) vit dans ``nlu/tok.py`` — ce module
        ne segmente jamais (garde du runtime).

        Gate capacitaire : vocabulaire vide → ``ValueError`` (prédire sur
        E vide/OOV-total est refusé — la vérité graphe ne se brade pas).
        OOV > 5 % → ``UserWarning`` (R5 : warn en prod), taux exposé sur
        ``last_oov_rate`` (aucun slot CIR — assumé, pas silencieux).
        """
        if not isinstance(text, str):
            raise TypeError(f"text doit être str, reçu {type(text).__name__}")
        if self.emb.vocab_size == 0:
            raise ValueError(
                "E_tok vide : appelez build_vocab(textes) ou chargez un "
                "checkpoint s1+ avant forward_from_spans — l'inférence sans "
                "vocabulaire est refusée (CIR non fondé)."
            )
        clauses = list(zip(clause_spans, clause_textes, strict=False))
        if not clauses:
            return emit(text, [], [], [], [], [])
        self.emb.reset_oov()
        matrice = forward_ids(self.emb, ids)
        self.last_oov_rate = self.emb.oov_rate
        if self.last_oov_rate > OOV_WARN_RATE:
            warnings.warn(
                f"forward_from_spans : taux OOV {self.last_oov_rate:.2%} > 5 % "
                f"(source B : {source_b}) — vecteurs zéro, CIR dégradé.",
                UserWarning,
                stacklevel=2,
            )
        spans = [list(span) for span, _texte in clauses]
        alignes = align_spans(offsets, spans)
        pooled, _miroir = pool_clauses(matrice, alignes)

        # Têtes nœuds (batch si dispo — même sémantique que la boucle).
        if hasattr(self.encoder, "forward_batch"):
            node_logits = self.encoder.forward_batch(pooled)
        else:
            node_logits = np.stack([self.encoder.forward_node(v) for v in pooled])
        node_probs = _softmax_rows(node_logits)
        node_types = [NODE_TYPES[int(i)] for i in np.argmax(node_logits, axis=1)]
        node_labels = [
            (texte or "").strip() or f"clause_{k}" for k, (_span, texte) in enumerate(clauses)
        ]
        token_spans = [(min(aligne), max(aligne)) if aligne else (0, 0) for aligne in alignes]
        scopes = ["unknown"] * len(clauses)
        origins = [NODE_ORIGIN_VALUES[0]] * len(clauses)

        # R-GCN (sauté à 1 clause — miroir du legacy) + paires fenêtre k=2.
        n = len(clauses)
        if n > 1:
            paires = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 1 + WINDOW_K))]
            index = np.array([[p[0] for p in paires], [p[1] for p in paires]], dtype=np.int64)
            types = np.full(len(paires), self._no_edge, dtype=np.int64)
            enriched = self.rgcn.message_pass(pooled.astype(np.float32), index, types)
        else:
            paires = []
            enriched = pooled

        # Connecteurs B observés (span → token) pour les marqueurs.
        connecteurs = connecteurs or []
        triples = []
        for i, j in paires:
            gap = float(j - i)
            base = np.concatenate(
                [
                    pooled[i],
                    pooled[j],
                    np.array([gap], dtype=np.float32),
                ]
            )
            vec = np.concatenate([base, enriched[i], enriched[j], node_probs[i], node_probs[j]])
            logit = self.encoder.forward_edge(vec)
            probs = _softmax_rows((logit / self.temperature).reshape(1, -1))[0]
            rel_idx = int(np.argmax(probs))
            conf = float(probs[rel_idx])
            conf = 0.0 if not np.isfinite(conf) else min(1.0, max(0.0, conf))
            if conf < self.theta_ambiguity and conf >= self.edge_threshold:
                top2 = np.argsort(probs)[-2:][::-1]
                candidates = [(RELATION_TYPES[int(k)], float(probs[k])) for k in top2]
                ambiguous = True
            else:
                candidates = None
                ambiguous = False
            if conf < self.edge_threshold:
                continue
            marker = None
            for conn in connecteurs:
                cspan = conn.get("span") if isinstance(conn, dict) else None
                if cspan is None:
                    continue
                try:
                    ok = (
                        float(spans[i][1]) <= float(cspan[0])
                        and float(cspan[1]) <= float(spans[j][0]) + 1e-9
                    )
                except (TypeError, ValueError, IndexError):
                    continue
                if ok:
                    marker = self._marker_for_pair(cspan, offsets)
                    if marker is not None:
                        break
            triples.append(
                (
                    i,
                    j,
                    RELATION_TYPES[rel_idx],
                    conf,
                    False,
                    marker,
                    None,
                    ambiguous,
                    candidates,
                )
            )
        return emit(
            text,
            node_types,
            node_labels,
            token_spans,
            scopes,
            triples,
            node_origins=origins,
            doc_ref=f"tok:{source_b}",
        )
