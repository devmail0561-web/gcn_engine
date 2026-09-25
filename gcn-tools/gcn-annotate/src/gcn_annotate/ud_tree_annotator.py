# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Annotateur CIR sans LLM — basé sur l'arbre de dépendances UD (spaCy).

Stratégie :
  1. Parser la phrase avec spaCy
  2. Chercher les sous-arbres advcl/mark ou cc/conj qui indiquent une relation causale
  3. Identifier source et cible depuis la structure syntaxique (pas de regex surface)
  4. Extraire les spans depuis les sous-arbres (précis, 1-based)
  5. Classifier les types de nœuds depuis le POS+lemme du ROOT de chaque clause
  6. Calibrer la confiance selon la force du connecteur

Respecte le protocole LLMAnnotator — interchangeable avec AnthropicAnnotator.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Connecteur → (relation, direction)
# direction "fwd" : advcl est la source, main est la cible
# direction "bwd" : advcl est la cible, main est la source
_CONNECTORS: list[tuple[str, str, str]] = [
    # cause (advcl=cause, main=effet)
    ("parce que",    "cause",      "fwd"),
    ("parce qu",     "cause",      "fwd"),
    ("car",          "cause",      "fwd"),
    ("puisque",      "cause",      "fwd"),
    ("du fait que",  "cause",      "fwd"),
    ("en raison de", "cause",      "fwd"),
    ("à cause de",   "cause",      "fwd"),
    ("grâce à",      "cause",      "fwd"),
    ("étant donné",  "cause",      "fwd"),
    ("because",      "cause",      "fwd"),
    ("since",        "cause",      "fwd"),
    ("due to",       "cause",      "fwd"),
    ("owing to",     "cause",      "fwd"),
    # conséquence (main=cause, advcl/result=effet)
    ("donc",         "cause",      "bwd"),
    ("ainsi",        "cause",      "bwd"),
    ("par conséquent","cause",     "bwd"),
    ("therefore",    "cause",      "bwd"),
    ("thus",         "cause",      "bwd"),
    ("consequently", "cause",      "bwd"),
    # enable
    ("permet",       "enable",     "fwd"),
    ("facilite",     "enable",     "fwd"),
    ("favorise",     "enable",     "fwd"),
    ("enables",      "enable",     "fwd"),
    ("allows",       "enable",     "fwd"),
    ("facilitates",  "enable",     "fwd"),
    # prevent
    ("empêche",      "prevent",    "fwd"),
    ("bloque",       "prevent",    "fwd"),
    ("interdit",     "prevent",    "fwd"),
    ("prevents",     "prevent",    "fwd"),
    ("inhibits",     "prevent",    "fwd"),
    # condition
    ("si",           "condition",  "fwd"),
    ("if",           "condition",  "fwd"),
    ("unless",       "condition",  "fwd"),
    ("provided that","condition",  "fwd"),
    # concession
    ("bien que",     "concession", "fwd"),
    ("même si",      "concession", "fwd"),
    ("malgré",       "concession", "fwd"),
    ("although",     "concession", "fwd"),
    ("even though",  "concession", "fwd"),
    ("despite",      "concession", "fwd"),
    # sequence
    ("puis",         "sequence",   "bwd"),
    ("ensuite",      "sequence",   "bwd"),
    ("then",         "sequence",   "bwd"),
    ("subsequently", "sequence",   "bwd"),
    # motivation
    ("afin de",      "motivation", "bwd"),
    ("pour que",     "motivation", "bwd"),
    ("in order to",  "motivation", "bwd"),
    ("so that",      "motivation", "bwd"),
    # opposition
    ("alors que",    "opposition", "fwd"),
    ("tandis que",   "opposition", "fwd"),
    ("whereas",      "opposition", "fwd"),
]

# Confiance selon type de connecteur
_CONFIDENCE: dict[str, float] = {
    "cause":      0.90,
    "enable":     0.88,
    "prevent":    0.88,
    "condition":  0.85,
    "concession": 0.85,
    "motivation": 0.82,
    "sequence":   0.80,
    "opposition": 0.80,
}

_STATE_VERBS = {"être", "avoir", "rester", "demeurer", "sembler", "paraître",
                "be", "remain", "stay", "seem", "appear"}
_TRANSITION_VERBS = {"devenir", "changer", "évoluer", "transformer",
                     "become", "change", "evolve", "transform", "shift"}
_ACTION_VERBS = {"faire", "créer", "construire", "lancer", "décider",
                 "make", "create", "build", "launch", "decide", "execute", "run"}
_SYSTEMIC_NOUNS = {"loi", "règle", "norme", "politique", "système", "protocole",
                   "law", "rule", "norm", "policy", "system", "protocol", "framework"}


def _classify_root(root) -> str:
    """Classifie un nœud ROOT spaCy en NodeType CIR."""
    if root is None:
        return "entite"
    lemma = root.lemma_.lower()
    has_verb = root.pos_ in {"VERB", "AUX"}
    if not has_verb:
        return "entite"
    subtree_lemmas = {t.lemma_.lower() for t in root.subtree}
    if subtree_lemmas & _SYSTEMIC_NOUNS:
        return "etat_systemique"
    if lemma in _STATE_VERBS:
        return "etat"
    if lemma in _TRANSITION_VERBS:
        return "transition"
    if lemma in _ACTION_VERBS:
        return "action"
    return "processus"


def _subtree_span(root) -> list[int]:
    """Span 1-based du sous-arbre syntaxique d'un token."""
    tokens = list(root.subtree)
    return [min(t.i for t in tokens) + 1, max(t.i for t in tokens) + 1]


def _match_connector(text_lower: str) -> tuple[str, str] | None:
    """Retourne (relation, direction) pour le premier connecteur trouvé."""
    for conn, rel, direction in _CONNECTORS:
        if conn in text_lower:
            return rel, direction
    return None


def _annotate_one(text: str, nlp) -> dict | None:
    """Annote une phrase via l'arbre UD spaCy. Retourne None si non causale."""
    text = text.strip()
    if len(text.split()) < 5:
        return None

    doc = nlp(text)
    T = len(doc)
    text_lower = text.lower()

    conn_match = _match_connector(text_lower)
    if conn_match is None:
        return None
    relation, direction = conn_match

    # Stratégie 1 : chercher une relation advcl ou relcl dans l'arbre
    advcl_root = None
    main_root = None
    for token in doc:
        if token.dep_ in {"advcl", "advcl:relcl", "relcl"} and token.head.dep_ == "ROOT":
            advcl_root = token
            main_root = token.head
            break

    if advcl_root is None:
        # Stratégie 2 : chercher un ROOT et un SCONJ/CC marquant la frontière
        root_tok = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root_tok is None:
            return None
        conn_idx = text_lower.find(
            next((c for c, *_ in _CONNECTORS if c in text_lower), "")
        )
        if conn_idx < 0:
            return None
        # Séparer avant/après le connecteur en termes de tokens
        tokens_before = [t for t in doc if t.idx < conn_idx]
        tokens_after  = [t for t in doc if t.idx >= conn_idx]
        if not tokens_before or not tokens_after:
            return None
        # Trouver le root de chaque moitié
        def _local_root(toks):
            for t in toks:
                if t.dep_ == "ROOT":
                    return t
            return max(toks, key=lambda t: len(list(t.subtree)))
        main_root  = _local_root(tokens_before)
        advcl_root = _local_root(tokens_after)

    # Direction : fwd = advcl→src, main→dst ; bwd = main→src, advcl→dst
    if direction == "fwd":
        src_root, dst_root = advcl_root, main_root
    else:
        src_root, dst_root = main_root, advcl_root

    src_span = _subtree_span(src_root)
    dst_span = _subtree_span(dst_root)

    # Garantir spans distincts et non-vides
    if src_span == dst_span:
        mid = max(1, T // 2)
        src_span = [1, mid]
        dst_span = [min(T, mid + 1), T]

    src_type = _classify_root(src_root)
    dst_type  = _classify_root(dst_root)
    confidence = _CONFIDENCE.get(relation, 0.80)

    return {
        "text": text,
        "causal_pattern": relation,
        "cir": {
            "nodes": [
                {
                    "id": "n001", "type": src_type,
                    "label": src_root.lemma_ if src_root else text.split()[0],
                    "token_span": src_span,
                    "origin": "explicit", "scope": "specific",
                    "temporal_index": 0, "modifiers": [], "attributes": {},
                },
                {
                    "id": "n002", "type": dst_type,
                    "label": dst_root.lemma_ if dst_root else text.split()[-1],
                    "token_span": dst_span,
                    "origin": "explicit", "scope": "specific",
                    "temporal_index": 1, "modifiers": [], "attributes": {},
                },
            ],
            "edges": [
                {
                    "source": "n001", "target": "n002", "relation": relation,
                    "attributes": {
                        "confidence": confidence,
                        "explicit": True,
                        "negated": False,
                    },
                }
            ],
        },
    }


class UDTreeAnnotator:
    """Annotateur CIR sans LLM basé sur le parse tree UD (spaCy).

    Respecte le protocole LLMAnnotator — passe directement dans le CLI
    gcn-annotate à la place d'AnthropicAnnotator ou OpenAIAnnotator.

    Qualité cible : ~0.85 de précision sur phrases causales explicites FR/EN.
    Confiance calibrée par type de connecteur (0.80–0.90).
    """

    def __init__(self, model_fr: str = "fr_core_news_sm",
                 model_en: str = "en_core_web_sm") -> None:
        self._models = {"fr": model_fr, "en": model_en}
        self._cache: dict = {}

    def _get_nlp(self, lang: str):
        if lang not in self._cache:
            import spacy
            model = self._models.get(lang, self._models["fr"])
            try:
                self._cache[lang] = spacy.load(model)
            except OSError:
                # Repli sur le modèle FR si le modèle demandé est absent
                self._cache[lang] = spacy.load(self._models["fr"])
        return self._cache[lang]

    def annotate(self, sentences: list[str], lang: str = "fr") -> list[dict]:
        """Annote une liste de phrases. Retourne une liste de dicts sentences CIR.

        Conforme au protocole LLMAnnotator :
          annotate(sentences: list[str], lang: str) -> list[dict]
        """
        if not sentences:
            return []
        nlp = self._get_nlp(lang)
        results = []
        for i, text in enumerate(sentences):
            ann = _annotate_one(text, nlp)
            if ann is not None:
                ann["id"] = f"s{i+1:04d}"
                results.append(ann)
        return results
