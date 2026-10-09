# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""DialogueSession — état de dialogue Phases 9b/9c/9d/9e.

Moteur language-agnostic : AUCUN lexique de langue ici (ni pronoms, ni
marqueurs, ni phrases). Mécanismes structurels uniquement :

- 9b sujets : pile des 3 derniers sujets + règle de continuité (aucun
  concept reconnu → on garde le sujet en cours) + chiffres Unicode pour les
  ordinaux + répétition du nom (match structurel). Le sujet compris est
  annoncé en code (`topic: <label>`) pour permettre la correction.
- 9c clarification : une seule en attente ; réponse par chiffre ou nom ;
  survit aux tours vagues, abandonnée après 3 tours ou sur bascule nette
  de sujet (match exact différent), toujours avec annonce en code
  (`clarify:` / `abandoned:`).
- 9d langue : tag opaque (`lang: <code>`, consigne explicite persistante)
  + tag de script Unicode par message (latin, arabic, ...) — jamais de nom
  de langue détecté par lexique. Priorité : consigne > message > fil > "und".
  Les réponses restent structurelles (labels + codes) : le tag est exposé,
  pas traduit.
- 9e journal + correction : un tour = {n, question, sujet, résumé 1 ligne,
  statut, langue, ajouts}. Correction par commande `revoke: [tour]` —
  ne retire que les apports utilisateur (jamais le corpus), annoncée en code
  (`revoked:`). Résumé automatique des vieux échanges : Phase 10 (M2).

Réponses en codes : `topic:`, `clarify:`, `abandoned:`, `revoked:`, `lang:`,
`origin:` — même modèle que le DSL Pearl (mots-clés fixes, contenu libre).
"""
from __future__ import annotations

from collections import Counter

from .verbalizer.instructions import (
    _USER_ORIGINS,
    CausalGraph,
    InstructionHandler,
    parse_command,
)
from .verbalizer.query_report import QueryVerbalizer

# Fenêtre d'historique en mémoire (résumé automatique au-delà — jamais drop,
# jamais de refus ; promis par 9e, livré ici avec M2a).
JOURNAL_KEEP_TURNS = 200

# Plafonds de session M2a (configurables par constructeur) : au dépassement,
# rangement automatique ANNONCÉ (fusion doublons, résumé vieux échanges),
# jamais de refus d'entrée.
SESSION_MAX_NODES = 500
SESSION_MAX_EDGES = 2000

# Marques d'interrogation (ponctuation, pas du lexique).
_QUESTION_MARKS = ("?", "؟")

# M2c — paraphrases : coefficient d'overlap sur trigrammes (le petit contenu
# dans le grand : "fatigues" ⊂ "fatigue"). Jaccard punirait les différences
# de longueur, or une paraphrase change la longueur. Seuil FIGÉ à 0.75 :
# variantes morphologiques (0.83+) passent, inclusions courtes ("eau" ⊂
# "bateau" = 0.67 via "au#") non.
PARAPHRASE_THRESHOLD = 0.75
PARAPHRASE_MIN_LEN = 4


def _trigrams(s: str) -> set[str]:
    """Trigrammes bordés du normalisé (accents pliés, cf. CausalGraph)."""
    t = f"#{CausalGraph._normalize(s)}#"
    if len(t) < PARAPHRASE_MIN_LEN + 1:
        return set()
    return {t[i:i + 3] for i in range(len(t) - 2)}


def paraphrase_overlap(a: str, b: str) -> float:
    """Overlap |A∩B|/min(|A|,|B|) ∈ [0,1] ; 0 si trop court (bruit)."""
    ga, gb = _trigrams(a), _trigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))

# Ponctuation dépouillée pour les mots de contenu (mirroir nlu_routing + ¿؟).
_STRIP = "?.,!()«»:;\"'¿؟"


def content_words(text: str) -> list[str]:
    """Mots de contenu : longueur > 2 après strip ponctuation. Aucun lexique."""
    return [
        w.strip(_STRIP).lower()
        for w in text.split()
        if len(w.strip(_STRIP)) > 2
    ]


def is_question_surface(text: str) -> bool:
    """Heuristique P2 : marque d'interrogation finale (ponctuation universelle)."""
    return text.strip().endswith(_QUESTION_MARKS)


def script_tag(text: str) -> str | None:
    """Tag de script Unicode dominant (latin, arabic, ...) ou None.

    Dérivé de unicodedata — aucun nom de langue, aucun lexique.
    """
    import unicodedata as _ud
    scripts = []
    for ch in text:
        if ch.isalpha():
            try:
                scripts.append(_ud.name(ch).split()[0].lower())
            except ValueError:
                continue
    if not scripts:
        return None
    return Counter(scripts).most_common(1)[0][0]


def parse_digits(text: str) -> list[int]:
    """Entiers (chiffres Unicode, tous scripts) trouvés dans le texte."""
    import unicodedata as _ud
    nums, cur = [], ""
    for ch in text:
        if ch.isdigit():
            try:
                cur += str(_ud.digit(ch))
            except ValueError:
                continue
        elif cur:
            nums.append(int(cur))
            cur = ""
    if cur:
        nums.append(int(cur))
    return nums


def prefix_cir_ids(cir: dict, prefix: str) -> dict:
    """Préfixe les ids d'un CIR (anti-collision inter-blocs), sur place."""
    for _n in cir.get("nodes", []) or []:
        if isinstance(_n, dict) and _n.get("id") is not None:
            _n["id"] = prefix + str(_n["id"])
    remapped = []
    for _e in cir.get("edges", []) or []:
        if isinstance(_e, (list, tuple)) and len(_e) == 3:
            remapped.append([prefix + str(_e[0]), prefix + str(_e[1]), _e[2]])
        elif isinstance(_e, dict):
            _e = dict(_e)
            if _e.get("source") is not None:
                _e["source"] = prefix + str(_e["source"])
            if _e.get("target") is not None:
                _e["target"] = prefix + str(_e["target"])
            remapped.append(_e)
    cir["edges"] = remapped
    return cir


def format_response(handler: InstructionHandler, question: str,
                    subject: str | None = None) -> str:
    """Répond à une question sur le graphe accumulé via QueryVerbalizer.

    Détecte le type de requête depuis la structure (pas la langue).
    subject : sujet déjà résolu par DialogueSession (sinon dernier mot,
    heuristique historique conservée en repli).
    """
    vb = QueryVerbalizer(handler.graph)
    _q = question.lower().strip().rstrip("?.,!")

    # Commandes formelles (explain:, effects:, chain:, counterfactual:)
    cmd, arg = parse_command(question)
    if cmd == "explain" and arg:
        return vb.causes(arg)
    if cmd == "effects" and arg:
        return vb.effects(arg)
    if cmd == "chain" and arg:
        parts = arg.split()
        if len(parts) >= 2:
            return vb.path(parts[0], parts[-1])
    if cmd == "counterfactual" and arg:
        return vb.counterfactual(arg)
    if cmd == "summarize":
        return vb.summarize()
    if cmd == "verbalize":
        return handler.execute("verbalize")

    if subject is None:
        # Repli historique : dernier mot non-vide = sujet probable
        words = [w.strip("?.,!()") for w in question.split() if len(w.strip("?.,!()")) > 2]
        if not words:
            return vb.summarize()
        subject = words[-1]

    # Chercher dans le graphe
    causes = handler.graph.find_causes(subject)
    effects = handler.graph.find_effects(subject)

    if causes and effects:
        return vb.causes(subject) + "\n\n" + vb.effects(subject)
    if causes:
        return vb.causes(subject)
    if effects:
        return vb.effects(subject)

    # Rien trouvé
    return (
        f"  No causal structure found for {subject!r} in the submitted corpus.\n"
        f"  Use /analyze <file> to add documents."
    )


_UNANSWERED_MARKERS = (
    "No causal structure", "no causes found", "no effects found",
    "no causal path", "not found", "not connected",
)


class DialogueSession:
    """État de dialogue : sujets, clarification, langue, journal (9b-e)."""

    def __init__(self, max_topics: int = 3, abandon_after: int = 3,
                 max_nodes: int = SESSION_MAX_NODES,
                 max_edges: int = SESSION_MAX_EDGES,
                 max_turns: int = JOURNAL_KEEP_TURNS) -> None:
        self.max_topics = max_topics
        self.abandon_after = abandon_after
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_turns = max_turns
        self.turn = 0
        self.topics: list[str] = []
        self.last_candidates: list[str] = []
        self.pending: dict | None = None
        self.lang_thread = "und"
        self.lang_override: str | None = None
        self.journal: list[dict] = []
        self._qblocks = 0

    # ------------------------------------------------------------------
    # 9d — langue (tags opaques, jamais de lexique)
    # ------------------------------------------------------------------
    def effective_lang(self) -> str:
        """Priorité figée : consigne explicite > fil > défaut."""
        return self.lang_override or self.lang_thread or "und"

    # ------------------------------------------------------------------
    # 9b — résolution du sujet (structurelle, sans pronoms en dur)
    # ------------------------------------------------------------------
    def _exact_labels(self, graph: CausalGraph, text: str) -> list[str]:
        """Labels du graphe matchés au niveau exact (1), dédupliqués, triés."""
        found = []
        for w in content_words(text):
            for nid, nd in graph.nodes.items():
                if CausalGraph._match_level(nd.get("label", ""), w) == 1:
                    lbl = graph._label(nid)
                    if lbl not in found:
                        found.append(lbl)
        return sorted(found)

    def _partial_label(self, graph: CausalGraph, text: str) -> str | None:
        """Un unique label en match word-boundary (2), sinon None."""
        found = []
        for w in content_words(text):
            for nid, nd in graph.nodes.items():
                if CausalGraph._match_level(nd.get("label", ""), w) == 2:
                    lbl = graph._label(nid)
                    if lbl not in found:
                        found.append(lbl)
        return found[0] if len(found) == 1 else None

    def _paraphrase_label(self, graph: CausalGraph, text: str) -> str | None:
        """M2c — un unique label en overlap paraphrase (≥ seuil), sinon None.

        Plusieurs candidats → None (jamais de clarification sur du flou :
        repli exact/continuité). Synonymes vrais hors portée (moteur+vecs).
        """
        found = []
        for w in content_words(text):
            for nid, nd in graph.nodes.items():
                lbl = graph._label(nid)
                if lbl in found:
                    continue
                if paraphrase_overlap(lbl, w) >= PARAPHRASE_THRESHOLD:
                    found.append(lbl)
        return found[0] if len(found) == 1 else None

    def resolve(self, text: str, graph: CausalGraph) -> str | None:
        """Sujet du message : chiffres → candidats ; exacts → sujet/ambigu ;
        partiel unique → sujet ; sinon continuité (sujet en cours) ou repli."""
        digits = parse_digits(text)
        if digits and self.last_candidates:
            idx = digits[0] - 1  # "1" = premier (index 1-based, universel)
            if 0 <= idx < len(self.last_candidates):
                return self.last_candidates[idx]
        exacts = self._exact_labels(graph, text)
        if len(exacts) == 1:
            return exacts[0]
        if len(exacts) >= 2:
            return None  # ambiguïté → clarify (appelant)
        partial = self._partial_label(graph, text)
        if partial is not None:
            return partial
        paraphrase = self._paraphrase_label(graph, text)
        if paraphrase is not None:
            return paraphrase
        if self.topics:
            return self.topics[-1]  # continuité : aucun concept reconnu
        words = content_words(text)
        return words[-1] if words else None

    def wants_clarify(self, text: str, graph: CausalGraph) -> list[str]:
        """Options si ≥ 2 labels exacts distincts, sinon []."""
        if parse_digits(text) and self.last_candidates:
            return []
        return self._exact_labels(graph, text)

    # ------------------------------------------------------------------
    # 9c — clarification unique
    # ------------------------------------------------------------------
    @staticmethod
    def _match_option(text: str, options: list[str], graph: CausalGraph) -> str | None:
        """Chiffre (1-based) ou répétition du nom (match ≤ 2, unique)."""
        digits = parse_digits(text)
        if digits and 1 <= digits[0] <= len(options):
            return options[digits[0] - 1]
        hits = []
        for w in content_words(text):
            for opt in options:
                if CausalGraph._match_level(opt, w) in (1, 2) and opt not in hits:
                    hits.append(opt)
        _ = graph  # signature homogène (match sur labels d'options, pas le graphe)
        return hits[0] if len(hits) == 1 else None

    # ------------------------------------------------------------------
    # 9e — journal + révocation
    # ------------------------------------------------------------------
    def _record(self, question: str, subject: str | None, response: str,
                status: str, added: int) -> tuple[dict, dict | None]:
        """Enregistre un tour ; résume les plus vieux au-delà du plafond.

        Retourne (entrée, résumé-ou-None). M2a : jamais drop, jamais de refus —
        l'excédent est replié en UNE entrée `summary` (sujets conservés).
        """
        first = next((ln for ln in response.splitlines() if ln.strip()), "")
        entry = {
            "n": self.turn,
            "event": "turn",
            "question": question[:500],
            "subject": subject,
            "summary": first[:120],
            "status": status,
            "lang": self.effective_lang(),
            "added": added,
        }
        self.journal.append(entry)
        summarized = self._summarize_oldest()
        return entry, summarized

    def _summarize_oldest(self) -> dict | None:
        """Replie les tours excédentaires en une entrée `summary` (sujets gardés)."""
        if len(self.journal) <= self.max_turns:
            return None
        n_drop = len(self.journal) - self.max_turns + 1
        dropped = self.journal[:n_drop]
        # Aplatir (un résumé replié ne doit jamais en écraser un autre) :
        # turns et subjects concaténés dans l'ordre.
        dropped_turns: list = []
        subjects = []
        for e in dropped:
            if e.get("event") == "summary":
                dropped_turns.extend(e.get("turns", []))
                subjects.extend(e.get("subjects", []))
            else:
                if e.get("n") is not None:
                    dropped_turns.append(e["n"])
                if e.get("subject"):
                    subjects.append(e["subject"])
        summary = {
            "n": dropped_turns[0] if dropped_turns else 0,
            "event": "summary",
            "question": None,
            "subject": None,
            "summary": f"{len(dropped_turns)} turn(s)",
            "status": "summarized",
            "lang": self.effective_lang(),
            "added": 0,
            "turns": dropped_turns,
            "subjects": subjects,
        }
        self.journal = [summary] + self.journal[n_drop:]
        return summary

    def _enforce_graph_caps(self, graph: CausalGraph) -> list[str]:
        """Rangement M2a : fusion doublons au-delà des plafonds (jamais de refus).

        Nœuds d'abord (le recâblage peut créer des arêtes doublons), puis arêtes.
        Retourne les morceaux d'annonce (`compacted: ...` assemblé par l'appelant).
        """
        parts = []
        if len(graph.nodes) > self.max_nodes:
            n = graph.merge_duplicate_nodes()
            if n:
                parts.append(f"merged {n} node(s)")
        if len(graph.edges) > self.max_edges:
            n = graph.merge_duplicate_edges()
            if n:
                parts.append(f"merged {n} duplicate edge(s)")
        return parts

    def _push_topic(self, subject: str | None) -> None:
        if subject and (not self.topics or self.topics[-1] != subject):
            self.topics.append(subject)
            del self.topics[: max(0, len(self.topics) - self.max_topics)]

    def from_history(self, history: list[dict]) -> None:
        """Reconstruit l'état depuis l'historique persisté (pending non repris)."""
        turns = [e for e in history if e.get("event") == "turn" and e.get("n")]
        if not turns:
            return
        self.turn = max(e["n"] for e in turns)
        self.topics = [e["subject"] for e in turns[-self.max_topics:] if e.get("subject")]
        last_lang = turns[-1].get("lang")
        if last_lang:
            self.lang_thread = last_lang
        self.journal = list(turns[-self.max_turns:])

    def revoke(self, graph: CausalGraph, turn: int | None = None) -> str:
        """Retire les apports utilisateur d'un tour (jamais le corpus)."""
        if turn is None:
            scored = [e for e in self.journal if e.get("added")]
            if not scored:
                return "  revoked: nothing (no user-added turn)"
            turn = scored[-1]["n"]
        removed = graph.drop_edges(
            lambda _s, _d, a: isinstance(a, dict)
            and a.get("session_turn") == turn
            and a.get("session_origin") in _USER_ORIGINS
        )
        if removed:
            return f"  revoked: turn {turn} ({removed} edge(s))"
        return f"  revoked: nothing (turn {turn})"

    # ------------------------------------------------------------------
    # Tour complet
    # ------------------------------------------------------------------
    def handle(self, text: str, handler: InstructionHandler,
               engine=None, vec_index: list | None = None) -> tuple[str, dict]:
        """Traite un message ; retourne (réponse, info journal).

        info = {turn, subject, summary, status, lang, added} pour
        SessionStore.record_exchange() et le log.
        """
        text = (text or "").strip()
        self.turn += 1
        n = self.turn
        tag = script_tag(text)
        if tag:
            self.lang_thread = tag  # message > fil (consigne > tout, cf. effective_lang)

        # Commandes de dialogue (interceptées avant le routage graphe)
        cmd, arg = parse_command(text)
        if cmd == "lang":
            code = (arg or "").strip().lower()
            if not code:
                resp = f"  lang: {self.effective_lang()}"
            elif code in ("auto", "none", "-"):
                self.lang_override = None
                resp = f"  lang: {self.effective_lang()}"
            else:
                self.lang_override = code
                resp = f"  lang: {code}"
            info, _summ = self._record(text, None, resp, "ok", 0)
            if _summ is not None:
                info["summarized"] = _summ
            return resp, info
        if cmd == "revoke":
            target = None
            if arg:
                digits = parse_digits(arg)
                if digits:
                    target = digits[0]
            resp = self.revoke(handler.graph, target)
            info, _summ = self._record(text, None, resp, "corrected", 0)
            if _summ is not None:
                info["summarized"] = _summ
            return resp, info

        notes: list[str] = []

        # 9c — abandon après N tours sans réponse
        if self.pending and n - self.pending["asked_turn"] > self.abandon_after:
            notes.append(f"  abandoned: {' | '.join(self.pending['options'])}")
            self.pending = None

        # 9c — le message répond-il à la clarification en attente ?
        subject: str | None = None
        clarified = False
        if self.pending:
            hit = self._match_option(text, self.pending["options"], handler.graph)
            if hit is not None:
                subject = hit
                self.pending = None
                clarified = True
            else:
                # Bascule nette ? match exact disjoint des options → abandon annoncé
                exacts = self._exact_labels(handler.graph, text)
                if exacts and not set(exacts) & set(self.pending["options"]):
                    notes.append(f"  abandoned: {' | '.join(self.pending['options'])}")
                    self.pending = None

        # 9b — ambiguïté : ≥ 2 labels exacts → clarify (jamais 2 en attente)
        if subject is None and not clarified:
            options = self.wants_clarify(text, handler.graph)
            if len(options) >= 2:
                if self.pending is None:
                    self.pending = {"question": text[:200], "options": options,
                                    "asked_turn": n}
                    self.last_candidates = list(options)
                    resp = f"  clarify: {' | '.join(options)}"
                    if notes:  # bascule annoncée (abandon juste consommé)
                        resp = "\n".join(notes) + "\n" + resp
                    info, _summ = self._record(text, None, resp, "clarification", 0)
                    if _summ is not None:
                        info["summarized"] = _summ
                    return resp, info
                # Une clarification est déjà en attente : rappel, pas de file
                resp = f"  clarify: {' | '.join(self.pending['options'])}"
                if notes:  # bascule annoncée conservée même sur le chemin précoce
                    resp = "\n".join(notes) + "\n" + resp
                info, _summ = self._record(text, None, resp, "clarification", 0)
                if _summ is not None:
                    info["summarized"] = _summ
                return resp, info
            subject = self.resolve(text, handler.graph)

        # 9a — une question enrichit le graphe (poids réduit, jamais un fait)
        added = 0
        if engine is not None and is_question_surface(text) and cmd == "text":
            try:
                cir = engine.analyze(text)
                if cir.get("edges"):
                    before = len(handler.graph.edges)
                    self._qblocks += 1
                    prefix_cir_ids(cir, f"q{self._qblocks:05d}_")
                    handler.add_cir(cir, session_origin="question_utilisateur")
                    for e in handler.graph.edges[before:]:
                        e[2]["session_turn"] = n
                    added = len(handler.graph.edges) - before
            except Exception:  # noqa: S110, BLE001  # enrichissement best-effort
                pass

        # M2a : rangement au-delà des plafonds — annoncé, jamais de refus.
        cap_parts = self._enforce_graph_caps(handler.graph)
        if cap_parts:
            notes.append(f"  compacted: {', '.join(cap_parts)}")

        # Routage : commande formelle > NLU (tête apprise puis surface) > repli
        nlu_cmd = None
        if cmd != "text":
            response = handler.execute(text)
            if cmd in ("explain", "effects"):
                subject = (arg or "").strip() or subject
            elif cmd in ("chain", "chain_t", "before", "delay", "diff", "analogy"):
                parts = (arg or "").replace("->", " ").replace(",", " ").split()
                subject = parts[-1] if parts else subject
            if response is None:
                response = format_response(handler, text, subject=subject)
        else:
            if engine is not None:
                try:
                    cir = engine.analyze(text)
                    _reps = getattr(engine._pipeline, "_cached_reps", None)
                    from .pipeline.nlu_routing import nlu_route
                    nlu_cmd = nlu_route(
                        text,
                        tokens=_reps[0].tokens if _reps else None,
                        pipeline=engine._pipeline,
                    )
                    if nlu_cmd and vec_index:
                        try:
                            from .pipeline.nlu_routing import semantic_resolve_concept
                            _qvecs = getattr(engine._pipeline, "_cached_enriched_vecs", None)
                            if _qvecs is not None and len(_qvecs) > 0:
                                _resolved = semantic_resolve_concept(_qvecs, vec_index)
                                if _resolved:
                                    _parts = nlu_cmd.split(": ", 1)
                                    if len(_parts) == 2:
                                        nlu_cmd = f"{_parts[0]}: {_resolved}"
                        except Exception:  # noqa: S110, BLE001
                            pass
                except Exception:  # noqa: S110, BLE001
                    pass
            if nlu_cmd:
                response = handler.execute(nlu_cmd) or format_response(
                    handler, text, subject=subject)
            else:
                response = format_response(handler, text, subject=subject)

        status = "clarified" if clarified else (
            "unanswered" if any(m in response for m in _UNANSWERED_MARKERS) else "ok")
        if subject is not None:
            response = f"  topic: {subject}\n" + response
            self.last_candidates = [subject]
        self._push_topic(subject)
        info, _summ = self._record(text, subject, response, status, added)
        if _summ is not None:
            info["summarized"] = _summ
        if notes:
            response = "\n".join(notes) + "\n" + response
            info["summary"] = next(
                (ln for ln in response.splitlines() if ln.strip()), "")[:120]
        return response, info
