# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Protocole et implémentations d'annotation LLM."""
from __future__ import annotations

import json
import time
import warnings
from typing import Protocol, runtime_checkable

from .normalize import normalize_annotation


# v4 — 8 types de nœuds (D5 ETUDE)
NODE_TYPES = [
    "etat", "action", "transition", "processus",
    "condition", "entite", "etat_systemique", "contrainte",
]
# v4 — 19 types de relations (D2 ETUDE)
RELATION_TYPES = [
    "cause", "enable", "prevent", "condition", "concession", "sequence",
    "motivation", "filter", "opposition", "data_dependency", "control_dependency",
    "analogy", "counterfactual",
    "conditional_cause", "mediated_cause", "joint_cause",
    "conditional_prevent", "mediated_prevent", "joint_prevent",
]

SYSTEM_PROMPT = f"""Tu es un annotateur expert de relations causales (FR/EN).
Pour chaque phrase, produis un CausalIR au format JSON v4 (schéma ETUDE §11).

=== TYPES DE NŒUDS (8) ===
{json.dumps(NODE_TYPES)}

=== TYPES DE RELATIONS (19) ===
{json.dumps(RELATION_TYPES)}

=== RÈGLES OBLIGATOIRES ===

NŒUDS :
- "id" : identifiant unique (ex: "n001")
- "type" : un des 8 NODE_TYPES
- "label" : expression du concept dans la phrase
- "token_span" : [index_premier_token, index_dernier_token] (0-based)
- "pos" : UPOS du token source (VERB/NOUN/ADJ/...)
- "morph" : dict des traits morphologiques UD (Tense, Mood, Voice, ...)

ARÊTES (schéma v4) :
- "sources" : LISTE d'ids de nœuds sources (ex: ["n001"] ou ["n001","n002"] pour joint_cause)
- "target" : id du nœud cible
- "relation" : un des 19 RELATION_TYPES
- "confidence" : certitude 0.0–1.0
- "polarity" : "positive" | "negative" (negative si la cible est niée)
- "voice" : "active" | "passive"
- "modality" : "indicative" | "subjunctive" | "conditional" | "imperative"
- "has_restriction" : true si restriction exclusive (ne...que / only if)
- "condition_prominence" : "foreground" | "background" | null (position de la condition)

RELATIONS TERNAIRES :
- conditional_cause / conditional_prevent : ajouter "third": {{"role": "condition", "node": "id_du_tiers"}}
- mediated_cause / mediated_prevent : ajouter "third": {{"role": "mediator", "node": "id_du_médiateur"}}
- joint_cause / joint_prevent : "sources": ["n001", "n002"], "third": null (PAS de third)

INTENT (questions uniquement) :
- Ajouter "intent" sur la phrase si c'est une question : explain|effects|chain|chain_t|counterfactual|abduct|summarize|analogy|spof|centrality|before|delay|density|coverage|reliability|diff|zoom_in|zoom_out|aggregate|verbalize
- Laisser "intent": "" pour les déclaratifs

IMPORTANT : utiliser "sources" (liste), JAMAIS "source" (singulier).

Format de sortie (JSON uniquement) :
{{
  "document": {{
    "id": "llm-001",
    "sentences": [
      {{
        "id": "s001",
        "text": "la phrase",
        "intent": "",
        "cir": {{
          "nodes": [...],
          "edges": [...]
        }}
      }}
    ]
  }}
}}

=== EXEMPLES ===

Phrase : "La pluie cause l'inondation."
{{
  "document": {{"id": "ex-001", "sentences": [{{
    "id": "s001", "text": "La pluie cause l'inondation.", "intent": "",
    "cir": {{
      "nodes": [
        {{"id": "n001", "type": "processus", "label": "pluie", "token_span": [1,1], "pos": "NOUN", "morph": {{}}}},
        {{"id": "n002", "type": "etat", "label": "inondation", "token_span": [3,3], "pos": "NOUN", "morph": {{}}}}
      ],
      "edges": [
        {{"sources": ["n001"], "target": "n002", "relation": "cause",
          "confidence": 0.9, "polarity": "positive", "voice": "active",
          "modality": "indicative", "has_restriction": false, "third": null}}
      ]
    }}
  }}]}}
}}

Phrase : "Si les traitements échouent, le médicament est prescrit."
{{
  "document": {{"id": "ex-002", "sentences": [{{
    "id": "s001", "text": "Si les traitements échouent, le médicament est prescrit.", "intent": "",
    "cir": {{
      "nodes": [
        {{"id": "n001", "type": "processus", "label": "échec traitements", "token_span": [1,2], "pos": "VERB", "morph": {{"Mood": "Ind"}}}},
        {{"id": "n002", "type": "evenement", "label": "prescription médicament", "token_span": [3,5], "pos": "VERB", "morph": {{"Voice": "Pass"}}}}
      ],
      "edges": [
        {{"sources": ["n001"], "target": "n002", "relation": "condition",
          "confidence": 0.85, "polarity": "positive", "voice": "passive",
          "modality": "indicative", "has_restriction": false,
          "condition_prominence": "foreground", "third": null}}
      ]
    }}
  }}]}}
}}

Phrase : "La pluie et le vent causent ensemble les inondations."
{{
  "document": {{"id": "ex-003", "sentences": [{{
    "id": "s001", "text": "La pluie et le vent causent ensemble les inondations.", "intent": "",
    "cir": {{
      "nodes": [
        {{"id": "n001", "type": "processus", "label": "pluie", "token_span": [1,1], "pos": "NOUN", "morph": {{}}}},
        {{"id": "n002", "type": "processus", "label": "vent", "token_span": [3,3], "pos": "NOUN", "morph": {{}}}},
        {{"id": "n003", "type": "etat", "label": "inondations", "token_span": [6,6], "pos": "NOUN", "morph": {{}}}}
      ],
      "edges": [
        {{"sources": ["n001", "n002"], "target": "n003", "relation": "joint_cause",
          "confidence": 0.80, "polarity": "positive", "voice": "active",
          "modality": "indicative", "has_restriction": false, "third": null}},
        {{"sources": ["n002"], "target": "n003", "relation": "joint_cause",
          "confidence": 0.80, "polarity": "positive", "voice": "active",
          "modality": "indicative", "has_restriction": false, "third": null}}
      ]
    }}
  }}]}}
}}

Phrase : "Pourquoi les ventes ont-elles baissé ?"
{{
  "document": {{"id": "ex-004", "sentences": [{{
    "id": "s001", "text": "Pourquoi les ventes ont-elles baissé ?", "intent": "explain",
    "cir": {{"nodes": [], "edges": []}}
  }}]}}
}}
"""


@runtime_checkable
class LLMAnnotator(Protocol):
    """Protocole pour les annotateurs LLM."""

    def annotate(self, sentences: list[str], lang: str = "fr") -> list[dict]:
        """Annote une liste de phrases et retourne une liste de CausalIR dicts."""
        ...


class AnthropicAnnotator:
    """Annotateur LLM utilisant l'API Anthropic (Claude)."""

    def __init__(
        self,
        model: str = "claude-sonnet-5-20250514",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic est requis. Installez avec : pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def annotate(self, sentences: list[str], lang: str = "fr") -> list[dict]:
        """Annote un batch de phrases via l'API Anthropic."""
        if not sentences:
            return []

        user_content = "\n\n".join(
            f"Phrase {i+1} : \"{s}\"" for i, s in enumerate(sentences)
        )

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                raw_text = response.content[0].text.strip()
                # Extraire le JSON même s'il est entouré de ```json ... ```
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                raw_doc = json.loads(raw_text)
                normalized = normalize_annotation(raw_doc)
                return normalized.get("document", {}).get("sentences", [])

            except json.JSONDecodeError as e:
                last_error = e
                warnings.warn(
                    f"Réponse LLM non-JSON (tentative {attempt+1}/{self._max_retries}): {e}",
                    UserWarning, stacklevel=2,
                )
            except Exception as e:
                last_error = e
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                is_server_error = "5" in str(e)[:3]
                if is_rate_limit or is_server_error:
                    delay = self._retry_delay * (2 ** attempt)
                    warnings.warn(
                        f"Erreur API (tentative {attempt+1}/{self._max_retries}), "
                        f"retry dans {delay:.1f}s : {e}",
                        UserWarning, stacklevel=2,
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"Annotation échouée après {self._max_retries} tentatives. "
            f"Dernière erreur : {last_error}"
        )


class OpenAIAnnotator:
    """Annotateur LLM utilisant l'API OpenAI."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai est requis. Installez avec : pip install openai"
            ) from e
        self._client = openai.OpenAI()
        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def annotate(self, sentences: list[str], lang: str = "fr") -> list[dict]:
        """Annote un batch de phrases via l'API OpenAI."""
        if not sentences:
            return []

        user_content = "\n\n".join(
            f"Phrase {i+1} : \"{s}\"" for i, s in enumerate(sentences)
        )

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=4096,
                )
                raw_text = response.choices[0].message.content.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                raw_doc = json.loads(raw_text)
                normalized = normalize_annotation(raw_doc)
                return normalized.get("document", {}).get("sentences", [])

            except json.JSONDecodeError as e:
                last_error = e
                warnings.warn(
                    f"Réponse LLM non-JSON (tentative {attempt+1}/{self._max_retries}): {e}",
                    UserWarning, stacklevel=2,
                )
            except Exception as e:
                last_error = e
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                if is_rate_limit:
                    delay = self._retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"Annotation échouée après {self._max_retries} tentatives. "
            f"Dernière erreur : {last_error}"
        )
