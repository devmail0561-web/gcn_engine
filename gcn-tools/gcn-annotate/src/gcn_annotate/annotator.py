"""Protocole et implémentations d'annotation LLM."""
from __future__ import annotations

import json
import time
import warnings
from typing import Protocol, runtime_checkable

from .normalize import normalize_annotation


NODE_TYPES = ["etat", "action", "transition", "processus", "condition", "entite", "etat_systemique"]
RELATION_TYPES = ["cause", "enable", "prevent", "condition", "concession", "sequence",
                  "motivation", "filter", "opposition", "data_dependency", "control_dependency"]

SYSTEM_PROMPT = f"""Tu es un annotateur de relations causales en français.
Pour chaque phrase, produis un CausalIR (Causal Intermediary Representation) au format JSON.

Types de nœuds disponibles : {json.dumps(NODE_TYPES)}
Types de relations disponibles : {json.dumps(RELATION_TYPES)}

Règles :
- Chaque nœud a un "id" (ex: "n001"), un "type" (un des NODE_TYPES), un "label", et un "token_span" [début, fin]
- Chaque arête a "source", "target", "relation" (un des RELATION_TYPES)
- Le token_span est [index_du_premier_token, index_du_dernier_token] (inclus)
- Si tu ne peux pas déterminer un token_span exact, utilise [0, 0]

Format de sortie attendu (JSON uniquement, pas de texte avant ou après) :
{{
  "document": {{
    "id": "llm-001",
    "sentences": [
      {{
        "id": "s001",
        "text": "la phrase d'entrée",
        "cir": {{
          "nodes": [...],
          "edges": [...]
        }}
      }}
    ]
  }}
}}

Exemples few-shot :

Phrase : "La pluie cause l'inondation"
{{
  "document": {{
    "id": "ex-001",
    "sentences": [{{
      "id": "s001",
      "text": "La pluie cause l'inondation",
      "cir": {{
        "nodes": [
          {{"id": "n001", "type": "processus", "label": "pluie", "token_span": [1, 1]}},
          {{"id": "n002", "type": "etat", "label": "inondation", "token_span": [3, 3]}}
        ],
        "edges": [
          {{"source": "n001", "target": "n002", "relation": "cause"}}
        ]
      }}
    }}]
  }}
}}

Phrase : "Le gouvernement a决定 d'interdire les plastiques"
{{
  "document": {{
    "id": "ex-002",
    "sentences": [{{
      "id": "s001",
      "text": "Le gouvernement a décidé d'interdire les plastiques",
      "cir": {{
        "nodes": [
          {{"id": "n001", "type": "action", "label": "décider", "token_span": [2, 2]}},
          {{"id": "n002", "type": "action", "label": "interdire", "token_span": [4, 4]}},
          {{"id": "n003", "type": "entite", "label": "plastiques", "token_span": [6, 6]}}
        ],
        "edges": [
          {{"source": "n001", "target": "n002", "relation": "motivation"}},
          {{"source": "n002", "target": "n003", "relation": "filter"}}
        ]
      }}
    }}]
  }}
}}

Phrase : "Bien que le chômage ait diminué, la pauvreté persiste"
{{
  "document": {{
    "id": "ex-003",
    "sentences": [{{
      "id": "s001",
      "text": "Bien que le chômage ait diminué, la pauvreté persiste",
      "cir": {{
        "nodes": [
          {{"id": "n001", "type": "processus", "label": "chômage", "token_span": [2, 2]}},
          {{"id": "n002", "type": "etat", "label": "pauvreté", "token_span": [5, 5]}}
        ],
        "edges": [
          {{"source": "n001", "target": "n002", "relation": "concession"}}
        ]
      }}
    }}]
  }}
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
