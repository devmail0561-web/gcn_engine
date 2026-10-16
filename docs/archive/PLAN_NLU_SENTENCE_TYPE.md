# Plan : Classifieur structurel de phrase — Intégration NLU dans GCN-Core

## Contexte

Le moteur GCN-Core traite toute entrée identiquement : commande DSL ou "texte à analyser". Il n'a aucune compréhension de l'intention ni de la structure phrastique.

**Objectifs :**
1. **Routage par intention** : interrogative → chercher/répondre, impérative → agir, déclarative → extraire
2. **Détection de complexité** : phrase simple vs complexe (subordination)
3. **Détection de subordination typée** : condition, cause, concession (via config externe)

**Exemple clé :**
> « Le médicament n'est prescrit que si les traitements conventionnels ont échoué. »
> → Déclarative, passive, complexe, condition ("si"), restriction ("ne...que")

---

## Architecture : 2 couches, zéro hardcoding

### Couche 1 — Features UD universelles (dans le moteur)

Détection purement structurelle, aucune constante linguistique :

| Feature UD | Signal | Détection |
|---|---|---|
| Ponctuation `?` `!` | Type de phrase | `_final_punctuation()` sur le dernier PUNCT |
| `morph.PronType=Int` | Interrogatif | Token portant un pronom/déterminant interrogatif |
| `morph.Polarity=Neg` | Négation | Root morph ou token-level |
| `morph.Voice=Pass` | Passif | Root morph |
| `morph.Mood=Imp/Sub/Cnd` | Modalité | Root morph |
| `dep_rel in {advcl,ccomp,xcomp,acl}` | Complexité | Présence d'une subordonnée |
| `dep_rel in {aux:pass,nsubj:pass}` | Passif | Structure dep |
| `dep_rel=nsubj` + position | Inversion S-V | `nsubj.id > root.id` |

### Couche 2 — Config JSON externe (dans gcn-datasets/)

Marqueurs linguistiques chargés depuis un fichier JSON. Le moteur ne contient AUCUN lemme — les lexiques sont des données.

**Fichier :** `gcn-datasets/configs/lang_markers.json`

```json
{
  "fr": {
    "interrogative_lemmas": ["pourquoi", "comment", "quand", "quel", "combien", "où"],
    "negation_particles": ["ne", "pas", "jamais", "plus", "rien", "aucun"],
    "restriction_patterns": [["ne", "que"]],
    "subordination_markers": {
      "condition": ["si", "à condition que", "pourvu que", "seulement si"],
      "cause": ["parce que", "car", "puisque", "comme", "étant donné que"],
      "concession": ["bien que", "malgré", "quoique", "même si"],
      "temporal": ["quand", "lorsque", "après que", "avant que"],
      "motivation": ["pour que", "afin que", "afin de"],
      "filter": ["à moins que", "sauf si", "excepté si"]
    }
  },
  "en": {
    "interrogative_lemmas": ["why", "how", "when", "what", "which", "where", "who"],
    "negation_particles": ["not", "never", "no", "neither", "nor"],
    "restriction_patterns": [["only"]],
    "subordination_markers": {
      "condition": ["if", "provided that", "unless"],
      "cause": ["because", "since", "as"],
      "concession": ["although", "despite", "even though"],
      "temporal": ["when", "after", "before", "once"],
      "motivation": ["so that", "in order to"],
      "filter": ["unless", "except"]
    }
  }
}
```

### Fonctionnement

```
classify(tokens, lang_markers=None) → SentenceProfile
    1. Couche 1 : features UD universelles (toujours)
    2. Couche 2 : si lang_markers fourni, fallback lexical pour
       - interrogatifs non détectés par PronType=Int
       - négation non détectée par Polarity=Neg
       - subordination typée (condition/cause/concession)
       - restriction (ne...que)
```

Le paramètre `lang_markers` est optionnel. Sans config, seules les features UD sont utilisées. Avec config, le fallback lexical enrichit la détection.

---

## Principe de non-régression

- `parse_command()` n'est **PAS modifié**
- Tous les paramètres ajoutés sont **optionnels** avec défaut
- Les call sites existants fonctionnent **à l'identique**
- Chaque phase auditée avant passage à la suivante

---

## Phase 1 — Réécrire `sentence_type.py`

**Fichier :** `gcn-python/src/gcn_python/layer1/sentence_type.py`

### 1a. Enums

```python
class SentenceType(str, Enum):
    DECLARATIVE / INTERROGATIVE / IMPERATIVE / EXCLAMATIVE

class Polarity(str, Enum):
    AFFIRMATIVE / NEGATIVE

class Voice(str, Enum):
    ACTIVE / PASSIVE

class Modality(str, Enum):
    INDICATIVE / SUBJUNCTIVE / CONDITIONAL / IMPERATIVE

class Complexity(str, Enum):       # NOUVEAU
    SIMPLE = "simple"
    COMPLEX = "complex"

class SubordinationType(str, Enum): # NOUVEAU
    NONE = "none"
    CONDITION = "condition"
    CAUSE = "cause"
    CONCESSION = "concession"
    TEMPORAL = "temporal"
    MOTIVATION = "motivation"
    FILTER = "filter"
```

### 1b. `LangMarkers` — structure pour la config externe

```python
@dataclass(frozen=True)
class LangMarkers:
    interrogative_lemmas: frozenset[str] = frozenset()
    negation_particles: frozenset[str] = frozenset()
    restriction_patterns: tuple[tuple[str, ...], ...] = ()
    subordination_markers: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> LangMarkers: ...
```

### 1c. `SentenceProfile` enrichi

```python
@dataclass(frozen=True)
class SentenceProfile:
    sentence_type: SentenceType
    polarity: Polarity
    voice: Voice
    modality: Modality
    complexity: Complexity                 # NOUVEAU
    subordination: SubordinationType       # NOUVEAU
    has_restriction: bool                  # NOUVEAU

    # Propriétés existantes inchangées
    expects_response / expects_action / asserts_fact / is_negated / is_passive

    # Nouvelles propriétés
    is_complex → complexity == COMPLEX
    has_condition → subordination == CONDITION
```

### 1d. Classifieurs réécrites

**`_classify_type(tokens, root, root_morph, final_punct, markers=None)`**

```
1. final_punct in ("?", "?!") → INTERROGATIVE
2. Token avec morph PronType=Int + dep_rel in {advmod, det, nsubj, obl}
   + final_punct == "?" → INTERROGATIVE
3. [Fallback si markers] Token avec lemma in markers.interrogative_lemmas
   + dep_rel structural + final_punct == "?" → INTERROGATIVE
4. Inversion S-V (nsubj.id > root.id, root=VERB) → INTERROGATIVE
5. final_punct == "!" et Mood != Imp → EXCLAMATIVE
6. Mood=Imp → IMPERATIVE
7. VERB sans nsubj + Person in {1,2} + Mood != Ind → IMPERATIVE
8. Défaut → DECLARATIVE
```

**`_classify_polarity(tokens, root_morph, markers=None)`**

```
1. root_morph Polarity=Neg → NEGATIVE
2. Token avec morph Polarity=Neg → NEGATIVE
3. [Fallback si markers] Token avec lemma in markers.negation_particles
   + dep_rel == "advmod" → NEGATIVE
4. Défaut → AFFIRMATIVE
```

**`_classify_complexity(tokens)`** — purement structurel, pas de markers

```
Token avec dep_rel in {advcl, ccomp, xcomp, acl} → COMPLEX
Token avec pos=SCONJ + dep_rel=mark → COMPLEX
Défaut → SIMPLE
```

**`_classify_subordination(tokens, markers=None)`**

```
1. Si pas COMPLEX (aucun dep_rel de subordination) → NONE
2. Trouver le token SCONJ/mark
3. [Si markers] Matcher son lemma contre markers.subordination_markers
   → CONDITION / CAUSE / CONCESSION / TEMPORAL / MOTIVATION / FILTER
4. Sans markers → NONE (subordination détectée mais type inconnu)
```

**`_classify_restriction(tokens, markers=None)`**

```
1. [Si markers] Chercher le pattern séquentiel markers.restriction_patterns
   dans les lemmes des tokens (ex: ["ne", "que"]) → True
2. Sans markers → False
```

**`classify(tokens, markers=None)`** — point d'entrée

```python
def classify(tokens: list[dict], markers: LangMarkers | None = None) -> SentenceProfile:
    ...
    stype = _classify_type(tokens, root, root_morph, final_punct, markers)
    polarity = _classify_polarity(tokens, root_morph, markers)
    voice = _classify_voice(tokens, root_morph)  # pas de markers, déjà structurel
    modality = _classify_modality(root_morph, stype)  # idem
    complexity = _classify_complexity(tokens)  # purement structurel
    subordination = _classify_subordination(tokens, markers)
    restriction = _classify_restriction(tokens, markers)
    return SentenceProfile(stype, polarity, voice, modality,
                           complexity, subordination, restriction != False)
```

### Risque Phase 1

`SentenceProfile` change de signature (3 nouveaux champs). Seul `classify()` le construit → zéro impact externe. Le paramètre `markers` est optionnel → callers existants inchangés.

---

## Phase 2 — Config JSON + tests

### 2a. Fichier config

**Fichier :** `gcn-datasets/configs/lang_markers.json` (NOUVEAU)

Contenu : lexiques FR + EN comme décrit dans la section Architecture.

### 2b. Tests exhaustifs

**Fichier :** `gcn-python/tests/test_sentence_type.py` (NOUVEAU, ~400 lignes, ~65 tests)

| Classe | Tests | Ce qui est testé |
|---|---|---|
| `TestSentenceProfile` | 10 | Propriétés existantes + `is_complex`, `has_condition`, `has_restriction`, `frozen` |
| `TestClassifyTypeUD` | 8 | Couche 1 pure UD : `?`, `PronType=Int`, inversion S-V, `Mood=Imp`, `!`, défaut |
| `TestClassifyTypeMarkers` | 5 | Couche 2 fallback : lemme interrogatif FR, lemme EN, sans markers = pas de fallback |
| `TestClassifyPolarityUD` | 3 | `Polarity=Neg` root, `Polarity=Neg` token, défaut |
| `TestClassifyPolarityMarkers` | 3 | Fallback "ne"+"advmod" avec markers, sans markers = pas de fallback |
| `TestClassifyVoice` | 4 | `Voice=Pass`, `aux:pass`, `nsubj:pass`, défaut actif |
| `TestClassifyModality` | 5 | `Sub`, `Cnd`, `Imp`, impératif par type, défaut indicatif |
| `TestClassifyComplexity` | 6 | `advcl`, `ccomp`, `xcomp`, `acl`, `SCONJ+mark`, simple défaut |
| `TestClassifySubordination` | 7 | Avec markers : condition/cause/concession/temporal/filter. Sans markers : NONE même si complexe. Pas complexe : NONE |
| `TestClassifyRestriction` | 4 | Pattern `["ne","que"]` avec markers, `["only"]` EN, sans markers = False, pattern partiel ("ne" sans "que") = False |
| `TestComplexSentences` | 5 | Phrase utilisateur complète, interrogative complexe, impérative+subordination, déclarative+ccomp |
| `TestEdgeCases` | 5 | Tokens vides, dict 4 clés, dict 7 clés, clés manquantes, sans markers |
| `TestLangMarkers` | 4 | `from_json()`, frozenset, config FR, config EN |

**Critère :** Tous passent avant Phase 3.

---

## Phase 3 — Exports et `UDRepresentation.sentence_profile`

### 3a. `layer1/__init__.py`

```python
from .sentence_type import (
    SentenceProfile, SentenceType, Complexity, SubordinationType,
    LangMarkers, classify as classify_sentence,
)
```

### 3b. `representation.py` — propriété

```python
@property
def sentence_profile(self):
    from .sentence_type import classify
    return classify(self.tokens)
```

Note : la propriété n'a pas accès aux `LangMarkers` — elle utilise uniquement la couche UD. Pour le fallback lexical, l'appelant doit utiliser `classify(rep.tokens, markers=...)` directement.

### Tests Phase 3

- `test_import_from_layer1`
- `test_udrepresentation_profile_declarative`
- `test_udrepresentation_profile_complex_advcl`

---

## Phase 4 — Intégration `InstructionHandler.execute()`

**Fichier :** `verbalizer/instructions.py`

### 4a. Signature (L395)

```python
def execute(self, text: str, tokens: list[dict] | None = None,
            markers: LangMarkers | None = None) -> str | None:
```

### 4b. Branche `cmd == "text"` (L401)

```python
if cmd == "text":
    if tokens is not None:
        from ..layer1.sentence_type import classify
        profile = classify(tokens, markers)
        return self._dispatch_by_profile(profile, text)
    return None
```

### 4c. `_dispatch_by_profile()` (NOUVEAU)

```python
def _dispatch_by_profile(self, profile, text: str) -> str | None:
    if profile.expects_response:
        return self._query_from_text(text)
    if profile.expects_action:
        return self._instruct_from_text(text)
    return None
```

### 4d. `_query_from_text()` — extraction sujet + query graphe
### 4e. `_instruct_from_text()` — MVP retourne `None` (hook documenté)

### Risques et tests

| Risque | Garde |
|---|---|
| DSL cassé | `parse_command()` inchangé, 10 tests existants |
| execute() sans tokens | Défaut `None` → `return None` identique |
| DSL avec tokens | `parse_command("explain: X")` → `"explain"` → branche text pas atteinte |

Tests dans `test_instructions.py` :
- `test_execute_text_without_tokens_returns_none`
- `test_execute_text_with_interrogative_tokens`
- `test_execute_dsl_ignores_tokens`
- `test_execute_text_with_declarative_returns_none`

---

## Phase 5 — Intégration `discuss.py`

### 5a. Chargement des markers à l'init du REPL

Dans `run_discuss()`, charger le JSON config une seule fois :

```python
markers = _load_lang_markers(lang="fr")  # ou détecté depuis le corpus
```

### 5b. `_format_response()` — détection par ponctuation

```python
is_question = question.strip().endswith("?")
```

### 5c. Fallback engine (L410-426) — routage NLU avec markers

```python
cir = engine.analyze(user_input)
tmp_handler = InstructionHandler()
tmp_handler.add_cir(cir)
nlu_response = tmp_handler.execute(user_input, tokens=cir_tokens, markers=markers)
```

### Tests

- `test_format_response_dsl_unchanged`
- `test_format_response_question_routing`

---

## Phase 6 — Documentation, régression, audit

### 6a. `test_sentence_type_integration.py` (NOUVEAU)

- CausalGraph + données → interrogative/déclarative/DSL
- Les 21 commandes DSL non cassées

### 6b. CHANGELOG.md / README.md

### 6c. `python -m pytest tests/ -v` — 0 régression

---

## Résumé des fichiers

| Fichier | Action | Risque |
|---|---|---|
| `layer1/sentence_type.py` | Réécrire : 2 couches, +LangMarkers, +Complexity, +SubordinationType | Faible |
| `gcn-datasets/configs/lang_markers.json` | NOUVEAU : lexiques FR/EN | Zéro |
| `tests/test_sentence_type.py` | NOUVEAU ~400 lignes, ~65 tests | Zéro |
| `layer1/__init__.py` | +exports | Zéro |
| `layer1/representation.py` | +propriété `sentence_profile` | Très faible |
| `verbalizer/instructions.py` | +params `tokens`+`markers`, +3 méthodes | Faible |
| `discuss.py` | +chargement markers, +ponctuation, +fallback NLU | Faible |
| `tests/test_instructions.py` | +4 tests régression | Zéro |
| `tests/test_layer1.py` | +2 tests | Zéro |
| `tests/test_sentence_type_integration.py` | NOUVEAU ~100 lignes | Zéro |
| `CHANGELOG.md` / `README.md` | Docs | Zéro |

## Vérification end-to-end

1. `python -m pytest tests/ -v` — 0 échecs
2. 10 `TestParseCommand` existants passent
3. 42 `test_instructions.py` existants passent
4. ~75 nouveaux tests passent
5. Aucun lemme hardcodé dans `sentence_type.py`
6. `lang_markers.json` valide et chargeable
