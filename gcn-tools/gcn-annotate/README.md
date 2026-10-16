# gcn-annotate — Outil d'annotation LLM v4

Outil externe (hors moteur) pour annoter des phrases en structure causale GCN-NL v4.
Utilise des LLMs (Anthropic Claude, OpenAI GPT) et produit le schéma ETUDE §11.

---

## Installation

```bash
pip install gcn-annotate[anthropic]   # avec Claude
pip install gcn-annotate[openai]      # avec GPT
pip install gcn-annotate[anthropic,openai]
```

---

## Usage

```bash
# Annotation LLM (Anthropic par défaut)
gcn-annotate annotate --input phrases.txt --output dataset/ --lang fr

# Avec backend OpenAI
gcn-annotate annotate --input phrases.txt --output dataset/ \
  --llm-backend openai --model gpt-4o

# Annotation UD structurelle (sans LLM — zéro coût API)
gcn-annotate ud-annotate --input phrases.txt --output dataset/ --lang fr

# Équilibrage des quotas (±10% par relation, N_min=20 pour ternaires)
gcn-annotate balance --input dataset/ --output dataset_balanced/

# Découpage train/val/test stratifié
gcn-annotate split --input dataset/ --output splits/ \
  --train 0.70 --val 0.15 --test 0.15 --seed 42

# Évaluation vs gold
gcn-annotate eval --gold gold.json --pred pred.json
```

### API Python — normalisation

```python
from gcn_annotate.normalize import normalize_node_type, normalize_relation_type

# Migration v2 → v4 automatique
normalize_node_type("action")          # → "processus"
normalize_node_type("etat")            # → "etat_local"
normalize_node_type("etat_systemique") # → "etat_global"
normalize_node_type("concept")         # → "concept"   (v4 valide)

# Relations — aliases PascalCase + anglais
normalize_relation_type("Cause")               # → "cause"
normalize_relation_type("ConditionalCause")    # → "conditional_cause"
normalize_relation_type("joint_cause")         # → "joint_cause"
```

---

## Format de sortie — Schéma v4 (ETUDE §11)

```json
{
  "document": {
    "id": "llm-001",
    "sentences": [
      {
        "id": "s001",
        "text": "Si les traitements échouent, le médicament est prescrit.",
        "intent": "",
        "cir": {
          "sentence_profile": {
            "sentence_type": "declarative",
            "subordination": "condition"
          },
          "salience": {
            "focus_node": "n001",
            "condition_prominence": "foreground"
          },
          "nodes": [
            {
              "id": "n001",
              "type": "processus",
              "label": "échec traitements",
              "token_span": [0, 2],
              "pos": "VERB",
              "morph": {"Mood": "Ind", "Tense": "Pres"}
            },
            {
              "id": "n002",
              "type": "evenement",
              "label": "prescription médicament",
              "token_span": [3, 5],
              "pos": "VERB",
              "morph": {"Voice": "Pass"}
            }
          ],
          "edges": [
            {
              "sources": ["n001"],
              "target": "n002",
              "relation": "conditional_cause",
              "third": {"role": "condition", "node": "n001"},
              "confidence": 0.85,
              "polarity": "positive",
              "voice": "passive",
              "modality": "indicative",
              "has_restriction": false,
              "condition_prominence": "foreground"
            }
          ]
        }
      }
    ]
  }
}
```

---

## Types supportés

### NODE_TYPES D5 — 8 types (ETUDE §8)

| Type | Description | Direction causale |
|------|-------------|-------------------|
| `processus` | Processus/action — absorbe action+transition v2 | Both |
| `etat_local` | État local stable (était "etat" en v2) | Backward |
| `etat_global` | Propriété systémique (était "etat_systemique" en v2) | Accumulative |
| `entite` | Entité non-causale (acteur, objet) | None |
| `condition` | Condition nécessaire ou suffisante | Suspended |
| `concept` | Concept abstrait (nouveau D5) | None |
| `evenement` | Événement ponctuel (nouveau D5) | Forward |
| `contrainte` | Contrainte réglementaire ou physique | Suspended |

### RELATION_TYPES — 19 types (v3.0)

**11 directes :** `cause`, `enable`, `prevent`, `condition`, `concession`, `sequence`, `motivation`, `filter`, `opposition`, `data_dependency`, `control_dependency`

**8 ternaires v3.0 :** `analogy`, `counterfactual`, `conditional_cause`, `mediated_cause`, `joint_cause`, `conditional_prevent`, `mediated_prevent`, `joint_prevent`

---

## Règles d'annotation obligatoires

### `sources` — liste (jamais `source` singulier)

```json
"sources": ["n001"]          // relation directe
"sources": ["n001", "n002"]  // joint_cause — deux sources
```

### `third` — tiers pour relations ternaires

```json
// conditional_cause / conditional_prevent
"third": {"role": "condition", "node": "n003"}

// mediated_cause / mediated_prevent
"third": {"role": "mediator", "node": "n003"}

// joint_cause / joint_prevent — PAS de third, deux arêtes séparées
"third": null
```

### `intent` — uniquement pour les questions

```json
"intent": "explain"   // "Pourquoi X ?"
"intent": "chain"     // "Comment A mène à B ?"
"intent": ""          // déclaratif (laisser vide)
```

Valeurs : `explain`, `effects`, `abduct`, `counterfactual`, `chain`, `chain_t`,
`before`, `delay`, `spof`, `centrality`, `analogy`, `summarize`, `density`,
`coverage`, `reliability`, `diff`, `zoom_in`, `zoom_out`, `aggregate`, `verbalize`, `none`

### Qualifications d'arête

| Champ | Valeurs |
|-------|---------|
| `polarity` | `"positive"` \| `"negative"` |
| `voice` | `"active"` \| `"passive"` |
| `modality` | `"indicative"` \| `"subjunctive"` \| `"conditional"` \| `"imperative"` |
| `has_restriction` | `true` \| `false` (ne...que / only if) |
| `condition_prominence` | `"foreground"` \| `"background"` \| `null` |

---

## Normalisation automatique

Le module `normalize.py` applique :
- Migration v2→D5 : `"etat"→"etat_local"`, `"action"→"processus"`, `"etat_systemique"→"etat_global"`, `"transition"→"processus"`
- `source` singulier → `sources: [...]` (shim backward compat)
- Validation `third.role` ∈ `{"condition","mediator"}`
- Défauts : `polarity="positive"`, `voice="active"`, `modality="indicative"`, `has_restriction=false`
- Types invalides supprimés avec warning

---

## Intégration avec gcn-python

```bash
# 1. Annoter
gcn-annotate annotate --input corpus.txt --output dataset_v4/

# 2. Entraîner
cd gcn-python
gcn-train \
  --data-dir dataset_v4/ \
  --val-dir val_v4/ \
  --epochs 100 --lr 0.0005 \
  --weighted-loss --use-attention --bidirectional \
  --output model_v4.npz

# 3. Tête intention (si questions annotées avec intent)
gcn-train --data-dir dataset_v4/ --n-intent-types 21 --output model_v4_intent.npz
```

---

## Architecture

```
gcn-annotate/
├── src/gcn_annotate/
│   ├── annotator.py      # SYSTEM_PROMPT v4, AnthropicAnnotator, OpenAIAnnotator
│   ├── normalize.py      # NODE_TYPES D5, RELATION_TYPES 19, migration v2→D5, validate third
│   ├── auto_annotate.py  # analyse_sentence() — patterns structurels (FR+EN)
│   ├── smart_annotate.py # annotation guidée avec contexte
│   ├── balanced_auto.py  # annotation équilibrée par type
│   ├── split_dataset.py  # split train/val/test stratifié
│   └── cli.py            # CLI : annotate, eval
└── pyproject.toml
```
