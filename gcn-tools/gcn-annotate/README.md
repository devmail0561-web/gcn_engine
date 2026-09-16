# gcn-annotate — Outil d'annotation LLM pour GCN-NL

Outil externe (hors moteur) qui utilise des LLMs (Anthropic Claude, OpenAI GPT) pour annoter automatiquement des phrases en structure causale GCN-NL.

## Installation

```bash
# Basique (CLI only)
pip install gcn-annotate

# Avec support Anthropic
pip install gcn-annotate[anthropic]

# Avec support OpenAI
pip install gcn-annotate[openai]

# Tout
pip install gcn-annotate[anthropic,openai]
```

## Usage

### Annotation

```bash
# Annotations depuis un fichier de phrases
gcn-annotate annotate --input phrases.txt --output dataset_llm/ --lang fr

# Avec un backend spécifique
gcn-annotate annotate --input phrases.txt --output dataset_llm/ --llm-backend openai --model gpt-4o

# Batch size personnalisé
gcn-annotate annotate --input phrases.txt --output dataset_llm/ --batch-size 20
```

Le fichier `phrases.txt` contient une phrase par ligne. L'outil produit un fichier JSON au format gcn-nl compatible avec `gcn-python` :

```json
{
  "document": {
    "id": "llm-phrases",
    "lang": "fr",
    "sentences": [
      {
        "id": "s001",
        "text": "La pluie cause l'inondation",
        "cir": {
          "nodes": [
            {"id": "n001", "type": "processus", "label": "pluie", "token_span": [1, 1]},
            {"id": "n002", "type": "etat", "label": "inondation", "token_span": [3, 3]}
          ],
          "edges": [
            {"source": "n001", "target": "n002", "relation": "cause"}
          ]
        }
      }
    ]
  }
}
```

### Évaluation

```bash
# Comparer annotations LLM vs gold standard
gcn-annotate eval --gold gold.json --pred pred_llm.json
```

Sortie :
```
Nodes : 150 alignés, accuracy = 0.820
Edges : 89 alignés, accuracy = 0.719
Gold nodes: 150 | Pred nodes: 148
Gold edges: 92 | Pred edges: 89
```

## Types supportés

**Types de nœuds** (7) : `etat`, `action`, `transition`, `processus`, `condition`, `entite`, `etat_systemique`

**Types de relations** (11) : `cause`, `enable`, `prevent`, `condition`, `concession`, `sequence`, `motivation`, `filter`, `opposition`, `data_dependency`, `control_dependency`

## Normalisation

L'outil normalise automatiquement les variantes LLM :
- `"État"` → `"etat"`, `"enables"` → `"enable"`, etc.
- Les types invalides sont supprimés avec un warning
- Les nœuds/arêtes corrompus sont filtrés

## Intégration avec gcn-python

```bash
# 1. Annoter
gcn-annotate annotate --input corpus_brut.txt --output dataset_llm/

# 2. Entraîner
gcn-train --data-dir dataset_llm/ --epochs 50 --output model.npz

# 3. Évaluer
gcn-eval --data-dir dataset_llm/ --model-path model.npz
```

## Coût estimé

Pour 1000 phrases avec `--batch-size 10` (100 appels API) :
- Input ~350k tokens + output ~150k tokens
- Claude Sonnet : ~$0.07–$0.15
- GPT-4o : ~$0.10–$0.25

## Architecture

```
gcn-annotate/
├── src/gcn_annotate/
│   ├── __init__.py
│   ├── annotator.py    # LLMAnnotator protocol + AnthropicAnnotator + OpenAIAnnotator
│   ├── normalize.py    # Normalisation types nœuds/relations, validation
│   └── cli.py          # CLI click : annotate + eval
└── pyproject.toml
```

## Développement

```bash
cd gcn-tools/gcn-annotate
pip install -e ".[dev]"
pytest tests/
```
