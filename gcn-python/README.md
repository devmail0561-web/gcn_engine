# gcn-python — GCN Causal Engine

[![PyPI version](https://img.shields.io/pypi/v/gcn-python)](https://pypi.org/project/gcn-python/)
[![Version](https://img.shields.io/badge/version-2.5.0-blue.svg)](https://pypi.org/project/gcn-python/)
[![Python](https://img.shields.io/pypi/pyversions/gcn-python)](https://pypi.org/project/gcn-python/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Tests](https://img.shields.io/badge/tests-424-passing)](tests/)

---

## Ce qu'est GCN

GCN est un **moteur d'extraction et de raisonnement causal vérifiable**.

Il prend du texte brut, extrait la structure causale, et répond à des questions sur cette structure — avec traçabilité jusqu'aux sources.

```
Texte brut (FR, EN, code — autres langues : frontend Rust + connector_lemmas à fournir)
          │
          ▼  GCN Causal Engine
          │
          ▼
CausalIR — graphe causal structuré
  ├── Nœuds typés  (7 types : etat, action, processus…)
  ├── Relations typées  (11 types : cause, enable, prevent…)
  ├── Confiance par arête
  └── Source exacte par relation
          │
          ├── Interrogeable : "what causes X ?"
          ├── Traceable    : "selon quel document ?"
          ├── Analyse contrefactuelle structurelle : "sans X, que se passe-t-il ?" (do-calculus complet dans gcn-backend Rust)
          └── Contradiction detection entre sources
```

**Ce que GCN fait qu'un LLM ne garantit pas :**

| | LLM | GCN |
|-|-----|-----|
| Réponse causale | Plausible, non vérifiable | Tracée jusqu'à la source |
| Requêtes formelles sur le graphe | ❌ | ✅ |
| Détection de contradictions entre sources | ❌ | ✅ (heuristique : co-occurrence prevent/filter+negated) |
| Raisonnement Pearl (do-calculus, gcn-backend Rust) | ❌ | ✅ |
| Entraînable sur corpus spécifique | Coûteux | ✅ léger (NumPy) |

**Pour qui :**
- Analystes CTI / SOC — chaînes d'attaque depuis des rapports de menace
- Auditeurs — obligations causales dans des référentiels réglementaires
- Investigateurs — chaînes de responsabilité depuis des dossiers
- Chercheurs — extraction de claims causaux depuis la littérature
- Ingénieurs — dépendances causales dans le code source

---

## Installation

```bash
pip install gcn-python

# Avec support PyTorch (RGCNLayerGAT) :
pip install "gcn-python[torch]"
```

---

## Usage

### 1. Session interactive — analyser des fichiers et poser des questions

```bash
gcn-discuss --checkpoint model.npz
```

```
  GCN Causal Engine
  ─────────────────────────────────────────────────────
  Corpus : vide  —  utilisez /analyze pour charger des documents

  > /analyze incident_report.txt
    incident_report.txt : 47 relation(s) extraite(s)

  > /analyze threat_reports/
    APT28_2024.txt   : 23 relation(s)
    Mandiant_Q3.txt  : 31 relation(s)
    Total session : 101 relation(s)

  > What causes data exfiltration?
    causes_of: 'data exfiltration'
    ────────────────────────────────────────────────────
    1. [action] authentication_bypass  --[enable]-->  (conf=~0.72)
       source: APT28_2024.txt
    2. [processus] credential_theft  --[cause]-->  (conf=~0.65)
       source: Mandiant_Q3.txt
    ⚠ CONTRADICTION : firewall_rule --[prevent]--> (SecPolicy.txt)
    Note : les scores de confiance sont des softmax non calibrés — préférence relative, pas probabilité absolue.
    ────────────────────────────────────────────────────

  > /save session.json
    Graphe sauvegardé : 101 relations

  > /quit

# Reprendre la session précédente
gcn-discuss --checkpoint model.npz --graph session.json
```

**Commandes disponibles :**

| Commande | Description |
|----------|-------------|
| `/analyze <fichier_ou_répertoire>` | Analyser du texte brut, enrichir le graphe |
| `/save <path.json>` | Persister le graphe de session |
| `/load <path.json>` | Charger un graphe existant |
| `/summarize` | Résumé du corpus courant |
| `/help` | Aide |
| `/quit` | Quitter |

**Questions causales (sans préfixe) :**
```
What causes X?           explain: X
Effects of X?            effects: X
Chain from A to B?       chain: A B
Without X?               counterfactual: X
```

### 2. Indexer un corpus en batch

```bash
gcn-index --corpus rapports/ --checkpoint model.npz --output graph.json
```

Construit le graphe causal depuis un répertoire entier sans interaction.
Utile pour pré-indexer avant une session `gcn-discuss --graph graph.json`.

### 3. Entraîner sur votre corpus

```bash
# Configuration de référence (v2.4.0) — val_edge_macro_f1 = 0.468
gcn-train \
  --data-dir gcn-datasets/real/train_c1_oversampled/ \
  --val-dir  gcn-datasets/real/val/ \
  --epochs 100 --lr 0.0005 \
  --weighted-loss --use-attention --bidirectional \
  --output model.npz
```

Voir `gcn-train --help` pour toutes les options (label smoothing, dropout R-GCN, etc.).

### 3b. Charger un checkpoint

```python
from gcn_python import GCNEngine

# Chargement automatique — l'architecture est encodée dans le .npz
engine = GCNEngine.from_pretrained("model.npz", trusted=True)

# Ou charger manuellement dans un pipeline existant
from gcn_python.training.checkpoint import load_checkpoint
from gcn_python.pipeline.cgnp import CGNPipeline
# (pipeline déjà construit avec la même architecture)
load_checkpoint(pipeline, "model.npz")
```

Le checkpoint `.npz` contient :
- Tous les poids (`encoder_*`, `graph_0`, `graph_extra_*`)
- Les métadonnées d'architecture (`_arch_json`) : dimensions, bidirectionnel, n_relations
- Le vocabulaire de features (`_vocab_json`) pour reconstruire `FeatureVocabulary`

**Compatibilité** : `GCNEngine.from_pretrained()` reconstruit l'architecture depuis `_arch_json`
— aucun paramètre à passer manuellement. Requiert `gcn-python >= 2.1.0`.

### 3c. Inférence sur texte brut (sans annotation UD)

```python
from gcn_python import GCNEngine

engine = GCNEngine.from_pretrained("model.npz", trusted=True, gcn_bin="gcn")
# gcn_bin="gcn" : chemin vers le binaire gcn-cli Rust (doit être dans le PATH)
# Sans gcn_bin : passe en mode heuristique GCNBridgeParser (~80-85% qualité)

cir = engine.analyze("Le gel détruit les cultures, provoquant des pénuries.")
print(cir)  # dict CausalIR JSON-serializable
```

> **Note** : pour la qualité maximale, utiliser des données pré-annotées via `GCNDataLoader`
> plutôt que le bridge heuristique. Voir section [Limitations connues](#limitations-connues--gcnbridgeparser).

### 4. API Python

```python
from gcn_python import GCNEngine
from gcn_python.verbalizer.instructions import CausalGraph
from gcn_python.verbalizer.query_report import QueryVerbalizer

# Charger le moteur
engine = GCNEngine.from_pretrained("model.npz", trusted=True)

# Analyser un corpus
cirs  = engine.analyze_batch(open("corpus.txt").readlines())
graph = CausalGraph.from_cirs(cirs)
graph.save("graph.json")

# Interroger
vb = QueryVerbalizer(graph)
print(vb.causes("data_exfiltration"))
print(vb.path("phishing", "ransomware"))
print(vb.contradictions())

# Reprendre une session
graph2 = CausalGraph.load("graph.json")
```

### 5. Substituer votre propre encodeur

```python
from gcn_python.layer2.interface import CausalEncoder

class MyEncoder(CausalEncoder):
    def forward_node(self, x): ...
    def forward_edge(self, x): ...
    def parameters(self): ...

from gcn_python.pipeline.cgnp import CGNPipeline
pipeline = CGNPipeline(encoder=MyEncoder(), graph=graph, vocabulary=vocab)
```

---

## Format de données (gcn-nl)

Vos données d'entraînement : fichiers JSON avec tokens UD et CIR gold.

```json
{
  "document": {
    "sentences": [{
      "id": "s001",
      "text": "The auth bypass enables data exfiltration.",
      "tokens": [
        {"id": 1, "form": "The",   "lemma": "the",    "pos": "DET",  "dep_rel": "det",   "dep_head": 3, "morph": {}},
        {"id": 2, "form": "auth",  "lemma": "auth",   "pos": "NOUN", "dep_rel": "compound","dep_head": 3,"morph": {}},
        {"id": 3, "form": "bypass","lemma": "bypass", "pos": "NOUN", "dep_rel": "nsubj", "dep_head": 4, "morph": {}}
      ],
      "cir": {
        "nodes": [
          {"id": "n001", "type": "entite",   "label": "auth_bypass",        "token_span": [1, 3]},
          {"id": "n002", "type": "processus","label": "data_exfiltration",   "token_span": [5, 7]}
        ],
        "edges": [{
          "source": "n001", "target": "n002", "relation": "enable",
          "attributes": {"confidence": 1.0, "explicit": true, "negated": false}
        }]
      }
    }]
  }
}
```

**7 types de nœuds :** `etat` · `action` · `transition` · `processus` · `condition` · `entite` · `etat_systemique`

**11 relations :** `cause` · `enable` · `prevent` · `condition` · `concession` · `sequence` · `motivation` · `filter` · `opposition` · `data_dependency` · `control_dependency`

---

## Architecture

```
Texte brut (fr/en/code)
     │
     ▼  [gcn-frontend-fr / gcn-frontend-en / gcn-frontend-code — Rust]
UD tokens (pos, dep_rel, morph, lemma)
     │
     ▼  Layer 1 — FeatureVocabulary  (gcn-python)
207-dim vector par clause par défaut (79-dim syntaxe + 128-dim lexical apprenable ;
voir FeatureVocabulary.d_clause_effective). Sans embeddings (--embedding-dim 0,
déconseillé) : 79-dim syntaxe seule.
     │
     ▼  Layer 2 — MLPEncoder  (remplaçable)
node_logits (N×7) + edge_logits (E×11)
     │
     ▼  Layer 3 — RGCNLayer / RGCNLayerGAT  (remplaçable)
message passing — enrichissement des représentations
     │
     ▼  CGNPipeline.forward() → CausalIR
```

**Précisions architecturales :**

- **Traitement phrase par phrase** : le ML traite une phrase à la fois. Le graphe document est la réunion des CIR individuels — aucune coréférence inter-phrase, aucun raisonnement cross-sentence.
- **Classification, pas prédiction** : le moteur classifie les nœuds (7 types) et les arêtes (11 relations) depuis le texte complet déjà disponible. Il ne prédit pas d'événements futurs.
- **Features syntaxiques + lexicales** : le vecteur clause contient POS, dep_rel, morphologie UD
  **plus, par défaut, un embedding lexical apprenable** (`--embedding-dim 128`). Mesuré sur
  1665 phrases (`BENCHMARK.md` §11) : emb128 apporte +34% val_edge_f1 vs baseline syntaxique.
  Pour un gain supplémentaire : `--embedding-file wiki.fr.vec` sans `--freeze-embeddings`
  (fine-tuning, +45% val_edge_f1 mesuré). Le moteur voit les lemmes mais pas le contexte
  phrastique complet.
- **Teacher forcing** : en entraînement, le R-GCN reçoit les vrais types de relations pour stabiliser les premières epochs. À l'inférence, le two-pass prédit les types sans or. Utiliser `--scheduled-sampling` pour réduire progressivement cette asymétrie.
- **Moteur vs checkpoint** : le moteur est le pipeline (chassis). Le checkpoint `.npz` contient les poids du classifieur embarqué. Charger uniquement des checkpoints de sources fiables (contient du JSON sérialisé, risque équivalent à un pickle).

---

## CLI

| Commande | Description |
|----------|-------------|
| `gcn-discuss` | Session interactive — `/analyze`, Q&A causale |
| `gcn-index` | Indexation batch d'un corpus → graphe JSON |
| `gcn-train` | Entraîner sur un corpus annoté |
| `gcn-eval` | Évaluer un checkpoint |
| `gcn-bootstrap` | Générer des données d'entraînement depuis texte brut |
| `gcn-verbalize` | CIR → texte (ReferenceDecoder, templates) |

---

## Métriques d'évaluation

```python
from gcn_python.evaluation.metrics import (
    node_macro_f1,      # F1 macro sur les 7 types de nœuds
    edge_macro_f1,      # F1 macro sur les 11 relations
    graph_exact_match,  # fraction de phrases avec graphe complet correct
    confusion_matrix,
    per_class_report,
)
```

---

## Limitations connues — Classifieur ML

| Limitation | Impact | Contournement |
|-----------|--------|---------------|
| Features syntaxiques uniquement (si `--embedding-dim 0`) | Sans embeddings, le classifieur ne voit pas le sens des mots. Deux phrases avec le même patron UD reçoivent le même vecteur. | Ne pas désactiver : les embeddings sont ACTIFS PAR DÉFAUT (`--embedding-dim 128`, +34% val_edge_f1 mesuré). Pour un signal plus riche : `--embedding-file wiki.fr.vec` sans `--freeze-embeddings` (fine-tuning, +45% val_edge_f1 mesuré, `BENCHMARK.md` §11). |
| Causalité implicite (sans connecteur) | Difficile à classifier — les features UD ne portent pas l'information implicite. | Annoter des exemples explicitement sans connecteur dans le dataset. |
| Traitement phrase par phrase | Aucune coréférence inter-phrase. Une chaîne causale sur 3 phrases ne sera pas résolue automatiquement. | Réunion manuelle des CIR via `CausalGraph.from_cirs()`. |
| Scores de confiance non calibrés | Les probabilités softmax ne sont pas des probabilités épistémiques. conf=0.7 ≠ 70% de chance d'être correct. | Ne pas utiliser les scores comme seuils de décision absolus. |
| Langues non FR/EN | ES, DE, etc. n'ont ni frontend Rust ni `connector_lemmas`. Nœuds extraits, arêtes absentes. | Implémenter un frontend Rust pour la langue cible + fournir `connector_lemmas`. |

---

## Limitations connues — GCNBridgeParser

`GCNBridgeParser` (dans `gcn_python.frontend.bridge`) permet d'utiliser le pipeline
sans données pré-annotées en appelant le binaire Rust `gcn` via subprocess. Cette
approche est **heuristique** et présente les limitations suivantes :

| Limitation | Impact | Contournement |
|-----------|--------|---------------|
| `root_morph` toujours `{}` | Tense, Aspect, Mood, Polarity absents (14 dims à zéro) | Utiliser `GCNDataLoader` avec des données annotées UD |
| `root_pos` / `dep_rel` heuristiques | Approximation depuis le type de nœud CIR | Idem |
| `is_negative` toujours `False` | Négations non détectées | Idem |
| Qualité globale ~80-85% | Représentation appauvrie vs annotations manuelles | Annoter des données via `gcn-train` |
| Requiert le binaire `gcn` dans le PATH | `GCNBridgeError` si absent | Installer `gcn-core` (Rust) et l'ajouter au PATH |
| Binding PyO3 direct non implémenté | Appel subprocess (latence) | Prévu hors scope v2.x |

**Utilisation recommandée :**

```python
# Qualité maximale — données annotées
from gcn_python.data.loader import GCNDataLoader
loader = GCNDataLoader("gcn-datasets/real/train/")

# Qualité réduite (~80-85%) — texte brut sans annotation
from gcn_python.frontend.bridge import GCNBridgeParser
parser = GCNBridgeParser(gcn_bin="gcn")  # gcn doit être dans le PATH
reps, connectors = parser.parse("Le gel détruit les cultures, provoquant des pénuries.")
```

Pour la production, privilégier les données annotées via `gcn-train` avec `GCNDataLoader`.

---

## Stack complète

| Package | Rôle | Langage |
|---------|------|---------|
| `gcn-python` | Couches ML, entraînement, évaluation, gcn-discuss | Python |
| `gcn-ir` | Types fondamentaux CausalIR | Rust |
| `gcn-knowledge` | Taxonomies, lexique causal | Rust |
| `gcn-frontend-fr` | Parser causal français | Rust |
| `gcn-frontend-en` | Parser causal anglais | Rust |
| `gcn-frontend-code` | Parser code (Python/Rust/JS) | Rust |
| `gcn-middleend` | Graphe causal, cycles, validation | Rust |
| `gcn-backend` | Pearl niveaux 1-2-3, GCN-QL | Rust |
| `gcn-verbalizer` | CausalIR → texte | Rust/Python |

---

## Licence

Apache-2.0 — © Michel Tendeng
