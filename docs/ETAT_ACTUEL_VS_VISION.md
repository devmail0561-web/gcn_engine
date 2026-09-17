# État actuel du moteur GCN vs Vision

**Date :** 2026-09-17  
**Auteur :** Michel Tendeng

---

## La vision

**Même usage que les LLMs/Transformers**, mais avec une sortie vérifiable.

Un LLM prend du texte brut et produit du texte plausible.
GCN Causal Engine prend du texte brut et produit un graphe causal
structuré, vérifiable, interrogeable, raisonnable — dans n'importe quel
domaine, sur n'importe quel corpus.

---

## Ce qui est complètement fonctionnel ✅

### Infrastructure ML (gcn-python)

| Composant | État | Détail |
|-----------|------|--------|
| Pipeline d'entraînement | ✅ | train/val/test, early stopping, graph_exact_match |
| Couche 1 — FeatureVocabulary | ✅ | 79-dim, UD syntax, word embeddings optionnels |
| Couche 2 — MLPEncoder | ✅ | MLP 3 couches, dropout, weight decay |
| Couche 3 — RGCNLayer | ✅ | message passing, bidirectionnel, 22 relations |
| Couche 3 — RGCNLayerGAT | ✅ | attention par relation |
| Loss — cross-entropie pondérée | ✅ | label smoothing, class weights |
| Backward / SGD | ✅ | mini-batch, accumulation, decoder |
| Checkpoint save/load | ✅ | validation des shapes, rollback |
| Métriques | ✅ | node/edge macro-F1, graph_exact_match, confusion matrix |
| Interfaces extensibles | ✅ | CausalEncoder, CausalGraph, TextParser |
| Évaluation correcte | ✅ | val set séparé, pas de fuite train→val |

### Frontends symboliques (gcn-core — Rust)

| Composant | État | Détail |
|-----------|------|--------|
| Parser FR | ✅ | tokenisation, POS, annotation causale, taxonomies |
| Parser EN | ✅ | idem pour l'anglais |
| Parser code | ✅ | Python/Rust/JS via tree-sitter |
| gcn-knowledge | ✅ | 11 taxonomies, lexique causal chargé depuis YAML |
| gcn-middleend | ✅ | construction graphe, cycles, validation |
| gcn-backend | ✅ | Pearl 1-2-3, GCN-QL, export JSON/DOT |
| gcn-cli | ✅ | `gcn analyze`, `gcn query` opérationnels |

### Outils (gcn-tools)

| Outil | État | Détail |
|-------|------|--------|
| gcn-scraper | ✅ | 6 sources (Wikipedia FR/EN, HAL, Education, GitHub, Docs) |
| gcn-annotate | ✅ | LLM (Anthropic/OpenAI), retry, validation |
| Pipeline UD | ✅ | rederive_spans + annotate_real_dataset + split |

### Publication

| Package | Version | Plateforme |
|---------|---------|------------|
| gcn-python | 2.1.0 | PyPI ✅ |
| gcn-ir, gcn-knowledge, gcn-frontend-*, gcn-middleend, gcn-backend, gcn-verbalizer, gcn-cli | 2.1.0 | crates.io ✅ |

---

## Ce qui est partiellement fonctionnel ⚠️

### Modèle pré-entraîné

| Aspect | État | Problème |
|--------|------|---------|
| Checkpoint existant | ⚠️ | Entraîné sur 767 phrases (678 clima. + 89 divers) |
| Couverture node types | ⚠️ | Seuls `processus` et `entite` bien appris. `etat`, `action`, `transition`, `etat_systemique`, `condition` : F1 = 0 |
| Couverture relations | ⚠️ | 5/11 relations représentées. `sequence`, `motivation`, `filter`, `opposition`, `data_dependency`, `control_dependency` : F1 = 0 |
| Généralisation domaines | ⚠️ | Texte climatique uniquement. Médecine, droit, code : non testés |
| graph_exact_match (test) | ⚠️ | 0.52 — acceptable mais sur données déséquilibrées |

### Frontend bridge Python

| Aspect | État | Problème |
|--------|------|---------|
| `pipeline.analyze(text)` | ⚠️ | Disponible via subprocess gcn-cli, mais qualité heuristique |
| Sans gcn-cli | ⚠️ | `GCNBridgeParser` fonctionne mais approximatif |

---

## Ce qui est absent ❌

### Ce qui sépare GCN d'un LLM utilisable

| Manque | Impact | Priorité |
|--------|--------|---------|
| **Dataset massif diversifié** | Sans données couvrant médecine, droit, science, code, économie — le modèle ne généralise pas. Un LLM pré-entraîné fonctionne sur tout domaine. GCN ne fonctionne que sur le texte climatique | **Critique** |
| **Modèle pré-entraîné distribuable** | L'utilisateur ne peut pas faire `load_checkpoint("gcn_v1.npz")` et analyser du texte médical. Il doit entraîner son propre modèle | **Critique** |
| **API text-in → CIR-out sans friction** | Aujourd'hui l'utilisateur doit connaître l'architecture (MLPEncoder, RGCNLayer, FeatureVocabulary) pour charger un modèle. Un LLM : `model.generate(text)` | **Haute** |
| **Évaluation sur texte naturel non vu** | Toutes nos métriques sont sur les 117 phrases du test set. Aucune mesure sur texte "sauvage" hors dataset | **Haute** |
| **Support multilingue étendu** | FR et EN parsés, mais pas de données d'entraînement EN. Le modèle est de facto FR uniquement | **Moyenne** |

---

## Écart quantifié : GCN actuel vs vision

```
VISION (= LLM niveau)
  ┌─────────────────────────────────────────────────────────┐
  │ Texte brut (tout domaine, toute langue)                  │
  │   → CIR correct, 7 types de nœuds, 11 relations         │
  │   → graph_exact_match > 0.80 sur texte non vu            │
  │   → modèle pré-entraîné distribuable, sans réentraîn.   │
  └─────────────────────────────────────────────────────────┘

ÉTAT ACTUEL
  ┌─────────────────────────────────────────────────────────┐
  │ Texte brut climatique (FR uniquement)                    │
  │   → CIR correct sur processus/entite/cause/enable        │
  │   → graph_exact_match = 0.52 (test set déséquilibré)     │
  │   → modèle non distribuable (trop spécialisé)            │
  └─────────────────────────────────────────────────────────┘

ÉCART PRINCIPAL : les données
  Actuel  : ~800 phrases annotées, 2 domaines, 2 node types dominants
  Nécessaire : 50 000–500 000 phrases, tous domaines, tous types équilibrés
```

---

## Chemin vers la vision

### Ce qui fonctionne déjà et n'a pas besoin de changer

- L'architecture ML (MLP + R-GCN) — prouvée à fonctionner avec les bonnes données
- Le pipeline d'entraînement complet — val set, early stopping, métriques
- Le stack Rust (frontends, backend, GCN-QL, Pearl)
- Les outils de scraping et d'annotation
- L'API d'extensibilité (remplacer encodeur, graphe)

### Ce qui doit être construit

**1. Dataset massif et diversifié**
Le goulot d'étranglement est les données, pas l'architecture.
Objectif : 50 000+ phrases annotées couvrant :
- Médecine, biologie, pharmacologie
- Droit, réglementation
- Économie, finance
- Sciences physiques, chimie
- Code source (Python, Rust, JavaScript)
- Histoire, sociologie
- Techniques et ingénierie

**2. Modèle pré-entraîné de référence**
Une fois le dataset massif constitué, entraîner un checkpoint de référence
distribué avec `gcn-python` — équivalent d'un modèle de base.

**3. API simplifiée**
```python
# Ce que l'utilisateur devrait pouvoir faire :
from gcn_python import GCNEngine
engine = GCNEngine.from_pretrained()      # charge le checkpoint de référence
cir = engine.analyze("any text here")     # c'est tout
```

**4. Fine-tuning sur domaine spécifique**
Comme un LLM peut être fine-tuné, GCN doit pouvoir être adapté à un domaine
en quelques centaines d'exemples supplémentaires :
```bash
gcn-train --encoder-checkpoint base_model.npz --data-dir medical/ --epochs 20
```
