# État actuel du moteur GCN vs Vision

**Date :** 2026-09-18  
**Auteur :** Michel Tendeng

---

## La vision

**GCN est un outil d'analyse causale vérifiable.**

Il extrait des structures causales depuis du texte ou du code, construit un graphe causal interrogeable, et répond à des questions sur ce graphe avec traçabilité jusqu'aux sources.

Ce que GCN fait qu'aucun LLM ne garantit :
- Chaque réponse causale est **tracée jusqu'au document source**
- Les **contradictions entre sources** sont détectées automatiquement
- Le raisonnement est **formel** (Pearl niveaux 1-2-3, GCN-QL)
- Les résultats sont **vérifiables** — pas de texte plausible non fondé

Cas d'usage réels : analyse CTI / SOC, audit de conformité, investigation judiciaire, revue de littérature scientifique, analyse de dépendances dans le code source.

---

## Ce qui est complètement fonctionnel ✅

### Infrastructure ML (gcn-python)

| Composant | État | Détail |
|-----------|------|--------|
| Pipeline d'entraînement | ✅ | train/val/test, early stopping, graph_exact_match |
| `FeatureVocabulary` language-agnostic | ✅ | connector_lemmas vide par défaut, configurable |
| Couche 1 — vectorisation UD | ✅ | 79-dim, universels UD, word embeddings optionnels |
| Couche 2 — MLPEncoder | ✅ | dropout, weight decay, label smoothing |
| Couche 3 — RGCNLayer / GAT | ✅ | message passing, bidirectionnel, 22 relations |
| `GCNEngine.from_pretrained()` | ✅ | architecture déduite depuis `_arch_json` sans hardcoding |
| `gcn-discuss` | ✅ | /analyze fichier/répertoire, Q&A causale, /save, --log |
| `gcn-index` | ✅ | indexation batch corpus → graphe JSON persistant |
| `QueryVerbalizer` | ✅ | rapports multi-lignes avec sources et contradictions |
| CIR → verbalizer direct | ✅ | `decoder.decode_cir(dict)` sans fichier intermédiaire |
| `CausalGraph` persistable | ✅ | save/load/from_cirs |
| Métriques | ✅ | node/edge macro-F1, graph_exact_match, confusion matrix |
| Interfaces extensibles | ✅ | CausalEncoder, CausalGraph, TextParser |
| Checkpoints | ✅ | validation shapes, rollback, _arch_json |
| 206 tests | ✅ | 2 skipped stables |

### Frontends symboliques (gcn-core — Rust)

| Composant | État | Détail |
|-----------|------|--------|
| Parser FR | ✅ | tokenisation, POS, annotation causale, taxonomies |
| Parser EN | ✅ | idem pour l'anglais |
| Parser code | ✅ | Python/Rust/JS via tree-sitter |
| gcn-knowledge | ✅ | 11 taxonomies, lexique causal depuis YAML |
| gcn-middleend | ✅ | construction graphe, cycles (Tarjan SCC), validation |
| gcn-backend | ✅ | Pearl 1-2-3, GCN-QL, export JSON/DOT |
| gcn-cli | ✅ | `gcn analyze`, `gcn query` |

### Outils (gcn-tools)

| Outil | État | Détail |
|-------|------|--------|
| gcn-scraper | ✅ | 6 sources (Wikipedia FR/EN, HAL, Education, GitHub, Docs) |
| gcn-annotate | ✅ | LLM (Anthropic/OpenAI), annotation spaCy, retry |
| Pipeline UD (Phase 1) | ✅ | rederive_spans + annotate_real_dataset + split + validate |

### Publication

| Package | Version | Plateforme |
|---------|---------|------------|
| gcn-python | 2.2.0 | PyPI ✅ |
| gcn-ir, gcn-knowledge, gcn-frontend-*, gcn-middleend, gcn-backend, gcn-verbalizer, gcn-cli | 2.1.0 | crates.io ✅ |

---

## Ce qui est partiellement fonctionnel ⚠️

### Modèle pré-entraîné

| Aspect | État | Problème |
|--------|------|---------|
| Checkpoint existant | ⚠️ | Entraîné sur 767 phrases (678 climatiques + 89 annotées manuellement) |
| Couverture node types | ⚠️ | `processus` et `entite` bien appris. 5 autres types : F1 ≈ 0 |
| Couverture relations | ⚠️ | 5/11 relations représentées. 6 absentes du dataset réel |
| Généralisation domaines | ⚠️ | Fonctionne sur texte climatique FR. Autres domaines : non testés |
| graph_exact_match (test) | ⚠️ | 0.52 sur données climatiques déséquilibrées |

**Rappel :** GCN est un moteur — le modèle pré-entraîné est un artefact de test. L'utilisateur entraîne son propre modèle sur son corpus avec `gcn-train`.

### Frontend bridge Python

| Aspect | État | Problème |
|--------|------|---------|
| Sans gcn-cli | ⚠️ | `GCNBridgeParser` heuristique Python (qualité approximative) |
| Avec gcn-cli | ✅ | Parsing symbolique Rust complet |

---

## Ce qui est absent (non-bloquant pour le moteur)

| Manque | Impact | Priorité |
|--------|--------|---------|
| Dataset 5000+ phrases diversifiées | Améliore le modèle de démonstration, pas le moteur | Données (Phase 4) |
| Modèle pré-entraîné distribuable | Facilite l'usage out-of-the-box | Données + entraînement |

---

## Écart vision vs état actuel

```
VISION : outil d'analyse causale vérifiable
  ┌────────────────────────────────────────────────────┐
  │ Corpus (texte brut, toute langue)                  │
  │   → CIR extrait, graphe causal construit            │
  │   → Questions causales avec traçabilité             │
  │   → Contradictions détectées                        │
  │   → Raisonnement Pearl                              │
  └────────────────────────────────────────────────────┘

ÉTAT ACTUEL :
  ┌────────────────────────────────────────────────────┐
  │ Infrastructure complète ✅                          │
  │ Interfaces language-agnostic ✅                     │
  │ gcn-discuss + gcn-index opérationnels ✅            │
  │                                                    │
  │ Modèle de démo : limité (767 phrases, 2 domaines)  │
  │ → Normal pour un moteur — l'utilisateur entraîne   │
  │   sur son propre corpus                            │
  └────────────────────────────────────────────────────┘

ÉCART PRINCIPAL : les données (travail d'annotation, pas de dev)
```

---

## Ce que l'utilisateur doit faire pour utiliser GCN en production

```bash
# 1. Annoter son corpus (gcn-annotate ou manuel)
gcn-annotate annotate --input phrases.txt --out corpus/

# 2. Entraîner sur son corpus
gcn-train --data-dir corpus/train/ --val-dir corpus/val/ \
          --epochs 100 --output model.npz

# 3. Analyser ses documents
gcn-discuss --checkpoint model.npz
> /analyze mes_rapports/
> What causes X?
```

L'infrastructure est prête. Les données sont la seule variable.
