# Architecture GCN-Core : Moteur vs Outils

**Date** : 2026-09-16  
**Contexte** : Clarification des composants essentiels vs auxiliaires après phase 9

---

## 🎯 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ÉCOSYSTÈME GCN                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │              MOTEUR GCN-CORE (Python ML)                  │    │
│  │                   🔴 CRITIQUE                             │    │
│  │                                                           │    │
│  │  Couche 1 : UDRepresentation + FeatureVocabulary         │    │
│  │  Couche 2 : MLPEncoder (MLP causale)                     │    │
│  │  Couche 3 : R-GCN (message passing relationnel)          │    │
│  │  Couche 4 : CausalIR Emitter + TrainableDecoder (P2d)    │    │
│  │                                                           │    │
│  │  → Forward : UDRep[] → CausalIR                          │    │
│  │  → Training : GCNDataLoader → Backprop → Checkpoints     │    │
│  │  → Inference : Vecteurs enrichis → Verbalisation         │    │
│  └───────────────────────────────────────────────────────────┘    │
│                              ↑                                      │
│                              │ utilise                              │
│                              │                                      │
│  ┌───────────────────────────┴───────────────────────────────┐    │
│  │              FRONTENDS SYMBOLIQUES (Rust)                 │    │
│  │                   🟡 IMPORTANT                            │    │
│  │                                                           │    │
│  │  gcn-frontend-fr : Parser français (UDPipe + règles)     │    │
│  │  gcn-frontend-en : Parser anglais                        │    │
│  │  gcn-frontend-code : Parser Python/Rust (Treesitter)     │    │
│  │                                                           │    │
│  │  → Texte brut → UDRepresentation                         │    │
│  │  → 137 tests Rust passent                                │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │              OUTILS AUXILIAIRES                           │    │
│  │                   🟢 NICE-TO-HAVE                         │    │
│  │                                                           │    │
│  │  bootstrap.py : Wrapper CLI Rust → JSON datasets         │    │
│  │  eval_runner.py : Métriques end-to-end                   │    │
│  │  verbalize_loader.py : Chargeur datasets verbalisation   │    │
│  │                                                           │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Flux de Données — Vue Détaillée

### **Option A : Pipeline Complet (avec bootstrap)**

```
┌──────────────────────────────────────────────────────────────────────┐
│                    GÉNÉRATION DATASETS                               │
└──────────────────────────────────────────────────────────────────────┘

Texte brut                    CLI Rust                    Python
(phrases.txt)                (gcn analyze)              (bootstrap.py)
     │                            │                           │
     │  1. Lecture fichier        │                           │
     ├────────────────────────────>                           │
     │                            │                           │
     │                            │  2. Analyse UD            │
     │                            │     + Annotation causale  │
     │                            │                           │
     │                            │  3. CausalIR (stdout)     │
     │                            ├──────────────────────────>│
     │                            │                           │
     │                            │                           │  4. Conversion
     │                            │                           │     CIR → gcn-nl
     │                            │                           │
     │                            │                           │  5. Écriture JSON
     │                            │                           ├──────────┐
     │                            │                           │          │
     │                            │                           │          ↓
     │                            │                           │    generated_0001.json
     │                            │                           │    generated_0002.json
     │                            │                           │    ...
     │                            │                           │
     └──────────────────────────────────────────────────────────────────┘

                                    ↓ datasets générés

┌──────────────────────────────────────────────────────────────────────┐
│                    ENTRAÎNEMENT MOTEUR ML                            │
└──────────────────────────────────────────────────────────────────────┘

JSON datasets                GCNDataLoader              CGNPipeline
(gcn-datasets/)                   │                          │
     │                            │                          │
     │  1. Charge *.json          │                          │
     ├────────────────────────────>                          │
     │                            │                          │
     │                            │  2. Parse → TrainingSample│
     │                            │     (UDRep + gold labels)│
     │                            │                          │
     │                            │  3. Batch forward        │
     │                            ├─────────────────────────>│
     │                            │                          │
     │                            │                          │  4. Couche 1-4
     │                            │                          │     (UD → MLP → RGCN → IR)
     │                            │                          │
     │                            │  5. Loss + Backward      │
     │                            │<─────────────────────────┤
     │                            │                          │
     │                            │  6. Update poids         │
     │                            │     (SGD)                │
     │                            │                          │
     └────────────────────────────────────────────────────────────────┘

                                    ↓ checkpoint

                            model.npz (poids entraînés)
```

### **Option B : Pipeline Direct (sans bootstrap)**

```
┌──────────────────────────────────────────────────────────────────────┐
│              UTILISATION DIRECTE DU MOTEUR                           │
└──────────────────────────────────────────────────────────────────────┘

Datasets existants            GCNDataLoader              CGNPipeline
(gcn-datasets/examples/)          │                          │
     │                            │                          │
     │  Fichiers JSON             │                          │
     │  déjà annotés              │                          │
     │  (fr_causal_basic.json,    │                          │
     │   fr_causal_cycles.json)   │                          │
     │                            │                          │
     ├────────────────────────────>                          │
     │                            │                          │
     │                            │  Training direct         │
     │                            ├─────────────────────────>│
     │                            │                          │
     │                            │  Inférence directe       │
     │                            │  (si UDRep disponibles)  │
     │                            │                          │
     └────────────────────────────────────────────────────────────────┘

OU

Frontend Rust                UDRepresentation           CGNPipeline
(gcn-frontend-fr)                 │                          │
     │                            │                          │
     │  Texte → UDPipe            │                          │
     │          + règles causales │                          │
     │                            │                          │
     │  Produit UDRep             │                          │
     ├────────────────────────────>                          │
     │                            │                          │
     │                            │  Forward direct          │
     │                            ├─────────────────────────>│
     │                            │                          │
     │                            │  CausalIR (sortie)       │
     │                            │<─────────────────────────┤
     │                            │                          │
     └────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Matrice de Dépendances

### **Composants Essentiels (Moteur)**

| Composant | Dépend de | Est requis par | Peut fonctionner sans |
|-----------|-----------|----------------|----------------------|
| **UDRepresentation** | — | MLPEncoder | ✅ Tout le reste |
| **FeatureVocabulary** | — | MLPEncoder | ✅ Tout le reste |
| **MLPEncoder** | FeatureVocabulary | CGNPipeline | ✅ Tout le reste |
| **RGCNLayer** | — | CGNPipeline | ✅ Tout le reste |
| **TrainableDecoder** | SurfaceVocabulary | CGNPipeline (optionnel) | ✅ Tout le reste |
| **CGNPipeline** | Encoder, Graph, Vocab | Training, Inference | ✅ Bootstrap |
| **GCNDataLoader** | json_reader | Training | ✅ Bootstrap |

### **Composants Auxiliaires (Outils)**

| Composant | Dépend de | Est requis par | Impact si cassé |
|-----------|-----------|----------------|-----------------|
| **bootstrap.py** | CLI Rust (gcn analyze) | — | ⚠️ Génération datasets automatisée |
| **eval_runner.py** | CGNPipeline, metrics | — | ⚠️ Métriques end-to-end |
| **verbalize_loader.py** | json_reader | — | ⚠️ Datasets verbalisation |

---

## 🎯 Scénarios d'Utilisation

### **Scénario 1 : Recherche ML (typique)**

**Besoin** : Entraîner le moteur sur nouveaux datasets

**Workflow** :
1. ✅ Créer JSON manuellement OU utiliser bootstrap
2. ✅ `gcn-train --data-dir ./data --epochs 50`
3. ✅ Évaluer avec `eval_runner.py`

**Bootstrap requis ?** ❌ NON (option de convenance)

---

### **Scénario 2 : Inférence Production**

**Besoin** : Analyser des phrases en temps réel

**Workflow** :
1. ✅ Charger checkpoint entraîné
2. ✅ Frontend Rust : texte → UDRepresentation
3. ✅ `pipeline.forward(reps)` → CausalIR
4. ✅ `decoder.decode(enriched_vecs)` → texte

**Bootstrap requis ?** ❌ NON (jamais utilisé en production)

---

### **Scénario 3 : Génération Datasets Batch**

**Besoin** : Annoter 10000 phrases automatiquement

**Workflow** :
1. ✅ Préparer `phrases.txt`
2. ✅ `gcn-bootstrap --input phrases.txt --out-dir ./data`
3. ✅ Review manuel des JSON générés
4. ✅ Training

**Bootstrap requis ?** ✅ OUI (ou alternative manuelle)

---

## 📉 Impact des Bugs Bootstrap

### **Bugs Identifiés (Issues #1-3)**

```
┌────────────────────────────────────────────────────────────────┐
│                    IMPACT DES BUGS                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Bugs bootstrap.py :                                           │
│  ❌ Issue #1 : stdin au lieu d'arg positionnel                 │
│  ❌ Issue #2 : encodage locale au lieu de UTF-8               │
│  ❌ Issue #3 : crash sur token_span null                      │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         COMPOSANTS AFFECTÉS                          │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │                                                      │    │
│  │  bootstrap.py         → ❌ CASSÉ                     │    │
│  │  gcn-bootstrap CLI    → ❌ CASSÉ                     │    │
│  │                                                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │      COMPOSANTS NON AFFECTÉS (fonctionnent)         │    │
│  ├──────────────────────────────────────────────────────┤    │
│  │                                                      │    │
│  │  UDRepresentation     → ✅ OK                        │    │
│  │  MLPEncoder           → ✅ OK                        │    │
│  │  RGCNLayer            → ✅ OK                        │    │
│  │  TrainableDecoder     → ✅ OK                        │    │
│  │  CGNPipeline          → ✅ OK                        │    │
│  │  GCNDataLoader        → ✅ OK                        │    │
│  │  gcn-train CLI        → ✅ OK                        │    │
│  │  gcn-forward CLI      → ✅ OK (si impl.)            │    │
│  │  Frontend Rust        → ✅ OK                        │    │
│  │                                                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### **Alternatives Fonctionnelles**

Même avec bootstrap cassé, ces workflows fonctionnent :

```bash
# Alternative 1 : CLI Rust direct
gcn analyze "Les ventes baissent." > output.json

# Alternative 2 : Script shell
for line in $(cat phrases.txt); do
  gcn analyze "$line" | python3 convert_to_gcn_nl.py >> dataset.json
done

# Alternative 3 : Datasets manuels
# Éditer JSON à la main dans gcn-datasets/

# Alternative 4 : Frontend Rust programmatique
# Appeler bibliothèque gcn-frontend-fr depuis Rust
```

---

## 🏗️ Architecture Simplifiée — Noyau vs Périphérie

```
                    ┌─────────────────────────┐
                    │                         │
                    │    MOTEUR GCN-CORE      │
                    │                         │
                    │  ┌───────────────────┐  │
                    │  │  UDRepresentation │  │
                    │  │  FeatureVocab     │  │
                    │  └─────────┬─────────┘  │
                    │            │            │
                    │  ┌─────────▼─────────┐  │
                    │  │   MLPEncoder      │  │
                    │  └─────────┬─────────┘  │
                    │            │            │
                    │  ┌─────────▼─────────┐  │
                    │  │   RGCNLayer       │  │
                    │  └─────────┬─────────┘  │
                    │            │            │
                    │  ┌─────────▼─────────┐  │
                    │  │  CausalIR Emitter │  │
                    │  │  TrainableDecoder │  │
                    │  └───────────────────┘  │
                    │                         │
                    └────────┬────────────────┘
                             │
                             │ Utilisé par
             ┌───────────────┼───────────────┐
             │               │               │
             │               │               │
    ┌────────▼────────┐ ┌───▼────────┐ ┌───▼────────┐
    │                 │ │            │ │            │
    │ GCNDataLoader   │ │ bootstrap  │ │ eval_runner│
    │ (datasets JSON) │ │ (wrapper)  │ │ (metrics)  │
    │                 │ │            │ │            │
    │  🟡 IMPORTANT   │ │🟢 OPTIONAL │ │🟢 OPTIONAL │
    │                 │ │            │ │            │
    └─────────────────┘ └────────────┘ └────────────┘
```

### **Légende**
- **Noyau** : Moteur ML (4 couches) — DOIT fonctionner
- **Important** : GCNDataLoader — requis pour training
- **Optional** : Outils de convenance — améliorent l'expérience

---

## 📝 Conclusion

### **Moteur GCN-Core (Essentiel)**

✅ **Complètement fonctionnel après phase 9**
- 4 couches ML opérationnelles
- P2d attention pooling implémenté
- Training/inférence validés
- 127 tests passent

### **Bootstrap (Auxiliaire)**

✅ **Bugs #1-3 corrigés**
- Workflow restauré
- UTF-8 robuste
- Null handling sécurisé
- 10 tests ajoutés

### **Relation**

```
Moteur ≠ Bootstrap

Moteur peut fonctionner SANS bootstrap
Bootstrap NE PEUT PAS fonctionner sans moteur

Bootstrap = wrapper de convenance
Moteur = cœur du système ML
```

---

## 🎯 Pour Aller Plus Loin

**Si vous voulez désactiver bootstrap complètement :**

1. Supprimer `training/bootstrap.py`
2. Supprimer `tests/test_bootstrap.py`
3. Retirer CLI `gcn-bootstrap` de `setup.py`

**Impact** : ❌ AUCUN sur le moteur ML

Le moteur continuerait à fonctionner normalement pour training, inférence, et verbalisation avec datasets JSON existants.

---

## 📚 Références

- **Moteur** : `gcn-python/src/gcn_python/{layer1,layer2,layer3,pipeline}/`
- **Bootstrap** : `gcn-python/src/gcn_python/training/bootstrap.py`
- **Tests moteur** : 127 tests (dont 117 indépendants de bootstrap)
- **Tests bootstrap** : 10 tests isolés
