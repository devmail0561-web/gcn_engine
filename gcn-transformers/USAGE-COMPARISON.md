# gcn-transformers : Usages vs Modules Précédents

**Comparaison** : gcn-transformers vs gcn-python (MLPEncoder)

> ## ⚠️ Avertissement — aucune mesure dans ce document
>
> **Aucun benchmark ni entraînement n'a été exécuté pour gcn-transformers**
> (aucun artefact de résultats dans le dépôt, `benchmark_comparison.py` jamais
> lancé). **Tous les chiffres de ce document** (latences, throughputs, mémoire,
> `val_edge_f1`, gains) sont des **estimations ou hypothèses**, pas des mesures.
> Seule exception : `0.468` est une mesure du **projet principal**
> (`BENCHMARK.md` racine, harnais différent — ne pas le comparer directement).
> De plus `pip install gcn-transformers` renvoie 404 : le paquet n'est
> **pas publié**.

---

## 📊 Vue d'Ensemble

| Aspect | **gcn-python (MLPEncoder)** | **gcn-transformers** |
|--------|----------------------------|---------------------|
| **Package** | gcn-python (inclus) | gcn-transformers (optionnel) |
| **Encodeurs** | MLPEncoder, TransformerMLPEncoder | XLMRoberta, CamemBERT, CodeBERT |
| **Paramètres** | 50k (MLP) / 400k (MHA) | 110M-280M |
| **Features** | Syntaxiques UD (79-dim) | Syntaxiques UD (v1.0) |
| **Embeddings** | One-hot POS/dep_rel/morph | Pré-entraînés (v1.1) |
| **Performance** | 0.468 (mesuré dans `BENCHMARK.md` racine, harnais distinct) | **non mesuré** : ~0.47 (v1.0) / >0.70 (v1.1) — hypothèses |
| **Mémoire** | ~50 MB (ordre de grandeur) | estimé ~500-2000 MB (**non mesuré**) |
| **Vitesse** | Rapide (ordre de grandeur) | supposé lent (**non mesuré**) |
| **Installation** | pip install gcn-python | **depuis la source** : `pip install -e .` (PyPI → 404) |
| **Dépendances** | NumPy | NumPy + PyTorch + transformers |

---

## 🎯 Quand Utiliser Quoi ?

### ✅ Utiliser **MLPEncoder** (gcn-python) si :

1. **Ressources limitées**
   - GPU < 4 GB
   - CPU uniquement
   - Mémoire RAM < 8 GB

2. **Vitesse prioritaire**
   - Inférence temps réel
   - Prototypage rapide
   - Expérimentations itératives

3. **Baseline simple**
   - Première approche
   - Validation proof-of-concept
   - Dataset < 1000 phrases

4. **Production légère**
   - Serveur CPU
   - Latence critique (<100ms)
   - Pas de GPU disponible

**Exemple** :
```python
from gcn_python.layer2.reference import MLPEncoder

encoder = MLPEncoder(d_clause=79, d_edge=365)
# ✓ 50k params, ~5ms/phrase, 50 MB RAM
```

---

### ✅ Utiliser **gcn-transformers** si :

1. **Performance maximale recherchée**
   - Dataset > 2000 phrases
   - Fine-tuning possible
   - GPU disponible (recommandé)

2. **Embeddings sémantiques requis**
   - Causalité implicite (sans connecteur)
   - Synonymes/paraphrases
   - Contexte sémantique important

3. **Multilingue**
   - FR + EN mixte
   - Code + texte
   - 100 langues (XLM-RoBERTa)

4. **Recherche / Benchmarks**
   - État de l'art
   - Comparaison avec littérature
   - Publication scientifique

**Exemple** :
```python
from gcn_transformers import XLMRobertaEncoder

encoder = XLMRobertaEncoder(d_clause=79, d_edge=365)
# ✓ 280M params — (~50 ms/phrase et 2 GB : estimations non mesurées)
# ✓ Embeddings pré-entraînés (v1.1)
```

---

## 🔄 Remplacement Direct (Drop-in)

### Architecture Identique

Les deux implémentent le **Protocol CausalEncoder** :

```python
# Architecture gcn-python (4 couches)
UDRepresentation → FeatureVocabulary → CausalEncoder → R-GCN → Pearl

# gcn-transformers s'insère au même endroit :
UDRepresentation → FeatureVocabulary → XLMRobertaEncoder → R-GCN → Pearl
                                        ^^^^^^^^^^^^^^^^^^^^
                                        Seul changement ici !
```

### Migration 1 Ligne

**Avant (MLPEncoder)** :
```python
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder  # ← Ligne à changer
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline

vocab = FeatureVocabulary()
d_eff = vocab.d_clause_effective(0, False)  # 79
d_edge = vocab.d_edge_closed_loop(d_eff, 7, 0, False)  # 365

encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11)
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
```

**Après (gcn-transformers)** :
```python
from gcn_python.layer1.features import FeatureVocabulary
from gcn_transformers import XLMRobertaEncoder  # ← SEUL CHANGEMENT
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline

vocab = FeatureVocabulary()
d_eff = vocab.d_clause_effective(0, False)  # 79
d_edge = vocab.d_edge_closed_loop(d_eff, 7, 0, False)  # 365

encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge)
graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11)
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)

# Le reste (forward, backward, checkpoint) IDENTIQUE
```

**Tout le reste reste identique** :
- Chargement données (GCNDataLoader)
- Forward : `pipeline.forward(reps, sentence)`
- Loss : `pipeline.loss(gold_node_labels, gold_edge_map)`
- Backward : `pipeline.backward(d_node, d_edge, lr)`
- Checkpoint : `save_checkpoint(pipeline, "model.npz")`

---

## 📈 Cas d'Usage Détaillés

### Use Case 1 : Prototypage Rapide

**Besoin** : Tester l'architecture GCN sur nouveau dataset

**Solution** : **MLPEncoder** (gcn-python)

**Pourquoi** :
- Installation rapide (pas de PyTorch/transformers)
- Entraînement rapide (~5 min sur 500 phrases)
- Résultats baseline en quelques itérations

**Performance attendue** : 0.45-0.50 edge_f1

```python
# 10 lignes de code, 5 min entraînement
from gcn_python.layer2.reference import MLPEncoder
encoder = MLPEncoder(d_clause=79, d_edge=365)
# ... train 10 epochs → baseline prête
```

---

### Use Case 2 : Production Temps Réel

**Besoin** : API REST pour détection causale temps réel

**Solution** : **MLPEncoder** (gcn-python)

**Pourquoi** :
- Latence < 100ms par phrase
- CPU suffisant (pas de GPU requis)
- Mémoire < 100 MB
- Déploiement simple (Docker léger)

**Architecture** :
```python
# API Flask/FastAPI
@app.post("/analyze")
def analyze(text: str):
    reps = parse_ud(text)  # spaCy/Stanza
    result = mlp_pipeline.forward(reps, text)  # ~50ms
    return {"causal_graph": result}
```

**Déploiement** :
- Docker image : ~500 MB (vs 5 GB avec Transformers)
- RAM : 512 MB (vs 4 GB)
- Latence : 50-100ms (vs 300-500ms)

---

### Use Case 3 : Recherche État de l'Art

**Besoin** : Publication scientifique, benchmarks compétitifs

**Solution** : **gcn-transformers** (XLMRoberta/CamemBERT)

**Pourquoi** :
- Performance visée (>0.70 edge_f1 en v1.1) — **hypothèse non mesurée**
- Embeddings sémantiques contextuels
- Comparaison avec littérature (BERT-based)
- Fine-tuning sur dataset spécifique

**Protocole** :
```python
from gcn_transformers import CamembertEncoder

# Fine-tuning sur 5000 phrases
encoder = CamembertEncoder(
    d_clause=79, d_edge=365,
    freeze_layers=8,  # Fine-tune 4/12 couches
    learning_rate=2e-5
)

# 20 epochs : ~2h sur GPU = hypothèse, non mesuré
# → val_edge_f1 : objectif 0.72 (hypothèse v1.1, non mesuré)
```

---

### Use Case 4 : Multilingue FR+EN

**Besoin** : Corpus mixte français/anglais

**Solution** : **XLMRobertaEncoder** (gcn-transformers)

**Pourquoi** :
- Pré-entraîné sur 100 langues
- Pas de séparation FR/EN nécessaire
- Transfer learning multilingue

**Exemple** :
```python
from gcn_transformers import XLMRobertaEncoder

encoder = XLMRobertaEncoder(d_clause=79, d_edge=365)

# Dataset mixte FR+EN
sentences = [
    "La pluie cause des inondations.",  # FR
    "Rain causes floods.",              # EN
    "Le code génère une erreur.",       # FR
    "The code throws an error.",        # EN
]

# Un seul modèle pour les deux langues
```

---

### Use Case 5 : Analyse Code (Docstrings)

**Besoin** : Causalité dans commentaires code Python/Rust

**Solution** : **CodeBERTEncoder** (gcn-transformers)

**Pourquoi** :
- Pré-entraîné sur 6.4M fonctions GitHub
- Comprend vocabulaire technique
- Conçu pour les docstrings (propriété du modèle, non mesurée ici)

**Exemple** :
```python
from gcn_transformers import CodeBERTEncoder

encoder = CodeBERTEncoder(d_clause=79, d_edge=365)

# Docstrings Python/Rust
code_comments = [
    "This function fails if the file is missing.",
    "Returns None when the connection times out.",
    "Panic occurs if the mutex is poisoned.",
]

# CodeBERT comprend "fails", "panic", "timeout" → causalité
```

---

### Use Case 6 : Dataset < 500 Phrases

**Besoin** : Petit corpus spécialisé (médical, juridique)

**Solution** : **MLPEncoder** (gcn-python)

**Pourquoi** :
- Transformers sur-apprennent sur petits datasets
- 50k params plus adapté que 280M
- Régularisation plus simple
- Pas de risque d'overfitting catastrophique

**Recommandation** :
```python
# Dataset < 500 phrases
encoder = MLPEncoder(d_clause=79, d_edge=365)

# Dataset > 2000 phrases
encoder = XLMRobertaEncoder(d_clause=79, d_edge=365)
```

---

## 🔍 Différences Techniques

### 1. Features Utilisées

**MLPEncoder** (gcn-python) :
```python
# Features syntaxiques UD uniquement
- POS tags (18 classes)
- Dependency relations (38 classes)
- Morphology (Tense/Aspect/Mood)
- Sujet/Objet binaires

→ Vecteur 79-dim one-hot
→ Pas de sémantique lexicale
```

**gcn-transformers v1.0** (Option A) :
```python
# MÊMES features que MLPEncoder (v1.0)
- Features UD 79-dim
- Projection vers 768-dim
- Transformers voit la projection, pas le texte

→ Performance similaire (~0.47)
→ Limitation documentée
```

**gcn-transformers v1.1** (Option B, futur) :
```python
# Tokenization texte + embeddings pré-entraînés
- Texte brut : "La pluie cause l'inondation"
- Tokenization : ["La", "pluie", "cause", "l'", "inondation"]
- Embeddings 768-dim contextuels (BERT-based)
- Self-attention sur tokens

→ Sémantique lexicale exploitée
→ Performance >0.70 visée (hypothèse de roadmap, non mesurée)
```

---

### 2. Architecture Interne

**MLPEncoder** :
```
Input (79-dim) → Linear(128) → ReLU → Dropout
               → Linear(7) node logits
               → Linear(11) edge logits
```

**gcn-transformers** :
```
Input (79-dim) → proj_ud(768)
               → Transformer 12 couches (280M params)
                 - 10 gelées (pré-entraînées)
                 - 2 fine-tunées
               → Linear(256) → ReLU → Dropout
               → Linear(7) node logits / Linear(11) edge logits
```

---

### 3. Backward et Optimizer

**MLPEncoder** :
```python
# Backward manuel NumPy
def backward_node(self, d_logits):
    # Calcul gradients à la main
    dW = np.outer(d_logits, self.cached_h)
    db = d_logits
    # ...
    return grads

# Optimizer simple
def update_node(self, grads, lr):
    for (dW, db), (W, b) in zip(grads, self.params):
        W -= lr * dW
        b -= lr * db
```

**gcn-transformers** :
```python
# Backward autograd PyTorch
def backward_node_dx(self, d_logits):
    d_logits_t = torch.as_tensor(d_logits)
    self._cached_output.backward(d_logits_t)
    dx = self._cached_input.grad.cpu().numpy()
    # Gradients extraits depuis .grad
    return grads, dx

# Optimizer AdamW (standard Transformers)
def update_node(self, grads, lr):
    self.optimizer.step()  # AdamW
    # lr du pipeline ignoré (warning)
```

---

### 4. Checkpoint et Persistance

**MLPEncoder** :
```python
# Checkpoint NumPy léger
encoder.parameters()  # → 10 arrays (50k params)
# Fichier .npz : ~200 KB

# Load instantané
encoder = MLPEncoder(...)
# Restauration en <1s
```

**gcn-transformers** :
```python
# Checkpoint PyTorch + HuggingFace
encoder.parameters()  # → 199 arrays (280M params)
# Fichier .npz : ~1 GB

# Load lent (télécharge modèle)
encoder = XLMRobertaEncoder(...)
# Premier load : ~30s (télécharge xlm-roberta-base)
# Loads suivants : ~5s (cache local)
```

---

## 💰 Coût Calcul

### Entraînement

### Entraînement (**estimations — aucune mesure**)

| Encodeur | Temps/Epoch (1000 phrases) | GPU | RAM |
|----------|----------------------------|-----|-----|
| **MLPEncoder** | ~2 min (estimé) | Optionnel | ~2 GB (estimé) |
| **TransformerMLPEncoder** | ~3 min (estimé) | Optionnel | ~2 GB (estimé) |
| **XLMRobertaEncoder** | ~15 min (estimé) | Recommandé | ~6 GB (estimé) |
| **CamembertEncoder** | ~12 min (estimé) | Recommandé | ~5 GB (estimé) |

### Inférence (**estimations — aucune mesure**)

| Encodeur | Latence/Phrase | Throughput |
|----------|----------------|------------|
| **MLPEncoder** | ~5 ms (estimé) | ~200 phrases/s (estimé) |
| **TransformerMLPEncoder** | ~8 ms (estimé) | ~125 phrases/s (estimé) |
| **XLMRobertaEncoder** | ~50 ms (estimé) | ~20 phrases/s (estimé) |
| **CamembertEncoder** | ~40 ms (estimé) | ~25 phrases/s (estimé) |

> Ces deux tableaux **n'ont jamais été mesurés** : aucune exécution de
> `benchmark_comparison.py` ou `example_train.py` n'a produit de chiffre.

---

## 🎯 Matrice de Décision (avis qualitatif — aucune case n'est mesurée)

```
                  MLPEncoder          gcn-transformers
                  (gcn-python)        (v1.1)
                  ───────────         ────────────────
Performance       ★★☆☆☆ (0.47 est.)   ★★★★★ (>0.70 visé)   ← non mesuré
Vitesse           ★★★★★ (~5 ms est.)  ★★☆☆☆ (~50 ms est.)  ← non mesuré
Mémoire           ★★★★★ (~50 MB est.) ★★☆☆☆ (~2 GB est.)   ← non mesuré
Installation      ★★★★★ (simple)     ★★★☆☆ (PyTorch)
Sémantique        ★☆☆☆☆ (syntax)     ★★★★★ (embeddings)
Multilingue       ★★★☆☆ (UD)         ★★★★★ (100 langues)
Production        ★★★★★ (CPU)        ★★★☆☆ (GPU)
Recherche         ★★☆☆☆ (baseline)   ★★★★★ (SOTA)
```

**Choix** :
- **Prototypage / Production** → MLPEncoder
- **Recherche / Performance** → gcn-transformers

---

## 📚 Compatibilité

### ✅ Compatible (Drop-in Replacement)

- ✅ **CGNPipeline** : forward/backward/loss
- ✅ **GCNDataLoader** : chargement données
- ✅ **RGCNLayer** : message passing
- ⚠️ **Checkpoint** : `save_checkpoint` OK ; `load_checkpoint()` inutilisable
  tel quel sur CUDA → workaround `load_parameters()` (**sans test automatisé**)
- ✅ **Metrics** : node_accuracy, edge_f1, etc.
- ✅ **Verbalizer** : génération texte Pearl

### ⚠️ Différences Mineures

- ⚠️ **Learning rate** : AdamW utilise son propre lr (warning émis)
- ⚠️ **Snapshots** : consume ~2× mémoire (acceptable N≤40)
- ⚠️ **Eval mode** : appeler `encoder.eval()` explicitement

### ❌ Non Compatible

- ❌ **gcn-train CLI** : pas d'option --encoder-class (v1.0)
  - Solution : utiliser `example_train.py`
- ❌ **TransformerMLPEncoder** : architectures différentes
  - Pas d'interopérabilité checkpoints

---

## 🔄 Migration Stratégie

### Étape 1 : Baseline MLPEncoder

```python
# 1. Baseline rapide (1h)
from gcn_python.layer2.reference import MLPEncoder
encoder = MLPEncoder(d_clause=79, d_edge=365)
# Train 20 epochs → val_edge_f1 = ? (0.47 = hypothèse jamais mesurée)
```

### Étape 2 : Essai gcn-transformers

```python
# 2. Essai Transformers (1 ligne changée, 2h train)
from gcn_transformers import CamembertEncoder
encoder = CamembertEncoder(d_clause=79, d_edge=365)
# Train 20 epochs → val_edge_f1 : à mesurer (hypothèses 0.47 v1.0 / >0.70 v1.1)
```

### Étape 3 : Comparaison

```python
# 3. Comparer métriques
# ⚠️ Structure à REMPLIR avec vos propres mesures — valeurs d'exemple de
# format uniquement, pas des résultats.
results = {
    "MLPEncoder": {"edge_f1": 0.468, "time": "10 min"},   # 0.468 : BENCHMARK.md racine
    "CamembertEncoder": {"edge_f1": None, "time": None},  # jamais mesuré
}

# Si gain < 0.10 → rester sur MLPEncoder (production)
# Si gain > 0.15 → adopter gcn-transformers (recherche)
```

---

## 🎓 Recommandations

### Pour Débuter
1. **Commencer avec MLPEncoder** (baseline rapide)
2. Valider pipeline end-to-end
3. Mesurer performance baseline

### Pour Optimiser
4. **Essayer gcn-transformers** si dataset > 1000 phrases
5. Comparer val_edge_f1 (gain visé +0.20 en v1.1 — hypothèse non mesurée)
6. Fine-tuner freeze_layers et learning_rate

### Pour Production
7. **MLPEncoder si** : latence critique, CPU, petit dataset
8. **gcn-transformers si** : performance critique, GPU, dataset large

---

## 📞 Support

**Questions** :
- MLPEncoder → Voir gcn-python/README.md
- gcn-transformers → Voir gcn-transformers/README.md

**Comparaison benchmark** : **aucun benchmark exécuté** pour ce paquet ; voir `BENCHMARK-README.md`.

---

**TL;DR** :
- **MLPEncoder** = référence rapide et légère (0.468 mesuré dans `BENCHMARK.md` racine)
- **gcn-transformers v1.0** = code écrit, performance **jamais mesurée**
  (hypothèse ~0.47, Option A)
- **gcn-transformers v1.1** = objectif >0.70, **hypothèse de roadmap**
- **Migration** = 1 ligne changée côté API ; **aucune validation par run**
- **Publication** : le paquet n'est **pas sur PyPI** (404)
