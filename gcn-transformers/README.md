# gcn-transformers

Encodeurs Transformer pour [GCN Causal Engine](https://github.com/devmail0561-web/gcn_engine) (gcn-python).

Ce package fournit des encodeurs basés sur des modèles Transformer pré-entraînés (CamemBERT, XLM-RoBERTa, CodeBERT) comme alternatives au `MLPEncoder` de gcn-python.

**Version 1.0.0** : Option A (projection features UD → hidden_size)  
**Roadmap v1.1** : Option B (tokenization texte brut → embeddings pré-entraînés)

---

## 📦 Installation

> **Ce paquet n'est pas publié sur PyPI** (vérifié le 2026-09-25 :
> `https://pypi.org/pypi/gcn-transformers/json` → 404).
> `pip install gcn-transformers` échoue donc. Installation depuis la source :

```bash
# depuis un clone du dépôt
pip install -e /chemin/vers/gcn-transformers

# avec l'extra benchmark (psutil, pour la mémoire RSS de benchmark_comparison.py)
pip install -e "/chemin/vers/gcn-transformers[benchmark]"
```

> Le dossier `gcn-transformers/` n'est pas encore versionné dans git :
> l'installation `pip install git+https://github.com/devmail0561-web/gcn_engine#subdirectory=gcn-transformers`
> ne fonctionnera qu'après commit.

**Dépendances** :
- `gcn-python >= 2.5.0`
- `transformers >= 4.30.0`
- `torch >= 2.0.0`

**Extra** :
- `benchmark` → `psutil >= 5.9` (optionnel ; sans lui,
  `benchmark_comparison.py` imprime `Mémoire : 0.0 MB`)

---

## 🚀 Utilisation

### Migration depuis MLPEncoder

**Avant (MLPEncoder)** :
```python
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline

vocab = FeatureVocabulary()
d_eff = vocab.d_clause_effective(d_emb=0, subject_object_emb=False)  # 79
d_edge = vocab.d_edge_closed_loop(d_eff, n_node_types=7, d_emb=0,
                                   subject_object_emb=False)  # 365

encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11)
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
```

**Après (XLM-RoBERTa)** :
```python
from gcn_python.layer1.features import FeatureVocabulary
from gcn_transformers import XLMRobertaEncoder  # ← SEUL CHANGEMENT
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline

vocab = FeatureVocabulary()
d_eff = vocab.d_clause_effective(d_emb=0, subject_object_emb=False)  # 79
d_edge = vocab.d_edge_closed_loop(d_eff, n_node_types=7, d_emb=0,
                                   subject_object_emb=False)  # 365

encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge)  # ← Transformers
# ATTENTION : d_out DOIT rester == d_eff (79), PAS 768 !
graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11)  # ← INCHANGÉ
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)

# Le reste (forward, backward, checkpoint) fonctionne sans modification
```

### Entraînement complet (script Python)

**Aucun CLI fourni** : gcn-transformers est une library uniquement. Écrivez votre propre script d'entraînement.

```python
from pathlib import Path
import numpy as np

from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.data.loader import GCNDataLoader
from gcn_transformers import XLMRobertaEncoder

# Configuration (embeddings lexicaux ACTIFS PAR DÉFAUT, comme gcn-train)
from gcn_python.data.loader import reps_from_sentence
from gcn_python.layer1.embedding import WordEmbedding
vocab = FeatureVocabulary()
d_emb = 128  # 0 = désactivé (déconseillé : aucun signal lexical)
d_eff = vocab.d_clause_effective(d_emb, False)  # 79 + 128 = 207
d_edge = vocab.d_edge_closed_loop(d_eff, 7, d_emb, False)

# Pipeline avec XLM-RoBERTa
encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge, learning_rate=1e-5)
graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11)
word_embedding = WordEmbedding(d_emb=d_emb) if d_emb > 0 else None
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                       word_embedding=word_embedding)

# Charger données
train_loader = GCNDataLoader(Path("gcn-datasets/real/train"))
train_samples = list(train_loader)

# Vocabulaire d'embeddings : lemmes racines du train
if word_embedding is not None:
    _lemmas = []
    for s in train_samples:
        _reps, _, _ = reps_from_sentence(s.sentence)
        _lemmas.extend(r.root_lemma for r in _reps)
    word_embedding.build_vocab(_lemmas)

# Entraînement (boucle conforme à gcn-python/training/train.py)
epochs = 20
for epoch in range(epochs):
    total_loss = 0.0
    for sample in train_samples:
        # Forward : reps construites depuis la phrase (PAS de sample.sentence.ud_reps,
        # cet attribut n'existe pas), gold aligné sur les clauses valides
        reps, valid_clause_idxs, connector_reps = reps_from_sentence(sample.sentence)
        if not reps:
            continue
        pipeline.forward(
            reps, sample.sentence.text,
            clause_positions=valid_clause_idxs,
            n_total_clauses=len(sample.sentence.clauses),
            connector_reps=connector_reps,
            gold_edge_map=sample.edge_map,
        )

        # Récupérer logits
        node_logits = pipeline._cached_node_logits
        edge_logits = pipeline._cached_edge_logits
        if node_logits is None or len(node_logits) == 0:
            continue

        if valid_clause_idxs:
            gold_node = sample.gold_node_labels[np.array(valid_clause_idxs, dtype=np.int64)]
        else:
            gold_node = sample.gold_node_labels

        # Filtrer arêtes gold (paires consécutives, indices ORIGINAUX des clauses)
        gold_edge, edge_logits_filtered, valid_idx = None, None, None
        if (len(valid_clause_idxs) >= 2 and sample.edge_map
                and edge_logits is not None and len(edge_logits) > 0):
            pairs = [(valid_clause_idxs[k], valid_clause_idxs[k + 1])
                     for k in range(len(valid_clause_idxs) - 1)]
            gold_edge_full = np.array([sample.edge_map.get(p, -1) for p in pairs],
                                      dtype=np.int64)
            mask = gold_edge_full >= 0
            if mask.any():
                valid_idx = np.where(mask)[0]
                gold_edge = gold_edge_full[valid_idx]
                edge_logits_filtered = edge_logits[valid_idx]

        # Loss
        loss, d_node, d_edge_filtered = pipeline.loss(
            node_logits, edge_logits_filtered,
            gold_node, gold_edge
        )
        total_loss += loss

        # Backward
        d_edge_full = None
        if d_edge_filtered is not None and edge_logits is not None:
            d_edge_full = np.zeros_like(edge_logits)
            d_edge_full[valid_idx] = d_edge_filtered
        pipeline.backward(d_node, d_edge_full, lr=1e-5)

    print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_samples):.4f}")

# Sauvegarder checkpoint
from gcn_python.training.checkpoint import save_checkpoint
save_checkpoint(pipeline, "model_xlmroberta.npz")

# Charger checkpoint (script custom requis)
# NOTE : gcn-eval/from_pretrained incompatibles (hardcodent MLPEncoder)
encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge)
graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11)
word_embedding = WordEmbedding(d_emb=d_emb) if d_emb > 0 else None
pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                       word_embedding=word_embedding)

# IMPORTANT : load_checkpoint() ne fonctionne PAS directement avec gcn-transformers
# (bug CUDA : paramètres copiés dans array NumPy, pas dans tensors PyTorch).
# Utiliser load_parameters() à la place :
import numpy as np
data = np.load("model_xlmroberta.npz", allow_pickle=True)
encoder_params = [data[f"encoder_{i}"] for i in range(len(encoder.parameters()))]
encoder.load_parameters(encoder_params)
# Puis charger graph/vocabulary manuellement ou via load_checkpoint après encodeur
```

---

## 🧠 Encodeurs Disponibles

| Encodeur | Langue | Paramètres | Modèle HuggingFace | Taille |
|----------|--------|------------|---------------------|--------|
| `XLMRobertaEncoder` | FR/EN/Code (100 langues) | 280M | `xlm-roberta-base` | 270 MB |
| `CamembertEncoder` | FR uniquement | 110M | `camembert-base` | 440 MB |
| `CodeBERTEncoder` | Code (Python/Java/JS/...) | 125M | `microsoft/codebert-base` | 500 MB |

### Choix de l'encodeur

- **Multilingue (FR + EN)** : `XLMRobertaEncoder` (recommandé)
- **Français uniquement** : `CamembertEncoder` (hypothèse non mesurée : aucune comparaison de performance n'a été exécutée)
- **Code / docstrings techniques** : `CodeBERTEncoder`

---

## ⚙️ Paramètres

```python
encoder = XLMRobertaEncoder(
    d_clause=79,              # Dimension features UD (vocab.d_clause_effective)
    d_edge=365,               # Dimension edge vectors (vocab.d_edge_closed_loop)
    freeze_layers=10,         # Geler 10/12 couches Transformer (économie calcul)
    learning_rate=1e-5,       # Learning rate AdamW (recommandé pour Transformers)
    device=None,              # "cuda", "cpu", ou None (auto-détection)
    n_node_types=7,           # Nombre de types de nœuds (NODE_TYPES)
    n_relation_types=11,      # Nombre de relations causales (RELATION_TYPES)
)
```

---

## ⚠️ Limitation v1.0 : Option A (Projection UD)

**Version actuelle (1.0.0)** : L'encodeur projette les **features de clause** vers hidden_size
(768-dim), mais **ne tokenise PAS le texte brut**. Depuis la remédiation, les vecteurs de
clause incluent les **embeddings lexicaux apprenables** (79-dim syntaxe + 128-dim lexical = 207-dim
par défaut, comme `gcn-train`) — le déficit lexical total est corrigé, mais il n'y a toujours
pas d'embeddings **contextuels pré-entraînés** (pas de tokenization XLM-RoBERTa du texte).

**Impact** :
- Les poids pré-entraînés XLM-RoBERTa sur texte brut ne sont **pas utilisés** (gelés sur autre chose)
- Performance attendue : **~0.47 val_edge_f1** (similaire à MLPEncoder) — hypothèse non mesurée
- Le Transformer fonctionne sur features UD + embeddings apprenables, sans contexte phrastique

**Roadmap v1.1 (Option B)** :
- Tokenization texte brut → embeddings contextuels
- Ajout de `raw_text` dans `UDRepresentation` (gcn-python 2.6.0)
- Objectif : **val_edge_f1 > 0.70** — hypothèse de roadmap, **aucune base
  empirique** (jamais mesuré)

**Utilisation v1.0** :
```python
# v1.0 : projection clause (79 syntaxe + 128 lexical apprenable par défaut)
encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge)  # d_eff=207, d_edge calculé via vocab
result = pipeline.forward(reps, sentence)  # sentence ignoré

# v1.1 (futur, NON IMPLÉMENTÉ) : tokenization texte
encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge)
# result = pipeline.forward(reps, sentence, texts=["Il pleut", "Inondation"])
#              ↑↑↑ le paramètre texts= n'existe pas — ne pas utiliser
```

---

## 🧪 Tests

```bash
# Installer avec dépendances dev
pip install -e ".[dev]"

# Tests unitaires
pytest tests/ -v

# Tests backward (gradients critiques)
pytest tests/test_backward.py -v

# Tests intégration complète
pytest tests/test_integration.py -v
```

**Ce qu'on peut dire des résultats de tests (sans lancer pytest aujourd'hui)** :
- **35 ids pytest** collectés (25 fonctions ; 5 paramétrées × 3 encodeurs) —
  décompte vérifié dans `.pytest_cache/v/cache/nodeids`.
- L'exécution réelle dépend du cache HuggingFace : sur cette machine, seul
  `models--xlm-roberta-base` est présent → les 10 variantes
  CamemBERT/CodeBERT sont **sautées** (exécution attendue : 25 exécutés /
  10 sautés, échec éventuel non vérifié ici).
- Aucune mesure de couverture n'existe (pas de `pytest-cov` actif, pas de
  `.coverage`) : ne citez aucun pourcentage de couverture.

---

## 📊 Benchmark (aucune mesure à ce jour)

> **Aucun benchmark ni entraînement n'a été exécuté pour ce paquet.**
> Aucun artefact de résultats (`.json` / `.csv` / `.npz`) n'existe dans le
> dossier. Les valeurs ci-dessous sont des **hypothèses de roadmap**, pas des
> mesures ; elles sont conservées uniquement pour indiquer ce qu'il restera à
> mesurer (voir `BENCHMARK-README.md`).

| Encodeur | Val Edge F1 | Params | Mémoire GPU |
|----------|-------------|--------|-------------|
| MLPEncoder (baseline) | **non mesuré** | 50k | **non mesuré** |
| XLMRobertaEncoder (v1.0, Option A) | **non mesuré** (hypothèse ~0.47) | 280M | **non mesuré** |
| CamembertEncoder (v1.0, Option A) | **non mesuré** (hypothèse ~0.47) | 110M | **non mesuré** |

**Note** : v1.0 est limitée par l'Option A (pas de tokenization). L'objectif
`> 0.70` pour v1.1 est une **hypothèse de roadmap sans base empirique**.

---

## 🔧 Architecture

### TransformerEncoderBase (classe abstraite)

Factorise la logique PyTorch ↔ NumPy :
- Conversion `requires_grad` pour autograd
- `backward_node_dx` et `backward_edge_dx` via `torch.autograd`
- Cache snapshots avec graphe autograd préservé
- Optimizer AdamW unique pour node+edge (évite double step)

### Sous-classes concrètes

- `XLMRobertaEncoder` : instancie `AutoModel.from_pretrained("xlm-roberta-base")`
- `CamembertEncoder` : instancie `AutoModel.from_pretrained("camembert-base")`
- `CodeBERTEncoder` : instancie `AutoModel.from_pretrained("microsoft/codebert-base")`

---

## 📄 Licence

Apache License 2.0 — texte intégral dans le fichier `LICENSE`
(copyright « Copyright 2026 Michel Tendeng »).

---

## 🙏 Citation

```bibtex
@software{gcn_transformers,
  author = {Tendeng, Michel},
  title = {gcn-transformers: Transformer Encoders for GCN Causal Engine},
  year = {2026},
  url = {https://github.com/devmail0561-web/gcn_engine}
}
```

---

## 🐛 Bugs / Contributions

Issues et PR : https://github.com/devmail0561-web/gcn_engine/issues

---

## 📚 Voir aussi

- [gcn-python](https://pypi.org/project/gcn-python/) : Moteur GCN de base (MLPEncoder)
- [Transformers](https://huggingface.co/docs/transformers/) : Library HuggingFace
- [GCN Engine Guide](https://github.com/devmail0561-web/gcn_engine) : Documentation complète
