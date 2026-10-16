# Corrections Appliquées au Plan d'Intégration gcn-transformers

## 📋 Vue d'Ensemble

Le plan original contenait **6 bugs bloquants** et **2 contradictions d'architecture majeures** qui auraient causé des crashs dès la première exécution. Toutes les corrections sont documentées ci-dessous.

---

## 🚨 Bugs Bloquants Corrigés

### 1. **snapshot_edge_cache manquant** ❌→✅
**Problème** : `snapshot_edge_cache()` et `restore_edge_cache()` non implémentés  
**Impact** : `AttributeError` dès la 2ᵉ clause (cgnp.py:910)  
**Correction** :
```python
def snapshot_edge_cache(self) -> dict:
    return {
        'input_tensor': self._cached_edge_input_tensor,
        'output_logits': self._cached_edge_output_logits,
    }

def restore_edge_cache(self, snapshot: dict) -> None:
    self._cached_edge_input_tensor = snapshot['input_tensor']
    self._cached_edge_output_logits = snapshot['output_logits']
```

### 2. **Ordre d'initialisation sous-classe** ❌→✅
**Problème** : `super().__init__()` appelé avant `self.model = ...`  
**Impact** : `'NoneType' has no attribute 'config'` (base.py accède à `self.model.config`)  
**Correction** :
```python
class XLMRobertaEncoder(TransformerEncoderBase):
    def __init__(self, ...):
        # 1. AVANT : instancier self.model
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 2. APRÈS : appeler super().__init__()
        super().__init__(...)
```

### 3. **Dimension edge_head incorrecte** ❌→✅
**Problème** : `nn.Linear(hidden*2=1536, 256)` alors que `forward_edge` reçoit 365-dim  
**Impact** : Shape mismatch crash  
**Correction** :
```python
# AVANT (faux)
self._edge_head = nn.Sequential(
    nn.Linear(self.model.config.hidden_size * 2, 256),  # 1536 ❌
    ...
)

# APRÈS (correct)
self._edge_head = nn.Sequential(
    nn.Linear(d_edge, 256),  # 365 ✅
    ...
)
```

### 4. **Dimensions d_edge/d_clause incorrectes** ❌→✅
**Problème** : `d_edge=100` et `d_clause=80` dans tous les exemples  
**Valeurs réelles** : `d_clause=79`, `d_edge=365` (calculés par `FeatureVocabulary`)  
**Correction** : Tous les exemples mis à jour avec valeurs correctes

### 5. **RGCNLayer d_out incorrect** ❌→✅
**Problème** : Migration suggère `RGCNLayer(d_in=768, d_out=768)`  
**Impact** : `ValueError` (pipeline impose `d_out == vocabulary.d_clause_effective`)  
**Correction** :
```python
# AVANT (faux)
graph = RGCNLayer(d_in=768, d_out=768, n_relations=11)  # ❌

# APRÈS (correct)
graph = RGCNLayer(d_in=79, d_out=79, n_relations=11)  # ✅
# d_out DOIT rester == d_clause (79), pas 768 !
```

### 6. **API backward() inexistante** ❌→✅
**Problème** : Test appelle `pipeline.backward(gold_node_labels=..., gold_edge_map=...)`  
**API réelle** : `loss()` puis `backward(d_node, d_edge, lr)` (2 étapes)  
**Correction** :
```python
# AVANT (faux)
loss, metrics = pipeline.backward(
    gold_node_labels=gold_node_labels,
    gold_edge_map=gold_edge_map,
    lr=1e-5
)

# APRÈS (correct)
loss, d_node, d_edge = pipeline.loss(
    gold_node_labels=gold_node_labels,
    gold_edge_map=gold_edge_map
)
pipeline.backward(d_node, d_edge, lr=1e-5)
```

---

## ⚠️ Bugs Fonctionnels Corrigés

### 7. **Autograd détruit au mauvais endroit** ❌→✅
**Problème** : `torch.no_grad()` dans forward alors que training=False (cgnp.py:876-880)  
**Impact** : Pas de graphe autograd → `RuntimeError` sur `.backward()`  
**Correction** :
```python
# AVANT (faux)
if self.training:
    outputs = self.model(inputs_embeds=H)
else:
    with torch.no_grad():  # ❌ Détruit le graphe
        outputs = self.model(inputs_embeds=H)

# APRÈS (correct)
# PAS de torch.no_grad() — graphe construit toujours
outputs = self.model(inputs_embeds=H)  # ✅
```

### 8. **optimizer.zero_grad() dans backward_node_dx** ❌→✅
**Problème** : `zero_grad()` appelé après chaque nœud → seuls les gradients du dernier survivent  
**Impact** : Gradients des N-1 premiers nœuds perdus  
**Correction** :
```python
# AVANT (faux)
def backward_node_dx(self, d_logits):
    # ... backward ...
    self.optimizer.zero_grad()  # ❌ Efface les gradients accumulés
    return grads, dx

# APRÈS (correct)
def backward_node_dx(self, d_logits):
    # ... backward ...
    # PAS de zero_grad() ici ✅
    return grads, dx

def update_node(self, grads, lr):
    self.optimizer.step()
    self.optimizer.zero_grad()  # ✅ Après update
```

### 9. **Double optimizer.step()** ❌→✅
**Problème** : `step()` dans `update_node()` ET `update_edge()` → momentum/weight_decay appliqués 2×  
**Correction** :
```python
def update_node(self, grads, lr):
    self.optimizer.step()
    self.optimizer.zero_grad()

def update_edge(self, grads, lr):
    # No-op : optimizer déjà step() dans update_node
    pass
```

### 10. **proj non persistant** ❌→✅
**Problème** : `proj = nn.Linear(...)` créé dans `forward_batch()` → poids aléatoires à chaque appel  
**Impact** : Gradients d'une projection hasardeuse, jamais optimisée  
**Correction** :
```python
# AVANT (faux)
def forward_batch(self, X):
    proj = nn.Linear(X.shape[1], hidden_size)  # ❌ Recréé à chaque fois
    H = proj(X_t)

# APRÈS (correct)
def __init__(self, ...):
    self.proj_ud = nn.Linear(d_clause, hidden_size).to(device)  # ✅ Persistant

def forward_batch(self, X):
    H = self.proj_ud(X_t)  # ✅ Utilise le module persistant
```

### 11. **backward_edge_dx = pass** ❌→✅
**Problème** : Méthode non implémentée → `TypeError` au unpack  
**Correction** : Implémentation complète (similaire à `backward_node_dx`)

### 12. **Dropout non désactivé** ❌→✅
**Problème** : `encoder.training = False` ne désactive pas `self.model`  
**Correction** :
```python
def eval(self):
    self.training = False
    self.model.eval()  # ✅ Désactive dropout Transformer
    self._node_head.eval()
    self._edge_head.eval()
```

### 13. **Attributs manquants pour --weighted-loss** ❌→✅
**Problème** : `n_relation_types` et `n_node_types` absents  
**Impact** : `AttributeError` avec `--weighted-loss` (train.py:530)  
**Correction** :
```python
def __init__(self, ..., n_node_types=7, n_relation_types=11):
    self.n_node_types = n_node_types
    self.n_relation_types = n_relation_types
```

### 14. **Warning lr émis 2000×/epoch** ❌→✅
**Problème** : `warnings.warn()` à chaque appel `update_node()`  
**Correction** :
```python
def __init__(self, ...):
    self._lr_warning_emitted = False

def update_node(self, grads, lr):
    if not self._lr_warning_emitted and abs(lr - ...) > 1e-6:
        warnings.warn(...)
        self._lr_warning_emitted = True  # ✅ Une seule fois
```

---

## 🏗️ Contradictions d'Architecture Résolues

### Contradiction #1 : "Une seule ligne" impossible ❌→✅

**Problème** : Pas de point d'injection encodeur dans gcn-python  
- `train.py:317-327` instancie `TransformerMLPEncoder/MLPEncoder` en dur
- `eval_runner.py:97` instancie `MLPEncoder` en dur
- `engine.py:167-171` instancie encodeurs en dur
- Aucune factory, aucun entry-point

**Solution retenue** : **Option 1 - Factory dans gcn-python** (~40 lignes)

#### Modifications gcn-python requises :

**1. checkpoint.py** (~2 lignes)
```python
arch = {
    # ... existant ...
    "encoder_class": type(pipeline.encoder).__name__,  # ← AJOUT
}
```

**2. train.py** (~15 lignes)
```python
@click.option('--encoder-class', type=str, default=None)
def train(..., encoder_class):
    if encoder_class is not None:
        encoder = _load_encoder_class(encoder_class, d_eff, d_edge, ...)
    elif global_attention:
        encoder = TransformerMLPEncoder(...)
    else:
        encoder = MLPEncoder(...)

def _load_encoder_class(class_path: str, ...):
    module_name, class_name = class_path.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)(d_clause=..., d_edge=...)
```

**3. eval_runner.py** (~10 lignes)
```python
_encoder_class = _arch.get("encoder_class", "MLPEncoder")
if _encoder_class not in ("MLPEncoder", "TransformerMLPEncoder"):
    encoder = _load_encoder_class(f"gcn_transformers.{_encoder_class}", ...)
```

**4. engine.py** (~10 lignes, similaire)

**5. pyproject.toml** (entry-point)
```toml
[project.entry-points."gcn_python.encoders"]
xlm_roberta = "gcn_transformers:XLMRobertaEncoder"
camembert = "gcn_transformers:CamembertEncoder"
codebert = "gcn_transformers:CodeBERTEncoder"
```

#### Avantages :
- ✅ 1 CLI, 1 eval, 1 checkpoint (cohérence)
- ✅ Repli `MLPEncoder` par défaut (rétro-compatible)
- ✅ "Une seule ligne" devient vrai (après installation)

### Contradiction #2 : Option A annule les bénéfices Transformers ❌→✅

**Problème** : Plan revendique val_edge_f1 > 0.70 avec Option A (projection UD)

**Réalité** :
- Option A : `proj(79-dim UD) → 768-dim` → **pas de texte brut**
- Embeddings pré-entraînés **inutilisés**
- Gel 10/12 couches **contre-productif** (calibré sur texte, pas UD)
- Performance attendue : **~0.47** (même que MLPEncoder)

**Corrections** :
1. **Option B ajoutée** (tokenization réelle) :
```python
def forward_batch(self, X, texts=None):
    if texts is not None:
        # Option B : tokenization (recommandé)
        inputs = self.tokenizer(texts, ...)
        H = self.model(**inputs).last_hidden_state[:, 0, :]
    else:
        # Option A : projection UD (fallback)
        H = self.proj_ud(X_t)
```

2. **Performance documentée** :
   - v1.0 (Option A) : **val_edge_f1 ≈ 0.47** — **HYPOTHÈSE NON MESURÉE** (aucun
     entraînement exécuté ; cf `BENCHMARK.md` §11 : meilleur résultat mesuré avec
     embeddings simples = 0.2292 sur 1665 phrases, 20 ep)
   - v1.1 (Option B) : **objectif val_edge_f1 > 0.70** — **NON MESURÉE**

3. **Roadmap claire** :
   - v1.0 : publier avec Option A, documenter limitation
   - v1.1 : ajouter `raw_text` à `UDRepresentation` (gcn-python 2.6.0)

---

## 📊 Tableau Comparatif : Avant → Après

| Aspect | Version Originale ❌ | Version Corrigée ✅ |
|--------|---------------------|-------------------|
| **d_clause** | 80 | **79** |
| **d_edge** | 100 | **365** |
| **API backward** | `pipeline.backward(gold_*, lr)` | `loss()` puis `backward(d_*, lr)` |
| **snapshot_edge_cache** | Absent | Implémenté |
| **Ordre init** | `super().__init__()` puis `self.model` | `self.model` puis `super().__init__()` |
| **Edge head dim** | 1536 (hidden*2) | 365 (d_edge) |
| **RGCNLayer d_out** | 768 | 79 (d_clause) |
| **torch.no_grad()** | Détruit graphe en training=False | Jamais appelé |
| **zero_grad()** | Dans backward_node_dx | Dans update_node |
| **proj** | Recréé à chaque forward | Persistant (nn.Module) |
| **backward_edge_dx** | `pass` | Implémenté |
| **Warning lr** | 2000×/epoch | 1×/session |
| **Point injection** | Absent ("une seule ligne" faux) | Factory (~40 lignes) |
| **Performance v1.0** | ">0.70" (fiction) | "~0.47" (hypothèse non mesurée — cf BENCHMARK §11) |
| **URLs** | github.com/vous | devmail0561-web/gcn_engine |
| **base.py lignes** | 700 | 350 |

---

## ✅ Checklist Tests Ajoutés

### Tests Critiques :
- [ ] `torch.autograd.gradcheck` sur `backward_node_dx`
- [ ] `torch.autograd.gradcheck` sur `backward_edge_dx`
- [ ] Accumulation gradients N nœuds (vérifier sum)
- [ ] Snapshot node/edge save/restore
- [ ] Checkpoint save/load avec `encoder_class`
- [ ] Pipeline forward/backward sans crash
- [ ] `n_node_types` et `n_relation_types` accessibles
- [ ] Warning lr émis 1× seulement
- [ ] Mode eval désactive dropout

---

## 📅 Timeline Révisée

| Semaine | Action | Livrables |
|---------|--------|-----------|
| **S0** | PR gcn-python (factory) | ~40 lignes dans train/checkpoint/eval/engine |
| **S1-2** | `TransformerEncoderBase` | base.py (350 lignes), tests backward |
| **S3** | 3 sous-classes | xlm_roberta/camembert/codebert |
| **S4** | Tests intégration | test_integration, test_backward (gradcheck) |
| **S5** | Benchmarks v1.0 | Mesure réelle val_edge_f1 (Option A) |
| **S6** | Documentation | README, limitation Option A claire |
| **S7** | Publication v1.0 | PyPI, annonce |
| **S8+** | Roadmap v1.1 | Option B (tokenization) |

---

## 🎯 Actions Immédiates

### Avant tout code :
1. **PR gcn-python** : factory + entry-point (~40 lignes)
2. **Tests non-régression** : vérifier que MLPEncoder fonctionne toujours

### Développement gcn-transformers :
1. **base.py** : suivre code corrigé (350 lignes, pas 700)
2. **Tests backward** : `torch.autograd.gradcheck` obligatoire
3. **Documentation** : clairement marquer v1.0 = Option A (~0.47)

### Publication v1.0 :
1. **README** : limitation Option A en gros (pas de texte brut)
2. **Roadmap** : v1.1 avec Option B (>0.70)
3. **Exemples** : dimensions correctes (79, 365)

---

## 📁 Fichiers Générés

- **Integration-post-pub-CORRECTED.txt** : Plan complet corrigé (prêt à implémenter)
- **CORRECTIONS-Integration-post-pub.md** : Ce document (récapitulatif)

---

## 🔑 Messages Clés

1. **Le plan original aurait crashé** dès la 1ʳᵉ exécution (6 bugs bloquants)
2. **"Une seule ligne" nécessite ~40 lignes dans gcn-python** (factory)
3. **v1.0 ne dépassera pas MLPEncoder** (~0.47 hypothèse) avec Option A — **non mesuré** ;
   pour référence, le meilleur résultat mesuré avec embeddings simples = 0.2292 (`BENCHMARK.md` §11)
4. **v1.1 avec Option B atteindra >0.70** (tokenization texte) — **non mesuré**
5. **Tests gradients critiques** : `torch.autograd.gradcheck` obligatoire

---

**Status** : ✅ Plan corrigé, prêt à implémenter après PR factory gcn-python
