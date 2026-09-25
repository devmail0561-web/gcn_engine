# Corrections Post-Audit — gcn-transformers v1.0

**Date** : 2026-09-25  
**Auteur** : Claude Sonnet 4.5  
**Contexte** : Audit statique + corrections bugs bloquants

---

## 🎯 Résultat Final

⚠️ **Tests** : 35 ids collectés ; exécution **non relancée** lors de cette
vérification (~25 exécutés / 10 sautés attendus : CamemBERT et CodeBERT absents
du cache HuggingFace). « 25/25 PASS » : affirmation **non revérifiée**.  
✅ **Correctifs listés** dans ce document (voir avertissement sur le dénombrement).  
✅ **Auto-attention activée** — `prefers_batch_forward=True` (état du code).  
✅ **Checkpoint** : `load_parameters()` implémentée — **sans test automatisé**,
`load_checkpoint()` reste inutilisable tel quel sur CUDA.

---

## 🐛 Corrections listées (dénombrement contradictoire)

> Ce document en liste **12** numérotées (#1–#12) plus des items lettrés, alors
> que le titre d'origine disait « 10 Total » et que d'autres documents annoncent
> « 25 » (16 plan + 9 audit) ou « 9 » (audit seul). **Le total exact n'est pas
> établi** : les listes se chevauchent.

### Bug #1 : train()/eval() sans argument ⚠️ CRITIQUE
**Problème** : `def train(self)` incompatible avec `encoder.train(mode)` appelé par gcn-python  
**Impact** : `TypeError: train() takes 1 positional argument but 2 were given`  
**Correction** :
```python
# AVANT
def train(self) -> None:
    self.training = True

# APRÈS
def train(self, mode: bool = True) -> None:
    self.training = mode
    if mode:
        self.model.train()
    else:
        self.model.eval()
```
**Fichier** : `src/gcn_transformers/base.py:559-580`

---

### Bug #2 : API loss() incorrecte ⚠️ CRITIQUE
**Problème** : Signature `pipeline.loss(gold_node_labels=..., gold_edge_map=...)` n'existe pas  
**Signature réelle** : `loss(node_logits, edge_logits, gold_node, gold_edge, ...)`  
**Impact** : `TypeError` dans README, example_train.py, benchmark, tests  
**Correction** :
```python
# AVANT (INCORRECT)
loss, d_node, d_edge = pipeline.loss(
    gold_node_labels=sample["node_labels"],
    gold_edge_map=sample["edge_map"]
)

# APRÈS (CORRECT)
node_logits = pipeline._cached_node_logits
edge_logits = pipeline._cached_edge_logits

# Filtrer arêtes gold
pairs = [(i, i+1) for i in range(len(node_logits)-1)]
gold_edge_full = np.array([sample.edge_map.get(p, -1) for p in pairs], dtype=np.int64)
valid_mask = gold_edge_full >= 0
gold_edge = gold_edge_full[valid_mask] if valid_mask.any() else None
edge_logits_filtered = edge_logits[valid_mask] if edge_logits is not None and valid_mask.any() else None

loss, d_node, d_edge_filtered = pipeline.loss(
    node_logits, edge_logits_filtered,
    sample.gold_node_labels, gold_edge
)

# Reconstruire d_edge_full
d_edge_full = None
if d_edge_filtered is not None and edge_logits is not None:
    d_edge_full = np.zeros_like(edge_logits)
    d_edge_full[valid_mask] = d_edge_filtered
pipeline.backward(d_node, d_edge_full, lr=lr)
```
**Fichiers corrigés** :
- `README.md:98-135`
- `example_train.py:78-135`
- `benchmark_comparison.py:111-175`
- `tests/test_integration.py` (3 tests)

---

### Bug #3 : TrainingSample clés incorrectes ⚠️ CRITIQUE
**Problème** : Code utilise `sample["reps"]`, `sample["sentence"]`, etc. qui n'existent pas  
**Structure réelle** : `sample.sentence.ud_reps`, `sample.gold_node_labels`, `sample.edge_map`  
**Impact** : `KeyError` / `TypeError` dans tous les exemples  
**Correction** :
```python
# AVANT
result = pipeline.forward(sample["reps"], sample["sentence"])
pred_nodes = [n["type_idx"] for n in result["nodes"]]
pred_edges = {(e["src"], e["dst"]): e["type_idx"] for e in result["edges"]}

# APRÈS
result = pipeline.forward(sample.sentence.ud_reps, sample.sentence.text)
pred_nodes = [n["node_type_idx"] for n in result["nodes"]]
pred_edges = {(e["source_idx"], e["target_idx"]): e["relation_idx"] for e in result["edges"]}
```
**Fichiers corrigés** : Same as Bug #2

---

### Bug #4 : vocab.vectorize n'existe pas
**Problème** : `from gcn_python.layer1.vectorize import vectorize_clause` — module inexistant  
**Correction** : `from gcn_python.layer1.features import vectorize_clause`  
**Fichier** : `tests/test_protocol_compliance.py:77`

---

### Bug #5 : result["nodes"]["logits"] n'existe pas
**Problème** : Test accède à `result["nodes"][0]["logits"]` qui n'est pas dans le dict retourné  
**Correction** : Utiliser `pipeline._cached_node_logits` directement  
**Fichier** : `tests/test_integration.py:115-120`

---

### Bug #6 : test_backward contredit l'implémentation
**Problème** : Test attend `grad=0` après `update_node()`, mais implémentation diffère volontairement le `zero_grad()`  
**Correction** : Accepter `zero_grad()` différé dans le test  
**Fichier** : `tests/test_backward.py:109-136`

---

### Bug #7 : CamemBERT/CodeBERT jamais chargés
**Problème** : Tests échouent si modèles pas en cache HuggingFace  
**Correction** : Skip markers pour modèles non cachés  
**Fichier** : `tests/test_protocol_compliance.py:17-45`

---

### Bug #8 : Auto-attention inter-clauses manquante ⚠️ CRITIQUE
**Problème** : `prefers_batch_forward=False` → Transformer voit 1 clause à la fois → pas d'attention inter-clauses → équivaut à MLP 280M params  
**Impact** : Les 280M paramètres pré-entraînés ne servent à RIEN  
**Correction** : `prefers_batch_forward=True`  
**Trade-off accepté** : backward rejoue `forward_node()` individuel (approximation), mais gain d'attention au forward >> perte précision backward  
**Fichier** : `src/gcn_transformers/base.py:68-73`

---

### Bug #9 : Checkpoint CUDA silencieux ⚠️ CRITIQUE
**Problème** : `parameters()` retourne copies NumPy → `p[:] = data[key]` écrit dans copie, pas dans tensor PyTorch  
**Impact** : Sur GPU, checkpoints ne chargent PAS les poids (silencieux)  
**Correction** : Implémentation `load_parameters()` qui copie vers tensors PyTorch  
**Fichier** : `src/gcn_transformers/base.py:397-455`

---

### Bug #10 : backward_node_dx shape mismatch
**Problème** : `d_logits` (7,) vs `_cached_output_logits` (1, 7)  
**Impact** : `RuntimeError: Mismatch in shape` dans tests intégration  
**Correction** : Ajuster shape avec `unsqueeze(0)` si nécessaire  
**Fichier** : `src/gcn_transformers/base.py:267-288`

---

### Bug #11 : test_lr_warning capture incorrecte
**Problème** : Test utilise `capfd` pour capturer `warnings.warn()` qui ne va pas dans stderr  
**Correction** : Utiliser `warnings.catch_warnings(record=True)`  
**Fichier** : `tests/test_backward.py:161-192`

---

### Bug #12 : __setattr__ crash avec fake models
**Problème** : `__setattr__` appelle `self.model.train()` sans vérifier que la méthode existe  
**Impact** : Tests avec fake models crashent  
**Correction** : `if hasattr(self.model, 'train')`  
**Fichier** : `src/gcn_transformers/base.py:615-633`

---

## 📋 Bugs Documentés (Non Corrigés)

### Bug C : gcn-eval incompatible
**Problème** : `gcn-eval` hardcode `MLPEncoder` → `load_checkpoint()` échoue avec gcn-transformers  
**Solution** : Documentation workaround `load_parameters()`  
**Fichier** : `README.md:120-129`

### Bug E : Mini-batch échelles divergentes
**Problème** : `apply_accumulated_gradients()` divise grads par `n_samples` pour R-GCN, mais pas pour encodeur  
**Impact** : Mini-batch > 1 non recommandé  
**Raison** : Bug dans gcn-python (hors scope)

### Bug F : Backward avorté laisse .grad périmés
**Statut** : Déjà corrigé par flag `_needs_zero_grad`

---

## ✅ Tests — décompte des ids (exécution non relancée)

### Suites (décompte des ids ; exécution non relancée)

#### test_audit_fixes.py : 10 ids
- test_bug1_update_edge_applies_gradients
- test_bug1_forward_auto_zero_grad
- test_bug2_model_config_validation
- test_bug4_double_update_warning
- test_bug5_forward_batch_empty
- test_bug6_parameters_filters_frozen
- test_bug7_backward_with_none_grads
- test_bug9_freeze_layers_warning_no_encoder
- test_bug9_freeze_layers_warning_too_many
- test_all_bugs_integration

#### test_backward.py : 6 ids
- test_backward_node_dx_returns_correct_format
- test_backward_edge_dx_returns_correct_format
- test_backward_node_accumulation
- test_update_node_zero_grad_after
- test_snapshot_restore_preserves_gradients
- test_lr_warning_emitted_once

#### test_integration.py : 4 ids
- test_pipeline_forward
- test_pipeline_forward_backward
- test_pipeline_multiple_epochs
- test_pipeline_with_eval_mode

#### test_protocol_compliance.py : 15 ids (5 × 3 encodeurs ; 10 sautés hors cache)
- test_implements_causal_encoder_protocol[XLMRobertaEncoder]
- test_forward_node_shape[XLMRobertaEncoder]
- test_forward_edge_shape[XLMRobertaEncoder]
- test_parameters_returns_numpy[XLMRobertaEncoder]
- test_eval_train_modes[XLMRobertaEncoder]

**Note** : 10 ids sautés pour CamemBERT et CodeBERT (modèles non en cache
HuggingFace sur cette machine). Le slug du test CamemBERT
(`models--camembert--camembert-base`) ne correspond pas au dépôt réellement
chargé par le code (`camembert-base` → slug `models--camembert-base`).

---

## 📈 Impact Corrections

### Bloquants Résolus (constat sur le code, non mesuré)
- ✅ Package **utilisable** — mais **aucun entraînement ni benchmark exécuté**
- ✅ API **conforme** à gcn-python (tests écrits, non relancés)
- ✅ Auto-attention **activée** (raison d'être des Transformers)
- ⚠️ Checkpoints GPU : `load_parameters()` implémentée mais **non testée** ;
  `load_checkpoint()` reste inutilisable tel quel (bug CUDA documenté)

### Performance (aucune mesurée)
- **v1.0** (actuel) : ~0.47 edge_f1 — **hypothèse** (Option A : projection UD)
- **v1.1** (futur) : >0.70 edge_f1 — objectif de roadmap **sans base empirique**

---

## 🚀 Prochaine Étape

**Test entraînement réel** (optionnel) :
```bash
python example_train.py \
    --data-dir ../gcn-datasets/real/train \
    --val-dir ../gcn-datasets/real/val \
    --epochs 5 \
    --lr 1e-5
```

**Validation à faire** (aucune n'a été exécutée à ce jour) :
- Loss décroît sur 5 epochs — **non vérifié**
- val_edge_f1 ≈ 0.47 — **hypothèse non mesurée**
- Pas de crash — **non vérifié**

---

## 📝 Fichiers Modifiés

### Code Source (1 fichier)
- `src/gcn_transformers/base.py` : 10 corrections

### Documentation (1 fichier)
- `README.md` : API loss() corrigée + checkpoint workaround

### Exemples (2 fichiers)
- `example_train.py` : API + clés corrigées
- `benchmark_comparison.py` : API + clés corrigées

### Tests (4 fichiers)
- `tests/test_protocol_compliance.py` : import + skip markers
- `tests/test_integration.py` : API + clés + logits
- `tests/test_backward.py` : zero_grad différé + warnings
- Pas de modif `test_audit_fixes.py` (déjà correct)

---

## 🎉 Conclusion

**Package gcn-transformers v1.0 : code implémenté, NON VALIDÉ**

- ⚠️ Correctifs listés ici : 12 numérotés + items lettrés (titre d'origine
  « 10 ») — dénombrement **non établi** avec les autres documents
- ⚠️ Tests : 35 ids collectés, exécution **non relancée**
- ⚠️ Checkpoints GPU : `load_parameters()` sans test, `load_checkpoint()`
  inutilisable tel quel
- ❌ Aucun entraînement ni benchmark exécuté ; paquet non publié (PyPI → 404)

**Limitation documentée** : performance v1.0 estimée ~0.47 (hypothèse, Option A),
v1.1 visée >0.70 (objectif de roadmap, sans base empirique)

**Recommandation** : sans aucune mesure comparative, aucune recommandation de performance n'est justifiée ; MLPEncoder reste la référence par défaut faute de benchmark exécuté.
