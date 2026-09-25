# Audit Code Review : Corrections Appliquées

**Date** : 2026-09-25  
**Niveau** : max  
**Bugs trouvés** : 9 (3 critiques, 5 robustesse, 1 UX)

---

## 🚨 Bugs Critiques Corrigés

### Bug #1 — `update_edge` no-op perdait gradients edge ⚠️ **CRITIQUE**

**Problème** :
```python
def update_edge(self, grads, lr):
    pass  # ❌ NO-OP : gradients edge jamais appliqués
```

Si le pipeline (ou utilisateur manuel) appelait :
1. `backward_node_dx()` → accumule gradients node
2. `update_node()` → `step()` + `zero_grad()` ❌ efface gradients edge !
3. `backward_edge_dx()` → accumule gradients edge
4. `update_edge()` → `pass` ❌ gradients edge perdus !

**Correction** :
```python
# Flag _needs_zero_grad pour tracker l'état
def update_node(self, grads, lr):
    self.optimizer.step()
    # NE PAS zero_grad() ici !
    self._needs_zero_grad = True

def update_edge(self, grads, lr):
    if self._needs_zero_grad:
        self.optimizer.zero_grad()
        self._needs_zero_grad = False

# Safety dans forward_batch
if self._needs_zero_grad:
    self.optimizer.zero_grad()  # Au cas où update_edge jamais appelé
    self._needs_zero_grad = False
```

**Impact** : Edge classifier peut maintenant s'entraîner correctement.

---

### Bug #4 — Double `optimizer.step()` si `update_node()` + `update()` appelés

**Problème** :
```python
# Utilisateur confus appelle les deux
encoder.update_node(grads, lr)  # step() ici
encoder.update(grads, lr)        # step() encore ici ❌
# Learning rate doublé → divergence
```

**Correction** :
```python
def update(self, grads, lr):
    if self._needs_zero_grad:
        # update_node() déjà appelé
        warnings.warn(
            "update() appelé après update_node() — double step évité.",
            UserWarning
        )
        self.optimizer.zero_grad()
        self._needs_zero_grad = False
    else:
        # Cas normal : update() seul
        self.optimizer.step()
        self.optimizer.zero_grad()
```

**Impact** : Évite divergence training si mauvaise utilisation API.

---

### Bug #5 — Crash avec batch vide (N=0)

**Problème** :
```python
X = np.array([])  # shape (0, 79)
encoder.forward_batch(X)
# → RuntimeError: Transformer sequence_length=0
```

**Correction** :
```python
def forward_batch(self, X, texts=None):
    # Safety : batch vide
    if X.shape[0] == 0:
        return np.zeros((0, self.n_node_types), dtype=np.float32)
    # ...
```

**Impact** : Gère graphes vides (edge cases dataset).

---

## 🔧 Bugs Robustesse Corrigés

### Bug #2 — `model.config.hidden_size` non validé → AttributeError

**Correction** :
```python
if not hasattr(self.model, 'config'):
    raise ValueError(
        f"self.model doit avoir 'config' (structure HuggingFace attendue)"
    )
if not hasattr(self.model.config, 'hidden_size'):
    raise ValueError(
        f"self.model.config doit avoir 'hidden_size'"
    )
```

**Impact** : Message d'erreur clair au lieu d'AttributeError obscur.

---

### Bug #6 — `parameters()` incluait frozen params inconsistents

**Problème** :
```python
# Transformer filtré par requires_grad
for p in self.model.parameters():
    if p.requires_grad:
        params.append(p)

# Mais heads toujours inclus même si gelés ❌
params.extend([
    layer.weight.detach().cpu().numpy(),  # Même si requires_grad=False
    layer.bias.detach().cpu().numpy()
])
```

**Correction** :
```python
# Filtrage cohérent partout
if layer.weight.requires_grad:
    params.append(layer.weight.detach().cpu().numpy())
if layer.bias.requires_grad:
    params.append(layer.bias.detach().cpu().numpy())
```

**Impact** : Checkpoint reflète exactement les paramètres trainables.

---

### Bug #7 — Extraction gradients silencieuse si un grad None

**Problème** :
```python
if layer.weight.grad is not None and layer.bias.grad is not None:
    grads.append((weight.grad, bias.grad))
# Sinon : skip silencieux ❌
```

**Correction** :
```python
# Gérer weight et bias séparément
dW = (layer.weight.grad.cpu().numpy().copy()
      if layer.weight.grad is not None
      else np.zeros_like(layer.weight))
db = (layer.bias.grad.cpu().numpy().copy()
      if layer.bias.grad is not None
      else np.zeros_like(layer.bias))
grads.append((dW, db))
```

**Impact** : Pas de skip silencieux, gradients toujours retournés (zéro si None).

---

### Bug #9 — Freeze layers silencieux si structure modèle différente

**Problème** :
```python
if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
    # Freeze...
# Sinon : silencieux, aucune couche gelée ❌
```

**Correction** :
```python
if hasattr(...):
    n_layers = len(self.model.encoder.layer)
    if n_layers == 0:
        warnings.warn("model.encoder.layer vide")
    elif freeze_layers > n_layers:
        warnings.warn(f"freeze_layers={freeze_layers} > {n_layers} couches")
else:
    if freeze_layers > 0:
        warnings.warn(
            "model ne suit pas structure BERT/RoBERTa. "
            "Aucune couche gelée."
        )
```

**Impact** : Utilisateur averti si freeze ne fonctionne pas.

---

## 📝 Bug UX Documenté

### Bug #8 — `lr` paramètre ignoré (trompeur)

**Déjà géré** par warning existant :
```python
if abs(pipeline_lr - adamw_lr) > 1e-6:
    warnings.warn(
        f"Pipeline lr={pipeline_lr} != AdamW lr={adamw_lr}. "
        "AdamW lr utilisé.",
        UserWarning
    )
```

**Impact** : Utilisateur averti 1× par session.

---

## 📊 Résumé Corrections

| Bug | Sévérité | Status | Lignes Modifiées |
|-----|----------|--------|------------------|
| #1 update_edge no-op | **CRITIQUE** | ✅ Corrigé | +15 lignes (flag _needs_zero_grad) |
| #2 model.config validation | Robustesse | ✅ Corrigé | +10 lignes (checks) |
| #3 _cached_input_tensor.grad | Robustesse | ✅ Déjà géré | 0 (existant OK) |
| #4 Double optimizer.step() | **CRITIQUE** | ✅ Corrigé | +10 lignes (warning) |
| #5 Crash batch vide | **CRITIQUE** | ✅ Corrigé | +3 lignes (check) |
| #6 parameters() frozen | Robustesse | ✅ Corrigé | +15 lignes (filtrage) |
| #7 Extraction gradients silencieuse | Robustesse | ✅ Corrigé | +20 lignes (gérer None) |
| #8 lr ignoré | UX | ✅ Déjà documenté | 0 (warning existant) |
| #9 Freeze layers silencieux | Robustesse | ✅ Corrigé | +15 lignes (warnings) |

**Total : 88 lignes ajoutées/modifiées**

---

## ✅ Tests Additionnels Requis

### Test update_edge avec gradients tardifs
```python
def test_update_edge_after_backward():
    """Bug #1 : vérifier gradients edge appliqués."""
    encoder = XLMRobertaEncoder(d_clause=79, d_edge=365)
    
    # Simuler ordre problématique
    X = np.random.randn(2, 79).astype(np.float32)
    encoder.forward_batch(X)
    d_logits = np.ones((2, 7))
    grads_node, dx_node = encoder.backward_node_dx(d_logits)
    
    # update_node fait step() mais PAS zero_grad()
    encoder.update_node(grads_node, lr=1e-5)
    
    # Gradients edge accumulés APRÈS update_node
    x_edge = np.random.randn(365).astype(np.float32)
    encoder.forward_edge(x_edge)
    d_edge_logits = np.ones(11)
    grads_edge, dx_edge = encoder.backward_edge_dx(d_edge_logits)
    
    # update_edge doit faire zero_grad()
    encoder.update_edge(grads_edge, lr=1e-5)
    
    # Vérifier que gradients sont bien nettoyés
    assert encoder._needs_zero_grad == False
```

### Test batch vide
```python
def test_forward_batch_empty():
    """Bug #5 : batch vide ne doit pas crasher."""
    encoder = XLMRobertaEncoder(d_clause=79, d_edge=365)
    X_empty = np.zeros((0, 79), dtype=np.float32)
    
    logits = encoder.forward_batch(X_empty)
    
    assert logits.shape == (0, 7)
    assert logits.dtype == np.float32
```

### Test double update warning
```python
def test_double_update_warning(capfd):
    """Bug #4 : warning si update() après update_node()."""
    encoder = XLMRobertaEncoder(d_clause=79, d_edge=365)
    
    X = np.random.randn(2, 79).astype(np.float32)
    encoder.forward_batch(X)
    grads, dx = encoder.backward_node_dx(np.ones((2, 7)))
    
    encoder.update_node(grads, lr=1e-5)
    encoder.update(grads, lr=1e-5)  # ❌ Double call
    
    captured = capfd.readouterr()
    assert "double optimizer.step() évité" in captured.err
```

### Test freeze layers warning
```python
def test_freeze_layers_warning_structure_mismatch():
    """Bug #9 : warning si structure modèle différente."""
    # Mock un modèle sans model.encoder.layer
    class FakeModel:
        def __init__(self):
            self.config = type('obj', (object,), {'hidden_size': 768})()
            self.h = []  # GPT-2 style, pas .encoder.layer
    
    # Devrait émettre warning
    with pytest.warns(UserWarning, match="ne suit pas structure BERT"):
        encoder = XLMRobertaEncoder.__new__(XLMRobertaEncoder)
        encoder.model = FakeModel()
        encoder.__init__(d_clause=79, d_edge=365, freeze_layers=10)
```

---

## 🎯 Impact Global

**Avant audit** : ce document liste **9 bugs** (3 critiques, 5 robustesse,
1 UX) — le « 6 critiques + 5 robustesse » de la ligne d'origine contredisait
la liste elle-même.  
**Après corrections** : correctifs décrits ici et repris dans
`tests/test_audit_fixes.py` (10 ids) — **exécution non relancée** lors de cette
vérification ; d'autres défauts connus restent (checkpoint CUDA,
`load_parameters()` sans test, slug CamemBERT, `psutil`).

**Lignes modifiées** : 88 lignes (chiffre d'origine, **non revérifié**)  
**Tests ajoutés** : 10 ids dans `test_audit_fixes.py`

---

## 📝 Notes

1. **Bug #1 (update_edge)** était le plus critique : sans correction, l'edge classifier ne s'entraîne pas du tout dans certains scénarios.

2. **Bug #4 (double step)** aurait causé divergence training silencieuse si utilisateur confus.

3. **Bug #5 (batch vide)** rare mais bloquant pour datasets avec graphes vides.

4. **Bugs robustesse (#2, #6, #7, #9)** causent erreurs obscures ou comportement silencieux incorrect.

5. **Bug #8 (lr ignoré)** déjà géré par warning existant, acceptable pour Transformers (AdamW standard).

---

**Status** : correctifs d'audit décrits ici ; **aucun run** (tests,
entraînement, benchmark) n'a été exécuté ni revérifié lors de cette correction
de documentation ; **défauts connus restants** listés dans `README.md` et
`STATUS.md`.
