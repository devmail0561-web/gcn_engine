# TODO v2.5.1 — Correctifs Post-Publication

Suite à l'audit pré-publication v2.5.0, ces corrections sont recommandées pour v2.5.1.

---

## H-1 : Exceptions silencieuses (HAUTE PRIORITÉ)

**Fichiers concernés** :
- `gcn-python/src/gcn_python/discuss.py:91`
- `gcn-python/src/gcn_python/taxonomy/loader.py:43`
- `gcn-python/src/gcn_python/cli/session.py:82-83`
- `gcn-python/src/gcn_python/data/graph_vecs.py:39,77,99`

**Problème** : Blocs `except Exception: pass` masquent des erreurs légitimes

**Action** :
```python
# Remplacer tous les
except Exception:
    pass

# Par
except Exception as exc:
    warnings.warn(f"Opération échouée: {exc}", UserWarning, stacklevel=2)
```

**Impact** : Améliore debuggabilité, évite pertes de données silencieuses

---

## H-2 : Validation gradient shapes (HAUTE PRIORITÉ)

**Fichier** : `gcn-python/src/gcn_python/pipeline/cgnp.py:1175-1178`

**Problème** : Accumulation gradient silencieusement ignorée si shapes incompatibles

**Action** :
```python
if (_d_node_embs is not None
        and d_enriched is not None
        and _d_node_embs.shape == d_enriched.shape):
    d_enriched += _d_node_embs
else:
    if _d_node_embs is not None and d_enriched is not None:
        warnings.warn(
            f"Shape mismatch gradient décodeur: {_d_node_embs.shape} vs {d_enriched.shape}",
            UserWarning
        )
```

**Impact** : Détecte erreurs convergence ML

---

## H-3 : Tests modules critiques (HAUTE PRIORITÉ)

**Modules sans tests dédiés** :
- `gcn-python/src/gcn_python/data/schema.py`
- `gcn-python/src/gcn_python/verbalizer/instructions.py`

**Action** : Créer fichiers de test dédiés
- `gcn-python/tests/test_schema.py` — validation JSON SentenceRecord, EdgeRecord
- `gcn-python/tests/test_instructions.py` — génération texte Q&A

**Impact** : Évite bugs silencieux validation données + génération texte

---

## M-1 : Unifier epsilon numérique (MOYENNE PRIORITÉ)

**Fichiers** :
- `gcn-python/src/gcn_python/layer3/gat.py:51` — `1e-9`
- `gcn-python/src/gcn_python/pipeline/cgnp.py:1305` — `1e-9`

**Action** : Définir constante
```python
# gcn-python/src/gcn_python/constants.py
EPSILON = 1e-9  # Stabilité numérique (anti-division par zéro, log(0))
```

Remplacer toutes les magic numbers par `EPSILON`.

---

## M-2 : Documenter FIFO cache limitations (MOYENNE PRIORITÉ)

**Fichier** : `gcn-python/src/gcn_python/data/graph_vecs.py:68-80`

**Action** : Ajouter docstring
```python
# Cache FIFO de 8 handles NpzFile max. Si >8 fichiers actifs simultanément,
# les plus anciens sont fermés (rechargement si réaccès).
# Limitation acceptable pour usage typique (≤5 checkpoints actifs).
```

---

## M-3 : Section migration CHANGELOG (MOYENNE PRIORITÉ)

**Fichier** : `CHANGELOG.md`

**Action** : Ajouter section pour v2.5.0
```markdown
### Migration depuis 2.4.x

Aucune action requise — tous les nouveaux paramètres ont des valeurs par défaut
backward-compatibles :
- `--fasttext` : opt-in (défaut None)
- `--pairnorm`, `--compgcn` : opt-in (défaut False)
- `--global-attention` : opt-in (défaut False)

Les checkpoints 2.4.x restent compatibles via `GCNEngine.from_pretrained()`.
```

---

## Vérification finale v2.5.1

Après correctifs :
- [ ] `pytest tests/` → 412+ passed
- [ ] `ruff check gcn-python/src/` → 0 warnings
- [ ] Bump version 2.5.1 dans pyproject.toml, __init__.py, badges
- [ ] Git tag v2.5.1
- [ ] Publication PyPI `twine upload dist/*`
