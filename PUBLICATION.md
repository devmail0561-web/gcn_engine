# Guide de Publication — GCN v2.5.0

**Date** : 2026-09-25  
**Version** : 2.5.0  
**Status** : ✅ PRÊT

---

## 📦 Python Package (PyPI)

### Pré-requis

```bash
pip install build twine
```

### Publication

```bash
cd gcn-python/

# 1. Build
python -m build
# Génère : dist/gcn_python-2.5.0-py3-none-any.whl (154 KB)
#          dist/gcn_python-2.5.0.tar.gz (21 MB)

# 2. Vérification locale (recommandé)
pip install dist/gcn_python-2.5.0-py3-none-any.whl
gcn-discuss --help
gcn-train --help
python -c "from gcn_python import GCNEngine; print(GCNEngine.__doc__)"

# 3. TestPyPI (fortement recommandé)
twine upload --repository testpypi dist/*
# URL: https://test.pypi.org/project/gcn-python/
# Test install : pip install --index-url https://test.pypi.org/simple/ gcn-python

# 4. Production PyPI
twine upload dist/gcn_python-2.5.0*
# URL finale : https://pypi.org/project/gcn-python/
```

### Vérifications Post-Publication

```bash
# Install depuis PyPI
pip install gcn-python==2.5.0

# Tests smoke
gcn-discuss --version  # Devrait afficher 2.5.0
python -c "import gcn_python; print(gcn_python.__version__)"  # 2.5.0
pytest --pyargs gcn_python  # Si tests inclus (non recommandé pour PyPI)
```

---

## 🦀 Rust Crates (crates.io)

### Pré-requis

```bash
# Authentification crates.io (une fois)
cargo login <your-token>
```

### Ordre de Publication (dépendances)

Les crates doivent être publiés dans l'ordre des dépendances :

```
1. gcn-ir           (aucune dépendance interne)
2. gcn-knowledge    (dépend de gcn-ir)
3. gcn-verbalizer   (dépend de gcn-ir)
4. gcn-frontend-fr  (dépend de gcn-ir)
5. gcn-frontend-en  (dépend de gcn-ir)
6. gcn-frontend-code (dépend de gcn-ir)
7. gcn-middleend    (dépend de gcn-ir)
8. gcn-backend      (dépend de gcn-ir, gcn-knowledge)
9. gcn-cli          (dépend de tous)
```

### Publication

```bash
cd gcn-core/

# 1. Vérification build
cargo build --workspace --release
cargo test --workspace

# 2. Publication séquentielle
cargo publish -p gcn-ir
# Attendre ~1 minute pour indexation crates.io

cargo publish -p gcn-knowledge
cargo publish -p gcn-verbalizer
cargo publish -p gcn-frontend-fr
cargo publish -p gcn-frontend-en
cargo publish -p gcn-frontend-code
cargo publish -p gcn-middleend
# Attendre ~1 minute

cargo publish -p gcn-backend
# Attendre ~1 minute

cargo publish -p gcn-cli
```

### Vérifications Post-Publication

```bash
# Install binaire depuis crates.io
cargo install gcn-cli
gcn --version  # Devrait afficher 2.5.0

# Test fonctionnel
echo "La pluie cause l'inondation." | gcn analyze --lang fr
```

---

## 🏷️ Git Tags

```bash
# Tag déjà créé
git tag v2.5.0

# Push (si pas encore fait)
git push origin master
git push origin v2.5.0

# Vérifier sur GitHub
# https://github.com/devmail0561-web/gcn_engine/releases
# Créer release notes depuis CHANGELOG.md section v2.5.0
```

---

## ✅ Checklist Pré-Publication

### Python (gcn-python)

- [x] Version 2.5.0 dans pyproject.toml
- [x] Version 2.5.0 dans __init__.py
- [x] Version 2.5.0 dans README badges
- [x] Badge licence Apache 2.0 (pas MIT)
- [x] Exemples README avec trusted=True
- [x] Dépendance fasttext-wheel déclarée
- [x] Upper bounds dépendances (click<9.0, pyyaml<7.0)
- [x] Metadata authors + keywords
- [x] Tests : 412 passed, 4 skipped
- [x] Build wheel : 154 KB clean
- [x] .gitignore + MANIFEST.in : fichiers dev exclus

### Rust (gcn-core)

- [x] Workspace version 2.5.0
- [x] Dépendances internes 2.5.0 (8 Cargo.toml)
- [x] cargo build --release : ✅
- [x] cargo test --workspace : ✅ (144 tests)
- [x] .gitignore : target/, fixtures exclus
- [x] cargo package --list : propre (aucun temporaire)

### Git

- [x] Commits : 3 commits pré-publication
- [x] Tag v2.5.0 créé
- [x] TODO-v2.5.1.md : correctifs post-publication documentés

---

## 📝 Post-Publication

### Annonce

**PyPI** : https://pypi.org/project/gcn-python/2.5.0/  
**crates.io** : https://crates.io/crates/gcn-cli/2.5.0  
**GitHub Release** : https://github.com/devmail0561-web/gcn_engine/releases/tag/v2.5.0

### Documentation

Mettre à jour :
- README.md racine : badge version 2.5.0
- Installation instructions testées
- Exemples fonctionnels vérifiés

### Monitoring

Surveiller pendant 48h :
- Issues PyPI/crates.io
- Downloads analytics
- Bug reports utilisateurs

### Patch v2.5.1 (si nécessaire)

Voir `TODO-v2.5.1.md` pour correctifs haute priorité :
- H-1 : Exceptions silencieuses → warnings
- H-2 : Validation shapes gradients
- H-3 : Tests schema.py + instructions.py

---

## 🚨 Rollback (en cas de problème critique)

### PyPI

```bash
# Impossible de supprimer — publier v2.5.1 avec fix
# Ou yank la version (masque mais garde fichiers)
twine upload --repository pypi --skip-existing dist/*
```

### crates.io

```bash
# Yank version (masque sans supprimer)
cargo yank --vers 2.5.0 gcn-cli
```

### Git

```bash
# Supprimer tag local
git tag -d v2.5.0

# Supprimer tag remote
git push origin :refs/tags/v2.5.0
```

---

## 📊 Métriques Succès

- [ ] PyPI : 100+ downloads première semaine
- [ ] crates.io : 50+ downloads première semaine
- [ ] GitHub : 0 issues critiques première semaine
- [ ] Tests : 412 tests passent sur CI
- [ ] Documentation : 0 liens cassés

---

**Préparé par** : Audit exhaustif pré-publication 2026-09-25  
**Confiance** : 95% — Tous bloquants corrigés, tests passent
