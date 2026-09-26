# Guide de Publication — GCN v2.5.0

**Date** : 2026-09-25  
**Version** : 2.5.0  
**Status** : ⚠️ PUBLIÉ PARTIELLEMENT — PyPI 2.5.0 ✓, crates.io 1/9 à 2.5.0 (voir encadré ci-dessous)

---

## 📌 État réel au 2026-09-25

> Vérifié le **2026-09-25** sur les sources primaires : API crates.io, API PyPI,
> `git ls-remote origin`, dépôt local. **Aucune case de ce document n'est cochée
> sans preuve vérifiable** ; les cases fausses sont décochées, la réalité est
> écrite en face.

| Élément | Réalité constatée |
|---|---|
| Workspace `gcn-core` | `version = "2.5.0"` (`gcn-core/Cargo.toml:16`) |
| crates.io — **gcn-ir** | **2.5.0** publié le 25/09/2026 ✓ → **1/9 crate à 2.5.0** |
| crates.io — **8 autres crates** | **2.1.0** publié le 17/09/2026 : gcn-cli, gcn-backend, gcn-frontend-fr, gcn-frontend-en, gcn-frontend-code, gcn-knowledge, gcn-middleend, gcn-verbalizer — **leur 2.5.0 n'existe pas** |
| Licence sur crates.io | Les 8 crates en 2.1.0 déclarent encore **MIT** ; seul gcn-ir 2.5.0 déclare Apache-2.0 |
| `cargo install gcn-cli` | installe **2.1.0** → `gcn --version` affiche **2.1.0**, pas 2.5.0 (à corriger après publication des 8 crates) |
| PyPI — gcn-python | **2.5.0** publié le 25/09/2026 08:35 ✓ |
| Badges README | `gcn-python/README.md:4` = 2.5.0 ✓ ; **`README.md` racine `:8` = 2.4.1 → à corriger** (fichier racine hors périmètre de correction) |
| Git tag / Release | tag `v2.5.0` poussé sur `origin` ✓ et release GitHub ✓ — mais il pointe `6755ccf` (08:23), **antérieur** au bump des dépendances internes (`faf488e`, 08:41) |

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

> ✅ **Déjà fait** : `gcn-python` **2.5.0** est sur PyPI depuis le 2026-09-25 08:35
> (wheel 156 903 o + sdist). Ne pas re-upload sans nouvelle version.

### Vérifications Post-Publication

```bash
# Install depuis PyPI
pip install gcn-python==2.5.0

# Tests smoke
python -c "import gcn_python; print(gcn_python.__version__)"  # 2.5.0
gcn-discuss --help     # ⚠️ gcn-discuss n'a PAS d'option --version (aucune déclaration
                       # version_option dans gcn-python/src) : vérifier via Python ci-dessus
pytest --pyargs gcn_python  # ⚠️ les tests ne sont PAS embarqués dans le wheel
                            # (MANIFEST.in : « prune tests ») — cette commande ne
                            # collectera rien depuis un paquet installé
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

> ⚠️ **Partiellement fait** : au 2026-09-25, seul **gcn-ir 2.5.0** est publié
> (25/09). Les 8 autres crates sont toujours en **2.1.0** (17/09) : la séquence
> `cargo publish -p …` ci-dessus reste **à exécuter pour les 8 crates**.

### Vérifications Post-Publication

```bash
# Install binaire depuis crates.io
cargo install gcn-cli
gcn --version  # Réel au 2026-09-25 : 2.1.0 (2.5.0 indisponible tant que
               # les 8 crates ne sont pas publiés — voir encadré « État réel »)

# Test fonctionnel
gcn analyze "La pluie cause l'inondation." --data-dir gcn-references/taxonomies/
# (gcn analyze prend un argument positionnel, pas un stdin ; il n'a pas d'option --lang)
```

---

## 🏷️ Git Tags

```bash
# Tag déjà créé (local + origin, release GitHub existante)
git tag v2.5.0   # ⚠️ déjà présent : « fatal: tag 'v2.5.0' already exists »
                 # le tag courant pointe 6755ccf, AVANT le bump des dépendances
                 # internes (faf488e) → à retagger si on veut un tag à jour

# Push (si pas encore fait)
git push origin master
git push origin v2.5.0

# Vérifier sur GitHub
# https://github.com/devmail0561-web/gcn_engine/releases → release v2.5.0 existante ✓
# Release notes à vérifier/actualiser depuis CHANGELOG.md section v2.5.0
```

---

## ✅ Checklist Pré-Publication

### Python (gcn-python)

- [x] Version 2.5.0 dans pyproject.toml (preuve : `gcn-python/pyproject.toml:7`)
- [x] Version 2.5.0 dans __init__.py (preuve : `gcn-python/src/gcn_python/__init__.py:4`)
- [ ] Version 2.5.0 dans README badges (réel : `gcn-python/README.md:4` = 2.5.0 ✓, mais badge du `README.md` racine `:8` = **2.4.1** — à corriger, fichier hors périmètre)
- [x] Badge licence Apache 2.0 (pas MIT) (preuve : `gcn-python/README.md:6` et `:390` ; anomalie résiduelle : la description PyPI 2.5.0 publiée se termine encore par « Licence MIT »)
- [x] Exemples README avec trusted=True (preuve : 3 occurrences dans `gcn-python/README.md`)
- [x] Dépendance fasttext-wheel déclarée (preuve : metadata PyPI, extras `all`/`embeddings`)
- [x] Upper bounds dépendances (click<9.0, pyyaml<7.0) (preuve : `requires_dist` PyPI : `click<9.0,>=8.1`, `pyyaml<7.0,>=6.0`)
- [x] Metadata authors + keywords (preuve : `info.author` = Michel Tendeng, `keywords` renseignés sur PyPI)
- [x] Tests : 412 passed, 4 skipped (preuve : `AUDIT-ET-PREPARATION-v2.5.0.md:309` ; incohérence interne : `:627` indique 409 passed)
- [x] Build wheel : 154 KB clean (preuve : `gcn_python-2.5.0-py3-none-any.whl` = 156 903 o, publié 25/09 08:35)
- [x] .gitignore + MANIFEST.in : fichiers dev exclus (preuve : `gcn-python/MANIFEST.in` — prune tests, exclude *.npz/*.csv)

### Rust (gcn-core)

- [x] Workspace version 2.5.0 (preuve : `gcn-core/Cargo.toml:16`)
- [x] Dépendances internes 2.5.0 (8 Cargo.toml) (preuve : `version = "2.5.0"` dans les 8 Cargo.toml ayant des dépendances internes ; gcn-ir n'en a pas) — **non publié** : crates.io en est encore à 2.1.0 pour ces 8 crates
- [x] cargo build --release : ✅ (preuve : `gcn-core/target/release/gcn`, build du 25/09 12:19)
- [x] cargo test --workspace : ✅ (144 tests) (preuve : `AUDIT-ET-PREPARATION-v2.5.0.md:631`)
- [x] .gitignore : target/, fixtures exclus (preuve : `gcn-core/.gitignore`)
- [x] cargo package --list : propre (aucun temporaire) (preuve : `AUDIT-ET-PREPARATION-v2.5.0.md:488-494`)

### Git

- [ ] Commits : 3 commits pré-publication (réel : **5** commits le 2026-09-25 — 6755ccf, 95e7e64, faf488e, 0f9ff39, 4ec3d39 ; le décompte « 3 » n'est pas vérifiable)
- [x] Tag v2.5.0 créé (preuve : `git tag` local + `refs/tags/v2.5.0` sur origin + release GitHub ; réserve : le tag pointe 6755ccf, avant le bump des dépendances internes faf488e)
- [x] TODO-v2.5.1.md : correctifs post-publication documentés (archivé dans `ancien/TODO-v2.5.1.md`)

---

## 📝 Post-Publication

### Annonce

**PyPI** : https://pypi.org/project/gcn-python/2.5.0/ — ✅ réel (publié 2026-09-25 08:35)

**crates.io** — état réel au 2026-09-25 (aucune URL 2.5.0 n'existe pour gcn-cli) :

- gcn-ir 2.5.0 : https://crates.io/crates/gcn-ir/2.5.0 — ✅ réel
- gcn-cli (et les 7 autres) : https://crates.io/crates/gcn-cli/2.1.0 — dernière version réellement publiée
- https://crates.io/crates/gcn-cli/2.5.0 — ❌ **inexistant** : ne deviendra valide qu'après `cargo publish -p gcn-cli` (voir ordre de publication plus haut)

**GitHub Release** : https://github.com/devmail0561-web/gcn_engine/releases/tag/v2.5.0 — ✅ réel (tag pointant 6755ccf)

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

Voir `ancien/TODO-v2.5.1.md` pour correctifs haute priorité :
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
# ⚠️ valable seulement APRÈS publication de gcn-cli 2.5.0 (inexistant au 2026-09-25)
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
**Révision vérité** : 2026-09-25 — réserves ouvertes : 8/9 crates encore à 2.1.0 sur
crates.io, badge `README.md` racine à 2.4.1, tag `v2.5.0` antérieur au bump des
dépendances internes, `gcn forward` retirée de la CLI.
