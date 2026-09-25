# Changelog

All notable changes to gcn-transformers will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-25

### Added
- **TransformerEncoderBase** : Classe abstraite pour encodeurs Transformer (633 lignes)
  - Support PyTorch ↔ NumPy avec autograd
  - backward_node_dx et backward_edge_dx via torch.autograd
  - Snapshots avec graphe autograd préservé
  - Optimizer AdamW optimisé (un seul step pour node+edge)
  - Gestion eval()/train() pour dropout

- **XLMRobertaEncoder** : Encodeur multilingue XLM-RoBERTa-base
  - 280M paramètres, 100 langues (FR/EN/Code)
  - Freeze layers configurable (défaut : 10/12)
  - Learning rate AdamW 1e-5 par défaut

- **CamembertEncoder** : Encodeur français CamemBERT-base
  - 110M paramètres, corpus français OSCAR
  - Optimisé pour français uniquement

- **CodeBERTEncoder** : Encodeur code CodeBERT-base
  - 125M paramètres, 6 langages (Python/Java/JS/...)
  - Optimisé pour docstrings et commentaires techniques

- **Tests complets** (786 lignes hors `conftest.py`, 60 lignes)
  - test_protocol_compliance.py : Protocol CausalEncoder
  - test_integration.py : CGNPipeline complet
  - test_backward.py : Validation gradients avec torch.autograd.gradcheck
  - test_audit_fixes.py : Tests corrections audit (9 bugs)

- **Documentation**
  - README.md : Guide utilisateur complet (366 lignes)
  - IMPLEMENTATION.md : Documentation technique
  - AUDIT-SUMMARY.md : Résumé audit --level max
  - AUDIT-CORRECTIONS.md : Détails corrections bugs
  - example_train.py : Script d'entraînement complet (253 lignes)

### Fixed

#### Corrections Plan Original (16 items listés)

> **Dénombrement non établi** : cette liste (16) chevauche la liste « audit » (9)
> et la liste « post-audit » (12) de `CORRECTIONS-POST-AUDIT.md` — doublons
> vérifiables (warning `lr`, extraction de gradients `None`, validation
> `model.config.hidden_size`). Les totaux qui circulent (25 / 12 / 10 / 9) sont
> donc contradictoires ; il n'existe pas de liste unique.
- Dimensions correctes : d_clause=79, d_edge=365 (pas 80/100)
- snapshot_edge_cache() + restore_edge_cache() implémentés
- Ordre init : self.model AVANT super().__init__()
- Edge head : nn.Linear(d_edge=365) au lieu de 1536
- RGCNLayer d_out=79 (pas 768)
- API backward : loss() puis backward(d_node, d_edge, lr)
- Pas de torch.no_grad() en training (graphe requis)
- zero_grad() dans update_node, PAS backward_node_dx
- Optimizer AdamW unique (update_edge no-op)
- proj_ud persistant (nn.Module), pas recréé
- backward_edge_dx implémenté (pas pass)
- model.eval() désactive dropout Transformer
- n_node_types et n_relation_types ajoutés
- Warning lr émis 1× seulement

#### Corrections Audit --level max (9 bugs)

**Critiques (Production Bloquants)**
1. **update_edge no-op perdait gradients edge** (⚠️ CRITIQUE)
   - Problème : update_edge() = pass → edge classifier ne s'entraîne pas
   - Solution : Flag _needs_zero_grad + step() dans update_node, zero_grad() dans update_edge
   - Impact : Edge classifier fonctionnel

2. **Double optimizer.step() possible** (⚠️ CRITIQUE)
   - Problème : update_node() + update() → double step → divergence
   - Solution : Warning dans update() si _needs_zero_grad=True
   - Impact : Évite divergence training

3. **Crash avec batch vide (N=0)** (⚠️ CRITIQUE)
   - Problème : forward_batch([]) → RuntimeError Transformer
   - Solution : Check X.shape[0]==0, retourne array vide
   - Impact : Gère graphes vides

**Robustesse**
4. **model.config.hidden_size non validé**
   - Solution : Validation explicite avec ValueError clair

5. **parameters() incluait frozen params inconsistents**
   - Solution : Filtrage requires_grad cohérent partout

6. **Extraction gradients silencieuse si None**
   - Solution : Gérer weight.grad et bias.grad séparément, retourner zeros si None

7. **Freeze layers silencieux si structure différente**
   - Solution : Warnings si model sans encoder.layer ou freeze_layers > n_layers

**UX**
8. **lr paramètre ignoré (trompeur)**
   - Status : Déjà documenté par warning existant (acceptable pour AdamW)

### Known Limitations

**v1.0 : Option A (Projection UD)**
- Transformers projettent features syntaxiques UD → hidden_size
- **Pas de tokenization texte brut** → embeddings pré-entraînés inutilisés
- Performance **hypothétique** : ~0.47 val_edge_f1 — hypothèse non mesurée (aucun run)
- Les 280M paramètres pré-entraînés ne s'appliquent PAS dans cette version

**Raison** : UDRepresentation ne contient pas `raw_text` (gcn-python 2.5.0)

**Roadmap v1.1** :
- Ajouter `raw_text: str | None` à UDRepresentation (gcn-python 2.6.0)
- Implémenter Option B : tokenization texte → embeddings pré-entraînés
- Objectif : **val_edge_f1 > 0.70** (hypothèse de roadmap, sans base empirique)

### Notes

**Approche** : Library uniquement (Option 3)
- Pas de CLI intégré (utilisateur écrit son propre script)
- Pas de modification gcn-python requise
- Import direct : `from gcn_transformers import XLMRobertaEncoder`

**Tests** (état vérifiable sans les relancer) :
- 35 ids collectés ; exécution non relancée (~25 exécutés / 10 sautés attendus
  selon le cache HuggingFace de la machine)
- Conformité Protocol CausalEncoder : testée pour les 3 encodeurs, mais
  CamemBERT/CodeBERT sont sautés hors cache
- Aucune mesure de couverture n'existe

**Performance (aucune non mesurée)** :
- v1.0 (Option A) : ~0.47 val_edge_f1 — hypothèse
- v1.1 (Option B) : >0.70 val_edge_f1 — objectif de roadmap

## [Unreleased]

### Planned for v1.1.0

**Option B : Tokenization Texte**
- [ ] Modifier forward_batch pour accepter texts: list[str]
- [ ] Tokenization réelle : self.tokenizer(texts, ...)
- [ ] Utiliser embeddings [CLS] au lieu de projection UD
- [ ] Dépend de : gcn-python 2.6.0 (raw_text dans UDRepresentation)

**Optimisations**
- [ ] Batch size dynamique pour Transformers
- [ ] Mixed precision training (torch.amp)
- [ ] Gradient accumulation pour GPU limité

**Documentation**
- [ ] Tutoriels Jupyter notebooks
- [ ] Benchmarks complets vs MLPEncoder
- [ ] Guide fine-tuning avancé

### Planned for v2.0.0

**Factory gcn-python**
- [ ] PR gcn-python : encoder factory (~40 lignes)
- [ ] Option --encoder-class dans gcn-train
- [ ] Entry-point setuptools pour plugins
- [ ] CLI intégré gcn-train compatible

**Nouveaux Encodeurs**
- [ ] FlaubertEncoder (français, GPT-2 architecture)
- [ ] DeBERTa-v3 (meilleure performance)
- [ ] mBERT (multilingue, plus léger)

---

## Contribution Guidelines

**Bugs** : https://github.com/devmail0561-web/gcn_engine/issues  
**Pull Requests** : https://github.com/devmail0561-web/gcn_engine/pulls

**Tests requis** :
```bash
pytest tests/ -v           # Tous les tests
pytest tests/test_backward.py -v  # Gradients critiques
python example_train.py --data-dir ../gcn-datasets/real/train --epochs 5  # test
```

---

## Credits

**Author** : Michel Tendeng  
**License** : Apache 2.0  
**Based on** : gcn-python 2.5.0, transformers 4.30+, PyTorch 2.0+

---

## Version History

- **1.0.0** (2026-09-25) : version du code (**non publiée** : absent de PyPI,
  build jamais fait, aucun tag)
  - 3 encodeurs (XLM-R, CamemBERT, CodeBERT)
  - Option A (projection UD)
  - Correctifs listés : 16 plan + 9 audit + 12 post-audit (dénombrement
    contradictoire, total unique non établi)
  - 786 lignes de tests (hors `conftest.py`)
  - Documentation (14 fichiers `.md`)
