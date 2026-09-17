# Plan — Production Readiness GCN Engine

**Date :** 2026-09-17  
**Objectif :** Passer de l'état actuel (infrastructure prête, modèle non entraîné sur données réelles)
à un moteur déployable produisant des CIR fiables sur du texte naturel non vu.

---

## État de départ (vérifié)

```
Dataset réel       : 678 phrases, tokens UD = 0, causal_pattern absent
                     Relations : cause(402) enable(167) condition(56) concession(40) prevent(13)
                     6 relations absentes : motivation, sequence, filter, opposition, data_*, control_*

Checkpoints        : 5 existants — tous entraînés sur corpus synthétique (66 templates)
                     → mémorisation confirmée, non utilisables en prod

Tests              : 204 passent, infrastructure solide
Scripts Phase 1    : implémentés, non encore exécutés sur les données
```

**Critères "production ready" :**
```
val_edge_macro_f1 (réel)    > 0.40   (au-dessus de la baseline triviale)
val_graph_exact_match (réel) > 0.20   (au moins 1 graphe complet sur 5 est correct)
val_node_macro_f1 (réel)    > 0.60
Gap train−val node           < 0.15   (pas de mémorisation)
Test robustesse : "si" → condition ≠ "bien que" → concession (connecteur change la relation)
```

---

## Étape 1 — Annoter le dataset réel avec tokens UD

**Durée estimée réelle :** 2–4h (debug spaCy + alignement spans + vérification)

**Avertissement `prevent` :** 13 phrases seulement. Le split stratifié peut produire
0 exemples de `prevent` dans val ou test. Si c'est le cas, les métriques
`val_edge_macro_f1` exclueront silencieusement cette classe. C'est documenté et
acceptable pour le premier déploiement — `prevent` sera renforcé en Phase 4.

```bash
cd gcn-tools/gcn-scraper/src/gcn_scraper

# 1a. Diagnostiquer les token_span existants
python diagnose_spans.py \
  --input gcn-datasets/real/annotated.json

# Résultat attendu : taux d'invalides > 50% → confirme que les spans doivent être re-dérivés

# 1b. Re-dériver les token_span via spaCy (structure dépendances syntaxiques)
python rederive_spans.py \
  --input  gcn-datasets/real/annotated.json \
  --output gcn-datasets/real/annotated_spans_fixed.json

# Résultat attendu : >= 400 phrases corrigées sur 678
# Si < 400 : les spans sont trop corrompus → re-annotation CIR complète nécessaire (hors scope)

# 1c. Valider la qualité des CIR après correction spans
python validate_cir.py \
  --input gcn-datasets/real/annotated_spans_fixed.json

# 1d. Annoter avec les tokens UD (spaCy fr_core_news_sm)
python annotate_real_dataset.py \
  --input  gcn-datasets/real/annotated_spans_fixed.json \
  --output gcn-datasets/real/annotated_ud.json \
  --lang   fr

# 1e. Vérifier que reps_from_sentence() retourne des représentations non vides
python validate_reps.py \
  --input gcn-datasets/real/annotated_ud.json

# 1f. Re-splitter en train/val/test stratifié sur la relation des edges
python split_real_dataset.py \
  --input   gcn-datasets/real/annotated_ud.json \
  --out-dir gcn-datasets/real/ \
  --seed    42
```

**Critère de succès :** `gcn-train --data-dir gcn-datasets/real/ --epochs 1` produit `loss > 0`

---

## Étape 2 — Entraînement sur données réelles

**Durée estimée :** 10–30 min (selon matériel)

**Note sur le déséquilibre :** `--weighted-loss` compense automatiquement le biais
`cause=60%` en pondérant les classes rares inversement à leur fréquence. `edge-loss-weight=3.0`
favorise l'apprentissage des arêtes face aux nœuds — à descendre à 2.0 si le modèle
ne converge pas sur les nœuds.

```bash
gcn-train \
  --data-dir  gcn-datasets/real/train.json \
  --val-dir   gcn-datasets/real/val.json \
  --epochs    150 \
  --lr        0.001 \
  --use-attention \
  --bidirectional \
  --embedding-dim  50 \
  --edge-loss-weight 3.0 \
  --weighted-loss \
  --weight-decay   0.001 \
  --rgcn-dropout   0.1 \
  --label-smoothing 0.05 \
  --patience       20 \
  --output         checkpoints/prod_v1.npz \
  --log-csv        logs/prod_v1.csv
```

**Reproductibilité :** fixer le seed avant l'exécution :
```bash
export PYTHONHASHSEED=42
```
Le seed du split (42) est déjà fixé dans `split_real_dataset.py`.

**Ce qui peut mal se passer et quoi faire :**

| Symptôme | Cause probable | Action |
|----------|---------------|--------|
| `loss=0` dès la 1ère epoch | tokens UD toujours absents | Re-vérifier étape 1e |
| `val_node_macro_f1 = 0` | Toutes les clauses ont 0 nœud valide | Vérifier alignement token_span |
| `val_edge_macro_f1 < 0.15` | 6 relations absentes + déséquilibre cause (60%) | Normal au 1er run — continuer |
| gap train−val > 0.30 | Mémorisation | Augmenter weight_decay à 0.005 |

---

## Étape 3 — Évaluation et diagnostic

**Durée estimée :** 5 min

```bash
# Métriques sur le test set (jamais vu pendant l'entraînement)
gcn-eval \
  --data-dir    gcn-datasets/real/test.json \
  --checkpoint  checkpoints/prod_v1.npz \
  --output-json logs/prod_v1_test_metrics.json

# Matrice de confusion edges
python -c "
from gcn_python.evaluation.metrics import confusion_matrix, per_class_report
# (après avoir récupéré les prédictions via eval_runner)
"
```

**Lecture des résultats :**

- `val_edge_macro_f1 > 0.40` ET `gap < 0.15` → le modèle généralise → **Prêt pour tests en conditions réelles**
- `val_edge_macro_f1` entre 0.25 et 0.40 → insuffisant mais améliorable → aller à l'étape 4
- `val_edge_macro_f1 < 0.25` → problème fondamental (données, alignement) → diagnostiquer

---

## Étape 4 — Si métriques insuffisantes : enrichir le dataset

**Condition :** `val_edge_macro_f1 < 0.40` après étape 2–3

Le problème principal est la **distribution déséquilibrée et la couverture incomplète** des relations :
- `cause` domine à 60 % → le modèle prédit toujours "cause"
- 6 relations absentes → impossible à prédire

**Action : scraper des phrases ciblées par relation manquante**

```bash
# Scraper 500+ phrases par relation manquante
gcn-scrape \
  --sources wikipedia_fr wikipedia_en hal \
  --out-dir gcn-datasets/raw/targeted/ \
  --limit 5000

# Annoter UD + CIR (gcn-annotate)
gcn-annotate annotate \
  --input  gcn-datasets/raw/targeted/ \
  --out    gcn-datasets/real/targeted_cir.json \
  --model  claude-sonnet-5

# Combiner avec les 678 phrases existantes et re-splitter
python gcn-tools/gcn-scraper/src/gcn_scraper/split_real_dataset.py \
  --input   gcn-datasets/real/targeted_cir.json \
  --out-dir gcn-datasets/real/ \
  --seed    42
```

Répéter l'étape 2 avec le dataset enrichi.

---

## Étape 5 — Tests de robustesse

**Condition :** métriques de l'étape 3 satisfaisantes  
**Durée estimée :** 2–3h (implémentation + exécution)

Ces tests valident que le modèle apprend la **sémantique du connecteur**, pas sa fréquence.
Ils sont indépendants d'un checkpoint — ils s'exécutent avec `pytest` sur le checkpoint `prod_v1.npz`.

Fichier à créer : `gcn-python/tests/test_robustness.py`

```python
"""
Tests de robustesse prod — valident que le modèle discrimine les connecteurs
et la direction des arêtes. Nécessitent un checkpoint entraîné sur données réelles.
Sautés si le checkpoint n'existe pas.
"""
import pytest
import numpy as np
from pathlib import Path
from gcn_python.layer1.representation import UDRepresentation
from gcn_python.layer1.features import FeatureVocabulary, vectorize_clause, vectorize_edge

CHECKPOINT = Path("checkpoints/prod_v1.npz")


def _make_rep(lemma, pos="VERB", dep_rel="root", morph=None) -> UDRepresentation:
    """Construit une UDRepresentation minimale pour les tests de robustesse."""
    return UDRepresentation(
        tokens=[{"lemma": lemma, "pos": pos, "dep_rel": dep_rel, "morph": morph or {}}],
        root_lemma=lemma, root_pos=pos, root_dep_rel=dep_rel,
        root_morph=morph or {}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(1, 1),
    )


def _make_connector(lemma, pos="SCONJ") -> UDRepresentation:
    return _make_rep(lemma, pos=pos, dep_rel="mark")


# ---------------------------------------------------------------------------
# Test 1 — Le connecteur change le vecteur edge
# ---------------------------------------------------------------------------

def test_connecteur_si_vs_bien_que_change_vecteur():
    """
    Deux arêtes identiques sauf le connecteur ("si" vs "bien que") doivent
    produire des vecteurs différents.
    Prouve que le connecteur est bien encodé dans vectorize_edge.
    """
    vocab = FeatureVocabulary()
    src = _make_rep("baisser", morph={"Tense": "Pres"})
    dst = _make_rep("réduire", morph={"Tense": "Pres"})
    conn_si    = _make_connector("si")
    conn_bien  = _make_connector("bien")  # "bien que"

    vec_si   = vectorize_edge(src, dst, conn_si,   1, 2, 3, vocab)
    vec_bien = vectorize_edge(src, dst, conn_bien,  1, 2, 3, vocab)

    assert not np.array_equal(vec_si, vec_bien), (
        "Les vecteurs edge sont identiques malgré des connecteurs différents — "
        "le connecteur n'est pas encodé."
    )


# ---------------------------------------------------------------------------
# Test 2 — L'inversion src/dst change le vecteur edge
# ---------------------------------------------------------------------------

def test_inversion_src_dst_change_vecteur():
    """
    vectorize_edge(A, B, conn, src=1, dst=2, n=3)
    ≠ vectorize_edge(B, A, conn, src=2, dst=1, n=3)
    
    La direction est encodée dans pos_vec[0] = float(src_idx < dst_idx).
    Si le vecteur est symétrique, le modèle ne peut pas distinguer la direction.
    """
    vocab = FeatureVocabulary()
    rep_a = _make_rep("baisser")
    rep_b = _make_rep("réduire")
    conn  = _make_connector("parce")

    vec_ab = vectorize_edge(rep_a, rep_b, conn, 1, 2, 3, vocab)
    vec_ba = vectorize_edge(rep_b, rep_a, conn, 2, 1, 3, vocab)

    assert not np.array_equal(vec_ab, vec_ba), (
        "Les vecteurs A→B et B→A sont identiques — "
        "la direction de l'arête n'est pas encodée."
    )


# ---------------------------------------------------------------------------
# Test 3 — Le modèle prédit "concession" pour "bien que" (si checkpoint dispo)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Checkpoint prod_v1.npz absent")
def test_bien_que_predit_concession():
    """
    Avec le checkpoint prod_v1.npz entraîné sur données réelles,
    une arête avec connecteur "bien que" doit être prédite "concession",
    pas "cause" (classe dominante à 60%).

    Ce test détecte le biais vers "cause" : si le modèle n'a pas appris
    à discriminer les connecteurs, il prédit toujours "cause".
    """
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import load_checkpoint
    from gcn_python.constants import RELATION_TYPES, NODE_TYPES

    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES))
    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
    graph   = RGCNLayer(d_in=d_eff, d_out=d_eff)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    load_checkpoint(pipeline, CHECKPOINT)
    pipeline.encoder.training = False

    src  = _make_rep("baisser", morph={"Tense": "Pres"})
    dst  = _make_rep("réduire", morph={"Tense": "Pres"})
    conn = _make_connector("bien")  # connecteur de concession

    cir = pipeline.forward([src, dst], "Les ventes baissent bien qu'on réduise les coûts.",
                           connector_reps=[conn])
    assert cir["edges"], "Aucune arête produite"
    predicted_relation = cir["edges"][0][2]["relation"]
    assert predicted_relation == "concession", (
        f"Attendu 'concession' pour 'bien que', obtenu '{predicted_relation}'. "
        f"Le modèle est probablement biaisé vers la classe dominante 'cause'."
    )


# ---------------------------------------------------------------------------
# Test 4 — Phrase hors-template (vocabulaire non vu à l'entraînement)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Checkpoint prod_v1.npz absent")
def test_phrase_hors_template_produit_cir_valide():
    """
    Une phrase avec des lemmes absents du dataset d'entraînement doit
    quand même produire un CIR structurellement valide (nodes + edges non vides).
    Prouve que le modèle généralise sur les features syntaxiques, pas les lemmes.
    """
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import load_checkpoint
    from gcn_python.constants import NODE_TYPES

    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES))
    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
    graph   = RGCNLayer(d_in=d_eff, d_out=d_eff)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    load_checkpoint(pipeline, CHECKPOINT)
    pipeline.encoder.training = False

    # Lemmes volontairement hors-vocabulaire d'entraînement
    src  = _make_rep("désinhiber",  pos="VERB", morph={"Tense": "Pres"})
    dst  = _make_rep("exacerber",   pos="VERB", morph={"Tense": "Fut"})
    conn = _make_connector("parce")

    cir = pipeline.forward([src, dst], "...", connector_reps=[conn])
    assert len(cir["nodes"]) == 2,  f"Attendu 2 nœuds, obtenu {len(cir['nodes'])}"
    assert len(cir["edges"]) == 1,  f"Attendu 1 arête, obtenu {len(cir['edges'])}"
    for node in cir["nodes"]:
        assert node["node_type"] in NODE_TYPES, (
            f"Type de nœud inconnu : {node['node_type']!r}"
        )
```

**Exécution :**
```bash
# Sans checkpoint (teste uniquement la vectorisation)
python -m pytest gcn-python/tests/test_robustness.py -k "not prod" -v

# Avec checkpoint (teste le modèle entraîné)
python -m pytest gcn-python/tests/test_robustness.py -v
```

**Interprétation du test 3 :**
- Si `predicted_relation == "cause"` → le modèle n'a pas appris la sémantique des connecteurs → entraîner plus longtemps ou augmenter le dataset de concession
- Si `predicted_relation == "concession"` → généralisation confirmée pour cette classe

---

## Étape 6 — Packaging checkpoint prod

**Condition :** étapes 3 et 5 satisfaisantes

```bash
# Renommer le checkpoint validé
cp checkpoints/prod_v1.npz checkpoints/prod_stable.npz

# Documenter les métriques du checkpoint
cat logs/prod_v1_test_metrics.json

# Vérifier que le checkpoint se charge sans erreur
python -c "
from gcn_python.training.checkpoint import load_checkpoint
from gcn_python.pipeline.cgnp import CGNPipeline
# ... charger et faire un forward sur une phrase test
print('Checkpoint OK')
"
```

---

## Étape 7 — Versioning et monitoring post-déploiement

**Versionner les checkpoints :**
```bash
# Convention : prod_v{N}.npz + prod_v{N}_metrics.json
cp checkpoints/prod_v1.npz checkpoints/prod_v1_stable.npz
echo '{"val_edge_macro_f1": X, "val_graph_exact_match": Y, "date": "2026-09-17"}' \
  > checkpoints/prod_v1_stable_metrics.json
```

**Rollback si dégradation :**
- Garder `prod_v1_stable.npz` intact tant qu'un `prod_v2` n'est pas validé
- Ne jamais écraser un checkpoint stable avec un checkpoint non-évalué

**Monitoring :**
- Loguer `confidence` et `relation` prédites par phrase en production
- Alerter si `confidence` moyen < 0.5 sur une fenêtre de 100 phrases (signal de dérive)
- Re-entraîner si la distribution des relations prédites s'écarte > 20% de la distribution val

---

## Résumé du chemin critique

```
Étape 1 : annoter UD dataset réel                [BLOQUANT — 2–4h]
    ↓
Étape 2 : entraîner sur données réelles          [20–30 min]
    ↓
Étape 3 : évaluer — métriques OK ?
    ├── OUI (val_edge_f1 > 0.40)  ──────→ Étape 5 : robustesse [2–3h]
    │                                              ↓
    │                                      Étape 6 : packaging [15 min]
    │                                              ↓
    │                                      Étape 7 : monitoring [15 min]
    │
    └── NON (val_edge_f1 < 0.40)  ──────→ Étape 4 : enrichir dataset [4–8h]
                                                   ↓
                                           Étape 2 (itération)

Durée totale chemin nominal : ~6–8h
Durée totale avec enrichissement : ~14–20h
```

---

## Métriques minimales de production

| Métrique | Minimum acceptable | Idéal |
|----------|-------------------|-------|
| `val_node_macro_f1` (réel) | > 0.60 | > 0.80 |
| `val_edge_macro_f1` (réel) | > 0.40 | > 0.60 |
| `val_graph_exact_match` (réel) | > 0.20 | > 0.40 |
| Gap train−val node | < 0.15 | < 0.10 |
| Test robustesse connecteurs | Pass | Pass |
| Baseline triviale edge (cause 60%) | Battu | Battu |
