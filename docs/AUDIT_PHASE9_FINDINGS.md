# Audit SPEC_PHASE9_PIPELINE_ML.md — Findings

**Date :** 2026-09-15
**Réviseur :** Claude (audit interne)
**Cible :** docs/SPEC_PHASE9_PIPELINE_ML.md rev.3

---

## Statut : RÉSOLU

Tous les findings ont été intégrés dans la spec rev.3 (F1-F10) et la spec rev.4 (F11).
L'implémentation code est complète (commits 3bafb75 + 12691c6).

---

## Findings critiques

### F1 — C1 : correction logiquement défectueuse

**Problème :** Le spec propose d'appeler `_backward_rgcn(d_enriched, lr)` même quand l'encodeur est custom (sans `backward_node_dx`). Mais `d_enriched` est rempli par `encoder.backward_node_dx()` — sans ce backward, `d_enriched` reste un tensor de zéros. Passer des zéros à `backward_message_pass` ne met à jour rien d'utile dans le R-GCN.

**Affirmation erronée dans le spec :** "Backward R-GCN — indépendant de l'encodeur" — c'est faux, le gradient vers le R-GCN vient du backward du MLP encoder.

**Correction requise :** Soit documenter que `backward_message_pass` n'est utilisable qu'avec `MLPEncoder` de référence, soit exiger que tout `CausalEncoder` implémente `backward_node_dx` (contrainte d'interface).

---

### F2 — C6 : confusion sémantique `explicit` vs confiance du modèle

**Problème :**
```python
if confidence < 0.5:
    return "inferred"
```
`explicit` vs `inferred` est une propriété du texte source — soit le marqueur causal est présent dans la phrase, soit la relation est inférée. La confiance du modèle n'a rien à voir. Un connecteur "parce que" détecté avec confiance 0.3 reste un marqueur explicite.

**Correction requise :** Lier `origin="explicit"` à la présence d'un `connector_rep` non-None, pas à `confidence`. Exemple :
```python
def _infer_origin(rep, node_type, connector_rep) -> str:
    if node_type == "condition" and connector_rep is None:
        return "inferred"
    return "explicit"
```

---

## Findings modérés

### F3 — C7 : `_find_patient_lemma` non définie

La correction propose `_find_patient_lemma(rep)` dans `build_label`. Cette fonction n'existe pas dans `label_builder.py` et n'est pas définie dans le spec. La correction est incomplète.

**Action requise :** Définir `_find_patient_lemma` — chercher un token avec `dep_rel in {"obj", "iobj", "nobj"}`.

---

### F4 — C2 : sévérité surévaluée

Le spec présente le padding de `d_enriched` comme un "bug silencieux critique". En pratique, `n = min(len(d_node_logits), len(vecs))` vaut toujours `N` dans le flow d'entraînement normal : `d_node_logits` provient de `loss()` qui reçoit les logits du `forward()`, lesquels ont exactement `N` lignes. Ce cas ne se produit que si le caller tronque `d_node_logits` manuellement.

**Action requise :** Reclasser comme amélioration défensive, pas comme bug critique.

---

### F5 — C8 : `build_label` a déjà le paramètre `taxonomies_dir`

La signature actuelle (`label_builder.py:10-13`) est déjà :
```python
def build_label(rep, node_type: str, taxonomies_dir: Path | None = None) -> str:
```
Le seul travail réel est d'ajouter `taxonomies_dir` à `CGNPipeline.__init__` et de le passer à l'appel ligne 197-200 de `cgnp.py`. Le spec laisse entendre qu'il faut modifier l'interface de `build_label`, ce qui est inexact.

**Action requise :** Reformuler C8 — seul `CGNPipeline.__init__` est à modifier.

---

## Findings légers

### F6 — C5 : import inutilisé + portée uniquement FR

```python
from ..constants import SCOPE_VALUES  # importé, jamais utilisé
```
`_SCOPE_HINTS` est un dict local avec uniquement des lemmes français. Pas de support multilingue documenté.

**Action requise :** Supprimer l'import. Documenter explicitement la limitation FR-only ou ajouter des lemmes EN.

---

### F7 — C3/C4 : négation analytique non couverte

`is_negative` vérifie `root_morph.get("Polarity", "") == "Neg"` (morphologie UD). Les négations analytiques du français ("ne...pas") dont "pas" n'est pas le root ne seront pas détectées par cette heuristique.

**Action requise :** Documenter la limitation. Optionnel : chercher un token avec `dep_rel == "advmod"` et `lemma in {"pas", "jamais", "plus", "guère"}` dans le span.

---

### F8 — Assertion end-to-end insuffisante

```python
assert len(set(n['scope'] for n in cir['nodes'])) >= 1
```
Cette assertion passe toujours, même si tous les scopes restent "specific". Elle ne vérifie pas que la correction C5 fonctionne.

**Action requise :** Remplacer par une phrase de test contenant "tous" ou "chaque" et asserter `any(n['scope'] == 'universal' for n in cir['nodes'])`.

---

## Findings structurels

### F9 — C10 mal positionné

C10 apparaît dans la cartographie des corrections et dans l'ordre d'implémentation (position 10), mais la section dit "ne pas changer". Un lecteur suivant l'ordre d'implémentation arrive à C10 pour ne rien faire.

**Action requise :** Retirer C10 de la cartographie des corrections et de l'ordre d'implémentation. Le placer uniquement dans la table des limitations.

---

### F10 — Double numérotation P-x / C-x ambiguë

La cartographie initiale utilise P1-P14. Les sections de correction utilisent C1-C10. La correspondance n'est pas documentée de façon directe dans le spec.

**Action requise :** Ajouter une table de correspondance P→C, ou unifier la numérotation.

---

### F11 — P2d et P3e sans code dans rev.3

P2d (décodeur autorégressif) est la correction structurellement la plus lourde. Le spec délègue à "Détails inchangés depuis rev.2." Une rev.3 marquée "audit exhaustif intégré" qui n'inclut pas le code de ses cas complexes est incomplète comme référence d'implémentation.

**Action requise :** Inclure le code de P2d et P3e, ou créer des specs dédiés référencés explicitement.

---

## Tableau de synthèse

| Finding | Correction cible | Sévérité | Action |
|---|---|---|---|
| F1 | C1 | Critique | Revoir la logique backward R-GCN avec custom encoder |
| F2 | C6 | Critique | Lier `explicit` à `connector_rep`, pas à `confidence` |
| F3 | C7 | Modérée | Définir `_find_patient_lemma` |
| F4 | C2 | Modérée | Reclasser en amélioration défensive |
| F5 | C8 | Modérée | Reformuler — seul `CGNPipeline.__init__` change |
| F6 | C5 | Légère | Supprimer import inutilisé, documenter FR-only |
| F7 | C3/C4 | Légère | Documenter la limitation négation analytique |
| F8 | Vérif. e2e | Légère | Assertion scope insuffisante |
| F9 | C10 | Structurel | Retirer de la cartographie des corrections |
| F10 | Cartographie | Structurel | Unifier numérotation P-x / C-x |
| F11 | P2d/P3e | Structurel | Inclure le code ou référencer un spec dédié |
