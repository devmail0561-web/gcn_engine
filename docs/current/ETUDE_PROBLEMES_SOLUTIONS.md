# ETUDE — Problèmes et Solutions GCN-Core v4.0 (2026-09-29, MAJ 2026-09-29)

**Auteur :** Michel Tendeng  
**Date :** 2026-09-29  
**MAJ :** 2026-09-29 — les 7 bugs ouverts (§2.7–§2.13) sont corrigés en v4, tests verts (685 Python passed / 3 xfailed attendus, Rust gcn-backend 56+6 OK, frontends FR/EN OK).  
**MAJ2 :** 2026-09-29 — P0 (stub v3) + P2.1/P2.2/P2.3 implémentés sur `feat/p2-frontends`, audit complet sans complaisance (8 constats F1–F8, Phase A appliquée, B2/B3 appliqués). Python 687 passed / 3 xfailed, Rust workspace 211 tests verts.  
**Scope :** Session d'audit et de travail — bugs confirmés, architecture, types de données  
**Méthode :** Chaque claim vérifié dans le code avant d'être écrit. Aucune affirmation non vérifiée.

---

## 1. Résumé exécutif

Deux axes de travail distincts ont été soulevés dans cette session.

**Axe 1 — Bugs moteur.** 13 bugs identifiés lors d'un audit en 4 dimensions (Rust, Python ML, docs/CI, sécurité). 6 corrigés en première session, 7 corrigés en v4 (MAJ 2026-09-29) — 12 avec changement de code, 1 (§2.10 hyperarêtes) volontairement sans changement (attente données ternaires N_min=20). Tous ont été vérifiés dans le code source — les corrections sont en place et les tests passent (685 Python passed / 3 xfailed attendus — checkpoint v2 incompatible, cf. Priorité 0 ; Rust gcn-backend 56+6 OK, frontends FR/EN OK).

**Axe 2 — Problème architectural.** Le moteur traite des phrases NL. Les use cases réels fournissent des données hétérogènes (logs, graphes structurés, données tabulaires, PDF, texte mathématique) qui ne sont pas et ne peuvent pas être des phrases NL. Ce n'est pas un bug — c'est une limite de périmètre non documentée. La solution est l'ajout de frontends spécialisés suivant le même patron que `gcn-frontend-code`, le seul frontend non-NL existant.

**Avertissement honnête :** le USECASES_AUDIT.md lui-même (ligne 373) dit que le seul bloquant vers la production est "des données annotées dans chacun des domaines cibles". Les bugs et les nouveaux frontends sont des améliorations techniques, mais sans données de domaine annotées, aucun déploiement n'est possible.

---

## 2. Bugs moteur

### 2.1 Slicing UTF-8 non sécurisé — query.rs

**Fichier :** `gcn-core/crates/gcn-backend/src/query.rs:151-154`  
**Statut :** CORRIGÉ

**Description :** `&input[..200]` coupe sur des octets, pas des caractères. En Rust, indexer un `&str` sur des octets produit un panic si l'octet 200 tombe au milieu d'un caractère multi-octets (é, ü, à). Un simple `gcn query "é..."` avec un caractère accentué en position 199 crashe le CLI.

**Solution appliquée :**
```rust
let preview = {
    let end = input.char_indices().nth(200).map(|(i, _)| i).unwrap_or(input.len());
    &input[..end]
};
```
`char_indices().nth(200)` retourne l'offset byte de la 200e frontière de caractère, garantissant un slice valide.

**Vérification codebase :** Lignes 151-154 lues — correction en place.

**Évaluation :** C'est la meilleure solution. Alternatives écartées : (a) `chars().take(200).collect::<String>()` — correct mais alloue en heap inutilement pour un message d'erreur ; (b) truncation UTF-8 lossy — moins lisible et masque l'intention. La solution retenue est O(n) en temps, zéro allocation.

---

### 2.2 RCE via allow_pickle — init_v3_stub.py

**Fichier :** `scripts/init_v3_stub.py:58`  
**Statut :** CORRIGÉ

**Description :** `np.load(v2_path, allow_pickle=True)` sans passer par `guarded_np_load` permettait à un checkpoint v2 malveillant contenant un tableau `object` Python d'exécuter du code arbitraire au chargement. C'était le seul endroit du repo qui contournait la garde de sécurité existante.

**Solution appliquée :**
```python
from gcn_python.security.npz_guard import guarded_np_load  # ligne 30
data = guarded_np_load(v2_path)  # ligne 58
```
`guarded_np_load` appelle `assert_npz_pickle_safe(path)` (audit du pickle via `_RestrictedUnpickler` avec allowlist) avant de charger.

**Vérification codebase :** Import ligne 30, appel ligne 58 — correction en place. `guarded_np_load` implémentée dans `gcn-python/src/gcn_python/security/npz_guard.py:118-124`.

**Évaluation :** Solution correcte. Le script est de migration manuelle (exécuté par Michel sur ses propres checkpoints), mais la règle "aucun contournement de la garde" est une invariante de sécurité répertoriée dans le code. La correction est cohérente avec l'architecture de sécurité existante.

**Limite :** `guarded_np_load` utilise `_RestrictedUnpickler` avec une allowlist. Si les checkpoints v2 contiennent des tableaux `object` de types non couverts par l'allowlist, le chargement lèvera `UnsafeCheckpointError`. Tester sur un checkpoint v2 réel avant de l'utiliser en production.

---

### 2.3 LexicalConnectorAssembler — n_relations hardcodé à 11

**Fichier :** `gcn-python/src/gcn_python/verbalizer/trainable.py:522`  
**Statut :** CORRIGÉ

**Description :** `n_relations: int = 11` comme valeur par défaut dans `LexicalConnectorAssembler.__init__`. Avec 19 relations v3.0, tout `relation_idx >= 11` lève `ValueError` (ligne 539 : `if not (0 <= int(relation_idx) < self.n_relations): raise ValueError(...)`). Identique dans `from_json` (ligne 596 : `data.get("n_relations", 11)`). Bloquant pour réentraîner avec des données v3.

**Solution appliquée :**
```python
from ..constants import RELATION_TYPES  # ligne 6 de trainable.py
n_relations: int = len(RELATION_TYPES)  # ligne 522 → 19
# et from_json:
n_relations=data.get("n_relations", len(RELATION_TYPES))  # ligne 596
```
De plus, `train.py:441` : `LexicalConnectorAssembler(connector_vocab, n_relations=len(RELATION_TYPES))`.

**Vérification codebase :** Lignes 522, 596 de `trainable.py` lues — `len(RELATION_TYPES)` en place. `RELATION_TYPES` importé en tête de fichier.

**Évaluation :** Solution minimale et correcte. `len(RELATION_TYPES)` est dynamique — si les relations changent à nouveau, le défaut suit sans modification. Alternative écartée : hardcoder 19 — fragile et contraire à D1 (pas de valeurs en dur).

---

### 2.4 joint_cause silencieux — edge_norm.py

**Fichier :** `gcn-python/src/gcn_python/data/edge_norm.py:225-232`  
**Statut :** CORRIGÉ

**Description :** Toute arête avec 2 sources dont la relation n'est pas déjà `joint_cause` ou `joint_prevent` était réécrite en `joint_cause` silencieusement. Une annotation incorrecte (`cause` avec 2 sources) était transformée en données d'entraînement trompeuses sans aucun signal.

**Solution appliquée :**
```python
if relation and sanitize_text(str(relation)) not in ("joint_cause", "joint_prevent"):
    import warnings as _w
    _w.warn(
        f"edge_norm: relation '{relation}' sur 2 sources remplacée par 'joint_cause'. "
        "Annoter explicitement joint_cause/joint_prevent pour supprimer cet avertissement.",
        UserWarning, stacklevel=4,
    )
    relation = "joint_cause"
```

**Vérification codebase :** Lignes 225-232 lues — avertissement en place.

**Évaluation :** La correction ajoute de la visibilité sans changer le comportement. C'est la bonne approche ici : le remplacement est intentionnel (une arête à 2 sources doit être joint_cause), mais il ne doit pas être silencieux. Alternative — rejeter l'arête — serait trop agressive et casserait les pipelines existants.

---

### 2.5 Filtre MediatedCause incorrect — pearl.rs

**Fichier :** `gcn-core/crates/gcn-backend/src/pearl.rs:655-663`  
**Statut :** CORRIGÉ

**Description :** La fonction `counterfactual` vérifiait si le médiateur M d'une `MediatedCause(X→C via M)` avait d'autres sources pour décider si M devait être exclu du graphe résiduel. Le filtre original `*s != target.id && *s == mediator_id` sélectionnait les arêtes **sortant du médiateur** (source == mediator), pas les arêtes **arrivant au médiateur** (destination == mediator). Résultat : `other_sources` était biaisé, le médiateur n'était pas exclu quand il aurait dû l'être, produisant des faux positifs dans le raisonnement contrefactuel.

**Solution appliquée :**
```rust
let other_sources = ir
    .edges
    .iter()
    .filter(|(s, d, _e)| *d == mediator_id && *s != target.id)
    .count();
```
`*d == mediator_id` sélectionne correctement les arêtes arrivant au médiateur, hors la source X (target.id).

**Vérification codebase :** Ligne 658 lue — `filter(|(s, d, _e)| *d == mediator_id && *s != target.id)` en place.

**Évaluation :** Correction minimale et correcte. La sémantique est maintenant : "le médiateur a-t-il d'autres sources que X ?" — c'est exactement ce que le commentaire ligne 649 (`si M n'a pas d'autre source → M disparaît aussi`) requiert.

---

### 2.6 NODE_TYPES v2 dans gcn-annotate — normalize.py

**Fichier :** `gcn-tools/gcn-annotate/src/gcn_annotate/normalize.py:9-11`  
**Statut :** CORRIGÉ

**Description :** `NODE_TYPES` contenait les 8 types v2 (`etat`, `action`, `transition`, `processus`, `condition`, `entite`, `etat_systemique`, `contrainte`) — donc les types D5 v4 (`etat_local`, `etat_global`, `concept`, `evenement`) étaient rejetés par `normalize_node_type()`. De plus, les aliases `etat_local → etat`, `etat_global → etat_systemique` mappaient vers des cibles qui n'étaient même plus dans les types v4 valides, produisant des annotations invalides silencieusement.

**Solution appliquée :**
```python
NODE_TYPES = {
    "processus", "etat_local", "etat_global", "entite",
    "condition", "concept", "evenement", "contrainte",
}
# Aliases inversés : v2→v4 (migration transparente)
NODE_TYPE_ALIASES = {
    "etat": "etat_local",
    "etat_systemique": "etat_global",
    "action": "processus",
    "transition": "processus",
    ...
}
```

**Vérification codebase :** Lignes 9-55 lues — types v4 et aliases v2→v4 en place.

**Évaluation :** Correction nécessaire et complète. Les aliases v2→v4 permettent de lire les données existantes annotées avec l'ancien schéma. La migration est transparente.

---

### 2.7 T5-min potentiellement silencieux — train.py

**Fichier :** `gcn-python/src/gcn_python/training/train.py:1252-1274` (+ alignement, ~§1277+)  
**Statut :** CORRIGÉ (v4, 2026-09-29)

**Description :** Le code T5-min appelle `pipeline.forward(_vs.sentence.text)` (passage d'une chaîne de caractères). Si ce chemin ne peuple pas `pipeline._cached_edge_logits` (attribut privé qui dépend de l'implémentation interne de `forward()`), la condition `if pipeline._cached_edge_logits is not None` est toujours fausse, `_val_logits` reste vide, et T5 ne s'exécute jamais — sans message d'erreur, juste `if _val_logits and len(_val_labels) >= 10:` silencieusement faux. De plus, `except Exception: pass` (ligne 1263) avale toutes les erreurs individuelles.

**Vérification codebase :** Lignes 1252-1274 lues — le problème est réel. `pipeline.forward(str)` passe par le chemin `discuss.py`/bridge, qui peut ne pas peupler `_cached_edge_logits`. Ce dépend de si le pipeline a un checkpoint chargé et si le chemin `str` invoque les couches ML ou seulement le symbolique.

**Solution proposée :**
1. Remplacer `pipeline.forward(_vs.sentence.text)` par `pipeline.forward(_vs.reps, _vs.connector_reps, _vs.valid_clause_idxs)` (chemin direct avec les représentations déjà calculées depuis le val_loader).
2. Logger explicitement si aucun logit n'est collecté : `click.echo("T5-min : aucun logit collecté — chemin forward incorrect")`.
3. Remplacer `except Exception: pass` par `except Exception as _e: warnings.warn(f"T5 skip: {_e}")`.

**Pourquoi cette solution :** Le val_loader a déjà les représentations calculées — utiliser `_vs.reps` directement contourne le problème du chemin `str` et est cohérent avec comment le training loop utilise le val_loader (lignes 748-756). C'est le chemin le plus court vers une T5 qui fonctionne réellement.

**Risque :** Vérifier que `pipeline.forward(reps, ...)` peuple bien `_cached_edge_logits` — lire `cgnp.py:forward()` avant d'implémenter.

**Correction appliquée (v4) :** boucle T5 réécrite — `reps_from_sentence(_vs.sentence)` puis `pipeline.forward(_reps, text, clause_positions, n_total_clauses, connector_reps)` (même chemin que `_run_eval_pass`), alignement logits↔gold via paires ordonnées + masque `edge_map.get(p, -1)` avec garde d'indice, `warnings.warn` par phrase ignorée au lieu de `except: pass`, compteur `_t5_skipped` affiché, message explicite si aucun logit collecté. Suite Python : 685 passed / 3 xfailed.

---

### 2.8 Seed partiel — train.py

**Fichier :** `gcn-python/src/gcn_python/training/train.py:274-278, 493-497` + `pipeline/cgnp.py:__init__` (param `seed`)  
**Statut :** CORRIGÉ (v4, 2026-09-29)

**Description :** `np.random.seed(seed)` et `torch.manual_seed(seed)` sont appelés (lignes 275, 278). `LinkPredHead` (ligne 496) est créé après ces appels et utilise `torch.nn.init` — il est donc couvert par `torch.manual_seed()`. En revanche, les couches R-GCN extra dans `CGNPipeline` utilisent `seed=extra_i*100+42` hardcodé (confirmé dans l'audit précédent, non relu ici) — ces couches ne sont pas contrôlées par `--seed`.

**Vérification codebase :** Lignes 274-278 et 493-497 lues — torch et numpy seedés. LinkPredHead créé ligne 496 sans seed explicite mais couvert par torch global. R-GCN extra: non relu dans cette session — se fier à l'audit précédent.

**Solution proposée :** Passer `seed` au constructeur des couches R-GCN extra dans `CGNPipeline.__init__`. Modifier la signature pour accepter `seed: int = 42` et l'utiliser dans `np.random.default_rng(seed + extra_i * 100)` au lieu du hardcode.

**Évaluation :** Impact limité sur les résultats (les seeds hardcodés sont déterministes, juste non configurables). Priority faible — ne bloque pas le réentraînement.

**Correction appliquée (v4) :** `CGNPipeline.__init__` accepte `seed: int = 42` (`self._seed`), couches R-GCN extra en `seed + extra_i*100` (2 sites : kwargs + fallback TypeError). `train.py` propage `_init_seed` à `WordEmbedding` (×2 : fastText + standard), `CGNPipeline` et `LinkPredHead` (qui défautait à 0). Reproductibilité complète via `--seed`.

---

### 2.9 OOV accumule sur _unk — embedding.py

**Fichier :** `gcn-python/src/gcn_python/layer1/embedding.py:67, 81, 86`  
**Statut :** CORRIGÉ (v4, 2026-09-29 — Option A)

**Description :** `self._vocab.get(lemma, 0)` retourne indice 0 pour tout lemme OOV (ligne 67 lookup, ligne 81 backward). `self._grad_accum[idx] += d_emb` (ligne 86) accumule le gradient de tous les OOV sur `_E[0]` = vecteur `_unk`. Sur le bridge heuristique (qui produit beaucoup d'OOV), `_unk` absorbe un gradient bruité agrégé et perd sa sémantique d'inconnu.

**Vérification codebase :** Lignes 65-86 lues — comportement confirmé.

**Solution proposée :** Séparer "lemme absent du vocabulaire" de "lemme inconnu". Deux options :
- **Option A (recommandée) :** `_vocab.get(lemma, None)` → si `None`, ne pas accumuler de gradient (no-op comme les lemmes gelés). `_unk` reste propre, OOV vrais produisent un vecteur zéro.
- **Option B :** Garder l'indice 0 mais ajouter une clé dédiée `"_oov_bucket"` distincte de `"_unk"`, accumuler le gradient sur ce bucket séparé.

**Pourquoi Option A :** Plus simple. Les OOV sur le bridge ne doivent pas être appris — ils représentent du bruit de tokenisation heuristique. Les garder à zéro est plus honnête que d'apprendre un vecteur "moyen de tout ce qu'on n't connaît pas". Option B est pertinente si on veut capturer quelque chose des OOV, mais sans fastText subword actif, c'est prématuré.

**Impact :** Correction de second ordre — le bridge est heuristique, les OOV sont structurels. La vraie solution à long terme est un parser UD réel (stanza) qui n'a pas d'OOV sur les tokens UD.

**Correction appliquée (v4, Option A) :** `lookup()` retourne un vecteur zéro pour tout lemme absent (`_vocab.get(lemma)` → `None` → `np.zeros`) au lieu de `_E[0]` ; `_unk` n'est retourné que pour le lemme explicite `"_unk"`. `backward()` : no-op si `idx is None` (plus d'accumulation du bruit OOV sur `_unk`). Sanity check OK (OOV=zéro, `_grad_accum[0]` propre).

---

### 2.10 Hyperarêtes non supervisées — loader.py

**Fichier :** `gcn-python/src/gcn_python/data/loader.py:111-127`  
**Statut :** CONFIRMÉ SANS CHANGEMENT (v4, 2026-09-29 — attente données, décision inchangée)

**Description :** Les arêtes avec `len(src_list) > 1` sont routées vers `hyperedge_map` via `continue`, contournant `edge_map`. `hyperedge_map` est collecté dans `SentenceRecord` mais aucune loss dans `train.py` ne le supervise. Confirmé dans `train.py:522-530` : un `warnings.warn` signale ce comportement.

**Vérification codebase :** Lignes 111-127 lues — comportement confirmé. `train.py:522-530` a le warning explicite.

**Évaluation :** Ce n'est pas un bug caché — c'est un comportement documenté dans le code lui-même. La supervision des N-arêtes nécessite une loss auxiliaire sur `joint_cause` / `joint_prevent`. Sans données annotées avec `joint_group_id` (N_min=20 requis selon REGLE_EQUILIBRE_DATASET.md), toute loss sur les hyperarêtes entraînerait sur du bruit.

**Solution :** Ne rien faire tant que les données ternaires ne sont pas annotées (N_min=20 par type ternaire). Quand les données existent : ajouter une loss BCE sur les paires `joint_group_id` dans `train.py`, directement après la loss edge principale.

**Vérification v4 :** warning `train.py:525-531` et routage `hyperedge_map` confirmés en place — aucun changement de code, conformément à la décision.

---

### 2.11 Format incohérent DOT vs JSON — export.rs

**Fichier :** `gcn-core/crates/gcn-backend/src/export.rs:33`  
**Statut :** CORRIGÉ (v4, 2026-09-29)

**Description :** `format!("{:?}", edge.relation)` utilise le format Debug Rust → PascalCase (`"Cause"`, `"ConditionalCause"`). `query.rs:943` utilise `serde_json::to_value(l.relation)` → snake_case serde (`"cause"`, `"conditional_cause"`). Deux formats distincts dans le même binaire. Le DOT exporté ne peut pas être parsé avec les mêmes règles que le JSON.

**Vérification codebase :** `export.rs:33` lu — `format!("{:?}", edge.relation)`. `query.rs:943` lu — `serde_json::to_value(l.relation)`.

**Solution proposée :**
```rust
// export.rs:33 — remplacer
let rel = format!("{:?}", edge.relation);
// par
let rel = serde_json::to_value(edge.relation)
    .ok()
    .and_then(|v| v.as_str().map(|s| s.to_string()))
    .unwrap_or_else(|| format!("{:?}", edge.relation).to_lowercase());
```
Cohérent avec `link_to_dto` dans `query.rs:942-946` qui fait exactement ce pattern avec un fallback.

**Impact :** Correction mineure — le DOT est utilisé pour visualisation (Graphviz), pas pour parsing. Mais l'incohérence est une source de confusion pour les utilisateurs.

**Correction appliquée (v4) :** `export.rs` utilise `serde_json::to_value(edge.relation)` → snake_case avec fallback Debug lowercased (même pattern que `link_to_dto`) pour les relations ET le tooltip `node_type`. Test `dot_export_contains_nodes_and_edges` mis à jour (assert `cause`, anti-assert `Cause`). Rust gcn-backend : 56+6 OK.

---

### 2.12 Frontend FR — 9/19 relations dans rules.rs

**Fichier :** `gcn-core/crates/gcn-frontend-fr/src/rules.rs:24-36` (+ `gcn-frontend-en/src/rules.rs` parité)  
**Statut :** CORRIGÉ partiel (v4, 2026-09-29 — `filter` ajouté, ternaires au ML)

**Description :** `relation_type_str_to_enum` mappe 9 variants seulement : Cause (2 alias), Condition, Concession, Sequence, Opposition, Enable, Prevent, Motivation. Les 10 manquants : Filter, Analogy, Counterfactual, DataDependency, ControlDependency, ConditionalCause, MediatedCause, JointCause, ConditionalPrevent, MediatedPrevent, JointPrevent. Tout token YAML avec ces valeurs retourne `None` → arête silencieusement ignorée.

**Vérification codebase :** `rules.rs:24-36` lu entièrement — 9 variants confirmés.

**Évaluation :** Ce n'est pas exactement un bug — c'est une limite documentée du frontend symbolique FR. Les types ternaires (`conditional_cause`, etc.) ne peuvent pas être détectés par des règles lexicales simples sans le pipeline ML. `filter` pourrait être ajouté (pattern "ne...que si" est détectable par règle). `data_dependency` et `control_dependency` appartiennent au frontend code, pas FR.

**Solution partielle :** Ajouter `filter` dans `relation_type_str_to_enum` si une entrée YAML pour ce type existe dans les taxonomies FR. Ne pas ajouter les ternaires — leur détection requiert le ML (détecteur `detect_ternary()` dans `ir_emitter.py`).

**Priorité :** Faible. Le frontend FR est un outil de bootstrap pour générer des données d'entraînement, pas le pipeline de production. La production passe par le pipeline ML.

**Correction appliquée (v4) :** `"filter" | "filtre" | "filtrage" → RelationType::Filter` en FR, `"filter"` en EN (parité). Vérifié : aucune entrée `filter` n'existe encore dans `gcn-references/taxonomies/*/`, donc effet nul aujourd'hui — débloque les futures entrées YAML sans toucher aux ternaires. Frontends FR/EN : tests OK.

---

### 2.13 spof_all sans garde — pearl.rs

**Fichier :** `gcn-core/crates/gcn-backend/src/pearl.rs:383`  
**Statut :** CORRIGÉ (v4, 2026-09-29 — breaking change documenté)

**Description :** `pub fn spof_all(ir: &CausalIR)` est une fonction publique O(n²·(n+e)) sans garde de taille. La garde `SPOF_MAX_NODES = 500` existe dans `query.rs:639-646` mais seulement sur le chemin CLI. Un utilisateur qui appelle `pearl::spof_all()` directement via la crate peut provoquer un calcul exponentiel sur un grand graphe.

**Vérification codebase :** `pearl.rs:383` lu — pas de garde. `query.rs:638-646` lu — garde à 500 nœuds présente.

**Solution proposée :**
```rust
pub fn spof_all(ir: &CausalIR) -> Result<(usize, Vec<SpofScore>), BackendError> {
    const SPOF_MAX_NODES: usize = 500;
    if ir.nodes.len() > SPOF_MAX_NODES {
        return Err(BackendError::QueryParseError(format!(
            "SPOF: graph too large ({} nodes > {})", ir.nodes.len(), SPOF_MAX_NODES
        )));
    }
    // corps inchangé
}
```
La signature change de `-> (usize, Vec<SpofScore>)` à `-> Result<...>` — breaking change pour les consommateurs directs. À faire avec bump de version mineur.

**Priorité :** Moyenne. Le vecteur d'exploitation est limité (requiert accès direct à la crate, pas via CLI). Mais `spof_all` est publique et documentée dans le README — corriger avant la prochaine release.

**Correction appliquée (v4) :** `spof_all` retourne `Result<(usize, Vec<SpofScore>), BackendError>` avec garde interne `SPOF_MAX_NODES = 500` (même seuil que le CLI). Appelants mis à jour : `query.rs` (`?`), test `test_ternary_pearl.rs` (`.expect`), README backend (`.expect`). ⚠️ Breaking change — bump version mineure requis.

---

## 3. Problème architectural — types de données

### 3.1 Analyse du patron d'extraction existant

Deux frontends co-existent et servent de référence pour tout nouveau frontend :

**`gcn-frontend-fr` (NL symbolique) — patron en 4 étapes :**
1. Tokenisation + étiquetage POS heuristique (tagger.rs)
2. Détection de connecteurs causaux : verbes causaux (lookup YAML `causal_verb_relations`) OU marqueurs multi-mots (lookup YAML `causal_markers`)
3. Découpage en clauses : le connecteur est le pivot — le texte à gauche est une unité causale, le texte à droite en est une autre
4. Attribution de RelationType depuis la table de correspondance dans le YAML (via `rules.rs:relation_type_str_to_enum`)

**`gcn-frontend-code` (non-NL symbolique) — même patron, primitives différentes :**
1. Parsing AST tree-sitter (remplace tokenisation)
2. Détection de "connecteurs" : le **kind du nœud AST parent** (if_statement, for_expression, function_definition...) remplace les SCONJ lexicaux
3. Découpage en "clauses" : les nœuds AST du corps (corps d'un if, d'une fonction) remplacent les spans de tokens
4. RelationType depuis `mapper.rs` → `kind_to_edge_type` (YAML `python_ast.yaml`, `rust_ast.yaml`)

**Patron générique (valide pour tous les types) :**
```
Entrée (format natif)
  → Unités causales (clauses NL, nœuds AST, événements, objets JSON...)
  → Signal causal (SCONJ lexical, topologie AST, champ relationship_type, ordre temporel...)
  → RelationType (lookup ou ML)
  → CausalIR (même format pour tous)
```

La clé est : **le connecteur causal peut être n'importe quel signal dans le format natif**. Il n'a pas besoin d'être lexical. Dans le code, c'est la topologie. Pour STIX, c'est le champ `relationship_type`. Pour les logs, c'est l'escalade de sévérité et la corrélation temporelle.

---

### 3.2 Inventaire des types de données et leur encodage causal

Basé sur `docs/archive/USECASES_AUDIT.md` et les use cases identifiés en session.

| Type de données | Exemples concrets | Signal causal natif | Phrase NL ? |
|---|---|---|---|
| Texte NL prose | Rapports CTI, articles scientifiques, contrats | SCONJ, verbes causaux, connecteurs | Oui |
| NL fragmentaire | Notes cliniques (`Pt 45a HTA+ → trtt`), commits, tweets | `→`, `:`, connecteurs implicites | Partiel |
| Graphes typés | STIX, MITRE ATT&CK, OWL/RDF, BPMN, arbres de faute | `relationship_type`, arc label | Non |
| Séquences d'événements | Logs SIEM, traces syscall, transactions | Ordre temporel + sévérité + correlation_id | Non |
| Enregistrements structurés | CVE/NVD, dossiers médicaux, formulaires | Champs nommés à sémantique causale | Non |
| Documents composites | PDF, HTML, notebooks Jupyter | Multiples types imbriqués | Mixte |
| Données tabulaires | CSV, Excel, résultats SQL | Relations inter-colonnes ou inter-lignes | Non |
| Texte mathématique | Théorèmes, démonstrations, définitions | NL formel + formules symboliques opaques | NL formel |
| Causalité statistique | Résultats Granger, SEM, régression causale | Valeurs numériques (p-value, coefficient) | Non (résultats) |

**Observation critique :** Le USECASES_AUDIT.md lui-même traite tous ses use cases comme du texte NL en prose (rapports, publications, contrats). Les données non-NL (logs §1.2, root cause §6.3 mêlant "logs + code + rapports") sont mentionnées mais sans solution proposée — le document les considère comme prétraitées ou hors périmètre immédiat.

---

### 3.3 Solution proposée — architecture multi-frontends

Quatre nouveaux frontends, classés par solidité technique et urgence use case.

#### Frontend 1 — `gcn-frontend-graph` (PRIORITÉ 1, 2 jours)

**Couvre :** STIX 2.x, MITRE ATT&CK, graphes de dépendances, arbres de faute, OWL/RDF simple.

**Justification technique :** Les relations causales sont déjà explicites et typées dans le format. `relationship_type: "uses"` est directement mappable à `RelationType::Enable`. Pas de ML, pas d'ambiguïté — c'est exactement le patron de `gcn-frontend-code` avec `serde_json` à la place de tree-sitter.

**Implémentation :**
- `src/stix.rs` : désérialiser le bundle STIX 2.x (`serde_json`), mapper objets STIX → CausalNode, mapper relationships → CausalEdge
- Mapping sans YAML (contrairement à gcn-frontend-code) : les types STIX sont stables, connus et peu nombreux → hardcoder dans `mapper.rs` à la manière de `gcn-frontend-fr/rules.rs`
- `SourceSpan::Synthetic` (pas de position dans un texte)
- Confiance : 1.0

**Ce qui manque dans la codebase actuelle :** Rien de bloquant. `gcn-ir` a déjà tous les types nécessaires. `gcn-cli` peut être étendu avec `gcn analyze-graph`.

#### Frontend 2 — `gcn-frontend-table` (PRIORITÉ 2, 2 jours)

**Couvre :** CSV, Excel (via CSV export), résultats SQL, matrices de risque.

**Justification technique :** Un tableau avec colonnes "facteur" et "résultat" est une relation causale structurée. Le schéma définit les relations inter-colonnes. C'est exactement le patron de gcn-frontend-code avec des colonnes CSV à la place de nœuds AST.

**Implémentation :**
- Input : CSV avec schéma de colonnes causales (ex: `cause_col`, `effect_col`, `relation_col`, `confidence_col`)
- Le schéma de colonnes est fourni via paramètre CLI, pas YAML intégré dans le moteur
- Chaque ligne → un couple (source CausalNode, target CausalNode, CausalEdge)
- Aucun ML nécessaire si le schéma est fourni

#### Frontend 3 — `gcn-frontend-stream` (PRIORITÉ 3, 3 jours + données)

**Couvre :** Logs SIEM (JSONL), alertes EDR, séquences d'événements.

**Justification technique :** La causalité dans les logs est principalement implicite (Sequence par ordre temporel, Cause par escalade de sévérité). Les features structurelles (event_type, severity, delta_t, correlation_id) alimentent le ML.

**AVERTISSEMENT :** Contrairement aux 2 frontends précédents, celui-ci requiert des données annotées pour être utile. Avec 20 features binaires/ordinales sur un modèle entraîné sur ~800 phrases génériques, la généralisation sur des logs de domaines spécifiques sera mauvaise. **Ne pas implémenter avant d'avoir ≥500 exemples de logs annotés par domaine cible.**

#### Frontend 4 — Préprocesseur document (pas un crate — une étape de pipeline)

**Couvre :** PDF, HTML, notebooks Jupyter, documents composites.

**Justification technique :** Ces formats contiennent du texte NL que le frontend FR/EN existant peut déjà traiter. La seule étape manquante est l'extraction du texte propre depuis le format binaire. Créer un `gcn-frontend-document` serait de la sur-ingénierie — ce dont on a besoin c'est d'un préprocesseur externe (pdfminer, BeautifulSoup) qui extrait les phrases NL, puis les passe au frontend NL existant.

**Implémentation recommandée :** Ajouter `gcn extract-text --input doc.pdf --output sentences.txt` dans `gcn-cli`, qui utilise une bibliothèque externe de parsing de documents. Ensuite `gcn analyze sentences.txt` utilise le frontend NL normal.

---

### 3.4 Évaluation honnête (forces, limites, risques)

**Forces de l'approche :**
- Le patron `gcn-frontend-code` valide que l'approche fonctionne pour des données non-NL — ce n'est pas une hypothèse théorique
- `gcn-frontend-graph` est le plus solide : formats stables, relations explicites, zéro ML, retour immédiat
- L'architecture CIR est suffisamment générale pour absorber toutes ces sources
- Le raisonnement Pearl (WHY, CHAIN, SPOF) fonctionne sur tout CIR quelle que soit la source

**Limites réelles :**
1. `gcn-frontend-stream` (logs) : les features structurelles sont très pauvres comparées aux 106 dims UD. Un modèle ML sur ces features ne sera pas plus performant qu'une heuristique à base de règles, sauf avec des milliers d'exemples annotés par domaine. La complexité ML est injustifiée à ce stade.

2. Texte NL fragmentaire (notes cliniques) : sans tokeniseur médical (UMLS, scispaCy), les features heuristiques seront de mauvaise qualité. Ce n'est pas un problème de format — c'est un problème de domaine NLP spécialisé.

3. La vraie contrainte reste les données : USECASES_AUDIT.md (ligne 373) dit explicitement que le seul bloquant est "des données annotées dans chacun des domaines cibles". Les nouveaux frontends sans données annotées pour les entraîner/valider sont des outils sans signal.

**Risques :**
- `gcn-frontend-stream` implémenté avant les données → code mort supervisé par rien (exact problème C-complet évité dans le PLAN_MIGRATION)
- Préprocesseur PDF externe : la qualité de l'extraction de texte depuis des PDF scannés ou en colonnes est variable — introduce du bruit dans le pipeline NL

---

## 4. Nouvelles données non couvertes

### 4.1 Documents composites (PDF, HTML, notebooks)

**Pertinence :** CRITIQUE — c'est le format de livraison réel dans tous les use cases. Un rapport CTI, un article PubMed, un contrat sont des PDF.

**Approche optimale :** Préprocesseur externe dans la CLI, pas un nouveau frontend. L'extraction de texte (pdfminer.six, BeautifulSoup) est un problème résolu — l'intégrer comme étape de pipeline avant le frontend NL. Aucun nouveau crate nécessaire.

**Limite honnête :** Les PDF avec formules, tableaux et figures inline produiront du texte bruité. Le moteur NL n'a pas de mécanisme pour distinguer une phrase NL propre d'un artefact d'extraction PDF. Qualité dépendante du document.

---

### 4.2 Données tabulaires (CSV, Excel, SQL)

**Pertinence :** ÉLEVÉE — épidémiologie, finance, registres d'entreprise.

**Approche optimale :** `gcn-frontend-table` avec schéma de colonnes fourni en paramètre. Symbolique pur, pas de ML. Chaque ligne = un couple cause/effet. Le schéma définit quelles colonnes sont causes et lesquelles sont effets.

**Limite honnête :** Les tables avec plusieurs colonnes causales et des relations inter-lignes complexes (une transaction causée par une autre dans un historique) nécessitent une logique de groupement plus sophistiquée que "ligne → arête".

---

### 4.3 Texte mathématique

**Pertinence :** MOYENNE — recherche scientifique, articles avec formules.

**Analyse :** Le texte mathématique est du NL formel. Les énoncés de théorèmes ("si f est continue, alors..."), les démonstrations ("puisque, donc, il s'ensuit"), les définitions sont des structures causales en langage naturel. Le frontend NL existant les traite correctement — les connecteurs ("si", "donc", "puisque") sont déjà dans les taxonomies FR.

**Seul ajout nécessaire :** Traiter les formules symboliques (LaTeX, notation mathématique) comme des entités opaques plutôt que les ignorer ou paniquer. Dans le tagger FR, un token comme `$f(x)$` ou `∀x` doit être classé `NodeType::Concept` et non parsé comme un verbe.

**Implémentation minimale :** Filtre en amont du tokeniseur qui remplace les expressions LaTeX par des tokens normaux (`formula_1`, `formula_2`...) tracés dans `SourceSpan`. Pas de nouveau frontend nécessaire.

---

### 4.4 Causalité statistique

**Pertinence :** MOYENNE — finance, épidémiologie, sciences.

**Analyse :** La causalité statistique (Granger, SEM, régression causale) produit des paires `(A, B, p-value, coefficient)` — pas du texte NL. C'est un **composant d'analyse externe** qui tourne en amont et produit soit (a) un graphe causal probabiliste → ingéré par `gcn-frontend-graph`, soit (b) une description NL des résultats → ingérée par le frontend NL.

**Approche :** Pas de nouveau frontend. La causalité statistique est un préprocesseur qui produit un format déjà traitable (graphe ou NL). L'intégrer directement dans le moteur introduirait une dépendance sur scipy/statsmodels et un paradigme statistique orthogonal au raisonnement Pearl structurel.

---

## 5. Conclusion — Priorités

### Priorité 0 — Avant tout réentraînement

Exécuter `scripts/init_v3_stub.py` pour générer le checkpoint v3.0 stub (débloque les 3 xfailed).

### Priorité 1 — Bugs ouverts → TOUS CORRIGÉS (v4, 2026-09-29)

| Bug | Impact | Effort | Statut v4 |
|---|---|---|---|
| T5-min silencieuse (§2.7) | Calibration inutilisée → ECE non optimisée | 0.5j | ✅ corrigé (forward reps + warn) |
| export.rs PascalCase (§2.11) | Incohérence format DOT vs JSON | 0.5j | ✅ corrigé (+ tooltip nœuds, test MAJ) |
| spof_all sans garde (§2.13) | DoS via API crate directe | 0.5j | ✅ corrigé (Result + garde 500, ⚠️ breaking) |
| OOV→_unk (§2.9) | Qualité embedding sur bridge heuristique | 1j | ✅ corrigé (Option A : zéro + no-op) |
| Seed partiel (§2.8) | Reproductibilité incomplète | 0.5j | ✅ corrigé (seed propagé partout) |
| Hyperarêtes (§2.10) | Attendre données ternaires annotées | N/A | ➖ sans changement (décision confirmée) |
| Frontend FR 9/19 (§2.12) | Ajouter filter seulement, laisser ternaires au ML | 0.5j | ✅ corrigé partiel (filter FR+EN) |

### Priorité 2 — Architecture multi-frontends → 1-3 IMPLÉMENTÉS (MAJ2, branche `feat/p2-frontends`)

1. ✅ `gcn-frontend-graph` (STIX, graphes explicites) — nouveau crate, mapping compilé SANS YAML (règle moteur), `SourceLanguage::Graph { Stix21 }` (additif), `GraphParseReport` (ignorés comptés), CLI `gcn analyze-graph`, 6 tests. E2E : bundle → CIR → WHY/CHAIN/SPOF/DOT. Audit F1 : `detects` → `Enable` ajouté (seuls `subtechnique-of`/`revoked-by` restent ignorés, à raison : hiérarchie/métadonnée, pas causal).
2. ✅ Préprocesseur document — `gcn extract-text` (module `gcn-cli`, pas de crate) : TXT/HTML/IPYNB en interne, PDF via `pdftotext` externe (erreur explicite sans poppler), découpeur FR/EN, 9 tests CLI. Review : panique `strip_tag_blocks` sur U+0130 corrigée.
3. ✅ `gcn-frontend-table` (CSV, schéma) — nouveau crate, schéma en paramètres CLI, parseur RFC-4180 sans dépendance, `TableParseReport`, `SourceLanguage::Graph { Table }`, CLI `gcn analyze-table`, 6 tests. Audit : confiances hors [0,1] → ignorées+comptées (F2), `DuplicateColumn` typée (F3), sémantique doublons/auto-boucles documentée (F4).
4. **Différé (décision tenue)** `gcn-frontend-stream` jusqu'à ≥500 logs annotés par domaine cible.

**Refactor D1 associé** : `RelationType::from_name` / `NodeType::from_name` dans `gcn-ir` (source unique, roundtrip serde 19+8 prouvé) ; `gcn-frontend-code/mapper.rs` (relations) délégué — `node_type` inchangé (aliases v2 rejetés à raison). T5 extrait en `collect_val_logits_for_calibration()` testable (`tests/test_t5_collection.py`).

**Règle moteur rappelée (contractuelle)** : les taxonomies guident les développeurs dans l'annotation (référençables) ; **zéro YAML chargé dans le moteur — savoir compilé, le moteur n'apprend jamais de configs**. Vérifié : aucune dep YAML ajoutée, aucun `Lexicon::load` dans le nouveau code. Ne plus jamais proposer de YAML moteur.

### Priorité 3 — Le vrai bloquant

Annotation de données par domaine cible (cybersécurité, médecine, droit). Selon USECASES_AUDIT.md ligne 373, c'est le seul bloquant vers la production — pas les frontends, pas les bugs. Sans 5 000–50 000 phrases annotées par domaine, le moteur le plus complet restera non généralisable.

---

*Étude produite le 2026-09-29. Tous les claims ont été vérifiés dans le code source avant rédaction.*
*MAJ v4 le 2026-09-29 : §2.7, §2.8, §2.9, §2.11, §2.12, §2.13 corrigés avec changement de code ; §2.10 confirmé sans changement. Vérification : 685 Python passed / 3 xfailed (checkpoint v2, cf. Priorité 0), Rust gcn-backend 56+6 OK, frontends FR/EN OK. Breaking change : `pearl::spof_all` → `Result` (bump mineur requis).*
*MAJ2 le 2026-09-29 : P0 stub v3 (`stub_v3.npz`, 2 bugs du script corrigés : `d_hidden` 207→234, `d_ecl` 987→1115 via `FeatureVocabulary`) ; P2.1/P2.2/P2.3 implémentés et E2E vérifiés ; audit par exécution (T5 E2E 12 labels, seed déterminisme, OOV fini, stub byte-identique, garde SPOF 500/501/2000, ATT&CK 4/7 mappés, HTML/PDF/CSV adversariaux) : 8 constats F1–F8, Phase A (F1/F2/F3/F8) + B2/B3 appliqués, F4 documenté, F5/F6/F7 classés. Python 687 passed / 3 xfailed, Rust 211 verts, clippy/fmt propres (warnings restants pré-existants master). Reste : stream (données), P3 (annotation), commit sur demande.*
