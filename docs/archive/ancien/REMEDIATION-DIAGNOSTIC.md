# DIAGNOSTIC COMPLET — Déficit Sémantique Moteur GCN

**Date** : 2026-09-25  
**Révision audit factuel** : 2026-09-25 (chronologie, chiffres et preuves §7 corrigés ; voir notes `[AUDIT]`)  
**Clôture remédiation** : 2026-09-25 — statut **VALIDÉ, EN COURS D'OPTIMISATION** (voir §12)  
**Criticité** : ⚠️ MOYENNE (gain mesuré +45%, mais loin de cible 0.40)  
**Status** : 🟢 VALIDÉ — 4 scénarios exécutés, résultats mesurés (`BENCHMARK.md` §11)

---

## 1. IDENTIFICATION DU PROBLÈME

### 1.1 Symptôme Observé
**Performance plafonnée autour de ~0.47 `val_edge_macro_f1` sur la meilleure configuration historique**
(`BENCHMARK.md` §2, combinaison #1 : GAT + bidi + oversampling C1, 2026-09-18).

> `[AUDIT]` Précisions de mesure (sources vérifiées le 2026-09-25) :
> - Le 0.4676 date du 2026-09-18 (ancien harnais) ; ré-évalué avec le harnais actuel, le même
>   checkpoint donne **0.1867** (`BENCHMARK.md` §9b) et les runs récents donnent **0.1615–0.2508** (§9a).
> - Dernier résultat comparable : modèle supervision v2 = **0.3124 val / 0.3392 test**, baseline prod
>   mêmes données = 0.1880 (`BENCHMARK.md` §10).
> - **« MLPEncoder = 0.47 » n'a jamais été mesuré** : 0 occurrence de MLPEncoder dans `BENCHMARK.md`.
>   Le 0.47 est une *attente* héritée (`Integration-post-pub.txt:571` → `CORRECTIONS-Integration-post-pub.md:296`),
>   pas une mesure. Ne pas comparer §1 et §9 de `BENCHMARK.md` (deux règles de mesure différentes).

### 1.2 Cause Racine
Le moteur GCN utilise **uniquement des features syntaxiques** (POS, dep_rel, morph) au niveau clause
et **ignore la sémantique lexicale des lemmes** (hors connecteurs, voir nuance ci-dessous).

**Exemple concret** :
```
Phrase 1 : "La pluie cause l'inondation"
Phrase 2 : "Le soleil cause l'évaporation"

Représentation moteur actuel (clause, gcn-python/src/gcn_python/layer1/features.py:251-263) :
  Phrase 1 : NOUN ←nsubj VERB →obj NOUN   (POS + dep_rel + morph : identiques)
  Phrase 2 : NOUN ←nsubj VERB →obj NOUN   (POS + dep_rel + morph : identiques)

  → Vecteurs de clause IDENTIQUES pour le moteur (pas de lemme dans le vecteur)
  → Impossible de distinguer sémantiquement
```

> `[AUDIT]` Nuance : `vectorize_connector` (`features.py:288-292`) inclut un one-hot du **lemme du
> connecteur** (`vocab.connector_lemmas`, vocabulaire fermé). Un signal lexical borné existe donc
> sur les connecteurs ; le déficit porte sur les lemmes des clauses (sujet/verbe/objet).
> `gcn-python/README.md:325` documente déjà ce déficit et recommande `--fasttext wiki.fr.bin`.

### 1.3 Impact Mesuré
- **Edge F1 (val, harnais actuel)** : 0.16–0.25 (runs §9), 0.3124 (supervision v2, §10) ; meilleur
  historique 0.4676 (ancien harnais, §1). Cible prod : > 0.40.
- **Node F1 (val)** : 0.313 max (`BENCHMARK.md` §1, run #2). Cible prod : > 0.60 — bloqué par les
  données (types rares : 4–35 exemples, `BENCHMARK.md` §5), pas seulement par les features.
- **Patterns causaux** : syntaxiques + lemme du connecteur (vocabulaire fermé).
- **Sémantique lexicale des clauses** : 0% capturée (par défaut, `--embedding-dim 0`).

> `[AUDIT]` Les valeurs « 0.47 (plafond dur) » et « node acc 0.65 » de la version précédente
> mélangeaient métriques train/val et chiffres non mesurés (voir §5 révisé).

---

## 2. SOLUTIONS EXISTANTES (NON ACTIVÉES PAR DÉFAUT `[AUDIT : titre corrigé]`)

### 2.1 WordEmbedding (gcn-python)

**Status** : ✅ Implémenté, ❌ Désactivé par défaut (`--embedding-dim 0`)

**Localisation** :
- Code : `gcn-python/src/gcn_python/layer1/embedding.py` (227 lignes)
- Intégration : `gcn-python/src/gcn_python/layer1/features.py:220-280`
  (`vectorize_clause` : concatène l'embedding du `root_lemma`, + pooling A et sujet/objet B)
- CLI : `gcn-python/src/gcn_python/training/train.py:60-67` (options), validations `train.py:258-285`
- Checkpoint : clés `word_emb_E` + `_word_emb_vocab_json` (`training/checkpoint.py:152,244,450-453`)

**Capacités** :
1. **Embeddings apprenables** : Table (vocab_size, d_emb) apprise par backprop
2. **FastText pré-entraîné** : Chargement cc.fr.300.bin (2M mots)
3. **Fine-tuning** : Option `--fine-tune-embeddings`
4. **Pooling avancé** : root/mean/max sur tokens clause
5. **Subject/Object embeddings** : Ajout embeddings sujet/objet explicites

**Performance attendue (HYPOTHÈSE — non validée expérimentalement, et partiellement contredite, voir §5bis)** :
- Apprenables : 0.55-0.60 edge_f1 (+17-28%)
- FastText gelé : 0.60-0.65 edge_f1 (+28-39%)
- FastText fine-tuné : 0.62-0.68 edge_f1 (+32-45%)

**Historique réel du code** (vérifié `git log`, dépôt créé le 2026-09-14) :
- Ajout : `c7bb578` — 2026-09-16 (« feat(Phase10) : correction des 12 défauts structurels moteur (v1.2.0) »)
- Rapport : `689513f` — 2026-09-16 (« embeddings réduisent le biais (54% → 99% node acc) »),
  **il y a 9 jours, pas 6 mois**. Ce rapport précise aussi : « Edge classification reste à ~30%
  (problème ouvert) » — mise en garde à ne pas ignorer.

> `[AUDIT]` Correction : la version précédente attribuait à tort l'introduction du code au commit
> `689513f` (qui ne modifie qu'un rapport) et le datait de « 2025-12 ». La mention « node acc
> 54% → 99% » concerne la **node accuracy d'entraînement** des runs pré-v2.3.0, identifiée depuis
> comme **mémorisation pure** (`BENCHMARK.md` §4 : `val_node_f1 ≈ 0.15–0.20` sur ces runs).

---

### 2.2 gcn-transformers (Nouveau Projet)

**Status** : 🆕 Créé récemment, ⚠️ v1.0 inefficace

**Objectif déclaré** : Résoudre le déficit sémantique via Transformers pré-entraînés

**Réalité v1.0** (attentes documentées, **jamais mesurées** — aucun entraînement n'a été exécuté) :
- ❌ Projection features UD → 768-dim (toujours syntaxe)
- ❌ Performance attendue identique à la baseline (~0.47 edge_f1, hypothèse non mesurée)
- ❌ Coût estimé élevé (**« 12× plus lent, 35× plus lourd » : chiffres issus de
  `gcn-transformers/BENCHMARK-README.md`, eux-mêmes jamais mesurés** — aucune exécution, pas de GPU)
- ❌ Complexité accrue sans bénéfice démontré

**Réalité v1.1 (futur)** :
- ✅ Tokenization texte brut → embeddings contextuels
- ✅ Performance attendue >0.70 edge_f1
- ⚠️ Coût reste élevé (GPU requis)
- ⚠️ Complexité élevée

**Diagnostic** : Solution sur-dimensionnée créée **sans identifier que WordEmbedding existait**.

---

## 3. HISTORIQUE CHRONOLOGIQUE

### Timeline (vérifiée `git log` — le dépôt date du 2026-09-14, pas de 2025)

```
2026-09-14 : Création gcn-python (initial commit)
             └─ Encodeurs syntaxe seule (79-dim clause)
             └─ Baseline R-GCN : 0.2333 val_edge_macro_f1 (BENCHMARK.md §1, run #11)

2026-09-16 : Ajout WordEmbedding (commit c7bb578, « Phase10 »)
             └─ Implémentation complète (apprenables + FastText + pooling A/B)
             └─ Rapport 689513f : node acc train 54% → 99% sur runs pré-v2.3.0…
             └─ …mais « Edge classification reste à ~30% (problème ouvert) »

2026-09-18 : Benchmark : meilleur historique 0.4676 (GAT + bidi + oversamp C1)
             └─ Runs pré-v2.3.0 avec emb50 : train_edge_f1 0.475–0.485, train_node_acc 93–99%
             └─ ⚠️ Sans val monitoring → mémorisation, pas généralisation (BENCHMARK.md §1b, §4)

2026-09-20 : Filtrage des fichiers de développement internes (commit 89a62aa)
             └─ docs/EXPERIMENT_990_DATASET.md (cité par 689513f) supprimé du dépôt

2026-09-21 : Re-run harnais actuel (BENCHMARK.md §9) : échelle 0.16–0.25 val_edge_f1
             └─ Expérience wiki.fr 300d : 5 epochs seulement, non concluant (run #17)

2026-09-24 : PROGRESS.md signale déjà le problème (§« Comparaison pré-v2.3.0 ») et
             recommande les embeddings pré-entraînés en priorité n°1

2026-09-25 : Création gcn-transformers (projet jeune) + ce diagnostic
```

> `[AUDIT]` La timeline précédente (« 2025-09 », « 2025-12 », « 2026-01 », « 2026-03 »,
> « 6 mois de sous-performance ») était incompatible avec l'historique git (11 jours) et a été
> remplacée. Conséquence : le « coût d'opportunité de 6 mois » n'existe pas ; le vrai coût est
> l'effort investi dans gcn-transformers v1.0 (jamais exécuté) au lieu de valider les embeddings.

### Commits Clés

```bash
c7bb578 - 2026-09-16 feat(Phase10) : AJOUT RÉEL de layer1/embedding.py (12 défauts structurels, v1.2.0)
689513f - 2026-09-16 Rapport : embeddings réduisent le biais (54% → 99% node acc train),
          avec mise en garde "Edge classification reste à ~30% (problème ouvert)"
27a69af - 2026-09-21 fix(eval+backward): gradient embedding post-RGCN, training=False en éval
c05ea22 - 2026-09-24 feat(v2.5.0): 4 phases architecturales + correctifs post-audit (385 tests)
```

**Observation** : WordEmbedding mentionné comme correction de biais, mais `PROGRESS.md`
(2026-09-24) et `BENCHMARK.md` (§7 rec. n°3) l'ont depuis promu au rang de piste prioritaire.
Le déficit de promotion pointé ici était donc déjà en cours de correction.

---

## 4. CAUSES RACINES (ANALYSE 5 POURQUOI)

### Pourquoi 1 : Pourquoi WordEmbedding n'est-il pas utilisé ?
**Réponse** : Pas activé par défaut (`--embedding-dim 0`), et les checkpoints de production
actuels (`model_v2.4.0.npz`, runs §9) sont entraînés sans.

### Pourquoi 2 : Pourquoi pas activé par défaut ?
**Réponse** : Introduit comme « feature optionnelle » de réduction de biais le 2026-09-16 (projet
âgé de 2 jours) ; les résultats pré-v2.3.0 avec embeddings se sont révélés être de la
mémorisation (`BENCHMARK.md` §4), ce qui a — à juste titre — freiné son activation en production.

### Pourquoi 3 : Pourquoi l'apport sémantique n'est-il toujours pas tranché ?
**Réponse** : Les seules expériences embeddings avec val monitoring sont : emb50 aléatoires
(**négatif** : 0.318 → 0.222) et wiki.fr 300d sur **5 epochs** (non concluant). Le protocole
correct est documenté (`BENCHMARK.md` §5 « Chemin pour tester correctement ») mais **jamais
exécuté** (100 ep × 12 min ≈ 20 h CPU, ou GPU requis).

### Pourquoi 4 : Pourquoi gcn-transformers a-t-il été créé sans exploiter cette piste ?
**Réponse** : Analyse incomplète de l'existant (WordEmbedding + `BENCHMARK.md` §3-5 non pris
en compte) ; objectifs hérités d'un plan contenant des chiffres non mesurés
(`CORRECTIONS-Integration-post-pub.md`).

### Pourquoi 5 : Pourquoi les chiffres non mesurés circulent-ils comme des faits ?
**Réponse** : Chaîne de copies sans traçabilité (`Integration-post-pub.txt` → `CORRECTIONS-*` →
`BENCHMARK-README.md` de gcn-transformers → ce diagnostic, avant révision).

**CAUSE RACINE FONDAMENTALE** : **Absence de traçabilité mesure → affirmation, et non-exécution
des protocoles de validation déjà documentés** (`BENCHMARK.md` §5, `train.py` CLI prête).

---

## 5. IMPACT QUANTIFIÉ (mesures réelles + hypothèses étiquetées)

### 5.1 État mesuré (sources : `BENCHMARK.md`, `PROGRESS.md`)

| Métrique | Baseline prod (`model_v2.4.0`, §1) | Harnais actuel, mêmes données (§9b/§10) | Supervision v2 (§10) | Cible prod |
|----------|-------------------------------------|------------------------------------------|----------------------|------------|
| val_edge_macro_f1 | 0.4676 (ep92, ancien harnais) | 0.1867 (mêmes poids) / 0.1880–0.2508 (runs neufs) | **0.3124** (test 0.3392) | > 0.40 |
| val_node_macro_f1 | 0.273 | 0.127–0.162 | 0.2357 | > 0.60 |
| val_graph_exact_match | 0.123 | 0.009 | — | > 0.20 |

> `[AUDIT]` Le 0.4676 n'est **pas** un « plafond dur du moteur » : `BENCHMARK.md` §9 montre que
> l'échelle a changé avec les données/le harnais, et §5 attribue le blocage node_f1 aux
> **données** (types rares : 4–35 exemples → ~800 phrases à annoter), pas aux seuls hyperparamètres.

### 5.2 Résultats mesurés (2026-09-25, `BENCHMARK.md` §11)

**Dataset** : `DATA/supervision/train_v2/` — 1665 train / 350 val, GAT+bidi+weighted-loss,
20 epochs, seed 42. Détails complets : `BENCHMARK.md` §11.

| Solution | best val_edge_f1 | Δ vs baseline | Δ relatif | Statut |
|----------|-----------------|---------------|-----------|--------|
| Baseline (dim 0) | 0.1579 | — | — | Mesuré ✅ |
| Emb128 apprenables | 0.2120 | +0.054 | +34% | Mesuré ✅ |
| wiki.fr 300d gelé | 0.2027 | +0.045 | +28% | Mesuré ✅ |
| wiki.fr 300d fine-tuné | **0.2292** | **+0.071** | **+45%** | Mesuré ✅ |
| Transformers v1.1 | — | — | — | Code jamais exécuté |

**Les projections 0.55–0.65 de la version précédente sont abandonnées.** Le meilleur résultat
mesuré (0.2292) reste loin de la cible production (> 0.40). Les embeddings apportent un gain
réel mais modeste ; le bloquant principal reste les données (§5.1, ~800 phrases).

> Note : 20 epochs ne suffisent pas — aucune configuration n'a convergé (`BENCHMARK.md` §11e).
> Le fine-tuning est encore en progression à ep20. Des runs 50–100 ep sont nécessaires.

### 5.3 Coûts réels constatés

```
Effort gcn-transformers v1.0 (jamais exécuté) :
  - Code + tests + documentation volumineuse, 0 exécution d'entraînement
  - Résultat : 0 mesure, chiffres du README non mesurés

Effort de documentation redondante :
  - 5 fichiers REMEDIATION-* générés le 2026-09-25 en ~15 min, propageant
    les mêmes chiffres non mesurés (corrigés par la présente révision)
    → coût = risque de décision sur de fausses bases, pas du temps CPU

Opportunité manquée (réelle, courte) :
  - Depuis le 2026-09-16, aucun run avec embeddings pré-entraînés + val monitoring
    n'a été lancé, alors que le protocole existe (BENCHMARK.md §5)
```

---

## 5bis. ÉVIDENCE CONTRAIRE ET NUANCES `[AUDIT : section ajoutée, mise à jour post-validation]`

Le gain mesuré (+34% à +45%) est réel mais doit être lu avec les nuances suivantes :

1. **Embeddings aléatoires 50d sur petit dataset : effet NÉGATIF historique.** GAT + bidi +
   emb50rand sur 536 phrases : val_edge_macro_f1 **0.318 → 0.222 (−0.096)**
   (`BENCHMARK.md` §1, run #12). **Inversé sur 1665 phrases** : emb128 apprenables = +34%
   (`BENCHMARK.md` §11). La taille du dataset est déterminante.

2. **Overfitting wiki.fr 300d.** Les configs C/D ont le plus gros gap loss (~0.5 à ep20,
   `BENCHMARK.md` §11d). Les 300 dimensions supplémentaires mémorisent le train sans
   proportionnellement généraliser. À surveiller sur runs longs.

3. **99.4% node accuracy historique = mémorisation** (`BENCHMARK.md` §4) :
   mêmes runs → `val_node_f1 ≈ 0.15–0.20`. Règle absolue : ne jamais évaluer sans val séparé.

4. **Le bloquant documenté n°1 reste les données** (`BENCHMARK.md` §5) :
   5 types de nœuds à 4–35 exemples, val_node_f1 max 0.1519 (emb128, §11).
   Les embeddings améliorent les edges mais ne résolvent pas le déficit de nœuds rares.
   Annoter ~800 phrases ciblées reste indispensable.

5. **20 epochs insuffisantes.** Aucune config n'a convergé. Le fine-tuning (D) est encore
   en progression active à ep20 (val_edge_f1 = best à la dernière epoch). Les résultats
   actuels sont des **bornes inférieures**, pas des performances finales.

---

## 6. VALIDATION EXPÉRIMENTALE — EXÉCUTÉE 2026-09-25

### 6.1 Statut des scénarios

Les 3 scénarios définis en §8 + un 4e ajouté ont été **exécutés** le 2026-09-25.
Résultats complets : `BENCHMARK.md` §11. CSVs : `benchmark-validation/v2_*.csv`.

| Scénario | Statut | Config | best val_edge_f1 | Critère atteint ? |
|----------|--------|--------|-----------------|-------------------|
| §8.1 — Emb apprenables | ✅ Exécuté | emb128, 20ep | 0.2120 | ✅ +0.054 > baseline+0.05 |
| §8.2 — FastText gelé | ✅ Exécuté | wiki.fr 300d frozen | 0.2027 | ❌ 0.2027 < emb128+0.03 (0.242) |
| §8.3 — Fine-tuning | ✅ Exécuté | wiki.fr 300d unfrozen | **0.2292** | ✅ 0.2292 ≥ frozen 0.2027 |
| Baseline (référence) | ✅ Exécuté | dim 0, 20ep | 0.1579 | — |

### 6.2 Ce que les résultats montrent

1. **Scénario 1 VALIDÉ** : emb128 surpasse la baseline de +0.054 (> seuil +0.05).
   Les embeddings apprenables apportent un gain mesurable sur ce dataset, contrairement
   aux résultats antérieurs sur 536 phrases (run #12, §1).

2. **Scénario 2 NON VALIDÉ** : wiki.fr gelé (0.2027) < emb128 (0.2120).
   Le gel des poids pré-entraînés ne surpasse pas l'apprentissage from scratch.
   Cause probable : vocabulaire wiki.fr trop généraliste (22k mots, couverture partielle
   du vocabulaire causal spécialisé).

3. **Scénario 3 VALIDÉ** : fine-tuning (0.2292) > gelé (0.2027).
   L'adaptation des poids pré-entraînés au corpus cible est le meilleur compromis.

4. **Convergence non atteinte** : 20 epochs insuffisantes (val_loss encore décroissante).
   Le potentiel réel de chaque config n'est pas encore connu.

### 6.3 Commandes exécutées

```bash
# Baseline
gcn-train --data-dir DATA/supervision/train_v2/train/ \
  --val-dir DATA/supervision/train_v2/val/ \
  --epochs 20 --lr 0.001 --seed 42 \
  --weighted-loss --use-attention --bidirectional \
  --embedding-dim 0 \
  --output benchmark-validation/v2_baseline_20ep.npz \
  --log-csv benchmark-validation/v2_baseline_20ep.csv

# Emb128 apprenables
gcn-train ... --embedding-dim 128 \
  --output benchmark-validation/v2_emb128_20ep.npz \
  --log-csv benchmark-validation/v2_emb128_20ep.csv

# wiki.fr 300d gelé
gcn-train ... --embedding-file gcn-python/models/wiki.fr.vec \
  --freeze-embeddings \
  --output benchmark-validation/v2_fasttext_frozen_20ep.npz \
  --log-csv benchmark-validation/v2_fasttext_frozen_20ep.csv

# wiki.fr 300d fine-tuné
gcn-train ... --embedding-file gcn-python/models/wiki.fr.vec \
  --output benchmark-validation/v2_fasttext_finetune_20ep.npz \
  --log-csv benchmark-validation/v2_fasttext_finetune_20ep.csv
```

---

## 7. PREUVES DOCUMENTAIRES

### 7.1 Code WordEmbedding Existe (sorties réelles, vérifiées 2026-09-25)

```bash
# Fichier principal
$ wc -l gcn-python/src/gcn_python/layer1/embedding.py
227 gcn-python/src/gcn_python/layer1/embedding.py

# Tests existants (unitaires : OK — à distinguer d'une validation en production)
$ grep -rn "WordEmbedding" gcn-python/tests/ --include=*.py
test_lot1_corrections.py:    we = WordEmbedding(d_emb=...)   (4 occurrences)
test_pipeline.py:            we = WordEmbedding(d_emb=8, seed=0)
test_update_phases.py:       we = WordEmbedding(d_emb=1) ; load_from_fasttext(..., d_emb=300, frozen=True)

# CLI existe (texte réel de gcn-train --help)
$ gcn-train --help | grep -A2 "embedding-dim"
  --embedding-dim INTEGER         Activer les word embeddings apprenables
                                  (S1/S2). 0 = désactivé.  [default: 0]
```

> `[AUDIT]` La version précédente citait « 184 lignes », un test `d_emb=50` inexistant et un
> texte d'aide différent (« Dimension embeddings apprenables ») : trois sorties non reproductibles.

### 7.2 Quels Modèles Utilisent Déjà les Embeddings (vérifié 2026-09-25)

```bash
$ python3 -c "... vérifie la clé 'word_emb_E' dans chaque .npz ..."
48 checkpoints trouvés, dont 14 AVEC embeddings, ex. :
  gcn-datasets/checkpoints/prod_v1.npz, gat_emb50.npz, rgcn_emb50.npz,
  bidi_emb50.npz, gat_bidi_oversamp_emb50rand.npz, gat_bidi_oversamp_emb300_5ep.npz,
  gat_cl_v2.npz, gat_cl_bwd.npz, ...
34 SANS embeddings, ex. : model_v2.4.0.npz (prod), runs §9, model_baseline.npz
```

> `[AUDIT]` La version précédente affirmait « tous : NO embeddings » à partir d'un `find`
> dans `../models`, **répertoire inexistant**, et ignorait `gcn-datasets/checkpoints/`.
> Nuance : les checkpoints AVEC embeddings relèvent de runs pré-v2.3.0 **sans val monitoring**
> (mémorisation, §5bis) — ils prouvent l'usage, pas l'efficacité.

### 7.3 Documentation : Recommandation Présente, Protocole Non Exécuté

```bash
# Le README documente DÉJÀ le déficit et la solution :
$ grep -n "fasttext" gcn-python/README.md
325:| Features syntaxiques uniquement | [...] | Ajouter `--fasttext wiki.fr.bin` pour injecter des embeddings sémantiques. |
```

> `[AUDIT]` La version précédente citait une ligne `word_embedding=None` **absente du README**
> et concluait « aucune section ». Conclusion corrigée : la recommandation existe
> (`README.md:325`, `BENCHMARK.md` §5 et §7-rec.3, `PROGRESS.md:636`) ; ce qui manque, c'est
> **l'exécution du protocole de validation**, pas sa documentation.

---

## 8. SCÉNARIOS DE VALIDATION — RÉSULTATS

### Scénario 1 : Validation Embeddings Apprenables — ✅ VALIDÉ

**Résultat** : emb128 best val_edge_f1 = **0.2120** vs baseline **0.1579** → Δ = +0.054.
**Critère** : edge_f1 > baseline + 0.05 → **0.054 ≥ 0.05 ✅**

Les embeddings apprenables apportent un gain mesurable. Contrairement aux résultats antérieurs
sur 536 phrases (§1 run #12, emb50 : −0.096), le dataset plus grand (1665 phrases) et le
backend GAT+bidi permettent d'exploiter le signal lexical sans overfitting destructeur.

Bonus : meilleur val_node_f1 (0.1519 vs 0.1176).

---

### Scénario 2 : Validation FastText Pré-entraîné — ❌ NON VALIDÉ (20 ep)

**Résultat** : wiki.fr 300d gelé best val_edge_f1 = **0.2027** vs emb128 **0.2120** → Δ = −0.009.
**Critère** : edge_f1 > emb128 + 0.03 → **0.2027 < 0.242 ❌**

À 20 epochs, le gel des poids pré-entraînés ne surpasse pas l'apprentissage from scratch.
Cause probable : vocabulaire wiki.fr (22 218 mots, troncature) mal adapté.
**Réserve** : convergence non atteinte — un run plus long pourrait inverser le classement.

---

### Scénario 3 : Validation Fine-tuning — ✅ VALIDÉ

**Résultat** : wiki.fr 300d fine-tuné best val_edge_f1 = **0.2292** vs gelé **0.2027** → Δ = +0.027.
**Critère** : edge_f1 (fine-tuned) ≥ edge_f1 (gelé) → **0.2292 > 0.2027 ✅**

Le fine-tuning est la meilleure configuration mesurée. Il combine le bénéfice des poids
pré-entraînés (initialisation informée) avec l'adaptation au vocabulaire causal du corpus.

---

### Synthèse des scénarios

| Classement | Config | val_edge_f1 | Verdict |
|-----------|--------|-------------|---------|
| 1 | wiki.fr 300d fine-tuné (D) | **0.2292** | Meilleur edges |
| 2 | emb128 apprenables (B) | 0.2120 | Meilleur nœuds |
| 3 | wiki.fr 300d gelé (C) | 0.2027 | Intermédiaire |
| 4 | Baseline dim 0 (A) | 0.1579 | Référence basse |

**Tous restent loin de la cible production (> 0.40).** Prochaine étape : runs 50–100 epochs
+ seed-sweep + annotation 800 phrases (piste données, `BENCHMARK.md` §5).

---

## 9. MÉTRIQUES CLÉS

### Métriques Primaires
- **Edge F1 (val)** : Métrique principale (comparer à la baseline du même harnais : 0.16–0.25
  sur runs §9, 0.3124 sur supervision v2 — pas à 0.47, voir §1.1)
- **Node F1 (val)** : Secondaire (max mesuré 0.313 ; cible prod > 0.60 — piste données, `BENCHMARK.md` §5)

### Métriques Secondaires
- **Temps training/epoch** : Overhead embeddings acceptable (<50%)
- **Temps inférence/sample** : Overhead acceptable (<100%)
- **Taille checkpoint** : Embeddings ajoutent vocab_size × d_emb (~6 MB pour 50k vocab × 128-dim)

### Métriques Qualitatives
- **Interprétabilité** : Visualisation embeddings (t-SNE)
- **Généralisation** : Performance sur domaines différents
- **Robustesse** : Performance avec vocabulaire inconnu

---

## 10. RISQUES IDENTIFIÉS

### Risque 1 : WordEmbedding N'Apporte Rien (voire Dégrade) en Val
**Probabilité** : Moyenne — embeddings aléatoires mesurés à **−0.096** (`BENCHMARK.md` §1 run #12)  
**Impact** : Moyen (trancherait pour la piste « données » ou Transformers v1.1)  
**Mitigation** : Tester les **pré-entraînés à convergence** (pas les aléatoires), protocole `BENCHMARK.md` §5

### Risque 2 : Overhead Temps Training Trop Élevé
**Probabilité** : Faible (<50% attendu)  
**Impact** : Faible (acceptable)  
**Mitigation** : Optimisations backward embeddings si nécessaire

### Risque 3 : Vocabulaire Hors Dataset Trop Grand
**Probabilité** : Moyenne  
**Impact** : Moyen (vocabulaire inconnu = _unk)  
**Mitigation** : FastText pré-entraîné (2M mots couvert)

### Risque 4 : Complexité Configuration Utilisateur
**Probabilité** : Moyenne  
**Impact** : Moyen (adoption lente)  
**Mitigation** : Presets recommandés (`--preset semantic`)

---

## 11. CONCLUSION DIAGNOSTIC (mise à jour 2026-09-25 post-validation)

### Résumé
Le déficit lexical des features de clause est **réel** (vecteurs sans lemmes, §1.2) et une
solution **existe, est câblée, et a été mesurée** (WordEmbedding, CLI, checkpoints — §2.1, §7).

**Résultats de la validation (§6, §8, `BENCHMARK.md` §11) :**
- Les embeddings **améliorent les edges** : +34% à +45% relatif selon la config.
- Le **fine-tuning wiki.fr 300d** est la meilleure config mesurée (0.2292 val_edge_f1).
- Les **emb128 apprenables** sont les meilleurs pour les nœuds (0.1519 val_node_f1).
- **Aucune config n'atteint la cible production** (> 0.40).
- Les projections 0.55–0.65 sont **abandonnées**.
- **20 epochs ne suffisent pas** — convergence non atteinte sur les 4 configs.

Le vrai bloquant pour node_f1 reste **les données** (~800 phrases à annoter, `BENCHMARK.md` §5).
Les embeddings ne dispensent pas de cet effort d'annotation.

### Recommandation
1. **Poursuivre les embeddings en production** : conserver le défaut `--embedding-dim 128` et
   recommander `--embedding-file wiki.fr.vec` (sans `--freeze-embeddings`) pour les cas d'usage
   où le vocabulaire pré-entraîné est pertinent.
2. **Runs longs** : 50–100 epochs sur le fine-tuning (config D) pour mesurer la convergence.
3. **Seed-sweep** : 3 seeds (7, 42, 123) sur configs B et D pour quantifier la variance.
4. **Annoter 800 phrases** ciblant les nœuds rares — seul levier pour node_f1 > 0.30.
5. **Ne pas investir davantage dans gcn-transformers** tant que les embeddings simples
   n'ont pas été poussés à convergence — le gain potentiel restant est non négligeable.

---

## 12. CLÔTURE REMÉDIATION 2026-09-25 — STATUT : VALIDÉ, EN COURS D'OPTIMISATION

### Réalisé
1. **Embeddings non optionnels** : `gcn-train --embedding-dim` défaut 0 → 128 + warning si
   désactivé (`train.py:60-61,257-290`) ; `--fasttext`/`--embedding-file` seuls préservés ;
   round-trip checkpoint vérifié empiriquement (`word_emb_E` restauré à l'identique).
2. **Validation complète des 4 scénarios** (2026-09-25, dataset `DATA/supervision/train_v2/`,
   1665 train / 350 val, GAT+bidi+weighted-loss, 20 epochs) :
   - Baseline (dim 0) : val_edge_f1 = 0.1579
   - Emb128 apprenables : val_edge_f1 = 0.2120 (+34%)
   - wiki.fr 300d gelé : val_edge_f1 = 0.2027 (+28%)
   - wiki.fr 300d fine-tuné : val_edge_f1 = **0.2292 (+45%)**
3. **Documentation mise à jour** : `BENCHMARK.md` §11 (résultats complets, 7 sous-sections),
   `REMEDIATION-DIAGNOSTIC.md` §5.2/§6/§8/§11/§12, `gcn-python/README.md`, `PROGRESS.md`.
4. **Tests** : gcn-python 411+ passed / 4 skipped ; gcn-transformers 25 passed / 10 skipped.
5. **CSVs archivés** : `benchmark-validation/v2_*.csv` (4 fichiers, epoch-by-epoch).

### Mesures (détail `BENCHMARK.md` §11, GAT+bidi, train_v2, 20 ep, seed 42)

| Config | best val_edge_f1 | best val_node_f1 | best val_loss |
|--------|-----------------|------------------|---------------|
| Baseline (dim 0) | 0.1579 (ep3) | 0.1176 (ep20) | 4.0564 (ep15) |
| Emb128 apprenables | 0.2120 (ep17) | **0.1519** (ep20) | 3.9744 (ep19) |
| wiki.fr 300d gelé | 0.2027 (ep14) | 0.1152 (ep19) | 3.9445 (ep18) |
| wiki.fr 300d fine-tuné | **0.2292** (ep20) | 0.1190 (ep12) | **3.9271** (ep18) |

### Décision
- **Piste « apprenables » : VALIDÉE** (+34% edge, +29% node vs baseline).
  Le défaut 128 est confirmé comme bénéfique.
- **Piste « pré-entraînés fine-tunés » : MEILLEURE CONFIG MESURÉE** (+45% edge vs baseline).
  Recommandée via `--embedding-file wiki.fr.vec` (sans `--freeze-embeddings`).
- **Piste « pré-entraînés gelés » : INFÉRIEURE aux apprenables** à 20 epochs.
  Le gel ne surpasse pas l'apprentissage from scratch — à revoir sur runs plus longs.
- **Piste « données » : MAINTENUE comme bloquant principal node_f1** (~800 phrases,
  `BENCHMARK.md` §5) — non traitée par cette remédiation.
- **Projections 0.55–0.65 : ABANDONNÉES.** Le meilleur mesuré est 0.2292 (cible > 0.40).

### Reste ouvert
- Runs 50–100 epochs sur config D (convergence non atteinte)
- Seed-sweep (7, 42, 123) sur configs B et D
- Test `cc.fr.300.bin` (2M mots vs 22k wiki.fr.vec)
- Annotation ~800 phrases ciblant nœuds rares (piste données, hors scope code)

---

**Date** : 2026-09-25  
**Auteur** : Claude Sonnet 4.5  
**Classification** : CONFIDENTIEL — Usage Interne Uniquement
