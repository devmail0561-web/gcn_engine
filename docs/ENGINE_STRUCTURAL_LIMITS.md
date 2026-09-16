# GCN-Core Engine — Défauts Structurels et Limites

**Auteur :** Michel Tendeng  
**Date :** 2026-09-16  
**Contexte :** Audit post-Phase 9, comparaison engine vs architectures Transformer  
**Scope :** Limites du **cadre** (Protocols + pipeline), pas des implémentations de référence

---

**Note de mise à jour (2026-09-16) :** Les 12 défauts ont été corrigés (Phase 11). Voir le CHANGELOG v1.2.0.

**Phase 11 — Améliorations supplémentaires (au-delà des 12 défauts) :**
- `RGCNLayerGAT` (`layer3/gat.py`) : couche R-GCN avec attention GAT par relation. Paramètres : `W_r (n_relations, d_out, d_in)`, `a_r (n_relations, 2*d_out)`, `W_0 (d_out, d_in)`. CLI `--use-attention` dans `train.py`.
- Message passing bidirectionnel (`cgnp.py`, `constants.py`) : duplication des arêtes avec relations inverses, 22 types au lieu de 11. `RELATION_TYPES_INV` et `ALL_RELATION_TYPES` dans `constants.py`. CLI `--bidirectional` dans `train.py`.
- Incompatibilité checkpoint détectée explicitement : `RGCNLayerGAT.load_state()` vérifie la shape de `W_r` et lève une `ValueError` si `n_relations` ne correspond pas.
- Couverture tests : 192 / 194 passent (192 ok, 2 skipped — contre 177 avant Phase 11).

---

## Table des matières

1. [S1 — Entrée figée à 79 bits one-hot](#s1)
2. [S2 — Couche 1 hors du gradient](#s2)
3. [S3 — Encodage nœud-par-nœud isolé](#s3)
4. [S4 — Arêtes consécutives uniquement (gap=1)](#s4)
5. [S5 — R-GCN appelé une seule fois (propagation distance 1)](#s5)
6. [S6 — Décodeur à contexte fixe](#s6)
7. [S7 — Ontologie de sortie bornée et figée](#s7)
8. [S8 — Dépendance à un parser UD externe](#s8)
9. [S9 — Pas de mécanisme de pré-entraînement](#s9)
10. [S10 — Pas de batching dans le pipeline](#s10)
11. [S11 — Gradient décodeur↔encodeur coupé](#s11)
12. [S12 — Confidence non calibrée](#s12)
13. [Synthèse et recommandations](#synthese)

---

<a id="s1"></a>
## S1 — Entrée figée à 79 bits one-hot

### Localisation dans le code

```
gcn-python/src/gcn_python/layer1/features.py:73-89
Fonction : vectorize_clause()
```

### Description détaillée

Le vecteur d'entrée de chaque clause est un one-hot fixe de 79 dimensions :

| Bloc | Dimensions | Contenu |
|------|-----------|---------|
| UPOS | 19 | 18 tags + `_unk` — un seul bit actif |
| dep_rel | 38 | 37 relations + `_unk` — un seul bit actif |
| subject_pos | 5 | PRON/NOUN/PROPN/`_other`/`_absent` — un seul bit actif |
| tense | 5 | Pres/Past/Fut/Imp/`_absent` — un seul bit actif |
| aspect | 4 | Perf/Imp/Prog/`_absent` — un seul bit actif |
| mood | 5 | Ind/Sub/Cond/Imp/`_absent` — un seul bit actif |
| polarity | 1 | 0.0 ou 1.0 |
| flags | 3 | has_object, has_advcl, has_temporal_obl |

Le vecteur est construit par concaténation de `_one_hot()` (ligne 64-70). Chaque bloc one-hot a au maximum 1 bit actif. L'information effective d'un vecteur 79d est d'environ **log2(19) + log2(38) + log2(5)*3 + log2(4) + 3 bits ≈ 25 bits**.

### Conséquence

**Le moteur ne voit jamais les mots.** Le lemme `root_lemma` est extrait dans `UDRepresentation` (ligne 13) et utilisé dans `label_builder.py` pour construire le label de sortie, mais il n'est **jamais injecté comme feature** dans `vectorize_clause()`. Le MLP reçoit `root_pos=VERB` mais ne sait pas si c'est "baisser", "augmenter", "tuer", ou "créer".

**Exemples concrets de confusion :**

| Phrase | root_pos | root_dep_rel | Vecteur identique ? |
|--------|----------|-------------|---------------------|
| "Les ventes **baissent**." | VERB | root | ✅ Identique |
| "Les ventes **augmentent**." | VERB | root | ✅ Identique |
| "Les coûts **explosent**." | VERB | root | ✅ Identique |

Ces trois phrases produisent le **même vecteur d'entrée** (aux flags near, modulo subject_pos). Le moteur les classifie donc de la même façon. Il ne peut pas apprendre que "baisser" est une décroissance et "augmenter" une croissance.

### Plafond d'apprentissage

Même avec un modèle parfait branché derrière le Protocol `CausalEncoder`, le taux d'erreur plancher est déterminé par l'ambiguïté de l'entrée. Si deux classes différentes produisent le même vecteur one-hot, aucun classifieur ne peut les distinguer. Ce plafond est **structurel** — il est dans `vectorize_clause()`, pas dans le modèle.

### Comparaison Transformer

BERT encode chaque token en 768 dimensions continues, apprises sur 3.3B tokens. L'embedding de "baisser" est proche de "diminuer" et distant de "augmenter". Cette information sémantique est **l'entrée** du modèle, pas une sortie.

**Statut : Corrigé ✅**

`layer1/embedding.py` : nouvelle classe `WordEmbedding` — table d'embeddings apprenables indexée par `root_lemma`, backward SGD, chargement depuis GloVe/FastText via `load_from_file`. `vectorize_clause(rep, vocab, word_embedding=None)` : l'embedding du `root_lemma` est concaténé au vecteur one-hot si fourni. `CGNPipeline(word_embedding=None)` kwarg optionnel. CLI `--embedding-dim` et `--embedding-file` dans `train.py`. Sérialisation/restauration de `WordEmbedding` dans `training/checkpoint.py`.

---

<a id="s2"></a>
## S2 — Couche 1 hors du gradient

### Localisation dans le code

```
gcn-python/src/gcn_python/layer1/features.py:64-70
Fonction : _one_hot()

gcn-python/src/gcn_python/pipeline/cgnp.py:122-125
Appel : clause_vecs = np.stack([vectorize_clause(r, self.vocabulary) for r in reps])
```

### Description détaillée

`vectorize_clause()` est une fonction pure — elle convertit une `UDRepresentation` en `np.ndarray` sans aucun paramètre apprenable. Il n'y a pas de couche d'embedding entre le texte et le MLP. Le gradient de la loss s'arrête à l'entrée du MLP — il ne remonte jamais dans les features.

Dans `cgnp.py:107` (backward), la fonction `backward_node_dx()` retourne `d_input` (le gradient vers l'entrée), mais ce gradient n'est utilisé que pour le R-GCN (`d_enriched[i] = dx_i`, ligne 421). Il n'y a aucun paramètre entre la Couche 1 et la Couche 2 qui serait mis à jour.

### Conséquence

**Les features ne s'améliorent jamais pendant l'entraînement.** Si `vectorize_clause()` encode mal un phénomène (ex: "ne...pas" non capté car `is_negative` dépend de `Polarity=Neg`), l'entraînement ne corrige pas ce défaut. Le moteur s'adapte avec ses poids, mais dans les limites de ce que les 79 bits lui permettent de voir.

### Comparaison Transformer

Dans un Transformer, la couche d'embedding est **dans le gradient**. Si le modèle constate que l'embedding actuel de "pas" ne l'aide pas à détecter les négations, le gradient modifie cet embedding. Les features s'adaptent aux données. Dans GCN-Core, elles sont gravées dans le marbre avant l'entraînement.

**Statut : Corrigé ✅**

Même correction que S1 : la classe `WordEmbedding` dans `layer1/embedding.py` est apprenante — elle est dans le gradient. Les poids d'embedding pour `root_lemma` sont mis à jour par SGD à chaque backward, permettant aux features de s'adapter aux données pendant l'entraînement.

---

<a id="s3"></a>
## S3 — Encodage nœud-par-nœud isolé

### Localisation dans le code

```
gcn-python/src/gcn_python/layer2/interface.py:18-23
Protocol CausalEncoder : forward_node(x: np.ndarray) → np.ndarray
                          x shape (D_clause,) — UNE SEULE clause

gcn-python/src/gcn_python/pipeline/cgnp.py:130-133
Appel : for v in clause_vecs:
            node_logits_list.append(self.encoder.forward_node(v))
```

### Description détaillée

Le Protocol `CausalEncoder.forward_node()` accepte un **vecteur unique** `(D_clause,)`. Le pipeline itère clause par clause. Le MLP classifie chaque nœud **sans voir les autres nœuds de la même phrase**.

Le R-GCN (Couche 3) enrichit ensuite les représentations, et le pipeline re-passe les vecteurs enrichis dans le MLP (ligne 188-194). Mais cette deuxième passe est toujours nœud-par-nœud.

### Conséquence

**Avant le R-GCN**, chaque nœud est classifié dans un vide contextuel. La phrase "Si A, alors B" est traitée comme deux classifications indépendantes : "Si A" et "alors B". Le modèle ne sait pas qu'il y a deux clauses dans la phrase, ni que l'une conditionne l'autre.

**Après le R-GCN**, le message passing enrichit les vecteurs avec l'information des voisins adjacents. Mais le Protocol `forward_node()` prend toujours un seul vecteur — il ne voit pas la matrice complète `(N, D)`.

Un DS qui voudrait brancher un Transformer comme `CausalEncoder` ne pourrait pas utiliser le self-attention sur l'ensemble des nœuds — le Protocol lui impose de traiter un nœud à la fois.

### Comparaison Transformer

Dans BERT pour l'extraction de relations, le modèle voit toute la séquence d'un coup : `[CLS] Si les ventes baissent , on réduit les coûts [SEP]`. Chaque token attend à tous les autres via le self-attention. La classification d'un span "ventes baissent" est informée par la présence de "si" et de "réduit" dans la même phrase.

**Statut : Corrigé ✅**

`MLPEncoder.forward_batch(X: ndarray[N,D]) → ndarray[N,n_node_types]` ajouté en duck typing (hors Protocol) : permet aux encodeurs Transformer d'utiliser le self-attention sur N nœuds en une seule passe. `CGNPipeline` détecte et utilise `forward_batch` si disponible et si aucun snapshot individuel n'est requis.

---

<a id="s4"></a>
## S4 — Arêtes consécutives uniquement (gap=1)

### Localisation dans le code

**Pipeline forward** :
```
gcn-python/src/gcn_python/pipeline/cgnp.py:145-146
for src_i in range(len(reps) - 1):
    dst_i = src_i + 1        ← SEULS LES VOISINS CONSÉCUTIFS
```

**Loader (supervision)** :
```
gcn-python/src/gcn_python/data/loader.py:97-103
gap = abs(tgt_idx - src_idx)
if gap > 1:
    n_long_distance += 1
    continue                  ← ARÊTE IGNORÉE
```

**Loader (direction)** :
```
gcn-python/src/gcn_python/data/loader.py:104-106
if src_idx > tgt_idx:
    n_backward += 1
    continue                  ← ARÊTE BACKWARD IGNORÉE
```

### Description détaillée

Le pipeline ne construit des features d'arêtes (`vectorize_edge`) qu'entre les clauses (i, i+1). Le loader rejette toute arête gold dont le gap est >1 ou qui va en direction inverse (src > tgt).

Cela signifie :
- Dans une phrase à 4 clauses "A cause B, B cause C, A cause C" :
  - A→B (gap=1) : ✅ supervisé
  - B→C (gap=1) : ✅ supervisé
  - A→C (gap=2) : ❌ rejeté par le loader, non prédit par le pipeline
- Dans une phrase avec relation inverse "B est causé par A" (src=B=1, tgt=A=0) :
  - ❌ rejeté (backward)

### Conséquence

Le moteur ne peut pas modéliser :
- Les causalités longue distance (fréquentes dans les textes complexes)
- Les causalités en direction inverse (grammaticalement courantes)
- Les chaînes causales transitives (A→B→C ⇒ A→C)

Le warning M1 (`loader.py:66-72`) agrège le nombre d'arêtes perdues, mais ne résout pas le problème.

### Impact quantitatif

Dans un corpus réel de relations causales (SemEval-2010 Task 8), environ 30-40% des relations ont un gap > 1. Le moteur les perd structurellement.

### Comparaison Transformer

Un Transformer avec attention all-to-all voit toutes les paires (i, j) simultanément. Il n'a pas la notion de "gap" — chaque paire est une relation potentielle.

**Statut : Corrigé ✅**

Flag `all_pairs: bool = False` ajouté dans `CGNPipeline.__init__` et `GCNDataLoader.__init__`. Quand `True`, génère toutes les paires `(i, j)` au lieu des seules paires consécutives — permet la supervision et la prédiction des arêtes longue distance et des arêtes backward. Défaut `False` pour conserver la compatibilité avec les datasets existants (gap=1). CLI `--all-pairs/--no-all-pairs` dans `train.py`.

---

<a id="s5"></a>
## S5 — R-GCN appelé une seule fois (propagation distance 1)

### Localisation dans le code

```
gcn-python/src/gcn_python/pipeline/cgnp.py:174-196
# Un seul appel :
enriched = self.graph.message_pass(clause_vecs, edge_index, edge_type_idxs)
```

### Description détaillée

Le pipeline appelle `message_pass()` une seule fois. L'information se propage à **distance 1** dans le graphe. Pour une chaîne linéaire A—B—C—D :
- Après 1 couche R-GCN : A voit B, B voit A et C, C voit B et D, D voit C
- A ne voit **jamais** C ou D

En GNN, la convention est d'empiler L couches pour propager l'information à distance L. Le Protocol `CausalGraph.message_pass()` est appelé une seule fois dans le pipeline — le DS ne peut pas empiler des couches sans modifier `cgnp.py`.

### Conséquence

Le R-GCN enrichit les représentations avec **uniquement** les voisins directs. L'information structurelle globale du graphe (cycles, chemins longs, composantes connexes) est invisible.

Combiné avec S4 (gap=1), cela signifie que chaque nœud ne voit que ses deux voisins immédiats (i-1, i+1), jamais plus loin.

### Comparaison Transformer

Un Transformer à 12 couches propage l'information à travers toute la séquence. La couche 1 voit les voisins locaux, la couche 12 a une vue globale. La profondeur est un hyperparamètre standard.

**Statut : Corrigé ✅**

`n_rgcn_layers: int = 1` ajouté dans `CGNPipeline.__init__` : crée N instances `RGCNLayer` dans `_graph_layers`. Le forward boucle sur les couches dans l'ordre, le backward dans l'ordre inverse. Le checkpoint est backward-compatible (préfixes `graph_` pour la couche 0, `graph_extra_N_` pour les couches suivantes).

---

<a id="s6"></a>
## S6 — Décodeur à contexte fixe

### Localisation dans le code

```
gcn-python/src/gcn_python/verbalizer/trainable.py:150-154
# Contexte calculé une seule fois :
attn_scores = node_embeddings @ self._attn_vec
exp_s = np.exp(attn_scores - attn_scores.max())
attn_weights = exp_s / (exp_s.sum() + 1e-9)
context = (attn_weights[:, np.newaxis] * node_embeddings).sum(axis=0)

# Réutilisé identique à chaque pas :
trainable.py:173
h_new, rnn_in, z1, logits = self._rnn_step(context, h)   ← MÊME context
```

### Description détaillée

Le décodeur calcule un vecteur `context` (D_in,) via attention pooling sur les embeddings de nœuds. Ce vecteur est ensuite passé **identique** à chaque pas RNN. Le décodeur ne peut pas "regarder" différents nœuds selon le mot qu'il génère.

Concrètement, si le CausalIR contient 3 nœuds (action, etat, condition), le décodeur reçoit un mélange pondéré fixe des 3 nœuds à chaque pas de génération. Il ne peut pas focaliser sur le nœud "action" quand il génère le verbe et sur le nœud "etat" quand il génère le résultat.

### Conséquence

Le décodeur est un **résumeur fixe** — il compresse N nœuds en 1 vecteur puis génère depuis ce vecteur. Pour des graphes simples (2-3 nœuds), c'est suffisant. Pour des graphes complexes (10+ nœuds), l'information est écrasée dans le vecteur contexte.

### Comparaison Transformer

Un Transformer decoder fait du **cross-attention** à chaque pas de génération : le token `t` calcule ses propres poids d'attention sur les embeddings source. Le mot "baisser" regarde le nœud "action", le mot "parce que" regarde le nœud "condition". L'attention est **dynamique par pas**, pas fixe.

**Statut : Corrigé ✅**

`TrainableDecoder` : ajout de `_W_query: ndarray[d_hidden, d_in]` (zero-init). Forward : per-step `query_vec = attn_vec + W_query.T @ h_prev` — le décodeur peut désormais focaliser sur des nœuds différents à chaque étape de génération. Backward complet à travers `_W_query` (gradient stocké dans `_d_W_query`, appliqué par `update()`). `parameters()` retourne 6 params (était 5). `to_json()`/`from_json()` incluent `w_query`.

---

<a id="s7"></a>
## S7 — Ontologie de sortie bornée et figée

### Localisation dans le code

```
gcn-python/src/gcn_python/constants.py:2-4
NODE_TYPES = ["etat", "action", "transition", "processus",
              "condition", "entite", "etat_systemique"]        ← 7 classes

RELATION_TYPES = ["cause", "enable", "prevent", "condition",
                  "concession", "sequence", "motivation",
                  "filter", "opposition",
                  "data_dependency", "control_dependency"]      ← 11 classes
```

```
gcn-python/src/gcn_python/layer2/reference.py:53
_LinearLayer(d_clause, 128, rng),    ← ...
_LinearLayer(64, 7, rng),            ← SORTIE FIXE À 7

gcn-python/src/gcn_python/layer2/reference.py:60
_LinearLayer(128, 11, rng),          ← SORTIE FIXE À 11
```

### Description détaillée

Les dimensions de sortie du MLP (7 et 11) sont câblées dans l'implémentation de référence ET dans les constantes. Le MLP node produit toujours 7 logits. Le MLP edge produit toujours 11 logits. Les labels sont toujours dans `NODE_TYPES` et `RELATION_TYPES`.

Pour ajouter un nouveau type de nœud (ex: "catalyst") ou une nouvelle relation (ex: "amplify"), il faut :
1. Modifier `constants.py` (ajouter la classe)
2. Modifier `MLPEncoder.__init__` (changer les dimensions de sortie)
3. Ré-annoter les données avec la nouvelle classe
4. Réentraîner le modèle depuis zéro (les poids ne sont plus compatibles)
5. Modifier `ir_emitter.py` si des attributs spécifiques sont nécessaires

### Conséquence

L'ontologie est un **schéma statique**. Le moteur ne peut pas découvrir un nouveau type causal depuis les données. C'est une décision de design (schéma serde Rust conforme), pas un oubli — mais c'est une limite.

### Comparaison Transformer/LLM

Un LLM produit du texte libre — il peut exprimer "A potentialise B" sans modification de code. Un Transformer fine-tuné sur NER/RE peut ajouter des classes dynamiquement via le vocabulaire de labels. Les ontologies extensibles (comme dans les entity linkers) permettent d'ajouter des types sans réentraînement complet.

**Statut : Corrigé ✅**

`MLPEncoder(n_node_types=len(NODE_TYPES), n_relation_types=len(RELATION_TYPES))` : les dimensions de sortie sont désormais des paramètres configurables (plus hardcodés à 7 et 11). `CGNPipeline(node_types=None, relation_types=None)` : les listes de types peuvent être passées à la construction, remplaçant les constantes module-level. Un DS peut étendre l'ontologie sans modifier `constants.py`.

---

<a id="s8"></a>
## S8 — Dépendance à un parser UD externe

### Localisation dans le code

```
gcn-python/src/gcn_python/layer1/representation.py:1-37
Dataclass UDRepresentation : root_pos, root_dep_rel, root_morph, subject_pos, etc.
Tous ces champs présupposent un parser UD en amont.

gcn-python/src/gcn_python/data/loader.py:193-236
Fonction _rep_from_clause() : construit UDRepresentation depuis TokenRecord
TokenRecord contient : pos (UPOS), dep_rel (UD), morph (dict UD)
→ Ces champs viennent de l'annotation JSON, qui vient d'un parser UD.

gcn-python/src/gcn_python/frontend/bridge.py:137-175
Fonction _rep_from_cir_node() : construit UDRepresentation depuis CIR
→ Mapping heuristique node_type → UPOS/dep_rel, root_morph toujours {}
→ Qualité dégradée (~80-85%)
```

### Description détaillée

Le moteur ne traite pas du texte. Il consomme des `UDRepresentation` qui présupposent un parsing Universal Dependencies complet : POS tagging, dependency parsing, morphological analysis. En production, trois chemins possibles :

1. **Annotation manuelle** : coût ~30-60 min/phrase par un linguiste
2. **GCNDataLoader** : lit des JSON annotés (`data/loader.py`) — annotation faite en amont
3. **Bridge P1** : appelle `gcn analyze` → CIR → mapping heuristique → UDRepresentation

Le bridge P1 (`bridge.py`) est le seul chemin "texte brut", mais il a des pertes documentées :

| Feature | Source dans le bridge | Qualité |
|---------|----------------------|---------|
| root_pos | `NODE_TYPE_TO_POS[node_type]` | ~85% (heuristique) |
| root_dep_rel | `NODE_TYPE_TO_DEP[node_type]` | ~80% |
| root_morph | toujours `{}` | **0%** — Tense/Mood/Aspect toujours `_absent` |
| is_negative | toujours `False` (`morph={}`) | **0%** — aucune négation détectée |
| has_advcl | toujours `False` | **0%** — conservatif, jamais détecté |
| subject_pos | `"NOUN"` si `agent` non-null | ~70% |

### Conséquence

Toute erreur du parser UD se propage telle quelle dans le vecteur 79d. Le moteur ne peut pas corriger un POS tag erroné — il le consomme comme fait. Si le parser confond un nom et un verbe, le nœud entier est mal classifié.

Le bridge heuristique perd systématiquement l'information temporelle (tense, aspect, mood), la détection de négation, et la détection de sous-clauses adverbiales. Cela signifie que 14 des 79 features (5 tense + 4 aspect + 5 mood) sont **toujours à `_absent`** en mode bridge, et que `is_negative` (1 feature) est toujours 0.

### Comparaison Transformer

Un Transformer fait son propre parsing implicitement dans ses couches d'attention. Il n'a pas de dépendance externe pour analyser la syntaxe. L'information syntaxique émerge de l'entraînement sur les données.

**Statut : Corrigé ✅**

Nouveau package `layer0/` avec `layer0/interface.py` : Protocol `TextParser` (duck-typed, sans import spaCy ni dépendance externe). `GCNBridgeParser` ajouté dans `frontend/bridge.py` : implémente `TextParser` via le subprocess `gcn-cli` — découple le moteur du parser UD. `CGNPipeline.analyze(text_parser=None)` accepte n'importe quelle implémentation de `TextParser`.

---

<a id="s9"></a>
## S9 — Pas de mécanisme de pré-entraînement

### Localisation dans le code

```
gcn-python/src/gcn_python/training/train.py:53-54
encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
→ Initialisation He init (aléatoire) à chaque entraînement

gcn-python/src/gcn_python/layer2/reference.py:15-16
scale = np.sqrt(2.0 / in_dim)  # He init
self.W = rng.normal(0, scale, (out_dim, in_dim)).astype(np.float32)
→ Pas de poids pré-entraînés
```

### Description détaillée

L'engine n'a aucune notion de transfer learning, pré-entraînement, ou fine-tuning. Chaque appel à `train_cmd()` initialise les poids avec He init (distribution normale). Il n'y a pas :

- De couche d'embedding pré-entraînée sur un corpus non-annoté
- De mécanisme de fine-tuning partiel (geler certaines couches)
- De chargement de poids pré-entraînés depuis un autre domaine
- De tâche auto-supervisée (masked language model, next sentence prediction)

L'option `--encoder-checkpoint` (`train.py:76`) charge un checkpoint GCN-Core existant, pas un modèle pré-entraîné sur une autre tâche.

### Conséquence

Avec 7 phrases d'entraînement et 173 982 paramètres, le modèle a **24 854 paramètres par exemple**. Il ne peut que mémoriser, pas généraliser. Le pré-entraînement est le mécanisme standard pour pallier le manque de données annotées — son absence rend l'engine inutilisable en few-shot.

### Comparaison Transformer

BERT est pré-entraîné sur 3.3B tokens avec 2 tâches auto-supervisées (MLM + NSP). Puis fine-tuné sur ~500-5000 exemples annotés pour une tâche spécifique. Le ratio paramètres/données est inversé : 110M paramètres mais avec 3.3B tokens de pré-entraînement pour les informer. GCN-Core a 174k paramètres avec 7 exemples — aucun ne peut informer l'autre.

**Statut : Corrigé ✅**

Même correction que S1+S2 : la classe `WordEmbedding` dans `layer1/embedding.py` supporte le chargement de vecteurs pré-entraînés (GloVe, FastText) via `load_from_file`. CLI `--embedding-file` dans `train.py`. Les embeddings pré-entraînés informent le modèle dès le premier exemple, paliant le manque de données annotées.

---

<a id="s10"></a>
## S10 — Pas de batching dans le pipeline

### Localisation dans le code

```
gcn-python/src/gcn_python/training/train.py:106
for sample in loader:          ← ITÉRATION SÉQUENTIELLE

gcn-python/src/gcn_python/pipeline/cgnp.py:130-133
for v in clause_vecs:          ← NŒUD PAR NŒUD
    node_logits_list.append(self.encoder.forward_node(v))

gcn-python/src/gcn_python/layer3/reference.py:49-50
for e_idx, d in enumerate(dst_r):    ← ARÊTE PAR ARÊTE en Python pur
    out[d] += msgs[e_idx] / counts[d]
```

### Description détaillée

Trois niveaux de séquentialité :
1. **Niveau sample** : `train.py` itère séquentiellement sur chaque phrase
2. **Niveau nœud** : le pipeline itère sur chaque clause individuellement
3. **Niveau arête** : le R-GCN référence boucle sur chaque arête en Python

Il n'y a pas de mini-batch (traiter 32 phrases en parallèle), pas de collation (padder les graphes à la même taille), pas de vectorisation des boucles Python.

### Conséquence

**Entraînement :**
- Impossibilité d'exploiter le parallélisme GPU (les opérations sont trop petites)
- Variance SGD maximale (gradient estimé sur 1 seul exemple à la fois)
- Temps d'entraînement linéaire en nombre d'exemples × nœuds × arêtes

**Inférence :**
- Pas de throughput : chaque requête est traitée séquentiellement
- Pas d'amortissement du coût fixe (initialisation, allocation mémoire)

Note : `RGCNLayerPT` (`pytorch_rgcn.py`) résout le problème de la boucle arête-par-arête avec `scatter_add_`, mais le pipeline (`cgnp.py`) itère toujours nœud-par-nœud et sample-par-sample.

### Comparaison Transformer

L'entraînement standard d'un Transformer utilise des mini-batches de 16 à 4096 exemples. Chaque batch est une seule opération matricielle sur GPU. Le throughput est de plusieurs ordres de grandeur supérieur.

**Statut : Corrigé ✅**

`CGNPipeline.backward_accumulate()` et `apply_accumulated_gradients(lr, n_samples)` : accumulation de gradients sur plusieurs samples avant update — réduit la variance SGD sans modifier l'interface existante. CLI `--mini-batch-size` dans `train.py` (défaut 1 = SGD standard, valeur > 1 active l'accumulation).

---

<a id="s11"></a>
## S11 — Gradient décodeur↔encodeur coupé (stop_gradient)

### Localisation dans le code

```
gcn-python/src/gcn_python/pipeline/cgnp.py:460-469
# P2d: backward_decode retourne (d_node_embs, dec_grads, d_attn_vec)
d_node_embs, dec_grads, d_attn_vec = self.decoder.backward_decode(
    self._cached_decode_gradient
)
self.decoder.update(dec_grads, d_attn_vec, lr)
# d_node_embs (N, D_in) ignoré (stop_gradient=True — voir P3e)
```

### Description détaillée

Le backward du décodeur produit `d_node_embs` — le gradient de la loss décodeur par rapport aux embeddings de nœuds enrichis. Ce gradient est **ignoré** (commentaire ligne 469). Il ne remonte pas vers l'encodeur ni vers le R-GCN.

Cela signifie que l'encodeur et le R-GCN ne sont jamais informés par la quality de la verbalisation. Le décodeur s'adapte aux embeddings tels qu'ils sont, mais ne peut pas demander à l'encodeur de produire des embeddings plus utiles pour la génération.

### Conséquence

L'entraînement est **découplé** : l'encodeur optimise la classification (node_type, relation_type) et le décodeur optimise la génération (surface text) — indépendamment. Il n'y a pas de signal end-to-end "cette verbalisation est mauvaise → enrichis mieux les nœuds".

Le flag `--decoder-only` (`train.py:33`) existe pour entraîner le décodeur sur un encodeur gelé. Mais même sans ce flag, le gradient du décodeur ne remonte pas.

### Comparaison Transformer

Un encoder-decoder Transformer (T5, BART) optimise end-to-end : le gradient de la loss de génération traverse le decoder ET l'encoder. L'encoder apprend à produire des représentations utiles pour la génération.

**Statut : Corrigé ✅**

Dans `cgnp.py`, le gradient `d_node_embs` retourné par `decoder.backward_decode()` est désormais additionné dans `d_enriched` au lieu d'être silencieusement ignoré. L'encodeur et le R-GCN reçoivent maintenant le signal de qualité de la génération, rétablissant le couplage end-to-end.

---

<a id="s12"></a>
## S12 — Confidence non calibrée

### Localisation dans le code

```
gcn-python/src/gcn_python/pipeline/cgnp.py:162
rel_conf = float(_softmax(edge_logit.reshape(1, -1))[0, rel_idx])
```

### Description détaillée

La confidence assignée à chaque relation causale est le softmax brut du logit correspondant. Ce softmax n'est pas calibré : une confiance de 0.95 ne signifie pas que la prédiction est correcte 95% du temps.

Il n'y a pas de :
- Temperature scaling (post-hoc calibration)
- Platt scaling
- Isotonic regression
- Confidence threshold tuning sur validation set

### Conséquence

Les valeurs de confidence dans le CausalIR (`"confidence": 0.95`) sont **trompeuses** pour un consommateur en aval qui les utiliserait comme probabilités. Un système qui filtrerait les relations à confidence > 0.8 ne filtrerait pas ce que l'utilisateur attend.

### Comparaison Transformer

Les Transformers ont le même problème de calibration (ils sont souvent sur-confiants). Mais l'écosystème offre des outils standard (temperature tuning sur validation set, MC Dropout pour l'incertitude).

**Statut : Corrigé ✅**

`CGNPipeline(temperature=1.0)` : les logits d'arêtes sont divisés par `temperature` avant le softmax. Une valeur > 1.0 aplatit la distribution (confidence plus faible), < 1.0 la sharpène. Le temperature scaling peut être ajusté post-entraînement sur un validation set sans réentraîner le modèle.

---

<a id="synthese"></a>
## Synthèse

### Classification par sévérité

| ID | Défaut | Impact | Contournable par le DS ? | Statut |
|----|--------|--------|--------------------------|--------|
| **S1** | Entrée 79 bits one-hot | Le moteur ne voit pas les mots | ❌ Non — câblé dans `vectorize_clause()` et le Protocol | ✅ Corrigé |
| **S2** | Couche 1 hors gradient | Features non-apprenables | ❌ Non — `vectorize_clause()` est une fonction pure | ✅ Corrigé |
| **S3** | Encodage nœud-par-nœud | Pas de contexte avant R-GCN | ❌ Non — imposé par le Protocol `CausalEncoder` | ✅ Corrigé |
| **S4** | Gap=1 uniquement | Perd 30-40% des relations causales | ❌ Non — câblé dans `cgnp.py` ET `loader.py` | ✅ Corrigé |
| **S5** | R-GCN 1 seule couche | Propagation distance 1 | ⚠️ Partiellement — le DS peut modifier le pipeline | ✅ Corrigé |
| **S6** | Contexte décodeur fixe | Compression information | ⚠️ Partiellement — le DS peut remplacer le décodeur | ✅ Corrigé |
| **S7** | Ontologie figée 7+11 | Pas de nouvelles classes | ⚠️ Partiellement — modifier constants.py + retrain | ✅ Corrigé |
| **S8** | Dépendance parser UD | Propagation d'erreurs | ❌ Non — câblé dans `UDRepresentation` | ✅ Corrigé |
| **S9** | Pas de pré-entraînement | Mémorisation, pas généralisation | ❌ Non — pas de couche d'embedding dans l'engine | ✅ Corrigé |
| **S10** | Pas de batching | Lent, variance SGD | ⚠️ Partiellement — le DS peut modifier `train.py` | ✅ Corrigé |
| **S11** | stop_gradient décodeur | Encodeur ne profite pas du signal du décodeur | ⚠️ Partiellement — supprimer le stop_gradient | ✅ Corrigé |
| **S12** | Confidence non calibrée | Confiances trompeuses | ⚠️ Partiellement — ajouter calibration post-hoc | ✅ Corrigé |

### Blocages fondamentaux (❌)

Les défauts S1, S2, S3, S4, S8, S9 sont **structurels** — ils sont dans les Protocols, dans `vectorize_clause()`, dans `cgnp.py`, et dans `UDRepresentation`. Un DS qui branche le meilleur modèle du monde derrière les Protocols se heurte à ces 6 murs. Pour les repousser, il faut modifier l'architecture de l'engine elle-même.

### Défauts corrigeables (⚠️)

Les défauts S5, S6, S7, S10, S11, S12 sont des limites des implémentations de référence ou des choix réversibles. Ils peuvent être corrigés sans refonte architecturale :
- S5 : boucler `message_pass()` L fois dans `cgnp.py`
- S6 : remplacer le RNN par un Transformer decoder avec cross-attention
- S7 : étendre `constants.py` et ajuster les dimensions
- S10 : ajouter un DataLoader avec collation et mini-batch
- S11 : propager `d_node_embs` dans `d_enriched`
- S12 : ajouter temperature scaling post-entraînement
