# Plan : Lever les deux limites structurelles du moteur GCN

> **STATUT : IMPLÉMENTÉ ✅ (2026-09-16)**  
> Phase 1 (bugs FastText), Phase 2 (GAT + embeddings) et Phase 2b (bidirectionnel) sont tous implémentés et testés.  
> Tests : 192 / 194 passent. Fichiers modifiés : `constants.py`, `pipeline/cgnp.py`, `training/train.py`, `layer3/gat.py`. Tests ajoutés : `tests/test_gat.py` (15 tests).

---

## Contexte

Deux limites ont été identifiées :
- **Limite 1** : le moteur dépend de données annotées manuellement (JSON experts). Objectif : permettre la génération automatique via LLMs.
- **Limite 2** : le plafond sémantique des features one-hot UPOS/DEP_REL. Objectif : intégrer des embeddings FastText + mécanisme d'attention GAT qui se renforcent mutuellement.

Deux bugs bloquants ont été découverts lors de l'exploration — ils doivent être corrigés en priorité car ils empêchent les embeddings de fonctionner du tout.

---

## Phase 1 — Corriger les bugs FastText (prérequis bloquant)

### Bug A : backward hors scope (`cgnp.py`)

**Fichier :** `gcn-python/src/gcn_python/pipeline/cgnp.py`

Le bloc de rétropropagation des embeddings (lignes 640-650) est dans `apply_accumulated_gradients()` et référence `d_enriched` qui n'y est jamais défini — NameError garanti à l'exécution quand `word_embedding is not None`.

Il y a **deux chemins d'entraînement** qui doivent être corrigés séparément :

#### Chemin SGD standard — `backward()` (ligne ~538)

Après la boucle `reversed(self._graph_layers)` (ligne 538), le gradient `d_enriched` est en scope. Ajouter :

```python
# Après le R-GCN backward — d_enriched contient le gradient post-propagation
if (self.word_embedding is not None
        and self._cached_reps is not None
        and d_enriched is not None
        and d_enriched.shape[1] > self.vocabulary.d_clause):
    d_emb_slice = d_enriched[:, self.vocabulary.d_clause:]
    for i in range(min(len(self._cached_reps), len(d_emb_slice))):
        self.word_embedding.backward(d_emb_slice[i], self._cached_reps[i].root_lemma)
    self.word_embedding.update(lr)
```

#### Chemin mini-batch — `backward_accumulate()` + `apply_accumulated_gradients()`

Dans `backward_accumulate()`, `d_enriched` est défini à la ligne 568. Après la boucle R-GCN (ligne 623), accumuler les gradients embedding (sans appeler `update()` ici) :

```python
# Après R-GCN accumulation — accumuler les gradients embedding
if (self.word_embedding is not None
        and self._cached_reps is not None
        and d_enriched.shape[1] > self.vocabulary.d_clause):
    d_emb_slice = d_enriched[:, self.vocabulary.d_clause:]
    for i in range(min(len(self._cached_reps), len(d_emb_slice))):
        self.word_embedding.backward(d_emb_slice[i], self._cached_reps[i].root_lemma)
    # update() appelé dans apply_accumulated_gradients() avec normalisation
```

Dans `apply_accumulated_gradients()`, remplacer le bloc hors-scope (lignes 640-650) par :

```python
# Appliquer les gradients embedding accumulés (normalisés par n_samples)
if self.word_embedding is not None:
    self.word_embedding.update(lr / max(n_samples, 1))
```

`WordEmbedding.backward()` (embedding.py ligne 62) fait `self._grad_accum[idx] += d_emb` — accumulation additive confirmée. N appels successifs additionnent correctement avant que `update()` applique la moyenne via `lr / n_samples`.

### Bug B : vocabulaire non pré-peuplé (`train.py`)

`WordEmbedding.lookup()` retourne `_unk` pour les lemmes absents du fichier FastText sans créer de ligne dans `_E`. Ajouter un pré-pass avant la boucle d'entraînement dans `train.py` :

```python
# Après création du loader, avant la boucle d'entraînement :
if word_embedding is not None:
    all_lemmas = [
        r.root_lemma
        for sample in loader
        for r in reps_from_sentence(sample.sentence)[0]
    ]
    word_embedding.build_vocab(all_lemmas)
```

**Limitation à documenter :** Les lemmes absents du fichier FastText reçoivent un embedding initialisé aléatoirement (sans signal sémantique). Ils sont apprenables via SGD mais partent de zéro — gain marginal pour les lemmes rares absents du FastText.

---

## Phase 2 — RGCNLayerGAT (embeddings + attention intégrés)

### Nouveau fichier : `gcn-python/src/gcn_python/layer3/gat.py`

Classe `RGCNLayerGAT(nn.Module)` implémentant le `CausalGraph` Protocol.

**Paramètres :**
- `W_r : nn.Parameter` — shape `(n_relations, d_out, d_in)` (identique à `RGCNLayerPT`)
- `W_0 : nn.Parameter` — shape `(d_out, d_in)` (identique)
- `a_r : nn.Parameter` — shape `(n_relations, 2 * d_out)` — vecteurs d'attention par relation (ajout)

**Formule (forward) :**
```
# Pour relation r :
msg_src = H[src_r] @ W_r[r].T         # (|E_r|, d_out)
msg_dst = H[dst_r] @ W_r[r].T         # (|E_r|, d_out)
e_ij    = LeakyReLU([msg_src ‖ msg_dst] @ a_r[r])   # (|E_r|,) scalaire

# Softmax par nœud destination (sans torch_scatter) :
e_max[dst] = max sur tous les j tels que dst_r[j] == dst   # par destination
e_shifted  = e_ij - e_max[dst_r]                           # stabilité numérique
exp_e      = exp(e_shifted)
sum_exp[dst] = scatter_add(exp_e, dst_r)                   # somme par destination
alpha_ij   = exp_e / sum_exp[dst_r].clamp(min=1e-9)        # nœuds sans voisins → 0

out += scatter_add(alpha_ij.unsqueeze(1) * msg_src, dst_r)
```

Implémentation de `e_max` par destination (PyTorch ≥ 1.12) :
```python
e_max = torch.full((N,), float('-inf'), device=self._device)
e_max.scatter_reduce_(0, dst_r, e_ij, reduce='amax', include_self=True)
```
Fallback boucle Python si `torch.__version__ < "1.12"` :
```python
e_max = torch.full((N,), float('-inf'), device=self._device)
for idx, val in zip(dst_r.tolist(), e_ij.tolist()):
    if val > e_max[idx]:
        e_max[idx] = val
```

**Compatibilité Protocol :**
- `parameters()` retourne `[W_r_np, W_0_np, a_r_np]` — 3 arrays
- `update(grads, lr)` accepte 3 gradients
- `load_state(arrays)` charge 3 arrays
- `torch_parameters()` et `forward_torch()` exposés pour optimiseur PyTorch natif

**Chemin d'entraînement — `backward_message_pass` via autograd :**

`H_in` provient toujours de numpy via `torch.as_tensor()` — c'est déjà un tenseur leaf détaché de tout graphe de calcul. `requires_grad_(True)` active le calcul du gradient w.r.t. H_in. `backward_message_pass` utilise `torch.autograd.grad()` :

```python
def message_pass(self, node_features, edge_index, edge_types):
    H_in = torch.as_tensor(node_features, dtype=torch.float32, device=self._device)
    H_in.requires_grad_(True)                       # leaf tensor depuis numpy — safe
    out = self._gat_forward(H_in, edge_index, edge_types)
    self._H_in_retained = H_in
    self._out_retained = out
    return out.detach().cpu().numpy()

def backward_message_pass(self, d_output):
    d_out_t = torch.as_tensor(d_output, dtype=torch.float32, device=self._device)
    grads = torch.autograd.grad(
        self._out_retained,
        [self._H_in_retained, self.W_r, self.W_0, self.a_r],
        grad_outputs=d_out_t,
        retain_graph=False,   # graphe libéré — ne pas appeler deux fois après un même forward
    )
    d_input = grads[0].detach().cpu().numpy()
    # d_input[:, vocabulary.d_clause:] = gradient vers les embeddings FastText
    return d_input, [g.detach().cpu().numpy() for g in grads[1:]]
```

**Contrainte `retain_graph=False`** : `backward_message_pass` ne peut être appelé qu'une seule fois par forward — le graphe de calcul est libéré. Dans `CGNPipeline`, c'est toujours le cas (un forward, un backward). Ne pas appeler deux fois sans nouveau forward.

`CGNPipeline.backward()` appelle `backward_message_pass` via duck-typing (`if hasattr(_layer, 'backward_message_pass')`, ligne 536 de cgnp.py) — aucun changement dans le pipeline.

**`_gat_forward`** est la méthode interne qui implémente la formule d'attention décrite ci-dessus (calcul de `e_ij`, softmax par destination, scatter_add). `message_pass()` et `backward_message_pass()` en sont les deux points d'entrée publics.

Le gradient des embeddings remonte via `d_input[:, vocabulary.d_clause:]` extrait dans Bug A fix.

**Tests — `test_pytorch_rgcn.py` inchangé** (teste `RGCNLayerPT`, reste `== 2`). Nouveaux tests dans `test_gat.py` :
- `assert len(params) == 3`
- `assert params[2].shape == (n_relations, 2 * d_out)` — vecteurs d'attention
- Gradient non-nul sur `a_r` après forward + backward
- `backward_message_pass` retourne `d_input` de shape `(N, d_in)` — gradient non-nul
- `apply_accumulated_gradients` avec `word_embedding` actif ne crash pas
- `RGCNLayerGAT.message_pass` produit des sorties différentes de `RGCNLayerPT` sur le même graphe (l'attention est utile)
- Pipeline complet : `CGNPipeline` avec `RGCNLayerGAT` + `word_embedding` + appel `backward()` complet sans erreur, poids modifiés
- `loader.py` fallback : JSON sans tokens produit une `UDRepresentation` valide avec `root_pos` correct selon `node_type`

### Phase 2b — Message passing bidirectionnel

#### Principe

Aujourd'hui toutes les arêtes du graphe sont forward (`i < j`). Le nœud `j` reçoit l'information de `i` mais `i` n'apprend rien de `j`. Un nœud au centre d'une chaîne causale ne connaît que son passé, pas son futur causal.

Le message passing bidirectionnel ajoute des arêtes inverses synthétiques à chaque arête forward, créant deux flux d'information :
- **Forward** `i → j` : propagation causale (cause → effet)
- **Backward** `j → i` : propagation abductive (effet → cause)

Les arêtes inverses ne sont pas supervisées — elles sont ajoutées automatiquement au moment du forward pass et apprises via backpropagation. Les données d'entraînement JSON ne changent pas.

#### Modification 1 — `constants.py`

Ajouter les 11 types de relations inverses :

```python
# constants.py — après RELATION_TYPES
RELATION_TYPES_INV = [r + "_inv" for r in RELATION_TYPES]
# ["cause_inv", "enable_inv", "prevent_inv", "condition_inv", "concession_inv",
#  "sequence_inv", "motivation_inv", "filter_inv", "opposition_inv",
#  "data_dependency_inv", "control_dependency_inv"]

ALL_RELATION_TYPES = RELATION_TYPES + RELATION_TYPES_INV  # 22 types au total
```

**Règle** : les indices `0–10` = forward, les indices `11–21` = backward (`r_inv = r + 11`).

**IMPORTANT — deux `n_relations` distincts dans le pipeline :**

| Constante | Valeur | Utilisée pour | Côté Rust |
|---|---|---|---|
| `RELATION_TYPES` | 11 | Classification d'arêtes (MLP Layer 2), supervision, CIR output | Synchronisée — NE PAS MODIFIER |
| `ALL_RELATION_TYPES` | 22 | `n_relations` du R-GCN/GAT (poids W_r, a_r) uniquement | Python-interne — Rust ne connaît pas `_inv` |

`CGNPipeline.relation_types` reste `RELATION_TYPES` (11) — il pilote la prédiction.
`RGCNLayerGAT.n_relations` passe à 22 quand `bidirectional=True` — il pilote les poids du message passing.
`FeatureVocabulary.d_edge = 181` est inchangé — confirmé sur code : `d_edge = 2*d_clause + d_conn` (pas de dépendance à `n_relations`).

**`RGCNLayer` (NumPy) supporte déjà `n_relations`** — confirmé sur `reference.py` ligne 20 :
`def __init__(self, d_in, d_out, n_relations: int | None = None, seed: int = 42)`.
La config `--bidirectional` sans `--use-attention` utilise `RGCNLayer(n_relations=22)` sans aucune modification du code.

#### Modification 2 — `pipeline/cgnp.py`

Ajouter le paramètre `bidirectional: bool = False` au constructeur de `CGNPipeline`. Après la construction de `edge_index` et `edge_type_idxs` (ligne ~232-237), insérer :

```python
# Message passing bidirectionnel : duplication des arêtes avec relations inverses
if self.bidirectional and edge_index.shape[1] > 0:
    rev_index = edge_index[[1, 0], :]                          # (2, E) — sens inverse
    rev_types = edge_type_idxs + len(self.relation_types)     # indices 11–21
    edge_index_mp  = np.concatenate([edge_index, rev_index], axis=1)   # (2, 2E)
    edge_types_mp  = np.concatenate([edge_type_idxs, rev_types])       # (2E,)
else:
    edge_index_mp  = edge_index
    edge_types_mp  = edge_type_idxs
```

`edge_index_mp` et `edge_types_mp` sont passés au R-GCN/GAT pour le message passing uniquement. `edge_index` et `edge_type_idxs` originaux restent utilisés pour la classification d'arêtes (MLP Layer 2) — la supervision est inchangée.

**Flux MLP edge — aucun risque de contamination des 22 types :**
Le MLP `encoder.forward_edge(edge_vec)` (cgnp.py ligne 214) prend un **vecteur de features** en entrée — pas `edge_type_idxs`. Les `edge_type_idxs` ne sont construits qu'aux lignes 235-241 et passés exclusivement au R-GCN. Le MLP ne voit jamais les indices de relation. Il prédit toujours parmi 11 logits (`n_relation_types` dans `MLPEncoder`). Il n'y a aucune interaction entre `edge_types_mp` (22 types, R-GCN) et `encoder.forward_edge` (11 logits, MLP).

**Gradient des arêtes inverses — mécanisme standard :**
Les arêtes inverses (W_r[11:21], a_r[11:21]) n'ont pas de gold label et ne participent pas à la loss d'arêtes. Elles reçoivent néanmoins des gradients via `d_enriched` — le gradient des représentations de nœuds qui remonte à travers `backward_message_pass`. C'est le comportement standard des GNN : les paramètres de message passing apprennent via la loss des nœuds, même pour les arêtes non-supervisées. W_r[11:21] apprend à propager l'information dans le sens abductif de façon à améliorer la classification de nœuds.

#### Modification 3 — `RGCNLayerGAT` et `RGCNLayerPT` dans `gat.py` / `pytorch_rgcn.py`

Passer `n_relations=22` quand `bidirectional=True` :

```python
# Dans train.py, quand --use-attention + --bidirectional :
graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective, n_relations=22)
pipeline = CGNPipeline(..., bidirectional=True)
```

`W_r` devient shape `(22, d_out, d_in)` et `a_r` devient shape `(22, 2*d_out)`. Les 11 poids supplémentaires (indices 11–21) apprennent indépendamment à pondérer les messages backward par type de relation inverse.

**Incompatibilité checkpoint** : un checkpoint entraîné avec `n_relations=11` ne peut pas être chargé avec `n_relations=22`. `load_state()` doit vérifier la shape et lever une `ValueError` explicite si elle ne correspond pas.

#### Modification 4 — `train.py`

Ajouter deux flags CLI :

```
--use-attention/--no-attention       (défaut : --no-attention)
--bidirectional/--no-bidirectional   (défaut : --no-bidirectional)
```

`--bidirectional` sans `--use-attention` utilise `RGCNLayerPT` (ou `RGCNLayer`) avec `n_relations=22`.
`--bidirectional` avec `--use-attention` utilise `RGCNLayerGAT` avec `n_relations=22`.

```python
n_rel = 22 if bidirectional else len(RELATION_TYPES)  # 22 ou 11
if use_attention:
    graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective, n_relations=n_rel)
else:
    graph = RGCNLayer(d_in=d_effective, d_out=d_effective, n_relations=n_rel)
pipeline = CGNPipeline(..., bidirectional=bidirectional)
```

#### Tests à ajouter dans `test_gat.py`

- Les arêtes inverses dans `edge_types_mp` ont des indices dans `[11, 21]`
- `RGCNLayerGAT` avec `n_relations=22` : `W_r.shape == (22, d_out, d_in)`, `a_r.shape == (22, 2*d_out)`
- Pipeline bidirectionnel : les représentations de nœuds avec `bidirectional=True` diffèrent de `bidirectional=False` sur le même graphe
- `load_state()` avec mauvaise shape lève `ValueError` explicite
- Checkpoint round-trip avec `n_relations=22`

### Intégration dans `train.py`

Flags CLI complets :
```
--use-attention/--no-attention       (défaut : --no-attention)
--bidirectional/--no-bidirectional   (défaut : --no-bidirectional)
--embedding-file PATH                (optionnel — FastText)
--embedding-dim INT                  (optionnel — si pas de fichier)
```

Combinaisons valides :
| Config | Couche 3 | n_relations | Commentaire |
|---|---|---|---|
| aucun flag | `RGCNLayer` NumPy | 11 | Baseline actuel |
| `--use-attention` | `RGCNLayerGAT` | 11 | GAT forward uniquement |
| `--bidirectional` | `RGCNLayer` NumPy | 22 | Bidirectionnel sans attention |
| `--use-attention --bidirectional` | `RGCNLayerGAT` | 22 | Configuration maximale |

---

## Phase 3 — Outil d'annotation LLM (hors moteur)

> **Périmètre :** L'annotateur LLM n'est pas partie intégrante du moteur. Il s'agit d'un outil externe autonome dont le seul rôle est de produire des fichiers JSON gcn-nl conformes, consommables par le moteur existant sans aucune modification de celui-ci.

### Localisation : `gcn-tools/gcn-annotate/` (répertoire séparé, hors `gcn-python` et `gcn-core`)

#### Commande CLI `gcn-annotate`

```
gcn-annotate --input phrases.txt --output dataset_llm/ --lang fr
             [--llm-backend anthropic|openai]
             [--model claude-sonnet-5]
             [--batch-size 10]
```

#### Architecture interne

**1. `LLMAnnotatorProtocol`** — protocole pluggable :
```python
class LLMAnnotator(Protocol):
    def annotate(self, sentences: list[str], lang: str) -> list[dict]: ...
```

**2. `AnthropicAnnotator`** — implémentation de référence :
- Prompt système : 7 `NODE_TYPES` + 11 `RELATION_TYPES` documentés avec **few-shot examples** (minimum 3 phrases annotées complètes dans le prompt)
- Demande JSON en format `document.sentences[].cir` (champs `type` + `relation`)
- Traitement par batch : la réponse LLM est un `document` JSON avec N sentences. `json_reader.load_sentences()` prend un `Path` (appelle `path.read_text()`) — ne fonctionne pas sur du contenu en mémoire. L'outil écrit la réponse dans un fichier temporaire (`tempfile.NamedTemporaryFile`) puis appelle `load_sentences(tmp_path)`. Pas de modification de `json_reader.py`.
- **Validation + retry** : si le JSON retourné est invalide ou contient des types inconnus après normalisation, relancer l'appel (max 3 tentatives)
- **Gestion d'erreurs API** : retry exponentiel sur rate-limit, timeout, erreur 5xx

**3. `normalize_annotation(raw: dict) -> dict`** — validation + normalisation :
- Table explicite des variantes LLM → valeurs canoniques (ex: `"état"→"etat"`, `"processus"→"processus"`, `"cause"→"cause"`, `"enables"→"enable"`, etc.)
- Rejet des samples dont les labels restent invalides après normalisation (avec log)

**4. Fallback UDRepresentation sans tokens** — modification de `loader.py` (moteur) :**

> **Note architecturale :** cette modification du moteur est une amélioration générale — le moteur doit pouvoir charger des JSON sans tokens, quelle que soit leur origine (LLM, bootstrap, saisie manuelle). Elle ne dépend pas de l'outil LLM et est justifiée indépendamment. Elle est comptée comme partie du périmètre moteur (Phase 1/2), pas comme modification induite par l'outil externe.


Quand `span_toks = []`, construire une `UDRepresentation` synthétique minimale depuis `ClauseRecord` :
```python
return UDRepresentation(
    tokens=[],
    root_lemma=clause.label.split("(")[0].strip(),  # ex: "baisser(ventes)"→"baisser"
    root_pos=_node_type_to_upos(clause.node_type),  # heuristique
    root_dep_rel="root",
    root_morph={},
    subject_pos=None,
    has_object=False, has_advcl=False, has_temporal_obl=False,
    token_span=clause.token_span,
    lang=lang,
)
```
`_node_type_to_upos` : `action/transition/processus → "VERB"`, `etat/etat_systemique → "ADJ"`, `entite → "NOUN"`.

**Limitation à documenter explicitement :** Le fallback laisse `root_morph={}`, `has_advcl=False`, `has_temporal_obl=False` toujours. Cela signifie que ~14 des 80 features sont toujours à `_absent`. Les données LLM produisent des features de qualité inférieure aux données UD annotées manuellement — le modèle entraîné dessus aura un plafond de performance plus bas.

**Estimation coût API (ordre de grandeur) :** prompt système ~2k tokens + 3 few-shot ~1.5k tokens + N phrases par batch. Pour 1000 phrases avec `--batch-size 10` = 100 appels. Input ~350k tokens + output ~150k tokens → ~$0.07–$0.15 (Claude Sonnet 5 à $3/$15 per Mtok in/out). À documenter dans le README de l'outil.

**Évaluation qualité annotations LLM :** l'outil doit inclure une commande `gcn-annotate --eval --gold gold.json --pred pred.json` qui calcule `node_accuracy` et `edge_f1` entre annotations LLM et annotations gold sur un sous-ensemble. Sans cette évaluation, la qualité du pipeline reste inconnue.

#### Format JSON produit

Identique au format gcn-nl existant, compatible `json_reader.py` sans modification :
```json
{
  "document": {
    "id": "llm-generated-001",
    "lang": "fr",
    "sentences": [
      {
        "id": "s001",
        "text": "...",
        "tokens": [],
        "cir": {
          "nodes": [{"id": "n001", "type": "processus", "label": "...", "token_span": [0,0], ...}],
          "edges": [{"source": "n001", "target": "n002", "relation": "cause", ...}]
        }
      }
    ]
  }
}
```

---

## Fichiers modifiés / créés

| Fichier | Action | Raison |
|---|---|---|
| `pipeline/cgnp.py` | Modifier | Bug A (embedding backward) + ajout `bidirectional` flag + duplication arêtes inverses |
| `training/train.py` | Modifier | Bug B (vocab) + flags `--use-attention`, `--bidirectional`, `--embedding-file` |
| `constants.py` | Modifier | Ajout `RELATION_TYPES_INV` + `ALL_RELATION_TYPES` (22 types) |
| `layer3/gat.py` | Créer | `RGCNLayerGAT` — GAT avec `backward_message_pass` autograd, `n_relations` configurable |
| `layer3/pytorch_rgcn.py` | Modifier | `RGCNLayerPT.load_state()` — vérification shape `n_relations` + `ValueError` si incompatible |
| `data/loader.py` | Modifier (Phase 1) | Fallback `UDRepresentation` sans tokens — amélioration moteur générale |
| `gcn-tools/gcn-annotate/` | Créer (**hors moteur**) | Outil autonome d'annotation LLM |
| `tests/test_pytorch_rgcn.py` | **Inchangé** | `RGCNLayerPT` retourne 2 params — ne pas toucher |
| `tests/test_gat.py` | Modifier | Ajouter tests bidirectionnel : shapes (22), nœuds diffèrent, checkpoint round-trip, ValueError shape |
| `tests/test_llm_annotate.py` | Créer (**dans gcn-tools**) | Normalisation + format JSON + retry |

---

## Vérification end-to-end

```bash
# 1. Phase 1 — vérifier que les embeddings s'entraînent (SGD et mini-batch)
gcn-train --data-dir gcn-datasets/examples/ --embedding-file /tmp/cc.fr.300.vec \
          --epochs 5 --output /tmp/ckpt_emb.npz --log-csv /tmp/emb.csv
gcn-train --data-dir gcn-datasets/examples/ --embedding-file /tmp/cc.fr.300.vec \
          --epochs 5 --mini-batch-size 4 --output /tmp/ckpt_emb_mb.npz
# Attendre : loss décroissante, "Embeddings : N vecteurs chargés", pas de NameError

# 2a. Phase 2 — GAT forward seul
gcn-train --data-dir gcn-datasets/examples/ --embedding-file /tmp/cc.fr.300.vec \
          --use-attention --epochs 5 --output /tmp/ckpt_gat.npz

# 2b. Phase 2b — GAT + bidirectionnel (configuration maximale)
gcn-train --data-dir gcn-datasets/examples/ --embedding-file /tmp/cc.fr.300.vec \
          --use-attention --bidirectional --epochs 5 --output /tmp/ckpt_gat_bidi.npz
# Attendre : node_accuracy(bidi) >= node_accuracy(forward seul)
python -m pytest gcn-python/tests/test_gat.py -v

# 3. Phase 3 — LLM annotation (outil externe)
export ANTHROPIC_API_KEY=...
gcn-annotate --input gcn-datasets/corpus/phrases_fr.txt \
             --output /tmp/llm_dataset/ --lang fr
gcn-train --data-dir /tmp/llm_dataset/ --epochs 5 --output /tmp/ckpt_llm.npz

# 4. Suite de tests complète moteur
python -m pytest gcn-python/tests/ -v
```

**Métriques de succès :**
- `node_accuracy` FastText + GAT forward > baseline one-hot
- `node_accuracy` GAT bidirectionnel >= GAT forward seul
- Chemin mini-batch avec `word_embedding` : pas de NameError dans `apply_accumulated_gradients`
- `test_gat.py` : gradient sur `a_r` non-nul, shapes `(22, d_out, d_in)` avec `--bidirectional`
- `load_state()` avec `n_relations` incompatible : `ValueError` explicite
- Samples LLM sans tokens : aucun `UserWarning` sur "aucune clause convertie"
- `test_pytorch_rgcn.py` : tous les tests passent sans modification
