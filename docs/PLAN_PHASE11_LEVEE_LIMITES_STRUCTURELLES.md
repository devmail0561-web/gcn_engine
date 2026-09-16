# Plan : Lever les deux limites structurelles du moteur GCN

## Contexte

Deux limites ont été identifiées :
- **Limite 1** : le moteur dépend de données annotées manuellement (JSON experts). Objectif : permettre la génération automatique via LLMs.
- **Limite 2** : le plafond sémantique des features one-hot UPOS/DEP_REL. Objectif : intégrer des embeddings FastText + mécanisme d'attention GAT qui se renforcent mutuellement.

Deux bugs bloquants ont été découverts lors de l'exploration — ils doivent être corrigés en priorité car ils empêchent les embeddings de fonctionner du tout.

---

## Phase 1 — Corriger les bugs FastText (prérequis bloquant)

### Bug A : backward hors scope (`cgnp.py`)

**Fichier :** `gcn-python/src/gcn_python/pipeline/cgnp.py`, méthode `apply_accumulated_gradients()` (~ligne 641)

Le bloc de rétropropagation des embeddings référence `d_enriched` et `lr` qui n'existent pas dans cette méthode. Déplacer ce bloc dans `backward()` (après la ligne ~538 où `d_enriched` est en scope) :

```python
# Dans backward(), après la boucle reversed(self._graph_layers) :
if (self.word_embedding is not None
        and self._cached_reps is not None
        and d_enriched is not None
        and d_enriched.shape[1] > self.vocabulary.d_clause):
    d_emb_slice = d_enriched[:, self.vocabulary.d_clause:]
    for i in range(min(len(self._cached_reps), len(d_emb_slice))):
        self.word_embedding.backward(d_emb_slice[i], self._cached_reps[i].root_lemma)
    self.word_embedding.update(lr)
```

### Bug B : vocabulaire non pré-peuplé (`train.py`)

`WordEmbedding.lookup()` retourne `_unk` pour les lemmes absents du fichier FastText — ils ne reçoivent jamais de ligne dans `_E`. Ajouter un pré-pass avant la boucle d'entraînement dans `train.py` :

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

# Softmax par nœud destination — sans dépendance torch_scatter :
e_max   = scatter_reduce(e_ij, dst_r, reduce='amax')   # stabilité numérique
alpha   = softmax manuel via logsumexp sur dst_r

out += scatter_add(alpha.unsqueeze(1) * msg_src, dst_r)
```

**Compatibilité Protocol :**
- `parameters()` retourne `[W_r_np, W_0_np, a_r_np]` — 3 arrays
- `update(grads, lr)` accepte 3 gradients
- `load_state(arrays)` charge 3 arrays
- `torch_parameters()` et `forward_torch()` exposés pour optimiseur PyTorch natif

**Chemin d'entraînement — `backward_message_pass` via autograd (Option A) :**

Le layer retient le graphe de calcul PyTorch pendant le forward. `backward_message_pass(d_output)` utilise `torch.autograd.grad()` pour calculer les gradients sur `W_r`, `W_0`, `a_r` et les features d'entrée :

```python
def _forward_pt_retained(self, H_in, edge_index, edge_types):
    H = H_in.detach().requires_grad_(True)   # retenir pour d_input
    out = self._gat_forward(H, edge_index, edge_types)
    self._H_in_retained = H
    self._out_retained = out
    return out

def backward_message_pass(self, d_output):
    d_out_t = torch.as_tensor(d_output, dtype=torch.float32, device=self._device)
    grads = torch.autograd.grad(
        self._out_retained, [self._H_in_retained, self.W_r, self.W_0, self.a_r],
        grad_outputs=d_out_t, retain_graph=False,
    )
    d_input = grads[0].detach().cpu().numpy()
    return d_input, [g.detach().cpu().numpy() for g in grads[1:]]
```

`message_pass()` appelle `_forward_pt_retained` au lieu de `_forward_pt`. Aucun changement dans `CGNPipeline` ni dans `train.py`.

**Contrainte PyTorch ≥ 1.12** — la softmax par destination utilise `scatter_reduce_(..., reduce='amax')` disponible depuis PyTorch 1.12. Vérifier la version au démarrage ou implémenter un fallback boucle Python si `torch.__version__ < "1.12"`.

**Tests — `test_pytorch_rgcn.py` inchangé** (teste `RGCNLayerPT`, reste `== 2`). Les nouveaux tests GAT vont dans `test_gat.py` avec `assert len(params) == 3`.

### Intégration dans `train.py`

Ajouter un flag CLI `--use-attention/--no-attention` (défaut : `--no-attention`). Quand actif :
```python
from ..layer3.gat import RGCNLayerGAT
graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective)
```
Sinon : comportement actuel avec `RGCNLayer` / `RGCNLayerPT`.

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
- Prompt système : explique les 7 `NODE_TYPES` et 11 `RELATION_TYPES` avec exemples
- Demande JSON en format `document.sentences[].cir` (champs `type` + `relation`)
- Pas de token annotations requis (voir point 4 ci-dessous)
- Traitement par batch pour limiter les appels API (réponse = `document` avec N sentences → parsée via `json_reader.load_sentences` sur le contenu en mémoire)

**3. `normalize_annotation(raw: dict) -> dict`** — validation + normalisation :
- Mappe les variantes LLM (`"état"` → `"etat"`, `"process"` → `"processus"`, etc.)
- Rejette les samples dont les labels restent invalides après normalisation
- Table de normalisation pour node_types et relation_types

**4. Fallback UDRepresentation sans tokens** — modification dans `loader.py` du moteur (seul point de contact autorisé) :

Le moteur doit pouvoir charger des JSON sans tokens. Quand `span_toks = []`, construire une `UDRepresentation` synthétique minimale depuis `ClauseRecord` :
```python
# Fallback dans loader.py:_rep_from_clause() — quand span_toks est vide :
return UDRepresentation(
    tokens=[],
    root_lemma=clause.label.split("(")[0].strip(),  # mot principal, ex: "baisser(ventes)"→"baisser"
    root_pos=_node_type_to_upos(clause.node_type),  # heuristique
    root_dep_rel="root",
    root_morph={},
    subject_pos=None,
    has_object=False, has_advcl=False, has_temporal_obl=False,
    token_span=clause.token_span,
    lang=lang,
)
```
Avec `_node_type_to_upos` : `action/transition/processus → "VERB"`, `etat/etat_systemique → "ADJ"`, `entite → "NOUN"`.

#### Format JSON produit par l'outil

Identique au format gcn-nl existant, compatible `json_reader.py` sans modification de celui-ci :
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
| `pipeline/cgnp.py` | Modifier | Bug A : déplacer embedding backward dans `backward()` |
| `training/train.py` | Modifier | Bug B : pré-peuplement vocab + flag `--use-attention` |
| `layer3/gat.py` | Créer | `RGCNLayerGAT` — nouvelle couche GAT |
| `data/loader.py` | Modifier | Fallback UDRepresentation sans tokens (pour JSON sans token annotations) |
| `gcn-tools/gcn-annotate/` | Créer (**hors moteur**) | Outil autonome d'annotation LLM — produit des JSON gcn-nl |
| `tests/test_pytorch_rgcn.py` | **Inchangé** | `RGCNLayerPT` retourne toujours 2 params — ne pas toucher |
| `tests/test_gat.py` | Créer | Tests Protocol + attention + gradient |
| `tests/test_llm_annotate.py` | Créer (**dans gcn-tools**) | Tests normalisation + format JSON |

---

## Vérification end-to-end

```bash
# 1. Corriger les bugs (Phase 1) — vérifier que les embeddings s'entraînent
gcn-train --data-dir gcn-datasets/examples/ --embedding-file /tmp/cc.fr.300.vec \
          --epochs 5 --output /tmp/ckpt_emb.npz --log-csv /tmp/emb.csv
# Attendre : loss décroissante, "Embeddings : N vecteurs chargés"

# 2. GAT (Phase 2)
gcn-train --data-dir gcn-datasets/examples/ --embedding-file /tmp/cc.fr.300.vec \
          --use-attention --epochs 5 --output /tmp/ckpt_gat.npz
python -m pytest gcn-python/tests/test_gat.py -v

# 3. LLM annotation (Phase 3)
export ANTHROPIC_API_KEY=...
gcn-annotate --input gcn-datasets/corpus/phrases_fr.txt \
             --output /tmp/llm_dataset/ --lang fr
# Puis entraîner sur le dataset généré :
gcn-train --data-dir /tmp/llm_dataset/ --epochs 5 --output /tmp/ckpt_llm.npz

# 4. Suite de tests complète
python -m pytest gcn-python/tests/ -v
```

**Métriques de succès :**
- `node_accuracy` avec FastText + GAT > node_accuracy baseline (features one-hot seules)
- Samples LLM sans tokens : aucun `UserWarning` sur "aucune clause convertie"
- `test_pytorch_rgcn.py` : tous les tests passent sans modification
