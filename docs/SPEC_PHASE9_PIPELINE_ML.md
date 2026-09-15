# Spécification Phase 9 — Correction des 3 problèmes structurels du pipeline ML

**Auteur :** Michel Tendeng  
**Date :** 2026-09-15  
**Statut :** Approuvé — en attente d'implémentation  
**Branche cible :** `master`

---

## Vue d'ensemble

Trois problèmes structurels ont été identifiés dans le pipeline ML de GCN-Core. Ils concernent le chemin complet du texte brut jusqu'au conflit d'entraînement encodeur-décodeur.

```
PROBLÈME 1          PROBLÈME 2          PROBLÈME 3
Texte brut          Décodeur            Conflit gradients
    │               mean-pool           encodeur ↔ décodeur
    ✗ manque            │                   │
    spaCy           perd structure      R-GCN optimisé
                    du graphe           pour 2 objectifs
```

**Ordre d'implémentation :** Problème 3 → Problème 2 → Problème 1

---

## Problème 1 — Texte brut non utilisable avec l'encodeur entraîné

### Manifestation

Après entraînement, l'appel suivant est impossible :
```python
reps, idxs, connectors = reps_from_raw_text("Si les ventes baissent...")
cir = pipeline.forward(reps, text=text)
```
Cette fonction n'existe pas. Le seul chemin vers l'encodeur passe par un JSON pré-annoté avec UPOS, DEP_REL et morphologie.

### Cause profonde

`reps_from_sentence(rec: SentenceRecord)` lit les features syntaxiques depuis `rec.tokens` — des `TokenRecord` produits manuellement. Sans ce fichier, la fonction retourne `([], [], [])` immédiatement (ligne 118 de `loader.py`).

spaCy est déclaré dans `pyproject.toml` (`spacy>=3.7`) mais **aucune ligne de code ne l'appelle à l'inférence**. Cette décision était correcte pour l'entraînement (indépendance du moteur), mais crée un trou à l'inférence : l'encodeur entraîné est inutilisable sur du texte brut.

### Solution

Ajouter `reps_from_raw_text(text, lang)` dans `gcn-python/src/gcn_python/data/loader.py`.

**Principe :**
1. Charge spaCy automatiquement selon la langue (lazy, caché dans `_NLP_CACHE` au niveau module — invisible pour l'appelant)
2. Parse le texte → features syntaxiques automatiques (UPOS, DEP_REL, morphologie)
3. Utilise le frontend symbolique (FrenchParser/EnglishParser) pour détecter les **spans de clauses** uniquement — leur classification symbolique est ignorée
4. Construit les `UDRepresentation` depuis les tokens spaCy pour chaque span
5. Retourne `(reps, valid_idxs, connector_reps)` — **signature identique à `reps_from_sentence()`**

**Interface publique (ce que voit l'appelant) :**
```python
def reps_from_raw_text(
    text: str,
    lang: str = "fr",
) -> tuple[list[UDRepresentation], list[int], list[UDRepresentation | None]]:
    ...
```
Pas de modèle spaCy en paramètre. Pas de taxonomies. Juste du texte.

**Chargement spaCy — interne :**
```python
_NLP_CACHE: dict[str, Any] = {}

def _get_nlp(lang: str):
    if lang not in _NLP_CACHE:
        model = "fr_core_news_sm" if lang == "fr" else "en_core_web_sm"
        import spacy
        _NLP_CACHE[lang] = spacy.load(model)
    return _NLP_CACHE[lang]
```

**Mapping spaCy → UDRepresentation :**

| Champ spaCy | → Champ UDRepresentation |
|---|---|
| `token.pos_` | `root_pos`, `subject_pos` |
| `token.dep_` | `root_dep_rel`, `has_object`, `has_advcl`, `has_temporal_obl` |
| `token.morph.to_dict()` | `root_morph` → `tense`, `aspect`, `mood` |
| `token.lemma_` | `root_lemma` |
| `token.morph.get("Polarity") == ["Neg"]` | `is_negative` |

**Mise à jour CLI Python** (`gcn-python/src/gcn_python/pipeline/cli.py`) :
- Ajouter option `--raw-text TEXT` à `gcn-forward`
- Quand présent : appelle `reps_from_raw_text(text, lang)` au lieu de `load_sentences(path)`

**Mise à jour CLI Rust** (`gcn-core/crates/gcn-cli/src/main.rs`) :
- Ajouter flag `--raw` à `gcn forward`
- Passe `--raw-text "<text>"` au sous-processus `gcn-forward` au lieu d'un chemin de fichier

**Usage final après implémentation :**
```bash
gcn-forward --raw-text "Si les ventes baissent, on réduit les coûts." \
  --lang fr --model-path model.npz
# → CausalIR JSON, sans aucun fichier annoté
```

### Ce qui NE CHANGE PAS

- `reps_from_sentence()` — identique, tous les tests existants passent
- `GCNDataLoader` — identique
- Format JSON des datasets — identique
- `CGNPipeline.forward()` — reçoit des `UDRepresentation` quelle qu'en soit l'origine

### Risques de régression

| Risque | Mitigation |
|---|---|
| spaCy tokenise différemment du tokenizer symbolique | Utiliser les spans de FrenchParser comme référence, aligner les tokens spaCy par position de caractère |
| `reps_from_raw_text` appelée dans les tests existants | Impossible — la fonction n'existe pas encore |
| `fr_core_news_sm` non installé | `ImportError` clair avec message d'installation |

---

## Problème 2 — Mean-pool écrase la structure causale du graphe

### Manifestation

`TrainableDecoder.decode(ir_json)` retourne des tokens dans le désordre, pas une phrase :
```
Attendu  : "Si les ventes baissent, on réduit les coûts."
Obtenu   : "réduire coûts ventes si baissent"
```

### Cause profonde

Dans `trainable.py` ligne 111 :
```python
mean = node_embeddings.mean(axis=0)   # (N, 80) → (80,) par moyenne simple
```

**Problème a — Perte de structure :** la moyenne traite le nœud `Action` et le nœud `Condition` comme équivalents, avec le même poids `1/N`. Le décodeur ne sait pas quelle clause causale est la plus pertinente pour reconstruire la surface.

**Problème b — Gradient pauvre :** `backward_decode()` retourne `d_mean (80,)` — un seul vecteur identique pour tous les N nœuds. Le gradient ne différencie pas les nœuds, ce qui ralentit l'apprentissage et crée un couplage avec le problème 3.

### Principe de la solution : Attention pooling sur nœuds causaux

> Contrairement aux Transformers qui font de l'attention sur chaque **mot** (50–512 tokens), ici l'attention opère sur N **nœuds causaux** (2–8 clauses). Chaque nœud représente une clause entière avec son type causal. L'attention apprend quelle clause est la plus pertinente pour générer la surface — une attention au niveau du **sens**, pas du lexique.

Un vecteur appris `_attn_vec (80,)` calcule un score d'importance pour chaque nœud :

```
node_embeddings (N, 80)
       │  @ _attn_vec (80,)
       ↓
scores (N,)  →  softmax  →  weights (N,)
       │
Σ weights[i] * node[i]  →  pooled (80,)
       │
MLP 2 couches  →  logits (|V|,)
```

L'initialisation de `_attn_vec` à zéro garantit un comportement identique au mean-pool au premier forward (softmax uniforme = poids 1/N). Le décodeur apprend progressivement à pondérer les nœuds.

### Ce qui doit changer

**Fichier :** `gcn-python/src/gcn_python/verbalizer/trainable.py`

**a) `_init_layers(d_in)`** :
```python
self._attn_vec = np.zeros(d_in, dtype=np.float64)
# + stocker d_in pour la sérialisation
```

**b) `forward_decode(node_embeddings)`** — remplacer ligne 111 :
```python
# Avant
mean = node_embeddings.mean(axis=0)

# Après
attn_scores = node_embeddings @ self._attn_vec     # (N,)
exp_s = np.exp(attn_scores - attn_scores.max())   # stabilité numérique
attn_weights = exp_s / exp_s.sum()                 # (N,) — somme = 1
pooled = (attn_weights[:, np.newaxis] * node_embeddings).sum(axis=0)  # (80,)
# Cache pour le backward :
self._cached_attn_weights = attn_weights
self._cached_node_emb = node_embeddings
```

**c) `backward_decode(d_logits)`** — ajouter backward de l'attention avant backward MLP :
```python
# d_pooled (80,) issu du backward MLP (déjà calculé)
d_weighted = d_pooled[np.newaxis, :] * self._cached_attn_weights[:, np.newaxis]  # (N, 80)
d_attn_weights = (d_pooled * self._cached_node_emb).sum(axis=1)                   # (N,)
# Backward softmax
d_attn_scores = self._cached_attn_weights * (
    d_attn_weights - (self._cached_attn_weights * d_attn_weights).sum()
)                                                                                   # (N,)
d_attn_vec = self._cached_node_emb.T @ d_attn_scores                              # (80,)
d_input = d_weighted                                                                # (N, 80)
# Retourner : (d_input, grads_MLP, d_attn_vec)
```

**d) `parameters()`** : inclure `self._attn_vec`.

**e) `update(grads, lr)`** : appliquer SGD sur `_attn_vec`.

**f) `to_json()` / `from_json()`** : sérialiser `_attn_vec.tolist()` et `d_in`.

### Ce qui NE CHANGE PAS

- `forward_decode()` retourne toujours `(|V|,)` — interface publique identique
- `decode(ir_json)` — identique
- L'appel dans `cgnp.py:forward()` — identique
- Les tests de forme existants dans `test_trainable_decoder.py` passent

### Risques de régression

| Risque | Mitigation |
|---|---|
| Ancien checkpoint sans `_attn_vec` | Détection dans `load_checkpoint` : si clé absente → initialiser à zéro |
| `backward_decode` change de signature de retour | Adapter `cgnp.py:backward()` dans le même commit (problème 3) |
| Test `checkpoint_roundtrip` ne teste pas `_attn_vec` | Étendre le test existant |

---

## Problème 3 — Conflit de gradients encodeur ↔ décodeur

### Manifestation

Après entraînement conjoint encodeur + décodeur, les deux composants convergent moins bien qu'avec un entraînement séparé. L'encodeur classe moins bien les types causaux.

### Cause profonde

Dans `cgnp.py:backward()`, lignes 368–371 :
```python
d_enriched[:min(n, N_dec)] += d_mean[np.newaxis, :] / max(N_dec, 1)
```

Ce code propage le gradient de la loss décodeur (`d_mean`) dans `d_enriched`, qui est transmis au backward du R-GCN. Le R-GCN reçoit des gradients de **deux objectifs contradictoires** :

```
Backward complet (situation actuelle) :

Loss NodeType + RelationType
         │
   d_node_logits, d_edge_logits
         │
   backward MLP encodeur  →  d_enriched (encodeur)
         │
         +──────────────────────────────────┐
                                            │
Loss surface texte (décodeur)               │
         │                                  │
   backward_decode() → d_mean              │
         │                                  │
   d_enriched += d_mean / N  ◄──────────────┘
         │
   backward_message_pass(d_enriched)
         │
   W_r mis à jour selon : classifier NodeType ET prédire vocabulaire
   → CONFLIT → convergence dégradée pour les deux
```

### Solution : Stop-gradient (découplage des entraînements)

Ajouter `decoder_stop_gradient: bool = True` dans `CGNPipeline.__init__()`.

**Avec `decoder_stop_gradient=True` (défaut recommandé) :**

```
Forward  : texte → encodeur → R-GCN → embeddings enrichis → décodeur  (couplé ✓)
Backward : décodeur → STOP ✗  |  R-GCN ← encodeur uniquement           (découplé ✓)

Résultat :
- W_r (R-GCN) : optimisé uniquement pour NodeType / RelationType  ✓
- W_dec (MLP décodeur + _attn_vec) : optimisé uniquement pour la surface  ✓
- Le décodeur bénéficie toujours des embeddings causaux enrichis au forward  ✓
```

### Ce qui doit changer

**Fichier :** `gcn-python/src/gcn_python/pipeline/cgnp.py`

**a) `__init__()`** :
```python
def __init__(
    self,
    encoder: CausalEncoder,
    graph: CausalGraph,
    lang: str,
    vocabulary: FeatureVocabulary,
    *,
    decoder=None,
    decoder_stop_gradient: bool = True,   # ← nouveau paramètre
):
    ...
    self.decoder_stop_gradient = decoder_stop_gradient
```

**b) `backward()`** — conditionner lignes 368–371 :
```python
# Backward décodeur
if self.decoder is not None and self._cached_decode_gradient is not None:
    d_input_dec, dec_grads, d_attn_vec = self.decoder.backward_decode(
        self._cached_decode_gradient
    )
    self.decoder.update(dec_grads, d_attn_vec, lr)

    if not self.decoder_stop_gradient and self._cached_enriched_vecs is not None:
        # Propagation vers R-GCN (déconseillé — désactivé par défaut)
        N_dec, n = d_input_dec.shape[0], d_enriched.shape[0]
        d_enriched[:min(n, N_dec)] += d_input_dec[:min(n, N_dec)]
# else : gradient décodeur s'arrête ici — R-GCN non perturbé
```

### Ce qui NE CHANGE PAS

- `CGNPipeline(encoder, graph, lang, vocabulary)` — appel sans `decoder_stop_gradient` → comportement plus correct qu'avant, rétro-compatible
- `CGNPipeline(encoder, graph, lang, vocabulary, decoder=dec)` — idem
- Le forward est inchangé
- `filter_edge_cache()` est inchangé

### Risques de régression

| Risque | Mitigation |
|---|---|
| Code qui passait `decoder_stop_gradient=False` explicitement | Aucun code existant ne fait ça — le paramètre est nouveau |
| Test `test_pipeline_with_decoder` vérifie le backward | Mettre à jour le test pour vérifier que `W_r` ne change pas sous loss décodeur seule avec `stop_gradient=True` |

---

## Récapitulatif des fichiers modifiés

| Fichier | Modification | Problème |
|---|---|---|
| `gcn-python/src/gcn_python/data/loader.py` | + `reps_from_raw_text(text, lang)` + `_get_nlp()` cache | 1 |
| `gcn-python/src/gcn_python/pipeline/cli.py` | + option `--raw-text TEXT` | 1 |
| `gcn-core/crates/gcn-cli/src/main.rs` | + flag `--raw` dans `gcn forward` | 1 |
| `gcn-python/src/gcn_python/verbalizer/trainable.py` | mean-pool → attention pooling, backward `(N, 80)`, `_attn_vec` sérialisé | 2 |
| `gcn-python/src/gcn_python/pipeline/cgnp.py` | `decoder_stop_gradient=True`, backward adapté | 3 |

---

## Nouveaux tests requis

| Test | Fichier | Ce qu'il vérifie |
|---|---|---|
| `test_reps_from_raw_text_returns_reps` | `test_data_loader.py` | Retourne des UDRepresentation non vides depuis texte brut |
| `test_reps_from_raw_text_same_clause_count` | `test_data_loader.py` | Même nombre de clauses que FrenchParser sur la même phrase |
| `test_attention_weights_nonuniform_after_update` | `test_trainable_decoder.py` | Après updates, `attn_weights != 1/N` |
| `test_backward_decode_returns_N_D_gradient` | `test_trainable_decoder.py` | `d_input.shape == (N, 80)` |
| `test_checkpoint_attn_vec_roundtrip` | `test_trainable_decoder.py` | `_attn_vec` sauvegardé et restauré |
| `test_stop_gradient_rgcn_unchanged` | `test_pipeline.py` | Avec `stop_gradient=True`, `W_r` ne varie pas sous loss décodeur seule |

---

## Vérification end-to-end

```bash
# Régression globale (doit passer à chaque étape)
cd gcn-core && cargo test --workspace   # 137/137
cd .. && python3 -m pytest gcn-python/tests/ -q   # 112+ tests

# Problème 3 — stop-gradient (après implémentation)
python3 -m pytest gcn-python/tests/test_pipeline.py::test_stop_gradient_rgcn_unchanged -v

# Problème 2 — attention pooling (après implémentation)
python3 -m pytest gcn-python/tests/test_trainable_decoder.py -v

# Problème 1 — texte brut (après implémentation, requiert fr_core_news_sm)
python3 -c "
from gcn_python.data.loader import reps_from_raw_text
reps, idxs, connectors = reps_from_raw_text('Si les ventes baissent, on réduit les coûts.')
print(f'{len(reps)} clauses extraites')  # attendu : 2
"
```
