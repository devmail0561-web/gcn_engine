# Spécification Phase 9 — Correction des 3 problèmes structurels du pipeline ML

**Auteur :** Michel Tendeng  
**Date :** 2026-09-15  
**Révision :** 2 — réécriture complète après critique  
**Statut :** En révision  
**Branche cible :** `feat/phase9-pipeline-fixes`

---

## Avertissement préliminaire

La révision 1 de ce document contenait trois erreurs fondamentales qui auraient conduit à des régressions ou à des corrections sans effet réel :

1. **Problème 1 (rev.1)** : mixer FrenchParser (spans) + spaCy (features) crée un problème d'alignement d'indices de tokens et perpétue la dépendance au symbolique. La cause profonde réelle est un **mismatch train/inférence** non identifié.
2. **Problème 2 (rev.1)** : l'attention pooling est une amélioration marginale sur N=2–8 nœuds qui ne résout pas le problème fondamental : le décodeur ne génère pas de séquences ordonnées.
3. **Problème 3 (rev.1)** : le stop-gradient coupe les gradients sans les remplacer. La boucle standalone décodeur (`train.py` lignes 187–198) fait déjà un entraînement découplé — l'analyse de ce mécanisme existant était absente.

---

## Vue d'ensemble corrigée

```
Texte brut
    │
    ▼
spaCy (tokenisation + POS + DEP + morph)      ← PROBLÈME 1
    │
    ▼
Segmentation en clauses (dépendance syntaxique)
    │
    ▼
UDRepresentation (features syntaxiques)
    │                    ┌─────────────────────┐
    ▼                    │  PROBLÈME 3          │
Encodeur MLP             │  Entraînement phasé  │
    │                    │  (pas conjoint)      │
    ▼                    └─────────────────────┘
R-GCN message passing
    │
    ▼
Embeddings nœuds (N, 80)
    │
    ▼
Décodeur autorégressif                        ← PROBLÈME 2
token[0] → token[1] → token[2] → ... → <eos>
```

**Ordre d'implémentation obligatoire :** Problème 3 → Problème 2 → Problème 1  
(chaque étape doit passer les 112+ tests Python et 137 tests Rust avant de passer à la suivante)

---

## Problème 1 — Mismatch train/inférence + absence de chemin texte brut

### Manifestation réelle (deux symptômes distincts)

**Symptôme A :** il est impossible d'appeler `pipeline.forward()` sur du texte brut — la fonction `reps_from_raw_text` n'existe pas.

**Symptôme B (non identifié en rev.1) :** même si on ajoutait cette fonction en utilisant spaCy, l'encodeur entraîné sur des annotations manuelles recevrait des features produites par un modèle statistique. La distribution des features serait différente à l'entraînement et à l'inférence → dégradation silencieuse des prédictions.

### Cause profonde complète

`reps_from_sentence()` construit les `UDRepresentation` depuis `rec.tokens` — des annotations syntaxiques produites **manuellement** par les annotateurs du corpus. Ces annotations suivent le standard Universal Dependencies, mais leur précision et leur cohérence sont celles d'un humain expert.

À l'inférence sur texte brut, on devrait utiliser un annotateur automatique (spaCy). Or spaCy produit des annotations selon le même standard UD mais avec une précision statistique (~95% POS, ~90% DEP sur fr). Cette différence de source crée un décalage entre ce que l'encodeur a appris et ce qu'il reçoit à l'inférence.

**La solution correcte adresse les deux symptômes simultanément :**  
Utiliser spaCy comme source unique de features, **aussi bien à l'entraînement qu'à l'inférence**.

### Solution

#### Étape 1 — Modifier `reps_from_sentence()` pour utiliser spaCy

Fichier : `gcn-python/src/gcn_python/data/loader.py`

`reps_from_sentence(rec)` doit ré-extraire les features syntaxiques depuis `rec.text` via spaCy, **en ignorant `rec.tokens`** pour les features (UPOS, DEP_REL, morph). Il utilise toujours `rec.clauses` pour les **spans et les labels causaux gold** (NodeType, RelationType).

```python
def reps_from_sentence(
    rec: SentenceRecord,
) -> tuple[list[UDRepresentation], list[int], list[UDRepresentation | None]]:
    """Construit les UDRepresentation depuis spaCy sur rec.text.
    Les spans de clauses et les labels causaux viennent de rec.clauses (gold).
    Les features syntaxiques (UPOS, DEP, morph) viennent de spaCy.
    """
    if not rec.clauses:
        return [], [], []
    nlp = _get_nlp(rec.lang)
    doc = nlp(rec.text)
    # Aligner tokens spaCy sur les spans de clauses via positions de caractères
    ...
```

**Alignement par position de caractère (pas par index de token) :**
- FrenchParser retourne des `token_span (start_token_idx, end_token_idx)` dans sa propre tokenisation
- spaCy retourne des tokens avec `token.idx` (position caractère début) et `token.idx + len(token.text)` (fin)
- L'alignement se fait via les positions caractère du texte source `rec.text` → univoque, robuste

#### Étape 2 — Ajouter `reps_from_raw_text(text, lang)`

Une fois l'étape 1 faite, `reps_from_raw_text` devient simple : il faut seulement détecter les spans de clauses sans avoir de `rec.clauses` gold.

```python
def reps_from_raw_text(
    text: str,
    lang: str = "fr",
) -> tuple[list[UDRepresentation], list[int], list[UDRepresentation | None]]:
```

**Détection de clauses via spaCy (sans FrenchParser) :**  
La dépendance syntaxique permet de segmenter en clauses :
- Un nouveau span de clause commence à chaque token dont `dep_` est dans `{"advcl", "csubj", "mark", "relcl"}` ou après une ponctuation forte (`,`, `;`)
- Le token racine de la phrase (`dep_ == "ROOT"`) définit la clause principale
- Chaque verbe fléchi (`pos_ in {"VERB", "AUX"}`) non inclus dans un span existant génère un nouveau span

Cette segmentation est purement syntaxique — elle ne dépend pas de FrenchParser.

**Interface finale :**
```bash
gcn-forward --raw-text "Si les ventes baissent, on réduit les coûts." --lang fr --model-path model.npz
# → CausalIR JSON — aucune annotation manuelle requise
```

#### Étape 3 — Mise à jour CLI

- `gcn-forward` (`cli.py`) : ajouter option `--raw-text TEXT`
- `gcn forward` Rust (`main.rs`) : ajouter flag `--raw` → passe `--raw-text` au sous-processus

#### Prérequis d'installation (à documenter)

```bash
pip install gcn-python
python -m spacy download fr_core_news_md   # ~43 MB — modèle fr (md = meilleure morphologie)
python -m spacy download en_core_web_md    # pour l'anglais
```

`fr_core_news_sm` est insuffisant pour la morphologie (Tense/Aspect/Mood) — utiliser `md` minimum.

### Ce qui NE CHANGE PAS

- Format JSON des datasets — identique (les `rec.clauses` avec labels causaux gold sont toujours lus)
- `GCNDataLoader` — identique
- `CGNPipeline.forward()` — identique

### Risques

| Risque | Gravité | Mitigation |
|---|---|---|
| spaCy `md` non installé | Bloquant | `ImportError` avec message d'installation explicite |
| Légères différences spaCy vs annotation manuelle sur les données d'entraînement | Modéré | Ré-entraîner le modèle après ce changement — les features d'entraînement seront désormais cohérentes avec l'inférence |
| Segmentation spaCy produit un nombre différent de clauses que les spans gold | Possible sur texte brut | Warning si détection ≠ spans gold sur les données de test — pas d'erreur |

---

## Problème 2 — Décodeur ne génère pas de séquences ordonnées

### Manifestation

`TrainableDecoder.decode(ir_json)` retourne des tokens dans le désordre :
```
Attendu  : "Si les ventes baissent, on réduit les coûts."
Obtenu   : "réduire coûts ventes si baissent"
```

### Cause profonde complète

Deux problèmes distincts, confondus en rev.1 :

**A — Mean-pool (problème identifié en rev.1, mais solution insuffisante)**  
`mean(axis=0)` traite tous les nœuds avec poids 1/N. L'attention pooling améliore la sélection du nœud le plus saillant mais **ne change pas le fait qu'on génère depuis un seul vecteur**.

**B — Absence de génération séquentielle (problème non adressé en rev.1)**  
Le MLP produit une distribution `(|V|,)` depuis un seul vecteur. Il n'y a aucun état récurrent, aucun mécanisme de position, aucune cohérence entre les tokens générés. Même avec une attention pooling parfaite, les tokens seront dans le désordre car rien ne encode l'ordre de génération.

L'attention pooling (rev.1) ne résout que A. Le problème B est la cause du désordre.

### Solution : décodeur autorégressif (NumPy référence)

Un décodeur autorégressif génère les tokens **un par un**, en maintenant un état caché qui encode "ce qui a déjà été généré" :

```
Contexte (nœuds causaux) → état initial h_0
    │
    ▼
token[0] = argmax(W_o @ h_0)
    │
    ▼
h_1 = tanh(W_h @ h_0 + W_e @ embed(token[0]))
    │
    ▼
token[1] = argmax(W_o @ h_1)
    │   ...
    ▼
<eos> → arrêt
```

#### Modifications dans `trainable.py`

**a) `SurfaceVocabulary`** : ajouter `<eos>` comme token spécial (index 2, après `<pad>=0` et `<unk>=1`). `build()` doit toujours inclure `<eos>` dans le vocabulaire.

**b) `TrainableDecoder` — nouvelle architecture :**

Paramètres apprenants :
- `W_h (d_hidden, d_hidden)` — transition d'état caché
- `W_e (d_hidden, d_emb)` — embedding du token précédent (`d_emb = 32` par défaut)
- `W_c (d_hidden, d_in)` — projection du contexte (nœuds causaux) vers l'état initial
- `W_o (|V|, d_hidden)` — projection vers le vocabulaire
- `E (|V|, d_emb)` — matrice d'embeddings de tokens

Initialisation de l'état `h_0` :
```python
# Context = attention-pooled node embeddings (si disponible) ou zeros
context = attention_pool(node_embeddings)  # (d_in,)
h_0 = tanh(W_c @ context)                 # (d_hidden,)
```

**c) `forward_decode(node_embeddings, gold_tokens=None, max_len=20)` :**
- Si `gold_tokens` fourni (entraînement) : **teacher forcing** — utilise le token gold au pas t-1
- Sinon (inférence) : **greedy decoding** — utilise argmax au pas t-1
- S'arrête à `<eos>` ou `max_len`
- Retourne `logits_sequence (T, |V|)` en entraînement, `tokens (T,)` en inférence

**d) `loss_decode(logits_seq, gold_tokens)` :**
- Cross-entropie sur toute la séquence : `mean(CE(logits_t, gold_t))` pour t=0..T-1
- Masque les positions padding

**e) `backward_decode(d_logits_seq)` :**
- BPTT (Backpropagation Through Time) sur la séquence
- Retourne les gradients de tous les paramètres

**f) `decode(ir_json)` — inférence :**
- Construit les embeddings one-hot des NodeTypes depuis le CausalIR
- Appelle `forward_decode(node_embeddings)` en mode greedy
- Retourne la séquence de tokens jointe par espaces (jusqu'à `<eos>`)

**g) Sérialisation :** `to_json()` / `from_json()` incluent tous les nouveaux paramètres et `d_emb`.

### Ce qui NE CHANGE PAS

- Interface `decode(ir_json) -> str` — identique pour l'appelant
- `CGNPipeline.forward()` appelle toujours `decoder.forward_decode(enriched_vecs)` — identique
- Protocol `VerbalizerDecoder` — inchangé

### Risques

| Risque | Gravité | Mitigation |
|---|---|---|
| Checkpoints existants incompatibles (nouvelle architecture) | Bloquant | `load_checkpoint` détecte l'ancienne architecture (absence de `W_h`, `W_e`, `E`) et refuse le chargement avec message clair |
| BPTT instable sur longues séquences | Modéré | Gradient clipping : `np.clip(grad, -1.0, 1.0)` sur tous les gradients du décodeur |
| `max_len` trop petit pour certaines phrases | Faible | Défaut 20, configurable via `--max-decode-len` dans `gcn-train` |
| Tests existants passent des `(|V|,)` logits — maintenant `(T, |V|)` | Bloquant | Adapter tous les tests de `test_trainable_decoder.py` dans le même commit |

---

## Problème 3 — Entraînement conjoint contre-productif

### Manifestation

Après entraînement conjoint encodeur + décodeur, les deux composants convergent moins bien qu'avec un entraînement séparé. L'encodeur perd de la précision sur la classification causale.

### Cause profonde complète

**Cause A — Gradients conflictuels (identifiée en rev.1) :**  
`cgnp.py:backward()` lignes 368–371 propagent `d_mean` du décodeur dans `d_enriched`, perturbant les poids R-GCN.

**Cause B — Non identifiée en rev.1 :**  
La boucle standalone décodeur (`train.py` lignes 187–198) entraîne déjà le décodeur indépendamment après la boucle encodeur. Ce mécanisme fait une partie du découplage. Mais il est exécuté **après** la boucle couplée, qui a déjà perturbé les poids R-GCN. L'ordre est incorrect.

**Cause C — Non identifiée en rev.1 :**  
L'encodeur et le décodeur utilisent le même learning rate. Or la loss décodeur (génération de texte) est sur une échelle différente de la loss encodeur (classification à 7 classes). Sans normalisation des losses, le décodeur peut dominer le gradient même avec le stop-gradient.

### Solution : entraînement séquentiel en deux phases explicites

Remplacer l'entraînement conjoint par deux phases séquentielles distinctes dans `gcn-train` :

**Phase 1 — Encodeur seul :**
```bash
gcn-train --data-dir corpus/ --epochs 50 --output model_encoder.npz
# Pas de --verbalize-dir → décodeur absent → pipeline purement encodeur
```

**Phase 2 — Décodeur seul, encodeur gelé :**
```bash
gcn-train --verbalize-dir corpus/ --decoder-only --encoder-checkpoint model_encoder.npz \
  --epochs 30 --output model_full.npz
```

`--decoder-only` charge l'encodeur depuis le checkpoint, **gèle ses poids** (aucune mise à jour sur `encoder_*` et `graph_*`), et entraîne uniquement les paramètres du décodeur.

#### Modifications dans `train.py`

Ajouter `--decoder-only` et `--encoder-checkpoint PATH` :

```python
@click.option("--decoder-only", is_flag=True, default=False)
@click.option("--encoder-checkpoint", type=click.Path(), default=None)
```

Quand `decoder_only=True` :
1. Charger les poids encodeur depuis `encoder_checkpoint`
2. `encoder.frozen = True` — ne pas appeler `update_node()` / `update_edge()`
3. `graph.frozen = True` — ne pas appeler `graph.update()`
4. Entraîner uniquement sur les paires verbalize
5. Sauvegarder le checkpoint complet (encodeur + décodeur)

#### Nettoyage de la boucle existante

Supprimer la boucle standalone décodeur (lignes 187–198 de `train.py`) — elle est remplacée par la phase 2 explicite ci-dessus. La garder créerait une confusion sur l'ordre d'entraînement.

#### Suppression du couplage dans `cgnp.py`

Supprimer purement les lignes 368–371 (`d_enriched += d_mean / N`). Pas de paramètre `decoder_stop_gradient` — le couplage est simplement retiré car il n'a jamais été utile. La suppression est définitive.

### Ce qui NE CHANGE PAS

- `CGNPipeline.forward()` — identique
- `CGNPipeline.loss()` — identique (calcule toujours la loss combinée pour le logging)
- `CGNPipeline.backward()` — simplifié (suppression des lignes 368–371 uniquement)
- Entraînement sans décodeur (`gcn-train` sans `--verbalize-dir`) — identique

### Risques

| Risque | Gravité | Mitigation |
|---|---|---|
| Utilisateurs qui relancent un entraînement conjoint existant | Faible | Les deux phases peuvent être enchaînées dans un script — documenter dans README |
| `frozen` non implémenté sur `MLPEncoder` et `RGCNLayer` | Bloquant | Ajouter attribut `frozen: bool = False` et vérifier dans `update_node()`, `update_edge()`, `graph.update()` |
| La loss décodeur n'est plus dans `epoch_loss` en phase 1 | Informatif | Normal — la loss décodeur est loguée séparément en phase 2 |

---

## Récapitulatif des fichiers modifiés

| Fichier | Modification | Problème | Commit atomique |
|---|---|---|---|
| `gcn-python/src/gcn_python/data/loader.py` | `reps_from_sentence` utilise spaCy + `reps_from_raw_text()` | 1 | Commit P1 |
| `gcn-python/src/gcn_python/pipeline/cli.py` | + `--raw-text` | 1 | Commit P1 |
| `gcn-core/crates/gcn-cli/src/main.rs` | + `--raw` dans `gcn forward` | 1 | Commit P1 |
| `gcn-python/src/gcn_python/verbalizer/trainable.py` | décodeur autorégressif complet | 2 | Commit P2 |
| `gcn-python/src/gcn_python/pipeline/cgnp.py` | suppression lignes 368–371 | 3 | Commit P3 |
| `gcn-python/src/gcn_python/training/train.py` | + `--decoder-only` + `--encoder-checkpoint`, suppression boucle standalone | 3 | Commit P3 |
| `gcn-python/src/gcn_python/layer2/reference.py` | + attribut `frozen` dans `MLPEncoder` | 3 | Commit P3 |
| `gcn-python/src/gcn_python/layer3/reference.py` | + attribut `frozen` dans `RGCNLayer` | 3 | Commit P3 |

**Règle absolue :** chaque commit doit passer `pytest gcn-python/tests/ -q` (112+ tests) et `cargo test --workspace` (137 tests) **avant** de passer au commit suivant. Aucune exception.

---

## Nouveaux tests requis

### Problème 1

| Test | Ce qu'il vérifie |
|---|---|
| `test_reps_from_sentence_uses_spacy_features` | Les features UPOS/DEP viennent de spaCy, pas de `rec.tokens` |
| `test_reps_from_raw_text_returns_N_reps` | Retourne autant de reps que de clauses détectées |
| `test_reps_from_raw_text_no_manual_annotation_needed` | Fonctionne sans `rec.tokens` rempli |
| `test_feature_consistency_sentence_vs_raw_text` | Même features pour le même texte via les deux chemins |

### Problème 2

| Test | Ce qu'il vérifie |
|---|---|
| `test_forward_decode_returns_sequence` | `forward_decode` retourne `(T, |V|)` logits |
| `test_greedy_decode_produces_eos` | `decode()` s'arrête à `<eos>` |
| `test_teacher_forcing_vs_greedy_same_first_token` | Cohérence entre modes train et inférence |
| `test_backward_bptt_gradients_nonzero` | Gradients non nuls sur tous les paramètres |
| `test_checkpoint_autoregressive_roundtrip` | Sauvegarder + restaurer → même prédictions |

### Problème 3

| Test | Ce qu'il vérifie |
|---|---|
| `test_encoder_weights_frozen_in_decoder_only_mode` | `W_r` ne change pas pendant `--decoder-only` |
| `test_decoder_weights_update_in_decoder_only_mode` | `W_h`, `W_e` changent bien pendant `--decoder-only` |
| `test_no_gradient_coupling_in_backward` | Supprimer les lignes 368–371 n'affecte pas la loss encodeur |

---

## Vérification end-to-end

**Après Commit P3 (problème 3) :**
```bash
python3 -m pytest gcn-python/tests/ -q       # 112+ tests
python3 -m pytest gcn-python/tests/test_pipeline.py -v
```

**Après Commit P2 (problème 2) :**
```bash
python3 -m pytest gcn-python/tests/test_trainable_decoder.py -v
python3 -c "
from gcn_python.verbalizer.trainable import TrainableDecoder, SurfaceVocabulary
import numpy as np
sv = SurfaceVocabulary(); sv.build(['si les ventes baissent on réduit les coûts'])
dec = TrainableDecoder(sv, d_hidden=32)
emb = np.random.randn(2, 80)
tokens = dec.decode('{\"nodes\":[{\"node_type\":\"processus\"},{\"node_type\":\"action\"}],\"edges\":[]}')
print('tokens générés :', tokens)
assert '<eos>' not in tokens  # <eos> ne doit pas apparaître dans la sortie
"
```

**Après Commit P1 (problème 1) :**
```bash
# Requiert : python -m spacy download fr_core_news_md
python3 -c "
from gcn_python.data.loader import reps_from_raw_text
reps, idxs, connectors = reps_from_raw_text('Si les ventes baissent, on réduit les coûts.')
assert len(reps) >= 2, f'Attendu ≥2 clauses, obtenu {len(reps)}'
assert reps[0].root_pos in {'VERB', 'NOUN', 'AUX'}, f'POS inattendu : {reps[0].root_pos}'
print(f'{len(reps)} clauses — OK')
"
```

**Pipeline complet :**
```bash
# Phase 1 : entraîner l'encodeur
gcn-train --data-dir gcn-datasets/examples/ --epochs 20 --output model_enc.npz

# Phase 2 : entraîner le décodeur
gcn-train --decoder-only --encoder-checkpoint model_enc.npz \
  --verbalize-dir gcn-datasets/examples/ --epochs 20 --output model_full.npz

# Inférence sur texte brut
gcn-forward --raw-text "Si les ventes baissent, on réduit les coûts." \
  --lang fr --model-path model_full.npz
```
