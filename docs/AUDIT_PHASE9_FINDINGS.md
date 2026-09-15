# Audit SPEC_PHASE9_PIPELINE_ML.md — Findings

**Date :** 2026-09-15  
**Réviseur :** Claude Opus 4.6 (audit indépendant)  
**Cible :** docs/SPEC_PHASE9_PIPELINE_ML.md rev.5  
**Code audité :** commits `3bafb75`, `12691c6`, `fedfab7`  
**Tests :** 120 / 120 Python — 137 / 137 Rust

---

## Findings précédents (F1–F11) — tous résolus

| Finding | Correction | Vérification code | Statut |
|---|---|---|---|
| F1 — C1 backward custom encoder | `cgnp.py:319-328` émet `UserWarning` + return | `warnings.warn("encodeur sans backward_node_dx")` | Résolu |
| F2 — C6 confidence vs connector | `cgnp.py:485-489` utilise `connector_rep is None` | `_infer_origin` ne lit plus `confidence` | Résolu |
| F3 — C7 `_find_patient_lemma` | `label_builder.py:55-60` | Fonction définie, cherche `dep_rel in {obj, iobj, nobj}` | Résolu |
| F4 — C2 sévérité | Spec rev.5 ligne 125 | Reclassé "amélioration défensive, pas un bug actif" | Résolu |
| F5 — C8 reformulation | Spec rev.5 lignes 306-313 | Précise que seul `CGNPipeline.__init__` change | Résolu |
| F6 — C5 import SCOPE_VALUES | `grep -rn SCOPE_VALUES pipeline/` → aucun résultat | Import supprimé, FR-only documenté `cgnp.py:475` | Résolu |
| F7 — C3/C4 négation analytique | `cgnp.py:453-455` + spec ligne 192 | Limitation documentée in-code et dans la spec | Résolu |
| F8 — Assertion scope e2e | `test_pipeline.py:261` | `assert result["nodes"][0]["scope"] == "universal"` | Résolu |
| F9 — C10 cartographie | Spec ligne 52 + table limitations ligne 530 | Retiré de l'ordre d'implémentation | Résolu |
| F10 — P→C numérotation | Spec lignes 56-71 | Table de correspondance ajoutée | Résolu |
| F11 — P2d/P3e code | Spec lignes 375-458 | Code RNN autorégressif + P3e complets | Résolu |

---

## Nouveaux findings (N1–N6)

---

### N1 — `label_builder.py` lit du YAML dans le moteur

**Sévérité :** Modérée — violation architecturale potentielle

**Fichier :** `gcn-python/src/gcn_python/pipeline/label_builder.py`

**Code fautif — ligne 3 :**
```python
import yaml
```

**Code fautif — lignes 89-101 :**
```python
def _load_nominalizations(taxonomies_dir: Path, lang: str) -> dict[str, str]:
    table: dict[str, str] = {}
    for search_dir in [taxonomies_dir / lang, taxonomies_dir]:
        nom_path = search_dir / "nominalizations.yaml"
        if nom_path.exists():
            doc = yaml.safe_load(nom_path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                examples_key = "examples_fr" if lang == "fr" else "examples"
                for _cls, cls_data in (doc.get("classes") or {}).items():
                    for entry in (cls_data or {}).get(examples_key) or []:
                        if isinstance(entry, dict) and "lemma" in entry and "note" in entry:
                            table[entry["lemma"].lower()] = entry["note"]
            break
    return table
```

**Règle enfreinte :** La décision architecturale v0.9.3 (mémoire `feedback_no_hardcoding.md`) dit :
> "Le moteur ne lit JAMAIS les YAML. YAML = outils annotateurs uniquement."

`label_builder.py` est dans `gcn_python/pipeline/` — c'est le moteur. Il importe `yaml` et appelle `yaml.safe_load()` pour charger `nominalizations.yaml` depuis `taxonomies_dir`.

**Ce qui est lu :** Le fichier `nominalizations.yaml` contient des tables de nominalisation (verbe → nom : "augmenter" → "augmentation"). Ce n'est pas de la donnée d'annotation (gold labels), c'est une ressource de référence linguistique.

**Conséquence pratique :** Le package `gcn-python` dépend de `pyyaml` à cause de cette seule fonction. Si `taxonomies_dir=None` (par défaut), le code n'est jamais exécuté — mais l'import ligne 3 est inconditionnel.

**Correction proposée :** Convertir `nominalizations.yaml` en `nominalizations.json` dans `gcn-references/`. Remplacer dans `label_builder.py` :
- `import yaml` → `import json`
- `yaml.safe_load(nom_path.read_text(...))` → `json.load(open(nom_path, encoding="utf-8"))`
- Rechercher `nominalizations.json` au lieu de `nominalizations.yaml`

Alternativement, si la règle ne s'applique qu'aux YAML d'annotation (pas aux ressources de référence), documenter cette exception explicitement.

---

### N2 — `forward_decode` appelé deux fois pendant l'entraînement

**Sévérité :** Modérée — calcul gaspillé à chaque sample

**Fichier :** `gcn-python/src/gcn_python/pipeline/cgnp.py`

**Premier appel — `_forward_from_reps()`, lignes 222-228 :**
```python
# Décodeur (optionnel) — utilise les embeddings R-GCN enrichis si disponibles
if self.decoder is not None:
    _vecs = (self._cached_enriched_vecs
             if self._cached_enriched_vecs is not None
             else self._cached_clause_vecs)
    if _vecs is not None and len(_vecs) > 0:
        self._cached_decode_logits = self.decoder.forward_decode(_vecs)
```

Cet appel est en mode inférence (`gold_tokens=None`). Il exécute un pas RNN et retourne `(|V|,)` logits. Le résultat est stocké dans `self._cached_decode_logits`.

**Deuxième appel — `loss()`, lignes 286-296 :**
```python
if (self.decoder is not None
        and gold_surface is not None
        and len(gold_surface) > 0):
    _vecs = (self._cached_enriched_vecs
             if self._cached_enriched_vecs is not None
             else self._cached_clause_vecs)
    if _vecs is not None and len(_vecs) > 0:
        dec_logits = self.decoder.forward_decode(_vecs, gold_surface)
        dec_loss, d_dec = self.decoder.loss_decode(dec_logits, gold_surface)
        total_loss += dec_loss
        self._cached_decode_gradient = d_dec
```

Ce deuxième appel est en mode teacher forcing (`gold_tokens=gold_surface`). Il exécute T pas RNN et retourne `(T, |V|)` logits. Il écrase le `_rnn_step_cache` du premier appel.

**Problème :** Pendant l'entraînement, le premier appel (ligne 228) est du calcul perdu :
1. Son résultat `_cached_decode_logits` n'est jamais lu par aucune méthode
2. Son `_rnn_step_cache` est écrasé par le deuxième appel
3. Il exécute `mean_pool` + 1 pas RNN + 1 matmul output pour rien, à chaque sample

**Quantification :** Pour un vocabulaire de 500 tokens, `d_hidden=64`, `d_in=95` : chaque appel gaspillé = 1 matmul `(159, 64)` + 1 matmul `(64, 500)` + 1 mean_pool. Sur 100 samples × 50 epochs = 5000 forward gaspillés.

**Correction proposée :** Supprimer le bloc lignes 222-228 entièrement. L'appel en mode inférence n'a aucun consommateur. Le `decode()` du `TrainableDecoder` (utilisé pour la verbalisation à l'inférence) fait son propre greedy decoding indépendamment (`trainable.py:279-303`) — il n'utilise pas `_cached_decode_logits`.

---

### N3 — `--decoder-only` : la loss reportée compte la decoder loss deux fois

**Sévérité :** Modérée — la loss affichée est fausse, ce qui fausse le diagnostic d'entraînement

**Fichier :** `gcn-python/src/gcn_python/training/train.py`

**Chemin 1 — pipeline.loss(), lignes 173-176 :**
```python
loss_val, d_node, d_edge = pipeline.loss(
    node_logits, edge_logits_arg, gold_node, gold_edge,
    gold_surface=_gold_surface,
)
```

Quand `_gold_surface` est non-None, `pipeline.loss()` appelle `forward_decode(vecs, gold_surface)` et ajoute `dec_loss` au `total_loss` retourné (`cgnp.py:295`). Le gradient est caché dans `_cached_decode_gradient`.

**Ligne 178-180 — backward sauté :**
```python
if not decoder_only:
    pipeline.backward(d_node, d_edge, lr=lr)
```

Quand `decoder_only=True`, `pipeline.backward()` n'est pas appelé. Donc `_cached_decode_gradient` n'est jamais consommé. Le decoder n'est pas mis à jour par ce chemin. Mais `dec_loss` a quand même été ajouté à `loss_val`.

**Chemin 2 — boucle standalone, lignes 202-213 :**
```python
for vsample in verb_loader:
    if len(vsample.gold_tokens) == 0:
        continue
    dec_logits = pipeline.decoder.forward_decode(vsample.node_type_embeddings)
    dec_loss, d_dec = pipeline.decoder.loss_decode(dec_logits, vsample.gold_tokens)
    if not np.isfinite(dec_loss):
        continue
    _, dec_grads = pipeline.decoder.backward_decode(d_dec)
    pipeline.decoder.update(dec_grads, lr)
    epoch_loss += dec_loss
    n_samples += 1
```

Ici le décodeur est entraîné pour de vrai (forward + loss + backward + update). Et `dec_loss` est aussi ajouté à `epoch_loss`.

**Résultat concret :** Pour un sample qui a une surface gold et qui apparaît aussi dans `verb_loader` :
- `epoch_loss` += `dec_loss` du chemin 1 (pipeline.loss, non entraîné)
- `epoch_loss` += `dec_loss` du chemin 2 (standalone, entraîné)

La loss reportée à l'écran (`avg_loss = epoch_loss / max(n_samples, 1)`) est gonflée. Un data scientist qui observe la courbe de loss voit une valeur plus haute que la réalité, et la convergence apparaît plus lente qu'elle ne l'est.

**Ce que ça ne casse PAS :** L'entraînement lui-même est correct — le décodeur est bien mis à jour via le chemin 2. Seul le reporting est faux.

**Correction proposée — `train.py` ligne 175 :**
```python
loss_val, d_node, d_edge = pipeline.loss(
    node_logits, edge_logits_arg, gold_node, gold_edge,
    gold_surface=_gold_surface if not decoder_only else None,
)
```

Ne passer `gold_surface` au pipeline que si le backward va consommer le gradient. Quand `decoder_only=True`, la loss pipeline ne doit contenir que la classification causale (gelée, pour monitoring).

---

### N4 — Test e2e de la spec : ordre des clauses inversé par rapport au texte

**Sévérité :** Légère — documentation incorrecte, pas de bug code

**Fichier :** `docs/SPEC_PHASE9_PIPELINE_ML.md`, lignes 559-582

**Code fautif dans la spec :**
```python
rep1 = UDRepresentation(
    tokens=[
        {'lemma': 'tous', 'pos': 'DET', 'dep_rel': 'det', 'morph': {}},
        {'lemma': 'coût', 'pos': 'NOUN', 'dep_rel': 'nsubj', 'morph': {}},
        {'lemma': 'augmenter', 'pos': 'VERB', 'dep_rel': 'root', 'morph': {}},
    ],
    ...
    token_span=(5, 8), lang='fr',         # ← span 5-8 = 2ème clause du texte
)
rep2 = UDRepresentation(
    tokens=[
        {'lemma': 'vente', 'pos': 'NOUN', 'dep_rel': 'nsubj', 'morph': {}},
        {'lemma': 'baisser', 'pos': 'VERB', 'dep_rel': 'root', 'morph': {}},
    ],
    ...
    token_span=(1, 3), lang='fr',         # ← span 1-3 = 1ère clause du texte
)

cir = pipeline.forward([rep1, rep2], 'Si les ventes baissent, tous les coûts augmentent.')
#                        ^^^^  ^^^^
#                     span 5-8  span 1-3 → passées dans l'ordre inverse du texte
```

**Problème :** Dans le texte source, "les ventes baissent" (span ~1-3) vient avant "tous les coûts augmentent" (span ~5-8). Mais `rep1` (span 5-8) est passé en premier et `rep2` (span 1-3) en second. L'arête créée va de rep1→rep2, soit de la clause tardive vers la clause précoce.

Le test passe quand même car les assertions portent sur les attributs (`entity is not None`) et le scope (`!= "specific"`), qui ne dépendent pas de l'ordre. Mais comme documentation d'un usage correct du pipeline, c'est trompeur.

**Correction proposée :** Inverser l'appel :
```python
cir = pipeline.forward([rep2, rep1], 'Si les ventes baissent, tous les coûts augmentent.')
```
Ou réassigner les spans pour que rep1 soit la première clause textuelle.

---

### N5 — `_SCOPE_HINTS` : "un"/"une" mappé à "existential"

**Sévérité :** Légère — faux positifs sur la majorité des clauses françaises

**Fichier :** `gcn-python/src/gcn_python/pipeline/cgnp.py`, lignes 463-468

**Code :**
```python
_SCOPE_HINTS: dict[str, str] = {
    "tous": "universal", "toutes": "universal", "chaque": "universal",
    "tout": "universal", "aucun": "null", "aucune": "null",
    "certains": "existential", "certaines": "existential",
    "un": "existential", "une": "existential",       # ← problème
    "quelques": "partial",
}
```

**Condition de déclenchement — lignes 477-481 :**
```python
for tok in rep.tokens:
    if tok.get("dep_rel") in {"det", "nsubj"} and tok.get("pos") in {"DET", "PRON"}:
        hint = _SCOPE_HINTS.get(tok["lemma"].lower())
        if hint:
            return hint
```

**Problème :** "un" et "une" sont les articles indéfinis les plus fréquents du français. Toute clause contenant un token `{lemma: "un"/"une", pos: "DET", dep_rel: "det"}` sort avec `scope="existential"`.

Exemples concrets :
- "**Une** hausse des coûts entraîne **une** baisse des ventes" → les deux nœuds reçoivent `scope="existential"`
- "**Un** processus de recrutement s'accélère" → `scope="existential"`

En sémantique formelle, c'est correct ("un" = ∃). En modélisation causale, ces phrases décrivent des relations spécifiques, pas des quantifications existentielles. Le scope "existential" est sémantiquement réservé à des énoncés comme "certains coûts augmentent" (∃x, coût(x) ∧ augmente(x)).

**Impact :** Sans corpus annoté pour vérifier, l'estimation est que 30-50% des clauses françaises contiennent "un/une" comme article. Toutes ces clauses auraient un scope incorrect.

**Correction proposée :** Retirer "un" et "une" de `_SCOPE_HINTS`. Les articles indéfinis simples restent à `"specific"` (valeur par défaut). Si la sémantique formelle est voulue, la documenter explicitement.

---

### N6 — `_cached_decode_logits` : écriture sans lecture

**Sévérité :** Légère — dead store, calcul mineur gaspillé

**Fichier :** `gcn-python/src/gcn_python/pipeline/cgnp.py`, ligne 228

**Code :**
```python
self._cached_decode_logits = self.decoder.forward_decode(_vecs)
```

**Qui lit `_cached_decode_logits` :** Personne. Aucune méthode de `CGNPipeline` ne lit cet attribut. `loss()` fait son propre `forward_decode()`. `backward()` utilise `_cached_decode_gradient`. `decode()` du `TrainableDecoder` fait son propre greedy decoding.

`grep -n "_cached_decode_logits" cgnp.py` :
- Ligne 71 : `self._cached_decode_logits = None` (init)
- Ligne 116 : `self._cached_decode_logits = None` (reset)
- Ligne 228 : `self._cached_decode_logits = self.decoder.forward_decode(_vecs)` (écriture)

Trois occurrences, zéro lecture. L'attribut est initialisé, réinitialisé, et écrit — mais jamais consommé.

**Ce finding est lié à N2 :** Le premier `forward_decode` (ligne 228) produit un résultat qui n'a aucun consommateur. Supprimer le bloc lignes 222-228 résout à la fois N2 (double forward) et N6 (dead store).

---

## Tableau de synthèse

| Finding | Fichier | Sévérité | Statut |
|---|---|---|---|
| N1 | `label_builder.py` | Modérée | **Corrigé** — `yaml` → `json`, fichiers convertis en `.json` |
| N2 | `cgnp.py` | Modérée | **Corrigé** — bloc `forward_decode` supprimé de `_forward_from_reps` |
| N3 | `train.py` | Modérée | **Corrigé** — `gold_surface=None` quand `decoder_only=True` |
| N4 | spec lignes 559-582 | Légère | **Corrigé** — `[rep2, rep1]` au lieu de `[rep1, rep2]` |
| N5 | `cgnp.py` | Légère | **Corrigé** — "un"/"une" retirés de `_SCOPE_HINTS` |
| N6 | `cgnp.py` | Légère | **Corrigé** — attribut `_cached_decode_logits` supprimé (résolu par N2) |
| N7 | `SPEC_PHASE9_PIPELINE_ML.md` | Modérée | **Corrigé** — P1 recadré : problème d'intégration, pas défaut du moteur |

**Tests post-correction :** 120 / 120 Python — 137 / 137 Rust

---

### N7 — P1 confond moteur et modèle

**Sévérité :** Modérée — erreur conceptuelle dans la spec qui fausse l'analyse architecturale

**Fichier :** `docs/SPEC_PHASE9_PIPELINE_ML.md`, section P1 (lignes 462-492)

**Le problème :** La spec traitait P1 comme un défaut architectural du moteur :

- Titre : "Le **modèle** entraîné ne peut pas faire de prédictions sur texte non-annoté"
- "rend le **modèle** inutilisable en production"
- "C'est l'équivalent d'entraîner un **LLM** puis d'exiger qu'on étiquette manuellement chaque prompt"
- "La solution doit intégrer la production des features UD dans l'architecture **du moteur lui-même**"

**Pourquoi c'est faux :** GCN-Core est un moteur (engine), pas un modèle. Un moteur définit le graphe de calcul et le contrat d'interface. Le Transformer ne contient pas le tokenizer. PyTorch ne contient pas le data loader de votre tâche. Le moteur GCN-Core prend des `UDRepresentation` en entrée — c'est son contrat, et c'est la bonne séparation de responsabilités.

Les frontends Rust (`gcn-frontend-fr`, `gcn-frontend-en`) produisent déjà des représentations structurées depuis du texte brut (tokenisation, POS-tagging, analyse syntaxique, annotation causale). Ils existent et sont testés (137 tests Rust).

**Ce qui manquait réellement :** un adaptateur d'intégration `frontend Rust → UDRepresentation Python → CGNPipeline.forward()`. C'est un problème de plomberie entre deux composants existants, pas un défaut architectural du moteur.

**L'analogie LLM était trompeuse :** un LLM est un modèle qui prend du texte brut — c'est dans son contrat. Le moteur GCN-Core prend de l'entrée structurée — c'est aussi dans son contrat. Comparer les deux revient à dire que PyTorch est cassé parce qu'il ne tokenize pas le texte.

**Corrections appliquées dans la spec :**

| Emplacement | Avant | Après |
|---|---|---|
| Cartographie ligne 25 | "Modèle inutilisable sur texte non-annoté" | "Pas de chemin d'intégration frontend → moteur pour l'inférence" |
| Table P→C ligne 70 | "Modèle inutilisable (**non résolu** — problème architectural)" | "Pas de chemin d'intégration (**non résolu** — hors scope moteur)" |
| Section P1 titre | "Le modèle entraîné ne peut pas faire de prédictions" | "Pas de chemin d'intégration frontend → moteur pour l'inférence" |
| Section P1 corps | Analogie LLM + "solution doit s'intégrer dans le moteur" | Séparation moteur/frontend + adaptateur manquant |
| Table limitations | "Problème architectural" | "Hors scope moteur" |
| P2d ligne 430 | "Le modèle ne peut pas apprendre" | "Le décodeur ne peut pas apprendre" |

---

## Détail des corrections appliquées

| Finding | Fichier modifié | Changement |
|---|---|---|
| N1 | `label_builder.py` | `import yaml` → `import json` ; `yaml.safe_load()` → `json.load()` ; cherche `nominalizations.json` |
| N1 | `gcn-references/taxonomies/fr/nominalizations.yaml` | Converti en `nominalizations.json` (même contenu) |
| N1 | `gcn-references/taxonomies/en/nominalizations.yaml` | Converti en `nominalizations.json` (même contenu) |
| N1 | `gcn-python/README.md:797` | Commentaire mis à jour `nominalizations.yaml` → `.json` |
| N2+N6 | `cgnp.py` | Supprimé le bloc lignes 222-228 (`forward_decode` en inférence) + attribut `_cached_decode_logits` |
| N2+N6 | `test_trainable_decoder.py` | 2 assertions `_cached_decode_logits` retirées |
| N3 | `train.py:175` | `gold_surface=_gold_surface if not decoder_only else None` |
| N4 | `SPEC_PHASE9_PIPELINE_ML.md:581` | `[rep1, rep2]` → `[rep2, rep1]` |
| N5 | `cgnp.py:467-468` | `"un": "existential", "une": "existential"` retirés de `_SCOPE_HINTS` |
| N7 | `SPEC_PHASE9_PIPELINE_ML.md` | P1 recadré — titre, corps, cartographie, table limitations, P2d |

---

## Verdict

Tous les findings (F1-F11 précédents + N1-N7 nouveaux) sont **résolus**. La spec est cohérente avec le code corrigé et avec l'architecture moteur. 120 tests Python et 137 tests Rust passent sans régression.
