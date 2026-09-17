# Plan Global — GCN Causal Engine (v2, corrigé)

**Date :** 2026-09-17  
**Auteur :** Michel Tendeng  
**Version :** 2 — toutes les ambiguïtés de la v1 corrigées

---

## État actuel vérifié par code (valeurs exactes)

| Composant | État | Détail vérifié |
|-----------|------|----------------|
| Pipeline ML (layers 1–3, training) | ✅ Fonctionnel | 195 tests Python |
| Tests Rust | ✅ Fonctionnel | 137 tests, build OK |
| gcn-scraper (6 sources) | ✅ Fonctionnel | Wikipedia FR/EN, HAL, Python/Rust docs |
| gcn-annotate (LLM) | ✅ Fonctionnel | Anthropic/OpenAI, retry, validation |
| Dataset synthétique | ✅ Utilisable | 990 phrases, 11 classes parfaitement balancées (90/classe) |
| Dataset réel | ❌ **Inutilisable** | **678 phrases** (pas 1356), 0 token UD, 0 `causal_pattern`, 6 relations absentes |
| Évaluation | ❌ **Cassée** | Métriques calculées sur le train uniquement |

### Valeurs techniques exactes

```
d_clause actuel     = 79   (18 UPOS + 38 DEP + 5 SUBJ_POS + 5 TENSE + 4 ASPECT + 5 MOOD + 1 POL + 3 FLAGS)
d_clause après P5   = 90   (79 + 11 RELATION_TYPES)
d_conn              = 85
d_edge actuel       = 247  (2×79 + 85 + 4)
NODE_TYPES          = 7
RELATION_TYPES      = 11   ["cause","enable","prevent","condition","concession","sequence",
                             "motivation","filter","opposition","data_dependency","control_dependency"]
```

### État du dataset réel (vérifié)

```
Fichiers : annotated.json (678 phrases source), train.json (542), val.json (67), test.json (69)
Tokens UD : 0 pour toutes les phrases
causal_pattern : absent (chaîne vide) pour toutes les 678 phrases
Relations dans le CIR :
  cause        = 402 (59.3 %)
  enable       = 167 (24.6 %)
  condition    =  56  (8.3 %)
  concession   =  40  (5.9 %)
  prevent      =  13  (1.9 %)
  motivation   =   0  — ABSENT
  sequence     =   0  — ABSENT
  filter       =   0  — ABSENT
  opposition   =   0  — ABSENT
  data_dependency    = 0  — ABSENT
  control_dependency = 0  — ABSENT
Baseline triviale (prédire toujours "cause") : edge accuracy = 59.3 %
```

### Problèmes centraux (ordonnés par impact)

1. **[BLOQUANT] Dataset réel : 0 token UD** → `reps_from_sentence()` retourne `[]` → loss=0
2. **[BLOQUANT] Alignement `token_span` CIR ↔ IDs spaCy non résolu** → même avec des tokens UD, les filtres par span échouent si les offsets ne correspondent pas
3. **[BLOQUANT] Évaluation sur le train set uniquement** → aucune mesure fiable de la généralisation
4. **[HAUT] Dataset réel : 0 `causal_pattern` et 6 relations absentes** → Phase 5 ne peut pas s'appliquer
5. **[HAUT] Qualité CIR réel non validée** → entraîner sur du bruit produit un modèle incorrect
6. **[MOYEN] Aucune régularisation** → mémorisation probable sur le corpus synthétique

---

## Phase 1 — Rendre le dataset réel utilisable (BLOQUANT)

**Objectif précis :** Pour chaque phrase du dataset réel, produire la liste complète de
tokens UD (id, form, lemma, pos, dep_rel, dep_head, morph) **avec les IDs alignés sur les
`token_span` existants dans le CIR**, de façon que `_rep_from_clause` retourne des
représentations non vides.

**Contrainte architecturale :** spaCy est interdit dans le moteur (`gcn-python`).
Il est utilisé ici uniquement dans `gcn-tools/gcn-scraper/` — conforme à la règle.

---

### Étape 1.1 — Analyser l'alignement `token_span` CIR ↔ tokenisation spaCy

**Avant d'écrire une seule ligne de code**, exécuter ce script de diagnostic :

```python
# gcn-tools/gcn-scraper/src/gcn_scraper/diagnose_spans.py
import json, spacy
nlp = spacy.load("fr_core_news_sm")
data = json.loads(open("gcn-datasets/real/annotated.json").read())
sents = data["document"]["sentences"]

span_types = {"char": 0, "token_1based": 0, "token_0based": 0, "unknown": 0}
for s in sents[:50]:
    text = s["text"]
    doc = nlp(text)
    tokens = list(doc)
    for node in s.get("cir", {}).get("nodes", []):
        start, end = node["token_span"]
        # Tester chaque hypothèse
        if 0 <= start < end <= len(text):
            span_types["char"] += 1
        if 1 <= start <= end <= len(tokens):
            span_types["token_1based"] += 1
        if 0 <= start < end <= len(tokens):
            span_types["token_0based"] += 1

print("Diagnostic spans:", span_types)
```

**Les trois cas possibles :**

- **Cas A — offsets de caractères** : `token_span = [char_start, char_end]`
  → Mapping : utiliser `token.idx` (offset début char) et `token.idx + len(token.text)`
  → Affecter à chaque token UD : `id = token.i + 1` ; garder les spans caractères tels quels
  et convertir en token IDs dans `_rep_from_clause` (nécessite une modification du moteur)

- **Cas B — IDs de tokens 0-based** : `token_span = [tok_idx_start, tok_idx_end]`
  → Affecter : `id = token.i` (0-based) ; `_rep_from_clause` filtre avec `span_start <= t.id <= span_end` → fonctionne directement

- **Cas C — IDs de tokens 1-based** : `token_span = [tok_idx_start+1, tok_idx_end+1]`
  → Affecter : `id = token.i + 1` (1-based) → fonctionne directement

Le diagnostic détermine le cas avant tout autre développement.

---

### Étape 1.2 — Créer `gcn-tools/gcn-scraper/src/gcn_scraper/ud_annotator.py`

Ce module reçoit un texte et retourne la liste de tokens UD **avec les IDs alignés** sur
la convention détectée en étape 1.1.

```python
import spacy
from typing import Literal

_NLP_CACHE: dict = {}

def _load_nlp(lang: str):
    if lang not in _NLP_CACHE:
        model = {"fr": "fr_core_news_sm", "en": "en_core_web_sm"}.get(lang)
        if model is None:
            raise ValueError(f"Langue non supportée : {lang!r}. Valeurs: 'fr', 'en'")
        _NLP_CACHE[lang] = spacy.load(model)
    return _NLP_CACHE[lang]


def annotate_ud(
    text: str,
    lang: str = "fr",
    id_convention: Literal["0based", "1based"] = "1based",
) -> list[dict]:
    """
    Texte brut → liste de dicts token compatibles gcn-nl.

    Chaque dict contient : id, form, lemma, pos, dep_rel, dep_head, morph.
    Le champ `gcn` (causal_type, causal_class) est ABSENT — le modèle l'infère.

    id_convention :
      "1based" → ids commencent à 1 (défaut, compatible avec le CIR annoté)
      "0based" → ids commencent à 0

    dep_head = 0 si le token est la racine syntaxique (root), id du parent sinon.
    morph = dict avec clés "Tense", "Aspect", "Mood", "Polarity" ; valeur "_absent"
            si la catégorie morphologique est absente.
    """
    nlp = _load_nlp(lang)
    doc = nlp(text)
    offset = 1 if id_convention == "1based" else 0
    tokens = []
    for tok in doc:
        tok_id = tok.i + offset
        # dep_head = 0 si root, sinon id du parent
        if tok.dep_.lower() == "root":
            dep_head = 0
        else:
            dep_head = tok.head.i + offset
        morph = {
            "Tense":    tok.morph.get("Tense",    ["_absent"])[0],
            "Aspect":   tok.morph.get("Aspect",   ["_absent"])[0],
            "Mood":     tok.morph.get("Mood",     ["_absent"])[0],
            "Polarity": tok.morph.get("Polarity", ["_absent"])[0],
        }
        tokens.append({
            "id":       tok_id,
            "form":     tok.text,
            "lemma":    tok.lemma_,
            "pos":      tok.pos_,        # UPOS (ADJ, VERB, NOUN, …)
            "dep_rel":  tok.dep_,        # UD dep relation (nsubj, root, obj, …)
            "dep_head": dep_head,
            "morph":    morph,
            # gcn absent intentionnellement — le modèle doit l'inférer
        })
    return tokens
```

**Dépendances à ajouter dans `gcn-tools/gcn-scraper/pyproject.toml` :**
```toml
[tool.poetry.dependencies]
spacy = ">=3.7,<4.0"
```

**Modèles à installer séparément (ne pas ajouter comme dépendances PyPI) :**
```bash
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

---

### Étape 1.3 — Valider la qualité du CIR réel avant annotation UD

**Avant d'annoter 678 phrases**, valider que les CIR existants sont exploitables.
Créer `gcn-tools/gcn-scraper/src/gcn_scraper/validate_cir.py` :

```python
"""
Valide le CIR de chaque phrase du dataset réel :
1. Chaque node a un token_span [start, end] avec start < end
2. Les token_span de nodes distincts ne se chevauchent pas à l'identique
3. Chaque edge référence des node_id qui existent
4. Les relations sont dans RELATION_TYPES

Produit un rapport : nb phrases valides, invalides, exemples d'erreurs.
"""
```

Toute phrase avec un CIR invalide doit être **exclue ou re-annotée** avant Phase 1.4.
Le plan ne peut pas progresser sur des CIR incorrects.

---

### Étape 1.4 — Script d'annotation UD du dataset réel

Créer `gcn-tools/gcn-scraper/src/gcn_scraper/annotate_real_dataset.py` :

```python
"""
Lit gcn-datasets/real/annotated.json (678 phrases, CIR présent, 0 token UD).
Pour chaque phrase :
  1. Appelle annotate_ud(text, lang, id_convention) avec id_convention déterminée en 1.1
  2. Injecte la liste de tokens dans sentence["tokens"]
  3. NE MODIFIE PAS le CIR (nodes, edges)
Écrit gcn-datasets/real/annotated_ud.json.
Rapport final : nb succès, nb échecs, nb tokens moyens par phrase.
"""
```

**Fichier en entrée :** `gcn-datasets/real/annotated.json` (678 phrases, CIR présent)
**Fichier en sortie :** `gcn-datasets/real/annotated_ud.json` (678 phrases, CIR + tokens UD)

---

### Étape 1.5 — Valider que `reps_from_sentence` retourne des reps non vides

```python
# Script de validation post-annotation
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
import json, pathlib

data = json.loads(pathlib.Path("gcn-datasets/real/annotated_ud.json").read_text())
sents_data = data["document"]["sentences"]

n_ok = n_empty = 0
for s in sents_data:
    # Construire un SentenceRecord minimal pour test
    from gcn_python.data.json_reader import load_sentences
    # Écrire une phrase dans un fichier tmp et la charger
    ...
    reps, idxs, conn = reps_from_sentence(sample.sentence)
    if reps:
        n_ok += 1
    else:
        n_empty += 1
        print("VIDE:", s["text"][:80])

print(f"OK: {n_ok}, VIDE: {n_empty}")
assert n_ok > n_empty * 10, "Trop de reps vides — vérifier l'alignement token_span"
```

---

### Étape 1.6 — Re-générer les splits train/val/test avec tokens UD

À partir de `annotated_ud.json` (uniquement les phrases dont le CIR est valide et les
reps non vides), créer les 3 splits :

**Règles de split :**
- Ratio : 70 % train / 15 % val / 15 % test
- Stratification : sur la relation de chaque edge (pas sur `causal_pattern` qui est absent)
- Seed fixe (42) pour reproductibilité
- Les 5 relations présentes (cause, enable, condition, concession, prevent) doivent
  être représentées dans val et test avec au minimum 5 exemples chacune
- Si `prevent` (13 phrases) ne peut pas satisfaire ce critère, mentionner explicitement
  que cette classe est sous-représentée dans les métriques de validation

**Fichiers produits :**
```
gcn-datasets/real/train.json   (~473 phrases  — 70 %)
gcn-datasets/real/val.json     (~101 phrases  — 15 %)
gcn-datasets/real/test.json    (~104 phrases  — 15 %)
```

**Validation du split :**
```bash
# Vérifier la distribution par relation dans chaque split
python -c "
import json, pathlib
from collections import Counter
for split in ['train', 'val', 'test']:
    d = json.loads(pathlib.Path(f'gcn-datasets/real/{split}.json').read_text())
    sents = d.get('document',{}).get('sentences', d.get('sentences',[]))
    rels = Counter(e['relation'] for s in sents for e in s.get('cir',{}).get('edges',[]))
    print(f'{split}: {len(sents)} phrases, relations: {dict(rels)}')
"
```

---

## Phase 2 — Évaluation correcte sur val set

**Prérequis :** Phase 2 peut démarrer **maintenant** (indépendant de Phase 1).
Pour tester l'implémentation avant que le dataset réel soit prêt, utiliser un split
manuel du corpus synthétique (800 train / 190 val).

---

### Étape 2.0 — Créer le split synthétique train/val

```bash
# Créer les répertoires
mkdir -p gcn-datasets/corpus_train gcn-datasets/corpus_val

# Splitter generated_1000.json (990 phrases balancées, 90/classe)
# → 792 train (80 %) + 198 val (20 %), stratifié sur causal_pattern
python -c "
import json, pathlib, random
random.seed(42)
data = json.loads(pathlib.Path('gcn-datasets/corpus/generated_1000.json').read_text())
sents = data['document']['sentences']
# Grouper par causal_pattern (90 phrases par classe)
from collections import defaultdict
by_pattern = defaultdict(list)
for s in sents:
    by_pattern[s['causal_pattern']].append(s)
train_sents, val_sents = [], []
for pattern, group in by_pattern.items():
    random.shuffle(group)
    cut = int(0.8 * len(group))   # 72 train, 18 val par classe
    train_sents.extend(group[:cut])
    val_sents.extend(group[cut:])
# Écrire les fichiers
for path, sents_list in [
    ('gcn-datasets/corpus_train/train.json', train_sents),
    ('gcn-datasets/corpus_val/val.json', val_sents),
]:
    pathlib.Path(path).write_text(json.dumps(
        {'document': {'sentences': sents_list}}, ensure_ascii=False, indent=2))
print(f'Train: {len(train_sents)}, Val: {len(val_sents)}')
"
```

---

### Étape 2.1 — `--val-dir` dans `training/train.py`

Ajouter l'argument CLI :
```python
@click.option("--val-dir", default=None, type=click.Path(path_type=Path),
              help="Répertoire val JSON (optionnel). Métriques val calculées à chaque epoch.")
```

Comportement exact :
1. Si `--val-dir` est fourni, créer `val_loader = GCNDataLoader(val_dir, shuffle=False)`
2. À chaque fin d'epoch, avant de loguer les métriques train :
   a. Désactiver le dropout sur TOUS les composants (encodeur + layers R-GCN)
   b. Faire un pass forward complet sur val_loader (pas de backward, pas d'accumulation)
   c. Calculer val_loss, val_node_accuracy, val_node_macro_f1, val_edge_accuracy, val_edge_macro_f1
   d. Réactiver le dropout

Mode eval/train à basculer :
```python
def _set_training_mode(pipeline: CGNPipeline, training: bool) -> None:
    """Bascule TOUS les composants avec dropout en mode eval ou train."""
    if hasattr(pipeline.encoder, 'training'):
        pipeline.encoder.training = training
    for layer in pipeline._graph_layers:
        if hasattr(layer, 'training'):
            layer.training = training
```

Appeler `_set_training_mode(pipeline, False)` avant le val pass et
`_set_training_mode(pipeline, True)` après.

---

### Étape 2.2 — Early stopping + best model

Ajouter l'argument CLI :
```python
@click.option("--patience", default=0, show_default=True, type=int,
              help="Epochs sans amélioration de val_node_macro_f1 avant arrêt. "
                   "0 = désactivé (défaut). Requiert --val-dir.")
```

Comportement exact :
- Si `--patience > 0` ET `--val-dir` est fourni :
  - Tracker `best_val_f1 = -1.0` et `best_epoch = 0`
  - À chaque epoch : si `val_node_macro_f1 > best_val_f1` :
    - Sauvegarder un checkpoint temporaire dans `<output>.best.npz`
    - Mettre à jour `best_val_f1` et `best_epoch`
  - Si `epoch - best_epoch >= patience` : arrêter la boucle
  - À la fin : copier `<output>.best.npz` vers `<output>`
  - Supprimer `<output>.best.npz`
- Si `--patience > 0` mais pas de `--val-dir` : lever `ClickException` explicite
- Si `--patience == 0` : pas d'early stopping, pas de sauvegarde intermédiaire

---

### Étape 2.3 — Shuffling dans `data/loader.py`

Ajouter au constructeur de `GCNDataLoader` :
```python
def __init__(self, data_dir: Path, repeat: bool = False,
             all_pairs: bool = False, shuffle: bool = False, seed: int = 42):
    ...
    self._shuffle = shuffle
    self._rng = np.random.default_rng(seed) if shuffle else None
```

Dans `__iter__` : si `self._shuffle`, créer une permutation des records au début
de chaque passage (y compris le premier) :
```python
def __iter__(self):
    while True:
        records = self._records
        if self._shuffle:
            idxs = self._rng.permutation(len(records))
            records = [records[i] for i in idxs]
        for rec in records:
            ...
```

Dans `train.py` : passer `shuffle=True` au train loader, `shuffle=False` au val loader.

---

### Étape 2.4 — Étendre `evaluation/recorder.py`

Champs à ajouter dans le CSV et le JSON :
```
val_loss, val_node_accuracy, val_node_macro_f1, val_edge_accuracy, val_edge_macro_f1
```

Ces champs sont `None` (ou `-1`) si `--val-dir` n'est pas fourni.

Nouvelle méthode :
```python
def best_epoch(self, metric: str = "val_node_macro_f1") -> tuple[int, float]:
    """
    Retourne (epoch, valeur) de l'epoch avec la meilleure valeur de la métrique.
    Lève ValueError si la métrique n'a aucune valeur non-None.
    """
```

---

### Étape 2.5 — Tests

Dans `tests/test_training.py` :

1. Test `--val-dir` produit CSV avec colonnes `val_*` non nulles
2. Test best model sauvegardé quand `val_node_macro_f1` s'améliore
3. Test early stopping arrête avant `--epochs` quand patience atteinte
4. Test early stopping lève `ClickException` si `--patience > 0` sans `--val-dir`
5. Test shuffle permute les samples à chaque epoch (seeds différents → ordres différents)
6. Test `_set_training_mode(pipeline, False)` → dropout désactivé sur encodeur et R-GCN

---

## Phase 3 — Régularisation

**Prérequis :** indépendant — peut démarrer en parallèle avec Phases 1 et 2.

---

### Étape 3.1 — Weight decay (L2) dans `layer2/reference.py`

Ajouter au constructeur de `MLPEncoder` :
```python
def __init__(self, ..., weight_decay: float = 0.0):
    self.weight_decay = weight_decay
```

Modifier `_apply_grads` (appliqué après normalisation mini-batch le cas échéant) :
```python
def _apply_grads(self, layers, grads, lr):
    for layer, (dW, db) in zip(layers, grads):
        # Weight decay appliqué sur W uniquement (pas sur les biais)
        layer.W -= lr * dW + lr * self.weight_decay * layer.W
        layer.b -= lr * db
```

**Note :** `lr * self.weight_decay * layer.W` est indépendant du gradient — c'est la
régularisation L2. Cela est correct même en mode mini-batch car weight decay est une
contrainte sur les poids, pas sur les gradients.

Ajouter dans `train.py` :
```python
@click.option("--weight-decay", default=0.0, show_default=True, type=float,
              help="Coefficient de régularisation L2 sur les poids MLP. 0 = désactivé.")
```
Passer `weight_decay` au constructeur `MLPEncoder`.

---

### Étape 3.2 — Dropout sur les features d'entrée R-GCN

**Dans `layer3/reference.py` — `RGCNLayer` :**

Ajouter au constructeur :
```python
def __init__(self, ..., dropout: float = 0.0):
    self.dropout = dropout
    self.training = True
```

Ajouter en début de `message_pass`, avant tout calcul :
```python
if self.dropout > 0.0 and self.training:
    mask = (np.random.random(node_features.shape) > self.dropout).astype(np.float32)
    node_features = node_features * mask / (1.0 - self.dropout)
    # Copie locale — ne modifie pas le tableau d'entrée
```

**Dans `layer3/gat.py` — `RGCNLayerGAT` :**

Ajouter au constructeur :
```python
self.dropout_rate = dropout  # float, défaut 0.0
self.training = True
```

Dans `message_pass` (ou `forward`), avant `H @ W_r` :
```python
if self.dropout_rate > 0.0 and self.training:
    mask = torch.bernoulli(
        torch.full(H.shape, 1.0 - self.dropout_rate, device=H.device)
    ) / (1.0 - self.dropout_rate)
    H = H * mask
```

Ajouter dans `train.py` :
```python
@click.option("--rgcn-dropout", default=0.0, show_default=True, type=float,
              help="Dropout sur les features d'entrée des couches R-GCN/GAT. 0 = désactivé.")
```
Passer `dropout=rgcn_dropout` aux constructeurs de `RGCNLayer` et `RGCNLayerGAT`.

**Important :** `_set_training_mode` (Phase 2) bascule `layer.training` sur chaque layer —
ce flag désactivera le dropout R-GCN pendant le val pass.

---

### Étape 3.3 — Label smoothing dans `pipeline/cgnp.py`

Modifier la signature de `_cross_entropy` :
```python
def _cross_entropy(
    logits: np.ndarray,
    labels: np.ndarray,
    class_weights: np.ndarray | None = None,
    label_smoothing: float = 0.0,
) -> tuple[float, np.ndarray]:
```

Logique :
```python
N, C = logits.shape
probs = _softmax(logits)

if label_smoothing > 0.0:
    # Distribution lissée : masse (1 - eps) sur la vraie classe, eps/(C-1) sur les autres
    y_smooth = np.full((N, C), label_smoothing / max(C - 1, 1), dtype=np.float32)
    y_smooth[np.arange(N), labels] = 1.0 - label_smoothing
    if class_weights is not None:
        w_n = class_weights[labels]   # (N,) — poids de la classe gold
        per_sample_loss = -(y_smooth * np.log(probs + 1e-9)).sum(axis=1) * w_n
    else:
        per_sample_loss = -(y_smooth * np.log(probs + 1e-9)).sum(axis=1)
    # Gradient : d(L)/d(logit) = probs - y_smooth (normalisé par N, pondéré si weights)
    d_logits = probs - y_smooth
    if class_weights is not None:
        d_logits *= class_weights[labels][:, np.newaxis]
    d_logits /= N
else:
    # Comportement identique à l'implémentation actuelle
    per_sample_loss = -np.log(probs[np.arange(N), labels] + 1e-9)
    if class_weights is not None:
        per_sample_loss *= class_weights[labels]
    d_logits = probs.copy()
    d_logits[np.arange(N), labels] -= 1.0
    if class_weights is not None:
        d_logits *= class_weights[labels][:, np.newaxis]
    d_logits /= N

loss = float(per_sample_loss.mean())
return loss, d_logits
```

Propager `label_smoothing` depuis `CGNPipeline.loss()` jusqu'aux deux appels de
`_cross_entropy` (nœuds et arêtes).

Ajouter dans `train.py` :
```python
@click.option("--label-smoothing", default=0.0, show_default=True, type=float,
              help="Lissage des labels [0, 1]. 0 = one-hot strict. Recommandé : 0.05–0.1.")
```

---

### Étape 3.4 — Tests

Dans `tests/test_pipeline.py` (ou `test_training.py`) :

1. Test weight decay : après 20 epochs avec `weight_decay=0.1`, `norm(W) < norm(W_init)`
2. Test dropout désactivé : `encoder.training = False` → forward déterministe (même sortie deux fois)
3. Test dropout activé : `encoder.training = True` → forward stochastique (sorties différentes)
4. Test label smoothing neutre : avec `label_smoothing=0`, comportement identique à sans label smoothing
5. Test label smoothing actif : avec `label_smoothing=0.1` et prédiction parfaite,
   `loss > 0` (contrairement au cas sans smoothing où log(1) = 0)

---

## Phase 4 — Dataset réel à grande échelle (5 000+ phrases)

**Prérequis :** Phase 1 complète (pipeline UD opérationnel).

---

### Étape 4.1 — Annotation `causal_pattern` du dataset réel existant

**Problème :** les 678 phrases réelles n'ont aucun `causal_pattern`. Ce champ est
nécessaire pour la Phase 5.

**Solution :** dériver `causal_pattern` automatiquement depuis le CIR existant.

Règle : `causal_pattern` = relation la plus fréquente parmi les edges de la phrase.
En cas d'égalité : prendre la relation avec le plus haut `confidence` moyen.

```python
from collections import Counter

def derive_causal_pattern(sentence: dict) -> str:
    """
    Derive causal_pattern depuis les edges du CIR.
    Retourne la relation dominante, ou "" si pas d'edges.
    """
    edges = sentence.get("cir", {}).get("edges", [])
    if not edges:
        return ""
    rel_counts = Counter(e["relation"] for e in edges)
    return rel_counts.most_common(1)[0][0]
```

Ce script s'exécute sur `annotated_ud.json` et enrichit chaque phrase avec
`causal_pattern` avant de produire `annotated_ud_cp.json`.

**Limitation documentée :** le `causal_pattern` ainsi dérivé = relation du CIR gold.
Pendant l'entraînement, le modèle reçoit cette information pour les données réelles.
**Ce n'est pas un data leak pour les nœuds** (node_type est une cible différente de
causal_pattern). **C'est un data leak partiel pour les arêtes** sur les phrases à une
seule arête. La stratégie est de le documenter, entraîner avec, et mesurer l'impact en
Phase 6 en comparant avec/sans cette feature.

---

### Étape 4.2 — Scraping de nouvelles phrases

```bash
gcn-scrape \
  --sources wikipedia_fr wikipedia_en hal python_docs rust_docs github_issues \
  --out-dir gcn-datasets/raw/batch2/ \
  --limit 20000
```

**Critères de filtrage des phrases brutes** (à implémenter dans le scraper) :
- Longueur : entre 10 et 150 tokens (≈ 50–900 caractères)
- Présence d'au moins un connecteur causal (liste : `FR_DURATIVE_MARKERS`,
  `FR_INFINITIVE_MARKERS`, SCONJ, CCONJ du lexique gcn-knowledge)
- Langue détectée correctement (langdetect ou spaCy)
- Pas de doublon exact avec les phrases existantes
- Pas de fragments (pas de verbe conjugué = rejeté)
- Pas de pure URL / formule mathématique / code source

Objectif : 15 000 phrases brutes pour en garder ~7 000 après filtrage.

---

### Étape 4.3 — Annotation UD + CausalIR

```bash
# Annotation UD
python -m gcn_scraper.annotate_real_dataset \
  --input gcn-datasets/raw/batch2/ \
  --out gcn-datasets/real/batch2_ud.json

# Annotation CausalIR via LLM
gcn-annotate annotate \
  --input gcn-datasets/real/batch2_ud.json \
  --out gcn-datasets/real/batch2_cir.json \
  --model claude-sonnet-5

# Dériver causal_pattern
python -m gcn_scraper.derive_causal_pattern \
  --input gcn-datasets/real/batch2_cir.json \
  --out gcn-datasets/real/batch2_full.json
```

Objectif : 5 000 phrases avec tokens UD + CIR valide + causal_pattern.

---

### Étape 4.4 — Combiner et splitter

Combiner `annotated_ud_cp.json` (678 phrases) + `batch2_full.json` (~5 000 phrases).

Validation de la distribution avant split :
```python
# Vérifier que les 11 relations sont représentées
# Minimum acceptable : 30 exemples par relation dans le train
# Si une relation < 30 exemples → cibler des phrases supplémentaires pour cette relation
```

Split stratifié 70/15/15 sur la relation des edges :
```
gcn-datasets/real/train_v2.json  (~3 900 phrases)
gcn-datasets/real/val_v2.json    (~840 phrases)
gcn-datasets/real/test_v2.json   (~840 phrases)
```

---

## Phase 5 — `causal_pattern` comme feature d'apprentissage

**Prérequis :** Phase 1 terminée (dataset réel avec tokens UD ET `causal_pattern` dérivé).

**Avertissement data leak :** sur les phrases à une seule arête, `causal_pattern` = relation
gold → le modèle reçoit partiellement la réponse. Ce comportement est documenté et sera
mesuré en Phase 6 (comparaison avec/sans `--causal-pattern-feature`).

**Impact sur les dimensions :**
```
d_clause AVANT Phase 5  = 79
RELATION_TYPES          = 11  (one-hot du causal_pattern)
d_clause APRÈS Phase 5  = 90  (79 + 11)
d_edge_closed_loop change automatiquement via vocab.d_edge_closed_loop()
Checkpoints existants   → INVALIDÉS (d_clause change)
```

---

### Étape 5.1 — `data/schema.py`

```python
@dataclass
class SentenceRecord:
    id: str
    text: str
    tokens: list[TokenRecord]
    clauses: list[ClauseRecord]
    edges: list[EdgeRecord]
    lang: str = ""
    causal_pattern: str = ""
    # Valeur "" → vecteur nul dans vectorize_clause
    # Valeur "cause", "condition", etc. → one-hot sur RELATION_TYPES
```

---

### Étape 5.2 — `data/json_reader.py`

Dans `_parse_dataset_sentence`, ajouter :
```python
causal_pattern=s.get("causal_pattern", ""),
```

---

### Étape 5.3 — `layer1/representation.py`

```python
@dataclass
class UDRepresentation:
    tokens: list[dict]
    root_lemma: str
    root_pos: str
    root_dep_rel: str
    root_morph: dict
    subject_pos: str | None
    has_object: bool
    has_advcl: bool
    has_temporal_obl: bool
    token_span: tuple[int, int]
    causal_pattern: str = ""   # ← NOUVEAU
```

---

### Étape 5.4 — `data/loader.py` — propagation dans `reps_from_sentence`

Modifier `reps_from_sentence` pour accepter `causal_pattern` en paramètre et le
propager à chaque rep :

```python
def reps_from_sentence(
    rec: SentenceRecord,
) -> tuple[list[UDRepresentation], list[int], list[UDRepresentation | None]]:
    ...
    for i, clause in enumerate(rec.clauses):
        rep = _rep_from_clause(clause, rec.tokens)
        if rep is not None:
            rep.causal_pattern = rec.causal_pattern   # ← propagation
            result.append(rep)
            valid_indices.append(i)
    ...
```

---

### Étape 5.5 — `data/loader.py` — supprimer `gcn_causal_type` des heuristiques

**`_rep_from_clause`** — le critère `gcn_causal_type == "verbe"` est supprimé.
Le fallback POS est le seul critère :

```python
# AVANT
root_tok = (
    next((t for t in span_toks
          if t.gcn_causal_type == "verbe" and t.pos in {"VERB", "AUX"}), None)
    or next((t for t in span_toks if t.pos in {"VERB", "AUX"}), None)
    or next((t for t in span_toks if t.pos in {"NOUN", "PROPN"}), None)
    or span_toks[0]
)

# APRÈS
root_tok = (
    next((t for t in span_toks if t.pos in {"VERB", "AUX"}), None)
    or next((t for t in span_toks if t.pos in {"NOUN", "PROPN"}), None)
    or span_toks[0]
)
```

**`_connector_between`** — le critère `gcn_causal_type == "conjonction"` est supprimé :

```python
# AVANT
tok = (
    next((t for t in gap_toks if t.gcn_causal_type == "conjonction"), None)
    or next((t for t in gap_toks if t.pos in {"SCONJ", "CCONJ", "ADP"}), None)
)

# APRÈS
tok = next((t for t in gap_toks if t.pos in {"SCONJ", "CCONJ", "ADP"}), None)
```

---

### Étape 5.6 — `layer1/features.py` — `FeatureVocabulary`

Ajouter l'import de `RELATION_TYPES` si absent, puis le champ :

```python
from ..constants import (
    UPOS_TAGS, UD_DEP_RELS, UD_TENSE_VALUES, UD_ASPECT_VALUES,
    UD_MOOD_VALUES, SUBJECT_POS_CATS, RELATION_TYPES,   # RELATION_TYPES ajouté
)

@dataclass
class FeatureVocabulary:
    upos_tags: list[str] = field(default_factory=lambda: list(UPOS_TAGS))
    dep_rels: list[str] = field(default_factory=lambda: list(UD_DEP_RELS))
    tense_values: list[str] = field(default_factory=lambda: list(UD_TENSE_VALUES))
    aspect_values: list[str] = field(default_factory=lambda: list(UD_ASPECT_VALUES))
    mood_values: list[str] = field(default_factory=lambda: list(UD_MOOD_VALUES))
    subject_pos_cats: list[str] = field(default_factory=lambda: list(SUBJECT_POS_CATS))
    relation_types: list[str] = field(default_factory=lambda: list(RELATION_TYPES))  # ← NOUVEAU

    @property
    def d_clause(self) -> int:
        return (
            len(self.upos_tags)            # 18
            + len(self.dep_rels)           # 38
            + len(self.subject_pos_cats)   #  5
            + len(self.tense_values)       #  5
            + len(self.aspect_values)      #  4
            + len(self.mood_values)        #  5
            + 1                            #  1  polarity
            + 3                            #  3  structural flags
            + len(self.relation_types)     # 11  causal_pattern one-hot  ← NOUVEAU
        )
        # 79 + 11 = 90
```

Inclure `relation_types` dans `to_json()` et `from_json()`.

---

### Étape 5.7 — `layer1/features.py` — `vectorize_clause`

Ajouter en fin de liste `parts`, **après le word embedding** :
```python
parts.append(_one_hot(rep.causal_pattern or "", vocab.relation_types))
# "" → vecteur nul (neutre, représente l'absence de causal_pattern)
# "cause" → index 0 = 1.0, reste = 0.0
# "condition" → index 3 = 1.0, reste = 0.0
```

---

### Étape 5.8 — Ajouter `causal_pattern` dans `fr_causal_basic.json`

Les 3 phrases de `gcn-datasets/examples/fr_causal_basic.json` n'ont pas de
`causal_pattern`. Dériver depuis les edges du CIR (même règle qu'en Phase 4.1) :
- s001 : 1 edge `"condition"` → `"causal_pattern": "condition"`
- s002 : 1 edge `"cause"` → `"causal_pattern": "cause"`
- s003 : 1 edge `"motivation"` → `"causal_pattern": "motivation"`

`fr_causal_cycles.json` : causalité inter-phrases, pas de relation intra-phrase
→ laisser `causal_pattern` absent (vecteur nul = neutre, ce qui est correct).

---

### Étape 5.9 — Tests — mise à jour des hardcodes `d_clause`

Les tests qui hardcodent `d_clause == 79` doivent être mis à jour vers `90`.
Les tests qui utilisent `vocab.d_clause` directement sont automatiquement corrects.

Fichiers à inspecter :
```bash
grep -rn "d_clause\|== 79\|== 75" gcn-python/tests/
```

Fichiers impactés connus : `test_layer2.py`, `test_pipeline.py`, `test_checkpoint.py`,
`test_trainable_decoder.py`, `test_frontend_bridge.py`, `test_training.py`.

---

## Phase 6 — Entraînement et évaluation

**Prérequis :** Phases 1 à 5 complètes.

---

### Étape 6.0 — Créer le split synthétique train/val (requis avant 6.1)

Déjà décrit en Phase 2 étape 2.0. Le split doit exister avant tout entraînement de
validation.

---

### Étape 6.1 — Baseline synthétique (avec val set séparé)

```bash
gcn-train \
  --data-dir  gcn-datasets/corpus_train/ \
  --val-dir   gcn-datasets/corpus_val/ \
  --epochs 100 \
  --lr 0.001 \
  --use-attention \
  --bidirectional \
  --embedding-dim 50 \
  --edge-loss-weight 2.0 \
  --weighted-loss \
  --weight-decay 0.001 \
  --rgcn-dropout 0.1 \
  --label-smoothing 0.05 \
  --patience 15 \
  --output checkpoints/baseline_synth.npz \
  --log-csv logs/baseline_synth.csv
```

**Métriques cibles (baseline synthétique) :**
- `val_node_macro_f1 > 0.80` (nœuds : 5 classes sur 7 représentées dans le synthétique)
- `val_edge_macro_f1 > 0.70` (11 classes parfaitement balancées)
- `gap train_node_macro_f1 − val_node_macro_f1 < 0.10`

**Note :** le corpus synthétique est balancé et issu de templates réguliers.
Un gap < 0.10 est attendu — s'il est plus grand, la régularisation est insuffisante.

---

### Étape 6.2 — Entraînement sur dataset réel (1356 phrases avec tokens UD)

```bash
gcn-train \
  --data-dir  gcn-datasets/real/train.json \
  --val-dir   gcn-datasets/real/val.json \
  --epochs 100 \
  --lr 0.001 \
  --use-attention \
  --bidirectional \
  --embedding-dim 50 \
  --edge-loss-weight 2.0 \
  --weighted-loss \
  --weight-decay 0.001 \
  --rgcn-dropout 0.1 \
  --label-smoothing 0.05 \
  --patience 15 \
  --output checkpoints/real.npz \
  --log-csv logs/real.csv
```

**Métriques cibles (données réelles, 5 relations sur 11) :**
- `val_node_macro_f1 > 0.50` (7 classes, données déséquilibrées)
- `val_edge_macro_f1 > 0.40` (5 classes représentées, cause domine à 59 %)
- **Note :** `val_edge_accuracy > 0.59` n'est PAS un objectif utile car la baseline
  triviale (toujours prédire "cause") atteint déjà 59 % sur ce dataset.
  `val_edge_macro_f1` mesure la performance sur toutes les classes, pas seulement
  la classe dominante.

---

### Étape 6.3 — Mesurer l'impact du data leak de `causal_pattern`

Entraîner deux modèles identiques avec et sans la feature `causal_pattern` :
```bash
# Sans causal_pattern
gcn-train --data-dir gcn-datasets/real/train.json --val-dir gcn-datasets/real/val.json \
  ... --output checkpoints/real_no_cp.npz

# Avec causal_pattern (Phase 5)
gcn-train --data-dir gcn-datasets/real/train.json --val-dir gcn-datasets/real/val.json \
  ... --output checkpoints/real_with_cp.npz
```

Comparer `val_edge_macro_f1` entre les deux. Si l'écart est > 0.15, le data leak est
significatif et la feature doit être retirée ou le `causal_pattern` doit être annoté
indépendamment des edges.

---

### Étape 6.4 — Diagnostics dans `evaluation/metrics.py`

Ajouter `confusion_matrix` :
```python
def confusion_matrix(pred: list[str], gold: list[str], classes: list[str]) -> np.ndarray:
    """
    Retourne une matrice (N_classes × N_classes) de type int.
    cm[i, j] = nombre de fois où gold=classes[i] et pred=classes[j].
    """
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=np.int64)
    for p, g in zip(pred, gold):
        if g in idx and p in idx:
            cm[idx[g], idx[p]] += 1
    return cm
```

Ajouter `per_class_report` (réutilise `_f1_per_class` existant) :
```python
def per_class_report(pred: list[str], gold: list[str], classes: list[str]) -> str:
    """Tableau texte : classe | precision | recall | f1 | support."""
```

---

### Étape 6.5 — Tests de robustesse (`tests/test_robustness.py`)

Ces tests valident que les features encodent correctement la variance des inputs.
Ils sont **indépendants d'un checkpoint** — ils testent uniquement la vectorisation.

```python
def test_connecteur_change_vecteur_edge():
    """
    Construire deux UDRepresentation identiques sauf que le connecteur
    de l'une est SCONJ "si" et l'autre est SCONJ "bien_que".
    Le vecteur edge produit par vectorize_edge doit être différent.
    """

def test_inversion_clauses_change_vecteur():
    """
    vectorize_edge(src, dst, ...) != vectorize_edge(dst, src, ...)
    car la direction (src_idx < dst_idx) change la feature pos_vec[0].
    """

def test_causal_pattern_change_vecteur_clause():
    """
    Deux UDRepresentation identiques sauf causal_pattern ("cause" vs "condition").
    vectorize_clause doit retourner des vecteurs différents (après Phase 5).
    """

def test_morph_tense_change_vecteur():
    """
    UDRepresentation avec root_morph["Tense"]="Pres" vs "Past".
    vectorize_clause doit retourner des vecteurs différents.
    """
```

---

## Ordre d'exécution avec dépendances exactes

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Phase 1             │  │  Phase 2             │  │  Phase 3             │
│  (UD tokens)         │  │  (Val set)           │  │  (Régularisation)    │
│                      │  │                      │  │                      │
│  1.1 diagnose_spans  │  │  2.0 split synthéti- │  │  3.1 weight decay    │
│  1.2 ud_annotator    │  │      que train/val   │  │  3.2 dropout R-GCN   │
│  1.3 validate_cir    │  │  2.1 --val-dir       │  │  3.3 label smoothing │
│  1.4 annotate script │  │  2.2 early stopping  │  │  3.4 tests           │
│  1.5 validation reps │  │  2.3 shuffle         │  │                      │
│  1.6 re-split        │  │  2.4 recorder        │  │                      │
│                      │  │  2.5 tests           │  │                      │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                          │
           └─────────────────────────┴──────────────────────────┘
                                     │
                            Phases 1+2+3 complètes
                                     │
                         ┌───────────┴───────────┐
                         │  Phase 4               │
                         │  (Dataset 5000+)       │
                         │  4.1 annotation cp     │
                         │  4.2 scraping          │
                         │  4.3 annotation UD+CIR │
                         │  4.4 split stratifié   │
                         └───────────┬────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │  Phase 5               │
                         │  (causal_pattern feat) │
                         │  d_clause : 79 → 90    │
                         │  Checkpoints invalidés │
                         └───────────┬────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │  Phase 6               │
                         │  6.0 split synthétique │
                         │  6.1 baseline synth    │
                         │  6.2 entr. réel        │
                         │  6.3 mesure data leak  │
                         │  6.4 diagnostics       │
                         │  6.5 tests robustesse  │
                         └────────────────────────┘
```

---

## Livrables

| # | Livrable | Phase | Fichier(s) | Lignes est. |
|---|---------|-------|-----------|-------------|
| L1 | Script diagnostic spans | 1.1 | `gcn-scraper/.../diagnose_spans.py` (nouveau) | ~40 |
| L2 | `ud_annotator.py` | 1.2 | `gcn-scraper/.../ud_annotator.py` (nouveau) | ~80 |
| L3 | `validate_cir.py` | 1.3 | `gcn-scraper/.../validate_cir.py` (nouveau) | ~60 |
| L4 | Script annotation UD dataset réel | 1.4 | `gcn-scraper/.../annotate_real_dataset.py` (nouveau) | ~60 |
| L5 | Dataset réel avec tokens UD | 1.5–1.6 | `gcn-datasets/real/annotated_ud.json` + splits | — |
| L6 | Split synthétique train/val | 2.0 | `gcn-datasets/corpus_train/`, `gcn-datasets/corpus_val/` | ~30 |
| L7 | `--val-dir` + `_set_training_mode` | 2.1 | `training/train.py` | ~80 |
| L8 | Early stopping + best model | 2.2 | `training/train.py` | ~60 |
| L9 | Shuffle + recorder étendu | 2.3–2.4 | `data/loader.py`, `evaluation/recorder.py` | ~70 |
| L10 | Tests Phase 2 | 2.5 | `tests/test_training.py` | ~80 |
| L11 | Weight decay | 3.1 | `layer2/reference.py`, `training/train.py` | ~25 |
| L12 | Dropout R-GCN + GAT | 3.2 | `layer3/reference.py`, `layer3/gat.py`, `training/train.py` | ~40 |
| L13 | Label smoothing | 3.3 | `pipeline/cgnp.py`, `training/train.py` | ~35 |
| L14 | Tests Phase 3 | 3.4 | `tests/test_pipeline.py` | ~60 |
| L15 | Dérivation `causal_pattern` réel | 4.1 | `gcn-scraper/.../derive_causal_pattern.py` (nouveau) | ~30 |
| L16 | Dataset 5 000+ | 4.2–4.4 | `gcn-datasets/real/` | — |
| L17 | `causal_pattern` dans schema + reader | 5.1–5.2 | `data/schema.py`, `data/json_reader.py` | ~10 |
| L18 | `causal_pattern` dans UDRepresentation | 5.3–5.4 | `layer1/representation.py`, `data/loader.py` | ~10 |
| L19 | Suppression `gcn_causal_type` heuristiques | 5.5 | `data/loader.py` | ~15 |
| L20 | `FeatureVocabulary` + `vectorize_clause` | 5.6–5.7 | `layer1/features.py` | ~20 |
| L21 | Tests mis à jour `d_clause == 90` | 5.9 | 6 fichiers tests | ~20 |
| L22 | Diagnostics + `per_class_report` | 6.4 | `evaluation/metrics.py` | ~50 |
| L23 | Tests robustesse | 6.5 | `tests/test_robustness.py` (nouveau) | ~100 |

---

## Métriques de succès

| Métrique | Actuel | Objectif | Note |
|----------|--------|----------|------|
| Dataset réel utilisable | 0/678 phrases | 678/678 | Phase 1 |
| `val_node_macro_f1` (synthétique) | N/A | > 0.80 | Phase 6.1 |
| `val_edge_macro_f1` (synthétique) | N/A | > 0.70 | Phase 6.1 — 11 classes balancées |
| Gap train–val node (synthétique) | Inconnu | < 0.10 | Phase 6.1 |
| `val_node_macro_f1` (réel) | N/A | > 0.50 | Phase 6.2 — données déséquilibrées |
| `val_edge_macro_f1` (réel) | N/A | > 0.40 | Phase 6.2 — 5 classes sur 11 |
| Baseline triviale edge (réel) | 59.3 % accuracy | — | Prédire toujours "cause" |
| Taille dataset réel utilisable | 0 | 5 000+ | Phase 4 |
| Tests Python | 195 | ≥ 195 | Chaque phase |
| Tests robustesse | 0 | 4 | Phase 6.5 |
