# Benchmark Comparatif — Mode d'Emploi

**Script** : `benchmark_comparison.py`  
**Objectif** : Comparer MLPEncoder vs gcn-transformers (performance, vitesse, mémoire)

---

## 🚀 Utilisation Rapide

### Test Rapide (~5 min, ordre de grandeur non mesuré)

```bash
python benchmark_comparison.py \
    --train-dir ../gcn-datasets/real/train \
    --val-dir ../gcn-datasets/real/val \
    --epochs 5 \
    --max-samples 50 \
    --encoders mlp,xlmroberta
```

### Ce que le script produit (structure de sortie)

> **Aucun run n'a été exécuté à ce jour** ; aucun artefact de résultats
> (`.json` / `.csv` / `.npz`) n'existe dans ce dépôt. Les valeurs chiffrées qui
> circulent dans les autres documents sont des **hypothèses**, pas des mesures.

Le script n'a pas encore tourné : ce qui suit décrit la **structure** de sa
sortie console (et non une exécution réelle). À chaque lancement, il écrit :

1. **En-tête de configuration** — répertoires train/val, epochs, learning rate,
   liste des encodeurs demandés, dimensions détectées (`d_clause`, `d_edge`,
   `d_emb`) et embeddings lexicaux activés ou non.
2. **Bloc par encodeur** :
   - temps d'initialisation de l'encodeur ;
   - mémoire RSS du processus (via `psutil`, dépendance optionnelle : si elle
     est absente, le script imprime `Mémoire : 0.0 MB`) ;
   - courbe de loss par epoch (temps/epoch + loss) ;
   - évaluation train puis val : `node_acc`, `edge_acc`, `edge_f1` ;
   - temps moyen par epoch et par échantillon.
3. **Tableaux de comparaison** — une ligne par encodeur pour les métriques de
   performance, de vitesse et de mémoire.
4. **Recommandation** — calculée par le script à partir des mesures obtenues
   (meilleur `edge_f1`, plus rapide, plus léger).

Le script n'écrit **aucun fichier** de résultats : tout part en stdout. Pour
conserver les chiffres, il faut rediriger la sortie (`python benchmark_comparison.py
... > results.txt`).

---

## 📋 Options

### Options Principales

```bash
--train-dir PATH          # Répertoire données train (requis)
--val-dir PATH            # Répertoire données val (requis)
--epochs N                # Nombre d'epochs (défaut: 5)
--lr FLOAT                # Learning rate (défaut: 0.001)
--max-samples N           # Limite samples (défaut: tous)
--encoders LIST           # Encodeurs à tester (défaut: mlp,xlmroberta)
--embedding-dim N         # Dimension embeddings lexicaux (défaut: 128)
```

> Les durées indiquées plus bas (`~2 min`, `~1 h`, …) sont des **ordres de
> grandeur estimés**, pas des mesures : aucun de ces runs n'a été exécuté.

### Encodeurs Disponibles

- `mlp` — MLPEncoder (gcn-python)
- `xlmroberta` — XLMRobertaEncoder (gcn-transformers)
- `camembert` — CamembertEncoder (gcn-transformers)
- `codebert` — CodeBERTEncoder (gcn-transformers)

---

## 📊 Exemples

### 1. MLPEncoder Uniquement

```bash
python benchmark_comparison.py \
    --train-dir ../gcn-datasets/real/train \
    --val-dir ../gcn-datasets/real/val \
    --epochs 10 \
    --encoders mlp
```

**Temps estimé** : ~2 min (non mesuré)  
**Usage** : Baseline rapide

---

### 2. Tous les Encodeurs

```bash
python benchmark_comparison.py \
    --train-dir ../gcn-datasets/real/train \
    --val-dir ../gcn-datasets/real/val \
    --epochs 5 \
    --max-samples 100 \
    --encoders mlp,xlmroberta,camembert,codebert
```

**Temps estimé** : ~1 h (GPU) / ~3 h (CPU) — non mesuré  
**Usage** : Comparaison exhaustive

---

### 3. Benchmark Complet (Dataset Entier)

```bash
python benchmark_comparison.py \
    --train-dir ../gcn-datasets/real/train \
    --val-dir ../gcn-datasets/real/val \
    --epochs 20 \
    --encoders mlp,xlmroberta
```

**Temps estimé** : ~2 h (GPU) — non mesuré  
**Usage** : Benchmark publication

---

## 🔍 Métriques (définitions — aucune n'a été mesurée)

### Performance
- **Val Edge F1** : F1-score macro arêtes (métrique principale)
- **Val Node Acc** : Accuracy nœuds
- **Val Edge Acc** : Accuracy arêtes

### Vitesse
- **Init Time** : Temps initialisation encodeur
- **Train Time/Epoch** : Temps entraînement par epoch
- **Inference Time** : Temps inférence par sample (ms)

### Mémoire
- **Memory (MB)** : Mémoire RSS additionnelle (encodeur + overhead)

---

## 💡 Interprétation

### Différence < 5% Edge F1
→ **Équivalent en performance**
- Choisir le plus rapide/léger (MLPEncoder)

### Différence > 10% Edge F1
→ **Gain significatif**
- Évaluer trade-off performance vs vitesse

### Exemple de lecture (valeurs **inventées**, aucune n'est mesurée)

Ces chiffres illustrent le **format** de lecture d'un résultat. Ils ne
proviennent d'aucune exécution et ne doivent pas être cités comme des mesures.

```
MLPEncoder        : 0.468 edge_f1   ← valeur d'exemple, non mesurée
XLMRoberta (v1.0) : 0.475 edge_f1   ← valeur d'exemple, non mesurée
```
→ Gain **< 5%** : rester MLPEncoder (plus rapide/léger)

```
XLMRoberta (v1.1) : >0.70 edge_f1   ← objectif de roadmap, sans base empirique
```
→ Gain **> 10%** : adopter gcn-transformers (si GPU disponible)

---

## ⚙️ Configuration Hardware

### CPU Uniquement
```bash
# Ajouter --max-samples pour limiter temps
python benchmark_comparison.py \
    --train-dir ... \
    --val-dir ... \
    --epochs 5 \
    --max-samples 50 \
    --encoders mlp,xlmroberta
```

**Temps estimés** (non mesurés) :
- MLPEncoder : ~2 min
- XLMRoberta : ~30 min

---

### GPU Disponible
```bash
# Peut tester plus d'epochs et samples
python benchmark_comparison.py \
    --train-dir ... \
    --val-dir ... \
    --epochs 10 \
    --max-samples 200 \
    --encoders mlp,xlmroberta,camembert
```

**Temps estimés** (non mesurés) :
- MLPEncoder : ~5 min
- XLMRoberta : ~20 min
- CamemBERT : ~15 min

---

## 🐛 Dépannage

### Erreur : "transformers not found"
```bash
# Le paquet n'est PAS publié sur PyPI : installation depuis la source
pip install -e "/chemin/vers/gcn-transformers[benchmark]"   # + psutil (mémoire)
# ou sans l'extra
pip install -e /chemin/vers/gcn-transformers
# ou directement
pip install "transformers>=4.30" "torch>=2.0"
```

### Erreur : "Out of memory"
```bash
# Réduire max-samples
python benchmark_comparison.py \
    --max-samples 20 \
    --encoders mlp,xlmroberta
```

### XLMRoberta très lent
```bash
# Normal sur CPU (facteur 50-100× : hypothèse, non mesuré ici)
# Solution : utiliser GPU ou tester MLPEncoder seul
python benchmark_comparison.py \
    --encoders mlp  # MLPEncoder uniquement
```

---

## 📈 Résultats Attendus

> `BENCHMARK-RESULTS.md` (chiffres non mesurés) a été **supprimé** le 2026-09-25.
> Aucun benchmark n'a encore été exécuté : les valeurs ci-dessous sont des **attentes
> non mesurées**, pas des résultats. Lancer le script pour obtenir de vrais chiffres.

**Attentes v1.0 (non mesurées)** :
- **Performance** : MLPEncoder ≈ gcn-transformers (~0.47 supposé)
- **Vitesse** : MLPEncoder supposé plus rapide (facteur non mesuré)
- **Mémoire** : MLPEncoder supposé plus léger (facteur non mesuré)

**Objectifs v1.1 (futur, hypothèses non mesurées)** :
- **Performance** : viser `>0.70` — objectif de roadmap **sans base empirique**
- **Vitesse** : MLPEncoder supposé plus rapide (facteur non mesuré)
- **Mémoire** : MLPEncoder supposé plus léger (facteur non mesuré)

---

## 📞 Support

**Bugs** : https://github.com/devmail0561-web/gcn_engine/issues  
**Docs** : README.md, USAGE-COMPARISON.md

---

## ✅ Checklist Benchmark

- [ ] Installer dépendances : `pip install -e ".[benchmark]"` (paquet non publié sur PyPI)
- [ ] Préparer données : train + val dirs
- [ ] Lancer benchmark : `python benchmark_comparison.py --train-dir ... --val-dir ...`
- [ ] Analyser résultats : voir output console (aucun fichier de résultats pré-rempli)
- [ ] Décider encodeur : à faire **après** un run réel (aucune mesure à ce jour)

---

**TL;DR** : Script automatique qui compare MLPEncoder vs gcn-transformers sur vos données et recommande le meilleur encodeur selon vos contraintes (performance, vitesse, mémoire).
