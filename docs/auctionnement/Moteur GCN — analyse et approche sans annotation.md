# Moteur GCN — analyse exhaustive et approche sans annotation

Date : 2026-10-14 · Sources : `auctionnement.zip` (comparaison moteur vs Transformers, étude de l'approche actuelle, SAD, architecture refacto)

> Périmètre : lecture des quatre documents uniquement, pas du code. Les références `fichier:ligne` viennent des docs et n'ont pas été revérifiées.

## Verdict

Le problème n'est pas « moteur contre Transformers ». Tout l'apprentissage repose sur une seule source de signal : du gold structuré (UD, spans, arêtes). Pourtant les représentations sont apprises from scratch sur \~570 phrases. Ce qui manque, c'est le **pré-entraînement auto-supervisé**, pas l'attention. Il est indépendant de l'architecture, donc compatible avec les contraintes du moteur (CPU, moins de 10 Mo, offline, spans).

## 1. Lecture critique des documents

Le diagnostic chiffré est cohérent d'un document à l'autre : val 0,239 contre train 0,99, OOV 23,7 %, ECE 0,33-0,55, 8 relations à zéro. Quatre points sont en revanche faux ou surestimés, et ils orientent la décision.

1. **« Bytes bruts, prétraitement aucun » (comparaison §1) est inexact.** Le SAD (§7-8) décrit un split, une normalisation par script, des ids hashés et un vocabulaire E\_tok appris sur le train (d'où l'OOV). Surtout, le chemin tok ne reçoit que du texte **pré-segmenté en clauses par B** (spaCy 3.8 ou fallback regex, 6 langues, pt/it vides). B est un frontend par langue déguisé : P1 est déplacé, pas résolu.
2. **Asymétrie train/inférence probable.** À l'entraînement, les clauses viennent du gold UD (`span_clause`) ; à l'inférence, elles viennent de B. Si la val utilise les clauses gold, 0,239 est une borne haute. Aucun doc ne mesure texte brut → CIR de bout en bout. À vérifier dans le code.
3. **Le concurrent est mal choisi.** On compare à un Transformer génératif de 100M. Le vrai concurrent est un encodeur pré-entraîné avec une tête de relation, sur les mêmes 572 phrases. « Boucles impossibles » et « ECE non applicable » sont faux : une tête de classification se calibre, et un modèle qui produit un graphe peut produire des cycles.
4. **Incohérences mineures.** L'étude §2 cite 21 requêtes (README) alors que le code en a 25. Le doc annonce « \~80× » puis « \~100× » plus léger. Les tests passent de 260/887 à 932/952 selon la source. L'étude devrait appliquer sa propre règle P10.

## 2. Causes racines, par impact

**R1. La représentation est un sac de mots.** Le lookup suivi d'une moyenne par clause efface l'ordre, la négation et le connecteur. L'arête est classée sur `[src|dst|gap]`, soit deux sacs de mots, donc le modèle mémorise des thèmes. C'est ce qui explique 0,99 contre 0,24. Estimation de paramètres (calcul à partir des dimensions du SAD, à confirmer) :

- Le R-GCN (21 matrices 128×128) pèse environ 0,34M de paramètres, dont 8 relations qui n'ont jamais de gradient.
- E\_tok pèse environ 0,9M (vocabulaire de l'ordre de 7k mots).
- Ces deux blocs font \~95 % des 1,3M, soit \~2 300 paramètres par phrase d'entraînement.

**R2. L'espace d'étiquettes est plat alors que l'IR est déjà factorisé.** L'arête porte `third`, `joint_group`, `negated` et `derivation`. Or 8 des 19 relations (les v3) sont des composites : ConditionalCause = Cause + tiers Condition, JointCause est un motif de graphe, MediatedCause est un chemin déjà couvert par `Derivation`. Les classer à plat est ce qui produit des classes vides. C'est un choix de modélisation, pas une contrainte du schéma.

**R3. Nœud = clause, imposé par B.** Les relations intra-clause (« la sécheresse a provoqué la famine ») sont invisibles, donc le rappel est plafonné par construction.

**R4. Un seul signal d'apprentissage.** Il faut du gold UD et des spans, et `SpanError` ne devine jamais (bonne rigueur). Le coût, lui, est celui du ROADMAP : 3M de phrases à 5 % annotées, soit 50-95 jours (si la lecture est bonne). Les 285k phrases sans CIR ne servent à rien, et le silver est pondéré 0,0.

**R5. La mesure n'est pas fiable.**

- Le meilleur epoch est restauré sur `val_edge_macro_f1` avec N=114 : 0,239 est déjà optimiste.
- L'ECE sur 10 bins avec N=81 donne \~8 points par bin, donc la plage 0,33-0,55 est en partie du bruit d'estimateur.
- Les harnais passés sont incomparables (0,47 → 0,12).

On ne peut donc pas dire aujourd'hui si 0,239 est un plafond d'architecture ou de données.

**R6. Dette de conception** : double chemin, constantes D6/D7, normalisation qui collisionne. C'est réel, mais ça ne déplacera pas 0,239.

## 3. À conserver

- CIR, provenance et span obligatoire.
- Middleend (Tarjan, validation).
- `UserResponse` et les gates GEL/hash/format.
- Le CPU offline.
- Les ingestors code/table/STIX : ils parsent des formats structurés, ce ne sont pas des frontends de langue. Ils sortent du noyau mais restent légitimes.

## 4. Approche proposée

**Principe : apprendre sans étiquettes ce qui est la langue, étiqueter peu et seulement pour mesurer.** Un graphe typé à zéro étiquette n'existe pas. Ce qui est atteignable :

- zéro annotation pour l'encodeur ;
- zéro annotation humaine pour le signal de relation ;
- \~300 phrases gold pour évaluer et calibrer (déjà prévu dans l'étude §7) ;
- du silver LLM en complément, sous le gate IC95 ≤ 20 %.

```text
octets UTF-8
 → embeddings octets + n-grammes hashés        (pas de vocabulaire, OOV = 0)
 → encodeur contextuel ≤ 1M params             (CNN dilaté ou 2-4 couches d'attention, d=128)
 → états contextuels (connecteurs inclus)
 → têtes : ancres/bornes de nœuds + biaffine factorisé (existe / famille / flags)
 → calibration conforme + abstention
 → CIR → middleend → backend                   (inchangés)
```

### Objectifs d'apprentissage

- **O1. Pré-entraînement sur texte brut** (321k phrases plus le scraper) : span-MLM ou détection de tokens remplacés (ELECTRA), plus efficace en échantillons pour un petit modèle. À trancher par ablation.
- **O2. Prédiction de marqueur masqué**, dans l'esprit de DisSent et Discovery (Sileo et al., 2019). Les deux côtés d'un marqueur de discours (« parce que », « donc », « malgré »…) forment la paire, on masque le marqueur et on prédit sa famille. On le masque 50-70 % du temps pour forcer la lecture du contenu et pas seulement du marqueur. Aucun parseur n'est nécessaire.
  - Le lexique (\~100 entrées par langue) vit **uniquement dans `pretrain/`**, jamais à l'inférence. Pour une nouvelle langue, on le génère par LLM puis on le vérifie : la langue devient une donnée (cible T6).
  - Ça couvre environ 6 familles. **Enable, Prevent et Filter** (déclencheurs verbaux) ne sont pas couverts : il faut du silver LLM sous gate ou des motifs verbaux.
- **O3. Ordre des clauses** (détection de permutation) : donne un a priori de direction pour Sequence. Les cycles ne s'apprennent pas, Tarjan les calcule. Le gap est reporté à plus tard.
- **O4. Fine-tuning court sur gold** (219 + 85), sur les têtes.

### Étiquettes factorisées

Le modèle prédit existence × famille (≈9) × flags (négation, hypothétique, tiers). Une fonction déterministe `compose()` reconstruit les 19 relations pour l'IR :

- Joint est dérivé (≥2 arêtes vers la même cible avec le même marqueur).
- Mediated vient de `Derivation`.
- Counterfactual est le flag hypothétique.

Le biaffine doit être factorisé (rang faible), sinon on recrée les 0,34M du R-GCN. L'apport du R-GCN n'est d'ailleurs ablaté nulle part dans les docs : à tester.

### Calibration

Prédiction conforme sur gold-300 (couverture garantie, mais sous échangeabilité, donc fragile en dérive). On rapporte Brier/NLL et un ECE à bins équi-massiques avec bootstrap. C'est cohérent avec l'usage du SA : il a besoin de précision à faible couverture avec repli sur UNCONFIRMED, pas d'un F1 global.

### Entraînement

En torch offline, puis export `.npz` et inférence NumPy/Rust. Une backward NumPy écrite à la main ne tiendra pas pour du pré-entraînement. Torch existe déjà en optionnel (GAT). Il est interdit dans le chemin d'inférence, par un guard CI du même style que `subprocess==0`.

### Distillation optionnelle

Un teacher (encodeur pré-entraîné branché via le protocole `CausalEncoder`, fine-tuné sur le gold) étiquette les 285k phrases, et un student de moins de 10 Mo l'apprend. Ça ne supprime pas le gold : il en faut pour le teacher.

### Alternatives écartées

| Option | Raison |
| --- | --- |
| Transformer/LLM de bout en bout | Viole offline/CPU/spans/calibration. Reste utile comme teacher et baseline. |
| Découverte causale non supervisée | Pas de relations typées. |
| Annoter 3M phrases | 50-95 j, et ne corrige pas R1. |
| Mining par motifs seul (type CausalBank) | C'est P3, un moteur de règles déguisé. |

## 5. Contraintes à faire respecter

- Poids ≤ 10 Mo, p99 `validate` < 200 ms et `causes` < 50 ms, réseau coupé.
- Gold-300 gelé par hash, jamais utilisé pour la sélection d'epoch ni la calibration d'hyperparamètres.
- 3 seeds, IC bootstrap, et un seul harnais de référence versionné.
- Métrique décomposée : existence, famille, flags. Aucun chiffre comparé à l'ancien harnais.
- Aucun import spaCy ni fichier de règles de langue dans `infer/`.
- Toute étiquette d'entraînement porte sa source (marqueur, silver, gold, teacher).

## 6. Plan d'expériences avec critères d'arrêt

| # | Expérience | Critère de décision |
| --- | --- | --- |
| E0 (1-2 j) | Baseline tête-à-tête : encodeur pré-entraîné via `CausalEncoder` + biaffine sur les 572/132, 3 seeds. | Gain net : la représentation est le goulot, on continue. Reste vers 0,25 : le problème est le schéma ou les données, on traite R2/R5 d'abord. |
| E1 | Geler gold-300, appliquer la métrique décomposée, ré-évaluer le modèle actuel. | Devient la baseline de référence. |
| E2 | Encodeur octets, initialisation aléatoire contre pré-entraîné O1, sur le même train de 536. | Le pré-entraînement doit améliorer la val de façon significative (IC bootstrap). |
| E3 | Ajout de O2 (marqueur masqué). Mesurer d'abord la part de phrases avec marqueur dans le corpus. | Part inférieure à \~2 % : signal trop rare, on abandonne cette voie. |
| E4 | Distillation et courbe précision/couverture avec abstention. | Précision cible atteinte à couverture exploitable pour le SA. |
| E5 | Zéro-shot EN/pt/it sans code nouveau. | Prouve (ou non) l'universalité, aujourd'hui non mesurée. |

## 7. Correctifs indépendants, à faire sans attendre

- **Middleend** : diagnostiquer les arêtes pendantes avant de les sortir du graphe, rejeter conf ∉ \[0,1\], IR vide et version absente.
- **`normalize`** : ne plus s'en servir comme identité (offset + hash à la place). Normalisation réservée au matching de requêtes.
- **DO / COUNTERFACTUAL** : les renommer (CUT / WITHOUT), puisque ce n'est pas du do-calculus. Plus tard, d-séparation sur le graphe condensé par SCC.
- **`backend` → verbalizer** : passer par un trait injecté plutôt que par PATH.
- **Chemin legacy** (UD, 106d, D6/D7) : le supprimer après E0.
- **Fusion multi-phrases** (P7) : objectif contrastif sans label, plus tard.

## 8. Limites de l'analyse

- Les estimations de paramètres sont un calcul à partir des dimensions du SAD.
- « 536/6,9M » est ambigu : lu comme 536 phrases de train dans un pool de 6,9M.
- Les gains attendus sont des hypothèses que E0 et E2 tranchent.
- Risque principal : le signal des marqueurs est biaisé vers la causalité explicite, et l'implicite reste difficile. Une dérive de domaine peut aussi casser la calibration conforme.
