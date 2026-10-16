# Plan LOT-SOCLE-NI — combler les zéros nœuds + intent (v1.0)

> Statut : VALIDÉ par owner (2026-10-13, session chemin tok). Exécution en cours.
> Références : `plan-collecte-dataset-annotation.md` (pilotage unifié, T4-nouveau gelé
> pendant ce lot, §6 amendé), `plan-annotation-cir.md` (chaîne silver→gold),
> `guide-annotation-v0.md` (règles spans, κ ≥ 0,70), `audit-gold.md` (pools),
> `GATES.md` (R8/R10, silver 0→0,3), `table-11-19.md` (méthode minage→arbitrage).
> Hypothèses par défaut (owner silencieux) : 80 phrases FR seul, elicited autorisé
> taggé `elicited`. Réviser ici si amendé.

## 0. Objectif et règle d'arrêt

Servir les classes à zéro (3 types nœuds + intent) pour rendre le chemin tok
évaluable, puis trancher données-vs-modèle par un run go/no-go.
**Règle d'arrêt** : si le run go/no-go ne décolle pas malgré des classes servies,
on ne monte pas en charge — on rouvre le dossier modèle (sortie écrite, GATES.md).
Aucun T4 nouveau avant le verdict (runs fond engagés laissés en cours).

Contexte mesuré (2026-10-13, v4/final2, 704 phrases) : `concept`/`evenement`/
`contrainte` = 0 occ. ; `intent` = 1 occ. (`explain`) ; spans `auto` 99,6 % ;
train 504× declarative / 1 question / 1 exclamation. Run coarse `3ded799` :
train 2.52→1.62, F1 val gelée à l'init — mémorisation sans généralisation.

## 1. Composition du lot (80 phrases FR, 100 % manuelles)

| Sous-lot | N | Contenu |
|---|---|---|
| N-nœuds | 30 | 10× `concept`, 10× `evenement`, 10× `contrainte` (spans manuelles ±7, marqueur exclu) |
| N-renfort | 10 | 5× `etat_local`, 5× `condition` (actuellement 39/41 au train) |
| N-intent | 30 | 10× `assert`, 10× `question`, 10× `request` (R9 : `!` exclamatif≠request, `Si+Pres`→assert, question sans `?`→assert) |
| N-mixte | 10 | 2-clauses : nœud cible + arête 11-relations existantes |

Registres du train (encyclo/science/wikipedia), aucun domaine neuf.
`request` elicitées autorisées si introuvables (tag `registre: elicited`, jamais
mélangées sans tag).

## 2. Sourcing (`g0/fr_a_annoter.jsonl`, 28 273 phrases)

Script : `gcn-datasets/scripts/source_socle_ni.py` (déterministe, seed 42).
Filtre causal existant (≥1 connecteur, sinon quarantaine) → déclencheurs
lexicaux larges **frontières de mots obligatoires** (leçon `viaux/viande`) →
sur-échantillonnage `?`/`!`/impératifs pour N-intent → dédup MD5 contre
train/val/test final2 + `deja_annote` → **lot socle = val-only** (zéro fuite train).
Sortie : `gcn-datasets/DATA/supervision/socle_ni/candidats_*.jsonl` + manifest.

Déclencheurs initiaux (le minage propose, l'humain dispose) :
concept : `concept|notion|idée|théorie` · evenement : `événement|survenue|
accident|crise|incident` · contrainte : `contrainte|limite|interdiction|
obligation|restriction`.

## 3. Protocole d'annotation (v4 éprouvé, sans pré-remplissage)

Double annotation aveugle (owner + assistant, sans voir les verdicts adverses),
grille schéma v4 strict + `sentence.intent ∈ {assert, question, request}`.
κ ≥ 0,70 par strate (nœuds / relations / intent), première mesure réelle ;
κ < 0,70 → révision de la règle litigieuse avant de continuer.
Arbitrage tiers : `GARDER | QUARANTAINE + motif`, tombstones conservés.
Cas à trancher versés au guide v1 : `entite` vs `concept`, `etat_local` vs
`etat_global`, `condition`-nœud vs `condition`-relation.
Livrables : `lot_socle_ni.jsonl` + `ARBITRAGE_socle_ni.md` + manifest
(seed 42, GARDER/QUARANTAINE, κ par strate).

## 4. Sorties dérivées (sans surcoût)

Mini-table nœuds v0 (miroir table-11-19) · exemples or 3×type + 3×acte (guide v1) ·
témoin 30 spans `auto` ré-annotées en aveugle (1re mesure d'erreur des spans auto).

## 5. Run go/no-go (harnais unique, seeds 42/7/123, `PROTOCOLE_MESURE.md`)

Train = final2-train + socle (absorption) ; val = socle tenu-out + val final2.
Espace **fin d'abord** (R8 se lève-t-il ?), coarse en second (réf. F1 0,1067).
**GO** : 0 GelError ET val_edge_F1 fin > init + ≥0,05 avec loss train ↓ →
T4 relancé à quotas corrigés (ciblage nœuds), montée gold-3000.
**NO-GO** : GelError persistant (revoir sourcing) OU F1 gelée malgré classes
servies (STOP collecte, dossier modèle, sortie écrite). Effet < 0,05 = bruit.
Traçabilité CSV (data_hash, git_sha, seed).

## 6. Moyens et effort (~3 j)

Owner : Michel Tendeng (décideur solo GATES.md) ; 2nd annotateur aveugle :
assistant. Sourcing ~0,5 j · annotation ~1–1,5 j · arbitrage+κ ~0,5 j ·
run ×3 ~0,5 j. Outils existants : `gcn-annotate normalize/eval/split`,
`scripts/audit_spans_tok.py`, `train_tok --espace fin`.
