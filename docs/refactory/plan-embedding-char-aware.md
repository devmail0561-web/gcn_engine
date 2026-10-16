# Plan embedding char-aware (morphologie OOV) — 2026-10-15

## V0. Vérification préalable (faite AVANT ce plan — chiffres, pas opinions)
- Vocab train P9 : 5076 formes. OOV val 21.6 %, test 20.7 % (tokens).
- OOV partageant un radical (stemming naïf ≥6 lettres) avec une forme connue :
  **83/658 = 12.6 %** des distincts test. Top OOV : `cm`, noms propres
  (`Detroit`, `Colbert`), nombres (`1943`), termes de domaine — NON morphologiques.
- Borne : la morphologie ne peut récupérer que ~13-20 % de l'OOV, soit
  **~3 points de couverture tokens**. Gain F1 attendu ≤ +0.02 < seuil 0.05 :
  le gate dira probablement NEUTRE — accepté d'avance (honnêteté).
- Composés : déjà découpés au runtime (`pare-feu` = 3 tokens, tiret = ponctuation).
- Docs : aucun principe `BYTES+T` écrit ; seule mention voisine = fastText pour la
  recherche de labels (`ETUDE_SA.md:597`, autre sujet). Ce plan ne s'appuie sur
  aucun slogan, seulement sur les mesures ci-dessus.

## V1. Diagnostic (où ça fuit, exactement)
- `layer1/tok_embedding.py:44-133` : lexique de formes entières (id sha256/forme).
  `protège`/`protéger`/`protégeaient` = 3 étrangers. Aucun partage sub-lexical.
- `layer0/tokenizer.py:119-183` : split char-aware correct (CJK, apostrophes,
  ponctuation, offsets exacts) — RIEN à changer ici.
- `l1_forward.py:75-87` : pooling moyenne brute — hors périmètre (autre chantier).
- Statut : NON prioritaire (E7/E7-bis neutres, saturation par N/val) — chantier
  post-gold-3000, timeboxé, réversible.

## V2. Design (3 options, zéro lexique externe — doctrine maintenue)
- **A (recommandée) : char-CNN additive.** `E = E_form + E_char`, où E_char =
  embedding chars (128 symboles, 16d) → conv1D k=3,5 → max-pool → projection d_model.
  NumPy pur, seedé, ~5k params. Ablation exigée : form-seul vs char-seul vs somme.
- **B (repli) : hash-subword.** N-grammes chars 3-5 hashés → table apprise
  (même pattern que `oov-hash`, `tok_embedding.py:72`). Moins expressif, 0 risque.
- **C (statu quo documenté).** Ne rien faire, réévaluer à N/val ≥ 300.
  Choisir C si A et B sont neutres — c'est un verdict valide, pas un échec.

## V3. Mesures strictes (garde-fous anti-fausse-victoire)
- M1 : suites vertes (`gcn-python/tests`, 981) + `make gates` vert avant/après.
- M2 : protocole E gelé — coarse 30ep `--context`, seeds 7/42/123, val P5 gelée,
  3 runs mini (`--epochs` identiques). Comparaison au run E5-P6E archivé, jamais entre vals.
- M3 : métriques DOUBLES — F1 edge (seuil 0.05) **ET** taux OOV-récupéré
  (formes OOV dont le nearest-neighbor char est la bonne famille ; cible +10 pts relatifs).
- M4 : rollback automatique — `E_form` intact par défaut, `--char-emb` opt-in ;
  régression > 0.02 sur F1 ⇒ revert, pas de tuning.
- M5 : interdiction du vocabulaire externe (pas de fastText pré-entraîné, pas de BPE
  sur corpus externe — le plan s'arrête là où commence la dépendance).

## V4. Jalons (timebox totale ~2 h machine + revue)
- J0 [FAIT] : mesure de cadrage (V0 ci-dessus).
- J1 : implémentation A (+ tests unitaires : déterminisme seed, roundtrip npz,
  shapes, OOV-char non nul) — derrière `--char-emb`, défaut off.
- J2 : mesure 3 seeds + ablation (form/char/somme) + M3.
- J3 : décision — adopter si (M3 F1 ≥ +0.02 ET OOV-récupéré +10 %) ou si neutre
  strict sans régression (défaut off conservé) ; sinon C + entrée baseline.
- J4 : si adopté, re-mesure B (silver gate) — la morphologie change les rendements.

## V5. Non-objectifs explicites
- Pas de BPE/SentencePiece externe, pas de lemmatiseur au runtime (zéro frontend),
  pas de norme L2 imposée aux embeddings, pas de changement tokenizer/pooling/loaders.
