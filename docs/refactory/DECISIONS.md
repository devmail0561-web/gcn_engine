# Registre des décisions bloquantes — phase 0

> Doc locale (docs/ non versionné). Statut : D-ORIG, D-SPACY, D-S03, D-PH8 FIGÉES le 2026-10-08 ;
> D-TABLE : format figé, contenu (8 déclencheurs + cas ambigus) en cours via procédure lots ;
> D-AGNOSTIC FIGÉE le 2026-10-09 (moteur ni FR ni EN).

## D-ORIG — Composition des deux poids d'origine backend (7b × 9a)

- Question : `plan-backend-durcissement.md:86` (Explicit/Inferred/Hypothetical 1.0/0.7/0.3) et `:145`
  (document/affirmation/question 1.0/0.8/0.5) multiplient tous deux le tri WHY/WHAT/CHAIN. Comment composer ?
- Options : (a) multiplier les deux ; (b) 9a écrase 7b en session ; (c) champs séparés, un seul point
  de multiplication (poids 9a appliqué à l'écriture session, 7b au ranking backend).
- Retenu : (c). Motif : deux axes sémantiques distincts (provenance d'extraction vs provenance
  de dialogue, vérifié : `NODE_ORIGIN_VALUES` en `constants.py:62` vs `add_cir` sans origine
  `instructions.py:96-124`).
- Statut : FIGÉE 2026-10-08. Appliquer dans `spec-origine-addcir.md` (déjà conforme) + backend 7b/9a.

## D-SPACY — T1 vs interdiction spaCy dans `gcn_python`

- Question : T1 exige `spacy==3.8` lazy pour le moteur B ; `layer0/interface.py:20-22` interdit
  tout import spaCy dans `gcn_python.*` (vérifié).
- Options : (a) amender la contrainte (import lazy optionnel, test sans-spacy vert) ; (b) B hors
  package (implémentation Protocol en code utilisateur) ; (c) B = règles/regex uniquement, sans spaCy.
- Retenu : (a). Motif : T1 prévoit déjà « test sans-spacy passe » ; (b) complique silver-B et NLU ;
  (c) dégrade le silver sous le niveau auto-v2 existant.
- Statut : FIGÉE 2026-10-08 (contrainte amendée dans `interface.py`). Débloque socle 0a + spec-moteur-B.

## D-S03 — Suivi des amendements s0→s3 côté Rust

- Question : le writer Python évolue (s0→s3, `format_refacto=20261007`), le lecteur Rust (`parse`,
  `query.rs`) ne connaît que le CIR actuel (vérifié : `format_refacto` 0 hit).
- Options : (a) lecteur tolérant (`serde(default)` + gate de version en CI) ; (b) Rust épinglé s0,
  amendements s1-s3 Python-only.
- Retenu : (a). Motif : M1 verrouille déjà contrat versionné + test CI ; (b) casse PROVENANCE/7f
  qui lit `model_version`/`span`.
- Statut : FIGÉE 2026-10-08. Bloque socle 0b (roundtrip) — à câbler dans le pont M1.

## D-PH8 — Séquencement contrat UserResponse vs socle 0b

- Question : R9 exige PR Rust + roundtrip (dont UserResponse Ph8) AVANT la tête ; Ph8 est
  « parallélisable avec P0 » sans ordre explicite (vérifié : `UserResponse`/`respond(` absents).
- Options : (a) figer le contrat UserResponse (+ test roundtrip) avant socle 0b ; (b) 0b avec CIR seul,
  UserResponse après.
- Retenu : (a). Motif : le pont M1 inclut UserResponse dès le temps 1 ; 0b sans lui = roundtrip partiel.
- Statut : FIGÉE 2026-10-08. Reporter dans ROADMAP (dépendance Ph8→tête act).

## D-TABLE — Format et emplacement de la table 11→19

- Question : qui consomme (fusion C8/J7, silver, gold humain) et où vit la source de vérité ?
- Options : (a) YAML versionné + section doc ; (b) code seul (`normalize.py`) ; (c) doc seul.
- Retenu : (a-bis) `docs/refactory/table-11-19.md` (doc locale : humains + scripts data)
  + aliases en constantes code (`normalize.py`, versionné). Principe acté 2026-10-08 : le moteur
  ne charge jamais de données configurées (pas de YAML) — le runtime n'utilise que le code.
- Statut : format FIGÉ 2026-10-08 ; contenu (8 déclencheurs + cas ambigus) : procédure lots, EN COURS.

## D-AGNOSTIC — Moteur language-agnostic (ni FR ni EN)

- Question : Phase 9 (sujets, clarification, langue, correction) avec pronoms français/anglais
  en dur, catalogue de messages FR/EN, déclencheurs en français (« réponds en wolof ») ?
- Options : (a) lexiques FR/EN dans le moteur ; (b) mécanismes structurels + codes.
- Retenu : (b). Motif : note M. Tendeng 2026-10-09 — le moteur n'est pas pour le FR ou l'AN.
  Sujets par continuité + chiffres Unicode + répétition de nom ; clarification par chiffre/nom ;
  langue = tag opaque (`lang: <code>`) + script Unicode (jamais de nom de langue détecté) ;
  correction par `revoke:` ; réponses en codes (`topic:/clarify:/abandoned:/revoked:/lang:/origin:`).
  Identifiants figés antérieurs (`session_origin`, `AgentType`) conservés (specs gelées).
- Portée 2026-10-09 : chemins refacto (moteur + dialogue) ; l'existant hérité
  (`RELATION_KEYWORDS`, taxonomies fr/en, noms de modèles spaCy, données/configs)
  est grand-paternité, pas de la logique moteur — pas de retrofit.
- Statut : FIGÉE 2026-10-09. Appliquée dans `gcn_python/dialogue.py` + README gcn-python.
