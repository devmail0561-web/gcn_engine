# RÈGLE STRICTE — Équilibre du dataset GCN-NL (NON VIOLABLE)

Décidée le 2026-09-21. Mise à jour v3.0 le 2026-09-27 (8 nœuds, 19 relations).
S'applique à toute production de données d'entraînement.

## 1. Quotas uniformes obligatoires (v3.0)

### Relations (19 — v3.0)

- Chaque dataset livré (train/val/test, batch, merge) DOIT contenir le **même nombre
  de phrases par relation** pour les **11 relations directes** (cause, enable, prevent,
  condition, concession, sequence, motivation, filter, opposition, data_dependency,
  control_dependency).
- Écart maximal toléré : **±10 %** autour de la moyenne par relation.
- **8 nouvelles relations v3.0** (analogy, counterfactual, conditional_cause,
  mediated_cause, joint_cause, conditional_prevent, mediated_prevent, joint_prevent) :
  - Minimum **N_min = 20 exemples** par type avant activation du logit correspondant.
  - En dessous de N_min : relation ABSENTE du dataset livré (pas de 1-5 exemples
    qui biaiseraient l'entraînement sans être représentatifs).
  - Interdiction de livrer des exemples ternaires avec `joint_group_id` manquant.

### Types de nœuds (8 — v3.0)

- Types de nœuds : minimum **5 %** du total pour chacun des 8 types.
- Nouveau type `contrainte` : minimum N_min = 20 exemples (même règle que nouvelles relations).
- Renommage v3.0 : les datasets existants utilisant `etat`/`etat_systemique` restent
  valides (serde alias configuré côté Rust) — migration progressive.

- Interdiction de livrer un dataset où une **relation directe** (11 premières) est
  absente (0 exemple), sauf justification écrite + plan de comblement.

## 2. Méthode imposée

1. Filtrage large multi-langues (jamais de filtre étroit qui affame les classes rares).
2. Annotation (manuelle = gold, ou auto = silver) avec suivi live des quotas.
3. **Sous-échantillonnage des classes fréquentes** (plafond = quota) — jamais
   l'inverse (ni duplication massive d'une classe pour "compenser").
4. Si une classe est introuvable en données réelles : quotas uniformes BAISSÉS
   au niveau atteignable (ex. 90/relation), PAS de classe à 0 + autres à 150.
5. Tout merge avec l'existant (536 phrases) passe par un **rééquilibrage stratifié**
   (sous/sur-échantillonnage avec seed fixe) — jamais d'ajout brut qui casse l'équilibre.

## 3. Validation obligatoire avant livraison

- Tableau relations × dataset + tableau node types (comptes + %).
- Si annotateur auto : accuracy mesurée contre le gold (219 phrases FR/EN/ES),
  par relation. Seuil : relation < 0.50 = rejetée du silver, ré-annotée ou retirée
  (et quotas rebaissés en conséquence).
- `validate_cir.py` : 100 % valides (spans réels, refs arêtes, ≥2 nœuds ≥1 arête).

## 4. Traçabilité

- Chaque phrase : source (scrap URL ou `gold-manual`), langue, relation, confiance.
- Données provisoires (tokens regex, spans provisoires) MARQUÉES `_note` et
  interdites d'entraînement sans injection UD + réalignement des spans.

Historique : batch_fr_001 (118, full UD), batch_en_001_partial (77), batch_es_001_partial (24).
Gold manuel total : 219 phrases FR/EN/ES → jeu de validation officiel.

## 5. Journal d'application (à ne pas violer — leçons 2026-10-02)

- **Constat 2026-10-02** : 1345 phrases à ratio 19x (cause 260 / control 14).
  Violation §1 par accumulation de lots non plafonnés. Jamais présenter
  des ajouts de rares comme de l'équilibre.
- **Interdit** : sur-échantillonnage (duplication) des rares pour compenser
  (§2.3 : « jamais l'inverse »). Plan proposé puis abandonné le 2026-10-02.
- **Interdit** : seuils de sélection qui tuent les rares patiemment collectés
  (seuil conf<0.9 : 226 phrases valides jetées, réparé — seuil <0.8).
- **Méthode conforme** : quotas uniformes au niveau atteignable, appliqués
  AU SPLIT (`scripts/make_split.py --quota`, seed fixe, gold intact).
  Control à 20 → plafond ~20/classe jusqu'à campagne control 30-40.
- **Interdit** : `git add` de dossier (pollution `.pyc`/caches) — fichiers
  explicites + `git add -f` pour l'unique exception `docs/PROTOCOLE*`.
- **Interdit** : réécrire les labels arbitrés (01-04) avec la règle mécanique
  (régénère les bugs corrigés : parens, `(développeuse(aoun)`).
- Rappel : gold hors git (DATA ignoré), tombstones jamais suppression,
  défauts moteur inchangés sans mesure 3 seeds (Δ < 0.05 = bruit).
