# DEMAIN — reprise programme GCN (état au 2026-10-16 soir)

## Aveugle humain (EN ATTENTE, prioritaire)
- **Ph5 (11 cas)** : `gcn-datasets/silver_ph5_AVEUGLE.md` — phrases COMPLÈTES exigées
  (retour utilisateur : pas de vote sans phrase entière ; UI questions écartée 2×,
  passer en **texte libre** : `1 oui, 2 non, 3 corriger→X…`). Gate 11/11.
- **Groupé (34 cas)** : `gcn-datasets/silver_AVEUGLE_GROUPE.md` (vagues J/S/T/bulk/cond).
  Gate ≥95%.
- À la réception des verdicts : comptabiliser, archiver en `validation_pv1/verdicts_aveugle*.json`,
  clôturer Ph5 (si 11/11) puis merger.

## CI / PR
- PR draft #1 (branche `refacto/arch-cible-refacto-2026-10`, 171+ commits) : CI self-hosted
  lente mais active (runner local OK). Quand verte → revue ciblée déjà faite (pas de
  secrets, pas de /tmp, pas de debug) → `gh pr merge`.
- Règle : NE PLUS PUSHER en rafale (chaque push relance un run complet).

## État programme (tout commité/pushé, arbre propre)
- Calibration GO (ECE 0.086, T*=2.50, N=107) — réf prod `e9ter_calibre.npz`.
- Précision production ~37% stable (4 vagues, 166+60 golds). Test best 0.261 (aug).
- Segmentation v2+B-c1 (986 tests). Voie A : détection NO-GO, infra gardée.
- Données : 3910 phrases, 19 relations ≥30. p12 (980/207/207, lot37 exclu, lot39 gardé).
- Protocole anti-fuite inter-splits (exclusion + comptage systématiques).

## Prochaines vagues (si besoin volume)
- Réservoir : ~2400 phrases OK (≈1900 science/educ + presse). Rendement ~37%.
- Bulk : re-spanner AVANT d'inclure (style whitespace obligatoire).
- Ne jamais remettre de mot-matching dans le moteur (règle projet absolue).
