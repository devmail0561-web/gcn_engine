# Guide d'annotation v0 — phrases → CIR gold (phase 0, P0-5)

> Statut : v0 squelette — règles minimales pour démarrer gold_mini 60-100. Références schéma :
> `gcn_annotate/normalize.py` (8 types, 19 relations, aliases v2→v4), `gcn-datasets/schemas/*.yaml`.
> Doctrine existante à consolider : lots arbitrés 08-10, 20-24 (ex. `ARBITRAGE_lot24.md`).

## 1. Périmètre

Annoter en CIR 8 types / 19 relations. Tout le reste (silver auto, contrôle LLM) est hors guide.

## 2. Règles de spans (doctrine lots 08-10, à figer)

- Span nœud : ±7 tokens autour de la tête, **marqueur causal exclu** (vérifié : verdicts
  « spans ±7 (marqueur exclu) », `ARBITRAGE_lot24.md:11-12`).
- Span >15 tokens → LITIGE obligatoire (raccourcir ou quarantaine motivée).
- `token_span` + `marker_token` obligatoires sur chaque arête ; `confidence` ∈ {0.5…1.0} revalidée
  (pas de 1.0 automatique — cf. silver `confidence_calibree: 1.0`, à bannir du gold).

## 3. Champs obligatoires par CIR

`nodes[].{id, type ∈ 8, label, token_span, origin, scope}` · `edges[].{source, target,
relation ∈ 19, attributes.{confidence, explicit, negated, marker_token}}` ·
`{text, lang, source, url, hash, lot}` (traçabilité, cf. annotation §6).

## 4. Cas ambigus

- Indice causal → relation : appliquer `docs/refactory/table-11-19.md` ; cas marqué `ambigu`
  → quarantaine motivée, jamais de choix silencieux.
- Types limites (`etat_local` vs `etat_global`, `entite` vs `concept`) : exemples à constituer
  (source : lots arbitrés + tables `balanced_auto.py`).

## 5. Protocole qualité (kappa)

- Double annotation indépendante sur N=100 stratifié (langue×relation×domaine) par campagne ;
  seuil accord FIGÉ (2026-10-08) : κ ≥ 0.70, sinon révision du guide avant de continuer.
- Arbitrage tiers sur désaccords, verdicts `GARDER | QUARANTAINE + motif` ; **tombstones conservés,
  jamais de suppression** (pratique lots 20-24).
- Outillage : `gcn-annotate normalize` (migration v2→v4), `gcn-annotate eval --gold --pred`
  (`cli.py:92`), `split`/`balance`, `ud-annotate` (zéro coût).

## 6. Gabarit de lot

`lot_<lang>_<nnn>.jsonl` + `ARBITRAGE_lot<nn>.md` (proposées → LITIGE → après → verdict) +
entrée `manifest` (seed, caps rares-first, compteurs GARDER/QUARANTAINE).

## TBD (v1)

Exemples or par relation (19×3), guide scopes/temporal/modifiers, grille `!`/impératif (actes),
ponctuation non-latine.
