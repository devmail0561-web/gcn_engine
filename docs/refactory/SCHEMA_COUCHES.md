# Schéma complet des couches — formats d'entrée / sortie

> Généré le 2026-10-07. Source vérifiée dans `gcn-python/src/gcn_python/` et `gcn-core/crates/`.
> Statut (2026-10-08) : décrit l'état LEGACY pré-refacto (UD, vocabs, lemmes, lattice) — pas la cible refacto-2026-10 (split fixe, `d_model`, 3 actes) : voir `plan-remodelage-*.md`. Les extensions DTO des phases backend 1-7 (`blocked_by`, `caveats`…) n'y figurent pas : voir `plan-backend-durcissement.md`.

## 1. Vue globale : les 2 chemins

```text
═══════════════════════════════════════════════════════════
  TEXTE BRUT : "Si les ventes baissent, on réduit les coûts."
═══════════════════════════════════════════════════════════
                          │
              ┌───────────┴───────────┐
              │                       │
     CHEMIN RUST                CHEMIN PYTHON ML
     (symbolique)               (vecteurs)
```

## 2. Chemin Python ML

```text
┌─────────────────────────────────────────────────┐
│ LAYER 0 — Portier                               │
│                                                 │
│  ENTRÉE : text: str                             │
│    ex: "Si les ventes..."                       │
│                                                 │
│  SORTIE : (clauses, connecteurs)                │
│    clauses = [fiche1, fiche2]                   │
│    connecteurs = ["Si"] ou [None]               │
│                                                 │
│  FICHE = { root_lemma, root_pos,                │
│             root_morph?, span? }                │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ LAYER 1 — Vectoriseur                           │
│                                                 │
│  ENTRÉE :                                       │
│   + fiche (Layer0)        ← OBLIGATOIRE         │
│   + vocab (18 POS, 38 deps) ← OBLIGATOIRE       │
│   + embedding (V,50)      ← OBLIGATOIRE         │
│     sans embedding = CRASH                      │
│                                                 │
│  SORTIE : VECTEUR                               │
│   1 fiche ──► [0.2, 0.0, 1.0 ...]               │
│   taille = 156 nombres                          │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ LAYER 2 — Classeur                              │
│                                                 │
│  ENTRÉE :                                       │
│   x_noeud : (156,)  ← 1 vecteur                 │
│   x_lien  : (297,)  ← [2*106 + 81 + 4 = 297]    │
│                                                 │
│  SORTIE : SCORES                                │
│   noeud : (8,)  → ex [0.1, 0.8..] = "processus" │
│   lien  : (19,) → ex [0.7, 0.05] = "cause"      │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ LAYER 3 — Mélangeur (voisins)                   │
│                                                 │
│  ENTRÉE : les 3 ensemble, OBLIGATOIRES          │
│   H          : (N, 156)  ← N vecteurs            │
│   edge_index : (2, E)    ← qui → qui            │
│   edge_types : (E,)      ← type 0..19           │
│                                                 │
│  SORTIE : (N, 156)                              │
│   mêmes vecteurs, mélangés avec voisins         │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ PIPELINE — Assembleur                           │
│                                                 │
│  ENTRÉE : reps + text + connecteurs             │
│                                                 │
│  SORTIE : CausalIR                              │
│   { nodes: [id, type, label],                   │
│     edges: [[src, dst, {relation, conf}]] }     │
└─────────────────────────────────────────────────┘
```

> Note dims (vérifié 2026-10-08) : 156 = `d_clause_effective(50)` (config du schéma : embedding 50) — défaut train `--embedding-dim 128` → 234 ; `x_lien` = `d_edge_effective(50)` = 2·106+81+4 = 297 (`features.py:122-128`) ; 247 = `d_edge_effective(0)` ; 1115 = `d_edge_closed_loop(234,8,128)` ; FICHE §2 simplifiée (champs complets : §5.1 + `representation.py`).

## 3. Chemin Rust symbolique

```text
┌─────────────────────────────────────────────────┐
│ RUST — Chemin direct (sans ML)                  │
│                                                 │
│  FRONTEND                                       │
│   ENTRÉE : source: str (+ schema si CSV)        │
│   SORTIE : CausalIR (confiance 1.0)             │
│   FR/EN seuls : SORTIE = Lattice (pas CausalIR) │
│    Lattice = {tokens, clauses}                  │
│                                                 │
│  MIDDLEEND                                      │
│   ENTRÉE : CausalIR (ids uniques)               │
│   SORTIE : CausalIR + cycles[] + diagnostics[]  │
│                                                 │
│  BACKEND                                        │
│   ENTRÉE : CausalIR + "WHY x?" (str)            │
│   SORTIE : réponse selon question               │
│    WHY → {causes} / DO → {severed, effects} /   │
│    SPOF → {paths_cut} ... (21 questions)        │
│                                                 │
│  VERBALIZER                                     │
│   ENTRÉE : CausalIR                             │
│   SORTIE : phrase: str                          │
└─────────────────────────────────────────────────┘
```

## 4. Résumé une ligne par couche

```text
L0 : str ────────────────► [fiches]
L1 : fiche + vocab + emb ──► (156,)
L2 : (156,) ─────────────► (8,) + (19,)
L3 : (N,156)+(2,E)+(E,) ──► (N,156)
PIPE : fiches ───────────► {nodes, edges}
RUST : source ───────────► CausalIR ──► +cycles ──► réponse
```

## 5. Formats de référence

### 5.1 Fiche UD (Layer 0 → Layer 1)

```text
FICHE = {
  root_lemma: str,      ex "baisser"   ← OBLIGATOIRE
  root_pos: str,        ex "VERB"      ← OBLIGATOIRE
  root_dep_rel: str,    ex "root"      ← OBLIGATOIRE
  root_morph: dict,     ex {Tense: Past} ← optionnel
  subject_pos: str|None
  token_span: (int, int)
}
```

### 5.2 CausalIR (sortie finale commune)

```json
{
  "nodes": [{ "id": 0, "node_type": "processus", "label": "ventes" }],
  "edges": [[0, 1, { "relation": "condition", "confidence": 0.82 }]],
  "cycles": [],
  "unresolved": []
}
```

### 5.3 Dictionnaire (constants.py, pas un layer qui tourne)

```text
8 types noeuds  : processus, etat_local, etat_global, entite,
                  condition, concept, evenement, contrainte
19 relations    : cause, enable, prevent, condition, concession,
                  sequence, motivation, filter, opposition,
                  data_dependency, control_dependency + 8 v3
Vocab UD        : 18 POS + 38 deps + 5 temps + ...
```
