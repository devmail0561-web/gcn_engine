# gcn-frontend-table

Frontend **tabulaire → CausalIR** (P2.3 ETUDE). Chaque ligne CSV avec colonnes
cause/effet produit un couple (source, target, edge). Symbolique pur, zéro ML.

## Schéma (paramètres, pas config)

```rust
use gcn_frontend_table::{TableSchema, parse_table};

let schema = TableSchema {
    cause_col: "facteur".to_string(),
    effect_col: "resultat".to_string(),
    relation_col: Some("relation".to_string()),     // snake_case, sinon `Cause`
    confidence_col: Some("confiance".to_string()),  // [0,1], sinon 1.0
    ..TableSchema::default()
};
let (ir, report) = parse_table(&csv_text, &schema, Some("risques.csv".into()))?;
```

Sans schéma explicite, une table n'a pas de lecture causale définie : le
schéma est donc **obligatoire en paramètres** — jamais deviné, jamais en dur,
jamais en YAML (le moteur n'apprend pas de configs).

## Transparence

Lignes vides, relations inconnues, confiances illisibles ou hors [0, 1] :
**ignorées mais comptées** dans `TableParseReport`. Lignes à nombre de
champs incohérent, en-têtes dupliqués : erreur typée (fichier malformé,
pas de devinette).

## Sémantique assumée (F4 audit)

- Lignes identiques → arêtes dupliquées (la fréquence est un signal :
  deux lignes = deux observations, pas une).
- Auto-boucles (`x → x`) acceptées telles quelles — le middleend les
  traite comme tout cycle potentiel.
- Cellules de même libellé → même nœud (déduplication par label exact).
