# gcn-frontend-graph

Frontend **graphes typés → CausalIR** (P2.1 ETUDE). Premier client : bundles **STIX 2.x**
(MITRE ATT&CK, rapports CTI structurés).

## Pourquoi pas de ML

Dans STIX, les relations causales sont déjà explicites et typées
(`relationship_type: "uses"` → `RelationType::Enable`). Même patron que
`gcn-frontend-code` avec `serde_json` à la place de tree-sitter :

```
objets STIX (unités) → relationship_type (signal) → RelationType → CausalIR
```

`SourceSpan::Synthetic`, confiance `1.0`, provenance = id du bundle.

## Usage

```rust
use gcn_frontend_graph::parse_stix_bundle_with_report;

let (ir, report) = parse_stix_bundle_with_report(&bundle_json)?;
println!("{} nœuds, {} arêtes ({} ignorées)",
    ir.nodes.len(), ir.edges.len(), report.total_skipped());
```

## Transparence (anti-silence, §2.12 ETUDE)

Relations non causales (`related-to`, `located-at`, `impersonates`,
`investigates`, `belongs-to`, `resolves-to`), références pendantes et objets
sans `id`/`type` : **ignorés mais comptés** dans `GraphParseReport`.
Types d'objets inconnus (extensions STIX futures) : repli `Concept`, jamais d'échec.
