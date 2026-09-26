// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Causal Intermediate Representation types for GCN-Core.

pub mod code;
pub mod edge;
pub mod error;
pub mod ir;
pub mod modifier;
pub mod node;
pub mod normalize;
pub mod scope;
pub mod temporal;

pub use code::{CodeAttributes, CodeCausalType, ControlFlow, DataFlow};
pub use edge::{CausalEdge, CycleId, Derivation, ExtractionMethod, Provenance, RelationType};
pub use error::{GcnError, GcnResult};
pub use ir::{
    Ambiguity, AmbiguityCandidate, AmbiguousField, CausalCycle, CausalIR, CycleType, IrMetadata,
    NaturalLanguage, ProgrammingLanguage, SourceLanguage,
};
pub use modifier::{FrequencyKind, LocationScope, Maturity, Modifier};
pub use node::{
    AgentType, CausalDirection, CausalNode, NodeAttributes, NodeId, NodeOrigin, NodeType,
    SourceSpan,
};
pub use normalize::normalize_label;
pub use scope::Scope;
pub use temporal::{GapNature, TemporalAnchor, TemporalGap, TemporalRef};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_type_causal_gap_flags() {
        assert!(RelationType::Concession.signals_causal_gap());
        assert!(RelationType::Opposition.signals_causal_gap());
        assert!(!RelationType::Cause.signals_causal_gap());
        assert!(!RelationType::Enable.signals_causal_gap());
        assert!(!RelationType::Sequence.signals_causal_gap());
    }

    #[test]
    fn node_type_serde_roundtrip() {
        for nt in [
            NodeType::Etat,
            NodeType::Action,
            NodeType::Transition,
            NodeType::Processus,
            NodeType::Condition,
            NodeType::Entite,
            NodeType::EtatSystemique,
        ] {
            let s = serde_json::to_string(&nt).expect("serialize NodeType");
            let back: NodeType = serde_json::from_str(&s).expect("deserialize NodeType");
            assert_eq!(nt, back);
        }
    }

    #[test]
    fn relation_type_serde_roundtrip() {
        for rt in [
            RelationType::Cause,
            RelationType::Enable,
            RelationType::Prevent,
            RelationType::Condition,
            RelationType::Filter,
            RelationType::DataDependency,
            RelationType::ControlDependency,
        ] {
            let s = serde_json::to_string(&rt).expect("serialize RelationType");
            let back: RelationType = serde_json::from_str(&s).expect("deserialize RelationType");
            assert_eq!(rt, back);
        }
    }

    #[test]
    fn provenance_serde_roundtrip() {
        let p = Provenance {
            doc_ref: Some("docs/spec.pdf".to_string()),
            span: SourceSpan::TokenSpan { start: 0, end: 10 },
            extraction_method: ExtractionMethod::SymbolicRust,
            model_version: "2.5.0".to_string(),
            extracted_at: "2026-09-26T12:00:00Z".to_string(),
        };
        let s = serde_json::to_string(&p).expect("serialize Provenance");
        let back: Provenance = serde_json::from_str(&s).expect("deserialize Provenance");
        assert_eq!(back.model_version, "2.5.0");
        // "ref" serialized as "ref" in JSON
        assert!(s.contains("\"ref\""));
        assert!(s.contains("docs/spec.pdf"));
    }

    #[test]
    fn causal_edge_provenance_optional_roundtrip() {
        // Old CIR without provenance must still deserialize (backward compat)
        let json = r#"{"relation":"cause","confidence":0.9,"temporal_gap":null,"explicit":true,"negated":false,"marker_token":null,"in_cycle":null}"#;
        let edge: CausalEdge = serde_json::from_str(json).expect("deserialize old CausalEdge");
        assert!(edge.provenance.is_none());
        assert!(edge.derivation.is_none());
        // Round-trip with provenance
        let edge2 = CausalEdge {
            relation: RelationType::Cause,
            confidence: 0.85,
            temporal_gap: None,
            explicit: true,
            negated: false,
            marker_token: None,
            in_cycle: None,
            provenance: Some(Provenance {
                doc_ref: None,
                span: SourceSpan::Synthetic,
                extraction_method: ExtractionMethod::SymbolicRust,
                model_version: "2.5.0".to_string(),
                extracted_at: "2026-09-26T00:00:00Z".to_string(),
            }),
            derivation: None,
        };
        let s2 = serde_json::to_string(&edge2).expect("serialize CausalEdge with provenance");
        let back2: CausalEdge = serde_json::from_str(&s2).expect("deserialize CausalEdge with provenance");
        assert!(back2.provenance.is_some());
    }

    #[test]
    fn causal_ir_empty_counts() {
        let ir = CausalIR {
            source_lang: SourceLanguage::Natural {
                lang: NaturalLanguage::French,
            },
            source_text: String::new(),
            nodes: vec![],
            edges: vec![],
            cycles: vec![],
            unresolved: vec![],
            metadata: IrMetadata::default(),
        };
        assert_eq!(ir.node_count(), 0);
        assert_eq!(ir.edge_count(), 0);
        assert!(!ir.has_cycles());
        assert!(!ir.has_gaps());
    }
}
