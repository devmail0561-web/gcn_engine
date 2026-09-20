//! Causal Intermediate Representation types for GCN-Core.

pub mod code;
pub mod edge;
pub mod error;
pub mod ir;
pub mod modifier;
pub mod node;
pub mod scope;
pub mod temporal;

pub use code::{CodeAttributes, CodeCausalType, ControlFlow, DataFlow};
pub use edge::{CausalEdge, CycleId, RelationType};
pub use error::{GcnError, GcnResult};
pub use ir::{
    Ambiguity, AmbiguousField, AmbiguityCandidate, CausalCycle, CausalIR, CycleType, IrMetadata,
    NaturalLanguage, ProgrammingLanguage, SourceLanguage,
};
pub use modifier::{FrequencyKind, LocationScope, Maturity, Modifier};
pub use node::{AgentType, CausalDirection, CausalNode, NodeAttributes, NodeId, NodeOrigin, NodeType, SourceSpan};
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
    fn causal_ir_empty_counts() {
        let ir = CausalIR {
            source_lang: SourceLanguage::Natural { lang: NaturalLanguage::French },
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
