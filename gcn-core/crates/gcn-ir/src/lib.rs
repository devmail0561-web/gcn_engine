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
