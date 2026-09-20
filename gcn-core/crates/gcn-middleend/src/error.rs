// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::NodeId;

#[derive(Debug, thiserror::Error)]
pub enum MiddleendError {
    #[error("graph construction failed: {0}")]
    GraphBuildError(String),
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub node_id: Option<NodeId>,
    pub severity: DiagnosticSeverity,
    pub kind: DiagnosticKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone)]
pub enum DiagnosticKind {
    SelfLoop { node: NodeId },
    OrphanedNode { node: NodeId },
    LowConfidenceEdge { src: NodeId, dst: NodeId, confidence: f32 },
    DanglingCondition { node: NodeId },
    DanglingEdge { src: NodeId, dst: NodeId, index: usize },
    TemporalOrderViolation { src: NodeId, dst: NodeId },
}
