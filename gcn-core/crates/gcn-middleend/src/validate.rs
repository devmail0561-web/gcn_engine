// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{CausalIR, NodeType};
use std::collections::HashSet;

use crate::error::{Diagnostic, DiagnosticKind, DiagnosticSeverity};
use crate::graph::CausalGraph;

const MIN_CONFIDENCE: f32 = 0.3;

pub fn run(ir: &CausalIR, _g: &CausalGraph, diagnostics: &mut Vec<Diagnostic>) {
    check_dangling_edges(ir, diagnostics);
    check_self_loops(ir, diagnostics);
    check_orphaned_nodes(ir, diagnostics);
    check_low_confidence(ir, diagnostics);
    check_dangling_conditions(ir, diagnostics);
    check_empty_labels(ir, diagnostics);
    check_missing_provenance(ir, diagnostics);
}

fn check_dangling_edges(ir: &CausalIR, diagnostics: &mut Vec<Diagnostic>) {
    use std::collections::HashSet;
    let node_set: HashSet<u32> = ir.nodes.iter().map(|n| n.id.0).collect();
    for (idx, (src, dst, _)) in ir.edges.iter().enumerate() {
        if !node_set.contains(&src.0) || !node_set.contains(&dst.0) {
            diagnostics.push(Diagnostic {
                node_id: None,
                severity: DiagnosticSeverity::Error,
                kind: DiagnosticKind::DanglingEdge {
                    src: *src,
                    dst: *dst,
                    index: idx,
                },
            });
        }
    }
}

fn check_self_loops(ir: &CausalIR, diagnostics: &mut Vec<Diagnostic>) {
    for (src, dst, _) in &ir.edges {
        if src == dst {
            diagnostics.push(Diagnostic {
                node_id: Some(*src),
                severity: DiagnosticSeverity::Error,
                kind: DiagnosticKind::SelfLoop { node: *src },
            });
        }
    }
}

fn check_orphaned_nodes(ir: &CausalIR, diagnostics: &mut Vec<Diagnostic>) {
    if ir.nodes.len() <= 1 {
        return;
    }
    let connected: HashSet<u32> = ir.edges.iter().flat_map(|(s, d, _)| [s.0, d.0]).collect();
    for node in &ir.nodes {
        if !connected.contains(&node.id.0) {
            diagnostics.push(Diagnostic {
                node_id: Some(node.id),
                severity: DiagnosticSeverity::Warning,
                kind: DiagnosticKind::OrphanedNode { node: node.id },
            });
        }
    }
}

fn check_low_confidence(ir: &CausalIR, diagnostics: &mut Vec<Diagnostic>) {
    for (src, dst, edge) in &ir.edges {
        if edge.confidence < MIN_CONFIDENCE {
            diagnostics.push(Diagnostic {
                node_id: Some(*src),
                severity: DiagnosticSeverity::Warning,
                kind: DiagnosticKind::LowConfidenceEdge {
                    src: *src,
                    dst: *dst,
                    confidence: edge.confidence,
                },
            });
        }
    }
}

fn check_dangling_conditions(ir: &CausalIR, diagnostics: &mut Vec<Diagnostic>) {
    for node in &ir.nodes {
        if node.node_type == NodeType::Condition {
            let has_outgoing = ir.edges.iter().any(|(s, _, _)| *s == node.id);
            if !has_outgoing {
                diagnostics.push(Diagnostic {
                    node_id: Some(node.id),
                    severity: DiagnosticSeverity::Warning,
                    kind: DiagnosticKind::DanglingCondition { node: node.id },
                });
            }
        }
    }
}

fn check_empty_labels(ir: &CausalIR, diagnostics: &mut Vec<Diagnostic>) {
    for node in &ir.nodes {
        if node.label.trim().is_empty() {
            diagnostics.push(Diagnostic {
                node_id: Some(node.id),
                severity: DiagnosticSeverity::Warning,
                kind: DiagnosticKind::EmptyLabel { node: node.id },
            });
        }
    }
}

fn check_missing_provenance(ir: &CausalIR, diagnostics: &mut Vec<Diagnostic>) {
    for (index, (src, dst, edge)) in ir.edges.iter().enumerate() {
        if edge.provenance.is_none() {
            diagnostics.push(Diagnostic {
                node_id: None,
                severity: DiagnosticSeverity::Warning,
                kind: DiagnosticKind::MissingProvenance { index, src: *src, dst: *dst },
            });
        }
    }
}
