use gcn_ir::{CausalIR, GapNature, RelationType, TemporalGap};
use std::collections::HashMap;

use crate::error::{Diagnostic, DiagnosticKind, DiagnosticSeverity};
use crate::graph::CausalGraph;

pub fn run(ir: &mut CausalIR, _g: &CausalGraph, diagnostics: &mut Vec<Diagnostic>) {
    propagate_temporal_gaps(ir);
    check_sequence_ordering(ir, diagnostics);
}

/// Edges that signal a causal gap (Concession, Opposition) receive an explicit TemporalGap
/// if none was set by the frontend.
fn propagate_temporal_gaps(ir: &mut CausalIR) {
    for (_src, _dst, edge) in &mut ir.edges {
        if edge.relation.signals_causal_gap() && edge.temporal_gap.is_none() {
            edge.temporal_gap = Some(TemporalGap {
                min: None,
                max: None,
                nature: GapNature::Deferred,
            });
        }
    }
}

/// Sequence edges imply temporal ordering: src must precede dst.
/// Warn when temporal indices contradict this.
fn check_sequence_ordering(ir: &CausalIR, diagnostics: &mut Vec<Diagnostic>) {
    let ti: HashMap<u32, i32> = ir
        .nodes
        .iter()
        .filter_map(|n| n.temporal_index.map(|t| (n.id.0, t)))
        .collect();

    for (src, dst, edge) in &ir.edges {
        if edge.relation == RelationType::Sequence {
            if let (Some(&si), Some(&di)) = (ti.get(&src.0), ti.get(&dst.0)) {
                if si > di {
                    diagnostics.push(Diagnostic {
                        node_id: Some(*src),
                        severity: DiagnosticSeverity::Warning,
                        kind: DiagnosticKind::TemporalOrderViolation {
                            src: *src,
                            dst: *dst,
                        },
                    });
                }
            }
        }
    }
}
