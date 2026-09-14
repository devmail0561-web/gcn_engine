// Phase 3: Constraint propagation, causal graph construction, cycle detection, validation

pub mod cycle;
pub mod error;
pub mod graph;
pub mod propagate;
pub mod validate;

pub use error::{Diagnostic, DiagnosticKind, DiagnosticSeverity, MiddleendError};

use gcn_ir::CausalIR;

pub struct MiddleendResult {
    pub ir: CausalIR,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn process(mut ir: CausalIR) -> Result<MiddleendResult, MiddleendError> {
    let g = graph::build(&ir);

    let (cycles, edge_cycle_map) = cycle::detect_and_classify(&ir, &g);
    ir.cycles = cycles;
    for (edge_idx, cycle_id) in edge_cycle_map {
        ir.edges[edge_idx].2.in_cycle = Some(cycle_id);
    }

    let mut diagnostics = Vec::new();
    propagate::run(&mut ir, &g, &mut diagnostics);
    validate::run(&ir, &g, &mut diagnostics);

    ir.metadata.pipeline.push("gcn-middleend".to_string());

    Ok(MiddleendResult { ir, diagnostics })
}
