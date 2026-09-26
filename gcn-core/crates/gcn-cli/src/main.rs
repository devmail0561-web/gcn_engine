// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser, Subcommand, ValueEnum};
use gcn_backend::{Query, execute, to_dot, to_json};
use gcn_frontend_fr::FrenchParser;
use gcn_ir::CausalIR;
use gcn_middleend::process as middleend_process;

#[derive(Parser)]
#[command(
    name = "gcn",
    version,
    about = "Grammaire Causale Naturelle — Causal reasoning engine",
    long_about = "GCN-Core CLI\n\
                  \nBootstrap annotation (symbolic, builds gcn-datasets/):\n\
                  gcn analyze  — French text → CausalIR via symbolic rules\n\
                  gcn query    — GCN-QL queries on a CausalIR\n\
                  gcn export   — Export CausalIR to JSON/DOT\n\
                  \nML inference: via l'API Python (GCNEngine) ou `gcn-discuss` \
                  (`gcn forward` retiré avec le binaire `gcn-forward`)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// [Bootstrap] Auto-annotate French text → CausalIR using symbolic rules (builds gcn-datasets/)
    Analyze {
        /// Text to analyze
        text: String,
        /// Path to gcn-references/taxonomies/ (or set GCN_TAXONOMY_DIR)
        #[arg(long, env = "GCN_TAXONOMY_DIR")]
        data_dir: PathBuf,
        /// Output format
        #[arg(long, default_value = "json")]
        format: ExportFormat,
        /// Emit middleend diagnostics
        #[arg(long)]
        diagnostics: bool,
    },
    /// Run a GCN-QL query on a CausalIR
    Query {
        /// GCN-QL query string (e.g. "WHY ventes?", "CYCLES?")
        query: String,
        /// Path to a CausalIR JSON file
        #[arg(long)]
        ir: PathBuf,
    },
    /// Export a CausalIR to a different format
    Export {
        /// Path to a CausalIR JSON file
        #[arg(long)]
        ir: PathBuf,
        /// Output format
        #[arg(long, default_value = "dot")]
        format: ExportFormat,
    },
    /// Retirée : le binaire `gcn-forward` n'existe plus (utilisez l'API Python
    /// `GCNEngine` ou `gcn-discuss`). Conservée pour un message d'erreur clair.
    Forward {
        /// Text to process
        text: String,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum ExportFormat {
    Json,
    Dot,
}

fn main() {
    let cli = Cli::parse();
    let result = run(cli.command);
    if let Err(e) = result {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn run(cmd: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        Commands::Analyze {
            text,
            data_dir,
            format,
            diagnostics,
        } => {
            let parser = FrenchParser::new(&data_dir)?;
            let ir = parser.parse_with_ref(&text, Some("cli:inline".to_string()))?;
            let result = middleend_process(ir)?;

            if diagnostics && !result.diagnostics.is_empty() {
                for d in &result.diagnostics {
                    eprintln!("[{:?}] {:?}", d.severity, d.kind);
                }
            }

            let output = match format {
                ExportFormat::Json => to_json(&result.ir)?,
                ExportFormat::Dot => to_dot(&result.ir)?,
            };
            println!("{output}");
        }

        Commands::Query { query, ir } => {
            let raw = load_ir(&ir)?;
            // Validate + recompute cycles — même chemin qu'analyze, évite IDs dupliqués
            // silencieux et ir.cycles périmés (audit adversarial P0).
            let processed = middleend_process(raw)?;
            for d in &processed.diagnostics {
                eprintln!("[{:?}] {:?}", d.severity, d.kind);
            }
            let q = Query::parse(&query)?;
            let result = execute(&q, &processed.ir)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Export { ir, format } => {
            let raw = load_ir(&ir)?;
            let processed = middleend_process(raw)?;
            for d in &processed.diagnostics {
                eprintln!("[{:?}] {:?}", d.severity, d.kind);
            }
            let output = match format {
                ExportFormat::Json => to_json(&processed.ir)?,
                ExportFormat::Dot => to_dot(&processed.ir)?,
            };
            println!("{output}");
        }

        Commands::Forward { .. } => {
            // `gcn-forward` a été supprimé de gcn-python (redondant avec
            // `gcn-discuss` et l'API Python `GCNEngine`). Ne plus tenter
            // d'exécuter un binaire inexistant (anciennement via GCN_PYTHON_BIN).
            return Err(
                "gcn forward a été retiré : le binaire `gcn-forward` n'existe plus. \
                 Utilisez l'API Python (GCNEngine.from_pretrained(..., trusted=True)) \
                 ou `gcn-discuss` pour l'inférence ML."
                    .into(),
            );
        }
    }
    Ok(())
}

fn load_ir(path: &Path) -> Result<CausalIR, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let ir = serde_json::from_str(&content)
        .map_err(|e| format!("invalid CausalIR JSON in {}: {e}", path.display()))?;
    Ok(ir)
}
