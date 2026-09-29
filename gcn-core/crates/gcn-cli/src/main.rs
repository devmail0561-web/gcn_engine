// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser, Subcommand, ValueEnum};
use gcn_backend::{Query, execute, to_dot, to_json};
use gcn_frontend_fr::FrenchParser;
use gcn_frontend_graph::parse_stix_bundle_with_report;
use gcn_frontend_table::{TableSchema, parse_table};
use gcn_ir::CausalIR;
use gcn_middleend::process as middleend_process;

mod extract;

#[derive(Parser)]
#[command(
    name = "gcn",
    version,
    about = "Grammaire Causale Naturelle — Causal reasoning engine",
    long_about = "GCN-Core CLI\n\
                  \nBootstrap annotation (symbolic, builds gcn-datasets/):\n\
                   gcn analyze  — French text → CausalIR via symbolic rules\n\
                   gcn analyze-graph — STIX 2.x bundle → CausalIR (typed graphs)\n\
                   gcn analyze-table — cause/effect CSV → CausalIR (schema via flags)\n\
                   gcn extract-text  — PDF/HTML/notebook/txt → sentences for analyze\n\
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
    /// [P2.1] Ingest a STIX 2.x bundle file → CausalIR (typed graphs, zero ML)
    AnalyzeGraph {
        /// Path to a STIX 2.x bundle JSON file
        bundle: PathBuf,
        /// Output format
        #[arg(long, default_value = "json")]
        format: ExportFormat,
    },
    /// [P2.3] Ingest a cause/effect CSV file → CausalIR (schema from CLI params)
    AnalyzeTable {
        /// Input CSV path
        #[arg(long)]
        input: PathBuf,
        /// Cause column header
        #[arg(long, default_value = "cause")]
        cause_col: String,
        /// Effect column header
        #[arg(long, default_value = "effect")]
        effect_col: String,
        /// Relation column header (snake_case values; defaults to cause)
        #[arg(long)]
        relation_col: Option<String>,
        /// Confidence column header (floats in [0, 1]; defaults to 1.0)
        #[arg(long)]
        confidence_col: Option<String>,
        /// Default relation when no column (snake_case)
        #[arg(long, default_value = "cause")]
        default_relation: String,
        /// Node type for all cells (snake_case)
        #[arg(long, default_value = "entite")]
        node_type: String,
        /// CSV delimiter (single ASCII char)
        #[arg(long, default_value = ",")]
        delimiter: String,
        /// Output format
        #[arg(long, default_value = "json")]
        format: ExportFormat,
    },
    /// [P2.2] Extract plain-text sentences from a document (PDF/HTML/notebook/txt)
    /// — pipeline step before `gcn analyze`
    ExtractText {
        /// Input document path
        #[arg(long)]
        input: PathBuf,
        /// Output file (one sentence per line). Defaults to stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Force format (auto-detected from extension by default)
        #[arg(long, value_enum)]
        format: Option<DocFormatArg>,
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

#[derive(ValueEnum, Clone, Copy, Debug)]
enum DocFormatArg {
    Txt,
    Html,
    Pdf,
    Ipynb,
}

impl From<DocFormatArg> for extract::DocFormat {
    fn from(f: DocFormatArg) -> Self {
        match f {
            DocFormatArg::Txt => extract::DocFormat::Txt,
            DocFormatArg::Html => extract::DocFormat::Html,
            DocFormatArg::Pdf => extract::DocFormat::Pdf,
            DocFormatArg::Ipynb => extract::DocFormat::Ipynb,
        }
    }
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

        Commands::AnalyzeGraph { bundle, format } => {
            let raw = std::fs::read_to_string(&bundle)
                .map_err(|e| format!("cannot read {}: {e}", bundle.display()))?;
            let (ir, report) = parse_stix_bundle_with_report(&raw)
                .map_err(|e| format!("invalid STIX bundle in {}: {e}", bundle.display()))?;
            let result = middleend_process(ir)?;
            eprintln!(
                "{} nodes, {} edges ({} relationships/objects skipped: {} unknown type, {} dangling, {} malformed)",
                result.ir.nodes.len(),
                result.ir.edges.len(),
                report.total_skipped(),
                report.skipped_unknown_relationships,
                report.skipped_dangling,
                report.skipped_objects,
            );
            let output = match format {
                ExportFormat::Json => to_json(&result.ir)?,
                ExportFormat::Dot => to_dot(&result.ir)?,
            };
            println!("{output}");
        }

        Commands::AnalyzeTable {
            input,
            cause_col,
            effect_col,
            relation_col,
            confidence_col,
            default_relation,
            node_type,
            delimiter,
            format,
        } => {
            use gcn_ir::{NodeType, RelationType};
            let default_relation = RelationType::from_name(&default_relation.to_lowercase())
                .ok_or_else(|| format!("unknown default relation: {default_relation}"))?;
            let node_type = NodeType::from_name(&node_type.to_lowercase())
                .ok_or_else(|| format!("unknown node type: {node_type}"))?;
            let delimiter = delimiter
                .as_bytes()
                .first()
                .copied()
                .filter(|b| {
                    delimiter.len() == 1 && b.is_ascii() && *b != b'"' && *b != b'\n' && *b != b'\r'
                })
                .ok_or_else(|| {
                    format!("delimiter must be a single ASCII char (got {delimiter:?})")
                })?;
            let raw = std::fs::read_to_string(&input)
                .map_err(|e| format!("cannot read {}: {e}", input.display()))?;
            let schema = TableSchema {
                cause_col,
                effect_col,
                relation_col,
                confidence_col,
                default_relation,
                default_node_type: node_type,
                delimiter,
            };
            let doc_ref = input.file_name().map(|n| n.to_string_lossy().into_owned());
            let (ir, report) = parse_table(&raw, &schema, doc_ref)
                .map_err(|e| format!("invalid CSV table in {}: {e}", input.display()))?;
            let result = middleend_process(ir)?;
            eprintln!(
                "{} nodes, {} edges ({} rows skipped: {} empty, {} unknown relation, {} bad confidence)",
                result.ir.nodes.len(),
                result.ir.edges.len(),
                report.total_skipped(),
                report.skipped_empty,
                report.skipped_unknown_relation,
                report.skipped_bad_confidence,
            );
            let output = match format {
                ExportFormat::Json => to_json(&result.ir)?,
                ExportFormat::Dot => to_dot(&result.ir)?,
            };
            println!("{output}");
        }

        Commands::ExtractText {
            input,
            output,
            format,
        } => {
            let doc_format = match format {
                Some(f) => f.into(),
                None => extract::detect_format(&input).map_err(|e| format!("extract-text: {e}"))?,
            };
            let raw = extract::extract_text(&input, doc_format)
                .map_err(|e| format!("extract-text: {e}"))?;
            let sentences = extract::split_sentences(&raw);
            let out = sentences.join("\n") + "\n";
            match output {
                Some(path) => std::fs::write(&path, &out)
                    .map_err(|e| format!("cannot write {}: {e}", path.display()))?,
                None => print!("{out}"),
            }
            eprintln!(
                "{} sentence(s) from {} ({})",
                sentences.len(),
                input.display(),
                doc_format.as_str()
            );
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

const MAX_IR_FILE_SIZE: u64 = 50 * 1024 * 1024; // 50 MB

fn load_ir(path: &Path) -> Result<CausalIR, Box<dyn std::error::Error>> {
    let meta =
        std::fs::metadata(path).map_err(|e| format!("cannot stat {}: {e}", path.display()))?;
    if meta.len() > MAX_IR_FILE_SIZE {
        return Err(format!(
            "{}: file too large ({} bytes, max {})",
            path.display(),
            meta.len(),
            MAX_IR_FILE_SIZE
        )
        .into());
    }
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let ir = serde_json::from_str(&content)
        .map_err(|e| format!("invalid CausalIR JSON in {}: {e}", path.display()))?;
    Ok(ir)
}
