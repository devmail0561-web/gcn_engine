use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser, Subcommand, ValueEnum};
use gcn_backend::{execute, to_dot, to_json, Query};
use gcn_frontend_fr::FrenchParser;
use gcn_ir::CausalIR;
use gcn_middleend::process as middleend_process;

#[derive(Parser)]
#[command(
    name = "gcn",
    version = "0.1.0",
    about = "Grammaire Causale Naturelle — Causal reasoning engine",
    long_about = "GCN-Core: parse natural language or code into causal graphs, \
                  query them with GCN-QL, and export results."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse text into a CausalIR graph
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
    /// Run the Python ML pipeline (gcn-python) on text — Rust↔Python interface
    Forward {
        /// Text to process
        text: String,
        /// Language code
        #[arg(long, default_value = "fr")]
        lang: String,
        /// Path to gcn-references/taxonomies/ (or set GCN_TAXONOMY_DIR)
        #[arg(long, env = "GCN_TAXONOMY_DIR")]
        taxonomy_dir: Option<PathBuf>,
        /// Apply middleend processing on the result
        #[arg(long)]
        enrich: bool,
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
        Commands::Analyze { text, data_dir, format, diagnostics } => {
            let parser = FrenchParser::new(&data_dir)?;
            let ir = parser.parse(&text)?;
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
            let ir = load_ir(&ir)?;
            let q = Query::parse(&query)?;
            let result = execute(&q, &ir)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Export { ir, format } => {
            let ir = load_ir(&ir)?;
            let output = match format {
                ExportFormat::Json => to_json(&ir)?,
                ExportFormat::Dot => to_dot(&ir)?,
            };
            println!("{output}");
        }

        Commands::Forward { text, lang, taxonomy_dir, enrich } => {
            let mut cmd = std::process::Command::new("gcn-forward");
            cmd.arg("--lang").arg(&lang);
            if let Some(dir) = &taxonomy_dir {
                cmd.arg("--taxonomy-dir").arg(dir);
            }
            cmd.arg(&text);

            let output = cmd.output().map_err(|e| {
                format!("failed to run gcn-forward (is gcn-python installed?): {e}")
            })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("gcn-forward failed: {stderr}").into());
            }

            let ir: CausalIR = serde_json::from_slice(&output.stdout)?;

            if enrich {
                let result = middleend_process(ir)?;
                println!("{}", to_json(&result.ir)?);
            } else {
                println!("{}", to_json(&ir)?);
            }
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
