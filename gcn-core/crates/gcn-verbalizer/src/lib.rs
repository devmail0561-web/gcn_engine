use std::io::Write;
use std::process::{Command, Stdio};

use gcn_ir::CausalIR;

#[derive(Debug, thiserror::Error)]
pub enum VerbalizerError {
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("failed to run gcn-verbalize (is gcn-python installed?): {0}")]
    ProcessSpawn(std::io::Error),
    #[error("gcn-verbalize failed: {0}")]
    DecoderError(String),
    #[error("invalid UTF-8 in decoder output: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

/// Decode a CausalIR to a surface form via the Python decoder.
///
/// Sends the CausalIR as JSON to the `gcn-verbalize` subprocess (stdin)
/// and returns its stdout. What the decoder produces depends on its
/// training data — no output format is presupposed by the architecture.
pub fn decode(ir: &CausalIR) -> Result<String, VerbalizerError> {
    let ir_json = serde_json::to_string(ir)?;

    let mut child = Command::new("gcn-verbalize")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(VerbalizerError::ProcessSpawn)?;

    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(ir_json.as_bytes())
        .map_err(VerbalizerError::ProcessSpawn)?;

    let output = child.wait_with_output().map_err(VerbalizerError::ProcessSpawn)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(VerbalizerError::DecoderError(stderr));
    }

    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}
