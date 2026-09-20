// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

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

    let mut stdin = child.stdin.take().ok_or_else(|| {
        VerbalizerError::ProcessSpawn(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "stdin du décodeur non pipé",
        ))
    })?;
    stdin
        .write_all(ir_json.as_bytes())
        .map_err(VerbalizerError::ProcessSpawn)?;
    // Fermer stdin AVANT wait : l'enfant lit jusqu'à EOF (`gcn-verbalize -`
    // fait `sys.stdin.read()`). Sans ce drop, parent et enfant s'attendent
    // mutuellement → deadlock systématique.
    drop(stdin);

    let output = child.wait_with_output().map_err(VerbalizerError::ProcessSpawn)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(VerbalizerError::DecoderError(stderr));
    }

    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_decoder_error() {
        let e = VerbalizerError::DecoderError("exit code 1".into());
        assert!(e.to_string().contains("exit code 1"));
    }

    #[test]
    fn error_display_serialization() {
        let inner: serde_json::Error =
            serde_json::from_str::<serde_json::Value>("{bad}").unwrap_err();
        let e = VerbalizerError::Serialization(inner);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn decode_returns_process_spawn_when_binary_absent() {
        use gcn_ir::{CausalIR, IrMetadata, NaturalLanguage, SourceLanguage};
        let ir = CausalIR {
            source_lang: SourceLanguage::Natural { lang: NaturalLanguage::Und },
            source_text: String::new(),
            nodes: vec![],
            edges: vec![],
            cycles: vec![],
            unresolved: vec![],
            metadata: IrMetadata::default(),
        };
        // gcn-verbalize n'est pas dans le PATH en environnement de test
        // SAFETY : test mono-thread, aucun autre thread ne lit PATH simultanément.
        unsafe { std::env::set_var("PATH", "") };
        let result = decode(&ir);
        assert!(matches!(
            result,
            Err(VerbalizerError::ProcessSpawn(_))
        ));
    }
}
