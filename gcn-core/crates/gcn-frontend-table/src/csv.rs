// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Mini-parseur CSV RFC-4180 (sans dépendance) : délimiteur configurable,
//! champs quotés, guillemets doublés (`""`), CRLF, champs multilignes.

use crate::error::TableParserError;

/// Parse un texte CSV en lignes de champs. La première ligne est l'en-tête.
pub fn parse_csv(text: &str, delimiter: u8) -> Result<Vec<Vec<String>>, TableParserError> {
    let bytes = text.as_bytes();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut field = String::new();
    let mut row: Vec<String> = Vec::new();
    // Numéro de ligne physique (pour les erreurs) — 1-based.
    let mut line_no: usize = 1;
    let mut i = 0;

    // BOM UTF-8 éventuelle ignorée.
    if bytes.starts_with(b"\xef\xbb\xbf") {
        i = 3;
    }

    macro_rules! end_field {
        () => {{
            row.push(std::mem::take(&mut field));
        }};
    }
    macro_rules! end_row {
        () => {{
            end_field!();
            // Ligne entièrement vide (que du whitespace) : ignorée SAUF l'en-tête.
            let all_empty = row.iter().all(|f| f.trim().is_empty());
            if !rows.is_empty() && all_empty && row.len() <= 1 {
                row.clear();
            } else {
                rows.push(std::mem::take(&mut row));
            }
        }};
    }

    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' {
            // Champ quoté : doit commencer en début de champ (tolérant sinon).
            let mut buf = std::mem::take(&mut field);
            i += 1;
            let mut closed = false;
            while i < bytes.len() {
                match bytes[i] {
                    b'"' => {
                        if bytes.get(i + 1) == Some(&b'"') {
                            buf.push('"');
                            i += 2;
                        } else {
                            closed = true;
                            i += 1;
                            break;
                        }
                    }
                    b'\n' => {
                        line_no += 1;
                        buf.push('\n');
                        i += 1;
                    }
                    _ => {
                        // Décodage UTF-8 par caractère (le CSV est supposé UTF-8).
                        let s = &text[i..];
                        match s.chars().next() {
                            Some(c) => {
                                buf.push(c);
                                i += c.len_utf8();
                            }
                            None => {
                                i += 1;
                            }
                        }
                    }
                }
            }
            if !closed {
                return Err(TableParserError::UnterminatedQuote(line_no));
            }
            field = buf;
        } else if b == delimiter {
            end_field!();
            i += 1;
        } else if b == b'\r' {
            // CRLF ou CR seul : fin de ligne.
            if bytes.get(i + 1) == Some(&b'\n') {
                i += 2;
            } else {
                i += 1;
            }
            line_no += 1;
            end_row!();
        } else if b == b'\n' {
            line_no += 1;
            i += 1;
            end_row!();
        } else {
            let s = &text[i..];
            match s.chars().next() {
                Some(c) => {
                    field.push(c);
                    i += c.len_utf8();
                }
                None => {
                    i += 1;
                }
            }
        }
    }
    // Dernier champ/ligne sans newline final.
    if !field.is_empty() || !row.is_empty() {
        end_row!();
    }
    Ok(rows)
}
