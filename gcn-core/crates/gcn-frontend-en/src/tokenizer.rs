// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

/// Simple whitespace + punctuation tokenizer for English.
#[derive(Debug, Clone)]
pub struct Token {
    pub form: String,
    pub lower: String,
    pub index: u32,
}

pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut idx: u32 = 0;

    for raw in text.split_whitespace() {
        // Strip trailing punctuation but preserve it as a separate token
        let (word, punct) = split_punct(raw);
        if !word.is_empty() {
            tokens.push(Token {
                form: word.to_string(),
                lower: word.to_lowercase(),
                index: idx,
            });
            idx += 1;
        }
        if let Some(p) = punct {
            tokens.push(Token {
                form: p.to_string(),
                lower: p.to_lowercase(),
                index: idx,
            });
            idx += 1;
        }
    }
    tokens
}

fn split_punct(s: &str) -> (&str, Option<&str>) {
    if let Some(last) = s.chars().last()
        && matches!(last, '.' | ',' | ';' | ':' | '!' | '?') {
        let split = s.len() - last.len_utf8();
        return (&s[..split], Some(&s[split..]));
    }
    (s, None)
}
