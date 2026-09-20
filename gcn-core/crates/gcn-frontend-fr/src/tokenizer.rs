// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

/// A single token in the tokenized stream.
#[derive(Debug, Clone)]
pub struct Token {
    /// 1-based index.
    pub index: u32,
    /// Surface form as it appears in text (punctuation stripped from edges).
    pub form: String,
    /// Lower-cased form for matching.
    pub lower: String,
}

/// Tokenize a French sentence.
///
/// Rules applied in order:
/// 1. Split on ASCII whitespace.
/// 2. For each raw chunk, strip leading/trailing ASCII punctuation (.,;:!?«»).
/// 3. Split clitics on the apostrophe (n', l', s', qu', j', m', t', c', d').
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut idx: u32 = 1;

    for raw in text.split_whitespace() {
        // Strip leading punctuation
        let trimmed_start = raw.trim_start_matches(|c: char| "«»\"'".contains(c));
        // Extract trailing punctuation tokens separately
        let (core, trailing) = split_trailing_punct(trimmed_start);

        if !core.is_empty() {
            for t in split_clitic(core) {
                tokens.push(make_token(idx, t));
                idx += 1;
            }
        }
        if let Some(p) = trailing {
            // Keep sentence boundaries and clause-splitting commas.
            if p == "." || p == "," {
                tokens.push(make_token(idx, p));
                idx += 1;
            }
        }
    }

    tokens
}

fn make_token(index: u32, form: &str) -> Token {
    Token {
        index,
        form: form.to_string(),
        lower: form.to_lowercase(),
    }
}

/// Strip trailing punctuation from a token, returning (core, Option<punct>).
fn split_trailing_punct(s: &str) -> (&str, Option<&str>) {
    if s.is_empty() {
        return (s, None);
    }
    let last = s.chars().last().unwrap();
    if ".,;:!?»\"'".contains(last) {
        let end = s.len() - last.len_utf8();
        (&s[..end], Some(&s[end..]))
    } else {
        (s, None)
    }
}

/// Split a token on apostrophe if it starts with a known clitic prefix.
fn split_clitic(s: &str) -> Vec<&str> {
    // Find apostrophe position (handle both ASCII ' and Unicode ')
    let apos_pos = s
        .char_indices()
        .find(|(_, c)| *c == '\'' || *c == '\u{2019}')
        .map(|(i, c)| (i, c.len_utf8()));

    if let Some((pos, apos_len)) = apos_pos {
        let prefix = &s[..pos];
        let lower_prefix = prefix.to_lowercase();
        let clitics = ["n", "l", "s", "j", "m", "t", "c", "d", "qu"];
        if clitics.contains(&lower_prefix.as_str()) {
            let clitic = &s[..pos + apos_len]; // e.g. "n'"
            let rest = &s[pos + apos_len..];    // e.g. "a"
            if !rest.is_empty() {
                return vec![clitic, rest];
            }
        }
    }
    vec![s]
}
