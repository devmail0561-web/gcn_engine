// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

/// Normalise un label CIR pour la comparaison et la déduplication.
///
/// Transformations appliquées dans l'ordre :
/// 1. Trim des espaces en bord
/// 2. Fold des accents courants (FR/ES/PT)
/// 3. Lowercase ASCII
/// 4. Suppression des caractères hors [a-z0-9 _-]
///
/// Exemples :
/// ```
/// use gcn_ir::normalize_label;
/// assert_eq!(normalize_label("Authentification"), "authentification");
/// assert_eq!(normalize_label("  économie  "), "economie");
/// assert_eq!(normalize_label("auth/login"), "authlogin");
/// assert_eq!(normalize_label("État-Système"), "etat-systeme");
/// ```
pub fn normalize_label(s: &str) -> String {
    // Substituer les ligatures multi-char avant le traitement char-à-char
    let pre = s.trim()
        .replace('œ', "oe").replace('Œ', "oe")
        .replace('æ', "ae").replace('Æ', "ae");
    pre.chars()
        .map(fold_accent)
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '-'))
        .collect::<String>()
        .to_ascii_lowercase()
}

fn fold_accent(c: char) -> char {
    match c {
        'À' | 'Â' | 'Ä' | 'à' | 'â' | 'ä' => 'a',
        'É' | 'È' | 'Ê' | 'Ë' | 'é' | 'è' | 'ê' | 'ë' => 'e',
        'Î' | 'Ï' | 'î' | 'ï' => 'i',
        'Ô' | 'Ö' | 'ô' | 'ö' => 'o',
        'Ù' | 'Û' | 'Ü' | 'ù' | 'û' | 'ü' => 'u',
        'Ç' | 'ç' => 'c',
        'Ñ' | 'ñ' => 'n',
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_ascii() {
        assert_eq!(normalize_label("Authentification"), "authentification");
        assert_eq!(normalize_label("  login  "), "login");
        assert_eq!(normalize_label("auth_token"), "auth_token");
    }

    #[test]
    fn normalize_accents() {
        assert_eq!(normalize_label("économie"), "economie");
        assert_eq!(normalize_label("État-Système"), "etat-systeme");
        assert_eq!(normalize_label("bœuf"), "boeuf");
    }

    #[test]
    fn normalize_strips_punct() {
        assert_eq!(normalize_label("auth/login"), "authlogin");
        assert_eq!(normalize_label("cause (principale)"), "cause principale");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize_label(""), "");
        assert_eq!(normalize_label("   "), "");
    }
}
