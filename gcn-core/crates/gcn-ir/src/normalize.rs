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
    let pre = s
        .trim()
        .replace(['œ', 'Œ'], "oe")
        .replace(['æ', 'Æ'], "ae");
    pre.chars()
        .map(fold_accent)
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '-'))
        .collect::<String>()
        .to_ascii_lowercase()
}

fn fold_accent(c: char) -> char {
    match c {
        'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'à' | 'á' | 'â' | 'ã' | 'ä' | 'Ą' | 'ą' => 'a',
        'É' | 'È' | 'Ê' | 'Ë' | 'é' | 'è' | 'ê' | 'ë' | 'Ě' | 'ě' | 'Ę' | 'ę' => 'e',
        'Î' | 'Ï' | 'Í' | 'î' | 'ï' | 'í' => 'i',
        'Ô' | 'Ö' | 'Ó' | 'ô' | 'ö' | 'ó' => 'o',
        'Ù' | 'Û' | 'Ü' | 'Ú' | 'ù' | 'û' | 'ü' | 'ú' | 'Ů' | 'ů' => 'u',
        'Ç' | 'ç' | 'Č' | 'č' => 'c',
        'Ñ' | 'ñ' | 'Ń' | 'ń' | 'Ň' | 'ň' => 'n',
        'Š' | 'š' | 'Ś' | 'ś' | 'Ş' | 'ş' => 's',
        'Ž' | 'ž' | 'Ź' | 'ź' | 'Ż' | 'ż' => 'z',
        'Ř' | 'ř' => 'r',
        'Ď' | 'ď' => 'd',
        'Ť' | 'ť' => 't',
        'Ý' | 'ý' => 'y',
        'Ğ' | 'ğ' => 'g',
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
    fn normalize_extended_accents() {
        assert_eq!(normalize_label("přístup"), "pristup");
        assert_eq!(normalize_label("żądanie"), "zadanie");
        assert_eq!(normalize_label("güvenlik"), "guvenlik");
        assert_eq!(normalize_label("señal"), "senal");
        assert_eq!(normalize_label("último"), "ultimo");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize_label(""), "");
        assert_eq!(normalize_label("   "), "");
    }
}
