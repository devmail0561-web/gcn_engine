// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Préprocesseur document (P2.2 ETUDE — pas un crate, une étape de pipeline).
//!
//! Les use cases réels livrent des PDF/HTML/notebooks, pas du `.txt` prêt pour
//! le frontend NL. Ce module extrait le texte brut puis découpe en phrases
//! (une par ligne), prêtes pour `gcn analyze`.
//!
//! Stratégie par format — honnêteté affichée :
//! - `.txt`/`.md` : recopie telle quelle ;
//! - `.html` : détaggage interne (script/style/commentaires supprimés, entités
//!   décodées), sans dépendance ;
//! - `.ipynb` : cellules markdown via `serde_json` ;
//! - `.pdf` : `pdftotext` (poppler) externe en mode `-layout`. **Sans poppler,
//!   erreur explicite** — aucun extracteur naïf (un parseur de streams sans
//!   gestion des encodages produit du mojibake qui polluerait le pipeline NL,
//!   cf. risque §3.4 ETUDE).

use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExtractError {
    #[error("cannot read {0}: {1}")]
    Io(String, std::io::Error),
    #[error(
        "unsupported extension for {0} (expected .txt/.md/.html/.htm/.ipynb/.pdf, or pass --format)"
    )]
    UnsupportedExtension(String),
    #[error(
        "pdftotext (poppler) introuvable dans PATH — installez poppler-utils \
         (apt install poppler-utils / dnf install poppler-utils / brew install poppler) \
         pour extraire les PDF"
    )]
    PdfToolMissing,
    #[error("pdftotext failed on {0}: {1}")]
    PdfToolFailed(String, String),
    #[error("invalid notebook {0}: {1}")]
    InvalidNotebook(String, serde_json::Error),
    #[error("notebook {0} has no cells array (not a Jupyter notebook)")]
    NotebookWithoutCells(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocFormat {
    Txt,
    Html,
    Pdf,
    Ipynb,
}

impl DocFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            DocFormat::Txt => "txt",
            DocFormat::Html => "html",
            DocFormat::Pdf => "pdf",
            DocFormat::Ipynb => "ipynb",
        }
    }
}

/// Détecte le format par extension (minuscules). Erreur si inconnue.
pub fn detect_format(path: &Path) -> Result<DocFormat, ExtractError> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .as_deref()
    {
        Some("txt") | Some("md") | Some("markdown") => Ok(DocFormat::Txt),
        Some("html") | Some("htm") => Ok(DocFormat::Html),
        Some("pdf") => Ok(DocFormat::Pdf),
        Some("ipynb") => Ok(DocFormat::Ipynb),
        _ => Err(ExtractError::UnsupportedExtension(
            path.display().to_string(),
        )),
    }
}

fn read_file(path: &Path) -> Result<String, ExtractError> {
    std::fs::read_to_string(path).map_err(|e| ExtractError::Io(path.display().to_string(), e))
}

/// Extrait le texte brut d'un document selon son format.
pub fn extract_text(path: &Path, format: DocFormat) -> Result<String, ExtractError> {
    match format {
        DocFormat::Txt => read_file(path),
        DocFormat::Html => Ok(strip_html(&read_file(path)?)),
        DocFormat::Ipynb => extract_notebook(&read_file(path)?, path),
        DocFormat::Pdf => extract_pdf(path),
    }
}

// ---------------------------------------------------------------------------
// PDF — pdftotext externe
// ---------------------------------------------------------------------------

fn extract_pdf(path: &Path) -> Result<String, ExtractError> {
    let out = std::process::Command::new("pdftotext")
        .args(["-layout", &path.display().to_string(), "-"])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ExtractError::PdfToolMissing
            } else {
                ExtractError::Io(path.display().to_string(), e)
            }
        })?;
    if !out.status.success() {
        let detail = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(ExtractError::PdfToolFailed(
            path.display().to_string(),
            if detail.is_empty() {
                format!("exit {}", out.status)
            } else {
                detail.chars().take(300).collect()
            },
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

// ---------------------------------------------------------------------------
// HTML — détaggage interne
// ---------------------------------------------------------------------------

/// Supprime scripts, styles, commentaires et balises ; décode les entités ;
/// une ligne logique par bloc (p/div/br/h1-h6/li/tr...).
pub fn strip_html(html: &str) -> String {
    let no_comments = strip_between(html, "<!--", "-->");
    let no_script = strip_tag_blocks(&no_comments, "script");
    let no_style = strip_tag_blocks(&no_script, "style");
    // Frontières de blocs → saut de ligne avant suppression des balises.
    let with_breaks = replace_block_tags(&no_style);
    let no_tags = strip_tags(&with_breaks);
    let decoded = decode_entities(&no_tags);
    // Normalise : espaces tassés, lignes vides supprimées.
    decoded
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Supprime les occurrences `open...close` (commentaires).
fn strip_between(s: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find(open) {
        out.push_str(&rest[..start]);
        match rest[start..].find(close) {
            Some(end) => rest = &rest[start + end + close.len()..],
            None => return out, // commentaire non fermé : tronque la fin
        }
    }
    out.push_str(rest);
    out
}

/// Recherche ASCII insensible à la casse — retourne un index d'octet valide
/// pour `haystack` (les motifs sont purs ASCII : aucun décalage possible,
/// contrairement à une ombre `to_lowercase()` dont la longueur d'octets
/// peut différer — panique sur caractères type U+0130).
fn find_ascii_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let hb = haystack.as_bytes();
    let nb = needle.as_bytes();
    if nb.is_empty() || nb.len() > hb.len() {
        return None;
    }
    (0..=hb.len() - nb.len()).find(|&i| {
        hb[i..i + nb.len()]
            .iter()
            .zip(nb.iter())
            .all(|(h, n)| h.to_ascii_lowercase() == *n)
    })
}

/// Supprime les blocs `<tag ...>...</tag>` (insensible à la casse, attributs OK).
fn strip_tag_blocks(s: &str, tag: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    let open_pat = format!("<{tag}");
    let close_pat = format!("</{tag}>");
    while let Some(start) = find_ascii_insensitive(rest, &open_pat) {
        // Vérifie que c'est bien une balise ouvrante (<tag suivi de espace/>/>).
        let after = rest[start + open_pat.len()..].chars().next();
        match after {
            Some(c) if c.is_whitespace() || c == '>' || c == '/' => {}
            _ => {
                out.push_str(&rest[..start + 1]);
                rest = &rest[start + 1..];
                continue;
            }
        }
        out.push_str(&rest[..start]);
        match find_ascii_insensitive(&rest[start..], &close_pat) {
            Some(end) => {
                rest = &rest[start + end + close_pat.len()..];
            }
            None => {
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

const BLOCK_TAGS: &[&str] = &[
    "p",
    "div",
    "br",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "tr",
    "section",
    "article",
    "header",
    "footer",
    "blockquote",
    "pre",
    "hr",
    "table",
];

/// Remplace les balises de bloc par `\n` (le reste des balises est supprimé après).
fn replace_block_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find('<') {
        out.push_str(&rest[..start]);
        let tag_end = rest[start..].find('>').map(|i| start + i);
        match tag_end {
            Some(end) => {
                let inner = rest[start + 1..end].trim().to_lowercase();
                let name: String = inner
                    .trim_start_matches('/')
                    .chars()
                    .take_while(|c| c.is_alphanumeric())
                    .collect();
                if BLOCK_TAGS.contains(&name.as_str()) {
                    out.push('\n');
                }
                rest = &rest[end + 1..];
            }
            None => {
                out.push_str(&rest[start..]);
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Supprime toutes les balises `<...>` restantes.
fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find('<') {
        out.push_str(&rest[..start]);
        match rest[start..].find('>') {
            Some(end) => rest = &rest[start + end + 1..],
            None => {
                out.push_str(&rest[start..]);
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Décode les entités nommées courantes + numériques (`&#233;`, `&#xE9;`).
fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        match tail.find(';') {
            Some(end) if end <= 12 => {
                let entity = &tail[1..end];
                if let Some(decoded) = decode_entity(entity) {
                    out.push_str(&decoded);
                } else {
                    out.push_str(&tail[..=end]); // entité inconnue : inchangée
                }
                rest = &tail[end + 1..];
            }
            _ => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn decode_entity(entity: &str) -> Option<String> {
    match entity {
        "amp" => Some("&".to_string()),
        "lt" => Some("<".to_string()),
        "gt" => Some(">".to_string()),
        "quot" => Some("\"".to_string()),
        "apos" => Some("'".to_string()),
        "nbsp" => Some(" ".to_string()),
        "hellip" => Some("…".to_string()),
        "mdash" => Some("—".to_string()),
        "ndash" => Some("–".to_string()),
        "laquo" => Some("«".to_string()),
        "raquo" => Some("»".to_string()),
        "lsquo" => Some("'".to_string()),
        "rsquo" => Some("'".to_string()),
        "ldquo" => Some("\"".to_string()),
        "rdquo" => Some("\"".to_string()),
        // Latin-1 accentué (FR courant dans les documents)
        "agrave" => Some("à".to_string()),
        "aacute" => Some("á".to_string()),
        "acirc" => Some("â".to_string()),
        "auml" => Some("ä".to_string()),
        "ccedil" => Some("ç".to_string()),
        "egrave" => Some("è".to_string()),
        "eacute" => Some("é".to_string()),
        "ecirc" => Some("ê".to_string()),
        "euml" => Some("ë".to_string()),
        "igrave" => Some("ì".to_string()),
        "iacute" => Some("í".to_string()),
        "icirc" => Some("î".to_string()),
        "iuml" => Some("ï".to_string()),
        "ograve" => Some("ò".to_string()),
        "oacute" => Some("ó".to_string()),
        "ocirc" => Some("ô".to_string()),
        "ouml" => Some("ö".to_string()),
        "ugrave" => Some("ù".to_string()),
        "uacute" => Some("ú".to_string()),
        "ucirc" => Some("û".to_string()),
        "uuml" => Some("ü".to_string()),
        "yuml" => Some("ÿ".to_string()),
        "szlig" => Some("ß".to_string()),
        "Agrave" => Some("À".to_string()),
        "Egrave" => Some("È".to_string()),
        "Eacute" => Some("É".to_string()),
        "Ecirc" => Some("Ê".to_string()),
        "Ccedil" => Some("Ç".to_string()),
        "sect" => Some("§".to_string()),
        "copy" => Some("©".to_string()),
        _ if entity.starts_with('#') => {
            let num = entity.strip_prefix('#')?;
            let code = if let Some(hex) = num.strip_prefix('x').or_else(|| num.strip_prefix('X')) {
                u32::from_str_radix(hex, 16).ok()?
            } else {
                num.parse::<u32>().ok()?
            };
            char::from_u32(code).map(|c| c.to_string())
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Notebooks Jupyter — cellules markdown
// ---------------------------------------------------------------------------

/// Concatène les sources des cellules `markdown` (séparées par une ligne vide).
/// Les cellules de code sont ignorées : ce n'est pas du NL.
fn extract_notebook(json_text: &str, path: &Path) -> Result<String, ExtractError> {
    let doc: serde_json::Value = serde_json::from_str(json_text)
        .map_err(|e| ExtractError::InvalidNotebook(path.display().to_string(), e))?;
    let cells = doc
        .get("cells")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ExtractError::NotebookWithoutCells(path.display().to_string()))?;
    let mut parts: Vec<String> = Vec::new();
    for cell in cells {
        let is_markdown = cell.get("cell_type").and_then(|t| t.as_str()) == Some("markdown");
        if !is_markdown {
            continue;
        }
        match cell.get("source") {
            Some(serde_json::Value::String(s)) => parts.push(s.clone()),
            Some(serde_json::Value::Array(lines)) => {
                let joined: String = lines.iter().filter_map(|l| l.as_str()).collect();
                parts.push(joined);
            }
            _ => {}
        }
    }
    Ok(parts.join("\n\n"))
}

// ---------------------------------------------------------------------------
// Découpage en phrases (FR/EN, règles)
// ---------------------------------------------------------------------------

/// Abréviations qui ne terminent pas une phrase (comparaison insensible à la casse).
const ABBREVIATIONS: &[&str] = &[
    "m", "mme", "mlle", "dr", "pr", "me", "mm", "st", "ste", "ex", "cf", "fig", "vol", "chap",
    "art", "av", "bd", "mr", "mrs", "ms", "jr", "sr", "vs", "etc", "ed", "no", "nos",
];

/// Découpe en phrases : frontière sur `.`/`!`/`?`/`…` suivie d'espace ou fin de
/// texte, sauf décimaux (`3.14`), abréviations (`M.`, `Dr.`) et pointillés
/// internes (`e.g.`). Lignes vides et résidus < 2 alphanumériques écartés.
pub fn split_sentences(text: &str) -> Vec<String> {
    // Aplatit d'abord en un flux (les phrases peuvent chevaucher des lignes).
    let flat: String = text
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let chars: Vec<char> = flat.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if matches!(c, '.' | '!' | '?' | '…') {
            // Groupe `...`, `?!`, `!!` : avance jusqu'à la fin du groupe.
            let mut j = i;
            while j < chars.len() && matches!(chars[j], '.' | '!' | '?') {
                j += 1;
            }
            // `…` seul (pas suivi de .!?) : frontière immédiate.
            let end = if c == '…' && j == i { i + 1 } else { j };
            let next = chars.get(end).copied();
            let is_boundary = next.is_none() || next.is_some_and(|n| n.is_whitespace());
            if is_boundary && !is_decimal(&chars, i, end) && !is_abbreviation(&chars, start, i) {
                let sentence: String = chars[start..end].iter().collect();
                let trimmed = sentence.trim().to_string();
                if trimmed.chars().filter(|ch| ch.is_alphanumeric()).count() >= 2 {
                    out.push(trimmed);
                }
                // Saute les espaces après la frontière.
                let mut k = end;
                while k < chars.len() && chars[k].is_whitespace() {
                    k += 1;
                }
                start = k;
                i = k;
                continue;
            }
            i = end;
            continue;
        }
        i += 1;
    }
    let tail: String = chars[start..].iter().collect();
    let trimmed = tail.trim().to_string();
    if trimmed.chars().filter(|ch| ch.is_alphanumeric()).count() >= 2 {
        out.push(trimmed);
    }
    out
}

/// Vrai si le `.` est entre deux chiffres (`3.14`).
fn is_decimal(chars: &[char], dot: usize, group_end: usize) -> bool {
    // Ne s'applique qu'à un point seul, pas à `...`.
    if group_end - dot != 1 {
        return false;
    }
    let prev = dot.checked_sub(1).and_then(|p| chars.get(p)).copied();
    let next = chars.get(dot + 1).copied();
    matches!(prev, Some(p) if p.is_ascii_digit()) && matches!(next, Some(n) if n.is_ascii_digit())
}

/// Vrai si le mot avant le point est une abréviation ou contient un point interne (`e.g.`).
fn is_abbreviation(chars: &[char], sent_start: usize, dot: usize) -> bool {
    let mut wstart = dot;
    while wstart > sent_start && !chars[wstart - 1].is_whitespace() {
        wstart -= 1;
    }
    let word: String = chars[wstart..dot].iter().collect();
    let lower = word.to_lowercase();
    if lower.contains('.') {
        return true; // e.g., i.e., a.c. — point interne
    }
    ABBREVIATIONS.contains(&lower.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_fr_avec_abreviations_et_decimaux() {
        let text = "M. Dupont a raison. La valeur est 3.14, c'est prouvé ! Vraiment ? Oui.";
        let s = split_sentences(text);
        assert_eq!(s.len(), 4, "attendu 4 phrases, reçu {s:?}");
        assert!(s[0].starts_with("M. Dupont"));
        assert!(s[1].contains("3.14"));
    }

    #[test]
    fn split_eg_et_ellipsis() {
        let s = split_sentences("Voir e.g. ce cas. Puis la suite… Fin.");
        assert_eq!(s.len(), 3, "{s:?}");
    }

    #[test]
    fn split_filtre_residus() {
        let s = split_sentences("... ! 42. Phrase ok.");
        assert_eq!(s, vec!["42.", "Phrase ok."]);
    }

    #[test]
    fn html_strip_script_style_entites() {
        let html = "<html><head><style>.a{color:red}</style><script>alert(1)</script></head>\
            <body><!-- commentaire --><h1>Titre &amp; caf&eacute;</h1>\
            <p>La pluie <b>cause</b> des d&#233;g&#226;ts.</p></body></html>";
        let t = strip_html(html);
        assert!(!t.contains("alert"), "{t:?}");
        assert!(!t.contains("color:red"), "{t:?}");
        assert!(!t.contains("commentaire"), "{t:?}");
        assert!(t.contains("Titre & café"), "{t:?}");
        assert!(t.contains("La pluie cause des dégâts."), "{t:?}");
    }

    #[test]
    fn notebook_extrait_markdown_seulement() {
        let nb = r##"{"cells": [
            {"cell_type": "markdown", "source": ["# Titre\n", "Une phrase."]},
            {"cell_type": "code", "source": ["print('x')"]},
            {"cell_type": "markdown", "source": "Seconde phrase."}
        ]}"##;
        let t = extract_notebook(nb, Path::new("demo.ipynb")).expect("notebook valide");
        assert!(t.contains("Une phrase."), "{t:?}");
        assert!(t.contains("Seconde phrase."), "{t:?}");
        assert!(!t.contains("print"), "{t:?}");
    }

    #[test]
    fn notebook_invalide_erreur_typee() {
        assert!(extract_notebook("pas du json", Path::new("x.ipynb")).is_err());
        assert!(extract_notebook(r#"{"cells": 42}"#, Path::new("x.ipynb")).is_err());
    }

    #[test]
    fn detect_format_par_extension() {
        assert_eq!(
            detect_format(Path::new("doc.PDF")).expect("pdf"),
            DocFormat::Pdf
        );
        assert_eq!(
            detect_format(Path::new("page.htm")).expect("htm"),
            DocFormat::Html
        );
        assert!(detect_format(Path::new("archive.zip")).is_err());
        assert!(detect_format(Path::new("sans_extension")).is_err());
    }

    #[test]
    fn script_apres_caractere_multioctet_sans_panique() {
        // U+0130 : to_lowercase() change la longueur d'octets — une ombre
        // minuscule désalignerait les index (panique). Ne doit pas paniquer.
        let html = "<p>İstanbul</p><script>var x = 1;</script><p>Suite.</p>";
        let t = strip_html(html);
        assert!(t.contains("İstanbul"), "{t:?}");
        assert!(!t.contains("var x"), "{t:?}");
        assert!(t.contains("Suite."), "{t:?}");
        // Pire cas : l'ancienne ombre minuscule calculait un index hors limites
        // (panique garantie) — doit produire un résultat exact.
        assert_eq!(strip_html("İ<script>a</script><p>OK.</p>"), "İ\nOK.");
    }

    #[test]
    fn entites_inconnues_inchangees() {
        assert_eq!(decode_entities("a &truc; b"), "a &truc; b");
        assert_eq!(decode_entities("&#x41;&#66;"), "AB");
    }
}
