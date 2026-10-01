// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Dictionary-free token lattice (EN frontend).
//!
//! Zero lemma lists, zero YAML. Everything derives from FORM (verbal
//! suffixes, length, punctuation) and POSITION (order relative to verbs,
//! comma splits). Same doctrine as the FR lattice:
//! ETUDE_LINGUISTIQUE_NLU §14, P2.

use gcn_ir::{Lattice, LatticeDep, LatticePos, LatticeToken};

fn tokenize_lattice(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split_whitespace() {
        let mut core = raw;
        while let Some(first) = core.chars().next() {
            if "«»\"'".contains(first) {
                core = &core[first.len_utf8()..];
            } else {
                break;
            }
        }
        let mut trailing: Vec<char> = Vec::new();
        while let Some(last) = core.chars().last() {
            if ".,?!".contains(last) {
                trailing.push(last);
                core = &core[..core.len() - last.len_utf8()];
            } else {
                break;
            }
        }
        if !core.is_empty() {
            // Hyphen-inversion split by form (no list): last segment ≤ 4
            // and first segment > 2. Single-letter tokens are inert
            // (never det/mark/advmod).
            if core.contains('-') {
                let parts: Vec<&str> = core.split('-').collect();
                if parts.len() >= 2
                    && parts.first().map(|s| s.chars().count()).unwrap_or(0) > 2
                    && parts.last().map(|s| s.chars().count()).unwrap_or(99) <= 4
                {
                    for part in parts.iter() {
                        if !part.is_empty() {
                            out.push(part.to_string());
                        }
                    }
                    for p in trailing.iter().rev() {
                        out.push(p.to_string());
                    }
                    continue;
                }
            }
            // Generic contraction split on ASCII apostrophe, no list:
            // "doesn't" → ["doesn", "t"] by form (lemma rule fixes n't below).
            if let Some(pos) = core.find('\'') {
                let (pre, rest) = core.split_at(pos);
                let after = &rest[1..];
                if !pre.is_empty() {
                    out.push(pre.to_string());
                }
                if !after.is_empty() {
                    out.push(after.to_string());
                }
            } else {
                out.push(core.to_string());
            }
        }
        for p in trailing.iter().rev() {
            out.push(p.to_string());
        }
    }
    out
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// English verb by form: -ing, -ed (len>3), -ize/-ise/-ify/-ate.
/// Bare -s is NOT enough (plural nouns). No lists.
fn looks_like_verb(lower: &str) -> bool {
    lower.ends_with("ing")
        || (lower.ends_with("ed") && lower.len() > 3)
        || lower.ends_with("ize")
        || lower.ends_with("ise")
        || lower.ends_with("ify")
        || lower.ends_with("ate")
}

/// High-precision form lemmatization, no lists: n't contraction,
/// -ied→-y, -ies→-y, double-consonant -ed/-ing. Rest = surface form
/// (train vocab covers it; OOV embedding is a safe zero vector).
fn lemmatize_form(lower: &str) -> String {
    if let Some(s) = lower.strip_suffix("n't") {
        return s.to_string();
    }
    if let Some(s) = lower.strip_suffix("ied") {
        return format!("{}y", s);
    }
    if let Some(s) = lower.strip_suffix("ies") {
        return format!("{}y", s);
    }
    // 3rd person singular -s ("reduces" → "reduce"). Applied only to
    // tokens already promoted to Verb by position (promote_s_verbs) —
    // never as a detection signal (plural nouns).
    if let Some(s) = lower.strip_suffix('s')
        && s.len() > 2
    {
        return s.to_string();
    }
    if let Some(s) = lower.strip_suffix("ed")
        && s.len() > 2
    {
        let c: Vec<char> = s.chars().collect();
        let n = c.len();
        if n >= 2 && c[n - 1] == c[n - 2] {
            return c[..n - 1].iter().collect();
        }
        return s.to_string();
    }
    if let Some(s) = lower.strip_suffix("ing")
        && s.len() > 2
    {
        let c: Vec<char> = s.chars().collect();
        let n = c.len();
        if n >= 2 && c[n - 1] == c[n - 2] {
            return c[..n - 1].iter().collect();
        }
        return s.to_string();
    }
    lower.to_string()
}

fn is_punct_form(s: &str) -> bool {
    matches!(s, "." | "," | "?" | "!" | ";" | ":")
}

struct WorkToken {
    form: String,
    lower: String,
    pos: LatticePos,
    lemma: String,
}

/// Passe 2a : gérondif sujet ("Flooding is causing") — un -ing en tête de
/// clause suivi d'un autre verbe plus loin = Noun, pas verbe.
fn demote_subject_gerunds(work: &mut [WorkToken]) {
    let n = work.len();
    for i in 0..n {
        if work[i].pos != LatticePos::Verb || !work[i].lower.ends_with("ing") {
            continue;
        }
        let at_head =
            i == 0 || (i > 0 && work[i - 1].pos == LatticePos::Punct && work[i - 1].lower == ",");
        if at_head && work[i + 1..].iter().any(|w| w.pos == LatticePos::Verb) {
            work[i].pos = LatticePos::Noun;
            work[i].lemma = work[i].lower.clone();
        }
    }
}

fn is_content(pos: LatticePos) -> bool {
    matches!(pos, LatticePos::Noun | LatticePos::Verb | LatticePos::Other)
}

/// Passe 2b : présent en -s ("drug reduces pain") — le -s exige un Noun
/// à gauche STRICT (jamais de pronom : "The dogs bark" ne doit pas
/// inverser les rôles). Parmi les qualifiés avec Noun à droite, le
/// DERNIER devient verbe ("always" survit comme Noun).
/// Bruit assumé : sujets pronominaux ("He causes") → racine nominale.
fn promote_s_verbs(work: &mut [WorkToken]) {
    let n = work.len();
    let mut qual: Vec<usize> = Vec::new();
    for i in 0..n {
        if work[i].pos != LatticePos::Noun {
            continue;
        }
        let lower = work[i].lower.clone();
        if !(lower.ends_with('s') && lower.len() > 3 && !lower.ends_with("ss")) {
            continue;
        }
        let prev = (0..i).rev().find(|&k| is_content(work[k].pos));
        if !matches!(prev, Some(k) if work[k].pos == LatticePos::Noun) {
            continue;
        }
        if !work[i + 1..].iter().any(|w| w.pos == LatticePos::Noun) {
            continue;
        }
        qual.push(i);
    }
    if let Some(&v) = qual.last() {
        work[v].pos = LatticePos::Verb;
        work[v].lemma = lemmatize_form(&work[v].lower);
    }
}

fn tag_forms(forms: &[String]) -> Vec<WorkToken> {
    forms
        .iter()
        .map(|f| {
            let lower = f.to_lowercase();
            if is_punct_form(&lower) {
                WorkToken {
                    form: f.clone(),
                    lower: lower.clone(),
                    pos: LatticePos::Punct,
                    lemma: lower,
                }
            } else if looks_like_verb(&lower) {
                WorkToken {
                    form: f.clone(),
                    lower: lower.clone(),
                    pos: LatticePos::Verb,
                    lemma: lemmatize_form(&lower),
                }
            } else if char_len(&lower) > 3 {
                WorkToken {
                    form: f.clone(),
                    lower: lower.clone(),
                    pos: LatticePos::Noun,
                    lemma: lower,
                }
            } else {
                WorkToken {
                    form: f.clone(),
                    lower: lower.clone(),
                    pos: LatticePos::Other,
                    lemma: lower,
                }
            }
        })
        .collect()
}

/// Same positional role rules as the FR lattice (see fr/lattice.rs).
pub fn parse_lattice(text: &str) -> Lattice {
    let forms = tokenize_lattice(text.trim());
    let mut work = tag_forms(&forms);
    demote_subject_gerunds(&mut work);
    promote_s_verbs(&mut work);
    let n = work.len();

    let mut clauses: Vec<(u32, u32)> = Vec::new();
    let mut clause_of = vec![0u32; n];
    if n > 0 {
        let mut start = 0usize;
        let mut cid = 0u32;
        for (i, w) in work.iter().enumerate() {
            if w.pos == LatticePos::Punct && w.lower == "," {
                clauses.push(((start + 1) as u32, (i + 1) as u32));
                for slot in clause_of.iter_mut().take(i + 1).skip(start) {
                    *slot = cid;
                }
                cid += 1;
                start = i + 1;
            }
        }
        clauses.push(((start + 1) as u32, (n) as u32));
        for slot in clause_of
            .iter_mut()
            .skip(start)
            .take(n.saturating_sub(start))
        {
            *slot = cid;
        }
    }

    // Single-letter tokens ("t", "s", "a") : inert everywhere.
    let is_short_other = |i: usize| -> bool {
        work[i].pos == LatticePos::Other
            && char_len(&work[i].lower) > 1
            && char_len(&work[i].lower) <= 3
    };
    let is_det = |i: usize| -> bool {
        if !is_short_other(i) {
            return false;
        }
        let next_content = work[i + 1..].iter().position(|v| {
            v.pos == LatticePos::Noun || v.pos == LatticePos::Verb || v.pos == LatticePos::Other
        });
        match next_content {
            Some(off) => {
                let j = i + 1 + off;
                work[j].pos == LatticePos::Noun
                    && work[j + 1..].iter().any(|v| v.pos == LatticePos::Verb)
            }
            None => false,
        }
    };
    // Previous content: walk left skipping 1-letter tokens, stop at comma.
    let prev_content = |i: usize| -> Option<usize> {
        let mut k = i;
        while k > 0 {
            k -= 1;
            let w = &work[k];
            if w.pos == LatticePos::Punct {
                if w.lower == "," {
                    return None;
                }
                continue;
            }
            if char_len(&w.lower) <= 1 {
                continue;
            }
            return Some(k);
        }
        None
    };
    let is_content_pos = |p: LatticePos| -> bool {
        matches!(p, LatticePos::Noun | LatticePos::Verb | LatticePos::Other)
    };
    // Marker eligibility: short, non-det, not verb-adjacent,
    // non-nominal previous, glued verb allowed ("say that": offset 1),
    // distant verb refused.
    let eligible = |c: usize| -> bool {
        if c >= n || !is_short_other(c) || is_det(c) {
            return false;
        }
        if c + 1 < n
            && let Some(off) = work[c + 1..].iter().position(|v| is_content_pos(v.pos))
            && work[c + 1 + off].pos == LatticePos::Verb
        {
            return false;
        }
        if let Some(p) = prev_content(c) {
            if work[p].pos == LatticePos::Noun {
                return false;
            }
            if work[p].pos == LatticePos::Verb && c - p > 1 {
                return false;
            }
        }
        true
    };
    let verb_at: Vec<usize> = work
        .iter()
        .enumerate()
        .filter(|(_, w)| w.pos == LatticePos::Verb)
        .map(|(i, _)| i)
        .collect();

    let mut mark_at: Option<usize> = None;
    let mut subordinate: std::collections::HashSet<usize> = std::collections::HashSet::new();
    {
        // Ordre : virgule+1 et verbe+1 AVANT la position 0 — un marqueur
        // attaché ("si" après verbe) prime sur un initial ("Que" interrogatif).
        let mut candidates: Vec<usize> = Vec::new();
        for (i, w) in work.iter().enumerate() {
            if w.pos == LatticePos::Punct && w.lower == "," && i + 1 < n {
                candidates.push(i + 1);
            }
        }
        for &v in &verb_at {
            if v + 1 < n {
                candidates.push(v + 1);
            }
        }
        if n > 0 {
            candidates.push(0);
        }
        for &c in candidates.iter() {
            if !eligible(c) {
                continue;
            }
            if let Some(&sv) = verb_at.iter().find(|&&v| v > c) {
                // Initial marker (position 0): rejected if another eligible
                // marker sits further on.
                if c == 0
                    && (c + 1..n).any(|k| k != c && eligible(k) && verb_at.iter().any(|&v| v > k))
                {
                    continue;
                }
                mark_at = Some(c);
                subordinate.insert(sv);
                break;
            }
        }
        // Second pass: mid-clause markers. Position 0 never re-enters.
        if mark_at.is_none() {
            for c in 1..n {
                if candidates.contains(&c) || !eligible(c) {
                    continue;
                }
                if let Some(&sv) = verb_at.iter().find(|&&v| v > c) {
                    mark_at = Some(c);
                    subordinate.insert(sv);
                    break;
                }
            }
        }
    }

    let main_verb: Option<usize> = verb_at
        .iter()
        .rev()
        .find(|&&v| !subordinate.contains(&v))
        .copied()
        .or_else(|| verb_at.last().copied())
        .or_else(|| work.iter().position(|w| w.pos == LatticePos::Noun));

    let mut dep = vec![LatticeDep::Other; n];
    let mut head: Vec<i32> = vec![-1; n];

    if let Some(mv) = main_verb {
        dep[mv] = LatticeDep::Root;
        head[mv] = 0;
        let mv_head = (mv + 1) as i32;

        for (i, w) in work.iter().enumerate() {
            if i != mv && w.pos == LatticePos::Verb {
                dep[i] = LatticeDep::Advcl;
                head[i] = mv_head;
            }
        }
        if let Some(m) = mark_at
            && let Some(&sv) = subordinate.iter().next()
        {
            dep[m] = LatticeDep::Mark;
            head[m] = (sv + 1) as i32;
        }
        if let Some(s) = work[..mv].iter().position(|w| w.pos == LatticePos::Noun) {
            dep[s] = LatticeDep::Nsubj;
            head[s] = mv_head;
        }
        if let Some(o) = work[mv + 1..]
            .iter()
            .position(|w| w.pos == LatticePos::Noun)
        {
            let o = mv + 1 + o;
            dep[o] = LatticeDep::Obj;
            head[o] = mv_head;
        }
        for (i, w) in work.iter().enumerate() {
            if w.pos != LatticePos::Other || dep[i] != LatticeDep::Other {
                continue;
            }
            // det : tête = le Noun déterminé (is_det vérifie le verbe aval).
            let det_head: Option<usize> = work[i + 1..]
                .iter()
                .position(|v| is_content_pos(v.pos))
                .map(|off| i + 1 + off)
                .filter(|&j| work[j].pos == LatticePos::Noun && is_det(i));
            if let Some(j) = det_head {
                dep[i] = LatticeDep::Det;
                head[i] = (j + 1) as i32;
                continue;
            }
            if char_len(&w.lower) > 1
                && (i + 1 == mv || (i + 1 < n && dep[i + 1] == LatticeDep::Root))
            {
                dep[i] = LatticeDep::Advmod;
                head[i] = mv_head;
            }
        }
        for (i, w) in work.iter().enumerate() {
            if w.pos == LatticePos::Punct {
                dep[i] = LatticeDep::Punct;
                head[i] = -1;
            }
        }
    }

    let tokens = work
        .into_iter()
        .enumerate()
        .map(|(i, w)| LatticeToken {
            index: (i + 1) as u32,
            pos: w.pos,
            dep_rel: dep[i],
            dep_head: head[i],
            lemma: w.lemma,
            clause: clause_of[i],
            flags: vec![],
            form: w.form,
        })
        .collect();

    Lattice {
        source_text: text.trim().to_string(),
        tokens,
        clauses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_en_simple_roles_by_position() {
        // "reduces" : -s avec Noun à gauche (drug) et à droite (pain) → verbe.
        let lat = parse_lattice("The drug reduces pain.");
        let reduces = lat.tokens.iter().find(|t| t.form == "reduces").unwrap();
        assert_eq!(reduces.pos, LatticePos::Verb);
        assert_eq!(reduces.dep_rel, LatticeDep::Root);
        assert_eq!(reduces.lemma, "reduce");
        let drug = lat.tokens.iter().find(|t| t.form == "drug").unwrap();
        assert_eq!(drug.dep_rel, LatticeDep::Nsubj);
        let pain = lat.tokens.iter().find(|t| t.form == "pain").unwrap();
        assert_eq!(pain.dep_rel, LatticeDep::Obj);
    }

    #[test]
    fn lattice_en_ing_verb_detected() {
        let lat = parse_lattice("Flooding is causing damage.");
        let causing = lat.tokens.iter().find(|t| t.form == "causing").unwrap();
        assert_eq!(causing.pos, LatticePos::Verb);
        assert_eq!(causing.lemma, "caus");
    }

    #[test]
    fn lattice_never_matches_lemma_lists() {
        let src = include_str!("lattice.rs");
        let lt = "<";
        for b in [
            format!("HashSet{lt}String"),
            format!("HashMap{lt}String"),
            format!("BTreeMap{lt}String"),
            format!("BTreeSet{lt}String"),
            ["&[", "&str]"].concat(),
            ["phf:", ":"].concat(),
        ] {
            assert!(!src.contains(&b), "forbidden word table: {}", b);
        }
    }
}
