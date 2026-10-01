// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Lattice de tokens sans dictionnaire (frontend FR).
//!
//! Zéro liste de lemmes, zéro YAML. Tout est dérivé de la FORME
//! (suffixes verbaux, longueur, ponctuation) et de la POSITION
//! (ordre relatif aux verbes, découpes virgule). Doctrine :
//! ETUDE_LINGUISTIQUE_NLU §14 (habitudes positionnelles), P2
//! (la position distingue les rôles).
//!
//! Les rôles (`nsubj/obj/mark/advcl/...`) sont des pseudo-dep_rel
//! positionnels : bruités mais réels. Le ML apprend à les pondérer ;
//! le parseur UD (couche B) les remplacera par des dep_rel authentiques.

use gcn_ir::{Lattice, LatticeDep, LatticePos, LatticeToken};

// ---------------------------------------------------------------------------
// Tokenisation locale (sans liste de clitiques)
// ---------------------------------------------------------------------------

fn tokenize_lattice(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split_whitespace() {
        // Ponctuation de tête conservée comme token séparé si elle porte du sens
        let mut core = raw;
        while let Some(first) = core.chars().next() {
            if "«»\"'".contains(first) {
                core = &core[first.len_utf8()..];
            } else {
                break;
            }
        }
        // Ponctuation finale : ., ,, ?, ! conservés (le ?/! porte le type de phrase)
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
            // Split d'inversion hyphenée par la forme (aucune liste) :
            // "passe-t-il" → ["passe", "t", "il"] si dernier segment ≤ 4
            // et premier segment > 2. Les tokens d'une lettre sont inertes
            // (jamais det/mark/advmod).
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
            // Split générique sur l'apostrophe (aucune liste de préfixes) :
            // "n'est" → ["n'", "est"], quel que soit le préfixe.
            if let Some(pos) = core.find(['\'', '\u{2019}']) {
                let (pre, rest) = core.split_at(pos);
                let apos_len = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                let after = &rest[apos_len..];
                if !pre.is_empty() {
                    out.push(format!("{}'", pre));
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

// ---------------------------------------------------------------------------
// Morphologie de forme (suffixes uniquement, aucune liste)
// ---------------------------------------------------------------------------

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn is_imparfait(lower: &str) -> bool {
    lower.ends_with("ait") || lower.ends_with("aient") || lower.ends_with("ais")
}

fn is_infinitive(lower: &str) -> bool {
    lower.ends_with("er")
        || lower.ends_with("ir")
        || lower.ends_with("re")
        || lower.ends_with("oir")
}

fn is_participle_er(lower: &str) -> bool {
    lower.ends_with('é') || lower.ends_with("ée") || lower.ends_with("és") || lower.ends_with("ées")
}

/// Verbe par la forme : imparfait, participe -é, infinitif, -ent (tige ne
/// finissant PAS par 'm' — exclut médicament/traitement/récemment),
/// -ez. Le -e seul et -ons ne suffisent PAS (noms en -e, "maisons").
/// Participes 3e groupe -it/-ut/-is dès 3 lettres ("dit" inclus ;
/// bruit assumé : nuit/fruit).
fn looks_like_verb(lower: &str) -> bool {
    if lower.len() <= 2 {
        return false;
    }
    if is_imparfait(lower) || is_participle_er(lower) || is_infinitive(lower) {
        return true;
    }
    if lower.ends_with("ent") && lower.len() > 4 {
        let stem = &lower[..lower.len() - 3];
        if !stem.ends_with('m') && stem.len() > 2 {
            return true;
        }
    }
    if lower.ends_with("ez") && lower.len() > 3 {
        return true;
    }
    if (lower.ends_with("it") || lower.ends_with("ut") || lower.ends_with("is")) && lower.len() >= 3
    {
        return true;
    }
    false
}

/// Passe 2 : le -e final (char>4) devient VERBE si contexte sujet à gauche :
/// token de contenu précédent = Noun ("chaleur dilate"), ou court en tête
/// de clause ("Il mange", "se" de "Que se passe" — chaîne clitique initiale).
/// "la douleur" survit (court non en tête). Bruit assumé : "La cause est X".
fn upgrade_e_verbs(work: &mut [WorkToken]) {
    let n = work.len();
    let mut clause_start = 0usize;
    for i in 0..n {
        if work[i].pos == LatticePos::Punct && work[i].lower == "," {
            clause_start = i + 1;
        }
        if work[i].pos != LatticePos::Noun {
            continue;
        }
        let lower = work[i].lower.clone();
        if !(lower.ends_with('e') && char_len(&lower) > 4) {
            continue;
        }
        let prev = (0..i).rev().find(|&k| {
            work[k].pos == LatticePos::Noun
                || work[k].pos == LatticePos::Verb
                || work[k].pos == LatticePos::Other
        });
        let ok = match prev {
            Some(k) if work[k].pos == LatticePos::Noun => true,
            Some(k)
                if work[k].pos == LatticePos::Other
                    && char_len(&work[k].lower) > 1
                    && char_len(&work[k].lower) <= 3
                    && k <= clause_start + 1 =>
            {
                true
            }
            _ => false,
        };
        if ok {
            work[i].pos = LatticePos::Verb;
            // Lemme = surface (déterministe ; "cause" nom/verbe partagés).
            work[i].lemma = lower;
        }
    }
}

/// Lemmatisation haute-précision sans liste : participes -é et imparfaits
/// réguliers → stem + er ; infinitifs tels quels ; le reste = forme.
/// (3e groupe irrégulier gardé en surface : l'embedding OOV rend un vecteur
/// nul sûr, jamais une fausse règle.)
fn lemmatize_form(lower: &str) -> String {
    if lower.ends_with("ées") || lower.ends_with("és") {
        let end = if lower.ends_with("ées") { 4 } else { 3 };
        return format!("{}er", &lower[..lower.len() - end]);
    }
    if let Some(s) = lower.strip_suffix("ée") {
        return format!("{}er", s);
    }
    if let Some(s) = lower.strip_suffix('é') {
        return format!("{}er", s);
    }
    if let Some(s) = lower.strip_suffix("aient") {
        return format!("{}er", s);
    }
    if let Some(s) = lower.strip_suffix("ait") {
        return format!("{}er", s);
    }
    if let Some(s) = lower.strip_suffix("ais") {
        return format!("{}er", s);
    }
    lower.to_string()
}

fn is_punct_form(s: &str) -> bool {
    matches!(s, "." | "," | "?" | "!" | ";" | ":")
}

// ---------------------------------------------------------------------------
// Lattice
// ---------------------------------------------------------------------------

struct WorkToken {
    form: String,
    lower: String,
    pos: LatticePos,
    lemma: String,
    flags: Vec<String>,
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
                    flags: vec![],
                }
            } else if looks_like_verb(&lower) {
                let mut flags = Vec::new();
                if is_imparfait(&lower) {
                    flags.push("imparfait".to_string());
                }
                if is_infinitive(&lower) {
                    flags.push("infinitive".to_string());
                }
                if is_participle_er(&lower) {
                    flags.push("participle".to_string());
                }
                WorkToken {
                    form: f.clone(),
                    lower: lower.clone(),
                    pos: LatticePos::Verb,
                    lemma: lemmatize_form(&lower),
                    flags,
                }
            } else if char_len(&lower) > 3 {
                // Mot de contenu par défaut (pas de dictionnaire de noms).
                WorkToken {
                    form: f.clone(),
                    lower: lower.clone(),
                    pos: LatticePos::Noun,
                    lemma: lower,
                    flags: vec![],
                }
            } else {
                // Token court : déterminant / connecteur / modificateur —
                // tranché par position ci-dessous, jamais par liste.
                WorkToken {
                    form: f.clone(),
                    lower: lower.clone(),
                    pos: LatticePos::Other,
                    lemma: lower,
                    flags: vec![],
                }
            }
        })
        .collect()
}

/// Construit le lattice : POS morphologique + pseudo-dep_rel positionnels.
///
/// Règles de position (indices 0-based ici, convertis en 1-based à l'émission) :
/// - verbe subordonné = premier VERB après un marqueur candidat ;
/// - marqueur candidat = token court (Other) non-det en position {0,
///   virgule+1, verbe+1}, NON suivi directement d'un verbe ;
/// - verbe principal = dernier VERB non subordonné (repli : dernier VERB,
///   puis premier Noun) ;
/// - autres VERB sans marqueur = advcl rattachés au principal (juxtaposition) ;
/// - premier Noun avant le principal = nsubj ; premier Noun après = obj ;
/// - Other court devant un Noun pré-verbal (verbe plus loin) = det ;
/// - Other court juste devant le principal (sinon) = advmod.
///
/// Limites assumées (sans liste, indécidables par position seule) :
/// - chaînes multi-courtes ("que si...") : seul le premier est marqué ;
/// - connecteurs à mot long ("parce", "lorsque") : invisibles ;
/// - "Le/Il" initiaux exclus du marquage par la garde non-det + adjacence.
pub fn parse_lattice(text: &str) -> Lattice {
    let forms = tokenize_lattice(text.trim());
    let mut work = tag_forms(&forms);
    upgrade_e_verbs(&mut work);
    let n = work.len();

    // Découpes de clauses sur la virgule (spans 1-based inclusifs).
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

    // Tokens d'une lettre ("t" d'inversion, "y", "a") : inertes partout.
    let is_short_other = |i: usize| -> bool {
        work[i].pos == LatticePos::Other
            && char_len(&work[i].lower) > 1
            && char_len(&work[i].lower) <= 3
    };
    // det : court devant un Noun pré-verbal avec verbe plus loin.
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
    // Contenu précédent : remonte en sautant les 1-lettre, stop à la virgule.
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
    // Éligibilité d'un marqueur : court, non-det, non adjacent à un verbe,
    // précédent non nominal ("n'", "est", "ont" exclus), verbe collé admis
    // ("dit que" : offset 1), verbe lointain refusé ("il" après "passe-t-").
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

    // Marqueurs : positions candidates {0, virgule+1, verbe+1}.
    // Un candidat marque s'il est court, non-det, et NON suivi d'un verbe
    // (garde anti "Il/Le" initiaux et anti auxiliaires).
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
            // Le verbe subordonné = premier VERB après le marqueur.
            if let Some(&sv) = verb_at.iter().find(|&&v| v > c) {
                // Marqueur initial (position 0) : rejeté si un autre marqueur
                // éligible existe plus loin ("Que ... si ..." → "si" gagne).
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
        // Second passage : marqueurs médians ("si" après inversion).
        // La position 0 ne repasse jamais (déjà tranchée ci-dessus).
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

    // Verbe principal : dernier VERB non subordonné ; replis documentés.
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

        // Verbes subordonnés (marqueur) ou juxtaposés (advcl sans marqueur).
        for (i, w) in work.iter().enumerate() {
            if i != mv && w.pos == LatticePos::Verb {
                dep[i] = LatticeDep::Advcl;
                head[i] = mv_head;
            }
        }
        // Marqueur unique v1.
        if let Some(m) = mark_at
            && let Some(&sv) = subordinate.iter().next()
        {
            dep[m] = LatticeDep::Mark;
            head[m] = (sv + 1) as i32;
        }
        // Premier Noun avant / après le principal.
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
        // det puis advmod pré-verbal pour les Other restants.
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
        // Ponctuation : non rattachée.
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
            flags: w.flags,
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
    fn lattice_simple_roles_by_position() {
        let lat = parse_lattice("Le médicament réduit la douleur.");
        let by_form = |f: &str| lat.tokens.iter().find(|t| t.form == f).unwrap().clone();
        assert_eq!(by_form("réduit").pos, LatticePos::Verb);
        assert_eq!(by_form("réduit").dep_rel, LatticeDep::Root);
        assert_eq!(by_form("médicament").dep_rel, LatticeDep::Nsubj);
        assert_eq!(by_form("douleur").dep_rel, LatticeDep::Obj);
        // Aucun lemme comparé à une liste : le verdict vient des positions.
        assert!(lat.tokens.iter().all(|t| !t.lemma.is_empty()));
    }

    #[test]
    fn lattice_subordinate_mark_and_advcl() {
        // "si" seul token court entre les verbes → marqueur non ambigu.
        let lat = parse_lattice("Il venait si tu venais.");
        let si = lat.tokens.iter().find(|t| t.form == "si").unwrap();
        assert_eq!(si.dep_rel, LatticeDep::Mark);
        let sub = lat.tokens.iter().find(|t| t.form == "venais").unwrap();
        assert_eq!(sub.dep_rel, LatticeDep::Advcl);
        assert_eq!(
            sub.dep_head,
            lat.tokens
                .iter()
                .find(|t| t.form == "venait")
                .unwrap()
                .index as i32
        );
        let main = lat
            .tokens
            .iter()
            .find(|t| t.dep_rel == LatticeDep::Root)
            .unwrap();
        assert_eq!(main.form, "venait");
    }

    #[test]
    fn lattice_interrogative_inversion_and_true_mark() {
        // "passe-t-il" : inversion hyphenée → verbe ; "Que" initial rejeté
        // (candidat "si" plus loin) ; "si" = marqueur, "baisse" subordonnée.
        let lat = parse_lattice("Que se passe-t-il si la demande baisse ?");
        let passe = lat.tokens.iter().find(|t| t.form == "passe").unwrap();
        assert_eq!(passe.pos, LatticePos::Verb);
        let si = lat.tokens.iter().find(|t| t.form == "si").unwrap();
        assert_eq!(si.dep_rel, LatticeDep::Mark);
        let baisse = lat.tokens.iter().find(|t| t.form == "baisse").unwrap();
        assert_eq!(baisse.dep_rel, LatticeDep::Advcl);
        let main = lat
            .tokens
            .iter()
            .find(|t| t.dep_rel == LatticeDep::Root)
            .unwrap();
        assert_eq!(main.form, "passe");
        // Le "?" survit (type interrogatif lisible en aval).
        assert!(lat.tokens.iter().any(|t| t.form == "?"));
    }

    #[test]
    fn lattice_never_matches_lemma_lists() {
        // Garde : aucune table de chaînes (le seul set est HashSet<usize>
        // de positions ; les ".contains" restants portent sur des chars
        // de ponctuation, pas des mots).
        let src = include_str!("lattice.rs");
        // Motifs assemblés par concaténation : le garde ne doit pas
        // contenir lui-même les fragments interdits en clair.
        let lt = "<";
        for b in [
            format!("HashSet{lt}String"),
            format!("HashMap{lt}String"),
            format!("BTreeMap{lt}String"),
            format!("BTreeSet{lt}String"),
            ["&[", "&str]"].concat(),
            ["phf:", ":"].concat(),
        ] {
            assert!(!src.contains(&b), "table de mots interdite: {}", b);
        }
    }
}
