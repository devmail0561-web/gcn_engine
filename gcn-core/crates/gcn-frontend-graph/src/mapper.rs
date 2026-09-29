// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Correspondances STIX → CIR, compilées dans le moteur (SANS YAML).
//!
//! Règle moteur : le savoir est compilé, jamais chargé de configs.
//! Les types STIX 2.x sont une spec externe stable — même statut que
//! `gcn-frontend-fr/rules.rs`. Les tables mappent DIRECTEMENT vers les
//! enums (les conversions canoniques `snake_case → enum` vivent dans
//! `gcn_ir::RelationType::from_name` / `NodeType::from_name`, source unique).

use gcn_ir::{NodeType, RelationType};

/// Type d'objet STIX (SDO/SCO) → `NodeType`.
///
/// - acteurs et infrastructures → `Entite` ;
/// - comportements et actions (attaque, malware, campagne, contre-mesure) → `Processus` ;
/// - vulnérabilité → `Condition` (état qui permet l'exploitation) ;
/// - artefacts de connaissance (indicateur, rapport...) → `Concept` ;
/// - observations (données observées, sightings) → `Evenement` ;
/// - objets observables cyber (SCO : file, url, ipv4-addr...) → `Entite` ;
/// - tout autre type inconnu → `Concept` (jamais d'échec sur un STIX futur).
pub fn stix_object_type_to_node_type(stix_type: &str) -> NodeType {
    match stix_type {
        // Acteurs — entités
        "threat-actor" | "intrusion-set" | "identity" | "infrastructure" | "location" => {
            NodeType::Entite
        }
        // Comportements et actions — processus
        "attack-pattern" | "malware" | "tool" | "campaign" | "course-of-action" => {
            NodeType::Processus
        }
        // État permissif — condition
        "vulnerability" => NodeType::Condition,
        // Artefacts de connaissance — concepts
        "indicator" | "report" | "note" | "opinion" | "grouping" | "malware-analysis" => {
            NodeType::Concept
        }
        // Observations — événements
        "observed-data" | "sighting" => NodeType::Evenement,
        // Objets observables cyber (SCO STIX 2.1, liste stable) — entités
        "artifact"
        | "autonomous-system"
        | "directory"
        | "domain-name"
        | "email-addr"
        | "email-message"
        | "file"
        | "ipv4-addr"
        | "ipv6-addr"
        | "mac-addr"
        | "mutex"
        | "network-traffic"
        | "process"
        | "software"
        | "url"
        | "user-account"
        | "windows-registry-key"
        | "x509-certificate" => NodeType::Entite,
        // Inconnu (extensions STIX futures) — concept, jamais d'échec
        _ => NodeType::Concept,
    }
}

/// `relationship_type` STIX → `RelationType`.
///
/// Seuls les types à lecture causale directe sont mappés :
/// - `uses` → `Enable` (exemple canonique P2.1 ETUDE) ;
/// - `targets`, `attributed-to`, `derived-from`, `originates-from` → `Cause` ;
/// - `mitigates`, `remediates` → `Prevent` ;
/// - `indicates`, `detects` → `Enable` (l'indicateur/la détection permet la réponse) ;
/// - `duplicate-of`, `variant-of` → `Analogy`.
///
/// Les types locatifs/vagues (`related-to`, `located-at`, ...) retournent
/// `None` → relation ignorée mais comptée (`GraphParseReport`).
pub fn stix_relationship_to_relation_type(relationship_type: &str) -> Option<RelationType> {
    match relationship_type {
        "uses" | "indicates" | "detects" => Some(RelationType::Enable),
        "targets" | "attributed-to" | "derived-from" | "originates-from" => {
            Some(RelationType::Cause)
        }
        "mitigates" | "remediates" => Some(RelationType::Prevent),
        "duplicate-of" | "variant-of" => Some(RelationType::Analogy),
        _ => None,
    }
}

/// Vrai si ce `relationship_type` STIX produit une arête causale.
pub fn is_supported_relationship(relationship_type: &str) -> bool {
    stix_relationship_to_relation_type(relationship_type).is_some()
}
