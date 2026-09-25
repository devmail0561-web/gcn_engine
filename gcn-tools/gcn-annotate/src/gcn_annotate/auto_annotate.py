# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Annotateur de phrases causales - FR/EN.

Ce script lit les phrases extraites et genere les annotations CausalIR
au format GCN-NL pour l'entrainement du modele.
"""

import json
import re
from pathlib import Path


# Types de nœuds valides
NODE_TYPES = ["etat", "action", "transition", "processus", "condition", "entite", "etat_systemique"]

# Types de relations valides
RELATION_TYPES = ["cause", "enable", "prevent", "condition", "concession", "sequence", "motivation", "filter", "opposition", "data_dependency", "control_dependency"]


def analyze_sentence(text: str) -> dict | None:
    """Analyse une phrase et retourne l'annotation CausalIR si elle contient une relation causale."""
    text_lower = text.lower().strip()
    
    # Patterns de relations causales avec leur type (FR + EN)
    causal_patterns = [
        # FR - Cause
        (r'parce que', 'cause'),
        (r'\bcar\b', 'cause'),
        (r'puisque', 'cause'),
        (r'du fait que', 'cause'),
        (r'en raison de', 'cause'),
        (r'à cause de', 'cause'),
        (r'grâce à', 'cause'),
        
        # FR - Conséquence
        (r'\bdonc\b', 'cause'),
        (r'\bainsi\b', 'cause'),
        (r'c\'est pourquoi', 'cause'),
        (r'par conséquent', 'cause'),
        (r'entraîne', 'cause'),
        (r'provoque', 'cause'),
        (r'conduit à', 'cause'),
        (r'mène à', 'cause'),
        (r'engendre', 'cause'),
        (r'détermine', 'cause'),
        (r'rend possible', 'enable'),
        (r'permet', 'enable'),
        (r'favorise', 'enable'),
        
        # FR - Empêche
        (r'empêche', 'prevent'),
        (r'interdit', 'prevent'),
        (r'bloque', 'prevent'),
        
        # FR - Condition
        (r'\bsi\b', 'condition'),
        (r'à condition que', 'condition'),
        
        # FR - Concession
        (r'bien que', 'concession'),
        (r'malgré', 'concession'),
        (r'quoique', 'concession'),
        (r'même si', 'concession'),
        
        # EN - Cause
        (r'\bbecause\b', 'cause'),
        (r'\bsince\b', 'cause'),
        (r'\bas\b', 'cause'),
        (r'\bdue to\b', 'cause'),
        (r'\bthanks to\b', 'cause'),
        (r'\bcaused by\b', 'cause'),
        (r'\bresults? in\b', 'cause'),
        (r'\blead to\b', 'cause'),
        (r'\bcontribute to\b', 'cause'),
        (r'\bbring about\b', 'cause'),
        (r'\baccount for\b', 'cause'),
        
        # EN - Conséquence
        (r'\btherefore\b', 'cause'),
        (r'\bthus\b', 'cause'),
        (r'\bhence\b', 'cause'),
        (r'\bconsequently\b', 'cause'),
        (r'\bso\b', 'cause'),
        (r'\bthis means\b', 'cause'),
        (r'\bthis leads to\b', 'cause'),
        
        # EN - Enable
        (r'\benable\b', 'enable'),
        (r'\ballow\b', 'enable'),
        (r'\bmake it possible\b', 'enable'),
        (r'\bfacilitate\b', 'enable'),
        (r'\bsupport\b', 'enable'),
        (r'\bhelp\b', 'enable'),
        (r'\bimprove\b', 'enable'),
        (r'\benhance\b', 'enable'),
        (r'\bincrease\b', 'enable'),
        (r'\bboost\b', 'enable'),
        
        # EN - Prevent
        (r'\bprevent\b', 'prevent'),
        (r'\bavoid\b', 'prevent'),
        (r'\bblock\b', 'prevent'),
        (r'\bstop\b', 'prevent'),
        (r'\bprohibit\b', 'prevent'),
        (r'\bforbid\b', 'prevent'),
        (r'\bexclude\b', 'prevent'),
        (r'\bignore\b', 'prevent'),
        (r'\bskip\b', 'prevent'),
        (r'\bdrop\b', 'prevent'),
        (r'\bfilter out\b', 'prevent'),
        
        # EN - Condition
        (r'\bif\b', 'condition'),
        (r'\bwhen\b', 'condition'),
        (r'\bwhenever\b', 'condition'),
        (r'\bunless\b', 'condition'),
        (r'\bprovided that\b', 'condition'),
        (r'\bas long as\b', 'condition'),
        
        # EN - Concession
        (r'\balthough\b', 'concession'),
        (r'\beven though\b', 'concession'),
        (r'\bdespite\b', 'concession'),
        (r'\bin spite of\b', 'concession'),
        (r'\bregardless of\b', 'concession'),
        (r'\bnevertheless\b', 'concession'),
        (r'\bhowever\b', 'concession'),
        (r'\bnonetheless\b', 'concession'),
    ]
    
    # Chercher les patterns
    found_relations = []
    for pattern, rel_type in causal_patterns:
        if re.search(pattern, text_lower):
            found_relations.append(rel_type)
    
    if not found_relations:
        return None
    
    # Prendre la premiere relation trouvee
    relation = found_relations[0]
    
    # Extraire les entites (noms et verbes principaux)
    # Methode simplifiee : prendre les mots significatifs
    words = re.findall(r'\b[a-zéèêëàâäùûüôöîïçœæ]{4,}\b', text_lower)
    
    # Filtrer les mots courants
    stop_words = {
        'dans', 'pour', 'avec', 'sur', 'les', 'des', 'une', 'est', 'sont', 'qui', 'que', 'par', 'pas', 'plus', 'tout', 'fait', 'être', 'avoir', 'comme', 'cette', 'ces', 'entre', 'même', 'aussi', 'bien', 'encore', 'très', 'tous', 'toutes', 'autre', 'autres', 'mais', 'donc', 'ainsi', 'donc', 'alors', 'car', 'puis',
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'shall', 'not', 'but', 'or', 'nor', 'yet', 'both', 'either', 'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because', 'as', 'until', 'while', 'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now'
    }
    significant_words = [w for w in words if w not in stop_words][:5]
    
    if len(significant_words) < 2:
        return None
    
    # Localiser chaque mot significatif dans les tokens 1-based
    tokens = text.split()
    T = len(tokens)
    tokens_lower = [t.lower() for t in tokens]
    used_positions: set[int] = set()

    def _find_pos(word: str) -> int:
        for j, tok in enumerate(tokens_lower):
            if word in tok and (j + 1) not in used_positions:
                used_positions.add(j + 1)
                return j + 1
        # Fallback : position distribuée, sans collision
        pos = max(1, min(T, (len(used_positions) + 1) * max(1, T // (len(significant_words[:3]) + 1))))
        while pos in used_positions and pos < T:
            pos += 1
        used_positions.add(pos)
        return pos

    nodes = []
    for i, word in enumerate(significant_words[:3]):
        node_type = "entite" if i == 0 else "processus"
        pos = _find_pos(word)
        nodes.append({
            "id": f"n{i+1:03d}",
            "type": node_type,
            "label": word,
            "token_span": [pos, pos],
        })
    
    # Creer les aretes
    edges = []
    if len(nodes) >= 2:
        edges.append({
            "source": nodes[0]["id"],
            "target": nodes[1]["id"],
            "relation": relation
        })
    
    return {
        "nodes": nodes,
        "edges": edges
    }


def annotate_file(input_path: Path, output_path: Path, max_sentences: int = 5000):
    """Annotate un fichier de phrases et genere le dataset GCN-NL."""
    with open(input_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    
    # Limiter le nombre de phrases
    if len(lines) > max_sentences:
        lines = lines[:max_sentences]
    
    annotated = []
    skipped = 0
    
    for i, line in enumerate(lines):
        result = analyze_sentence(line)
        if result:
            annotated.append({
                "id": f"s{i+1:04d}",
                "text": line,
                "cir": result
            })
        else:
            skipped += 1
    
    # Creer le document GCN-NL
    document = {
        "document": {
            "id": "real_data",
            "lang": "fr",
            "sentences": annotated
        }
    }
    
    # Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    
    print(f"Phrases traitees: {len(lines)}")
    print(f"Phrases annotees: {len(annotated)}")
    print(f"Phrases ignorees: {skipped}")
    print(f"Fichier: {output_path}")
    
    return document


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parents[4]
    input_file = base / "gcn-datasets/raw_test/sentences_raw.txt"
    output_file = base / "gcn-datasets/real/annotated.json"
    annotate_file(input_file, output_file, max_sentences=5000)
