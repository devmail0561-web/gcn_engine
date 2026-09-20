# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Métriques d'évaluation CGNP — NumPy pur, framework-agnostique.

Le data scientist appelle ces fonctions après chaque epoch pour
mesurer la qualité des prédictions. Aucune dépendance à PyTorch/sklearn.
"""
from __future__ import annotations
import warnings
import numpy as np
from ..constants import NODE_TYPES, RELATION_TYPES


# ---------------------------------------------------------------------------
# Types de base
# ---------------------------------------------------------------------------

Predictions = list[str]   # valeurs snake_case ex: ["action", "processus"]
GoldLabels = list[str]


# ---------------------------------------------------------------------------
# Métriques nœuds
# ---------------------------------------------------------------------------

def node_accuracy(pred: Predictions, gold: GoldLabels) -> float:
    """Exactitude globale sur la prédiction des NodeType."""
    if not gold:
        return 0.0
    if len(pred) != len(gold):
        raise ValueError(
            f"node_accuracy: pred ({len(pred)} éléments) et gold ({len(gold)} éléments) "
            f"doivent avoir la même longueur."
        )
    correct = sum(p == g for p, g in zip(pred, gold))
    return correct / len(gold)


def node_f1_per_class(
    pred: Predictions, gold: GoldLabels
) -> dict[str, dict[str, float]]:
    """
    F1, précision, rappel par NodeType.

    Retourne :
      {"action": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 12}, …}
    """
    return _f1_per_class(pred, gold, NODE_TYPES)


def node_macro_f1(pred: Predictions, gold: GoldLabels) -> float:
    """F1 macro-moyenné sur tous les NodeType présents dans gold."""
    per_class = node_f1_per_class(pred, gold)
    scores = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Métriques arêtes
# ---------------------------------------------------------------------------

def edge_accuracy(pred: Predictions, gold: GoldLabels) -> float:
    """Exactitude globale sur la prédiction des RelationType."""
    if not gold:
        return 0.0
    if len(pred) != len(gold):
        raise ValueError(
            f"edge_accuracy: pred ({len(pred)} éléments) et gold ({len(gold)} éléments) "
            f"doivent avoir la même longueur."
        )
    correct = sum(p == g for p, g in zip(pred, gold))
    return correct / len(gold)


def edge_f1_per_class(
    pred: Predictions, gold: GoldLabels
) -> dict[str, dict[str, float]]:
    """F1, précision, rappel par RelationType."""
    return _f1_per_class(pred, gold, RELATION_TYPES)


def edge_macro_f1(pred: Predictions, gold: GoldLabels) -> float:
    per_class = edge_f1_per_class(pred, gold)
    scores = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Similarité de graphe causal
# ---------------------------------------------------------------------------

def causal_graph_similarity(pred_ir: dict, gold_ir: dict) -> dict[str, float]:
    """
    Mesure la similarité entre deux CausalIR (dicts JSON).

    Retourne :
      {
        "node_count_ratio": float,      # |pred_nodes| / |gold_nodes|
        "node_type_accuracy": float,    # % de nœuds avec le bon type (par position)
        "edge_count_ratio": float,
        "edge_relation_accuracy": float,
        "overall": float,               # moyenne des 4 mesures
      }
    """
    pred_nodes = pred_ir.get("nodes", [])
    gold_nodes = gold_ir.get("nodes", [])
    pred_edges = pred_ir.get("edges", [])
    gold_edges = gold_ir.get("edges", [])

    # Node count ratio (capped at 1.0)
    node_count_ratio = (
        min(len(pred_nodes), len(gold_nodes)) / max(len(gold_nodes), 1)
    )

    # Node type accuracy (align by position)
    if len(pred_nodes) != len(gold_nodes):
        warnings.warn(
            f"causal_graph_similarity: pred_nodes ({len(pred_nodes)}) ≠ gold_nodes "
            f"({len(gold_nodes)}) — alignement par position potentiellement trompeur. "
            f"Envisager un alignement par identité pour des métriques plus fiables.",
            UserWarning,
            stacklevel=2,
        )
    n = min(len(pred_nodes), len(gold_nodes))
    node_type_acc = (
        sum(
            pred_nodes[i].get("node_type") == gold_nodes[i].get("node_type")
            for i in range(n)
        ) / max(n, 1)
    )

    # Edge count ratio
    edge_count_ratio = (
        min(len(pred_edges), len(gold_edges)) / max(len(gold_edges), 1)
    )

    # Edge relation accuracy (align by position)
    if len(pred_edges) != len(gold_edges):
        warnings.warn(
            f"causal_graph_similarity: pred_edges ({len(pred_edges)}) ≠ gold_edges "
            f"({len(gold_edges)}) — alignement par position potentiellement trompeur. "
            f"Envisager un alignement par identité pour des métriques plus fiables.",
            UserWarning,
            stacklevel=2,
        )
    def _edge_attrs(e) -> dict:
        if isinstance(e, dict):
            return e
        if isinstance(e, (list, tuple)) and len(e) > 2:
            return e[2] if isinstance(e[2], dict) else {}
        return {}

    e = min(len(pred_edges), len(gold_edges))
    edge_rel_acc = (
        sum(
            _edge_attrs(pred_edges[i]).get("relation")
            == _edge_attrs(gold_edges[i]).get("relation")
            for i in range(e)
        )
        / max(e, 1)
    )

    overall = float(np.mean([node_count_ratio, node_type_acc,
                              edge_count_ratio, edge_rel_acc]))

    return {
        "node_count_ratio": round(node_count_ratio, 4),
        "node_type_accuracy": round(node_type_acc, 4),
        "edge_count_ratio": round(edge_count_ratio, 4),
        "edge_relation_accuracy": round(edge_rel_acc, 4),
        "overall": round(overall, 4),
    }


# ---------------------------------------------------------------------------
# Métriques décodeur (CausalIR → texte)
# ---------------------------------------------------------------------------

def decoder_causal_fidelity(decoded_ir: dict, gold_ir: dict) -> dict[str, float]:
    """
    Fidélité causale du décodeur : compare le CausalIR obtenu en re-parsant
    la sortie du décodeur avec le CausalIR gold d'origine.

    Le data scientist appelle cette fonction après avoir re-parsé la surface
    générée par le décodeur. Utilise causal_graph_similarity en interne.

    Retourne les mêmes clés que causal_graph_similarity + "causal_fidelity"
    (alias de "overall" pour clarté sémantique).
    """
    sim = causal_graph_similarity(decoded_ir, gold_ir)
    sim["causal_fidelity"] = sim["overall"]
    return sim


def cross_modal_consistency(ir_a: dict, ir_b: dict) -> dict[str, float]:
    """
    Cohérence cross-modale : mesure si deux CausalIR issus de surfaces
    différentes (ex: fr + python) encodent la même structure causale.

    Les deux IR doivent avoir été produits depuis le même graphe causal.
    Un score "consistency" proche de 1.0 indique que les deux surfaces
    encodent fidèlement la même structure.

    Retourne les mêmes clés que causal_graph_similarity + "consistency".
    """
    sim = causal_graph_similarity(ir_a, ir_b)
    sim["consistency"] = sim["overall"]
    return sim


def roundtrip_similarity(source_ir: dict, decoded_ir: dict) -> dict[str, float]:
    """
    Similarité round-trip : mesure la fidélité du cycle complet
    texte → CausalIR → texte → CausalIR.

    source_ir  : CausalIR produit par l'encodeur depuis le texte original
    decoded_ir : CausalIR produit par l'encodeur depuis le texte généré
                 par le décodeur

    Un score "roundtrip" proche de 1.0 indique que l'architecture
    encodeur-décodeur préserve la structure causale.
    """
    sim = causal_graph_similarity(decoded_ir, source_ir)
    sim["roundtrip"] = sim["overall"]
    return sim


def generation_bleu(hypothesis: str, references: list[str], max_n: int = 4) -> float:
    """
    BLEU score simplifié — NumPy pur, sans dépendance externe.

    Mesure la qualité de surface de la sortie du décodeur par rapport
    aux surfaces gold du dataset. Utile pour les surfaces en langage naturel.
    Pour le code source, préférer decoder_causal_fidelity.
    """
    import math
    from collections import Counter

    hyp = hypothesis.split()
    refs = [r.split() for r in references]

    if not hyp or not refs:
        return 0.0

    hyp_len = len(hyp)
    ref_len = min((len(r) for r in refs), key=lambda rl: (abs(rl - hyp_len), rl))
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)

    # Note : on itère jusqu'à min(max_n, len(hyp)) — pas jusqu'à max_n.
    # Comportement intentionnel : BLEU tronqué aux n-grammes disponibles pour
    # les courtes hypothèses NLP (labels causaux, clauses). Standard BLEU-4
    # retournerait 0 pour toute hypothèse < 4 tokens ; ce n'est pas utile ici.
    precisions: list[float] = []
    for n in range(1, min(max_n, len(hyp)) + 1):
        hyp_ng = Counter(_ngrams(hyp, n))
        clipped_total = 0
        for ng, cnt in hyp_ng.items():
            max_ref = max(Counter(_ngrams(r, n))[ng] for r in refs)
            clipped_total += min(cnt, max_ref)
        total = sum(hyp_ng.values())
        precisions.append(clipped_total / total if total > 0 else 0.0)

    if not precisions or min(precisions) == 0.0:
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return round(bp * math.exp(log_avg), 4)


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Métrique au niveau phrase : graph_exact_match
# ---------------------------------------------------------------------------

def graph_exact_match(
    sentences_node_pred: list[list[str]],
    sentences_node_gold: list[list[str]],
    sentences_edge_pred: list[list[str]],
    sentences_edge_gold: list[list[str]],
) -> float:
    """
    Proportion de phrases dont le graphe causal complet est prédit exactement.

    Une phrase est "exacte" si et seulement si :
    1. Le nombre de nœuds prédit == le nombre de nœuds gold
    2. Chaque type de nœud est correct (aligné par position)
    3. Le nombre d'arêtes prédites == le nombre d'arêtes gold
    4. Chaque relation d'arête est correcte (alignée par paire)

    Retourne float [0.0, 1.0]. Retourne 0.0 si la liste est vide.
    """
    if not sentences_node_gold:
        return 0.0
    if not (len(sentences_node_pred) == len(sentences_node_gold)
            == len(sentences_edge_pred) == len(sentences_edge_gold)):
        raise ValueError(
            "graph_exact_match : les 4 listes doivent avoir la même longueur. "
            f"Reçu : node_pred={len(sentences_node_pred)}, "
            f"node_gold={len(sentences_node_gold)}, "
            f"edge_pred={len(sentences_edge_pred)}, "
            f"edge_gold={len(sentences_edge_gold)}"
        )
    n_exact = sum(
        node_pred == node_gold and edge_pred == edge_gold
        for node_pred, node_gold, edge_pred, edge_gold
        in zip(sentences_node_pred, sentences_node_gold,
               sentences_edge_pred, sentences_edge_gold)
    )
    return n_exact / len(sentences_node_gold)


# ---------------------------------------------------------------------------
# Diagnostics : confusion matrix, per-class report
# ---------------------------------------------------------------------------

def confusion_matrix(pred: list[str], gold: list[str], classes: list[str]) -> np.ndarray:
    """
    Retourne une matrice (N_classes × N_classes) de type int.
    cm[i, j] = nombre de fois où gold=classes[i] et pred=classes[j].
    """
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=np.int64)
    for p, g in zip(pred, gold):
        if g in idx and p in idx:
            cm[idx[g], idx[p]] += 1
    return cm


def per_class_report(pred: list[str], gold: list[str], classes: list[str]) -> str:
    """Tableau texte : classe | precision | recall | f1 | support."""
    per_class = _f1_per_class(pred, gold, classes)
    lines = ["classe          prec    recall     f1  support"]
    for cls in classes:
        m = per_class[cls]
        lines.append(
            f"{cls:<16s} {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}  {m['support']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _f1_per_class(
    pred: Predictions, gold: GoldLabels, classes: list[str]
) -> dict[str, dict[str, float]]:
    result = {}
    for cls in classes:
        tp = sum(p == cls and g == cls for p, g in zip(pred, gold))
        fp = sum(p == cls and g != cls for p, g in zip(pred, gold))
        fn = sum(p != cls and g == cls for p, g in zip(pred, gold))
        support = sum(g == cls for g in gold)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        result[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return result
