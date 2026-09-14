"""
Métriques d'évaluation CGNP — NumPy pur, framework-agnostique.

Le data scientist appelle ces fonctions après chaque epoch pour
mesurer la qualité des prédictions. Aucune dépendance à PyTorch/sklearn.
"""
from __future__ import annotations
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
    n = min(len(pred_nodes), len(gold_nodes))
    node_type_acc = (
        sum(pred_nodes[i]["node_type"] == gold_nodes[i]["node_type"] for i in range(n)) / max(n, 1)
    )

    # Edge count ratio
    edge_count_ratio = (
        min(len(pred_edges), len(gold_edges)) / max(len(gold_edges), 1)
    )

    # Edge relation accuracy (align by position)
    e = min(len(pred_edges), len(gold_edges))
    edge_rel_acc = (
        sum(pred_edges[i][2]["relation"] == gold_edges[i][2]["relation"] for i in range(e))
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
