from gcn_python.evaluation.metrics import (
    node_accuracy, node_f1_per_class, node_macro_f1,
    edge_accuracy, edge_macro_f1, causal_graph_similarity,
    decoder_causal_fidelity, cross_modal_consistency,
    roundtrip_similarity, generation_bleu,
    graph_exact_match, confusion_matrix, per_class_report,
)
from gcn_python.evaluation.recorder import TrainingRecorder
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Métriques nœuds
# ---------------------------------------------------------------------------

def test_node_accuracy_perfect():
    pred = ["action", "processus", "etat"]
    gold = ["action", "processus", "etat"]
    assert node_accuracy(pred, gold) == 1.0


def test_node_accuracy_partial():
    pred = ["action", "action", "etat"]
    gold = ["action", "processus", "etat"]
    assert node_accuracy(pred, gold) == pytest_approx(2 / 3)


def test_node_f1_per_class_keys():
    from gcn_python.constants import NODE_TYPES
    pred = ["action", "processus"]
    gold = ["action", "etat"]
    result = node_f1_per_class(pred, gold)
    assert set(result.keys()) == set(NODE_TYPES)


def test_node_macro_f1_perfect():
    pred = ["action", "processus", "condition"]
    gold = ["action", "processus", "condition"]
    assert node_macro_f1(pred, gold) == 1.0


def test_empty_inputs():
    assert node_accuracy([], []) == 0.0
    assert node_macro_f1([], []) == 0.0
    assert edge_accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# Métriques arêtes
# ---------------------------------------------------------------------------

def test_edge_accuracy_perfect():
    pred = ["cause", "condition"]
    gold = ["cause", "condition"]
    assert edge_accuracy(pred, gold) == 1.0


def test_edge_macro_f1():
    pred = ["cause", "condition", "cause"]
    gold = ["cause", "enable", "cause"]
    f1 = edge_macro_f1(pred, gold)
    assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# Similarité de graphe
# ---------------------------------------------------------------------------

def _make_ir(node_types, edge_relations):
    nodes = [{"node_type": t} for t in node_types]
    edges = [[i, i + 1, {"relation": r}] for i, r in enumerate(edge_relations)]
    return {"nodes": nodes, "edges": edges}


def test_graph_similarity_identical():
    ir = _make_ir(["action", "processus"], ["cause"])
    s = causal_graph_similarity(ir, ir)
    assert s["overall"] == 1.0
    assert s["node_type_accuracy"] == 1.0
    assert s["edge_relation_accuracy"] == 1.0


def test_graph_similarity_empty_pred():
    gold = _make_ir(["action", "etat"], ["cause"])
    pred = _make_ir([], [])
    s = causal_graph_similarity(pred, gold)
    assert s["node_count_ratio"] == 0.0


def test_graph_similarity_partial():
    gold = _make_ir(["action", "processus"], ["cause"])
    pred = _make_ir(["action", "etat"], ["enable"])
    s = causal_graph_similarity(pred, gold)
    assert 0.0 <= s["overall"] <= 1.0


# ---------------------------------------------------------------------------
# TrainingRecorder
# ---------------------------------------------------------------------------

def test_recorder_record_and_curve():
    rec = TrainingRecorder()
    for i in range(5):
        rec.record(i, loss=2.0 - i * 0.3, metrics={"node_accuracy": 0.5 + i * 0.1})
    curve = rec.learning_curve()
    assert curve["epoch"] == [0, 1, 2, 3, 4]
    assert len(curve["loss"]) == 5
    assert len(curve["node_accuracy"]) == 5


def test_recorder_best_epoch():
    rec = TrainingRecorder()
    rec.record(0, loss=1.5, metrics={})
    rec.record(1, loss=0.8, metrics={})
    rec.record(2, loss=1.1, metrics={})
    best = rec.best_epoch("loss", "min")
    assert best.epoch == 1


def test_recorder_to_csv():
    rec = TrainingRecorder()
    rec.record(0, 1.0, {"node_accuracy": 0.6, "edge_macro_f1": 0.5})
    rec.record(1, 0.7, {"node_accuracy": 0.75, "edge_macro_f1": 0.65})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "history.csv"
        rec.to_csv(path)
        content = path.read_text()
        assert "epoch" in content
        assert "loss" in content
        assert "node_accuracy" in content
        lines = content.strip().splitlines()
        assert len(lines) == 3  # header + 2 epochs


def test_recorder_summary():
    rec = TrainingRecorder()
    rec.record(0, 2.0, {"node_accuracy": 0.4})
    rec.record(1, 1.0, {"node_accuracy": 0.8})
    s = rec.summary()
    assert s["n_epochs"] == 2
    assert s["best_epoch"] == 1
    assert s["best_loss"] == 1.0


def test_recorder_to_json():
    rec = TrainingRecorder()
    rec.record(0, 1.5, {"node_accuracy": 0.7})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "history.json"
        rec.to_json(path)
        import json
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["epoch"] == 0
        assert data[0]["node_accuracy"] == 0.7


# ---------------------------------------------------------------------------
# Métriques décodeur
# ---------------------------------------------------------------------------

_IR_A = {
    "nodes": [
        {"node_type": "processus", "id": 0, "label": "ventes"},
        {"node_type": "action",    "id": 1, "label": "coûts"},
    ],
    "edges": [[0, 1, {"relation": "condition"}]],
}
_IR_B = {  # même structure, légèrement différente (surface différente)
    "nodes": [
        {"node_type": "processus", "id": 0, "label": "sales"},
        {"node_type": "action",    "id": 1, "label": "costs"},
    ],
    "edges": [[0, 1, {"relation": "condition"}]],
}
_IR_DIFFERENT = {
    "nodes": [{"node_type": "etat", "id": 0, "label": "x"}],
    "edges": [],
}


def test_decoder_causal_fidelity_perfect():
    result = decoder_causal_fidelity(_IR_A, _IR_A)
    assert result["causal_fidelity"] == 1.0


def test_decoder_causal_fidelity_has_expected_keys():
    result = decoder_causal_fidelity(_IR_A, _IR_B)
    assert "causal_fidelity" in result
    assert "node_type_accuracy" in result
    assert "edge_relation_accuracy" in result


def test_cross_modal_consistency_same_structure():
    result = cross_modal_consistency(_IR_A, _IR_B)
    assert result["consistency"] == 1.0  # same types and relations
    assert "node_type_accuracy" in result


def test_cross_modal_consistency_different_structure():
    result = cross_modal_consistency(_IR_A, _IR_DIFFERENT)
    assert result["consistency"] < 1.0


def test_roundtrip_similarity_perfect():
    result = roundtrip_similarity(_IR_A, _IR_A)
    assert result["roundtrip"] == 1.0


def test_roundtrip_similarity_has_key():
    result = roundtrip_similarity(_IR_A, _IR_B)
    assert "roundtrip" in result
    assert 0.0 <= result["roundtrip"] <= 1.0


def test_generation_bleu_perfect():
    score = generation_bleu("le chat mange", ["le chat mange"])
    assert score == 1.0


def test_generation_bleu_empty_hypothesis():
    score = generation_bleu("", ["le chat mange"])
    assert score == 0.0


def test_generation_bleu_partial():
    score = generation_bleu("le chat", ["le chat mange"])
    assert 0.0 < score < 1.0


def test_generation_bleu_multiple_references():
    score = generation_bleu("if x: y()", ["if x: y()", "if x:\n    y()"])
    assert score > 0.0


# ---------------------------------------------------------------------------
# pytest_approx helper
# ---------------------------------------------------------------------------

def pytest_approx(x, rel=1e-6):
    class Approx:
        def __eq__(self, other):
            return abs(other - x) <= rel * max(abs(x), 1e-12)
        def __repr__(self):
            return f"~{x}"
    return Approx()


# ---------------------------------------------------------------------------
# graph_exact_match (Phase 2.5)
# ---------------------------------------------------------------------------

def test_graph_exact_match_all_correct():
    node_pred = [["action", "processus"], ["etat"]]
    node_gold = [["action", "processus"], ["etat"]]
    edge_pred = [["cause"], []]
    edge_gold = [["cause"], []]
    assert graph_exact_match(node_pred, node_gold, edge_pred, edge_gold) == 1.0


def test_graph_exact_match_none_correct():
    node_pred = [["action"], ["action"]]
    node_gold = [["etat"], ["processus"]]
    edge_pred = [["cause"], ["enable"]]
    edge_gold = [["enable"], ["cause"]]
    assert graph_exact_match(node_pred, node_gold, edge_pred, edge_gold) == 0.0


def test_graph_exact_match_nodes_correct_edges_wrong():
    node_pred = [["action", "processus"]]
    node_gold = [["action", "processus"]]
    edge_pred = [["cause"]]
    edge_gold = [["enable"]]
    assert graph_exact_match(node_pred, node_gold, edge_pred, edge_gold) == 0.0


def test_graph_exact_match_empty_list():
    assert graph_exact_match([], [], [], []) == 0.0


def test_graph_exact_match_mismatched_lengths():
    try:
        graph_exact_match([["action"]], [["action"], ["etat"]], [[]], [[]])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# confusion_matrix (Phase 6.3)
# ---------------------------------------------------------------------------

def test_confusion_matrix_diagonal():
    pred = ["action", "processus", "etat"]
    gold = ["action", "processus", "etat"]
    classes = ["action", "processus", "etat"]
    cm = confusion_matrix(pred, gold, classes)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1
    assert cm.sum() == 3


def test_confusion_matrix_off_diagonal():
    pred = ["action", "action"]
    gold = ["action", "processus"]
    classes = ["action", "processus"]
    cm = confusion_matrix(pred, gold, classes)
    assert cm[0, 0] == 1  # action→action
    assert cm[1, 0] == 1  # processus→action
    assert cm.sum() == 2


# ---------------------------------------------------------------------------
# per_class_report (Phase 6.3)
# ---------------------------------------------------------------------------

def test_per_class_report():
    pred = ["action", "action", "processus"]
    gold = ["action", "processus", "processus"]
    classes = ["action", "processus"]
    report = per_class_report(pred, gold, classes)
    assert "action" in report
    assert "processus" in report
    assert "prec" in report
