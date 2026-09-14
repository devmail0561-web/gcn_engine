from gcn_python.evaluation.metrics import (
    node_accuracy, node_f1_per_class, node_macro_f1,
    edge_accuracy, edge_macro_f1, causal_graph_similarity,
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
# pytest_approx helper
# ---------------------------------------------------------------------------

def pytest_approx(x, rel=1e-6):
    class Approx:
        def __eq__(self, other):
            return abs(other - x) <= rel * max(abs(x), 1e-12)
        def __repr__(self):
            return f"~{x}"
    return Approx()
