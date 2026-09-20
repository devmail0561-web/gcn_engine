# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
TrainingRecorder — suivi des métriques et courbes d'apprentissage par epoch.

Framework-agnostique (NumPy pur). Le data scientist appelle .record() dans
sa boucle d'entraînement. Le recorder stocke l'historique et permet
l'export CSV pour visualisation externe (matplotlib, Excel, etc.).
"""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    metrics: dict[str, float]   # {"node_accuracy": 0.87, "edge_macro_f1": 0.72, …}


class TrainingRecorder:
    """
    Enregistre loss et métriques CGNP par epoch.

    Usage typique dans la boucle du DS :

        recorder = TrainingRecorder()
        for epoch in range(n_epochs):
            pipeline.forward(reps, text)
            node_logits = pipeline._cached_node_logits
            edge_logits = pipeline._cached_edge_logits
            loss_val, d_node, d_edge = pipeline.loss(
                node_logits, edge_logits, gold_node, gold_edge
            )
            pipeline.backward(d_node, d_edge, lr=0.001)
            metrics = {
                "node_accuracy": node_accuracy(pred_types, gold_types),
                "edge_macro_f1": edge_macro_f1(pred_rels, gold_rels),
            }
            recorder.record(epoch, loss_val, metrics)

        recorder.to_csv("training_history.csv")
        curve = recorder.learning_curve()
    """

    def __init__(self) -> None:
        self._history: list[EpochRecord] = []

    def record(self, epoch: int, loss: float, metrics: dict[str, float]) -> None:
        """Enregistre une epoch."""
        self._history.append(EpochRecord(epoch=epoch, loss=loss, metrics=dict(metrics)))

    def learning_curve(self) -> dict[str, list]:
        """
        Retourne les séries temporelles pour tracer les courbes d'apprentissage.

        Exemple de retour :
          {
            "epoch": [0, 1, 2, …],
            "loss": [2.1, 1.8, 1.5, …],
            "node_accuracy": [0.3, 0.55, 0.72, …],
            "node_macro_f1": […],
            "edge_accuracy": […],
            "edge_macro_f1": […],
          }
        """
        if not self._history:
            return {}

        curve: dict[str, list] = {
            "epoch": [r.epoch for r in self._history],
            "loss": [r.loss for r in self._history],
        }
        # Collect all metric keys that appear across epochs
        all_keys: set[str] = set()
        for r in self._history:
            all_keys.update(r.metrics.keys())

        for key in sorted(all_keys):
            curve[key] = [r.metrics.get(key, float("nan")) for r in self._history]

        return curve

    def best_epoch(self, metric: str = "loss", mode: str = "min") -> EpochRecord | None:
        """
        Retourne l'epoch avec la meilleure valeur d'une métrique.

        mode="min" → minimize (pour loss)
        mode="max" → maximize (pour accuracy, f1)
        """
        if not self._history:
            return None

        def get_val(r: EpochRecord) -> float:
            if metric == "loss":
                return r.loss
            return r.metrics.get(metric, float("nan"))

        valid = [r for r in self._history if not _is_nan(get_val(r))]
        if not valid:
            return None

        if mode == "min":
            return min(valid, key=get_val)
        return max(valid, key=get_val)

    def summary(self) -> dict:
        """Résumé : première epoch, dernière epoch, meilleure loss."""
        if not self._history:
            return {}
        first = self._history[0]
        last = self._history[-1]
        best = self.best_epoch("loss", "min")
        return {
            "n_epochs": len(self._history),
            "first_loss": first.loss,
            "last_loss": last.loss,
            "best_loss": best.loss if best else None,
            "best_epoch": best.epoch if best else None,
            "last_metrics": last.metrics,
        }

    def to_csv(self, path: Path | str) -> None:
        """Export CSV — une ligne par epoch, toutes les métriques en colonnes."""
        curve = self.learning_curve()
        if not curve:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(curve.keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            n = len(curve["epoch"])
            for i in range(n):
                writer.writerow([curve[k][i] for k in keys])

    def to_json(self, path: Path | str) -> None:
        """Export JSON — liste d'objets {epoch, loss, …metrics}."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for r in self._history:
            row = {"epoch": r.epoch, "loss": r.loss}
            row.update(r.metrics)
            records.append(row)
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"TrainingRecorder(epochs={len(self._history)})"


def _is_nan(v: float) -> bool:
    return v != v  # NaN check without math import
