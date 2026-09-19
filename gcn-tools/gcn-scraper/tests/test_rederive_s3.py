"""C6/C7 : S3 valide même si n_nodes > T ; les phrases vides sont conservées."""
import json

import pytest

spacy = pytest.importorskip("spacy")
from gcn_scraper import rederive_spans as rs


class _FakeDoc:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __iter__(self):
        return iter([])


def test_s3_valid_when_more_nodes_than_tokens():
    spans = rs._detect_clauses_s3(_FakeDoc(2), 5)
    assert len(spans) == 5
    for s, e in spans:
        assert 1 <= s <= e <= 2


def test_empty_sentences_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(spacy, "load", lambda *a, **k: (lambda t: _FakeDoc(10)))
    data = {
        "document": {
            "id": "d",
            "sentences": [
                {"id": "s0", "text": "courte.", "cir": {"nodes": []}},
                {
                    "id": "s1",
                    "text": "La pluie cause des inondations majeures dans toute la vallée du fleuve.",
                    "cir": {"nodes": [{"id": "n1"}, {"id": "n2"}]},
                },
            ],
        }
    }
    src = tmp_path / "in.json"
    dst = tmp_path / "out.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    report = rs.rederive_all_spans(str(src), str(dst))
    out = json.loads(dst.read_text(encoding="utf-8"))
    ids = [s["id"] for s in out["document"]["sentences"]]
    assert "s0" in ids  # phrase vide conservée, pas jetée (cf C7)
    assert report["corrected"] >= 1
