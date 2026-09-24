# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""C2.1 : _methode silver-auto-v2 au niveau phrase dans balanced_auto.run_file."""
from __future__ import annotations

import json
from pathlib import Path

from gcn_annotate.balanced_auto import run_file


def _write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t, "lang": "fr"}, ensure_ascii=False) + "\n")


def test_run_file_writes_methode_at_sentence_level(tmp_path: Path):
    inp = tmp_path / "inp.jsonl"
    outp = tmp_path / "out.jsonl"
    _write_jsonl(inp, [
        "Les ventes baissent parce que les prix montent.",
        "La pluie tombe donc le sol est mouillé.",
        "Si tu étudies alors tu réussiras.",
    ])
    run_file(inp, outp)
    lines = [
        json.loads(l) for l in outp.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert lines, "aucune annotation produite par run_file"
    for line in lines:
        assert "_methode" in line, "_methode absent au niveau phrase"
        assert line["_methode"].startswith("silver"), line["_methode"]
        assert "_methode" not in line.get("cir", {}), (
            "_methode ne doit pas être écrit dans cir"
        )


def test_run_file_keeps_silver_subdict_intact(tmp_path: Path):
    inp = tmp_path / "inp.jsonl"
    outp = tmp_path / "out.jsonl"
    _write_jsonl(inp, ["Les ventes baissent parce que les prix montent."])
    run_file(inp, outp)
    lines = [
        json.loads(l) for l in outp.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert lines, "aucune annotation"
    cir = lines[0].get("cir", {})
    assert isinstance(cir.get("_silver"), dict), "sous-dict cir._silver absent"
    assert cir["_silver"].get("method") == "auto-v2"


def test_gcndataloader_applies_silver_weight_on_balanced_auto_output(tmp_path: Path):
    """Sortie run_file → document JSON → GCNDataLoader(silver_weight=0.5)."""
    from gcn_python.data.loader import GCNDataLoader

    inp = tmp_path / "inp.jsonl"
    outp = tmp_path / "out.jsonl"
    _write_jsonl(inp, [
        "Les ventes baissent parce que les prix montent.",
        "La pluie tombe donc le sol est mouillé.",
    ])
    run_file(inp, outp)
    sentences = [
        json.loads(l) for l in outp.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert sentences, "run_file n'a produit aucune phrase"
    for s in sentences:
        assert s.get("_methode", "").startswith("silver")

    data_dir = tmp_path / "train_data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text(
        json.dumps({"document": {"lang": "fr", "sentences": sentences}},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    loader = GCNDataLoader(data_dir, silver_weight=0.5)
    samples = list(loader)
    assert samples, "GCNDataLoader n'a chargé aucun sample"
    assert samples[0].sentence.weight == 0.5, (
        f"attendu silver_weight=0.5, obtenu {samples[0].sentence.weight}"
    )
