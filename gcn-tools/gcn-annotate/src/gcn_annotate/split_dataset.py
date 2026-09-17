"""Split du dataset en train/val/test."""

import json
import random
from pathlib import Path


def split_dataset(input_path: Path, output_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """Divise un dataset GCN-NL en train/val/test."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    
    sentences = data["document"]["sentences"]
    random.seed(seed)
    random.shuffle(sentences)
    
    n = len(sentences)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": sentences[:n_train],
        "val": sentences[n_train:n_train + n_val],
        "test": sentences[n_train + n_val:],
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_sents in splits.items():
        split_data = {
            "document": {
                "id": f"real_{split_name}",
                "lang": "fr",
                "sentences": split_sents
            }
        }
        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"{split_name}: {len(split_sents)} phrases -> {out_path}")
    
    return splits


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[4]
    input_file = base / "gcn-datasets/real/annotated.json"
    output_dir = base / "gcn-datasets/real"
    split_dataset(input_file, output_dir)
