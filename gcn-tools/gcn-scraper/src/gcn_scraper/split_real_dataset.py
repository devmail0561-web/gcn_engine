"""
Re-génère les splits train/val/test avec tokens UD.

Règles de split :
- Ratio : 70 % train / 15 % val / 15 % test
- Stratification : sur la relation de chaque edge
- Seed fixe (42) pour reproductibilité
- Les 5 relations présentes doivent être représentées dans val et test
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def split_dataset(
    annotated_ud_path: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    random.seed(seed)
    data = json.loads(Path(annotated_ud_path).read_text())
    sents = data["document"]["sentences"]

    by_relation = defaultdict(list)
    for s in sents:
        edges = s.get("cir", {}).get("edges", [])
        if edges:
            dominant = Counter(e["relation"] for e in edges).most_common(1)[0][0]
        else:
            dominant = "_no_edge"
        by_relation[dominant].append(s)

    train_sents, val_sents, test_sents = [], [], []
    for rel, group in by_relation.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        # Garantir au moins 1 exemple en val si le groupe a ≥ 3 phrases
        if n_val == 0 and n >= 3:
            n_val = 1
            n_train = n - n_val  # ajuster train
        if n_val == 0 and n > 0:
            print(
                f"ATTENTION : relation '{rel}' a {n} phrase(s) — "
                f"aucune assignée au val set (minimum 3 phrases requis par classe pour val≥1)."
            )
        train_sents.extend(group[:n_train])
        val_sents.extend(group[n_train:n_train + n_val])
        test_sents.extend(group[n_train + n_val:])

    os.makedirs(output_dir, exist_ok=True)

    for name, sents_list in [
        ("train.json", train_sents),
        ("val.json", val_sents),
        ("test.json", test_sents),
    ]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            json.dump({"document": {"sentences": sents_list}}, f, ensure_ascii=False, indent=2)

    # Rapport
    report = {}
    for name, sents_list in [("train", train_sents), ("val", val_sents), ("test", test_sents)]:
        rels = Counter(
            e["relation"]
            for s in sents_list
            for e in s.get("cir", {}).get("edges", [])
        )
        report[name] = {"count": len(sents_list), "relations": dict(rels)}
        print(f"{name}: {len(sents_list)} phrases, relations: {dict(rels)}")

    return report


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "gcn-datasets/real/annotated_ud.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "gcn-datasets/real"
    split_dataset(inp, out_dir)
