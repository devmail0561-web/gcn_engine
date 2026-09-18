"""
Oversampling des classes rares dans les données d'entraînement.

Duplique les phrases contenant des arêtes de relation rare pour équilibrer
le dataset sans introduire de nouvelles données.

Cibles minimales (exemples d'arêtes) :
  prevent : 14 → ~42 (×3)
  concession : 30 → déjà OK
  opposition : 4  → ~32 (×8)
  motivation : 3  → ~30 (×10)
  sequence : 2   → ~30 (×15)
"""
import json
import copy
from pathlib import Path
from collections import Counter


RARE_TARGETS = {
    "prevent": 42,
    "opposition": 32,
    "motivation": 30,
    "sequence": 30,
}


def count_edges(sentence: dict) -> Counter:
    c = Counter()
    for edge in sentence.get("cir", {}).get("edges", []):
        rel = edge.get("relation")
        if rel:
            c[rel] += 1
    return c


def oversample(input_path: Path, output_path: Path) -> None:
    with open(input_path) as f:
        raw = json.load(f)

    sentences = raw["document"]["sentences"]
    edge_counts = Counter()
    for s in sentences:
        edge_counts.update(count_edges(s))

    print("Distribution originale :")
    for rel, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel:<22} {cnt:4d}")

    # Calculer les facteurs de duplication par phrase
    extra: list[dict] = []
    for s in sentences:
        s_edges = count_edges(s)
        max_factor = 0
        for rel, target in RARE_TARGETS.items():
            if s_edges.get(rel, 0) > 0:
                current = edge_counts[rel]
                factor = max(0, target // current - 1)
                max_factor = max(max_factor, factor)
        for _ in range(max_factor):
            dup = copy.deepcopy(s)
            dup["id"] = f"{s['id']}_dup{_ + 1}"
            extra.append(dup)

    augmented = sentences + extra
    print(f"\nAprès oversampling : {len(sentences)} → {len(augmented)} phrases (+{len(extra)})")

    # Recalculer la distribution
    new_counts = Counter()
    for s in augmented:
        new_counts.update(count_edges(s))
    print("\nDistribution après oversampling :")
    for rel, cnt in sorted(new_counts.items(), key=lambda x: -x[1]):
        old = edge_counts[rel]
        marker = " ↑" if cnt > old else ""
        print(f"  {rel:<22} {cnt:4d}  (était {old}){marker}")

    out = copy.deepcopy(raw)
    out["document"]["sentences"] = augmented
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegardé : {output_path}")


if __name__ == "__main__":
    base = Path(__file__).parent
    oversample(
        base / "real" / "train" / "train.json",
        base / "real" / "train_oversampled" / "train.json",
    )
