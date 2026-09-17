"""
Script d'annotation UD du dataset réel.

Lit annotated_spans_fixed.json (CIR corrigé, 0 token UD).
Pour chaque phrase :
  1. Appelle annotate_ud(text, lang, id_convention="1based")
  2. Injecte la liste de tokens dans sentence["tokens"]
  3. NE MODIFIE PAS le CIR (nodes, edges)
Écrit annotated_ud.json.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from ud_annotator import annotate_ud


def annotate_real_dataset(input_path: str, output_path: str, lang: str = "fr") -> dict:
    data = json.loads(Path(input_path).read_text())
    sents = data["document"]["sentences"]

    success = 0
    errors = 0
    total_tokens = 0

    for s in sents:
        text = s.get("text", "")
        if not text:
            errors += 1
            continue
        try:
            tokens = annotate_ud(text, lang=lang, id_convention="1based")
            s["tokens"] = tokens
            success += 1
            total_tokens += len(tokens)
        except Exception as e:
            print(f"Erreur [{s.get('id', '?')}]: {e}")
            errors += 1
            s["tokens"] = []

    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    avg_tokens = total_tokens / max(success, 1)
    report = {
        "success": success,
        "errors": errors,
        "avg_tokens_per_sentence": round(avg_tokens, 1),
    }

    print(f"Succès : {success}, Erreurs : {errors}, Tokens moyens : {avg_tokens:.1f}")
    print(f"Écrit : {output_path}")

    return report


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "gcn-datasets/real/annotated_spans_fixed.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "gcn-datasets/real/annotated_ud.json"
    annotate_real_dataset(inp, out)
