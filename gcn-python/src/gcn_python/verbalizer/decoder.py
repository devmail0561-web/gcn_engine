from __future__ import annotations
import json


class ReferenceDecoder:
    """
    Reference verbalizer decoder — format-agnostic linearization placeholder.

    Implements the VerbalizerDecoder protocol.
    Replace with a trained model: the trained model learns what to produce
    from its training data — no output format is presupposed here.

    This reference produces a structural linearization of the CausalIR graph:
    one fragment per edge, joined by newlines. It makes no assumption about
    whether the output should be natural language, code, or any other form.
    """

    def decode(self, ir_json: str) -> str:
        ir = json.loads(ir_json)
        nodes: list[dict] = ir.get("nodes", [])
        edges: list[list] = ir.get("edges", [])

        node_map: dict[int, str] = {n["id"]: n["label"] for n in nodes}

        fragments: list[str] = []
        for edge_tuple in edges:
            src_id, dst_id, edge = edge_tuple
            src = node_map.get(src_id, str(src_id))
            dst = node_map.get(dst_id, str(dst_id))
            relation: str = edge.get("relation", "?")
            negated: bool = edge.get("negated", False)
            neg = "¬" if negated else ""
            fragments.append(f"{src} -{neg}[{relation}]-> {dst}")

        if not fragments:
            return " | ".join(n["label"] for n in nodes)

        return "\n".join(fragments)
