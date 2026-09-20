# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json


class ReferenceDecoder:
    """
    Reference verbalizer decoder — linearisation structurée d'un CausalIR.

    Deux points d'entrée :
      - decode_cir(cir: dict)  : flux direct depuis le moteur (aucun fichier)
      - decode(ir_json: str)   : depuis un fichier JSON (pipeline batch)

    Le CIR produit par engine.analyze() passe directement ici —
    pas de sérialisation/désérialisation intermédiaire.

    À remplacer par un TrainableDecoder entraîné pour produire
    du texte naturel dans le domaine cible.
    """

    def __init__(self, max_source_chars: int = 200):
        self.max_source_chars = int(max_source_chars)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        import re as _re
        text = _re.sub(r"\x1b\[[0-9;]*m", "", text or "").replace("\n", " ").replace("\r", " ").strip()
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars].rsplit(" ", 1)[0]
        return cut if cut else text[:max_chars]

    def decode_cir(self, cir: dict) -> str:
        """
        CausalIR dict → texte structuré lisible.

        Flux direct depuis engine.analyze() sans fichier intermédiaire.
        """
        nodes: list[dict] = cir.get("nodes", [])
        edges = cir.get("edges", [])
        source_text: str = cir.get("source_text", "")

        if not nodes:
            return ""

        node_map: dict = {n["id"]: (n.get("label", ""), n.get("node_type", "?"))
                          for n in nodes}

        lines = []
        if source_text:
            lines.append(f'  "{self._truncate(source_text, self.max_source_chars)}"')

        if not edges:
            labels = [n.get("label", "") for n in nodes if n.get("label")]
            lines.append("  → " + "  |  ".join(labels))
            return "\n".join(lines)

        for edge_tuple in edges:
            if not (isinstance(edge_tuple, (list, tuple)) and len(edge_tuple) == 3):
                continue
            src_id, dst_id, attrs = edge_tuple
            src_lbl, src_type = node_map.get(src_id, (str(src_id), "?"))
            dst_lbl, dst_type = node_map.get(dst_id, (str(dst_id), "?"))
            relation: str = attrs.get("relation", "?")
            confidence = attrs.get("confidence")
            conf_s = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "?"
            negated: bool = attrs.get("negated", False)
            neg = " [negated]" if negated else ""
            lines.append(
                f"  {src_lbl} [{src_type}]"
                f"  →[{relation}{neg}]→"
                f"  {dst_lbl} [{dst_type}]"
                f"  ({conf_s})"
            )

        return "\n".join(lines)

    def decode(self, ir_json: str) -> str:
        """Depuis un fichier JSON (pipeline batch). Délègue à decode_cir."""
        return self.decode_cir(json.loads(ir_json))
