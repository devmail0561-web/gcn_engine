# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Phase D — T4 unitaire : has_advcl bridge fix (B0.1).

Vérifie que le fix bridge.py:187 (has_advcl=(node_type=="condition"))
est bien en place. La partie end-to-end de T4 (20 phrases conditionnelles
annotées) est dans gcn-datasets/test/ et sera exécutée séparément (nécessite D.0).
"""
import pytest

from gcn_python.frontend.bridge import NODE_TYPE_TO_DEP, NODE_TYPE_TO_POS


class TestBridgeFix:
    """B0.1 — node_type=condition → has_advcl=True via mapping."""

    def test_condition_pos_is_sconj(self):
        """node_type=condition → UPOS SCONJ (SCONJ déclenche advcl)."""
        assert NODE_TYPE_TO_POS.get("condition") == "SCONJ"

    def test_condition_dep_is_advcl(self):
        """node_type=condition → dep_rel advcl."""
        assert NODE_TYPE_TO_DEP.get("condition") == "advcl"

    def test_action_dep_is_root(self):
        """node_type=action → dep_rel root (inchangé)."""
        assert NODE_TYPE_TO_DEP.get("processus") == "root"

    def test_entite_dep_is_nsubj(self):
        """node_type=entite → dep_rel nsubj (inchangé)."""
        assert NODE_TYPE_TO_DEP.get("entite") == "nsubj"

    def test_all_v3_node_types_covered(self):
        """Tous les 8 types v3.0 ont un mapping dans NODE_TYPE_TO_POS."""
        from gcn_python.constants import NODE_TYPES
        for nt in NODE_TYPES:
            assert nt in NODE_TYPE_TO_POS, f"node_type '{nt}' absent de NODE_TYPE_TO_POS"

    def test_all_v3_node_types_covered_dep(self):
        """Tous les 8 types v3.0 ont un mapping dans NODE_TYPE_TO_DEP."""
        from gcn_python.constants import NODE_TYPES
        for nt in NODE_TYPES:
            assert nt in NODE_TYPE_TO_DEP, f"node_type '{nt}' absent de NODE_TYPE_TO_DEP"

    def test_new_v3_types_present(self):
        """Types ajoutés en v3.0 : contrainte, concept, evenement."""
        for nt in ("contrainte", "concept", "evenement"):
            assert nt in NODE_TYPE_TO_POS
            assert nt in NODE_TYPE_TO_DEP

    def test_no_positional_index_access(self):
        """Le fichier bridge.py n'utilise plus NODE_TYPES[i] pour ces dicts."""
        from pathlib import Path
        bridge_src = (
            Path(__file__).parent.parent
            / "src/gcn_python/frontend/bridge.py"
        ).read_text(encoding="utf-8")
        # Les dicts doivent utiliser des strings, pas des index
        assert 'NODE_TYPE_TO_POS: dict[str, str] = {' in bridge_src
        # Les clés sont des strings littérales, pas NODE_TYPES[i]
        assert '"condition"' in bridge_src
        assert '"processus"' in bridge_src
