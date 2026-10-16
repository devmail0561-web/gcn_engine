# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Frontend : texte brut → UDRepresentation via lattice gcn-cli (zéro dictionnaire)."""
from .bridge import GCNBridgeError, GCNLatticeParser, reps_from_lattice_text

__all__ = ["GCNBridgeError", "GCNLatticeParser", "reps_from_lattice_text"]
