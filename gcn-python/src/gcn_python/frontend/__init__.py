# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Frontend bridge : texte brut → UDRepresentation via gcn-cli subprocess."""
from .bridge import GCNBridgeError, reps_from_text

__all__ = ["GCNBridgeError", "reps_from_text"]
