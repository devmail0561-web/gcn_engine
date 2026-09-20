# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Frontend bridge : texte brut → UDRepresentation via gcn-cli subprocess."""
from .bridge import reps_from_text, GCNBridgeError

__all__ = ["reps_from_text", "GCNBridgeError"]
