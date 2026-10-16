# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Pré-entraînement sans labels (O1/O3), zéro lexique."""
from .masking import permute_clauses, span_mask

__all__ = ["permute_clauses", "span_mask"]
