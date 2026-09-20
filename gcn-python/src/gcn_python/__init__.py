# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""gcn-python — GCN Causal Engine (Python ML layers)."""
__version__ = "2.4.0"

from .engine import GCNEngine  # noqa: F401  — API haut niveau

__all__ = ["GCNEngine", "__version__"]
