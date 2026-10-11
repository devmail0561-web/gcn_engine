# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""R1 — raisonnement causal borné (fonction pure, cf. plan-raisonnement-r1.md)."""
from .engine import MAX_EDGES, MAX_NODES, TOPK, reason, reason_question, verify_path
from .steps import REASON_VERSION, STEP_CODES, reason_response
from .trace import dump_trace, load_trace

__all__ = [
    "MAX_EDGES", "MAX_NODES", "TOPK",
    "REASON_VERSION", "STEP_CODES",
    "dump_trace", "load_trace",
    "reason", "reason_question", "reason_response", "verify_path",
]
