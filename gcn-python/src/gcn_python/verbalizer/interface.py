# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class VerbalizerDecoder(Protocol):
    """
    Protocol for verbalizer decoders: CausalIR JSON → surface string.

    The reference implementation is ReferenceDecoder (linearization placeholder).
    Replace with a trained model: what the model produces depends on its training data.
    """

    def decode(self, ir_json: str) -> str:
        """
        Decode a CausalIR JSON string into a surface form.

        What the decoder produces depends entirely on its training data.
        No output format is presupposed by the architecture.
        """
        ...
