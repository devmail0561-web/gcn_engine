# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""NLU sans UD (refacto étape 6) : voir ``infer``."""

from .infer import EXPECTS, content_words, infer_nlu, silver_annotate

__all__ = ["EXPECTS", "content_words", "infer_nlu", "silver_annotate"]
