# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Contrôles de sûreté sur les artefacts chargés depuis le disque."""
from .npz_guard import UnsafeCheckpointError, assert_npz_pickle_safe, guarded_np_load

__all__ = ["UnsafeCheckpointError", "assert_npz_pickle_safe", "guarded_np_load"]
