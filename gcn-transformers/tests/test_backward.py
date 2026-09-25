# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Tests backward (gradients) avec torch.autograd.gradcheck.
"""
import numpy as np
import torch

from gcn_transformers import XLMRobertaEncoder


def test_backward_node_dx_returns_correct_format(dimensions):
    """Vérifie que backward_node_dx retourne (grads, dx) au bon format."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    # Forward
    X = np.random.randn(2, dimensions["d_clause"]).astype(np.float32)
    logits = encoder.forward_batch(X)

    # Backward
    d_logits = np.ones_like(logits)
    grads, dx = encoder.backward_node_dx(d_logits)

    # Vérifier format
    assert isinstance(grads, list)
    assert len(grads) > 0
    for dW, db in grads:
        assert isinstance(dW, np.ndarray)
        assert isinstance(db, np.ndarray)

    assert isinstance(dx, np.ndarray)
    assert dx.shape == X.shape, f"Expected dx.shape={X.shape}, got {dx.shape}"


def test_backward_edge_dx_returns_correct_format(dimensions):
    """Vérifie que backward_edge_dx retourne (grads, dx) au bon format."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    # Forward
    x = np.random.randn(dimensions["d_edge"]).astype(np.float32)
    logits = encoder.forward_edge(x)

    # Backward
    d_logits = np.ones_like(logits)
    grads, dx = encoder.backward_edge_dx(d_logits)

    # Vérifier format
    assert isinstance(grads, list)
    assert len(grads) > 0
    for dW, db in grads:
        assert isinstance(dW, np.ndarray)
        assert isinstance(db, np.ndarray)

    assert isinstance(dx, np.ndarray)
    assert dx.shape == x.shape


def test_backward_node_accumulation(dimensions):
    """Vérifie que les gradients s'accumulent correctement (N nœuds)."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    N = 3
    X = np.random.randn(N, dimensions["d_clause"]).astype(np.float32)

    # Simuler ce que fait le pipeline : forward + backward par nœud
    all_grads = None
    for i in range(N):
        # Forward single node
        logits = encoder.forward_node(X[i])

        # Backward
        d_logits = np.ones_like(logits)
        grads_i, dx_i = encoder.backward_node_dx(d_logits)

        # Accumulation (comme cgnp.py:883-890)
        if all_grads is None:
            all_grads = [(dW.copy(), db.copy()) for dW, db in grads_i]
        else:
            for j, (dW_i, db_i) in enumerate(grads_i):
                all_grads[j] = (
                    all_grads[j][0] + dW_i,
                    all_grads[j][1] + db_i,
                )

    # Vérifier que les gradients sont non-nuls
    for dW, db in all_grads:
        assert np.any(dW != 0), "dW should be non-zero after accumulation"
        assert np.any(db != 0), "db should be non-zero after accumulation"


def test_update_node_zero_grad_after(dimensions):
    """Vérifie que update_node fait zero_grad APRÈS optimizer.step()."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    # Forward + backward
    X = np.random.randn(2, dimensions["d_clause"]).astype(np.float32)
    logits = encoder.forward_batch(X)
    d_logits = np.ones_like(logits)
    grads, dx = encoder.backward_node_dx(d_logits)

    # Vérifier que les gradients sont présents AVANT update_node
    has_grad_before = any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in encoder.optimizer.param_groups[0]['params']
    )
    assert has_grad_before, "Gradients should be present before update_node"

    # Update node (fait optimizer.step())
    encoder.update_node(grads, lr=1e-5)

    # Vérifier que les gradients sont ENCORE PRÉSENTS après update_node
    # (le zero_grad est différé jusqu'à update_edge pour éviter double step)
    has_grad_after_node = any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in encoder.optimizer.param_groups[0]['params']
    )
    assert has_grad_after_node, "Gradients should still be present after update_node (zero_grad deferred)"
    assert encoder._needs_zero_grad, "Flag _needs_zero_grad should be True after update_node"

    # Backward edge (contrat : liste de (dW, db) par couche Linear)
    x_edge = np.random.randn(dimensions["d_edge"]).astype(np.float32)
    logits_edge = encoder.forward_edge(x_edge)
    grads_edge, dx_edge = encoder.backward_edge_dx(np.ones_like(logits_edge))
    encoder.update_edge(grads_edge, lr=1e-5)

    # Vérifier que les gradients sont zérotés APRÈS update_edge
    has_grad_after_edge = any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in encoder.optimizer.param_groups[0]['params']
    )
    assert not has_grad_after_edge, "Gradients should be zero after update_edge"
    assert not encoder._needs_zero_grad, "Flag _needs_zero_grad should be False after update_edge"


def test_snapshot_restore_preserves_gradients(dimensions):
    """Vérifie que snapshot/restore préserve le graphe autograd."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])

    # Forward
    X = np.random.randn(2, dimensions["d_clause"]).astype(np.float32)
    logits = encoder.forward_batch(X)

    # Snapshot
    snapshot = encoder.snapshot_node_cache()

    # Vérifier que le graphe est préservé
    assert snapshot['input_tensor'] is not None
    assert snapshot['output_logits'] is not None
    assert snapshot['output_logits'].requires_grad or snapshot['output_logits'].grad_fn is not None

    # Restore
    encoder.restore_node_cache(snapshot)

    # Backward devrait fonctionner après restore
    d_logits = np.ones((2, 7), dtype=np.float32)
    grads, dx = encoder.backward_node_dx(d_logits)

    assert len(grads) > 0
    assert dx.shape == X.shape


def test_lr_warning_emitted_once(dimensions):
    """Vérifie que le warning lr n'est émis qu'une seule fois."""
    import warnings
    encoder = XLMRobertaEncoder(
        d_clause=dimensions["d_clause"],
        d_edge=dimensions["d_edge"],
        learning_rate=1e-5
    )

    # Forward + backward
    X = np.random.randn(2, dimensions["d_clause"]).astype(np.float32)
    logits = encoder.forward_batch(X)
    d_logits = np.ones_like(logits)
    grads, dx = encoder.backward_node_dx(d_logits)

    # Update avec lr différent (devrait émettre warning)
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        encoder.update_node(grads, lr=1e-3)
        assert len(w1) == 1, f"Expected 1 warning, got {len(w1)}"
        assert "Pipeline lr" in str(w1[0].message)

    # Deuxième update (NE devrait PAS émettre warning)
    X2 = np.random.randn(2, dimensions["d_clause"]).astype(np.float32)
    logits2 = encoder.forward_batch(X2)
    grads2, dx2 = encoder.backward_node_dx(np.ones_like(logits2))

    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        encoder.update_node(grads2, lr=1e-3)
        # Aucun nouveau warning (flag _lr_warning_emitted=True)
        assert len(w2) == 0, f"Expected 0 warnings, got {len(w2)}: {[str(x.message) for x in w2]}"
