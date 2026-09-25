# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Contrat update_node()/update_edge() et backward_*_dx() (audit 2026-09-25).

Couvre les trois défauts corrigés :
1. backward_node_dx/backward_edge_dx retournent la contribution de l'APPEL
   (delta) — le pipeline somme les retours sur N nœuds, un cumul serait
   double-compté.
2. update_node() applique réellement l'argument `grads` (il était ignoré).
3. la normalisation par n_samples du pipeline est propagée à TOUS les
   paramètres torch, pas seulement aux têtes.
"""
import warnings

import numpy as np
import torch

from gcn_transformers import XLMRobertaEncoder


def _node_linear_layers(encoder):
    return [m for m in encoder._node_head if isinstance(m, torch.nn.Linear)]


def _accumulate_over_nodes(encoder, X, n_nodes):
    """Boucle identique à cgnp.py:869-890 (forward + backward par nœud)."""
    all_grads = None
    for i in range(n_nodes):
        encoder.forward_node(X[i])
        grads_i, _ = encoder.backward_node_dx(np.ones(7, dtype=np.float32))
        if all_grads is None:
            all_grads = [(dW.copy(), db.copy()) for dW, db in grads_i]
        else:
            for j, (dW_i, db_i) in enumerate(grads_i):
                all_grads[j] = (all_grads[j][0] + dW_i, all_grads[j][1] + db_i)
    return all_grads


def test_backward_node_dx_returns_delta_not_cumul(dimensions):
    """La somme des retours doit valoir les .grad autograd (pas de double comptage)."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])
    encoder.optimizer.zero_grad()

    n_nodes = 3
    X = np.random.randn(n_nodes, dimensions["d_clause"]).astype(np.float32)
    all_grads = _accumulate_over_nodes(encoder, X, n_nodes)

    layer = _node_linear_layers(encoder)[0]
    assert layer.weight.grad is not None
    np.testing.assert_allclose(
        layer.weight.grad.numpy(), all_grads[0][0], rtol=1e-4, atol=1e-6,
        err_msg="la somme des retours doit égaler les .grad autograd",
    )


def test_backward_edge_dx_returns_delta(dimensions):
    """Delta edge : deux backward successifs doivent s'additionner proprement."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])
    encoder.optimizer.zero_grad()

    layer = [m for m in encoder._edge_head if isinstance(m, torch.nn.Linear)][0]
    total = None
    for _ in range(2):
        x = np.random.randn(dimensions["d_edge"]).astype(np.float32)
        logits = encoder.forward_edge(x)
        grads_i, dx = encoder.backward_edge_dx(np.ones_like(logits))
        assert dx.shape == (dimensions["d_edge"],)
        total = grads_i if total is None else [
            (a + b, c + d) for (a, c), (b, d) in zip(total, grads_i)
        ]

    np.testing.assert_allclose(
        layer.weight.grad.numpy(), total[0][0], rtol=1e-4, atol=1e-6
    )


def test_update_node_applies_provided_grads(dimensions):
    """update_node() doit appliquer l'argument `grads` (contrat CausalEncoder)."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])
    encoder.optimizer.zero_grad()

    X = np.random.randn(2, dimensions["d_clause"]).astype(np.float32)
    logits = encoder.forward_batch(X)
    grads, _ = encoder.backward_node_dx(np.ones_like(logits))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder.update_node(grads, lr=encoder.optimizer.param_groups[0]["lr"])

    layer = _node_linear_layers(encoder)[0]
    np.testing.assert_allclose(
        layer.weight.grad.numpy(), grads[0][0], rtol=1e-5, atol=1e-7,
        err_msg="update_node doit recopier les gradients fournis dans .grad",
    )
    assert encoder._needs_zero_grad is True


def test_update_node_normalizes_all_params_by_n_samples(dimensions):
    """La normalisation /n_samples doit toucher le backbone ET les têtes."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])
    encoder.optimizer.zero_grad()

    n_samples = 3
    X = np.random.randn(n_samples, dimensions["d_clause"]).astype(np.float32)
    all_grads = _accumulate_over_nodes(encoder, X, n_samples)

    # Paramètre hors node_head (normalement non normalisé par le pipeline)
    backbone = encoder.proj_ud.weight
    assert backbone.grad is not None
    backbone_before = backbone.grad.numpy().copy()
    normalized = [(dW / n_samples, db / n_samples) for dW, db in all_grads]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder.update_node(normalized, lr=encoder.optimizer.param_groups[0]["lr"])

    layer = _node_linear_layers(encoder)[0]
    np.testing.assert_allclose(
        layer.weight.grad.numpy(), normalized[0][0], rtol=1e-5, atol=1e-7
    )

    mask = np.abs(backbone_before) > 0
    assert mask.any(), "proj_ud doit recevoir un gradient non nul"
    scale = np.median(
        backbone.grad.numpy()[mask] / backbone_before[mask]
    )
    assert abs(1.0 / scale - n_samples) < 1e-3, (
        f"backbone mis à l'échelle {1.0 / scale:.4f}, attendu {n_samples}"
    )


def test_update_node_rejects_malformed_grads(dimensions):
    """Format invalide : on conserve les .grad autograd et on warningne une fois."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])
    encoder.optimizer.zero_grad()

    X = np.random.randn(2, dimensions["d_clause"]).astype(np.float32)
    logits = encoder.forward_batch(X)
    encoder.backward_node_dx(np.ones_like(logits))
    layer = _node_linear_layers(encoder)[0]
    autograd_grad = layer.weight.grad.numpy().copy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder.update_node(None, lr=encoder.optimizer.param_groups[0]["lr"])
        encoder.update_node([np.zeros(1)], lr=encoder.optimizer.param_groups[0]["lr"])

    contract_warnings = [w for w in caught if "incomplet" in str(w.message)]
    assert len(contract_warnings) == 1, "un seul warning de contrat"
    np.testing.assert_allclose(layer.weight.grad.numpy(), autograd_grad)


def test_update_edge_detects_incoherent_grads(dimensions):
    """update_edge() contrôle les gradients fournis (au lieu de les ignorer)."""
    encoder = XLMRobertaEncoder(d_clause=dimensions["d_clause"], d_edge=dimensions["d_edge"])
    encoder.optimizer.zero_grad()

    X = np.random.randn(2, dimensions["d_clause"]).astype(np.float32)
    logits = encoder.forward_batch(X)
    grads, _ = encoder.backward_node_dx(np.ones_like(logits))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder.update_node(grads, lr=encoder.optimizer.param_groups[0]["lr"])

    x_edge = np.random.randn(dimensions["d_edge"]).astype(np.float32)
    logits_edge = encoder.forward_edge(x_edge)
    grads_edge, _ = encoder.backward_edge_dx(np.ones_like(logits_edge))

    # Gradients cohérents (ceux réellement accumulés) → aucun warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder.update_edge(grads_edge, lr=encoder.optimizer.param_groups[0]["lr"])
    assert not [w for w in caught if "incomplet" in str(w.message)]
    assert encoder._needs_zero_grad is False

    # Gradients incohérents (valeurs arbitraires) → warning une seule fois
    encoder._grad_contract_warned = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder.update_node(grads, lr=encoder.optimizer.param_groups[0]["lr"])
    encoder.forward_edge(x_edge)
    encoder.backward_edge_dx(np.ones_like(logits_edge))
    fake = [(np.zeros_like(dW), np.zeros_like(db)) for dW, db in grads_edge]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder.update_edge(fake, lr=encoder.optimizer.param_groups[0]["lr"])
        encoder.update_edge(fake, lr=encoder.optimizer.param_groups[0]["lr"])
    incoherent = [w for w in caught if "incohérents" in str(w.message)]
    assert len(incoherent) == 1, "le warning doit n'être émis qu'une seule fois"
    assert encoder._needs_zero_grad is False
