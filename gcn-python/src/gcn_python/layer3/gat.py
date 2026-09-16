"""
R-GCN + GAT — couche 3 avec attention par relation.

Comme RGCNLayerPT mais ajoute des vecteurs d'attention par relation
permettant au modèle de pondérer dynamiquement l'importance de chaque
voisin dans le message passing.

Formule (par relation r) :
  msg_src = H[src] @ W_r[r].T
  msg_dst = H[dst] @ W_r[r].T
  e_ij    = LeakyReLU([msg_src || msg_dst] @ a_r[r])
  α_ij    = softmax(e_ij) par nœud destination
  out    += scatter_add(α_ij * msg_src, dst)

Compatible avec le Protocol CausalGraph de layer3/interface.py.
Utilise torch.autograd pour la rétropropagation.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch est requis pour RGCNLayerGAT. "
        "Installez-le avec : pip install torch"
    ) from e

import numpy as np
from ..constants import RELATION_TYPES


def _softmax_per_dst(
    e_ij: torch.Tensor, dst: torch.Tensor, N: int,
) -> torch.Tensor:
    """Softmax par nœud destination, sans torch_scatter.

    Utilise scatter_reduce (fonctionnel, pas in-place) pour max et sum.
    Le max est détaché du graphe autograd — la soustraction max sert
    uniquement à la stabilité numérique et ne doit pas affecter les gradients.
    Le dénominateur est également détaché — standard GAT.
    """
    init_max = torch.full((N,), float('-inf'), device=e_ij.device)
    e_max = init_max.scatter_reduce(0, dst, e_ij, reduce='amax', include_self=True)
    e_shifted = e_ij - e_max[dst].detach()
    exp_e = torch.exp(e_shifted)
    init_sum = torch.zeros(N, device=e_ij.device)
    sum_exp = init_sum.scatter_reduce(0, dst, exp_e, reduce='sum', include_self=True)
    return exp_e / sum_exp[dst].detach().clamp(min=1e-9)


class RGCNLayerGAT(nn.Module):
    """
    Couche R-GCN avec attention GAT par relation.

    Paramètres :
      W_r  : (n_relations, d_out, d_in)  — poids de message par relation
      W_0  : (d_out, d_in)               — poids self-loop
      a_r  : (n_relations, 2 * d_out)    — vecteurs d'attention par relation
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_relations: int | None = None,
        device: str | torch.device | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_relations = n_relations or len(RELATION_TYPES)

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self._device = torch.device(device)

        torch.manual_seed(seed)
        scale = (2.0 / d_in) ** 0.5

        # Poids de message par relation : (R, D_out, D_in)
        self.W_r = nn.Parameter(
            torch.empty(self.n_relations, d_out, d_in, device=self._device).normal_(0, scale)
        )
        # Poids self-loop : (D_out, D_in)
        self.W_0 = nn.Parameter(
            torch.empty(d_out, d_in, device=self._device).normal_(0, scale)
        )
        # Vecteurs d'attention par relation : (R, 2 * D_out)
        self.a_r = nn.Parameter(
            torch.empty(self.n_relations, 2 * d_out, device=self._device).normal_(0, scale)
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Cache pour backward_message_pass
        self._H_in_retained: torch.Tensor | None = None
        self._out_retained: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # GAT forward interne (sur tenseurs PyTorch)
    # ------------------------------------------------------------------

    def _gat_forward(
        self,
        H: torch.Tensor,           # (N, D_in) avec grad
        edge_index: np.ndarray,
        edge_types: np.ndarray,
    ) -> torch.Tensor:             # (N, D_out) — raw logits (pas de sigmoid)
        N = H.shape[0]

        # Self-loop
        out = H @ self.W_0.t()     # (N, D_out)

        if edge_index.shape[1] > 0:
            src = torch.as_tensor(edge_index[0], dtype=torch.long, device=self._device)
            dst = torch.as_tensor(edge_index[1], dtype=torch.long, device=self._device)
            rtypes = torch.as_tensor(edge_types, dtype=torch.long, device=self._device)

            for r in range(self.n_relations):
                mask = rtypes == r
                if not mask.any():
                    continue
                src_r = src[mask]
                dst_r = dst[mask]

                # Messages source et destination
                msg_src = H[src_r] @ self.W_r[r].t()   # (|E_r|, D_out)
                msg_dst = H[dst_r] @ self.W_r[r].t()   # (|E_r|, D_out)

                # Scores d'attention : LeakyReLU([msg_src || msg_dst] @ a_r[r])
                concat = torch.cat([msg_src, msg_dst], dim=1)  # (|E_r|, 2*D_out)
                e_ij = self.leaky_relu(concat @ self.a_r[r])   # (|E_r|,)

                # Softmax par nœud destination (stabilité numérique)
                alpha = _softmax_per_dst(e_ij, dst_r, N)

                # Agrégation pondérée — requires_grad=True pour garder le graphe autograd
                weighted_msgs = alpha.unsqueeze(1) * msg_src
                agg = torch.zeros(N, self.d_out, device=self._device, requires_grad=True)
                agg = agg.scatter_add(0, dst_r.unsqueeze(1).expand_as(weighted_msgs), weighted_msgs)
                out = out + agg

        return out  # raw logits — sigmoid appliqué dans message_pass()

    # ------------------------------------------------------------------
    # CausalGraph Protocol — message_pass
    # ------------------------------------------------------------------

    def message_pass(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_types: np.ndarray,
    ) -> np.ndarray:
        """Passe les messages avec attention et retourne les représentations enrichies."""
        H_in = torch.as_tensor(node_features, dtype=torch.float32, device=self._device)
        H_in.requires_grad_(True)
        out = self._gat_forward(H_in, edge_index, edge_types)
        self._H_in_retained = H_in
        self._out_retained = out
        return torch.sigmoid(out).detach().cpu().numpy()

    def backward_message_pass(
        self,
        d_output: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Rétropropagation via autograd. Retourne (d_input, [dW_r, dW_0, da_r]).

        message_pass() retourne sigmoid(out), mais _out_retained contient out
        (pré-sigmoid). On doit appliquer la dérivée de sigmoid :
          d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        pour convertir les gradients post-sigmoid en gradients pré-sigmoid.
        """
        assert self._H_in_retained is not None, "backward_message_pass appelé avant message_pass"
        assert self._out_retained is not None, "backward_message_pass appelé avant message_pass"

        out = self._out_retained
        sig = torch.sigmoid(out)
        sigmoid_deriv = sig * (1.0 - sig)

        d_out_t = torch.as_tensor(d_output, dtype=torch.float32, device=self._device)
        d_pre_sigmoid = d_out_t * sigmoid_deriv

        grads = torch.autograd.grad(
            out,
            [self._H_in_retained, self.W_r, self.W_0, self.a_r],
            grad_outputs=d_pre_sigmoid,
            retain_graph=False,
        )
        d_input = grads[0].detach().cpu().numpy()
        return d_input, [g.detach().cpu().numpy() for g in grads[1:]]

    # ------------------------------------------------------------------
    # CausalGraph Protocol — parameters / update (compatibilité NumPy)
    # ------------------------------------------------------------------

    def parameters(self) -> list[np.ndarray]:
        return [
            self.W_r.detach().cpu().numpy(),
            self.W_0.detach().cpu().numpy(),
            self.a_r.detach().cpu().numpy(),
        ]

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        """Mise à jour manuelle des poids (gradient descent numpy)."""
        with torch.no_grad():
            self.W_r -= lr * torch.as_tensor(grads[0], dtype=torch.float32, device=self._device)
            self.W_0 -= lr * torch.as_tensor(grads[1], dtype=torch.float32, device=self._device)
            if len(grads) > 2:
                self.a_r -= lr * torch.as_tensor(grads[2], dtype=torch.float32, device=self._device)

    def load_state(self, arrays: list[np.ndarray]) -> None:
        """Charge les poids depuis des arrays NumPy. Vérifie la shape."""
        if arrays[0].shape != (self.n_relations, self.d_out, self.d_in):
            raise ValueError(
                f"W_r shape incompatible : {arrays[0].shape} "
                f"attendu ({self.n_relations}, {self.d_out}, {self.d_in})"
            )
        with torch.no_grad():
            self.W_r.copy_(torch.as_tensor(arrays[0], dtype=torch.float32, device=self._device))
            self.W_0.copy_(torch.as_tensor(arrays[1], dtype=torch.float32, device=self._device))
            if len(arrays) > 2:
                self.a_r.copy_(torch.as_tensor(arrays[2], dtype=torch.float32, device=self._device))

    # ------------------------------------------------------------------
    # API PyTorch native
    # ------------------------------------------------------------------

    def torch_parameters(self) -> list[nn.Parameter]:
        return list(super().parameters())

    def forward_torch(
        self,
        H: torch.Tensor,
        edge_index: np.ndarray,
        edge_types: np.ndarray,
    ) -> torch.Tensor:
        return self._gat_forward(H, edge_index, edge_types)

    def to_device(self, device: str | torch.device) -> "RGCNLayerGAT":
        self._device = torch.device(device)
        return self.to(self._device)

    def __repr__(self) -> str:
        return (
            f"RGCNLayerGAT(d_in={self.d_in}, d_out={self.d_out}, "
            f"n_relations={self.n_relations}, device={self._device})"
        )
