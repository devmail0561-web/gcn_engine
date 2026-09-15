"""
R-GCN PyTorch — implémentation haute performance de la couche 3.

Remplace RGCNLayer (NumPy référence) par une implémentation PyTorch optimisée :
- GPU / MPS support via device placement
- Gradient automatique (autograd) pour la rétropropagation
- Scatter-add vectorisé (pas de boucle Python par relation)
- Compatible avec le Protocol CausalGraph de layer3/interface.py

Usage (à la place de RGCNLayer) :
    from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
    layer = RGCNLayerPT(d_in=64, d_out=128)
    out = layer.message_pass(node_feat, edge_index, edge_types)

Intégration dans une boucle d'entraînement PyTorch standard :
    optimizer = torch.optim.Adam(layer.torch_parameters(), lr=1e-3)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()

Note : torch est une dépendance optionnelle. Si PyTorch n'est pas installé,
l'import de ce module lève ImportError avec un message explicite.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch est requis pour RGCNLayerPT. "
        "Installez-le avec : pip install torch  "
        "(voir https://pytorch.org/get-started/locally/ pour les options GPU)"
    ) from e

import numpy as np
from typing import Any
from ..constants import RELATION_TYPES


class RGCNLayerPT(nn.Module):
    """
    Couche R-GCN PyTorch.

    Formule :
      h_i^(out) = σ( Σ_r (1/c_{i,r}) Σ_{j∈N_r(i)} W_r h_j  +  W_0 h_i )

    Implémentée avec scatter_add_ vectorisé pour éviter les boucles Python
    par relation. Supportant nativement les cycles.

    Compatible avec le Protocol CausalGraph : message_pass / parameters / update.
    Les utilisateurs PyTorch utiliseront de préférence torch_parameters() et
    un optimizer PyTorch standard plutôt que update() (qui reste disponible
    pour la compatibilité avec CGNPipeline).
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

        # Relation-specific weights: (R, D_out, D_in)
        self.W_r = nn.Parameter(
            torch.empty(self.n_relations, d_out, d_in, device=self._device).normal_(0, scale)
        )
        # Self-loop weight: (D_out, D_in)
        self.W_0 = nn.Parameter(
            torch.empty(d_out, d_in, device=self._device).normal_(0, scale)
        )

    # ------------------------------------------------------------------
    # CausalGraph Protocol — message_pass
    # ------------------------------------------------------------------

    def message_pass(
        self,
        node_features: np.ndarray,  # (N, D_in)
        edge_index: np.ndarray,     # (2, E)
        edge_types: np.ndarray,     # (E,) int
    ) -> np.ndarray:                # (N, D_out)
        """Passe les messages sur le graphe causal et retourne les représentations enrichies."""
        H = torch.as_tensor(node_features, dtype=torch.float32, device=self._device)
        out = self._forward_pt(H, edge_index, edge_types)
        return out.detach().cpu().numpy()

    def _forward_pt(
        self,
        H: torch.Tensor,           # (N, D_in)
        edge_index: np.ndarray,
        edge_types: np.ndarray,
    ) -> torch.Tensor:             # (N, D_out)
        """Version PyTorch native (pour l'entraînement avec autograd)."""
        N = H.shape[0]

        # Self-loop : H @ W_0^T
        out = H @ self.W_0.t()  # (N, D_out)

        if edge_index.shape[1] > 0:
            src = torch.as_tensor(edge_index[0], dtype=torch.long, device=self._device)
            dst = torch.as_tensor(edge_index[1], dtype=torch.long, device=self._device)
            rtypes = torch.as_tensor(edge_types, dtype=torch.long, device=self._device)

            # Pour chaque type de relation, scatter_add vectorisé
            for r in range(self.n_relations):
                mask = rtypes == r
                if not mask.any():
                    continue
                src_r = src[mask]   # arêtes de type r
                dst_r = dst[mask]

                # Comptage des voisins par destination (normalisation)
                counts = torch.zeros(N, device=self._device, dtype=torch.float32)
                counts.scatter_add_(0, dst_r, torch.ones_like(dst_r, dtype=torch.float32))
                counts = counts.clamp(min=1.0)

                # Messages : (|E_r|, D_out) = H[src_r] @ W_r[r]^T
                msgs = H[src_r] @ self.W_r[r].t()

                # Aggrégation dans les nœuds destination
                agg = torch.zeros(N, self.d_out, device=self._device)
                agg.scatter_add_(0, dst_r.unsqueeze(1).expand_as(msgs), msgs)
                out = out + agg / counts.unsqueeze(1)

        return torch.sigmoid(out)

    # ------------------------------------------------------------------
    # CausalGraph Protocol — parameters / update (compatibilité NumPy)
    # ------------------------------------------------------------------

    def parameters(self) -> list[np.ndarray]:  # type: ignore[override]
        """Retourne les poids sous forme NumPy (compatibilité Protocol)."""
        return [
            self.W_r.detach().cpu().numpy(),
            self.W_0.detach().cpu().numpy(),
        ]

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        """Mise à jour manuelle des poids (gradient descent numpy).
        Les utilisateurs PyTorch préféreront optimizer.step() via torch_parameters().
        """
        with torch.no_grad():
            self.W_r -= lr * torch.as_tensor(grads[0], dtype=torch.float32, device=self._device)
            self.W_0 -= lr * torch.as_tensor(grads[1], dtype=torch.float32, device=self._device)

    # ------------------------------------------------------------------
    # API PyTorch native
    # ------------------------------------------------------------------

    def torch_parameters(self) -> list[nn.Parameter]:
        """Retourne les paramètres PyTorch pour un optimizer standard."""
        return list(super().parameters())

    def forward_torch(
        self,
        H: torch.Tensor,
        edge_index: np.ndarray,
        edge_types: np.ndarray,
    ) -> torch.Tensor:
        """Forward pass PyTorch natif, conserve le graphe de calcul pour backward()."""
        return self._forward_pt(H, edge_index, edge_types)

    def to_device(self, device: str | torch.device) -> "RGCNLayerPT":
        self._device = torch.device(device)
        return self.to(self._device)

    def __repr__(self) -> str:
        return (
            f"RGCNLayerPT(d_in={self.d_in}, d_out={self.d_out}, "
            f"n_relations={self.n_relations}, device={self._device})"
        )
