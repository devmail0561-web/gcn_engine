# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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
        pairnorm: bool = False,
        drop_edge: float = 0.0,
        use_compgcn: bool = False,
        d_rel_emb: int = 32,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_relations = n_relations or len(RELATION_TYPES)
        self.pairnorm = bool(pairnorm)
        self.drop_edge = float(drop_edge)
        if not (0.0 <= self.drop_edge < 1.0):
            raise ValueError(
                f"drop_edge doit être dans [0, 1[ (reçu {drop_edge!r})."
            )
        self.use_compgcn = bool(use_compgcn)
        self.d_rel_emb = int(d_rel_emb)
        if self.use_compgcn and self.d_rel_emb < 1:
            raise ValueError(
                f"d_rel_emb={d_rel_emb} doit être >= 1 quand use_compgcn=True."
            )

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self._device = torch.device(device)

        # M3 : générateur local — ne pollue plus le seed global torch.
        _gen = torch.Generator(device="cpu")
        _gen.manual_seed(seed)
        scale = (2.0 / d_in) ** 0.5

        # Relation-specific weights: (R, D_out, D_in)
        if self.use_compgcn:
            # Phase D (CompGCN) : E_r ∈ R^{n_rel × d_rel}, W_effective = Linear(e_r).
            self.E_r = nn.Parameter(
                torch.randn(self.n_relations, self.d_rel_emb, generator=_gen,
                            dtype=torch.float32, device=self._device) * 0.01
            )
            self.W_comp = nn.Parameter(
                torch.randn(self.d_rel_emb, d_out * d_in, generator=_gen,
                            dtype=torch.float32, device=self._device) * scale
            )
            # W_r conservé comme dummy non-Parameter pour que load_state() reste robuste.
            self.register_buffer("W_r", torch.zeros(self.n_relations, d_out, d_in))
        else:
            self.W_r = nn.Parameter(
                torch.randn(self.n_relations, d_out, d_in, generator=_gen, dtype=torch.float32, device=self._device) * scale
            )
        # Self-loop weight: (D_out, D_in)
        self.W_0 = nn.Parameter(
            torch.randn(d_out, d_in, generator=_gen, dtype=torch.float32, device=self._device) * scale
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
        if node_features.ndim != 2 or node_features.shape[1] != self.d_in:
            raise ValueError(
                f"RGCNLayerPT.message_pass : node_features.shape={node_features.shape} "
                f"incompatible avec d_in={self.d_in}."
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                f"RGCNLayerPT.message_pass : edge_index.shape={edge_index.shape} (attendu (2, E))."
            )
        if edge_index.shape[1] > 0 and (
            int(edge_types.min()) < 0 or int(edge_types.max()) >= self.n_relations
        ):
            raise ValueError(
                f"RGCNLayerPT.message_pass : edge_types hors bornes [0, {self.n_relations}[."
            )
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

        # Phase B (DropEdge) : masque NumPy avant conversion tensor
        # (edge_index/edge_types arrivent comme np.ndarray ici).
        # Appliqué en une seule expression (atomique), en train uniquement.
        if self.training and self.drop_edge > 0.0:
            mask = np.random.rand(edge_index.shape[1]) > self.drop_edge
            edge_index, edge_types = edge_index[:, mask], edge_types[mask]

        # Phase D (CompGCN) : W_r effectif depuis E_r @ W_comp.
        if self.use_compgcn:
            W_r = (self.E_r @ self.W_comp).view(self.n_relations, self.d_out, self.d_in)
        else:
            W_r = self.W_r

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
                msgs = H[src_r] @ W_r[r].t()

                # Aggrégation dans les nœuds destination
                agg = torch.zeros(N, self.d_out, device=self._device)
                agg.scatter_add_(0, dst_r.unsqueeze(1).expand_as(msgs), msgs)
                out = out + agg / counts.unsqueeze(1)

        # Phase B (PairNorm) : recentrage + normalisation L2 par nœud,
        # avant activation (anti-over-smoothing, couches profondes).
        if self.pairnorm:
            out = out - out.mean(dim=0, keepdim=True)
            out = out / (out.norm(dim=1, keepdim=True) + 1e-8)

        return torch.sigmoid(out)

    # ------------------------------------------------------------------
    # CausalGraph Protocol — parameters / update (compatibilité NumPy)
    # ------------------------------------------------------------------

    def parameters(self) -> list[np.ndarray]:  # type: ignore[override]
        """Retourne les poids sous forme NumPy (compatibilité Protocol)."""
        if self.use_compgcn:
            return [
                self.E_r.detach().cpu().numpy(),
                self.W_comp.detach().cpu().numpy(),
                self.W_0.detach().cpu().numpy(),  # FIX-1 : W_0 toujours entraînable
            ]
        return [
            self.W_r.detach().cpu().numpy(),
            self.W_0.detach().cpu().numpy(),
        ]

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        """Mise à jour manuelle des poids (gradient descent numpy).
        Les utilisateurs PyTorch préféreront optimizer.step() via torch_parameters().
        """
        with torch.no_grad():
            if self.use_compgcn:
                self.E_r    -= lr * torch.as_tensor(grads[0], dtype=torch.float32, device=self._device)
                self.W_comp -= lr * torch.as_tensor(grads[1], dtype=torch.float32, device=self._device)
                self.W_0    -= lr * torch.as_tensor(grads[2], dtype=torch.float32, device=self._device)  # FIX-1
            else:
                self.W_r -= lr * torch.as_tensor(grads[0], dtype=torch.float32, device=self._device)
                self.W_0 -= lr * torch.as_tensor(grads[1], dtype=torch.float32, device=self._device)

    def load_state(self, arrays: list[np.ndarray]) -> None:
        """
        Charge les poids depuis des arrays NumPy (utilisé par checkpoint.load).

        H5 correction : parameters() retourne des copies détachées, load_checkpoint
        écrivait dans ces copies. Cette méthode copie directement dans les tenseurs
        PyTorch W_r et W_0.
        """
        # C1.2 : garde explicite — 2 arrays attendus (W_r, W_0), sinon IndexError
        # cryptique sur arrays[1]. Avec CompGCN : [E_r, W_comp].
        if len(arrays) < 2:
            raise ValueError(
                f"load_state : 2 arrays attendus (W_r, W_0), reçu {len(arrays)}."
            )
        if self.use_compgcn:
            # arrays = [E_r, W_comp, W_0]  (FIX-1 : W_0 inclus)
            if len(arrays) < 3:
                raise ValueError(
                    f"load_state (CompGCN) : 3 arrays attendus (E_r, W_comp, W_0), reçu {len(arrays)}."
                )
            e_r = np.asarray(arrays[0])
            w_c = np.asarray(arrays[1])
            w0  = np.asarray(arrays[2])
            if e_r.shape != (self.n_relations, self.d_rel_emb):
                raise ValueError(
                    f"E_r shape incompatible : {e_r.shape} "
                    f"attendu ({self.n_relations}, {self.d_rel_emb})"
                )
            if w_c.shape != (self.d_rel_emb, self.d_out * self.d_in):
                raise ValueError(
                    f"W_comp shape incompatible : {w_c.shape} "
                    f"attendu ({self.d_rel_emb}, {self.d_out * self.d_in})"
                )
            if w0.shape != (self.d_out, self.d_in):
                raise ValueError(
                    f"W_0 shape incompatible (CompGCN) : {w0.shape} "
                    f"attendu ({self.d_out}, {self.d_in})"
                )
            with torch.no_grad():
                self.E_r.copy_(torch.as_tensor(e_r, dtype=torch.float32, device=self._device))
                self.W_comp.copy_(torch.as_tensor(w_c, dtype=torch.float32, device=self._device))
                self.W_0.copy_(torch.as_tensor(w0, dtype=torch.float32, device=self._device))
            return
        if arrays[0].shape != (self.n_relations, self.d_out, self.d_in):
            raise ValueError(
                f"W_r shape incompatible : {arrays[0].shape} "
                f"attendu ({self.n_relations}, {self.d_out}, {self.d_in})"
            )
        # C1.2 : check W_0 shape (symétrique au check GAT) — sinon crash torch
        # cryptique sur copy_ au lieu d'un ValueError clair.
        w0 = np.asarray(arrays[1])
        if w0.shape != (self.d_out, self.d_in):
            raise ValueError(
                f"W_0 shape incompatible : {w0.shape} "
                f"attendu ({self.d_out}, {self.d_in})"
            )
        with torch.no_grad():
            self.W_r.copy_(torch.as_tensor(arrays[0], dtype=torch.float32, device=self._device))
            self.W_0.copy_(torch.as_tensor(w0, dtype=torch.float32, device=self._device))

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
