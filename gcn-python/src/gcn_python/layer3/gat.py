# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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
    Le dénominateur reste dans le graphe autograd pour un gradient correct.
    """
    init_max = torch.full((N,), float('-inf'), device=e_ij.device)
    e_max = init_max.scatter_reduce(0, dst, e_ij, reduce='amax', include_self=True)
    e_shifted = e_ij - e_max[dst].detach()
    exp_e = torch.exp(e_shifted)
    init_sum = torch.zeros(N, device=e_ij.device)
    sum_exp = init_sum.scatter_reduce(0, dst, exp_e, reduce='sum', include_self=True)
    return exp_e / sum_exp[dst].clamp(min=1e-9)


class RGCNLayerGAT(nn.Module):
    """
    Couche R-GCN avec attention GAT par relation (Amélioration D : multi-tête).

    Paramètres :
      W_r  : (n_relations, d_out, d_in)  — poids de message par relation
      W_0  : (d_out, d_in)               — poids self-loop
      a_r  : (n_heads, n_relations, 2 * d_head) — vecteurs d'attention (D)
             d_head = d_out // n_heads. n_heads=1 : (1, R, 2*D).
             load_state() accepte l'ancien format (R, 2*D) si n_heads=1.
      norm : nn.LayerNorm(d_out) optionnel (E2, use_layernorm=True).

    output_activation (E1) : "sigmoid" (défaut, rétrocompatible), "relu"
    (couches intermédiaires empilées) ou "none".
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_relations: int | None = None,
        device: str | torch.device | None = None,
        seed: int = 42,
        dropout: float = 0.0,
        n_heads: int = 1,
        output_activation: str = "sigmoid",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_relations = n_relations or len(RELATION_TYPES)
        self.dropout_rate = dropout
        self.n_heads = n_heads
        self.output_activation = output_activation
        self.use_layernorm = use_layernorm
        if output_activation not in ("sigmoid", "relu", "none"):
            raise ValueError(
                f"output_activation inconnu : {output_activation!r} "
                "(attendu 'sigmoid', 'relu' ou 'none')."
            )
        # Amélioration D — contrainte de divisibilité (CORRECTION défaut #1)
        if d_out % n_heads != 0:
            compatibles = [n for n in [1, 2, 4, 8] if d_out % n == 0]
            raise ValueError(
                f"d_out={d_out} non divisible par n_heads={n_heads}. "
                f"n_heads compatibles : {compatibles}. "
                "Avec d_emb=49 : d_out=128 compatible avec n_heads ∈ {1,2,4,8}."
            )
        self.d_head = d_out // n_heads

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

        # Poids de message par relation : (R, D_out, D_in)
        self.W_r = nn.Parameter(
            torch.randn(self.n_relations, d_out, d_in, generator=_gen, dtype=torch.float32, device=self._device) * scale
        )
        # Poids self-loop : (D_out, D_in)
        self.W_0 = nn.Parameter(
            torch.randn(d_out, d_in, generator=_gen, dtype=torch.float32, device=self._device) * scale
        )
        # Vecteurs d'attention par tête et relation : (H, R, 2 * D_head)
        self.a_r = nn.Parameter(
            torch.randn(self.n_heads, self.n_relations, 2 * self.d_head, generator=_gen, dtype=torch.float32, device=self._device) * scale
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        # E2 : LayerNorm optionnelle (identité à l'init : weight=1, bias=0)
        self.norm = nn.LayerNorm(d_out).to(self._device) if use_layernorm else None

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
    ) -> torch.Tensor:             # (N, D_out) — raw logits (pré-activation)
        N = H.shape[0]

        # Amélioration D : chaque tête h accède à la tranche
        # [h*d_head:(h+1)*d_head] de la sortie. W_r inchangé (R, D_out, D_in).
        out_heads = []
        for h_idx in range(self.n_heads):
            s, e = h_idx * self.d_head, (h_idx + 1) * self.d_head
            out_h = H @ self.W_0[s:e].t()   # self-loop tête h

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

                    # Messages source et destination (tranche tête h)
                    msg_src = H[src_r] @ self.W_r[r][s:e].t()   # (|E_r|, d_head)
                    msg_dst = H[dst_r] @ self.W_r[r][s:e].t()

                    # Scores d'attention : LeakyReLU([msg_src || msg_dst] @ a_r[h, r])
                    concat = torch.cat([msg_src, msg_dst], dim=1)  # (|E_r|, 2*d_head)
                    e_ij = self.leaky_relu(concat @ self.a_r[h_idx, r])   # (|E_r|,)

                    # Softmax par nœud destination (stabilité numérique)
                    alpha = _softmax_per_dst(e_ij, dst_r, N)

                    weighted_msgs = alpha.unsqueeze(1) * msg_src
                    agg = torch.zeros(N, self.d_head, device=self._device)
                    agg = agg.scatter_add(0, dst_r.unsqueeze(1).expand_as(weighted_msgs), weighted_msgs)
                    out_h = out_h + agg
            out_heads.append(out_h)

        out = torch.cat(out_heads, dim=1)   # (N, d_out)
        # E2 : normalisation après concaténation des têtes
        if self.norm is not None:
            out = self.norm(out)
        return out  # raw logits — activation appliquée dans message_pass()

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
        if node_features.ndim != 2 or node_features.shape[1] != self.d_in:
            raise ValueError(
                f"RGCNLayerGAT.message_pass : node_features.shape={node_features.shape} "
                f"incompatible avec d_in={self.d_in}."
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                f"RGCNLayerGAT.message_pass : edge_index.shape={edge_index.shape} (attendu (2, E))."
            )
        if edge_index.shape[1] > 0 and (
            int(edge_types.min()) < 0 or int(edge_types.max()) >= self.n_relations
        ):
            raise ValueError(
                f"RGCNLayerGAT.message_pass : edge_types hors bornes [0, {self.n_relations}[."
            )
        H_in = torch.as_tensor(node_features, dtype=torch.float32, device=self._device)
        H_in.requires_grad_(True)  # BUG-2 : feuille AVANT dropout — masque inclus dans autograd

        # Dropout sur les features d'entrée
        if self.dropout_rate > 0.0 and self.training:
            mask = torch.bernoulli(
                torch.full(H_in.shape, 1.0 - self.dropout_rate, device=self._device)
            ) / (1.0 - self.dropout_rate)
            H_in = H_in * mask  # opération dans le graphe autograd, gradient correct
        out = self._gat_forward(H_in, edge_index, edge_types)
        self._H_in_retained = H_in
        self._out_retained = out
        # E1 : activation configurable (sigmoid = comportement historique)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(out).detach().cpu().numpy()
        elif self.output_activation == "relu":
            return torch.relu(out).detach().cpu().numpy()
        else:  # "none"
            return out.detach().cpu().numpy()

    def backward_message_pass(
        self,
        d_output: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Rétropropagation via autograd. Retourne (d_input, [dW_r, dW_0, da_r, ...]).

        message_pass() retourne activation(out), mais _out_retained contient out
        (pré-activation). La dérivée de l'activation (E1) est appliquée selon
        output_activation : sigmoid → sig*(1-sig), relu → (out>0), none → 1.
        Avec LayerNorm (E2), les gradients de norm.weight/norm.bias sont
        ajoutés en fin de liste (autograd couvre la nouvelle shape).
        """
        if self._H_in_retained is None or self._out_retained is None:
            raise RuntimeError("backward_message_pass appelé avant message_pass")

        out = self._out_retained
        d_out_t = torch.as_tensor(d_output, dtype=torch.float32, device=self._device)
        # E1 : dérivée conditionnelle
        if self.output_activation == "sigmoid":
            sig = torch.sigmoid(out)
            d_pre = d_out_t * sig * (1.0 - sig)
        elif self.output_activation == "relu":
            d_pre = d_out_t * (out > 0).float()
        else:
            d_pre = d_out_t

        _params: list = [self._H_in_retained, self.W_r, self.W_0, self.a_r]
        if self.norm is not None:
            _params += [self.norm.weight, self.norm.bias]
        try:
            grads = torch.autograd.grad(
                out,
                _params,
                grad_outputs=d_pre,
                retain_graph=False,
            )
            d_input = grads[0].detach().cpu().numpy()
            return d_input, [g.detach().cpu().numpy() for g in grads[1:]]
        finally:
            # M2 : libère le graphe autograd retenu entre deux forwards.
            self._H_in_retained = None
            self._out_retained = None

    # ------------------------------------------------------------------
    # CausalGraph Protocol — parameters / update (compatibilité NumPy)
    # ------------------------------------------------------------------

    def parameters(self) -> list[np.ndarray]:
        # E2 : 5 arrays si LayerNorm (Cas E : graph_0…graph_4 au checkpoint)
        params = [
            self.W_r.detach().cpu().numpy(),
            self.W_0.detach().cpu().numpy(),
            self.a_r.detach().cpu().numpy(),
        ]
        if self.norm is not None:
            params += [
                self.norm.weight.detach().cpu().numpy(),
                self.norm.bias.detach().cpu().numpy(),
            ]
        return params

    def update(self, grads: list[np.ndarray], lr: float) -> None:
        """Mise à jour manuelle des poids (gradient descent numpy)."""
        with torch.no_grad():
            self.W_r -= lr * torch.as_tensor(grads[0], dtype=torch.float32, device=self._device)
            self.W_0 -= lr * torch.as_tensor(grads[1], dtype=torch.float32, device=self._device)
            if len(grads) > 2:
                self.a_r -= lr * torch.as_tensor(grads[2], dtype=torch.float32, device=self._device)
            if self.norm is not None and len(grads) > 4:
                self.norm.weight -= lr * torch.as_tensor(grads[3], dtype=torch.float32, device=self._device)
                self.norm.bias -= lr * torch.as_tensor(grads[4], dtype=torch.float32, device=self._device)

    def load_state(self, arrays: list[np.ndarray]) -> None:
        """Charge les poids depuis des arrays NumPy. Vérifie la shape.

        Cas D (§1) : a_r 2-D = ancien checkpoint mono-tête → unsqueeze si
        n_heads=1, ValueError sinon. a_r 3-D : shape exacte requise.
        Cas E (§1) : 3 arrays + use_layernorm → LayerNorm reste identité ;
        5 arrays → restauration complète.
        """
        if arrays[0].shape != (self.n_relations, self.d_out, self.d_in):
            raise ValueError(
                f"W_r shape incompatible : {arrays[0].shape} "
                f"attendu ({self.n_relations}, {self.d_out}, {self.d_in})"
            )
        a_r = np.asarray(arrays[2])
        if a_r.ndim == 2:  # ancien checkpoint (n_heads=1)
            if self.n_heads != 1:
                raise ValueError(
                    "Checkpoint mono-tête incompatible avec n_heads>1. "
                    "Ré-entraîner ou utiliser --n-gat-heads 1."
                )
            if a_r.shape != (self.n_relations, 2 * self.d_out):
                raise ValueError(
                    f"a_r shape incompatible : {a_r.shape} "
                    f"attendu ({self.n_relations}, {2 * self.d_out})"
                )
            a_r = a_r[np.newaxis, :, :]
        elif a_r.ndim == 3:
            expected = (self.n_heads, self.n_relations, 2 * self.d_head)
            if a_r.shape != expected:
                raise ValueError(
                    f"a_r shape incompatible : {a_r.shape} attendu {expected}."
                )
        else:
            raise ValueError(f"a_r ndim inattendu : {a_r.ndim} (attendu 2 ou 3).")
        with torch.no_grad():
            self.W_r.copy_(torch.as_tensor(arrays[0], dtype=torch.float32, device=self._device))
            self.W_0.copy_(torch.as_tensor(arrays[1], dtype=torch.float32, device=self._device))
            self.a_r.copy_(torch.as_tensor(a_r, dtype=torch.float32, device=self._device))
            if self.norm is not None and len(arrays) > 4:
                self.norm.weight.copy_(torch.as_tensor(arrays[3], dtype=torch.float32, device=self._device))
                self.norm.bias.copy_(torch.as_tensor(arrays[4], dtype=torch.float32, device=self._device))

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
        # M2 : les caches retiennent des tenseurs sur l'ancien device — les invalider.
        self._H_in_retained = None
        self._out_retained = None
        return self.to(self._device)

    def __repr__(self) -> str:
        return (
            f"RGCNLayerGAT(d_in={self.d_in}, d_out={self.d_out}, "
            f"n_relations={self.n_relations}, n_heads={self.n_heads}, "
            f"output_activation={self.output_activation!r}, "
            f"layernorm={self.norm is not None}, device={self._device})"
        )
