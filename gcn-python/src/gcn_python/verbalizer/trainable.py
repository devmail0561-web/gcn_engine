from __future__ import annotations
import json
import re
import numpy as np

from ..layer2.reference import _LinearLayer, _relu, _relu_grad


class SurfaceVocabulary:
    """Word-level vocabulary built from gold surface texts."""

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self) -> None:
        self._t2i: dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self._i2t: list[str] = [self.PAD, self.UNK]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def build(self, surfaces: list[str]) -> None:
        for text in surfaces:
            for tok in self._tokenize(text):
                if tok not in self._t2i:
                    self._t2i[tok] = len(self._i2t)
                    self._i2t.append(tok)

    def encode(self, text: str) -> list[int]:
        return [self._t2i.get(tok, 1) for tok in self._tokenize(text)]

    def decode(self, indices: list[int] | np.ndarray) -> str:
        return " ".join(self._i2t[int(i)] for i in indices if 0 <= int(i) < len(self._i2t))

    def __len__(self) -> int:
        return len(self._i2t)

    def to_json(self) -> str:
        return json.dumps(self._i2t)

    @classmethod
    def from_json(cls, s: str) -> "SurfaceVocabulary":
        v = cls()
        tokens: list[str] = json.loads(s)
        v._i2t = tokens
        v._t2i = {t: i for i, t in enumerate(tokens)}
        return v


class TrainableDecoder:
    """
    Reference NumPy decoder — node embeddings (N, D_in) → surface token logits (|V|,).

    Architecture: mean-pool → fc(D_in → d_hidden) + ReLU → fc(d_hidden → |V|)

    Implements VerbalizerDecoder (inference via decode()) AND the training interface:
    forward_decode / loss_decode / backward_decode / parameters / update.

    Layers are lazily initialized at the first forward_decode call.
    Pass d_in to pre-initialize (required for checkpoint restore).
    """

    def __init__(
        self,
        vocab: SurfaceVocabulary,
        d_hidden: int = 64,
        d_in: int | None = None,
        seed: int = 0,
    ) -> None:
        self.vocab = vocab
        self.d_hidden = d_hidden
        self._rng = np.random.default_rng(seed)
        self._layers: list[_LinearLayer] | None = None
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []
        self._last_d_in: int | None = None
        if d_in is not None:
            self._init_layers(d_in)

    def _init_layers(self, d_in: int) -> None:
        if self._layers is None:
            self._layers = [
                _LinearLayer(d_in, self.d_hidden, self._rng),
                _LinearLayer(self.d_hidden, len(self.vocab), self._rng),
            ]
            self._last_d_in = d_in

    @staticmethod
    def _node_type_embeddings_from_ir(ir_json: str) -> np.ndarray:
        from ..constants import NODE_TYPES
        ir = json.loads(ir_json)
        nodes = ir.get("nodes", [])
        if not nodes:
            return np.zeros((1, len(NODE_TYPES)), dtype=np.float32)
        embs = []
        for node in nodes:
            nt = node.get("node_type", "")
            idx = NODE_TYPES.index(nt) if nt in NODE_TYPES else 0
            onehot = np.zeros(len(NODE_TYPES), dtype=np.float32)
            onehot[idx] = 1.0
            embs.append(onehot)
        return np.stack(embs)  # (N, 7)

    def forward_decode(self, node_embeddings: np.ndarray) -> np.ndarray:
        """(N, D_in) → (|V|,) logits."""
        if len(node_embeddings) == 0:
            return np.zeros(len(self.vocab), dtype=np.float32)
        d_in = node_embeddings.shape[1]
        self._init_layers(d_in)
        assert self._layers is not None
        mean = node_embeddings.mean(axis=0)  # (D_in,)
        self._cache = []
        h = mean
        for i, layer in enumerate(self._layers):
            z = layer.forward(h)
            if i < len(self._layers) - 1:
                h = _relu(z)
                self._cache.append((z, h))
            else:
                h = z
                self._cache.append((z, z))
        return h  # (|V|,)

    def loss_decode(
        self, logits: np.ndarray, gold_tokens: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Average cross-entropy over gold tokens. Returns (loss, d_logits)."""
        if len(gold_tokens) == 0 or len(logits) == 0:
            return 0.0, np.zeros_like(logits)
        V = len(logits)
        e = np.exp(logits - logits.max())
        probs = e / (e.sum() + 1e-9)
        total_loss = 0.0
        d_logits = np.zeros(V, dtype=np.float32)
        valid = [int(t) for t in gold_tokens if 0 <= int(t) < V]
        if not valid:
            return 0.0, d_logits
        for t in valid:
            total_loss -= float(np.log(probs[t] + 1e-9))
            d_t = probs.copy()
            d_t[t] -= 1.0
            d_logits += d_t
        N = len(valid)
        return total_loss / N, d_logits / N

    def backward_decode(
        self, d_logits: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Backward through MLP. Returns (d_mean, param_grads).

        d_mean has shape (D_in,) — gradient w.r.t. the mean-pooled node embedding.
        Caller must divide by N and broadcast to each node when propagating upstream.
        """
        assert self._layers is not None, "backward_decode called before forward_decode"
        grads: list[tuple[np.ndarray, np.ndarray]] = []
        d = d_logits.copy()
        for i in reversed(range(len(self._layers))):
            z, _ = self._cache[i]
            if i < len(self._layers) - 1:
                d = d * _relu_grad(z)
            dx, dW, db = self._layers[i].backward(d)
            grads.insert(0, (dW, db))
            d = dx
        return d, grads  # d: (D_in,)

    def parameters(self) -> list[np.ndarray]:
        if self._layers is None:
            return []
        params: list[np.ndarray] = []
        for layer in self._layers:
            params.extend([layer.W, layer.b])
        return params

    def update(self, grads: list[tuple[np.ndarray, np.ndarray]], lr: float) -> None:
        assert self._layers is not None
        for layer, (dW, db) in zip(self._layers, grads):
            layer.W -= lr * dW
            layer.b -= lr * db

    # ── VerbalizerDecoder inference interface ─────────────────────────────────

    def decode(self, ir_json: str) -> str:
        """CausalIR JSON → surface string (inference)."""
        node_embs = self._node_type_embeddings_from_ir(ir_json)
        logits = self.forward_decode(node_embs)
        top_indices = np.argsort(logits)[-10:][::-1]
        filtered = [int(i) for i in top_indices if int(i) >= 2][:5]
        return self.vocab.decode(filtered)

    # ── Checkpoint serialization ──────────────────────────────────────────────

    def to_json(self) -> str:
        d_in = self._layers[0].W.shape[1] if self._layers else None
        return json.dumps({
            "vocab": self.vocab.to_json(),
            "d_hidden": self.d_hidden,
            "d_in": d_in,
        })

    @classmethod
    def from_json(cls, s: str) -> "TrainableDecoder":
        data = json.loads(s)
        vocab = SurfaceVocabulary.from_json(data["vocab"])
        return cls(vocab, d_hidden=data["d_hidden"], d_in=data.get("d_in"))
