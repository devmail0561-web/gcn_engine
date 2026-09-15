from __future__ import annotations
import json
import re
import numpy as np

from ..layer2.reference import _LinearLayer


class SurfaceVocabulary:
    """Word-level vocabulary built from gold surface texts."""

    PAD = "<pad>"
    UNK = "<unk>"
    EOS = "<eos>"

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
        # EOS appended at end after all user tokens, to avoid shifting indices
        if self.EOS not in self._t2i:
            self._t2i[self.EOS] = len(self._i2t)
            self._i2t.append(self.EOS)

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
    Reference NumPy autoregressive decoder.

    Architecture: h_0 = zeros(d_hidden); context = mean_pool(node_embs)
    RNN step: h_t = tanh(W_rnn @ [context; h_{t-1}] + b_rnn)
    Output:   logit_t = W_out @ h_t + b_out

    Stored as 2 _LinearLayer objects (4 params total):
      layer0 : _LinearLayer(d_in + d_hidden, d_hidden)  — RNN cell
      layer1 : _LinearLayer(d_hidden, |V|)               — output

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
        max_decode_len: int = 20,
    ) -> None:
        self.vocab = vocab
        self.d_hidden = d_hidden
        self.max_decode_len = max_decode_len
        self._rng = np.random.default_rng(seed)
        self._layers: list[_LinearLayer] | None = None
        self._rnn_step_cache: list[dict] = []  # per-step cache for BPTT
        self._last_d_in: int | None = None
        if d_in is not None:
            self._init_layers(d_in)

    def _init_layers(self, d_in: int) -> None:
        if self._layers is None:
            self._layers = [
                _LinearLayer(d_in + self.d_hidden, self.d_hidden, self._rng),  # RNN cell
                _LinearLayer(self.d_hidden, len(self.vocab), self._rng),        # output
            ]
            self._last_d_in = d_in

    def _rnn_step(
        self, context: np.ndarray, h_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One RNN step. Returns (h_new, rnn_in, z1, logits)."""
        assert self._layers is not None
        rnn_in = np.concatenate([context, h_prev])          # (d_in + d_hidden,)
        z1 = self._layers[0].forward(rnn_in)                 # (d_hidden,)
        h_new = np.tanh(z1)
        logits = self._layers[1].forward(h_new)              # (|V|,)
        return h_new, rnn_in, z1, logits

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

    def forward_decode(
        self,
        node_embeddings: np.ndarray,
        gold_tokens: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        (N, D_in) → (|V|,) logits in inference mode, or (T, |V|) in teacher-forcing mode.

        Inference (gold_tokens=None): runs one RNN step, returns first-step logits (|V|,).
        Training (gold_tokens provided): runs T steps with teacher forcing, returns (T, |V|).
        """
        if len(node_embeddings) == 0:
            return np.zeros(len(self.vocab), dtype=np.float32)
        d_in = node_embeddings.shape[1]
        self._init_layers(d_in)
        context = node_embeddings.mean(axis=0).astype(np.float32)  # (d_in,)
        h = np.zeros(self.d_hidden, dtype=np.float32)

        if gold_tokens is None:
            # Inference: single step
            h_new, rnn_in, z1, logits = self._rnn_step(context, h)
            self._rnn_step_cache = [{"rnn_in": rnn_in, "h1": h_new, "z1": z1}]
            return logits  # (|V|,)
        else:
            # Teacher forcing: T steps
            T = len(gold_tokens)
            all_logits = np.zeros((T, len(self.vocab)), dtype=np.float32)
            self._rnn_step_cache = []
            for t in range(T):
                h_new, rnn_in, z1, logits = self._rnn_step(context, h)
                all_logits[t] = logits
                self._rnn_step_cache.append({"rnn_in": rnn_in, "h1": h_new, "z1": z1})
                h = h_new
            return all_logits  # (T, |V|)

    def loss_decode(
        self, logits: np.ndarray, gold_tokens: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Cross-entropy loss. Handles 1D logits (|V|,) or 2D (T, |V|)."""
        if len(gold_tokens) == 0 or len(logits) == 0:
            return 0.0, np.zeros_like(logits)

        if logits.ndim == 1:
            # Single-step: average cross-entropy over gold tokens (backward compat)
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
        else:
            # Multi-step (T, |V|): per-step cross-entropy
            T, V = logits.shape
            total_loss = 0.0
            d_logits = np.zeros_like(logits)
            valid_steps = 0
            for t in range(T):
                tok = int(gold_tokens[t]) if t < len(gold_tokens) else -1
                if tok < 0 or tok >= V:
                    continue
                e = np.exp(logits[t] - logits[t].max())
                probs = e / (e.sum() + 1e-9)
                total_loss -= float(np.log(probs[tok] + 1e-9))
                d_t = probs.copy()
                d_t[tok] -= 1.0
                d_logits[t] = d_t
                valid_steps += 1
            if valid_steps == 0:
                return 0.0, d_logits
            return total_loss / valid_steps, d_logits / valid_steps

    def backward_decode(
        self, d_logits: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """
        Backward through RNN. Returns (d_mean, param_grads).

        d_mean has shape (D_in,) — gradient w.r.t. the mean-pooled node embedding.
        Handles 1D d_logits (|V|,) for single-step or 2D (T, |V|) for multi-step.
        """
        assert self._layers is not None, "backward_decode called before forward_decode"
        assert self._last_d_in is not None

        dW0_total = np.zeros_like(self._layers[0].W)
        db0_total = np.zeros_like(self._layers[0].b)
        dW1_total = np.zeros_like(self._layers[1].W)
        db1_total = np.zeros_like(self._layers[1].b)
        d_mean_total = np.zeros(self._last_d_in, dtype=np.float32)

        steps = self._rnn_step_cache
        if d_logits.ndim == 1:
            # Single step
            step = steps[0]
            self._layers[0]._cache["x"] = step["rnn_in"]
            self._layers[1]._cache["x"] = step["h1"]
            dx_h1, dW1, db1 = self._layers[1].backward(d_logits)
            dW1_total += dW1; db1_total += db1
            d_pre_tanh = dx_h1 * (1.0 - step["h1"] ** 2)
            dx_rnn, dW0, db0 = self._layers[0].backward(d_pre_tanh)
            dW0_total += dW0; db0_total += db0
            d_mean_total += dx_rnn[:self._last_d_in]
            n_steps = 1
        else:
            # Multi-step BPTT — propagate h_prev gradient through time
            T = d_logits.shape[0]
            n_steps = max(T, 1)
            d_h_next = np.zeros(self.d_hidden, dtype=np.float32)
            for t in range(T - 1, -1, -1):
                step = steps[t]
                self._layers[0]._cache["x"] = step["rnn_in"]
                self._layers[1]._cache["x"] = step["h1"]
                dx_h1, dW1, db1 = self._layers[1].backward(d_logits[t])
                dx_h1 += d_h_next  # gradient from future step's h_prev
                dW1_total += dW1; db1_total += db1
                d_pre_tanh = dx_h1 * (1.0 - step["h1"] ** 2)
                dx_rnn, dW0, db0 = self._layers[0].backward(d_pre_tanh)
                dW0_total += dW0; db0_total += db0
                d_mean_total += dx_rnn[:self._last_d_in]
                d_h_next = dx_rnn[self._last_d_in:]  # gradient w.r.t. h_prev → previous step

        grads = [
            (dW0_total / n_steps, db0_total / n_steps),
            (dW1_total / n_steps, db1_total / n_steps),
        ]
        return d_mean_total / n_steps, grads

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
        """CausalIR JSON → surface string (greedy decode)."""
        node_embs = self._node_type_embeddings_from_ir(ir_json)
        if len(node_embs) == 0:
            return ""
        eos_idx = self.vocab._t2i.get(SurfaceVocabulary.EOS, -1)
        _skip = {0, 1, eos_idx}
        if self._layers is None:
            logits = self.forward_decode(node_embs)
            top_indices = np.argsort(logits)[-10:][::-1]
            filtered = [int(i) for i in top_indices if int(i) not in _skip][:5]
            return self.vocab.decode(filtered)
        # Greedy multi-step decode
        context = node_embs.mean(axis=0).astype(np.float32)
        h = np.zeros(self.d_hidden, dtype=np.float32)
        tokens: list[int] = []
        for _ in range(self.max_decode_len):
            h_new, _, _, logits = self._rnn_step(context, h)
            token = int(np.argmax(logits))
            if token == eos_idx:
                break
            tokens.append(token)
            h = h_new
        filtered = [i for i in tokens if i not in _skip][:5]
        return self.vocab.decode(filtered)

    # ── Checkpoint serialization ──────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps({
            "vocab": self.vocab.to_json(),
            "d_hidden": self.d_hidden,
            "d_in": self._last_d_in,
            "max_decode_len": self.max_decode_len,
        })

    @classmethod
    def from_json(cls, s: str) -> "TrainableDecoder":
        data = json.loads(s)
        vocab = SurfaceVocabulary.from_json(data["vocab"])
        return cls(vocab, d_hidden=data["d_hidden"], d_in=data.get("d_in"),
                   max_decode_len=data.get("max_decode_len", 20))
