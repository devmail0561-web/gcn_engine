from __future__ import annotations
import json
import re
import warnings
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
        # M9 : Warning si indices hors-bornes (modèle possiblement corrompu)
        valid = [self._i2t[int(i)] for i in indices if 0 <= int(i) < len(self._i2t)]
        invalid_count = len(indices) - len(valid)
        if invalid_count > 0:
            warnings.warn(
                f"SurfaceVocabulary.decode : {invalid_count} indice(s) hors-bornes ignoré(s). "
                f"Modèle possiblement corrompu.",
                UserWarning,
                stacklevel=2,
            )
        return " ".join(valid)

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

    Architecture (P2d - attention pooling):
      scores = node_embs @ _attn_vec
      attn_weights = softmax(scores)
      context = sum(attn_weights[i] * node_embs[i])
      h_0 = zeros(d_hidden)
      RNN step: h_t = tanh(W_rnn @ [context; h_{t-1}] + b_rnn)
      Output:   logit_t = W_out @ h_t + b_out

    Stored as 1 attention vector + 2 _LinearLayer objects (5 params total):
      _attn_vec : (d_in,) — attention vector (initialized to zero)
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
        self._attn_vec: np.ndarray | None = None  # P2d: attention vector
        self._W_query: np.ndarray | None = None   # S6: per-step attention query
        self._d_W_query: np.ndarray | None = None  # S6: accumulated W_query gradient
        self._cached_attn_weights: np.ndarray | None = None  # P2d: for backward (last step)
        self._cached_node_embs: np.ndarray | None = None  # P2d: for backward
        self._n_wq_steps: int = 0  # accumulation counter for _d_W_query
        if len(vocab) <= 2:
            warnings.warn(
                "TrainableDecoder : vocab contient seulement PAD/UNK — "
                "appeler vocab.build(surfaces) avant de construire le décodeur.",
                UserWarning,
                stacklevel=2,
            )
        if d_in is not None:
            self._init_layers(d_in)

    def _init_layers(self, d_in: int) -> None:
        if self._layers is None:
            self._attn_vec = np.zeros(d_in, dtype=np.float32)       # P2d: attention pooling
            _scale = np.sqrt(2.0 / (self.d_hidden + d_in))
            self._W_query = self._rng.normal(0, _scale, (self.d_hidden, d_in)).astype(np.float32)
            self._layers = [
                _LinearLayer(d_in + self.d_hidden, self.d_hidden, self._rng),  # RNN cell
                _LinearLayer(self.d_hidden, len(self.vocab), self._rng),        # output
            ]
            self._last_d_in = d_in

    def _rnn_step(
        self, context: np.ndarray, h_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One RNN step. Returns (h_new, rnn_in, z1, logits)."""
        if self._layers is None:
            raise RuntimeError("_rnn_step appelé avant l'initialisation des couches")
        rnn_in = np.concatenate([context, h_prev])          # (d_in + d_hidden,)
        z1 = self._layers[0].forward(rnn_in)                 # (d_hidden,)
        h_new = np.tanh(z1)
        logits = self._layers[1].forward(h_new)              # (|V|,)
        return h_new, rnn_in, z1, logits


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
            self._rnn_step_cache = []
            self._cached_node_embs = None
            return np.zeros(len(self.vocab), dtype=np.float32)
        d_in = node_embeddings.shape[1]
        self._init_layers(d_in)
        if self._last_d_in is not None and node_embeddings.shape[1] != self._last_d_in:
            raise ValueError(
                f"forward_decode : d_in={node_embeddings.shape[1]} incompatible "
                f"avec d_in={self._last_d_in} verrouillé à l'init."
            )

        if self._attn_vec is None or self._W_query is None:
            raise RuntimeError("forward_decode : état interne non initialisé après _init_layers")
        # Cacher les node embeddings pour backward
        self._cached_node_embs = node_embeddings
        h = np.zeros(self.d_hidden, dtype=np.float32)

        def _step_attention(h_prev: np.ndarray):
            """S6 : attention per-step — query = attn_vec + W_query.T @ h_prev."""
            query_vec = self._attn_vec + self._W_query.T @ h_prev  # (d_in,)
            scores = node_embeddings @ query_vec                    # (N,)
            exp_s = np.exp(scores - scores.max())
            step_attn = exp_s / (exp_s.sum() + 1e-9)               # (N,)
            context = (step_attn[:, np.newaxis] * node_embeddings).sum(axis=0).astype(np.float32)
            return context, step_attn, query_vec

        if gold_tokens is None:
            # Inference: single step (h_prev = zeros → query_vec = attn_vec, rétrocompat)
            context, step_attn, query_vec = _step_attention(h)
            self._cached_attn_weights = step_attn
            h_new, rnn_in, z1, logits = self._rnn_step(context, h)
            self._rnn_step_cache = [{"rnn_in": rnn_in, "h1": h_new, "z1": z1,
                                     "step_attn": step_attn, "h_prev": h.copy(),
                                     "query_vec": query_vec}]
            return logits  # (|V|,)
        else:
            # Teacher forcing: T steps with per-step attention
            T = len(gold_tokens)
            all_logits = np.zeros((T, len(self.vocab)), dtype=np.float32)
            self._rnn_step_cache = []
            for t in range(T):
                context, step_attn, query_vec = _step_attention(h)
                h_new, rnn_in, z1, logits = self._rnn_step(context, h)
                all_logits[t] = logits
                self._rnn_step_cache.append({"rnn_in": rnn_in, "h1": h_new, "z1": z1,
                                             "step_attn": step_attn, "h_prev": h.copy(),
                                             "query_vec": query_vec})
                h = h_new
            self._cached_attn_weights = self._rnn_step_cache[-1]["step_attn"]
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
            return total_loss / N, d_logits
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
            return total_loss / valid_steps, d_logits

    def backward_decode(
        self, d_logits: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Backward through RNN + attention. Returns (d_node_embs, param_grads, d_attn_vec).

        P2d: d_node_embs shape (N, D_in) — gradient différencié par nœud
        (vs d_mean (D_in,) en mean-pool). Handles 1D d_logits (|V|,) for single-step
        or 2D (T, |V|) for multi-step.
        """
        if self._layers is None:
            raise RuntimeError("backward_decode appelé avant forward_decode")
        if self._last_d_in is None or self._cached_node_embs is None or self._W_query is None:
            raise RuntimeError("backward_decode : état du cache invalide — forward_decode requis d'abord")

        dW0_total = np.zeros_like(self._layers[0].W)
        db0_total = np.zeros_like(self._layers[0].b)
        dW1_total = np.zeros_like(self._layers[1].W)
        db1_total = np.zeros_like(self._layers[1].b)
        d_attn_vec_total = np.zeros(self._last_d_in, dtype=np.float32)
        d_W_query_total = np.zeros_like(self._W_query)    # (d_hidden, d_in)
        d_node_embs_total = np.zeros_like(self._cached_node_embs)  # (N, d_in)

        steps = self._rnn_step_cache
        if d_logits.ndim == 1:
            T, n_steps = 1, 1
            d_logits_2d = d_logits.reshape(1, -1)
        else:
            T = d_logits.shape[0]
            n_steps = max(T, 1)
            d_logits_2d = d_logits

        d_h_next = np.zeros(self.d_hidden, dtype=np.float32)
        for t in range(min(T, len(steps)) - 1, -1, -1):
            step = steps[t]
            step_logits = d_logits_2d[t]

            # Couche output backward
            self._layers[0]._cache["x"] = step["rnn_in"]
            self._layers[1]._cache["x"] = step["h1"]
            dx_h1, dW1, db1 = self._layers[1].backward(step_logits)
            dx_h1 += d_h_next  # gradient depuis l'étape future
            dW1_total += dW1; db1_total += db1

            # Couche RNN backward
            d_pre_tanh = dx_h1 * (1.0 - step["h1"] ** 2)
            dx_rnn, dW0, db0 = self._layers[0].backward(d_pre_tanh)
            dW0_total += dW0; db0_total += db0

            d_context_t = dx_rnn[:self._last_d_in]   # (d_in,)
            d_h_from_rnn = dx_rnn[self._last_d_in:]  # (d_hidden,) → h_prev

            # S6 : Backward attention per-step
            step_attn = step["step_attn"]   # (N,)
            h_prev_t = step["h_prev"]       # (d_hidden,)
            query_vec_t = step["query_vec"] # (d_in,)

            # d_context_t → d_step_attn_t
            d_step_attn_t = (d_context_t * self._cached_node_embs).sum(axis=1)  # (N,)

            # d_step_attn_t → d_scores_t (inverse softmax)
            d_scores_t = step_attn * (
                d_step_attn_t - (step_attn * d_step_attn_t).sum()
            )  # (N,)

            # d_scores_t → d_query_vec_t
            d_query_vec_t = self._cached_node_embs.T @ d_scores_t  # (d_in,)

            # d_query_vec → d_attn_vec (direct)
            d_attn_vec_total += d_query_vec_t

            # d_query_vec → d_W_query : outer(h_prev, d_query_vec) — shape (d_hidden, d_in)
            d_W_query_total += np.outer(h_prev_t, d_query_vec_t)

            # d_query_vec → d_h_prev via W_query
            d_h_from_attn = self._W_query @ d_query_vec_t  # (d_hidden,)

            # Gradient combiné vers h_prev
            d_h_next = d_h_from_rnn + d_h_from_attn

            # Gradient vers node_embeddings (via context et via scores)
            d_node_embs_total += step_attn[:, np.newaxis] * d_context_t[np.newaxis, :]
            d_node_embs_total += d_scores_t[:, np.newaxis] * query_vec_t[np.newaxis, :]

        # Accumuler d_W_query (mini-batch : plusieurs backward_decode avant update)
        if self._d_W_query is None:
            self._d_W_query = d_W_query_total.copy()
            self._n_wq_steps = n_steps
        else:
            self._d_W_query += d_W_query_total
            self._n_wq_steps += n_steps

        grads = [
            (dW0_total / n_steps, db0_total / n_steps),
            (dW1_total / n_steps, db1_total / n_steps),
        ]
        d_attn_vec = d_attn_vec_total / n_steps
        return d_node_embs_total / n_steps, grads, d_attn_vec

    def parameters(self) -> list[np.ndarray]:
        if self._layers is None:
            return []
        # Ordre : attn_vec, layers(W,b)×2, W_query (en dernier pour compat checkpoints antérieurs)
        params: list[np.ndarray] = [self._attn_vec] if self._attn_vec is not None else []
        for layer in self._layers:
            params.extend([layer.W, layer.b])
        if self._W_query is not None:
            params.append(self._W_query)
        return params

    def update(self, grads: list[tuple[np.ndarray, np.ndarray]], d_attn_vec: np.ndarray, lr: float) -> None:
        if self._layers is None:
            raise RuntimeError("update appelé avant l'initialisation — forward_decode requis d'abord")
        if len(grads) != len(self._layers):
            raise ValueError(
                f"update() : {len(grads)} groupes de gradients pour "
                f"{len(self._layers)} couches."
            )
        # P2d: Mettre à jour attn_vec
        if self._attn_vec is not None:
            self._attn_vec -= lr * d_attn_vec
        # S6: Mettre à jour W_query depuis le gradient stocké par backward_decode
        if self._W_query is not None and self._d_W_query is not None:
            _nwq = self._n_wq_steps if self._n_wq_steps > 0 else 1
            self._W_query -= lr * (self._d_W_query / _nwq)
            self._d_W_query = None
            self._n_wq_steps = 0
        for layer, (dW, db) in zip(self._layers, grads):
            layer.W -= lr * dW
            layer.b -= lr * db

    # ── VerbalizerDecoder inference interface ─────────────────────────────────

    def decode(self, node_embeddings: np.ndarray) -> str:
        """
        Vecteurs enrichis (N, D_in) → surface string (greedy decode).

        IMPORTANT (H1 correction) : node_embeddings doit provenir de l'encodeur+R-GCN,
        pas du one-hot. Workflow correct :
          1. cir_json = pipeline.forward(reps, text)
          2. surface = decoder.decode(pipeline.get_enriched_vectors())

        Un décodeur entraîné conjointement avec l'encodeur (d_in=75) ne peut PAS
        décoder depuis du one-hot (d_in=7) → dimension mismatch.
        """
        if len(node_embeddings) == 0:
            return ""
        eos_idx = self.vocab._t2i.get(SurfaceVocabulary.EOS, -1)
        _skip = {0, 1, eos_idx}
        if self._layers is None:
            self._init_layers(node_embeddings.shape[1])

        # Greedy multi-step decode with S6 per-step attention
        if self._attn_vec is None or self._W_query is None:
            raise RuntimeError("infer : état interne non initialisé — forward_decode requis d'abord")

        h = np.zeros(self.d_hidden, dtype=np.float32)
        tokens: list[int] = []
        for _ in range(self.max_decode_len):
            query_vec = self._attn_vec + self._W_query.T @ h
            attn_scores = node_embeddings @ query_vec
            exp_s = np.exp(attn_scores - attn_scores.max())
            attn_weights = exp_s / (exp_s.sum() + 1e-9)
            context = (attn_weights[:, np.newaxis] * node_embeddings).sum(axis=0).astype(np.float32)
            h_new, _, _, logits = self._rnn_step(context, h)
            token = int(np.argmax(logits))
            if token == eos_idx:
                break
            tokens.append(token)
            h = h_new
        # H2 correction : pas de troncature [:5], longueur contrôlée par max_decode_len
        filtered = [i for i in tokens if i not in _skip]
        return self.vocab.decode(filtered)

    # ── Checkpoint serialization ──────────────────────────────────────────────

    def to_json(self) -> str:
        attn_vec_list = self._attn_vec.tolist() if self._attn_vec is not None else None
        w_query_list = self._W_query.tolist() if self._W_query is not None else None
        return json.dumps({
            "vocab": self.vocab.to_json(),
            "d_hidden": self.d_hidden,
            "d_in": self._last_d_in,
            "max_decode_len": self.max_decode_len,
            "attn_vec": attn_vec_list,
            "w_query": w_query_list,
        })

    @classmethod
    def from_json(cls, s: str) -> "TrainableDecoder":
        data = json.loads(s)
        vocab = SurfaceVocabulary.from_json(data["vocab"])
        dec = cls(vocab, d_hidden=data["d_hidden"], d_in=data.get("d_in"),
                  max_decode_len=data.get("max_decode_len", 20))
        # P2d: Restaurer attn_vec si présent (compatibilité checkpoints antérieurs)
        if data.get("attn_vec") and dec._layers is not None:
            dec._attn_vec = np.array(data["attn_vec"], dtype=np.float32)
        # S6: Restaurer W_query si présent
        if data.get("w_query") and dec._layers is not None:
            dec._W_query = np.array(data["w_query"], dtype=np.float32)
        return dec
