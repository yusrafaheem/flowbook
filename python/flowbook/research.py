"""A short-horizon LOB movement classifier: a minimal, from-scratch
self-attention model (single head, hand-derived forward/backward pass, no
autodiff framework) trained to predict whether the mid-price will be up,
flat, or down `horizon` steps ahead, given a window of LOB feature vectors.

Why hand-rolled instead of a framework: the whole point of this module is
the same one behind the `vectorgrad` project this repo's author already
built (a from-scratch reverse-mode autodiff engine) -- being able to derive
and verify the gradients yourself is what lets you trust and debug a model,
rather than trusting a framework's autograd blindly. `grad_check()` below
verifies every analytic gradient against central-difference numerical
gradients; see tests/test_research.py, where it's run as an actual test.

Architecture (deliberately small -- this is a from-scratch reference
implementation, not a production model):

    input (T, F)
      -> linear embed to (T, d_model)
      -> + fixed sinusoidal positional encoding
      -> single-head self-attention (Q, K, V all (T, d_model)); no causal
         mask, since we predict from a completed window, not
         autoregressively -- every position may attend to every other
      -> residual add
      -> mean-pool over time -> (d_model,)
      -> linear classifier -> (3,) logits: {down, flat, up}
      -> softmax cross-entropy

This mirrors the shape of the 2025 transformer-based LOB forecasters (LiT,
TLOB -- see docs/RESEARCH.md) at a scale that trains in seconds on a CPU
with no external ML framework: one attention block instead of a deep stack,
a hand-derived backward pass instead of autograd. The public interface
(`build_dataset`, `TinyAttentionClassifier`, `train`) is what you'd keep if
you swapped this out for a deeper, framework-backed model later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flowbook.features import FEATURE_DIM, FeatureWindow
from flowbook.simulator import MarketSimulator, SimulatorConfig, research_sim_config

N_CLASSES = 3  # 0 = down, 1 = flat, 2 = up


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Standard fixed sinusoidal positional encoding (Vaswani et al. 2017),
    shape (seq_len, d_model). No learnable parameters.
    """
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class ModelConfig:
    feature_dim: int = FEATURE_DIM
    d_model: int = 16
    seq_len: int = 24
    seed: int = 0


class TinyAttentionClassifier:
    """Single-head self-attention encoder + linear classifier, with a fully
    hand-derived forward and backward pass (see module docstring). Weights
    are stored in `self.params` (a dict of named arrays) so `train()` can
    treat this as a generic parametric model.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.cfg = config or ModelConfig()
        rng = np.random.default_rng(self.cfg.seed)
        d, f = self.cfg.d_model, self.cfg.feature_dim

        def init(shape):
            fan_in = shape[0]
            return rng.normal(0, 1.0 / np.sqrt(fan_in), size=shape)

        self.params = {
            "W_emb": init((f, d)),
            "b_emb": np.zeros(d),
            "W_q": init((d, d)),
            "W_k": init((d, d)),
            "W_v": init((d, d)),
            "W_out": init((d, N_CLASSES)),
            "b_out": np.zeros(N_CLASSES),
        }
        self._pe_cache: dict[int, np.ndarray] = {}

    def _pe(self, seq_len: int) -> np.ndarray:
        if seq_len not in self._pe_cache:
            self._pe_cache[seq_len] = positional_encoding(seq_len, self.cfg.d_model)
        return self._pe_cache[seq_len]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """X: (T, F). Returns (logits: (N_CLASSES,), cache for backward)."""
        p = self.params
        T, F = X.shape
        d = self.cfg.d_model

        emb = X @ p["W_emb"] + p["b_emb"]           # (T, d)
        h0 = emb + self._pe(T)                        # (T, d)  -- + positional encoding

        Q = h0 @ p["W_q"]                              # (T, d)
        K = h0 @ p["W_k"]                              # (T, d)
        V = h0 @ p["W_v"]                              # (T, d)

        scores = (Q @ K.T) / np.sqrt(d)                # (T, T)
        attn = softmax(scores, axis=-1)                # (T, T), rows sum to 1
        attn_out = attn @ V                             # (T, d)

        h1 = h0 + attn_out                              # residual
        pooled = h1.mean(axis=0)                        # (d,) -- mean pool over time

        logits = pooled @ p["W_out"] + p["b_out"]       # (N_CLASSES,)

        cache = dict(X=X, emb=emb, h0=h0, Q=Q, K=K, V=V, scores=scores,
                     attn=attn, attn_out=attn_out, h1=h1, pooled=pooled, T=T, d=d)
        return logits, cache

    def backward(self, dlogits: np.ndarray, cache: dict) -> dict:
        """dlogits: (N_CLASSES,) = dL/dlogits. Returns a dict of gradients
        with the same keys/shapes as self.params.
        """
        p = self.params
        X, h0, Q, K, V = cache["X"], cache["h0"], cache["Q"], cache["K"], cache["V"]
        attn, attn_out, pooled, T, d = (
            cache["attn"], cache["attn_out"], cache["pooled"], cache["T"], cache["d"]
        )

        grads = {k: np.zeros_like(v) for k, v in p.items()}

        # logits = pooled @ W_out + b_out
        grads["W_out"] = np.outer(pooled, dlogits)
        grads["b_out"] = dlogits
        d_pooled = p["W_out"] @ dlogits                  # (d,)

        # pooled = mean_t h1[t]  ->  d_h1[t] = d_pooled / T  for every t
        d_h1 = np.tile(d_pooled / T, (T, 1))              # (T, d)

        # h1 = h0 + attn_out  (residual)
        d_h0 = d_h1.copy()
        d_attn_out = d_h1.copy()

        # attn_out = attn @ V
        d_attn = d_attn_out @ V.T                          # (T, T)
        d_V = attn.T @ d_attn_out                           # (T, d)

        # attn = softmax(scores, axis=-1); softmax Jacobian per row.
        # d_scores[i, :] = (diag(attn[i]) - outer(attn[i], attn[i])) @ d_attn[i, :]
        d_scores = np.empty_like(d_attn)
        for i in range(T):
            a = attn[i]                                     # (T,)
            jac = np.diag(a) - np.outer(a, a)                # (T, T)
            d_scores[i] = jac @ d_attn[i]

        # scores = (Q @ K.T) / sqrt(d)
        scale = 1.0 / np.sqrt(d)
        d_Q = (d_scores @ K) * scale                          # (T, d)
        d_K = (d_scores.T @ Q) * scale                        # (T, d)

        # Q = h0 @ W_q, K = h0 @ W_k, V = h0 @ W_v
        grads["W_q"] = h0.T @ d_Q
        grads["W_k"] = h0.T @ d_K
        grads["W_v"] = h0.T @ d_V
        d_h0 += d_Q @ p["W_q"].T + d_K @ p["W_k"].T + d_V @ p["W_v"].T

        # h0 = emb + positional_encoding (pe has no params) -> d_emb = d_h0
        d_emb = d_h0

        # emb = X @ W_emb + b_emb
        grads["W_emb"] = X.T @ d_emb
        grads["b_emb"] = d_emb.sum(axis=0)

        return grads

    def loss_and_grad(self, X: np.ndarray, y: int) -> tuple[float, dict]:
        logits, cache = self.forward(X)
        probs = softmax(logits)
        loss = -np.log(max(probs[y], 1e-12))
        dlogits = probs.copy()
        dlogits[y] -= 1.0  # softmax + cross-entropy gradient
        grads = self.backward(dlogits, cache)
        return loss, grads

    def predict(self, X: np.ndarray) -> int:
        logits, _ = self.forward(X)
        return int(np.argmax(logits))


class Adam:
    """Minimal Adam optimizer operating on a params dict, matched key-for-key
    against a grads dict of the same shapes.
    """

    def __init__(self, params: dict, lr: float = 1e-2, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def grad_check(model: TinyAttentionClassifier, X: np.ndarray, y: int,
                eps: float = 1e-5, tol: float = 2e-2) -> dict:
    """Verifies every analytic gradient in `backward` against a central
    finite-difference numerical gradient. Returns a dict of per-parameter
    max relative errors; raises AssertionError if any exceeds `tol`.

    This is the from-scratch equivalent of trusting an autograd framework:
    since there is no autograd here, this check is what stands in for it.
    """
    _, analytic_grads = model.loss_and_grad(X, y)
    errors = {}

    for name, param in model.params.items():
        flat = param.reshape(-1)
        analytic = analytic_grads[name].reshape(-1)
        numeric = np.zeros_like(flat)

        # Full check on small params, a random subset on larger ones (kept
        # fast enough to run in CI on every push).
        n_check = flat.size if flat.size <= 40 else 40
        idxs = np.arange(flat.size) if flat.size <= 40 else \
            np.random.default_rng(0).choice(flat.size, size=n_check, replace=False)

        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            loss_plus, _ = model.loss_and_grad(X, y)
            flat[i] = orig - eps
            loss_minus, _ = model.loss_and_grad(X, y)
            flat[i] = orig
            numeric[i] = (loss_plus - loss_minus) / (2 * eps)

        checked = analytic[idxs]
        num_checked = numeric[idxs]
        denom = np.maximum(np.abs(checked), np.abs(num_checked))
        denom = np.where(denom == 0, 1.0, denom)
        rel_err = np.max(np.abs(checked - num_checked) / denom)
        errors[name] = float(rel_err)
        assert rel_err < tol, f"gradient check failed for {name}: rel_err={rel_err}"

    return errors


def label_from_return(log_return: float, alpha: float) -> int:
    """3-class label following the DeepLOB / FI-2010 convention: compares
    the future log-return to a threshold `alpha` (in the same units as the
    return) to decide up/flat/down, rather than just its sign -- this
    avoids labeling microscopic noise as a directional move.
    """
    if log_return > alpha:
        return 2  # up
    if log_return < -alpha:
        return 0  # down
    return 1  # flat


def build_dataset(n_sequences: int, seq_len: int = 24, horizon: int = 150,
                   alpha: float | None = None, balance_quantile: float = 0.33,
                   sim_config: SimulatorConfig | None = None,
                   background_flow_per_step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Generates (X, y) from the synthetic simulator: X has shape
    (n_sequences, seq_len, FEATURE_DIM), y has shape (n_sequences,) with
    values in {0, 1, 2}.

    `alpha` is the flat/directional threshold on the horizon log-return
    (see `label_from_return`). If left as None (the default), it is set
    automatically to the `balance_quantile` quantile of |log-return| over
    the generated series, i.e. roughly `balance_quantile` of examples land
    in the "flat" bucket and the rest split across up/down -- this avoids
    the degenerate case of a fixed alpha producing an almost-all-flat
    dataset a model can "solve" by always predicting the majority class.

    Swapping in real data: replace this function's body with a loader that
    yields (feature_window, future_mid_price) pairs from a real LOB feed
    (e.g. LOBSTER/ITCH replay) through the same `flowbook.features`
    extraction -- `TinyAttentionClassifier` and `train()` are agnostic to
    where the windows came from.
    """
    sim = MarketSimulator(sim_config or research_sim_config(seed=0))
    window = FeatureWindow(maxlen=seq_len)

    all_vectors: list[np.ndarray] = []
    all_mids: list[float] = []

    # Run long enough to have n_sequences + horizon usable windows.
    n_steps = (n_sequences + horizon + seq_len) * 2
    steps_done = 0
    while len(all_vectors) < n_sequences + horizon and steps_done < n_steps * 3:
        for _ in range(background_flow_per_step):
            sim.step()
        steps_done += 1
        if window.push(sim.engine):
            all_vectors.append(window.as_array()[-1].copy())
            all_mids.append(sim.engine.mid_price())

    # First pass: compute every candidate horizon log-return so we can
    # calibrate `alpha` before assigning any labels.
    candidate_rets = []
    for t in range(seq_len, len(all_vectors) - horizon):
        mid_now = all_mids[t - 1]
        mid_future = all_mids[t - 1 + horizon]
        candidate_rets.append(np.log(mid_future / mid_now) if mid_now > 0 else 0.0)
    candidate_rets = np.array(candidate_rets)

    if alpha is None:
        alpha = float(np.quantile(np.abs(candidate_rets), balance_quantile)) if len(candidate_rets) else 0.0

    X_list, y_list = [], []
    for idx, t in enumerate(range(seq_len, len(all_vectors) - horizon)):
        seq = np.stack(all_vectors[t - seq_len:t])
        X_list.append(seq)
        y_list.append(label_from_return(candidate_rets[idx], alpha))
        if len(X_list) >= n_sequences:
            break

    return np.stack(X_list).astype(np.float64), np.array(y_list, dtype=np.int64)


def train(model: TinyAttentionClassifier, X: np.ndarray, y: np.ndarray,
          epochs: int = 10, lr: float = 5e-3, verbose: bool = True) -> list[float]:
    """Full-batch-per-epoch (but per-sample gradient step, i.e. SGD with
    batch size 1) training loop. Small dataset + small model -> this is
    fast enough without minibatching machinery.
    """
    opt = Adam(model.params, lr=lr)
    history = []
    n = len(X)
    rng = np.random.default_rng(0)

    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        for i in order:
            loss, grads = model.loss_and_grad(X[i], int(y[i]))
            opt.step(model.params, grads)
            total_loss += loss
        avg_loss = total_loss / n
        history.append(avg_loss)
        if verbose:
            acc = evaluate(model, X, y)
            print(f"epoch {epoch + 1:>2}/{epochs}  loss={avg_loss:.4f}  train_acc={acc:.3f}")
    return history


def evaluate(model: TinyAttentionClassifier, X: np.ndarray, y: np.ndarray) -> float:
    preds = np.array([model.predict(X[i]) for i in range(len(X))])
    return float(np.mean(preds == y))


def class_balance(y: np.ndarray) -> dict:
    values, counts = np.unique(y, return_counts=True)
    labels = {0: "down", 1: "flat", 2: "up"}
    return {labels[int(v)]: int(c) for v, c in zip(values, counts)}
