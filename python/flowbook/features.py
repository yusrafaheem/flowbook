"""LOB feature extraction.

Turns a `flowbook._core.MatchingEngine` snapshot into the numeric features
used both by the market-making strategies and by the research model
(flowbook.research). Features are the standard ones used across the deep
LOB forecasting literature (Kercheval & Zhang 2015; Zhang, Zohren & Roberts
2019 "DeepLOB"; Berti et al. 2025 "LiT"):

  - order book imbalance (OBI) at multiple depths
  - microprice / mid-price deviation
  - relative spread
  - depth (total resting quantity) at each level, both sides

`snapshot_to_vector` returns a fixed-length numeric vector per book state;
`FeatureWindow` accumulates a rolling window of these vectors, which is the
unit both the Avellaneda-Stoikov strategy (for a volatility estimate) and
the transformer model (as its input sequence) consume.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from flowbook import _core

N_LEVELS = 5           # depth levels considered per side
FEATURE_DIM = 2 * N_LEVELS + 3  # bid/ask qty per level + imbalance + rel_spread + micro_dev


def order_book_imbalance(snapshot: "_core.BookSnapshot", levels: int = N_LEVELS) -> float:
    """OBI = (bid_depth - ask_depth) / (bid_depth + ask_depth) over the top
    `levels` price levels. In [-1, 1]; positive means buy-side pressure.
    """
    bid_depth = sum(l.total_quantity for l in snapshot.bids[:levels])
    ask_depth = sum(l.total_quantity for l in snapshot.asks[:levels])
    total = bid_depth + ask_depth
    if total == 0:
        return 0.0
    return (bid_depth - ask_depth) / total


def relative_spread(engine: "_core.MatchingEngine") -> float | None:
    bid = engine.best_bid()
    ask = engine.best_ask()
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2.0
    if mid == 0:
        return None
    return (ask - bid) / mid


def microprice_deviation(engine: "_core.MatchingEngine") -> float | None:
    """(microprice - mid) / tick_size -- signed pressure indicator in ticks."""
    micro = engine.microprice()
    mid = engine.mid_price()
    if micro is None or mid is None:
        return None
    tick = engine.tick_size or 1
    return (micro - mid) / tick


def snapshot_to_vector(engine: "_core.MatchingEngine", levels: int = N_LEVELS) -> np.ndarray | None:
    """Fixed-length feature vector for the current book state, or None if
    either side of the book is empty (features are undefined without a
    two-sided market).

    Layout: [bid_qty_0..L, ask_qty_0..L, imbalance, rel_spread, micro_dev]
    Quantities are log1p-scaled to tame the heavy right tail from the
    simulator's lognormal order sizes.
    """
    snap = engine.snapshot(levels)
    if not snap.bids or not snap.asks:
        return None

    bid_q = [np.log1p(l.total_quantity) for l in snap.bids[:levels]]
    ask_q = [np.log1p(l.total_quantity) for l in snap.asks[:levels]]
    bid_q += [0.0] * (levels - len(bid_q))
    ask_q += [0.0] * (levels - len(ask_q))

    imb = order_book_imbalance(snap, levels)
    spread = relative_spread(engine)
    micro_dev = microprice_deviation(engine)
    if spread is None or micro_dev is None:
        return None

    return np.array(bid_q + ask_q + [imb, spread, micro_dev], dtype=np.float32)


class FeatureWindow:
    """Rolling window of feature vectors, used as the transformer's input
    sequence and for realized-volatility estimates used by the market maker.
    """

    def __init__(self, maxlen: int = 100):
        self.maxlen = maxlen
        self._vectors: deque[np.ndarray] = deque(maxlen=maxlen)
        self._mids: deque[float] = deque(maxlen=maxlen)

    def push(self, engine: "_core.MatchingEngine") -> bool:
        vec = snapshot_to_vector(engine)
        mid = engine.mid_price()
        if vec is None or mid is None:
            return False
        self._vectors.append(vec)
        self._mids.append(mid)
        return True

    def is_full(self) -> bool:
        return len(self._vectors) == self.maxlen

    def as_array(self) -> np.ndarray:
        """Shape (T, FEATURE_DIM), T <= maxlen."""
        return np.stack(self._vectors) if self._vectors else np.zeros((0, FEATURE_DIM), dtype=np.float32)

    def realized_volatility(self) -> float:
        """Std of mid-price log-returns over the window -- used by the
        Avellaneda-Stoikov maker as its sigma estimate.
        """
        if len(self._mids) < 3:
            return 0.0
        mids = np.array(self._mids, dtype=np.float64)
        rets = np.diff(np.log(mids))
        return float(np.std(rets)) if len(rets) > 0 else 0.0
