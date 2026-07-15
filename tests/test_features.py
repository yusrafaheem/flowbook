"""Tests for LOB feature extraction."""

import numpy as np

from flowbook import _core
from flowbook._core import Side, TimeInForce
from flowbook.features import (
    FEATURE_DIM,
    FeatureWindow,
    microprice_deviation,
    order_book_imbalance,
    relative_spread,
    snapshot_to_vector,
)


def _book_with_bid_ask(bid_qty=100, ask_qty=100, bid_px=99, ask_px=101):
    eng = _core.MatchingEngine(tick_size=1)
    eng.submit_limit(Side.Buy, bid_px, bid_qty, TimeInForce.GTC)
    eng.submit_limit(Side.Sell, ask_px, ask_qty, TimeInForce.GTC)
    return eng


def test_imbalance_is_zero_symmetric():
    eng = _book_with_bid_ask(100, 100)
    snap = eng.snapshot(5)
    assert order_book_imbalance(snap) == 0.0


def test_imbalance_sign_reflects_pressure():
    eng = _book_with_bid_ask(bid_qty=300, ask_qty=100)
    snap = eng.snapshot(5)
    assert order_book_imbalance(snap) > 0  # more resting buy interest

    eng2 = _book_with_bid_ask(bid_qty=100, ask_qty=300)
    snap2 = eng2.snapshot(5)
    assert order_book_imbalance(snap2) < 0


def test_imbalance_bounded():
    eng = _book_with_bid_ask(bid_qty=1, ask_qty=10_000)
    snap = eng.snapshot(5)
    imb = order_book_imbalance(snap)
    assert -1.0 <= imb <= 1.0


def test_relative_spread_and_microprice_deviation_none_when_one_sided():
    eng = _core.MatchingEngine(tick_size=1)
    eng.submit_limit(Side.Buy, 99, 10, TimeInForce.GTC)  # no ask yet
    assert relative_spread(eng) is None
    assert microprice_deviation(eng) is None


def test_relative_spread_positive_and_reasonable():
    eng = _book_with_bid_ask(bid_px=99, ask_px=101)
    spread = relative_spread(eng)
    assert spread is not None and spread > 0
    assert spread == (101 - 99) / 100.0


def test_snapshot_to_vector_shape_and_none_handling():
    eng = _core.MatchingEngine(tick_size=1)
    assert snapshot_to_vector(eng) is None  # empty book

    eng.submit_limit(Side.Buy, 99, 10, TimeInForce.GTC)
    assert snapshot_to_vector(eng) is None  # one-sided book

    eng.submit_limit(Side.Sell, 101, 10, TimeInForce.GTC)
    vec = snapshot_to_vector(eng)
    assert vec is not None
    assert vec.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(vec))


def test_feature_window_accumulates_and_reports_volatility():
    eng = _book_with_bid_ask()
    window = FeatureWindow(maxlen=5)
    for _ in range(5):
        assert window.push(eng) is True
    assert window.is_full()
    arr = window.as_array()
    assert arr.shape[0] == 5
    # Constant mid price -> zero realized volatility.
    assert window.realized_volatility() == 0.0


def test_feature_window_push_returns_false_on_empty_book():
    eng = _core.MatchingEngine(tick_size=1)
    window = FeatureWindow(maxlen=5)
    assert window.push(eng) is False
