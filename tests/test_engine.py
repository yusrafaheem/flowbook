"""Correctness tests for the C++ matching engine (via pybind11 bindings).

Covers price-time priority, partial fills, cancellation, market-order
sweeps, and the derived mid/microprice quantities -- the properties any
matching engine must get right before anything built on top of it (market
making, ML features) can be trusted.
"""

import pytest

from flowbook import _core
from flowbook._core import Side, TimeInForce


def make_engine(tick_size=1):
    return _core.MatchingEngine(tick_size=tick_size)


def test_resting_order_no_trade():
    eng = make_engine()
    oid, trades = eng.submit_limit(Side.Buy, 100, 10, TimeInForce.GTC)
    assert trades == []
    assert eng.best_bid() == 100
    assert eng.best_ask() is None


def test_crossing_limit_order_generates_trade():
    eng = make_engine()
    eng.submit_limit(Side.Buy, 100, 10, TimeInForce.GTC)
    _, trades = eng.submit_limit(Side.Sell, 100, 4, TimeInForce.GTC)
    assert len(trades) == 1
    assert trades[0].price == 100
    assert trades[0].quantity == 4
    # Partial fill: 6 left resting on the bid side.
    snap = eng.snapshot(1)
    assert snap.bids[0].total_quantity == 6


def test_price_time_priority_fifo():
    eng = make_engine()
    id1, _ = eng.submit_limit(Side.Buy, 100, 5, TimeInForce.GTC)
    id2, _ = eng.submit_limit(Side.Buy, 100, 5, TimeInForce.GTC)
    # An incoming sell for 5 should fill the *first* resting order (id1),
    # not the second, even though both are at the same price.
    _, trades = eng.submit_limit(Side.Sell, 100, 5, TimeInForce.GTC)
    assert len(trades) == 1
    assert trades[0].resting_id == id1
    snap = eng.snapshot(1)
    assert snap.bids[0].total_quantity == 5  # id2 untouched


def test_price_priority_best_price_first():
    eng = make_engine()
    eng.submit_limit(Side.Sell, 102, 5, TimeInForce.GTC)
    eng.submit_limit(Side.Sell, 100, 5, TimeInForce.GTC)  # better (lower) ask
    eng.submit_limit(Side.Sell, 101, 5, TimeInForce.GTC)
    _, trades = eng.submit_limit(Side.Buy, 102, 5, TimeInForce.GTC)
    assert len(trades) == 1
    assert trades[0].price == 100  # best ask, not first-submitted


def test_cancel_removes_order():
    eng = make_engine()
    oid, _ = eng.submit_limit(Side.Buy, 100, 10, TimeInForce.GTC)
    assert eng.cancel(oid) is True
    assert eng.cancel(oid) is False  # already cancelled
    assert eng.best_bid() is None


def test_cancelled_order_is_skipped_not_matched():
    eng = make_engine()
    id1, _ = eng.submit_limit(Side.Buy, 100, 5, TimeInForce.GTC)
    id2, _ = eng.submit_limit(Side.Buy, 100, 5, TimeInForce.GTC)
    eng.cancel(id1)
    _, trades = eng.submit_limit(Side.Sell, 100, 5, TimeInForce.GTC)
    assert len(trades) == 1
    assert trades[0].resting_id == id2


def test_market_order_sweeps_multiple_levels():
    eng = make_engine()
    eng.submit_limit(Side.Sell, 100, 5, TimeInForce.GTC)
    eng.submit_limit(Side.Sell, 101, 5, TimeInForce.GTC)
    _, trades = eng.submit_market(Side.Buy, 8)
    assert len(trades) == 2
    assert trades[0].price == 100 and trades[0].quantity == 5
    assert trades[1].price == 101 and trades[1].quantity == 3
    assert eng.best_ask() == 101  # 2 lots remain at 101


def test_market_order_never_rests():
    eng = make_engine()
    _, trades = eng.submit_market(Side.Buy, 100)  # empty book, nothing to hit
    assert trades == []
    assert eng.best_bid() is None and eng.best_ask() is None


def test_ioc_limit_does_not_rest():
    eng = make_engine()
    eng.submit_limit(Side.Sell, 100, 5, TimeInForce.GTC)
    _, trades = eng.submit_limit(Side.Buy, 100, 10, TimeInForce.IOC)
    assert len(trades) == 1 and trades[0].quantity == 5
    # Unfilled remainder (5 lots) must NOT rest, since it was IOC.
    assert eng.best_bid() is None


def test_mid_and_microprice():
    eng = make_engine()
    eng.submit_limit(Side.Buy, 99, 100, TimeInForce.GTC)
    eng.submit_limit(Side.Sell, 101, 100, TimeInForce.GTC)
    assert eng.mid_price() == pytest.approx(100.0)
    # Symmetric depth -> microprice should equal mid.
    assert eng.microprice() == pytest.approx(100.0)


def test_microprice_leans_toward_thin_side():
    eng = make_engine()
    eng.submit_limit(Side.Buy, 99, 10, TimeInForce.GTC)   # thin bid
    eng.submit_limit(Side.Sell, 101, 1000, TimeInForce.GTC)  # deep ask
    # Stoikov's microprice weights P_bid by ask volume and P_ask by bid
    # volume: P_micro = (P_bid * V_ask + P_ask * V_bid) / (V_bid + V_ask).
    # A thin bid (small V_bid) gives P_bid a near-1 weight -> the estimate
    # is pulled toward the thin side's own quote (below mid here), not
    # away from it.
    assert eng.microprice() < eng.mid_price()


def test_snapshot_depth_and_ordering():
    eng = make_engine()
    for i, px in enumerate([98, 99, 100]):
        eng.submit_limit(Side.Buy, px, 10 + i, TimeInForce.GTC)
    snap = eng.snapshot(levels=10)
    prices = [l.price for l in snap.bids]
    assert prices == sorted(prices, reverse=True)  # best (highest) bid first
    assert prices[0] == 100
