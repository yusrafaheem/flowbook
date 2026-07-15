"""Tests for the market-making strategies and the backtest loop.

The key regression this guards against: fills that happen on a resting
quote *after* it was submitted (i.e. someone else's order walks into it
later) must still be recognized and applied to inventory/cash. An earlier
version of this code only accounted for fills returned directly by the
strategy's own `submit_limit` call, which silently produced zero fills
whenever a quote joined (rather than crossed) the book -- the common case.
See `MakerBase.on_trade` and its docstring.
"""

import numpy as np

from flowbook import _core
from flowbook._core import Side, TimeInForce
from flowbook.backtest import run_backtest
from flowbook.features import FeatureWindow
from flowbook.simulator import MarketSimulator, SimulatorConfig
from flowbook.strategies import AvellanedaStoikovConfig, AvellanedaStoikovMaker, FixedSpreadMaker


def test_maker_recognizes_fill_on_resting_order():
    """A quote that rests (does not cross on submission) must still be
    marked filled when a later, independent order matches against it.

    Ambient liquidity is seeded well away from the price the maker will
    quote (90 / 110) so that when the maker joins the touch at 99 / 101
    its order is the *only* one at that price level -- guaranteeing it is
    at the front of the FIFO queue and must be the one hit next, rather
    than racing an ambient order already resting at the same price.
    """
    eng = _core.MatchingEngine(tick_size=1)
    eng.submit_limit(Side.Buy, 90, 50, TimeInForce.GTC)
    eng.submit_limit(Side.Sell, 110, 50, TimeInForce.GTC)

    maker = FixedSpreadMaker(half_spread_ticks=1, quote_size=5)
    window = FeatureWindow(maxlen=10)
    window.push(eng)

    result = maker.step(eng, window)  # mid = 100 -> quotes at 99 / 101
    assert result["quoted"] is True
    assert result["bid_px"] == 99
    assert maker.n_fills == 0  # quote rested, did not cross

    # An independent aggressive sell now walks into the maker's bid at 99
    # (the ambient bid at 90 is worse and untouched).
    _, trades = eng.submit_market(Side.Sell, 3)
    for t in trades:
        maker.on_trade(t)

    assert maker.n_fills == 1
    assert maker.inventory == 3  # bought 3 lots


def test_backtest_produces_nonzero_fills_and_finite_metrics():
    cfg = SimulatorConfig(seed=7)
    result = run_backtest(
        AvellanedaStoikovMaker(AvellanedaStoikovConfig(session_length=3000)),
        n_steps=3000,
        sim_config=cfg,
    )
    assert result.n_fills > 0, "expected at least some fills over a 3000-step session"
    assert np.isfinite(result.final_pnl)
    assert np.isfinite(result.sharpe)
    assert np.isfinite(result.max_drawdown)
    assert result.max_drawdown <= 0.0  # drawdown is measured as a non-positive number


def test_inventory_limit_is_respected():
    cfg = AvellanedaStoikovConfig(session_length=2000, inventory_limit=5, quote_size=5)
    result = run_backtest(AvellanedaStoikovMaker(cfg), n_steps=2000, sim_config=SimulatorConfig(seed=3))
    assert np.max(np.abs(result.inventory_path)) <= 5 + cfg.quote_size
    # (allow one extra fill's worth of slop between the limit check and the
    # fill that pushes inventory over it)


def test_backtest_is_deterministic_given_a_seed():
    cfg = SimulatorConfig(seed=123)
    r1 = run_backtest(AvellanedaStoikovMaker(), n_steps=1000, sim_config=SimulatorConfig(seed=123))
    r2 = run_backtest(AvellanedaStoikovMaker(), n_steps=1000, sim_config=SimulatorConfig(seed=123))
    assert r1.final_pnl == r2.final_pnl
    assert np.array_equal(r1.pnl_path, r2.pnl_path)
