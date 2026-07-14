"""Event loop tying the simulator, a market-making strategy, and feature
extraction together, plus the performance/risk metrics used to evaluate a
strategy run.

Metrics reported (`BacktestResult`):
  - final / path of mark-to-market PnL
  - Sharpe ratio of per-step PnL changes (unannualized -- there is no
    calendar here, only simulated steps; reported as a relative comparison
    between strategies, not as an absolute annualized number)
  - max drawdown
  - inventory path + time-average |inventory| (risk taken)
  - fill count / quote-to-trade ratio (how often quotes actually execute)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flowbook.features import FeatureWindow
from flowbook.simulator import MarketSimulator, SimulatorConfig


@dataclass
class BacktestResult:
    pnl_path: np.ndarray
    inventory_path: np.ndarray
    mid_path: np.ndarray
    n_quotes: int
    n_fills: int

    @property
    def final_pnl(self) -> float:
        return float(self.pnl_path[-1]) if len(self.pnl_path) else 0.0

    @property
    def sharpe(self) -> float:
        if len(self.pnl_path) < 3:
            return 0.0
        rets = np.diff(self.pnl_path)
        std = np.std(rets)
        if std == 0:
            return 0.0
        return float(np.mean(rets) / std * np.sqrt(len(rets)))

    @property
    def max_drawdown(self) -> float:
        if len(self.pnl_path) == 0:
            return 0.0
        running_max = np.maximum.accumulate(self.pnl_path)
        drawdown = self.pnl_path - running_max
        return float(drawdown.min())

    @property
    def mean_abs_inventory(self) -> float:
        return float(np.mean(np.abs(self.inventory_path))) if len(self.inventory_path) else 0.0

    @property
    def quote_to_trade_ratio(self) -> float:
        return self.n_fills / self.n_quotes if self.n_quotes else 0.0

    def summary(self) -> str:
        return (
            f"final_pnl={self.final_pnl:.2f}  sharpe={self.sharpe:.3f}  "
            f"max_dd={self.max_drawdown:.2f}  mean|inv|={self.mean_abs_inventory:.2f}  "
            f"quote_to_trade={self.quote_to_trade_ratio:.3f}  n_quotes={self.n_quotes}"
        )


def run_backtest(strategy, n_steps: int = 5000, window_len: int = 100,
                  sim_config: SimulatorConfig | None = None,
                  background_flow_per_quote: int = 3) -> BacktestResult:
    """Runs `strategy` against a fresh `MarketSimulator` for `n_steps`.

    Each "step" of the backtest is: let the simulator generate a burst of
    background order flow (other participants), then let the strategy
    observe the resulting book and re-quote. This models the strategy as a
    slower participant relative to the ambient flow, which is the right
    regime for a market maker (it reacts to flow; it doesn't dominate it).
    """
    sim = MarketSimulator(sim_config or SimulatorConfig(seed=0))
    window = FeatureWindow(maxlen=window_len)

    pnl_path = []
    inventory_path = []
    mid_path = []
    n_quotes = 0
    n_fills = 0

    for _ in range(n_steps):
        for _ in range(background_flow_per_quote):
            event = sim.step()
            # A background participant's order may fill (or partially
            # fill) one of the strategy's resting quotes -- the strategy
            # only finds out by being shown the trade stream, since it did
            # not initiate this call itself. See MakerBase.on_trade.
            if hasattr(strategy, "on_trade"):
                for t in event.get("trades", []):
                    strategy.on_trade(t)

        window.push(sim.engine)
        result = strategy.step(sim.engine, window)

        if result.get("quoted"):
            n_quotes += 1

        mid = sim.engine.mid_price()
        pnl = strategy.mark_to_market(sim.engine)
        pnl_path.append(pnl)
        inventory_path.append(getattr(strategy, "inventory", 0))
        mid_path.append(mid if mid is not None else (mid_path[-1] if mid_path else 0.0))

    n_fills = getattr(strategy, "n_fills", None)
    if n_fills is None:
        # Fallback: infer fills from inventory changes (works for any
        # strategy that doesn't track n_fills itself).
        inv = np.array(inventory_path)
        n_fills = int(np.sum(np.diff(inv, prepend=0) != 0))

    return BacktestResult(
        pnl_path=np.array(pnl_path, dtype=np.float64),
        inventory_path=np.array(inventory_path, dtype=np.float64),
        mid_path=np.array(mid_path, dtype=np.float64),
        n_quotes=n_quotes,
        n_fills=n_fills,
    )
