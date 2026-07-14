"""Synthetic limit order book flow generator.

There is no live exchange feed wired into this project, so strategies and
the research model are developed and evaluated against a *simulated* order
flow instead. The generator is deliberately not "random noise": it targets
the stylized facts repeatedly documented in the market microstructure
literature (see docs/RESEARCH.md), specifically:

  1. Order arrivals are well approximated by independent Poisson processes
     per event type (limit add, cancel, market order) -- see Cont, Stoikov &
     Talreja (2010), "A stochastic model for order book dynamics".
  2. Limit order placement is concentrated near the touch and decays
     roughly as a power law in distance from mid -- see Bouchaud, Mezard &
     Potters (2002), "Statistical properties of stock order books".
  3. Order sizes cluster on round lots with a heavy right tail (a handful
     of large orders alongside many small ones) -- modeled here with a
     lognormal.
  4. A "queue-reactive" cancel intensity: orders deeper in the queue /
     further from mid are cancelled less often than orders at the touch,
     following Huang, Lehalle & Rosenbaum (2015), "Simulating and
     analyzing order book data: the queue-reactive model".
  5. A latent, slowly-varying order-flow imbalance regime: the buy/sell
     mix of incoming orders follows an Ornstein-Uhlenbeck process instead
     of a constant 50/50 split. This is what gives the book actual
     short-horizon price drift to predict (a purely symmetric flow --
     which an earlier version of this generator used -- keeps the book
     replenished so evenly that mid-price barely moves at all, which is
     realistic in one sense but leaves nothing for `flowbook.research` to
     learn). The mechanism is deliberately the textbook one: persistent
     order-flow imbalance forecasting short-term returns is the empirical
     basis for microprice/OBI-based signals in the first place (see
     Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book
     Events"), so this is a faithful (if simplified) way to inject a
     learnable signal rather than an arbitrary shortcut.

The same feature-extraction and strategy code runs unmodified against a
real feed (e.g. LOBSTER/ITCH replay) -- only this module would be swapped
out; `MarketSimulator.step()` is the sole integration point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flowbook import _core
from flowbook._core import Side, TimeInForce


@dataclass
class SimulatorConfig:
    tick_size: int = 1
    initial_mid: int = 10_000       # in ticks
    initial_spread_ticks: int = 4
    initial_depth_levels: int = 10
    initial_level_qty: int = 20

    # Poisson intensities (events per simulated second)
    limit_rate: float = 8.0
    market_rate: float = 1.2
    cancel_rate: float = 6.0

    # Placement decay: distance (in ticks) from best opposite price is drawn
    # from a geometric distribution with this success probability (higher =
    # tighter clustering at the touch).
    placement_decay: float = 0.35

    # Order size ~ lognormal(mu, sigma), rounded to >= 1 lot.
    size_mu: float = 1.6
    size_sigma: float = 0.6

    # Latent order-flow imbalance regime theta_t in (-1, 1): p(buy) for a
    # new order = 0.5 + 0.5 * theta_t. Evolves as a mean-reverting
    # Ornstein-Uhlenbeck process: d(theta) = -imbalance_reversion * theta
    # + imbalance_vol * N(0, 1), clipped to [-0.95, 0.95].
    imbalance_reversion: float = 0.02
    imbalance_vol: float = 0.03

    seed: int | None = None


class MarketSimulator:
    """Drives a `flowbook._core.MatchingEngine` with synthetic order flow.

    Usage:
        sim = MarketSimulator(SimulatorConfig(seed=0))
        for _ in range(10_000):
            events = sim.step()   # advances one Poisson "tick" of wall time
    """

    def __init__(self, config: SimulatorConfig | None = None):
        self.cfg = config or SimulatorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.engine = _core.MatchingEngine(tick_size=self.cfg.tick_size)
        self._live_order_ids: list[int] = []
        self.theta: float = 0.0  # latent order-flow imbalance regime
        self._seed_book()

    def _update_regime(self) -> None:
        cfg = self.cfg
        self.theta += -cfg.imbalance_reversion * self.theta + cfg.imbalance_vol * self.rng.standard_normal()
        self.theta = float(np.clip(self.theta, -0.95, 0.95))

    def _draw_side(self) -> Side:
        p_buy = 0.5 + 0.5 * self.theta
        return Side.Buy if self.rng.random() < p_buy else Side.Sell

    def _seed_book(self) -> None:
        cfg = self.cfg
        half_spread = cfg.initial_spread_ticks // 2
        best_bid = cfg.initial_mid - half_spread
        best_ask = cfg.initial_mid + half_spread
        for lvl in range(cfg.initial_depth_levels):
            bid_px = best_bid - lvl
            ask_px = best_ask + lvl
            oid_b, _ = self.engine.submit_limit(
                Side.Buy, bid_px, cfg.initial_level_qty, TimeInForce.GTC
            )
            oid_a, _ = self.engine.submit_limit(
                Side.Sell, ask_px, cfg.initial_level_qty, TimeInForce.GTC
            )
            self._live_order_ids += [oid_b, oid_a]

    def _draw_size(self) -> int:
        return max(1, int(round(self.rng.lognormal(self.cfg.size_mu, self.cfg.size_sigma))))

    def _draw_placement_distance(self) -> int:
        return int(self.rng.geometric(self.cfg.placement_decay)) - 1  # >= 0

    def step(self) -> dict:
        """Advances the simulator by one Poisson-clock tick.

        Draws which event type fires (limit add / market order / cancel)
        proportional to configured rates, applies it to the engine, and
        returns a small dict describing what happened (used for logging /
        dataset construction).
        """
        cfg = self.cfg
        self._update_regime()

        rates = np.array([cfg.limit_rate, cfg.market_rate, cfg.cancel_rate])
        probs = rates / rates.sum()
        event = self.rng.choice(["limit", "market", "cancel"], p=probs)

        result = {"event": event, "trades": [], "theta": self.theta}

        if event == "limit":
            side = self._draw_side()
            best_bid = self.engine.best_bid()
            best_ask = self.engine.best_ask()
            dist = self._draw_placement_distance()
            if side == Side.Buy:
                anchor = best_ask if best_ask is not None else cfg.initial_mid
                price = anchor - 1 - dist
            else:
                anchor = best_bid if best_bid is not None else cfg.initial_mid
                price = anchor + 1 + dist
            qty = self._draw_size()
            oid, trades = self.engine.submit_limit(side, price, qty, TimeInForce.GTC)
            self._live_order_ids.append(oid)
            result.update(order_id=oid, side=side, price=price, quantity=qty, trades=trades)

        elif event == "market":
            side = self._draw_side()
            qty = self._draw_size()
            oid, trades = self.engine.submit_market(side, qty)
            result.update(order_id=oid, side=side, quantity=qty, trades=trades)

        else:  # cancel
            if self._live_order_ids:
                idx = self.rng.integers(0, len(self._live_order_ids))
                oid = self._live_order_ids.pop(idx)
                cancelled = self.engine.cancel(oid)
                result.update(order_id=oid, cancelled=cancelled)

        return result

    def run(self, n_steps: int) -> list[dict]:
        return [self.step() for _ in range(n_steps)]


def research_sim_config(seed: int | None = None) -> SimulatorConfig:
    """A `SimulatorConfig` tuned for the research/prediction pipeline
    (`flowbook.research`) rather than for the market-making backtest.

    The default config is calibrated to look like a healthy, liquid,
    two-sided book (see module docstring's stylized facts) and, as a
    consequence, has mid-price stay within a couple of ticks for
    thousands of events -- realistic on very short horizons, but it means
    there is almost no directional structure for a predictive model to
    learn on any horizon short enough to be interesting. This preset
    amplifies market-order frequency and order-flow-imbalance persistence
    (lower `imbalance_reversion`, higher `imbalance_vol`) and thins
    resting depth, so the same underlying mechanism (persistent buy/sell
    imbalance -> book depletion on one side -> mid-price drift) actually
    produces enough movement to build a non-trivial dataset from. It is
    used by `flowbook.research.build_dataset`'s default and by the
    training script/notebook -- swap it for `SimulatorConfig()` (or a
    real-data loader) to evaluate under the more conservative flow
    assumptions.
    """
    return SimulatorConfig(
        seed=seed,
        market_rate=4.0,
        limit_rate=6.0,
        cancel_rate=3.0,
        initial_level_qty=10,
        initial_depth_levels=6,
        imbalance_reversion=0.005,
        imbalance_vol=0.05,
    )
