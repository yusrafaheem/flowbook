"""Market-making strategies.

Implements Avellaneda & Stoikov (2008), "High-frequency trading in a limit
order book" -- the canonical inventory-risk-aware market-making model, still
the standard reference point/baseline in market-making research (and a
frequent whiteboard topic in quant interviews).

Reservation price:      r = mid - q * gamma * sigma^2 * (T - t)
Optimal half-spread:    delta = (gamma * sigma^2 * (T - t)) / 2
                               + (1 / gamma) * ln(1 + gamma / kappa)

  q        current inventory (signed; +long / -short)
  gamma    risk aversion
  sigma    volatility of the mid-price (realized, estimated from the window)
  T - t    remaining trading horizon (as a fraction of the session)
  kappa    order-book liquidity parameter (arrival rate decay of market
           orders vs. quoted distance) -- controls how aggressively the
           strategy tightens quotes to compete for flow

A subtlety that is easy to get wrong (and did, during development of this
module -- see the fill-tracking design note on `MakerBase`): a resting quote
can be filled by *someone else's* incoming order, arbitrarily long after it
was placed. The strategy only learns about that fill by inspecting the
trade stream the matching engine produces on *every* call, not just the
calls it makes itself. `MakerBase.on_trade` is the single place fills are
recognized and applied to inventory/cash, whether the fill happened
instantly (the strategy's own order crossed on submission) or later (an
outside order walked into a resting quote).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from flowbook import _core
from flowbook._core import Side, TimeInForce
from flowbook.features import FeatureWindow


class MakerBase:
    """Shared order-tracking and fill-accounting logic for two-sided
    quoting strategies.

    Subclasses implement `compute_quotes(engine, window) -> (bid_px, ask_px)`
    and get `step()`, fill tracking, and mark-to-market for free.
    """

    def __init__(self, quote_size: int = 5, inventory_limit: int = 200):
        self.quote_size = quote_size
        self.inventory_limit = inventory_limit
        self.inventory: int = 0
        self.cash: float = 0.0
        self.n_fills: int = 0

        self._bid_order_id: int | None = None
        self._ask_order_id: int | None = None
        self._bid_px: int | None = None
        self._ask_px: int | None = None
        # Every order ID this strategy has ever submitted, mapped to the
        # side it was submitted on. Never pruned: engine order IDs are
        # never reused, so stale entries are harmless and let us recognize
        # a fill on an order that has since been fully consumed and
        # dropped from the book.
        self._my_orders: dict[int, Side] = {}

    def compute_quotes(self, engine: "_core.MatchingEngine", window: FeatureWindow):
        raise NotImplementedError

    def on_trade(self, trade) -> None:
        """Recognizes and applies a fill if `trade` involves one of this
        strategy's orders, on either the aggressor or resting side (a
        strategy order can be either, depending on whether it crossed
        immediately on submission or was hit later while resting).
        """
        side = self._my_orders.get(trade.aggressor_id)
        if side is None:
            side = self._my_orders.get(trade.resting_id)
        if side is None:
            return
        signed_qty = trade.quantity if side == Side.Buy else -trade.quantity
        self.inventory += signed_qty
        self.cash -= signed_qty * trade.price
        self.n_fills += 1

    def _requote_side(self, engine: "_core.MatchingEngine", side: Side, price: int) -> list:
        """Cancels the existing order on `side` if the price changed and
        submits a fresh one. Returns any trades from the new submission
        (already applied via on_trade by the caller's convention: the
        caller is responsible for feeding these back through on_trade).
        """
        current_id = self._bid_order_id if side == Side.Buy else self._ask_order_id
        current_px = self._bid_px if side == Side.Buy else self._ask_px

        if price == current_px:
            return []

        if current_id is not None:
            engine.cancel(current_id)

        order_id, trades = engine.submit_limit(side, price, self.quote_size, TimeInForce.GTC)
        self._my_orders[order_id] = side
        if side == Side.Buy:
            self._bid_order_id, self._bid_px = order_id, price
        else:
            self._ask_order_id, self._ask_px = order_id, price
        return trades

    def _withdraw_side(self, engine: "_core.MatchingEngine", side: Side) -> None:
        current_id = self._bid_order_id if side == Side.Buy else self._ask_order_id
        if current_id is not None:
            engine.cancel(current_id)
        if side == Side.Buy:
            self._bid_order_id, self._bid_px = None, None
        else:
            self._ask_order_id, self._ask_px = None, None

    def step(self, engine: "_core.MatchingEngine", window: FeatureWindow) -> dict:
        mid = engine.mid_price()
        if mid is None:
            return {"quoted": False}

        bid_px, ask_px = self.compute_quotes(engine, window)

        # Clamp to be at least as competitive as the current touch: a
        # theoretical quote that is wider than the prevailing NBBO would
        # otherwise just rest behind the touch and never trade. This is
        # "join, or improve if the model says to", not "quote into dead
        # air" -- see module docstring.
        best_bid = engine.best_bid()
        best_ask = engine.best_ask()
        if best_bid is not None:
            bid_px = max(bid_px, best_bid)
        if best_ask is not None:
            ask_px = min(ask_px, best_ask)
        if bid_px >= ask_px:
            ask_px = bid_px + 1

        all_trades = []
        if self.inventory < self.inventory_limit:
            all_trades += self._requote_side(engine, Side.Buy, bid_px)
        else:
            self._withdraw_side(engine, Side.Buy)

        if self.inventory > -self.inventory_limit:
            all_trades += self._requote_side(engine, Side.Sell, ask_px)
        else:
            self._withdraw_side(engine, Side.Sell)

        for t in all_trades:
            self.on_trade(t)

        return {"quoted": True, "bid_px": bid_px, "ask_px": ask_px, "mid": mid}

    def mark_to_market(self, engine: "_core.MatchingEngine") -> float:
        """Cash + inventory valued at the current mid price."""
        mid = engine.mid_price() or 0.0
        return self.cash + self.inventory * mid


@dataclass
class AvellanedaStoikovConfig:
    gamma: float = 0.1          # risk aversion
    kappa: float = 1.5          # book liquidity parameter
    session_length: int = 5000  # total steps in a session, for the (T - t) term
    quote_size: int = 5
    min_half_spread_ticks: float = 1.0
    inventory_limit: int = 200  # hard cap: stop adding to inventory beyond this


class AvellanedaStoikovMaker(MakerBase):
    """Quotes both sides of the book using the A&S reservation price and
    optimal spread, recomputed every step but only re-submitted to the
    engine when the resulting price actually changes (see MakerBase).
    """

    def __init__(self, config: AvellanedaStoikovConfig | None = None):
        self.cfg = config or AvellanedaStoikovConfig()
        super().__init__(quote_size=self.cfg.quote_size, inventory_limit=self.cfg.inventory_limit)
        self._t: int = 0

    def reservation_price(self, mid: float, sigma: float) -> float:
        cfg = self.cfg
        tau = max(0.0, 1.0 - self._t / cfg.session_length)
        return mid - self.inventory * cfg.gamma * (sigma ** 2) * tau

    def optimal_half_spread(self, sigma: float) -> float:
        cfg = self.cfg
        tau = max(0.0, 1.0 - self._t / cfg.session_length)
        inventory_term = 0.5 * cfg.gamma * (sigma ** 2) * tau
        liquidity_term = (1.0 / cfg.gamma) * math.log(1.0 + cfg.gamma / cfg.kappa)
        return inventory_term + liquidity_term

    def compute_quotes(self, engine: "_core.MatchingEngine", window: FeatureWindow):
        self._t += 1
        mid = engine.mid_price()
        sigma = window.realized_volatility() or 1e-6
        tick = engine.tick_size or 1

        r = self.reservation_price(mid, sigma * mid)  # scale sigma (log-return) to price units
        half_spread_ticks = max(
            self.cfg.min_half_spread_ticks, self.optimal_half_spread(sigma * mid) / tick
        )
        bid_px = int(round(r / tick)) - int(round(half_spread_ticks))
        ask_px = int(round(r / tick)) + int(round(half_spread_ticks))
        return bid_px, ask_px


class FixedSpreadMaker(MakerBase):
    """Naive baseline: quotes a constant tick spread around mid regardless
    of inventory or volatility. A control to show what the A&S
    inventory/volatility terms actually buy you.
    """

    def __init__(self, half_spread_ticks: int = 3, quote_size: int = 5, inventory_limit: int = 200):
        super().__init__(quote_size=quote_size, inventory_limit=inventory_limit)
        self.half_spread_ticks = half_spread_ticks

    def compute_quotes(self, engine: "_core.MatchingEngine", window: FeatureWindow):
        mid = engine.mid_price()
        tick = engine.tick_size or 1
        mid_ticks = int(round(mid / tick))
        return mid_ticks - self.half_spread_ticks, mid_ticks + self.half_spread_ticks
