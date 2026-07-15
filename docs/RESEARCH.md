# Research background

This document maps each modeling choice in flowbook to the paper or result
it comes from, and is honest about where the project's own experiments
land relative to that literature. If you only read one section, read
"Known limitations" -- it explains what the shipped numbers do and don't
show.

## Order flow and book dynamics (`flowbook/simulator.py`)

There is no live exchange feed in this project, so `MarketSimulator`
generates synthetic order flow calibrated to match documented empirical
regularities, rather than to any specific instrument:

- **Poisson arrivals per event type.** Cont, Stoikov & Talreja (2010), *A
  stochastic model for order book dynamics*, models limit orders, market
  orders, and cancellations as independent Poisson processes with
  constant intensities. flowbook uses the same structure
  (`limit_rate`, `market_rate`, `cancel_rate`).
- **Placement concentrated near the touch.** Bouchaud, Mezard & Potters
  (2002), *Statistical properties of stock order books: empirical results
  and models*, documents that new limit order placement decays roughly as
  a power law in distance from the opposite touch. flowbook approximates
  this with a geometric distribution (`placement_decay`), which has the
  same qualitative "most mass near zero" shape and is simpler to calibrate.
- **Queue-reactive cancellation.** Huang, Lehalle & Rosenbaum (2015),
  *Simulating and analyzing order book data: the queue-reactive model*,
  ties cancellation intensity to queue depth/position rather than treating
  it as uniform. flowbook's cancellation is currently a simplification
  (uniform over live orders) -- a documented gap, see below.
- **Order-flow imbalance regime.** The Ornstein-Uhlenbeck "theta" process
  driving the buy/sell mix (`imbalance_reversion`, `imbalance_vol`) is a
  simplified stand-in for the empirical fact that order flow is
  autocorrelated and that this autocorrelation is what makes order-book
  imbalance predictive of short-horizon returns in the first place -- see
  Cont, Kukanov & Stoikov (2014), *The Price Impact of Order Book Events*,
  and the broader OBI literature it sits in.

## Market making (`flowbook/strategies.py`)

`AvellanedaStoikovMaker` implements Avellaneda & Stoikov (2008),
*High-frequency trading in a limit order book*: a reservation price that
shifts away from mid in proportion to signed inventory (so the strategy
naturally leans into unwinding a position) and an optimal half-spread that
trades off inventory risk against fill probability via a book-liquidity
parameter kappa. It remains the standard reference point for
inventory-aware market making and a frequent basis for quant interview
questions on the topic.

`FixedSpreadMaker` is a deliberately naive control (constant spread,
no inventory or volatility awareness) to isolate what the A&S terms
actually buy you in a backtest.

**Design note surfaced during development:** a market maker's resting
quote can be filled by someone else's order arbitrarily long after it was
placed. An early version of this code only recognized fills returned
directly by the strategy's own `submit_limit` call, which is correct only
when a quote crosses immediately -- and silently produces zero fills
whenever a quote joins (rather than crosses) the book, which is the common
case for a maker. See `MakerBase.on_trade` and `tests/test_strategies.py::
test_maker_recognizes_fill_on_resting_order`, which is a regression test
for exactly this.

## Short-horizon prediction (`flowbook/research.py`)

The model is a small, single-head self-attention encoder over a window of
LOB feature vectors, in the same family as the 2025 transformer-based LOB
forecasters:

- Zhang, Zohren & Roberts (2019), *DeepLOB: Deep Convolutional Neural
  Networks for Limit Order Books* -- the CNN-based predecessor these
  transformer models replace the convolutional stack of.
- Berti et al. (2025), *LiT: limit order book transformer* -- structured
  patches + self-attention over LOB data, no convolutional layers.
- *TLOB: A Novel Transformer Model with Dual Attention for Stock Price
  Trend Prediction with Limit Order Book Data* (2025).
- Backhouse et al. (2025), *Painting the market: generative diffusion
  models for financial limit order book simulation and forecasting* --
  a different (generative, not discriminative) 2025 approach to the same
  data, worth knowing about even though flowbook doesn't implement it.

flowbook's model is intentionally much smaller than any of the above (one
attention block, `d_model` in the tens, no framework) because the point of
this module is a **from-scratch, verified** implementation rather than a
competitive benchmark result -- see `grad_check()` in `research.py`, which
validates every analytic gradient in the hand-derived backward pass
against numerical (finite-difference) gradients before anything is
trusted to train on real data. This mirrors the author's `vectorgrad`
project (a from-scratch reverse-mode autodiff engine): the belief being
tested here is that you should be able to derive and verify your own
gradients, not just trust a framework's autograd.

The classification target follows the FI-2010 / DeepLOB convention:
3-class (down / flat / up) based on comparing the horizon log-return to a
threshold, rather than using its raw sign, so near-zero moves aren't
forced into a direction.

## Known limitations

**Order-flow-imbalance out-of-sample accuracy is close to the majority-
class baseline, not clearly above it.** `scripts/train_research_model.py`
trains on 30 independent simulation seeds and evaluates on 15 disjoint
seeds. In-sample accuracy climbs well above baseline (the model fits its
training data -- confirms the optimizer/forward/backward wiring works).
Out-of-sample accuracy on the held-out seeds lands within a couple of
points of the majority-class baseline. Two things are true at once here:

1. The underlying correlation between order-book imbalance and future
   returns is real and measurable in this simulator (~0.08-0.16 at the
   horizons used, consistent in sign and rough magnitude with the
   empirical OBI literature).
2. A small model trained on a few thousand synthetic examples, evaluated
   against *independently sampled* regime realizations, does not reliably
   turn that correlation into classification accuracy above chance here.
   The most likely causes, roughly in order of suspected impact: (a) the
   3-class threshold framing discards much of the linear signal a
   regression or 2-class framing would retain, (b) the OU regime's long
   persistence relative to the prediction horizon means each seed
   contributes only a handful of effectively-independent "episodes", so
   30 training seeds is less data diversity than it sounds like, and
   (c) the model has no explicit regularization (no dropout, no weight
   decay) given how small it already is.

This is reported rather than hidden or tuned away because a synthetic
backtest that "works" on the first honest evaluation is a bigger red flag
than one that doesn't -- and because the correctness claim this module
actually stands behind is the gradient check, not the accuracy number.
Ideas for closing the gap, roughly in order of expected effort-to-payoff:
switch to a continuous regression target (predict the horizon return, not
its bucketed class) and evaluate with information coefficient / rank
correlation instead of classification accuracy; add L2 regularization;
replace the OU regime with several independent regimes per simulated
episode so more of the training set is effectively i.i.d.; or swap the
synthetic generator for a real LOBSTER/ITCH replay through the same
feature interface, where the imbalance-return relationship is well
documented rather than something this project has to induce synthetically
in the first place.

**Cancellation is uniform, not queue-reactive.** The queue-reactive model
(Huang, Lehalle & Rosenbaum 2015) ties cancel probability to a resting
order's queue position and distance from mid; flowbook currently cancels
uniformly at random among all live orders. This is the most likely next
change to the simulator if it's extended.

**Concurrency is demonstrated at the ingestion layer, not inside the
matching engine itself.** `SpscRingBuffer` decouples order intake from
matching (see `cpp/include/concurrent_feed_handler.hpp`), but
`MatchingEngine` itself is single-threaded internally -- this is a
deliberate scope decision (a genuinely concurrent matching core, e.g. with
per-price-level locking or a different data structure entirely, is a
substantially larger project) rather than an oversight.
