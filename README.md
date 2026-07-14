# flowbook

A limit order book microstructure lab: a C++ matching engine (with a
lock-free concurrent ingestion path), Python market-making strategies
backtested against a calibrated synthetic order-flow simulator, and a
from-scratch (no ML framework) transformer-style model for short-horizon
price prediction, with every gradient in its hand-derived backward pass
verified against numerical finite differences.

This project exists to cover, in one codebase, the three things that
actually show up in quant/quant-dev interview loops: a correct, fast
matching engine (systems + C++), a market-making strategy with real
inventory/PnL accounting (quantitative finance), and a small ML model you
can actually explain the math of end to end (research). Each piece is
independently tested; none of it is a toy that only works in a demo script.

## Why this exists

Most "limit order book" repos on GitHub are one of: a C++ engine with no
strategy or research layer, a market-making notebook with no engine behind
it, or an ML repo that imports a framework and reports an accuracy number
with no discussion of whether it's real. This project tries to be
end-to-end and honest about which parts are solid and which parts are
proof-of-concept -- see `docs/RESEARCH.md`, "Known limitations", for an
explicit discussion of where the ML result is and isn't meaningful.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │      cpp/  (C++17, pybind11)         │
                    │                                       │
                    │  MatchingEngine                       │
                    │    - price-time priority (std::map    │
                    │      + per-level FIFO deque)          │
                    │    - lazy tombstone cancellation       │
                    │    - mid / microprice / depth snapshot │
                    │                                       │
                    │  SpscRingBuffer<4096>                 │
                    │    - wait-free, atomic head/tail       │
                    │    - decouples order intake from       │
                    │      matching (see concurrency below)  │
                    └───────────────┬───────────────────────┘
                                    │ pybind11 (flowbook._core)
                    ┌───────────────▼───────────────────────┐
                    │      python/flowbook/                 │
                    │                                        │
                    │  simulator.py   synthetic order flow   │
                    │                 (Poisson arrivals +    │
                    │                 OU imbalance regime)   │
                    │  features.py    OBI / microprice /     │
                    │                 spread extraction      │
                    │  strategies.py  Avellaneda-Stoikov +    │
                    │                 fixed-spread baseline   │
                    │  backtest.py    event loop, PnL/Sharpe/ │
                    │                 drawdown/fill tracking  │
                    │  research.py    from-scratch attention  │
                    │                 model + verified        │
                    │                 backward pass           │
                    └────────────────────────────────────────┘
```

## What's actually tested (34 tests, all passing)

- **Matching engine correctness** (`tests/test_engine.py`): price-time
  priority (FIFO within a level), price priority across levels, partial
  fills, cancellation (including that a cancelled order cannot be matched
  and does not linger in `best_bid()`/`snapshot()` -- see the lazy-reap
  design note in `cpp/src/matching_engine.cpp`), market-order sweeps
  across multiple levels, IOC semantics, and the microprice/mid-price
  calculations (including *direction*: verified against Stoikov (2018)'s
  microprice, which pulls the estimate toward the thinner side, not away
  from it -- easy to get backwards, which is exactly what happened during
  development and is now a regression test).
- **Concurrency** (`tests/test_ring_buffer.py`): a real multi-threaded
  stress test -- two actual OS threads (GIL released on `push`/`pop`)
  moving 200,000 messages through the SPSC ring buffer with an exact
  order/no-loss/no-duplication assertion at the end, not just a "it didn't
  crash" smoke test.
- **Strategy fill accounting** (`tests/test_strategies.py`): including a
  regression test for a real bug found during development -- a resting
  quote filled by *someone else's* later order wasn't being recognized as
  a fill at all (see `MakerBase.on_trade` and the design note in
  `strategies.py`).
- **ML gradient correctness** (`tests/test_research.py`): every parameter
  gradient in the hand-derived attention backward pass checked against
  central-difference numerical gradients, for every class label.

Run them yourself:

```bash
pip install -e .
pytest tests/ -v
```

## Benchmarks

Measured in this project's own dev sandbox (Ubuntu 22.04, aarch64) -- not
a tuned benchmark rig, and not a claim about production latency on real
hardware. Reproduce with `python benchmarks/bench_engine.py` and
`python benchmarks/bench_ring_buffer.py`; relative numbers (pure insertion
vs. mixed workload, single- vs multi-threaded) are more meaningful here
than the absolute figures.

| Workload                                   | Throughput        | p50      | p99      |
|---------------------------------------------|------------------:|---------:|---------:|
| Pure limit-order insertion (no crossing)     | ~2.2M ops/sec      | 375 ns   | 916 ns   |
| Mixed (55% limit / 15% market / 30% cancel)  | ~0.61M ops/sec     | 1.25 μs  | 7 μs     |
| SPSC ring buffer, 2 real threads             | ~0.7M msgs/sec     | --       | --       |

These are Python-binding-level numbers (i.e. what a caller of this package
actually experiences through pybind11), not raw C++ numbers -- a
pure-C++ microbenchmark would show tighter latencies but wouldn't reflect
real usage of this package.

## Market making example

```python
from flowbook.strategies import AvellanedaStoikovMaker, AvellanedaStoikovConfig, FixedSpreadMaker
from flowbook.backtest import run_backtest
from flowbook.simulator import SimulatorConfig

cfg = SimulatorConfig(seed=1)
as_result = run_backtest(AvellanedaStoikovMaker(AvellanedaStoikovConfig(session_length=5000)),
                          n_steps=5000, sim_config=cfg)
fixed_result = run_backtest(FixedSpreadMaker(), n_steps=5000, sim_config=SimulatorConfig(seed=1))

print("Avellaneda-Stoikov:", as_result.summary())
print("Fixed spread      :", fixed_result.summary())
```

## Research model

```bash
python scripts/train_research_model.py
```

Trains the from-scratch attention model (`flowbook.research`) on 30
independent synthetic order-flow seeds, evaluates on 15 disjoint seeds, and
prints in-sample vs. out-of-sample accuracy against majority-class
baselines. Read `docs/RESEARCH.md` before interpreting the numbers --
the honest summary is: the gradient implementation is verified correct,
the underlying OBI-return correlation the model is trying to learn is
real and measurable (~0.08-0.16 in this simulator), and out-of-sample
classification accuracy on held-out regime realizations is close to,
not clearly above, the majority-class baseline with this model size and
dataset size. That gap, and some ideas for closing it, are discussed
there rather than papered over.

## Research this project is built on

See `docs/RESEARCH.md` for the full mapping. Short version:

- Cont, Stoikov & Talreja (2010) -- Poisson order-flow model
- Bouchaud, Mezard & Potters (2002) -- order placement decay near the touch
- Huang, Lehalle & Rosenbaum (2015) -- queue-reactive cancellation model
- Cont, Kukanov & Stoikov (2014) -- order-book-imbalance price impact
- Avellaneda & Stoikov (2008) -- inventory-aware market making
- Stoikov (2018) -- the microprice
- Zhang, Zohren & Roberts (2019, DeepLOB), Berti et al. (2025, LiT), and
  TLOB (2025) -- deep/transformer LOB forecasting

## Project layout

```
cpp/            C++17 matching engine + SPSC ring buffer + pybind11 bindings
python/flowbook/  simulator, features, strategies, backtest, research
tests/          pytest suite (engine, strategies, features, ring buffer, research)
benchmarks/     throughput/latency scripts
scripts/        end-to-end training run
docs/           research background and known limitations
```

## Building

```bash
pip install -e .            # compiles the C++ extension via pybind11 + setuptools
pip install -e ".[research]"  # + numpy/pandas/matplotlib for the research scripts
```

Requires a C++17 compiler (tested with g++ 11). No GPU, and no ML
framework dependency -- `flowbook.research`'s model is pure NumPy.

## License

MIT -- see `LICENSE`.
