"""Throughput/latency micro-benchmark for the C++ matching engine.

Run with:  python benchmarks/bench_engine.py

Measures two things separately, since they stress different parts of the
engine:
  1. Pure limit-order insertion (no crossing) -- exercises the map lookup +
     deque push_back path.
  2. A mixed workload (limit adds, cancels, market orders, at roughly the
     ratios flowbook.simulator uses) -- exercises matching, the tombstone
     path, and reap_front together.

Reports throughput (ops/sec) and per-op latency percentiles (p50/p99/p999)
measured with time.perf_counter_ns, since Python-level timing dominates the
overhead here anyway (this benchmarks the *binding-level* engine, i.e. what
a Python caller actually experiences -- a pure-C++ benchmark would show
tighter numbers but wouldn't reflect real usage from this package).
"""

from __future__ import annotations

import time

import numpy as np

from flowbook import _core
from flowbook._core import Side, TimeInForce


def percentiles(latencies_ns: np.ndarray) -> dict:
    return {
        "p50_ns": float(np.percentile(latencies_ns, 50)),
        "p99_ns": float(np.percentile(latencies_ns, 99)),
        "p999_ns": float(np.percentile(latencies_ns, 99.9)),
    }


def bench_pure_insertion(n: int = 200_000) -> dict:
    eng = _core.MatchingEngine(tick_size=1)
    latencies = np.empty(n, dtype=np.float64)
    rng = np.random.default_rng(0)
    prices = rng.integers(9000, 10000, size=n)  # all below any ask -> never crosses

    for i in range(n):
        t0 = time.perf_counter_ns()
        eng.submit_limit(Side.Buy, int(prices[i]), 10, TimeInForce.GTC)
        latencies[i] = time.perf_counter_ns() - t0

    total_s = latencies.sum() / 1e9
    result = {"n": n, "ops_per_sec": n / total_s}
    result.update(percentiles(latencies))
    return result


def bench_mixed_workload(n: int = 200_000) -> dict:
    eng = _core.MatchingEngine(tick_size=1)
    rng = np.random.default_rng(0)
    live_ids: list[int] = []

    # Seed a two-sided book so market orders / crossing limits have something to hit.
    for i in range(50):
        eng.submit_limit(Side.Buy, 9990 - i, 10, TimeInForce.GTC)
        eng.submit_limit(Side.Sell, 10010 + i, 10, TimeInForce.GTC)

    latencies = np.empty(n, dtype=np.float64)
    event_types = rng.choice(["limit", "market", "cancel"], size=n, p=[0.55, 0.15, 0.30])
    sides = rng.choice([Side.Buy, Side.Sell], size=n)

    for i in range(n):
        ev = event_types[i]
        t0 = time.perf_counter_ns()
        if ev == "limit":
            price = int(rng.integers(9985, 10015))
            oid, _ = eng.submit_limit(sides[i], price, 5, TimeInForce.GTC)
            live_ids.append(oid)
        elif ev == "market":
            eng.submit_market(sides[i], 5)
        else:
            if live_ids:
                idx = int(rng.integers(0, len(live_ids)))
                eng.cancel(live_ids.pop(idx))
        latencies[i] = time.perf_counter_ns() - t0

    total_s = latencies.sum() / 1e9
    result = {"n": n, "ops_per_sec": n / total_s}
    result.update(percentiles(latencies))
    return result


if __name__ == "__main__":
    print("flowbook matching engine benchmark")
    print("-" * 60)

    r1 = bench_pure_insertion()
    print(f"pure insertion   : {r1['ops_per_sec']:>12,.0f} ops/sec   "
          f"p50={r1['p50_ns']:>7,.0f}ns  p99={r1['p99_ns']:>8,.0f}ns  p99.9={r1['p999_ns']:>9,.0f}ns")

    r2 = bench_mixed_workload()
    print(f"mixed workload   : {r2['ops_per_sec']:>12,.0f} ops/sec   "
          f"p50={r2['p50_ns']:>7,.0f}ns  p99={r2['p99_ns']:>8,.0f}ns  p99.9={r2['p999_ns']:>9,.0f}ns")
