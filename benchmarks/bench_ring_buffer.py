"""Throughput benchmark for the SPSC ring buffer under real concurrent
producer/consumer threads (GIL released on both sides -- see bindings.cpp).

Run with:  python benchmarks/bench_ring_buffer.py
"""

from __future__ import annotations

import threading
import time

from flowbook import _core
from flowbook._core import CommandType, Side, TimeInForce


def bench_concurrent(n_items: int = 2_000_000) -> float:
    rb = _core.RingBuffer()
    received = [0]

    def producer():
        i = 0
        while i < n_items:
            if rb.push(CommandType.Cancel, Side.Buy, 0, 0, TimeInForce.GTC, i):
                i += 1

    def consumer():
        count = 0
        while count < n_items:
            if rb.pop() is not None:
                count += 1
        received[0] = count

    t0 = time.perf_counter()
    tp = threading.Thread(target=producer)
    tc = threading.Thread(target=consumer)
    tc.start()
    tp.start()
    tp.join()
    tc.join()
    elapsed = time.perf_counter() - t0

    assert received[0] == n_items
    return n_items / elapsed


if __name__ == "__main__":
    print("flowbook SPSC ring buffer benchmark")
    print("-" * 60)
    throughput = bench_concurrent()
    print(f"concurrent producer/consumer: {throughput:>14,.0f} msgs/sec "
          f"(2 real OS threads, GIL released on push/pop)")
