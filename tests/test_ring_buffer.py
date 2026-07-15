"""Correctness tests for the SPSC lock-free ring buffer
(cpp/include/concurrent_feed_handler.hpp), exercised with real OS threads.

`RingBuffer.push`/`pop` release the GIL (see bindings.cpp), so a Python
producer thread and consumer thread genuinely run concurrently here rather
than just taking turns on a single core -- this is actually testing the
atomic head/tail protocol under real concurrent access, not simulating it.
"""

import threading

from flowbook import _core
from flowbook._core import CommandType, Side, TimeInForce


def test_single_threaded_push_pop_preserves_order_and_values():
    rb = _core.RingBuffer()
    assert rb.pop() is None  # empty

    for i in range(10):
        ok = rb.push(CommandType.SubmitLimit, Side.Buy, 100 + i, 5, TimeInForce.GTC, 0)
        assert ok

    for i in range(10):
        item = rb.pop()
        assert item is not None
        cmd_type, side, price, qty, tif, cancel_id = item
        assert cmd_type == CommandType.SubmitLimit
        assert price == 100 + i

    assert rb.pop() is None  # drained


def test_push_fails_when_full():
    rb = _core.RingBuffer()
    n_pushed = 0
    while rb.push(CommandType.Cancel, Side.Buy, 0, 0, TimeInForce.GTC, n_pushed):
        n_pushed += 1
        if n_pushed > rb.capacity + 10:
            break  # safety valve; should never trigger
    # Capacity is a power of two and the buffer reserves one slot to
    # distinguish full from empty, so usable capacity is capacity - 1.
    assert n_pushed == rb.capacity - 1
    assert rb.push(CommandType.Cancel, Side.Buy, 0, 0, TimeInForce.GTC, 999) is False


def test_concurrent_producer_consumer_no_loss_no_duplication():
    """Runs a real producer thread and a real consumer thread against one
    ring buffer and checks that every value produced is received exactly
    once, in order -- the property a wait-free SPSC queue must guarantee.
    """
    rb = _core.RingBuffer()
    n_items = 200_000
    received = []
    produced_count = [0]

    def producer():
        i = 0
        while i < n_items:
            # cancel_id doubles as a monotonically increasing payload here.
            if rb.push(CommandType.Cancel, Side.Buy, 0, 0, TimeInForce.GTC, i):
                i += 1
        produced_count[0] = i

    def consumer():
        while len(received) < n_items:
            item = rb.pop()
            if item is not None:
                received.append(item[5])  # cancel_id payload

    t_producer = threading.Thread(target=producer)
    t_consumer = threading.Thread(target=consumer)
    t_consumer.start()
    t_producer.start()
    t_producer.join(timeout=30)
    t_consumer.join(timeout=30)

    assert produced_count[0] == n_items
    assert len(received) == n_items
    # FIFO order must be exactly preserved (single producer, single consumer).
    assert received == list(range(n_items))
