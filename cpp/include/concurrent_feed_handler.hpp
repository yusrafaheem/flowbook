// flowbook/cpp/include/concurrent_feed_handler.hpp
//
// A single-producer/single-consumer (SPSC) lock-free ring buffer that
// decouples "order intake" from "matching". This models how real feed
// handlers/gateways are architected: a network/IO thread enqueues incoming
// order messages as fast as they arrive, while a dedicated matching thread
// dequeues and applies them to the book at its own pace, so a slow matching
// step never backs up the network read loop (and vice versa).
//
// Concurrency approach: a fixed-capacity circular buffer of trivially
// copyable `Command` structs, with `head_`/`tail_` as std::atomic<size_t>
// using acquire/release ordering. No mutex, no condition variable on the
// hot path -> push()/pop() are wait-free for a single producer and single
// consumer. Capacity must be a power of two so index wraparound is a
// bitmask instead of a modulo.
//
// This is intentionally scoped to SPSC (one feed thread, one matching
// thread) rather than a general MPMC queue: that is the actual topology of
// a single-symbol matching engine, and SPSC lets us drop all of the
// compare-and-swap retry loops an MPMC queue would need.

#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

#include "order.hpp"

namespace flowbook {

enum class CommandType : std::uint8_t { SubmitLimit, SubmitMarket, Cancel };

struct Command {
    CommandType type;
    Side side;
    Price price;
    Qty quantity;
    TimeInForce tif;
    OrderId cancel_id; // only used when type == Cancel
};

template <std::size_t Capacity>
class SpscRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");

public:
    // Producer side. Returns false if the buffer is full (caller should
    // back off / apply flow control rather than block).
    bool push(const Command& cmd) {
        std::size_t head = head_.load(std::memory_order_relaxed);
        std::size_t next = (head + 1) & mask_;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // full
        }
        buffer_[head] = cmd;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer side. Returns std::nullopt if the buffer is empty.
    std::optional<Command> pop() {
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // empty
        }
        Command cmd = buffer_[tail];
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return cmd;
    }

    std::size_t capacity() const { return Capacity; }

    // Approximate occupancy; safe for monitoring/metrics, not for
    // correctness decisions (head/tail can move between the two loads).
    std::size_t size_approx() const {
        std::size_t h = head_.load(std::memory_order_acquire);
        std::size_t t = tail_.load(std::memory_order_acquire);
        return (h - t) & mask_;
    }

private:
    static constexpr std::size_t mask_ = Capacity - 1;
    std::vector<Command> buffer_{Capacity};
    alignas(64) std::atomic<std::size_t> head_{0};  // written by producer
    alignas(64) std::atomic<std::size_t> tail_{0};  // written by consumer
};

}  // namespace flowbook
