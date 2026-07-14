// flowbook/cpp/include/order.hpp
//
// Core value types shared by the matching engine: sides, orders, trades,
// and book-level snapshots. Kept POD-ish and allocation-light so the hot
// path (order_book.cpp) can stay cache-friendly.

#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace flowbook {

using OrderId = std::uint64_t;
using Price = std::int64_t;   // price expressed in integer ticks (no floats on the hot path)
using Qty = std::int64_t;     // quantity in integer lots
using Timestamp = std::uint64_t;  // monotonic sequence number assigned by the engine

enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

inline Side opposite(Side s) { return s == Side::Buy ? Side::Sell : Side::Buy; }

enum class OrderType : std::uint8_t { Limit = 0, Market = 1 };

enum class TimeInForce : std::uint8_t { GTC = 0, IOC = 1 };

// A resting or incoming order. `remaining` is mutated in place as fills occur.
struct Order {
    OrderId id;
    Side side;
    OrderType type;
    TimeInForce tif;
    Price price;      // ignored for Market orders
    Qty quantity;      // original quantity
    Qty remaining;     // quantity left to fill
    Timestamp arrival; // sequence number -> enforces price-time priority

    bool is_filled() const { return remaining <= 0; }
};

// Emitted every time an incoming order crosses the book.
struct Trade {
    OrderId aggressor_id;
    OrderId resting_id;
    Side aggressor_side;
    Price price;
    Qty quantity;
    Timestamp timestamp;
};

// Aggregated view of a single price level, used for book snapshots.
struct BookLevel {
    Price price;
    Qty total_quantity;
    std::size_t order_count;
};

// Top-of-book / depth snapshot returned to callers (Python included).
struct BookSnapshot {
    std::vector<BookLevel> bids; // best first (highest price)
    std::vector<BookLevel> asks; // best first (lowest price)
};

}  // namespace flowbook
