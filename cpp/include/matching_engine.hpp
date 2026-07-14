// flowbook/cpp/include/matching_engine.hpp
//
// A single-symbol price-time-priority limit order book with a matching
// engine on top. Design notes:
//
//   - Bids are kept in a std::map<Price, Level, std::greater<>> (best bid
//     first); asks in a std::map<Price, Level> (best ask first). Each
//     Level is a FIFO deque of resting orders -> O(log P) to find a price
//     level, O(1) amortized to match/insert/cancel within it.
//   - Cancellation is O(log P) via an id -> (side, price) index; we do not
//     shrink the deque immediately on cancel, we tombstone and skip, which
//     keeps cancel latency O(log P) instead of O(level depth). Tombstones
//     are reaped lazily when a level is walked.
//   - All arithmetic is on integer ticks/lots. Python-facing code converts
//     from float price/qty using a configurable tick size.
//
// This mirrors, at a much smaller scale, how production matching engines
// (e.g. exchange cores) separate "book maintenance" from "order routing":
// MatchingEngine owns both here for simplicity, but the two responsibilities
// are kept in clearly separated methods so they could be split later.

#pragma once

#include <map>
#include <deque>
#include <unordered_map>
#include <vector>
#include <functional>
#include <optional>

#include "order.hpp"

namespace flowbook {

class MatchingEngine {
public:
    explicit MatchingEngine(Price tick_size = 1) : tick_size_(tick_size) {}

    // Submits a limit order. Matches immediately against the opposite side
    // while price compatible, then rests any remainder (unless IOC).
    // Returns the assigned OrderId and appends any resulting trades to
    // `out_trades`.
    OrderId submit_limit(Side side, Price price, Qty quantity,
                          TimeInForce tif, std::vector<Trade>& out_trades);

    // Submits a market order: matches against the book until filled or the
    // book on that side is exhausted. Never rests.
    OrderId submit_market(Side side, Qty quantity, std::vector<Trade>& out_trades);

    // Cancels a resting order. Returns true if it was found and live.
    bool cancel(OrderId id);

    // Best bid / ask price; nullopt if that side is empty.
    //
    // Not const: a cancelled order is tombstoned rather than immediately
    // erased from its price level's deque (see class-level comment), so a
    // level can be entirely "dead" (every order in it cancelled) while
    // still structurally present in the map. These accessors reap such
    // dead entries from the front of the book on read, which is the
    // other half of the lazy-cancel design -- match_against() reaps
    // during matching, these reap during queries. Both must agree, or
    // best_bid()/best_ask() can report a price with zero live quantity.
    std::optional<Price> best_bid();
    std::optional<Price> best_ask();

    // Mid price and microprice (volume-weighted between best bid/ask).
    // Both return nullopt if either side is empty.
    std::optional<double> mid_price();
    std::optional<double> microprice();

    // Depth snapshot, top `levels` price levels per side.
    BookSnapshot snapshot(std::size_t levels = 10);

    Price tick_size() const { return tick_size_; }
    Timestamp clock() const { return clock_; }

private:
    struct Resting {
        Side side;
        Price price;
    };

    // Ascending-price map for asks (best ask = begin()), descending for bids.
    using AskBook = std::map<Price, std::deque<Order>>;
    using BidBook = std::map<Price, std::deque<Order>, std::greater<Price>>;

    AskBook asks_;
    BidBook bids_;
    std::unordered_map<OrderId, Resting> index_; // live orders only
    std::unordered_map<OrderId, bool> tombstoned_; // cancelled-but-not-reaped

    OrderId next_id_ = 1;
    Timestamp clock_ = 0;
    Price tick_size_;

    Timestamp tick() { return ++clock_; }

    // Matches `incoming` against `book` (the opposite side), mutating
    // `incoming.remaining` and appending trades. `price_ok` decides whether
    // the best resting price is still marketable against the incoming order.
    template <typename BookMap>
    void match_against(Order& incoming, BookMap& book,
                        std::vector<Trade>& out_trades);

    // Pops tombstoned orders from the front of the best level's queue
    // (and erases levels that become fully empty) until the book is
    // empty or the front level has at least one live order. Idempotent
    // and cheap when there is nothing to reap.
    template <typename BookMap>
    void reap_front(BookMap& book);

    void rest(Order order);
    void erase_from_index(OrderId id);
};

}  // namespace flowbook
