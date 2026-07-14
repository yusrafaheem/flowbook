#include "matching_engine.hpp"

#include <algorithm>

namespace flowbook {

template <typename BookMap>
void MatchingEngine::reap_front(BookMap& book) {
    while (!book.empty()) {
        auto level_it = book.begin();
        std::deque<Order>& fifo = level_it->second;
        while (!fifo.empty()) {
            auto tomb_it = tombstoned_.find(fifo.front().id);
            if (tomb_it != tombstoned_.end() && tomb_it->second) {
                tombstoned_.erase(tomb_it);
                fifo.pop_front();
                continue;
            }
            break;  // front order is live
        }
        if (fifo.empty()) {
            book.erase(level_it);
            continue;  // level fully dead, move to the next one
        }
        break;  // front level now has a live order at its front
    }
}

template <typename BookMap>
void MatchingEngine::match_against(Order& incoming, BookMap& book,
                                    std::vector<Trade>& out_trades) {
    while (incoming.remaining > 0 && !book.empty()) {
        auto level_it = book.begin();
        Price level_price = level_it->first;

        // Price-compatibility check: a limit order only crosses while the
        // best resting price is at least as good as its own limit; market
        // orders always cross.
        if (incoming.type == OrderType::Limit) {
            bool crosses = (incoming.side == Side::Buy)
                               ? (level_price <= incoming.price)
                               : (level_price >= incoming.price);
            if (!crosses) break;
        }

        std::deque<Order>& fifo = level_it->second;
        while (incoming.remaining > 0 && !fifo.empty()) {
            Order& resting = fifo.front();

            // Skip tombstoned (cancelled) orders lazily.
            auto tomb_it = tombstoned_.find(resting.id);
            if (tomb_it != tombstoned_.end() && tomb_it->second) {
                tombstoned_.erase(tomb_it);
                fifo.pop_front();
                continue;
            }

            Qty fill_qty = std::min(incoming.remaining, resting.remaining);
            incoming.remaining -= fill_qty;
            resting.remaining -= fill_qty;

            Trade t;
            t.aggressor_id = incoming.id;
            t.resting_id = resting.id;
            t.aggressor_side = incoming.side;
            t.price = level_price;
            t.quantity = fill_qty;
            t.timestamp = tick();
            out_trades.push_back(t);

            if (resting.is_filled()) {
                erase_from_index(resting.id);
                fifo.pop_front();
            }
        }

        if (fifo.empty()) book.erase(level_it);
    }
}

OrderId MatchingEngine::submit_limit(Side side, Price price, Qty quantity,
                                      TimeInForce tif,
                                      std::vector<Trade>& out_trades) {
    Order incoming{next_id_++, side, OrderType::Limit, tif,
                    price,      quantity, quantity, tick()};

    if (side == Side::Buy) {
        match_against(incoming, asks_, out_trades);
    } else {
        match_against(incoming, bids_, out_trades);
    }

    if (incoming.remaining > 0 && tif != TimeInForce::IOC) {
        rest(incoming);
    }
    return incoming.id;
}

OrderId MatchingEngine::submit_market(Side side, Qty quantity,
                                       std::vector<Trade>& out_trades) {
    Order incoming{next_id_++, side, OrderType::Market, TimeInForce::IOC,
                    0,          quantity, quantity, tick()};

    if (side == Side::Buy) {
        match_against(incoming, asks_, out_trades);
    } else {
        match_against(incoming, bids_, out_trades);
    }
    // Market orders never rest; any unfilled remainder is simply dropped
    // (as on a real exchange with no resting market-order book).
    return incoming.id;
}

bool MatchingEngine::cancel(OrderId id) {
    auto it = index_.find(id);
    if (it == index_.end()) return false;
    tombstoned_[id] = true;
    index_.erase(it);
    return true;
}

void MatchingEngine::rest(Order order) {
    OrderId id = order.id;
    Side side = order.side;
    Price price = order.price;
    if (side == Side::Buy) {
        bids_[price].push_back(std::move(order));
    } else {
        asks_[price].push_back(std::move(order));
    }
    index_[id] = Resting{side, price};
}

void MatchingEngine::erase_from_index(OrderId id) { index_.erase(id); }

std::optional<Price> MatchingEngine::best_bid() {
    reap_front(bids_);
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<Price> MatchingEngine::best_ask() {
    reap_front(asks_);
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<double> MatchingEngine::mid_price() {
    auto b = best_bid();
    auto a = best_ask();
    if (!b || !a) return std::nullopt;
    return (static_cast<double>(*b) + static_cast<double>(*a)) / 2.0;
}

std::optional<double> MatchingEngine::microprice() {
    reap_front(bids_);
    reap_front(asks_);
    if (bids_.empty() || asks_.empty()) return std::nullopt;

    // Volume-weighted "microprice": weights the best ask by resting bid
    // volume and vice versa, which leans the estimate toward the side with
    // less standing liquidity (i.e. the side more likely to be walked
    // through next). See Stoikov (2018), "The micro-price: a high
    // frequency estimator of future prices".
    Qty bid_vol = 0;
    for (const auto& o : bids_.begin()->second) bid_vol += o.remaining;
    Qty ask_vol = 0;
    for (const auto& o : asks_.begin()->second) ask_vol += o.remaining;

    double bid_p = static_cast<double>(bids_.begin()->first);
    double ask_p = static_cast<double>(asks_.begin()->first);
    double total = static_cast<double>(bid_vol + ask_vol);
    if (total <= 0) return (bid_p + ask_p) / 2.0;

    return (bid_p * static_cast<double>(ask_vol) +
            ask_p * static_cast<double>(bid_vol)) /
           total;
}

BookSnapshot MatchingEngine::snapshot(std::size_t levels) {
    reap_front(bids_);
    reap_front(asks_);
    BookSnapshot snap;
    std::size_t count = 0;
    for (const auto& [price, fifo] : bids_) {
        if (count++ >= levels) break;
        Qty total = 0;
        std::size_t n = 0;
        for (const auto& o : fifo) {
            auto tomb = tombstoned_.find(o.id);
            if (tomb != tombstoned_.end() && tomb->second) continue;
            total += o.remaining;
            ++n;
        }
        if (n > 0) snap.bids.push_back(BookLevel{price, total, n});
    }
    count = 0;
    for (const auto& [price, fifo] : asks_) {
        if (count++ >= levels) break;
        Qty total = 0;
        std::size_t n = 0;
        for (const auto& o : fifo) {
            auto tomb = tombstoned_.find(o.id);
            if (tomb != tombstoned_.end() && tomb->second) continue;
            total += o.remaining;
            ++n;
        }
        if (n > 0) snap.asks.push_back(BookLevel{price, total, n});
    }
    return snap;
}

}  // namespace flowbook
