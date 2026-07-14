// flowbook/cpp/bindings.cpp
//
// pybind11 bindings exposing MatchingEngine and the SPSC ring buffer to
// Python as `flowbook._core`. Kept thin on purpose: all research logic
// (simulation, strategies, ML) lives in Python where iteration is fast;
// only the latency-sensitive book/matching logic lives in C++.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <tuple>

#include "matching_engine.hpp"
#include "concurrent_feed_handler.hpp"

namespace py = pybind11;
using namespace flowbook;

namespace {
using Ring = SpscRingBuffer<4096>;

using CommandTuple = std::tuple<CommandType, Side, Price, Qty, TimeInForce, OrderId>;

CommandTuple to_tuple(const Command& c) {
    return {c.type, c.side, c.price, c.quantity, c.tif, c.cancel_id};
}
}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "flowbook C++ matching engine core (pybind11 bindings)";

    py::enum_<Side>(m, "Side")
        .value("Buy", Side::Buy)
        .value("Sell", Side::Sell);

    py::enum_<TimeInForce>(m, "TimeInForce")
        .value("GTC", TimeInForce::GTC)
        .value("IOC", TimeInForce::IOC);

    py::enum_<CommandType>(m, "CommandType")
        .value("SubmitLimit", CommandType::SubmitLimit)
        .value("SubmitMarket", CommandType::SubmitMarket)
        .value("Cancel", CommandType::Cancel);

    py::class_<Trade>(m, "Trade")
        .def_readonly("aggressor_id", &Trade::aggressor_id)
        .def_readonly("resting_id", &Trade::resting_id)
        .def_readonly("aggressor_side", &Trade::aggressor_side)
        .def_readonly("price", &Trade::price)
        .def_readonly("quantity", &Trade::quantity)
        .def_readonly("timestamp", &Trade::timestamp)
        .def("__repr__", [](const Trade& t) {
            return "<Trade price=" + std::to_string(t.price) +
                   " qty=" + std::to_string(t.quantity) + ">";
        });

    py::class_<BookLevel>(m, "BookLevel")
        .def_readonly("price", &BookLevel::price)
        .def_readonly("total_quantity", &BookLevel::total_quantity)
        .def_readonly("order_count", &BookLevel::order_count);

    py::class_<BookSnapshot>(m, "BookSnapshot")
        .def_readonly("bids", &BookSnapshot::bids)
        .def_readonly("asks", &BookSnapshot::asks);

    py::class_<MatchingEngine>(m, "MatchingEngine")
        .def(py::init<Price>(), py::arg("tick_size") = 1)
        .def(
            "submit_limit",
            [](MatchingEngine& self, Side side, Price price, Qty qty,
               TimeInForce tif) {
                std::vector<Trade> trades;
                OrderId id = self.submit_limit(side, price, qty, tif, trades);
                return std::make_pair(id, trades);
            },
            py::arg("side"), py::arg("price"), py::arg("quantity"),
            py::arg("tif") = TimeInForce::GTC,
            "Submit a limit order. Returns (order_id, trades).")
        .def(
            "submit_market",
            [](MatchingEngine& self, Side side, Qty qty) {
                std::vector<Trade> trades;
                OrderId id = self.submit_market(side, qty, trades);
                return std::make_pair(id, trades);
            },
            py::arg("side"), py::arg("quantity"),
            "Submit a market order. Returns (order_id, trades).")
        .def("cancel", &MatchingEngine::cancel, py::arg("order_id"))
        .def("best_bid", &MatchingEngine::best_bid)
        .def("best_ask", &MatchingEngine::best_ask)
        .def("mid_price", &MatchingEngine::mid_price)
        .def("microprice", &MatchingEngine::microprice)
        .def("snapshot", &MatchingEngine::snapshot, py::arg("levels") = 10)
        .def_property_readonly("tick_size", &MatchingEngine::tick_size)
        .def_property_readonly("clock", &MatchingEngine::clock);

    // Wait-free SPSC ring buffer -- see cpp/include/concurrent_feed_handler.hpp
    // for the design rationale. Bound mainly so the concurrency behavior is
    // independently testable/benchmarkable from Python with real OS threads
    // (see tests/test_ring_buffer.py, benchmarks/bench_ring_buffer.py):
    // push()/pop() release the GIL so two Python threads genuinely run the
    // producer and consumer sides in parallel, not just interleaved.
    py::class_<Ring>(m, "RingBuffer")
        .def(py::init<>())
        .def(
            "push",
            [](Ring& self, CommandType type, Side side, Price price, Qty qty,
               TimeInForce tif, OrderId cancel_id) {
                Command cmd{type, side, price, qty, tif, cancel_id};
                return self.push(cmd);
            },
            py::arg("type"), py::arg("side") = Side::Buy, py::arg("price") = 0,
            py::arg("quantity") = 0, py::arg("tif") = TimeInForce::GTC,
            py::arg("cancel_id") = 0, py::call_guard<py::gil_scoped_release>(),
            "Enqueue a command. Returns False if the buffer is full.")
        .def(
            "pop",
            [](Ring& self) -> std::optional<CommandTuple> {
                auto cmd = self.pop();
                if (!cmd) return std::nullopt;
                return to_tuple(*cmd);
            },
            py::call_guard<py::gil_scoped_release>(),
            "Dequeue a command as (type, side, price, quantity, tif, cancel_id), or None if empty.")
        .def_property_readonly("capacity", &Ring::capacity)
        .def("size_approx", &Ring::size_approx);
}
