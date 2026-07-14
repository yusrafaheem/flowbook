"""flowbook: a limit order book microstructure lab.

    flowbook._core        C++ matching engine (pybind11 extension)
    flowbook.simulator     synthetic order-flow generator (stylized facts)
    flowbook.features      LOB feature extraction (imbalance, microprice, ...)
    flowbook.strategies    market-making strategies (Avellaneda-Stoikov, ...)
    flowbook.backtest      event loop + PnL/risk metrics
    flowbook.research      transformer-lite short-horizon predictor + training

See README.md for the research this project is built on and how the pieces
fit together.
"""

from flowbook import _core
from flowbook._core import Side, TimeInForce

__all__ = ["_core", "Side", "TimeInForce", "__version__"]
__version__ = "0.1.0"
