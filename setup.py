"""Build script for flowbook's C++ matching engine extension.

Uses pybind11's setuptools helpers so `pip install .` (or `pip install -e .`)
compiles cpp/*.cpp into the `flowbook._core` extension module. C++17 is
required (structured bindings, std::optional).
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

ext_modules = [
    Pybind11Extension(
        "flowbook._core",
        sources=[
            "cpp/src/matching_engine.cpp",
            "cpp/bindings.cpp",
        ],
        include_dirs=["cpp/include"],
        cxx_std=17,
    ),
]

setup(
    name="flowbook",
    version="0.1.0",
    description="Limit order book microstructure lab: C++ matching engine, "
    "market-making strategies, and a transformer-based short-horizon "
    "price predictor.",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
