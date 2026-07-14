import os
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

root = Path(__file__).resolve().parents[2]

# The v0.2.0 engine is header-only C++17. -ffp-contract=off is what makes the scalar and SIMD
# paths return bit-identical results (see the paper); -march selects the SIMD width. The
# default tunes to the build machine, which is right for a from-source install; set
# VOLFI_MARCH="" for a portable scalar build, or e.g. VOLFI_MARCH=x86-64-v3 for an AVX2 wheel.
if sys.platform == "win32":
    extra = ["/O2", "/std:c++17"]
else:
    extra = ["-O3", "-std=c++17", "-ffp-contract=off", "-fno-fast-math"]
    march = os.environ.get("VOLFI_MARCH", "native")
    if march:
        extra.append(f"-march={march}")

ext_modules = [
    Extension(
        "volfi._volfi",
        ["src/volfi_py.cpp"],
        include_dirs=[str(root / "include"), pybind11.get_include()],
        language="c++",
        extra_compile_args=extra,
    )
]

setup(
    name="volfi",
    version="0.2.0",
    description="Python bindings for volfi v0.2.0 (routed machine-precision Black-Scholes IV inverter)",
    packages=["volfi"],
    package_dir={"": "src_py"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["numpy"],
)
