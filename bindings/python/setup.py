from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import pybind11
import numpy

root = Path(__file__).resolve().parents[2]

ext_modules = [
    Extension(
        "volfi._volfi",
        ["src/volfi_py.cpp"],
        include_dirs=[str(root / "include"), pybind11.get_include(), numpy.get_include()],
        language="c++",
        extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O3", "-std=c++11"],
    )
]

setup(
    name="volfi",
    version="0.1.8.9000",
    description="Python bindings for volfi",
    packages=["volfi"],
    package_dir={"": "src_py"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["numpy"],
)
