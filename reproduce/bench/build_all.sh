#!/bin/bash
# build_all.sh -- builds the routed-inverter benchmarks that need Let's Be Rational.
#
#   bash bench/build_all.sh /path/to/LetsBeRational
#
# Binaries go to reproduce/out and must be run from reproduce/data (oracle files load by relative path):
#   cd data && for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ../out/bench_512 > run512_$i.txt; done
# Only the four sources the solver needs are compiled. Its folder also ships Excel, Octave and Python
# bindings, so a *.cpp glob fails. The gates without external inputs are in ../Makefile (make check).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
LBR="${1:-}"; [ -f "$LBR/lets_be_rational.cpp" ] || { echo "usage: bash bench/build_all.sh /path/to/LetsBeRational"; exit 2; }
OUT="$ROOT/out"; mkdir -p "$OUT"
FLAGS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -DNO_XL_API -w -I$ROOT/../include/volfi -I$LBR"
L="$LBR/lets_be_rational.cpp $LBR/normaldistribution.cpp $LBR/rationalcubic.cpp $LBR/erf_cody.cpp"
build() { printf "  %-14s " "$3"; if g++ $FLAGS $2 "$1" ${4:+$L} -o "$OUT/$3" 2> "$OUT/$3.build.log"; then echo OK; else echo FAIL; grep -m4 error "$OUT/$3.build.log"; return 1; fi; }
build "$HERE/benchmark_vec.cpp" "-march=native"               bench_512   lbr
build "$HERE/benchmark_vec.cpp" "-mavx2 -mfma -mno-avx512f"   bench_256   lbr
build "$HERE/bench_sweep.cpp"   "-march=native"               bench_sweep lbr
build "$HERE/bench_phases.cpp"  "-march=native"               bench_phases
build "$ROOT/accuracy/accuracy_vs_lbr.cpp" "-march=native"    accuracy_vs_lbr lbr
build "$ROOT/accuracy/wb_accuracy.cpp"     "-march=native"    wb_accuracy     lbr
echo "built into $OUT. Run from $ROOT/data."
