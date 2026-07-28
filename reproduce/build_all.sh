#!/bin/bash
# Builds every harness for the clean CPU campaign.  Run from bench_run/.
#
#   ./build_all.sh "/path/to/LetsBeRational"
#
# The reference solver's folder ships Excel / Octave / Python bindings alongside the four
# sources we need; a plain *.cpp glob pulls those in and fails on oct.h / Python.h, so the
# four are listed explicitly.  Quoting is deliberate throughout: the paths contain spaces.
set -u
LBR="${1:-}"
if [ -z "$LBR" ] || [ ! -f "$LBR/lets_be_rational.cpp" ]; then
  echo "usage: ./build_all.sh /path/to/LetsBeRational   (must contain lets_be_rational.cpp)"
  exit 1
fi

# Headers live in ../include/volfi in the published repository and one level up in
# the development tree; add whichever exists so this script runs in either layout.
if   [ -d ../include/volfi ]; then INC="-I../include/volfi"
elif [ -f ../volfi_annulus.hpp ]; then INC="-I.."
else echo "cannot find the volfi headers (../include/volfi or ..)"; exit 1; fi

FLAGS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -DNO_XL_API -w $INC"
OUT="${OUT:-.}"

build() {                      # build <name> <isa-flags> <output> [extra sources]
  local src="$1"; shift
  local isa="$1"; shift
  local out="$1"; shift
  printf "  %-26s " "$out"
  if g++ $FLAGS $isa -I"$LBR" "$src" "$@" -o "$OUT/$out" 2>/tmp/va_build.log; then
    echo OK
  else
    echo FAIL; grep -E "error" /tmp/va_build.log | head -4; return 1
  fi
}

L1="$LBR/lets_be_rational.cpp"
L2="$LBR/normaldistribution.cpp"
L3="$LBR/rationalcubic.cpp"
L4="$LBR/erf_cody.cpp"

AVX512="-march=native"
AVX2="-mavx2 -mfma -mno-avx512f"
SCALAR="-mno-avx2 -mno-avx512f"

echo "=== correctness gates (host-independent) ==="
build smoke_test.cpp "$AVX512" smoke_512
build smoke_test.cpp "$AVX2"   smoke_256
build smoke_test.cpp "$SCALAR" smoke_scl
build verify_vec.cpp "$AVX512" vv_512
build verify_vec.cpp "$AVX2"   vv_256
build verify_vec.cpp "$SCALAR" vv_scl

echo "=== throughput (needs the reference solver) ==="
build benchmark_vec.cpp "$AVX512" bench_512 "$L1" "$L2" "$L3" "$L4"
build benchmark_vec.cpp "$AVX2"   bench_256 "$L1" "$L2" "$L3" "$L4"
build bench_sweep.cpp   "$AVX512" bench_sweep "$L1" "$L2" "$L3" "$L4"

echo "=== driver phase breakdown (no reference solver) ==="
build bench_phases.cpp "$AVX512" bp

echo
echo "built. Now:"
echo "  ./smoke_512 | tail -1        (repeat for _256 / _scl: all must print SMOKE PASS)"
echo "  ./vv_512                     (repeat for _256 / _scl: bit-identity 0/0/0, 0 pts >1e-15)"
echo "  for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ./bench_512 > run512_\$i.txt; done"
echo "  for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ./bench_256 > run256_\$i.txt; done"
echo "  taskset -c 2 nice -n -5 ./bench_sweep > batchsweep.txt   (Figure 1 data; ~30 s)"
