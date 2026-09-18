#!/bin/bash
# run_cpu_tiles.sh -- the region-tile rows of the CPU like-for-like table only (cpu_all_bench.cpp,
# section [2]): Let's Be Rational, the PDE method, the routed inverter and the whole-book chart on
# the feed's Near / Far / Wing tiles and on the synthetic Upper tile, AVX-512 and AVX2 builds,
# release flags on every side, no LTO.  A subset of run_cpu_all.sh for the tile rows alone.
#
#   BENCH_PIN="taskset -c 2 nice -n -5" bash reproduce/book/run_cpu_tiles.sh [LBR dir] [PDE dir]
#
# Same inputs and licences as run_cpu_all.sh (market_feed.csv, the reference's sources, the PDE
# method's sources and loadPartition.txt, none redistributed).  Report: reproduce/book/out/cpu_tiles_<stamp>.txt.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC="$(cd "$HERE/../../include/volfi" && pwd)"
BR="$HERE"
LBR="${1:-$HERE/third_party/LetsBeRational}"
PDE="${2:-$HERE/third_party/PDE-method-for-implied-volatility}"
[ -f "$BR/market_feed.csv" ] || { echo "no market_feed.csv in $BR (not redistributed; see README.md)"; exit 2; }
[ -f "$LBR/lets_be_rational.cpp" ] || { echo "no lets_be_rational.cpp in $LBR"; exit 2; }
[ -f "$PDE/loadPartition.txt" ] || { echo "no loadPartition.txt in $PDE"; exit 2; }
OUT="$HERE/out"; mkdir -p "$OUT/lbr" "$OUT/pde"
REP="$OUT/cpu_tiles_$(date +%Y%m%d_%H%M%S).txt"
OURS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -fopenmp -w -I$INC -I$LBR -I$PDE"
THEIRS_LBR="-std=c++17 -O3 -DNDEBUG -ffp-contract=fast -finput-charset=UTF-8 -fextended-identifiers -DNO_XL_API -w -I$LBR"
THEIRS_PDE="-std=c++11 -O2 -fopenmp -w -I$PDE"
declare -A ISA=( [512]="-march=native" [256]="-mavx2 -mfma -mno-avx512f" )
PIN="${BENCH_PIN:-}"

objs() {
  local isa=$1 o=""
  for s in lets_be_rational normaldistribution rationalcubic erf_cody; do
    g++ $THEIRS_LBR ${ISA[$isa]} -c "$LBR/$s.cpp" -o "$OUT/lbr/${s}_${isa}.o" || return 1; o="$o $OUT/lbr/${s}_${isa}.o"; done
  for s in cImpVolBasic indexStructure; do
    g++ $THEIRS_PDE ${ISA[$isa]} -c "$PDE/$s.cpp" -o "$OUT/pde/${s}_${isa}.o" || return 1; o="$o $OUT/pde/${s}_${isa}.o"; done
  echo "$o"
}
cp -n "$PDE/loadPartition.txt" "$BR/loadPartition.txt" 2>/dev/null; COPIED_PART=$?
{
  echo "== CPU region tiles: LBR / PDE / routed / whole-book, Near / Far / Wing / Upper* ==  $(date)"
  echo "   cpu  $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')   load $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "   LBR  $LBR  (its Makefile's flags)   PDE $PDE  (its compile script's flags)   pin: ${PIN:-none}"
  for isa in 512 256; do
    echo "################ tiles $isa"
    O=$(objs $isa) && g++ $OURS ${ISA[$isa]} "$HERE/cpu_all_bench.cpp" $O -o "$OUT/cputiles_$isa" && ( cd "$BR" && BENCH_TILES_ONLY=1 $PIN "$OUT/cputiles_$isa" )
  done
  echo "   load@end $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "== done $(date) ==  report: $REP"
} 2>&1 | tee "$REP"
[ "$COPIED_PART" = "0" ] && rm -f "$BR/loadPartition.txt"
