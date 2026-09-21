#!/bin/bash
# run_lbr_compare.sh -- accuracy on the manuscript's oracle sets and like-for-like market-feed
# timing of the whole-book chart, the routed inverter and Let's Be Rational, one binary each.
#
#   bash reproduce/bench/run_lbr_compare.sh [/path/to/LetsBeRational]
# The reference's sources are not redistributed (http://www.jaeckel.org/); market_feed.csv is
# not redistributed either (licensed OptionMetrics origin, see README.md).
#
# LIKE-FOR-LIKE: the reference is compiled from its author's unmodified sources with the flags
# of its own Makefile (-O3 -DNDEBUG -ffp-contract=fast, instruction set per build), i.e. the
# release build its author ships; our translation unit keeps -ffp-contract=off, which the
# bit-identity contract needs and which costs our kernels nothing (every intended fusion is an
# explicit fma).  Both live in one binary and are timed by the same loop in the same session.
# A second build per ISA adds -flto to BOTH sides, so the reference may be inlined into the
# timing loop exactly as our header-only kernels are; if its row does not move, the call
# boundary is not what the table measures.
# Builds: wbacc (native), wblbr_{512,256,scl} and wblbr_{512,256,scl}_lto.  Runs from bench_run.
# Output: reproduce/bench/out/lbr_compare_<stamp>.txt (and to the terminal).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC="$(cd "$HERE/../../include/volfi" && pwd)"
BR="$(cd "$HERE/../data" && pwd)"
LBR="${1:-$HERE/third_party/LetsBeRational}"
[ -f "$BR/market_feed.csv" ] || { echo "no market_feed.csv in $BR (not redistributed; see README.md)"; exit 2; }
[ -f "$LBR/lets_be_rational.cpp" ] || { echo "no lets_be_rational.cpp in $LBR"; exit 2; }
OUT="$HERE/out"; mkdir -p "$OUT/lbr"
REP="$OUT/lbr_compare_$(date +%Y%m%d_%H%M%S).txt"
OURS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -w -I$INC -I$LBR"
THEIRS="-std=c++17 -O3 -DNDEBUG -ffp-contract=fast -finput-charset=UTF-8 -fextended-identifiers -DNO_XL_API -w -I$LBR"
declare -A ISA=( [512]="-march=native" [256]="-mavx2 -mfma -mno-avx512f" [scl]="-mno-avx2 -mno-avx512f" )
PIN="${BENCH_PIN:-}"

# lbr_objs <isa> <extra>  -> compiles the four reference sources, prints the object list
lbr_objs() {
  local isa=$1 extra=$2 objs=""
  for s in lets_be_rational normaldistribution rationalcubic erf_cody; do
    local o="$OUT/lbr/${s}_${isa}${extra:+_lto}.o"
    g++ $THEIRS ${ISA[$isa]} $extra -c "$LBR/$s.cpp" -o "$o" || return 1
    objs="$objs $o"
  done
  echo "$objs"
}
{
  echo "== whole-book chart vs routed inverter vs Let's Be Rational ==  $(date)"
  echo "   cpu  $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')   load $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "   LBR  $LBR   (compiled with its Makefile's flags: -O3 -DNDEBUG -ffp-contract=fast)"
  echo "   ours $OURS"
  echo
  echo "################ accuracy on the manuscript's oracle sets (native build)"
  O=$(lbr_objs 512 "") && g++ $OURS -march=native "$HERE/../accuracy/wb_accuracy.cpp" $O -o "$OUT/wbacc" && \
    ( cd "$BR" && "$OUT/wbacc" oracle_heat.bin oracle_stressed.bin oracle_edge.bin )
  for isa in 512 256 scl; do
    echo "################ timing $isa"
    O=$(lbr_objs $isa "") && g++ $OURS ${ISA[$isa]} "$HERE/wb_lbr_bench.cpp" $O -o "$OUT/wblbr_$isa" && ( cd "$BR" && $PIN "$OUT/wblbr_$isa" )
    echo "################ timing $isa, -flto on both sides (the reference may be inlined into the loop as well)"
    O=$(lbr_objs $isa "-flto") && g++ $OURS ${ISA[$isa]} -flto -DLTO_BUILD "$HERE/wb_lbr_bench.cpp" $O -o "$OUT/wblbr_${isa}_lto" && ( cd "$BR" && $PIN "$OUT/wblbr_${isa}_lto" )
  done
  echo "   load@end $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "== done $(date) ==  report: $REP"
} 2>&1 | tee "$REP"
