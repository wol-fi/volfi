#!/bin/bash
# run_cpu_all.sh -- the CPU like-for-like table: Let's Be Rational, the PDE method, the routed
# inverter and the whole-book chart in ONE binary per ISA, timed by one loop on the feed and on
# region-filtered tiles (cpu_all_bench.cpp), plus the accuracy of all of them on the manuscript's
# oracle sets (wb_accuracy.cpp for LBR/routed/whole-book, pde_regions.cpp for the PDE method).
#
#   bash reproduce/bench/run_cpu_all.sh [/path/to/LetsBeRational] [/path/to/PDE-method-for-implied-volatility]
#
# Third-party sources are NOT redistributed here (their licences are not ours to pass on):
#   Let's Be Rational: the author's sources from http://www.jaeckel.org/ (LetsBeRational.7z)
#   PDE method:        git clone https://github.com/maticivan/PDE-method-for-implied-volatility
#                      (MIT) and download its loadPartition.txt (46 MB) as its README says
# The market feed (market_feed.csv, 30,000 "h c" lines) derives from a licensed OptionMetrics
# file and is not redistributed either; see reproduce/README.md for the recipe.
#
# Flags: the reference at its Makefile's release flags (-O3 -DNDEBUG -ffp-contract=fast); the PDE
# method at its compile script's (-O2 -fopenmp, std=c++11); ours at -O3 -ffp-contract=off.  Each
# ISA also once with -flto on every side.  Report: reproduce/bench/out/cpu_all_<stamp>.txt.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC="$(cd "$HERE/../../include/volfi" && pwd)"
BR="$(cd "$HERE/../data" && pwd)"                                   # run directory: truth files, market_feed.csv, oracle_*.bin
LBR="${1:-$HERE/third_party/LetsBeRational}"
PDE="${2:-$HERE/third_party/PDE-method-for-implied-volatility}"
[ -f "$BR/market_feed.csv" ] || { echo "no market_feed.csv in $BR (not redistributed; see README.md)"; exit 2; }
[ -f "$LBR/lets_be_rational.cpp" ] || { echo "no lets_be_rational.cpp in $LBR"; exit 2; }
[ -f "$PDE/loadPartition.txt" ] || { echo "no loadPartition.txt in $PDE"; exit 2; }
OUT="$HERE/out"; mkdir -p "$OUT/lbr" "$OUT/pde"
REP="$OUT/cpu_all_$(date +%Y%m%d_%H%M%S).txt"
OURS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -fopenmp -w -I$INC -I$LBR -I$PDE"
THEIRS_LBR="-std=c++17 -O3 -DNDEBUG -ffp-contract=fast -finput-charset=UTF-8 -fextended-identifiers -DNO_XL_API -w -I$LBR"
THEIRS_PDE="-std=c++11 -O2 -fopenmp -w -I$PDE"
declare -A ISA=( [512]="-march=native" [256]="-mavx2 -mfma -mno-avx512f" [scl]="-mno-avx2 -mno-avx512f" )
PIN="${BENCH_PIN:-}"

objs() {   # objs <isa> <extra> -> the reference's and the PDE method's objects, at their own flags
  local isa=$1 extra=$2 o=""
  for s in lets_be_rational normaldistribution rationalcubic erf_cody; do
    g++ $THEIRS_LBR ${ISA[$isa]} $extra -c "$LBR/$s.cpp" -o "$OUT/lbr/${s}_${isa}${extra:+_lto}.o" || return 1; o="$o $OUT/lbr/${s}_${isa}${extra:+_lto}.o"; done
  for s in cImpVolBasic indexStructure; do
    g++ $THEIRS_PDE ${ISA[$isa]} $extra -c "$PDE/$s.cpp" -o "$OUT/pde/${s}_${isa}${extra:+_lto}.o" || return 1; o="$o $OUT/pde/${s}_${isa}${extra:+_lto}.o"; done
  echo "$o"
}
cp -n "$PDE/loadPartition.txt" "$BR/loadPartition.txt" 2>/dev/null; COPIED_PART=$?
{
  echo "== CPU like-for-like: LBR / PDE / routed / whole-book ==  $(date)"
  echo "   cpu  $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')   load $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "   LBR  $LBR  (its Makefile's flags)   PDE $PDE  (its compile script's flags)"
  echo
  echo "################ accuracy on the manuscript's oracle sets"
  O=$(objs 512 "") && g++ $OURS -march=native "$HERE/../accuracy/wb_accuracy.cpp" $O -o "$OUT/wbacc" && ( cd "$BR" && "$OUT/wbacc" oracle_heat.bin oracle_stressed.bin oracle_edge.bin )
  g++ $OURS -march=native "$HERE/pde_regions.cpp" $O -o "$OUT/pde_regions" && ( cd "$BR" && "$OUT/pde_regions" )
  for isa in 512 256 scl; do
    echo "################ timing $isa"
    O=$(objs $isa "") && g++ $OURS ${ISA[$isa]} "$HERE/cpu_all_bench.cpp" $O -o "$OUT/cpuall_$isa" && ( cd "$BR" && $PIN "$OUT/cpuall_$isa" )
    echo "################ timing $isa, -flto on every side"
    O=$(objs $isa "-flto") && g++ $OURS ${ISA[$isa]} -flto -DLTO_BUILD "$HERE/cpu_all_bench.cpp" $O -o "$OUT/cpuall_${isa}_lto" && ( cd "$BR" && $PIN "$OUT/cpuall_${isa}_lto" )
  done
  echo "   load@end $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "== done $(date) ==  report: $REP"
} 2>&1 | tee "$REP"
[ "$COPIED_PART" = "0" ] && rm -f "$BR/loadPartition.txt"
