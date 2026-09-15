#!/bin/bash
# run_cpu_bench.sh -- the CPU throughput numbers of the book kernel (v0.3.0), pinned runs.
#
#   BENCH_BUILD_ONLY=1 bash reproduce/book/run_cpu_bench.sh      build every benchmark binary into gate/out (once)
#   BENCH_NOBUILD=1    bash reproduce/book/run_cpu_bench.sh      time the prebuilt binaries, write one report
#                      bash reproduce/book/run_cpu_bench.sh      build + time in one go (loaded-machine quick look)
#
# Environment: BENCH_PIN="taskset -c 2 nice -n -5" pins and prioritises every timed
# binary; BENCH_TAG=run3 goes into the report name; BENCH_RESULTS overrides the
# report folder (default reproduce/book/results): bench_v030_<stamp>_<tag>.txt.
# Needs market_feed.csv in reproduce/book (not redistributed; see README.md).
#
# Meant for a QUIET machine: the report records the load average before and after
# and refuses to call itself clean if the 1-minute load was above 0.5 at the start.
# Every gate binary re-checks bit-identity before it times, so a report is never
# produced from a build that fails its own gate.
#
# What it times (best of 5 in every section):
#   wb_vec_gate   [4]  whole-book batch vs shipped batch on the FULL feed, at 64/256/1024/4096/30000, per ISA
#   wb_gate       [5]  scalar latency over the full feed: whole-book chart vs shipped
#   rec_vec_gate  [4]  recurrence batch vs book batch vs shipped batch, feed NEAR subset, per ISA
#   rec_gate      [3]  scalar latency: recurrence vs 12-cell vs book
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC="$(cd "$HERE/../../include/volfi" && pwd)"
BR="$HERE"
OUT="$HERE/out"; mkdir -p "$OUT"
BUILD_ONLY="${BENCH_BUILD_ONLY:-0}"
NOBUILD="${BENCH_NOBUILD:-0}"
PIN="${BENCH_PIN:-}"
FLAGS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -w -I$INC"
declare -A ISA=( [512]="-march=native" [256]="-mavx2 -mfma -mno-avx512f" [scl]="-mno-avx2 -mno-avx512f" )

# cc <binary> <source> <flags...>: compile unless BENCH_NOBUILD=1 and the binary exists
cc() {
  local bin=$1 src=$2; shift 2
  if [ "$NOBUILD" = "1" ]; then
    [ -x "$OUT/$bin" ] && return 0
    echo "MISSING $OUT/$bin (run BENCH_BUILD_ONLY=1 first)"; return 1
  fi
  if g++ $FLAGS "$@" "$HERE/$src" -o "$OUT/$bin" 2> "$OUT/$bin.build.log"; then
    [ "$BUILD_ONLY" = "1" ] && printf "  %-18s OK\n" "$bin"; return 0
  fi
  printf "  %-18s FAIL  (see out/$bin.build.log)\n" "$bin"; return 1
}

# ------------------------------------------------------------------ build phase
BUILD_RC=0
[ "$BUILD_ONLY" = "1" ] && echo "=== building the v0.3.0-test benchmark binaries into gate/out ==="
for isa in 512 256 scl; do
  cc bench_wbvg_$isa wb_vec_gate.cpp   ${ISA[$isa]}              || BUILD_RC=1
  cc bench_rvg_$isa  rec_vec_gate.cpp  ${ISA[$isa]} -DNB_LAYOUT=1 || BUILD_RC=1
done
cc bench_wbg       wb_gate.cpp   -march=native || BUILD_RC=1
cc bench_rg        rec_gate.cpp  -march=native || BUILD_RC=1
if [ "$BUILD_ONLY" = "1" ]; then
  [ "$BUILD_RC" = "0" ] && echo "built: 8 binaries in $OUT" || echo "BUILD FAILED"
  exit $BUILD_RC
fi
[ "$BUILD_RC" = "0" ] || { echo "cannot time: a binary is missing or failed to build"; exit 1; }

# ------------------------------------------------------------------ timing phase
STAMP=$(date +%Y%m%d_%H%M%S)
RESDIR="${BENCH_RESULTS:-$HERE/results}"; mkdir -p "$RESDIR"
REP="$RESDIR/bench_v030_$STAMP${BENCH_TAG:+_$BENCH_TAG}.txt"
LOAD0=$(cut -d' ' -f1 /proc/loadavg)
{
  echo "== volfi v0.3.0 CPU benchmark ==  $(date)"
  echo "   cpu     $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
  echo "   cores   $(nproc)   g++ $(g++ -dumpfullversion)"
  echo "   flags   $FLAGS"
  echo "   pin     ${PIN:-none}"
  echo "   load@0  $(cut -d' ' -f1-3 /proc/loadavg)"
  if awk -v l="$LOAD0" 'BEGIN{exit !(l+0 > 0.5)}'; then
    echo "   NOTE    1-minute load above 0.5 at start: this run is NOT a clean-machine measurement"
  else
    echo "   clean   1-minute load $LOAD0 at start"
  fi
  echo
} | tee "$REP"

[ -f "$BR/market_feed.csv" ] || { echo "no market_feed.csv in $BR (not redistributed; see README.md)"; exit 2; }
section() { echo | tee -a "$REP"; echo "################ $1" | tee -a "$REP"; }
timed()   { ( cd "$BR" && $PIN "$OUT/$1" ); }

for isa in 512 256 scl; do
  section "wb_vec_gate  $isa  (whole-book chart vs shipped batch, FULL feed)"
  timed bench_wbvg_$isa | tee -a "$REP"
done
section "wb_gate  (coverage [2], scalar latency [5]: whole-book vs shipped, full feed)"
timed bench_wbg | grep -E "^\[2\]|^\[5\]|region A|fallback" | tee -a "$REP"
for isa in 512 256 scl; do
  section "rec_vec_gate  $isa"
  timed bench_rvg_$isa | tee -a "$REP"
done
section "rec_gate  (scalar latency of the three charts)"
timed bench_rg | sed -n '/^\[3\]/,$p' | tee -a "$REP"

{
  echo
  echo "   load@end $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "== done $(date) ==  report: $REP"
} | tee -a "$REP"
