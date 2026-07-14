# Quiet-machine benchmark protocol — the paper's timing table

Produces the definitive ns-per-quote numbers (LBR scalar; routed inverter scalar, batch
AVX-512, batch AVX2) on the workloads of the paper's timing table. Accuracy and bit-identity
are host-independent and already final (see `verify_vec` in `README.md`) — this run is timing
only.

The library headers are in `../include/volfi`; the golden verification vectors
(`oracle_*.bin`) are in this folder. The market feed (`market_feed.csv`) and Jäckel's *Let's
Be Rational* sources are **not** distributed here — see `README.md` for the market-feed
regeneration note and the LBR drop-in.

## 0. Machine preparation
- Linux or WSL2, mains power, no browser / IDE / cloud sync active; close other WSL sessions.
- Optionally `sudo cpupower frequency-set -g performance`.
- Pin every run: prefix with `taskset -c 2 nice -n -5` (drop `nice` if it needs sudo; keep `taskset`).

## 1. Paths and flags
```bash
cd reproduce
LBR=/path/to/LetsBeRational      # obtain separately; not redistributed here
FLAGS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math"
INC="-I../include/volfi"
```
`-ffp-contract=off` is required (it makes every fusion the explicit `fma` in the source, so
codegen is deterministic across TUs, ISAs, and compilers). Never use `-ffast-math`.

## 2. Sanity check (must print SMOKE PASS)
```bash
g++ $FLAGS -march=native $INC smoke_test.cpp -o smoke && ./smoke | tail -2
```

## 3. Build the benchmark, both instruction sets
```bash
g++ $FLAGS -march=native $INC -I"$LBR" -DNO_XL_API -w benchmark_vec.cpp \
  "$LBR/lets_be_rational.cpp" "$LBR/erf_cody.cpp" \
  "$LBR/normaldistribution.cpp" "$LBR/rationalcubic.cpp" -o bench_512

g++ $FLAGS -mavx2 -mfma -mno-avx512f $INC -I"$LBR" -DNO_XL_API -w benchmark_vec.cpp \
  "$LBR/lets_be_rational.cpp" "$LBR/erf_cody.cpp" \
  "$LBR/normaldistribution.cpp" "$LBR/rationalcubic.cpp" -o bench_256
```

## 4. Run — 5 repetitions per binary
```bash
for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ./bench_512 > results/run512_$i.txt; done
for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ./bench_256 > results/run256_$i.txt; done
```
Medians across the 5 reps are used; spread should be under a few percent on a quiet host.

**Abort and report if any run violates an invariant:**
- `GRIDBATCH_VS_SCALAR ... mismatches=0 (>1ulp=0)`
- WING accuracy `max` ~2.5e-14 (the benchmark's c-rounding floor), no 1e-2 outliers.

## 5. Reading the output
- fixed-`h` surface rows (CENTRAL / small-`h` LEFT / large-`v` RIGHT) → the surface-workload
  rows (columns: LBR, ours scalar, batch AVX-512, batch AVX2);
- the `MARKET-REALISTIC FEED` line → the market-weighted row (the headline for a live book);
  present only if `market_feed.csv` is supplied (otherwise skipped);
- the `stream E_warm2_feed` / `E_warm3_feed` lines just after it → the warm streaming
  re-inversion rate on the same feed and host (warm-2step = the live steady state);
- the mixed broad-grid line → the coverage-stress row;
- `run512_*` fills the AVX-512 batch column, `run256_*` the AVX2 column (LBR and scalar
  columns should agree between the two builds — a cross-check).

The reference outputs from the paper's quiet-host run are in `results/`.

## Verification suite (host-independent; run once to confirm the build)
```bash
g++ $FLAGS -march=native $INC verify_vec.cpp -o vv && ./vv   # must print BIT-IDENTITY: PASS, pts>1e-15=0
```
`oracle_*.bin` are the `mpmath` golden vectors.
