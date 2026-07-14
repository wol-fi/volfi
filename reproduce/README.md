# Reproduce — verification and benchmarks for volfi v0.2.0

A self-contained bundle that certifies the two invariants of the routed inverter and
reproduces the paper's timing table. The library headers live in `../include/volfi`; the
harness sources here include them by name, so every command below adds `-I../include/volfi`.

Build environment: Linux (or WSL2), GCC ≥ 11 or Clang ≥ 14. `-ffp-contract=off` is
mandatory — it makes every fused multiply-add explicit in the source, which is what gives the
scalar and SIMD paths bit-identical output across translation units, instruction sets, and
compilers. Never use `-ffast-math`.

```bash
FLAGS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math"
INC="-I../include/volfi"
```

## Contents

| file                     | needs LBR | what it certifies / does                                    |
|--------------------------|-----------|-------------------------------------------------------------|
| `smoke_test.cpp`         | no        | quick self-check; prints `SMOKE PASS`                       |
| `verify_vec.cpp`         | no        | accuracy vs `mpmath` oracle + scalar==batch bit-identity    |
| `warm_test.cpp`          | no        | streaming warm-start accuracy basin                         |
| `fixed_bench.cpp`        | no        | fixed-`h` surface timing (no reference)                     |
| `benchmark_vec.cpp`      | yes       | full timing table incl. LBR head-to-head + market feed      |
| `accuracy_vs_lbr.cpp`    | yes       | side-by-side accuracy heatmap data vs LBR                   |
| `oracle_*.bin`           | —         | 40-digit `mpmath` golden vectors (loaded by the above)      |
| `results/run*.txt`       | —         | reference outputs from the paper's quiet-host run           |

(The v0.1 baseline kernel `paper_volfi.hpp`, pulled in by both the engine and the comparison
benchmark, ships in `../include/volfi` and is found via `-I../include/volfi`.)

The oracle `.bin` files are loaded by relative path, so run each binary from this directory.

## Standalone checks (no external dependencies)

```bash
g++ $FLAGS -march=native $INC smoke_test.cpp  -o smoke && ./smoke | tail -2
g++ $FLAGS -march=native $INC verify_vec.cpp  -o vv    && ./vv     # BIT-IDENTITY: PASS, pts>1e-15 = 0
g++ $FLAGS -march=native $INC warm_test.cpp   -o warm  && ./warm
g++ $FLAGS -march=native $INC fixed_bench.cpp -o fb    && ./fb
```

Repeat `verify_vec` on the other instruction sets to confirm cross-ISA bit-identity:

```bash
g++ $FLAGS -mavx2 -mfma -mno-avx512f $INC verify_vec.cpp -o vv_avx2 && ./vv_avx2
g++ $FLAGS -mno-sse4.1 -mno-avx $INC          verify_vec.cpp -o vv_scalar && ./vv_scalar
```

All three must print `BIT-IDENTITY: PASS` and `pts>1e-15 = 0`, with identical accuracy tables.

## Full timing table (requires Let's Be Rational)

The LBR head-to-head is not runnable out of the box: Jäckel's *Let's Be Rational* is his own
copyrighted work and is **not** redistributed here. Obtain the sources
(`lets_be_rational.cpp`, `erf_cody.cpp`, `normaldistribution.cpp`, `rationalcubic.cpp` and the
`lets_be_rational.h` header) from the author's site, put them in a directory, and point `LBR`
at it:

```bash
LBR=/path/to/LetsBeRational
g++ $FLAGS -march=native $INC -I"$LBR" -DNO_XL_API -w benchmark_vec.cpp \
    "$LBR"/lets_be_rational.cpp "$LBR"/erf_cody.cpp \
    "$LBR"/normaldistribution.cpp "$LBR"/rationalcubic.cpp -o bench_512

g++ $FLAGS -mavx2 -mfma -mno-avx512f $INC -I"$LBR" -DNO_XL_API -w benchmark_vec.cpp \
    "$LBR"/lets_be_rational.cpp "$LBR"/erf_cody.cpp \
    "$LBR"/normaldistribution.cpp "$LBR"/rationalcubic.cpp -o bench_256

for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ./bench_512 > results/run512_$i.txt; done
for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ./bench_256 > results/run256_$i.txt; done
```

See `BENCHMARK_PROTOCOL.md` for machine preparation, the per-line → Table-4 mapping, and the
invariants each run must satisfy. The reference outputs from the paper's run are in
`results/`.

## Market-feed row

The paper's market row is driven by `market_feed.csv` — 30,000 normalized `(h, c)` pairs
resampled from the 2024 SPX end-of-day *tradeable* option population. That population is
derived from a **licensed OptionMetrics feed and is not distributed here.** When the file is
absent, `benchmark_vec` prints `market_feed.csv not found (skipped)` and reports every other
row normally.

To regenerate it from your own licensed data: filter to a trading year, drop zero-volume /
zero-open-interest quotes, match each option to its vendor forward, project puts and ITM calls
to their OTM-call twin by put–call parity (`C - P = F - K`, undiscounted), and write one
`h c` pair per line, where `h = |log(K/F)|` and `c` is the undiscounted OTM price divided by
the forward. A resample of ~30k rows reproduces the paper's route mix (~90/5/5 percent
left/wing/central).
