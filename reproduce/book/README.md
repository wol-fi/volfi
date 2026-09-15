# reproduce/book — the v0.3.0 campaign (book kernel)

Everything behind the v0.3.0 numbers in the paper: gates, truth sets, harnesses, generators
and the raw outputs of the quiet-host and H100 runs. Headers are in `../../include/volfi`;
every build line below is run from this directory with

```bash
FLAGS="-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops"
INC="-I../../include/volfi"
```

`-ffp-contract=off` is mandatory (bit-identity), `-ffast-math` is forbidden.

## Contents

| file | needs | what it does |
|---|---|---|
| `wb_vec_gate.cpp` | nothing | book kernel SIMD twin == scalar on 393,806 quotes; accuracy on the 1,200-point truth set (`wb_truth.bin`) |
| `wb_gate.cpp` | feed | truth set per region, feed coverage (29,997 of 30,000), `expm1` kernel ULP, scalar latency vs the routed charts |
| `rec_gate.cpp`, `rec_vec_gate.cpp` | feed | the 11-row recurrence chart (the `NEAR`-band form): truth set, bit-identity, timing vs the 12-cell table and the routed chart |
| `wb_truth_score.cpp` | nothing | scores any truth file per class; `wb_truth_boundary.bin` is the 20,000-point campaign on every switch of the kernel (writes `.worst`) |
| `wb_accuracy.cpp` | LBR | book kernel, routed charts and Let's Be Rational on the oracle sets, per region (Table "accuracy") |
| `cpu_all_bench.cpp`, `run_cpu_all.sh` | feed, LBR, PDE | THE CPU TABLE: all four methods in one binary per ISA, full feed and region tiles, each also with `-flto` (Tables "book" and "pde") |
| `wb_lbr_bench.cpp`, `run_lbr_compare.sh` | feed, LBR | the same without the PDE method |
| `run_cpu_bench.sh` | feed | the pinned repeated runs of the four gates above (`results/bench_v030_*`) |
| `pde_compare.cpp`, `pde_branch.cpp`, `pde_regions.cpp` | feed, PDE | the PDE method's accuracy and CPU timing on the feed, branch-wise tiles, and per-region accuracy on the oracle sets; with `-DPDE_GPU` its OpenCL path (see `../../gpu/run_pde_gpu.sh`) |
| `persistence.cpp` | vendor file | node persistence of the `a = 2 pi` partition over consecutive quoting days (needs the licensed yearly file, not redistributed) |
| `dump_feed_ref.cpp`, `fastvollib_feed.py` | feed | writes `h c sigma region` for scoring third-party Python libraries; fast-vollib's Jäckel and Halley paths scored and timed on it |
| `gen/rows_exact.py` | mpmath, sympy | the exact rational rows `P_m` of Proposition 1 by the recursion of its proof (`rows_P.json`) |
| `gen/certified_wb.py` | mpmath | the two-cell table `G` / `V0/sqrt(a)` and the conformal rows, oracle-gated, emitted as `volfi_wb_tables.hpp` |
| `gen/make_readme_figures.py` | matplotlib | the two README figures (`docs/figures/cpu_one_binary.png`, `gpu_book_kernel.png`) from the numbers in `results/` |
| `gen/make_truth.py`, `gen/make_truth_wb_boundary.py` | mpmath | the 40-digit truth sets; `--recheck FILE` re-inverts a `.worst` file at 60 digits |
| `near_truth.bin`, `wb_truth.bin`, `wb_truth_boundary.bin` | | truth sets (`int64 n`, then `h c v` doubles) |
| `results/` | | raw outputs: `bench_v030_*` (three pinned runs), `cpu_all_*_clean.txt` (the CPU table), `gpu_near_run_*`, `pde_gpu_run_*`, `pde_regions_*`, `pde_branch_cpu_*`, `persistence_*`, `wb_boundary_score_*`, `wb_boundary_recheck60_*`, `fastvollib_cpu_*` |

## Standalone (no external inputs)

```bash
g++ $FLAGS -march=native $INC wb_vec_gate.cpp -o wbvg && ./wbvg      # mismatches = 0, 2.39 ULP worst
g++ $FLAGS -mavx2 -mfma -mno-avx512f $INC wb_vec_gate.cpp -o wbvg2 && ./wbvg2
g++ $FLAGS -mno-avx2 -mno-avx512f $INC wb_vec_gate.cpp -o wbvgs && ./wbvgs
g++ $FLAGS -march=native $INC wb_truth_score.cpp -o wbts && ./wbts wb_truth_boundary.bin
```

Expected: `wb_truth_boundary.bin  n=20000`, book kernel max `5.695e-16` (4 ULP, 0 over
`1e-15`), routed charts `9.203e-16`.

## With the market feed

`market_feed.csv` is 30,000 lines `h c` (normalized OTM-call log-moneyness and price) resampled
from the tradeable 2024 S&P 500 population of a licensed OptionMetrics file. It is **not
redistributed**; `../README.md` (section "Market-feed row") gives the filter, the forward and
discount conventions and the parity projection that regenerate it from your own licence. Any
file of `h c` pairs in that format works for every harness here. Put it in this directory, then

```bash
g++ $FLAGS -march=native $INC wb_gate.cpp -o wbg && ./wbg
BENCH_PIN="taskset -c 2 nice -n -5" BENCH_TAG=run1 bash run_cpu_bench.sh     # a pinned repetition
```

## The CPU table (Let's Be Rational and the PDE method)

Neither third-party code is redistributed. Put Jäckel's sources (http://www.jaeckel.org/,
revision 1520 was used) in `third_party/LetsBeRational/` and a clone of
https://github.com/maticivan/PDE-method-for-implied-volatility with its `loadPartition.txt` in
`third_party/PDE-method-for-implied-volatility/` (or pass both paths), then

```bash
bash run_cpu_all.sh [/path/to/LetsBeRational] [/path/to/PDE-method-for-implied-volatility]
```

One binary per ISA (AVX-512, AVX2, no SIMD, each also with `-flto` on every side) times the
reference at its Makefile's flags (`-O3 -DNDEBUG -ffp-contract=fast`), the PDE method at its
compile script's (`-O2 -fopenmp`), and ours at `-O3 -ffp-contract=off`, by the same loop on
the feed and on `NEAR` / `FAR` / `WING` tiles. The report lands in `out/`. The checked-in
`results/cpu_all_20260914_185859_clean.txt` is the quiet-host run behind the paper.

## GPU

`../../gpu/run_near_gpu.sh` (book kernel, recurrence kernel, 12-cell table) and
`../../gpu/run_pde_gpu.sh` (the PDE method's OpenCL path in the same container). Generate the
device tables first with `python3 ../../gpu/make_near_cuda.py`. The host self-check of the port
runs without a GPU:

```bash
cd ../../gpu && python3 make_near_cuda.py && cd .. && \
g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -march=native -DNCG_HOST_CHECK=1 -DNB_LAYOUT=1 \
    -x c++ -Iinclude/volfi -Igpu gpu/volfi_near_certified_gpu.cu -o ncheck && (cd reproduce/book && ../../ncheck)
```

Expected: `mismatches = 0` in all four sections (344,178 `NEAR` quotes; 832,287 covered probes
of the 849,818-point price box for the whole-book kernel).
