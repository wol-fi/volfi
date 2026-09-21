# reproduce: checks, accuracy, benchmarks, generators, results

Everything behind the numbers in the paper and in the top-level README. The folders are by
purpose. `results/` is by release, because the paper's appendix still cites older campaigns.

```
check/      gates without external inputs: accuracy, scalar == batch bit identity, seams
accuracy/   scoring against the 40-digit oracle sets, with and without Let's Be Rational
bench/      timing harnesses and their run scripts (need the feed, LBR and/or the PDE method)
gen/        generators of every shipped table and truth file, and of the README figures
data/       oracle and truth files; also the run directory of every binary
results/    raw outputs: v0.3.1/, v0.3.0/, v0.2/
```

Build flags everywhere: `-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops`, headers
from `../include/volfi`. `-ffp-contract=off` is mandatory: it makes every fused multiply-add
explicit in the source, which is what gives scalar, AVX2, AVX-512 and CUDA identical doubles.
Never use `-ffast-math`. Linux or WSL2, GCC 11 or later.

## Checks (no external inputs)

```bash
make check        # native build: smoke, verify, warm, book, seam
make check-isa    # the identity gates on AVX-512, AVX2 and without SIMD
```

Binaries are built into `out/` and run from `data/`, because the truth files load by relative path.

| file | what it certifies |
|---|---|
| `check/smoke_test.cpp` | quick self-check, prints `SMOKE PASS` |
| `check/verify_vec.cpp` | accuracy on `oracle_vec.bin` per chart, and scalar == grid == permuted == fixed-h bit for bit |
| `check/warm_test.cpp` | the warm-start basin and its bit identity |
| `check/wb_vec_gate.cpp` | book kernel: SIMD twin == scalar on 363,806 quotes (393,806 with the feed), 1.99 ULP on the 1,200-point truth set |
| `check/wb_gate.cpp` | book kernel per region on the truth set (1.69 / 1.99 ULP); with the feed also coverage (29,997 of 30,000) |
| `check/seam_gate.cpp` | 806,000 quotes on both routing seams and the band edges at -2..+2 ulp: batch == scalar, and the fast scalar route never contradicts the exact route |
| `check/rec_gate.cpp`, `rec_vec_gate.cpp` | the 11-row recurrence chart, an intermediate design the paper's appendix still measures (needs the feed) |
| `check/fingerprint.cpp` | bitwise dump of every driver, for cross-compiler comparison or to prove that a refactor changed nothing |

Expected from `make check`: `SMOKE PASS`, `BIT-IDENTITY: PASS` with `pts>1e-15 = 0`, `WARM PASS`,
`mismatches = 0` three times, and `contradictions 0`.

## Accuracy

| file | needs | what it does |
|---|---|---|
| `accuracy/wb_accuracy.cpp` | LBR | book kernel in front, routed charts alone and Let's Be Rational on any oracle file, per region (the paper's accuracy table) |
| `accuracy/accuracy_vs_lbr.cpp` | LBR | routed charts against LBR, with `--dump` for the heat map |
| `accuracy/wb_truth_score.cpp` | nothing | scores any truth file per class and writes the 40 worst points to `<file>.worst`; `data/wb_truth_boundary.bin` is the 20,000-point campaign on every switch of the book kernel |
| `accuracy/make_oracle_sets.py` | mpmath | the oracle itself: `heat` regenerates `oracle_heat.bin`, `recheck FILE` re-inverts a set at 60 digits |

```bash
mkdir -p out && g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -march=native -I../include/volfi accuracy/wb_truth_score.cpp -o out/score
cd data && ../out/score wb_truth_boundary.bin      # book kernel 5.254e-16 (4 ULP), routed charts 9.203e-16, 0 above 1e-15
```

Every LBR figure forms its input as `beta = c * exp(-h/2)`. Forming it as `c / exp(h/2)` doubles
its worst error.

`oracle_real.bin` (1,150 real quotes) is not redistributed, because it derives from the licensed feed.

## Benchmarks

Three inputs are not redistributed.

- **Let's Be Rational** (Jäckel, http://www.jaeckel.org/, revision 1520 was used). Put the
  sources in `bench/third_party/LetsBeRational/` or pass the path.
- **The PDE table method** (https://github.com/maticivan/PDE-method-for-implied-volatility) with
  its `loadPartition.txt`, in `bench/third_party/PDE-method-for-implied-volatility/` or by path.
- **The market feed** `data/market_feed.csv`: 30,000 lines `h c`, resampled from the tradeable
  2024 S&P 500 population of a licensed OptionMetrics file. To regenerate it from your own
  licence: one trading year, positive volume and open interest, the tradeability screen of the
  paper, each option matched to its vendor forward by expiry and settlement convention, puts and
  in-the-money calls projected to the out-of-the-money call by parity, then `h = |log(K/F)|` and
  `c` = undiscounted OTM price over `min(F, K)`. Any file of `h c` pairs in that format works.

| file | needs | what it does |
|---|---|---|
| `bench/cpu_all_bench.cpp`, `run_cpu_all.sh` | feed, LBR, PDE | THE CPU TABLE: all methods in one binary per instruction set, full feed, batches of 64, `NEAR` / `FAR` / `WING` tiles and a synthetic `Upper*` tile, each also with `-flto` |
| `bench/run_cpu_tiles.sh` | feed, LBR, PDE | the tile rows alone |
| `bench/wb_lbr_bench.cpp`, `run_lbr_compare.sh` | feed, LBR | the same without the PDE method |
| `bench/benchmark_vec.cpp`, `build_all.sh` | LBR | routed inverter: chart-pure fixed-moneyness batches, the broad 120,000-quote grid, market feed, warm start |
| `bench/bench_sweep.cpp` | LBR | batch-size sweep of the routed drivers |
| `bench/bench_phases.cpp` | nothing | phase breakdown of the cold routed driver |
| `bench/pde_compare.cpp`, `pde_branch.cpp`, `pde_regions.cpp` | feed, PDE | the PDE method's accuracy and timing, whole feed, per tile and per region; with `-DPDE_GPU` its OpenCL path |
| `bench/persistence.cpp` | vendor file | persistence of the `a = 2 pi` partition over consecutive days (needs the licensed yearly file) |
| `bench/feed_route_mix.cpp` | feed | route mix of a quote file |
| `bench/dump_feed_ref.cpp`, `fastvollib_feed.py` | feed | reference values for scoring third-party Python libraries |

```bash
BENCH_RUNS=31 REPEATS=8 BENCH_PIN="taskset -c 2 nice -n -5" bash bench/run_cpu_all.sh [LBR dir] [PDE dir]
```

Use 31 passes of eight sweeps. With the older 7 x 4 a batch cell lasts 13 ms at 16 ns per quote and
its median is bimodal. `BENCH_TILES_ONLY=1` and `BENCH_WB_ONLY=1` restrict the run. Machine
preparation and the invariants each run must satisfy are in `BENCHMARK_PROTOCOL.md`.

GPU: `../gpu/run_near_gpu.sh` (book driver) and `../gpu/run_gpu.sh` (routed driver), after
`python3 make_near_cuda.py`, `make_device_constants.py`, `make_device_tables.py` and
`make_upper1_cuda.py` in `../gpu`. Both drivers have a host self-check that needs no GPU
(`-DNCG_HOST_CHECK=1`, `-DVGB_HOST_CHECK=1`).

## Generators

| file | needs | output |
|---|---|---|
| `gen/rows_exact.py` | mpmath, sympy | the exact rational rows `P_m` of Proposition 1 (`rows_P.json`) |
| `gen/certified_wb.py` | mpmath | the two-cell table and the conformal rows: `include/volfi/volfi_wb_tables.hpp`, `data/wb_truth.bin` |
| `gen/gen_upper_v031.py`, `gen_upper_one.py` | numpy, scipy, mpmath | the one-step `UPPER` chart: `erfcx` piece, quantile series, 13-coefficient seed, `data/upper_truth.bin` |
| `gen/gen_fastroute.cpp` | nothing | the routing seams in the book coordinate, with margins |
| `gen/make_truth.py`, `make_truth_wb_boundary.py` | mpmath | `data/near_truth.bin`, `data/wb_truth_boundary.bin`; `--recheck FILE` re-inverts a `.worst` file at 60 digits |
| `gen/near_ceiling_sweep.cpp` | nothing | the measurement that fixes the shared ceiling at `v = 1.85` |
| `gen/make_readme_figures.py` | matplotlib | the two README figures |

`certified_wb.py` emits the full row arrays of v0.3.0. The shipped header sets `NW_MA = 15` and
`NW_NB = 16`, so v0.3.1 evaluates sixteen rows in each region and leaves the last one (region A)
and last two (region B) unused. The `UPPER` generators write research-format tables next to
themselves. The shipped headers `volfi_annulus_upper1_tables.hpp` and
`volfi_annulus_fastroute_tables.hpp` hold the same doubles.

## Results, and which table they back

| paper | harness | result file |
|---|---|---|
| Table 2 (accuracy), boundary campaign, market prices | `accuracy/wb_accuracy.cpp`, `wb_truth_score.cpp` | `results/v0.3.1/accuracy_v031_2026-09-18.txt`, `wb_boundary_recheck60_v031_2026-09-18.txt` |
| Table 3 (CPU) and appendix Table 6, feed and batches-of-64 rows | `bench/cpu_all_bench.cpp` | `results/v0.3.1/cpu_all_v031_20260918_171645_clean.txt` |
| appendix Table 6, LTO rows | `bench/cpu_all_bench.cpp` | `results/v0.3.1/paper_rows_2026-09-21/B_lto_*.txt` |
| Table 4 (GPU), resident rows | `gpu/*.cu` | `results/v0.3.1/gpu_v031_run_2026-09-18.txt` |
| Table 4, rows with host transfers, and the PDE method on the H100 | `gpu/run_near_gpu.sh`, `run_pde_gpu.sh` | `results/v0.3.0/gpu_near_run_2026-09-14c.txt`, `pde_gpu_run_2026-09-14b.txt` |
| identity and seam gates | `check/` | `results/v0.3.1/gates_v031_20260918_171537.txt` |
| one-step `UPPER` chart, seed margin | development harness, not shipped | `results/v0.3.1/upper_one_step_gate_2026-09-17_clean.txt`, `upper_seed_scan_2026-09-17.txt` |
| appendix Table 7 (routed inverter, five repetitions), warm start | `bench/benchmark_vec.cpp` | `results/v0.3.1/paper_rows_2026-09-21/C_run512_*.txt`, `C_run256_*.txt` |
| batch-size sweep | `bench/bench_sweep.cpp` | `results/v0.3.1/paper_rows_2026-09-21/D_batchsweep.txt` |
| bit identity across GCC 11 and Clang 14 | `check/fingerprint.cpp`, `check/wb_vec_gate.cpp --hex` | `results/v0.3.1/cross_compiler_2026-09-21.txt` |
| routed charts on the H100, July campaign | `gpu/volfi_gpu_book.cu` | `results/v0.2/gpu_run_2026-07-27.txt` |
| PDE method per tile and per region, node persistence, fast-vollib | `bench/pde_*.cpp`, `persistence.cpp`, `fastvollib_feed.py` | `results/v0.3.0/` |

`results/v0.2` and `results/v0.3.0` are the campaigns of the earlier releases, kept for comparison. The
paper no longer cites them, except for the two transfer-bound GPU rows and the third-party runs
listed above. The `v0.2` files carry the old chart labels `CENTRAL` / `LEFT` / `RIGHT` for `FAR` /
`NEAR` / `UPPER`.
