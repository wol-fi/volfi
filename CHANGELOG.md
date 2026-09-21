# Changelog

## v0.3.1

Faster everywhere, same accuracy contract. On one core the book kernel's batch path takes 16.6 ns
per quote on AVX-512 and 23.2 on AVX2 on the market feed (v0.3.0: 29 and 51), which is 11.6 and
8.3 times the rate of Let's Be Rational. Every entry, scalar and batch, is now faster than the
reference on every tile, including `UPPER`. Results are NOT bit-identical to v0.3.0 (rows trimmed,
reconstruction fused, `UPPER` rebuilt); scalar, AVX2, AVX-512 and CUDA remain bit-identical to
one another.

### Changed
- **Book kernel.** Region A evaluates 16 exact rows (was 17) and region B 16 (was 18); the dropped
  rows contributed nothing at double precision. The reconstruction is `v = fma(t x, S1, t)` with
  the rows summed from `m = 1`, so the leading term is not rounded twice. Truth-set errors fall
  from 2.13 / 2.39 to 1.69 / 1.99 ULP; the 20,000-point boundary campaign from `5.7e-16` to
  `5.3e-16` (40 worst rechecked at 60 digits).
- **Book batch driver.** Each tile is processed in passes with short loop bodies: the coordinate
  `a` for every register, then the kernels, with pure region-A registers evaluated in place and
  all other lanes collected into per-tile lists; region B runs as a table pass and a row pass.
  Lanes the kernel will decline are no longer evaluated by it (the reach test is repeated in
  vector form, operation for operation). Per lane the arithmetic is unchanged.
- **`UPPER` chart.** `a_U = h - 2 log(1 - c)`, `x0 = -Phi^{-1}(exp(-a_U/2)/2)` as a 12-term Chebyshev
  series in `sqrt(a_U)`, a 13-coefficient seed in two bounded coordinates (worst seed error
  `4.9e-5` over 5.2 million points), and exactly ONE Householder-3 step. The step's error
  constant is bounded by 217/18 on the chart, so a seed within `9.5e-5` reaches `1e-15`. The
  residual needs one exponential and two `erfcx`. No quantile kernel, no Mills ratio. Oracle gate
  1.62 ULP (v0.3.0: 2.15); scalar 122 ns against 485. The old chart is kept as
  `br::upper_variance_v030`.
- **Routing.** A vector pre-filter (`detail::grid_upper_first`) classifies `UPPER` lanes with the
  seam twins and evaluates them directly; it serves the grid batch, the speculative driver's
  drain and the book batch's declined lanes. Single-quote calls test the seams in the book
  coordinate (`volfi_annulus_fastroute_tables.hpp`): outside a margin of four fit errors the
  decision is certain and equals the exact seam comparison, inside it the exact seams are used.
  The context constructor no longer computes `exp(-h/2)` and `exp(h)`.
- **GPU.** Device tables regenerated; one-step `UPPER` mirror; the book driver gained the
  synthetic `UPPER` tile and a host-check mode exists for the routed driver
  (`-DVGB_HOST_CHECK=1`). H100 PCIe: book kernel 0.061 ns per quote in file order and 0.031
  sorted by `a` (0.072 / 0.032); `UPPER` chart alone 0.045 against 0.173.
- **Timing protocol.** 31 passes of eight sweeps per cell (`BENCH_RUNS=31 REPEATS=8`). With seven
  passes of four sweeps a batch cell lasts 13 ms at 16 ns per quote and its median is bimodal.

### Added
- `include/volfi/volfi_annulus_upper1_tables.hpp`, `volfi_annulus_fastroute_tables.hpp`.
- `reproduce/book/seam_gate.cpp` (806,000 quotes on the seams and band edges at -2..+2 ulp: batch
  == scalar, fast route never contradicts the exact route), `run_cpu_tiles.sh`, and the synthetic
  `Upper*` tile plus `BENCH_TILES_ONLY` / `BENCH_WB_ONLY` in `cpu_all_bench.cpp`.
- `reproduce/upper/`: the generators of the `UPPER` tables and of the fast-route tables, and the
  2,465-point 40-digit truth file.
- `gpu/make_upper1_cuda.py`.
- `reproduce/book/results/*v031*`: the clean CPU campaign, the gates, the accuracy recomputation
  on every oracle set, the 60-digit recheck, the `UPPER` gate and seed scan, the H100 session.

### Completed on 2026-09-21
- The link-time-optimized builds, the routed inverter's chart-pure table, warm start and
  batch-size sweep were re-measured on the release (`reproduce/results/v0.3.1/paper_rows_2026-09-21/`), and
  bit identity across GCC 11 and Clang 14 was re-verified (`cross_compiler_2026-09-21.txt`).
- `docs/volfi_v0.3.1_paper.pdf` replaces the v0.3.0 paper. Every number in it is a v0.3.1 measurement,
  except the two transfer-bound GPU rows and the third-party runs.

## v0.3.0 documentation updates (2026-09-16)

### Changed
- `reproduce/` reorganized by purpose: `check/`, `accuracy/`, `bench/`, `gen/`, `data/`, `results/`
  (per release). One README maps every table of the paper to its harness and result file, and one
  Makefile runs every gate (`make check`, `make check-isa`). Binaries build into `out/` and run from
  `data/`. `reproduce/book/` and `reproduce/upper/` no longer exist. Older entries below use the old paths.
- Generators write to `data/` and `include/volfi/` (they pointed at development-tree folders).

### Removed
- `include/volfi/volfi.hpp`, `volfi_fastpatch.hpp`, `volfi_logc_libm.hpp`: the v0.1 engine, referenced
  by nothing. `paper_volfi.hpp` remains, the routed charts use it.
- `reproduce/fixed_bench.cpp` (covered by `benchmark_vec.cpp`), `run_cpu_bench.sh` and its three
  `bench_v030_*` outputs, and superseded result files (`gpu_near_run_2026-09-14` and `b`,
  the first `pde_gpu_run_2026-09-14`). All remain in the git history.

## v0.3.0

The book kernel. One straight-line evaluation with a single branch and a 63-term table that
answers all but three quotes in thirty thousand of a real S&P 500 book, without routing and
without iteration. The routed four-chart inverter of v0.2.4 is unchanged and stands behind it as
the fallback and as the coverage of the whole feasible domain. The paper was retitled and
rewritten around it (`docs/volfi_v0.3.0_paper.pdf`).

### Added
- `include/volfi/volfi_wb.hpp`, `volfi_wb_tables.hpp`, `volfi_wb_vec.hpp` — the whole-book
  kernel: `volfi_wb::implied_variance_wb(h, c, &code)` and its lane-for-lane AVX-512 / AVX2 twin
  `implied_variance_wb_batch`. Region A (`a < 2 pi`, `theta = h/a <= 0.35`): 17 exact rational
  rows in `t^2 = h^2/A0^2`. Region B (`a >= 2 pi`, `h <= min(4, 0.16 a + 0.85)`): 18 rows
  re-expanded in the conformal variable `q = z/(sqrt(1+z)+1)^2`, `z = h^2/(4 pi^2)`. One
  two-cell table (`G(a)` on `[0, 2 pi]`, 23 terms; `V0/sqrt(a)` in `u = 1/sqrt(a)` on
  `[2 pi, 700]`, 40 terms). Two new kernels for the vector path: a division-free `log1p` with
  a Sterbenz-exact reduction (0.87 ULP) and a division-free `expm1` on `(0, 16.5]` (1.00 ULP).
- `volfi_near_certified*.hpp`, `volfi_near_rec*.hpp`, `volfi_near_book*.hpp` — the intrinsic
  `NEAR` chart in its three forms (certified matched chart, 11-row recurrence chart, 12-cell
  table), which the book kernel grew out of and which the paper's tables still measure.
- `reproduce/book/` — the v0.3.0 campaign: gates, 40-digit truth sets (including the
  20,000-point campaign on every switch of the kernel and its 60-digit recheck), the one-binary
  CPU harness with Let's Be Rational and the PDE method, branch-wise harnesses, node persistence,
  the exact-row generator (`gen/rows_exact.py`, the recursion of Proposition 1), the table
  generator, and the raw outputs of the quiet-host and H100 runs.
- `gpu/volfi_near_certified_gpu.cu`, `gpu/make_near_cuda.py`, `gpu/run_near_gpu.sh`,
  `gpu/run_pde_gpu.sh` — the device port of the intrinsic kernels with its host self-check, and
  the container recipe that builds the PDE method's OpenCL path on the same card.
- `make book` and the CMake test `book`: the SIMD twin's bit-identity and the truth-set accuracy
  of the book kernel, no external inputs.

### Numbers (quiet host, one binary, medians; H100 PCIe, 300 passes)
- Accuracy: entry point worst `5.7e-16` (4 ULP) on the 20,000-point boundary campaign,
  `5.3e-16` (3 ULP) on the regular grid, `4.9e-16` on the traded feed; no point above `1e-15`
  on any set.
- CPU, full feed, ns per quote: book kernel batch **29** (AVX-512) / **51** (AVX2), scalar
  118 / 142; Let's Be Rational at its release flags 194 / 193; PDE method scalar 76 / 86;
  routed charts 45 / 89 batch. Book kernel batch is 6.7× / 3.8× the reference.
- H100: book kernel 0.032 ns per quote on the feed sorted by `a`, 0.072 in file order, 1.12
  with uploads and readback on every pass; recurrence kernel 0.021 on the `NEAR` tile; the PDE
  method 7.68 under its own transfer-inclusive convention, 4.93 kernels plus readback.
- Every device result bit-identical to the CPU scalar entry (host self-check: 0 mismatches on
  832,287 covered probes; device runs: 0 mismatches).

### Changed
- Version strings, CMake project version and binding versions to 0.3.0. The Python binding
  gains `implied_variance_book` / `implied_volatility_book` and the R binding `volfi_w_book` /
  `volfi_iv_book` (the book kernel through its SIMD twin, with the region code per quote); the
  routed v0.2.4 API is unchanged. Both bindings build and pass their tests (pybind11 3.0 under
  WSL; R 4.4.0 with Rtools 4.4 on Windows).
- The reference is now timed at its author's release flags (`-O3 -DNDEBUG -ffp-contract=fast`),
  where it is 17% faster than the v0.2 campaign's build (contraction off, assertions on). Every
  ratio in the v0.3.0 paper uses the release build; the v0.2.4 tables keep their own build and
  say so.

### Removed
- `include/volfi/tmp_base.hpp`, `current_libm.hpp`, `volfi_reorder.hpp`: referenced by nothing.
- `reproduce/oracle_real.bin`: the 1,150 real-quote oracle points derive from the licensed
  OptionMetrics feed and are no longer redistributed, consistent with the feed itself. The
  harnesses skip it when absent; the "real" rows of the paper are reproducible from a licensed
  copy through the recipe in `reproduce/README.md`.
- `docs/volfi_v0.2.3_paper.pdf`, superseded by `docs/volfi_v0.3.0_paper.pdf`.

### Not redistributed
The market feed and everything derived from it (OptionMetrics licence), Jäckel's Let's Be
Rational sources, and the PDE method's sources and 46 MB table. The comparison harnesses take
their paths as arguments; see `NOTICE.md`.

### Verification status
From the release tree with only `-Iinclude/volfi`: v0.2.4 suite `SMOKE PASS`, bit-identity
`grid=0 permuted=0 fixed-h=0`, `pts>1e-15=0`, `WARM PASS`; `wb_vec_gate` 0 mismatches on
393,806 quotes and 2.39 ULP worst on the truth set, on AVX-512, AVX2 and scalar builds;
`wb_gate` 29,997 of 30,000 feed quotes covered; `rec_vec_gate` 0 mismatches on 344,178 `NEAR`
quotes; `wb_truth_score` 4 ULP worst on 20,000 boundary points; the device port's host
self-check 0 mismatches in all four sections. The device runs themselves are the 2026-09-14
H100 sessions in `reproduce/book/results/`.

## v0.2.4

Chart rename, and nothing else. `LEFT` -> `NEAR`, `CENTRAL` -> `FAR`, `RIGHT` -> `UPPER`;
`WING` is unchanged. The pair `LEFT`/`RIGHT` read as two ends of one axis and was not —
`LEFT` was a condition on moneyness `h`, `RIGHT` a condition on price `c` — which misled
readers of both the paper and the source. Paper and source now use the same four names.

**No behaviour changed.** This was verified, not assumed. `reproduce/fingerprint.cpp` dumps
every driver's output as `%a` hex doubles over 66,000 points spanning all four charts — the
scalar entry, the checked entry, the mixed-`h` and fixed-`h` batch drivers, the 2- and 3-step
warm drivers, the raw-data wrapper, and the routing label per point. The resulting
339,430-line file is **byte-identical before and after the rename, on AVX-512, AVX2 and
scalar alike**. The reference numbers in `reproduce/BENCHMARK_PROTOCOL.md` were measured on
v0.2.3 and carry over unchanged for that reason.

### Changed
- ~750 identifiers across the headers, both SIMD twins, the CUDA port and every harness.
- Public API is untouched: `implied_variance_otm`, `implied_volatility`,
  `implied_variance_grid_batch`, `implied_variance_warm_batch`, `iv_status` and friends carry
  no chart name. Only internal `br::` / `detail::` names and device kernels changed
  (`br::left_variance` -> `br::near_variance`, `central_kernel` -> `far_kernel`, and so on).
- `reproduce/left_ceiling_sweep.cpp` -> `reproduce/near_ceiling_sweep.cpp`.

### Added
- `reproduce/fingerprint.cpp` — the bitwise-fingerprint tool above. Useful for any future
  refactor that claims to be neutral.

### Deliberately not renamed
- The per-point chart tag inside `oracle_*.bin`. It is certification data; `verify_vec`
  translates it for display (`C`->`F`, `L`->`N`, `R`->`U`), so the gate now prints
  `[A] [F] [N] [U] [W]`.
- `br::cwing_price` and `br::c2_price`, which name seams of the retired v0.2.2 geometry and
  exist only so `old_covered()` can measure what the pre-v0.2.3 architecture covered.

### Note on the checked-in results
`reproduce/results/*.txt` were produced by v0.2.3 and still show the old per-chart row labels.
The numbers are unaffected; a fresh run reproduces them row for row under the new names.

### Verification status
Full CPU re-gate passed: three ISAs, `SMOKE PASS`, bit-identity `grid=0 permuted=0
fixed-h=0`, `pts>1e-15=0`, identical per-chart maxima, plus the fingerprint diff above. The
CUDA source was renamed and **statically** checked — every `g::` name it references is
declared by the regenerated device headers, and no pre-rename identifier survives on the
device side — but it has **not been recompiled or re-run on a GPU** since the rename. The
device numbers in the paper and in `BENCHMARK_PROTOCOL.md` were measured on v0.2.3.

## v0.2.3

Routing rework. The chart boundaries moved so that a real option book almost never reaches
the expensive deep-wing evaluator, and the two tabulated charts were unified under a single
ceiling. Accuracy, the API and the bit-identity contract are unchanged.

The paper renames three of the four charts for readability: `NEAR` (was LEFT), `FAR` (was
CENTRAL), `UPPER` (was RIGHT); `WING` keeps its name. **The source keeps the original
identifiers** — see the mapping table at the top of `reproduce/BENCHMARK_PROTOCOL.md`.

### Changed
- Wing seam moved from the iso-`W` contour `W = 3` to `W = 3.8`, i.e. `c_w(h) = C(h, h/sqrt(7.6))`.
- The small-moneyness and table charts now share **one** ceiling, `c_top(h) = C(h, 1.85)`,
  instead of stepping between `C(h,1.70)` and `C(h,2)` at `h = 0.3`. The height is where both
  charts are certified: the large-volatility chart is machine-precise from `v = 1.55`, while
  the small-moneyness finisher starts to extrapolate as `v -> 2`.
- Routing now costs two frozen polynomial sweeps instead of three.
- Central table grown from 181 to 239 cells (+42 KB) to cover the deeper region the lower
  wing seam hands it.
- `Binv` gained a third Chebyshev regime (`b3`, 13 coefficients), reaching three octaves
  deeper in `rho` at `1.07e-16` worst error in `A`.

### Effect on a real book
On the 2024 S&P 500 tradeable population the route mix moves from `90.0 / 5.1 / 4.9 / 0.0`
to `92.0 / 6.8 / 1.2 / 0.0` percent (near / far / wing / upper) — the deep wing falls from
one quote in twenty to one in eighty.

Measured on the same quiet host as the v0.2.0 numbers (medians of five, ns per quote,
AVX-512 / AVX2):

| workload | v0.2.0 | v0.2.3 |
|---|---|---|
| market feed (2024 SPX) | 55 / 90 | **48 / 86** |
| market feed, warm 2-step | 39 / 83 | **37 / 79** |
| large-volatility batch, h=1 | 77 / 152 | 74 / 148 |
| small-moneyness batch, h=0.2 | 26 / 42 | 26 / 42 |

Nothing got slower: the third `Binv` chain, charged to ~92% of the book, and the larger table
are both free at measurement resolution, and the feed gain comes from the collapse of the wing
share. The two `FAR` fixed-`h` rows are **not** comparable across versions, because their
`v`-range had to move with the ceiling.

### Added
- `gpu/` — the CUDA port used for the paper's device table: `volfi_gpu_book.cu`, the two
  device-header generators, and the container run script. Every device result is bitwise
  identical to the CPU scalar entry (zero mismatches at zero ULP, on every chart).
  Full book on one H100 PCIe: **0.078 ns per quote**.
- `reproduce/bench_sweep.cpp` — batch-size sweep behind the paper's latency figure.
- `reproduce/feed_route_mix.cpp` — route mix of a quote file under the live classifier.
- `reproduce/left_ceiling_sweep.cpp` — the measurement that fixes the ceiling at 1.85.
- `reproduce/bench_phases.cpp`, `reproduce/build_all.sh` — driver phase breakdown and a
  one-command build of every harness.
- `reproduce/results/batchsweep.txt`, `reproduce/results/gpu_run_2026-07-27.txt`.

### Fixed
- The timing harness carried its own copy of the routing predicate, which silently kept the
  v0.2.2 seams: reported route mixes and per-chart buckets were labelled with the old
  boundaries while the timings themselves were correct. `route()` now delegates to the shipped
  classifier, and every fixed-`h` surface self-checks for chart purity.

## v0.2.0

New engine: a **routed, vectorizable table inverter** for Black–Scholes implied volatility
that supersedes the v0.1 quantile-identity kernel as the current release. The v0.1 kernel
remains in the tree — the new engine is built on top of it.

### Added
- Broad-domain inverter covering `v = sigma*sqrt(T)` up to ~8 and `h = |log(K/F)|` up to
  ~16.2, routing every quote to one of four charts (CENTRAL / LEFT / RIGHT / WING) whose
  boundaries are fixed by branch-point analysis of the inverse map.
- Machine-precision accuracy: zero points worse than `1e-15` relative in `sigma` against a
  40-digit `mpmath` oracle over the whole domain (worst ~`8.3e-16`), certified on every build
  by the golden vectors in `reproduce/`.
- **Bit-identical** scalar and SIMD (AVX-512 / AVX2) output, and bit-identity across
  gcc/clang and all three instruction sets, via `-ffp-contract=off`.
- Vectorized batch drivers (`implied_variance_grid_batch`) with an adaptive
  speculative/sort-then-batch dispatcher, and streaming warm-restart drivers
  (`implied_variance_warm_batch`, `implied_variance_book_step`, `implied_variance_book_tick`)
  for re-inverting a persistent book each snapshot.
- Total input contract: `iv_status` classification, `implied_variance_otm_checked`, and a
  raw-data wrapper `implied_volatility(F, K, price, T, is_call, ...)` with put–call parity
  projection to the OTM-call twin.
- `docs/volfi_v0.2.0_paper.pdf`: technical paper (method, certification, timing methodology).
- `reproduce/`: self-contained accuracy / bit-identity suite, timing benchmark, golden oracle
  vectors, build protocol, and reference run outputs.

### Removed
- The v0.1 `bench/` and `tests/` harness, superseded by the self-contained `reproduce/`
  suite. The root `make` / CMake targets now build and run the v0.2.0 checks in `reproduce/`.

### Notes
- The Python binding (`bindings/python`) is a native v0.2.0 NumPy binding: array inputs go
  through the vectorized batch driver, bit-identical to scalar, with the checked-status API,
  the put-call-parity wrapper, and the warm-restart driver. The R binding still wraps v0.1.
- *Let's Be Rational* (reference solver for the comparison benchmark) and the licensed
  OptionMetrics market feed are not redistributed; see `NOTICE.md` and `reproduce/README.md`.

## v0.1.8
- Research reference implementation of the Black–Scholes implied-variance quantile identity
  on a deliberately narrow domain, with C++ header library, tests, and Python/R bindings.
