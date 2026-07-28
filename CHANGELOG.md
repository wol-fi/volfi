# Changelog

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
