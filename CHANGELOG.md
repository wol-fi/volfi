# Changelog

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
- Language bindings (`bindings/python`, `bindings/r`) still expose the v0.1 kernel; native
  v0.2.0 bindings are a follow-up.
- *Let's Be Rational* (reference solver for the comparison benchmark) and the licensed
  OptionMetrics market feed are not redistributed; see `NOTICE.md` and `reproduce/README.md`.

## v0.1.8
- Research reference implementation of the Black–Scholes implied-variance quantile identity
  on a deliberately narrow domain, with C++ header library, tests, and Python/R bindings.
