# volfi v0.1.7

`volfi` is a C++ research prototype for fast Black-Scholes implied variance. [Python](https://github.com/wol-fi/volfi/tree/main/bindings/python) and [R](https://github.com/wol-fi/volfi/tree/main/bindings/r) translations are also available.

It implements an efficient implied-volatility solver based on the generalized-inverse-Gaussian quantile representation in [An Explicit Solution to Black-Scholes Implied Volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6649499).

For out-of-the-money normalized calls with forward log-moneyness `k > 0`, normalized call price `c`, and total implied volatility `v = sigma sqrt(T)`, the variance-space representation is

```math
v(k,c)^2 = \mathcal{F}^{-1}_{GIG}\left(c; \frac{1}{2}, \frac{1}{4}, k^2\right), \qquad k > 0.
```

## Why variance space?

The formula above makes total variance the natural inversion variable. For fixed positive moneyness, write `F_h` for the corresponding GIG distribution function. Then the OTM inversion is simply the quantile problem `w = F_h^{-1}(c_*)`.

`volfi` fits and corrects this variance quantile directly. A volatility-space method instead targets its square root. This difference matters because the square-root map is strongly curved near small variance, while the GIG representation is already expressed in `w`.

The same choice also gives a compact correction step. With residual `R_h(w) = F_h(w) - c_*`, density `f_h(w) = F_h'(w)`, and log-density derivative `ell_h(w) = f_h'(w) / f_h(w)`, one Halley correction is

```math
w_1 = w_0 - \frac{2R_h(w_0)}{2f_h(w_0)-R_h(w_0)\ell_h(w_0)}.
```

For the GIG family used here,

```math
\ell_h(w) = -\frac{1}{2w} - \frac{1}{8} + \frac{h^2}{2w^2}.
```

Thus the correction is expressed through the GIG density and its logarithmic derivative in the native variable `w`. In volatility space, the corresponding residual is `G(v) = F_h(v^2) - c_*`, which introduces extra chain-rule terms from `w = v^2`.

The method still evaluates a residual. The point is that the residual is the GIG-CDF residual in total-variance space, evaluated through its equivalent closed form. In repeated inversions on fixed strike or moneyness grids, the moneyness-dependent quantities can also be precomputed in `otm_context`.

## Core setup

The normalized problem is

```math
h = |\log(K/F)| > 0, \qquad w = \sigma^2 T, \qquad w = Q_h(c_*),
```

where `c_*` is the normalized OTM-call price and `w` is total implied variance.

For normalized calls, ITM prices are projected exactly to the OTM side before inversion:

```math
\tilde c = 1 + \frac{F}{K}(c - 1), \qquad h = -\log(K/F).
```

At the forward strike the inversion reduces to

```math
v = 2\Phi^{-1}\left(\frac{1+c}{2}\right), \qquad w = v^2.
```

For put options, use put-call parity for a transformation into a call.

## What Changed In v0.1.7

This `v0.1.7` release is the wing-speed update: it extends the previous projected-OTM kernel with a precomputed log-`c` wing seed for the true raw OTM-call domain while keeping the projected ITM-to-OTM path.

- Keeps the precomputed `otm_context` fast path.
- Adds a wing seed for the true OTM-call region.
- Adds explicit fixed-grid and randomized true-OTM tests.
- Preserves the projected call-delta benchmark path from the earlier release.

## Layout

```text
include/volfi/volfi.hpp            flagship header
include/volfi/volfi_reorder.hpp    alternate ordering helpers
include/volfi/volfi_logc_libm.hpp  log-c wing support
tests/                             projected-OTM and true-OTM tests
bench/bench_otm_grid.cpp           fixed projected-grid benchmark
bench/bench_lbr_compare.cpp        optional volfi vs LetsBeRational comparison driver
docs/technical_note/               technical documentation (.tex and .pdf)
Makefile                           simple build entry point
CMakeLists.txt                     optional CMake build for core tests and standalone benchmark
```

## Build

```bash
make
make test
make bench
```

To override the compiler:

```bash
make CXX=g++
```

The optional CMake build covers the core tests and standalone `volfi` benchmark. The LetsBeRational comparison is a Makefile/manual target because it depends on a separately obtained local LetsBeRational build.

## Minimal API

```cpp
#include <volfi/volfi.hpp>

volfi::otm_context ctx(h);
double w = volfi::implied_variance_otm(ctx, c_otm);
```

For normalized calls:

```cpp
double w = volfi::implied_variance_call_normalised(k, c);
```

## Performance Test

The performance comparison was run against LetsBeRational through its native shared-library interface using `NormalisedImpliedBlackVolatility`.

The repository includes `bench/bench_lbr_compare.cpp`, but not the LetsBeRational source or binary. To reproduce the side-by-side comparison, fetch [LetsBeRational.7z](http://www.jaeckel.org/LetsBeRational.7z) separately and build or link it locally.

Minimal setup used for the comparison:

- obtain the upstream LetsBeRational source from the archive above
- build LetsBeRational locally as a native library or compile its source into a local benchmark build
- point `bench/bench_lbr_compare.cpp` at that local copy and benchmark `NormalisedImpliedBlackVolatility`

Example build command:

```bash
make bench_lbr_compare CXX=g++ \
  LBR_INCLUDE=/path/to/LetsBeRational \
  LBR_LIBDIR=/path/to/LetsBeRational/Linux
```

Test grids:

- Fixed grid:

```math
v \in \{0.01, 0.05, 0.10, \ldots, 2.00\}, \qquad
\Delta \in \{0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99\}.
```

- Random grid:

```math
v \sim U(0.01, 2.0), \qquad \Delta \sim U(0.01, 0.99).
```

### Fixed Grid

Benchmark setting:

```text
cases: 533
repetitions per timing run: 5000
evaluations per timing run: 2665000
runs: 9
reported unit: nanoseconds per implied-volatility evaluation
```

Accuracy:

| method | mean abs vol error | max abs vol error | max rel vol error |
|---|---:|---:|---:|
| volfi | `1.86e-16` | `1.33e-15` | `5.48e-14` |
| LetsBeRational | `1.62e-16` | `7.77e-16` | `2.32e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `48.29` | `48.53` | `46.29` | `49.42` |
| LetsBeRational | `169.37` | `169.62` | `163.93` | `174.12` |

Median speed ratio:

```math
\frac{169.62}{48.53} \approx 3.49.
```

### Random Grid

Benchmark setting:

```text
accuracy cases: 200000
random seed: 20260502
timing cases: 5000
repetitions per timing run: 1000
evaluations per timing run: 5000000
runs: 9
reported unit: nanoseconds per implied-volatility evaluation
```

Accuracy:

| method | mean abs vol error | max abs vol error | max rel vol error |
|---|---:|---:|---:|
| volfi | `1.61e-16` | `1.55e-15` | `6.51e-14` |
| LetsBeRational | `1.72e-16` | `1.33e-15` | `4.80e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `51.89` | `51.60` | `51.31` | `53.42` |
| LetsBeRational | `159.44` | `157.70` | `153.91` | `180.18` |

Median speed ratio:

```math
\frac{157.70}{51.60} \approx 3.06.
```

On Black prices generated over these fixed and random grids, `volfi v0.1.7` recovers the input volatility to near machine precision and is about `3.1x` to `3.5x` faster than LetsBeRational in this benchmark setup.

## Hardware/Software Setting

```text
OS/kernel: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 GNU/Linux
compiler: g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
CPU model: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz
CPU cores reported: 8
Threads per core: 2
Cores per socket: 4
Socket count: 1
Hypervisor vendor: Microsoft
Memory reported: 7.60 GiB
```

## Caveats

- This is a research kernel, not a drop-in replacement for well-tested existing routines.
- The comparison is domain-specific. The randomized tests were conducted over delta values from 0.01 to 0.99 and total implied volatility values from 0.01 to 2.0.
- Benchmarks were conducted on normalized Black calls. Empirical use requires additional computation for quote normalization, parity transformations, and supported-domain checks.
- Timings vary across machines, compilers, libm implementations, and host scheduling.
- The current implementation targets GCC/Clang-like compilers.

## License

`volfi` is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full license terms.

If this software, method, benchmark, or documentation influences research, software, internal development, or published results, please cite the repository and the accompanying technical note.

This software is provided as a research kernel and without warranty. It is not a drop-in replacement for production pricing, risk-management, execution, or model-validation systems without independent validation and appropriate integration work.
