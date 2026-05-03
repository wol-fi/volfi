# volfi v0.1.7

`volfi` is a C++ research prototype for fast Black-Scholes implied variance.

It implements an efficient implied-volatility solver based on the inverse-Gaussian / generalized-inverse-Gaussian quantile representation in [An Explicit Solution to Black-Scholes Implied Volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6649499).

For out-of-the-money normalized calls with forward log-moneyness `k > 0`, normalized call price `c`, and total implied volatility `v = sigma sqrt(T)`, the variance-space representation is

$$
v(k,c)^2 = \mathcal{F}^{-1}_{GIG}\left(c; \frac{1}{2}, \frac{1}{4}, k^2\right), \qquad k > 0.
$$

This `v0.1.7` release is the wing-speed update: it extends the previous projected-OTM kernel with a precomputed log-`c` wing seed for the true raw OTM-call domain while keeping the projected ITM-to-OTM path.

## Why variance space?

The representation above changes the inversion target from normalized volatility to total variance. Instead of solving $c \mapsto v$, `volfi` estimates $c_* \mapsto w = v^2$ directly. This has several practical advantages compared with volatility-space inversion methods (e.g., LetsBeRational):

- the probabilistic identity identifies total variance `w` itself as the
  natural quantile variable;
- the fitted target is the GIG variance quantile, not its square root;
- this avoids warping the approximation target by the square-root map,
  which is most nonlinear near small variance;
- the GIG density has simple logarithmic derivatives in `w`, giving a
  compact analytic Halley correction in variance space;
- moneyness-dependent quantities can be precomputed in `otm_context`,
  making repeated inversions on fixed strike or moneyness grids cheaper.

The distinction is not that `volfi` avoids price residual evaluation.
The single correction step evaluates the GIG-CDF residual

$$
F_h(w)-c_*,
\qquad
F_h(w)=F_{GIG}\left(w;\frac12,\frac14,h^2\right),
$$

using its equivalent closed-form evaluator. The algorithm is therefore
best viewed as a direct total-variance GIG-quantile method rather than a
volatility-space implied-volatility solver.

## Core setup

The normalized problem is

$$
h = |\log(K/F)| > 0, \qquad w = \sigma^2 T, \qquad w = Q_h(c_*),
$$

where `c_*` is the normalized OTM-call price and `w` is total implied variance.

For normalized calls, ITM prices are projected exactly to the OTM side before inversion:

$$
\tilde c = 1 + \frac{F}{K}(c - 1), \qquad h = -\log(K/F).
$$

At the forward strike the inversion reduces to

$$
v = 2\Phi^{-1}\left(\frac{1+c}{2}\right), \qquad w = v^2.
$$

## What Changed In v0.1.7

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

$$
v \in \{0.01, 0.05, 0.10, \ldots, 2.00\}, \qquad
\Delta \in \{0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90\}.
$$

- Random grid:

$$
v \sim U(0.01, 2.0), \qquad \Delta \sim U(0.01, 0.9).
$$

### Fixed Grid

Benchmark setting:

```text
cases: 451
repetitions per timing run: 5000
evaluations per timing run: 2255000
runs: 9
reported unit: nanoseconds per implied-volatility evaluation
```

Accuracy:

| method | mean abs vol error | max abs vol error | max rel vol error |
|---|---:|---:|---:|
| volfi | `1.86e-16` | `1.33e-15` | `5.46e-14` |
| LetsBeRational | `1.64e-16` | `7.77e-16` | `2.32e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `53.01` | `51.80` | `49.08` | `60.41` |
| LetsBeRational | `188.10` | `181.23` | `176.99` | `211.33` |

Median speed ratio:

$$
\frac{181.23}{51.80} \approx 3.50.
$$

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
| volfi | `1.60e-16` | `1.55e-15` | `5.01e-14` |
| LetsBeRational | `1.72e-16` | `1.33e-15` | `3.68e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `55.88` | `55.48` | `54.19` | `59.38` |
| LetsBeRational | `168.77` | `166.35` | `164.75` | `179.06` |

Median speed ratio:

$$
\frac{166.35}{55.48} \approx 3.00.
$$

On Black prices generated over these fixed and random grids, `volfi v0.1.7` recovers the input volatility to near machine precision and is about `3x` to `3.5x` faster than LetsBeRational in this benchmark setup.

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

- This is a research kernel, not a global replacement for LetsBeRational.
- The comparison is domain-specific.
- Timings vary across machines, compilers, libm implementations, and host scheduling.
- The current implementation targets GCC/Clang-like compilers.
