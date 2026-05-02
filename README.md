# volfi v0.1.7
`volfi` is a C++ research prototype for fast Black-Scholes implied variance on the normalized out-of-the-money call side.

This `v0.1.7` release is the wing-speed update: it extends the previous projected-OTM kernel with a precomputed log-`c` wing seed for the true raw OTM-call domain while keeping the projected ITM-to-OTM path.

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
Makefile                           simple build entry point
CMakeLists.txt                     optional CMake build
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

## Benchmarks Vs. LetsBeRational

LetsBeRational was evaluated through its native shared-library interface using `NormalisedImpliedBlackVolatility`.

### Fixed Projected OTM Grid

Grid:

$$
v \in \{0.01, 0.05, 0.10, \ldots, 2.00\}, \qquad
\Delta \in \{0.55, 0.70, 0.80, 0.95\}.
$$

These are ITM call deltas before exact projection to the normalized OTM-call side.

Accuracy:

| method | mean abs vol error | max abs vol error | max rel vol error |
|---|---:|---:|---:|
| volfi | `1.52e-16` | `6.66e-16` | `1.42e-14` |
| LetsBeRational | `1.53e-16` | `6.66e-16` | `1.46e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `44.04` | `41.03` | `39.23` | `64.28` |
| LetsBeRational | `166.26` | `150.62` | `143.90` | `288.77` |

Median speed ratio:

$$
\frac{150.62}{41.03} \approx 3.67.
$$

### Fixed True OTM Grid

Grid:

$$
v \in \{0.01, 0.05, 0.10, \ldots, 2.00\}, \qquad
\Delta \in \{0.05, 0.20, 0.30, 0.45\}.
$$

Accuracy:

| method | mean abs vol error | max abs vol error | max rel vol error |
|---|---:|---:|---:|
| volfi | `1.80e-16` | `8.88e-16` | `2.31e-14` |
| LetsBeRational | `1.54e-16` | `4.44e-16` | `2.22e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `57.89` | `57.53` | `52.89` | `67.19` |
| LetsBeRational | `184.38` | `183.32` | `168.94` | `195.39` |

Median speed ratio:

$$
\frac{183.32}{57.53} \approx 3.19.
$$

### Random True OTM Grid

Randomization:

$$
v \sim U(0.01,2.0), \qquad \Delta \sim U(0.01,0.5).
$$

Accuracy:

| method | mean abs vol error | max abs vol error | max rel vol error |
|---|---:|---:|---:|
| volfi | `1.75e-16` | `1.11e-15` | `4.00e-14` |
| LetsBeRational | `1.58e-16` | `1.11e-15` | `2.10e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `66.82` | `67.73` | `60.81` | `73.87` |
| LetsBeRational | `198.29` | `187.69` | `184.84` | `237.09` |

Median speed ratio:

$$
\frac{187.69}{67.73} \approx 2.77.
$$

Across these projected and true OTM benchmarks, `volfi v0.1.7` remains at machine-precision accuracy and is roughly `2.8x` to `3.7x` faster than LetsBeRational on this domain.

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
- Median latency is the preferred scalar speed summary.
