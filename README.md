# volfi v0.1.6

`volfi` is a C++ research prototype for fast Black-Scholes implied variance on the projected out-of-the-money call side.

The core problem is reduced to

$$
h = |\log(K/F)| > 0, \qquad w = \sigma^2T, \qquad w = Q_h(c_*),
$$

where $c_*$ is the normalized OTM-call price and $w$ is total implied variance. ITM calls are mapped exactly to the OTM side before entering the kernel:

$$
\tilde c = 1 + \frac{F}{K}(c - 1), \qquad h=-\log(K/F).
$$

Working paper: [An Explicit Solution to Black-Scholes Implied Volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6649499).
The closed-form identity underlying the kernel is the variance-space quantile representation

$$
w = v^2 = \mathcal F^{-1}_{GIG}\left(c_*;\frac12,\frac14,h^2\right),\qquad h>0.
$$

Here "closed-form" means analytically explicit: the Black-Scholes root search is replaced by a specified distributional quantile. Numerically, that quantile is still a non-elementary special-function.

The implementation evaluates this map $Q_h(c_*)$ directly and refines the result in implied-variance space.

At the forward strike, the inversion collapses to

$$
v = 2\Phi^{-1}\left(\frac{1+c}{2}\right),\qquad w=v^2.
$$

## Version

`volfi v0.1.6` is a precomputed-context OTM implied-variance kernel for fast Black-Scholes inversion on fixed and randomized projected OTM grids.

## Method

The implementation uses:

1. exact OTM projection for non-ATM calls;
2. a precomputed OTM context for fixed moneyness $h$;
3. a domain-specialized rational seed for implied variance $w$;
4. a local rational seed in a small transition region;
5. one Halley refinement using analytic density-derivative structure.

The precomputed context stores moneyness-only quantities:

$$
h,\qquad h^2,\qquad \exp(h/2),\qquad \exp(h).
$$

This is intended for production-style surface construction where many prices are inverted on a fixed strike/moneyness grid.

`volfi` is a research kernel, not a global production replacement for Jaeckel's LetsBeRational. The comparison below is restricted to the stated projected OTM domain.

## Scope

Fixed call-delta grid, projected to the OTM side:

$$
v\in\{0.01,0.05,0.10,\ldots,2.00\},\qquad
\Delta\in\{0.55,0.70,0.80,0.95\}.
$$

These are call-delta levels above `0.5`, so they correspond to ITM calls before exact projection to the OTM-call representation.

Random call-delta grid, projected to the OTM side:

$$
v\sim U(0.01,2.0),\qquad \Delta\sim U(0.5,0.99).
$$

## Files

```text
include/volfi/volfi.hpp          flagship C++ header
bench/bench_otm_grid.cpp         C++ fixed-grid benchmark
tests/                           C++ accuracy and robustness tests
Makefile                         simple C++ build entry point
CMakeLists.txt                   optional C++ CMake build
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

The precomputed context is the preferred fast path in `v0.1.6`.

The header also contains a normalized-call helper that projects ITM calls to the OTM side:

```cpp
double w = volfi::implied_variance_call_normalised(k, c);
```

## Test results

The LetsBeRational reference was evaluated through its native shared-library interface using `NormalisedImpliedBlackVolatility`, not through `py_vollib` or a Python wrapper.

All benchmark cases are parameterized by call delta first and then projected exactly to the normalized OTM-call representation before inversion.

### Fixed Call-Delta Grid Projected To OTM vs. LetsBeRational

Grid:

$$
v\in\{0.01,0.05,0.10,\ldots,2.00\},\qquad
\Delta\in\{0.55,0.70,0.80,0.95\}.
$$

Equivalent OTM call-delta levels would be `0.45, 0.30, 0.20, 0.05`.

Benchmark setting:

```text
LBR revision: 1520
compiler: g++ 11.4.0
cases: 164
repetitions per timing run: 5000
evaluations per timing run: 820000
runs: 9
reported unit: nanoseconds per implied-volatility evaluation
```

Accuracy:

| method | mean abs volatility error | max abs volatility error | max rel volatility error | errors > 1e-14 |
|---|---:|---:|---:|---:|
| volfi | `1.53e-16` | `6.66e-16` | `1.18e-14` | 0 |
| LetsBeRational | `1.61e-16` | `8.88e-16` | `1.73e-14` | 0 |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `46.56` | `46.38` | `46.27` | `47.69` |
| LetsBeRational | `133.31` | `132.54` | `131.98` | `137.14` |

Median speed ratio:

$$
\frac{132.54}{46.38}\approx 2.86.
$$

### Random Call-Delta Grid Projected To OTM vs. LetsBeRational

Randomization:

$$
v\sim U(0.01,2.0),\qquad \Delta\sim U(0.5,0.99).
$$

This is a random ITM call-delta grid before exact projection to the normalized OTM-call side.

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

| method | mean abs volatility error | max abs volatility error | max rel volatility error | errors > 1e-14 |
|---|---:|---:|---:|---:|
| volfi | `1.52e-16` | `1.11e-15` | `6.12e-14` | 0 |
| LetsBeRational | `2.02e-16` | `1.55e-15` | `4.01e-14` | 0 |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `50.46` | `50.47` | `50.14` | `50.88` |
| LetsBeRational | `145.35` | `145.09` | `142.50` | `148.26` |

Median speed ratio:

$$
\frac{145.09}{50.47}\approx 2.87.
$$

On both the fixed and randomized call-delta benchmarks, after exact projection to the OTM side, `volfi v0.1.6` is roughly `2.9x` faster than LetsBeRational on this projected OTM domain while retaining absolute volatility errors around machine precision.

## Hardware/software setting

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
CPU flags include: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr
```

The benchmark was run on a local laptop inside WSL2. Timings will still vary with compiler, libm implementation, power state, and host scheduling.

## Caveats

- The comparison is domain-specific.
- LetsBeRational is a global solver; this implementation is specialized to the stated OTM-projected domain.
- Benchmark timings may differ across machines, compilers, WSL configurations, and libm implementations.
- Median timings are the preferred scalar-latency summary.
