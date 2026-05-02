# volfi v0.1.5

`volfi` is a C++ research prototype for fast Black-Scholes implied variance on the projected out-of-the-money call side.

The core problem is reduced to

$$
h = |\log(K/F)| > 0, \qquad w = \sigma^2T, \qquad w = Q_h(c_*),
$$

where $c_*$ is the normalized OTM-call price and $w$ is total implied variance.

ITM calls are mapped exactly to the OTM side before entering the kernel:

$$
\tilde c = 1 + \frac{F}{K}(c - 1), \qquad h=-\log(K/F).
$$

At the forward strike, the inversion collapses to

$$
v = 2\Phi^{-1}\left(\frac{1+c}{2}\right),\qquad w=v^2.
$$

## Version

`volfi v0.1.5` is a precomputed-context OTM implied-variance kernel with a local transition seed for robustness on fixed and randomized OTM grids.

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

Fixed OTM grid:

$$
v\in\{0.01,0.05,0.10,\ldots,2.00\},\qquad
\Delta\in\{0.55,0.70,0.80,0.95\}.
$$

Random OTM grid:

$$
v\sim U(0.01,2.0),\qquad \Delta\sim U(0.5,0.99).
$$

## Files

```text
include/volfi/volfi.hpp          core header
bench/bench_otm_grid.cpp         fixed-grid benchmark
tests/test_otm_grid.cpp          fixed-grid accuracy check
tests/test_edge_grid.cpp         delta 0.52 and 0.99 checks
tests/test_random_delta.cpp      random-delta robustness check
tests/test_random_v_delta.cpp    random-delta and random-volatility robustness check
tests/test_atm.cpp               ATM formula check
Makefile                         simple build entry point
CMakeLists.txt                   optional CMake build
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

The precomputed context is the preferred fast path in `v0.1.5`.

The header also contains a normalized-call helper that projects ITM calls to the OTM side:

```cpp
double w = volfi::implied_variance_call_normalised(k, c);
```

## Test results

The LetsBeRational reference was evaluated through its native shared-library interface using `NormalisedImpliedBlackVolatility`, not through `py_vollib` or a Python wrapper.

An additional fixed-grid and random-grid rerun on another Linux machine is included in the `Additional Tests` section at the end of this README.

### Fixed OTM grid vs. LetsBeRational

Grid:

$$
v\in\{0.01,0.05,0.10,\ldots,2.00\},\qquad
\Delta\in\{0.55,0.70,0.80,0.95\}.
$$

Benchmark setting:

```text
LBR revision: 1520
compiler: g++ 14.2.0
flags: -Ofast -march=native -ffp-contract=fast -fno-math-errno
cases: 164
repetitions per timing run: 5000
evaluations per timing run: 820000
runs: 9
reported unit: nanoseconds per implied-volatility evaluation
```

Accuracy:

| method | mean abs volatility error | max abs volatility error | max rel volatility error | errors > 1e-14 |
|---|---:|---:|---:|---:|
| volfi | `1.41e-16` | `6.66e-16` | `1.42e-14` | 0 |
| LetsBeRational normalised | `1.46e-16` | `6.66e-16` | `1.44e-14` | 0 |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `60.81` | `60.25` | `59.79` | `64.35` |
| LetsBeRational normalised | `165.04` | `164.15` | `163.65` | `168.77` |

Median speed ratio:

$$
\frac{164.15}{60.25}\approx 2.72.
$$

### Random OTM grid vs. LetsBeRational

Randomization:

$$
v\sim U(0.01,2.0),\qquad \Delta\sim U(0.5,0.99).
$$

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
| volfi | `1.44e-16` | `1.11e-15` | `6.80e-14` | 0 |
| LetsBeRational normalised | `1.77e-16` | `1.33e-15` | `4.51e-14` | 0 |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `65.81` | `65.75` | `65.48` | `66.19` |
| LetsBeRational normalised | `171.45` | `171.54` | `170.32` | `172.39` |

Median speed ratio:

$$
\frac{171.54}{65.75}\approx 2.61.
$$

On both the fixed and randomized OTM benchmarks, `volfi v0.1.5` is faster than the normalized LetsBeRational call while retaining absolute volatility errors around machine precision.

## Hardware/software setting

```text
OS/kernel: Linux 4.4.0, x86_64
compiler: g++ (Debian 14.2.0-19) 14.2.0
CPU vendor: GenuineIntel
CPU count reported by container: 56
Threads per core: 1
Cores per socket: 56
Socket count: 1
Reported CPU MHz: 2793.439
Hypervisor vendor: Microsoft
Memory reported by container: 4 GiB
CPU flags include: AVX2, AVX512F, AVX512DQ, AVX512CD, AVX512BW, AVX512VL, FMA
```

The benchmark was run inside a virtualized container. The reported CPU information should therefore be interpreted as environment metadata, not as a precise physical-machine specification.

## Caveats

- The comparison is domain-specific.
- LetsBeRational is a global solver; this implementation is specialized to the stated OTM-projected domain.
- Benchmark timings were measured inside a virtualized container and may differ across machines, compilers, and libm implementations.
- Median timings are the preferred scalar-latency summary.

## Potential further speed improvements

Further improvements will likely require changing the approximation structure rather than only simplifying algebra.

The most promising direction is to split the OTM domain into additional regions and fit lower-degree rational seeds. A more accurate branchwise seed may allow replacing Halley by a cheaper Newton step, or omitting refinement in parts of the domain.

A second opportunity is to reduce the cost of the residual evaluation. The Halley step still evaluates normal-CDF-like terms. Tailored Cody/Remez approximations on the restricted benchmark domain, or branch rules that drop negligible terms, could reduce this cost.

For fixed strike grids, more quantities can be precomputed. Since the seed depends on $a=h/(1+h)$, the bivariate rational approximation could be collapsed further into univariate price-transform polynomials for each fixed $h$.

Finally, a batch/SIMD API may improve throughput for volatility-surface construction. The current benchmark measures scalar latency; vectorized evaluation over many strikes and maturities could give larger gains than further scalar micro-optimizations.

## Additional Tests

Additional Linux rerun on a separate laptop using `volfi v0.1.5` and the native LetsBeRational shared library.

### Fixed OTM grid vs. LetsBeRational

Grid:

$$
v\in\{0.01,0.05,0.10,\ldots,2.00\},\qquad
\Delta\in\{0.55,0.70,0.80,0.95\}.
$$

Accuracy:

| method | mean abs volatility error | max abs volatility error | max rel volatility error |
|---|---:|---:|---:|
| volfi | `1.41e-16` | `6.66e-16` | `1.18e-14` |
| LetsBeRational normalised | `1.61e-16` | `8.88e-16` | `1.73e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `64.82` | `64.52` | `63.00` | `68.26` |
| LetsBeRational normalised | `153.97` | `152.60` | `148.59` | `164.07` |

Median speed ratio:

$$
\frac{152.60}{64.52}\approx 2.37.
$$

### Random OTM grid vs. LetsBeRational

Randomization:

$$
v\sim U(0.01,2.0),\qquad \Delta\sim U(0.5,0.99).
$$

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

| method | mean abs volatility error | max abs volatility error | max rel volatility error |
|---|---:|---:|---:|
| volfi | `1.52e-16` | `1.11e-15` | `6.12e-14` |
| LetsBeRational normalised | `2.02e-16` | `1.55e-15` | `4.01e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi | `72.75` | `70.86` | `67.18` | `88.03` |
| LetsBeRational normalised | `168.68` | `166.51` | `163.19` | `192.10` |

Median speed ratio:

$$
\frac{166.51}{70.86}\approx 2.35.
$$
