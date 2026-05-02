# volfi v0.1.4

`volfi` is a small C++ research prototype for fast Black-Scholes implied variance on the projected out-of-the-money call side.

The core problem is reduced to

$$
h = |\log(K/F)| > 0, \qquad w = \sigma^2T, \qquad w = Q_h(c_*),
$$

where $c_*$ is the normalized OTM-call price and $w$ is total implied variance.

ITM calls are mapped exactly to the OTM side before entering the kernel:

$$
\tilde c = 1 + \frac{F}{K}(c - 1), \qquad h=-\log(K/F).
$$

At the forward strike, the inversion collapses to the normal-quantile formula

$$
v = 2\Phi^{-1}\left(\frac{1+c}{2}\right),
\qquad w=v^2.
$$

Thus the implemented fast path only has to solve the positive-moneyness OTM variance-quantile problem.

## Version

This release is `volfi v0.1.4`: a precomputed-context OTM implied-variance kernel with randomized robustness tests over both delta and total volatility.

## Method

The implementation uses:

1. an OTM projection for all non-ATM calls;
2. a precomputed OTM context for fixed moneyness $h$;
3. a domain-specialized rational seed for implied variance $w$;
4. one Halley refinement using analytic density-derivative structure;
5. a narrow extra-refinement fallback for a small transition region needed for the randomized $v$-$\Delta$ test.

The precomputed context stores moneyness-only quantities such as

$$
h,\qquad h^2,\qquad \exp(h/2),\qquad \exp(h).
$$

This is intended for production-style surface construction where many prices are inverted on a fixed strike/moneyness grid.

`volfi` is a research kernel, not a global production replacement for Jaeckel's LetsBeRational. The comparison below is intentionally restricted to the stated projected domain.

## Scope

The fixed benchmark grid is

$$
v\in\{0.01,0.05,0.10,\ldots,2.00\},
\qquad
\Delta\in\{0.55,0.70,0.80,0.95\}.
$$

This gives $41\times4=164$ OTM test cases after projection to $h>0$.

The randomized robustness benchmark uses

$$
v\sim U(0.01,2.0),
\qquad
\Delta\sim U(0.5,0.99).
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
```

To override the compiler:

```bash
make CXX=g++
```

Run the accuracy tests:

```bash
make test
```

Run the fixed-grid benchmark:

```bash
make bench
```

## Minimal API

```cpp
#include <volfi/volfi.hpp>

volfi::otm_context ctx(h);
double w = volfi::implied_variance_otm(ctx, c_otm);
```

The precomputed context is the preferred fast path in `v0.1.4`.

The header also contains a normalized-call helper that projects ITM calls to the OTM side:

```cpp
double w = volfi::implied_variance_call_normalised(k, c);
```

## Test results

### Fixed OTM grid

Grid:

$$
v\in\{0.01,0.05,0.10,\ldots,2.00\},
\qquad
\Delta\in\{0.55,0.70,0.80,0.95\}.
$$

Benchmark setting:

```text
compiler: g++ 14.2.0
flags: -Ofast -march=native -ffp-contract=fast -fno-math-errno
cases: 164
repetitions per timing run: 5000
evaluations per timing run: 820000
runs: 9
reported unit: nanoseconds per implied-volatility evaluation
```

Accuracy:

| method | mean abs variance error | max abs variance error | max rel variance error | mean abs volatility error | max abs volatility error | max rel volatility error |
|---|---:|---:|---:|---:|---:|---:|
| volfi precomputed OTM context | `3.15e-16` | `2.66e-15` | `2.83e-14` | `1.34e-16` | `6.66e-16` | `1.42e-14` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi precomputed OTM context | `62.43` | `61.69` | `60.12` | `67.16` |

### Edge deltas

The same $v\in\{0.01,0.05,\ldots,2.00\}$ grid was tested at two edge deltas.

| delta | cases | mean abs volatility error | max abs volatility error | max rel volatility error |
|---:|---:|---:|---:|---:|
| `0.52` | 41 | `1.41e-16` | `4.44e-16` | `1.25e-15` |
| `0.99` | 41 | `2.14e-16` | `7.49e-16` | `3.11e-14` |

### Random delta grid

Random test with fixed total-volatility grid:

```text
random deltas: 5000
cases: 205000
Delta ~ Uniform(0.5, 0.99)
v in {0.01, 0.05, 0.10, ..., 2.00}
seed: 1234567
```

| method | mean abs volatility error | max abs volatility error | max rel volatility error | abs error > 1e-14 |
|---|---:|---:|---:|---:|
| volfi precomputed OTM context | `1.51e-16` | `1.11e-15` | `9.61e-14` | 0 |

### Random delta and random volatility, compared with LetsBeRational

Randomized comparison against LetsBeRational revision 1520:

```text
accuracy cases: 200000
Delta ~ Uniform(0.5, 0.99)
v ~ Uniform(0.01, 2.0)
random seed: 20260502
LBR routine: NormalisedImpliedBlackVolatility from LetsBeRational.so
no Python wrapper
```

Accuracy:

| method | mean abs volatility error | max abs volatility error | max rel volatility error | abs error > 1e-14 |
|---|---:|---:|---:|---:|
| volfi precomputed OTM context | `1.44e-16` | `1.11e-15` | `6.80e-14` | 0 |
| LetsBeRational normalised | `1.77e-16` | `1.33e-15` | `4.51e-14` | 0 |

Timing:

```text
timing cases: 5000
repetitions per timing run: 1000
evaluations per timing run: 5000000
runs: 9
reported unit: nanoseconds per implied-volatility evaluation
```

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi precomputed OTM context | `73.98` | `68.34` | `66.58` | `93.07` |
| LetsBeRational normalised | `201.62` | `194.12` | `173.09` | `243.29` |

Median speed ratio:

$$
\frac{194.12}{68.34}\approx 2.84.
$$

On this randomized benchmark, `volfi v0.1.4` is about `2.8x` faster than the normalized LetsBeRational call while retaining absolute volatility errors around machine precision.

## Hardware/software setting

Hardware/software setting reported by the execution environment:

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

## Potential further speed improvements

The current implementation is already close to the scalar fast path: a rational variance seed plus Halley refinement. Further improvements would likely require changing the approximation structure rather than only simplifying algebra.

The most promising direction is to split the OTM domain into additional regions and fit lower-degree rational seeds. A more accurate branchwise seed may allow replacing Halley by a cheaper Newton step, or omitting refinement in parts of the domain.

A second opportunity is to reduce the cost of the residual evaluation. The Halley step still evaluates normal-CDF-like terms. Tailored Cody/Remez approximations on the restricted benchmark domain, or branch rules that drop negligible terms, could reduce this cost.

For fixed strike grids, more quantities can be precomputed. Since the seed depends on $a=h/(1+h)$, the bivariate rational approximation could be collapsed further into univariate price-transform polynomials for each fixed $h$.

Finally, a batch/SIMD API may improve throughput for volatility-surface construction. The current benchmark measures scalar latency; vectorized evaluation over many strikes and maturities could give larger gains than further scalar micro-optimizations.
