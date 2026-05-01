# volfi v0.1.0

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

Thus the implemented fast path only has to solve the positive-moneyness OTM problem.

## Version

This release is `volfi v0.1.0`: the first public research prototype of the OTM-projected implied-variance kernel.

## Method

The implementation uses:

1. a domain-specialized rational seed for implied variance $w$;
2. one Halley refinement;
3. analytic GIG-density derivative structure;
4. an OTM-only fast kernel;
5. an optional precomputed OTM context for fixed $h$.

The current implementation is a research kernel, not a global production replacement for Jaeckel's LetsBeRational.

## Scope

The benchmark grid is

$$
v\in\{0.01,0.05,0.10,\ldots,2.00\},
\qquad
\Delta\in\{0.55,0.70,0.80,0.95\}.
$$

This gives $41\times4=164$ OTM test cases after projection to $h>0$.

## Files

```text
include/volfi/volfi.hpp      core header
bench/bench_otm_grid.cpp     OTM-grid benchmark
tests/test_otm_grid.cpp      OTM-grid accuracy check
Makefile                     simple build entry point
CMakeLists.txt               optional CMake build
```

## Build

```bash
make
```

To override the compiler:

```bash
make CXX=g++
```

Run the accuracy test:

```bash
make test
```

Run the benchmark:

```bash
make bench
```

## Minimal API

```cpp
#include <volfi/volfi.hpp>

double w = volfi::implied_variance_otm(h, c_otm);
double v = volfi::implied_volatility_otm(h, c_otm, t);
```

For repeated evaluations at the same moneyness $h$, use the precomputed context:

```cpp
volfi::otm_context ctx(h);
double w = volfi::implied_variance_otm(ctx, c_otm);
```

The context stores $h$, $h^2$, $\exp(h/2)$, and $\exp(h)$, avoiding repeated computation of these terms in the hot path.

The header also contains a normalized-call helper that projects ITM calls to the OTM side:

```cpp
double w = volfi::implied_variance_call_normalised(k, c);
```

## Test results

There are two benchmark blocks below. They should not be mixed:

1. **Release benchmark**: uses the current packaged `volfi v0.1.0` code and compares the direct OTM kernel with the precomputed-context kernel.
2. **Reference comparison with LetsBeRational**: an earlier same-run comparison against LetsBeRational. That comparison used the direct OTM kernel, not the later precomputed-context micro-optimization.

### Release benchmark: packaged `volfi v0.1.0` code

This benchmark uses the bundled OTM code with 5000 repetitions of the 164-case grid.

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

| variant | mean abs variance error | max abs variance error | max rel variance error | mean abs volatility error | max abs volatility error | max rel volatility error |
|---|---:|---:|---:|---:|---:|---:|
| direct OTM kernel | `4.17e-16` | `2.66e-15` | `9.49e-14` | `1.96e-16` | `8.88e-16` | `4.74e-14` |
| precomputed OTM context | `4.17e-16` | `2.66e-15` | `9.49e-14` | `1.96e-16` | `8.88e-16` | `4.74e-14` |

Timing:

| variant | mean ns/eval | median ns/eval |
|---|---:|---:|
| direct OTM kernel | `93.98` | `93.75` |
| precomputed OTM context | `89.81` | `89.97` |

The precomputed context improves scalar speed by about

$$
\frac{93.75-89.97}{93.75}\approx 4.0\%.
$$

### Reference comparison with LetsBeRational

This comparison used the same 164-case OTM grid and native C++ loops.

```text
compiler: g++
flags: -O3 -march=native
runs: 7
evaluations per run: 30000 x 164
reported unit: nanoseconds per implied-volatility evaluation
```

The LetsBeRational comparison used the normalized implied-volatility routine from the supplied LetsBeRational shared library, not py_vollib and not a Python wrapper.

Accuracy:

| method | mean abs volatility error | max abs volatility error | max rel volatility error |
|---|---:|---:|---:|
| volfi direct OTM kernel | `1.59e-16` | `5.60e-16` | `5.60e-14` |
| LetsBeRational normalised | `1.67e-16` | `6.66e-16` | `1.14e-14` |

Timing from that same LBR-comparison run:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| volfi direct OTM kernel | `102.85` | `102.26` | `101.30` | `105.77` |
| LetsBeRational normalised | `233.43` | `231.41` | `228.30` | `248.69` |

Median speed ratio in the same LBR-comparison run:

$$
\frac{231.41}{102.26} \approx 2.26.
$$

So, on this benchmark grid, the specialized direct OTM implied-variance kernel was about `2.3x` faster than the normalized LetsBeRational call while retaining absolute volatility errors below `1e-14`.

The later precomputed-context kernel was not benchmarked in the same run against LetsBeRational. Using the same LetsBeRational median only as an orientation gives

$$
\frac{231.41}{89.97} \approx 2.57,
$$

but the conservative same-run comparison is the `2.26x` figure above.

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
- LetsBeRational is a global solver; this implementation is specialized to the stated OTM-projected grid.
- The source bundle does not include LetsBeRational source or binaries.
- No license has been selected in this prototype bundle. Add a license before making reuse terms explicit.
