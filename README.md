# vol-fi

`vol-fi` is a small C++ research prototype for fast Black-Scholes implied variance on the projected out-of-the-money call side.

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

## Method

The implementation uses:

1. a domain-specialized rational seed for implied variance $w$;
2. one Halley refinement;
3. analytic GIG-density derivative structure;
4. an OTM-only fast kernel.

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

The header also contains a normalized-call helper that projects ITM calls to the OTM side:

```cpp
double w = volfi::implied_variance_call_normalised(k, c);
```

## Reference benchmark with LetsBeRational comparison

The reference comparison used the same 164-case OTM grid and native C++ loops.

Benchmark setup:

```text
compiler: g++
flags: -O3 -march=native
runs: 7
evaluations per run: 30000 x 164
reported unit: nanoseconds per implied-volatility evaluation
```

The LetsBeRational comparison used the normalized implied-volatility routine from the supplied LetsBeRational shared library, not py_vollib and not a Python wrapper.

Hardware/software setting reported by the execution environment:

```text
OS/kernel: Linux 4.4.0, x86_64
CPU vendor: GenuineIntel
CPU model name reported by container: unknown
CPU count reported by container: 56
Threads per core: 1
Cores per socket: 56
Socket count: 1
Reported CPU MHz: 2793.439
Hypervisor vendor: Microsoft
Memory reported by container: 4 GiB
```

Accuracy on the 164-case grid:

| method | mean abs volatility error | max abs volatility error | max rel volatility error |
|---|---:|---:|---:|
| vol-fi OTM kernel | `1.59e-16` | `5.60e-16` | `5.60e-14` |
| LetsBeRational normalised | `1.67e-16` | `6.66e-16` | `1.14e-14` |

Variance accuracy for the vol-fi OTM kernel:

| metric | value |
|---|---:|
| mean absolute variance error | `3.26e-16` |
| max absolute variance error | `1.78e-15` |
| max relative variance error | `1.12e-13` |

Timing:

| method | mean ns/eval | median ns/eval | min ns/eval | max ns/eval |
|---|---:|---:|---:|---:|
| vol-fi OTM kernel | `102.85` | `102.26` | `101.30` | `105.77` |
| LetsBeRational normalised | `233.43` | `231.41` | `228.30` | `248.69` |

Median speed ratio:

$$
\frac{231.41}{102.26} \approx 2.26.
$$

On this benchmark grid, the specialized OTM implied-variance kernel is therefore about `2.3x` faster than the normalized LetsBeRational call while retaining absolute volatility errors below `1e-14`.

## Caveats

- The comparison is domain-specific.
- LetsBeRational is a global solver; this implementation is specialized to the stated OTM-projected grid.
- The source bundle does not include LetsBeRational source or binaries.
- No license has been selected in this prototype bundle. Add a license before making reuse terms explicit.
