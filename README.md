# volfi v0.1.8

`volfi` is a C++ research prototype for fast Black-Scholes implied variance. [Python](https://github.com/wol-fi/volfi/tree/main/bindings/python) and [R](https://github.com/wol-fi/volfi/tree/main/bindings/r) translations are also available.

It implements an efficient implied-volatility solver based on the generalized-inverse-Gaussian quantile representation in [An Explicit Solution to Black-Scholes Implied Volatility](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6649499).

For out-of-the-money normalized calls with forward log-moneyness `k > 0`, normalized call price `c`, and total implied volatility `v = sigma sqrt(T)`, the variance-space representation is

```math
v(k,c)^2 = \mathcal{F}^{-1}_{GIG}\left(c; \frac{1}{2}, \frac{1}{4}, k^2\right), \qquad k > 0.
```

**Where does the speed come from?**
- A good seed.
- Plus a single Halley refinement already attains near machine precision.
  
## Why variance space?

The formula above makes total variance the natural inversion variable. For fixed positive moneyness, write `F_h` for the corresponding GIG distribution function. Then the OTM inversion is simply the quantile problem `w = F_h^{-1}(c_*)`.

`volfi` fits and corrects this variance quantile directly. A volatility-space method instead targets its square root. This difference matters because the square-root map is more curved near small variance, while the GIG representation is already expressed in `w`.

The same choice also gives a compact correction step. With residual `R_h(w) = F_h(w) - c_*`, density `f_h(w) = F_h'(w)`, and log-density derivative `ell_h(w) = f_h'(w) / f_h(w)`, one Halley correction is

```math
w_1 = w_0 - \frac{2R_h(w_0)}{2f_h(w_0)-R_h(w_0)\ell_h(w_0)}.
```

For the GIG family used here,

```math
\ell_h(w) = -\frac{1}{2w} - \frac{1}{8} + \frac{h^2}{2w^2}.
```

Thus the correction is expressed through the GIG density and its logarithmic derivative in the native variable `w`. In volatility space, the corresponding residual is `G(v) = F_h(v^2) - c_*`, which introduces extra chain-rule terms from `w = v^2`. This matters because the direct volatility quantile contains a nested variance structure.

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

## What Changed In v0.1.8

This `v0.1.8` release adds a guarded high-speed approximate-Halley patch and hardens the projected wing seeds while preserving the one-Halley fallback design.

- Adds `volfi_fastpatch.hpp` with a certified regional approximate-Halley path on `h in [0.05,0.40]` and `c in [0.20,0.55]`.
- Keeps exact Halley as the fallback outside the certified region.
- Adds narrow low-price and wing seed branches for low-delta and high-delta projected wing cases.
- Extends fixed and randomized tests down to `Delta=0.001` and up to `Delta=0.999`.
- Preserves the precomputed `otm_context` path and the existing one-refinement structure.

## Layout

```text
include/volfi/volfi.hpp            flagship header
include/volfi/volfi_reorder.hpp    alternate ordering helpers
include/volfi/volfi_logc_libm.hpp  log-c wing support
tests/                             projected-OTM and true-OTM tests
bench/bench_otm_grid.cpp           fixed projected-grid benchmark
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

The optional CMake build covers the core tests and standalone `volfi` benchmark.

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

The `w` functions return total implied variance. Use the `implied_volatility_*` functions when volatility is required.

## Domain conventions

- OTM functions require `h > 0` and normalized OTM-call price `0 < c < 1`.
- Normalized call functions use `k = log(K/F)` and `c = C/F`; they require `max(0, 1 - exp(k)) < c < 1`.
- The full call function uses positive `f`, `k`, `d`, `t`, and an undiscounted normalized check after `c = price / (d f)`.
- Invalid domains return `NaN` by default. Compile with `VOLFI_STRICT_DOMAIN` to throw `std::domain_error` instead.
- The OTM API is not the ATM API. Use `implied_variance_call_normalised(0, c)` or `implied_volatility_call_normalised(0, c, t)` at the forward strike.

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
- Empirical behavior is domain-specific and should not be generalized outside supported regimes without independent testing.
- Benchmarks were conducted on normalized Black calls. Empirical use requires additional computation for quote normalization, parity transformations, and supported-domain checks.
- Timings vary across machines, compilers, libm implementations, and host scheduling.
- The current implementation targets GCC/Clang-like compilers.

## License

`volfi` is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full license terms.

If this software, method, benchmark, or documentation influences research, software, internal development, or published results, please cite the repository and the accompanying technical note.

This software is provided as a research kernel and without warranty. It is not a drop-in replacement for production pricing, risk-management, execution, or model-validation systems without independent validation and appropriate integration work.
