# volfi v0.1.8

`volfi` is a research reference implementation for the Black-Scholes implied-variance quantile identity.

The repository is intended to make the quantile representation concrete and reproducible. It is not presented as a like-for-like performance comparison with other implementations.

## Identity

For out-of-the-money normalized calls with forward log-moneyness `k > 0`, normalized call price `c`, and total implied volatility `v = sigma sqrt(T)`, the implemented representation is

```math
v(k,c)^2 = \mathcal{F}^{-1}_{GIG}\left(c; \frac{1}{2}, \frac{1}{4}, k^2\right), \qquad k > 0.
```

At the forward strike the inversion reduces to

```math
v = 2\Phi^{-1}\left(\frac{1+c}{2}\right), \qquad w = v^2.
```

## Build

Use `make`, `make test`, and `make bench` from the repository root.

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
- Invalid domains return `NaN` by default. Compile with `VOLFI_STRICT_DOMAIN` to throw `std::domain_error` instead.
- The OTM API is not the ATM API. Use the normalized-call API at the forward strike.

## Benchmarks

The included benchmark is a same-codebase microbenchmark for repeated inversions on fixed generated grids. It is intended for local regression checks and implementation diagnostics only.

It is not a like-for-like comparison with other libraries, and it should not be used to support a general speed claim.

## Caveats

This is a research reference implementation. The supported domain is narrower than full industrial implied-volatility libraries. Empirical behavior is domain-specific and should not be generalized outside supported regimes without independent testing.

## License

`volfi` is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full license terms.
