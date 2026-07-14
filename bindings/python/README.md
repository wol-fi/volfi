# volfi Python binding (v0.2.0)

NumPy binding for the volfi v0.2.0 routed, machine-precision Black-Scholes implied-volatility
inverter. Array inputs are inverted through the vectorized batch driver; the result is
bit-identical to per-quote inversion and accurate to the last few ULP.

## Install

From this directory (a C++17 compiler is required):

```bash
pip install .
```

The default build tunes to the build machine (`-march=native`) and compiles with
`-ffp-contract=off`, which is what makes the scalar and SIMD paths agree bit-for-bit. For a
portable, SIMD-free build set `VOLFI_MARCH=` (empty); for an AVX2 wheel set
`VOLFI_MARCH=x86-64-v3`.

## Conventions

| symbol | meaning |
|--------|---------|
| `h`    | `\|log(K/F)\|`, absolute log-moneyness, ≥ 0 |
| `c`    | `C/F`, undiscounted OTM-call price over the forward, in (0, 1) |
| `w`    | `v² = (σ√T)²`, total implied variance (what the core returns) |

Every function takes scalars or NumPy arrays. Scalars return a `float`, arrays a `float64`
`ndarray`; a length-1 input broadcasts against a length-n one.

## API

```python
import numpy as np
import volfi

# total implied variance and volatility
w     = volfi.implied_variance(h, c)              # w = v²
sigma = volfi.implied_volatility(h, c, T)         # σ = sqrt(w / T)

# vectorized over a whole book (SIMD batch driver, bit-identical to scalar)
w = volfi.implied_variance(h_array, c_array)

# from raw option data: ITM options and puts are put-call-parity-projected to the OTM twin
sigma = volfi.implied_volatility_from_option(F, K, price, T, is_call=False)

# checked: never raises; returns (w, status) against the iv_status enum
w, status = volfi.implied_variance_checked(h, c)
#   volfi.iv_status.{ok, below_intrinsic, above_max, bad_input, out_of_domain, near_saturation}

# streaming warm restart from the previous snapshot's variance (few-ULP accuracy contract)
w = volfi.implied_variance_warm(h, c, w_prev, steps=2)

volfi.version()   # "0.2.0"
```

The plain `implied_variance` / `implied_volatility` raise `RuntimeError` on out-of-range input
(`h ≤ 0`, `c ∉ (0,1)`); use `implied_variance_checked` for the non-raising, status-returning
form. For `v = σ√T > 8` the accuracy floor is set by the double-precision representation of
`1 − c` and the result is flagged `near_saturation` — a limit intrinsic to a double price
input, shared by any inverter.

## Test

```bash
pip install pybind11 numpy
python tests/test_volfi.py      # or: pytest tests/
```

The tests check accuracy against an independent 50-digit mpmath oracle, the bit-identical
batch-vs-scalar invariant, put-call-parity agreement, the warm restart, and the status codes.
