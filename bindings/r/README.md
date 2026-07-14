# volfiR

R (Rcpp) binding for the volfi v0.2.0 routed, machine-precision Black-Scholes
implied-volatility inverter. It wraps the same C++17 engine as the rest of the repository,
built with `-ffp-contract=off` so results are bit-identical to the scalar reference and
accurate to the last few ULP.

The engine headers are vendored under `src/volfi_annulus/` (an R package must be
self-contained); they are kept byte-identical to `include/volfi/`.

## Install

From this directory, with a C++17 toolchain and the `Rcpp` package:

```r
install.packages("Rcpp")
install.packages(".", repos = NULL, type = "source")   # or: R CMD INSTALL .
```

## Conventions

- `h = abs(log(K / F))`  — absolute log-moneyness, ≥ 0
- `c = C / F`            — undiscounted OTM-call price over the forward, in (0, 1)
- `w = v² = (σ√T)²`      — total implied variance (what the `volfi_w*` functions return)

Functions named `volfi_w*` return total variance; `volfi_iv*` return volatility. Every
function is vectorized (a length-1 argument broadcasts against a length-n one).

## Usage

```r
library(volfiR)

# normalized OTM inversion
volfi_w_otm(h = 1.0, c = 0.02)                 # total variance w
volfi_iv_otm(h = 1.0, c = 0.02, t = 1.0)       # volatility sigma

# a reusable per-h context, then the native batch engine over many prices
ctx <- volfi_ctx(0.1)
volfi_w_batch(ctx, c(0.001, 0.01, 0.1))

# from raw option data: ITM options and puts are put-call-parity-projected to the OTM twin
volfi_iv_option(forward = 100, strike = 90, price = 12.5, t = 1.0, is_call = TRUE)
volfi_iv_option(forward = 100, strike = 110, price = 4.2, t = 1.0, is_call = FALSE)

# checked inversion: a machine-readable status per quote (never errors on bad input)
volfi_w_otm_checked(h = c(0.1, -1, 0.1), c = c(0.01, 0.01, 1))
#   $status: "ok", "bad_input", "above_max"; $variance carries NaN where not ok

volfi_version()   # "0.2.0"
```

Status codes returned by `volfi_w_otm_checked()` are `ok`, `below_intrinsic`, `above_max`,
`bad_input`, `out_of_domain`, and `near_saturation`. For `v = σ√T > 8` the accuracy floor is
set by the double-precision representation of `1 − c` (`near_saturation`), a limit intrinsic
to a double-precision price input and shared by any inverter.

## Test

```r
# install.packages("testthat")
testthat::test_dir("tests/testthat")
```

The tests check accuracy against reference values, the batch-equals-scalar invariant, the
status codes, and put-call-parity round-trips on both sides of the money.

## License

BSD 3-Clause (see `LICENSE`), matching the parent repository.
