# volfiR

`volfiR` is the R translation/package layer for the flagship C++ `volfi` kernel. It exposes the precomputed out-of-the-money implied-variance and implied-volatility path through a small `Rcpp` interface.

## What It Wraps

The package targets the projected OTM convention used by the C++ core:

```text
h = abs(log(K / F))
c_otm in (0, 1)
w = sigma^2 * T
```

For repeated inversion on a fixed strike or moneyness grid, create the context once and reuse it:

```r
library(volfiR)

h <- abs(log(K / F))
ctx <- volfi_ctx(h)
w <- volfi_w(ctx, c_otm)
iv <- volfi_iv(ctx, c_otm, T)
```

Direct non-precomputed helpers are also available:

```r
w  <- volfi_w_otm(h, c_otm)
iv <- volfi_iv_otm(h, c_otm, T)
```

## Package API

- `volfi_ctx(h)`: precompute one or many OTM contexts
- `volfi_ctx_size(ctx)`: inspect the context length
- `volfi_w(ctx, c)`: implied total variance from a precomputed context
- `volfi_iv(ctx, c, t)`: implied volatility from a precomputed context
- `volfi_w_otm(h, c)`: direct implied total variance without precomputation
- `volfi_iv_otm(h, c, t)`: direct implied volatility without precomputation
- `volfi_version()`: report the wrapped core version

All numeric inputs are vectorized with standard R recycling rules for scalar-or-equal-length arguments.

## Installation

Requirements:

- `R`
- package `Rcpp`
- a working C++17 toolchain

Install from the unpacked repo directory:

```r
install.packages("Rcpp")
install.packages("bindings/R/volfiR", repos = NULL, type = "source")
```

Or build and install with `devtools`:

```r
devtools::install("bindings/R/volfiR", upgrade = "never")
```

## Validation And Benchmarking

Package tests live in `tests/testthat/`.

A simple local benchmark script is available one level above the package at:

```text
bindings/R/bench_volfiR.R
```

The package also ships an internal benchmark helper at:

```text
inst/bench/bench_volfiR.R
```

## Relation To The C++ Core

The C++ implementation in the repo root remains canonical. This package is intended to make the same precomputed OTM path available from R without changing the core algorithm or repository focus.

## License

`volfiR` is distributed under the MIT license, consistent with the main repository.
