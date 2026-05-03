# volfiR

Minimal Rcpp wrapper for `volfi` v0.1.7.

```r
install.packages("remotes")
remotes::install_github("wol-fi/volfi", subdir = "bindings/r")
```

```r
library(volfiR)
h <- abs(log(K / F))
ctx <- volfi_ctx(h)
w <- volfi_w(ctx, c_otm)
iv <- volfi_iv(ctx, c_otm, T)
```

Inputs for the precomputed OTM path use

```text
h = abs(log(K / F))
c_otm in (0, 1)
w = sigma^2 * T
```

The normalized call helpers are also exposed:

```r
w <- volfi_w_call_norm(log(K / F), c)
iv <- volfi_iv_call_norm(log(K / F), c, T)
iv2 <- volfi_iv_call(F, K, D, T, price)
```

Use `volfi_ctx()` once and benchmark `volfi_iv(ctx, c_otm, T)` for the precomputed OTM path.
