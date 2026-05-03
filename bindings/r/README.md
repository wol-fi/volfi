# volfiR

Minimal Rcpp wrapper for `volfi` v0.1.7.

```r
install.packages("remotes")
remotes::install_github("wol-fi/volfi", subdir = "bindings/r")
```

## Call-price roundtrip

Create a Black call price from known volatility and invert it with `volfi_iv_call()`.

```r
library(volfiR)

bs_call <- function(f, k, d, t, v) {
  s <- v * sqrt(t)
  d1 <- (log(f / k) + 0.5 * s^2) / s
  d2 <- d1 - s
  d * (f * pnorm(d1) - k * pnorm(d2))
}

f <- 100
k <- 105
d <- 0.98
t <- 1.2
v <- 0.2
p <- bs_call(f, k, d, t, v)
v_hat <- volfi_iv_call(f, k, d, t, p)

c(price = p, true_vol = v, estimated_vol = v_hat, error = v_hat - v)
```

## Accuracy and speed on 10,000 calls

Generate 10,000 Black call prices with known volatility, invert them, and measure error and runtime.

```r
set.seed(1)

n <- 1e4
f <- rep(100, n)
k <- exp(runif(n, log(70), log(140)))
d <- rep(0.98, n)
t <- runif(n, 0.05, 3)
v <- runif(n, 0.05, 1)
p <- bs_call(f, k, d, t, v)

tm <- system.time(v_hat <- volfi_iv_call(f, k, d, t, p))
e <- v_hat - v

c(
  n = n,
  elapsed_sec = unname(tm["elapsed"]),
  max_abs_error = max(abs(e)),
  mean_abs_error = mean(abs(e)),
  q99_abs_error = unname(quantile(abs(e), 0.99))
)
```

## OTM path

For repeated OTM inversions, precompute the OTM context once.

```r
h <- abs(log(k / f))
c_otm <- p / (d * f)
ctx <- volfi_ctx(h)
w <- volfi_w(ctx, c_otm)
iv <- volfi_iv(ctx, c_otm, t)
```

Inputs for the precomputed OTM path use

```text
h = abs(log(K / F))
c_otm in (0, 1)
w = sigma^2 * T
```
