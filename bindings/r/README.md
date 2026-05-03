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

Generate 10,000 Black call prices on the random grid

```text
v ~ U(0.01, 2.0), Delta ~ U(0.01, 0.9)
```

then invert them and measure error and runtime with `bench`.

```r
install.packages("bench")
library(bench)

set.seed(1)

n <- 1e4
f <- rep(100, n)
d <- rep(1, n)
t <- rep(1, n)
v <- runif(n, 0.01, 2.0)
delta <- runif(n, 0.01, 0.9)
s <- v * sqrt(t)
k <- f * exp(0.5 * s^2 - qnorm(delta) * s)
p <- bs_call(f, k, d, t, v)

v_hat <- volfi_iv_call(f, k, d, t, p)
e <- v_hat - v

c(
  n = n,
  na = sum(is.na(v_hat)),
  nonfinite = sum(!is.finite(v_hat)),
  max_abs_error = max(abs(e)),
  mean_abs_error = mean(abs(e)),
  q99_abs_error = unname(quantile(abs(e), 0.99))
)

bm <- bench::mark(
  volfi_iv_call(f, k, d, t, p),
  iterations = 100,
  check = FALSE
)

ns_per_eval <- median(as.numeric(bm$time)) * 1e9 / n
c(ns_per_eval = ns_per_eval)
```

## OTM path

For repeated OTM inversions, precompute the OTM context once. Use OTM prices: calls for `K >= F`, puts transformed by put-call parity for `K < F`.

```r
h <- abs(log(k / f))
c_otm <- ifelse(k >= f, p, p + d * (k - f)) / (d * pmin(f, k))
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
