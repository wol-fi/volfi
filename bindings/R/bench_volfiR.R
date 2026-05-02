# volfiR benchmark
#
# Requirements:
#   - R >= 4.4
#   - a working C++ toolchain: Rtools 4.4 on Windows, Xcode Command Line Tools on macOS, or g++ on Linux
#   - packages: volfiR and bench
#
# Install volfiR from the local source package:
#   install.packages(".../volfiR_0.1.6.9000.tar.gz", repos = NULL, type = "source")
#
library(volfiR)
library(bench)

# Normalized OTM Black price as a function of log-moneyness h and total variance w.
b_otm <- function(h, w) {
  s <- sqrt(w)
  pnorm(-h / s + s / 2) - exp(h) * pnorm(-h / s - s / 2)
}

# Convert bench timings for a vector call into nanoseconds per single evaluation.
peval <- function(x, n) {
  z <- as.numeric(x) * 1e9 / n
  c(
    min = min(z),
    q25 = unname(quantile(z, 0.25)),
    median = median(z),
    mean = mean(z),
    q75 = unname(quantile(z, 0.75)),
    max = max(z)
  )
}

set.seed(321)

# Generate an OTM test grid by sampling delta and total volatility.
n <- 1e4
d <- runif(n, 0.5, 0.99)
v <- runif(n, 0.01, 2)
h <- abs(v * (0.5 * v - qnorm(d)))
w0 <- v^2
c <- b_otm(h, w0)
t <- 1

# Precompute all h-dependent quantities once; only the price c varies in the hot path.
ctx <- volfi_ctx(h)

# Accuracy check against the known input total variance w0 and volatility v.
w1 <- volfi_w(ctx, c)
iv1 <- volfi_iv(ctx, c, t)

print(summary(abs(w1 - w0)))
print(summary(abs(iv1 - v)))

# Benchmark only the precomputed implied-volatility path.
r <- bench::mark(
  iv_precomp = volfi_iv(ctx, c, t),
  iterations = 1000,
  check = FALSE
)

# Report timing statistics in nanoseconds per implied-vol evaluation.
out <- do.call(rbind, lapply(seq_len(nrow(r)), function(i) peval(r$time[[i]], n)))
out <- data.frame(
  method = as.character(r$expression),
  unit = "ns/eval",
  out,
  row.names = NULL
)

# Time in Nanoseconds: 
print(out, digits = 4)

