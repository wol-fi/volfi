# Demo for the volfiR package.
#
# Install the package from the repository with:
# remotes::install_github("wol-fi/volfi", subdir = "bindings/r", ref = "main", force = TRUE, upgrade = "never")
#
# The public R API is:
#   iv(k, c)       total implied variance w
#   vol(k, c, t)   implied volatility sqrt(w / t)
#   iv_atm(c)      ATM total implied variance
#   black(w, h)    normalized OTM Black call price, h = abs(k)

library(volfiR)

if (!requireNamespace("bench", quietly = TRUE)) {
  stop("Package 'bench' is required for this demo. Install it with install.packages('bench').")
}

# Normalized Black call price C/F as a function of signed log-moneyness k = log(K/F)
# and total variance w. Positive k is OTM call; negative k is ITM call.
black_call <- function(k, w) {
  s <- sqrt(w)
  pnorm(-k / s + s / 2) - exp(k) * pnorm(-k / s - s / 2)
}

# Convert a normalized ITM call price to the corresponding normalized OTM-side price.
# For k >= 0 the call is already OTM. For k < 0 use the exact projection used by volfi.
otm_projection <- function(k, c) {
  ifelse(k >= 0, c, 1 + exp(-k) * (c - 1))
}

# Convert bench timing output to nanoseconds per scalar evaluation.
ns_eval <- function(x, n) {
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

# Random test grid.
# d spans the wide delta-style range used in the C++ tests.
# sign_k creates both OTM calls (k > 0) and ITM calls (k < 0).
n <- 1e4
d <- runif(n, 0.01, 0.90)
v <- runif(n, 0.01, 2.00)
sign_k <- sample(c(-1, 1), n, replace = TRUE)
h <- abs(v * (0.5 * v - qnorm(d)))
k <- sign_k * h
w0 <- v^2
c <- black_call(k, w0)
t <- 1

# Invert normalized call prices back to total variance and volatility.
w1 <- iv(k, c)
v1 <- vol(k, c, t)

# Check pricing consistency on the OTM side.
c_otm <- otm_projection(k, c)
c1 <- black(w1, abs(k))

print(summary(abs(w1 - w0)))
print(summary(abs(v1 - v)))
print(summary(abs(c1 - c_otm)))

# Microbenchmark vectorized calls. Increase iterations for more stable timing.
r <- bench::mark(
  iv = iv(k, c),
  vol = vol(k, c, t),
  black = black(w0, abs(k)),
  iterations = 1000,
  check = FALSE
)

out <- do.call(rbind, lapply(seq_len(nrow(r)), function(i) ns_eval(r$time[[i]], n)))
out <- data.frame(
  method = as.character(r$expression),
  unit = "ns/eval",
  out,
  row.names = NULL
)

print(out, digits = 4)
