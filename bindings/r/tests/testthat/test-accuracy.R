test_that("Annulus reference values match", {
  h <- c(0, 0.01, 0.1, 0.5, 1, 4)
  cp <- c(0.1, 0.01, 0.001, 0.05, 0.2, 1e-8)
  tt <- c(1, 0.5, 2, 1.25, 0.75, 3)
  ww <- c(
    0.063163096373724886, 0.0013032051608100842,
    0.0033012778463021642, 0.24189968319602104,
    1.4308324613409953, 0.51343900206561777
  )
  ctx <- volfi_ctx(h)
  expect_equal(volfi_w(ctx, cp), ww, tolerance = 1e-14)
  expect_equal(volfi_iv(ctx, cp, tt), sqrt(ww / tt), tolerance = 1e-14)
  expect_equal(volfi_w_otm(h, cp), ww, tolerance = 1e-14)
  expect_equal(volfi_iv_otm(h, cp, tt), sqrt(ww / tt), tolerance = 1e-14)
  expect_equal(volfi_version(), "0.2.0")
})

test_that("scalar contexts use the native batch engine", {
  h <- 0.1
  cp <- c(0.001, 0.01, 0.1)
  ctx <- volfi_ctx(h)
  expect_equal(volfi_w_batch(ctx, cp), volfi_w(ctx, cp), tolerance = 0)
  expect_error(volfi_w_batch(volfi_ctx(c(0.1, 0.2)), cp), "scalar context")
})

test_that("normalized call helpers and checked OTM statuses work", {
  expect_equal(volfi_w_call_norm(0.1, 0.01), 0.010983412121237268, tolerance = 1e-14)
  expect_equal(volfi_iv_call_norm(0.1, 0.01, 2), sqrt(0.010983412121237268 / 2), tolerance = 1e-14)
  checked <- volfi_w_otm_checked(c(0.1, -1, 0.1), c(0.01, 0.01, 1))
  expect_equal(checked$status, c("ok", "bad_input", "above_max"))
  expect_true(is.finite(checked$variance[1]))
  expect_true(is.nan(checked$variance[2]))
  expect_true(is.nan(checked$variance[3]))
})

test_that("forward-price call and put interfaces round trip", {
  forward <- 100
  strike <- c(105, 95)
  maturity <- 1.2
  sigma <- c(0.2, 0.35)
  s <- sigma * sqrt(maturity)
  d1 <- (log(forward / strike) + 0.5 * s^2) / s
  d2 <- d1 - s
  call <- forward * pnorm(d1) - strike * pnorm(d2)
  put <- call - forward + strike
  expect_equal(volfi_iv_option(forward, strike[1], call[1], maturity, TRUE), sigma[1], tolerance = 1e-12)
  expect_equal(volfi_iv_option(forward, strike[2], put[2], maturity, FALSE), sigma[2], tolerance = 1e-12)
  expect_equal(volfi_iv_call(forward, strike[1], 0.98, maturity, call[1] * 0.98), sigma[1], tolerance = 1e-12)
})
