test_that("reference values match", {
  h <- c(0, 0.01, 0.1, 0.5, 1)
  cp <- c(0.1, 0.01, 0.001, 0.05, 0.2)
  tt <- c(1, 0.5, 2, 1.25, 0.75)
  ww <- c(0.063163096373724928, 0.0013032051608100883, 0.0033012778463021937, 0.24189968319602106, 1.4308324613409955)
  vv <- c(0.25132269371014815, 0.051053014814212261, 0.040628055862311413, 0.43990879345248018, 1.381222869460728)
  ctx <- volfi_ctx(h)
  expect_equal(volfi_w(ctx, cp), ww, tolerance = 1e-14)
  expect_equal(volfi_iv(ctx, cp, tt), vv, tolerance = 1e-14)
  expect_equal(volfi_w_otm(h, cp), ww, tolerance = 1e-14)
  expect_equal(volfi_iv_otm(h, cp, tt), vv, tolerance = 1e-14)
})

test_that("scalar context is vectorized over prices", {
  h <- 0.1
  cp <- c(0.001, 0.01, 0.1)
  ctx <- volfi_ctx(h)
  expect_equal(volfi_w(ctx, cp), volfi_w_otm(h, cp), tolerance = 1e-14)
})

test_that("wing and call helpers match references", {
  expect_equal(volfi_w_otm(1, 0.01), 0.28357398093281544, tolerance = 1e-14)
  expect_equal(volfi_iv_otm(1, 0.01, 2), 0.37654613325116981, tolerance = 1e-14)
  expect_equal(volfi_w_call_norm(0.1, 0.01), 0.010983412121237294, tolerance = 1e-14)
  expect_equal(volfi_w_call_norm(-0.1, 0.1), 0.0072973603266672377, tolerance = 1e-14)
  expect_equal(volfi_iv_call(100, 105, 0.98, 1.2, 7), 0.21108908075172517, tolerance = 1e-14)
  expect_equal(volfi_version(), "0.1.7")
})
