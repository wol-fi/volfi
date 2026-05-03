test_that("basic volfi calls return finite values", {
 h <- 0.2
 c <- 0.05
 w <- iv_otm(h, c)
 expect_true(is.finite(w))
 expect_true(w > 0)
 expect_equal(black_otm(w, h), c, tolerance = 1e-12)
 expect_true(is.finite(vol_otm(h, c, 1)))
 expect_true(is.finite(iv_atm(0.1)))
})
