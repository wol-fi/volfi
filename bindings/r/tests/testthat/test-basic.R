test_that("basic volfi calls return finite values", {
 k <- log(1.2)
 c <- 0.05
 w <- iv(k, c)
 expect_true(is.finite(w))
 expect_true(w > 0)
 expect_true(is.finite(vol(k, c, 1)))
 expect_true(is.finite(iv_atm(0.1)))
})

test_that("otm aliases remain available", {
 h <- 0.2
 c <- 0.05
 w <- iv_otm(h, c)
 expect_true(is.finite(w))
 expect_true(w > 0)
 expect_equal(black_otm(w, h), c, tolerance = 1e-12)
})
