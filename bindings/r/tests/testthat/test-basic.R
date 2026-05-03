test_that("basic volfi calls return finite values", {
 k <- log(1.2)
 c <- 0.05
 w <- iv(k, c)
 expect_true(is.finite(w))
 expect_true(w > 0)
 expect_true(is.finite(vol(k, c, 1)))
 expect_true(is.finite(iv_atm(0.1)))
 expect_equal(black(w, abs(k)), c, tolerance = 1e-10)
})

test_that("otm names are not exported", {
 expect_false("iv_otm" %in% getNamespaceExports("volfiR"))
 expect_false("vol_otm" %in% getNamespaceExports("volfiR"))
})
