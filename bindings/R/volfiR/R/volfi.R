volfi_ctx <- function(h) .Call("_volfiR_volfi_ctx", as.numeric(h), PACKAGE = "volfiR")
volfi_ctx_size <- function(ctx) .Call("_volfiR_volfi_ctx_size", ctx, PACKAGE = "volfiR")
volfi_w <- function(ctx, c) .Call("_volfiR_volfi_w", ctx, as.numeric(c), PACKAGE = "volfiR")
volfi_iv <- function(ctx, c, t) .Call("_volfiR_volfi_iv", ctx, as.numeric(c), as.numeric(t), PACKAGE = "volfiR")
volfi_w_otm <- function(h, c) .Call("_volfiR_volfi_w_otm", as.numeric(h), as.numeric(c), PACKAGE = "volfiR")
volfi_iv_otm <- function(h, c, t) .Call("_volfiR_volfi_iv_otm", as.numeric(h), as.numeric(c), as.numeric(t), PACKAGE = "volfiR")
volfi_version <- function() .Call("_volfiR_volfi_version", PACKAGE = "volfiR")
print.volfi_ctx <- function(x, ...) {
  cat("<volfi_ctx: ", volfi_ctx_size(x), ">"); cat("\n")
  invisible(x)
}
