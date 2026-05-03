iv <- function(k, c) .Call(`_volfiR_iv_call_norm`, k, c)
vol <- function(k, c, t) sqrt(iv(k, c) / t)
iv_atm <- function(c) .Call(`_volfiR_iv_atm`, c)
black <- function(w, h) .Call(`_volfiR_black_otm`, w, h)
iv_otm <- function(h, c) .Call(`_volfiR_iv_otm`, h, c)
vol_otm <- function(h, c, t) .Call(`_volfiR_vol_otm`, h, c, t)
iv_call_norm <- iv
black_otm <- black
