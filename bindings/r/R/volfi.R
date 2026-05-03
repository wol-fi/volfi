iv <- function(k, c) .Call(`_volfiR_iv`, k, c)
vol <- function(k, c, t) .Call(`_volfiR_vol`, k, c, t)
iv_atm <- function(c) .Call(`_volfiR_iv_atm`, c)
black <- function(w, h) .Call(`_volfiR_black`, w, h)
