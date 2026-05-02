\name{volfiR}
\alias{volfiR}
\alias{volfi_ctx}
\alias{volfi_ctx_size}
\alias{volfi_w}
\alias{volfi_iv}
\alias{volfi_w_otm}
\alias{volfi_iv_otm}
\alias{volfi_version}
\title{Rcpp wrapper for volfi v0.1.6}
\usage{
volfi_ctx(h)
volfi_ctx_size(ctx)
volfi_w(ctx, c)
volfi_iv(ctx, c, t)
volfi_w_otm(h, c)
volfi_iv_otm(h, c, t)
volfi_version()
}
\arguments{
\item{h}{Non-negative normalized log-moneyness, \code{abs(log(K / F))}.}
\item{ctx}{External pointer returned by \code{volfi_ctx}.}
\item{c}{Normalized OTM price in \code{(0, 1)}.}
\item{t}{Positive maturity or total variance divisor.}
}
\description{
Minimal Rcpp wrapper around the precomputed OTM implied variance and implied volatility functions in volfi v0.1.6.
}
\value{
\code{volfi_ctx} returns an external pointer. \code{volfi_w} and \code{volfi_w_otm} return total variance. \code{volfi_iv} and \code{volfi_iv_otm} return implied volatility.
}
\examples{
h <- c(0.1, 0.2)
cp <- c(0.01, 0.02)
tt <- c(1, 1)
ctx <- volfi_ctx(h)
volfi_w(ctx, cp)
volfi_iv(ctx, cp, tt)
}
