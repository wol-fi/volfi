\name{volfiR}
\alias{volfiR}
\alias{volfi_ctx}
\alias{volfi_ctx_size}
\alias{volfi_w}
\alias{volfi_iv}
\alias{volfi_w_book}
\alias{volfi_iv_book}
\alias{volfi_w_otm}
\alias{volfi_iv_otm}
\alias{volfi_w_call_norm}
\alias{volfi_iv_call_norm}
\alias{volfi_iv_call}
\alias{volfi_w_batch}
\alias{volfi_w_otm_checked}
\alias{volfi_iv_option}
\alias{volfi_version}
\title{Rcpp wrapper for volfi Annulus v0.2.4}
\usage{
volfi_ctx(h)
volfi_ctx_size(ctx)
volfi_w(ctx, c)
volfi_iv(ctx, c, t)
volfi_w_book(h, c)
volfi_iv_book(h, c, t)
volfi_w_otm(h, c)
volfi_iv_otm(h, c, t)
volfi_w_call_norm(k, c)
volfi_iv_call_norm(k, c, t)
volfi_iv_call(f, k, d, t, price)
volfi_w_batch(ctx, c)
volfi_w_otm_checked(h, c)
volfi_iv_option(forward, strike, price, t, is_call = TRUE)
volfi_version()
}
\arguments{
\item{h}{Non-negative normalized log-moneyness, \code{abs(log(K / F))}.}
\item{ctx}{External pointer returned by \code{volfi_ctx}.}
\item{c}{Normalized OTM or call price in \code{(0, 1)}.}
\item{t}{Positive maturity or total variance divisor.}
\item{k}{Normalized log-moneyness, \code{log(K / F)}, for the normalized call helpers.}
\item{f}{Positive forward.}
\item{d}{Positive discount factor.}
\item{price}{Positive undiscounted call price input for \code{volfi_iv_call}.}
\item{forward}{Positive undiscounted forward price.}
\item{strike}{Positive strike.}
\item{is_call}{Logical flag selecting a call or put for \code{volfi_iv_option}.}
}
\description{
Rcpp wrapper around volfi v0.3.1. \code{volfi_w_book} and \code{volfi_iv_book} are the book kernel: one straight-line evaluation with exact rational rows that answers all but a few quotes in ten thousand of a traded book itself and hands the rest to the routed charts (\code{region} 1 = raw rows, 2 = conformal rows, 0 = routed). The remaining functions are the routed four-chart inverter: the OTM path, normalized calls, and raw undiscounted forward-price call or put quotes.
}
\value{
\code{volfi_w_book} returns a list with \code{variance} and the integer \code{region} per quote. \code{volfi_ctx} returns an external pointer. Variance functions return total variance. Volatility functions return implied volatility. \code{volfi_w_otm_checked} returns a list containing variance and a status string for each quote.
}
\examples{
h <- c(0.1, 0.2)
cp <- c(0.01, 0.02)
tt <- c(1, 1)
volfi_w_book(h, cp)
volfi_iv_book(h, cp, tt)
ctx <- volfi_ctx(h)
volfi_w(ctx, cp)
volfi_iv(ctx, cp, tt)
volfi_w_call_norm(c(0.1, -0.1), c(0.01, 0.1))
}
