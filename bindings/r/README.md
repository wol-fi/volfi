# volfiR

Minimal R binding for the volfi v0.2.0 research reference implementation.

This binding follows the repository scope: it exposes the quantile-identity implementation and is not presented as a head-to-head performance comparison with other libraries.

The OTM path accepts non-negative absolute log-moneyness and normalized OTM prices inside the open unit interval. The normalized-call path handles calls across moneyness, including the forward strike.

Functions named `volfi_w` return total implied variance. Functions named `volfi_iv` return implied volatility. `volfi_iv_option()` accepts undiscounted forward-price calls and puts, while `volfi_w_otm_checked()` returns a machine-readable status for each quote.

```r
library(volfiR)

ctx <- volfi_ctx(0.1)
volfi_w_batch(ctx, c(0.001, 0.01, 0.1))

volfi_iv_option(
  forward = 100,
  strike = 105,
  price = 6.839446,
  t = 1.2,
  is_call = TRUE
)
```
