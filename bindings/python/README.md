# volfi Python binding

Minimal Python binding for the volfi research reference implementation.

This binding follows the repository scope: it exposes the quantile-identity implementation and is not presented as a head-to-head performance comparison with other libraries.

The OTM path is for positive moneyness and normalized OTM prices inside the open unit interval. The normalized-call path should be used at the forward strike.

Functions named `w` return total implied variance. Functions named `iv` return implied volatility.
