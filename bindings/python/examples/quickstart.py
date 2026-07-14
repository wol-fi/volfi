"""volfi v0.2.0 quickstart: invert a small option book to implied volatility."""
import numpy as np

import volfi

# A tiny book at one expiry: forward F, strikes K, undiscounted mid prices, all calls.
F, T = 100.0, 0.5
K = np.array([90.0, 100.0, 110.0, 125.0])
price = np.array([11.20, 4.15, 1.02, 0.14])
is_call = True

sigma = volfi.implied_volatility_from_option(F, K, price, T, is_call)
for k, p, s in zip(K, price, sigma):
    print(f"K={k:6.1f}  price={p:6.2f}  implied vol={s:7.4f}")

# The normalized core, vectorized over the whole book at once:
h = np.abs(np.log(K / F))
c = price / F                      # undiscounted OTM-call price / F (calls are OTM for K > F)
w = volfi.implied_variance(h[K > F], c[K > F])
print("\nOTM-call total variances:", w)

print("\nvolfi version:", volfi.version())
