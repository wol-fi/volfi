# fastvollib_feed.py -- score and time fast-vollib's vectorized Jaeckel path (arXiv:2604.27210) on the
# manuscript's 30,000-quote feed, against the whole-book reference (market_feed_ref.txt written by
# gate/dump_feed_ref.cpp).  Normalized inputs: F=1, K=e^h, t=1, r=0, call, so sigma = sqrt(v).
#   python3 gate/fastvollib_feed.py bench_run/market_feed_ref.txt
import sys, time, numpy as np, fast_vollib
from fast_vollib import jackel as J
fn = sys.argv[1] if len(sys.argv) > 1 else "market_feed_ref.txt"
D = np.loadtxt(fn); h, c, vref, reg = D[:, 0], D[:, 1], D[:, 2], D[:, 3].astype(int)
n = len(h); K = np.exp(h); F = np.ones(n); t = np.ones(n); r = np.zeros(n)
print("fast-vollib", getattr(fast_vollib, "__version__", "?"), "quotes", n)
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("numba: none", e)
names = ["Wing", "Near", "Far", "Upper", "n/a"]
def score(tag, sig):
    rel = np.abs(sig - vref) / vref; bad = ~np.isfinite(sig)
    print(f"  {tag:34s} max rel {np.nanmax(rel):.2e} at h={h[np.nanargmax(rel)]:.3e} c={c[np.nanargmax(rel)]:.3e}  mean {np.nanmean(rel):.2e}  nonfinite {bad.sum()}")
    for k in range(4):
        m = reg == k
        if m.any(): print(f"     {names[k]:6s} n={m.sum():6d} max {np.nanmax(rel[m]):.2e} mean {np.nanmean(rel[m]):.2e}")
def timeit(tag, f, reps=7):
    f(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    ts.sort(); print(f"  {tag:34s} median {ts[len(ts)//2]/n*1e9:8.1f} ns/quote   best {ts[0]/n*1e9:8.1f}   (reps {reps}, batch {n})")
# Jaeckel paths: normalized (beta = price/sqrt(F K) = c e^{-h/2}, x = log(F/K) = -h, sigma*sqrt(t) = s) and Black-76 wrapper
sig_n = np.asarray(J.jackel_iv_normalized(c * np.exp(-h / 2), -h), dtype=float)
score("jackel_iv_normalized (numba)", sig_n)
timeit("jackel_iv_normalized (numba)", lambda: J.jackel_iv_normalized(c * np.exp(-h / 2), -h))
sig_b = np.asarray(J.jackel_iv_black(c, 1.0, K, 1.0, True), dtype=float)
score("jackel_iv_black (numba)", sig_b)
timeit("jackel_iv_black (numba)", lambda: J.jackel_iv_black(c, 1.0, K, 1.0, True))
# default public API (Halley x8 path), for context
sig_h = np.asarray(fast_vollib.fast_implied_volatility(c, F, K, t, 0.0, "c", return_as="numpy", backend="numpy"), dtype=float)
score("fast_implied_volatility (Halley)", sig_h)
timeit("fast_implied_volatility (Halley)", lambda: fast_vollib.fast_implied_volatility(c, F, K, t, 0.0, "c", return_as="numpy", backend="numpy"))
