import time
from math import erf, sqrt
from statistics import NormalDist

import numpy as np
import volfi
from py_vollib.black.implied_volatility import implied_volatility as vollib_LBR_iv


def cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(f, k, d, t, v):
    s = v * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * s * s) / s
    d2 = d1 - s
    return d * (f * np.vectorize(cdf)(d1) - k * np.vectorize(cdf)(d2))


def med_ns(fn, n, runs=100):
    x = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        x.append((time.perf_counter_ns() - t0) / n)
    return float(np.median(x))


def main():
    n = 10_000
    rng = np.random.default_rng(1)
    f = np.full(n, 100.0)
    d = np.ones(n)
    r = np.zeros(n)
    t = np.ones(n)
    v = rng.uniform(0.01, 2.0, n)
    delta = rng.uniform(0.01, 0.9, n)
    q = np.vectorize(NormalDist().inv_cdf)(delta)
    s = v * np.sqrt(t)
    k = f * np.exp(0.5 * s * s - q * s)
    p = bs_call(f, k, d, t, v)

    vv = volfi.iv_call(f, k, d, t, p)
    vl = np.array([vollib_LBR_iv(float(p[i]), float(f[i]), float(k[i]), float(r[i]), float(t[i]), "c") for i in range(n)])

    ev = vv - v
    el = vl - v

    print({
        "n": n,
        "volfi_nan": int(np.isnan(vv).sum()),
        "vollib_LBR_nan": int(np.isnan(vl).sum()),
        "volfi_max_abs_error": float(np.max(np.abs(ev))),
        "vollib_LBR_max_abs_error": float(np.max(np.abs(el))),
        "volfi_mean_abs_error": float(np.mean(np.abs(ev))),
        "vollib_LBR_mean_abs_error": float(np.mean(np.abs(el))),
        "volfi_q99_abs_error": float(np.quantile(np.abs(ev), 0.99)),
        "vollib_LBR_q99_abs_error": float(np.quantile(np.abs(el), 0.99)),
    })

    volfi_ns = med_ns(lambda: volfi.iv_call(f, k, d, t, p), n)
    vollib_LBR_ns = med_ns(lambda: [vollib_LBR_iv(float(p[i]), float(f[i]), float(k[i]), 0.0, float(t[i]), "c") for i in range(n)], n)

    print({
        "volfi_median_ns_per_eval": volfi_ns,
        "vollib_LBR_median_ns_per_eval": vollib_LBR_ns,
        "speedup": vollib_LBR_ns / volfi_ns,
    })


if __name__ == "__main__":
    main()
