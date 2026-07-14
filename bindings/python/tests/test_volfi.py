"""Tests for the volfi v0.2.0 Python binding.

Runnable with pytest, or directly: ``python3 test_volfi.py`` (prints PASS/FAIL, exit code).
Reference (h, c) -> w triples were generated independently at 50 digits with mpmath.
"""
import math

import numpy as np

import volfi

# (h, c, w) with c and w computed at 50 digits from a chosen (h, sigma, T); spans all charts.
REFERENCE = [
    (1.0,  0.0012998981331268078984,   0.16),      # central
    (0.05, 0.058592868120983829202,    0.04),      # small-moneyness (left)
    (0.5,  0.82999580994769030778,     9.0),       # large-volatility (right)
    (8.0,  1.0486591789128747036e-57,  0.25),      # deep wing (c ~ 1e-57)
    (2.0,  0.11452457401399357173,     2.0),       # central/right
    (0.2,  0.0013435555142658678935,   0.01125),   # small-moneyness
]


def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _otm_call_c(h, v):
    """Normalized OTM-call price c = C/F for log-moneyness h and total vol v (double precision)."""
    return _phi(v / 2 - h / v) - math.exp(h) * _phi(-v / 2 - h / v)


def test_reference_values():
    h = np.array([r[0] for r in REFERENCE])
    c = np.array([r[1] for r in REFERENCE])
    w = np.array([r[2] for r in REFERENCE])
    got = volfi.implied_variance(h, c)
    assert np.allclose(got, w, rtol=1e-12, atol=0), np.abs(got / w - 1).max()


def test_scalar_returns_float():
    r = volfi.implied_variance(1.0, 0.0012998981331268078984)
    assert isinstance(r, float)
    assert abs(r / 0.16 - 1) < 1e-12
    v = volfi.implied_volatility(1.0, 0.0012998981331268078984, 1.0)
    assert isinstance(v, float)
    assert abs(v / 0.4 - 1) < 1e-12  # w=0.16, T=1 -> sigma=0.4


def test_broadcasting_and_shape():
    c = np.array([0.001, 0.05, 0.2])
    out = volfi.implied_variance(1.0, c)  # scalar h broadcasts
    assert isinstance(out, np.ndarray) and out.shape == (3,)


def test_volatility_is_sqrt_w_over_t():
    h, c = 1.0, 0.0012998981331268078984
    w = volfi.implied_variance(h, c)
    for T in (0.25, 1.0, 3.0):
        s = volfi.implied_volatility(h, c, T)
        assert abs(s - math.sqrt(w / T)) < 1e-15


def test_batch_equals_scalar_bit_identical():
    # The core invariant: the vectorized batch path is bit-for-bit identical to per-quote
    # inversion. Generate valid (h, c) directly -- any c in (0,1) is feasible for h > 0.
    rng = np.random.default_rng(20260714)
    n = 4000
    h = rng.uniform(0.01, 15.0, n)
    c = rng.uniform(0.005, 0.995, n)
    batch = volfi.implied_variance(h, c)
    single = np.array([volfi.implied_variance(h[i], c[i]) for i in range(n)])
    assert batch.tobytes() == single.tobytes(), "batch != scalar (bit level)"


def test_checked_status():
    w, st = volfi.implied_variance_checked(1.0, 0.5)
    assert st == volfi.iv_status.ok and math.isfinite(w)
    _, st = volfi.implied_variance_checked(1.0, 1.5)   # c >= 1
    assert st == volfi.iv_status.above_max
    _, st = volfi.implied_variance_checked(1.0, -0.1)  # c <= 0
    assert st == volfi.iv_status.below_intrinsic
    _, st = volfi.implied_variance_checked(-1.0, 0.5)  # h < 0
    assert st == volfi.iv_status.bad_input
    # array form returns integer codes matching the enum
    w, st = volfi.implied_variance_checked([1.0, 1.0], [0.5, 1.5])
    assert list(st) == [int(volfi.iv_status.ok), int(volfi.iv_status.above_max)]


def test_parity_call_put_agree():
    # Both sides of the money: K > F (OTM call / ITM put) and K < F (ITM call / OTM put).
    # The K < F branch is the one that exercises the OTM-twin normalization by K.
    F, T = 100.0, 1.0
    for K, sigma in [(110.0, 0.30), (90.0, 0.35), (125.0, 0.20), (80.0, 0.50)]:
        s = sigma * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * s * s) / s
        d2 = d1 - s
        call = F * _phi(d1) - K * _phi(d2)      # undiscounted Black
        put = call - (F - K)                    # parity: C - P = F - K
        s_call = volfi.implied_volatility_from_option(F, K, call, T, is_call=True)
        s_put = volfi.implied_volatility_from_option(F, K, put, T, is_call=False)
        assert abs(s_call - sigma) < 1e-10, (K, "call", s_call, sigma)
        assert abs(s_put - sigma) < 1e-10, (K, "put", s_put, sigma)


def test_warm_restart():
    h, c = 1.0, 0.0012998981331268078984
    w = volfi.implied_variance(h, c)            # cold, full accuracy
    w_prev = w * (1.01) ** 2                     # previous snapshot, ~1% vol higher
    w_warm = volfi.implied_variance_warm(h, c, w_prev, steps=2)
    assert abs(w_warm / w - 1) < 1e-11          # few-ULP warm floor for a small move


def test_version():
    assert volfi.version() == "0.2.0"
    assert volfi.__version__ == "0.2.0"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, "->", repr(e))
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
