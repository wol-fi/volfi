"""volfi v0.2.0 - routed, vectorizable machine-precision Black-Scholes implied volatility.

Conventions (matching the C++ library and the paper):
    h  = |log(K / F)|                absolute log-moneyness, >= 0
    c  = C / F                       undiscounted OTM-call price over the forward, in (0, 1)
    w  = v**2 = (sigma*sqrt(T))**2   total implied variance (what the core returns)

Every function accepts scalars or NumPy arrays; array inputs are inverted through the
vectorized batch driver. Scalar inputs return a float, array inputs a float64 ndarray.

    >>> import volfi
    >>> volfi.implied_variance(1.0, 0.2)              # -> float
    >>> volfi.implied_volatility(h_arr, c_arr, T)     # -> ndarray of sigma
    >>> sigma = volfi.implied_volatility_from_option(F, K, price, T, is_call=False)
    >>> w, status = volfi.implied_variance_checked(h, c)   # status vs volfi.iv_status
"""
import numpy as np

from . import _volfi
from ._volfi import iv_status, version

__version__ = "0.2.0"

__all__ = [
    "implied_variance",
    "implied_volatility",
    "implied_variance_checked",
    "implied_volatility_from_option",
    "implied_variance_warm",
    "iv_status",
    "version",
    "__version__",
]


def _scalar(*xs):
    return all(np.ndim(x) == 0 for x in xs)


def implied_variance(h, c):
    """Total implied variance w = v**2 from (h, c)."""
    out = _volfi.implied_variance(h, c)
    return float(out[0]) if _scalar(h, c) else out


def implied_volatility(h, c, t):
    """Total volatility sigma = sqrt(w / T) from (h, c, T)."""
    out = _volfi.implied_volatility(h, c, t)
    return float(out[0]) if _scalar(h, c, t) else out


def implied_variance_checked(h, c):
    """Return (w, status). Never raises; status entries match the ``iv_status`` enum."""
    w, st = _volfi.implied_variance_checked(h, c)
    if _scalar(h, c):
        return float(w[0]), iv_status(int(st[0]))
    return w, st


def implied_volatility_from_option(F, K, price, t, is_call):
    """sigma from raw option data. ITM options and puts are parity-projected internally."""
    out = _volfi.implied_volatility_from_option(F, K, price, t, is_call)
    return float(out[0]) if _scalar(F, K, price, t, is_call) else out


def implied_variance_warm(h, c, w_prev, steps=2):
    """Streaming warm restart from previous variances (few-ULP accuracy contract)."""
    out = _volfi.implied_variance_warm(h, c, w_prev, steps)
    return float(out[0]) if _scalar(h, c, w_prev) else out
