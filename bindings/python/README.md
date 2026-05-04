# volfi Python binding

Minimal `pybind11` wrapper for `volfi` v0.1.7.

```bash
pip install ./bindings/python
```

From GitHub:

```bash
pip install "git+https://github.com/wol-fi/volfi.git#subdirectory=bindings/python"
```

## Call-price roundtrip

```python
import numpy as np
import volfi
from math import erf, sqrt


def cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(f, k, d, t, v):
    s = v * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * s * s) / s
    d2 = d1 - s
    return d * (f * np.vectorize(cdf)(d1) - k * np.vectorize(cdf)(d2))

f = np.array([100.0])
k = np.array([105.0])
d = np.array([0.98])
t = np.array([1.2])
v = np.array([0.2])
p = bs_call(f, k, d, t, v)
v_hat = volfi.iv_call(f, k, d, t, p)

print(dict(price=p[0], true_vol=v[0], estimated_vol=v_hat[0], error=v_hat[0] - v[0]))
```

## Accuracy and speed on 10,000 calls

```python
import time
from statistics import NormalDist

n = 10_000
rng = np.random.default_rng(1)
f = np.full(n, 100.0)
d = np.ones(n)
t = np.ones(n)
v = rng.uniform(0.01, 2.0, n)
delta = rng.uniform(0.01, 0.99, n)
q = np.vectorize(NormalDist().inv_cdf)(delta)
s = v * np.sqrt(t)
k = f * np.exp(0.5 * s * s - q * s)
p = bs_call(f, k, d, t, v)

v_hat = volfi.iv_call(f, k, d, t, p)
e = v_hat - v

print({
    "n": n,
    "nan": int(np.isnan(v_hat).sum()),
    "nonfinite": int((~np.isfinite(v_hat)).sum()),
    "max_abs_error": float(np.max(np.abs(e))),
    "mean_abs_error": float(np.mean(np.abs(e))),
    "q99_abs_error": float(np.quantile(np.abs(e), 0.99)),
})

runs = []
for _ in range(100):
    t0 = time.perf_counter_ns()
    volfi.iv_call(f, k, d, t, p)
    runs.append((time.perf_counter_ns() - t0) / n)

print({"median_ns_per_eval": float(np.median(runs))})
```

## Benchmark against py_vollib LetsBeRational

Install `py_vollib` and run the included benchmark demo.

```bash
pip install py_vollib
python bindings/python/examples/bench_vollib.py
```

The benchmark uses the same random grid as above:

```text
v ~ U(0.01, 2.0), Delta ~ U(0.01, 0.99)
```

It reports accuracy summaries for both methods and median nanoseconds per evaluation.

## OTM context path

```python
h = np.abs(np.log(k / f))
c_otm = np.where(k >= f, p, p + d * (k - f)) / (d * np.minimum(f, k))
ctx = volfi.ctx(h)
w = ctx.w(c_otm)
iv = ctx.iv(c_otm, t)
```
