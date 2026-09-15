"""40-digit truth set for the NEAR band, sampled in the certified coordinates.

Emits near_truth.bin :  int64 n, then n * {double h; double c; double v;}
h and c are the ROUNDED doubles actually fed to the inverter; v is the exact
root of C(h_double, v) = c_double computed at 40 digits, so the file measures
the inverter and nothing else.
"""
import math, os, struct, random
import mpmath as mp
from scipy.special import erf
mp.mp.dps = 40

SQ2 = mp.sqrt(2)
THETA = mp.mpf(str(float(erf(1.85 / (2 * math.sqrt(2))))))
A_WING = mp.mpf('8.050968812381163')
H_NEAR = mp.mpf('0.3')

def Phi(x): return mp.erfc(-x / SQ2) / 2
def phi(x): return mp.e ** (-x * x / 2) / mp.sqrt(2 * mp.pi)
def price(k, v): return Phi(-k / v + v / 2) - mp.e ** k * Phi(-k / v - v / 2)

def solve(h, c, guess):
    v = mp.mpf(guess)
    for _ in range(120):
        d1 = -h / v + v / 2
        st = (price(h, v) - c) / phi(d1)
        v -= st
        if abs(st) < abs(v) * mp.mpf(10) ** (-34): break
    return v

rng = random.Random(20260907)
pts = []
# structured: a log-spaced, theta spanning the full range including both seams
NA, NT = 60, 34
for i in range(NA):
    a = mp.e ** (mp.log(mp.mpf('1e-5')) + (mp.log(A_WING) - mp.log(mp.mpf('1e-5'))) * i / (NA - 1))
    for j in range(NT):
        th = THETA * (j + mp.mpf('0.5')) / NT
        h = th * a
        if not (h > mp.mpf('1e-13')) or h >= H_NEAR: continue
        c = mp.expm1(h) / mp.expm1(a)
        if not (0 < c < 1): continue
        pts.append((h, c))
# random fill
while len(pts) < 2600:
    a = mp.e ** (mp.log(mp.mpf('1e-5')) + (mp.log(A_WING) - mp.log(mp.mpf('1e-5'))) * mp.mpf(rng.random()))
    th = THETA * mp.mpf(rng.random())
    h = th * a
    if not (mp.mpf('1e-13') < h < H_NEAR): continue
    c = mp.expm1(h) / mp.expm1(a)
    if not (0 < c < 1): continue
    pts.append((h, c))

out = []
for (h, c) in pts:
    hd = float(h); cd = float(c)                     # the doubles the inverter sees
    if not (1e-300 < cd < 1.0) or not (hd > 0.0): continue
    H = mp.mpf(repr(hd)); Cc = mp.mpf(repr(cd))
    # guess from the leading term v ~ h/Binv(rho): just bracket-bisect a few times
    lo, hi = mp.mpf('1e-9'), mp.mpf(20)
    for _ in range(90):
        m = (lo + hi) / 2
        if price(H, m) < Cc: lo = m
        else: hi = m
    v = solve(H, Cc, (lo + hi) / 2)
    if not (v > 0) or not mp.isfinite(v): continue
    res = abs(price(H, v) - Cc) / Cc
    if res > mp.mpf('1e-32'): continue
    out.append((hd, cd, float(v)))

p = os.path.join(os.path.dirname(__file__), "..", "gate", "near_truth.bin")
with open(p, "wb") as f:
    f.write(struct.pack("<q", len(out)))
    for (h, c, v) in out:
        f.write(struct.pack("<ddd", h, c, v))
print("wrote %d truth points -> %s" % (len(out), os.path.abspath(p)))
vs = [v for _, _, v in out]
print("  v range [%.6g, %.6g]" % (min(vs), max(vs)))
