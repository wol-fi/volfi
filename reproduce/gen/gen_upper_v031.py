#!/usr/bin/env python3
"""gen_upper_v031.py -- RESEARCH.  Offline generator of the three frozen pieces of the UPPER v0.3.1 candidate:
  (1) x0(s), s = sqrt(a_U), a_U = h - 2 log(1-c): the Gaussian quantile on the fixed-gbar manifold, a 1-D Chebyshev series
  (2) R(X, Z): the endpoint surface, x = x0 (1 + zeta R), degree (1,2) in the normalized (x0, zeta), 6 doubles
  (3) erfcx on [-0.40, 0]: the one new special-function piece the one-exponential residual needs (note: bound is -0.3591)
and of the oracle truth set upper_truth.bin (40-digit inverse of the GIVEN double price, seams over-sampled).
Writes upper_v031_tables.hpp next to this file and ../data/upper_truth.bin.  Needs numpy, mpmath.
"""
import math, os, struct, sys, random, time
import numpy as np
import mpmath as mp
mp.mp.dps = 45
HERE = os.path.dirname(os.path.abspath(__file__))
def C_mp(h, v): return mp.ncdf(v / 2 - h / v) - mp.exp(h) * mp.ncdf(-v / 2 - h / v)
def onemC_mp(h, v): return mp.ncdf(-(v / 2 - h / v)) + mp.exp(h) * mp.ncdf(-v / 2 - h / v)
def truth(h, c):
    h = mp.mpf(h); onec = 1 - mp.mpf(c); lo = mp.log(onec)
    v = mp.findroot(lambda v: mp.log(onemC_mp(h, v)) - lo, (mp.mpf("0.05"), mp.mpf(40)), solver="bisect", tol=mp.mpf(10) ** -30, maxsteps=300)
    for _ in range(3): v = v + (onemC_mp(h, v) - onec) / mp.npdf(v / 2 - h / v)
    return v

# ---------------------------------------------------------------- the UPPER service region, sampled with exact (h, v)
V_LO, V_HI, H_HI, W_SEAM = 1.85, 8.0, 16.2, 3.8
def in_region(h, v): return V_LO <= v <= V_HI and 0 < h <= H_HI and h * h / (2 * v * v) < W_SEAM
pts = []
for h in np.concatenate([np.linspace(1e-3, 0.5, 12), np.linspace(0.5, H_HI, 70)]):
    for v in np.concatenate([np.linspace(V_LO, 2.6, 16), np.linspace(2.6, V_HI, 34)]):
        if in_region(h, v): pts.append((float(h), float(v)))
for v in np.linspace(V_LO, 5.87, 60):                                   # the WING seam W -> 3.8 (h = sqrt(7.6) v <= 16.2)
    for W in (3.79, 3.6, 3.2):
        h = math.sqrt(2 * W) * v
        if in_region(h, v): pts.append((h, float(v)))
print("region sample: %d points" % len(pts)); t0 = time.time()
rows = []
for h, v in pts:
    onec = onemC_mp(mp.mpf(h), mp.mpf(v))
    if onec < mp.mpf(10) ** -15: continue                                # beyond the representable complement
    aU = mp.mpf(h) - 2 * mp.log(onec)
    rows.append((h, v, float(aU)))
rows = np.array(rows); H, V, AU = rows[:, 0], rows[:, 1], rows[:, 2]
print("a_U range [%.4f, %.4f], %.0f s" % (AU.min(), AU.max(), time.time() - t0))

# ---------------------------------------------------------------- (1) x0(s): Chebyshev interpolation at 45 digits
S_LO, S_HI = math.sqrt(AU.min()) * 0.995, math.sqrt(AU.max()) * 1.005
def x0_exact(s): return -mp.sqrt(2) * mp.erfinv(mp.exp(-mp.mpf(s) ** 2 / 2) - 1)       # -Phi^{-1}(exp(-a/2)/2)
NX0 = 7                                                                                   # degree 6
nodes = [mp.cos(mp.pi * (k + mp.mpf(1) / 2) / NX0) for k in range(NX0)]
fv = [x0_exact((S_LO + S_HI) / 2 + (S_HI - S_LO) / 2 * t) for t in nodes]
X0C = [float((2 if j else 1) * sum(fv[k] * mp.cos(j * mp.pi * (k + mp.mpf(1) / 2) / NX0) for k in range(NX0)) / NX0) for j in range(NX0)]
def x0_poly(s):
    t = (2 * s - (S_LO + S_HI)) / (S_HI - S_LO); return np.polynomial.chebyshev.chebval(t, X0C)
chk = np.linspace(S_LO, S_HI, 2001); ex = np.array([float(x0_exact(s)) for s in chk])
print("x0(s) degree %d: worst relative error %.2e on [%.4f, %.4f]" % (NX0 - 1, np.max(np.abs(x0_poly(chk) / ex - 1)), S_LO, S_HI))

# ---------------------------------------------------------------- (2) the endpoint surface, weighted least squares on x/x0 - 1 = zeta R
S = np.sqrt(AU); X0 = x0_poly(S); r = H / AU; ZETA = (r / (1 + np.sqrt(1 - r * r))) ** 2
X0MIN, X0MAX, ZMAX = float(X0.min()), float(X0.max()), float(ZETA.max()) * 1.0005
Xn = 2 * (X0 - X0MIN) / (X0MAX - X0MIN) - 1; Zn = 2 * ZETA / ZMAX - 1
T = lambda n, t: [np.ones_like(t), t, 2 * t * t - 1][n]
basis = np.stack([T(i, Xn) * T(j, Zn) for i in range(2) for j in range(3)], axis=1)
target = (V / 2) / X0 - 1.0                                              # = zeta R
A, *_ = np.linalg.lstsq(basis * ZETA[:, None], target, rcond=None)
seed_err = np.abs(X0 * (1 + ZETA * (basis @ A)) / (V / 2) - 1)
print("endpoint surface (1,2): worst seed error %.3e at (h, v) = (%.3f, %.3f); zeta_max %.5f" % (seed_err.max(), H[seed_err.argmax()], V[seed_err.argmax()], ZETA.max()))

# ---------------------------------------------------------------- (3) erfcx on [-0.40, 0]
EA, EB, NE = -0.40, 0.0, 15
nodes = [mp.cos(mp.pi * (k + mp.mpf(1) / 2) / NE) for k in range(NE)]
erfcx = lambda z: mp.exp(z * z) * mp.erfc(z)
fv = [erfcx((EA + EB) / 2 + mp.mpf(EB - EA) / 2 * t) for t in nodes]
EC = [float((2 if j else 1) * sum(fv[k] * mp.cos(j * mp.pi * (k + mp.mpf(1) / 2) / NE) for k in range(NE)) / NE) for j in range(NE)]
zz = np.linspace(EA, EB, 1001); tt = (2 * zz - (EA + EB)) / (EB - EA)
ee = np.array([float(erfcx(mp.mpf(z))) for z in zz])
print("erfcx on [%.2f, 0], degree %d: worst relative error %.2e (double evaluation of the rounded coefficients)" % (EA, NE - 1, np.max(np.abs(np.polynomial.chebyshev.chebval(tt, EC) / ee - 1))))

# ---------------------------------------------------------------- header
hx = lambda x: float(x).hex()
with open(os.path.join(HERE, "upper_v031_tables.hpp"), "w", newline="\n") as f:
    f.write("// upper_v031_tables.hpp -- GENERATED by gen_upper_v031.py (research).  Do not edit.\n#pragma once\nnamespace upper_v031 {\n")
    f.write("static const double S_LO = %s, S_HI = %s;   // sqrt(a_U) range\n" % (hx(S_LO), hx(S_HI)))
    f.write("static const int NX0 = %d;\nstatic const double X0C[%d] = { %s };\n" % (NX0, NX0, ", ".join(hx(c) for c in X0C)))
    f.write("static const double X0MIN = %s, X0MAX = %s, ZMAX = %s;\n" % (hx(X0MIN), hx(X0MAX), hx(ZMAX)))
    f.write("static const double RA[2][3] = { { %s }, { %s } };   // R = sum A_ij T_i(X) T_j(Z)\n" % (", ".join(hx(a) for a in A[:3]), ", ".join(hx(a) for a in A[3:])))
    f.write("static const double ENEG_A = %s, ENEG_B = %s;\nstatic const int NENEG = %d;\nstatic const double ENEG_C[%d] = { %s };\n" % (hx(EA), hx(EB), NE, NE, ", ".join(hx(c) for c in EC)))
    f.write("} // namespace upper_v031\n")
print("wrote upper_v031_tables.hpp")

# ---------------------------------------------------------------- oracle truth set, seams over-sampled
if "--no-truth" in sys.argv: sys.exit(0)
rng = random.Random(20260917); Q = []
for _ in range(900): Q.append((rng.uniform(1e-3, H_HI), rng.uniform(V_LO, V_HI)))                       # uniform fill
for _ in range(500): v = rng.uniform(V_LO, 5.87); Q.append((math.sqrt(2 * rng.uniform(3.5, 3.7999)) * v, v))   # WING seam
for _ in range(400): Q.append((rng.uniform(1e-3, 5.1), V_LO + rng.uniform(0, 0.02)))                   # v = 1.85 boundary
for _ in range(200): Q.append((rng.uniform(15.9, H_HI), rng.uniform(5.75, V_HI)))                      # h cap
for _ in range(200): Q.append((10 ** rng.uniform(-6, -2), rng.uniform(V_LO, V_HI)))                    # h -> 0
for _ in range(200): Q.append((rng.uniform(1e-3, 6.0), rng.uniform(7.5, V_HI)))                        # v -> 8
for _ in range(300):                                                                                     # the c = 1/2 switch
    h = rng.uniform(0.5, 12.0)
    try:
        v = float(mp.findroot(lambda v: C_mp(mp.mpf(h), v) - mp.mpf("0.5"), (mp.mpf(1.85), mp.mpf(8)), solver="bisect", tol=1e-20))
        Q.append((h, v * (1 + rng.uniform(-2e-3, 2e-3))))
    except Exception: pass
out = []; t0 = time.time()
for h, v in Q:
    if not in_region(h, v): continue
    c = float(C_mp(mp.mpf(h), mp.mpf(v)))
    if not (0.0 < c < 1.0) or 1.0 - c < 1e-15: continue
    out.append((h, c, float(truth(h, c))))
with open(os.path.join(HERE, "..", "data", "upper_truth.bin"), "wb") as f:
    f.write(struct.pack("<q", len(out)))
    for h, c, v in out: f.write(struct.pack("<3d", h, c, v))
print("wrote upper_truth.bin: %d points, %.0f s" % (len(out), time.time() - t0))
