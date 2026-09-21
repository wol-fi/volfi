"""The WHOLE-BOOK chart: the exact tau-row series over the entire traded book.

Coordinates as in the recurrence chart:
    E = expm1(h),  a = log(1 + E/c),  A0 = a G(a),  G = 1/V_0,  t = h/A0,  v = t S.
Two regions, one branch at a = 2 pi that also selects the table cell:

  A  (a < 2 pi):   x = t^2, y = A0^2,   S = sum_{m<=MA} x^m P_m(y)          [raw series]
                   valid for theta = h/a <= THETA_A  (the h-radius is a here)
  B  (a >= 2 pi):  q = (sqrt(1+z) - 1)/(sqrt(1+z) + 1),  z = h^2/(4 pi^2),  w = 1/A0^2,
                   S = 1 + sum_{n=1..NB} q^n d_n(w),  d_n(w) = sum_{k=1..n} D[n][k] w^k
                   valid for h <= H_B  (the h-radius is 2 pi here; answer_whole_book.md A5)
  with D[n][k] = sum_{m=k..n} 4^m C(n+m-1, 2m-1) (4 pi^2)^m p_{m, m-k},
  p_{m,j} the coefficients of P_m(y) = sum_j p_{m,j} y^j  (gen/rows_P.json, exact).

Table of G: cell 0 on [0, 2 pi] in a, cell 1 on [2 pi, A_MAX] in u = 1/sqrt(a).
Validation of the assembled chart against the 40-digit solve in both regions, a
conditioning check of the q-series (largest term over |S|), and a truth set for the
gates (data/wb_truth.bin).  Emits include/volfi/volfi_wb_tables.hpp.
"""
import os, sys, json, time, struct, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp
import certified_near as G
from fractions import Fraction as Fr

mp.mp.dps = 40
HERE = os.path.dirname(os.path.abspath(__file__))
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--ma', type=int, default=16)          # rows in region A (x^0..x^MA)
ap.add_argument('--nb', type=int, default=18)          # rows in region B (q^1..q^NB)
ap.add_argument('--theta_a', default='0.35')
ap.add_argument('--h_b', default='4.0')
ap.add_argument('--hb_slope', default='0.16')     # region B: h <= min(H_B, HB_SLOPE a + HB_INT)
ap.add_argument('--hb_int', default='0.85')
ap.add_argument('--amax', default='700')
ap.add_argument('--eps', default='1e-17')
ARGS, _ = ap.parse_known_args()
MA, NB = ARGS.ma, ARGS.nb
THETA_A, H_B, A_MAX, EPS = mp.mpf(ARGS.theta_a), mp.mpf(ARGS.h_b), mp.mpf(ARGS.amax), mp.mpf(ARGS.eps)
HB_SLOPE, HB_INT = mp.mpf(ARGS.hb_slope), mp.mpf(ARGS.hb_int)
def h_b(a): return min(H_B, HB_SLOPE * a + HB_INT)
TWO_PI = 2 * mp.pi
fmt = G.fmt

def V_ref(tau, a):
    """v/theta by a safeguarded solve of price(k, v) = tgt (price is increasing in v):
    bisection to 60 bits, then Newton.  G.V's plain Newton from the tau = 0 guess can
    run away near the ceiling and overflow erfc."""
    if tau == 0: return V0(a)
    th = mp.sqrt(tau); k = th * a
    tgt = mp.expm1(k) / mp.expm1(a)
    lo, hi = k / 45, mp.mpf(200)          # k/v = A0 <= 37.4 on the box, so d2 >= -45 here: no erfc overflow
    for _ in range(70):
        mid = (lo + hi) / 2
        if G.price(k, mid) > tgt: hi = mid
        else: lo = mid
    v = (lo + hi) / 2
    for _ in range(60):
        d1 = -k / v + v / 2
        st = (G.price(k, v) - tgt) / G.phi(d1)
        v = v - st
        if abs(st) < abs(v) * mp.mpf(10) ** (-mp.mp.dps + 6): break
    return v / th

_V0 = {}
def V0(a):
    k = str(a)
    if k not in _V0: _V0[k] = mp.sqrt(2 * mp.pi) if a == 0 else G.V_at_zero(a)
    return _V0[k]

# ------------------------------------------------------------- exact rows
P = json.load(open(os.path.join(HERE, "rows_P.json")))
assert max(int(k) for k in P) >= max(MA, NB)
Pq = {int(m): [Fr(c) for c in cf] for m, cf in P.items()}         # highest degree first
Pn = {m: [mp.mpf(c.numerator) / c.denominator for c in cf] for m, cf in Pq.items()}

# ------------------------------------------------------------- G table
def cheb_fit(f, lo, hi, N=96):
    xs = [mp.cos(mp.pi * (j + mp.mpf('0.5')) / N) for j in range(N)]
    vals = [f(lo + (hi - lo) * (x + 1) / 2) for x in xs]
    co = [2 * sum(vals[j] * mp.cos(mp.pi * k * (j + mp.mpf('0.5')) / N) for j in range(N)) / N for k in range(N)]
    sc = max(abs(c) for c in co)
    n = next(k + 1 for k in range(N - 1, -1, -1) if abs(co[k]) > EPS * sc)
    c = co[:n]; c[0] = c[0] / 2
    return c
def clen(c, xv):
    d0 = mp.mpf(0); d1 = mp.mpf(0)
    for k in range(len(c) - 1, 0, -1):
        d0, d1 = 2 * xv * d0 - d1 + c[k], d0
    return xv * d0 - d1 + c[0]

print("fitting G = 1/V_0 ...")
U_LO, U_HI = 1 / mp.sqrt(A_MAX), 1 / mp.sqrt(TWO_PI)
cG0 = cheb_fit(lambda a: 1 / V0(a), mp.mpf(0), TWO_PI)
cG1 = cheb_fit(lambda u: V0(1 / (u * u)) * u, U_LO, U_HI)      # F1 = V_0 / sqrt(a): flat, O(1)
print("  cell 0: a in [0, 2pi] in a,        G = 1/V_0      : %d terms" % len(cG0))
print("  cell 1: a in [2pi, %d] in 1/sqrt(a), F1 = V_0/sqrt a : %d terms" % (int(A_MAX), len(cG1)))
def G_tab(a):
    if a < TWO_PI:
        return clen(cG0, 2 * a / TWO_PI - 1)
    u = 1 / mp.sqrt(a)
    return clen(cG1, (2 * u - (U_LO + U_HI)) / (U_HI - U_LO))

# ------------------------------------------------------------- region B constants
print("region B constants D[n][k], n = 1..%d ..." % NB)
FP2 = 4 * mp.pi ** 2
D = {}
for n in range(1, NB + 1):
    D[n] = {}
    for k in range(1, n + 1):
        s = mp.mpf(0)
        for m in range(k, n + 1):
            c_nm = mp.mpf(4) ** m * mp.binomial(n + m - 1, 2 * m - 1)
            # p_{m, m-k}: coefficient of y^{m-k} in P_m; Pn[m] is highest-first, degree m-1
            deg = len(Pn[m]) - 1
            j = m - k
            p = Pn[m][deg - j] if 0 <= j <= deg else mp.mpf(0)
            s += c_nm * FP2 ** m * p
        D[n][k] = s

# ------------------------------------------------------------- evaluators (mp, with double-rounded constants)
def dbl(x): return mp.mpf(float(x))
cG0d = [dbl(c) for c in cG0]; cG1d = [dbl(c) for c in cG1]
Pd = {m: [dbl(c) for c in Pn[m]] for m in Pn}
Dd = {n: {k: dbl(D[n][k]) for k in D[n]} for n in D}
U_LOd, U_HId = dbl(U_LO), dbl(U_HI)
def chart(h, a, region_only=None):
    """v from (h, a) with the double-rounded tables, in 40-digit arithmetic (so the
    error reported is the representation error, not roundoff).  Returns (v, region, cond)."""
    if a < TWO_PI:
        g = clen(cG0d, 2 * a / TWO_PI - 1)
        A0 = a * g; t = h / A0
        x = t * t; y = A0 * A0
        s = mp.mpf(0)
        for m in range(MA, -1, -1):
            pm = mp.mpf(0)
            for c in Pd[m]: pm = pm * y + c
            s = s * x + pm
        return t * s, 'A', mp.mpf(1)
    u = 1 / mp.sqrt(a)
    f1 = clen(cG1d, (2 * u - (U_LOd + U_HId)) / (U_HId - U_LOd))
    pp = u * f1                      # = 1/A0
    t = h * pp
    w = pp * pp
    z = h * h / FP2
    q = (mp.sqrt(1 + z) - 1) / (mp.sqrt(1 + z) + 1)
    s = mp.mpf(0); big = mp.mpf(0)
    terms = []
    for n in range(NB, 0, -1):
        dn = mp.mpf(0)
        for k in range(n, 0, -1): dn = dn * w + Dd[n][k]
        dn = dn * w
        terms.append(abs(dn * q ** n))
        s = s * q + dn
    s = s * q + 1
    cond = max(terms) / abs(s)
    return t * s, 'B', cond

# ------------------------------------------------------------- validation
print("validating against the 40-digit solve ...")
rnd = random.Random(11)
t0 = time.time()
worst = {'A': (mp.mpf(0), None), 'B': (mp.mpf(0), None)}
cond_max = mp.mpf(0)
pts = []
for i in range(300):                                 # region A: a in (0, 2pi), theta <= THETA_A
    a = TWO_PI * mp.mpf(rnd.random()) ** 2 + mp.mpf('1e-3')
    if a >= TWO_PI: a = TWO_PI - mp.mpf('1e-6')
    th = THETA_A * mp.mpf(rnd.random()) ** 0.5
    pts.append((th * a, a))
for i in range(300):                                 # region B: a in [2pi, A_MAX], h <= H_B
    a = TWO_PI + (A_MAX - TWO_PI) * mp.mpf(rnd.random()) ** 3
    h = h_b(a) * mp.mpf(rnd.random()) ** 0.5
    pts.append((h, a))
for h, a in [(THETA_A * (TWO_PI - mp.mpf('1e-6')), TWO_PI - mp.mpf('1e-6')), (h_b(TWO_PI), TWO_PI), (h_b(mp.mpf(10)), mp.mpf(10)), (H_B, A_MAX), (THETA_A * mp.mpf('0.004'), mp.mpf('0.004')), (THETA_A * mp.mpf('0.01'), mp.mpf('0.01'))]:
    pts.append((h, a))
for h, a in pts:
    tau = (h / a) ** 2
    Vex = V_ref(tau, a); vex = mp.sqrt(tau) * Vex
    v, reg, cond = chart(h, a)
    e = abs(v - vex) / vex
    if e > worst[reg][0]: worst[reg] = (e, (float(h), float(a)))
    if reg == 'B' and cond > cond_max: cond_max = cond
print("  region A (%d pts): worst rel %.3e at (h=%.4g, a=%.4g)" % (300 + 3, float(worst['A'][0]), *worst['A'][1]))
print("  region B (%d pts): worst rel %.3e at (h=%.4g, a=%.4g);  q-series conditioning max|term|/|S| = %.2f"
      % (300 + 2, float(worst['B'][0]), *worst['B'][1], float(cond_max)))
print("  (%.0fs)" % (time.time() - t0))

# ------------------------------------------------------------- truth set
tp = os.path.join(HERE, "..", "data", "wb_truth.bin")
rnd = random.Random(23)
recs = []
for i in range(1200):
    if i % 2 == 0:
        a = TWO_PI * mp.mpf(rnd.random()) ** 2 + mp.mpf('2e-3')
        if a >= TWO_PI: a = TWO_PI - mp.mpf('1e-6')
        th = THETA_A * mp.mpf(rnd.random()) ** 0.5
        h = th * a
    else:
        a = TWO_PI + (A_MAX - TWO_PI) * mp.mpf(rnd.random()) ** 3
        h = h_b(a) * mp.mpf(rnd.random()) ** 0.5
    hd = mp.mpf(float(h))
    rho = 1 / mp.expm1(a)
    c = rho * mp.expm1(hd)
    cd = mp.mpf(float(c))
    if not (0 < cd < 1): continue
    # re-solve at the double-rounded (h, c)
    rho2 = cd / mp.expm1(hd); a2 = mp.log(1 + 1 / rho2); tau2 = (hd / a2) ** 2
    v = mp.sqrt(tau2) * V_ref(tau2, a2)
    recs.append((float(hd), float(cd), float(v)))
with open(tp, "wb") as f:
    f.write(struct.pack("<q", len(recs)))
    for r in recs: f.write(struct.pack("<3d", *r))
print("wrote %s (%d records, both regions)" % (tp, len(recs)))

# ------------------------------------------------------------- emit
L = []
L.append("// volfi_wb_tables.hpp   GENERATED by gen/certified_wb.py -- DO NOT EDIT")
L.append("// The whole-book chart: G = 1/V_0 on [0, 2pi], F1 = V_0/sqrt(a) on [2pi, A_MAX] in 1/sqrt(a),")
L.append("// the exact rows P_m (region A) and")
L.append("// the q-series constants D[n][k] (region B).  See gen/certified_wb.py.")
L.append("#ifndef VOLFI_WB_TABLES_HPP")
L.append("#define VOLFI_WB_TABLES_HPP")
L.append("namespace volfi_wb {")
L.append("")
L.append("static constexpr double NW_TWO_PI   = %s;" % fmt(TWO_PI))
L.append("static constexpr double NW_A_MAX    = %s;" % fmt(A_MAX))
L.append("static constexpr double NW_THETA_A  = %s;   // region A valid for h <= NW_THETA_A * a" % fmt(THETA_A))
L.append("static constexpr double NW_H_B      = %s;   // region B valid for h <= min(NW_H_B, NW_HB_SLOPE a + NW_HB_INT)" % fmt(H_B))
L.append("static constexpr double NW_HB_SLOPE = %s;" % fmt(HB_SLOPE))
L.append("static constexpr double NW_HB_INT   = %s;" % fmt(HB_INT))
L.append("static constexpr double NW_INV_4PI2 = %s;   // 1/(4 pi^2): z = h*h*NW_INV_4PI2" % fmt(1 / FP2))
L.append("static constexpr double NW_SA0      = %s;   // 2/(2 pi): xa = fma(a, NW_SA0, -1) in cell 0" % fmt(2 / TWO_PI))
L.append("static constexpr double NW_SU1      = %s;   // 2/(U_HI-U_LO): xu = fma(u, NW_SU1, NW_BU1), u = 1/sqrt(a)" % fmt(2 / (U_HI - U_LO)))
L.append("static constexpr double NW_BU1      = %s;   // -(U_LO+U_HI)/(U_HI-U_LO)" % fmt(-(U_LO + U_HI) / (U_HI - U_LO)))
L.append("static constexpr int    NW_NG0 = %d, NW_NF1 = %d, NW_MA = %d, NW_NB = %d;" % (len(cG0), len(cG1), MA, NB))
L.append("static constexpr double NW_G0[%d] = {" % len(cG0))
for i in range(0, len(cG0), 3): L.append("  " + ",".join(fmt(x) for x in cG0[i:i + 3]) + ",")
L.append("};")
L.append("static constexpr double NW_F1[%d] = {" % len(cG1))   # F1 = V_0/sqrt(a) on cell 1
for i in range(0, len(cG1), 3): L.append("  " + ",".join(fmt(x) for x in cG1[i:i + 3]) + ",")
L.append("};")
L.append("// region A rows: P_m coefficients, highest degree first, at NWRowA<m>::off, deg m-1")
L.append("template<int M> struct NWRowA;")
flatP = []
for m in range(0, MA + 1):
    L.append("template<> struct NWRowA<%d> { enum { deg = %d, off = %d }; };" % (m, len(Pn[m]) - 1, len(flatP)))
    flatP += Pn[m]
L.append("static constexpr int    NW_NP = %d;" % len(flatP))
L.append("static constexpr double NW_P[%d] = {" % len(flatP))
for i in range(0, len(flatP), 3): L.append("  " + ",".join(fmt(x) for x in flatP[i:i + 3]) + ",")
L.append("};")
L.append("// region B rows: d_n(w) = w (D[n][1] + w (D[n][2] + ... w D[n][n])), stored D[n][n] .. D[n][1]")
L.append("// (highest power first) at NWRowB<n>::off, n coefficients")
L.append("template<int N> struct NWRowB;")
flatD = []
for n in range(1, NB + 1):
    L.append("template<> struct NWRowB<%d> { enum { deg = %d, off = %d }; };" % (n, n, len(flatD)))
    flatD += [D[n][k] for k in range(n, 0, -1)]
L.append("static constexpr int    NW_ND = %d;" % len(flatD))
L.append("static constexpr double NW_D[%d] = {" % len(flatD))
for i in range(0, len(flatD), 3): L.append("  " + ",".join(fmt(x) for x in flatD[i:i + 3]) + ",")
L.append("};")
L.append("")
L.append("} // namespace volfi_wb")
L.append("#endif")
out = os.path.join(HERE, "..", "..", "include", "volfi", "volfi_wb_tables.hpp")
open(out, "w", encoding="utf-8").write(chr(10).join(L) + chr(10))
print("emitted: G %d + %d, P %d, D %d doubles -> %s" % (len(cG0), len(cG1), len(flatP), len(flatD), out))
