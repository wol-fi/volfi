"""make_truth_wb_boundary.py -- a boundary-concentrated 40-digit truth set for the whole-book chart.

The referee's point: the 1,200-point set of certified_wb.py is a random sample; the chart has
analytically meaningful switches (a = 2 pi between the raw and the conformal rows, theta = 0.35
below it, the reach line h = min(4, 0.16 a + 0.85) above it, the table's cell edge at 2 pi, the
ends a -> 0 and a -> 700) and the campaign should be dense there.  This writes
gate/wb_truth_boundary.bin (int64 n; n x {h, c, v}) with the same oracle as certified_wb.py:
bracketed bisection to 60 bits then Newton at 40 digits on price(k, v) = c, v = sigma sqrt T.
Usage: python make_truth_wb_boundary.py [n_total]   (default 20000; ~0.2 s per point)
A second pass, --recheck FILE, recomputes the points listed in FILE (h c per line) at 60 digits
and prints the change of the rounded double, the oracle's own rounding check.
"""
import os, sys, struct, random, time
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certified_near as G

HERE = os.path.dirname(os.path.abspath(__file__))
THETA_A = mp.mpf('0.35'); H_B = mp.mpf('4.0'); HB_SLOPE = mp.mpf('0.16'); HB_INT = mp.mpf('0.85'); A_MAX = mp.mpf('700')
def h_b(a): return min(H_B, HB_SLOPE * a + HB_INT)

def V_ref(tau, a):
    if tau == 0: return G.V_at_zero(a)
    th = mp.sqrt(tau); k = th * a
    tgt = mp.expm1(k) / mp.expm1(a)
    lo, hi = k / 45, mp.mpf(200)
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

def record(h, a):
    """(h, c, v) as doubles for the pair (h, a): c = expm1(h)/expm1(a) rounded to double, then
    the oracle inverts THAT double (so the truth is for the price the code sees)."""
    h = mp.mpf(float(h)); a_nom = a
    c = mp.mpf(float(mp.expm1(h) / mp.expm1(a_nom)))
    if not (0 < c < 1): return None
    a = mp.log1p(mp.expm1(h) / c)                       # the exact a of the rounded price
    tau = (h / a) ** 2
    v = V_ref(tau, a) * (h / a)                          # sigma sqrt T
    return (float(h), float(c), float(v)), a

if len(sys.argv) > 1 and sys.argv[1] == '--recheck':
    mp.mp.dps = 60
    pts = [tuple(map(float, l.split()[:2])) for l in open(sys.argv[2]) if l.strip() and not l.startswith('#')]
    print("recheck at 60 digits: h c v40 v60 |v60-v40|/v60 same_double")
    for h, c in pts:
        hh = mp.mpf(h); cc = mp.mpf(c); a = mp.log1p(mp.expm1(hh) / cc); tau = (hh / a) ** 2
        v60 = V_ref(tau, a) * (hh / a)
        mp.mp.dps = 40
        v40 = V_ref(tau, a) * (hh / a)
        mp.mp.dps = 60
        print("%.17g %.17g %.17g %.17g %.2e %s" % (h, c, float(v40), float(v60), float(abs(v60 - v40) / v60), float(v40) == float(v60)))
    sys.exit(0)

mp.mp.dps = 40
n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rnd = random.Random(2026)
TWO_PI = 2 * mp.pi
recs = []; tags = []
def add(h, a, tag):
    r = record(h, a)
    if r: recs.append(r[0]); tags.append(tag)
t0 = time.time()
n_each = n_total // 8
# 1. the a = 2 pi switch, both sides, at every scale of distance; theta / reach as allowed on each side
for i in range(n_each):
    d = mp.mpf(10) ** (-1 - 5 * rnd.random()) * (1 if rnd.random() < 0.5 else -1)   # 1e-6 .. 1e-1
    a = TWO_PI + d
    if a < TWO_PI: h = a * THETA_A * mp.mpf(rnd.random()) ** 0.5
    else: h = h_b(a) * mp.mpf(rnd.random()) ** 0.5
    add(h, a, 'a=2pi')
# 2. theta = 0.35 edge below 2 pi (theta in [0.34, 0.35]), a spread over (0, 2 pi)
for i in range(n_each):
    a = TWO_PI * mp.mpf(rnd.random()) ** 2 + mp.mpf('2e-3')
    th = THETA_A * (1 - mp.mpf('0.03') * rnd.random())
    add(a * th, a, 'theta=0.35')
# 3. the reach line above 2 pi (h within 2% below h_b(a)), a in [2pi, 700]
for i in range(n_each):
    a = TWO_PI + (A_MAX - TWO_PI) * mp.mpf(rnd.random()) ** 3
    h = h_b(a) * (1 - mp.mpf('0.02') * rnd.random())
    add(h, a, 'reach')
# 4. the h = 4 cap for a >= 20 (h in [3.9, 4])
for i in range(n_each):
    a = mp.mpf(20) + (A_MAX - 20) * mp.mpf(rnd.random()) ** 2
    h = H_B * (1 - mp.mpf('0.025') * rnd.random())
    add(h, a, 'h=4')
# 5. small a (the table's left end and tiny prices' opposite: a -> 0 is c -> 1 at fixed h... here h tiny too)
for i in range(n_each):
    a = mp.mpf(10) ** (-3 + 2 * rnd.random())            # 1e-3 .. 0.1
    h = a * THETA_A * mp.mpf(rnd.random())
    add(h, a, 'a->0')
# 6. large a (deep prices, a in [300, 700], h up to 4)
for i in range(n_each):
    a = mp.mpf(300) + 400 * mp.mpf(rnd.random())
    h = H_B * mp.mpf(rnd.random()) ** 0.5
    add(h, a, 'a->700')
# 7. tiny h at every a (the h floor side)
for i in range(n_each):
    a = TWO_PI * 2 * mp.mpf(rnd.random()) ** 2 + mp.mpf('2e-3')
    h = mp.mpf(10) ** (-6 + 5 * rnd.random())
    if a < TWO_PI and h > a * THETA_A: h = a * THETA_A * mp.mpf(rnd.random())
    if a >= TWO_PI and h > h_b(a): h = h_b(a) * mp.mpf(rnd.random())
    add(h, a, 'h->0')
# 8. uniform fill of both regions
for i in range(n_total - len(recs)):
    if rnd.random() < 0.5:
        a = TWO_PI * mp.mpf(rnd.random()) ** 2 + mp.mpf('2e-3'); h = a * THETA_A * mp.mpf(rnd.random()) ** 0.5
    else:
        a = TWO_PI + (A_MAX - TWO_PI) * mp.mpf(rnd.random()) ** 3; h = h_b(a) * mp.mpf(rnd.random()) ** 0.5
    add(h, a, 'fill')
    if len(recs) % 1000 == 0: print("  %d points, %.0f s" % (len(recs), time.time() - t0), flush=True)
out = os.path.join(HERE, "..", "data", "wb_truth_boundary.bin")
with open(out, "wb") as f:
    f.write(struct.pack("<q", len(recs)))
    for r in recs: f.write(struct.pack("<3d", *r))
with open(os.path.join(HERE, "..", "data", "wb_truth_boundary.tags"), "w") as f:
    for t in tags: f.write(t + "\n")
from collections import Counter
print("wrote %s: %d points in %.0f s; by class: %s" % (out, len(recs), time.time() - t0, dict(Counter(tags))))
