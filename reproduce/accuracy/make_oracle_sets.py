#!/usr/bin/env python3
"""make_oracle_sets.py -- the 40-digit mpmath oracle behind reproduce/oracle_*.bin.

The oracle inverts the GIVEN double-precision price c (never a round trip through a double
forward pricer): for each (h, c) pair it solves Phi(-h/v + v/2) - e^h Phi(-h/v - v/2) = c
for v = sigma*sqrt(T) at 40 significant digits and stores v as the nearest double.

  python3 make_oracle_sets.py heat [--dps 40]        regenerate oracle_heat.bin: the "regular" set,
                                                    150 x 110 grid, h in [0.02, 16], v in [0.02, 8],
                                                    c = double(price(h, v)), points with c in (0, 1)
  python3 make_oracle_sets.py recheck FILE [--dps N] re-invert every (h, c) of a shipped file at N
                                                    digits (default 60) and report the largest
                                                    relative difference to the stored v; use it on
                                                    oracle_stressed.bin, oracle_edge.bin and
                                                    oracle_vec.bin, whose point lists are hand-built
                                                    and shipped as data, and on oracle_heat.bin

Record layout: int64 n, then n records of {double h; double c; double v}; oracle_vec.bin carries
one extra byte per record, the chart tag of the point (kept unchanged by recheck).
"""
import sys, struct, math
from mpmath import mp

SQ2 = None
def Phi(x): return mp.erfc(-x / SQ2) / 2
def phi(x): return mp.e ** (-x * x / 2) / mp.sqrt(2 * mp.pi)
def price(h, v): return Phi(-h / v + v / 2) - mp.e ** h * Phi(-h / v - v / 2)

def solve(hd, cd):
    """v with price(h, v) == c to ~dps digits, for the doubles hd, cd; None if infeasible"""
    H = mp.mpf(hd); C = mp.mpf(cd)              # the exact binary doubles, not their shortest decimal strings
    lo, hi = mp.mpf('1e-9'), mp.mpf(40)
    for _ in range(120):                       # bracket first: the map is monotone in v
        m = (lo + hi) / 2
        if price(H, m) < C: lo = m
        else: hi = m
    v = (lo + hi) / 2
    for _ in range(200):                       # then Newton with the exact vega phi(d1)
        st = (price(H, v) - C) / phi(-H / v + v / 2)
        v -= st
        if abs(st) < abs(v) * mp.mpf(10) ** (-(mp.dps - 6)): break
    if not (v > 0) or not mp.isfinite(v): return None
    if abs(price(H, v) - C) / C > mp.mpf(10) ** (-(mp.dps - 8)): return None
    return v

def setdps(d):
    global SQ2
    mp.dps = d; SQ2 = mp.sqrt(2)

def read(path):
    b = open(path, "rb").read(); n = struct.unpack("<q", b[:8])[0]
    rec = (len(b) - 8) // n; out = []
    for i in range(n):
        r = b[8 + i * rec: 8 + (i + 1) * rec]
        h, c, v = struct.unpack("<ddd", r[:24]); out.append((h, c, v, r[24:]))
    return rec, out

def heat(out="oracle_heat.bin"):
    NH, NV = 150, 110
    pts = []
    for i in range(NH):
        hd = 0.02 + i * (16.0 - 0.02) / (NH - 1)
        for j in range(NV):
            vd = 0.02 + j * (8.0 - 0.02) / (NV - 1)
            cd = float(price(mp.mpf(hd), mp.mpf(vd)))
            if not (0.0 < cd < 1.0): continue
            v = solve(hd, cd)
            if v is None: continue
            pts.append((hd, cd, float(v)))
        print("  h=%.3f done, %d points so far" % (hd, len(pts)), flush=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<q", len(pts)))
        for h, c, v in pts: f.write(struct.pack("<ddd", h, c, v))
    print("wrote", out, "n =", len(pts))

def recheck(path, limit=None):
    rec, pts = read(path); worst = 0.0; worst_at = None; n = 0
    for h, c, v, _ in (pts if limit is None else pts[:limit]):
        w = solve(h, c)
        if w is None: print("  infeasible at h=%r c=%r" % (h, c)); continue
        d = abs(float(w) - v) / v; n += 1
        if d > worst: worst, worst_at = d, (h, c, v, float(w))
    print("%s: %d points rechecked at %d digits, largest relative difference to the stored v: %.3e" % (path, n, mp.dps, worst))
    if worst_at: print("  at h=%r c=%r stored v=%r rechecked v=%r" % worst_at)
    print("  a difference at the 1e-16 level is the double rounding of the stored v; anything larger is a defect")

if __name__ == "__main__":
    a = sys.argv[1:]
    dps = 40; lim = None
    if "--dps" in a: dps = int(a[a.index("--dps") + 1])
    if "--limit" in a: lim = int(a[a.index("--limit") + 1])
    if not a or a[0] == "heat":
        setdps(dps); heat()
    elif a[0] == "recheck":
        setdps(dps if "--dps" in a else 60); recheck(a[1], lim)
    else:
        print(__doc__)
