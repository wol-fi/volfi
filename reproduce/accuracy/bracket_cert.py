# bracket_cert.py -- pointwise residual-bracket certificate in Arb ball arithmetic (python-flint).
# Usage: make certify   (bracket_dump writes out/SET.hex, then  python accuracy/bracket_cert.py out)
# For each line "h c vhat code" (exact hexadecimal doubles) certify
#     C(h,(1-eta) vhat) < c < C(h,(1+eta) vhat),   eta = eps/(1+eps),
# which by strict monotonicity of C in v proves |vhat - v*|/v* <= eps. Inputs are exact binary64 values,
# every comparison is a certain ball comparison, precision is doubled until both signs resolve.
import sys, os
from flint import arb, ctx
HERE = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
SETS = ["wb_truth", "wb_truth_boundary", "near_truth", "upper_truth", "oracle_heat", "oracle_stressed", "oracle_edge"]
EPS = ["1e-15", "6e-16"]
def C(h, v):
    s2 = arb(2).sqrt(); x = h / v; d1 = v / 2 - x; d2 = -v / 2 - x
    return ((-d1 / s2).erfc() - h.exp() * (-d2 / s2).erfc()) / 2
def cert(h, c, vh, eps):
    prec = 256
    while prec <= 8192:
        ctx.prec = prec
        H, Cc, V, e = arb(h), arb(c), arb(vh), arb(eps); eta = e / (1 + e)
        lo = C(H, (1 - eta) * V) - Cc; hi = C(H, (1 + eta) * V) - Cc
        if (lo < 0 or lo > 0 or lo == 0) and (hi < 0 or hi > 0 or hi == 0):
            ok = bool(lo < 0) and bool(hi > 0)
            return ok, float((-lo / Cc).lower()), float((hi / Cc).lower()), prec
        prec *= 2
    return None, 0.0, 0.0, prec
tot = {e: [0, 0, 0] for e in EPS}
for s in SETS:
    rows = [l.split() for l in open(os.path.join(HERE, s + ".hex"))]
    for eps in EPS:
        n = ok = und = 0; mlo = mhi = float("inf"); pmax = 0; fails = []
        for hx, cx, vx, code in rows:
            h, c, v = float.fromhex(hx), float.fromhex(cx), float.fromhex(vx)
            if not (v > 0): und += 1; n += 1; continue
            r, a, b, p = cert(h, c, v, eps); n += 1; pmax = max(pmax, p)
            if r is None: und += 1
            elif r: ok += 1; mlo = min(mlo, a); mhi = min(mhi, b)
            else: fails.append((h, c, v, code))
        tot[eps][0] += n; tot[eps][1] += ok; tot[eps][2] += und
        print("%-18s eps=%-6s n=%6d certified=%6d failed=%4d unresolved=%d  min margin -F-/c=%.3e F+/c=%.3e  max prec=%d" % (s, eps, n, ok, len(fails), und, mlo, mhi, pmax), flush=True)
        for f in fails[:5]: print("     fail h=%.17g c=%.17g vhat=%.17g code=%s" % f)
for eps in EPS: print("TOTAL eps=%s n=%d certified=%d unresolved=%d" % (eps, *tot[eps]))
