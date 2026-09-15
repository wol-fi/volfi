"""Exact tau-row polynomials P_m(y) to high order, by truncated power series in q
with coefficients that are Laurent polynomials in A0 over the rationals.

Same derivation as rows_from_recurrence.py (answer_rational_a.md, eqs (10)-(13)):
    rho = B(A) + phi(A) R(A, q),   q = a^2 tau,   A(q) = A0 + sum_m c_m q^m,
every coefficient a Laurent polynomial in A0 with rational coefficients, so the
whole computation is exact rational arithmetic.  Output: gen/rows_P.json with
P_m(y), y = A0^2, highest degree first, as rational strings, m = 0..MMAX.
Checked against rows_from_recurrence.log for m <= 12.
"""
import os, sys, json, time
from fractions import Fraction as Fr
import sympy as sp

MMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N = MMAX + 1
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- Laurent polynomials in A0: dict exponent -> Fraction ----------------
def lp_add(u, v):
    w = dict(u)
    for e, c in v.items():
        w[e] = w.get(e, Fr(0)) + c
        if w[e] == 0: del w[e]
    return w
def lp_scale(u, s):
    return {e: c * s for e, c in u.items()} if s != 0 else {}
def lp_mul(u, v):
    w = {}
    for e1, c1 in u.items():
        for e2, c2 in v.items():
            e = e1 + e2
            w[e] = w.get(e, Fr(0)) + c1 * c2
    return {e: c for e, c in w.items() if c != 0}
def lp_shift(u, k):            # multiply by A0^k
    return {e + k: c for e, c in u.items()}
def lp_diff(u):
    return {e - 1: c * e for e, c in u.items() if e != 0}
ONE = {0: Fr(1)}
ZERO = {}

# ---------------- truncated series in q with Laurent coefficients ----------------
def s_mul(u, v):
    w = [ZERO] * N
    for i, ui in enumerate(u):
        if not ui: continue
        for j, vj in enumerate(v):
            if i + j >= N: break
            if not vj: continue
            w[i + j] = lp_add(w[i + j], lp_mul(ui, vj))
    return w
def s_pow(u, k):
    r = [ONE] + [ZERO] * (N - 1)
    for _ in range(k): r = s_mul(r, u)
    return r
def s_exp(u):                  # u[0] == 0
    e = [ONE] + [ZERO] * (N - 1)
    for n in range(1, N):
        acc = ZERO
        for k in range(1, n + 1):
            if u[k] and e[n - k]:
                acc = lp_add(acc, lp_scale(lp_mul(u[k], e[n - k]), Fr(k)))
        e[n] = lp_scale(acc, Fr(1, n))
    return e

def main():
    t0 = time.time()
    # b_j: B^{(j)} = phi b_j ; b_1 = -A0^-2 ; b_{j+1} = b_j' - A0 b_j
    b = [None, {-2: Fr(-1)}]
    for j in range(1, N):
        b.append(lp_add(lp_diff(b[j]), lp_scale(lp_shift(b[j], 1), Fr(-1))))
    # p(q) = sqrt(q) / (2 sinh(sqrt(q)/2)): even series in sqrt(q), rational coefficients
    sq = sp.symbols('sq')
    pser = sp.series(sq / (2 * sp.sinh(sq / 2)), sq, 0, 2 * N + 2).removeO()
    p_q = [{0: Fr(int(sp.Rational(pser.coeff(sq, 2 * m)).p), int(sp.Rational(pser.coeff(sq, 2 * m)).q))} for m in range(N)]
    p_q = [c if c[0] != 0 else ZERO for c in p_q]
    # binomial(-n, k) as Fractions
    def binom_neg(n, k):
        r = Fr(1)
        for i in range(k): r *= Fr(-n - i, i + 1)
        return r
    dser = [ZERO] * N
    for order in range(1, N):
        dA = [lp_shift(t, -1) for t in dser]                       # d / A0
        dA_pows = [s_pow(dA, k) for k in range(N)]
        def Apow_neg(n):                                            # (A0 + d)^-n
            r = [ZERO] * N
            for k in range(N):
                coef = binom_neg(n, k)
                for i in range(N):
                    if dA_pows[k][i]:
                        r[i] = lp_add(r[i], lp_scale(dA_pows[k][i], coef))
            return [lp_shift(t, -n) for t in r]
        beta = [[ZERO] * N]
        for m in range(1, N):
            an = Apow_neg(2 * m + 1)
            beta.append([lp_scale(lp_add(an[k], lp_scale(beta[m - 1][k], Fr(-1))), Fr(1, 2 * m + 1)) for k in range(N)])
        Rsum = [ZERO] * N
        for m in range(1, N):
            fac = Fr(-1, 8) ** m / sp.factorial(m)
            fac = Fr(int(sp.Rational(fac).p), int(sp.Rational(fac).q))
            for k in range(N - m):
                if beta[m][k]:
                    Rsum[k + m] = lp_add(Rsum[k + m], lp_scale(beta[m][k], fac))
        d_pows = [s_pow(dser, j) for j in range(N)]
        Bdiff = [ZERO] * N
        for j in range(1, N):
            fj = Fr(1, int(sp.factorial(j)))
            for k in range(N):
                if d_pows[j][k]:
                    Bdiff[k] = lp_add(Bdiff[k], lp_scale(lp_mul(b[j], d_pows[j][k]), fj))
        expo = [lp_scale(lp_shift(t, 1), Fr(-1)) for t in dser]     # -A0 d
        d2 = s_mul(dser, dser)
        expo = [lp_add(e, lp_scale(t, Fr(-1, 2))) for e, t in zip(expo, d2)]
        phir = s_exp(expo)
        rhs = s_mul(s_mul(phir, p_q), Rsum)
        eq = lp_add(Bdiff[order], rhs[order])
        # b_1 * c_order + eq = 0,  b_1 = -A0^-2  ->  c_order = eq * A0^2
        dser[order] = lp_shift(eq, 2)
        print("  c_%-2d done (%.0fs), %d Laurent terms" % (order, time.time() - t0, len(dser[order]))); sys.stdout.flush()
    # V/V0 = 1/(1 + d/A0) = sum_k (-1)^k (d/A0)^k ; P_m(y) = e_m A0^{2m}
    dA = [lp_shift(t, -1) for t in dser]
    ratio = [ZERO] * N
    for k in range(N):
        pk = s_pow(dA, k)
        for i in range(N):
            if pk[i]: ratio[i] = lp_add(ratio[i], lp_scale(pk[i], Fr((-1) ** k)))
    out = {}
    for m in range(N):
        Pm = lp_shift(ratio[m], 2 * m)
        assert all(e >= 0 and e % 2 == 0 for e in Pm), "P_%d is not a polynomial in y = A0^2: %s" % (m, sorted(Pm))
        deg = max(Pm) // 2 if Pm else 0
        coeffs = [str(Pm.get(2 * d, Fr(0))) for d in range(deg, -1, -1)]   # highest first
        out[m] = coeffs
    json.dump(out, open(os.path.join(HERE, "rows_P.json"), "w"), indent=0)
    print("wrote rows_P.json with P_0..P_%d  (%.0fs)" % (MMAX, time.time() - t0))
    # cross-check against the sympy log where available
    logf = os.path.join(HERE, "rows_from_recurrence.log")
    if os.path.exists(logf):
        import re
        y = sp.symbols('y')
        txt = open(logf, encoding='utf-8').read()
        ok = True
        for ms, expr in re.findall(r"P_(\d+)\s*\(y\) = (.+)", txt):
            m = int(ms)
            if m > MMAX: continue
            ref = sp.Poly(sp.expand(sp.sympify(expr, locals={'y': y})), y)
            mine = sp.Poly(sum(sp.Rational(c) * y ** (len(out[m]) - 1 - i) for i, c in enumerate(out[m])), y)
            if not (ref - mine).is_zero: ok = False; print("  MISMATCH at P_%d" % m)
        print("  agrees with rows_from_recurrence.log for m <= 12: %s" % ok)

if __name__ == "__main__":
    main()
