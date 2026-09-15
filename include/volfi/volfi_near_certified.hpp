// volfi_near_certified.hpp   --   v0.3.0-test  CERTIFIED NEAR chart and router
// ===========================================================================
// EXPERIMENTAL.  Not part of the shipped library.  Header-only, scalar
// reference implementation; the SIMD/GPU twins come after this passes its gate.
//
// WHAT THIS REPLACES
// ------------------
//   shipped v0.2.4 NEAR quote  =  router  +  br::near_variance
//     router  : cwstar_price (18-coeff Clenshaw + exp_neg) and
//               ctop_price   (32-coeff Clenshaw + exp_neg), BOTH evaluated on
//               every lane of every vector path      -> 82 fma + 2 ldexp
//     chart   : expm1_small(11) -> rho -> binv(full_log + 15/17/25/13 + sqrt)
//               -> s = h/A -> sigma0_poly(erfinv, 18) -> V2(29) -> V4(27)
//               -> clenshaw2(2,16) finisher           -> 176..199 fma scalar,
//                                                        234 fma + 1 sqrt vector
//
//   this file                 =  router  +  near_variance_certified
//     router  : two double compares against frozen constants   -> 0 fma
//     chart   : expm1_small(11) -> q = E/c -> a = log1p(q) -> theta = h/a
//               -> one graded 2-D Chebyshev cell        -> ~100 fma, no sqrt,
//                                                          3 divisions
//
// THE MATHEMATICS (handover Part I Thm 2.1, and Thm 15.1)
// -------------------------------------------------------
// For fixed normalised price rho = c/expm1(k) the physical branch v_rho(k) is
// holomorphic in |k| < R_rho = min(a_rho, 2 pi) with a_rho = log(1 + 1/rho),
// it vanishes at k = 0, and it is ODD:  v_rho(-k) = -v_rho(k)  (eq. 1.3).
// Therefore v/k is even and descends to k^2 (eq. 15.7).  Rescaling k^2 by the
// radius itself,
//         theta = h / a_rho,      tau = theta^2,      v = theta * V(tau, a),
// makes the tau-analyticity radius equal to 1 UNIFORMLY in the price level
// (Regime A, rho >= 1/(e^{2pi}-1) = 1.87e-3), so one Chebyshev design in tau
// serves every price.  V(0,a) = a/Binv(rho) is the h-free inverse the old chart
// computed separately; here it is simply the tau-constant row of the table, so
// Binv, its sqrt and the matched coordinate s all disappear.
//
// EXACT COLLISION SET.  Substituting rho = 1/expm1(a) into the three finite
// collision equations (handover 15.2) gives, with no approximation,
//         Sigma = { k = +-a + 2 pi i n }  U  { k = 2 pi i n },
// so the singular tau at a given a are (+-1 + 2 pi i n/a)^2 and (2 pi i n/a)^2.
// The generator certifies every cell against this exact lattice; that is the
// (k,a) re-coordinatisation of the sufficient product condition (15.14), and it
// is sharper because it uses the lattice rather than the pi-half-strip.
//
// THE TWO SEAMS ARE COORDINATE LINES.  Over the NEAR band h < 0.3:
//   * the certified ceiling v = 1.85 is the iso-theta curve theta = theta* with
//     theta* = erf(1.85/(2 sqrt2)) = 0.6450340925535354; freezing it puts the
//     seam at v in [1.85000, 1.85178]  (the NEAR fit reaches v <= 1.94);
//   * the wing seam W = h^2/(2v^2) = 3.8 is the iso-a curve a = A_WING;
//     freezing it puts the seam at W in [3.7963, 3.8000]  (WING is machine
//     precise for W >= 2.5, and the seam is a definition).
// Both replacements are exact statements about the chart, measured in
// gen/seam_report.txt, and they remove every polynomial from the NEAR router.
// ===========================================================================
#ifndef VOLFI_NEAR_CERTIFIED_HPP
#define VOLFI_NEAR_CERTIFIED_HPP

#include <cmath>
#include "volfi_annulus_wing.hpp"
#include "volfi_annulus.hpp"
#include "volfi_near_certified_tables.hpp"

namespace volfi_near_certified {

// ---------------------------------------------------------------------------
// 1. Shared scalar kernels.
//    The chart REUSES the shipped, already-validated kernels rather than
//    copying their constants: volfi_annulus::br::expm1_small (frozen deg-10
//    Chebyshev of expm1(h)/h on (0,0.32], 1.0e-16 rel) and
//    volfi_annulus::br::full_log (log2approx-based, <=3 ULP, vectorised and
//    scalar==SIMD bit-identical).  That guarantees the experimental chart and
//    the shipped one see identical inputs, so any measured difference is the
//    chart's.
//
//    log1p:  the chart needs a = log(1+q) to ~1 ulp because theta = h/a feeds
//    the result linearly.  Two variants are provided and BOTH are measured by
//    the gate:
//      * NC_LOG1P_LIBM = 1 -> std::log1p (reference; not vectorisable as-is)
//      * NC_LOG1P_LIBM = 0 -> the shared full_log kernel plus the exact
//        Sterbenz correction (vectorisable; inherits full_log's <=3 ULP)
//    Deliverable for the production port: a ~1-ulp vectorisable log1p.  Until
//    that exists the shared variant is the honest cost model and the libm
//    variant is the honest accuracy model.
// ---------------------------------------------------------------------------

// Which log1p the chart uses.  The twins in volfi_near_certified_vec.hpp support
// modes 0 and 2; mode 1 exists only as the scalar accuracy reference.
//   0 = shipped full_log + Sterbenz correction   vectorisable, 2.99 ULP -> chart FAILS
//   1 = std::log1p                               not vectorisable, chart passes
//   2 = nc_log1p (below)                         vectorisable, 1.00 ULP
//   3 = log1p_reduce (below)                     vectorisable, exact reduction on q,
//                                                one division fewer; the default
#ifndef NC_LOG1P_MODE
#define NC_LOG1P_MODE 3
#endif

namespace shipped = volfi_annulus_broadrange;

// --- expm1 ------------------------------------------------------------------
// The chart needs expm1(h) to well under 1 ULP, because a = log1p(expm1(h)/c)
// carries the relative error of expm1 straight through: d(a)/a ~ d(E)/E.  The
// shipped br::expm1_small is a frozen 11-coefficient Chebyshev of expm1(h)/h and
// measures 2.15 ULP worst over h in [1e-12, 0.3) against std::expm1, which is
// fine where it is used (rho feeds binv, whose sensitivity to rho is damped) but
// not here: substituting the exact `a` at the two worst chart points drops the
// error from 5.946 and 2.709 ULP to 1.081 and 1.084.
//
// This replacement is the plain Taylor series of expm1(h)/h in the MONOMIAL
// basis, Horner from the top.  All coefficients are positive and h > 0, so there
// is no cancellation anywhere, and the leading 1 + h/2 is formed last, which
// caps the relative error near half an ulp.  Fourteen terms give a tail below
// 9.3e-20 on [0, 0.32].  It is also cheaper: 14 fma against the Chebyshev's 10
// Clenshaw steps of two operations each plus an affine division.
static const double NCE_C[14] = {
  1.00000000000000000e+00, 5.00000000000000000e-01, 1.66666666666666667e-01,
  4.16666666666666667e-02, 8.33333333333333333e-03, 1.38888888888888889e-03,
  1.98412698412698413e-04, 2.48015873015873016e-05, 2.75573192239858907e-06,
  2.75573192239858907e-07, 2.50521083854417188e-08, 2.08767569878680990e-09,
  1.60590438368216146e-10, 1.14707455977297247e-11,
};
inline double nc_expm1(double h) {                     // 0 < h <= 0.32
    double p = NCE_C[13];
    for (int k = 12; k >= 1; --k) p = std::fma(p, h, NCE_C[k]);
    return std::fma(h, p * h, h);                      // h*(1 + h*p) with the 1 folded in
}

#ifndef NC_EXPM1_SHIPPED
#define NC_EXPM1_SHIPPED 0
#endif
inline double expm1_small(double h) {
#if NC_EXPM1_SHIPPED
    return volfi_annulus::br::expm1_small(h);
#else
    return nc_expm1(h);
#endif
}

// --- mode 0: the shipped kernel with the exact Sterbenz correction ----------
//   u = fl(1+q);  d = fl(u-1);  the part of q lost in the sum is (q-d), and
//   log(1+q) = log(u) + log1p((q-d)/u) = log(u) + (q-d)/u + O(2^-106).
// At q < 2^-53 this returns exactly q (u == 1 so full_log(u) == 0 and d == 0).
// The correction is exact; the residual 2.99 ULP is full_log's own error, which
// is why this mode overruns the chart's 0.72 ULP of remaining budget.
inline double log1p_shared(double q) {
    const double u = 1.0 + q;
    const double d = u - 1.0;
    return volfi_annulus::full_log(u) + (q - d) / u;
}

// --- mode 2: an accurate, vectorisable natural log --------------------------
// Argument reduction  x = 2^k m,  m in [1/sqrt2, sqrt2),  then the standard
// odd-atanh form in s = f/(2+f), f = m-1, with a two-word ln2.  The polynomial
// is the classical seven-term minimax used by fdlibm's __ieee754_log; the
// branchless "always hfsq" assembly is algebraically identical to fdlibm's other
// branch, since s*f == hfsq - s*hfsq exactly in exact arithmetic.
// Every step is a blend or an fma, so the SIMD twin reproduces it bit for bit.
static const double NCL_LN2_HI = 6.93147180369123816490e-01;
static const double NCL_LN2_LO = 1.90821492927058770002e-10;
static const double NCL_SQRT2  = 1.41421356237309504880;
static const double NCL_LG[7]  = {
    6.666666666666735130e-01, 3.999999999940941908e-01, 2.857142874366239149e-01,
    2.222219843214978396e-01, 1.818357216161805012e-01, 1.531383769920937332e-01,
    1.479819860511658591e-01,
};

inline double nc_log(double x) {                       // x > 0, normal
    uint64_t b = volfi_annulus::detail::bits_of(x);
    double   m = volfi_annulus::detail::mant12(b);     // mantissa in [1,2)
    double   k = (double)((int)((b >> 52) & 0x7FFULL) - 1023);   // signed, not an implementation-defined unsigned wrap
    const bool hi = (m > NCL_SQRT2);
    if (hi) { m *= 0.5; k += 1.0; }                    // m in [1/sqrt2, sqrt2)
    const double f = m - 1.0;
    const double s = f / (2.0 + f);
    const double z = s * s;
    const double w = z * z;
    double t1 = std::fma(w, NCL_LG[5], NCL_LG[3]);
    t1 = std::fma(w, t1, NCL_LG[1]);
    t1 = w * t1;
    double t2 = std::fma(w, NCL_LG[6], NCL_LG[4]);
    t2 = std::fma(w, t2, NCL_LG[2]);
    t2 = std::fma(w, t2, NCL_LG[0]);
    t2 = z * t2;
    const double R    = t2 + t1;
    const double hfsq = 0.5 * f * f;
    return std::fma(k, NCL_LN2_HI, -((hfsq - (std::fma(s, hfsq + R, k * NCL_LN2_LO))) - f));
}

inline double log1p_nc(double q) {                     // q > 0
    const double u = 1.0 + q;
    const double d = u - 1.0;
    return nc_log(u) + (q - d) / u;
}

// --- mode 3: exact reduction on q itself --------------------------------------
// log1p(q) = k ln2 + log(m) with 1+q = 2^k m and m in [0.75, 1.5).  The reduced
// argument f = m - 1 = (q - (2^k - 1)) 2^-k is formed from q directly, never from
// the rounded sum 1+q: for k = 0 it is q itself, and for k >= 1 Sterbenz's lemma
// applies to q - (2^k - 1), because 0.75 2^k - 1 >= (2^k - 1)/2 and
// 1.5 2^k - 1 <= 2 (2^k - 1) hold for every k >= 1.  So f is exact, the Sterbenz
// correction (q - d)/u of modes 0 and 2 disappears, and with it one division and
// three other operations.  The rounded u = 1+q is used only to pick k; a q that
// rounds across the m = 1.5 boundary lands one ulp outside [0.75, 1.5), well
// inside the polynomial's fitted window, and the one case where q - (2^k - 1) can
// round (q one ulp below 0.5) contributes 0.04 ULP, because log(1+q) is 0.4 there.
// The window |s| <= 0.2 (fdlibm's is 0.1716) needs its own fit of R(z) = z P(z):
// eight terms on z in [0, 0.0425], 6.6e-17 relative in P, the double-rounding floor
// of 2/3, and below 1e-18 in log(1+f).  See gen/fit_log_poly.py.
static const double NCL3_P[8] = {
    6.6666666666666663e-1, 4.0000000000012037e-1, 2.8571428565490653e-1,
    2.2222223336620348e-1, 1.8181715682579969e-1, 1.5389718263641888e-1,
    1.3193475586569775e-1, 1.3730049006664399e-1,
};
inline double log1p_reduce(double q) {                 // q > 0, 1+q normal
    const double   u = 1.0 + q;                        // only its exponent and half-bit are used
    const uint64_t b = volfi_annulus::detail::bits_of(u);
    const double   m = volfi_annulus::detail::mant12(b);
    int k = (int)((b >> 52) & 0x7FFULL) - 1023;
    if (m >= 1.5) k += 1;                              // 1+q in [0.75 2^k, 1.5 2^k)
    const double pk  = volfi_annulus::detail::from_bits((uint64_t)(k + 1023) << 52);   // 2^k
    const double ipk = volfi_annulus::detail::from_bits((uint64_t)(1023 - k) << 52);   // 2^-k
    const double f = (q - (pk - 1.0)) * ipk;           // exact: Sterbenz, see above
    const double s = f / (2.0 + f);
    const double z = s * s;
    const double w = z * z;
    double t1 = std::fma(w, NCL3_P[7], NCL3_P[5]);     // even powers of z in R
    t1 = std::fma(w, t1, NCL3_P[3]);
    t1 = std::fma(w, t1, NCL3_P[1]);
    t1 = w * t1;
    double t2 = std::fma(w, NCL3_P[6], NCL3_P[4]);     // odd powers of z in R
    t2 = std::fma(w, t2, NCL3_P[2]);
    t2 = std::fma(w, t2, NCL3_P[0]);
    t2 = z * t2;
    const double R    = t2 + t1;
    const double hfsq = 0.5 * f * f;
    const double kd   = (double)k;
    return std::fma(kd, NCL_LN2_HI, -((hfsq - (std::fma(s, hfsq + R, kd * NCL_LN2_LO))) - f));
}

inline double log1p_pos(double q) {
#if   NC_LOG1P_MODE == 1
    return std::log1p(q);
#elif NC_LOG1P_MODE == 0
    return log1p_shared(q);
#elif NC_LOG1P_MODE == 2
    return log1p_nc(q);
#else
    return log1p_reduce(q);
#endif
}

// ---------------------------------------------------------------------------
// 2. Cell lookup.  NC_A_EDGE is ascending with NC_A_EDGE[0] == 0, so a linear
//    scan is exact and branch-predictable; the SIMD twin will use the same
//    edges with a vectorised compare-and-count (no gather for the index).
// ---------------------------------------------------------------------------
inline int cell_of(double a) {
    int j = 0;
    while (j < NC_NCELL - 1 && a >= NC_A_EDGE[j + 1]) ++j;
    return j;
}

// ---------------------------------------------------------------------------
// 3. Graded 2-D Chebyshev.  Row m (order m in tau) carries NC_ROWDEG[.] a-coeffs.
//    Cost = sum_m NC_ROWDEG[m] + nrow fma, i.e. the triangular count, not the
//    rectangle: the a-resolution needed falls off with the tau-order because
//    the tau-coefficients themselves fall off geometrically at rate rho_tau.
// ---------------------------------------------------------------------------
inline double clenshaw2_graded(int cell, double xt, double xa) {
    const int    nrow = NC_NROW[cell];
    const int    roff = NC_ROWOFF[cell];
    const double xa2  = 2.0 * xa;
    const double xt2  = 2.0 * xt;
    double g0 = 0.0, g1 = 0.0;                       // outer (tau) Clenshaw state
    for (int m = nrow - 1; m >= 1; --m) {
        const int    nc = NC_ROWDEG[roff + m];
        const double* C = NC_COEFF + NC_ROWCOFF[roff + m];
        double e0 = 0.0, e1 = 0.0;                   // inner (a) Clenshaw state
        for (int j = nc - 1; j >= 1; --j) { double b = std::fma(xa2, e0, C[j]) - e1; e1 = e0; e0 = b; }
        const double r = (nc > 0) ? (std::fma(xa, e0, C[0]) - e1) : 0.0;
        const double b = std::fma(xt2, g0, r) - g1;
        g1 = g0; g0 = b;
    }
    {   // m = 0 row, then the Clenshaw finish
        const int    nc = NC_ROWDEG[roff];
        const double* C = NC_COEFF + NC_ROWCOFF[roff];
        double e0 = 0.0, e1 = 0.0;
        for (int j = nc - 1; j >= 1; --j) { double b = std::fma(xa2, e0, C[j]) - e1; e1 = e0; e0 = b; }
        const double r = (nc > 0) ? (std::fma(xa, e0, C[0]) - e1) : 0.0;
        return std::fma(xt, g0, r) - g1;
    }
}

// ---------------------------------------------------------------------------
// 4. The chart.   h in (0, NC_H_NEAR),  0 < c < 1,  theta <= NC_THETA,
//    a < NC_A_WING.  Returns w = v*v.
// ---------------------------------------------------------------------------
struct near_coords {
    double E;       // expm1(h)
    double a;       // log(1 + 1/rho) -- the exact analyticity radius
    double theta;   // h/a
    double tau;     // theta^2
};

inline near_coords near_coords_of(double h, double c) {
    near_coords z;
    z.E     = expm1_small(h);
    z.a     = log1p_pos(z.E / c);
    z.theta = h / z.a;
    z.tau   = z.theta * z.theta;
    return z;
}

inline double near_variance_from_coords(const near_coords& z) {
    const int    cell = cell_of(z.a);
    const double t1   = NC_TAU_HI[cell];
    const double a0   = NC_A_EDGE[cell], a1 = NC_A_EDGE[cell + 1];
    const double xt   = std::fma(2.0, z.tau, -t1) / t1;          // tau lower edge is 0
    const double xa   = std::fma(2.0, z.a, -(a0 + a1)) / (a1 - a0);
    const double v    = z.theta * clenshaw2_graded(cell, xt, xa);
    return v * v;
}

inline double near_variance_certified(double h, double c) {
    return near_variance_from_coords(near_coords_of(h, c));
}

// ---------------------------------------------------------------------------
// 5. The router for the NEAR band.  No polynomial, no transcendental beyond the
//    one log1p the chart needs anyway.  Codes match the shipped library:
//        0 = scalar/edge fallback, 1 = NEAR, 2 = UPPER, 3 = WING.
//    Only h < NC_H_NEAR is handled here; the wider box keeps the shipped seams.
// ---------------------------------------------------------------------------
enum { NCR_EDGE = 0, NCR_NEAR = 1, NCR_UPPER = 2, NCR_WING = 3, NCR_OUTOFBAND = -1 };

// The wing test in its overflow-free form.  a >= NC_A_WING is exactly
// rho <= RHO_WING, i.e. c * expm1(NC_A_WING) <= expm1(h), because a, A = Binv(rho)
// and rho are each functions of rho alone (see volfi_router_dropin.hpp).  Testing it
// that way FIRST matters: for c below expm1(h)/DBL_MAX -- 5.85e-310 at h = 0.1 --
// the quotient q = E/c overflows to +inf, so u and d are both inf, (q-d) is NaN and
// `a` comes back NaN.  Review found the old ordering returning NEAR with w = NaN at
// h = 0.1, c = 5e-324, where the true a is 742.19 and the quote is plainly WING.
static const double NC_INV_RHO_WING = 3135.8325037881695;   // = expm1(NC_A_WING)

// The chart returns w = v*v, and below v ~ 1e-155 the square loses bits, then
// underflows to zero below 2^-537 (review item 8; shared with the shipped API).
// The NEAR band therefore carries an explicit floor on h: with h >= 2^-500 and
// a < A_WING = 8.05 one has theta = h/a > 2^-504, and V >= 1.9 on the whole chart
// gives v > 2^-504 and w > 2^-1008, a normal double with fourteen bits to spare
// above DBL_MIN.  Below the floor the quote is EDGE and the caller's scalar path
// answers it; no traded quote is within three hundred orders of magnitude of it.
static const double NC_H_FLOOR = 0x1p-500;

inline int route_near_band(double h, double c, near_coords* z_out) {
    if (!(c > 0.0) || c >= 1.0)   return NCR_EDGE;
    if (!(h >= NC_H_FLOOR))       return NCR_EDGE;     // also NaN and h <= 0
    if (!(h < NC_H_NEAR))         return NCR_OUTOFBAND;
    const double E = expm1_small(h);
    // WING is a >= NC_A_WING, i.e. rho <= RHO_WING, i.e. c*INV_RHO_WING <= expm1(h).
    // Written with a negated > so that c -> 0 (where E/c overflows) lands here.
    if (!(c * NC_INV_RHO_WING > E)) return NCR_WING;
    near_coords z = near_coords_of(h, c);
    if (z_out) *z_out = z;
    if (z.a >= NC_A_WING)         return NCR_WING;     // W in [3.7963, 3.8000]
    if (z.theta > NC_THETA)       return NCR_UPPER;    // v in [1.85000, 1.85178]
    return NCR_NEAR;
}

// One-shot convenience: route and, when the answer is NEAR, evaluate.
// Returns w, or a negative code in *route when the quote is not NEAR.
inline double implied_variance_near_band(double h, double c, int* route) {
    near_coords z;
    int r = route_near_band(h, c, &z);
    if (route) *route = r;
    if (r != NCR_NEAR) return 0.0;
    return near_variance_from_coords(z);
}

} // namespace volfi_near_certified
#endif // VOLFI_NEAR_CERTIFIED_HPP
