// volfi_wb.hpp -- the WHOLE-BOOK chart: the exact tau-row series over the entire
// traded book.  v0.3.0-test, EXPERIMENTAL.
// =============================================================================
// WHAT.  The recurrence chart (volfi_near_rec.hpp) knows nothing about h < 0.3: the
// fixed-rho theorem gives the series an h-radius of min(a, 2 pi), never below 2 pi
// in the wing.  With one table of G = 1/V_0 over the whole a-range and a conformal
// variable on the wing side (gen/answer_whole_book.md, A5, tested in
// gen/test_answer_whole_book.py), ONE straight-line kernel covers NEAR, FAR and WING:
//
//   E = expm1(h),  a = log(1 + E/c),   A0 = a G(a),   t = h/A0
//   A  (a < 2 pi):   x = t^2, y = A0^2,  S = sum_{m<=MA} x^m P_m(y)     valid h <= THETA_A a
//   B  (a >= 2 pi):  z = h^2/(4 pi^2),  q = z/(sqrt(1+z)+1)^2,  1/A0 = u F1(u), u = 1/sqrt a, w = 1/A0^2,
//                    S = 1 + sum_{n<=NB} q^n d_n(w)     valid h <= min(H_B, 0.18 a + 0.85)
//   v = t S,  w = v^2
//
// The rows are exact rationals (region A) and exact combinations of them with
// binomials and powers of 4 pi^2 (region B, gen/certified_wb.py); the table of G has
// two cells, [0, 2 pi] in a and [2 pi, A_MAX] in 1/sqrt(a), and the ONE branch at
// a = 2 pi selects both the cell and the series.  Quotes outside (theta above
// THETA_A below 2 pi, h above H_B above it) fall back to the shipped library.
//
// SHARED SOURCE, as for the recurrence chart: host (g++), device (nvcc) and the SIMD
// twins compile this text.  Coordinates are supplied by the caller; nw_expm1 is
// here because the NEAR kernels' 14-term Taylor stops at h = 0.32 and the whole
// book needs expm1 to 1 ULP up to h = 16.5.
// =============================================================================
#ifndef VOLFI_WB_HPP
#define VOLFI_WB_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include "volfi_wb_tables.hpp"

#if defined(__CUDACC__)
#define NW_HD __host__ __device__
#else
#define NW_HD
#endif

namespace volfi_wb {

#if defined(__CUDA_ARCH__)
#define NW_G0C(i) (d_NW_G0[(i)])
#define NW_F1C(i) (d_NW_F1[(i)])
#define NW_PC(i)  (d_NW_P[(i)])
#define NW_DC(i)  (d_NW_D[(i)])
#define NW_FMA(a, b, c) fma((a), (b), (c))
#define NW_SQRT(x) sqrt((x))
#define NW_FLOOR(x) floor((x))
#else
#define NW_G0C(i) (NW_G0[(i)])
#define NW_F1C(i) (NW_F1[(i)])
#define NW_PC(i)  (NW_P[(i)])
#define NW_DC(i)  (NW_D[(i)])
#define NW_FMA(a, b, c) std::fma((a), (b), (c))
#define NW_SQRT(x) std::sqrt((x))
#define NW_FLOOR(x) std::floor((x))
#endif

// --- expm1 on (0, 16.5] to about 1 ULP, division-free ------------------------
// h = k ln2 + r with k = floor(h / ln2) and r in [0, ln2): the Cody-Waite split is
// exact (LN2_HI carries 32 trailing zeros, h - k LN2_HI is Sterbenz-exact), the
// 20-term Taylor of expm1(r)/r has all-positive terms (tail 1.3e-23 at ln2), and
// E = 2^k expm1(r) + (2^k - 1) adds two non-negative terms.  For h < ln2 it is the
// bare Taylor, so the NEAR band costs 6 fma more than nc_expm1 and nothing else.
static constexpr double NW_LN2_HI  = 6.93147180369123816490e-01;
static constexpr double NW_LN2_LO  = 1.90821492927058770002e-10;
static constexpr double NW_INV_LN2 = 1.44269504088896340736;
static constexpr double NWE_C[20] = {
  1.00000000000000000e+00, 5.00000000000000000e-01, 1.66666666666666667e-01,
  4.16666666666666667e-02, 8.33333333333333333e-03, 1.38888888888888889e-03,
  1.98412698412698413e-04, 2.48015873015873016e-05, 2.75573192239858907e-06,
  2.75573192239858907e-07, 2.50521083854417188e-08, 2.08767569878680990e-09,
  1.60590438368216146e-10, 1.14707455977297247e-11, 7.64716373181981648e-13,
  4.77947733238738530e-14, 2.81145725434552076e-15, 1.56192069685862265e-16,
  8.22063524662432972e-18, 4.11031762331216486e-19,
};
#if defined(__CUDA_ARCH__)
#define NWE_CC(i) (d_NWE_C[(i)])
#else
#define NWE_CC(i) (NWE_C[(i)])
#endif

NW_HD inline double nw_pow2(double k) {                // 2^k for integer-valued k in [0, 1023]
    const uint64_t b = ((uint64_t)((int)k + 1023)) << 52;
    double x;
#if defined(__CUDA_ARCH__)
    x = __longlong_as_double((long long)b);
#else
    std::memcpy(&x, &b, 8);
#endif
    return x;
}

NW_HD inline double nw_expm1(double h) {
    const double k  = NW_FLOOR(h * NW_INV_LN2);
    const double r  = NW_FMA(-k, NW_LN2_HI, h) - k * NW_LN2_LO;
    double p = NWE_CC(19);
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 32
#endif
    for (int j = 18; j >= 1; --j) p = NW_FMA(p, r, NWE_CC(j));
    const double er = NW_FMA(r, p * r, r);              // expm1(r)
    const double pk = nw_pow2(k);
    return NW_FMA(pk, er, pk - 1.0);
}

// --- the table ----------------------------------------------------------------
template<int N, bool CELL1>
NW_HD inline double nw_clen(double xv) {
    const double x2 = 2.0 * xv;
    double e0 = 0.0, e1 = 0.0;
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 64
#endif
    for (int j = N - 1; j >= 1; --j) {
        const double b = NW_FMA(x2, e0, CELL1 ? NW_F1C(j) : NW_G0C(j)) - e1;
        e1 = e0; e0 = b;
    }
    return NW_FMA(xv, e0, CELL1 ? NW_F1C(0) : NW_G0C(0)) - e1;
}

// --- region A: the raw series ------------------------------------------------
template<int M>
NW_HD inline double nw_PA(double y) {
    double p = NW_PC(NWRowA<M>::off);
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 32
#endif
    for (int j = 1; j <= NWRowA<M>::deg; ++j) p = NW_FMA(p, y, NW_PC(NWRowA<M>::off + j));
    return p;
}
template<int M>
NW_HD inline double nw_SA(double x, double y, double s) {
    if constexpr (M < 1) { (void)x; (void)y; return s; }      // S1: rows M..1, S = 1 + x S1
    else return nw_SA<M - 1>(x, y, NW_FMA(s, x, nw_PA<M>(y)));
}

// --- region B: the q-series --------------------------------------------------
// d_n(w) = w (D[n][1] + w (D[n][2] + ... + w D[n][n])), stored highest power first.
template<int N>
NW_HD inline double nw_dB(double w) {
    double p = NW_DC(NWRowB<N>::off);
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 32
#endif
    for (int j = 1; j < NWRowB<N>::deg; ++j) p = NW_FMA(p, w, NW_DC(NWRowB<N>::off + j));
    return p * w;
}
template<int N>
NW_HD inline double nw_SB(double q, double w, double s) {
    if constexpr (N < 1) { (void)q; (void)w; return s; }       // S1: S = 1 + q S1
    else return nw_SB<N - 1>(q, w, NW_FMA(s, q, nw_dB<N>(w)));
}

// --- regions ------------------------------------------------------------------
enum { NW_OUT = 0, NW_A = 1, NW_B = 2 };
NW_HD inline int nw_region(double h, double a) {
    if (a < NW_TWO_PI) return (h <= NW_THETA_A * a) ? NW_A : NW_OUT;
    const double hb = NW_FMA(NW_HB_SLOPE, a, NW_HB_INT);     // the real pair h = +-a bites near 2 pi
    return (h <= NW_H_B && h <= hb && a <= NW_A_MAX) ? NW_B : NW_OUT;
}

// w = v*v in region A (a < 2 pi).  a is clamped to the cell so that a lane of the
// wrong region still computes something finite.
NW_HD inline double nw_variance_A(double h, double a) {
    const double ac = (a < NW_TWO_PI) ? a : NW_TWO_PI;
    const double g  = nw_clen<NW_NG0, false>(NW_FMA(ac, NW_SA0, -1.0));
    const double A0 = ac * g;
    const double t  = h / A0;
    const double x  = t * t;
    const double y  = A0 * A0;
    const double v  = NW_FMA(t * x, nw_SA<NW_MA>(x, y, 0.0), t);
    return v * v;
}
// w = v*v in region B (a >= 2 pi).  The cell tabulates F1 = V_0/sqrt(a), flat and O(1)
// where 1/V_0 would fall to 0.05 and cost the Clenshaw ten times its relative error;
// then u F1 = 1/A0 directly, so t and w need no division.
NW_HD inline double nw_variance_B(double h, double a) {
    const double ac = (a > NW_TWO_PI) ? ((a < NW_A_MAX) ? a : NW_A_MAX) : NW_TWO_PI;
    const double u  = 1.0 / NW_SQRT(ac);
    const double f1 = nw_clen<NW_NF1, true>(NW_FMA(u, NW_SU1, NW_BU1));
    const double pp = u * f1;                                  // 1/A0
    const double t  = h * pp;
    const double w  = pp * pp;
    const double z  = h * h * NW_INV_4PI2;
    const double sp = NW_SQRT(1.0 + z) + 1.0;
    const double q  = z / (sp * sp);
    const double v  = NW_FMA(t * q, nw_SB<NW_NB>(q, w, 0.0), t);
    return v * v;
}
NW_HD inline double nw_variance(double h, double a, int region) {
    return (region == NW_A) ? nw_variance_A(h, a) : nw_variance_B(h, a);
}

#undef NW_G0C
#undef NW_F1C
#undef NW_PC
#undef NW_DC
#undef NW_FMA
#undef NW_SQRT
#undef NW_FLOOR
#undef NWE_CC

} // namespace volfi_wb

// ---------------------------------------------------------------------------
// Host wrapper: the whole-book chart with the shipped library as the fallback.
// Returns w; *code = 1 (A), 2 (B), 0 (answered by the shipped scalar entry).
// ---------------------------------------------------------------------------
#include "volfi_near_certified.hpp"
namespace volfi_wb {

static constexpr double NW_H_MAX = 16.5;              // the box's moneyness limit

inline double implied_variance_wb(double h, double c, int* code) {
    namespace nc = volfi_near_certified;
    if (!(c > 0.0) || c >= 1.0 || !(h >= nc::NC_H_FLOOR) || !(h <= NW_H_MAX)) {
        if (code) *code = NW_OUT;
        return volfi_annulus::implied_variance_otm(h, c);
    }
    const double E = nw_expm1(h);
    const double a = nc::log1p_pos(E / c);
    const int r = nw_region(h, a);
    if (code) *code = r;
    if (r == NW_OUT) return (volfi_annulus::fastroute::upper_certain(h, a) == 1) ? volfi_annulus::br::upper_variance(h, c)
                                                                                : volfi_annulus::implied_variance_otm(h, c);
    return nw_variance(h, a, r);
}

} // namespace volfi_wb
#endif // VOLFI_WB_HPP
