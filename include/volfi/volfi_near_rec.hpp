// volfi_near_rec.hpp -- the RECURRENCE chart: the certified NEAR chart with the tau
// direction done by exact algebra.  v0.3.0-test, EXPERIMENTAL.
// =============================================================================
// WHAT.  From gen/answer_rational_a.md (Q4), verified in gen/rows_from_recurrence.py:
// the tau-Taylor rows of V(tau, a) are V_m = V_0^{2m+1} P_m(A_0^2) with P_m exact
// rational polynomials and A_0 = a / V_0 = Binv(1/expm1(a)); no Gaussian density
// survives.  So with
//
//     A0 = a * G(a),   G = 1/V_0  (the ONE fitted function, 25 Chebyshev terms)
//     t  = h / A0,     x = t*t,   y = A0*A0
//     v  = t * S(x, y),   S = sum_{m=0}^{10} x^m P_m(y)   (55 exact rational fma)
//     w  = v*v
//
// the whole tau direction costs nothing to fit and nothing to certify beyond an
// explicit truncation term.  Compared with the book chart (127 coefficients on one
// cell) the table is 25 doubles, there are no cells, nothing to bucket, nothing to
// diverge on, and the operation count per quote is about 190 with three divisions
// (E/c, the log's own, h/A0).  Validated at 1.85e-17 against the 40-digit solve on
// the book box (gen/certified_rec.py).
//
// DOMAIN.  tau = (h/a)^2 <= NR_TAU_MAX = 0.035 (the feed maximum is 0.0327); above
// it the series slows toward the ceiling and the caller falls back to the 12-cell
// chart.  The in-range test needs no division: h*h <= NR_TAU_MAX * a*a.
//
// SHARED SOURCE.  Compiled for the host by g++, for host and device by nvcc, and
// instantiated on vector lanes by volfi_near_rec_vec.hpp: one text.  Coefficients
// are read through NR_C / NR_CP, which resolve to the __constant__ device copies
// under __CUDA_ARCH__ and to the constexpr host arrays otherwise.
// =============================================================================
#ifndef VOLFI_NEAR_REC_HPP
#define VOLFI_NEAR_REC_HPP

#include <cmath>
#include "volfi_near_rec_tables.hpp"

#if defined(__CUDACC__)
#define NR_HD __host__ __device__
#else
#define NR_HD
#endif

namespace volfi_near_rec {

#if defined(__CUDA_ARCH__)
#define NR_CG(i)  (d_NR_G[(i)])
#define NR_CP(i)  (d_NR_P[(i)])
#define NR_FMA(a, b, c) fma((a), (b), (c))
#else
#define NR_CG(i)  (NR_G[(i)])
#define NR_CP(i)  (NR_P[(i)])
#define NR_FMA(a, b, c) std::fma((a), (b), (c))
#endif

// G(a) = 1/V_0(a): Clenshaw over NR_NG coefficients (c_0 already halved), xa in [-1,1].
NR_HD inline double nr_G(double xa) {
    const double xa2 = 2.0 * xa;
    double e0 = 0.0, e1 = 0.0;
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 64
#endif
    for (int j = NR_NG - 1; j >= 1; --j) {
        const double b = NR_FMA(xa2, e0, NR_CG(j)) - e1;
        e1 = e0; e0 = b;
    }
    return NR_FMA(xa, e0, NR_CG(0)) - e1;
}

// P_m(y), Horner from the highest degree.
template<int M>
NR_HD inline double nr_P(double y) {
    double p = NR_CP(NRRow<M>::off);
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (int j = 1; j <= NRRow<M>::deg; ++j) p = NR_FMA(p, y, NR_CP(NRRow<M>::off + j));
    return p;
}

// S = (((P_M x + P_{M-1}) x + ...) x + P_0), recursion from M down to 0.
template<int M>
NR_HD inline double nr_S(double x, double y, double s) {
    if constexpr (M < 0) {
        (void)x; (void)y; return s;
    } else {
        return nr_S<M - 1>(x, y, NR_FMA(s, x, nr_P<M>(y)));
    }
}

// w from (h, a).  No range test here; see nr_in_range.
NR_HD inline double nr_variance(double h, double a) {
    const double xa = NR_FMA(a, NR_SA, -1.0);
    const double g  = nr_G(xa);
    const double A0 = a * g;
    const double t  = h / A0;
    const double x  = t * t;
    const double y  = A0 * A0;
    const double S  = nr_S<NR_M>(x, y, 0.0);
    const double v  = t * S;
    return v * v;
}

// (h/a)^2 <= NR_TAU_MAX without a division.
NR_HD inline bool nr_in_range(double h, double a) {
    return h * h <= NR_TAU_MAX * (a * a);
}

#undef NR_CG
#undef NR_CP
#undef NR_FMA

} // namespace volfi_near_rec

// ---------------------------------------------------------------------------
// Host wrapper: the recurrence chart with the 12-cell fallback above its tau
// range, on the standard router's coordinates.  The CPU reference the device and
// SIMD results must match bit for bit.
// ---------------------------------------------------------------------------
#include "volfi_near_certified.hpp"
namespace volfi_near_rec {

inline double near_variance_rec(double h, double c, int* used_fallback = nullptr) {
    namespace nc = volfi_near_certified;
    const nc::near_coords z = nc::near_coords_of(h, c);
    if (nr_in_range(h, z.a)) { if (used_fallback) *used_fallback = 0; return nr_variance(h, z.a); }
    if (used_fallback) *used_fallback = 1;
    return nc::near_variance_from_coords(z);
}

} // namespace volfi_near_rec
#endif // VOLFI_NEAR_REC_HPP
