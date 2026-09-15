// volfi_near_book_vec.hpp -- AVX-512 and AVX2 twins of the book chart.
// v0.3.0-test, EXPERIMENTAL.
// =============================================================================
// The book chart has ONE cell (layout L1), so its SIMD driver needs no counting
// sort, no bucket scratch and no scatter: every lane runs the same straight-line
// Clenshaw, which is the shape the AVX-512 twin of the 12-cell chart lacked and
// the reason it lost at small batches (0.58x at 64 quotes).  Lanes above the
// book's tau range, which do not occur on the feed, are answered afterwards by
// the scalar 12-cell chart.
//
// CONTRACT.  Lane by lane, the same operation sequence as
// volfi_near_book::near_variance_book, so the results are bit-identical to the
// scalar chart and, through it, to the CUDA kernel.
//
// Requires NB_LAYOUT == 1.  Include after volfi_near_certified_vec.hpp, which
// provides the vector coordinate kernels (expm1_small_*, log1p_pos_*).
// =============================================================================
#ifndef VOLFI_NEAR_BOOK_VEC_HPP
#define VOLFI_NEAR_BOOK_VEC_HPP

#include "volfi_near_book.hpp"
#include "volfi_near_certified_vec.hpp"

#if NB_LAYOUT != 1
#error "volfi_near_book_vec.hpp supports the one-cell layout only (NB_LAYOUT=1)"
#endif

namespace volfi_near_book {

namespace nv = volfi_near_certified::vec;      // the vector coordinate kernels and the scrub bounds
using nv::NC_TINY;
using nv::NC_CHI;
using nv::NC_HHI;
using volfi_near_certified::NC_H_NEAR;
using volfi_near_certified::NC_H_FLOOR;
using volfi_near_certified::NC_INV_RHO_WING;
using volfi_near_certified::NC_A_WING;
using volfi_near_certified::NC_THETA;
using volfi_near_certified::NCR_EDGE;
using volfi_near_certified::NCR_NEAR;
using volfi_near_certified::NCR_UPPER;
using volfi_near_certified::NCR_WING;
using volfi_near_certified::NCR_OUTOFBAND;

// ---------------------------------------------------------------------------
// A tiny vector-op layer so the templated Clenshaw of volfi_near_book.hpp can be
// written once for both widths.  fma/sub/bcast map to the lane-exact intrinsics.
// ---------------------------------------------------------------------------
#if defined(VA_SIMD512)
struct V8 {
    typedef __m512d T;
    static inline T bcast(double x)      { return _mm512_set1_pd(x); }
    static inline T zero()               { return _mm512_setzero_pd(); }
    static inline T fma(T a, T b, T c)   { return _mm512_fmadd_pd(a, b, c); }
    static inline T sub(T a, T b)        { return _mm512_sub_pd(a, b); }
    static inline T mul(T a, T b)        { return _mm512_mul_pd(a, b); }
};
#endif
#if defined(VA_SIMD256)
struct V4 {
    typedef __m256d T;
    static inline T bcast(double x)      { return _mm256_set1_pd(x); }
    static inline T zero()               { return _mm256_setzero_pd(); }
    static inline T fma(T a, T b, T c)   { return _mm256_fmadd_pd(a, b, c); }
    static inline T sub(T a, T b)        { return _mm256_sub_pd(a, b); }
    static inline T mul(T a, T b)        { return _mm256_mul_pd(a, b); }
};
#endif

template<class V, int DEG, int OFF>
inline typename V::T nbv_row(typename V::T xa, typename V::T xa2) {
    if constexpr (DEG <= 0) {
        (void)xa; (void)xa2;
        return V::zero();
    } else {
        typename V::T e0 = V::zero(), e1 = V::zero();
#pragma GCC unroll 64
        for (int j = DEG - 1; j >= 1; --j) {
            const typename V::T b = V::sub(V::fma(xa2, e0, V::bcast(NB_COEFF[OFF + j])), e1);
            e1 = e0; e0 = b;
        }
        return V::sub(V::fma(xa, e0, V::bcast(NB_COEFF[OFF])), e1);
    }
}

template<class V, int CELL, int M>
inline void nbv_rows(typename V::T xt2, typename V::T xa, typename V::T xa2,
                     typename V::T& g0, typename V::T& g1) {
    if constexpr (M >= 1) {
        const typename V::T r = nbv_row<V, NBRow<CELL, M>::deg, NBRow<CELL, M>::off>(xa, xa2);
        const typename V::T b = V::sub(V::fma(xt2, g0, r), g1);
        g1 = g0; g0 = b;
        nbv_rows<V, CELL, M - 1>(xt2, xa, xa2, g0, g1);
    } else {
        (void)xt2; (void)xa; (void)xa2; (void)g0; (void)g1;
    }
}

// V(tau, a) on cell 0, all lanes; mirror of nb_V<0>.
template<class V>
inline typename V::T nbv_V0(typename V::T tau, typename V::T a) {
    const typename V::T xt  = V::fma(tau, V::bcast(NBCell<0>::st), V::bcast(-1.0));
    const typename V::T xa  = V::fma(a,   V::bcast(NBCell<0>::sa), V::bcast(NBCell<0>::ba));
    const typename V::T xt2 = V::mul(V::bcast(2.0), xt);
    const typename V::T xa2 = V::mul(V::bcast(2.0), xa);
    typename V::T g0 = V::zero(), g1 = V::zero();
    nbv_rows<V, 0, NBCell<0>::nrow - 1>(xt2, xa, xa2, g0, g1);
    const typename V::T r = nbv_row<V, NBRow<0, 0>::deg, NBRow<0, 0>::off>(xa, xa2);
    return V::sub(V::fma(xt, g0, r), g1);
}

// ===========================================================================
// The batch driver.  route[i] uses the scalar codes:
//   -1 out of the NEAR band, 0 edge, 1 NEAR, 2 UPPER, 3 WING.
// w[i] is written only where route[i] == 1; elsewhere it is left untouched.
// One pass: coordinates, chart and store, lane-parallel; then the routing loop,
// which also answers the rare above-book lanes through the scalar 12-cell chart.
// ===========================================================================
inline void near_variance_book_batch(const double* h, const double* c,
                                     double* w, int* route, int n) {
    namespace nc = volfi_near_certified;
    constexpr int TILE = NC_VEC_TILE;
    static thread_local double sa[TILE], sth[TILE], stau[TILE], sw[TILE];

    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;
        int i = 0;
#if defined(VA_SIMD512)
        for (; i + 8 <= m; i += 8) {
            __m512d vh = _mm512_loadu_pd(h + base + i);
            __m512d vc = _mm512_loadu_pd(c + base + i);
            vc = _mm512_min_pd(_mm512_max_pd(vc, _mm512_set1_pd(NC_TINY)), _mm512_set1_pd(NC_CHI));
            vh = _mm512_min_pd(_mm512_max_pd(vh, _mm512_set1_pd(NC_TINY)), _mm512_set1_pd(NC_HHI));
            __m512d E  = nv::expm1_small_avx512(vh);
            __m512d q  = _mm512_div_pd(E, vc);
            __m512d a  = nv::log1p_pos_avx512(q);
            __m512d th = _mm512_div_pd(vh, a);
            __m512d tu = _mm512_mul_pd(th, th);
            __m512d v  = _mm512_mul_pd(th, nbv_V0<V8>(tu, a));
            _mm512_storeu_pd(sa + i, a);
            _mm512_storeu_pd(sth + i, th);
            _mm512_storeu_pd(stau + i, tu);
            _mm512_storeu_pd(sw + i, _mm512_mul_pd(v, v));
        }
#elif defined(VA_SIMD256)
        for (; i + 4 <= m; i += 4) {
            __m256d vh = _mm256_loadu_pd(h + base + i);
            __m256d vc = _mm256_loadu_pd(c + base + i);
            vc = _mm256_min_pd(_mm256_max_pd(vc, _mm256_set1_pd(NC_TINY)), _mm256_set1_pd(NC_CHI));
            vh = _mm256_min_pd(_mm256_max_pd(vh, _mm256_set1_pd(NC_TINY)), _mm256_set1_pd(NC_HHI));
            __m256d E  = nv::expm1_small_avx2(vh);
            __m256d q  = _mm256_div_pd(E, vc);
            __m256d a  = nv::log1p_pos_avx2(q);
            __m256d th = _mm256_div_pd(vh, a);
            __m256d tu = _mm256_mul_pd(th, th);
            __m256d v  = _mm256_mul_pd(th, nbv_V0<V4>(tu, a));
            _mm256_storeu_pd(sa + i, a);
            _mm256_storeu_pd(sth + i, th);
            _mm256_storeu_pd(stau + i, tu);
            _mm256_storeu_pd(sw + i, _mm256_mul_pd(v, v));
        }
#endif
        for (; i < m; ++i) {                       // scalar tail, identical kernel
            const double sh = std::fmin(std::fmax(h[base + i], NC_TINY), NC_HHI);
            const double sc = std::fmin(std::fmax(c[base + i], NC_TINY), NC_CHI);
            nc::near_coords z = nc::near_coords_of(sh, sc);
            sa[i] = z.a; sth[i] = z.theta; stau[i] = z.tau;
            const double v = z.theta * nb_V<0>(z.tau, z.a);
            sw[i] = v * v;
        }

        // ---- routing from the ORIGINAL h and c; NEAR lanes take the chart -----
        for (i = 0; i < m; ++i) {
            const double cc = c[base + i], hh = h[base + i];
            int r;
            if (!(cc > 0.0) || cc >= 1.0)        r = NCR_EDGE;
            else if (!(hh >= NC_H_FLOOR))        r = NCR_EDGE;     // the w = v*v floor, see the scalar router
            else if (!(hh < NC_H_NEAR))          r = NCR_OUTOFBAND;
            else if (!(cc * NC_INV_RHO_WING > nc::expm1_small(hh))) r = NCR_WING;
            else if (sa[i] >= NC_A_WING)         r = NCR_WING;
            else if (sth[i] > NC_THETA)          r = NCR_UPPER;
            else                                 r = NCR_NEAR;
            route[base + i] = r;
            if (r != NCR_NEAR) continue;
            if (stau[i] <= NBCell<0>::t1) {
                w[base + i] = sw[i];
            } else {                               // above the book: the 12-cell chart
                nc::near_coords z; z.E = 0.0; z.a = sa[i]; z.theta = sth[i]; z.tau = stau[i];
                w[base + i] = nc::near_variance_from_coords(z);
            }
        }
    }
}

} // namespace volfi_near_book
#endif // VOLFI_NEAR_BOOK_VEC_HPP
