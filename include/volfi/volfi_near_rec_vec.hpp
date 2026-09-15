// volfi_near_rec_vec.hpp -- AVX-512 and AVX2 twins of the recurrence chart.
// v0.3.0-test, EXPERIMENTAL.
// =============================================================================
// One interval, one 25-term Clenshaw and 55 exact fma per lane: no counting sort,
// no cells, no scatter.  Lane by lane the same operation sequence as
// volfi_near_rec::near_variance_rec, so the results are bit-identical to the
// scalar chart and, through it, to the CUDA kernel.  Lanes above the tau range
// (none on the feed) are answered afterwards by the scalar 12-cell chart.
// Include after volfi_near_certified_vec.hpp (vector coordinate kernels).
// =============================================================================
#ifndef VOLFI_NEAR_REC_VEC_HPP
#define VOLFI_NEAR_REC_VEC_HPP

#include "volfi_near_rec.hpp"
#include "volfi_near_certified_vec.hpp"

namespace volfi_near_rec {

namespace nv = volfi_near_certified::vec;
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

#if defined(VA_SIMD512)
struct R8 {
    typedef __m512d T;
    static inline T bcast(double x)      { return _mm512_set1_pd(x); }
    static inline T zero()               { return _mm512_setzero_pd(); }
    static inline T fma(T a, T b, T c)   { return _mm512_fmadd_pd(a, b, c); }
    static inline T sub(T a, T b)        { return _mm512_sub_pd(a, b); }
    static inline T mul(T a, T b)        { return _mm512_mul_pd(a, b); }
    static inline T div(T a, T b)        { return _mm512_div_pd(a, b); }
};
#endif
#if defined(VA_SIMD256)
struct R4 {
    typedef __m256d T;
    static inline T bcast(double x)      { return _mm256_set1_pd(x); }
    static inline T zero()               { return _mm256_setzero_pd(); }
    static inline T fma(T a, T b, T c)   { return _mm256_fmadd_pd(a, b, c); }
    static inline T sub(T a, T b)        { return _mm256_sub_pd(a, b); }
    static inline T mul(T a, T b)        { return _mm256_mul_pd(a, b); }
    static inline T div(T a, T b)        { return _mm256_div_pd(a, b); }
};
#endif

template<class V>
inline typename V::T nrv_G(typename V::T xa) {
    const typename V::T xa2 = V::mul(V::bcast(2.0), xa);
    typename V::T e0 = V::zero(), e1 = V::zero();
#pragma GCC unroll 64
    for (int j = NR_NG - 1; j >= 1; --j) {
        const typename V::T b = V::sub(V::fma(xa2, e0, V::bcast(NR_G[j])), e1);
        e1 = e0; e0 = b;
    }
    return V::sub(V::fma(xa, e0, V::bcast(NR_G[0])), e1);
}

template<class V, int M>
inline typename V::T nrv_P(typename V::T y) {
    typename V::T p = V::bcast(NR_P[NRRow<M>::off]);
#pragma GCC unroll 16
    for (int j = 1; j <= NRRow<M>::deg; ++j) p = V::fma(p, y, V::bcast(NR_P[NRRow<M>::off + j]));
    return p;
}

template<class V, int M>
inline typename V::T nrv_S(typename V::T x, typename V::T y, typename V::T s) {
    if constexpr (M < 0) {
        (void)x; (void)y; return s;
    } else {
        return nrv_S<V, M - 1>(x, y, V::fma(s, x, nrv_P<V, M>(y)));
    }
}

// w for all lanes from (h, a); mirror of nr_variance.
template<class V>
inline typename V::T nrv_variance(typename V::T h, typename V::T a) {
    const typename V::T xa = V::fma(a, V::bcast(NR_SA), V::bcast(-1.0));
    const typename V::T g  = nrv_G<V>(xa);
    const typename V::T A0 = V::mul(a, g);
    const typename V::T t  = V::div(h, A0);
    const typename V::T x  = V::mul(t, t);
    const typename V::T y  = V::mul(A0, A0);
    const typename V::T S  = nrv_S<V, NR_M>(x, y, V::zero());
    const typename V::T v  = V::mul(t, S);
    return V::mul(v, v);
}

// ===========================================================================
// The batch driver.  route[i] uses the scalar codes:
//   -1 out of the NEAR band, 0 edge, 1 NEAR, 2 UPPER, 3 WING.
// w[i] is written only where route[i] == 1; elsewhere it is left untouched.
// Routing uses theta = h/a as the scalar router does, so routes are bit-identical
// to route_near_band; the chart itself does not need theta.
// ===========================================================================
inline void near_variance_rec_batch(const double* h, const double* c,
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
            _mm512_storeu_pd(sa + i, a);
            _mm512_storeu_pd(sth + i, th);
            _mm512_storeu_pd(stau + i, _mm512_mul_pd(th, th));
            _mm512_storeu_pd(sw + i, nrv_variance<R8>(vh, a));
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
            _mm256_storeu_pd(sa + i, a);
            _mm256_storeu_pd(sth + i, th);
            _mm256_storeu_pd(stau + i, _mm256_mul_pd(th, th));
            _mm256_storeu_pd(sw + i, nrv_variance<R4>(vh, a));
        }
#endif
        for (; i < m; ++i) {                       // scalar tail, identical kernel
            const double sh = std::fmin(std::fmax(h[base + i], NC_TINY), NC_HHI);
            const double sc = std::fmin(std::fmax(c[base + i], NC_TINY), NC_CHI);
            nc::near_coords z = nc::near_coords_of(sh, sc);
            sa[i] = z.a; sth[i] = z.theta; stau[i] = z.tau;
            sw[i] = nr_variance(sh, z.a);
        }
        for (i = 0; i < m; ++i) {
            const double cc = c[base + i], hh = h[base + i];
            int r;
            if (!(cc > 0.0) || cc >= 1.0)        r = NCR_EDGE;
            else if (!(hh >= NC_H_FLOOR))        r = NCR_EDGE;
            else if (!(hh < NC_H_NEAR))          r = NCR_OUTOFBAND;
            else if (!(cc * NC_INV_RHO_WING > nc::expm1_small(hh))) r = NCR_WING;
            else if (sa[i] >= NC_A_WING)         r = NCR_WING;
            else if (sth[i] > NC_THETA)          r = NCR_UPPER;
            else                                 r = NCR_NEAR;
            route[base + i] = r;
            if (r != NCR_NEAR) continue;
            if (nr_in_range(hh, sa[i])) {
                w[base + i] = sw[i];
            } else {                               // above the series range: the 12-cell chart
                nc::near_coords z; z.E = 0.0; z.a = sa[i]; z.theta = sth[i]; z.tau = stau[i];
                w[base + i] = nc::near_variance_from_coords(z);
            }
        }
    }
}

} // namespace volfi_near_rec
#endif // VOLFI_NEAR_REC_VEC_HPP
