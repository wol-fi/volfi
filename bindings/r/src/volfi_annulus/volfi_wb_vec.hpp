// volfi_wb_vec.hpp -- AVX-512 and AVX2 twins of the whole-book chart.
// v0.3.0-test, EXPERIMENTAL.
// =============================================================================
// Lane by lane the same operation sequence as volfi_wb::implied_variance_wb, so the
// results are bit-identical to the scalar chart and, through it, to the CUDA kernel.
// The one branch at a = 2 pi is taken per vector: a vector whose lanes all lie on
// one side evaluates one region; a mixed vector evaluates both and blends.  Lanes
// the chart does not cover (region NW_OUT, or EDGE inputs) are answered by the
// shipped scalar entry, as in the scalar wrapper.
// =============================================================================
#ifndef VOLFI_WB_VEC_HPP
#define VOLFI_WB_VEC_HPP

#include "volfi_wb.hpp"
#include "volfi_near_certified_vec.hpp"

namespace volfi_wb {

namespace nv = volfi_near_certified::vec;
using nv::NC_TINY;
using nv::NC_CHI;
using volfi_near_certified::NC_H_FLOOR;
static constexpr double NW_HHI = 16.5;

#if defined(VA_SIMD512)
struct W8 {
    typedef __m512d T;
    static inline T bcast(double x)      { return _mm512_set1_pd(x); }
    static inline T zero()               { return _mm512_setzero_pd(); }
    static inline T fma(T a, T b, T c)   { return _mm512_fmadd_pd(a, b, c); }
    static inline T sub(T a, T b)        { return _mm512_sub_pd(a, b); }
    static inline T add(T a, T b)        { return _mm512_add_pd(a, b); }
    static inline T mul(T a, T b)        { return _mm512_mul_pd(a, b); }
    static inline T div(T a, T b)        { return _mm512_div_pd(a, b); }
    static inline T sqrt(T a)            { return _mm512_sqrt_pd(a); }
    static inline T floor(T a)           { return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
    static inline T min(T a, T b)        { return _mm512_min_pd(a, b); }
    static inline T max(T a, T b)        { return _mm512_max_pd(a, b); }
    static inline T pow2(T k)            { return _mm512_scalef_pd(_mm512_set1_pd(1.0), k); }   // exact 2^k
};
#endif
#if defined(VA_SIMD256)
struct W4 {
    typedef __m256d T;
    static inline T bcast(double x)      { return _mm256_set1_pd(x); }
    static inline T zero()               { return _mm256_setzero_pd(); }
    static inline T fma(T a, T b, T c)   { return _mm256_fmadd_pd(a, b, c); }
    static inline T sub(T a, T b)        { return _mm256_sub_pd(a, b); }
    static inline T add(T a, T b)        { return _mm256_add_pd(a, b); }
    static inline T mul(T a, T b)        { return _mm256_mul_pd(a, b); }
    static inline T div(T a, T b)        { return _mm256_div_pd(a, b); }
    static inline T sqrt(T a)            { return _mm256_sqrt_pd(a); }
    static inline T floor(T a)           { return _mm256_floor_pd(a); }
    static inline T min(T a, T b)        { return _mm256_min_pd(a, b); }
    static inline T max(T a, T b)        { return _mm256_max_pd(a, b); }
    static inline T pow2(T k) {          // 2^k for integer-valued k in [0, 1023]
        __m128i ki = _mm256_cvttpd_epi32(k);
        __m256i kl = _mm256_cvtepi32_epi64(ki);
        kl = _mm256_slli_epi64(_mm256_add_epi64(kl, _mm256_set1_epi64x(1023)), 52);
        return _mm256_castsi256_pd(kl);
    }
};
#endif

template<class V>
inline typename V::T nwv_expm1(typename V::T h) {
    const typename V::T k  = V::floor(V::mul(h, V::bcast(NW_INV_LN2)));
    const typename V::T r  = V::sub(V::fma(V::sub(V::zero(), k), V::bcast(NW_LN2_HI), h), V::mul(k, V::bcast(NW_LN2_LO)));
    typename V::T p = V::bcast(NWE_C[19]);
#pragma GCC unroll 32
    for (int j = 18; j >= 1; --j) p = V::fma(p, r, V::bcast(NWE_C[j]));
    const typename V::T er = V::fma(r, V::mul(p, r), r);
    const typename V::T pk = V::pow2(k);
    return V::fma(pk, er, V::sub(pk, V::bcast(1.0)));
}

template<class V, int N, bool CELL1>
inline typename V::T nwv_clen(typename V::T xv) {
    const typename V::T x2 = V::mul(V::bcast(2.0), xv);
    typename V::T e0 = V::zero(), e1 = V::zero();
#pragma GCC unroll 64
    for (int j = N - 1; j >= 1; --j) {
        const typename V::T b = V::sub(V::fma(x2, e0, V::bcast(CELL1 ? NW_F1[j] : NW_G0[j])), e1);
        e1 = e0; e0 = b;
    }
    return V::sub(V::fma(xv, e0, V::bcast(CELL1 ? NW_F1[0] : NW_G0[0])), e1);
}
template<class V, int M>
inline typename V::T nwv_PA(typename V::T y) {
    typename V::T p = V::bcast(NW_P[NWRowA<M>::off]);
#pragma GCC unroll 32
    for (int j = 1; j <= NWRowA<M>::deg; ++j) p = V::fma(p, y, V::bcast(NW_P[NWRowA<M>::off + j]));
    return p;
}
template<class V, int M>
inline typename V::T nwv_SA(typename V::T x, typename V::T y, typename V::T s) {
    if constexpr (M < 1) { (void)x; (void)y; return s; }
    else return nwv_SA<V, M - 1>(x, y, V::fma(s, x, nwv_PA<V, M>(y)));
}
template<class V, int N>
inline typename V::T nwv_dB(typename V::T w) {
    typename V::T p = V::bcast(NW_D[NWRowB<N>::off]);
#pragma GCC unroll 32
    for (int j = 1; j < NWRowB<N>::deg; ++j) p = V::fma(p, w, V::bcast(NW_D[NWRowB<N>::off + j]));
    return V::mul(p, w);
}
template<class V, int N>
inline typename V::T nwv_SB(typename V::T q, typename V::T w, typename V::T s) {
    if constexpr (N < 1) { (void)q; (void)w; return s; }
    else return nwv_SB<V, N - 1>(q, w, V::fma(s, q, nwv_dB<V, N>(w)));
}

template<class V>
inline typename V::T nwv_variance_A(typename V::T h, typename V::T a) {
    const typename V::T ac = V::min(a, V::bcast(NW_TWO_PI));
    const typename V::T g  = nwv_clen<V, NW_NG0, false>(V::fma(ac, V::bcast(NW_SA0), V::bcast(-1.0)));
    const typename V::T A0 = V::mul(ac, g);
    const typename V::T t  = V::div(h, A0);
    const typename V::T x  = V::mul(t, t);
    const typename V::T y  = V::mul(A0, A0);
    const typename V::T v  = V::fma(V::mul(t, x), nwv_SA<V, NW_MA>(x, y, V::zero()), t);
    return V::mul(v, v);
}
template<class V>
inline typename V::T nwv_B_front(typename V::T a) {                     // the backbone: pp = u F1(u), u = 1/sqrt(a)
    const typename V::T ac = V::max(V::min(a, V::bcast(NW_A_MAX)), V::bcast(NW_TWO_PI));
    const typename V::T u  = V::div(V::bcast(1.0), V::sqrt(ac));
    const typename V::T f1 = nwv_clen<V, NW_NF1, true>(V::fma(u, V::bcast(NW_SU1), V::bcast(NW_BU1)));
    return V::mul(u, f1);
}
template<class V>
inline typename V::T nwv_B_rows(typename V::T h, typename V::T pp) {    // the rows, given the backbone
    const typename V::T t  = V::mul(h, pp);
    const typename V::T w  = V::mul(pp, pp);
    const typename V::T z  = V::mul(V::mul(h, h), V::bcast(NW_INV_4PI2));
    const typename V::T sp = V::add(V::sqrt(V::add(V::bcast(1.0), z)), V::bcast(1.0));
    const typename V::T q  = V::div(z, V::mul(sp, sp));
    const typename V::T v  = V::fma(V::mul(t, q), nwv_SB<V, NW_NB>(q, w, V::zero()), t);
    return V::mul(v, v);
}
template<class V>
inline typename V::T nwv_variance_B(typename V::T h, typename V::T a) { return nwv_B_rows<V>(h, nwv_B_front<V>(a)); }

// ---- per-tile lane lists, evaluated as contiguous registers (region B in two passes)
#if defined(VA_SIMD512)
struct NWIO8 { typedef W8 V; enum { L = 8 }; static __m512d ld(const double* p) { return _mm512_loadu_pd(p); } static void st(double* p, __m512d v) { _mm512_storeu_pd(p, v); } };
#endif
#if defined(VA_SIMD256)
struct NWIO4 { typedef W4 V; enum { L = 4 }; static __m256d ld(const double* p) { return _mm256_loadu_pd(p); } static void st(double* p, __m256d v) { _mm256_storeu_pd(p, v); } };
#endif
template<class IO>
inline void nwv_eval_list_A(double* lh, double* la, const int* li, int cnt, double* sw) {
    const int pad = (cnt + IO::L - 1) / IO::L * IO::L;
    for (int k = cnt; k < pad; ++k) { lh[k] = 0.1; la[k] = 1.0; }                 // finite padding lanes, results discarded
    for (int k = 0; k < pad; k += IO::L) IO::st(lh + k, nwv_variance_A<typename IO::V>(IO::ld(lh + k), IO::ld(la + k)));
    for (int k = 0; k < cnt; ++k) sw[li[k]] = lh[k];
}
template<class IO>
inline void nwv_eval_list_B(double* lh, double* la, const int* li, int cnt, double* sw) {
    const int pad = (cnt + IO::L - 1) / IO::L * IO::L;
    for (int k = cnt; k < pad; ++k) { lh[k] = 0.1; la[k] = 10.0; }
    for (int k = 0; k < pad; k += IO::L) IO::st(la + k, nwv_B_front<typename IO::V>(IO::ld(la + k)));          // pass B1: backbone, in place
    for (int k = 0; k < pad; k += IO::L) IO::st(lh + k, nwv_B_rows<typename IO::V>(IO::ld(lh + k), IO::ld(la + k)));   // pass B2: rows
    for (int k = 0; k < cnt; ++k) sw[li[k]] = lh[k];
}

// ===========================================================================
// The batch driver.  Every lane receives a value: code[i] = 1 (A), 2 (B), or 0
// (answered by the shipped scalar entry: EDGE inputs and quotes outside the chart).
// ===========================================================================
inline void implied_variance_wb_batch(const double* h, const double* c,
                                      double* w, int* code, int n) {
    namespace nc = volfi_near_certified;
    constexpr int TILE = NC_VEC_TILE;
    static thread_local double sa[TILE], sw[TILE];
    static thread_local double dh[TILE], dc[TILE], dw[TILE];   // lanes the book kernel declines
    static thread_local int di[TILE];
    static thread_local double lAh[TILE + 8], lAa[TILE + 8], lBh[TILE + 8], lBa[TILE + 8];   // lane lists of the tile
    static thread_local int lAi[TILE + 8], lBi[TILE + 8];

    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;
        int i = 0;
        int nA = 0, nB = 0;
#if defined(VA_SIMD512)
        for (int j = 0; j + 8 <= m; j += 8) {                  // pass 1: the intrinsic coordinate a for the whole tile
            __m512d vh = _mm512_loadu_pd(h + base + j);
            __m512d vc = _mm512_loadu_pd(c + base + j);
            vc = _mm512_min_pd(_mm512_max_pd(vc, _mm512_set1_pd(NC_TINY)), _mm512_set1_pd(NC_CHI));
            vh = _mm512_min_pd(_mm512_max_pd(vh, _mm512_set1_pd(NC_TINY)), _mm512_set1_pd(NW_HHI));
            __m512d E = nwv_expm1<W8>(vh);
            __m512d q = _mm512_div_pd(E, vc);
            _mm512_storeu_pd(sa + j, nv::log1p_pos_avx512(q));
        }
        {
        const __m256i iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        for (; i + 8 <= m; i += 8) {                           // pass 2: pure region-A registers in place, everything else to the lists
            const __m512d vh = _mm512_min_pd(_mm512_max_pd(_mm512_loadu_pd(h + base + i), _mm512_set1_pd(NC_TINY)), _mm512_set1_pd(NW_HHI));
            const __m512d a = _mm512_loadu_pd(sa + i);
            const __mmask8 sA = _mm512_cmp_pd_mask(a, _mm512_set1_pd(NW_TWO_PI), _CMP_LT_OQ);
            // nw_region, operation for operation, on the raw h: lanes the book declines are not evaluated at all
            const __m512d hr = _mm512_loadu_pd(h + base + i);
            const __mmask8 mA = sA & _mm512_cmp_pd_mask(hr, _mm512_mul_pd(_mm512_set1_pd(NW_THETA_A), a), _CMP_LE_OQ);
            if (mA == 0xFF) { _mm512_storeu_pd(sw + i, nwv_variance_A<W8>(vh, a)); continue; }
            const __mmask8 mB = (__mmask8)(~sA) & _mm512_cmp_pd_mask(hr, _mm512_set1_pd(NW_H_B), _CMP_LE_OQ)
                              & _mm512_cmp_pd_mask(hr, _mm512_fmadd_pd(_mm512_set1_pd(NW_HB_SLOPE), a, _mm512_set1_pd(NW_HB_INT)), _CMP_LE_OQ)
                              & _mm512_cmp_pd_mask(a, _mm512_set1_pd(NW_A_MAX), _CMP_LE_OQ);
            const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32(i), iota);
            if (mA) { _mm512_mask_compressstoreu_pd(lAh + nA, mA, vh); _mm512_mask_compressstoreu_pd(lAa + nA, mA, a);
                      _mm256_mask_compressstoreu_epi32(lAi + nA, mA, idx); nA += __builtin_popcount((unsigned)mA); }
            if (mB) { _mm512_mask_compressstoreu_pd(lBh + nB, mB, vh); _mm512_mask_compressstoreu_pd(lBa + nB, mB, a);
                      _mm256_mask_compressstoreu_epi32(lBi + nB, mB, idx); nB += __builtin_popcount((unsigned)mB); }
        }
        }
        if (nA) nwv_eval_list_A<NWIO8>(lAh, lAa, lAi, nA, sw);
        if (nB) nwv_eval_list_B<NWIO8>(lBh, lBa, lBi, nB, sw);
#elif defined(VA_SIMD256)
        for (int j = 0; j + 4 <= m; j += 4) {                  // pass 1: the intrinsic coordinate a for the whole tile
            __m256d vh = _mm256_loadu_pd(h + base + j);
            __m256d vc = _mm256_loadu_pd(c + base + j);
            vc = _mm256_min_pd(_mm256_max_pd(vc, _mm256_set1_pd(NC_TINY)), _mm256_set1_pd(NC_CHI));
            vh = _mm256_min_pd(_mm256_max_pd(vh, _mm256_set1_pd(NC_TINY)), _mm256_set1_pd(NW_HHI));
            __m256d E = nwv_expm1<W4>(vh);
            __m256d q = _mm256_div_pd(E, vc);
            _mm256_storeu_pd(sa + j, nv::log1p_pos_avx2(q));
        }
        for (; i + 4 <= m; i += 4) {                           // pass 2: pure region-A registers in place, everything else to the lists
            const __m256d vh = _mm256_min_pd(_mm256_max_pd(_mm256_loadu_pd(h + base + i), _mm256_set1_pd(NC_TINY)), _mm256_set1_pd(NW_HHI));
            const __m256d a = _mm256_loadu_pd(sa + i);
            const __m256d sA = _mm256_cmp_pd(a, _mm256_set1_pd(NW_TWO_PI), _CMP_LT_OQ);
            const __m256d hr = _mm256_loadu_pd(h + base + i);                     // nw_region, operation for operation, on the raw h
            const int mk = _mm256_movemask_pd(_mm256_and_pd(sA, _mm256_cmp_pd(hr, _mm256_mul_pd(_mm256_set1_pd(NW_THETA_A), a), _CMP_LE_OQ)));
            const int mkB = _mm256_movemask_pd(_mm256_andnot_pd(sA, _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(hr, _mm256_set1_pd(NW_H_B), _CMP_LE_OQ),
                                _mm256_cmp_pd(hr, _mm256_fmadd_pd(_mm256_set1_pd(NW_HB_SLOPE), a, _mm256_set1_pd(NW_HB_INT)), _CMP_LE_OQ)),
                                _mm256_cmp_pd(a, _mm256_set1_pd(NW_A_MAX), _CMP_LE_OQ))));
            if (mk == 0xF) { _mm256_storeu_pd(sw + i, nwv_variance_A<W4>(vh, a)); continue; }
            alignas(32) double th[4], ta[4]; _mm256_store_pd(th, vh); _mm256_store_pd(ta, a);
            for (int k = 0; k < 4; ++k) {
                if (mk & (1 << k)) { lAh[nA] = th[k]; lAa[nA] = ta[k]; lAi[nA] = i + k; ++nA; }
                else if (mkB & (1 << k)) { lBh[nB] = th[k]; lBa[nB] = ta[k]; lBi[nB] = i + k; ++nB; }
            }
        }
        if (nA) nwv_eval_list_A<NWIO4>(lAh, lAa, lAi, nA, sw);
        if (nB) nwv_eval_list_B<NWIO4>(lBh, lBa, lBi, nB, sw);
#endif
        for (; i < m; ++i) {                       // scalar tail, identical kernel
            const double sh = std::fmin(std::fmax(h[base + i], NC_TINY), NW_HHI);
            const double sc = std::fmin(std::fmax(c[base + i], NC_TINY), NC_CHI);
            const double E = nw_expm1(sh);
            const double a = nc::log1p_pos(E / sc);
            sa[i] = a;
            sw[i] = (a < NW_TWO_PI) ? nw_variance_A(sh, a) : nw_variance_B(sh, a);
        }
        int nd = 0;
        for (i = 0; i < m; ++i) {
            const double cc = c[base + i], hh = h[base + i];
            int r = NW_OUT;
            if (cc > 0.0 && cc < 1.0 && hh >= NC_H_FLOOR && hh <= NW_H_MAX) r = nw_region(hh, sa[i]);
            code[base + i] = r;
            if (r != NW_OUT) w[base + i] = sw[i];
            else { dh[nd] = hh; dc[nd] = cc; di[nd] = base + i; ++nd; }
        }
        if (nd >= 8) {                                // routed vector driver: bit-identical to the scalar entry per quote
            volfi_annulus::detail::grid_upper_first(dh, dc, dw, nd);
            for (int k = 0; k < nd; ++k) w[di[k]] = dw[k];
        } else for (int k = 0; k < nd; ++k) w[di[k]] = volfi_annulus::implied_variance_otm(dh[k], dc[k]);
    }
}

} // namespace volfi_wb
#endif // VOLFI_WB_VEC_HPP
