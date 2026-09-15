// volfi_near_certified_vec.hpp -- AVX-512 and AVX2 twins of the certified NEAR chart
// ============================================================================
// EXPERIMENTAL.  Companion to volfi_near_certified.hpp.
//
// CONTRACT.  Every lane performs exactly the scalar operation sequence, in the
// same order, with the same std::fma fusions, so
//     near_variance_batch(...)  ==  near_variance_certified(h[i], c[i])
// bit for bit, on AVX-512, AVX2 and the scalar fallback.  That is checked by
// gate/vec_gate.cpp, which reports a mismatch count and must print 0.
//
// LOG KERNEL.  The twins support NC_LOG1P_MODE 0, 2 and 3; mode 1 is
// std::log1p, which has no vector form.  Mode 2 (nc_log1p) is the default and
// measures 1.00 ULP against std::log1p over the chart's whole q-range, versus
// 2.94 ULP for the shipped full_log.  That difference is what decides the
// accuracy gate: at h = 2.708e-09, c = 0.6438766 mode 0 returns 6.502 ULP and
// fails, mode 2 returns 2.709 ULP and is bit-identical to the std::log1p build.
//
// WHY THE DRIVER PARTITIONS.  The chart is a graded 2-D Chebyshev whose row
// profile differs per a-cell, so a branchless blend across all 12 cells would
// have to pad every cell to the per-row maximum -- about 150 coefficients per
// quote instead of the 64 the feed actually weights to -- and would still need
// a per-lane gather of the coefficients.  Instead the driver does what
// grid_table_pass already does for the FAR table: an index pass that is fully
// vectorised, a counting sort into 12 buckets, then a straight-line kernel per
// bucket with every coefficient broadcast.  12 buckets against FAR's 239, and
// the cell index is a comparison ladder on one scalar rather than a bit
// extraction from the price exponent plus an octave lookup.
// ============================================================================
#ifndef VOLFI_NEAR_CERTIFIED_VEC_HPP
#define VOLFI_NEAR_CERTIFIED_VEC_HPP

#include "volfi_near_certified.hpp"

#if NC_LOG1P_MODE == 1
#  error "volfi_near_certified_vec.hpp needs NC_LOG1P_MODE 0, 2 or 3 (std::log1p has no vector form)"
#endif

#ifndef NC_VEC_TILE
#define NC_VEC_TILE 4096
#endif

namespace volfi_near_certified {
namespace vec {

// In-register scrub bounds for lanes that cannot route NEAR.  A NEAR lane always
// satisfies 0 < c < 1 and 0 < h < NC_H_NEAR, so clamping to these is a no-op for it.
static const double NC_TINY = 2.2250738585072014e-308;      // DBL_MIN
static const double NC_CHI  = 0.99999999999999989;          // 1 - 2^-53
static const double NC_HHI  = 0.29999999999999999;          // < NC_H_NEAR

// ===========================================================================
// AVX-512, 8 lanes
// ===========================================================================
#if defined(VA_SIMD512)

inline __m512d nc_neg(__m512d x) {          // exact C++ unary minus, AVX512F only
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(x),
                                                _mm512_set1_epi64((long long)0x8000000000000000ULL)));
}

// mirror of nc_log
inline __m512d nc_log_avx512(__m512d x) {
    const __m512d one = _mm512_set1_pd(1.0);
    __m512i b  = _mm512_castpd_si512(x);
    __m512i j  = _mm512_and_si512(_mm512_srli_epi64(b, 52), _mm512_set1_epi64(0x7FFLL));
    __m512d dj = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(j, _mm512_set1_epi64(0x4330000000000000LL))),
                               _mm512_set1_pd(0x1.0p52));
    __m512d k  = _mm512_sub_pd(dj, _mm512_set1_pd(1023.0));
    __m512d m  = volfi_annulus::detail::mant12_avx512(x);
    __mmask8 hi = _mm512_cmp_pd_mask(m, _mm512_set1_pd(NCL_SQRT2), _CMP_GT_OQ);
    m = _mm512_mask_mul_pd(m, hi, m, _mm512_set1_pd(0.5));
    k = _mm512_mask_add_pd(k, hi, k, one);
    __m512d f = _mm512_sub_pd(m, one);
    __m512d s = _mm512_div_pd(f, _mm512_add_pd(_mm512_set1_pd(2.0), f));
    __m512d z = _mm512_mul_pd(s, s);
    __m512d w = _mm512_mul_pd(z, z);
    __m512d t1 = _mm512_fmadd_pd(w, _mm512_set1_pd(NCL_LG[5]), _mm512_set1_pd(NCL_LG[3]));
    t1 = _mm512_fmadd_pd(w, t1, _mm512_set1_pd(NCL_LG[1]));
    t1 = _mm512_mul_pd(w, t1);
    __m512d t2 = _mm512_fmadd_pd(w, _mm512_set1_pd(NCL_LG[6]), _mm512_set1_pd(NCL_LG[4]));
    t2 = _mm512_fmadd_pd(w, t2, _mm512_set1_pd(NCL_LG[2]));
    t2 = _mm512_fmadd_pd(w, t2, _mm512_set1_pd(NCL_LG[0]));
    t2 = _mm512_mul_pd(z, t2);
    __m512d R    = _mm512_add_pd(t2, t1);
    __m512d hfsq = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), f), f);   // (0.5*f)*f
    __m512d klo  = _mm512_mul_pd(k, _mm512_set1_pd(NCL_LN2_LO));
    __m512d inr  = _mm512_fmadd_pd(s, _mm512_add_pd(hfsq, R), klo);
    __m512d t    = _mm512_sub_pd(_mm512_sub_pd(hfsq, inr), f);
    return _mm512_fmadd_pd(k, _mm512_set1_pd(NCL_LN2_HI), nc_neg(t));
}

// mirror of log1p_reduce (mode 3): k from the exponent and half-bit of the
// rounded 1+q, then f = (q - (2^k-1)) 2^-k formed from q itself, exact.  2^k and
// 2^-k are built by integer shifts of the biased exponent, so there is no gather
// and no transcendental; every lane op is an add, sub, mul, fma or one division.
inline __m512d log1p_reduce_avx512(__m512d q) {
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d u  = _mm512_add_pd(one, q);
    __m512i b  = _mm512_castpd_si512(u);
    __m512i j  = _mm512_and_si512(_mm512_srli_epi64(b, 52), _mm512_set1_epi64(0x7FFLL));   // biased exponent
    __m512d m  = volfi_annulus::detail::mant12_avx512(u);
    __mmask8 hi = _mm512_cmp_pd_mask(m, _mm512_set1_pd(1.5), _CMP_GE_OQ);
    j = _mm512_mask_add_epi64(j, hi, j, _mm512_set1_epi64(1));                          // biased k
    __m512d pk  = _mm512_castsi512_pd(_mm512_slli_epi64(j, 52));                        // 2^k
    __m512d ipk = _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_sub_epi64(_mm512_set1_epi64(2046), j), 52)); // 2^-k
    __m512d kd  = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(j, _mm512_set1_epi64(0x4330000000000000LL))),
                                _mm512_set1_pd(0x1.0p52 + 1023.0));                     // (double)k, exact
    __m512d f = _mm512_mul_pd(_mm512_sub_pd(q, _mm512_sub_pd(pk, one)), ipk);
    __m512d s = _mm512_div_pd(f, _mm512_add_pd(_mm512_set1_pd(2.0), f));
    __m512d z = _mm512_mul_pd(s, s);
    __m512d w = _mm512_mul_pd(z, z);
    __m512d t1 = _mm512_fmadd_pd(w, _mm512_set1_pd(NCL3_P[7]), _mm512_set1_pd(NCL3_P[5]));
    t1 = _mm512_fmadd_pd(w, t1, _mm512_set1_pd(NCL3_P[3]));
    t1 = _mm512_fmadd_pd(w, t1, _mm512_set1_pd(NCL3_P[1]));
    t1 = _mm512_mul_pd(w, t1);
    __m512d t2 = _mm512_fmadd_pd(w, _mm512_set1_pd(NCL3_P[6]), _mm512_set1_pd(NCL3_P[4]));
    t2 = _mm512_fmadd_pd(w, t2, _mm512_set1_pd(NCL3_P[2]));
    t2 = _mm512_fmadd_pd(w, t2, _mm512_set1_pd(NCL3_P[0]));
    t2 = _mm512_mul_pd(z, t2);
    __m512d R    = _mm512_add_pd(t2, t1);
    __m512d hfsq = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), f), f);
    __m512d klo  = _mm512_mul_pd(kd, _mm512_set1_pd(NCL_LN2_LO));
    __m512d inr  = _mm512_fmadd_pd(s, _mm512_add_pd(hfsq, R), klo);
    __m512d t    = _mm512_sub_pd(_mm512_sub_pd(hfsq, inr), f);
    return _mm512_fmadd_pd(kd, _mm512_set1_pd(NCL_LN2_HI), nc_neg(t));
}

// mirror of log1p_pos for the vectorisable modes
inline __m512d log1p_pos_avx512(__m512d q) {
#if NC_LOG1P_MODE == 3
    return log1p_reduce_avx512(q);
#else
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d u = _mm512_add_pd(one, q);
    __m512d d = _mm512_sub_pd(u, one);
    __m512d corr = _mm512_div_pd(_mm512_sub_pd(q, d), u);
#if NC_LOG1P_MODE == 0
    return _mm512_add_pd(volfi_annulus::detail::full_log_avx512(u), corr);
#else
    return _mm512_add_pd(nc_log_avx512(u), corr);
#endif
#endif
}

// mirror of expm1_small (nc_expm1 by default, the shipped Chebyshev under
// -DNC_EXPM1_SHIPPED=1)
inline __m512d expm1_small_avx512(__m512d h) {
#if NC_EXPM1_SHIPPED
    return _mm512_mul_pd(h, volfi_annulus::detail::clenshaw1_avx512(
        volfi_annulus_broadrange::EXPM1G_C, 11,
        volfi_annulus_broadrange::EXPM1G_A, volfi_annulus_broadrange::EXPM1G_B, h));
#else
    __m512d p = _mm512_set1_pd(NCE_C[13]);
    for (int k = 12; k >= 1; --k) p = _mm512_fmadd_pd(p, h, _mm512_set1_pd(NCE_C[k]));
    return _mm512_fmadd_pd(h, _mm512_mul_pd(p, h), h);
#endif
}

// mirror of cell_of: the number of interior edges at or below a, capped at NC_NCELL-1.
inline __m512i cell_of_avx512(__m512d a) {
    __m512i idx = _mm512_setzero_si512();
    for (int j = 1; j < NC_NCELL; ++j) {
        __mmask8 m = _mm512_cmp_pd_mask(a, _mm512_set1_pd(NC_A_EDGE[j]), _CMP_GE_OQ);
        idx = _mm512_mask_add_epi64(idx, m, idx, _mm512_set1_epi64(1));
    }
    return idx;
}

// mirror of clenshaw2_graded.  `cell` is a compile-time-unknown but lane-invariant
// bucket id, so every coefficient is a broadcast and there is no gather.
inline __m512d clenshaw2_graded_avx512(int cell, __m512d xt, __m512d xa) {
    const int nrow = NC_NROW[cell];
    const int roff = NC_ROWOFF[cell];
    const __m512d xa2 = _mm512_add_pd(xa, xa);
    const __m512d xt2 = _mm512_add_pd(xt, xt);
    __m512d g0 = _mm512_setzero_pd(), g1 = _mm512_setzero_pd();
    for (int m = nrow - 1; m >= 1; --m) {
        const int     nc = NC_ROWDEG[roff + m];
        const double* C  = NC_COEFF + NC_ROWCOFF[roff + m];
        __m512d e0 = _mm512_setzero_pd(), e1 = _mm512_setzero_pd();
        for (int j = nc - 1; j >= 1; --j) {
            __m512d b = _mm512_sub_pd(_mm512_fmadd_pd(xa2, e0, _mm512_set1_pd(C[j])), e1);
            e1 = e0; e0 = b;
        }
        __m512d r = (nc > 0) ? _mm512_sub_pd(_mm512_fmadd_pd(xa, e0, _mm512_set1_pd(C[0])), e1)
                             : _mm512_setzero_pd();
        __m512d b = _mm512_sub_pd(_mm512_fmadd_pd(xt2, g0, r), g1);
        g1 = g0; g0 = b;
    }
    {
        const int     nc = NC_ROWDEG[roff];
        const double* C  = NC_COEFF + NC_ROWCOFF[roff];
        __m512d e0 = _mm512_setzero_pd(), e1 = _mm512_setzero_pd();
        for (int j = nc - 1; j >= 1; --j) {
            __m512d b = _mm512_sub_pd(_mm512_fmadd_pd(xa2, e0, _mm512_set1_pd(C[j])), e1);
            e1 = e0; e0 = b;
        }
        __m512d r = (nc > 0) ? _mm512_sub_pd(_mm512_fmadd_pd(xa, e0, _mm512_set1_pd(C[0])), e1)
                             : _mm512_setzero_pd();
        return _mm512_sub_pd(_mm512_fmadd_pd(xt, g0, r), g1);
    }
}

#endif // VA_SIMD512

// ===========================================================================
// AVX2, 4 lanes -- line-for-line mirror of the AVX-512 block
// ===========================================================================
#if defined(VA_SIMD256)

inline __m256d nc_neg_256(__m256d x) {
    return _mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(x),
                                                _mm256_set1_epi64x((long long)0x8000000000000000ULL)));
}

inline __m256d nc_log_avx2(__m256d x) {
    const __m256d one = _mm256_set1_pd(1.0);
    __m256i b  = _mm256_castpd_si256(x);
    __m256i j  = _mm256_and_si256(_mm256_srli_epi64(b, 52), _mm256_set1_epi64x(0x7FFLL));
    __m256d dj = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(j, _mm256_set1_epi64x(0x4330000000000000LL))),
                               _mm256_set1_pd(0x1.0p52));
    __m256d k  = _mm256_sub_pd(dj, _mm256_set1_pd(1023.0));
    __m256d m  = volfi_annulus::detail::mant12_avx2(x);
    __m256d hi = _mm256_cmp_pd(m, _mm256_set1_pd(NCL_SQRT2), _CMP_GT_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), hi);
    k = _mm256_blendv_pd(k, _mm256_add_pd(k, one), hi);
    __m256d f = _mm256_sub_pd(m, one);
    __m256d s = _mm256_div_pd(f, _mm256_add_pd(_mm256_set1_pd(2.0), f));
    __m256d z = _mm256_mul_pd(s, s);
    __m256d w = _mm256_mul_pd(z, z);
    __m256d t1 = _mm256_fmadd_pd(w, _mm256_set1_pd(NCL_LG[5]), _mm256_set1_pd(NCL_LG[3]));
    t1 = _mm256_fmadd_pd(w, t1, _mm256_set1_pd(NCL_LG[1]));
    t1 = _mm256_mul_pd(w, t1);
    __m256d t2 = _mm256_fmadd_pd(w, _mm256_set1_pd(NCL_LG[6]), _mm256_set1_pd(NCL_LG[4]));
    t2 = _mm256_fmadd_pd(w, t2, _mm256_set1_pd(NCL_LG[2]));
    t2 = _mm256_fmadd_pd(w, t2, _mm256_set1_pd(NCL_LG[0]));
    t2 = _mm256_mul_pd(z, t2);
    __m256d R    = _mm256_add_pd(t2, t1);
    __m256d hfsq = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), f), f);
    __m256d klo  = _mm256_mul_pd(k, _mm256_set1_pd(NCL_LN2_LO));
    __m256d inr  = _mm256_fmadd_pd(s, _mm256_add_pd(hfsq, R), klo);
    __m256d t    = _mm256_sub_pd(_mm256_sub_pd(hfsq, inr), f);
    return _mm256_fmadd_pd(k, _mm256_set1_pd(NCL_LN2_HI), nc_neg_256(t));
}

inline __m256d log1p_reduce_avx2(__m256d q) {
    const __m256d one = _mm256_set1_pd(1.0);
    __m256d u  = _mm256_add_pd(one, q);
    __m256i b  = _mm256_castpd_si256(u);
    __m256i j  = _mm256_and_si256(_mm256_srli_epi64(b, 52), _mm256_set1_epi64x(0x7FFLL));
    __m256d m  = volfi_annulus::detail::mant12_avx2(u);
    __m256d hi = _mm256_cmp_pd(m, _mm256_set1_pd(1.5), _CMP_GE_OQ);
    j = _mm256_sub_epi64(j, _mm256_castpd_si256(hi));                                    // mask is all-ones = -1
    __m256d pk  = _mm256_castsi256_pd(_mm256_slli_epi64(j, 52));
    __m256d ipk = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_sub_epi64(_mm256_set1_epi64x(2046), j), 52));
    __m256d kd  = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(j, _mm256_set1_epi64x(0x4330000000000000LL))),
                                _mm256_set1_pd(0x1.0p52 + 1023.0));
    __m256d f = _mm256_mul_pd(_mm256_sub_pd(q, _mm256_sub_pd(pk, one)), ipk);
    __m256d s = _mm256_div_pd(f, _mm256_add_pd(_mm256_set1_pd(2.0), f));
    __m256d z = _mm256_mul_pd(s, s);
    __m256d w = _mm256_mul_pd(z, z);
    __m256d t1 = _mm256_fmadd_pd(w, _mm256_set1_pd(NCL3_P[7]), _mm256_set1_pd(NCL3_P[5]));
    t1 = _mm256_fmadd_pd(w, t1, _mm256_set1_pd(NCL3_P[3]));
    t1 = _mm256_fmadd_pd(w, t1, _mm256_set1_pd(NCL3_P[1]));
    t1 = _mm256_mul_pd(w, t1);
    __m256d t2 = _mm256_fmadd_pd(w, _mm256_set1_pd(NCL3_P[6]), _mm256_set1_pd(NCL3_P[4]));
    t2 = _mm256_fmadd_pd(w, t2, _mm256_set1_pd(NCL3_P[2]));
    t2 = _mm256_fmadd_pd(w, t2, _mm256_set1_pd(NCL3_P[0]));
    t2 = _mm256_mul_pd(z, t2);
    __m256d R    = _mm256_add_pd(t2, t1);
    __m256d hfsq = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), f), f);
    __m256d klo  = _mm256_mul_pd(kd, _mm256_set1_pd(NCL_LN2_LO));
    __m256d inr  = _mm256_fmadd_pd(s, _mm256_add_pd(hfsq, R), klo);
    __m256d t    = _mm256_sub_pd(_mm256_sub_pd(hfsq, inr), f);
    return _mm256_fmadd_pd(kd, _mm256_set1_pd(NCL_LN2_HI), nc_neg_256(t));
}

inline __m256d log1p_pos_avx2(__m256d q) {
#if NC_LOG1P_MODE == 3
    return log1p_reduce_avx2(q);
#else
    const __m256d one = _mm256_set1_pd(1.0);
    __m256d u = _mm256_add_pd(one, q);
    __m256d d = _mm256_sub_pd(u, one);
    __m256d corr = _mm256_div_pd(_mm256_sub_pd(q, d), u);
#if NC_LOG1P_MODE == 0
    return _mm256_add_pd(volfi_annulus::detail::full_log_avx2(u), corr);
#else
    return _mm256_add_pd(nc_log_avx2(u), corr);
#endif
#endif
}

inline __m256d expm1_small_avx2(__m256d h) {
#if NC_EXPM1_SHIPPED
    return _mm256_mul_pd(h, volfi_annulus::detail::clenshaw1_avx2(
        volfi_annulus_broadrange::EXPM1G_C, 11,
        volfi_annulus_broadrange::EXPM1G_A, volfi_annulus_broadrange::EXPM1G_B, h));
#else
    __m256d p = _mm256_set1_pd(NCE_C[13]);
    for (int k = 12; k >= 1; --k) p = _mm256_fmadd_pd(p, h, _mm256_set1_pd(NCE_C[k]));
    return _mm256_fmadd_pd(h, _mm256_mul_pd(p, h), h);
#endif
}

inline void cell_of_avx2(__m256d a, int* out) {
    __m256d idx = _mm256_setzero_pd();
    for (int j = 1; j < NC_NCELL; ++j) {
        __m256d m = _mm256_cmp_pd(a, _mm256_set1_pd(NC_A_EDGE[j]), _CMP_GE_OQ);
        idx = _mm256_add_pd(idx, _mm256_and_pd(m, _mm256_set1_pd(1.0)));
    }
    alignas(32) double t[4];
    _mm256_store_pd(t, idx);
    for (int i = 0; i < 4; ++i) out[i] = (int)t[i];
}

inline __m256d clenshaw2_graded_avx2(int cell, __m256d xt, __m256d xa) {
    const int nrow = NC_NROW[cell];
    const int roff = NC_ROWOFF[cell];
    const __m256d xa2 = _mm256_add_pd(xa, xa);
    const __m256d xt2 = _mm256_add_pd(xt, xt);
    __m256d g0 = _mm256_setzero_pd(), g1 = _mm256_setzero_pd();
    for (int m = nrow - 1; m >= 1; --m) {
        const int     nc = NC_ROWDEG[roff + m];
        const double* C  = NC_COEFF + NC_ROWCOFF[roff + m];
        __m256d e0 = _mm256_setzero_pd(), e1 = _mm256_setzero_pd();
        for (int j = nc - 1; j >= 1; --j) {
            __m256d b = _mm256_sub_pd(_mm256_fmadd_pd(xa2, e0, _mm256_set1_pd(C[j])), e1);
            e1 = e0; e0 = b;
        }
        __m256d r = (nc > 0) ? _mm256_sub_pd(_mm256_fmadd_pd(xa, e0, _mm256_set1_pd(C[0])), e1)
                             : _mm256_setzero_pd();
        __m256d b = _mm256_sub_pd(_mm256_fmadd_pd(xt2, g0, r), g1);
        g1 = g0; g0 = b;
    }
    {
        const int     nc = NC_ROWDEG[roff];
        const double* C  = NC_COEFF + NC_ROWCOFF[roff];
        __m256d e0 = _mm256_setzero_pd(), e1 = _mm256_setzero_pd();
        for (int j = nc - 1; j >= 1; --j) {
            __m256d b = _mm256_sub_pd(_mm256_fmadd_pd(xa2, e0, _mm256_set1_pd(C[j])), e1);
            e1 = e0; e0 = b;
        }
        __m256d r = (nc > 0) ? _mm256_sub_pd(_mm256_fmadd_pd(xa, e0, _mm256_set1_pd(C[0])), e1)
                             : _mm256_setzero_pd();
        return _mm256_sub_pd(_mm256_fmadd_pd(xt, g0, r), g1);
    }
}

#endif // VA_SIMD256

// ===========================================================================
// The batch driver.  route[i] uses the scalar codes:
//   -1 out of the NEAR band, 0 edge, 1 NEAR, 2 UPPER, 3 WING.
// w[i] is written only where route[i] == 1; elsewhere it is left untouched.
// ===========================================================================
inline void near_variance_batch(const double* h, const double* c,
                                double* w, int* route, int n) {
    constexpr int TILE = NC_VEC_TILE;
    static thread_local double sa[TILE], sth[TILE], stau[TILE];
    static thread_local double pa[TILE], pth[TILE], ptau[TILE], pw[TILE];
    static thread_local int    scell[TILE], order[TILE];

    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;
        int cnt[NC_NCELL];
        for (int t = 0; t < NC_NCELL; ++t) cnt[t] = 0;

        // ---- pass A: coordinates, route, cell -----------------------------
        // Pass A evaluates E/c, log1p and h/a for EVERY lane before routing, while
        // the scalar entry short-circuits on c<=0, c>=1, h<=0 first.  Left alone the
        // batch would raise DIVBYZERO/INVALID the scalar never raises, and would die
        // with SIGFPE under feenableexcept(FE_INVALID|FE_DIVBYZERO) -- review
        // demonstrated that on a batch of 8 containing one c = 0.0 quote.  Rather than
        // pre-filtering into a side buffer (a whole extra pass over the tile), the
        // lanes are scrubbed in register: min/max clamp c into [NC_TINY, 1-2^-53] and h
        // into [NC_TINY, NC_H_NEAR].  For a lane that will route NEAR the clamp is a
        // no-op, so bit-identity is untouched; for any other lane the arithmetic is
        // merely harmless, and its route is decided from the ORIGINAL h and c below.
        // Intel min/max return the SECOND source when either input is NaN, so writing
        // the clamp as min(max(x, lo), hi) also scrubs NaN.
        int i = 0;
#if defined(VA_SIMD512)
        for (; i + 8 <= m; i += 8) {
            __m512d vh = _mm512_loadu_pd(h + base + i);
            __m512d vc = _mm512_loadu_pd(c + base + i);
            vc = _mm512_min_pd(_mm512_max_pd(vc, _mm512_set1_pd(NC_TINY)),
                               _mm512_set1_pd(NC_CHI));
            vh = _mm512_min_pd(_mm512_max_pd(vh, _mm512_set1_pd(NC_TINY)),
                               _mm512_set1_pd(NC_HHI));
            __m512d E  = expm1_small_avx512(vh);
            __m512d q  = _mm512_div_pd(E, vc);
            __m512d a  = log1p_pos_avx512(q);
            __m512d th = _mm512_div_pd(vh, a);
            __m512d tu = _mm512_mul_pd(th, th);
            _mm512_storeu_pd(sa + i, a);
            _mm512_storeu_pd(sth + i, th);
            _mm512_storeu_pd(stau + i, tu);
            // narrow the 8 int64 cell ids to int32 and store them in one go
            _mm256_storeu_si256((__m256i*)(scell + i), _mm512_cvtepi64_epi32(cell_of_avx512(a)));
        }
#elif defined(VA_SIMD256)
        for (; i + 4 <= m; i += 4) {
            __m256d vh = _mm256_loadu_pd(h + base + i);
            __m256d vc = _mm256_loadu_pd(c + base + i);
            vc = _mm256_min_pd(_mm256_max_pd(vc, _mm256_set1_pd(NC_TINY)),
                               _mm256_set1_pd(NC_CHI));
            vh = _mm256_min_pd(_mm256_max_pd(vh, _mm256_set1_pd(NC_TINY)),
                               _mm256_set1_pd(NC_HHI));
            __m256d E  = expm1_small_avx2(vh);
            __m256d q  = _mm256_div_pd(E, vc);
            __m256d a  = log1p_pos_avx2(q);
            __m256d th = _mm256_div_pd(vh, a);
            __m256d tu = _mm256_mul_pd(th, th);
            _mm256_storeu_pd(sa + i, a);
            _mm256_storeu_pd(sth + i, th);
            _mm256_storeu_pd(stau + i, tu);
            cell_of_avx2(a, scell + i);
        }
#endif
        for (; i < m; ++i) {                       // scalar tail, identical kernel
            const double sh = std::fmin(std::fmax(h[base + i], NC_TINY), NC_HHI);
            const double sc = std::fmin(std::fmax(c[base + i], NC_TINY), NC_CHI);
            near_coords z = near_coords_of(sh, sc);
            sa[i] = z.a; sth[i] = z.theta; stau[i] = z.tau;
            scell[i] = cell_of(z.a);
        }

        // ---- routing, and the counting sort over NEAR quotes ---------------
        for (i = 0; i < m; ++i) {
            const double cc = c[base + i], hh = h[base + i];
            int r;
            if (!(cc > 0.0) || cc >= 1.0)        r = NCR_EDGE;
            else if (!(hh >= NC_H_FLOOR))        r = NCR_EDGE;     // the w = v*v floor, see the scalar router
            else if (!(hh < NC_H_NEAR))          r = NCR_OUTOFBAND;
            else if (!(cc * NC_INV_RHO_WING > expm1_small(hh))) r = NCR_WING;
            else if (sa[i] >= NC_A_WING)         r = NCR_WING;
            else if (sth[i] > NC_THETA)          r = NCR_UPPER;
            else                                 r = NCR_NEAR;
            route[base + i] = r;
            if (r == NCR_NEAR) ++cnt[scell[i]];
        }
        int off[NC_NCELL], pos[NC_NCELL], run = 0;
        for (int t = 0; t < NC_NCELL; ++t) { off[t] = run; run += cnt[t]; pos[t] = off[t]; }
        for (i = 0; i < m; ++i)
            if (route[base + i] == NCR_NEAR) order[pos[scell[i]]++] = i;

        // ---- pass B: one straight-line kernel per bucket -------------------
        for (int cell = 0; cell < NC_NCELL; ++cell) {
            const int lo = off[cell], hiN = off[cell] + cnt[cell];
            if (lo == hiN) continue;
            const int    k  = hiN - lo;
            const double t1 = NC_TAU_HI[cell];
            const double a0 = NC_A_EDGE[cell], a1 = NC_A_EDGE[cell + 1];
            for (int e = 0; e < k; ++e) {
                const int s = order[lo + e];
                pa[e] = sa[s]; pth[e] = sth[s]; ptau[e] = stau[s];
            }
            int e = 0;
#if defined(VA_SIMD512)
            for (; e + 8 <= k; e += 8) {
                __m512d a  = _mm512_loadu_pd(pa + e);
                __m512d tu = _mm512_loadu_pd(ptau + e);
                __m512d xt = _mm512_div_pd(_mm512_fmadd_pd(_mm512_set1_pd(2.0), tu,
                                                           _mm512_set1_pd(-t1)), _mm512_set1_pd(t1));
                __m512d xa = _mm512_div_pd(_mm512_fmadd_pd(_mm512_set1_pd(2.0), a,
                                                           _mm512_set1_pd(-(a0 + a1))),
                                           _mm512_set1_pd(a1 - a0));
                __m512d v  = _mm512_mul_pd(_mm512_loadu_pd(pth + e),
                                           clenshaw2_graded_avx512(cell, xt, xa));
                _mm512_storeu_pd(pw + e, _mm512_mul_pd(v, v));
            }
#elif defined(VA_SIMD256)
            for (; e + 4 <= k; e += 4) {
                __m256d a  = _mm256_loadu_pd(pa + e);
                __m256d tu = _mm256_loadu_pd(ptau + e);
                __m256d xt = _mm256_div_pd(_mm256_fmadd_pd(_mm256_set1_pd(2.0), tu,
                                                           _mm256_set1_pd(-t1)), _mm256_set1_pd(t1));
                __m256d xa = _mm256_div_pd(_mm256_fmadd_pd(_mm256_set1_pd(2.0), a,
                                                           _mm256_set1_pd(-(a0 + a1))),
                                           _mm256_set1_pd(a1 - a0));
                __m256d v  = _mm256_mul_pd(_mm256_loadu_pd(pth + e),
                                           clenshaw2_graded_avx2(cell, xt, xa));
                _mm256_storeu_pd(pw + e, _mm256_mul_pd(v, v));
            }
#endif
            for (; e < k; ++e) {                   // scalar tail
                const double xt = std::fma(2.0, ptau[e], -t1) / t1;
                const double xa = std::fma(2.0, pa[e], -(a0 + a1)) / (a1 - a0);
                const double v  = pth[e] * clenshaw2_graded(cell, xt, xa);
                pw[e] = v * v;
            }
            for (e = 0; e < k; ++e) w[base + order[lo + e]] = pw[e];
        }
    }
}

} // namespace vec
} // namespace volfi_near_certified
#endif // VOLFI_NEAR_CERTIFIED_VEC_HPP
