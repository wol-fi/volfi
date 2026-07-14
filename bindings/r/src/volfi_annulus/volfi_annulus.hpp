// volfi_annulus.hpp  --  runtime for the volfi-annulus inverse Black-Scholes solver.
//
// Drop-in successor to volfi::implied_variance_otm.  Given
//     h = |log(K/F)| >= 0   and   c = normalized OTM call price in (0,1),
// returns the Black-Scholes total variance  w = sigma^2 * T.
//
// -------------------------------------------------------------------------
//  Architecture (Phase 6 -- B1 + A2 + A1 regeneration cycle).
//
//  SHARED KERNELS (bit-identical in the generator, the scalar runtime and the
//  SIMD runtime -- this is the machinery behind BOTH non-negotiable invariants,
//  machine precision AND scalar==batch bit-identity):
//    * log2approx(m), m in [1,2)     -- B1: OUR OWN branchless, accurate log2
//        t=(m-1)/(m+1); u=t*t; s=LOG2_S_A*u+LOG2_S_B; g=Clenshaw(LOG2_GC,s);
//        log2approx = (LOG2_SCALE*t)*g            (Chebyshev deg 10 in s)
//      Replaces the scalar-only std::log2 that was the batch index-pass
//      bottleneck.  Vectorized 8-wide (AVX-512) and 4-wide (AVX2+FMA) so the
//      index pass VECTORIZES; written with EXPLICIT std::fma / _mm*_fmadd so
//      scalar and every SIMD width are BIT-IDENTICAL per lane.
//    * sigma0_poly(c) = (SIGMA0_SCALE*c)*Clenshaw(ERFINV_GC, ERFINV_TSC*c*c+TBIAS)
//        -- A1: shared branchless erfinv/qnorm (sigma at h=0), deg 17.
//    * clenshaw2(dp,dl,C,xh,xl)      -- bivariate Chebyshev via Tl-basis + inner
//        Horner + outer Clenshaw, all EXPLICIT std::fma; the AVX-512 twin uses
//        the identical fma order per lane.
//
//  BROAD-RANGE (h,c) ROUTING  (Phase-7 -- extends v up to ~8, |x|=h up to ~16).
//  Four charts tile the domain; the seam prices cw=cwing(h), ct2=c2(h) (1-D in h,
//  precomputed once per context by br::) are the branchless route predicate:
//        c <  cw                    -> WING     (true W>=3; unchanged evaluator)
//        h <  H_ATM_HI:  c<=ct_left ? LEFT : RIGHT   (ct_left=C(h,1.70) seam, see below)
//        h <= H_BOX && c<=ct2       -> CENTRAL  (the Phase-6 main table, unchanged)
//        else                       -> RIGHT    (v>2 endpoint / h>H_BOX)
//    In the h<H_ATM_HI band the LEFT<->RIGHT seam is at v=1.70 (price ct_left),
//    NOT v=2: the LEFT finisher extrapolates for v->2 (the h=0.3/v=2 corner), while
//    RIGHT's exact-equation Newton is machine-precise for v>=1.55.  ct_left is a
//    per-context (h-only) price, so the predicate stays branchless in the hot loop.
//    (h==0 -> exact ATM line.)  See volfi_annulus_broadrange.hpp for the recipes.
//
//   * CENTRAL (main table, h in [H_ATM_HI,H_BOX], cw<=c<=ct2):  stores W=h^2/(2w).
//        b   = clamp((bits(h)>>H_SHIFT) - OFF_H, 0, NB-1);  xh = h*HSCALE[b]+HBIAS[b]
//        cell selected by (k,sub) A2 split; xl = CELL_LSCALE*log2approx(m)+CELL_LBIAS
//        W = clenshaw2(...);  w = h*h/(2W).  Its own k-routing still guards the
//        c->1 ATM edge (k>KHI) and the deep seam (k<KLO -> wing/refined).
//   * LEFT (h<H_ATM_HI, cw<=c<=ct2):  br::left_variance -- rho=c/expm1(h);
//        A=binv(rho); s=h/A; v=V0(s)+h^2 V2+h^4 V4 + h^6*clenshaw2 finisher; w=v*v.
//   * RIGHT (c>ct2 or h>H_BOX):  br::right_variance -- erf-free 3-term seed
//        (Mills ratio exact) + fixed-cap exact-equation Newton; w=v*v.
//   * WING (c<cw):  volfi_annulus::wing_variance (resurgent GL-40, UNCHANGED).
//   The old SMALL-h residual table (region 1) and OUT-OF-BOX clamp (region 2) are
//   SUPERSEDED by LEFT and RIGHT; the SMH_* tables remain but are off the routed path.
//
//  scalar==batch: the CENTRAL table core AND the LEFT/RIGHT endpoint charts are
//  vectorized on BOTH FMA ISAs -- AVX-512 (8-wide) and AVX2+FMA (4-wide, the
//  __m256d twins).  Their explicit-fma kernels are bit-identical scalar<->SIMD.
//  Only WING and the analytic ATM/deep edges drop to the noinline `scalar_fallback`
//  (== the scalar entry), so batch is bit-identical to scalar in every ISA.  The
//  route gate (central_cell_of / grid_central_cell) is the single source of truth
//  shared by the vector core and its fallback pass, so no path-divergent floating
//  branch can disagree.
//  NOTE (cross-ISA): every intended fusion is an EXPLICIT std::fma / _mm*_fmadd
//  (that is what makes the SIMD kernels bit-identical mirrors of the scalar
//  kernels); under the locked -ffp-contract=off no other op can fuse, so the
//  no-FMA build differs from the FMA builds only through those explicit calls
//  (libm fma vs hardware -- values identical).  AVX-512, AVX2+FMA and scalar
//  builds are bit-identical; gcc and clang produce identical values.
//  NOTE (MinGW/Windows only): GCC on Win64 can emit an aligned vector spill to a
//  16-byte-aligned stack slot in the heavily-spilled grid endpoint pass, which may
//  fault depending on caller frame layout (pre-existing; affects the AVX-512 build
//  too).  The SysV/Linux target -- where this header is verified and shipped -- is
//  unaffected.  On Windows, prefer the fixed-h surface driver, which is unaffected.
//
// Header-only, C++17.  Build with the LOCKED flags:
//   -std=c++17 -O3 -march=native -ffp-contract=off -fno-fast-math
// (also builds under -mno-avx512f and -mavx2 -mno-avx512f).
// -ffp-contract=off is REQUIRED, not advisory: GCC's fp-contract=fast forms FMAs
// at the GIMPLE level in a translation-unit-dependent way (inlining context
// changes which mul+adds fuse), so the same header can compile to two different
// floating-point functions in two TUs of one program -- observed as 1-5 ulp
// scalar-vs-batch divergence on ~7% of wing quotes in one TU and zero in another
// (2026-07-13).  With contraction off, every fusion is the explicit std::fma /
// _mm*_fmadd written in the source: codegen is deterministic across TUs and
// across gcc/clang, and the hot paths lose nothing because they were written
// fma-explicit from the start.  Do NOT enable -ffast-math.
// -------------------------------------------------------------------------

#ifndef VOLFI_ANNULUS_HPP
#define VOLFI_ANNULUS_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <limits>
#include "volfi_annulus_tables.hpp"
#include "volfi_annulus_broadrange.hpp"  // broad-range chart constants (Binv, LEFT, RIGHT, seams)
#include "volfi_annulus_endpoint_vec.hpp"  // NEW: LEFT/RIGHT vectorization constants (Cheb V2/V4, erfcx/exp/qnorm)
#include "paper_volfi.hpp"   // reuse volfi::qnorm, volfi::otm_context, volfi price/Halley

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// SIMD availability: 512-wide needs AVX-512F; 256-wide needs AVX2 *and* FMA
// (the third locked build, -mavx2 -mno-avx512f, has AVX2 but NOT FMA, so it
// takes the scalar path -- still correct and bit-identical via std::fma).
#if defined(__AVX512F__)
#  define VA_SIMD512 1
#endif
#if defined(__AVX2__) && defined(__FMA__)
#  define VA_SIMD256 1
#endif

namespace volfi_annulus {

// c-exponent field mask (11 bits).  (Tables define C_EXP_BIAS = 1023.)
static const uint64_t C_EXP_MASK = 0x7FFULL;

// ============================================================================
//  WING fallback (out-of-box deep-OTM c).  Provided by volfi_annulus_wing.hpp.
//  Only reached for OUT-OF-BOX c (k < KLO[b]); no in-box quote depends on it.
// ============================================================================
double wing_variance(double h, double c);


namespace detail {

// ---- IEEE bit helpers (transcendental-free index) ----
inline uint64_t bits_of(double x) { uint64_t u; std::memcpy(&u, &x, sizeof u); return u; }
inline double   from_bits(uint64_t u) { double x; std::memcpy(&x, &u, sizeof x); return x; }
// mantissa in [1,2): keep c's 52 mantissa bits, force the biased exponent to 1023.
inline double mant12(uint64_t bc) {
    uint64_t mb = (bc & 0x000FFFFFFFFFFFFFULL) | (uint64_t(C_EXP_BIAS) << 52);
    return from_bits(mb);
}

inline double erfcx(double x) { return std::exp(x * x) * std::erfc(x); }

// cancellation-free OTM price for the small-h analytic fallback (excluded corner).
inline double price_small_h(double h, double w) {
    const double IS2 = 0.70710678118654752440;      // 1/sqrt(2)
    const double TWO_OVER_SQRTPI = 1.1283791670955126;
    double s = std::sqrt(w);
    double a_plus = -h / s + 0.5 * s;
    double delta  = s * IS2;                          // = B - A
    if (delta <= 0.42) {
        double A = -a_plus * IS2;
        double g_prev = erfcx(A);
        double g_cur  = 2.0 * A * g_prev - TWO_OVER_SQRTPI;
        double term = delta;
        double sum  = term * g_cur;
        for (int nn = 1; nn < 40; ++nn) {
            double g_next = 2.0 * A * g_cur + 2.0 * nn * g_prev;
            g_prev = g_cur; g_cur = g_next;
            term *= delta / (nn + 1);
            sum  += term * g_cur;
        }
        return 0.5 * std::exp(-A * A) * (-sum);
    }
    return volfi::phi_cdf(a_plus) - std::exp(h) * volfi::phi_cdf(a_plus - s);
}

inline double vega_w(double h, double w) {
    const double INV_SQRT_2PI = 0.39894228040143267794;
    double s = std::sqrt(w), d1 = -h / s + 0.5 * s;
    return INV_SQRT_2PI * std::exp(-0.5 * d1 * d1) / (2.0 * s);
}

} // namespace detail


// ============================================================================
//  SHARED SCALAR KERNELS  (B1 log2approx, A1 sigma0_poly, bivariate clenshaw2).
//  All fused ops are EXPLICIT std::fma -> deterministic, and the SIMD twins
//  reproduce the identical fma order per lane => scalar==batch bit-identity.
// ============================================================================

// B1: OUR OWN branchless, accurate log2 on m in [1,2) (NO std::log2, NO range
// reduction).  abs err vs mpmath log2 ~4e-16 (matches std::log2's own 1.6e-16).
inline double log2approx(double m) {
    double t  = (m - 1.0) / (m + 1.0);
    double u  = t * t;
    double s  = std::fma(LOG2_S_A, u, LOG2_S_B);
    double s2 = s + s;                                 // 2s (exact)
    double b1 = 0.0, b2 = 0.0;
    for (int k = 10; k >= 1; --k) { double b0 = std::fma(s2, b1, LOG2_GC[k]) - b2; b2 = b1; b1 = b0; }
    double g  = std::fma(s, b1, LOG2_GC[0]) - b2;
    return (LOG2_SCALE * t) * g;
}

// FULL natural log by reuse: log(x) = ln2 * ( exponent_bits(x) + log2approx(mantissa(x)) ).
// Uses the SAME validated, already-vectorized log2approx as CENTRAL's index pass, so the
// LEFT binv L-branch and RIGHT qnorm0 tail call NO libm log yet stay scalar==SIMD bit-identical.
// (De-risked in dv_left/flog_check.cpp: <=3 ULP vs std::log over the binv L-branch domain.)
static const double VA_LN2 = 0.69314718055994530942;   // ln 2
inline double full_log(double x) {
    uint64_t bc = detail::bits_of(x);
    int e = (int)((bc >> 52) & 0x7FFULL) - C_EXP_BIAS;
    return std::fma(VA_LN2, (double)e + log2approx(detail::mant12(bc)), 0.0);   // frozen product
}
// SUBNORMAL-SAFE full_log for the wing's log c (prices reach ~1e-320, where the plain
// bit-extraction misreads the exponent field): scale subnormals by 2^54 first, then
// subtract 54 from the extracted exponent.  Normal inputs take the identical path.
static const double VA_DBL_MIN = 2.2250738585072014e-308;
static const double VA_TWO54   = 1.8014398509481984e16;   // 2^54
inline double full_log_sub(double x) {
    const bool sub = (x < VA_DBL_MIN);
    double xs = sub ? x * VA_TWO54 : x;
    uint64_t bc = detail::bits_of(xs);
    int e = (int)((bc >> 52) & 0x7FFULL) - C_EXP_BIAS - (sub ? 54 : 0);
    return std::fma(VA_LN2, (double)e + log2approx(detail::mant12(bc)), 0.0);   // frozen product
}

// A1: sigma at h=0, i.e. 2*sqrt(2)*erfinv(c) via a shared branchless poly.
inline double sigma0_poly(double c) {
    double t  = c * c;
    double x  = std::fma(ERFINV_TSC, t, ERFINV_TBIAS);
    double x2 = x + x;
    double b1 = 0.0, b2 = 0.0;
    for (int k = 17; k >= 1; --k) { double b0 = std::fma(x2, b1, ERFINV_GC[k]) - b2; b2 = b1; b1 = b0; }
    double g  = std::fma(x, b1, ERFINV_GC[0]) - b2;
    return (SIGMA0_SCALE * c) * g;
}

// Bivariate Chebyshev  V = sum_i sum_j C[i*(dl+1)+j] T_i(xh) T_j(xl)  (Clenshaw).
//   dp <= 13 (row[14]), dl <= 16 (Tl[17]).  EXPLICIT std::fma throughout.
inline double clenshaw2(int dp, int dl, const double* C, double xh, double xl) {
    const int stride = dl + 1;
    double Tl[17];
    Tl[0] = 1.0; Tl[1] = xl;
    const double xl2 = 2.0 * xl;
    for (int j = 2; j <= dl; ++j) Tl[j] = std::fma(xl2, Tl[j - 1], -Tl[j - 2]);
    double row[14];
    for (int i = 0; i <= dp; ++i) {
        const double* Ci = C + i * stride;
        double acc = Ci[0];
        for (int j = 1; j <= dl; ++j) acc = std::fma(Ci[j], Tl[j], acc);
        row[i] = acc;
    }
    double b1 = 0.0, b2 = 0.0;
    const double xh2 = 2.0 * xh;
    for (int i = dp; i >= 1; --i) { double b0 = std::fma(xh2, b1, row[i]) - b2; b2 = b1; b1 = b0; }
    return std::fma(xh, b1, row[0]) - b2;
}

inline double clenshaw2_cell(int ci, double xh, double xl) {
    return clenshaw2(CELL_DP[ci], CELL_DL[ci], &COEFFS[CELL_OFF[ci]], xh, xl);
}
inline double clenshaw2_smh_cell(int cell, double xh, double xl) {
    return clenshaw2(SMH_CELL_DP[cell], SMH_CELL_DL[cell], &SMH_COEFFS[SMH_CELL_OFF[cell]], xh, xl);
}


// ============================================================================
//  BROAD-RANGE CHART EVALUATORS  (br::).  Scalar, branchless-per-chart, built on
//  the SHARED kernels above (sigma0_poly, clenshaw2) plus volfi::qnorm/phi_cdf.
//  These are the LEFT matched chart, the RIGHT endpoint chart, the h-free Binv
//  inverse, and the 1-D-in-h seam price curves cwing(h),c2(h).  Ported verbatim
//  (identical fma order) from the VERIFIED generator float64 reference
//  (generate_broadrange.py: binv_f64/left_chart_f64/right_chart_f64/route).
//  Every fused op that the reference performs with fma() uses std::fma here.
// ============================================================================
namespace br {
namespace K = volfi_annulus_broadrange;   // constant pack

// 1-D Chebyshev series (Clenshaw) with affine [a,b]->[-1,1] baked in.  Matches
// generate_broadrange.py::clenshaw(c,a,b,x) exactly (fma order + plain affine).
inline double clenshaw1(const double* C, int n, double a, double b, double x) {
    double t  = std::fma(2.0, x, -(a + b)) / (b - a);   // EXPLICIT fma affine: contraction-independent
    double t2 = 2.0 * t;                                 //   -> the SIMD twin reproduces it bit-for-bit
    double d0 = 0.0, d1 = 0.0;
    for (int j = n - 1; j >= 1; --j) { double b0 = std::fma(t2, d0, C[j]) - d1; d1 = d0; d0 = b0; }
    return std::fma(t, d0, C[0]) - d1;
}

inline double exp_neg(double a);   // fwd decl (defined below); seams use the shared kernel
// ---- seam price curves (depend on h only) --------------------------------
// exp via the shared exp_neg kernel (NOT libm std::exp), so the vectorized router
// reproduces these bit-for-bit and the batch routing decision equals the scalar
// one; the ~1-ulp shift from libm exp only reclassifies quotes within a ulp of a
// seam, all inside the inter-chart overlap bands (accuracy-neutral, oracle-gated).
inline double cwing_price(double h) {            // C(h, h/sqrt6)  (W=3 wing seam)
    return h * exp_neg(clenshaw1(K::CW_COEFFS, 23, K::CW_A, K::CW_B, h));
}
inline double c2_price(double h) {               // C(h, 2)        (v=2 central seam)
    return exp_neg(clenshaw1(K::C2_COEFFS, 27, K::C2_A, K::C2_B, h));
}
inline double ctl_seam(double h) {               // C(h, 1.70)  LEFT/RIGHT seam (frozen poly)
    return clenshaw1(K::CTL_SEAM_C, 9, K::CTL_SEAM_A, K::CTL_SEAM_B, h);
}
inline double expm1_small(double h) {            // expm1(h) on (0,0.32] (frozen; no libm)
    return h * clenshaw1(K::EXPM1G_C, 11, K::EXPM1G_A, K::EXPM1G_B, h);
}

// ---- Binv:  A = B^{-1}(rho),  h-free, 3 branchless Chebyshev regimes ------
inline double binv(double rho) {
    if (rho > K::BINV_RHO_A) {
        double u = K::BR_K / (rho + 0.5);
        double x = u * u;
        return u * clenshaw1(K::BINV_CA, 15, K::BINV_U2_A, K::BINV_U2_B, x);
    }
    double L  = -full_log(rho);   // full-log by reuse (log2approx), replaces std::log for SIMD bit-identity
    double A2 = (rho > K::BINV_RHO_B)
              ? clenshaw1(K::BINV_CB1, 17, K::BINV_L_B1A, K::BINV_L_B1B, L)
              : clenshaw1(K::BINV_CB2, 25, K::BINV_L_B2A, K::BINV_L_B2B, L);
    return std::sqrt(A2);
}

// ---- LEFT chart matched-coordinate helpers -------------------------------
// V2(s), V4(s) are now 1-D Chebyshev series in s on [0, LEFT_S_CHEB_MAX] (NEW, from
// generate_endpoint_vec.py): they REPLACE the exp-bearing closed form AND its s<0.22
// series crossover, so the LEFT inner loop is branchless with NO libm exp -> vectorizable.
inline double V0_left(double s) { return sigma0_poly(K::BR_K * s); }   // shared erfinv kernel
inline double V2_left(double s) { return clenshaw1(K::LEFT_V2_CHEB, K::LEFT_V2_CHEB_N, 0.0, K::LEFT_S_CHEB_MAX, s); }
inline double V4_left(double s) { return clenshaw1(K::LEFT_V4_CHEB, K::LEFT_V4_CHEB_N, 0.0, K::LEFT_S_CHEB_MAX, s); }

// LEFT matched small-h chart -> w = sigma^2.  h in (0, H_ATM_HI), c<=c2(h).
// Reformulated inner loop (full_log binv + sigma0_poly + Cheb V2/V4 + frozen finisher):
// NO libm exp/log -> shared with the SIMD twin -> scalar==batch bit-identity.
inline double left_variance(double h, double c) {
    double rho = c / expm1_small(h);
    double A   = binv(rho);
    double s   = h / A;
    double x   = V0_left(s);
    double h2  = h * h, h4 = h2 * h2;
    double xt  = std::fma(2.0 * h2, 1.0 / K::LEFT_T_MAX, -1.0);
    double xs  = std::fma(2.0 * s,  1.0 / K::LEFT_S_MAX, -1.0);
    double fin = clenshaw2(K::LEFT_FIN_DP, K::LEFT_FIN_DL, K::LEFT_FIN_COEFFS, xt, xs);
    double v   = std::fma(h4, V4_left(s), std::fma(h2, V2_left(s), x));   // V0+h^2 V2+h^4 V4
    v          = std::fma(h4 * h2, fin, v);                                // + h^6 finisher
    return v * v;
}

// ---- RIGHT chart:  erf-free 3-term seed + FIXED-count shared-kernel Newton --
// The transcendentals in the Newton loop (Phi via erfc, and the vega g') are replaced by
// SHARED branchless kernels (erfcx_poly, exp_neg) so the loop has NO libm erfc/exp and the
// scalar path and its SIMD twin call the SAME inlines -> scalar==batch bit-identity.  The
// per-h exp(-h/2), exp(h) are libm scalars (index pass in batch), NOT in the per-c loop.
static const double BR_IS2 = 0.70710678118654752440;   // 1/sqrt(2)
// erfcx(z) on [0,5.2] via the frozen TWO-PIECE Chebyshev (C2: chain 36 -> 22).  The piece
// selection is a plain comparison whose selected coefficients/bounds are the same doubles the
// SIMD twins blend, and the Clenshaw order is unchanged -> scalar==SIMD bit-identity holds.
inline double erfcx_poly(double z) {
    const bool hi = (z > K::RIGHT_ERFCX2_SPLIT);
    const bool h2 = (z > K::RIGHT_ERFCX_B);       // third piece: warm-start domain only
    const double* C = h2 ? K::RIGHT_ERFCX2_C2 : (hi ? K::RIGHT_ERFCX2_C1 : K::RIGHT_ERFCX2_C0);
    const double a  = h2 ? K::RIGHT_ERFCX_B    : (hi ? K::RIGHT_ERFCX2_SPLIT : K::RIGHT_ERFCX_A);
    const double b  = h2 ? K::RIGHT_ERFCX2_B2  : (hi ? K::RIGHT_ERFCX_B      : K::RIGHT_ERFCX2_SPLIT);
    return clenshaw1(C, K::RIGHT_ERFCX2_N, a, b, z);
}
// exp(a), a<=0-ish: Cody-Waite ln2 reduction + Chebyshev poly on [-ln2/2, ln2/2], ldexp.
inline double exp_neg(double a) {
    double nf = std::floor(std::fma(a, K::INV_LN2, 0.5));   // round-to-nearest integer n
    double r  = std::fma(-nf, K::LN2_HI, a);
    r         = std::fma(-nf, K::LN2_LO, r);                // reduced arg in [-ln2/2, ln2/2]
    double p  = clenshaw1(K::EXP_C, K::EXP_C_N, -0.5 * VA_LN2, 0.5 * VA_LN2, r);
    return std::ldexp(p, (int)nf);
}
// erfc(t) via shared erfcx + exp_neg (branchless on the sign of t).
inline double erfc_shared(double t) {
    double a = std::fabs(t);
    double e = erfcx_poly(a) * exp_neg(-a * a);             // = erfc(|t|)
    return (t >= 0.0) ? e : (2.0 - e);
}
inline double phicdf_sh(double x) { return 0.5 * erfc_shared(-x * BR_IS2); }
// OTM price / (1-C) via shared Phi and the per-h scalar ehp=exp(h).  EXPLICIT fma so the
// `+/- ehp*Phi` combine cannot be contracted differently scalar-vs-SIMD.
inline double black_otm(double h, double v, double ehp) {   // normalized OTM price (c<0.5 safe)
    double hv = h / v;
    double y  = std::fma(0.5, v, -hv);
    double r  = std::fma(0.5, v,  hv);
    return std::fma(-ehp, phicdf_sh(-r), phicdf_sh(y));
}
inline double onem_otm(double h, double v, double ehp) {    // 1-C, cancellation-free (c>0.5 safe)
    double hv = h / v;
    double y  = std::fma(0.5, v, -hv);
    double r  = std::fma(0.5, v,  hv);
    return std::fma(ehp, phicdf_sh(-r), phicdf_sh(-y));
}
// Deduplicated residual  c - C(h,v)  for the HH3 finisher (and the warm-start polisher),
// in whichever cancellation-safe form applies: c<1/2 uses c - black_otm, else onem_otm - (1-c).
// black_otm and onem_otm share Phi(-r), and Phi(y)/Phi(-y) differ only in the sign selection of
// one erfc core, so everything is computed ONCE: 2 erfcx + 2 exp per call instead of the 4 + 4
// a branchless evaluation of both prices would cost.  Every subexpression and predicate below
// reproduces the black_otm/onem_otm/phicdf_sh/erfc_shared chain exactly (same fma order, same
// sign-flip forms), so the returned double is bit-identical to the two-function formulation.
inline double price_residual(double h, double v, double ehp, double c, double onec, bool lo) {
    double hv = h / v;
    double y  = std::fma(0.5, v, -hv);
    double r  = std::fma(0.5, v,  hv);
    double t1 = -y * BR_IS2;                                  // arg of erfc inside phicdf_sh(y)
    double t2 = -(-y) * BR_IS2;                               // arg inside phicdf_sh(-y)
    // erfc cores with a COMPENSATED squared argument: the rounding of a*a costs a relative
    // a^2*eps in exp(-a^2), which the deep-tail price cancellation amplifies to ~W*eps in
    // sigma.  Split a^2 = s + sc exactly (one fma) and multiply by (1 - sc) ~ e^{-sc}.
    double ay  = std::fabs(t1);
    double sy  = ay * ay;
    double scy = std::fma(ay, ay, -sy);                       // exact low part of ay^2
    double ey  = erfcx_poly(ay) * (exp_neg(-sy) * (1.0 - scy));
    double phi_y  = 0.5 * ((t1 >= 0.0) ? ey : (2.0 - ey));    // = phicdf_sh(y)
    double phi_my = 0.5 * ((t2 >= 0.0) ? ey : (2.0 - ey));    // = phicdf_sh(-y)
    double tr  = -(-r) * BR_IS2;                              // arg inside phicdf_sh(-r)
    double ar  = std::fabs(tr);
    double sr  = ar * ar;
    double scr = std::fma(ar, ar, -sr);
    double er  = erfcx_poly(ar) * (exp_neg(-sr) * (1.0 - scr));
    double phi_r = 0.5 * ((tr >= 0.0) ? er : (2.0 - er));     // = phicdf_sh(-r)
    double bl = std::fma(-ehp, phi_r, phi_y);                 // = black_otm(h,v,ehp)
    double om = std::fma( ehp, phi_r, phi_my);                // = onem_otm(h,v,ehp)
    return lo ? (c - bl) : (om - onec);
}
// Acklam rational quantile SEED (gbar in (0,0.5): tail p<QN_P_LOW uses full_log; else middle).
// Only a Newton seed -> its ~1e-9 accuracy is washed out by the 6 exact-equation steps.
inline double qnorm0_seed(double p) {
    // middle branch  (QN_P_LOW <= p <= 0.5 < QN_P_HIGH)
    double q = p - 0.5, r = q * q;
    double num = K::QN_A[0];
    for (int i = 1; i < 6; ++i) num = std::fma(num, r, K::QN_A[i]);
    num *= q;
    double den = K::QN_B[0];
    for (int i = 1; i < 5; ++i) den = std::fma(den, r, K::QN_B[i]);
    den = std::fma(den, r, 1.0);
    double mid = num / den;
    // tail branch  (p < QN_P_LOW):  q = sqrt(-2 log p)  via full_log (no libm log)
    double qt = std::sqrt(-2.0 * full_log(p));
    double numt = K::QN_C[0];
    for (int i = 1; i < 6; ++i) numt = std::fma(numt, qt, K::QN_C[i]);
    double dent = K::QN_D[0];
    for (int i = 1; i < 4; ++i) dent = std::fma(dent, qt, K::QN_D[i]);
    dent = std::fma(dent, qt, 1.0);
    double tail = numt / dent;
    return (p < K::QN_P_LOW) ? tail : mid;
}
// RIGHT endpoint chart -> w = sigma^2.  v>2 (c>c2(h)) or h>H_BOX.
inline double right_variance(double h, double c) {
    double eh   = std::exp(-0.5 * h);       // per-h libm scalar (hoisted to the index pass in batch)
    double ehp  = std::exp(h);              // per-h libm scalar
    double gbar = 0.5 * (1.0 - c) * eh;
    double x0   = -qnorm0_seed(gbar);       // lower-tail quantile seed
    double m    = K::SQRT_PI2 * erfcx_poly(x0 * BR_IS2);   // Mills ratio via shared erfcx
    double inv  = 1.0 / x0;
    double x2   = -0.125 * (inv - m);
    double B2p  = (std::fma(inv * inv, inv, -inv) + m) * (1.0 / 3.0);
    double ta   = 0.5 * x0 * x2 * x2;
    double tb   = 0.125 * x2 * inv * inv;
    double x4   = std::fma(B2p, (1.0 / 128.0), ta + tb);
    double h2   = h * h;
    double x    = std::fma(h2 * h2, x4, std::fma(h2, x2, x0));   // fixed fma order
    double onec = 1.0 - c;
    // Householder-3 (order-4) finisher.  The residual f and its first derivative gp are the same
    // as for Newton (one price evaluation + one g'); the 2nd/3rd derivative ratios come from the
    // closed-form logarithmic derivative log|g'| = const - x^2/2 - h^2/(8 x^2):
    //   a = g''/g'  = L   = -x + h^2/(4 x^3),   b = g'''/g' = L^2 + L' = L^2 - 1 - 3 h^2/(4 x^4).
    // EXPLICIT fma throughout (ffp-contract=fast would otherwise contract 1-0.5*au differently in
    // scalar vs SIMD); the AVX-512/AVX2 twins reproduce this exact op sequence lane for lane.
    const bool lo = (c < 0.5);
    for (int it = 0; it < K::RIGHT_HH3_STEPS; ++it) {            // FIXED count, no early-exit
        double vv  = 2.0 * x;
        double cmC = price_residual(h, vv, ehp, c, onec, lo);
        double f   = 0.5 * cmC * eh;
        double gp  = -K::BR_K * exp_neg(-0.5 * x * x) * exp_neg(-h2 / (8.0 * x * x));
        double u   = f / gp;
        double ix  = 1.0 / x;
        double ix2 = ix * ix;
        double a   = std::fma(0.25 * h2, ix2 * ix, -x);         // L  = -x + (h^2/4)/x^3
        double Lp  = std::fma(-0.75 * h2, ix2 * ix2, -1.0);     // L' = -1 - (3h^2/4)/x^4
        double b   = std::fma(a, a, Lp);                        // b  = L^2 + L'
        double au  = a * u;
        double num = std::fma(-0.5, au, 1.0);                   // 1 - 0.5*a*u
        double den = std::fma(b * u, u * (1.0 / 6.0), 1.0 - au);// 1 - a*u + (1/6) b u^2
        x = x - u * num / den;
    }
    double v = 2.0 * x;
    return v * v;
}

// ---- D: warm-start refinement (no routing, no seed machinery) ----------------
// Given the previous variance w_prev for the same (h, strike) -- e.g. the last calibration
// iteration's result -- refine to the new price c by WARM_HH3_STEPS fixed Householder-3 steps
// on the exact OTM equation, using the same shared kernels as the RIGHT finisher.  Valid when
// W = h^2/(2 w_prev) <= 40 and the vol moved by no more than a few percent (see the basin note
// in volfi_annulus_endpoint_vec.hpp); the caller re-inverts cold outside that basin.  The two
// per-quote exponentials go through the shared exp kernel (whose Cody--Waite reduction is
// sign-agnostic), NOT libm, so the mixed-h warm batch is bit-identical to this scalar entry.
inline double warm_variance(double h, double c, double w_prev,
                            int steps = K::WARM_HH3_STEPS, double* last_rel = nullptr) {
    double x    = 0.5 * std::sqrt(w_prev);
    double h2   = h * h;
    double eh   = exp_neg(-0.5 * h);
    double ehp  = exp_neg(h);              // e^{+h}: reduction and scaling are sign-agnostic
    double onec = 1.0 - c;
    const bool lo = (c < 0.5);
    double corr = 0.0;
    for (int it = 0; it < steps; ++it) {
        double vv  = 2.0 * x;
        double cmC = price_residual(h, vv, ehp, c, onec, lo);
        double f   = 0.5 * cmC * eh;
        double gp  = -K::BR_K * exp_neg(-0.5 * x * x) * exp_neg(-h2 / (8.0 * x * x));
        double u   = f / gp;
        double ix  = 1.0 / x;
        double ix2 = ix * ix;
        double a   = std::fma(0.25 * h2, ix2 * ix, -x);
        double Lp  = std::fma(-0.75 * h2, ix2 * ix2, -1.0);
        double b   = std::fma(a, a, Lp);
        double au  = a * u;
        double num = std::fma(-0.5, au, 1.0);
        double den = std::fma(b * u, u * (1.0 / 6.0), 1.0 - au);
        corr = u * num / den;
        x = x - corr;
    }
    if (last_rel) *last_rel = std::fabs(corr) / std::fabs(x);   // convergence indicator
    double v = 2.0 * x;
    return v * v;
}

// Fast warm refinement for the live streaming regime (sub-1% moves): 2 HH3 steps
// instead of 3.  Sets *ok=false when the move was too large for 2 steps to converge
// (see WARM_FAST_TOL); the caller then re-inverts cold.  When *ok is true the result
// matches the 3-step warm to the warm floor (de-risked for |dv/v| <= 1%, all W<=40).
inline double warm_variance_fast(double h, double c, double w_prev, bool* ok) {
    double lr;
    double w = warm_variance(h, c, w_prev, K::WARM_HH3_FAST_STEPS, &lr);
    if (ok) *ok = (lr <= K::WARM_FAST_TOL) && (w > 0.0);
    return w;
}

// ---- Rescue branch for the beta > WING_B_HI wedge (h > ~16.2) -----------------
// The fitted wing tables cover beta = h^2/16 <= 16.4 only; beyond that the
// (clamped) fit would return a plausible-looking but wrong variance (up to
// ~1.7e-2 rel-sigma, see uncovered_beta_gt_16p4.csv).  These quotes are
// unreachable by any real market (K/F > 1.1e7) but representable, so a mature
// inverter must answer them correctly or flag them -- never silently drift.
//
// Method (de-risked 2026-07-13, seed sweep + basin probe):
//   1. Phi-seed: drop the e^h Phi(-h/v-v/2) term of the OTM price and solve
//      Phi(v/2 - h/v) = c  =>  v0 = z + sqrt(z^2 + 2h), z = Phi^{-1}(c); then
//      restore the dropped term by 3 fixed-point passes.  Worst seed error over
//      h in [16.2, 30]: 0.23% (0.92% at the low-c edge) -- far inside the
//      3-step HH3 basin (~ +-10-15%).
//   2. Finish with the exact-equation Householder-3 refiner (warm_variance),
//      whose price kernel needs erfcx argument ay = (h/v + v/2)/sqrt2 <= ~7.1
//      (equivalently W <~ 40, the warm basin).  Deeper quotes (c <~ 1e-40 at
//      these h) return quiet NaN: OUT_OF_DOMAIN, never garbage.
// Scalar by design: every driver (scalar, AVX-512, AVX2 grid batch) calls THIS
// function for beta > WING_B_HI, so scalar==batch bit-identity is structural.
inline double wing_rescue_variance(double h, double c) {
    double ceff = c;
    double v0   = 1.0;
    for (int it = 0; it < 3; ++it) {
        double z = qnorm0_seed(ceff);                       // Phi^{-1}(ceff)
        v0 = z + std::sqrt(std::fma(z, z, 2.0 * h));        // v^2/2 - z v - h = 0
        double x  = h / v0 + 0.5 * v0;                      // -d2
        double ay = x * BR_IS2;
        if (!(ay <= 7.1)) break;              // dropped term negligible: seed done
        double d1 = h / v0 - 0.5 * v0;
        double t  = 0.5 * erfcx_poly(ay) * exp_neg(-0.5 * d1 * d1);
        double cn = c + t;                    // e^h Phi(-x) = 0.5 erfcx(ay) e^{-d1^2/2}
        ceff = (cn < 1.0) ? cn : ceff;        // stay in qnorm domain
    }
    double ayf = (h / v0 + 0.5 * v0) * BR_IS2;
    if (!(ayf <= 7.1)) return std::numeric_limits<double>::quiet_NaN();
    return warm_variance(h, c, v0 * v0);
}

} // namespace br

// ============================================================================
//  WING runtime (A): fixed-count Newton on  W = Ltilde - 1.5 log W + log H(W),
//  with log H from the two-piece Chebyshev fit of S(u,beta) = logH/u + (beta+3/2)
//  (volfi_annulus_wing.hpp) or, for the deep tail, the asymptotic series.  The
//  piece/series regime is chosen ONCE per quote from Ltilde (the overlap bands
//  absorb the seed error), all logarithms go through the shared full_log kernels,
//  and the step count is FIXED -> the SIMD twins are bit-identical per lane.
//  The Gauss-Laguerre quadrature is retired to the offline oracle.
// ============================================================================
namespace wing_rt {
namespace wd = wing_detail;

// S(u,beta) and dS/du on one piece: inner Clenshaw over beta per u-row, outer
// Clenshaw over u; d/du via the Chebyshev derivative-coefficient recurrence on
// the same rows (value and derivative are consistent by construction).
inline double wing_S_d(const double* C, int NU, int NB, double ulo, double uhi,
                       double u, double beta, double* dS_du) {
    double xu = std::fma(2.0, u,    -(ulo + uhi)) / (uhi - ulo);
    double xb = std::fma(2.0, beta, -(wd::WING_B_LO + wd::WING_B_HI))
                / (wd::WING_B_HI - wd::WING_B_LO);
    double rows[27];
    const double tb2 = 2.0 * xb;
    for (int i = 0; i < NU; ++i) {
        const double* Ci = C + i * NB;
        double d0 = 0.0, d1 = 0.0;
        for (int j = NB - 1; j >= 1; --j) { double b0 = std::fma(tb2, d0, Ci[j]) - d1; d1 = d0; d0 = b0; }
        rows[i] = std::fma(xb, d0, Ci[0]) - d1;
    }
    const double tu2 = 2.0 * xu;
    double d0 = 0.0, d1 = 0.0;
    for (int i = NU - 1; i >= 1; --i) { double b0 = std::fma(tu2, d0, rows[i]) - d1; d1 = d0; d0 = b0; }
    double S = std::fma(xu, d0, rows[0]) - d1;
    double dp[27];
    dp[NU - 1] = 0.0;
    dp[NU - 2] = 2.0 * (NU - 1) * rows[NU - 1];
    for (int k = NU - 3; k >= 0; --k)
        dp[k] = std::fma(2.0 * (k + 1), rows[k + 1], dp[k + 2]);   // EXPLICIT fma (SIMD mirrors)
    dp[0] *= 0.5;
    d0 = 0.0; d1 = 0.0;
    for (int i = NU - 1; i >= 1; --i) { double b0 = std::fma(tu2, d0, dp[i]) - d1; d1 = d0; d0 = b0; }
    double dS_dxu = std::fma(xu, d0, dp[0]) - d1;
    *dS_du = dS_dxu * (2.0 / (uhi - ulo));
    return S;
}

// log H + analytic (log H)'(W) from the fitted piece.  u is clamped into the
// piece domain so Newton intermediates cannot leave it; the final W is interior
// by the Ltilde bucketing overlap.
inline double wing_logH_fit_d(double W, double beta, bool pieceA, double* dlogH) {
    const double* C; int NU, NB; double ulo, uhi;
    if (pieceA) { C = wd::WING_SA; NU = wd::WING_SA_NU; NB = wd::WING_SA_NB;
                  ulo = wd::WING_UA_LO; uhi = wd::WING_UA_HI; }
    else        { C = wd::WING_SB; NU = wd::WING_SB_NU; NB = wd::WING_SB_NB;
                  ulo = wd::WING_UB_LO; uhi = wd::WING_UB_HI; }
    double u = 1.0 / W;
    u = std::fmin(std::fmax(u, ulo), uhi);
    double Su;
    double S  = wing_S_d(C, NU, NB, ulo, uhi, u, beta, &Su);
    double bp = beta + 1.5;
    *dlogH = std::fma(-(u * u), std::fma(u, Su, S) - bp, 0.0);   // frozen product
    return u * (S - bp);
}
} // namespace wing_rt

// Ltilde = log(1/c) + log C(h), the wing's regime selector and Newton constant.
// Shared by wing_variance and the batch drivers' wing bucketing, so both pick the
// same piece/series regime for every quote.
inline double wing_Lt(double h, double c) {
    double lc = full_log(h) - wing_detail::WING_LOG4;
    lc = std::fma(0.5, h, lc) - wing_detail::WING_HLOGPI;    // = log C(h)
    return -full_log_sub(c) + lc;
}

// wing_variance(h, c): w = sigma^2 for small OTM price c (c < c_wing(h)), h > 0.
// Valid down to the smallest positive doubles (subnormal-safe log c).
inline double wing_variance(double h, double c) {
    using namespace wing_detail;
    const double beta = h * h * (1.0 / 16.0);
    // beta beyond the fitted table range -> Phi-seed rescue branch (never clamp:
    // a clamped fit returns plausible-looking garbage; see wing_rescue_variance).
    if (!(beta <= WING_B_HI)) return br::wing_rescue_variance(h, c);
    const double Lt = wing_Lt(h, c);
    const double lt = std::fmax(Lt, 2.0);
    double W = std::fmax(std::fma(-1.5, full_log(lt), lt), 2.6);
    const bool series = (Lt > WING_LT_BS);
    const bool pieceA = (Lt <= WING_LT_AB);
    for (int it = 0; it < WING_NEWTON_STEPS; ++it) {
        double dlH;
        const double lH = series ? logH_series_d(W, beta, &dlH)
                                 : wing_rt::wing_logH_fit_d(W, beta, pieceA, &dlH);
        const double F  = (W - Lt) + std::fma(1.5, full_log(W), -lH);
        const double Fp = 1.0 + 1.5 / W - dlH;
        W = std::fmax(W - F / Fp, 2.6);
    }
    return (h * h) / (2.0 * W);
}

// Single source of truth for MAIN-table routing: the global cell id for an
// in-band quote, or -1 if it must go to a scalar fallback (wing / analytic-ATM /
// infeasible sub-cell / c-out-of-octave).  Scalar entry, band pass, grid pass
// and every fallback pass all agree by calling THIS.
inline int main_cell_of(int band, uint64_t bc) {
    int k = (int)((bc >> 52) & C_EXP_MASK) - C_EXP_BIAS;
    if (k > KHI[band] || k < KLO[band]) return -1;
    int oc = OCTBASE[band] + (k - KLO[band]);
    int sb = OCT_SBITS[oc];
    int sub = sb ? (int)((bc >> (52 - sb)) & ((1u << sb) - 1u)) : 0;
    if (sub >= OCT_NSUB[oc]) return -1;
    return CBASE[band] + OCT_CELLOFF[oc] + sub;
}

// small-h band index (0..SMH_NB-1) from h.
inline int smh_band_of(double h) {
    for (int b = 0; b < SMH_NB - 1; ++b) if (h < SMH_BAND_HI[b]) return b;
    return SMH_NB - 1;
}

// Single source of truth for SMALL-h routing (mirrors main_cell_of): the global
// small-h cell id for an in-band quote, or -1 if it must go to the analytic
// fallback (c out of octave / excluded deep corner).  The scalar entry
// (smh_table_variance), the smh band pass and the grid smh pass all agree by
// deriving the cell the same way; this helper lets the batch index pass and its
// fallback pass share the identical decision.
inline int smh_cell_of(int sb, uint64_t bc) {
    int k = (int)((bc >> 52) & C_EXP_MASK) - C_EXP_BIAS;
    if (k < SMH_KLO[sb] || k > SMH_KHI[sb]) return -1;
    int oc = SMH_OCTBASE[sb] + (k - SMH_KLO[sb]);
    return SMH_OCTCELL[oc];   // may be -1 (excluded corner)
}


// ============================================================================
//  Analytic fallbacks  [VALIDATED Phase-2/3, reused verbatim -- serve only the
//  excluded small-h corner, the c->1 analytic edge and the out-of-box sliver].
// ============================================================================
inline double atm_small_h_variance(const volfi::otm_context& vq, double c) {
    double w = volfi::seed_otm(vq, c);
    for (int it = 0; it < 4; ++it) {
        double f  = detail::price_small_h(vq.h, w) - c;
        double fp = detail::vega_w(vq.h, w);
        w -= f / fp;
    }
    return w;
}

inline double refined_general_variance(const volfi::otm_context& vq, double c) {
    double w = volfi::implied_variance_otm(vq, c);
    for (int it = 0; it < 3; ++it) {
        double f  = detail::price_small_h(vq.h, w) - c;
        double fp = detail::vega_w(vq.h, w);
        w -= f / fp;
    }
    return w;
}

// Exact ATM line (h==0):  c = erf(s/(2 sqrt2)).
inline double atm_line_variance(double c) {
    const double INV_SQRT_2PI = 0.39894228040143267794;
    const double INV_2SQRT2   = 0.35355339059327376220;   // 1/(2 sqrt2)
    double s = 2.0 * volfi::qnorm(0.5 * (1.0 + c));
    for (int it = 0; it < 2; ++it) {
        double f  = std::erf(s * INV_2SQRT2) - c;
        double fp = INV_SQRT_2PI * std::exp(-0.125 * s * s);
        s -= f / fp;
    }
    return s * s;
}

static const double WING_W_MIN = 3.0;


// ============================================================================
//  context: everything the hot path needs for a fixed h (resolved once by pure
//  bit-extraction).  The volfi context (analytic fallbacks only) is built lazily.
// ============================================================================
struct context {
    double h;
    double h2;

    // Broad-range routing precompute (depend on h only): the 1-D-in-h seam prices.
    //   c <  cw   -> WING ;   c <= ct2 (with c>=cw) -> LEFT/CENTRAL ;  else RIGHT.
    double cw;           // = br::cwing_price(h)  (W=3 wing seam price)
    double ct2;          // = br::c2_price(h)     (v=2 central/right seam price)
    double ct_left;      // = C(h, LEFT_RIGHT_VSEAM) (region-1 LEFT<->RIGHT seam, v=1.70)

    // region: 0 = central-capable band (H_ATM_HI<=h<=H_BOX); 1 = LEFT band (h<H_ATM_HI);
    //         2 = RIGHT-only band (h>H_BOX).  Only region 0 carries a vectorizable
    //         central subset; regions 1/2 are entirely LEFT/RIGHT/WING (scalar batch).
    int    region;
    int    band;         // main-table band (region 0 only)
    double xh;           // affine h -> [-1,1] (region 0 only)

    // Per-h scalars hoisted for the vectorized LEFT/RIGHT batch inner loops (libm, once per h).
    double eh;           // = exp(-h/2)   (RIGHT)
    double ehp;          // = exp(h)      (RIGHT price)
    double expm1h;       // = expm1(h)    (LEFT rho = c/expm1h)
    double xt_left;      // = 2*h^2/LEFT_T_MAX - 1  (LEFT finisher xh; region 1 only, else unused)

    mutable std::optional<volfi::otm_context> vq_cache;
    const volfi::otm_context& volfi_ctx() const {
        if (!vq_cache) vq_cache.emplace(h);
        return *vq_cache;
    }

    explicit context(double x)
        : h(x), h2(x * x), cw(0.0), ct2(0.0), ct_left(0.0), region(0), band(0), xh(0.0),
          eh(0.0), ehp(0.0), expm1h(0.0), xt_left(0.0)
    {
        cw  = br::cwing_price(x);
        ct2 = br::c2_price(x);
        eh     = std::exp(-0.5 * x);
        ehp    = std::exp(x);
        expm1h = br::expm1_small(x);   // frozen poly (LEFT band only); matches left_variance / SIMD
        xt_left = std::fma(2.0 * (x * x), 1.0 / volfi_annulus_broadrange::LEFT_T_MAX, -1.0);
        if (x < H_ATM_HI) {                                     // LEFT band
            region = 1;
            // v=1.70 LEFT<->RIGHT seam price = C(h,1.70), frozen degree-8 poly (see
            // broadrange CTL_SEAM_*); the grid router uses the SAME fit -> identical routing.
            ct_left = br::ctl_seam(x);
            return;
        }
        if (x > H_BOX)    { region = 2; band = NB - 1; return; } // RIGHT-only band
        // region 0: central-capable main-table band
        uint64_t bh = detail::bits_of(x);
        long bi = (long)(bh >> H_SHIFT) - (long)OFF_H;
        if (bi < 0) bi = 0; else if (bi > NB - 1) bi = NB - 1;
        band = (int)bi;
        double v = std::fma(x, HSCALE[bi], HBIAS[bi]);
        if (v < -1.0) v = -1.0; else if (v > 1.0) v = 1.0;
        xh = v;
    }
};

// Single source of truth for the CENTRAL vectorizable subset: returns the global
// main-table cell id iff the new (h,c) routing sends this quote to CENTRAL *and*
// it lands on a real table cell; -1 otherwise (WING / LEFT / RIGHT / analytic edge
// -> handled by the scalar fallback).  Used by the batch table pass AND its
// fallback pass so they agree bit-for-bit with the scalar entry.
inline int central_cell_of(const context& q, double c, uint64_t bc) {
    if (c < q.cw || c > q.ct2) return -1;     // WING (c<cw) or RIGHT (c>ct2)
    return main_cell_of(q.band, bc);          // CENTRAL table cell, or -1 (analytic edge)
}


// Sub-wing router (a CENTRAL quote whose table cell falls below KLO): deep-OTM c.
inline double subwing_variance(const context& q, double c) {
    double w_wing = wing_variance(q.h, c);
    double W_wing = q.h2 / (2.0 * w_wing);
    if (W_wing >= WING_W_MIN) return w_wing;
    return refined_general_variance(q.volfi_ctx(), c);
}

// CENTRAL table inverse (unchanged Phase-6 main-table body): reached only when the
// (h,c) routing selects CENTRAL (H_ATM_HI<=h<=H_BOX, cw<=c<=c2).  Its own internal
// k-routing still guards the analytic ATM edge (k>KHI) and the deep seam (k<KLO).
inline double central_variance(const context& q, double c) {
    uint64_t bc = detail::bits_of(c);
    int k = (int)((bc >> 52) & C_EXP_MASK) - C_EXP_BIAS;
    if (k > KHI[q.band]) return refined_general_variance(q.volfi_ctx(), c);  // c->1 ATM
    if (k < KLO[q.band]) return subwing_variance(q, c);                       // deep-OTM wing
    int oc  = OCTBASE[q.band] + (k - KLO[q.band]);
    int sb  = OCT_SBITS[oc];
    int sub = sb ? (int)((bc >> (52 - sb)) & ((1u << sb) - 1u)) : 0;
    if (sub >= OCT_NSUB[oc]) return refined_general_variance(q.volfi_ctx(), c); // infeasible sub
    int cell = CBASE[q.band] + OCT_CELLOFF[oc] + sub;
    double m  = detail::mant12(bc);
    double xl = std::fma(CELL_LSCALE[cell], log2approx(m), CELL_LBIAS[cell]);
    double W  = clenshaw2_cell(cell, q.xh, xl);
    return q.h2 / (2.0 * W);
}


// ============================================================================
//  Scalar entry -- the derived branchless (h,c) routing predicate.
//    c <  cw(h)                       -> WING            (true W>=3)
//    h <  H_ATM_HI :  c<=c2(h)?LEFT:RIGHT
//    h <= H_BOX && c<=c2(h)           -> CENTRAL         (v<=2 box)
//    else                             -> RIGHT           (v>2 / h>H_BOX)
//  cw=cwing(h), ct2=c2(h) are precomputed once in the context (h-only).
// ============================================================================
inline double implied_variance_otm(const context& q, double c) {
    // Total input contract: any (h, c) outside the feasible open box returns
    // quiet NaN -- never a plausible-looking number.  c >= 1 has no finite
    // variance (the forward bound); c <= 0 is at/below OTM intrinsic; h < 0
    // violates the h = |log(K/F)| convention (h = NaN also fails h >= 0).
    if (!(c > 0.0) || !(c < 1.0)) return std::numeric_limits<double>::quiet_NaN();
    const double h = q.h;
    if (!(h >= 0.0)) return std::numeric_limits<double>::quiet_NaN();
    if (h == 0.0) return atm_line_variance(c);                  // exact ATM line (rho=inf guard)
    if (c < q.cw) return wing_variance(h, c);                   // WING
    if (q.region == 1)                                          // h < H_ATM_HI
        // LEFT owns v<=1.70 (finisher in-domain); RIGHT's exact Newton owns v>1.70,
        // clearing the former h=0.3/v=2 LEFT-finisher extrapolation corner.
        return (c <= q.ct_left) ? br::left_variance(h, c) : br::right_variance(h, c);
    if (q.region == 0 && c <= q.ct2) return central_variance(q, c);  // CENTRAL (v<=2 box)
    return br::right_variance(h, c);                            // RIGHT (v>2 or h>H_BOX)
}

inline double implied_variance_otm(double h, double c) {
    context q(h);
    return implied_variance_otm(q, c);
}


// ============================================================================
//  Checked entry points -- the recommended API for production callers.
//  Same numerics as implied_variance_otm (bit-identical when status == ok or
//  near_saturation); adds a machine-readable diagnosis of every non-result.
// ============================================================================
enum class iv_status : int {
    ok = 0,
    below_intrinsic,  // c <= 0: at/below the OTM intrinsic value (crossed/stale quote)
    above_max,        // c >= 1: at/above the forward bound (no finite variance)
    bad_input,        // NaN input, or h < 0 (h = |log(K/F)| convention violated)
    out_of_domain,    // h > 16.2 deep-tail corner the rescue branch cannot certify
                      // (K/F > 1.1e7 AND c <~ 1e-40: unreachable by any market feed)
    near_saturation   // valid result, but v = sigma*sqrt(T) > 8: accuracy is now
                      // limited by the double representation of 1-c itself (from
                      // ~1e-13 rel at v=8.5 to ~1e-3 at the v~16.6 edge where
                      // 1-c reaches machine epsilon).  Any double-price inverter
                      // shares this floor; treat sigma as approximate.
};

inline double implied_variance_otm_checked(double h, double c, iv_status* status) {
    iv_status st = iv_status::ok;
    if (std::isnan(h) || std::isnan(c) || h < 0.0) st = iv_status::bad_input;
    else if (c <= 0.0)                             st = iv_status::below_intrinsic;
    else if (c >= 1.0)                             st = iv_status::above_max;
    if (st != iv_status::ok) {
        if (status) *status = st;
        return std::numeric_limits<double>::quiet_NaN();
    }
    double w = implied_variance_otm(h, c);
    if (std::isnan(w))      st = iv_status::out_of_domain;
    else if (w > 64.0)      st = iv_status::near_saturation;    // v > 8
    if (status) *status = st;
    return w;
}

// Industry-convention wrapper: raw UNDISCOUNTED (forward) option price.
//   forward F > 0, strike K > 0, price = option price / discount factor,
//   T > 0 in years, is_call.  Handles the ITM->OTM parity flip internally and
//   returns implied sigma (annualized).  On any non-ok status returns NaN
//   (near_saturation still returns the best-effort sigma).
inline double implied_volatility(double forward, double strike, double price,
                                 double T, bool is_call, iv_status* status = nullptr) {
    if (!(forward > 0.0) || !(strike > 0.0) || !(T > 0.0) || std::isnan(price)) {
        if (status) *status = iv_status::bad_input;
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double k = strike / forward;
    double p = price / forward;                       // normalized price
    // parity flip to the OTM twin: C - P = F - K  =>  c = p_call - (1 - k), etc.
    if ( is_call && k < 1.0) p -= (1.0 - k);          // ITM call -> OTM put value
    if (!is_call && k > 1.0) p -= (k - 1.0);          // ITM put  -> OTM call value
    if (k < 1.0) p /= k;                              // K<F: the OTM-call twin is normalized by K, not F
    const double h = std::fabs(std::log(k));
    iv_status st;
    double w = implied_variance_otm_checked(h, p, &st);
    if (status) *status = st;
    return std::sqrt(w / T);
}


// NON-INLINE fallback shims for the batch drivers (keeping opaque calls out of
// the hot loop bodies -- an inlined erfc/Newton/wing path bloats a driver and
// blocks optimization of its index/sort/kernel loops, measured ~6-7x).  Results
// are the exact scalar entry -> BIT-IDENTICAL.
namespace detail {
__attribute__((noinline)) inline double scalar_fallback(const context& q, double c) {
    return implied_variance_otm(q, c);
}
__attribute__((noinline)) inline double scalar_fallback(double h, double c) {
    return implied_variance_otm(h, c);
}
} // namespace detail


// ============================================================================
//  SHARED SIMD KERNELS.  Each reproduces the scalar kernel's EXPLICIT fma order
//  per lane, so the vector result is bit-identical to the scalar result.
// ============================================================================
#if defined(VA_SIMD512)
namespace detail {
// mantissa(c) in [1,2) for 8 lanes.
inline __m512d mant12_avx512(__m512d c) {
    __m512i cb = _mm512_castpd_si512(c);
    __m512i mb = _mm512_or_si512(_mm512_and_si512(cb, _mm512_set1_epi64(0x000FFFFFFFFFFFFFLL)),
                                 _mm512_set1_epi64((long long)((uint64_t)C_EXP_BIAS << 52)));
    return _mm512_castsi512_pd(mb);
}
// log2approx on 8 mantissas.
inline __m512d log2approx_avx512(__m512d m) {
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d t  = _mm512_div_pd(_mm512_sub_pd(m, one), _mm512_add_pd(m, one));
    __m512d u  = _mm512_mul_pd(t, t);
    __m512d s  = _mm512_fmadd_pd(_mm512_set1_pd(LOG2_S_A), u, _mm512_set1_pd(LOG2_S_B));
    __m512d s2 = _mm512_add_pd(s, s);
    __m512d b1 = _mm512_setzero_pd(), b2 = _mm512_setzero_pd();
    for (int k = 10; k >= 1; --k) {
        __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(s2, b1, _mm512_set1_pd(LOG2_GC[k])), b2);
        b2 = b1; b1 = b0;
    }
    __m512d g  = _mm512_sub_pd(_mm512_fmadd_pd(s, b1, _mm512_set1_pd(LOG2_GC[0])), b2);
    return _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(LOG2_SCALE), t), g);
}
// sigma0_poly on 8 prices.
inline __m512d sigma0_poly_avx512(__m512d c) {
    __m512d t  = _mm512_mul_pd(c, c);
    __m512d x  = _mm512_fmadd_pd(_mm512_set1_pd(ERFINV_TSC), t, _mm512_set1_pd(ERFINV_TBIAS));
    __m512d x2 = _mm512_add_pd(x, x);
    __m512d b1 = _mm512_setzero_pd(), b2 = _mm512_setzero_pd();
    for (int k = 17; k >= 1; --k) {
        __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(x2, b1, _mm512_set1_pd(ERFINV_GC[k])), b2);
        b2 = b1; b1 = b0;
    }
    __m512d g  = _mm512_sub_pd(_mm512_fmadd_pd(x, b1, _mm512_set1_pd(ERFINV_GC[0])), b2);
    return _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(SIGMA0_SCALE), c), g);
}
// bivariate clenshaw, scalar xh, 8-lane xl.  (row[] fused into the outer
// Clenshaw so the live set stays inside 32 zmm.)  Identical fma order to scalar.
inline __m512d clenshaw2_avx512(int dp, int dl, const double* C, double xh, __m512d xl) {
    const int stride = dl + 1;
    __m512d Tl[17];
    Tl[0] = _mm512_set1_pd(1.0);
    Tl[1] = xl;
    __m512d xl2 = _mm512_mul_pd(_mm512_set1_pd(2.0), xl);
    for (int j = 2; j <= dl; ++j) Tl[j] = _mm512_fmsub_pd(xl2, Tl[j - 1], Tl[j - 2]);
    const __m512d vxh2 = _mm512_set1_pd(2.0 * xh);
    __m512d b1 = _mm512_setzero_pd(), b2 = _mm512_setzero_pd();
    for (int i = dp; i >= 1; --i) {
        const double* Ci = C + i * stride;
        __m512d acc = _mm512_set1_pd(Ci[0]);
        for (int j = 1; j <= dl; ++j) acc = _mm512_fmadd_pd(_mm512_set1_pd(Ci[j]), Tl[j], acc);
        __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(vxh2, b1, acc), b2);
        b2 = b1; b1 = b0;
    }
    __m512d acc0 = _mm512_set1_pd(C[0]);
    for (int j = 1; j <= dl; ++j) acc0 = _mm512_fmadd_pd(_mm512_set1_pd(C[j]), Tl[j], acc0);
    return _mm512_sub_pd(_mm512_fmadd_pd(_mm512_set1_pd(xh), b1, acc0), b2);
}
// bivariate clenshaw, per-lane xh AND xl (mixed-h grid).  Identical fma order.
inline __m512d clenshaw2_avx512_vxh(int dp, int dl, const double* C, __m512d xh, __m512d xl) {
    const int stride = dl + 1;
    __m512d Tl[17];
    Tl[0] = _mm512_set1_pd(1.0);
    Tl[1] = xl;
    __m512d xl2 = _mm512_mul_pd(_mm512_set1_pd(2.0), xl);
    for (int j = 2; j <= dl; ++j) Tl[j] = _mm512_fmsub_pd(xl2, Tl[j - 1], Tl[j - 2]);
    const __m512d vxh2 = _mm512_mul_pd(_mm512_set1_pd(2.0), xh);
    __m512d b1 = _mm512_setzero_pd(), b2 = _mm512_setzero_pd();
    for (int i = dp; i >= 1; --i) {
        const double* Ci = C + i * stride;
        __m512d acc = _mm512_set1_pd(Ci[0]);
        for (int j = 1; j <= dl; ++j) acc = _mm512_fmadd_pd(_mm512_set1_pd(Ci[j]), Tl[j], acc);
        __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(vxh2, b1, acc), b2);
        b2 = b1; b1 = b0;
    }
    __m512d acc0 = _mm512_set1_pd(C[0]);
    for (int j = 1; j <= dl; ++j) acc0 = _mm512_fmadd_pd(_mm512_set1_pd(C[j]), Tl[j], acc0);
    return _mm512_sub_pd(_mm512_fmadd_pd(xh, b1, acc0), b2);
}

// ==== NEW: AVX-512 twins of the LEFT/RIGHT endpoint kernels (namespace-K alias) ====
namespace Kv = volfi_annulus_broadrange;

// full natural log by reuse: ln2*(exponent + log2approx(mantissa)).  Mirrors scalar full_log.
inline __m512d full_log_avx512(__m512d x) {
    __m512i bc = _mm512_castpd_si512(x);
    __m512i j  = _mm512_and_si512(_mm512_srli_epi64(bc, 52), _mm512_set1_epi64(0x7FFLL));
    __m512d dj = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(j, _mm512_set1_epi64(0x4330000000000000LL))),
                               _mm512_set1_pd(0x1.0p52));                 // double(j)
    __m512d e  = _mm512_sub_pd(dj, _mm512_set1_pd((double)C_EXP_BIAS));   // j - 1023
    __m512d l2 = log2approx_avx512(mant12_avx512(x));
    return _mm512_fmadd_pd(_mm512_set1_pd(VA_LN2), _mm512_add_pd(e, l2), _mm512_setzero_pd());   // frozen
}

// 1-D Chebyshev (Clenshaw) with affine [a,b]->[-1,1], per-lane x.  Mirrors br::clenshaw1.
inline __m512d clenshaw1_avx512(const double* C, int n, double a, double b, __m512d x) {
    __m512d t  = _mm512_div_pd(_mm512_fmadd_pd(_mm512_set1_pd(2.0), x, _mm512_set1_pd(-(a + b))),
                               _mm512_set1_pd(b - a));
    __m512d t2 = _mm512_add_pd(t, t);
    __m512d d0 = _mm512_setzero_pd(), d1 = _mm512_setzero_pd();
    for (int jx = n - 1; jx >= 1; --jx) {
        __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(t2, d0, _mm512_set1_pd(C[jx])), d1);
        d1 = d0; d0 = b0;
    }
    return _mm512_sub_pd(_mm512_fmadd_pd(t, d0, _mm512_set1_pd(C[0])), d1);
}

// A = Binv(rho): 3 branchless Chebyshev regimes with masked blend.  Mirrors br::binv.
inline __m512d binv_avx512(__m512d rho) {
    const __m512d zero = _mm512_setzero_pd();
    __m512d u  = _mm512_div_pd(_mm512_set1_pd(Kv::BR_K), _mm512_add_pd(rho, _mm512_set1_pd(0.5)));
    __m512d xa = _mm512_mul_pd(u, u);
    __m512d Aa = _mm512_mul_pd(u, clenshaw1_avx512(Kv::BINV_CA, 15, Kv::BINV_U2_A, Kv::BINV_U2_B, xa));
    __m512d L  = _mm512_sub_pd(zero, full_log_avx512(rho));
    __m512d A2_1 = clenshaw1_avx512(Kv::BINV_CB1, 17, Kv::BINV_L_B1A, Kv::BINV_L_B1B, L);
    __m512d A2_2 = clenshaw1_avx512(Kv::BINV_CB2, 25, Kv::BINV_L_B2A, Kv::BINV_L_B2B, L);
    __mmask8 mB  = _mm512_cmp_pd_mask(rho, _mm512_set1_pd(Kv::BINV_RHO_B), _CMP_GT_OQ);
    __m512d A2   = _mm512_mask_blend_pd(mB, A2_2, A2_1);     // rho>RHO_B -> CB1 branch
    __m512d AL   = _mm512_sqrt_pd(A2);
    __mmask8 mA  = _mm512_cmp_pd_mask(rho, _mm512_set1_pd(Kv::BINV_RHO_A), _CMP_GT_OQ);
    return _mm512_mask_blend_pd(mA, AL, Aa);                 // rho>RHO_A -> regime a
}

// erfcx(z) via the two-piece Chebyshev (C2).  Per-lane piece selection: the coefficient blend
// feeds the Clenshaw fma chain but is itself off the serial chain (independent of d0/d1), so
// the recurrence latency is that of 22 iterations, not 36.  Affine bounds are blended from the
// same compile-time double sums the scalar path computes -> per-lane bit-identity to scalar.
inline __m512d erfcx_poly_avx512(__m512d z) {
    const __m512d zero = _mm512_setzero_pd();
    __mmask8 hi = _mm512_cmp_pd_mask(z, _mm512_set1_pd(Kv::RIGHT_ERFCX2_SPLIT), _CMP_GT_OQ);
    __mmask8 h2 = _mm512_cmp_pd_mask(z, _mm512_set1_pd(Kv::RIGHT_ERFCX_B), _CMP_GT_OQ);
    __m512d napb = _mm512_mask_blend_pd(h2, _mm512_mask_blend_pd(hi,
        _mm512_set1_pd(-(Kv::RIGHT_ERFCX_A + Kv::RIGHT_ERFCX2_SPLIT)),
        _mm512_set1_pd(-(Kv::RIGHT_ERFCX2_SPLIT + Kv::RIGHT_ERFCX_B))),
        _mm512_set1_pd(-(Kv::RIGHT_ERFCX_B + Kv::RIGHT_ERFCX2_B2)));
    __m512d bma  = _mm512_mask_blend_pd(h2, _mm512_mask_blend_pd(hi,
        _mm512_set1_pd(Kv::RIGHT_ERFCX2_SPLIT - Kv::RIGHT_ERFCX_A),
        _mm512_set1_pd(Kv::RIGHT_ERFCX_B - Kv::RIGHT_ERFCX2_SPLIT)),
        _mm512_set1_pd(Kv::RIGHT_ERFCX2_B2 - Kv::RIGHT_ERFCX_B));
    __m512d t  = _mm512_div_pd(_mm512_fmadd_pd(_mm512_set1_pd(2.0), z, napb), bma);
    __m512d t2 = _mm512_add_pd(t, t);
    __m512d d0 = zero, d1 = zero;
    for (int j = Kv::RIGHT_ERFCX2_N - 1; j >= 1; --j) {
        __m512d cj = _mm512_mask_blend_pd(h2, _mm512_mask_blend_pd(hi,
                         _mm512_set1_pd(Kv::RIGHT_ERFCX2_C0[j]),
                         _mm512_set1_pd(Kv::RIGHT_ERFCX2_C1[j])),
                         _mm512_set1_pd(Kv::RIGHT_ERFCX2_C2[j]));
        __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(t2, d0, cj), d1);
        d1 = d0; d0 = b0;
    }
    __m512d c0 = _mm512_mask_blend_pd(h2, _mm512_mask_blend_pd(hi,
                     _mm512_set1_pd(Kv::RIGHT_ERFCX2_C0[0]),
                     _mm512_set1_pd(Kv::RIGHT_ERFCX2_C1[0])),
                     _mm512_set1_pd(Kv::RIGHT_ERFCX2_C2[0]));
    return _mm512_sub_pd(_mm512_fmadd_pd(t, d0, c0), d1);
}
// exp(a): Cody-Waite ln2 reduction + Chebyshev poly + scalef (== ldexp).  Mirrors scalar exp_neg.
inline __m512d exp_neg_avx512(__m512d a) {
    __m512d nf = _mm512_floor_pd(_mm512_fmadd_pd(a, _mm512_set1_pd(Kv::INV_LN2), _mm512_set1_pd(0.5)));
    __m512d nneg = _mm512_sub_pd(_mm512_setzero_pd(), nf);
    __m512d r  = _mm512_fmadd_pd(nneg, _mm512_set1_pd(Kv::LN2_HI), a);
    r          = _mm512_fmadd_pd(nneg, _mm512_set1_pd(Kv::LN2_LO), r);
    __m512d p  = clenshaw1_avx512(Kv::EXP_C, Kv::EXP_C_N, -0.5 * VA_LN2, 0.5 * VA_LN2, r);
    return _mm512_scalef_pd(p, nf);
}
// erfc(t) via shared erfcx+exp_neg (branchless on sign of t).  Mirrors scalar erfc_shared.
inline __m512d erfc_shared_avx512(__m512d t) {
    const __m512d zero = _mm512_setzero_pd();
    __m512d a = _mm512_abs_pd(t);
    __m512d e = _mm512_mul_pd(erfcx_poly_avx512(a),
                              exp_neg_avx512(_mm512_sub_pd(zero, _mm512_mul_pd(a, a))));
    __m512d two_minus = _mm512_sub_pd(_mm512_set1_pd(2.0), e);
    __mmask8 pos = _mm512_cmp_pd_mask(t, zero, _CMP_GE_OQ);
    return _mm512_mask_blend_pd(pos, two_minus, e);
}
inline __m512d phicdf_avx512(__m512d x) {
    __m512d arg = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(x, _mm512_set1_pd(br::BR_IS2)));
    return _mm512_mul_pd(_mm512_set1_pd(0.5), erfc_shared_avx512(arg));
}
inline __m512d black_otm_avx512(__m512d h, __m512d v, __m512d ehp) {
    const __m512d zero = _mm512_setzero_pd(), half = _mm512_set1_pd(0.5);
    __m512d hv = _mm512_div_pd(h, v);
    __m512d y  = _mm512_fmadd_pd(half, v, _mm512_sub_pd(zero, hv));
    __m512d r  = _mm512_fmadd_pd(half, v, hv);
    return _mm512_fmadd_pd(_mm512_sub_pd(zero, ehp), phicdf_avx512(_mm512_sub_pd(zero, r)), phicdf_avx512(y));
}
inline __m512d onem_otm_avx512(__m512d h, __m512d v, __m512d ehp) {
    const __m512d zero = _mm512_setzero_pd(), half = _mm512_set1_pd(0.5);
    __m512d hv = _mm512_div_pd(h, v);
    __m512d y  = _mm512_fmadd_pd(half, v, _mm512_sub_pd(zero, hv));
    __m512d r  = _mm512_fmadd_pd(half, v, hv);
    return _mm512_fmadd_pd(ehp, phicdf_avx512(_mm512_sub_pd(zero, r)), phicdf_avx512(_mm512_sub_pd(zero, y)));
}
// Deduplicated residual c - C(h,v) (mirror of br::price_residual): shares Phi(-r) and the |y|
// erfc core between the black/onem forms, 2 erfcx + 2 exp per call instead of 4 + 4.  Every
// subexpression reproduces the black_otm_avx512/onem_otm_avx512/phicdf/erfc_shared chain
// bit-for-bit (same sign-flip forms, same blend predicates).
inline __m512d price_residual_avx512(__m512d vh, __m512d vv, __m512d vehp,
                                     __m512d vc, __m512d onec, __mmask8 lo) {
    const __m512d zero = _mm512_setzero_pd(), half = _mm512_set1_pd(0.5);
    const __m512d is2  = _mm512_set1_pd(br::BR_IS2);
    __m512d hv = _mm512_div_pd(vh, vv);
    __m512d y  = _mm512_fmadd_pd(half, vv, _mm512_sub_pd(zero, hv));
    __m512d r  = _mm512_fmadd_pd(half, vv, hv);
    __m512d t1 = _mm512_sub_pd(zero, _mm512_mul_pd(y, is2));                          // phicdf(y)
    __m512d t2 = _mm512_sub_pd(zero, _mm512_mul_pd(_mm512_sub_pd(zero, y), is2));     // phicdf(-y)
    const __m512d one512 = _mm512_set1_pd(1.0);
    __m512d ay  = _mm512_abs_pd(t1);
    __m512d sy  = _mm512_mul_pd(ay, ay);
    __m512d scy = _mm512_fmsub_pd(ay, ay, sy);                // exact low part of ay^2
    __m512d ey  = _mm512_mul_pd(erfcx_poly_avx512(ay),
                    _mm512_mul_pd(exp_neg_avx512(_mm512_sub_pd(zero, sy)),
                                  _mm512_sub_pd(one512, scy)));
    __m512d tm = _mm512_sub_pd(_mm512_set1_pd(2.0), ey);
    __m512d phi_y  = _mm512_mul_pd(half, _mm512_mask_blend_pd(
                         _mm512_cmp_pd_mask(t1, zero, _CMP_GE_OQ), tm, ey));
    __m512d phi_my = _mm512_mul_pd(half, _mm512_mask_blend_pd(
                         _mm512_cmp_pd_mask(t2, zero, _CMP_GE_OQ), tm, ey));
    __m512d tr  = _mm512_sub_pd(zero, _mm512_mul_pd(_mm512_sub_pd(zero, r), is2));
    __m512d ar  = _mm512_abs_pd(tr);
    __m512d sr  = _mm512_mul_pd(ar, ar);
    __m512d scr = _mm512_fmsub_pd(ar, ar, sr);
    __m512d er  = _mm512_mul_pd(erfcx_poly_avx512(ar),
                    _mm512_mul_pd(exp_neg_avx512(_mm512_sub_pd(zero, sr)),
                                  _mm512_sub_pd(one512, scr)));
    __m512d tmr = _mm512_sub_pd(_mm512_set1_pd(2.0), er);
    __m512d phi_r = _mm512_mul_pd(half, _mm512_mask_blend_pd(
                        _mm512_cmp_pd_mask(tr, zero, _CMP_GE_OQ), tmr, er));
    __m512d bl = _mm512_fmadd_pd(_mm512_sub_pd(zero, vehp), phi_r, phi_y);
    __m512d om = _mm512_fmadd_pd(vehp, phi_r, phi_my);
    return _mm512_mask_blend_pd(lo, _mm512_sub_pd(om, onec), _mm512_sub_pd(vc, bl));
}
// Acklam rational quantile seed; tail branch (p<QN_P_LOW) via full_log.  Mirrors br::qnorm0_seed.
inline __m512d qnorm0_seed_avx512(__m512d p) {
    __m512d q = _mm512_sub_pd(p, _mm512_set1_pd(0.5));
    __m512d r = _mm512_mul_pd(q, q);
    __m512d num = _mm512_set1_pd(Kv::QN_A[0]);
    for (int i = 1; i < 6; ++i) num = _mm512_fmadd_pd(num, r, _mm512_set1_pd(Kv::QN_A[i]));
    num = _mm512_mul_pd(num, q);
    __m512d den = _mm512_set1_pd(Kv::QN_B[0]);
    for (int i = 1; i < 5; ++i) den = _mm512_fmadd_pd(den, r, _mm512_set1_pd(Kv::QN_B[i]));
    den = _mm512_fmadd_pd(den, r, _mm512_set1_pd(1.0));
    __m512d mid = _mm512_div_pd(num, den);
    __m512d qt = _mm512_sqrt_pd(_mm512_mul_pd(_mm512_set1_pd(-2.0), full_log_avx512(p)));
    __m512d numt = _mm512_set1_pd(Kv::QN_C[0]);
    for (int i = 1; i < 6; ++i) numt = _mm512_fmadd_pd(numt, qt, _mm512_set1_pd(Kv::QN_C[i]));
    __m512d dent = _mm512_set1_pd(Kv::QN_D[0]);
    for (int i = 1; i < 4; ++i) dent = _mm512_fmadd_pd(dent, qt, _mm512_set1_pd(Kv::QN_D[i]));
    dent = _mm512_fmadd_pd(dent, qt, _mm512_set1_pd(1.0));
    __m512d tail = _mm512_div_pd(numt, dent);
    __mmask8 tl = _mm512_cmp_pd_mask(p, _mm512_set1_pd(Kv::QN_P_LOW), _CMP_LT_OQ);
    return _mm512_mask_blend_pd(tl, mid, tail);
}

// LEFT chart, 8 lanes -> w.  Per-lane vectors for c and all h-derived quantities.
inline __m512d left_variance_avx512(__m512d vc, __m512d v_expm1h, __m512d vh,
                                    __m512d vh2, __m512d vh4, __m512d vxt) {
    __m512d rho = _mm512_div_pd(vc, v_expm1h);
    __m512d A   = binv_avx512(rho);
    __m512d s   = _mm512_div_pd(vh, A);
    __m512d x   = sigma0_poly_avx512(_mm512_mul_pd(_mm512_set1_pd(Kv::BR_K), s));
    __m512d xs  = _mm512_fmadd_pd(_mm512_add_pd(s, s), _mm512_set1_pd(1.0 / Kv::LEFT_S_MAX), _mm512_set1_pd(-1.0));
    __m512d V2  = clenshaw1_avx512(Kv::LEFT_V2_CHEB, Kv::LEFT_V2_CHEB_N, 0.0, Kv::LEFT_S_CHEB_MAX, s);
    __m512d V4  = clenshaw1_avx512(Kv::LEFT_V4_CHEB, Kv::LEFT_V4_CHEB_N, 0.0, Kv::LEFT_S_CHEB_MAX, s);
    __m512d fin = clenshaw2_avx512_vxh(Kv::LEFT_FIN_DP, Kv::LEFT_FIN_DL, Kv::LEFT_FIN_COEFFS, vxt, xs);
    __m512d v   = _mm512_fmadd_pd(vh4, V4, _mm512_fmadd_pd(vh2, V2, x));
    v           = _mm512_fmadd_pd(_mm512_mul_pd(vh4, vh2), fin, v);
    return _mm512_mul_pd(v, v);
}

// RIGHT chart, 8 lanes -> w.  Per-lane c and per-lane per-h scalars (eh,ehp,h,h2).
inline __m512d right_variance_avx512(__m512d vc, __m512d veh, __m512d vehp, __m512d vh, __m512d vh2) {
    const __m512d zero = _mm512_setzero_pd(), one = _mm512_set1_pd(1.0);
    __m512d gbar = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), _mm512_sub_pd(one, vc)), veh);
    __m512d x0   = _mm512_sub_pd(zero, qnorm0_seed_avx512(gbar));
    __m512d m    = _mm512_mul_pd(_mm512_set1_pd(Kv::SQRT_PI2),
                                 erfcx_poly_avx512(_mm512_mul_pd(x0, _mm512_set1_pd(br::BR_IS2))));
    __m512d inv  = _mm512_div_pd(one, x0);
    __m512d x2   = _mm512_mul_pd(_mm512_set1_pd(-0.125), _mm512_sub_pd(inv, m));
    __m512d B2p  = _mm512_mul_pd(_mm512_add_pd(_mm512_fmadd_pd(_mm512_mul_pd(inv, inv), inv,
                                                              _mm512_sub_pd(zero, inv)), m),
                                 _mm512_set1_pd(1.0 / 3.0));
    __m512d ta   = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), x0), x2), x2);
    __m512d tb   = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.125), x2), inv), inv);
    __m512d x4   = _mm512_fmadd_pd(B2p, _mm512_set1_pd(1.0 / 128.0), _mm512_add_pd(ta, tb));
    __m512d x    = _mm512_fmadd_pd(_mm512_mul_pd(vh2, vh2), x4, _mm512_fmadd_pd(vh2, x2, x0));
    __m512d onec = _mm512_sub_pd(one, vc);
    __mmask8 lo  = _mm512_cmp_pd_mask(vc, _mm512_set1_pd(0.5), _CMP_LT_OQ);
    for (int it = 0; it < Kv::RIGHT_HH3_STEPS; ++it) {          // HH3 (order 4); mirrors scalar
        __m512d vv  = _mm512_mul_pd(_mm512_set1_pd(2.0), x);
        __m512d cmC = price_residual_avx512(vh, vv, vehp, vc, onec, lo);
        __m512d f   = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), cmC), veh);
        __m512d eA  = exp_neg_avx512(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(-0.5), x), x));
        __m512d eB  = exp_neg_avx512(_mm512_div_pd(_mm512_sub_pd(zero, vh2),
                                                   _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(8.0), x), x)));
        __m512d gp  = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(-Kv::BR_K), eA), eB);
        __m512d u   = _mm512_div_pd(f, gp);
        __m512d ix  = _mm512_div_pd(one, x);
        __m512d ix2 = _mm512_mul_pd(ix, ix);
        __m512d a   = _mm512_fmadd_pd(_mm512_mul_pd(_mm512_set1_pd(0.25), vh2),
                                      _mm512_mul_pd(ix2, ix), _mm512_sub_pd(zero, x));   // L
        __m512d Lp  = _mm512_fmadd_pd(_mm512_mul_pd(_mm512_set1_pd(-0.75), vh2),
                                      _mm512_mul_pd(ix2, ix2), _mm512_set1_pd(-1.0));    // L'
        __m512d b   = _mm512_fmadd_pd(a, a, Lp);                                        // L^2 + L'
        __m512d au  = _mm512_mul_pd(a, u);
        __m512d num = _mm512_fmadd_pd(_mm512_set1_pd(-0.5), au, one);                   // 1 - 0.5 a u
        __m512d den = _mm512_fmadd_pd(_mm512_mul_pd(b, u),
                                      _mm512_mul_pd(u, _mm512_set1_pd(1.0 / 6.0)),
                                      _mm512_sub_pd(one, au));                          // 1 - a u + b u^2/6
        x = _mm512_sub_pd(x, _mm512_div_pd(_mm512_mul_pd(u, num), den));
    }
    __m512d v = _mm512_mul_pd(_mm512_set1_pd(2.0), x);
    return _mm512_mul_pd(v, v);
}

// Warm-start refinement, 8 lanes (mirror of br::warm_variance; see basin note there).
inline __m512d warm_variance_avx512(__m512d vh, __m512d vc, __m512d vw,
                                    int steps = Kv::WARM_HH3_STEPS) {
    const __m512d zero = _mm512_setzero_pd(), one = _mm512_set1_pd(1.0), half = _mm512_set1_pd(0.5);
    __m512d x    = _mm512_mul_pd(half, _mm512_sqrt_pd(vw));
    __m512d vh2  = _mm512_mul_pd(vh, vh);
    __m512d veh  = exp_neg_avx512(_mm512_mul_pd(_mm512_set1_pd(-0.5), vh));
    __m512d vehp = exp_neg_avx512(vh);
    __m512d onec = _mm512_sub_pd(one, vc);
    __mmask8 lo  = _mm512_cmp_pd_mask(vc, half, _CMP_LT_OQ);
    for (int it = 0; it < steps; ++it) {
        __m512d vv  = _mm512_mul_pd(_mm512_set1_pd(2.0), x);
        __m512d cmC = price_residual_avx512(vh, vv, vehp, vc, onec, lo);
        __m512d f   = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), cmC), veh);
        __m512d eA  = exp_neg_avx512(_mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(-0.5), x), x));
        __m512d eB  = exp_neg_avx512(_mm512_div_pd(_mm512_sub_pd(zero, vh2),
                                                   _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(8.0), x), x)));
        __m512d gp  = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(-Kv::BR_K), eA), eB);
        __m512d u   = _mm512_div_pd(f, gp);
        __m512d ix  = _mm512_div_pd(one, x);
        __m512d ix2 = _mm512_mul_pd(ix, ix);
        __m512d a   = _mm512_fmadd_pd(_mm512_mul_pd(_mm512_set1_pd(0.25), vh2),
                                      _mm512_mul_pd(ix2, ix), _mm512_sub_pd(zero, x));
        __m512d Lp  = _mm512_fmadd_pd(_mm512_mul_pd(_mm512_set1_pd(-0.75), vh2),
                                      _mm512_mul_pd(ix2, ix2), _mm512_set1_pd(-1.0));
        __m512d b   = _mm512_fmadd_pd(a, a, Lp);
        __m512d au  = _mm512_mul_pd(a, u);
        __m512d num = _mm512_fmadd_pd(_mm512_set1_pd(-0.5), au, one);
        __m512d den = _mm512_fmadd_pd(_mm512_mul_pd(b, u),
                                      _mm512_mul_pd(u, _mm512_set1_pd(1.0 / 6.0)),
                                      _mm512_sub_pd(one, au));
        x = _mm512_sub_pd(x, _mm512_div_pd(_mm512_mul_pd(u, num), den));
    }
    __m512d v = _mm512_mul_pd(_mm512_set1_pd(2.0), x);
    return _mm512_mul_pd(v, v);
}

// ---- WING SIMD twins (A): subnormal-safe log, series, fitted logH, Newton ----
inline __m512d full_log_sub_avx512(__m512d x) {
    __mmask8 sb = _mm512_cmp_pd_mask(x, _mm512_set1_pd(VA_DBL_MIN), _CMP_LT_OQ);
    __m512d xs = _mm512_mask_mul_pd(x, sb, x, _mm512_set1_pd(VA_TWO54));
    __m512i bc = _mm512_castpd_si512(xs);
    __m512i j  = _mm512_and_si512(_mm512_srli_epi64(bc, 52), _mm512_set1_epi64(0x7FFLL));
    __m512d dj = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(j, _mm512_set1_epi64(0x4330000000000000LL))),
                               _mm512_set1_pd(0x1.0p52));
    __m512d e  = _mm512_sub_pd(dj, _mm512_set1_pd((double)C_EXP_BIAS));
    e = _mm512_mask_sub_pd(e, sb, e, _mm512_set1_pd(54.0));
    __m512d l2 = log2approx_avx512(mant12_avx512(xs));
    return _mm512_fmadd_pd(_mm512_set1_pd(VA_LN2), _mm512_add_pd(e, l2), _mm512_setzero_pd());   // frozen
}
// asymptotic series logH + (logH)'(W) (mirror of wing_detail::logH_series_d).
inline __m512d wing_series_d_avx512(__m512d W, __m512d b, __m512d* dlogH) {
    const __m512d zero = _mm512_setzero_pd(), one = _mm512_set1_pd(1.0);
    __m512d b2 = _mm512_mul_pd(b, b), b3 = _mm512_mul_pd(b2, b), b4 = _mm512_mul_pd(b2, b2);
    __m512d g1 = _mm512_sub_pd(zero, _mm512_add_pd(b, _mm512_set1_pd(1.5)));
    __m512d g2 = _mm512_add_pd(b, _mm512_set1_pd(21.0/8.0));
    __m512d g3 = _mm512_sub_pd(zero, _mm512_fmadd_pd(_mm512_set1_pd(3.5), b, _mm512_set1_pd(69.0/8.0)));
    __m512d g4 = _mm512_fmadd_pd(_mm512_set1_pd(0.5), b2,
                 _mm512_fmadd_pd(_mm512_set1_pd(17.25), b, _mm512_set1_pd(2529.0/64.0)));
    __m512d g5 = _mm512_sub_pd(zero, _mm512_fmadd_pd(_mm512_set1_pd(5.5), b2,
                 _mm512_fmadd_pd(_mm512_set1_pd(105.375), b, _mm512_set1_pd(36243.0/160.0))));
    __m512d g6 = _mm512_fmadd_pd(_mm512_set1_pd(1.0/3.0), b3,
                 _mm512_fmadd_pd(_mm512_set1_pd(53.875), b2,
                 _mm512_fmadd_pd(_mm512_set1_pd(755.0625), b, _mm512_set1_pd(197127.0/128.0))));
    __m512d g7 = _mm512_sub_pd(zero, _mm512_fmadd_pd(_mm512_set1_pd(7.5), b3,
                 _mm512_fmadd_pd(_mm512_set1_pd(537.75), b2,
                 _mm512_fmadd_pd(_mm512_set1_pd(6160.21875), b, _mm512_set1_pd(10786527.0/896.0)))));
    __m512d g8 = _mm512_fmadd_pd(_mm512_set1_pd(0.25), b4,
                 _mm512_fmadd_pd(_mm512_set1_pd(122.5), b3,
                 _mm512_fmadd_pd(_mm512_set1_pd(5651.15625), b2,
                 _mm512_fmadd_pd(_mm512_set1_pd(56179.828125), b, _mm512_set1_pd(217179009.0/2048.0)))));
    __m512d x = _mm512_div_pd(one, W);
    __m512d s = g8;
    s = _mm512_fmadd_pd(s, x, g7); s = _mm512_fmadd_pd(s, x, g6); s = _mm512_fmadd_pd(s, x, g5);
    s = _mm512_fmadd_pd(s, x, g4); s = _mm512_fmadd_pd(s, x, g3); s = _mm512_fmadd_pd(s, x, g2);
    s = _mm512_fmadd_pd(s, x, g1);
    __m512d lH = _mm512_mul_pd(s, x);
    __m512d d = _mm512_mul_pd(_mm512_set1_pd(8.0), g8);
    d = _mm512_fmadd_pd(d, x, _mm512_mul_pd(_mm512_set1_pd(7.0), g7));
    d = _mm512_fmadd_pd(d, x, _mm512_mul_pd(_mm512_set1_pd(6.0), g6));
    d = _mm512_fmadd_pd(d, x, _mm512_mul_pd(_mm512_set1_pd(5.0), g5));
    d = _mm512_fmadd_pd(d, x, _mm512_mul_pd(_mm512_set1_pd(4.0), g4));
    d = _mm512_fmadd_pd(d, x, _mm512_mul_pd(_mm512_set1_pd(3.0), g3));
    d = _mm512_fmadd_pd(d, x, _mm512_mul_pd(_mm512_set1_pd(2.0), g2));
    d = _mm512_fmadd_pd(d, x, g1);
    *dlogH = _mm512_fmadd_pd(_mm512_mul_pd(_mm512_sub_pd(zero, x), x), d, zero);   // frozen
    return lH;
}
// fitted logH + (logH)'(W) on one piece (bucket-uniform; mirror of wing_rt::wing_logH_fit_d).
inline __m512d wing_fit_d_avx512(__m512d W, __m512d beta, const double* C, int NU, int NB,
                                 double ulo, double uhi, __m512d* dlogH) {
    namespace wd = wing_detail;
    const __m512d zero = _mm512_setzero_pd(), one = _mm512_set1_pd(1.0);
    __m512d u = _mm512_div_pd(one, W);
    u = _mm512_min_pd(_mm512_max_pd(u, _mm512_set1_pd(ulo)), _mm512_set1_pd(uhi));
    __m512d xu = _mm512_div_pd(_mm512_fmadd_pd(_mm512_set1_pd(2.0), u, _mm512_set1_pd(-(ulo + uhi))),
                               _mm512_set1_pd(uhi - ulo));
    __m512d xb = _mm512_div_pd(_mm512_fmadd_pd(_mm512_set1_pd(2.0), beta,
                               _mm512_set1_pd(-(wd::WING_B_LO + wd::WING_B_HI))),
                               _mm512_set1_pd(wd::WING_B_HI - wd::WING_B_LO));
    __m512d rows[27];
    __m512d tb2 = _mm512_add_pd(xb, xb);
    for (int i = 0; i < NU; ++i) {
        const double* Ci = C + i * NB;
        __m512d d0 = zero, d1 = zero;
        for (int j = NB - 1; j >= 1; --j) {
            __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(tb2, d0, _mm512_set1_pd(Ci[j])), d1);
            d1 = d0; d0 = b0;
        }
        rows[i] = _mm512_sub_pd(_mm512_fmadd_pd(xb, d0, _mm512_set1_pd(Ci[0])), d1);
    }
    __m512d tu2 = _mm512_add_pd(xu, xu);
    __m512d d0 = zero, d1 = zero;
    for (int i = NU - 1; i >= 1; --i) {
        __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(tu2, d0, rows[i]), d1);
        d1 = d0; d0 = b0;
    }
    __m512d S = _mm512_sub_pd(_mm512_fmadd_pd(xu, d0, rows[0]), d1);
    __m512d dp[27];
    dp[NU - 1] = zero;
    dp[NU - 2] = _mm512_mul_pd(_mm512_set1_pd(2.0 * (NU - 1)), rows[NU - 1]);
    for (int k = NU - 3; k >= 0; --k)
        dp[k] = _mm512_fmadd_pd(_mm512_set1_pd(2.0 * (k + 1)), rows[k + 1], dp[k + 2]);
    dp[0] = _mm512_mul_pd(dp[0], _mm512_set1_pd(0.5));
    d0 = zero; d1 = zero;
    for (int i = NU - 1; i >= 1; --i) {
        __m512d b0 = _mm512_sub_pd(_mm512_fmadd_pd(tu2, d0, dp[i]), d1);
        d1 = d0; d0 = b0;
    }
    __m512d dS_dxu = _mm512_sub_pd(_mm512_fmadd_pd(xu, d0, dp[0]), d1);
    __m512d Su = _mm512_mul_pd(dS_dxu, _mm512_set1_pd(2.0 / (uhi - ulo)));
    __m512d bp = _mm512_add_pd(beta, _mm512_set1_pd(1.5));
    *dlogH = _mm512_fmadd_pd(_mm512_sub_pd(zero, _mm512_mul_pd(u, u)),
                             _mm512_sub_pd(_mm512_fmadd_pd(u, Su, S), bp), zero);   // frozen
    return _mm512_mul_pd(u, _mm512_sub_pd(S, bp));
}
// wing bucket kernel, 8 lanes (regime uniform across the bucket: 0=A, 1=B, 2=series).
inline __m512d wing_variance_avx512(__m512d vh, __m512d vc, int regime) {
    namespace wd = wing_detail;
    const __m512d zero = _mm512_setzero_pd(), two = _mm512_set1_pd(2.0);
    __m512d beta = _mm512_min_pd(_mm512_mul_pd(_mm512_mul_pd(vh, vh), _mm512_set1_pd(1.0 / 16.0)),
                                 _mm512_set1_pd(wd::WING_B_HI));
    __m512d lc = _mm512_sub_pd(full_log_avx512(vh), _mm512_set1_pd(wd::WING_LOG4));
    lc = _mm512_sub_pd(_mm512_fmadd_pd(_mm512_set1_pd(0.5), vh, lc), _mm512_set1_pd(wd::WING_HLOGPI));
    __m512d Lt = _mm512_add_pd(_mm512_sub_pd(zero, full_log_sub_avx512(vc)), lc);
    __m512d lt = _mm512_max_pd(Lt, two);
    __m512d W = _mm512_max_pd(_mm512_fmadd_pd(_mm512_set1_pd(-1.5), full_log_avx512(lt), lt),
                              _mm512_set1_pd(2.6));
    for (int it = 0; it < wd::WING_NEWTON_STEPS; ++it) {
        __m512d dlH, lH;
        if (regime == 2)      lH = wing_series_d_avx512(W, beta, &dlH);
        else if (regime == 0) lH = wing_fit_d_avx512(W, beta, wd::WING_SA, wd::WING_SA_NU, wd::WING_SA_NB,
                                                     wd::WING_UA_LO, wd::WING_UA_HI, &dlH);
        else                  lH = wing_fit_d_avx512(W, beta, wd::WING_SB, wd::WING_SB_NU, wd::WING_SB_NB,
                                                     wd::WING_UB_LO, wd::WING_UB_HI, &dlH);
        __m512d F  = _mm512_add_pd(_mm512_sub_pd(W, Lt),
                                   _mm512_fmadd_pd(_mm512_set1_pd(1.5), full_log_avx512(W),
                                                   _mm512_sub_pd(zero, lH)));
        __m512d Fp = _mm512_sub_pd(_mm512_add_pd(_mm512_set1_pd(1.0),
                                                 _mm512_div_pd(_mm512_set1_pd(1.5), W)), dlH);
        W = _mm512_max_pd(_mm512_sub_pd(W, _mm512_div_pd(F, Fp)), _mm512_set1_pd(2.6));
    }
    return _mm512_div_pd(_mm512_mul_pd(vh, vh), _mm512_mul_pd(two, W));
}
} // namespace detail
#endif // VA_SIMD512

#if defined(VA_SIMD256)
namespace detail {
inline __m256d mant12_avx2(__m256d c) {
    __m256i cb = _mm256_castpd_si256(c);
    __m256i mb = _mm256_or_si256(_mm256_and_si256(cb, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL)),
                                 _mm256_set1_epi64x((long long)((uint64_t)C_EXP_BIAS << 52)));
    return _mm256_castsi256_pd(mb);
}
inline __m256d log2approx_avx2(__m256d m) {
    const __m256d one = _mm256_set1_pd(1.0);
    __m256d t  = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
    __m256d u  = _mm256_mul_pd(t, t);
    __m256d s  = _mm256_fmadd_pd(_mm256_set1_pd(LOG2_S_A), u, _mm256_set1_pd(LOG2_S_B));
    __m256d s2 = _mm256_add_pd(s, s);
    __m256d b1 = _mm256_setzero_pd(), b2 = _mm256_setzero_pd();
    for (int k = 10; k >= 1; --k) {
        __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(s2, b1, _mm256_set1_pd(LOG2_GC[k])), b2);
        b2 = b1; b1 = b0;
    }
    __m256d g  = _mm256_sub_pd(_mm256_fmadd_pd(s, b1, _mm256_set1_pd(LOG2_GC[0])), b2);
    return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(LOG2_SCALE), t), g);
}
// sigma0_poly on 4 prices.  (mirror of sigma0_poly_avx512 / scalar sigma0_poly.)
inline __m256d sigma0_poly_avx2(__m256d c) {
    __m256d t  = _mm256_mul_pd(c, c);
    __m256d x  = _mm256_fmadd_pd(_mm256_set1_pd(ERFINV_TSC), t, _mm256_set1_pd(ERFINV_TBIAS));
    __m256d x2 = _mm256_add_pd(x, x);
    __m256d b1 = _mm256_setzero_pd(), b2 = _mm256_setzero_pd();
    for (int k = 17; k >= 1; --k) {
        __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(x2, b1, _mm256_set1_pd(ERFINV_GC[k])), b2);
        b2 = b1; b1 = b0;
    }
    __m256d g  = _mm256_sub_pd(_mm256_fmadd_pd(x, b1, _mm256_set1_pd(ERFINV_GC[0])), b2);
    return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(SIGMA0_SCALE), c), g);
}
// bivariate clenshaw, scalar xh, 4-lane xl.  (mirror of clenshaw2_avx512.)
inline __m256d clenshaw2_avx2(int dp, int dl, const double* C, double xh, __m256d xl) {
    const int stride = dl + 1;
    __m256d Tl[17];
    Tl[0] = _mm256_set1_pd(1.0);
    Tl[1] = xl;
    __m256d xl2 = _mm256_mul_pd(_mm256_set1_pd(2.0), xl);
    for (int j = 2; j <= dl; ++j) Tl[j] = _mm256_fmsub_pd(xl2, Tl[j - 1], Tl[j - 2]);
    const __m256d vxh2 = _mm256_set1_pd(2.0 * xh);
    __m256d b1 = _mm256_setzero_pd(), b2 = _mm256_setzero_pd();
    for (int i = dp; i >= 1; --i) {
        const double* Ci = C + i * stride;
        __m256d acc = _mm256_set1_pd(Ci[0]);
        for (int j = 1; j <= dl; ++j) acc = _mm256_fmadd_pd(_mm256_set1_pd(Ci[j]), Tl[j], acc);
        __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(vxh2, b1, acc), b2);
        b2 = b1; b1 = b0;
    }
    __m256d acc0 = _mm256_set1_pd(C[0]);
    for (int j = 1; j <= dl; ++j) acc0 = _mm256_fmadd_pd(_mm256_set1_pd(C[j]), Tl[j], acc0);
    return _mm256_sub_pd(_mm256_fmadd_pd(_mm256_set1_pd(xh), b1, acc0), b2);
}
// bivariate clenshaw, per-lane xh AND xl.  (mirror of clenshaw2_avx512_vxh.)
inline __m256d clenshaw2_avx2_vxh(int dp, int dl, const double* C, __m256d xh, __m256d xl) {
    const int stride = dl + 1;
    __m256d Tl[17];
    Tl[0] = _mm256_set1_pd(1.0);
    Tl[1] = xl;
    __m256d xl2 = _mm256_mul_pd(_mm256_set1_pd(2.0), xl);
    for (int j = 2; j <= dl; ++j) Tl[j] = _mm256_fmsub_pd(xl2, Tl[j - 1], Tl[j - 2]);
    const __m256d vxh2 = _mm256_mul_pd(_mm256_set1_pd(2.0), xh);
    __m256d b1 = _mm256_setzero_pd(), b2 = _mm256_setzero_pd();
    for (int i = dp; i >= 1; --i) {
        const double* Ci = C + i * stride;
        __m256d acc = _mm256_set1_pd(Ci[0]);
        for (int j = 1; j <= dl; ++j) acc = _mm256_fmadd_pd(_mm256_set1_pd(Ci[j]), Tl[j], acc);
        __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(vxh2, b1, acc), b2);
        b2 = b1; b1 = b0;
    }
    __m256d acc0 = _mm256_set1_pd(C[0]);
    for (int j = 1; j <= dl; ++j) acc0 = _mm256_fmadd_pd(_mm256_set1_pd(C[j]), Tl[j], acc0);
    return _mm256_sub_pd(_mm256_fmadd_pd(xh, b1, acc0), b2);
}

// ==== NEW: AVX2 twins of the LEFT/RIGHT endpoint kernels (namespace-K alias) ====
namespace Kv = volfi_annulus_broadrange;

// absolute value: clear the sign bit.  (AVX2 has no _mm256_abs_pd.)
inline __m256d abs_avx2(__m256d x) {
    return _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
}
// 2^n for an integer-valued n, built directly in the exponent field.  VALID ONLY while
// the biased exponent n+1023 lands in [1,2046], i.e. n in [-1022,1023]; a single power-of-
// two factor cannot represent the subnormal region, so ldexp/scalef fidelity below n=-1022
// is obtained by the two-factor split in exp_neg_avx2, NOT by this helper alone.
inline __m256d pow2i_avx2(__m256d n) {
    __m128i i32 = _mm256_cvtpd_epi32(n);                  // 4x int32 (n is exact integer)
    __m256i i64 = _mm256_cvtepi32_epi64(i32);             // widen to 4x int64
    __m256i eb  = _mm256_slli_epi64(_mm256_add_epi64(i64, _mm256_set1_epi64x(1023)), 52);
    return _mm256_castsi256_pd(eb);                       // 2^n per lane (n in [-1022,1023])
}

// full natural log by reuse: ln2*(exponent + log2approx(mantissa)).  (mirror of full_log_avx512.)
inline __m256d full_log_avx2(__m256d x) {
    __m256i bc = _mm256_castpd_si256(x);
    __m256i j  = _mm256_and_si256(_mm256_srli_epi64(bc, 52), _mm256_set1_epi64x(0x7FFLL));
    __m256d dj = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(j, _mm256_set1_epi64x(0x4330000000000000LL))),
                               _mm256_set1_pd(0x1.0p52));                 // double(j)
    __m256d e  = _mm256_sub_pd(dj, _mm256_set1_pd((double)C_EXP_BIAS));   // j - 1023
    __m256d l2 = log2approx_avx2(mant12_avx2(x));
    return _mm256_fmadd_pd(_mm256_set1_pd(VA_LN2), _mm256_add_pd(e, l2), _mm256_setzero_pd());   // frozen
}

// 1-D Chebyshev (Clenshaw) with affine [a,b]->[-1,1], per-lane x.  (mirror of clenshaw1_avx512.)
inline __m256d clenshaw1_avx2(const double* C, int n, double a, double b, __m256d x) {
    __m256d t  = _mm256_div_pd(_mm256_fmadd_pd(_mm256_set1_pd(2.0), x, _mm256_set1_pd(-(a + b))),
                               _mm256_set1_pd(b - a));
    __m256d t2 = _mm256_add_pd(t, t);
    __m256d d0 = _mm256_setzero_pd(), d1 = _mm256_setzero_pd();
    for (int jx = n - 1; jx >= 1; --jx) {
        __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(t2, d0, _mm256_set1_pd(C[jx])), d1);
        d1 = d0; d0 = b0;
    }
    return _mm256_sub_pd(_mm256_fmadd_pd(t, d0, _mm256_set1_pd(C[0])), d1);
}

// A = Binv(rho): 3 branchless Chebyshev regimes with masked blend.  (mirror of binv_avx512.)
inline __m256d binv_avx2(__m256d rho) {
    const __m256d zero = _mm256_setzero_pd();
    __m256d u  = _mm256_div_pd(_mm256_set1_pd(Kv::BR_K), _mm256_add_pd(rho, _mm256_set1_pd(0.5)));
    __m256d xa = _mm256_mul_pd(u, u);
    __m256d Aa = _mm256_mul_pd(u, clenshaw1_avx2(Kv::BINV_CA, 15, Kv::BINV_U2_A, Kv::BINV_U2_B, xa));
    __m256d L  = _mm256_sub_pd(zero, full_log_avx2(rho));
    __m256d A2_1 = clenshaw1_avx2(Kv::BINV_CB1, 17, Kv::BINV_L_B1A, Kv::BINV_L_B1B, L);
    __m256d A2_2 = clenshaw1_avx2(Kv::BINV_CB2, 25, Kv::BINV_L_B2A, Kv::BINV_L_B2B, L);
    __m256d mB   = _mm256_cmp_pd(rho, _mm256_set1_pd(Kv::BINV_RHO_B), _CMP_GT_OQ);
    __m256d A2   = _mm256_blendv_pd(A2_2, A2_1, mB);        // rho>RHO_B -> CB1 branch
    __m256d AL   = _mm256_sqrt_pd(A2);
    __m256d mA   = _mm256_cmp_pd(rho, _mm256_set1_pd(Kv::BINV_RHO_A), _CMP_GT_OQ);
    return _mm256_blendv_pd(AL, Aa, mA);                    // rho>RHO_A -> regime a
}

// erfcx(z) via the two-piece Chebyshev (C2); mirror of erfcx_poly_avx512.
inline __m256d erfcx_poly_avx2(__m256d z) {
    const __m256d zero = _mm256_setzero_pd();
    __m256d hi = _mm256_cmp_pd(z, _mm256_set1_pd(Kv::RIGHT_ERFCX2_SPLIT), _CMP_GT_OQ);
    __m256d h2 = _mm256_cmp_pd(z, _mm256_set1_pd(Kv::RIGHT_ERFCX_B), _CMP_GT_OQ);
    __m256d napb = _mm256_blendv_pd(_mm256_blendv_pd(
        _mm256_set1_pd(-(Kv::RIGHT_ERFCX_A + Kv::RIGHT_ERFCX2_SPLIT)),
        _mm256_set1_pd(-(Kv::RIGHT_ERFCX2_SPLIT + Kv::RIGHT_ERFCX_B)), hi),
        _mm256_set1_pd(-(Kv::RIGHT_ERFCX_B + Kv::RIGHT_ERFCX2_B2)), h2);
    __m256d bma  = _mm256_blendv_pd(_mm256_blendv_pd(
        _mm256_set1_pd(Kv::RIGHT_ERFCX2_SPLIT - Kv::RIGHT_ERFCX_A),
        _mm256_set1_pd(Kv::RIGHT_ERFCX_B - Kv::RIGHT_ERFCX2_SPLIT), hi),
        _mm256_set1_pd(Kv::RIGHT_ERFCX2_B2 - Kv::RIGHT_ERFCX_B), h2);
    __m256d t  = _mm256_div_pd(_mm256_fmadd_pd(_mm256_set1_pd(2.0), z, napb), bma);
    __m256d t2 = _mm256_add_pd(t, t);
    __m256d d0 = zero, d1 = zero;
    for (int j = Kv::RIGHT_ERFCX2_N - 1; j >= 1; --j) {
        __m256d cj = _mm256_blendv_pd(_mm256_blendv_pd(
                         _mm256_set1_pd(Kv::RIGHT_ERFCX2_C0[j]),
                         _mm256_set1_pd(Kv::RIGHT_ERFCX2_C1[j]), hi),
                         _mm256_set1_pd(Kv::RIGHT_ERFCX2_C2[j]), h2);
        __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(t2, d0, cj), d1);
        d1 = d0; d0 = b0;
    }
    __m256d c0 = _mm256_blendv_pd(_mm256_blendv_pd(
                     _mm256_set1_pd(Kv::RIGHT_ERFCX2_C0[0]),
                     _mm256_set1_pd(Kv::RIGHT_ERFCX2_C1[0]), hi),
                     _mm256_set1_pd(Kv::RIGHT_ERFCX2_C2[0]), h2);
    return _mm256_sub_pd(_mm256_fmadd_pd(t, d0, c0), d1);
}
// exp(a): Cody-Waite ln2 reduction + Chebyshev poly + pow2i (== ldexp).  (mirror of exp_neg_avx512.)
inline __m256d exp_neg_avx2(__m256d a) {
    __m256d nf = _mm256_floor_pd(_mm256_fmadd_pd(a, _mm256_set1_pd(Kv::INV_LN2), _mm256_set1_pd(0.5)));
    __m256d nneg = _mm256_sub_pd(_mm256_setzero_pd(), nf);
    __m256d r  = _mm256_fmadd_pd(nneg, _mm256_set1_pd(Kv::LN2_HI), a);
    r          = _mm256_fmadd_pd(nneg, _mm256_set1_pd(Kv::LN2_LO), r);
    __m256d p  = clenshaw1_avx2(Kv::EXP_C, Kv::EXP_C_N, -0.5 * VA_LN2, 0.5 * VA_LN2, r);
    // p * 2^nf == std::ldexp(p,(int)nf) == _mm512_scalef_pd(p,nf), faithful across the WHOLE
    // finite range incl. the subnormal underflow region: split nf = n1 + n2 (n1 = floor(nf/2))
    // so each 2^n_i stays a normal double.  For the reachable domain (nf in [-1022,0]) both
    // scalings are exact, so this is bit-identical to a single p*2^nf; beyond it, it underflows
    // to the correct subnormal/0 instead of the +0/-inf/sign-flipped garbage a lone (nf+1023)<<52
    // would give.  (Two multiplies vs one; negligible cost in the RIGHT Newton g'.)
    __m256d n1 = _mm256_floor_pd(_mm256_mul_pd(nf, _mm256_set1_pd(0.5)));
    __m256d n2 = _mm256_sub_pd(nf, n1);
    return _mm256_mul_pd(_mm256_mul_pd(p, pow2i_avx2(n1)), pow2i_avx2(n2));
}
inline __m256d erfc_shared_avx2(__m256d t) {
    const __m256d zero = _mm256_setzero_pd();
    __m256d a = abs_avx2(t);
    __m256d e = _mm256_mul_pd(erfcx_poly_avx2(a),
                              exp_neg_avx2(_mm256_sub_pd(zero, _mm256_mul_pd(a, a))));
    __m256d two_minus = _mm256_sub_pd(_mm256_set1_pd(2.0), e);
    __m256d pos = _mm256_cmp_pd(t, zero, _CMP_GE_OQ);
    return _mm256_blendv_pd(two_minus, e, pos);
}
inline __m256d phicdf_avx2(__m256d x) {
    __m256d arg = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(x, _mm256_set1_pd(br::BR_IS2)));
    return _mm256_mul_pd(_mm256_set1_pd(0.5), erfc_shared_avx2(arg));
}
inline __m256d black_otm_avx2(__m256d h, __m256d v, __m256d ehp) {
    const __m256d zero = _mm256_setzero_pd(), half = _mm256_set1_pd(0.5);
    __m256d hv = _mm256_div_pd(h, v);
    __m256d y  = _mm256_fmadd_pd(half, v, _mm256_sub_pd(zero, hv));
    __m256d r  = _mm256_fmadd_pd(half, v, hv);
    return _mm256_fmadd_pd(_mm256_sub_pd(zero, ehp), phicdf_avx2(_mm256_sub_pd(zero, r)), phicdf_avx2(y));
}
inline __m256d onem_otm_avx2(__m256d h, __m256d v, __m256d ehp) {
    const __m256d zero = _mm256_setzero_pd(), half = _mm256_set1_pd(0.5);
    __m256d hv = _mm256_div_pd(h, v);
    __m256d y  = _mm256_fmadd_pd(half, v, _mm256_sub_pd(zero, hv));
    __m256d r  = _mm256_fmadd_pd(half, v, hv);
    return _mm256_fmadd_pd(ehp, phicdf_avx2(_mm256_sub_pd(zero, r)), phicdf_avx2(_mm256_sub_pd(zero, y)));
}
// Deduplicated residual c - C(h,v) (mirror of price_residual_avx512 / br::price_residual).
inline __m256d price_residual_avx2(__m256d vh, __m256d vv, __m256d vehp,
                                   __m256d vc, __m256d onec, __m256d lo) {
    const __m256d zero = _mm256_setzero_pd(), half = _mm256_set1_pd(0.5);
    const __m256d one  = _mm256_set1_pd(1.0);
    const __m256d is2  = _mm256_set1_pd(br::BR_IS2);
    __m256d hv = _mm256_div_pd(vh, vv);
    __m256d y  = _mm256_fmadd_pd(half, vv, _mm256_sub_pd(zero, hv));
    __m256d r  = _mm256_fmadd_pd(half, vv, hv);
    __m256d t1 = _mm256_sub_pd(zero, _mm256_mul_pd(y, is2));                          // phicdf(y)
    __m256d t2 = _mm256_sub_pd(zero, _mm256_mul_pd(_mm256_sub_pd(zero, y), is2));     // phicdf(-y)
    __m256d ay  = abs_avx2(t1);
    __m256d sy  = _mm256_mul_pd(ay, ay);
    __m256d scy = _mm256_fmsub_pd(ay, ay, sy);                // exact low part of ay^2
    __m256d ey  = _mm256_mul_pd(erfcx_poly_avx2(ay),
                    _mm256_mul_pd(exp_neg_avx2(_mm256_sub_pd(zero, sy)),
                                  _mm256_sub_pd(one, scy)));
    __m256d tm = _mm256_sub_pd(_mm256_set1_pd(2.0), ey);
    __m256d phi_y  = _mm256_mul_pd(half, _mm256_blendv_pd(tm, ey,
                         _mm256_cmp_pd(t1, zero, _CMP_GE_OQ)));
    __m256d phi_my = _mm256_mul_pd(half, _mm256_blendv_pd(tm, ey,
                         _mm256_cmp_pd(t2, zero, _CMP_GE_OQ)));
    __m256d tr  = _mm256_sub_pd(zero, _mm256_mul_pd(_mm256_sub_pd(zero, r), is2));
    __m256d ar  = abs_avx2(tr);
    __m256d sr  = _mm256_mul_pd(ar, ar);
    __m256d scr = _mm256_fmsub_pd(ar, ar, sr);
    __m256d er  = _mm256_mul_pd(erfcx_poly_avx2(ar),
                    _mm256_mul_pd(exp_neg_avx2(_mm256_sub_pd(zero, sr)),
                                  _mm256_sub_pd(one, scr)));
    __m256d tmr = _mm256_sub_pd(_mm256_set1_pd(2.0), er);
    __m256d phi_r = _mm256_mul_pd(half, _mm256_blendv_pd(tmr, er,
                        _mm256_cmp_pd(tr, zero, _CMP_GE_OQ)));
    __m256d bl = _mm256_fmadd_pd(_mm256_sub_pd(zero, vehp), phi_r, phi_y);
    __m256d om = _mm256_fmadd_pd(vehp, phi_r, phi_my);
    return _mm256_blendv_pd(_mm256_sub_pd(om, onec), _mm256_sub_pd(vc, bl), lo);
}
// Acklam rational quantile seed; tail branch (p<QN_P_LOW) via full_log.  (mirror of qnorm0_seed_avx512.)
inline __m256d qnorm0_seed_avx2(__m256d p) {
    __m256d q = _mm256_sub_pd(p, _mm256_set1_pd(0.5));
    __m256d r = _mm256_mul_pd(q, q);
    __m256d num = _mm256_set1_pd(Kv::QN_A[0]);
    for (int i = 1; i < 6; ++i) num = _mm256_fmadd_pd(num, r, _mm256_set1_pd(Kv::QN_A[i]));
    num = _mm256_mul_pd(num, q);
    __m256d den = _mm256_set1_pd(Kv::QN_B[0]);
    for (int i = 1; i < 5; ++i) den = _mm256_fmadd_pd(den, r, _mm256_set1_pd(Kv::QN_B[i]));
    den = _mm256_fmadd_pd(den, r, _mm256_set1_pd(1.0));
    __m256d mid = _mm256_div_pd(num, den);
    __m256d qt = _mm256_sqrt_pd(_mm256_mul_pd(_mm256_set1_pd(-2.0), full_log_avx2(p)));
    __m256d numt = _mm256_set1_pd(Kv::QN_C[0]);
    for (int i = 1; i < 6; ++i) numt = _mm256_fmadd_pd(numt, qt, _mm256_set1_pd(Kv::QN_C[i]));
    __m256d dent = _mm256_set1_pd(Kv::QN_D[0]);
    for (int i = 1; i < 4; ++i) dent = _mm256_fmadd_pd(dent, qt, _mm256_set1_pd(Kv::QN_D[i]));
    dent = _mm256_fmadd_pd(dent, qt, _mm256_set1_pd(1.0));
    __m256d tail = _mm256_div_pd(numt, dent);
    __m256d tl = _mm256_cmp_pd(p, _mm256_set1_pd(Kv::QN_P_LOW), _CMP_LT_OQ);
    return _mm256_blendv_pd(mid, tail, tl);
}

// LEFT chart, 4 lanes -> w.  (mirror of left_variance_avx512.)
inline __m256d left_variance_avx2(__m256d vc, __m256d v_expm1h, __m256d vh,
                                  __m256d vh2, __m256d vh4, __m256d vxt) {
    __m256d rho = _mm256_div_pd(vc, v_expm1h);
    __m256d A   = binv_avx2(rho);
    __m256d s   = _mm256_div_pd(vh, A);
    __m256d x   = sigma0_poly_avx2(_mm256_mul_pd(_mm256_set1_pd(Kv::BR_K), s));
    __m256d xs  = _mm256_fmadd_pd(_mm256_add_pd(s, s), _mm256_set1_pd(1.0 / Kv::LEFT_S_MAX), _mm256_set1_pd(-1.0));
    __m256d V2  = clenshaw1_avx2(Kv::LEFT_V2_CHEB, Kv::LEFT_V2_CHEB_N, 0.0, Kv::LEFT_S_CHEB_MAX, s);
    __m256d V4  = clenshaw1_avx2(Kv::LEFT_V4_CHEB, Kv::LEFT_V4_CHEB_N, 0.0, Kv::LEFT_S_CHEB_MAX, s);
    __m256d fin = clenshaw2_avx2_vxh(Kv::LEFT_FIN_DP, Kv::LEFT_FIN_DL, Kv::LEFT_FIN_COEFFS, vxt, xs);
    __m256d v   = _mm256_fmadd_pd(vh4, V4, _mm256_fmadd_pd(vh2, V2, x));
    v           = _mm256_fmadd_pd(_mm256_mul_pd(vh4, vh2), fin, v);
    return _mm256_mul_pd(v, v);
}

// RIGHT chart, 4 lanes -> w.  (mirror of right_variance_avx512.)
inline __m256d right_variance_avx2(__m256d vc, __m256d veh, __m256d vehp, __m256d vh, __m256d vh2) {
    const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0);
    __m256d gbar = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_sub_pd(one, vc)), veh);
    __m256d x0   = _mm256_sub_pd(zero, qnorm0_seed_avx2(gbar));
    __m256d m    = _mm256_mul_pd(_mm256_set1_pd(Kv::SQRT_PI2),
                                 erfcx_poly_avx2(_mm256_mul_pd(x0, _mm256_set1_pd(br::BR_IS2))));
    __m256d inv  = _mm256_div_pd(one, x0);
    __m256d x2   = _mm256_mul_pd(_mm256_set1_pd(-0.125), _mm256_sub_pd(inv, m));
    __m256d B2p  = _mm256_mul_pd(_mm256_add_pd(_mm256_fmadd_pd(_mm256_mul_pd(inv, inv), inv,
                                                              _mm256_sub_pd(zero, inv)), m),
                                 _mm256_set1_pd(1.0 / 3.0));
    __m256d ta   = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), x0), x2), x2);
    __m256d tb   = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.125), x2), inv), inv);
    __m256d x4   = _mm256_fmadd_pd(B2p, _mm256_set1_pd(1.0 / 128.0), _mm256_add_pd(ta, tb));
    __m256d x    = _mm256_fmadd_pd(_mm256_mul_pd(vh2, vh2), x4, _mm256_fmadd_pd(vh2, x2, x0));
    __m256d onec = _mm256_sub_pd(one, vc);
    __m256d lo   = _mm256_cmp_pd(vc, _mm256_set1_pd(0.5), _CMP_LT_OQ);
    for (int it = 0; it < Kv::RIGHT_HH3_STEPS; ++it) {          // HH3 (order 4); mirrors scalar
        __m256d vv  = _mm256_mul_pd(_mm256_set1_pd(2.0), x);
        __m256d cmC = price_residual_avx2(vh, vv, vehp, vc, onec, lo);
        __m256d f   = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), cmC), veh);
        __m256d eA  = exp_neg_avx2(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-0.5), x), x));
        __m256d eB  = exp_neg_avx2(_mm256_div_pd(_mm256_sub_pd(zero, vh2),
                                                 _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(8.0), x), x)));
        __m256d gp  = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-Kv::BR_K), eA), eB);
        __m256d u   = _mm256_div_pd(f, gp);
        __m256d ix  = _mm256_div_pd(one, x);
        __m256d ix2 = _mm256_mul_pd(ix, ix);
        __m256d a   = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_set1_pd(0.25), vh2),
                                      _mm256_mul_pd(ix2, ix), _mm256_sub_pd(zero, x));   // L
        __m256d Lp  = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_set1_pd(-0.75), vh2),
                                      _mm256_mul_pd(ix2, ix2), _mm256_set1_pd(-1.0));    // L'
        __m256d b   = _mm256_fmadd_pd(a, a, Lp);                                        // L^2 + L'
        __m256d au  = _mm256_mul_pd(a, u);
        __m256d num = _mm256_fmadd_pd(_mm256_set1_pd(-0.5), au, one);                   // 1 - 0.5 a u
        __m256d den = _mm256_fmadd_pd(_mm256_mul_pd(b, u),
                                      _mm256_mul_pd(u, _mm256_set1_pd(1.0 / 6.0)),
                                      _mm256_sub_pd(one, au));                          // 1 - a u + b u^2/6
        x = _mm256_sub_pd(x, _mm256_div_pd(_mm256_mul_pd(u, num), den));
    }
    __m256d v = _mm256_mul_pd(_mm256_set1_pd(2.0), x);
    return _mm256_mul_pd(v, v);
}

// Warm-start refinement, 4 lanes (mirror of warm_variance_avx512 / br::warm_variance).
inline __m256d warm_variance_avx2(__m256d vh, __m256d vc, __m256d vw,
                                  int steps = Kv::WARM_HH3_STEPS) {
    const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0), half = _mm256_set1_pd(0.5);
    __m256d x    = _mm256_mul_pd(half, _mm256_sqrt_pd(vw));
    __m256d vh2  = _mm256_mul_pd(vh, vh);
    __m256d veh  = exp_neg_avx2(_mm256_mul_pd(_mm256_set1_pd(-0.5), vh));
    __m256d vehp = exp_neg_avx2(vh);
    __m256d onec = _mm256_sub_pd(one, vc);
    __m256d lo   = _mm256_cmp_pd(vc, half, _CMP_LT_OQ);
    for (int it = 0; it < steps; ++it) {
        __m256d vv  = _mm256_mul_pd(_mm256_set1_pd(2.0), x);
        __m256d cmC = price_residual_avx2(vh, vv, vehp, vc, onec, lo);
        __m256d f   = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), cmC), veh);
        __m256d eA  = exp_neg_avx2(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-0.5), x), x));
        __m256d eB  = exp_neg_avx2(_mm256_div_pd(_mm256_sub_pd(zero, vh2),
                                                 _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(8.0), x), x)));
        __m256d gp  = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-Kv::BR_K), eA), eB);
        __m256d u   = _mm256_div_pd(f, gp);
        __m256d ix  = _mm256_div_pd(one, x);
        __m256d ix2 = _mm256_mul_pd(ix, ix);
        __m256d a   = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_set1_pd(0.25), vh2),
                                      _mm256_mul_pd(ix2, ix), _mm256_sub_pd(zero, x));
        __m256d Lp  = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_set1_pd(-0.75), vh2),
                                      _mm256_mul_pd(ix2, ix2), _mm256_set1_pd(-1.0));
        __m256d b   = _mm256_fmadd_pd(a, a, Lp);
        __m256d au  = _mm256_mul_pd(a, u);
        __m256d num = _mm256_fmadd_pd(_mm256_set1_pd(-0.5), au, one);
        __m256d den = _mm256_fmadd_pd(_mm256_mul_pd(b, u),
                                      _mm256_mul_pd(u, _mm256_set1_pd(1.0 / 6.0)),
                                      _mm256_sub_pd(one, au));
        x = _mm256_sub_pd(x, _mm256_div_pd(_mm256_mul_pd(u, num), den));
    }
    __m256d v = _mm256_mul_pd(_mm256_set1_pd(2.0), x);
    return _mm256_mul_pd(v, v);
}

// ==== AVX2 WING twins (op-for-op mirror of the AVX-512 wing kernels) ====
inline __m256d full_log_sub_avx2(__m256d x) {
    __m256d sb = _mm256_cmp_pd(x, _mm256_set1_pd(VA_DBL_MIN), _CMP_LT_OQ);
    __m256d xs = _mm256_blendv_pd(x, _mm256_mul_pd(x, _mm256_set1_pd(VA_TWO54)), sb);
    __m256i bc = _mm256_castpd_si256(xs);
    __m256i j  = _mm256_and_si256(_mm256_srli_epi64(bc, 52), _mm256_set1_epi64x(0x7FFLL));
    __m256d dj = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(j, _mm256_set1_epi64x(0x4330000000000000LL))),
                               _mm256_set1_pd(0x1.0p52));
    __m256d e  = _mm256_sub_pd(dj, _mm256_set1_pd((double)C_EXP_BIAS));
    e = _mm256_blendv_pd(e, _mm256_sub_pd(e, _mm256_set1_pd(54.0)), sb);
    __m256d l2 = log2approx_avx2(mant12_avx2(xs));
    return _mm256_fmadd_pd(_mm256_set1_pd(VA_LN2), _mm256_add_pd(e, l2), _mm256_setzero_pd());   // frozen
}
inline __m256d wing_series_d_avx2(__m256d W, __m256d b, __m256d* dlogH) {
    const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0);
    __m256d b2 = _mm256_mul_pd(b, b), b3 = _mm256_mul_pd(b2, b), b4 = _mm256_mul_pd(b2, b2);
    __m256d g1 = _mm256_sub_pd(zero, _mm256_add_pd(b, _mm256_set1_pd(1.5)));
    __m256d g2 = _mm256_add_pd(b, _mm256_set1_pd(21.0/8.0));
    __m256d g3 = _mm256_sub_pd(zero, _mm256_fmadd_pd(_mm256_set1_pd(3.5), b, _mm256_set1_pd(69.0/8.0)));
    __m256d g4 = _mm256_fmadd_pd(_mm256_set1_pd(0.5), b2,
                 _mm256_fmadd_pd(_mm256_set1_pd(17.25), b, _mm256_set1_pd(2529.0/64.0)));
    __m256d g5 = _mm256_sub_pd(zero, _mm256_fmadd_pd(_mm256_set1_pd(5.5), b2,
                 _mm256_fmadd_pd(_mm256_set1_pd(105.375), b, _mm256_set1_pd(36243.0/160.0))));
    __m256d g6 = _mm256_fmadd_pd(_mm256_set1_pd(1.0/3.0), b3,
                 _mm256_fmadd_pd(_mm256_set1_pd(53.875), b2,
                 _mm256_fmadd_pd(_mm256_set1_pd(755.0625), b, _mm256_set1_pd(197127.0/128.0))));
    __m256d g7 = _mm256_sub_pd(zero, _mm256_fmadd_pd(_mm256_set1_pd(7.5), b3,
                 _mm256_fmadd_pd(_mm256_set1_pd(537.75), b2,
                 _mm256_fmadd_pd(_mm256_set1_pd(6160.21875), b, _mm256_set1_pd(10786527.0/896.0)))));
    __m256d g8 = _mm256_fmadd_pd(_mm256_set1_pd(0.25), b4,
                 _mm256_fmadd_pd(_mm256_set1_pd(122.5), b3,
                 _mm256_fmadd_pd(_mm256_set1_pd(5651.15625), b2,
                 _mm256_fmadd_pd(_mm256_set1_pd(56179.828125), b, _mm256_set1_pd(217179009.0/2048.0)))));
    __m256d x = _mm256_div_pd(one, W);
    __m256d s = g8;
    s = _mm256_fmadd_pd(s, x, g7); s = _mm256_fmadd_pd(s, x, g6); s = _mm256_fmadd_pd(s, x, g5);
    s = _mm256_fmadd_pd(s, x, g4); s = _mm256_fmadd_pd(s, x, g3); s = _mm256_fmadd_pd(s, x, g2);
    s = _mm256_fmadd_pd(s, x, g1);
    __m256d lH = _mm256_mul_pd(s, x);
    __m256d d = _mm256_mul_pd(_mm256_set1_pd(8.0), g8);
    d = _mm256_fmadd_pd(d, x, _mm256_mul_pd(_mm256_set1_pd(7.0), g7));
    d = _mm256_fmadd_pd(d, x, _mm256_mul_pd(_mm256_set1_pd(6.0), g6));
    d = _mm256_fmadd_pd(d, x, _mm256_mul_pd(_mm256_set1_pd(5.0), g5));
    d = _mm256_fmadd_pd(d, x, _mm256_mul_pd(_mm256_set1_pd(4.0), g4));
    d = _mm256_fmadd_pd(d, x, _mm256_mul_pd(_mm256_set1_pd(3.0), g3));
    d = _mm256_fmadd_pd(d, x, _mm256_mul_pd(_mm256_set1_pd(2.0), g2));
    d = _mm256_fmadd_pd(d, x, g1);
    *dlogH = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_sub_pd(zero, x), x), d, zero);   // frozen
    return lH;
}
inline __m256d wing_fit_d_avx2(__m256d W, __m256d beta, const double* C, int NU, int NB,
                               double ulo, double uhi, __m256d* dlogH) {
    namespace wd = wing_detail;
    const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0);
    __m256d u = _mm256_div_pd(one, W);
    u = _mm256_min_pd(_mm256_max_pd(u, _mm256_set1_pd(ulo)), _mm256_set1_pd(uhi));
    __m256d xu = _mm256_div_pd(_mm256_fmadd_pd(_mm256_set1_pd(2.0), u, _mm256_set1_pd(-(ulo + uhi))),
                              _mm256_set1_pd(uhi - ulo));
    __m256d xb = _mm256_div_pd(_mm256_fmadd_pd(_mm256_set1_pd(2.0), beta,
                              _mm256_set1_pd(-(wd::WING_B_LO + wd::WING_B_HI))),
                              _mm256_set1_pd(wd::WING_B_HI - wd::WING_B_LO));
    __m256d rows[27];
    __m256d tb2 = _mm256_add_pd(xb, xb);
    for (int i = 0; i < NU; ++i) {
        const double* Ci = C + i * NB;
        __m256d d0 = zero, d1 = zero;
        for (int j = NB - 1; j >= 1; --j) {
            __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(tb2, d0, _mm256_set1_pd(Ci[j])), d1);
            d1 = d0; d0 = b0;
        }
        rows[i] = _mm256_sub_pd(_mm256_fmadd_pd(xb, d0, _mm256_set1_pd(Ci[0])), d1);
    }
    __m256d tu2 = _mm256_add_pd(xu, xu);
    __m256d d0 = zero, d1 = zero;
    for (int i = NU - 1; i >= 1; --i) {
        __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(tu2, d0, rows[i]), d1);
        d1 = d0; d0 = b0;
    }
    __m256d S = _mm256_sub_pd(_mm256_fmadd_pd(xu, d0, rows[0]), d1);
    __m256d dp[27];
    dp[NU - 1] = zero;
    dp[NU - 2] = _mm256_mul_pd(_mm256_set1_pd(2.0 * (NU - 1)), rows[NU - 1]);
    for (int k = NU - 3; k >= 0; --k)
        dp[k] = _mm256_fmadd_pd(_mm256_set1_pd(2.0 * (k + 1)), rows[k + 1], dp[k + 2]);
    dp[0] = _mm256_mul_pd(dp[0], _mm256_set1_pd(0.5));
    d0 = zero; d1 = zero;
    for (int i = NU - 1; i >= 1; --i) {
        __m256d b0 = _mm256_sub_pd(_mm256_fmadd_pd(tu2, d0, dp[i]), d1);
        d1 = d0; d0 = b0;
    }
    __m256d dS_dxu = _mm256_sub_pd(_mm256_fmadd_pd(xu, d0, dp[0]), d1);
    __m256d Su = _mm256_mul_pd(dS_dxu, _mm256_set1_pd(2.0 / (uhi - ulo)));
    __m256d bp = _mm256_add_pd(beta, _mm256_set1_pd(1.5));
    *dlogH = _mm256_fmadd_pd(_mm256_sub_pd(zero, _mm256_mul_pd(u, u)),
                             _mm256_sub_pd(_mm256_fmadd_pd(u, Su, S), bp), zero);   // frozen
    return _mm256_mul_pd(u, _mm256_sub_pd(S, bp));
}
inline __m256d wing_variance_avx2(__m256d vh, __m256d vc, int regime) {
    namespace wd = wing_detail;
    const __m256d zero = _mm256_setzero_pd(), two = _mm256_set1_pd(2.0);
    __m256d beta = _mm256_min_pd(_mm256_mul_pd(_mm256_mul_pd(vh, vh), _mm256_set1_pd(1.0 / 16.0)),
                                 _mm256_set1_pd(wd::WING_B_HI));
    __m256d lc = _mm256_sub_pd(full_log_avx2(vh), _mm256_set1_pd(wd::WING_LOG4));
    lc = _mm256_sub_pd(_mm256_fmadd_pd(_mm256_set1_pd(0.5), vh, lc), _mm256_set1_pd(wd::WING_HLOGPI));
    __m256d Lt = _mm256_add_pd(_mm256_sub_pd(zero, full_log_sub_avx2(vc)), lc);
    __m256d lt = _mm256_max_pd(Lt, two);
    __m256d W = _mm256_max_pd(_mm256_fmadd_pd(_mm256_set1_pd(-1.5), full_log_avx2(lt), lt),
                              _mm256_set1_pd(2.6));
    for (int it = 0; it < wd::WING_NEWTON_STEPS; ++it) {
        __m256d dlH, lH;
        if (regime == 2)      lH = wing_series_d_avx2(W, beta, &dlH);
        else if (regime == 0) lH = wing_fit_d_avx2(W, beta, wd::WING_SA, wd::WING_SA_NU, wd::WING_SA_NB,
                                                   wd::WING_UA_LO, wd::WING_UA_HI, &dlH);
        else                  lH = wing_fit_d_avx2(W, beta, wd::WING_SB, wd::WING_SB_NU, wd::WING_SB_NB,
                                                   wd::WING_UB_LO, wd::WING_UB_HI, &dlH);
        __m256d F  = _mm256_add_pd(_mm256_sub_pd(W, Lt),
                                   _mm256_fmadd_pd(_mm256_set1_pd(1.5), full_log_avx2(W),
                                                   _mm256_sub_pd(zero, lH)));
        __m256d Fp = _mm256_sub_pd(_mm256_add_pd(_mm256_set1_pd(1.0),
                                                 _mm256_div_pd(_mm256_set1_pd(1.5), W)), dlH);
        W = _mm256_max_pd(_mm256_sub_pd(W, _mm256_div_pd(F, Fp)), _mm256_set1_pd(2.6));
    }
    return _mm256_div_pd(_mm256_mul_pd(vh, vh), _mm256_mul_pd(two, W));
}
} // namespace detail
#endif // VA_SIMD256

namespace detail {
// Fill log2m[0..n) = log2approx(mantissa(c[i])) using the widest available SIMD.
// This is the B1 win: the once-scalar std::log2 of the index pass now vectorizes.
// The tail (and the whole thing on a no-FMA build) uses the scalar log2approx,
// which is bit-identical to the vector lanes.
inline void log2approx_block(const double* c, double* out, int n) {
    int i = 0;
#if defined(VA_SIMD512)
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(out + i, log2approx_avx512(mant12_avx512(_mm512_loadu_pd(c + i))));
#elif defined(VA_SIMD256)
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(out + i, log2approx_avx2(mant12_avx2(_mm256_loadu_pd(c + i))));
#endif
    for (; i < n; ++i) out[i] = log2approx(mant12(bits_of(c[i])));
}
} // namespace detail


// ============================================================================
//  FIXED-h batch (surface): one context -> one h -> one xh; only c varies.
//  SORT-THEN-BATCH.  Only the CENTRAL subset (route==CENTRAL && real table cell)
//  is vectorized here; WING/LEFT/RIGHT/analytic-edge quotes drop to the scalar
//  fallback pass (bit-identical: same shared kernels, same fma order).
// ============================================================================
namespace detail {

// (region 0) CENTRAL table core.  xlv is reused as scratch: the vectorized
// log2approx pre-pass writes log2m into xlv, then routing rewrites xlv[i] in
// place to the per-cell abscissa xl = fma(LSCALE, log2m, LBIAS).  A quote is
// taken here only when central_cell_of() accepts it (route==CENTRAL + cell hit).
__attribute__((noinline)) inline void band_table_pass(
        const context& q, const double* c, double* w_out, int n) {
    const int    band  = q.band;
    const int    cbase = CBASE[band];
    const int    ncell = ((band + 1 < NB) ? CBASE[band + 1] : NCELLS) - cbase;  // <=14
    const double h2    = q.h2;
    const double xh    = q.xh;

    constexpr int TILE = 4096;
    int    tloc[TILE];       // local cell (0..ncell-1), -1 = fallback
    double xlv [TILE];       // log2m -> xl (in place)
    int    order[TILE];

    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;

        log2approx_block(c + base, xlv, m);            // vectorized index-pass log2

        int cnt[32];
        for (int t = 0; t < ncell; ++t) cnt[t] = 0;
        for (int i = 0; i < m; ++i) {
            double ci = c[base + i];
            if (!(ci > 0.0) || ci >= 1.0) { tloc[i] = -1; continue; }
            uint64_t bc = bits_of(ci);
            int cell = central_cell_of(q, ci, bc);     // route==CENTRAL gate + table cell
            if (cell < 0) { tloc[i] = -1; continue; }
            int t = cell - cbase;
            tloc[i] = t;
            xlv[i]  = std::fma(CELL_LSCALE[cell], xlv[i], CELL_LBIAS[cell]);
            ++cnt[t];
        }

        int off[32], pos[32], run = 0;
        for (int t = 0; t < ncell; ++t) { off[t] = run; run += cnt[t]; }
        for (int t = 0; t < ncell; ++t) pos[t] = off[t];
        for (int i = 0; i < m; ++i) { int t = tloc[i]; if (t >= 0) order[pos[t]++] = i; }

        for (int t = 0; t < ncell; ++t) {
            const int b0 = off[t], nb = cnt[t];
            if (nb == 0) continue;
            const int ci_cell = cbase + t;
            int e = 0;
#if defined(VA_SIMD512)
            const __m512d vtwo = _mm512_set1_pd(2.0);
            const __m512d vh2  = _mm512_set1_pd(h2);
            for (; e + 8 <= nb; e += 8) {
                double xlbuf[8]; const int* od = order + b0 + e;
                for (int l = 0; l < 8; ++l) xlbuf[l] = xlv[od[l]];
                __m512d W = clenshaw2_avx512(CELL_DP[ci_cell], CELL_DL[ci_cell],
                                             &COEFFS[CELL_OFF[ci_cell]], xh, _mm512_loadu_pd(xlbuf));
                __m512d w = _mm512_div_pd(vh2, _mm512_mul_pd(vtwo, W));
                double wbuf[8]; _mm512_storeu_pd(wbuf, w);
                for (int l = 0; l < 8; ++l) w_out[base + od[l]] = wbuf[l];
            }
#elif defined(VA_SIMD256)
            const __m256d vtwo = _mm256_set1_pd(2.0);
            const __m256d vh2  = _mm256_set1_pd(h2);
            for (; e + 4 <= nb; e += 4) {
                double xlbuf[4]; const int* od = order + b0 + e;
                for (int l = 0; l < 4; ++l) xlbuf[l] = xlv[od[l]];
                __m256d W = clenshaw2_avx2(CELL_DP[ci_cell], CELL_DL[ci_cell],
                                           &COEFFS[CELL_OFF[ci_cell]], xh, _mm256_loadu_pd(xlbuf));
                __m256d w = _mm256_div_pd(vh2, _mm256_mul_pd(vtwo, W));
                double wbuf[4]; _mm256_storeu_pd(wbuf, w);
                for (int l = 0; l < 4; ++l) w_out[base + od[l]] = wbuf[l];
            }
#endif
            for (; e < nb; ++e) {
                int idx = order[b0 + e];
                double W = clenshaw2_cell(ci_cell, xh, xlv[idx]);
                w_out[base + idx] = h2 / (2.0 * W);
            }
        }
    }
}

// non-central band (region 1 LEFT / region 2 RIGHT) or non-table: everything scalar.
__attribute__((noinline)) inline void all_scalar_pass(
        const context& q, const double* c, double* w_out, int n) {
    for (int i = 0; i < n; ++i) w_out[i] = scalar_fallback(q, c[i]);
}

// ---- Fixed-h LEFT / RIGHT bucket processors (idx[] = absolute quote indices) ----
// The per-h scalars are broadcast from the context (index-pass libm, once per h).
// The vector lanes reproduce br::left_variance / br::right_variance bit-for-bit
// (verified in kernel_bit_test), and the tail calls the SAME scalar kernels, so
// batch == scalar entry per quote (invariant b).  On non-512 builds the whole
// bucket runs through the scalar tail (still bit-identical, just not vectorized).
__attribute__((noinline)) inline void left_pass_fixed(
        const context& q, const double* c, double* w_out, const int* idx, int nb) {
    int e = 0;
#if defined(VA_SIMD512)
    const __m512d vE  = _mm512_set1_pd(q.expm1h), vh = _mm512_set1_pd(q.h),
                  vh2 = _mm512_set1_pd(q.h2), vh4 = _mm512_set1_pd(q.h2 * q.h2),
                  vxt = _mm512_set1_pd(q.xt_left);
    double cb[8], wb[8];
    for (; e + 8 <= nb; e += 8) {
        for (int l = 0; l < 8; ++l) cb[l] = c[idx[e + l]];
        __m512d w = left_variance_avx512(_mm512_loadu_pd(cb), vE, vh, vh2, vh4, vxt);
        _mm512_storeu_pd(wb, w);
        for (int l = 0; l < 8; ++l) w_out[idx[e + l]] = wb[l];
    }
#elif defined(VA_SIMD256)
    const __m256d vE  = _mm256_set1_pd(q.expm1h), vh = _mm256_set1_pd(q.h),
                  vh2 = _mm256_set1_pd(q.h2), vh4 = _mm256_set1_pd(q.h2 * q.h2),
                  vxt = _mm256_set1_pd(q.xt_left);
    double cb[4], wb[4];
    for (; e + 4 <= nb; e += 4) {
        for (int l = 0; l < 4; ++l) cb[l] = c[idx[e + l]];
        __m256d w = left_variance_avx2(_mm256_loadu_pd(cb), vE, vh, vh2, vh4, vxt);
        _mm256_storeu_pd(wb, w);
        for (int l = 0; l < 4; ++l) w_out[idx[e + l]] = wb[l];
    }
#endif
    for (; e < nb; ++e) { int i = idx[e]; w_out[i] = br::left_variance(q.h, c[i]); }
}
__attribute__((noinline)) inline void right_pass_fixed(
        const context& q, const double* c, double* w_out, const int* idx, int nb) {
    int e = 0;
#if defined(VA_SIMD512)
    const __m512d veh = _mm512_set1_pd(q.eh), vehp = _mm512_set1_pd(q.ehp),
                  vh = _mm512_set1_pd(q.h), vh2 = _mm512_set1_pd(q.h2);
    double cb[8], wb[8];
    for (; e + 8 <= nb; e += 8) {
        for (int l = 0; l < 8; ++l) cb[l] = c[idx[e + l]];
        __m512d w = right_variance_avx512(_mm512_loadu_pd(cb), veh, vehp, vh, vh2);
        _mm512_storeu_pd(wb, w);
        for (int l = 0; l < 8; ++l) w_out[idx[e + l]] = wb[l];
    }
#elif defined(VA_SIMD256)
    const __m256d veh = _mm256_set1_pd(q.eh), vehp = _mm256_set1_pd(q.ehp),
                  vh = _mm256_set1_pd(q.h), vh2 = _mm256_set1_pd(q.h2);
    double cb[4], wb[4];
    for (; e + 4 <= nb; e += 4) {
        for (int l = 0; l < 4; ++l) cb[l] = c[idx[e + l]];
        __m256d w = right_variance_avx2(_mm256_loadu_pd(cb), veh, vehp, vh, vh2);
        _mm256_storeu_pd(wb, w);
        for (int l = 0; l < 4; ++l) w_out[idx[e + l]] = wb[l];
    }
#endif
    for (; e < nb; ++e) { int i = idx[e]; w_out[i] = br::right_variance(q.h, c[i]); }
}

// (region 0) endpoint + scalar fallback for the CENTRAL vector core's leftovers:
// RIGHT (c>ct2) -> vectorized RIGHT bucket; WING (c<cw) / analytic edge -> scalar.
__attribute__((noinline)) inline void band_endpoint_fallback(
        const context& q, const double* c, double* w_out, int n) {
    constexpr int TILE = 4096;
    int ridx[TILE];
    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;
        int rc = 0;
        for (int i = 0; i < m; ++i) {
            const int ai = base + i; double ci = c[ai];
            if (!(ci > 0.0) || ci >= 1.0) { w_out[ai] = scalar_fallback(q, ci); continue; }
            // WING has precedence over ct2 in the scalar routing: at large h the wing
            // seam cw can EXCEED the v=2 seam ct2, so c<cw must be tested FIRST.
            if (ci < q.cw) { w_out[ai] = scalar_fallback(q, ci); continue; }             // WING
            if (central_cell_of(q, ci, bits_of(ci)) >= 0) continue;   // CENTRAL: already written
            if (ci > q.ct2) ridx[rc++] = ai;                          // RIGHT (v>2)
            else            w_out[ai] = scalar_fallback(q, ci);       // analytic edge (in box, no cell)
        }
        right_pass_fixed(q, c, w_out, ridx, rc);
    }
}

// (region 1 LEFT band / region 2 RIGHT-only band) endpoint bucketer: LEFT and RIGHT
// through the vector kernels, WING (c<cw) and degenerate c through the scalar entry.
__attribute__((noinline)) inline void band_endpoint_region12(
        const context& q, const double* c, double* w_out, int n) {
    constexpr int TILE = 4096;
    int lidx[TILE], ridx[TILE];
    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;
        int lc = 0, rc = 0;
        for (int i = 0; i < m; ++i) {
            const int ai = base + i; double ci = c[ai];
            if (!(ci > 0.0) || ci >= 1.0) { w_out[ai] = scalar_fallback(q, ci); continue; }
            if (ci < q.cw) { w_out[ai] = scalar_fallback(q, ci); continue; }   // WING
            if (q.region == 1) { if (ci <= q.ct_left) lidx[lc++] = ai; else ridx[rc++] = ai; }
            else               { ridx[rc++] = ai; }                            // region 2: RIGHT
        }
        left_pass_fixed (q, c, w_out, lidx, lc);
        right_pass_fixed(q, c, w_out, ridx, rc);
    }
}
} // namespace detail

// Fixed-h surface batch.  CENTRAL (region 0) stays the frozen vector table core;
// LEFT/RIGHT endpoints now vectorize too (region 1/2, and region-0 v>2).  Only
// WING and the analytic ATM/deep edges drop to the scalar entry.  h==0 (exact ATM
// line) is entirely scalar.  batch == scalar entry per quote (invariant b).
inline void implied_variance_otm_batch(const context& q,
                                       const double* c, double* w_out, int n) {
    if (q.h == 0.0) { detail::all_scalar_pass(q, c, w_out, n); return; }  // exact ATM line
    if (q.region == 0) {
        detail::band_table_pass(q, c, w_out, n);        // CENTRAL vector core (route-gated)
        detail::band_endpoint_fallback(q, c, w_out, n); // RIGHT vector + WING/edge scalar
    } else {
        detail::band_endpoint_region12(q, c, w_out, n); // LEFT/RIGHT vector + WING scalar
    }
}

// ============================================================================
//  D: WARM-START entries for calibration loops.  Re-invert a moved surface from the
//  previous iteration's variances: no routing, no seed, no table -- three fixed
//  Householder-3 steps on the exact OTM equation per quote, fully SIMD in the batch
//  (mixed h, no sorting).  CALLER CONTRACT: valid when W = h^2/(2 w_prev) <= 40 and the
//  vol moved by no more than a few percent; accuracy is a FEW ULP (worst 2.9e-15 for
//  W<=8, ~1.3e-14 at W=40; median sub-ulp), not last-bit -- cold-invert the final iterate
//  if last-bit output is required (basin + floor notes in volfi_annulus_endpoint_vec.hpp);
//  quotes outside the basin must be re-inverted cold.  batch == scalar bit-identical.
// ============================================================================
inline double implied_variance_warm(double h, double c, double w_prev) {
    return br::warm_variance(h, c, w_prev);
}
// Warm batch.  steps=3 (default) is the general +-few-percent basin; steps=2 is the
// fast streaming path for sub-1% moves (see WARM_HH3_FAST_STEPS).  A batch is
// lane-uniform, so the caller picks the step count for the whole vector; for mixed
// move sizes use steps=3, or call the scalar warm_variance_fast per node and
// re-invert the flagged (out-of-basin) minority cold.
inline void implied_variance_warm_batch(const double* h, const double* c,
                                        const double* w_prev, double* w_out, int n,
                                        int steps = 3) {
    int i = 0;
#if defined(VA_SIMD512)
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(w_out + i,
            detail::warm_variance_avx512(_mm512_loadu_pd(h + i), _mm512_loadu_pd(c + i),
                                         _mm512_loadu_pd(w_prev + i), steps));
#elif defined(VA_SIMD256)
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(w_out + i,
            detail::warm_variance_avx2(_mm256_loadu_pd(h + i), _mm256_loadu_pd(c + i),
                                       _mm256_loadu_pd(w_prev + i), steps));
#endif
    for (; i < n; ++i) w_out[i] = br::warm_variance(h[i], c[i], w_prev[i], steps);
}

// ============================================================================
//  BOOK MODE: streaming re-inversion of a persistent node set.
//  A live book re-prices the SAME (strike, expiry) nodes every snapshot, so each
//  node's previous variance is a seed <1% away and warm refinement -- no routing,
//  no chart evaluation -- is the steady-state path (faster than a cold re-route,
//  and flat in batch size because there is no deferred drain).  The caller carries
//  exactly two things between snapshots: the STATIC per-node log-strike, and the
//  previous per-node variance (w_out feeds back as next snapshot's w_prev).  Per
//  snapshot it supplies the per-node log-forward (one value per expiry, broadcast
//  to that expiry's nodes) and the new OTM-call-branch normalized price c=price/F.
//  Contract: valid for sub-1% inter-snapshot vol moves; the 2-step warm then
//  matches the cold inversion to the warm floor (a few ULP, not the last bit -- see
//  the warm basin note).  For onboarding, a regime break, or a last-bit refresh,
//  invert cold with implied_variance_grid_batch and reseed w_prev.
// ----------------------------------------------------------------------------
// Batch snapshot.  h_scratch is caller-owned working space of size n (need not
// persist).  Uses the fast 2-step warm; assumes sub-1% moves (streaming regime).
inline void implied_variance_book_step(const double* logK, const double* logF,
                                       const double* c, const double* w_prev,
                                       double* h_scratch, double* w_out, int n) {
    for (int i = 0; i < n; ++i) h_scratch[i] = std::fabs(logK[i] - logF[i]);
    implied_variance_warm_batch(h_scratch, c, w_prev, w_out, n,
                                volfi_annulus_broadrange::WARM_HH3_FAST_STEPS);
}
// Single-node tick.  Recompute h, fast-warm; if the move left the 2-step basin
// (the guard fires) re-invert cold, so the result is always trustworthy.  This is
// the n=1 streaming path (a node whose price ticked); it still refines from the
// prior, unlike a cold one-off which would re-route from scratch.
inline double implied_variance_book_tick(double logK, double logF, double c, double w_prev) {
    double h = std::fabs(logK - logF);
    bool ok; double w = br::warm_variance_fast(h, c, w_prev, &ok);
    return ok ? w : implied_variance_otm(h, c);
}


// ============================================================================
//  MIXED-h grid batch -- BOTH h and c vary.  Only the CENTRAL subset is
//  vectorized: h in [H_ATM_HI,H_BOX], cwing(h)<=c<=c2(h), and a real main-table
//  cell (per-band affine xh -> per-lane-xh Clenshaw twin).  WING/LEFT/RIGHT and
//  every analytic edge drop to the scalar fallback pass (bit-identical: same
//  shared kernels + noinline scalar entry).  The seam prices cwing(h),c2(h) are
//  the single-source route gate, matching the scalar entry exactly.
// ============================================================================
#ifndef VOLFI_GRID_TILE
#define VOLFI_GRID_TILE 8192
#endif

namespace detail {
static constexpr int GRID_NBUCK = NCELLS;   // CENTRAL main-table cells only

// Grid CENTRAL gate (single source of truth for the table pass AND fallback pass):
// returns the global main-table cell for a quote routed to CENTRAL with a real
// cell hit; -1 otherwise (WING / LEFT / RIGHT / edge -> scalar fallback).  Also
// writes the band on a hit.  Mirrors the scalar routing bit-for-bit.
inline int grid_central_cell(double hh, double cc, uint64_t bc, int& band_out) {
    if (!(cc > 0.0) || cc >= 1.0) return -1;
    if (hh < H_ATM_HI || hh > H_BOX) return -1;         // LEFT band / RIGHT-only band
    if (cc < br::cwing_price(hh)) return -1;            // WING
    if (cc > br::c2_price(hh))    return -1;            // RIGHT (v>2)
    uint64_t bh = bits_of(hh);
    long bi = (long)(bh >> H_SHIFT) - (long)OFF_H;
    if (bi < 0) bi = 0; else if (bi > NB - 1) bi = NB - 1;
    band_out = (int)bi;
    return main_cell_of((int)bi, bc);                  // CENTRAL cell, or -1 (analytic edge)
}

// Grid ENDPOINT route for a NON-central quote (precondition: grid_central_cell<0):
// 0 = scalar (analytic ATM-deep edge / degenerate / h==0), 1 = LEFT, 2 = RIGHT,
// 3 = WING.  Mirrors the scalar implied_variance_otm routing bit-for-bit so the
// vectorized buckets reproduce the scalar entry exactly.
inline int grid_endpoint_route(double hh, double cc) {
    if (!(cc > 0.0) || cc >= 1.0) return 0;
    if (!(hh > 0.0)) return 0;                                  // h==0 ATM line, h<0/NaN guard -> scalar
    if (cc < br::cwing_price(hh)) return 3;                     // WING
    if (hh < H_ATM_HI) {                                        // region 1 (LEFT band)
        return (cc <= br::ctl_seam(hh)) ? 1 : 2;                // frozen seam poly (== scalar ct_left)
    }
    if (hh <= H_BOX) return (cc > br::c2_price(hh)) ? 2 : 0;    // region 0: v>2 RIGHT else edge
    return 2;                                                   // region 2 (h>H_BOX): RIGHT
}

__attribute__((noinline)) inline void grid_table_pass(
        const double* h, const double* c, double* w_out, int n) {
    constexpr int TILE = VOLFI_GRID_TILE;
    int    gcell[TILE];      // global main-table cell id, -1 = fallback
    double xhv  [TILE];      // per-quote (per-band affine) xh
    double xlv  [TILE];      // log2m -> xl (in place)
    double h2v  [TILE];
    int    order[TILE];

    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;

        log2approx_block(c + base, xlv, m);            // vectorized index-pass log2

        int cnt[GRID_NBUCK];
        for (int t = 0; t < GRID_NBUCK; ++t) cnt[t] = 0;

        for (int i = 0; i < m; ++i) {
            const double hh = h[base + i], cc = c[base + i];
            uint64_t bc = bits_of(cc);
            int band;
            int cell = grid_central_cell(hh, cc, bc, band);
            if (cell < 0) { gcell[i] = -1; continue; }
            double xh = std::fma(hh, HSCALE[band], HBIAS[band]);
            if (xh < -1.0) xh = -1.0; else if (xh > 1.0) xh = 1.0;
            gcell[i] = cell;
            xhv[i]   = xh;
            h2v[i]   = hh * hh;
            xlv[i]   = std::fma(CELL_LSCALE[cell], xlv[i], CELL_LBIAS[cell]);
            ++cnt[cell];
        }

        int off[GRID_NBUCK], pos[GRID_NBUCK], run = 0;
        for (int t = 0; t < GRID_NBUCK; ++t) { off[t] = run; run += cnt[t]; }
        for (int t = 0; t < GRID_NBUCK; ++t) pos[t] = off[t];
        for (int i = 0; i < m; ++i) { int ci = gcell[i]; if (ci >= 0) order[pos[ci]++] = i; }

        for (int t = 0; t < GRID_NBUCK; ++t) {
            const int b0 = off[t], nb = cnt[t];
            if (nb == 0) continue;
            int e = 0;
#if defined(VA_SIMD512)
            const __m512d vtwo = _mm512_set1_pd(2.0);
            for (; e + 8 <= nb; e += 8) {
                double xhb[8], xlb[8], h2b[8]; const int* od = order + b0 + e;
                for (int l = 0; l < 8; ++l) { int idx = od[l]; xhb[l] = xhv[idx]; xlb[l] = xlv[idx]; h2b[l] = h2v[idx]; }
                __m512d W = clenshaw2_avx512_vxh(CELL_DP[t], CELL_DL[t], &COEFFS[CELL_OFF[t]],
                                                 _mm512_loadu_pd(xhb), _mm512_loadu_pd(xlb));
                __m512d w = _mm512_div_pd(_mm512_loadu_pd(h2b), _mm512_mul_pd(vtwo, W));
                double wb[8]; _mm512_storeu_pd(wb, w);
                for (int l = 0; l < 8; ++l) w_out[base + od[l]] = wb[l];
            }
#elif defined(VA_SIMD256)
            const __m256d vtwo = _mm256_set1_pd(2.0);
            for (; e + 4 <= nb; e += 4) {
                double xhb[4], xlb[4], h2b[4]; const int* od = order + b0 + e;
                for (int l = 0; l < 4; ++l) { int idx = od[l]; xhb[l] = xhv[idx]; xlb[l] = xlv[idx]; h2b[l] = h2v[idx]; }
                __m256d W = clenshaw2_avx2_vxh(CELL_DP[t], CELL_DL[t], &COEFFS[CELL_OFF[t]],
                                               _mm256_loadu_pd(xhb), _mm256_loadu_pd(xlb));
                __m256d w = _mm256_div_pd(_mm256_loadu_pd(h2b), _mm256_mul_pd(vtwo, W));
                double wb[4]; _mm256_storeu_pd(wb, w);
                for (int l = 0; l < 4; ++l) w_out[base + od[l]] = wb[l];
            }
#endif
            for (; e < nb; ++e) {
                int idx = order[b0 + e];
                double W = clenshaw2_cell(t, xhv[idx], xlv[idx]);
                w_out[base + idx] = h2v[idx] / (2.0 * W);
            }
        }
    }
}

// Grid ENDPOINT pass: the CENTRAL vector core (grid_table_pass) leaves every
// non-central quote unwritten; this pass buckets them into LEFT and RIGHT and runs
// each through the per-lane-h vector kernels (per-h expm1/exp libm hoisted into the
// index pass, exactly as in kernel_bit_test), with WING / analytic edges / h==0
// dropping to the scalar entry.  Bucket-contiguous arrays feed a straight 8-wide
// load; the tail (and every non-512 build) uses the SAME br::left/right scalar
// kernels -> batch == scalar entry per quote (invariant b).
__attribute__((noinline)) inline void grid_fallback_pass(
        const double* h, const double* c, double* w_out, int n) {
    constexpr int TILE = VOLFI_GRID_TILE;
    static thread_local double lc_[TILE], le_[TILE], lh_[TILE], lh2_[TILE], lh4_[TILE], lxt_[TILE];
    static thread_local double rc_[TILE], reh_[TILE], rehp_[TILE], rh_[TILE], rh2_[TILE];
    static thread_local int    lo_[TILE], ro_[TILE];
    static thread_local double wc_[TILE], wh_[TILE];
    static thread_local int    wo_[TILE], wreg_[TILE];
    static thread_local double wc2_[TILE], wh2_[TILE];
    static thread_local int    wo2_[TILE];
    const double invTmax = 1.0 / volfi_annulus_broadrange::LEFT_T_MAX;

    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;
        int lc = 0, rc = 0, wcnt = 0;
        for (int i = 0; i < m; ++i) {
            const double hh = h[base + i], cc = c[base + i];
            int band;
            if (grid_central_cell(hh, cc, bits_of(cc), band) >= 0) continue;   // CENTRAL: done
            int rt = grid_endpoint_route(hh, cc);
            if (rt == 1) {                                                      // LEFT
                double h2 = hh * hh;
                lc_[lc] = cc; le_[lc] = br::expm1_small(hh); lh_[lc] = hh;
                lh2_[lc] = h2; lh4_[lc] = h2 * h2;
                lxt_[lc] = std::fma(2.0 * h2, invTmax, -1.0);
                lo_[lc] = base + i; ++lc;
            } else if (rt == 2) {                                              // RIGHT
                rc_[rc] = cc; reh_[rc] = std::exp(-0.5 * hh); rehp_[rc] = std::exp(hh);
                rh_[rc] = hh; rh2_[rc] = hh * hh;
                ro_[rc] = base + i; ++rc;
            } else if (rt == 3) {                                              // WING
                if (!(hh * hh * (1.0 / 16.0) <= wing_detail::WING_B_HI)) {     // beta wedge:
                    w_out[base + i] = wing_variance(hh, cc);                   // scalar rescue,
                    continue;                                                  // same fn as scalar path
                }
                double Ltq = wing_Lt(hh, cc);                                  // same selector as scalar
                wc_[wcnt] = cc; wh_[wcnt] = hh;
                wreg_[wcnt] = (Ltq > wing_detail::WING_LT_BS) ? 2
                            : ((Ltq <= wing_detail::WING_LT_AB) ? 0 : 1);
                wo_[wcnt] = base + i; ++wcnt;
            } else {                                                          // edge / h==0
                w_out[base + i] = scalar_fallback(hh, cc);
            }
        }
        int e = 0;
#if defined(VA_SIMD512)
        for (; e + 8 <= lc; e += 8) {
            __m512d w = left_variance_avx512(_mm512_loadu_pd(lc_ + e), _mm512_loadu_pd(le_ + e),
                        _mm512_loadu_pd(lh_ + e), _mm512_loadu_pd(lh2_ + e),
                        _mm512_loadu_pd(lh4_ + e), _mm512_loadu_pd(lxt_ + e));
            double wb[8]; _mm512_storeu_pd(wb, w);
            for (int l = 0; l < 8; ++l) w_out[lo_[e + l]] = wb[l];
        }
#elif defined(VA_SIMD256)
        for (; e + 4 <= lc; e += 4) {
            __m256d w = left_variance_avx2(_mm256_loadu_pd(lc_ + e), _mm256_loadu_pd(le_ + e),
                        _mm256_loadu_pd(lh_ + e), _mm256_loadu_pd(lh2_ + e),
                        _mm256_loadu_pd(lh4_ + e), _mm256_loadu_pd(lxt_ + e));
            double wb[4]; _mm256_storeu_pd(wb, w);
            for (int l = 0; l < 4; ++l) w_out[lo_[e + l]] = wb[l];
        }
#endif
        for (; e < lc; ++e) w_out[lo_[e]] = br::left_variance(lh_[e], lc_[e]);
        e = 0;
#if defined(VA_SIMD512)
        for (; e + 8 <= rc; e += 8) {
            __m512d w = right_variance_avx512(_mm512_loadu_pd(rc_ + e), _mm512_loadu_pd(reh_ + e),
                        _mm512_loadu_pd(rehp_ + e), _mm512_loadu_pd(rh_ + e), _mm512_loadu_pd(rh2_ + e));
            double wb[8]; _mm512_storeu_pd(wb, w);
            for (int l = 0; l < 8; ++l) w_out[ro_[e + l]] = wb[l];
        }
#elif defined(VA_SIMD256)
        for (; e + 4 <= rc; e += 4) {
            __m256d w = right_variance_avx2(_mm256_loadu_pd(rc_ + e), _mm256_loadu_pd(reh_ + e),
                        _mm256_loadu_pd(rehp_ + e), _mm256_loadu_pd(rh_ + e), _mm256_loadu_pd(rh2_ + e));
            double wb[4]; _mm256_storeu_pd(wb, w);
            for (int l = 0; l < 4; ++l) w_out[ro_[e + l]] = wb[l];
        }
#endif
        for (; e < rc; ++e) w_out[ro_[e]] = br::right_variance(rh_[e], rc_[e]);
        // WING: three regime sub-buckets (piece A / piece B / series), each lane-uniform.
        // The scalar tail (and every non-512 build) calls the SAME wing_variance, whose
        // regime choice comes from the SAME wing_Lt -> value-identical per quote.
#if defined(VA_SIMD512)
        for (int reg = 0; reg < 3; ++reg) {
            int m2 = 0;
            for (int k = 0; k < wcnt; ++k)
                if (wreg_[k] == reg) { wc2_[m2] = wc_[k]; wh2_[m2] = wh_[k]; wo2_[m2] = wo_[k]; ++m2; }
            int e2 = 0;
            for (; e2 + 8 <= m2; e2 += 8) {
                __m512d w = wing_variance_avx512(_mm512_loadu_pd(wh2_ + e2),
                                                 _mm512_loadu_pd(wc2_ + e2), reg);
                double wb[8]; _mm512_storeu_pd(wb, w);
                for (int l = 0; l < 8; ++l) w_out[wo2_[e2 + l]] = wb[l];
            }
            for (; e2 < m2; ++e2) w_out[wo2_[e2]] = wing_variance(wh2_[e2], wc2_[e2]);
        }
#elif defined(VA_SIMD256)
        for (int reg = 0; reg < 3; ++reg) {
            int m2 = 0;
            for (int k = 0; k < wcnt; ++k)
                if (wreg_[k] == reg) { wc2_[m2] = wc_[k]; wh2_[m2] = wh_[k]; wo2_[m2] = wo_[k]; ++m2; }
            int e2 = 0;
            for (; e2 + 4 <= m2; e2 += 4) {
                __m256d w = wing_variance_avx2(_mm256_loadu_pd(wh2_ + e2),
                                               _mm256_loadu_pd(wc2_ + e2), reg);
                double wb[4]; _mm256_storeu_pd(wb, w);
                for (int l = 0; l < 4; ++l) w_out[wo2_[e2 + l]] = wb[l];
            }
            for (; e2 < m2; ++e2) w_out[wo2_[e2]] = wing_variance(wh2_[e2], wc2_[e2]);
        }
#else
        for (int k = 0; k < wcnt; ++k) w_out[wo_[k]] = wing_variance(wh_[k], wc_[k]);
#endif
    }
}
// ---- vectorized frozen routing seams (bit-identical to scalar br:: seams) ----
// Only the LEFT route needs vectorizing: cwing(h) (WING gate) and ctl(h)
// (LEFT/RIGHT gate).  c2(h) (central/right gate) stays scalar in the deferred
// drain.  All use the shared clenshaw/exp_neg kernels, so the batch route mask
// equals grid_endpoint_route()'s LEFT decision quote-for-quote.
namespace Kv2 = volfi_annulus_broadrange;
#if defined(VA_SIMD512)
inline __m512d cwing_price_avx512(__m512d h){
    return _mm512_mul_pd(h, exp_neg_avx512(clenshaw1_avx512(Kv2::CW_COEFFS,23,Kv2::CW_A,Kv2::CW_B,h)));
}
inline __m512d ctl_seam_avx512(__m512d h){
    return clenshaw1_avx512(Kv2::CTL_SEAM_C,9,Kv2::CTL_SEAM_A,Kv2::CTL_SEAM_B,h);
}
inline __m512d expm1_small_avx512(__m512d h){
    return _mm512_mul_pd(h, clenshaw1_avx512(Kv2::EXPM1G_C,11,Kv2::EXPM1G_A,Kv2::EXPM1G_B,h));
}
#endif
#if defined(VA_SIMD256)
inline __m256d cwing_price_avx2(__m256d h){
    return _mm256_mul_pd(h, exp_neg_avx2(clenshaw1_avx2(Kv2::CW_COEFFS,23,Kv2::CW_A,Kv2::CW_B,h)));
}
inline __m256d ctl_seam_avx2(__m256d h){
    return clenshaw1_avx2(Kv2::CTL_SEAM_C,9,Kv2::CTL_SEAM_A,Kv2::CTL_SEAM_B,h);
}
inline __m256d expm1_small_avx2(__m256d h){
    return _mm256_mul_pd(h, clenshaw1_avx2(Kv2::EXPM1G_C,11,Kv2::EXPM1G_A,Kv2::EXPM1G_B,h));
}
#endif

// Speculative-LEFT streaming grid batch (the empirical live path: ~90% LEFT).
// Streams (h,c) 8/4-wide; routes via the frozen seam polynomials (no libm, no
// runtime Black eval); runs the LEFT kernel UNCONDITIONALLY on every lane and
// stores the LEFT lanes in place (no sort, no scatter for the majority); the
// non-LEFT minority is compressed into a deferred buffer that the existing
// table+fallback machinery drains and scatters back.  LEFT lanes are bit-identical
// to the scalar entry (left_variance on unclamped h<H_ATM_HI); deferred lanes are
// bit-identical (grid_table_pass/grid_fallback_pass).  h is clamped from ABOVE to
// H_ATM_HI before the speculative kernel: this is the identity for every stored
// LEFT lane (h<0.3) and merely keeps the discarded off-chart lanes out of the
// expm1 poly's tail.  NO lower clamp -- the LEFT route admits any h>0 (down to the
// smallest subnormal), so raising a tiny positive h would alter a stored lane and
// break bit-identity (a max(1e-300,.) floor did exactly that; removed 2026-07-13).
// Discarded lanes may go inf/NaN; the kernel is purely lane-wise so masked store
// and the deferred scatter ignore them safely.
__attribute__((noinline)) inline void speculative_grid_batch(
        const double* h, const double* c, double* w_out, int n) {
    constexpr int TILE = VOLFI_GRID_TILE;
    static thread_local double dh_[TILE], dc_[TILE], dw_[TILE];
    static thread_local int    didx_[TILE];
    const double invTmax = 1.0 / volfi_annulus_broadrange::LEFT_T_MAX;
    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;
        int ndef = 0, i = 0;
#if defined(VA_SIMD512)
        {
        const __m512d z=_mm512_setzero_pd(), one=_mm512_set1_pd(1.0);
        const __m512d hatm=_mm512_set1_pd(H_ATM_HI);
        const __m512d vinvT=_mm512_set1_pd(invTmax), two=_mm512_set1_pd(2.0), negone=_mm512_set1_pd(-1.0);
        for (; i + 8 <= m; i += 8) {
            __m512d vh=_mm512_loadu_pd(h+base+i), vc=_mm512_loadu_pd(c+base+i);
            __m512d vcw=cwing_price_avx512(vh), vctl=ctl_seam_avx512(vh);
            __mmask8 mv = _mm512_cmp_pd_mask(vc,z,_CMP_GT_OQ) & _mm512_cmp_pd_mask(vc,one,_CMP_LT_OQ)
                        & _mm512_cmp_pd_mask(vh,z,_CMP_GT_OQ);
            __mmask8 ml = mv & _mm512_cmp_pd_mask(vc,vcw,_CMP_GE_OQ)
                        & _mm512_cmp_pd_mask(vh,hatm,_CMP_LT_OQ)
                        & _mm512_cmp_pd_mask(vc,vctl,_CMP_LE_OQ);
            __m512d vhc=_mm512_min_pd(vh,hatm);   // upper clamp only; identity for stored LEFT lanes (h<0.3)
            __m512d vE=expm1_small_avx512(vhc), vh2=_mm512_mul_pd(vhc,vhc), vh4=_mm512_mul_pd(vh2,vh2);
            __m512d vxt=_mm512_fmadd_pd(_mm512_mul_pd(two,vh2),vinvT,negone);
            __m512d vw=left_variance_avx512(vc,vE,vhc,vh2,vh4,vxt);
            _mm512_mask_storeu_pd(w_out+base+i, ml, vw);
            unsigned dm=(unsigned)(uint8_t)(~ml);
            if (dm){ double hs[8],cs[8]; _mm512_storeu_pd(hs,vh); _mm512_storeu_pd(cs,vc);
                while(dm){ int l=__builtin_ctz(dm); dm&=dm-1;
                    dh_[ndef]=hs[l]; dc_[ndef]=cs[l]; didx_[ndef]=base+i+l; ++ndef; } }
        }
        }
#elif defined(VA_SIMD256)
        {
        const __m256d z=_mm256_setzero_pd(), one=_mm256_set1_pd(1.0);
        const __m256d hatm=_mm256_set1_pd(H_ATM_HI);
        const __m256d vinvT=_mm256_set1_pd(invTmax), two=_mm256_set1_pd(2.0), negone=_mm256_set1_pd(-1.0);
        for (; i + 4 <= m; i += 4) {
            __m256d vh=_mm256_loadu_pd(h+base+i), vc=_mm256_loadu_pd(c+base+i);
            __m256d vcw=cwing_price_avx2(vh), vctl=ctl_seam_avx2(vh);
            __m256d mv=_mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(vc,z,_CMP_GT_OQ),_mm256_cmp_pd(vc,one,_CMP_LT_OQ)),
                                     _mm256_cmp_pd(vh,z,_CMP_GT_OQ));
            __m256d ml=_mm256_and_pd(_mm256_and_pd(mv,_mm256_cmp_pd(vc,vcw,_CMP_GE_OQ)),
                                     _mm256_and_pd(_mm256_cmp_pd(vh,hatm,_CMP_LT_OQ),_mm256_cmp_pd(vc,vctl,_CMP_LE_OQ)));
            __m256d vhc=_mm256_min_pd(vh,hatm);   // upper clamp only; identity for stored LEFT lanes (h<0.3)
            __m256d vE=expm1_small_avx2(vhc), vh2=_mm256_mul_pd(vhc,vhc), vh4=_mm256_mul_pd(vh2,vh2);
            __m256d vxt=_mm256_fmadd_pd(_mm256_mul_pd(two,vh2),vinvT,negone);
            __m256d vw=left_variance_avx2(vc,vE,vhc,vh2,vh4,vxt);
            _mm256_maskstore_pd(w_out+base+i, _mm256_castpd_si256(ml), vw);
            unsigned dm=(~(unsigned)_mm256_movemask_pd(ml)) & 0xFu;
            if (dm){ double hs[4],cs[4]; _mm256_storeu_pd(hs,vh); _mm256_storeu_pd(cs,vc);
                while(dm){ int l=__builtin_ctz(dm); dm&=dm-1;
                    dh_[ndef]=hs[l]; dc_[ndef]=cs[l]; didx_[ndef]=base+i+l; ++ndef; } }
        }
        }
#endif
        for (; i < m; ++i) {                                   // scalar tail (and non-SIMD builds)
            double hh=h[base+i], cc=c[base+i];
            if (grid_endpoint_route(hh,cc)==1) w_out[base+i]=br::left_variance(hh,cc);
            else { dh_[ndef]=hh; dc_[ndef]=cc; didx_[ndef]=base+i; ++ndef; }
        }
        if (ndef){ grid_table_pass(dh_,dc_,dw_,ndef); grid_fallback_pass(dh_,dc_,dw_,ndef);
            for (int j=0;j<ndef;++j) w_out[didx_[j]]=dw_[j]; }
    }
}
} // namespace detail

// Mixed-h grid batch entry.  w_out must NOT alias h or c (results are stored
// while later inputs are still being read; this has always been the contract).
//
// ADAPTIVE DISPATCH: two internally-verified drivers produce BIT-IDENTICAL
// results (each equals the scalar entry per quote), so the choice between them
// affects speed only, never values.  The LEFT-speculative streaming driver wins
// when the feed is dominated by the small-moneyness chart (a real option book:
// ~90% LEFT), but taxes minority-chart feeds (it evaluates LEFT speculatively on
// every lane and defers the rest); the classic two-pass driver is better on
// central/wing-dominated feeds such as uniform stress grids.  A strided sample
// of the h array (<=2048 probes, h < H_ATM_HI as the LEFT-band proxy) picks the
// driver; the sample is deterministic in the input, so repeated calls on the
// same feed always take the same path.
inline void implied_variance_grid_batch(const double* h, const double* c,
                                         double* w_out, int n) {
#if defined(VA_SIMD512) || defined(VA_SIMD256)
    if (n > 0) {
        const int ns = (n < 2048) ? n : 2048;
        const int stride = n / ns;
        int nl = 0;
        for (int k = 0; k < ns; ++k) nl += (h[(long)k * stride] < H_ATM_HI);
        if (2 * nl >= ns) {                                   // >=50% LEFT-band
            detail::speculative_grid_batch(h, c, w_out, n);   // LEFT-speculative streaming
            return;
        }
    }
#endif
    detail::grid_table_pass(h, c, w_out, n);                  // classic two-pass
    detail::grid_fallback_pass(h, c, w_out, n);
}

// ---- Optional PERMUTED-OUTPUT fast path (item 5) -------------------------
// When the caller accepts results in sorted (bucket) order instead of input
// order, the scattered store back to w_out[orig] is replaced by a CONTIGUOUS
// store: bucket t's results land in w_perm[cursor..cursor+nb) and perm[] records
// each result's ORIGINAL input index.  Fallback quotes are appended.  Returns n.
// The default implied_variance_grid_batch keeps input order.
//   Reconstruct input order (if ever needed) with:  w_out[perm[j]] = w_perm[j].
inline int implied_variance_grid_batch_permuted(const double* h, const double* c,
                                                double* w_perm, int* perm, int n) {
    constexpr int TILE = VOLFI_GRID_TILE;
    static thread_local int    gcell[TILE];
    static thread_local double xhv[TILE], xlv[TILE], h2v[TILE];
    static thread_local int    order[TILE];
    int cur = 0;
    for (int base = 0; base < n; base += TILE) {
        const int m = (n - base < TILE) ? (n - base) : TILE;
        detail::log2approx_block(c + base, xlv, m);
        int cnt[NCELLS];
        for (int t = 0; t < NCELLS; ++t) cnt[t] = 0;
        for (int i = 0; i < m; ++i) {
            const double hh = h[base + i], cc = c[base + i];
            int band;
            int cell = detail::grid_central_cell(hh, cc, detail::bits_of(cc), band);
            if (cell < 0) { gcell[i] = -1; continue; }
            double xh = std::fma(hh, HSCALE[band], HBIAS[band]);
            if (xh < -1.0) xh = -1.0; else if (xh > 1.0) xh = 1.0;
            gcell[i] = cell; xhv[i] = xh; h2v[i] = hh * hh;
            xlv[i] = std::fma(CELL_LSCALE[cell], xlv[i], CELL_LBIAS[cell]);
            ++cnt[cell];
        }
        int off[NCELLS], pos[NCELLS], run = 0;
        for (int t = 0; t < NCELLS; ++t) { off[t] = run; run += cnt[t]; }
        for (int t = 0; t < NCELLS; ++t) pos[t] = off[t];
        for (int i = 0; i < m; ++i) { int ci = gcell[i]; if (ci >= 0) order[pos[ci]++] = i; }
        for (int t = 0; t < NCELLS; ++t) {
            const int b0 = off[t], nb = cnt[t];
            if (nb == 0) continue;
            int e = 0;
#if defined(VA_SIMD512)
            const __m512d vtwo = _mm512_set1_pd(2.0);
            for (; e + 8 <= nb; e += 8) {
                double xhb[8], xlb[8], h2b[8]; const int* od = order + b0 + e;
                for (int l = 0; l < 8; ++l) { int idx = od[l]; xhb[l] = xhv[idx]; xlb[l] = xlv[idx]; h2b[l] = h2v[idx]; }
                __m512d W = detail::clenshaw2_avx512_vxh(CELL_DP[t], CELL_DL[t], &COEFFS[CELL_OFF[t]],
                                                         _mm512_loadu_pd(xhb), _mm512_loadu_pd(xlb));
                __m512d w = _mm512_div_pd(_mm512_loadu_pd(h2b), _mm512_mul_pd(vtwo, W));
                _mm512_storeu_pd(w_perm + cur, w);
                for (int l = 0; l < 8; ++l) perm[cur + l] = base + od[l];
                cur += 8;
            }
#elif defined(VA_SIMD256)
            const __m256d vtwo = _mm256_set1_pd(2.0);
            for (; e + 4 <= nb; e += 4) {
                double xhb[4], xlb[4], h2b[4]; const int* od = order + b0 + e;
                for (int l = 0; l < 4; ++l) { int idx = od[l]; xhb[l] = xhv[idx]; xlb[l] = xlv[idx]; h2b[l] = h2v[idx]; }
                __m256d W = detail::clenshaw2_avx2_vxh(CELL_DP[t], CELL_DL[t], &COEFFS[CELL_OFF[t]],
                                                       _mm256_loadu_pd(xhb), _mm256_loadu_pd(xlb));
                __m256d w = _mm256_div_pd(_mm256_loadu_pd(h2b), _mm256_mul_pd(vtwo, W));
                _mm256_storeu_pd(w_perm + cur, w);
                for (int l = 0; l < 4; ++l) perm[cur + l] = base + od[l];
                cur += 4;
            }
#endif
            for (; e < nb; ++e) {
                int idx = order[b0 + e];
                double W = clenshaw2_cell(t, xhv[idx], xlv[idx]);
                w_perm[cur] = h2v[idx] / (2.0 * W); perm[cur] = base + idx; ++cur;
            }
        }
        // Endpoint (non-central) buckets: LEFT/RIGHT vectorized, WING/edge scalar.
        // Results are appended to w_perm in bucket order (perm records the origin).
        static thread_local double lc_[TILE], le_[TILE], lh_[TILE], lh2_[TILE], lh4_[TILE], lxt_[TILE];
        static thread_local double rc_[TILE], reh_[TILE], rehp_[TILE], rh_[TILE], rh2_[TILE];
        static thread_local int    lo_[TILE], ro_[TILE];
        const double invTmax = 1.0 / volfi_annulus_broadrange::LEFT_T_MAX;
        int lc = 0, rc = 0;
        for (int i = 0; i < m; ++i) {
            if (gcell[i] >= 0) continue;
            const double hh = h[base + i], cc = c[base + i];
            int rt = detail::grid_endpoint_route(hh, cc);
            if (rt == 1) {
                double h2 = hh * hh;
                lc_[lc] = cc; le_[lc] = br::expm1_small(hh); lh_[lc] = hh; lh2_[lc] = h2;
                lh4_[lc] = h2 * h2; lxt_[lc] = std::fma(2.0 * h2, invTmax, -1.0);
                lo_[lc] = base + i; ++lc;
            } else if (rt == 2) {
                rc_[rc] = cc; reh_[rc] = std::exp(-0.5 * hh); rehp_[rc] = std::exp(hh);
                rh_[rc] = hh; rh2_[rc] = hh * hh; ro_[rc] = base + i; ++rc;
            } else {
                w_perm[cur] = detail::scalar_fallback(hh, cc); perm[cur] = base + i; ++cur;
            }
        }
        int e = 0;
#if defined(VA_SIMD512)
        for (; e + 8 <= lc; e += 8) {
            __m512d w = detail::left_variance_avx512(_mm512_loadu_pd(lc_ + e), _mm512_loadu_pd(le_ + e),
                        _mm512_loadu_pd(lh_ + e), _mm512_loadu_pd(lh2_ + e),
                        _mm512_loadu_pd(lh4_ + e), _mm512_loadu_pd(lxt_ + e));
            _mm512_storeu_pd(w_perm + cur, w);
            for (int l = 0; l < 8; ++l) perm[cur + l] = lo_[e + l];
            cur += 8;
        }
#elif defined(VA_SIMD256)
        for (; e + 4 <= lc; e += 4) {
            __m256d w = detail::left_variance_avx2(_mm256_loadu_pd(lc_ + e), _mm256_loadu_pd(le_ + e),
                        _mm256_loadu_pd(lh_ + e), _mm256_loadu_pd(lh2_ + e),
                        _mm256_loadu_pd(lh4_ + e), _mm256_loadu_pd(lxt_ + e));
            _mm256_storeu_pd(w_perm + cur, w);
            for (int l = 0; l < 4; ++l) perm[cur + l] = lo_[e + l];
            cur += 4;
        }
#endif
        for (; e < lc; ++e) { w_perm[cur] = br::left_variance(lh_[e], lc_[e]); perm[cur] = lo_[e]; ++cur; }
        e = 0;
#if defined(VA_SIMD512)
        for (; e + 8 <= rc; e += 8) {
            __m512d w = detail::right_variance_avx512(_mm512_loadu_pd(rc_ + e), _mm512_loadu_pd(reh_ + e),
                        _mm512_loadu_pd(rehp_ + e), _mm512_loadu_pd(rh_ + e), _mm512_loadu_pd(rh2_ + e));
            _mm512_storeu_pd(w_perm + cur, w);
            for (int l = 0; l < 8; ++l) perm[cur + l] = ro_[e + l];
            cur += 8;
        }
#elif defined(VA_SIMD256)
        for (; e + 4 <= rc; e += 4) {
            __m256d w = detail::right_variance_avx2(_mm256_loadu_pd(rc_ + e), _mm256_loadu_pd(reh_ + e),
                        _mm256_loadu_pd(rehp_ + e), _mm256_loadu_pd(rh_ + e), _mm256_loadu_pd(rh2_ + e));
            _mm256_storeu_pd(w_perm + cur, w);
            for (int l = 0; l < 4; ++l) perm[cur + l] = ro_[e + l];
            cur += 4;
        }
#endif
        for (; e < rc; ++e) { w_perm[cur] = br::right_variance(rh_[e], rc_[e]); perm[cur] = ro_[e]; ++cur; }
    }
    return cur;
}


// ============================================================================
//  Convenience / reflection wrappers.
// ============================================================================
inline double implied_volatility_otm(const context& q, double c, double t) {
    return std::sqrt(implied_variance_otm(q, c) / t);
}
inline double implied_volatility_otm(double h, double c, double t) {
    context q(h);
    return std::sqrt(implied_variance_otm(q, c) / t);
}
inline double implied_variance_call_normalised(double k, double c) {
    if (k > 0) return implied_variance_otm(k, c);
    if (k < 0) return implied_variance_otm(-k, 1 + std::exp(-k) * (c - 1));
    return implied_variance_otm(0.0, c);
}

} // namespace volfi_annulus

#endif // VOLFI_ANNULUS_HPP
