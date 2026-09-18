// volfi_gpu_book.cu -- FULL-BOOK GPU port of the volfi-annulus inverter (v2).
// =============================================================================
// Extends the v1 FAR-only kernel (volfi_gpu.cu) to all four charts --
// NEAR / FAR / UPPER / WING -- with the BUCKET-ORDER COALESCED driver the
// CPU phase breakdown motivated: quotes are partitioned by chart ONCE on the
// host (in a live book this order is persistent), each chart runs as its own
// kernel over a CONTIGUOUS slice, and there is NO per-snapshot sort, gather or
// scatter on the device.  This is the "GPU-optimal variant" of the same
// algorithm: identical math, identical tables, different execution schedule.
//
// SAME ALGORITHM, OP-FOR-OP: every device function below mirrors the scalar
// CPU path in ../volfi_annulus.hpp line by line -- same explicit fma order,
// same frozen polynomials, same fixed iteration counts (UPPER: 3 Householder-3,
// WING: 6 Newton).  The only per-quote host precomputes are the SAME per-h
// libm scalars the CPU batch drivers hoist to their index pass (exp(-h/2),
// exp(h) for UPPER) plus the FAR band/xh context -- no math is moved.
// Correctness gate: every GPU result is ULP-compared against the UNTOUCHED CPU
// reference (../volfi_annulus_all.hpp compiled as host code in this same TU);
// mismatches(>1ulp) MUST be 0 -- see "What mismatches would mean" in README.md.
//
// Build (Brev / any CUDA 12.x box; H100 = sm_90, A100 = sm_80):
//   nvcc -O3 -arch=sm_90 -std=c++17 --fmad=false volfi_gpu_book.cu -o volfi_gpu_book
// --fmad=false mirrors the CPU's -ffp-contract=off: only the SOURCE fma()s fuse,
// so codegen cannot contract differently from the host reference.
//
// Run (market_feed.csv from ../bench_run/ in cwd):
//   ./volfi_gpu_book                 # defaults: tile feed to >=1e6, 100 iters
//   ./volfi_gpu_book 4000000 200     # optional: n_target n_iters
//
// Exclusions (reported, never silent): the scalar-only analytic fallbacks stay
// CPU-side exactly as in v1 -- the beta>16.4 rescue wedge (h>16.2), FAR
// cells outside the table (analytic ATM edge / sub-wing sliver), h==0.  On the
// SPX market feed these are ~0 quotes.
// =============================================================================

#ifndef VGB_HOST_CHECK
#define VGB_HOST_CHECK 0
#endif
#if !VGB_HOST_CHECK
#include <cuda_runtime.h>
#else            // host-only build: strip the qualifiers, so the device mirrors run on the CPU against the reference
#define __device__
#define __constant__
#define __host__
#define __global__
#include <cstring>
static inline long long __double_as_longlong(double x) { long long u; std::memcpy(&u, &x, 8); return u; }
static inline double __longlong_as_double(long long u) { double x; std::memcpy(&x, &u, 8); return x; }
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>

// ---- device tables + constants (namespace volfi_annulus_gpu, mechanical copies)
#include "volfi_annulus_tables_cuda.cuh"
#include "volfi_constants_cuda.cuh"
#include "volfi_upper1_cuda.cuh"

// ---- host reference: the real library (namespace volfi_annulus) -------------
#include "volfi_annulus_all.hpp"          // build with -I../include/volfi

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

namespace g = volfi_annulus_gpu;

// =============================================================================
//  DEVICE MIRRORS of the shared scalar kernels (volfi_annulus.hpp, op-for-op).
// =============================================================================
__device__ static const double D_VA_LN2     = 0.69314718055994530942;
__device__ static const double D_VA_DBL_MIN = 2.2250738585072014e-308;
__device__ static const double D_VA_TWO54   = 1.8014398509481984e16;
__device__ static const double D_BR_IS2     = 0.70710678118654752440;
__device__ static const int    D_C_EXP_MASK = 0x7FF;

__device__ inline uint64_t d_bits(double x)   { return (uint64_t)__double_as_longlong(x); }
__device__ inline double   d_from_bits(uint64_t u) { return __longlong_as_double((long long)u); }
__device__ inline double   d_mant12(uint64_t bc) {
    uint64_t mb = (bc & 0x000FFFFFFFFFFFFFULL) | (uint64_t(g::C_EXP_BIAS) << 52);
    return d_from_bits(mb);
}

__device__ inline double d_log2approx(double m) {                    // == log2approx
    double t  = (m - 1.0) / (m + 1.0);
    double u  = t * t;
    double s  = fma(g::LOG2_S_A, u, g::LOG2_S_B);
    double s2 = s + s;
    double b1 = 0.0, b2 = 0.0;
    for (int k = 10; k >= 1; --k) { double b0 = fma(s2, b1, g::LOG2_GC[k]) - b2; b2 = b1; b1 = b0; }
    double gg = fma(s, b1, g::LOG2_GC[0]) - b2;
    return (g::LOG2_SCALE * t) * gg;
}
__device__ inline double d_full_log(double x) {                      // == full_log
    uint64_t bc = d_bits(x);
    int e = (int)((bc >> 52) & 0x7FFULL) - g::C_EXP_BIAS;
    return fma(D_VA_LN2, (double)e + d_log2approx(d_mant12(bc)), 0.0);
}
__device__ inline double d_full_log_sub(double x) {                  // == full_log_sub
    const bool sub = (x < D_VA_DBL_MIN);
    double xs = sub ? x * D_VA_TWO54 : x;
    uint64_t bc = d_bits(xs);
    int e = (int)((bc >> 52) & 0x7FFULL) - g::C_EXP_BIAS - (sub ? 54 : 0);
    return fma(D_VA_LN2, (double)e + d_log2approx(d_mant12(bc)), 0.0);
}
__device__ inline double d_sigma0_poly(double c) {                   // == sigma0_poly
    double t  = c * c;
    double x  = fma(g::ERFINV_TSC, t, g::ERFINV_TBIAS);
    double x2 = x + x;
    double b1 = 0.0, b2 = 0.0;
    for (int k = 17; k >= 1; --k) { double b0 = fma(x2, b1, g::ERFINV_GC[k]) - b2; b2 = b1; b1 = b0; }
    double gg = fma(x, b1, g::ERFINV_GC[0]) - b2;
    return (g::SIGMA0_SCALE * c) * gg;
}
__device__ inline double d_clenshaw1(const double* C, int n, double a, double b, double x) {
    double t  = fma(2.0, x, -(a + b)) / (b - a);                     // == br::clenshaw1
    double t2 = 2.0 * t;
    double d0 = 0.0, d1 = 0.0;
    for (int j = n - 1; j >= 1; --j) { double b0 = fma(t2, d0, C[j]) - d1; d1 = d0; d0 = b0; }
    return fma(t, d0, C[0]) - d1;
}
__device__ inline double d_clenshaw2(int dp, int dl, const double* C, double xh, double xl) {
    const int stride = dl + 1;                                       // == clenshaw2
    double Tl[17];
    Tl[0] = 1.0; Tl[1] = xl;
    const double xl2 = 2.0 * xl;
    for (int j = 2; j <= dl; ++j) Tl[j] = fma(xl2, Tl[j - 1], -Tl[j - 2]);
    double row[14];
    for (int i = 0; i <= dp; ++i) {
        const double* Ci = C + i * stride;
        double acc = Ci[0];
        for (int j = 1; j <= dl; ++j) acc = fma(Ci[j], Tl[j], acc);
        row[i] = acc;
    }
    double b1 = 0.0, b2 = 0.0;
    const double xh2 = 2.0 * xh;
    for (int i = dp; i >= 1; --i) { double b0 = fma(xh2, b1, row[i]) - b2; b2 = b1; b1 = b0; }
    return fma(xh, b1, row[0]) - b2;
}

// =============================================================================
//  NEAR chart (== br::near_variance chain: expm1_small, binv, V0/V2/V4, finisher)
// =============================================================================
__device__ inline double d_expm1_small(double h) {
    return h * d_clenshaw1(g::EXPM1G_C, 11, g::EXPM1G_A, g::EXPM1G_B, h);
}
__device__ inline double d_binv(double rho) {
    if (rho > g::BINV_RHO_A) {
        double u = g::BR_K / (rho + 0.5);
        double x = u * u;
        return u * d_clenshaw1(g::BINV_CA, 15, g::BINV_U2_A, g::BINV_U2_B, x);
    }
    double L  = -d_full_log(rho);
    double A2 = (rho > g::BINV_RHO_B) ? d_clenshaw1(g::BINV_CB1, 17, g::BINV_L_B1A, g::BINV_L_B1B, L)
              : (rho > g::BINV_RHO_C) ? d_clenshaw1(g::BINV_CB2, 25, g::BINV_L_B2A, g::BINV_L_B2B, L)
              :                         d_clenshaw1(g::BINV_CB3, 13, g::BINV_L_B3A, g::BINV_L_B3B, L);
    return sqrt(A2);
}
__device__ inline double d_V0_near(double s) { return d_sigma0_poly(g::BR_K * s); }
__device__ inline double d_V2_near(double s) { return d_clenshaw1(g::NEAR_V2_CHEB, g::NEAR_V2_CHEB_N, 0.0, g::NEAR_S_CHEB_MAX, s); }
__device__ inline double d_V4_near(double s) { return d_clenshaw1(g::NEAR_V4_CHEB, g::NEAR_V4_CHEB_N, 0.0, g::NEAR_S_CHEB_MAX, s); }

__device__ inline double d_near_variance(double h, double c) {
    double rho = c / d_expm1_small(h);
    double A   = d_binv(rho);
    double s   = h / A;
    double x   = d_V0_near(s);
    double h2  = h * h, h4 = h2 * h2;
    double xt  = fma(2.0 * h2, 1.0 / g::NEAR_T_MAX, -1.0);
    double xs  = fma(2.0 * s,  1.0 / g::NEAR_S_MAX, -1.0);
    double fin = d_clenshaw2(g::NEAR_FIN_DP, g::NEAR_FIN_DL, g::NEAR_FIN_COEFFS, xt, xs);
    double v   = fma(h4, d_V4_near(s), fma(h2, d_V2_near(s), x));
    v          = fma(h4 * h2, fin, v);
    return v * v;
}

// =============================================================================
//  UPPER chart (== br:: erfcx_poly, exp_neg, qnorm0_seed, price_residual, HH3)
//  eh = exp(-h/2), ehp = exp(h) are host-hoisted per-h libm scalars, exactly as
//  in the CPU batch drivers' index pass.
// =============================================================================
__device__ inline double d_erfcx_poly(double z) {
    const bool hi = (z > g::UPPER_ERFCX2_SPLIT);
    const bool h2 = (z > g::UPPER_ERFCX_B);
    const double* C = h2 ? g::UPPER_ERFCX2_C2 : (hi ? g::UPPER_ERFCX2_C1 : g::UPPER_ERFCX2_C0);
    const double a  = h2 ? g::UPPER_ERFCX_B    : (hi ? g::UPPER_ERFCX2_SPLIT : g::UPPER_ERFCX_A);
    const double b  = h2 ? g::UPPER_ERFCX2_B2  : (hi ? g::UPPER_ERFCX_B      : g::UPPER_ERFCX2_SPLIT);
    return d_clenshaw1(C, g::UPPER_ERFCX2_N, a, b, z);
}
__device__ inline double d_exp_neg(double a) {
    double nf = floor(fma(a, g::INV_LN2, 0.5));
    double r  = fma(-nf, g::LN2_HI, a);
    r         = fma(-nf, g::LN2_LO, r);
    double p  = d_clenshaw1(g::EXP_C, g::EXP_C_N, -0.5 * D_VA_LN2, 0.5 * D_VA_LN2, r);
    return ldexp(p, (int)nf);
}
__device__ inline double d_qnorm0_seed(double p) {
    double q = p - 0.5, r = q * q;
    double num = g::QN_A[0];
    for (int i = 1; i < 6; ++i) num = fma(num, r, g::QN_A[i]);
    num *= q;
    double den = g::QN_B[0];
    for (int i = 1; i < 5; ++i) den = fma(den, r, g::QN_B[i]);
    den = fma(den, r, 1.0);
    double mid = num / den;
    double qt = sqrt(-2.0 * d_full_log(p));
    double numt = g::QN_C[0];
    for (int i = 1; i < 6; ++i) numt = fma(numt, qt, g::QN_C[i]);
    double dent = g::QN_D[0];
    for (int i = 1; i < 4; ++i) dent = fma(dent, qt, g::QN_D[i]);
    dent = fma(dent, qt, 1.0);
    double tail = numt / dent;
    return (p < g::QN_P_LOW) ? tail : mid;
}
__device__ inline double d_price_residual(double h, double v, double ehp,
                                          double c, double onec, bool lo) {
    double hv = h / v;
    double y  = fma(0.5, v, -hv);
    double r  = fma(0.5, v,  hv);
    double t1 = -y * D_BR_IS2;
    double t2 = -(-y) * D_BR_IS2;
    double ay  = fabs(t1);
    double sy  = ay * ay;
    double scy = fma(ay, ay, -sy);
    double ey  = d_erfcx_poly(ay) * (d_exp_neg(-sy) * (1.0 - scy));
    double phi_y  = 0.5 * ((t1 >= 0.0) ? ey : (2.0 - ey));
    double phi_my = 0.5 * ((t2 >= 0.0) ? ey : (2.0 - ey));
    double tr  = -(-r) * D_BR_IS2;
    double ar  = fabs(tr);
    double sr  = ar * ar;
    double scr = fma(ar, ar, -sr);
    double er  = d_erfcx_poly(ar) * (d_exp_neg(-sr) * (1.0 - scr));
    double phi_r = 0.5 * ((tr >= 0.0) ? er : (2.0 - er));
    double bl = fma(-ehp, phi_r, phi_y);
    double om = fma( ehp, phi_r, phi_my);
    return lo ? (c - bl) : (om - onec);
}
__device__ inline double d_upper_variance(double h, double c, double eh, double ehp) {
    double gbar = 0.5 * (1.0 - c) * eh;
    double x0   = -d_qnorm0_seed(gbar);
    double m    = g::SQRT_PI2 * d_erfcx_poly(x0 * D_BR_IS2);
    double inv  = 1.0 / x0;
    double x2   = -0.125 * (inv - m);
    double B2p  = (fma(inv * inv, inv, -inv) + m) * (1.0 / 3.0);
    double ta   = 0.5 * x0 * x2 * x2;
    double tb   = 0.125 * x2 * inv * inv;
    double x4   = fma(B2p, (1.0 / 128.0), ta + tb);
    double h2   = h * h;
    double x    = fma(h2 * h2, x4, fma(h2, x2, x0));
    double onec = 1.0 - c;
    const bool lo = (c < 0.5);
    for (int it = 0; it < g::UPPER_HH3_STEPS; ++it) {
        double vv  = 2.0 * x;
        double cmC = d_price_residual(h, vv, ehp, c, onec, lo);
        double f   = 0.5 * cmC * eh;
        double gp  = -g::BR_K * d_exp_neg(-0.5 * x * x) * d_exp_neg(-h2 / (8.0 * x * x));
        double u   = f / gp;
        double ix  = 1.0 / x;
        double ix2 = ix * ix;
        double a   = fma(0.25 * h2, ix2 * ix, -x);
        double Lp  = fma(-0.75 * h2, ix2 * ix2, -1.0);
        double b   = fma(a, a, Lp);
        double au  = a * u;
        double num = fma(-0.5, au, 1.0);
        double den = fma(b * u, u * (1.0 / 6.0), 1.0 - au);
        x = x - u * num / den;
    }
    double v = 2.0 * x;
    return v * v;
}

// =============================================================================
//  WING chart (== wing_rt:: fit + series + fixed 6-step Newton; wing_Lt)
// =============================================================================
__device__ inline double d_logH_series_d(double W, double beta, double* dlogH) {
    const double b = beta, b2 = b*b, b3 = b2*b, b4 = b2*b2;
    const double g1 = -(b + 1.5);
    const double g2 =  (b + 21.0/8.0);
    const double g3 = -fma(3.5, b, 69.0/8.0);
    const double g4 =  fma(0.5, b2, fma(17.25, b, 2529.0/64.0));
    const double g5 = -fma(5.5, b2, fma(105.375, b, 36243.0/160.0));
    const double g6 =  fma(1.0/3.0, b3, fma(53.875, b2, fma(755.0625, b, 197127.0/128.0)));
    const double g7 = -fma(7.5, b3, fma(537.75, b2, fma(6160.21875, b, 10786527.0/896.0)));
    const double g8 =  fma(0.25, b4, fma(122.5, b3, fma(5651.15625, b2,
                       fma(56179.828125, b, 217179009.0/2048.0))));
    const double x = 1.0 / W;
    double s = g8;
    s = fma(s, x, g7); s = fma(s, x, g6); s = fma(s, x, g5);
    s = fma(s, x, g4); s = fma(s, x, g3); s = fma(s, x, g2);
    s = fma(s, x, g1);
    const double logH = s * x;
    if (dlogH) {
        double d = 8.0*g8;
        d = fma(d, x, 7.0*g7); d = fma(d, x, 6.0*g6); d = fma(d, x, 5.0*g5);
        d = fma(d, x, 4.0*g4); d = fma(d, x, 3.0*g3); d = fma(d, x, 2.0*g2);
        d = fma(d, x, g1);
        *dlogH = fma(-x * x, d, 0.0);
    }
    return logH;
}
__device__ inline double d_wing_S_d(const double* C, int NU, int NB, double ulo, double uhi,
                                    double u, double beta, double* dS_du) {
    double xu = fma(2.0, u,    -(ulo + uhi)) / (uhi - ulo);
    double xb = fma(2.0, beta, -(g::WING_B_LO + g::WING_B_HI)) / (g::WING_B_HI - g::WING_B_LO);
    const double tb2 = 2.0 * xb;
    const double tu2 = 2.0 * xu;
    double S, dS_dxu;
#if VOLFI_WING_FUSED
    // FUSED backward sweep, mirroring the host twin exactly.  On the device the payoff is
    // structural: rows[27]/dp[27] are dynamically indexed, so nvcc placed them in LOCAL
    // memory (432-byte stack frame = 27*8*2) and the kernel needed 128 registers, capping
    // occupancy near 25%.  Without the arrays the state is O(1).
    {
        double sd0 = 0.0, sd1 = 0.0, dd0 = 0.0, dd1 = 0.0;
        double row_next = 0.0, dpp1 = 0.0, dpp2 = 0.0;
        for (int i = NU - 1; i >= 1; --i) {
            const double* Ci = C + i * NB;
            double e0 = 0.0, e1 = 0.0;
            for (int j = NB - 1; j >= 1; --j) { double b0 = fma(tb2, e0, Ci[j]) - e1; e1 = e0; e0 = b0; }
            const double row_i = fma(xb, e0, Ci[0]) - e1;
            { double b0 = fma(tu2, sd0, row_i) - sd1; sd1 = sd0; sd0 = b0; }
            const double dp_i = (i == NU - 1) ? 0.0
                              : (i == NU - 2) ? 2.0 * (NU - 1) * row_next
                                              : fma(2.0 * (i + 1), row_next, dpp2);
            { double b0 = fma(tu2, dd0, dp_i) - dd1; dd1 = dd0; dd0 = b0; }
            row_next = row_i; dpp2 = dpp1; dpp1 = dp_i;
        }
        double e0 = 0.0, e1 = 0.0;                                  // i = 0
        for (int j = NB - 1; j >= 1; --j) { double b0 = fma(tb2, e0, C[j]) - e1; e1 = e0; e0 = b0; }
        const double row0 = fma(xb, e0, C[0]) - e1;
        S = fma(xu, sd0, row0) - sd1;
        double dp0 = fma(2.0 * (0 + 1), row_next, dpp2);
        dp0 *= 0.5;
        dS_dxu = fma(xu, dd0, dp0) - dd1;
    }
#else
    double rows[27];
    for (int i = 0; i < NU; ++i) {
        const double* Ci = C + i * NB;
        double d0 = 0.0, d1 = 0.0;
        for (int j = NB - 1; j >= 1; --j) { double b0 = fma(tb2, d0, Ci[j]) - d1; d1 = d0; d0 = b0; }
        rows[i] = fma(xb, d0, Ci[0]) - d1;
    }
    double d0 = 0.0, d1 = 0.0;
    for (int i = NU - 1; i >= 1; --i) { double b0 = fma(tu2, d0, rows[i]) - d1; d1 = d0; d0 = b0; }
    S = fma(xu, d0, rows[0]) - d1;
    double dp[27];
    dp[NU - 1] = 0.0;
    dp[NU - 2] = 2.0 * (NU - 1) * rows[NU - 1];
    for (int k = NU - 3; k >= 0; --k)
        dp[k] = fma(2.0 * (k + 1), rows[k + 1], dp[k + 2]);
    dp[0] *= 0.5;
    d0 = 0.0; d1 = 0.0;
    for (int i = NU - 1; i >= 1; --i) { double b0 = fma(tu2, d0, dp[i]) - d1; d1 = d0; d0 = b0; }
    dS_dxu = fma(xu, d0, dp[0]) - d1;
#endif
    *dS_du = dS_dxu * (2.0 / (uhi - ulo));
    return S;
}
__device__ inline double d_wing_logH_fit_d(double W, double beta, bool pieceA, double* dlogH) {
    const double* C; int NU, NB; double ulo, uhi;
    if (pieceA) { C = g::WING_SA; NU = g::WING_SA_NU; NB = g::WING_SA_NB;
                  ulo = g::WING_UA_LO; uhi = g::WING_UA_HI; }
    else        { C = g::WING_SB; NU = g::WING_SB_NU; NB = g::WING_SB_NB;
                  ulo = g::WING_UB_LO; uhi = g::WING_UB_HI; }
    double u = 1.0 / W;
    u = fmin(fmax(u, ulo), uhi);
    double Su;
    double S  = d_wing_S_d(C, NU, NB, ulo, uhi, u, beta, &Su);
    double bp = beta + 1.5;
    *dlogH = fma(-(u * u), fma(u, Su, S) - bp, 0.0);
    return u * (S - bp);
}
__device__ inline double d_wing_Lt(double h, double c) {
    double lc = d_full_log(h) - g::WING_LOG4;
    lc = fma(0.5, h, lc) - g::WING_HLOGPI;
    return -d_full_log_sub(c) + lc;
}
__device__ inline double d_wing_variance(double h, double c) {
    const double beta = h * h * (1.0 / 16.0);
    const double Lt = d_wing_Lt(h, c);
    const double lt = fmax(Lt, 2.0);
    double W = fmax(fma(-1.5, d_full_log(lt), lt), 2.6);   // W0 cheap seed
    const bool series = (Lt > g::WING_LT_BS);
    const bool pieceA = (Lt <= g::WING_LT_AB);
    if (!series) {                              // v0.2.1: add the (8,8) seed-table residual
        const double* C  = pieceA ? g::WSEED_A      : g::WSEED_B;
        const int NU     = pieceA ? g::WSEED_A_NU   : g::WSEED_B_NU;
        const int NB     = pieceA ? g::WSEED_A_NB   : g::WSEED_B_NB;
        const double LO  = pieceA ? g::WSEED_A_LTLO : g::WSEED_B_LTLO;
        const double HI  = pieceA ? g::WSEED_A_LTHI : g::WSEED_B_LTHI;
        const double bLO = pieceA ? g::WSEED_A_BLO  : g::WSEED_B_BLO;
        const double bHI = pieceA ? g::WSEED_A_BHI  : g::WSEED_B_BHI;
        const double Ltc = fmin(fmax(Lt,   LO ), HI );
        const double bc  = fmin(fmax(beta, bLO), bHI);
        const double xL  = fma(2.0, Ltc, -(LO + HI)) / (HI - LO);
        const double xb  = fma(2.0, bc,  -(bLO + bHI)) / (bHI - bLO);
        W = fmax(W + d_clenshaw2(NU - 1, NB - 1, C, xL, xb), 2.6);
    }
    for (int it = 0; it < g::WING_NEWTON_STEPS; ++it) {
        double dlH;
        const double lH = series ? d_logH_series_d(W, beta, &dlH)
                                 : d_wing_logH_fit_d(W, beta, pieceA, &dlH);
        const double F  = (W - Lt) + fma(1.5, d_full_log(W), -lH);
        const double Fp = 1.0 + 1.5 / W - dlH;
        W = fmax(W - F / Fp, 2.6);
    }
    return (h * h) / (2.0 * W);
}

// =============================================================================
//  FAR chart (== far_variance table interior; band/xh host-precomputed,
//  host guarantees the quote lands on a real table cell -- v1 kernel logic).
// =============================================================================
__device__ inline double d_far_variance(double h2, double xh, int band, double c) {
    uint64_t bc = d_bits(c);
    int k = (int)((bc >> 52) & D_C_EXP_MASK) - g::C_EXP_BIAS;
    int oc  = g::OCTBASE[band] + (k - g::KLO[band]);
    int sb  = g::OCT_SBITS[oc];
    int sub = sb ? (int)((bc >> (52 - sb)) & ((1u << sb) - 1u)) : 0;
    int cell = g::CBASE[band] + g::OCT_CELLOFF[oc] + sub;
    double m  = d_mant12(bc);
    double xl = fma(g::CELL_LSCALE[cell], d_log2approx(m), g::CELL_LBIAS[cell]);
    double W  = d_clenshaw2(g::CELL_DP[cell], g::CELL_DL[cell], &g::COEFFS[g::CELL_OFF[cell]], xh, xl);
    return h2 / (2.0 * W);
}

// =============================================================================
//  Per-chart kernels over CONTIGUOUS bucket slices (grid-stride).
// =============================================================================
// ONE-STEP UPPER chart (== br::upper1_seed + br::upper_variance of v0.3.1, op for op).  No per-h host scalars.
__device__ inline double d_upper1_variance(double h, double c) {
    namespace U = volfi_annulus_gpu::upper1;
    const double onec = 1.0 - c, h2 = h * h; const bool lo = (c < 0.5);
    const double aU = h - 2.0 * d_full_log(onec);
    const double ia = 1.0 / aU;
    double s = sqrt(aU);
    s = (s < U::S_LO) ? U::S_LO : ((s > U::S_HI) ? U::S_HI : s);
    const double x0 = d_clenshaw1(U::X0C, U::NX0, U::S_LO, U::S_HI, s);
    const double q  = h * ia;
    const double om = 1.0 - q * q;
    const double t  = q / (1.0 + sqrt(om > 0.0 ? om : 0.0));
    const double zeta = t * t;
    const double S = fma(2.0 / (U::SGMAX - U::SGMIN), x0 * x0 * ia - U::SGMIN, -1.0);
    const double Z = fma(2.0 / U::ZMAX, zeta, -1.0);
    double r0 = U::R0[5]; for (int j = 4; j >= 0; --j) r0 = fma(r0, Z, U::R0[j]);
    double r1 = U::R1[3]; for (int j = 2; j >= 0; --j) r1 = fma(r1, Z, U::R1[j]);
    const double r2 = fma(U::R2[1], Z, U::R2[0]);
    const double R  = fma(fma(fma(U::R3[0], S, r2), S, r1), S, r0);
    double x = fma(x0 * zeta, R, x0);
    const double ix = 1.0 / x, d = 0.5 * h * ix, y = x - d, r = x + d;
    const double ty = y * U::IS2, ay = fabs(ty);
    const double sy = ay * ay, scy = fma(ay, ay, -sy);
    const double p  = d_exp_neg(-sy) * (1.0 - scy);
    const double er = d_erfcx_poly(r * U::IS2);
    const double z  = lo ? -ty : ty;
    const double ey = (z >= 0.0) ? d_erfcx_poly(z) : d_clenshaw1(U::ENEG_C, U::NENEG, U::ENEG_A, U::ENEG_B, z);
    const double resid = lo ? (c - 0.5 * p * (ey - er)) : (0.5 * p * (ey + er) - onec);
    const double u  = -resid / (U::TWO_K * p);
    const double ix2 = ix * ix;
    const double a  = fma(0.25 * h2, ix2 * ix, -x);
    const double b  = fma(a, a, fma(-0.75 * h2, ix2 * ix2, -1.0));
    const double au = a * u;
    x = x - u * fma(-0.5, au, 1.0) / fma(b * u, u * (1.0 / 6.0), 1.0 - au);
    const double v = 2.0 * x; return v * v;
}
#if VGB_HOST_CHECK
int main() {                                   // transcription check on the CPU: device mirrors == library, bit for bit
    unsigned long long gs = 0x9E3779B97F4A7C15ULL; auto rnd = [&]() { gs = gs * 6364136223846793005ULL + 1442695040888963407ULL; return (double)(gs >> 11) * (1.0 / 9007199254740992.0); };
    auto Phi = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    long n = 0, m1 = 0, m0 = 0;
    while (n < 1000000) { const double h = 1e-4 + 16.2 * rnd() * rnd(), v = 1.85 + 6.15 * rnd();
        const double c = Phi(0.5 * v - h / v) - std::exp(h) * Phi(-0.5 * v - h / v);
        if (!(c > 0.0 && c < 1.0) || 1.0 - c < 1e-15 || volfi_annulus::detail::grid_endpoint_route(h, c) != 2) continue; ++n;
        const double a = d_upper1_variance(h, c), b = volfi_annulus::br::upper_variance(h, c); if (std::memcmp(&a, &b, 8)) ++m1;
        const double a0 = d_upper_variance(h, c, std::exp(-0.5 * h), std::exp(h)), b0 = volfi_annulus::br::upper_variance_v030(h, c); if (std::memcmp(&a0, &b0, 8)) ++m0; }
    std::printf("host check, %ld UPPER quotes: d_upper1_variance == br::upper_variance (one step): mismatches = %ld | control, v0.3.0 mirror == v0.3.0 chart: mismatches = %ld\n", n, m1, m0);
    return (m1 == 0 && m0 == 0) ? 0 : 1;
}
#else
__global__ void near_kernel(const double* h, const double* c, double* w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        w[i] = d_near_variance(h[i], c[i]);
}
__global__ void far_kernel(const double* h, const double* c, const double* xh,
                               const int* band, double* w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        w[i] = d_far_variance(h[i] * h[i], xh[i], band[i], c[i]);
}
__global__ void upper_kernel(const double* h, const double* c, const double* eh,
                             const double* ehp, double* w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        w[i] = d_upper1_variance(h[i], c[i]);          // v0.3.1: one step, eh / ehp unused
}
__global__ void wing_kernel(const double* h, const double* c, double* w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        w[i] = d_wing_variance(h[i], c[i]);
}

// =============================================================================
//  HOST driver: load feed -> route via the CPU reference -> bucket-order ->
//  tile to n_target -> ULP cross-check -> per-chart + full-book timing.
// =============================================================================
enum { R_WING = 0, R_NEAR = 1, R_FAR = 2, R_UPPER = 3 };
static const char* RNAME[4] = { "WING", "NEAR", "FAR", "UPPER" };

struct Quote { double h, c, w_ref, xh, eh, ehp; int band, route; };

static int64_t ulp_diff(double a, double b) {
    int64_t ia, ib;
    std::memcpy(&ia, &a, 8); std::memcpy(&ib, &b, 8);
    if (ia < 0) ia = INT64_MIN - ia;
    if (ib < 0) ib = INT64_MIN - ib;
    int64_t d = ia - ib;
    return d < 0 ? -d : d;
}

// Route classifier + per-h host precomputes.  ONE definition, used by both the
// market feed and the synthetic chart-pure surfaces, so the two input stages can
// never drift apart.  Returns false for the scalar-only analytic fallbacks
// (rescue wedge / off-table FAR cell / non-positive reference), counting
// each exclusion so it is reported rather than silently dropped.
static bool classify_quote(double hh, double cc, Quote& Q,
                           long& excl_rescue, long& excl_cell, long& excl_edge) {
    using namespace volfi_annulus;
    if (!(hh > 1e-4 && hh < 16.5) || !(cc > 0.0 && cc < 1.0)) return false;
    context ctx(hh);
    Q.h = hh; Q.c = cc; Q.band = ctx.band; Q.xh = ctx.xh;
    Q.eh = std::exp(-0.5 * hh); Q.ehp = std::exp(hh);      // per-h libm hoists (as CPU batch)
    if (cc < ctx.cw) {                                       // WING
        double beta = hh * hh * (1.0 / 16.0);
        if (!(beta <= wing_detail::WING_B_HI)) { excl_rescue++; return false; }  // rescue wedge: CPU-side
        Q.route = R_WING;
    } else if (ctx.region == 1) {                            // h < H_ATM_HI
        Q.route = (cc <= ctx.ct_near) ? R_NEAR : R_UPPER;
    } else if (ctx.region == 0 && cc <= ctx.ct2) {           // FAR box
        if (far_cell_of(ctx, cc, detail::bits_of(cc)) < 0) { excl_cell++; return false; } // analytic edge
        Q.route = R_FAR;
    } else {
        Q.route = R_UPPER;
    }
    Q.w_ref = implied_variance_otm(ctx, cc);
    if (!(Q.w_ref > 0.0)) { excl_edge++; return false; }
    return true;
}

int main(int argc, char** argv) {
    using namespace volfi_annulus;
    int n_target = (argc > 1) ? std::atoi(argv[1]) : 1000000;
    int n_iters  = (argc > 2) ? std::atoi(argv[2]) : 100;

    // ---- load market feed (same filter as bench_phases.cpp) ----
    std::vector<double> fh, fc;
    std::ifstream fin("market_feed.csv");
    if (!fin) { std::printf("market_feed.csv not found in cwd -- aborting.\n"); return 1; }
    std::string ln;
    while (std::getline(fin, ln)) {
        if (ln.empty() || ln[0] == '#') continue;
        std::istringstream is(ln); double hh, cc;
        if (!(is >> hh >> cc)) continue;
        if (!(hh > 1e-4 && hh < 16.5) || !(cc > 0.0 && cc < 1.0)) continue;
        fh.push_back(hh); fc.push_back(cc);
    }
    int nf = (int)fh.size();
    if (!nf) { std::printf("empty feed.\n"); return 1; }

    // ---- route + reference via the UNTOUCHED CPU library ----
    std::vector<Quote> q;
    q.reserve(nf);
    long excl_rescue = 0, excl_cell = 0, excl_edge = 0;
    for (int i = 0; i < nf; ++i) {
        Quote Q;
        if (classify_quote(fh[i], fc[i], Q, excl_rescue, excl_cell, excl_edge)) q.push_back(Q);
    }
    int nq = (int)q.size();

    // ---- BUCKET ORDER: NEAR | FAR(by cell) | WING(by regime) | UPPER ----
    // In a live book this order is computed once and PERSISTS across snapshots;
    // there is no per-snapshot sort and no scatter (results stay in book order).
    std::stable_sort(q.begin(), q.end(), [](const Quote& a, const Quote& b) {
        static const int ord[4] = { 2, 0, 1, 3 };            // NEAR, FAR, WING, UPPER
        if (ord[a.route] != ord[b.route]) return ord[a.route] < ord[b.route];
        if (a.route == R_FAR) {                          // cell-coherent far slice
            context ca(a.h), cb(b.h);
            return far_cell_of(ca, a.c, volfi_annulus::detail::bits_of(a.c))
                 < far_cell_of(cb, b.c, volfi_annulus::detail::bits_of(b.c));
        }
        if (a.route == R_WING) {                             // regime-coherent wing slice
            double la = wing_Lt(a.h, a.c), lb = wing_Lt(b.h, b.c);
            auto reg = [](double L) { return L > wing_detail::WING_LT_BS ? 2 : (L <= wing_detail::WING_LT_AB ? 0 : 1); };
            return reg(la) < reg(lb);
        }
        return false;
    });

    // ---- tile the bucket-ordered feed to >= n_target (slice-wise: preserves contiguity)
    int rep = std::max(1, (n_target + nq - 1) / nq);
    int cnt[4] = {0,0,0,0};
    for (auto& Q : q) cnt[Q.route]++;
    int n = nq * rep;
    std::vector<double> H(n), C(n), XH(n), EH(n), EHP(n), Wref(n), Wgpu(n);
    std::vector<int> BAND(n);
    int off[5];   // slice offsets in bucket order NEAR|FAR|WING|UPPER (tiled)
    off[0] = 0; off[1] = cnt[R_NEAR] * rep; off[2] = off[1] + cnt[R_FAR] * rep;
    off[3] = off[2] + cnt[R_WING] * rep;    off[4] = off[3] + cnt[R_UPPER] * rep;
    {
        // q is already ordered NEAR,FAR,WING,UPPER contiguously; tile slice-wise
        int pos = 0, qpos = 0;
        int route_seq[4] = { R_NEAR, R_FAR, R_WING, R_UPPER };
        for (int blk = 0; blk < 4; ++blk) {
            int r = route_seq[blk];
            int cblk = cnt[r];
            for (int t = 0; t < rep; ++t)
                for (int k = 0; k < cblk; ++k) {
                    const Quote& Q = q[qpos + k];
                    H[pos] = Q.h; C[pos] = Q.c; XH[pos] = Q.xh; BAND[pos] = Q.band;
                    EH[pos] = Q.eh; EHP[pos] = Q.ehp; Wref[pos] = Q.w_ref;
                    ++pos;
                }
            qpos += cblk;
        }
    }
    int nL = off[1] - off[0], nC = off[2] - off[1], nW = off[3] - off[2], nR = off[4] - off[3];

    std::printf("=== volfi-annulus GPU FULL-BOOK port (bucket-order coalesced) ===\n");
    std::printf("feed=%d quotes  ->  eligible=%d  excluded: rescue=%ld cell=%ld edge=%ld\n",
                nf, nq, excl_rescue, excl_cell, excl_edge);
    std::printf("route mix: NEAR=%.2f%%  FAR=%.2f%%  WING=%.2f%%  UPPER=%.2f%%   (tiled x%d -> N=%d)\n",
                100.0*cnt[R_NEAR]/nq, 100.0*cnt[R_FAR]/nq, 100.0*cnt[R_WING]/nq,
                100.0*cnt[R_UPPER]/nq, rep, n);

    // ---- device buffers ----
    double *dH, *dC, *dXH, *dEH, *dEHP, *dW; int* dBAND;
    CUDA_CHECK(cudaMalloc(&dH,   n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC,   n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dXH,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dEH,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dEHP, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dW,   n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dBAND, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dH,   H.data(),   n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC,   C.data(),   n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXH,  XH.data(),  n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEH,  EH.data(),  n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEHP, EHP.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBAND, BAND.data(), n * sizeof(int),  cudaMemcpyHostToDevice));

    const int TPB = 256;
    auto blocks = [&](int m) { return std::min(4096, (m + TPB - 1) / TPB); };
    auto launch_all = [&]() {
        if (nL) near_kernel   <<<blocks(nL), TPB>>>(dH + off[0], dC + off[0], dW + off[0], nL);
        if (nC) far_kernel<<<blocks(nC), TPB>>>(dH + off[1], dC + off[1], dXH + off[1],
                                                    dBAND + off[1], dW + off[1], nC);
        if (nW) wing_kernel   <<<blocks(nW), TPB>>>(dH + off[2], dC + off[2], dW + off[2], nW);
        if (nR) upper_kernel  <<<blocks(nR), TPB>>>(dH + off[3], dC + off[3], dEH + off[3],
                                                    dEHP + off[3], dW + off[3], nR);
    };

    // ---- correctness pass: one run, ULP-compare vs CPU reference ----
    launch_all();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(Wgpu.data(), dW, n * sizeof(double), cudaMemcpyDeviceToHost));
    {
        long mism[4] = {0,0,0,0}; int64_t worst[4] = {0,0,0,0};
        int route_of_slice[4] = { R_NEAR, R_FAR, R_WING, R_UPPER };
        for (int s = 0; s < 4; ++s)
            for (int i = off[s]; i < off[s + 1]; ++i) {
                int64_t d = ulp_diff(Wgpu[i], Wref[i]);
                if (d > worst[s]) worst[s] = d;
                if (d > 1) mism[s]++;
            }
        long tot = 0;
        std::printf("\nGPU vs CPU reference (mismatches > 1 ulp MUST be 0):\n");
        for (int s = 0; s < 4; ++s) {
            int m = off[s + 1] - off[s];
            std::printf("  %-8s n=%8d  mismatches=%ld  worst_ulp=%lld\n",
                        RNAME[route_of_slice[s]], m, mism[s], (long long)worst[s]);
            tot += mism[s];
        }
        std::printf("  => %s\n", tot == 0 ? "PASS: ULP-IDENTICAL to CPU reference"
                                          : "FAIL: DO NOT trust the timing numbers");
    }

    // ---- timing: per-chart, then full book ----
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    auto time_ms = [&](auto&& body) {
        body();                                            // warm-up
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(t0));
        for (int it = 0; it < n_iters; ++it) body();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        return (double)ms;
    };
    std::printf("\nTIMING (%d iters, ns/eval):\n", n_iters);
    if (nL) { double ms = time_ms([&]{ near_kernel<<<blocks(nL),TPB>>>(dH+off[0],dC+off[0],dW+off[0],nL); });
              std::printf("  NEAR     n=%8d  %8.4f ns/eval\n", nL, 1e6*ms/((double)n_iters*nL)); }
    if (nC) { double ms = time_ms([&]{ far_kernel<<<blocks(nC),TPB>>>(dH+off[1],dC+off[1],dXH+off[1],dBAND+off[1],dW+off[1],nC); });
              std::printf("  FAR  n=%8d  %8.4f ns/eval\n", nC, 1e6*ms/((double)n_iters*nC)); }
    if (nW) { double ms = time_ms([&]{ wing_kernel<<<blocks(nW),TPB>>>(dH+off[2],dC+off[2],dW+off[2],nW); });
              std::printf("  WING     n=%8d  %8.4f ns/eval\n", nW, 1e6*ms/((double)n_iters*nW)); }
    if (nR) { double ms = time_ms([&]{ upper_kernel<<<blocks(nR),TPB>>>(dH+off[3],dC+off[3],dEH+off[3],dEHP+off[3],dW+off[3],nR); });
              std::printf("  UPPER    n=%8d  %8.4f ns/eval\n", nR, 1e6*ms/((double)n_iters*nR)); }
    {
        double ms = time_ms(launch_all);
        std::printf("  FULL BOOK (all four kernels, market mix): N=%d  %8.4f ns/eval\n",
                    n, 1e6*ms/((double)n_iters*n));
        // Hand-carried from the CPU campaign for orientation only -- update it when
        // bench_512's D_gridbatch_feed changes (v0.2.3, 2026-07-27: 48.4 ns/quote).
        std::printf("  (compare: CPU AVX-512 cold speculative driver = 48.4 ns/quote, same feed)\n");
    }
    CUDA_CHECK(cudaFree(dH)); CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dXH));
    CUDA_CHECK(cudaFree(dEH)); CUDA_CHECK(cudaFree(dEHP)); CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dBAND));

    // =========================================================================
    //  CHART-PURE FIXED-h SURFACES -- one row per chart, same footing.
    // =========================================================================
    // The market feed routes 0% to UPPER (high-variance quotes are not tradeable
    // in the 2024 SPX book), so the feed alone can never time that chart: nR==0
    // and its line is skipped.  These synthetic surfaces use the SAME fixed
    // (h, v-range) workloads as the CPU benchmark's per-chart rows, so all four
    // charts get a number, and the GPU table can mirror the CPU one.
    //
    // On the device this is a fair comparison across all four charts: unlike the
    // CPU, there is no fixed-h vs mixed-h driver split here -- every chart is its
    // own kernel over a contiguous slice either way -- so no chart is measured
    // through a different execution path than the others.
    //
    // Each set is route-filtered through the SAME classifier the feed uses, so a
    // row is chart-pure by construction, and every row is ULP-checked against the
    // CPU reference before it is timed.
    {
        struct Spec { int route; const char* name; double h, vlo, vhi; };
        // These v-ranges mirror the CPU benchmark's per-chart rows and MUST move
        // when a seam moves: v0.2.3 dropped the shared ceiling to 1.85 and lifted
        // the wing seam to v=h/sqrt(7.6), which put the top of the old FAR
        // ([.,1.95]) and WING ([.,0.40]) ranges into the neighbouring chart.  The
        // route filter below discards those quotes rather than mistiming them, so
        // a stale range costs sample count, not correctness -- watch off-chart.
        const Spec specs[4] = {
            { R_FAR, "FAR", 1.00, 0.41, 1.80 },   // CPU row: far batch, h=1
            { R_NEAR,    "NEAR",    0.20, 0.30, 1.60 },   // CPU row: near batch, h=0.2
            { R_WING,    "WING",    1.00, 0.15, 0.35 },   // below the wing seam h/sqrt(7.6)=0.363
            { R_UPPER,   "UPPER",   1.00, 2.10, 8.00 },   // CPU row: upper batch, h=1
        };
        // Matches volfi::black_otm_from_variance (paper_volfi.hpp) -- inlined here
        // because that header is not part of volfi_annulus_all.hpp.
        auto price = [](double hh, double w) {
            double s = std::sqrt(w), u = -hh / s;
            auto Phi = [](double x){ return 0.5 * std::erfc(-x * 0.70710678118654752440); };
            return Phi(u + 0.5 * s) - std::exp(hh) * Phi(u - 0.5 * s);
        };

        std::printf("\nCHART-PURE FIXED-h SURFACES (mirror of the CPU per-chart rows):\n");
        for (const Spec& S : specs) {
            const int M = 262144;
            std::vector<Quote> s; s.reserve(M);
            long er = 0, ec = 0, ee = 0, off_chart = 0;
            for (int i = 0; i < M; ++i) {
                double v = S.vlo + (S.vhi - S.vlo) * (i + 0.5) / M;
                Quote Q;
                if (!classify_quote(S.h, price(S.h, v * v), Q, er, ec, ee)) continue;
                if (Q.route != S.route) { off_chart++; continue; }
                s.push_back(Q);
            }
            int m = (int)s.size();
            if (m < 1024) {
                std::printf("  %-8s SKIPPED: only %d in-chart quotes from h=%.2f v in [%.2f,%.2f]\n",
                            S.name, m, S.h, S.vlo, S.vhi);
                continue;
            }
            // price-ascending: the cell-coherent order the CPU surface benchmark uses
            std::stable_sort(s.begin(), s.end(),
                             [](const Quote& a, const Quote& b) { return a.c < b.c; });

            int r2 = std::max(1, (n_target + m - 1) / m);
            int nn = m * r2;
            std::vector<double> sH(nn), sC(nn), sXH(nn), sEH(nn), sEHP(nn), sRef(nn), sOut(nn);
            std::vector<int> sB(nn);
            for (int t = 0, p = 0; t < r2; ++t)
                for (int k = 0; k < m; ++k, ++p) {
                    sH[p] = s[k].h;  sC[p] = s[k].c;   sXH[p] = s[k].xh; sB[p] = s[k].band;
                    sEH[p] = s[k].eh; sEHP[p] = s[k].ehp; sRef[p] = s[k].w_ref;
                }

            double *aH, *aC, *aXH, *aEH, *aEHP, *aW; int* aB;
            CUDA_CHECK(cudaMalloc(&aH,  nn*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&aC,  nn*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&aXH, nn*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&aEH, nn*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&aEHP,nn*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&aW,  nn*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&aB,  nn*sizeof(int)));
            CUDA_CHECK(cudaMemcpy(aH,  sH.data(),  nn*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(aC,  sC.data(),  nn*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(aXH, sXH.data(), nn*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(aEH, sEH.data(), nn*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(aEHP,sEHP.data(),nn*sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(aB,  sB.data(),  nn*sizeof(int),    cudaMemcpyHostToDevice));

            auto launch = [&]() {
                switch (S.route) {
                    case R_NEAR:    near_kernel   <<<blocks(nn),TPB>>>(aH, aC, aW, nn); break;
                    case R_FAR: far_kernel<<<blocks(nn),TPB>>>(aH, aC, aXH, aB, aW, nn); break;
                    case R_WING:    wing_kernel   <<<blocks(nn),TPB>>>(aH, aC, aW, nn); break;
                    case R_UPPER:   upper_kernel  <<<blocks(nn),TPB>>>(aH, aC, aEH, aEHP, aW, nn); break;
                }
            };

            launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sOut.data(), aW, nn*sizeof(double), cudaMemcpyDeviceToHost));
            long mism = 0; int64_t worst = 0;
            for (int i = 0; i < nn; ++i) {
                int64_t d = ulp_diff(sOut[i], sRef[i]);
                if (d > worst) worst = d;
                if (d > 1) mism++;
            }
            double ms = time_ms(launch);
            std::printf("  %-8s h=%.2f v in [%.2f,%.2f]  n=%8d  %8.4f ns/eval   "
                        "mismatches=%ld worst_ulp=%lld%s\n",
                        S.name, S.h, S.vlo, S.vhi, nn, 1e6*ms/((double)n_iters*nn),
                        mism, (long long)worst,
                        mism ? "   <-- FAIL: timing not trustworthy" : "");
            if (off_chart || er || ec || ee)
                std::printf("           (generated %d, kept %d; off-chart=%ld rescue=%ld cell=%ld edge=%ld)\n",
                            M, m, off_chart, er, ec, ee);

            CUDA_CHECK(cudaFree(aH));  CUDA_CHECK(cudaFree(aC));  CUDA_CHECK(cudaFree(aXH));
            CUDA_CHECK(cudaFree(aEH)); CUDA_CHECK(cudaFree(aEHP)); CUDA_CHECK(cudaFree(aW));
            CUDA_CHECK(cudaFree(aB));
        }
    }
    return 0;
}
#endif // !VGB_HOST_CHECK
