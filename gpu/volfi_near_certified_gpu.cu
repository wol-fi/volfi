// volfi_near_certified_gpu.cu -- CUDA twin of the v0.3.0-test certified NEAR chart
// =============================================================================
// Companion to include/volfi/volfi_near_certified.hpp, volfi_near_book.hpp
// and the AVX-512 / AVX2 twins in src/volfi_near_certified_vec.hpp.
//
// CONTRACT.  Every thread performs exactly the scalar operation sequence, in the
// same order, with the same fused multiply-adds, so the device result equals the
// CPU reference bit for bit.  Build with --fmad=false, the device analogue of
// -ffp-contract=off: without it nvcc contracts a*b+c on its own and the identity
// is lost.
//
// TWO KERNELS, AND WHAT THE FIRST H100 RUN TAUGHT.
//
// The 12-cell chart (near_cert_kernel_selfroute) measured 0.062 ns per quote on an
// H100 against the shipped NEAR kernel's 0.037, and the bucketed schedule that was
// supposed to help it measured 0.077: twelve launches cost more than the divergence
// they removed.  The certified chart does LESS arithmetic than the shipped one, so
// the loss is layout: graded rows with data-dependent loop bounds, indexed through
// three offset arrays, at addresses that differ across a warp.  The shipped kernel
// is straight-line code whose every coefficient is a compile-time array index.
//
// The BOOK kernel (near_book_kernel) gives the certified chart that shape, and the
// mathematics is what permits it.  The tau-radius of the fixed-rho theorem is 1
// everywhere, so on the book's tau range [0, 0.05] every cell needs about nine
// rows, and the a-direction singularities sit on the imaginary axis at height
// >= 2 pi/(1 + sqrt 0.05) = 5.1 independent of a, so ONE cell certifies over the
// whole a-range: 10 rows, 152 coefficients.  With one cell the row profile is a
// set of template constants, the whole 2-D Clenshaw unrolls into straight-line fma
// with constant-bank operands, and nothing diverges.  The chart's mode-3 log1p and
// reciprocal affine maps bring the division count from six to three.
//
// Quotes above the book's tau range (none on the feed; the ceiling region of the
// validation population) get w = -1 from the book kernel and are answered by a
// second pass through the 12-cell chart.  Both passes are timed, separately and
// together, on the feed tile and on the mixed probe tile, so the reader sees what
// the sentinel pass costs when it has nothing to do and when it has.
//
// BUILD (inside a CUDA devel container, /w mounted on the volfi tree):
//   nvcc -O3 -arch=sm_90 -std=c++17 --fmad=false -Xptxas -v -DNB_LAYOUT=1 \
//        -I.. volfi_near_certified_gpu.cu -o volfi_near_gpu_L1
// RUN (from a directory holding market_feed.csv and near_truth.bin):
//   ./volfi_near_gpu_L1 5000000 300
// HOST-ONLY SELF-CHECK, no GPU required:
//   nvcc -O3 -std=c++17 --fmad=false -DNCG_HOST_CHECK=1 -DNB_LAYOUT=1 -x c++ -I../include/volfi \
//        volfi_near_certified_gpu.cu -o ncheck_L1 && ./ncheck_L1
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#ifndef NCG_HOST_CHECK
#define NCG_HOST_CHECK 0
#endif
#ifndef NB_LAYOUT
#define NB_LAYOUT 1
#endif

#if !NCG_HOST_CHECK
#include <cuda_runtime.h>
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                 \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::exit(1); } } while (0)
#else
// Host-only build: strip the qualifiers so the same source can be run on the CPU
// and compared against the reference.  This is what verifies the transcription.
#define __device__
#define __constant__
#define __host__
#define __global__
#endif

// Device tables: the 12-cell chart (all arrays), and the book chart's coefficient
// array under the name d_NB_COEFF that volfi_near_book.hpp reads on the device.
#include "volfi_near_certified_tables_cuda.cuh"
#if   NB_LAYOUT == 1
#include "volfi_near_book_tables_L1_cuda.cuh"
#elif NB_LAYOUT == 2
#include "volfi_near_book_tables_L2_cuda.cuh"
#else
#include "volfi_near_book_tables_L3_cuda.cuh"
#endif

#include "volfi_near_rec_tables_cuda.cuh"
#include "volfi_wb_tables_cuda.cuh"
namespace volfi_wb {                      // the expm1 Taylor of volfi_wb.hpp, device copy
__constant__ double d_NWE_C[20] = {
  1.00000000000000000e+00, 5.00000000000000000e-01, 1.66666666666666667e-01,
  4.16666666666666667e-02, 8.33333333333333333e-03, 1.38888888888888889e-03,
  1.98412698412698413e-04, 2.48015873015873016e-05, 2.75573192239858907e-06,
  2.75573192239858907e-07, 2.50521083854417188e-08, 2.08767569878680990e-09,
  1.60590438368216146e-10, 1.14707455977297247e-11, 7.64716373181981648e-13,
  4.77947733238738530e-14, 2.81145725434552076e-15, 1.56192069685862265e-16,
  8.22063524662432972e-18, 4.11031762331216486e-19,
};
}

// The CPU reference, for routing on the host and for the bitwise cross-check.
#include "paper_volfi.hpp"
#include "volfi_annulus_wing.hpp"
#include "volfi_annulus.hpp"
#include "volfi_near_certified.hpp"
#include "volfi_near_book.hpp"
#include "volfi_near_rec.hpp"
#include "volfi_wb.hpp"

namespace g  = volfi_near_certified_gpu;
namespace nc = volfi_near_certified;
namespace nb = volfi_near_book;
namespace nr = volfi_near_rec;
namespace nw = volfi_wb;

// =============================================================================
//  Device kernels.  The coordinate kernels are line-for-line mirrors of their host
//  counterparts in src/volfi_near_certified.hpp, with std::fma written as fma().
//  The book Clenshaw is SHARED source (src/volfi_near_book.hpp), compiled for both.
// =============================================================================

__device__ static const double D_NCL_LN2_HI = 6.93147180369123816490e-01;
__device__ static const double D_NCL_LN2_LO = 1.90821492927058770002e-10;
__device__ static const double D_NCL_SQRT2  = 1.41421356237309504880;
__constant__ double D_NCL_LG[7] = {
    6.666666666666735130e-01, 3.999999999940941908e-01, 2.857142874366239149e-01,
    2.222219843214978396e-01, 1.818357216161805012e-01, 1.531383769920937332e-01,
    1.479819860511658591e-01,
};
__constant__ double D_NCL3_P[8] = {
    6.6666666666666663e-1, 4.0000000000012037e-1, 2.8571428565490653e-1,
    2.2222223336620348e-1, 1.8181715682579969e-1, 1.5389718263641888e-1,
    1.3193475586569775e-1, 1.3730049006664399e-1,
};
__constant__ double D_NCE_C[14] = {
    1.00000000000000000e+00, 5.00000000000000000e-01, 1.66666666666666667e-01,
    4.16666666666666667e-02, 8.33333333333333333e-03, 1.38888888888888889e-03,
    1.98412698412698413e-04, 2.48015873015873016e-05, 2.75573192239858907e-06,
    2.75573192239858907e-07, 2.50521083854417188e-08, 2.08767569878680990e-09,
    1.60590438368216146e-10, 1.14707455977297247e-11,
};

__device__ inline uint64_t d_bits(double x) {
#if NCG_HOST_CHECK
    uint64_t b; std::memcpy(&b, &x, 8); return b;
#else
    return (uint64_t)__double_as_longlong(x);
#endif
}
__device__ inline double d_from_bits(uint64_t b) {
#if NCG_HOST_CHECK
    double x; std::memcpy(&x, &b, 8); return x;
#else
    return __longlong_as_double((long long)b);
#endif
}

// mirror of nc_expm1
__device__ inline double d_nc_expm1(double h) {
    double p = D_NCE_C[13];
    for (int k = 12; k >= 1; --k) p = fma(p, h, D_NCE_C[k]);
    return fma(h, p * h, h);
}

// mirror of nc_log (mode 2)
__device__ inline double d_nc_log(double x) {
    const uint64_t b = d_bits(x);
    double m = d_from_bits((b & 0x000FFFFFFFFFFFFFULL) | ((uint64_t)1023 << 52));
    double k = (double)((int)((b >> 52) & 0x7FFULL) - 1023);
    if (m > D_NCL_SQRT2) { m *= 0.5; k += 1.0; }
    const double f = m - 1.0;
    const double s = f / (2.0 + f);
    const double z = s * s;
    const double w = z * z;
    double t1 = fma(w, D_NCL_LG[5], D_NCL_LG[3]);
    t1 = fma(w, t1, D_NCL_LG[1]);
    t1 = w * t1;
    double t2 = fma(w, D_NCL_LG[6], D_NCL_LG[4]);
    t2 = fma(w, t2, D_NCL_LG[2]);
    t2 = fma(w, t2, D_NCL_LG[0]);
    t2 = z * t2;
    const double R    = t2 + t1;
    const double hfsq = 0.5 * f * f;
    return fma(k, D_NCL_LN2_HI, -((hfsq - (fma(s, hfsq + R, k * D_NCL_LN2_LO))) - f));
}
// mirror of log1p_nc (mode 2)
__device__ inline double d_log1p_nc(double q) {
    const double u = 1.0 + q;
    const double d = u - 1.0;
    return d_nc_log(u) + (q - d) / u;
}

// mirror of log1p_reduce (mode 3): exact reduction on q, no Sterbenz division.
__device__ inline double d_log1p_reduce(double q) {
    const double   u = 1.0 + q;
    const uint64_t b = d_bits(u);
    const double   m = d_from_bits((b & 0x000FFFFFFFFFFFFFULL) | ((uint64_t)1023 << 52));
    int k = (int)((b >> 52) & 0x7FFULL) - 1023;
    if (m >= 1.5) k += 1;
    const double pk  = d_from_bits((uint64_t)(k + 1023) << 52);
    const double ipk = d_from_bits((uint64_t)(1023 - k) << 52);
    const double f = (q - (pk - 1.0)) * ipk;
    const double s = f / (2.0 + f);
    const double z = s * s;
    const double w = z * z;
    double t1 = fma(w, D_NCL3_P[7], D_NCL3_P[5]);
    t1 = fma(w, t1, D_NCL3_P[3]);
    t1 = fma(w, t1, D_NCL3_P[1]);
    t1 = w * t1;
    double t2 = fma(w, D_NCL3_P[6], D_NCL3_P[4]);
    t2 = fma(w, t2, D_NCL3_P[2]);
    t2 = fma(w, t2, D_NCL3_P[0]);
    t2 = z * t2;
    const double R    = t2 + t1;
    const double hfsq = 0.5 * f * f;
    const double kd   = (double)k;
    return fma(kd, D_NCL_LN2_HI, -((hfsq - (fma(s, hfsq + R, kd * D_NCL_LN2_LO))) - f));
}

// mirror of log1p_pos
__device__ inline double d_log1p_pos(double q) {
#if NC_LOG1P_MODE == 2
    return d_log1p_nc(q);
#else
    return d_log1p_reduce(q);
#endif
}

// mirror of cell_of (12-cell chart)
__device__ inline int d_cell_of(double a) {
    int j = 0;
    while (j < g::NC_NCELL - 1 && a >= g::NC_A_EDGE[j + 1]) ++j;
    return j;
}

// mirror of clenshaw2_graded (12-cell chart; ragged, data-dependent bounds)
__device__ inline double d_clenshaw2_graded(int cell, double xt, double xa) {
    const int    nrow = g::NC_NROW[cell];
    const int    roff = g::NC_ROWOFF[cell];
    const double xa2  = 2.0 * xa;
    const double xt2  = 2.0 * xt;
    double g0 = 0.0, g1 = 0.0;
    for (int m = nrow - 1; m >= 1; --m) {
        const int nco = g::NC_ROWDEG[roff + m];
        const int cof = g::NC_ROWCOFF[roff + m];
        double e0 = 0.0, e1 = 0.0;
        for (int j = nco - 1; j >= 1; --j) {
            const double b = fma(xa2, e0, g::NC_COEFF[cof + j]) - e1;
            e1 = e0; e0 = b;
        }
        const double r = (nco > 0) ? (fma(xa, e0, g::NC_COEFF[cof]) - e1) : 0.0;
        const double b = fma(xt2, g0, r) - g1;
        g1 = g0; g0 = b;
    }
    {
        const int nco = g::NC_ROWDEG[roff];
        const int cof = g::NC_ROWCOFF[roff];
        double e0 = 0.0, e1 = 0.0;
        for (int j = nco - 1; j >= 1; --j) {
            const double b = fma(xa2, e0, g::NC_COEFF[cof + j]) - e1;
            e1 = e0; e0 = b;
        }
        const double r = (nco > 0) ? (fma(xa, e0, g::NC_COEFF[cof]) - e1) : 0.0;
        return fma(xt, g0, r) - g1;
    }
}

// mirror of near_variance_from_coords (12-cell chart), coordinates given
__device__ inline double d_near_variance_12(double a, double theta, double tau) {
    const int    cell = d_cell_of(a);
    const double t1   = g::NC_TAU_HI[cell];
    const double a0   = g::NC_A_EDGE[cell], a1 = g::NC_A_EDGE[cell + 1];
    const double xt   = fma(2.0, tau, -t1) / t1;
    const double xa   = fma(2.0, a, -(a0 + a1)) / (a1 - a0);
    const double v    = theta * d_clenshaw2_graded(cell, xt, xa);
    return v * v;
}

// mirror of near_variance_certified (12-cell, self-routing)
__device__ inline double d_near_variance_certified(double h, double c) {
    const double E     = d_nc_expm1(h);
    const double a     = d_log1p_pos(E / c);
    const double theta = h / a;
    const double tau   = theta * theta;
    return d_near_variance_12(a, theta, tau);
}

// the book chart from (h, c): w, or -1 when the quote is above the book's tau
__device__ inline double d_near_variance_book(double h, double c) {
    const double E     = d_nc_expm1(h);
    const double a     = d_log1p_pos(E / c);
    const double theta = h / a;
    const double tau   = theta * theta;
    return nb::nb_variance_from_coords(theta, tau, a);
}

// the RECURRENCE chart from (h, c): w, or -1 above its tau range.  Three divisions
// per quote (E/c, the log's own, h/A0); no theta, no cells, 25 fitted doubles.
__device__ inline double d_near_variance_rec(double h, double c) {
    const double E = d_nc_expm1(h);
    const double a = d_log1p_pos(E / c);
    return nr::nr_in_range(h, a) ? nr::nr_variance(h, a) : -1.0;
}

// the WHOLE-BOOK chart from (h, c): w, or -1 where the chart does not apply
// (theta above THETA_A below 2 pi, h above the local limit above it, EDGE inputs).
__device__ inline double d_wb_variance(double h, double c) {
    if (!(c > 0.0) || c >= 1.0 || !(h >= nc::NC_H_FLOOR) || !(h <= nw::NW_H_MAX)) return -1.0;
    const double E = nw::nw_expm1(h);
    const double a = d_log1p_pos(E / c);
    const int r = nw::nw_region(h, a);
    return r ? nw::nw_variance(h, a, r) : -1.0;
}

#if !NCG_HOST_CHECK
// The whole-book chart: NEAR, FAR and WING in one straight-line kernel with one
// branch at a = 2 pi.  w = -1 outside its domain.
__global__ void wb_kernel(const double* h, const double* c, double* w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        w[i] = d_wb_variance(h[i], c[i]);
}
// The recurrence chart: one interval, straight-line, 190 operations.  w = -1 above range.
__global__ void near_rec_kernel(const double* h, const double* c, double* w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        w[i] = d_near_variance_rec(h[i], c[i]);
}
// The 12-cell chart, self-routing (the faster of the two schedules measured before).
__global__ void near_cert_kernel_selfroute(const double* h, const double* c,
                                           double* w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        w[i] = d_near_variance_certified(h[i], c[i]);
}
// The book chart: straight-line, one cell, no divergence.  w = -1 above the book.
__global__ void near_book_kernel(const double* h, const double* c, double* w, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        w[i] = d_near_variance_book(h[i], c[i]);
}
// The sentinel pass: answer the w = -1 quotes with the 12-cell chart.
__global__ void near_fallback_kernel(const double* h, const double* c, double* w, int n,
                                     unsigned int* count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (w[i] < 0.0) { w[i] = d_near_variance_certified(h[i], c[i]); atomicAdd(count, 1u); }
}
#endif

// =============================================================================
//  Host side
// =============================================================================
static bool same_bits(double a, double b) {
    uint64_t x, y; std::memcpy(&x, &a, 8); std::memcpy(&y, &b, 8);
    return x == y;
}
static double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rec { double h, c, v; };
static bool load_truth(const char* p, std::vector<Rec>& out) {
    FILE* f = std::fopen(p, "rb"); if (!f) return false;
    int64_t n = 0; if (std::fread(&n, 8, 1, f) != 1) { std::fclose(f); return false; }
    out.resize((size_t)n);
    for (int64_t i = 0; i < n; ++i)
        if (std::fread(&out[(size_t)i], 8, 3, f) != 3) { std::fclose(f); return false; }
    std::fclose(f); return true;
}
static bool load_feed(const char* p, std::vector<double>& h, std::vector<double>& c) {
    FILE* f = std::fopen(p, "r"); if (!f) return false;
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        double a, b;
        if (std::sscanf(line, "%lf %lf", &a, &b) == 2 &&
            a > 1e-4 && a < 16.5 && b > 0 && b < 1 && 1 - b > 1e-16) { h.push_back(a); c.push_back(b); }
    }
    std::fclose(f); return true;
}

// The probe population, identical in construction to gate/vec_gate.cpp so the
// builds are compared on the same quotes.  The feed comes first (n_feed).
static void build_probes(std::vector<double>& H, std::vector<double>& C,
                         size_t& n_feed, std::vector<Rec>& tr) {
    load_feed("market_feed.csv", H, C);
    n_feed = H.size();
    if (load_truth("near_truth.bin", tr) ||
        load_truth("../gate/near_truth.bin", tr))
        for (const Rec& r : tr) { H.push_back(r.h); C.push_back(r.c); }
    for (int i = 0; i < 900; ++i) {
        double hh = std::exp(std::log(1e-9) + (std::log(0.29999) - std::log(1e-9)) * i / 899.0);
        for (int j = 0; j < 900; ++j) {
            double cc = std::exp2(-70.0 + 69.5 * j / 899.0);
            if (cc > 0 && cc < 1) { H.push_back(hh); C.push_back(cc); }
        }
    }
    for (int i = 0; i <= 400; ++i) {
        double hh = 1e-7 + (0.29999 - 1e-7) * i / 400.0;
        double c_ceil = std::expm1(hh) / std::expm1(hh / nc::NC_THETA);
        double c_wing = std::expm1(hh) / std::expm1(nc::NC_A_WING);
        for (double base : {c_ceil, c_wing})
            for (int d = -4; d <= 4; ++d) {
                double cc = base;
                for (int k = 0; k < std::abs(d); ++k) cc = std::nextafter(cc, d < 0 ? 0.0 : 1.0);
                if (cc > 0 && cc < 1) { H.push_back(hh); C.push_back(cc); }
            }
    }
}

int main(int argc, char** argv) {
    std::printf("=====================================================================\n");
    std::printf(" volfi v0.3.0-test  certified NEAR chart, CUDA twin  (book kernel)\n");
    std::printf("   mode     : %s\n", NCG_HOST_CHECK ? "HOST self-check (no GPU)" : "device");
    std::printf("   log1p    : mode %d\n", (int)NC_LOG1P_MODE);
    std::printf("   book     : layout L%d, %d cell(s), %d coefficients, tau <= %.3g, fit target %.1e\n",
                nb::NB_LAYOUT_ID, nb::NB_NCELL, nb::NB_NCOEFF, nb::NB_TAU_BOOK, nb::NB_FIT_EPS);
    std::printf("   12-cell  : %d cells, %d coefficients (fallback above the book)\n",
                nc::NC_NCELL, nc::NC_NCOEFF);
    std::printf("=====================================================================\n\n");

    std::vector<double> H, C; size_t n_feed = 0; std::vector<Rec> tr;
    build_probes(H, C, n_feed, tr);
    const size_t N = H.size();

    // ---- route on the host; the CPU reference is the book chart with fallback ---
    std::vector<char>   isnear(N, 0);
    std::vector<double> Wref(N, 0.0), Wrec(N, 0.0);
    long n_fb_rec = 0;
    std::vector<int>    order;      order.reserve(N);   // all NEAR-routed probes
    std::vector<int>    order_feed; order_feed.reserve(n_feed);
    long n_fb = 0, n_fb_feed = 0;
    for (size_t i = 0; i < N; ++i) {
        nc::near_coords z;
        if (nc::route_near_band(H[i], C[i], &z) != nc::NCR_NEAR) continue;
        isnear[i] = 1;
        int fb = 0;
        Wref[i] = nb::near_variance_book(H[i], C[i], &fb);
        n_fb += fb;
        int fbr = 0;
        Wrec[i] = nr::near_variance_rec(H[i], C[i], &fbr);
        n_fb_rec += fbr;
        order.push_back((int)i);
        if (i < n_feed) { order_feed.push_back((int)i); n_fb_feed += fb; }
    }
    const int M  = (int)order.size();
    const int MF = (int)order_feed.size();
    std::printf("probe population: %zu quotes, %d routed NEAR, %ld of them above the book, %ld above the recurrence range\n", N, M, n_fb, n_fb_rec);
    std::printf("feed subset     : %d NEAR quotes, %ld above the book\n\n", MF, n_fb_feed);

#if NCG_HOST_CHECK
    // ---- the port itself: device functions compiled for the host ------------
    long mism = 0; double worst = 0; int at = -1;
    for (int e = 0; e < M; ++e) {
        const int i = order[e];
        double got = d_near_variance_book(H[i], C[i]);
        if (got < 0.0) got = d_near_variance_certified(H[i], C[i]);
        if (!same_bits(got, Wref[i])) {
            ++mism;
            double d = std::fabs(got - Wref[i]) / std::fabs(Wref[i]);
            if (d > worst) { worst = d; at = i; }
        }
    }
    std::printf("[1] PORT CHECK device functions == CPU reference over %d NEAR quotes\n", M);
    std::printf("    mismatches = %ld\n", mism);
    if (mism) std::printf("      worst rel %.3e at h=%.17g c=%.17g\n", worst, H[at], C[at]);
    // the 12-cell mirror must still agree with its own CPU reference
    long mism2 = 0;
    for (int e = 0; e < M; ++e) {
        const int i = order[e];
        if (!same_bits(d_near_variance_certified(H[i], C[i]), nc::near_variance_certified(H[i], C[i]))) ++mism2;
    }
    std::printf("[2] 12-cell mirror == CPU 12-cell chart          : mismatches = %ld\n", mism2);
    long mism3 = 0;
    for (int e = 0; e < M; ++e) {
        const int i = order[e];
        double got = d_near_variance_rec(H[i], C[i]);
        if (got < 0.0) got = d_near_variance_certified(H[i], C[i]);
        if (!same_bits(got, Wrec[i])) ++mism3;
    }
    std::printf("[3] recurrence chart + fallback == CPU reference : mismatches = %ld\n", mism3);
    long mism4 = 0, cov = 0, codem = 0;
    for (size_t i = 0; i < N; ++i) {
        int code; const double ref = nw::implied_variance_wb(H[i], C[i], &code);
        const double got = d_wb_variance(H[i], C[i]);
        if ((got < 0.0) != (code == nw::NW_OUT)) ++codem;
        if (code != nw::NW_OUT) { ++cov; if (!same_bits(got, ref)) ++mism4; }
    }
    std::printf("[4] whole-book chart == CPU reference on the %ld covered of %zu probes : mismatches = %ld, domain disagreements = %ld\n\n",
                cov, N, mism4, codem);
    return (mism == 0 && mism2 == 0 && mism3 == 0 && mism4 == 0 && codem == 0) ? 0 : 1;
#else
    // ---- device ------------------------------------------------------------
    int dev = 0; cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device: %s, sm_%d%d, %d SMs\n\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    const long n_target = (argc > 1) ? std::atol(argv[1]) : 5000000L;
    const int  iters    = (argc > 2) ? std::atoi(argv[2]) : 300;

    // two tiles: the feed (the like-for-like comparison with the shipped NEAR slice)
    // and the full probe population (exercises the fallback pass)
    auto make_tile = [&](const std::vector<int>& idx, std::vector<double>& th, std::vector<double>& tc,
                         std::vector<int>& src) {
        const int reps = (int)std::max(1L, n_target / std::max<long>(1, (long)idx.size()));
        for (int r = 0; r < reps; ++r)
            for (int i : idx) { th.push_back(H[i]); tc.push_back(C[i]); src.push_back(i); }
    };
    // The benchmark feed is a shuffled year sample, so its warps are never
    // single-cell; a live book arrives strike-ordered within each expiry, along
    // which a is monotone.  A second feed tile sorted by a stands in for that.
    std::vector<int> order_sorted = order_feed;
    {
        std::vector<double> akey(N, 0.0);
        for (int i : order_feed) { nc::near_coords z; nc::route_near_band(H[i], C[i], &z); akey[i] = z.a; }
        std::sort(order_sorted.begin(), order_sorted.end(), [&](int x, int y) { return akey[x] < akey[y]; });
    }
    std::vector<double> fh, fc, sh, sc, ph, pc; std::vector<int> fsrc, ssrc, psrc;
    make_tile(order_feed,   fh, fc, fsrc);
    make_tile(order_sorted, sh, sc, ssrc);
    make_tile(order,        ph, pc, psrc);
    const int TF = (int)fh.size(), TS = (int)sh.size(), TP = (int)ph.size();

    double *dh, *dc, *dw; unsigned int* dcount;
    const int TMAX = std::max(TF, std::max(TS, TP));
    CUDA_CHECK(cudaMalloc(&dh, TMAX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dc, TMAX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dw, TMAX * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dcount, sizeof(unsigned int)));

    auto blocks_for = [](int n) { return std::min(65535, (n + 255) / 256); };
    auto run_book = [&](int n) { near_book_kernel<<<blocks_for(n), 256>>>(dh, dc, dw, n); };
    auto run_fb   = [&](int n) { near_fallback_kernel<<<blocks_for(n), 256>>>(dh, dc, dw, n, dcount); };
    auto run_12   = [&](int n) { near_cert_kernel_selfroute<<<blocks_for(n), 256>>>(dh, dc, dw, n); };
    auto run_rec  = [&](int n) { near_rec_kernel<<<blocks_for(n), 256>>>(dh, dc, dw, n); };

    // ---- bitwise cross-check on the probe tile (book + fallback) -------------
    CUDA_CHECK(cudaMemcpy(dh, ph.data(), TP * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, pc.data(), TP * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dcount, 0, sizeof(unsigned int)));
    run_book(TP); run_fb(TP);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> back(TP);
    unsigned int nfb_dev = 0;
    CUDA_CHECK(cudaMemcpy(back.data(), dw, TP * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&nfb_dev, dcount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    long mism = 0; double worst = 0; int at = -1;
    for (int e = 0; e < TP; ++e) {
        const int i = psrc[e];
        if (!same_bits(back[e], Wref[i])) {
            ++mism;
            double d = std::fabs(back[e] - Wref[i]) / std::fabs(Wref[i]);
            if (d > worst) { worst = d; at = i; }
        }
    }
    const int reps_p = TP / M;
    std::printf("[1] BIT-IDENTITY device (book + fallback) == CPU over %d NEAR quotes x %d : mismatches = %ld\n",
                M, reps_p, mism);
    std::printf("    fallback pass answered %u quotes (host expected %ld x %d = %ld)\n",
                nfb_dev, n_fb, reps_p, n_fb * reps_p);
    if (mism) std::printf("      worst rel %.3e at h=%.17g c=%.17g\n", worst, H[at], C[at]);
    {
        CUDA_CHECK(cudaMemset(dcount, 0, sizeof(unsigned int)));
        run_rec(TP); run_fb(TP);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> back2(TP); unsigned int nfb2 = 0;
        CUDA_CHECK(cudaMemcpy(back2.data(), dw, TP * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&nfb2, dcount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        long mm = 0; double ww = 0; int aa = -1;
        for (int e = 0; e < TP; ++e) {
            const int i = psrc[e];
            if (!same_bits(back2[e], Wrec[i])) { ++mm; double d = std::fabs(back2[e] - Wrec[i]) / std::fabs(Wrec[i]); if (d > ww) { ww = d; aa = i; } }
        }
        std::printf("[1b] BIT-IDENTITY device (recurrence + fallback) == CPU over %d NEAR quotes x %d : mismatches = %ld\n", M, reps_p, mm);
        std::printf("     fallback pass answered %u quotes (host expected %ld x %d = %ld)\n", nfb2, n_fb_rec, reps_p, n_fb_rec * reps_p);
        if (mm) std::printf("      worst rel %.3e at h=%.17g c=%.17g\n", ww, H[aa], C[aa]);
        const double EPS = 2.220446049250313e-16;
        double wr = 0; long used = 0, bad = 0;
        for (size_t k = 0; k < tr.size(); ++k) {
            const size_t i = n_feed + k;
            if (!isnear[i]) continue;
            int fbr = 0; nr::near_variance_rec(H[i], C[i], &fbr);
            if (fbr) continue;
            ++used;
            double dv = -1;
            for (int e = 0; e < M; ++e) if (psrc[e] == (int)i) { dv = back2[e]; break; }
            const double err = std::fabs(std::sqrt(dv) - tr[k].v) / tr[k].v;
            if (err > 1e-15) ++bad; if (err > wr) wr = err;
        }
        std::printf("[2b] ACCURACY device recurrence path, truth points in range (n=%ld): max %.3e (%.2f ULP), pts>1e-15 = %ld\n", used, wr, wr / EPS, bad);
        mism += mm;
    }

    // ---- accuracy through the device path, on the truth subset -------------
    if (!tr.empty()) {
        const double EPS = 2.220446049250313e-16;
        double wr = 0, wr_book = 0; long used = 0, bad = 0, used_book = 0;
        std::vector<double> devw(N, 0.0);
        for (int e = 0; e < M; ++e) devw[psrc[e]] = back[e];     // first replica
        for (size_t k = 0; k < tr.size(); ++k) {
            const size_t i = n_feed + k;
            if (!isnear[i]) continue;
            ++used;
            const double err = std::fabs(std::sqrt(devw[i]) - tr[k].v) / tr[k].v;
            if (err > 1e-15) ++bad;
            if (err > wr) wr = err;
            int fb = 0; nb::near_variance_book(H[i], C[i], &fb);
            if (!fb) { ++used_book; if (err > wr_book) wr_book = err; }
        }
        std::printf("[2] ACCURACY device path, 40-digit truth set (n=%ld): max %.3e (%.2f ULP), pts>1e-15 = %ld\n",
                    used, wr, wr / EPS, bad);
        std::printf("    of which through the book kernel (n=%ld): max %.3e (%.2f ULP)\n",
                    used_book, wr_book, wr_book / EPS);
    }

    // ---- timing -------------------------------------------------------------
    auto time_it = [&](const char* label, int n, auto&& launch) {
        CUDA_CHECK(cudaDeviceSynchronize());
        const double t0 = now_s();
        for (int it = 0; it < iters; ++it) launch(n);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double t1 = now_s();
        std::printf("    %-52s : %.4f ns/quote\n", label, 1e9 * (t1 - t0) / ((double)n * iters));
    };

    std::printf("\n[3] THROUGHPUT, feed tile in file order, a SHUFFLED sample (%d quotes x %d iters; the like-for-like NEAR slice)\n", TF, iters);
    CUDA_CHECK(cudaMemcpy(dh, fh.data(), TF * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, fc.data(), TF * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dcount, 0, sizeof(unsigned int)));
    time_it("RECURRENCE kernel alone", TF, run_rec);
    time_it("recurrence kernel + fallback pass (nothing to do)", TF, [&](int n) { run_rec(n); run_fb(n); });
    time_it("book kernel alone", TF, run_book);
    time_it("book kernel + fallback pass (nothing to do)", TF, [&](int n) { run_book(n); run_fb(n); });
    time_it("12-cell chart, self-routing (previous design)", TF, run_12);
    unsigned int nfb_feed = 0;
    CUDA_CHECK(cudaMemcpy(&nfb_feed, dcount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::printf("    (fallback pass on the feed tile answered %u quotes over %d iters)\n", nfb_feed, iters);

    std::printf("\n[3b] THROUGHPUT, feed tile SORTED BY a (%d quotes x %d iters; a strike ladder's warp uniformity)\n", TS, iters);
    CUDA_CHECK(cudaMemcpy(dh, sh.data(), TS * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, sc.data(), TS * sizeof(double), cudaMemcpyHostToDevice));
    time_it("RECURRENCE kernel alone", TS, run_rec);
    time_it("book kernel alone", TS, run_book);
    time_it("12-cell chart, self-routing (previous design)", TS, run_12);

    std::printf("\n[4] THROUGHPUT, probe tile (%d quotes x %d iters; %.1f%% above the book)\n",
                TP, iters, 100.0 * n_fb / M);
    CUDA_CHECK(cudaMemcpy(dh, ph.data(), TP * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, pc.data(), TP * sizeof(double), cudaMemcpyHostToDevice));
    time_it("RECURRENCE kernel + fallback pass", TP, [&](int n) { run_rec(n); run_fb(n); });
    time_it("book kernel alone (sentinels written, not answered)", TP, run_book);
    time_it("book kernel + fallback pass", TP, [&](int n) { run_book(n); run_fb(n); });
    time_it("12-cell chart, self-routing (previous design)", TP, run_12);

    // ---- [5] the WHOLE-BOOK kernel: bit-identity on the full feed, then timing ----
    {
        auto run_wb = [&](int n) { wb_kernel<<<blocks_for(n), 256>>>(dh, dc, dw, n); };
        // full-feed tile: every quote of the feed, all routes
        std::vector<int> order_full; for (size_t i = 0; i < n_feed; ++i) order_full.push_back((int)i);
        std::vector<double> uh, uc; std::vector<int> usrc;
        make_tile(order_full, uh, uc, usrc);
        const int TU = (int)uh.size();
        double *eh, *ec, *ew;
        CUDA_CHECK(cudaMalloc(&eh, TU * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ec, TU * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ew, TU * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(eh, uh.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ec, uc.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
        wb_kernel<<<blocks_for(TU), 256>>>(eh, ec, ew, TU);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> back3(TU);
        CUDA_CHECK(cudaMemcpy(back3.data(), ew, TU * sizeof(double), cudaMemcpyDeviceToHost));
        long mm = 0, cov = 0, cA = 0, cB = 0, cO = 0, codem = 0;
        for (int e = 0; e < TU; ++e) {
            const int i = usrc[e];
            int code; const double ref = nw::implied_variance_wb(H[i], C[i], &code);
            if (code == nw::NW_A) ++cA; else if (code == nw::NW_B) ++cB; else ++cO;
            if ((back3[e] < 0.0) != (code == nw::NW_OUT)) ++codem;
            if (code != nw::NW_OUT) { ++cov; if (!same_bits(back3[e], ref)) ++mm; }
        }
        std::printf("\n[5] WHOLE-BOOK kernel on the FULL feed tile (%d quotes, all routes): region A %ld, B %ld, outside %ld\n", TU, cA, cB, cO);
        std::printf("    BIT-IDENTITY device == CPU on the %ld covered quotes : mismatches = %ld, domain disagreements = %ld\n", cov, mm, codem);
        mism += mm + codem;
        // timing: full feed, NEAR feed tile, sorted NEAR tile
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0 = now_s();
        for (int it = 0; it < iters; ++it) wb_kernel<<<blocks_for(TU), 256>>>(eh, ec, ew, TU);
        CUDA_CHECK(cudaDeviceSynchronize());
        double t1 = now_s();
        std::printf("    %-52s : %.4f ns/quote   (shipped full book, same class of host: 0.078)\n",
                    "whole-book kernel, FULL feed, all routes", 1e9 * (t1 - t0) / ((double)TU * iters));
        CUDA_CHECK(cudaMemcpy(dh, fh.data(), TF * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dc, fc.data(), TF * sizeof(double), cudaMemcpyHostToDevice));
        time_it("whole-book kernel, NEAR feed tile (vs recurrence above)", TF, run_wb);
        CUDA_CHECK(cudaMemcpy(dh, sh.data(), TS * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dc, sc.data(), TS * sizeof(double), cudaMemcpyHostToDevice));
        time_it("whole-book kernel, NEAR feed tile sorted by a", TS, run_wb);
        // [5b] the FULL feed tile sorted once by a (the persistent-book deployment: region A and B
        //      quotes no longer share warps), same TU quotes, same kernel
        {
            std::vector<int> ord(TU); for (int e = 0; e < TU; ++e) ord[e] = e;
            std::vector<double> av(TU);
            for (int e = 0; e < TU; ++e) { const double E = nw::nw_expm1(uh[e]); av[e] = (uc[e] > 0.0 && uc[e] < 1.0) ? std::log1p(E / uc[e]) : 1e300; }
            std::sort(ord.begin(), ord.end(), [&](int x, int y) { return av[x] < av[y]; });
            std::vector<double> sh2(TU), sc2(TU);
            for (int e = 0; e < TU; ++e) { sh2[e] = uh[ord[e]]; sc2[e] = uc[ord[e]]; }
            CUDA_CHECK(cudaMemcpy(eh, sh2.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ec, sc2.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            double t0 = now_s();
            for (int it = 0; it < iters; ++it) wb_kernel<<<blocks_for(TU), 256>>>(eh, ec, ew, TU);
            CUDA_CHECK(cudaDeviceSynchronize());
            double t1 = now_s();
            std::printf("    %-52s : %.4f ns/quote   [5b] the sorted-book deployment\n",
                        "whole-book kernel, FULL feed SORTED by a", 1e9 * (t1 - t0) / ((double)TU * iters));
            // [5c] transfers included: upload h and c (pageable, blocking) + kernel + download w, per pass,
            //      the convention of the PDE method's published GPU timing
            t0 = now_s();
            for (int it = 0; it < iters; ++it) {
                CUDA_CHECK(cudaMemcpy(eh, sh2.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(ec, sc2.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
                wb_kernel<<<blocks_for(TU), 256>>>(eh, ec, ew, TU);
                CUDA_CHECK(cudaMemcpy(back3.data(), ew, TU * sizeof(double), cudaMemcpyDeviceToHost));
            }
            t1 = now_s();
            std::printf("    %-52s : %.4f ns/quote   [5c] uploads + kernel + readback, per pass\n",
                        "whole-book kernel, FULL feed sorted, WITH transfers", 1e9 * (t1 - t0) / ((double)TU * iters));
        }
        // [6] BRANCH-WISE: the whole-book kernel on region-filtered tiles of the feed (the shipped
        //     router's Near / Far / Wing), each tiled to TU quotes by cycling the subset, the same
        //     tiles pde_branch.cpp feeds to the PDE method.  Kernel-resident, and with transfers.
        {
            const char* RN[3] = { "Near", "Far", "Wing" };
            for (int r = 0; r < 3; ++r) {
                std::vector<double> rh, rc;
                for (size_t i = 0; i < n_feed; ++i) {
                    nc::near_coords zz; const int rt = nc::route_near_band(H[i], C[i], &zz);
                    const int rs = volfi_annulus::detail::grid_endpoint_route(H[i], C[i]);   // 1 near, 2 upper, 3 wing, else far
                    (void)rt; (void)zz;
                    const bool pick = (r == 0 && rs == 1) || (r == 1 && rs != 1 && rs != 2 && rs != 3) || (r == 2 && rs == 3);
                    if (pick) { rh.push_back(H[i]); rc.push_back(C[i]); }
                }
                if (rh.empty()) continue;
                std::vector<double> th(TU), tc(TU);
                for (int e = 0; e < TU; ++e) { th[e] = rh[e % rh.size()]; tc[e] = rc[e % rc.size()]; }
                CUDA_CHECK(cudaMemcpy(eh, th.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(ec, tc.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaDeviceSynchronize());
                double t0 = now_s();
                for (int it = 0; it < iters; ++it) wb_kernel<<<blocks_for(TU), 256>>>(eh, ec, ew, TU);
                CUDA_CHECK(cudaDeviceSynchronize());
                double t1 = now_s();
                const double kr = 1e9 * (t1 - t0) / ((double)TU * iters);
                t0 = now_s();
                for (int it = 0; it < iters; ++it) {
                    CUDA_CHECK(cudaMemcpy(eh, th.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(ec, tc.data(), TU * sizeof(double), cudaMemcpyHostToDevice));
                    wb_kernel<<<blocks_for(TU), 256>>>(eh, ec, ew, TU);
                    CUDA_CHECK(cudaMemcpy(back3.data(), ew, TU * sizeof(double), cudaMemcpyDeviceToHost));
                }
                t1 = now_s();
                std::printf("[6] whole-book kernel, %-5s tile (%zu feed quotes cycled to %d): kernel-resident %.4f ns/quote, with transfers %.4f\n",
                            RN[r], rh.size(), TU, kr, 1e9 * (t1 - t0) / ((double)TU * iters));
            }
        }
        CUDA_CHECK(cudaFree(eh)); CUDA_CHECK(cudaFree(ec)); CUDA_CHECK(cudaFree(ew));
    }

    std::printf("\n    shipped v0.2.4 on an H100 PCIe, same session as the 12-cell run:\n");
    std::printf("    NEAR 0.037, FAR 0.135, UPPER 0.163, WING 0.661 ns/quote; 12-cell self-routing 0.062.\n");

    CUDA_CHECK(cudaFree(dh)); CUDA_CHECK(cudaFree(dc)); CUDA_CHECK(cudaFree(dw)); CUDA_CHECK(cudaFree(dcount));
    return mism == 0 ? 0 : 1;
#endif
}
