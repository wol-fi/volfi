// pde_compare.cpp -- the PDE method of Matic, Radoicic and Stefanica (QF 20(3), 2020; MIT-licensed
// code at github.com/maticivan/PDE-method-for-implied-volatility) on OUR market feed, like for like
// with the whole-book chart and the routed inverter, in one binary.
//
// Their conventions: iv(FF, px, KK, rr, texp, PC) with k = log(KK/FF) - rr*texp and x = px/FF; with
// FF = 1, KK = e^h, rr = 0, texp = 1, PC = 1 (call) that is exactly our (h, c) and the result is
// sigma sqrt(T) = v.  evaluate(k, x) is their single-threaded scalar polynomial evaluation (returns
// -1 outside the partition); the vector iv() is OpenMP-parallel and carries an SR-formula fallback;
// ivGC() is the OpenCL path INCLUDING per-call uploads of six arrays and the readback, and
// evaluateGC(n) re-runs the two kernels on the uploaded inputs and reads the results back.
//
// Sections:
//   [1] accuracy on the 30,000-quote feed: PDE (CPU scalar, CPU vector) against our whole-book chart,
//       which is validated to 3 ULP against a 40-digit oracle, so at their 1e-7 class the reference
//       error is immaterial; coverage (their -1 returns)
//   [2] CPU timing on the feed: PDE evaluate() scalar; PDE iv() vector with OMP_NUM_THREADS=1 and
//       with all threads; our whole-book scalar and batch; our routed scalar and batch
//   [3] GPU (if built with PDE_GPU): PDE ivGC (published convention, transfers included) and
//       evaluateGC (kernels + readback) on the feed tiled to N; accuracy of the device results
// Build: see gpu/run_pde_gpu.sh.  Run from volfi_annulus/bench_run with loadPartition.txt and the
// two .cl files present in the working directory (their loader reads them from cwd).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <omp.h>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
#include "volfi_wb_vec.hpp"
#include "cImpVolBasic.h"
#ifdef PDE_GPU
#include "gcImpVolBasic.h"
#endif

namespace nw = volfi_wb;
volatile double g_sink = 0.0;
static double now_s() { using namespace std::chrono; return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n ? (n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0; }
static int env_int(const char* n, int f) { if (const char* r = std::getenv(n)) { int v = std::atoi(r); if (v > 0) return v; } return f; }

int main(int argc, char** argv) {
    const long n_gpu = (argc > 1) ? std::atol(argv[1]) : 4980000L;   // tile size for the device (multiple of 256)
    const int  iters = (argc > 2) ? std::atoi(argv[2]) : 20;
    const int  runs  = env_int("BENCH_RUNS", 7), rep = env_int("REPEATS", 4);

    std::vector<double> H, C; std::ifstream fin("market_feed.csv");
    if (!fin) { std::printf("market_feed.csv not found\n"); return 2; }
    std::string ln;
    while (std::getline(fin, ln)) {
        if (ln.empty() || ln[0] == '#') continue;
        std::istringstream is(ln); double h, c; if (!(is >> h >> c)) continue;
        if (!(h > 1e-4 && h < 16.5) || !(c > 0.0 && c < 1.0 && 1.0 - c > 1e-16)) continue;
        H.push_back(h); C.push_back(c);
    }
    const int N = (int)H.size();
    std::printf("=== PDE method vs whole-book chart vs routed inverter, market feed N=%d ===\n", N);

    // our reference: the whole-book chart (validated 3 ULP vs 40-digit oracle), sigma = sqrt(w)
    std::vector<double> Vref(N), Wtmp(N); std::vector<int> code(N);
    for (int i = 0; i < N; ++i) { int cd; Vref[i] = std::sqrt(nw::implied_variance_wb(H[i], C[i], &cd)); }

    double t0 = now_s();
    CImpVol pde;                                         // loads loadPartition.txt from cwd
    double t1 = now_s();
    std::vector<double> bounds = pde.getIntervalBounds();
    std::printf("PDE partition loaded in %.1f s: c in [%.3g, %.3g], k in [%.3g, %.3g]\n", t1 - t0, bounds[0], bounds[1], bounds[2], bounds[3]);

    // ---- [1] accuracy, CPU scalar evaluate(k, x) and vector iv()
    std::vector<double> Vpde(N), FF(N, 1.0), KK(N), RR(N, 0.0), TT(N, 1.0); std::vector<long> PC(N, 1);
    for (int i = 0; i < N; ++i) KK[i] = std::exp(H[i]);
    long out = 0; double wmax = 0, wsum = 0; int wat = 0;
    for (int i = 0; i < N; ++i) {
        const double s = pde.evaluate(H[i], C[i]);
        Vpde[i] = s;
        if (s < 0.0) { ++out; continue; }
        const double e = std::fabs(s - Vref[i]) / Vref[i]; wsum += e;
        if (e > wmax) { wmax = e; wat = i; }
    }
    std::printf("[1] PDE evaluate(k,x) scalar on the feed: outside partition %ld, max rel err in sigma %.3e at h=%.4g c=%.3e (v=%.4f), mean %.3e\n",
                out, wmax, H[wat], C[wat], Vref[wat], wsum / (N - out));
    std::vector<double> Viv = pde.iv(FF, C, KK, RR, TT, PC, 0);
    double vmax = 0, vsum = 0; long vbad = 0; int vat = 0;
    for (int i = 0; i < N; ++i) {
        if (!(Viv[i] > 0.0) || !std::isfinite(Viv[i])) { ++vbad; continue; }
        const double e = std::fabs(Viv[i] - Vref[i]) / Vref[i]; vsum += e; if (e > vmax) { vmax = e; vat = i; }
    }
    std::printf("    PDE iv() vector (with its SR fallback): non-finite/non-positive %ld, max rel err %.3e at h=%.4g c=%.3e, mean %.3e\n",
                vbad, vmax, H[vat], C[vat], vsum / (N - vbad));
    std::printf("    first quotes: h=%.5f c=%.5e  ref v=%.10f  evaluate=%.10f  iv=%.10f\n", H[0], C[0], Vref[0], Vpde[0], Viv[0]);
    // our own accuracy statement on the same feed is in wb_gate / the paper: 3 ULP against a 40-digit oracle.

    // ---- [2] CPU timing on the feed, ns/quote, median over runs of rep sweeps
    auto bench = [&](auto&& fn) {
        std::vector<double> t; for (int r = 0; r < runs; ++r) { double a = now_s(); for (int k = 0; k < rep; ++k) fn(); double b = now_s(); t.push_back(1e9 * (b - a) / ((double)rep * N)); }
        return median(t);
    };
    const double tE = bench([&]() { double s = 0; for (int i = 0; i < N; ++i) s += pde.evaluate(H[i], C[i]); g_sink += s; });
    omp_set_num_threads(1);
    const double tV1 = bench([&]() { std::vector<double> v = pde.iv(FF, C, KK, RR, TT, PC, 0); g_sink += v[0]; });
    const int nth = omp_get_max_threads(); omp_set_num_threads(omp_get_num_procs());
    const double tVn = bench([&]() { std::vector<double> v = pde.iv(FF, C, KK, RR, TT, PC, 0); g_sink += v[0]; });
    const int nthall = omp_get_max_threads(); (void)nth;
    const double tWS = bench([&]() { double s = 0; for (int i = 0; i < N; ++i) { int cd; s += nw::implied_variance_wb(H[i], C[i], &cd); } g_sink += s; });
    const double tWB = bench([&]() { nw::implied_variance_wb_batch(H.data(), C.data(), Wtmp.data(), code.data(), N); g_sink += Wtmp[0]; });
    const double tRS = bench([&]() { double s = 0; for (int i = 0; i < N; ++i) s += volfi_annulus::implied_variance_otm(H[i], C[i]); g_sink += s; });
    const double tRB = bench([&]() { volfi_annulus::implied_variance_grid_batch(H.data(), C.data(), Wtmp.data(), N); g_sink += Wtmp[0]; });
    std::printf("[2] CPU, ns/quote on the feed (median of %d x %d):\n", runs, rep);
    std::printf("    PDE evaluate() scalar, 1 thread            %8.2f\n", tE);
    std::printf("    PDE iv() vector, OpenMP 1 thread           %8.2f   (includes its input transform and SR fallback)\n", tV1);
    std::printf("    PDE iv() vector, OpenMP %2d threads         %8.2f   (their published CPU convention)\n", nthall, tVn);
    std::printf("    whole-book chart scalar / batch            %8.2f / %8.2f\n", tWS, tWB);
    std::printf("    routed inverter scalar / batch             %8.2f / %8.2f\n", tRS, tRB);

#ifdef PDE_GPU
    // ---- [3] GPU: their OpenCL path on the feed tiled to n_gpu
    const long M = (n_gpu / 256) * 256;
    cl_double *gFF = new cl_double[M], *gpx = new cl_double[M], *gKK = new cl_double[M], *grr = new cl_double[M], *gtx = new cl_double[M]; myint* gPC = new myint[M];
    std::vector<int> src(M);
    for (long i = 0; i < M; ++i) { const int j = (int)(i % N); src[i] = j; gFF[i] = 1.0; gpx[i] = C[j]; gKK[i] = KK[j]; grr[i] = 0.0; gtx[i] = 1.0; gPC[i] = 1; }
    double g0 = now_s();
    GCImpVol pdeGC;                                      // OpenCL init + partition upload (needs parallelImpVol.cl, gc_parallel_rnum.cl in cwd)
    double g1 = now_s();
    std::printf("[3] GPU: OpenCL device initialised, partition uploaded in %.1f s; tile %ld quotes\n", g1 - g0, M);
    cl_double* res = pdeGC.ivGC(gFF, gpx, gKK, grr, gtx, gPC, M);   // warm-up (also uploads the inputs)
    long gout = 0; double gmax = 0, gsum = 0;
    for (long i = 0; i < M; ++i) { const double s = res[i]; if (!(s > 0.0) || !std::isfinite(s)) { ++gout; continue; } const double e = std::fabs(s - Vref[src[i]]) / Vref[src[i]]; gsum += e; if (e > gmax) gmax = e; }
    std::printf("    device results vs our reference: non-finite/non-positive %ld, max rel err %.3e, mean %.3e\n", gout, gmax, gsum / (M - gout));
    delete[] res;
    // (a) published convention: ivGC per call, uploads + kernels + readback
    std::vector<double> ta, tb;
    for (int r = 0; r < 3; ++r) {
        double a = now_s();
        for (int it = 0; it < iters; ++it) { cl_double* q = pdeGC.ivGC(gFF, gpx, gKK, grr, gtx, gPC, M); g_sink += q[0]; delete[] q; }
        double b = now_s(); ta.push_back(1e9 * (b - a) / ((double)iters * M));
        a = now_s();
        for (int it = 0; it < iters; ++it) { cl_double* q = pdeGC.evaluateGC(M); g_sink += q[0]; delete[] q; }
        b = now_s(); tb.push_back(1e9 * (b - a) / ((double)iters * M));
    }
    std::printf("    PDE ivGC, uploads + 2 kernels + readback (published convention)  %.4f ns/quote  (%d iters, median of 3)\n", median(ta), iters);
    std::printf("    PDE evaluateGC, 2 kernels + readback only                        %.4f ns/quote\n", median(tb));
    std::printf("    compare: whole-book kernel on the H100, kernel-resident 0.0724 (file order) / transfers-inclusive line [5c] of the CUDA driver\n");
#endif
    return 0;
}
