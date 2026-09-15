// pde_branch.cpp -- BRANCH-WISE latency, both methods on the SAME region-filtered tiles of the feed:
// the feed split by the shipped router into Near / Far / Wing, each subset tiled to the same size.
//   CPU: PDE evaluate() scalar (1 thread); whole-book chart scalar and batch; routed scalar and batch
//   GPU (-DPDE_GPU): PDE ivGC (uploads + kernels + readback) and evaluateGC (kernels + readback) per tile;
//        our per-tile device numbers come from the CUDA driver's [6] section on the identical tiles.
// Run from bench_run with loadPartition.txt and the .cl files present.  Args: tile size, iters.
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
enum { R_WING = 0, R_NEAR = 1, R_FAR = 2, R_UPPER = 3, R_EDGE = 4 };
static const char* RN[5] = { "Wing", "Near", "Far", "Upper", "Edge" };
static int route(double h, double c) {
    using namespace volfi_annulus;
    if (!(c > 0.0) || c >= 1.0 || !(h > 0.0)) return R_EDGE;
    switch (detail::grid_endpoint_route(h, c)) { case 1: return R_NEAR; case 2: return R_UPPER; case 3: return R_WING; default: return R_FAR; }
}
int main(int argc, char** argv) {
    const long tile = (argc > 1) ? std::atol(argv[1]) : 30000L;      // quotes per region tile (CPU default: feed-sized)
    const int iters = (argc > 2) ? std::atoi(argv[2]) : 20;
    const int runs = 7, rep = 4;
    std::vector<double> H, C; std::ifstream fin("market_feed.csv"); std::string ln;
    while (std::getline(fin, ln)) { if (ln.empty() || ln[0] == '#') continue; std::istringstream is(ln); double h, c; if (!(is >> h >> c)) continue;
        if (!(h > 1e-4 && h < 16.5) || !(c > 0.0 && c < 1.0 && 1.0 - c > 1e-16)) continue; H.push_back(h); C.push_back(c); }
    CImpVol pde;
#ifdef PDE_GPU
    GCImpVol pdeGC;
#endif
    std::printf("=== BRANCH-WISE latency on region-filtered tiles of %ld quotes (feed subsets cycled), ns/quote ===\n", tile);
    std::printf("  %-6s %8s | %10s | %10s %10s | %10s %10s", "region", "feed n", "PDE scalar", "wb scalar", "wb batch", "rt scalar", "rt batch");
#ifdef PDE_GPU
    std::printf(" | %12s %12s", "PDE ivGC", "PDE evalGC");
#endif
    std::printf("\n");
    for (int r : { R_NEAR, R_FAR, R_WING }) {
        std::vector<double> sh, sc; for (size_t i = 0; i < H.size(); ++i) if (route(H[i], C[i]) == r) { sh.push_back(H[i]); sc.push_back(C[i]); }
        const long n0 = (long)sh.size(); if (n0 == 0) continue;
        const long M = (tile / 256) * 256;
        std::vector<double> th(M), tc(M), tw(M); std::vector<int> code(M);
        for (long i = 0; i < M; ++i) { th[i] = sh[i % n0]; tc[i] = sc[i % n0]; }
        auto bench = [&](auto&& fn) { std::vector<double> t; for (int q = 0; q < runs; ++q) { double a = now_s(); for (int k = 0; k < rep; ++k) fn(); double b = now_s(); t.push_back(1e9 * (b - a) / ((double)rep * M)); } return median(t); };
        const double tP = bench([&]() { double s = 0; for (long i = 0; i < M; ++i) s += pde.evaluate(th[i], tc[i]); g_sink += s; });
        const double tWS = bench([&]() { double s = 0; for (long i = 0; i < M; ++i) { int cd; s += nw::implied_variance_wb(th[i], tc[i], &cd); } g_sink += s; });
        const double tWB = bench([&]() { nw::implied_variance_wb_batch(th.data(), tc.data(), tw.data(), code.data(), (int)M); g_sink += tw[0]; });
        const double tRS = bench([&]() { double s = 0; for (long i = 0; i < M; ++i) s += volfi_annulus::implied_variance_otm(th[i], tc[i]); g_sink += s; });
        const double tRB = bench([&]() { volfi_annulus::implied_variance_grid_batch(th.data(), tc.data(), tw.data(), (int)M); g_sink += tw[0]; });
        std::printf("  %-6s %8ld | %10.2f | %10.2f %10.2f | %10.2f %10.2f", RN[r], n0, tP, tWS, tWB, tRS, tRB);
#ifdef PDE_GPU
        cl_double *gFF = new cl_double[M], *gpx = new cl_double[M], *gKK = new cl_double[M], *grr = new cl_double[M], *gtx = new cl_double[M]; myint* gPC = new myint[M];
        for (long i = 0; i < M; ++i) { gFF[i] = 1.0; gpx[i] = tc[i]; gKK[i] = std::exp(th[i]); grr[i] = 0.0; gtx[i] = 1.0; gPC[i] = 1; }
        cl_double* w0 = pdeGC.ivGC(gFF, gpx, gKK, grr, gtx, gPC, M); delete[] w0;
        std::vector<double> ta, tb;
        for (int q = 0; q < 3; ++q) {
            double a = now_s(); for (int it = 0; it < iters; ++it) { cl_double* w = pdeGC.ivGC(gFF, gpx, gKK, grr, gtx, gPC, M); g_sink += w[0]; delete[] w; } double b = now_s(); ta.push_back(1e9 * (b - a) / ((double)iters * M));
            a = now_s(); for (int it = 0; it < iters; ++it) { cl_double* w = pdeGC.evaluateGC(M); g_sink += w[0]; delete[] w; } b = now_s(); tb.push_back(1e9 * (b - a) / ((double)iters * M));
        }
        std::printf(" | %12.4f %12.4f", median(ta), median(tb));
        delete[] gFF; delete[] gpx; delete[] gKK; delete[] grr; delete[] gtx; delete[] gPC;
#endif
        std::printf("\n");
    }
    return 0;
}
