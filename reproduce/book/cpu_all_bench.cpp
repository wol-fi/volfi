// cpu_all_bench.cpp -- THE CPU like-for-like table in ONE binary, ONE loop, ONE session:
//   LBR   Let's Be Rational, the author's public scalar implementation at its Makefile's release flags
//   PDE   the PDE table method of Matic/Radoicic/Stefanica (their code, their flags): evaluate() scalar,
//         and iv() with one OpenMP thread (their vector entry, with its input transform and SR fallback)
//   RT    the routed inverter, scalar entry and speculative batch
//   WB    the whole-book chart, scalar entry and batch
// Section [1]: the full 30,000-quote feed, all methods, plus the two batch drivers in 64-quote chunks.
// Section [2]: BRANCH-WISE on region-filtered tiles (shipped router's Near / Far / Wing), each tile the
//              feed's own size (30,000, the subset cycled), all methods.
// Every method starts from the raw (h, c) pair inside the timed loop.  Median over BENCH_RUNS (7)
// timing passes of REPEATS (4) sweeps: a few seconds per method, enough on a quiet machine, and
// the reference and the PDE method are several times slower per quote than the batch paths.
// Bit-identity of both batch drivers against their scalar entries is re-checked before timing.
// Build: gate/run_cpu_all.sh.  Run from bench_run (market_feed.csv, loadPartition.txt in cwd).
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
#include "lets_be_rational.h"
#include "cImpVolBasic.h"

namespace nw = volfi_wb;
volatile double g_sink = 0.0;
static int env_int(const char* n, int f) { if (const char* r = std::getenv(n)) { int v = std::atoi(r); if (v > 0) return v; } return f; }
static double now_s() { using namespace std::chrono; return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n ? (n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0; }
enum { R_WING = 0, R_NEAR = 1, R_FAR = 2, R_UPPER = 3, R_EDGE = 4 };
static const char* RN[5] = { "Wing", "Near", "Far", "Upper", "Edge" };
static int route(double h, double c) {
    using namespace volfi_annulus;
    if (!(c > 0.0) || c >= 1.0 || !(h > 0.0)) return R_EDGE;
    switch (detail::grid_endpoint_route(h, c)) { case 1: return R_NEAR; case 2: return R_UPPER; case 3: return R_WING; default: return R_FAR; }
}
struct Row { double lbr, pdeS, pdeV, rtS, rtB, wbS, wbB; };

static Row time_all(const std::vector<double>& H, const std::vector<double>& C, CImpVol& pde, int runs, int rep) {
    const int N = (int)H.size();
    std::vector<double> W(N), FF(N, 1.0), KK(N), RR(N, 0.0), TT(N, 1.0); std::vector<long> PC(N, 1); std::vector<int> code(N);
    for (int i = 0; i < N; ++i) KK[i] = std::exp(H[i]);
    auto bench = [&](auto&& fn) { std::vector<double> t; for (int r = 0; r < runs; ++r) { double a = now_s(); for (int k = 0; k < rep; ++k) fn(); double b = now_s(); t.push_back(1e9 * (b - a) / ((double)rep * N)); } return median(t); };
    Row o;
    o.lbr  = bench([&]() { double s = 0; for (int i = 0; i < N; ++i) { double v = NormalisedImpliedBlackVolatility(C[i] * std::exp(-0.5 * H[i]), -H[i], 1.0); s += v * v; } g_sink += s; });
    o.pdeS = bench([&]() { double s = 0; for (int i = 0; i < N; ++i) s += pde.evaluate(H[i], C[i]); g_sink += s; });
    omp_set_num_threads(1);
    o.pdeV = bench([&]() { std::vector<double> v = pde.iv(FF, C, KK, RR, TT, PC, 0); g_sink += v[0]; });
    o.rtS  = bench([&]() { double s = 0; for (int i = 0; i < N; ++i) s += volfi_annulus::implied_variance_otm(H[i], C[i]); g_sink += s; });
    o.rtB  = bench([&]() { volfi_annulus::implied_variance_grid_batch(H.data(), C.data(), W.data(), N); g_sink += W[0]; });
    o.wbS  = bench([&]() { double s = 0; for (int i = 0; i < N; ++i) { int cd; s += nw::implied_variance_wb(H[i], C[i], &cd); } g_sink += s; });
    o.wbB  = bench([&]() { nw::implied_variance_wb_batch(H.data(), C.data(), W.data(), code.data(), N); g_sink += W[0]; });
    return o;
}
static void print_row(const char* nm, long n, const Row& o) {
    std::printf("  %-10s %7ld | %8.2f | %8.2f %8.2f | %8.2f %8.2f | %8.2f %8.2f |  %5.2f %5.2f\n", nm, n, o.lbr, o.pdeS, o.pdeV, o.rtS, o.rtB, o.wbS, o.wbB, o.lbr / o.wbB, o.pdeS / o.wbB);
}

int main() {
#if defined(__AVX512F__)
    const char* isa = "AVX-512";
#elif defined(__AVX2__)
    const char* isa = "AVX2";
#else
    const char* isa = "scalar";
#endif
    const int runs = env_int("BENCH_RUNS", 7), rep = env_int("REPEATS", 4);
    std::vector<double> H, C; std::ifstream fin("market_feed.csv"); std::string ln;
    if (!fin) { std::printf("market_feed.csv not found\n"); return 2; }
    while (std::getline(fin, ln)) { if (ln.empty() || ln[0] == '#') continue; std::istringstream is(ln); double h, c; if (!(is >> h >> c)) continue;
        if (!(h > 1e-4 && h < 16.5) || !(c > 0.0 && c < 1.0 && 1.0 - c > 1e-16)) continue; H.push_back(h); C.push_back(c); }
    const int N = (int)H.size();
    CImpVol pde;
    std::printf("=== CPU like-for-like, ISA=%s, feed N=%d, %d runs x %d sweeps, one binary%s ===\n", isa, N, runs, rep,
#ifdef LTO_BUILD
        " (-flto)"
#else
        ""
#endif
    );
    // bit-identity of both batch drivers vs their scalar entries, on the feed
    std::vector<double> wd(N), wf(N); std::vector<int> code(N);
    volfi_annulus::implied_variance_grid_batch(H.data(), C.data(), wd.data(), N);
    nw::implied_variance_wb_batch(H.data(), C.data(), wf.data(), code.data(), N);
    long mD = 0, mF = 0;
    for (int i = 0; i < N; ++i) { const double sr = volfi_annulus::implied_variance_otm(H[i], C[i]); int cd; const double sw = nw::implied_variance_wb(H[i], C[i], &cd);
        if (std::memcmp(&sr, &wd[i], 8)) ++mD; if (std::memcmp(&sw, &wf[i], 8)) ++mF; }
    std::printf("  batch == scalar: routed mismatches=%ld  whole-book mismatches=%ld  [both must be 0]\n", mD, mF);
    std::printf("  ns/quote:  %-7s | %8s | %8s %8s | %8s %8s | %8s %8s |  %s\n", "tile", "LBR", "PDE scl", "PDE iv1", "RT scl", "RT batch", "WB scl", "WB batch", "LBR/WBb PDE/WBb");
    std::printf("[1] full feed\n");
    print_row("feed", N, time_all(H, C, pde, runs, rep));
    {   // the two batch drivers in 64-quote chunks (small-batch regime)
        std::vector<double> W(N); std::vector<int> cd(N);
        auto bench = [&](auto&& fn) { std::vector<double> t; for (int r = 0; r < runs; ++r) { double a = now_s(); for (int k = 0; k < rep; ++k) fn(); double b = now_s(); t.push_back(1e9 * (b - a) / ((double)rep * (N / 64 * 64))); } return median(t); };
        const double rb = bench([&]() { for (int b = 0; b + 64 <= N; b += 64) volfi_annulus::implied_variance_grid_batch(H.data() + b, C.data() + b, W.data() + b, 64); g_sink += W[0]; });
        const double wb = bench([&]() { for (int b = 0; b + 64 <= N; b += 64) nw::implied_variance_wb_batch(H.data() + b, C.data() + b, W.data() + b, cd.data() + b, 64); g_sink += W[0]; });
        std::printf("  %-10s %7d |          |                   |          %8.2f |          %8.2f |\n", "64-chunks", N, rb, wb);
    }
    std::printf("[2] branch-wise, region-filtered tiles of %d (subset cycled)\n", N);
    for (int r : { R_NEAR, R_FAR, R_WING }) {
        std::vector<double> sh, sc; for (int i = 0; i < N; ++i) if (route(H[i], C[i]) == r) { sh.push_back(H[i]); sc.push_back(C[i]); }
        if (sh.empty()) continue;
        const long n0 = (long)sh.size();
        std::vector<double> th(N), tc(N); for (int i = 0; i < N; ++i) { th[i] = sh[i % n0]; tc[i] = sc[i % n0]; }
        print_row(RN[r], n0, time_all(th, tc, pde, runs, rep));
    }
    return (mD == 0 && mF == 0) ? 0 : 1;
}
