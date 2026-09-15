// wb_lbr_bench.cpp -- LIKE-FOR-LIKE market-feed timing in ONE binary, ONE loop, ONE session:
//   B  Let's Be Rational, the author's public scalar implementation, compiled from its sources
//      with the identical flags into this binary
//   C  the routed inverter, scalar entry            (volfi_annulus::implied_variance_otm)
//   D  the routed inverter, batch (speculative)     (volfi_annulus::implied_variance_grid_batch)
//   E  the whole-book chart, scalar entry           (volfi_wb::implied_variance_wb, shipped fallback inside)
//   F  the whole-book chart, batch                  (volfi_wb::implied_variance_wb_batch)
// Every method starts from the raw (h, c) pair inside the timed loop (all-inside): LBR pays
// its beta = c e^{-h/2} transform, the scalar entries build their per-quote context, the
// batch drivers take the raw arrays.  Median over BENCH_RUNS (15) timing passes of REPEATS (8)
// sweeps, the protocol of bench_run/benchmark_vec.cpp, on the full 30,000-quote feed and on
// contiguous chunks of 64 quotes (the small-batch regime).
// Build twice, the second with -flto (cross-TU inlining allowed, so LBR's call boundary is
// removed as well): if B does not move, the call boundary is not what the table measures.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
#include "volfi_wb_vec.hpp"
#include "lets_be_rational.h"

namespace nw = volfi_wb;
volatile double g_sink = 0.0;
static int env_int(const char* n, int f) { if (const char* r = std::getenv(n)) { int v = std::atoi(r); if (v > 0) return v; } return f; }
template<class F> static double bench(int n, int rep, F&& fn) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0; auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < rep; r++) { int i = 0; for (; i + 3 < n; i += 4) { s0 += fn(i); s1 += fn(i + 1); s2 += fn(i + 2); s3 += fn(i + 3); } for (; i < n; i++) s0 += fn(i); }
    auto t1 = std::chrono::steady_clock::now(); g_sink += s0 + s1 + s2 + s3;
    return 1e9 * std::chrono::duration<double>(t1 - t0).count() / ((double)rep * n);
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n ? (n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0; }
static double quant(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); if (v.empty()) return 0; double pos = p * (v.size() - 1); size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos); double w = pos - lo; return v[lo] * (1 - w) + v[hi] * w; }
static void rpt(const char* g, const char* nm, std::vector<double>& s) {
    std::printf("TIMING %-8s %-26s median=%8.2f  IQR=[%8.2f,%8.2f]  ns/eval\n", g, nm, median(s), quant(s, 0.25), quant(s, 0.75));
}

int main() {
#if defined(__AVX512F__)
    const char* isa = "AVX-512";
#elif defined(__AVX2__)
    const char* isa = "AVX2";
#else
    const char* isa = "scalar";
#endif
    const int runs = env_int("BENCH_RUNS", 15), rep = env_int("REPEATS", 8);
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
    std::printf("=== LIKE-FOR-LIKE market feed, ISA=%s, N=%d, %d runs x %d repeats, one binary%s ===\n", isa, N, runs, rep,
#ifdef LTO_BUILD
                " (-flto)"
#else
                ""
#endif
    );
    // bit-identity of both batch drivers against their scalar entries, on this feed
    std::vector<double> wd(N), wf(N), wtmp(N); std::vector<int> code(N);
    volfi_annulus::implied_variance_grid_batch(H.data(), C.data(), wd.data(), N);
    nw::implied_variance_wb_batch(H.data(), C.data(), wf.data(), code.data(), N);
    long mD = 0, mF = 0, cA = 0, cB = 0, cO = 0;
    for (int i = 0; i < N; ++i) {
        const double sr = volfi_annulus::implied_variance_otm(H[i], C[i]); int cd; const double sw = nw::implied_variance_wb(H[i], C[i], &cd);
        if (std::memcmp(&sr, &wd[i], 8)) ++mD; if (std::memcmp(&sw, &wf[i], 8)) ++mF;
        if (code[i] == 1) ++cA; else if (code[i] == 2) ++cB; else ++cO;
    }
    std::printf("  batch == scalar: routed mismatches=%ld  whole-book mismatches=%ld  [both must be 0]\n", mD, mF);
    std::printf("  whole-book coverage: region A %ld, region B %ld, shipped fallback %ld\n", cA, cB, cO);

    std::vector<double> tB, tC, tD, tE, tF, tD64, tF64;
    for (int r = 0; r < runs; r++) {
        tB.push_back(bench(N, rep, [&](int i) { double s = NormalisedImpliedBlackVolatility(C[i] * std::exp(-0.5 * H[i]), -H[i], 1.0); return s * s; }));
        tC.push_back(bench(N, rep, [&](int i) { return volfi_annulus::implied_variance_otm(H[i], C[i]); }));
        tE.push_back(bench(N, rep, [&](int i) { int cd; return nw::implied_variance_wb(H[i], C[i], &cd); }));
        auto t0 = std::chrono::steady_clock::now();
        for (int rr = 0; rr < rep; rr++) volfi_annulus::implied_variance_grid_batch(H.data(), C.data(), wtmp.data(), N);
        auto t1 = std::chrono::steady_clock::now(); g_sink += wtmp[0];
        tD.push_back(1e9 * std::chrono::duration<double>(t1 - t0).count() / ((double)rep * N));
        t0 = std::chrono::steady_clock::now();
        for (int rr = 0; rr < rep; rr++) nw::implied_variance_wb_batch(H.data(), C.data(), wtmp.data(), code.data(), N);
        t1 = std::chrono::steady_clock::now(); g_sink += wtmp[0];
        tF.push_back(1e9 * std::chrono::duration<double>(t1 - t0).count() / ((double)rep * N));
        // small batches: the same feed in contiguous chunks of 64
        t0 = std::chrono::steady_clock::now();
        for (int rr = 0; rr < rep; rr++) for (int b = 0; b + 64 <= N; b += 64) volfi_annulus::implied_variance_grid_batch(H.data() + b, C.data() + b, wtmp.data() + b, 64);
        t1 = std::chrono::steady_clock::now(); g_sink += wtmp[0];
        tD64.push_back(1e9 * std::chrono::duration<double>(t1 - t0).count() / ((double)rep * (N / 64 * 64)));
        t0 = std::chrono::steady_clock::now();
        for (int rr = 0; rr < rep; rr++) for (int b = 0; b + 64 <= N; b += 64) nw::implied_variance_wb_batch(H.data() + b, C.data() + b, wtmp.data() + b, code.data() + b, 64);
        t1 = std::chrono::steady_clock::now(); g_sink += wtmp[0];
        tF64.push_back(1e9 * std::chrono::duration<double>(t1 - t0).count() / ((double)rep * (N / 64 * 64)));
    }
    rpt("market", "B_lbr_scalar", tB);
    rpt("market", "C_routed_scalar", tC);
    rpt("market", "D_routed_batch", tD);
    rpt("market", "E_wholebook_scalar", tE);
    rpt("market", "F_wholebook_batch", tF);
    rpt("market", "D_routed_batch_64", tD64);
    rpt("market", "F_wholebook_batch_64", tF64);
    std::printf("  ratios vs LBR: routed scalar %.2f  routed batch %.2f  whole-book scalar %.2f  whole-book batch %.2f  (64-chunks: routed %.2f  whole-book %.2f)\n\n",
                median(tB) / median(tC), median(tB) / median(tD), median(tB) / median(tE), median(tB) / median(tF), median(tB) / median(tD64), median(tB) / median(tF64));
    return (mD == 0 && mF == 0) ? 0 : 1;
}
