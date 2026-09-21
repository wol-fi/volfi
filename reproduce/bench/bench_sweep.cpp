// bench_sweep.cpp -- batch-size sweep behind Figure 1 of the paper.
//
// Emits "n cold_ns warm2_ns lbr_ns" (ns per quote, median of 7 repetitions) for
// batch sizes 1..16384, which make_latency_batch.py turns into fig_latency_batch.
// The point of the figure is that the guarded warm-restart driver is flat in
// batch size while the cold driver's sort-and-drain only amortizes over a few
// thousand quotes, and that a single isolated quote does not vectorize at all.
//
// The generator for the original batchsweep.txt was never checked in, and that
// file's small-n rows are visibly unusable (the cold column reads 294 ns at n=4
// and 66 ns at n=8; the reference column wanders between 173 and 460).  The
// cause is timing a region far shorter than the clock's resolution: at n=1 one
// driver call is a few hundred nanoseconds.  This harness fixes that by
// calibrating the inner repetition count per batch size so every timed region
// runs at least MIN_MS milliseconds, and by rotating the slice offset across
// repetitions so no batch size is measured entirely from L1.
//
// Build (from this dir, LBR sources on the include/link line -- or use build_all.sh):
//   LBR=".../LetsBeRational"
//   g++ -std=c++17 -O3 -march=native -ffp-contract=off -fno-fast-math \
//       -funroll-loops -DNO_XL_API -w -I.. -I"$LBR" bench_sweep.cpp \
//       "$LBR"/{lets_be_rational,normaldistribution,rationalcubic,erf_cody}.cpp -o bench_sweep
//   taskset -c 2 nice -n -5 ./bench_sweep > batchsweep.txt
#include "paper_volfi.hpp"
#include "volfi_annulus_wing.hpp"
#include "volfi_annulus.hpp"
#include "lets_be_rational.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
volatile double g_sink = 0.0;
constexpr double MIN_MS = 20.0;      // minimum length of a timed region
constexpr int    REPS   = 7;         // repetitions per batch size; median reported

bool feasible(double c) { return c > 0.0 && c < 1.0; }

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t m = v.size();
    return (m & 1) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
}

// Times fn() with enough inner repetitions to cover MIN_MS, and returns ns per
// quote.  The calibration pass is discarded, so it doubles as a warm-up.
template <class F> double timed(int n, F&& fn) {
    int inner = 1;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < inner; ++r) fn();
        auto t1 = std::chrono::steady_clock::now();
        const double ms = 1e3 * std::chrono::duration<double>(t1 - t0).count();
        if (ms >= MIN_MS || inner > (1 << 24)) {
            auto s0 = std::chrono::steady_clock::now();
            for (int r = 0; r < inner; ++r) fn();
            auto s1 = std::chrono::steady_clock::now();
            return 1e9 * std::chrono::duration<double>(s1 - s0).count() / ((double)inner * n);
        }
        inner *= (ms < MIN_MS / 8.0) ? 8 : 2;
    }
}
}  // namespace

int main() {
    using namespace volfi_annulus;

    std::vector<double> h, c;
    std::ifstream fin("market_feed.csv");
    if (!fin) { std::printf("market_feed.csv not found in cwd -- aborting.\n"); return 1; }
    std::string ln;
    while (std::getline(fin, ln)) {
        if (ln.empty() || ln[0] == '#') continue;
        std::istringstream is(ln); double hh, cc;
        if (!(is >> hh >> cc)) continue;
        if (!(hh > 1e-4 && hh < 16.5) || !feasible(cc)) continue;
        h.push_back(hh); c.push_back(cc);
    }
    const int N = (int)h.size();
    if (N < 16384) { std::printf("feed too small (%d).\n", N); return 1; }

    // Warm seed: each node's previous variance after a half-percent move in
    // sigma, exactly as the market warm row of the throughput benchmark.
    std::vector<double> wtrue(N), wprev(N), wout(N);
    implied_variance_grid_batch(h.data(), c.data(), wtrue.data(), N);
    for (int i = 0; i < N; ++i) { double v = std::sqrt(wtrue[i]) * 1.005; wprev[i] = v * v; }

    // Per-quote context for the reference solver's own input transform, hoisted
    // out of the timed loop for both methods alike (the reference is timed on
    // beta = c/e^{h/2}, its normalized variable).
    std::vector<volfi::otm_context> q; q.reserve(N);
    for (int i = 0; i < N; ++i) q.emplace_back(h[i]);

    std::printf("# n cold_ns warm2_ns lbr_ns  (median of %d runs, ns/quote; "
                "timed regions >= %.0f ms, rotating slice offsets)\n", REPS, MIN_MS);

    for (int n = 1; n <= 16384; n *= 2) {
        std::vector<double> tc, tw, tl;
        for (int r = 0; r < REPS; ++r) {
            const int off = (int)(((long)r * 4099) % (N - n));   // rotate the slice
            const double* hp = h.data() + off;
            const double* cp = c.data() + off;
            const double* wp = wprev.data() + off;
            double* op = wout.data() + off;

            tc.push_back(timed(n, [&] { implied_variance_grid_batch(hp, cp, op, n); }));
            tw.push_back(timed(n, [&] { implied_variance_warm_batch(hp, cp, wp, op, n, 2); }));
            tl.push_back(timed(n, [&] {
                double s = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double beta = cp[i] / q[off + i].eh2;
                    const double sig = NormalisedImpliedBlackVolatility(beta, -q[off + i].h, 1.0);
                    s += sig * sig;
                }
                g_sink += s;
            }));
            g_sink += op[0];
        }
        std::printf("%d %.3f %.3f %.3f\n", n, median(tc), median(tw), median(tl));
        std::fflush(stdout);
    }
    return 0;
}
