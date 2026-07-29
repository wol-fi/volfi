// bench_phases.cpp -- phase-instrumented breakdown of the COLD mixed-feed driver's
// per-quote cost, to locate the ~33 ns of routing/sort/scatter overhead that sits on
// top of the pure per-chart kernel work, and to size the driver optimizations.
//
// GPU-FRIENDLY BY DESIGN: every phase below is annotated with its GPU-primitive analog.
// This is deliberate -- the CPU overheads we want to cut (classification, counting
// sort, scatter) are exactly the primitives that are expensive on a GPU too, and the
// two optimizations this harness is meant to justify --
//     (A) route-caching for persistent (K,T) books, and
//     (B) storing the book in bucket order (no per-pass sort, no scatter) --
// are precisely what makes a future full-book GPU port COALESCED (per-chart kernels
// over contiguous data, no gather/scatter, no per-snapshot re-sort). So one breakdown
// guides both the CPU driver and the GPU port.
//
// NO LIBRARY CHANGES: this file only calls public volfi_annulus entry points plus the
// detail:: batch passes, and re-implements the sort/scatter micro-ops locally.
//
// BUILD (match BENCHMARK_PROTOCOL.md, AVX-512 build):
//   g++ -O3 -march=native -ffp-contract=off -fno-fast-math -std=c++17 \
//       bench_phases.cpp -o bench_phases
//   (AVX2: -mavx2 -mfma -mno-avx512f ; scalar: -mno-avx2 -mno-avx512f)
// RUN on a quiet machine (mains power, no browser/IDE/OneDrive sync), market_feed.csv
// in the working directory. Report medians over RUNS; spread should be <~1%.

#include "volfi_annulus_all.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

volatile double g_sink = 0.0;      // defeat dead-code elimination
volatile long  g_isink = 0;
using clk = std::chrono::steady_clock;

static inline uint64_t bd(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
static inline bool feasible(double c) { return c > 0.0 && c < 1.0; }

// route() classifier -- copied verbatim from benchmark_vec.cpp (mirrors the entry).
enum { R_WING = 0, R_NEAR = 1, R_FAR = 2, R_UPPER = 3, R_EDGE = 4 };
int route(double h, double c) {
    using namespace volfi_annulus;
    if (!(c > 0.0) || c >= 1.0) return R_EDGE;
    if (h == 0.0) return R_EDGE;
    double cw = br::cwing_price(h);
    if (c < cw) return R_WING;
    if (h < H_ATM_HI) {
        double ct_near = 1.0 - br::onem_otm(h, volfi_annulus_broadrange::NEAR_UPPER_VSEAM, std::exp(h));
        return (c <= ct_near) ? R_NEAR : R_UPPER;
    }
    double ct2 = br::c2_price(h);
    if (h <= H_BOX && c <= ct2) return R_FAR;
    return R_UPPER;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Time `fn` with an adaptive inner rep-count (~2 ms/block), medians over RUNS,
// return ns per quote. `label`/`gpu` printed with the number.
template <class F>
static double bench(const char* label, const char* gpu, int n, F&& fn) {
    const int RUNS = 9;
    // calibrate reps to ~2ms
    int rep = 1;
    for (;;) {
        auto t0 = clk::now();
        for (int r = 0; r < rep; ++r) fn();
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (ms > 2.0 || rep > (1 << 20)) break;
        rep = std::max(rep * 2, (int)(rep * 2.5 / std::max(ms, 1e-3)));
    }
    std::vector<double> t;
    for (int run = 0; run < RUNS; ++run) {
        auto t0 = clk::now();
        for (int r = 0; r < rep; ++r) fn();
        double sec = std::chrono::duration<double>(clk::now() - t0).count();
        t.push_back(1e9 * sec / ((double)rep * n));
    }
    double m = median(t);
    std::printf("  %-28s %8.3f ns/quote   [GPU: %s]\n", label, m, gpu);
    return m;
}

int main() {
    using namespace volfi_annulus;

    // ---- load the 30k market feed (same file/format as benchmark_vec.cpp) ----
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
    int n = (int)h.size();
    if (n == 0) { std::printf("empty feed.\n"); return 1; }

    long mc[5] = {0};
    std::vector<int> rid(n);
    for (int i = 0; i < n; ++i) { rid[i] = route(h[i], c[i]); mc[rid[i]]++; }

    std::printf("=== PHASE BREAKDOWN, cold mixed feed (N=%d) ===\n", n);
    std::printf("route mix: WING=%.2f%% NEAR=%.2f%% FAR=%.2f%% UPPER=%.2f%% EDGE=%.2f%%\n\n",
                100.0 * mc[R_WING] / n, 100.0 * mc[R_NEAR] / n, 100.0 * mc[R_FAR] / n,
                100.0 * mc[R_UPPER] / n, 100.0 * mc[R_EDGE] / n);

    std::vector<double> w(n), w_true(n), w_prev(n);
    implied_variance_grid_batch(h.data(), c.data(), w_true.data(), n);   // ground truth (untimed)
    for (int i = 0; i < n; ++i) w_prev[i] = w_true[i];

    // ---------- baselines & route-free floor ----------
    std::printf("-- drivers --\n");
    double t_cold = bench("cold speculative (baseline)", "specul. NEAR kernel + masked store", n,
        [&] { implied_variance_grid_batch(h.data(), c.data(), w.data(), n); });
    double t_two = bench("classic two-pass driver", "route + radix sort + per-bucket + scatter", n,
        [&] { detail::grid_table_pass(h.data(), c.data(), w.data(), n);
              detail::grid_fallback_pass(h.data(), c.data(), w.data(), n); });
    double t_warm = bench("warm driver (route-free)", "per-node kernel, no routing (COALESCED)", n,
        [&] { implied_variance_warm_batch(h.data(), c.data(), w_prev.data(), w.data(), n, 3); });

    // ---------- overhead components (micro-ops) ----------
    std::printf("\n-- overhead components --\n");
    // NB: scalar route() does 3 Chebyshev seam-evals (+exp) per quote -- a LOOSE upper
    // bound only; the driver classifies via frozen low-degree fits, vectorized 8-wide,
    // at a small fraction of this. Shown for reference; NOT used in any projection below.
    double t_cls = bench("classify (scalar route(), UB)", "classify kernel (driver: <<this, vectorized)", n,
        [&] { long s = 0; for (int i = 0; i < n; ++i) s += route(h[i], c[i]); g_isink += s; });

    // counting sort of precomputed route ids -> bucket order (isolates SORT, not classify)
    std::vector<int> off(6), perm(n), cnt(6);
    double t_sort = bench("counting-sort by route", "radix/counting sort (memory-bound)", n,
        [&] { for (int b = 0; b < 6; ++b) cnt[b] = 0;
              for (int i = 0; i < n; ++i) cnt[rid[i] + 1]++;
              for (int b = 1; b < 6; ++b) cnt[b] += cnt[b - 1];
              for (int i = 0; i < n; ++i) perm[cnt[rid[i]]++] = i;
              g_isink += perm[n - 1]; });

    // scatter results back to input order by the permutation (memory scatter)
    double t_scat = bench("scatter by permutation", "gather/scatter (memory-bound, GPU-costly)", n,
        [&] { for (int i = 0; i < n; ++i) w[perm[i]] = w_true[i]; g_sink += w[0]; });

    double t_copy = bench("feed copy (mem floor)", "H2D/D2H PCIe transfer floor", n,
        [&] { std::memcpy(w.data(), w_true.data(), n * sizeof(double)); g_sink += w[0]; });

    // ---------- attribution & optimization projections ----------
    // Absolute ns are host-specific; read the STRUCTURE (cold vs warm vs two-pass gaps).
    double ov_spec = t_cold - t_warm;     // speculative overhead over the route-free floor
    double ov_two  = t_two  - t_warm;     // classic two-pass overhead (full route+sort+scatter)
    std::printf("\n=== ATTRIBUTION (medians; run on the quiet machine for real ns) ===\n");
    std::printf("  route-free floor (warm)      : %8.3f ns/quote\n", t_warm);
    std::printf("  cold speculative (baseline)  : %8.3f  -> overhead %.2f  (classify + minority drain;\n",
                t_cold, ov_spec);
    std::printf("                                             sort/scatter ALREADY avoided for the 90%% NEAR)\n");
    std::printf("  classic two-pass driver      : %8.3f  -> overhead %.2f  (full per-quote route +\n",
                t_two, ov_two);
    std::printf("                                             counting sort + per-bucket + scatter)\n");
    std::printf("  measured memory ops          : sort %.2f   scatter %.2f   copy %.2f\n",
                t_sort, t_scat, t_copy);

    std::printf("\n=== READING ===\n");
    std::printf("  * Speculation already dodges sort/scatter for the majority (cold << two-pass).\n");
    std::printf("    The speculative residual (%.1f ns) is classification + the minority drain.\n", ov_spec);
    std::printf("  * sort+scatter are cheap on CPU (%.1f ns) but are the DOMINANT cost of a NAIVE\n", t_sort + t_scat);
    std::printf("    GPU port (radix sort + gather/scatter); the two-pass gap (%.1f ns) is what a\n", ov_two);
    std::printf("    coalesced port must avoid.\n");

    std::printf("\n=== OPTIMIZATIONS (both GPU-friendly) ===\n");
    std::printf("  (A) route-cache, persistent (K,T) book: skip re-classification on re-inversion;\n");
    std::printf("      only seam-band nodes re-check. Cold-repeat approaches the warm floor (%.1f ns).\n", t_warm);
    std::printf("      GPU: cached per-node route ids -> tiny re-classify kernel.\n");
    std::printf("  (B) bucket-order storage: no per-pass sort/scatter -> contiguous per-chart kernels.\n");
    std::printf("      CPU saving here is small (%.1f ns); on GPU it is DECISIVE -- it turns the\n", t_sort + t_scat);
    std::printf("      naive two-pass model into a coalesced one (no gather/scatter, the #1 GPU cost).\n");
    std::printf("  (E) v0.2.1 wing seed (6->3 Newton): ~7-8 ns off the feed's wing term, independent\n");
    std::printf("      of driver work (%.2f%% of feed is wing).\n", 100.0 * mc[R_WING] / n);
    std::printf("\n  (classify(UB) row above is scalar route(), a loose upper bound, not the driver's\n");
    std::printf("   vectorized classify -- excluded from all projections.)\n");
    return 0;
}
