// wb_gate.cpp -- the whole-book chart on the host.
//   [1] accuracy on the whole-book truth set (gate/wb_truth.bin, both regions)
//   [2] the FULL market feed: coverage by region, agreement with the shipped scalar
//       entry on every quote the chart covers, and the shipped routes of the rest
//   [3] the NEAR band: agreement with the recurrence chart (the same series)
//   [4] expm1 on (0, 16.5] against glibc, worst ULP
//   [5] scalar latency over the full feed: whole-book chart vs the shipped scalar
// Build: g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -w \
//        -I../../volfi_annulus -I../src -march=native wb_gate.cpp -o wbg
// Run from volfi_annulus/bench_run (market_feed.csv; wb_truth.bin from ../../volfi_test/gate).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include "volfi_annulus_all.hpp"
#include "volfi_near_certified.hpp"
#include "volfi_near_rec.hpp"
#include "volfi_wb.hpp"

namespace nc = volfi_near_certified;
namespace nr = volfi_near_rec;
namespace nw = volfi_wb;

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
static double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    const double EPS = 2.220446049250313e-16;
    std::printf("WHOLE-BOOK chart gate: G %d + %d terms, region A %d rows (theta <= %.2f), region B %d rows (h <= min(%.1f, %.2f a + %.2f))\n\n",
                (int)nw::NW_NG0, (int)nw::NW_NF1, (int)nw::NW_MA + 1, nw::NW_THETA_A, (int)nw::NW_NB, nw::NW_H_B, nw::NW_HB_SLOPE, nw::NW_HB_INT);

    // [1] truth set
    std::vector<Rec> tr;
    if (!load_truth("wb_truth.bin", tr) && !load_truth("../../volfi_test/gate/wb_truth.bin", tr)) {
        std::printf("FATAL: wb_truth.bin not found\n"); return 2;
    }
    double wA = 0, wB = 0; long nA = 0, nB = 0, nOut = 0, bad = 0;
    for (const Rec& r : tr) {
        int code; const double w = nw::implied_variance_wb(r.h, r.c, &code);
        const double e = std::fabs(std::sqrt(w) - r.v) / r.v;
        if (code == nw::NW_A) { ++nA; wA = std::max(wA, e); }
        else if (code == nw::NW_B) { ++nB; wB = std::max(wB, e); }
        else ++nOut;
        if (code != nw::NW_OUT && e > 1e-15) ++bad;
    }
    std::printf("[1] 40-DIGIT TRUTH SET, whole book: A %ld pts max %.3e (%.2f ULP), B %ld pts max %.3e (%.2f ULP), outside %ld, pts>1e-15 = %ld\n\n",
                nA, wA, wA / EPS, nB, wB, wB / EPS, nOut, bad);

    // [2] full feed
    std::vector<double> H, C;
    if (!load_feed("market_feed.csv", H, C)) { std::printf("FATAL: market_feed.csv not found\n"); return 2; }
    long cnt[3] = {0, 0, 0}; double wdiff = 0; size_t at = 0; long far_route = 0;
    std::vector<int> codes(H.size());
    for (size_t i = 0; i < H.size(); ++i) {
        int code; const double w = nw::implied_variance_wb(H[i], C[i], &code);
        codes[i] = code; ++cnt[code];
        if (code != nw::NW_OUT) {
            const double ws = volfi_annulus::implied_variance_otm(H[i], C[i]);
            const double d = std::fabs(std::sqrt(w) - std::sqrt(ws)) / std::sqrt(ws);
            if (d > wdiff) { wdiff = d; at = i; }
        }
    }
    std::printf("[2] MARKET FEED, %zu quotes: region A %ld (%.2f%%), region B %ld (%.2f%%), shipped fallback %ld (%.2f%%)\n",
                H.size(), cnt[1], 100.0 * cnt[1] / H.size(), cnt[2], 100.0 * cnt[2] / H.size(), cnt[0], 100.0 * cnt[0] / H.size());
    std::printf("    chart vs shipped scalar on the covered quotes: max rel diff in sigma %.3e (%.2f ULP) at h=%.6g c=%.3e\n",
                wdiff, wdiff / EPS, H[at], C[at]);
    // what the fallback quotes are
    long fb_h = 0, fb_theta = 0;
    for (size_t i = 0; i < H.size(); ++i) if (codes[i] == nw::NW_OUT) {
        const double E = nw::nw_expm1(H[i]); const double a = nc::log1p_pos(E / C[i]);
        if (a < nw::NW_TWO_PI) ++fb_theta; else ++fb_h;
    }
    std::printf("    fallback quotes: %ld above theta_A below 2pi (UPPER-like), %ld beyond the h-limit above 2pi (extreme wing)\n\n", fb_theta, fb_h);

    // [3] NEAR band vs the recurrence chart
    double dnear = 0; long nnear = 0;
    for (size_t i = 0; i < H.size(); ++i) {
        nc::near_coords z;
        if (nc::route_near_band(H[i], C[i], &z) != nc::NCR_NEAR) continue;
        int fb = 0; const double wr = nr::near_variance_rec(H[i], C[i], &fb);
        if (fb) continue;
        int code; const double w = nw::implied_variance_wb(H[i], C[i], &code);
        if (code != nw::NW_A) continue;
        ++nnear; dnear = std::max(dnear, std::fabs(w - wr) / wr);
    }
    std::printf("[3] NEAR band, %ld quotes: whole-book vs recurrence chart, max rel diff in w %.3e (%.2f ULP)\n", nnear, dnear, dnear / EPS);
    std::printf("    (same exact rows, different expm1 and a 2-pi cell of G; agreement is a consistency check, not a gate)\n\n");

    // [4] expm1
    double wex = 0; double atx = 0;
    for (int i = 0; i <= 200000; ++i) {
        const double x = 1e-6 + (16.5 - 1e-6) * i / 200000.0;
        const double got = nw::nw_expm1(x), ref = std::expm1(x);
        const double u = std::nextafter(ref, INFINITY) - ref;
        const double e = std::fabs(got - ref) / u;
        if (e > wex) { wex = e; atx = x; }
    }
    std::printf("[4] nw_expm1 vs glibc expm1 on (0, 16.5]: worst %.3f ULP at h=%.6g\n\n", wex, atx);

    // [5] latency
    const int reps = 100;
    volatile double sink = 0;
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) for (size_t i = 0; i < H.size(); ++i) { int cd; sink += nw::implied_variance_wb(H[i], C[i], &cd); }
    double t1 = now_s();
    for (int r = 0; r < reps; ++r) for (size_t i = 0; i < H.size(); ++i) sink += volfi_annulus::implied_variance_otm(H[i], C[i]);
    double t2 = now_s();
    const double q = (double)H.size() * reps;
    std::printf("[5] SCALAR LATENCY, full feed, %d reps: whole-book %.2f ns/quote   shipped %.2f   ratio %.2f\n",
                reps, 1e9 * (t1 - t0) / q, 1e9 * (t2 - t1) / q, (t2 - t1) / (t1 - t0));
    return (bad == 0 && wex < 1.5) ? 0 : 1;
}
