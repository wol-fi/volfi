// rec_gate.cpp -- the recurrence chart on the host: accuracy against the 40-digit
// truth set, agreement with the 12-cell and book charts on the feed, fallback count,
// and scalar latency of the three charts.  Build:
//   g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -w \
//       -I../../volfi_annulus -I../src -march=native rec_gate.cpp -o rg
// Run from volfi_annulus/bench_run (market_feed.csv, near_truth.bin).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include "volfi_annulus_all.hpp"
#include "volfi_near_certified.hpp"
#include "volfi_near_book.hpp"
#include "volfi_near_rec.hpp"

namespace nc = volfi_near_certified;
namespace nb = volfi_near_book;
namespace nr = volfi_near_rec;

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
    std::printf("RECURRENCE chart gate: G with %d Chebyshev terms, %d exact rows, tau <= %.3g\n\n",
                (int)nr::NR_NG, (int)nr::NR_M + 1, nr::NR_TAU_MAX);

    std::vector<Rec> tr;
    if (!load_truth("near_truth.bin", tr) && !load_truth("../../volfi_test/gate/near_truth.bin", tr)) {
        std::printf("FATAL: near_truth.bin not found\n"); return 2;
    }
    double wr = 0, wb = 0, wc = 0; long used = 0, fb = 0, bad = 0;
    for (const Rec& r : tr) {
        nc::near_coords z;
        if (nc::route_near_band(r.h, r.c, &z) != nc::NCR_NEAR) continue;
        ++used;
        int f1 = 0, f2 = 0;
        const double w_rec  = nr::near_variance_rec(r.h, r.c, &f1);
        const double w_book = nb::near_variance_book(r.h, r.c, &f2);
        const double w_cert = nc::near_variance_from_coords(z);
        fb += f1;
        if (!f1) {
            const double e = std::fabs(std::sqrt(w_rec) - r.v) / r.v;
            wr = std::max(wr, e); if (e > 1e-15) ++bad;
        }
        if (!f2) wb = std::max(wb, std::fabs(std::sqrt(w_book) - r.v) / r.v);
        wc = std::max(wc, std::fabs(std::sqrt(w_cert) - r.v) / r.v);
    }
    std::printf("[1] 40-DIGIT TRUTH SET: %ld NEAR points, %ld above the series range (12-cell fallback)\n", used, fb);
    std::printf("    recurrence: max rel %.3e (%.2f ULP)  pts>1e-15 = %ld   (on the %ld in-range points)\n", wr, wr / EPS, bad, used - fb);
    std::printf("    book      : max rel %.3e (%.2f ULP)   12-cell: %.3e (%.2f ULP)\n\n", wb, wb / EPS, wc, wc / EPS);

    std::vector<double> H, C;
    if (!load_feed("market_feed.csv", H, C)) { std::printf("FATAL: market_feed.csv not found\n"); return 2; }
    std::vector<double> hb, cb;
    long fb_feed = 0; double d12 = 0, dbk = 0;
    for (size_t i = 0; i < H.size(); ++i) {
        nc::near_coords z;
        if (nc::route_near_band(H[i], C[i], &z) != nc::NCR_NEAR) continue;
        hb.push_back(H[i]); cb.push_back(C[i]);
        int f1 = 0;
        const double w_rec = nr::near_variance_rec(H[i], C[i], &f1);
        fb_feed += f1;
        d12 = std::max(d12, std::fabs(w_rec - nc::near_variance_from_coords(z)) / w_rec);
        dbk = std::max(dbk, std::fabs(w_rec - nb::near_variance_book(H[i], C[i])) / w_rec);
    }
    std::printf("[2] MARKET FEED: %zu NEAR quotes, fallbacks = %ld\n", hb.size(), fb_feed);
    std::printf("    recurrence vs 12-cell: max rel diff in w %.3e (%.2f ULP);  vs book: %.3e (%.2f ULP)\n\n",
                d12, d12 / EPS, dbk, dbk / EPS);

    const int reps = 200;
    volatile double sink = 0;
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) for (size_t i = 0; i < hb.size(); ++i) sink += nr::near_variance_rec(hb[i], cb[i]);
    double t1 = now_s();
    for (int r = 0; r < reps; ++r) for (size_t i = 0; i < hb.size(); ++i) sink += nc::near_variance_certified(hb[i], cb[i]);
    double t2 = now_s();
    for (int r = 0; r < reps; ++r) for (size_t i = 0; i < hb.size(); ++i) sink += nb::near_variance_book(hb[i], cb[i]);
    double t3 = now_s();
    const double q = (double)hb.size() * reps;
    std::printf("[3] SCALAR LATENCY, feed NEAR subset, %d reps\n", reps);
    std::printf("    recurrence %.2f ns/quote   12-cell %.2f   book %.2f   (12-cell/recurrence %.2f)\n",
                1e9 * (t1 - t0) / q, 1e9 * (t2 - t1) / q, 1e9 * (t3 - t2) / q, (t2 - t1) / (t1 - t0));
    return (bad == 0 && fb_feed == 0) ? 0 : 1;
}
