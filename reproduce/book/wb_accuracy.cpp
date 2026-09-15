// wb_accuracy.cpp -- the whole-book chart, the routed inverter and Let's Be Rational
// scored on the SAME oracle files as the manuscript's Table "accuracy" (bench_run/oracle_*.bin:
// int64 n; then n x {double h, double c, double v_oracle}, v_oracle = sigma sqrt T at 40 digits
// for the GIVEN double price).  Relative error in sigma, RMS, worst ULP distance, per routed
// region of the shipped router (the live one, detail::grid_endpoint_route), plus the
// whole-book chart's own coverage (region A / region B / shipped fallback) per row.
//
// Build (from volfi_test/gate; LBR = the Let's Be Rational source folder):
//   g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -w -DNO_XL_API \
//       -I../../volfi_annulus -I../src -I"$LBR" -march=native wb_accuracy.cpp \
//       "$LBR/lets_be_rational.cpp" "$LBR/normaldistribution.cpp" "$LBR/rationalcubic.cpp" "$LBR/erf_cody.cpp" -o wbacc
// Run from volfi_annulus/bench_run:  ./wbacc oracle_heat.bin [oracle_real.bin ...]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
#include "lets_be_rational.h"

namespace nw = volfi_wb;

static uint64_t bitsd(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
static double ulps(double a, double b) {
    if (a == b) return 0.0;
    int64_t ia = (int64_t)bitsd(a), ib = (int64_t)bitsd(b);
    if (ia < 0) ia = (int64_t)0x8000000000000000ULL - ia;
    if (ib < 0) ib = (int64_t)0x8000000000000000ULL - ib;
    int64_t d = ia > ib ? ia - ib : ib - ia;
    return (double)d;
}
enum { R_WING = 0, R_NEAR = 1, R_FAR = 2, R_UPPER = 3, R_EDGE = 4 };
static const char* RN[5] = { "Wing", "Near", "Far", "Upper", "Edge" };
static int route(double h, double c) {          // the LIVE shipped router, as benchmark_vec.cpp
    using namespace volfi_annulus;
    if (!(c > 0.0) || c >= 1.0) return R_EDGE;
    if (!(h > 0.0)) return R_EDGE;
    switch (detail::grid_endpoint_route(h, c)) {
        case 1: return R_NEAR; case 2: return R_UPPER; case 3: return R_WING; default: return R_FAR;
    }
}
struct Stat {
    double maxrel = 0, sumsq = 0, maxulp = 0; long n = 0, over = 0;
    void add(double rel, double u) { maxrel = std::max(maxrel, rel); sumsq += rel * rel; maxulp = std::max(maxulp, u); ++n; if (rel > 1e-15) ++over; }
    double rms() const { return n ? std::sqrt(sumsq / n) : 0.0; }
};

static int score(const char* path) {
    FILE* f = std::fopen(path, "rb"); if (!f) { std::printf("cannot open %s\n", path); return 2; }
    int64_t n; if (std::fread(&n, 8, 1, f) != 1) return 2;
    std::vector<double> H(n), C(n), V(n);
    for (int64_t i = 0; i < n; ++i) { double b[3]; if (std::fread(b, 8, 3, f) != 3) return 2; H[i] = b[0]; C[i] = b[1]; V[i] = b[2]; }
    std::fclose(f);
    Stat sW[5], sR[5], sL[5], sWall, sRall, sLall; long cov[5][3] = {{0}};
    for (int64_t i = 0; i < n; ++i) {
        const double h = H[i], c = C[i], vo = V[i];
        const int rg = route(h, c);
        int code; const double ww = nw::implied_variance_wb(h, c, &code);
        const double wr = volfi_annulus::implied_variance_otm(h, c);
        const double sl = NormalisedImpliedBlackVolatility(c * std::exp(-0.5 * h), -h, 1.0);
        const double sw = std::sqrt(ww), sr = std::sqrt(wr);
        auto rel = [&](double s) { return std::fabs(s - vo) / std::fabs(vo); };
        sW[rg].add(rel(sw), ulps(sw, vo)); sR[rg].add(rel(sr), ulps(sr, vo)); sL[rg].add(rel(sl), ulps(sl, vo));
        sWall.add(rel(sw), ulps(sw, vo));  sRall.add(rel(sr), ulps(sr, vo));  sLall.add(rel(sl), ulps(sl, vo));
        ++cov[rg][code];
    }
    std::printf("=== %s  n=%lld ===\n", path, (long long)n);
    std::printf("  %-6s %6s | %-34s | %-34s | %-34s | whole-book coverage\n", "region", "n",
                "WHOLE-BOOK  max_rel  rms  ulp  >1e-15", "ROUTED      max_rel  rms  ulp  >1e-15", "LBR         max_rel  rms  ulp  >1e-15");
    auto row = [&](const char* nm, long nn, const Stat& a, const Stat& b, const Stat& l, const long* cv) {
        std::printf("  %-6s %6ld | %.2e %.2e %5.0f %4ld           | %.2e %.2e %5.0f %4ld           | %.2e %.2e %5.0f %4ld           |",
                    nm, nn, a.maxrel, a.rms(), a.maxulp, a.over, b.maxrel, b.rms(), b.maxulp, b.over, l.maxrel, l.rms(), l.maxulp, l.over);
        if (cv) std::printf(" A %ld  B %ld  fallback %ld", cv[1], cv[2], cv[0]);
        std::printf("\n");
    };
    for (int r : { R_NEAR, R_FAR, R_WING, R_UPPER, R_EDGE }) if (sW[r].n) row(RN[r], sW[r].n, sW[r], sR[r], sL[r], cov[r]);
    row("All", sWall.n, sWall, sRall, sLall, nullptr);
    std::printf("\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return score("oracle_heat.bin");
    int rc = 0; for (int i = 1; i < argc; ++i) rc |= score(argv[i]);
    return rc;
}
