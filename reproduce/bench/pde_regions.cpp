// pde_regions.cpp -- the PDE method's accuracy per routed region (the shipped router's Near/Far/Wing/Upper
// and the whole-book chart's A/B), on the market feed (reference: whole-book chart, 3 ULP) and on the
// manuscript's oracle sets (reference: the 40-digit oracle).  CPU only; their device results equal
// their CPU results on every quote of the feed run, so this is their accuracy on the card as well.
// Build: g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -march=native -fopenmp -w -I<volfi_annulus> -I<volfi_test/src> -I<pde> \
//        pde_regions.cpp cImpVolBasic.o indexStructure.o -o pde_regions ; run from bench_run with loadPartition.txt present.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
#include "cImpVolBasic.h"
namespace nw = volfi_wb;
enum { R_WING = 0, R_NEAR = 1, R_FAR = 2, R_UPPER = 3, R_EDGE = 4 };
static const char* RN[5] = { "Wing", "Near", "Far", "Upper", "Edge" };
static int route(double h, double c) {
    using namespace volfi_annulus;
    if (!(c > 0.0) || c >= 1.0 || !(h > 0.0)) return R_EDGE;
    switch (detail::grid_endpoint_route(h, c)) { case 1: return R_NEAR; case 2: return R_UPPER; case 3: return R_WING; default: return R_FAR; }
}
struct St { double mx = 0, sum = 0; long n = 0, out = 0; double hx = 0, cx = 0; void add(double e, double h, double c) { ++n; sum += e; if (e > mx) { mx = e; hx = h; cx = c; } } };
static void report(const char* title, const std::vector<double>& H, const std::vector<double>& C, const std::vector<double>& Vref, CImpVol& pde) {
    St s[5], sA, sB, all;
    for (size_t i = 0; i < H.size(); ++i) {
        const int r = route(H[i], C[i]);
        int code; nw::implied_variance_wb(H[i], C[i], &code);
        const double p = pde.evaluate(H[i], C[i]);
        if (p < 0.0) { ++s[r].out; ++all.out; if (code == nw::NW_A) ++sA.out; else if (code == nw::NW_B) ++sB.out; continue; }
        const double e = std::fabs(p - Vref[i]) / Vref[i];
        s[r].add(e, H[i], C[i]); all.add(e, H[i], C[i]);
        if (code == nw::NW_A) sA.add(e, H[i], C[i]); else if (code == nw::NW_B) sB.add(e, H[i], C[i]);
    }
    std::printf("=== %s (n=%zu) ===\n  %-8s %7s %7s   %-10s %-10s  worst at\n", title, H.size(), "region", "n", "outside", "max rel", "mean rel");
    auto row = [](const char* nm, const St& t) { if (t.n + t.out) std::printf("  %-8s %7ld %7ld   %.3e  %.3e  h=%.4g c=%.3e\n", nm, t.n, t.out, t.mx, t.n ? t.sum / t.n : 0.0, t.hx, t.cx); };
    for (int r : { R_NEAR, R_FAR, R_WING, R_UPPER, R_EDGE }) row(RN[r], s[r]);
    row("wb-A", sA); row("wb-B", sB); row("All", all);
}
int main() {
    CImpVol pde;
    std::vector<double> H, C, V; std::ifstream fin("market_feed.csv"); std::string ln;
    while (std::getline(fin, ln)) { if (ln.empty() || ln[0] == '#') continue; std::istringstream is(ln); double h, c; if (!(is >> h >> c)) continue;
        if (!(h > 1e-4 && h < 16.5) || !(c > 0.0 && c < 1.0 && 1.0 - c > 1e-16)) continue; H.push_back(h); C.push_back(c); int cd; V.push_back(std::sqrt(nw::implied_variance_wb(h, c, &cd))); }
    report("market feed, reference = whole-book chart (3 ULP)", H, C, V, pde);
    for (const char* f : { "oracle_heat.bin", "oracle_real.bin", "oracle_stressed.bin", "oracle_edge.bin" }) {
        FILE* fp = std::fopen(f, "rb"); if (!fp) continue; int64_t n; if (std::fread(&n, 8, 1, fp) != 1) return 2;
        std::vector<double> h(n), c(n), v(n); for (int64_t i = 0; i < n; ++i) { double b[3]; if (std::fread(b, 8, 3, fp) != 3) return 2; h[i] = b[0]; c[i] = b[1]; v[i] = b[2]; } std::fclose(fp);
        report((std::string(f) + ", reference = 40-digit oracle").c_str(), h, c, v, pde);
    }
    return 0;
}
