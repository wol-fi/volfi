// dump_feed_ref.cpp -- write the market feed with our reference sigma (whole-book chart, 3 ULP
// against the 40-digit oracle on this feed) and the shipped router's region, as text, for scoring
// third-party Python libraries (fast-vollib) on the same quotes:  "h c v_ref region" per line.
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
namespace nw = volfi_wb;
static int route(double h, double c) {
    using namespace volfi_annulus;
    if (!(c > 0.0) || c >= 1.0 || !(h > 0.0)) return 4;
    switch (detail::grid_endpoint_route(h, c)) { case 1: return 1; case 2: return 3; case 3: return 0; default: return 2; }  // 0 wing 1 near 2 far 3 upper
}
int main(int argc, char** argv) {
    const char* in = argc > 1 ? argv[1] : "market_feed.csv"; const char* out = argc > 2 ? argv[2] : "market_feed_ref.txt";
    std::ifstream fin(in); std::ofstream fo(out); std::string ln; long n = 0;
    fo.precision(17);
    while (std::getline(fin, ln)) { if (ln.empty() || ln[0] == '#') continue; std::istringstream is(ln); double h, c; if (!(is >> h >> c)) continue;
        if (!(h > 1e-4 && h < 16.5) || !(c > 0.0 && c < 1.0 && 1.0 - c > 1e-16)) continue;
        int cd; const double v = std::sqrt(nw::implied_variance_wb(h, c, &cd));
        fo << std::scientific << h << " " << c << " " << v << " " << route(h, c) << "\n"; ++n; }
    std::printf("wrote %ld quotes to %s\n", n, out); return 0;
}
