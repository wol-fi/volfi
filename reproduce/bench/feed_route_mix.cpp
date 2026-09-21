// Route mix of the FULL 2024 SPX tradeable population under the LIVE router.
// The paper quotes this mix in the "what a real book exercises" paragraph; it
// moves whenever a seam moves, so it must be recomputed from the shipped
// classifier rather than from a reimplementation (a hand-mirrored copy of the
// routing predicate went stale once already -- see benchmark_vec.cpp:72).
//
//   g++ -std=c++17 -O2 -march=native -ffp-contract=off -fno-fast-math -I.. \
//       feed_route_mix.cpp -o frm && ./frm spx2024_otmcall_tradeable.csv
//
// Columns used: cp_flag (3), projected (17), h (18), c (20).
#include "paper_volfi.hpp"
#include "volfi_annulus_wing.hpp"
#include "volfi_annulus.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "spx2024_otmcall_tradeable.csv";
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
    std::string line;
    std::getline(in, line);                                  // header
    long n = 0, mix[5] = {0}, wing_put = 0, projected = 0;
    double hmax = 0.0;
    std::vector<std::string> f;
    while (std::getline(in, line)) {
        f.clear();
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) f.push_back(tok);
        if (f.size() < 21) continue;
        const double h = std::atof(f[18].c_str());
        const double c = std::atof(f[20].c_str());
        int r;                                               // WING/NEAR/FAR/UPPER/EDGE
        if (!(c > 0.0) || c >= 1.0 || !(h > 0.0)) r = 4;
        else switch (volfi_annulus::detail::grid_endpoint_route(h, c)) {
            case 1:  r = 1; break;                           // NEAR
            case 2:  r = 3; break;                           // UPPER
            case 3:  r = 0; break;                           // WING
            default: r = 2;                                  // FAR (cell or analytic edge)
        }
        ++mix[r]; ++n;
        if (r == 0 && f[3] == "P") ++wing_put;
        if (f[17] == "True") ++projected;
        if (h > hmax) hmax = h;
    }
    const char* NM[5] = {"WING", "NEAR", "FAR", "UPPER", "EDGE"};
    std::printf("n = %ld tradeable quotes\n", n);
    for (int k = 0; k < 5; ++k)
        std::printf("  %-8s %8ld  %6.2f%%\n", NM[k], mix[k], 100.0 * mix[k] / n);
    std::printf("wing puts: %ld/%ld = %.1f%%   (1 quote in %.0f is a wing quote)\n",
                wing_put, mix[0], 100.0 * wing_put / (mix[0] ? mix[0] : 1),
                mix[0] ? (double)n / mix[0] : 0.0);
    std::printf("parity-projected (ITM): %.1f%%   max h = %.2f\n", 100.0 * projected / n, hmax);
    return 0;
}
