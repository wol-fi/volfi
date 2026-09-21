// seam_gate.cpp -- v0.3.1.  The vector UPPER pre-filter must reproduce the scalar route on every quote, in particular
// ON the seams.  Quotes are placed at the wing seam cw(h) and the ceiling ctop(h), each at -2..+2 ulp, for h across all
// three bands including H_ATM_HI and H_BOX at -2..+2 ulp, plus random fill.  grid batch and book batch must equal the
// scalar entry bit for bit.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
#include "volfi_wb_vec.hpp"
using namespace volfi_annulus;
int main() {
    std::vector<double> H, C;
    unsigned long long g = 0x853C49E6748FEA9BULL; auto rnd = [&]() { g = g * 6364136223846793005ULL + 1442695040888963407ULL; return (double)(g >> 11) * (1.0 / 9007199254740992.0); };
    auto ulp = [](double x, int k) { for (int j = 0; j < std::abs(k); ++j) x = std::nextafter(x, k > 0 ? 2.0 : -1.0); return x; };
    std::vector<double> hs;
    for (int i = 0; i < 60000; ++i) hs.push_back(std::exp(std::log(1e-4) + rnd() * (std::log(20.0) - std::log(1e-4))));
    for (int k = -2; k <= 2; ++k) { double a = H_ATM_HI, b = H_BOX; for (int j = 0; j < std::abs(k); ++j) { a = std::nextafter(a, k > 0 ? 99.0 : 0.0); b = std::nextafter(b, k > 0 ? 99.0 : 0.0); }
        for (int r = 0; r < 200; ++r) { hs.push_back(a); hs.push_back(b); } }
    for (double h : hs) {
        const double cw = br::cwstar_price(h), ct = br::ctop_price(h);
        for (int k = -2; k <= 2; ++k) { H.push_back(h); C.push_back(ulp(cw, k)); H.push_back(h); C.push_back(ulp(ct, k)); }
        H.push_back(h); C.push_back(rnd()); H.push_back(h); C.push_back(1.0 - 1e-6 * rnd()); H.push_back(h); C.push_back(ct + (1.0 - ct) * rnd());
    }
    const int n = (int)H.size(); std::vector<double> ws(n), wg(n), wb(n); std::vector<int> code(n);
    for (int i = 0; i < n; ++i) ws[i] = implied_variance_otm(H[i], C[i]);
    implied_variance_grid_batch(H.data(), C.data(), wg.data(), n);
    volfi_wb::implied_variance_wb_batch(H.data(), C.data(), wb.data(), code.data(), n);
    long mg = 0, mb = 0, nup = 0;
    for (int i = 0; i < n; ++i) { if (std::memcmp(&ws[i], &wg[i], 8)) ++mg;
        const double s = volfi_wb::implied_variance_wb(H[i], C[i], nullptr); if (std::memcmp(&s, &wb[i], 8)) ++mb;
        if (detail::grid_endpoint_route(H[i], C[i]) == 2) ++nup; }
    long wrong = 0, undec = 0, wrong0 = 0, undec0 = 0;                // the fast route must never contradict the exact route
    for (int i = 0; i < n; ++i) { int band; const bool up = detail::grid_far_cell(H[i], C[i], detail::bits_of(C[i]), band) < 0 && detail::grid_endpoint_route(H[i], C[i]) == 2;
        const double a = std::log1p(std::expm1(H[i]) / C[i]);
        const int f = fastroute::upper_certain(H[i], a); if (f < 0) ++undec; else if ((f == 1) != up) ++wrong;
        const int f0 = fastroute::upper_certain_band0(H[i], C[i]); if (H[i] < H_ATM_HI) { if (f0 < 0) ++undec0; else if ((f0 == 1) != up) ++wrong0; } }
    std::printf("seam gate, %d quotes (%ld route UPPER): grid batch != scalar: %ld | book batch != book scalar: %ld\n", n, nup, mg, mb);
    std::printf("fast route: contradictions %ld (undecided %ld) | band-0 price form: contradictions %ld (undecided %ld)\n", wrong, undec, wrong0, undec0);
    return (mg == 0 && mb == 0 && wrong == 0 && wrong0 == 0) ? 0 : 1;
}
