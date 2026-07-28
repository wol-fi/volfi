// Does the LEFT (matched small-h) chart stay machine-precise up to v=2?
// Sweep h in (1e-4, 0.3), v in [1.50, 2.00], call br::left_variance directly
// (bypassing the router), compare sqrt(w) against the construction v.
// Reports worst rel error per v-band, so the certified ceiling is read off.
#include "paper_volfi.hpp"
#include "volfi_annulus_wing.hpp"
#include "volfi_annulus.hpp"
#include <cstdio>
#include <cmath>

int main() {
    using namespace volfi_annulus;
    const int NB = 20;                 // v-bands of width 0.025 over [1.5,2.0]
    double worst[NB] = {0}, wh[NB] = {0}, wv[NB] = {0};
    const int NH = 600, NV = 400;
    for (int i = 0; i < NH; ++i) {
        double h = 1e-4 + (0.29995 - 1e-4) * i / (NH - 1.0);
        for (int j = 0; j < NV; ++j) {
            double v = 1.50 + 0.50 * (j + 0.5) / NV;
            double c = volfi::black_otm_from_variance(v * v, h);
            if (!(c > 0.0) || c >= 1.0) continue;
            double w = br::left_variance(h, c);
            double rel = std::fabs(std::sqrt(w) - v) / v;
            int b = (int)((v - 1.50) / 0.025); if (b < 0) b = 0; if (b >= NB) b = NB - 1;
            if (rel > worst[b]) { worst[b] = rel; wh[b] = h; wv[b] = v; }
        }
    }
    std::printf("v-band            worst rel-sigma   at (h, v)\n");
    for (int b = 0; b < NB; ++b)
        std::printf("[%.3f,%.3f)   %.3e        (%.4f, %.4f)%s\n",
                    1.50 + 0.025 * b, 1.525 + 0.025 * b, worst[b], wh[b], wv[b],
                    worst[b] > 1e-15 ? "   <-- FAILS 1e-15 gate" : "");
    return 0;
}
