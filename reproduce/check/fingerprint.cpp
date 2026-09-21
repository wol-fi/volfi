// fingerprint.cpp -- deterministic bitwise fingerprint of the whole library surface.
//
// Purpose: prove that a refactor (e.g. the v0.2.4 chart rename) changed no value
// anywhere.  Prints every result as a %a hex double, so the output file is an exact
// record of the returned bits; hash the file before and after and compare.
//
// Covers, on one fixed deterministic grid spanning all four charts:
//   * the scalar entry            implied_variance_otm
//   * the checked entry           implied_variance_otm_checked (value + status)
//   * the mixed-h batch driver    implied_variance_grid_batch
//   * the fixed-h batch driver    implied_variance_otm_batch
//   * the warm streaming driver   implied_variance_warm_batch (2 and 3 steps)
//   * the raw-data wrapper        implied_volatility
// plus the per-chart routing label, so a routing change shows up as a diff even if
// every value survives.
//
// Build: g++ -std=c++17 -O3 -march=native -ffp-contract=off -fno-fast-math -w \
//        -I<headers> fingerprint.cpp -o fp   (add -mno-avx512f / -mno-avx2 for the
//        other ISAs; the file it writes must be identical for all three).
#include "volfi_annulus_all.hpp"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

using namespace volfi_annulus;

int main(int argc, char** argv) {
    const char* out = (argc > 1) ? argv[1] : "fingerprint.txt";
    FILE* f = std::fopen(out, "w");
    if (!f) { std::printf("cannot open %s\n", out); return 1; }

    // ---- deterministic grid: log-spaced h, log-spaced c, every chart reached ----
    std::vector<double> hs, cs;
    for (int i = 0; i < 220; ++i)                        // h in [1e-4, 16.4], log-spaced
        hs.push_back(std::exp(std::log(1e-4) + (std::log(16.4) - std::log(1e-4)) * i / 219.0));
    for (int j = 0; j < 300; ++j)                        // c in (1e-300, 1), log-spaced
        cs.push_back(std::exp2(-996.0 + 995.5 * j / 299.0));

    std::vector<double> gh, gc;
    for (double h : hs)
        for (double c : cs)
            if (c > 0.0 && c < 1.0) { gh.push_back(h); gc.push_back(c); }
    const int N = (int)gh.size();
    std::fprintf(f, "N %d\n", N);

    // ---- scalar entry, checked entry, and the routing label -------------------
    for (int i = 0; i < N; ++i) {
        const double h = gh[i], c = gc[i];
        const double w = implied_variance_otm(h, c);
        iv_status st;
        const double wc = implied_variance_otm_checked(h, c, &st);
        int route;                                        // 0 wing 1 near 2 upper 3 far/edge
        {
            int band;
            if (detail::grid_far_cell(h, c, detail::bits_of(c), band) >= 0) route = 3;
            else switch (detail::grid_endpoint_route(h, c)) {
                case 1: route = 1; break; case 2: route = 2; break;
                case 3: route = 0; break; default: route = 4;
            }
        }
        std::fprintf(f, "S %a %a %d %d\n", w, wc, (int)st, route);
    }

    // ---- mixed-h batch driver -------------------------------------------------
    std::vector<double> wb(N);
    implied_variance_grid_batch(gh.data(), gc.data(), wb.data(), N);
    for (int i = 0; i < N; ++i) std::fprintf(f, "B %a\n", wb[i]);

    // ---- fixed-h batch driver, one row per h ----------------------------------
    for (double h : hs) {
        context q(h);
        std::vector<double> row, wr;
        for (double c : cs) if (c > 0.0 && c < 1.0) row.push_back(c);
        wr.resize(row.size());
        implied_variance_otm_batch(q, row.data(), wr.data(), (int)row.size());
        for (double v : wr) std::fprintf(f, "F %a\n", v);
    }

    // ---- warm streaming driver, seeded from a half-percent move ---------------
    std::vector<double> wprev(N), ww(N);
    for (int i = 0; i < N; ++i) {
        const double v = std::sqrt(wb[i]) * 1.005;
        wprev[i] = (wb[i] > 0.0 && std::isfinite(v)) ? v * v : wb[i];
    }
    for (int steps = 2; steps <= 3; ++steps) {
        implied_variance_warm_batch(gh.data(), gc.data(), wprev.data(), ww.data(), N, steps);
        for (int i = 0; i < N; ++i) std::fprintf(f, "W%d %a\n", steps, ww[i]);
    }

    // ---- raw-data wrapper, both sides, through put-call parity ----------------
    for (int i = 0; i < N; i += 7) {
        const double h = gh[i], c = gc[i];
        const double F0 = 100.0, K = F0 * std::exp(-h), T = 0.75;
        iv_status st;
        const double sc = implied_volatility(F0, K, c * F0, T, true,  &st);
        const double sp = implied_volatility(F0, F0 * std::exp(h), c * F0, T, false, &st);
        std::fprintf(f, "R %a %a\n", sc, sp);
    }

    std::fclose(f);
    std::printf("wrote %s  (N=%d grid points)\n", out, N);
    return 0;
}
