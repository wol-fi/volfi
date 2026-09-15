// wb_truth_score.cpp -- score the entry point (whole-book chart in front, routed charts behind) and
// the routed charts alone on ANY truth file (int64 n; n x {h, c, v_oracle}), per class tag if a
// .tags file sits next to it, with the ULP metric defined as in the manuscript: the number of
// representable doubles between the returned value and the oracle value rounded to double.
// Writes the worst K points (h c) to <file>.worst for the 60-digit recheck.
// Build: g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -march=native -w -I<volfi_annulus> -I<volfi_test/src> wb_truth_score.cpp -o wbscore
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
namespace nw = volfi_wb;
static uint64_t bitsd(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
static double ulps(double a, double b) {
    if (a == b) return 0.0;
    int64_t ia = (int64_t)bitsd(a), ib = (int64_t)bitsd(b);
    if (ia < 0) ia = (int64_t)0x8000000000000000ULL - ia;
    if (ib < 0) ib = (int64_t)0x8000000000000000ULL - ib;
    int64_t d = ia > ib ? ia - ib : ib - ia; return (double)d;
}
struct St { double mx = 0, sumsq = 0, mxu = 0; long n = 0, over = 0; double hx = 0, cx = 0;
    void add(double e, double u, double h, double c) { ++n; sumsq += e * e; if (u > mxu) mxu = u; if (e > 1e-15) ++over; if (e > mx) { mx = e; hx = h; cx = c; } }
    double rms() const { return n ? std::sqrt(sumsq / n) : 0; } };
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "wb_truth_boundary.bin";
    const int K = argc > 2 ? std::atoi(argv[2]) : 40;
    FILE* f = std::fopen(path, "rb"); if (!f) { std::printf("cannot open %s\n", path); return 2; }
    int64_t n; if (std::fread(&n, 8, 1, f) != 1) return 2;
    std::vector<double> H(n), C(n), V(n);
    for (int64_t i = 0; i < n; ++i) { double b[3]; if (std::fread(b, 8, 3, f) != 3) return 2; H[i] = b[0]; C[i] = b[1]; V[i] = b[2]; }
    std::fclose(f);
    std::vector<std::string> tag(n, "all");
    { std::string tp = std::string(path); size_t d = tp.rfind(".bin"); if (d != std::string::npos) tp = tp.substr(0, d) + ".tags";
      std::ifstream tf(tp); std::string l; int64_t i = 0; while (i < n && std::getline(tf, l)) tag[i++] = l; }
    std::map<std::string, St> sw, sr; St aw, ar; std::map<std::string, long> cov[3];
    std::vector<std::pair<double, int64_t>> worst;
    for (int64_t i = 0; i < n; ++i) {
        int code; const double w = nw::implied_variance_wb(H[i], C[i], &code);
        const double sW = std::sqrt(w), sR = std::sqrt(volfi_annulus::implied_variance_otm(H[i], C[i]));
        const double eW = std::fabs(sW - V[i]) / V[i], eR = std::fabs(sR - V[i]) / V[i];
        sw[tag[i]].add(eW, ulps(sW, V[i]), H[i], C[i]); sr[tag[i]].add(eR, ulps(sR, V[i]), H[i], C[i]);
        aw.add(eW, ulps(sW, V[i]), H[i], C[i]); ar.add(eR, ulps(sR, V[i]), H[i], C[i]);
        ++cov[code][tag[i]]; worst.push_back({eW, i});
    }
    std::printf("=== %s  n=%lld ===\n  %-10s %6s | %-9s %-9s %4s %5s | %-9s %-9s %4s %5s | coverage A/B/fallback\n", path, (long long)n, "class", "n",
                "WB max", "rms", "ULP", ">1e-15", "RT max", "rms", "ULP", ">1e-15");
    for (auto& kv : sw) { const St& a = kv.second; const St& b = sr[kv.first];
        std::printf("  %-10s %6ld | %.3e %.3e %4.0f %5ld | %.3e %.3e %4.0f %5ld | %ld/%ld/%ld\n", kv.first.c_str(), a.n, a.mx, a.rms(), a.mxu, a.over, b.mx, b.rms(), b.mxu, b.over,
                    cov[1][kv.first], cov[2][kv.first], cov[0][kv.first]); }
    std::printf("  %-10s %6ld | %.3e %.3e %4.0f %5ld | %.3e %.3e %4.0f %5ld |  worst WB at h=%.6g c=%.6e\n", "ALL", aw.n, aw.mx, aw.rms(), aw.mxu, aw.over, ar.mx, ar.rms(), ar.mxu, ar.over, aw.hx, aw.cx);
    std::sort(worst.begin(), worst.end(), [](auto& x, auto& y) { return x.first > y.first; });
    std::string wp = std::string(path) + ".worst"; FILE* wf = std::fopen(wp.c_str(), "w");
    for (int k = 0; k < K && k < (int)worst.size(); ++k) std::fprintf(wf, "%.17g %.17g %.3e %s\n", H[worst[k].second], C[worst[k].second], worst[k].first, tag[worst[k].second].c_str());
    std::fclose(wf); std::printf("  worst %d written to %s\n", K, wp.c_str());
    return aw.over == 0 ? 0 : 1;
}
