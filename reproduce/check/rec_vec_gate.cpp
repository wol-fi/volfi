// rec_vec_gate.cpp -- the SIMD twins of the recurrence chart against the scalar chart.
//   [1] bit-identity of the batch driver against near_variance_rec, lane by lane
//   [2] routing agreement with route_near_band
//   [3] accuracy through the batch path on the 40-digit truth set
//   [4] throughput: recurrence batch vs book batch vs shipped batch, feed NEAR subset
//   [5] a %a dump for cross-ISA diffing (--hex <file>)
// Build one binary per instruction set (see README); run from volfi_annulus/bench_run.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include "volfi_annulus_all.hpp"
#include "volfi_near_certified.hpp"
#include "volfi_near_certified_vec.hpp"
#include "volfi_near_book.hpp"
#include "volfi_near_book_vec.hpp"
#include "volfi_near_rec.hpp"
#include "volfi_near_rec_vec.hpp"

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
static bool same_bits(double a, double b) {
    uint64_t x, y; std::memcpy(&x, &a, 8); std::memcpy(&y, &b, 8); return x == y;
}

int main(int argc, char** argv) {
    const char* hexout = nullptr;
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], "--hex")) hexout = argv[i + 1];
#if defined(VA_SIMD512)
    const char* isa = "AVX-512";
#elif defined(VA_SIMD256)
    const char* isa = "AVX2";
#else
    const char* isa = "scalar";
#endif
    std::printf("RECURRENCE chart SIMD twin gate: %s, %d G terms, %d rows, log1p mode %d\n\n",
                isa, (int)nr::NR_NG, (int)nr::NR_M + 1, (int)NC_LOG1P_MODE);

    std::vector<double> H, C; std::vector<Rec> tr;
    load_feed("market_feed.csv", H, C);
    const size_t n_feed = H.size();
    if (!load_truth("near_truth.bin", tr) && !load_truth("../../volfi_test/gate/near_truth.bin", tr)) {
        std::printf("FATAL: near_truth.bin not found\n"); return 2;
    }
    for (const Rec& r : tr) { H.push_back(r.h); C.push_back(r.c); }
    for (int i = 0; i < 900; ++i) {
        double hh = std::exp(std::log(1e-9) + (std::log(0.29999) - std::log(1e-9)) * i / 899.0);
        for (int j = 0; j < 900; ++j) {
            double cc = std::exp2(-70.0 + 69.5 * j / 899.0);
            if (cc > 0 && cc < 1) { H.push_back(hh); C.push_back(cc); }
        }
    }
    for (int i = 0; i <= 400; ++i) {
        double hh = 1e-7 + (0.29999 - 1e-7) * i / 400.0;
        double c_ceil = std::expm1(hh) / std::expm1(hh / nc::NC_THETA);
        double c_wing = std::expm1(hh) / std::expm1(nc::NC_A_WING);
        for (double base : {c_ceil, c_wing})
            for (int d = -4; d <= 4; ++d) {
                double cc = base;
                for (int k = 0; k < std::abs(d); ++k) cc = std::nextafter(cc, d < 0 ? 0.0 : 1.0);
                if (cc > 0 && cc < 1) { H.push_back(hh); C.push_back(cc); }
            }
    }
    H.push_back(0.1); C.push_back(0.0);
    H.push_back(0.1); C.push_back(5e-324);
    H.push_back(0.0); C.push_back(0.5);
    H.push_back(0.1); C.push_back(1.0);
    const size_t N = H.size();

    std::vector<double> Ws(N, 0.0), Wb(N, 0.0);
    std::vector<int>    Rs(N, -9),  Rb(N, -9);
    long n_near = 0, n_fb = 0;
    for (size_t i = 0; i < N; ++i) {
        Rs[i] = nc::route_near_band(H[i], C[i], nullptr);
        if (Rs[i] == nc::NCR_NEAR) { int fb = 0; Ws[i] = nr::near_variance_rec(H[i], C[i], &fb); ++n_near; n_fb += fb; }
    }
    nr::near_variance_rec_batch(H.data(), C.data(), Wb.data(), Rb.data(), (int)N);

    long mism = 0, rmism = 0; size_t at = 0;
    for (size_t i = 0; i < N; ++i) {
        if (Rb[i] != Rs[i]) ++rmism;
        if (Rs[i] == nc::NCR_NEAR && !same_bits(Wb[i], Ws[i])) { if (!mism) at = i; ++mism; }
    }
    std::printf("[1] BIT-IDENTITY batch == scalar over %ld NEAR quotes (%ld above the series range) : mismatches = %ld\n",
                n_near, n_fb, mism);
    if (mism) std::printf("      first at h=%.17g c=%.17g batch=%a scalar=%a\n", H[at], C[at], Wb[at], Ws[at]);
    std::printf("[2] ROUTING batch == scalar over %zu quotes            : mismatches = %ld\n", N, rmism);
    {
        const double EPS = 2.220446049250313e-16;
        double wr = 0; long used = 0, bad = 0;
        for (size_t k = 0; k < tr.size(); ++k) {
            const size_t i = n_feed + k;
            if (Rs[i] != nc::NCR_NEAR) continue;
            ++used;
            const double e = std::fabs(std::sqrt(Wb[i]) - tr[k].v) / tr[k].v;
            if (e > 1e-15) ++bad;
            wr = std::max(wr, e);
        }
        std::printf("[3] ACCURACY batch path, 40-digit truth set (n=%ld): max rel %.3e (%.2f ULP)  pts>1e-15 = %ld\n\n",
                    used, wr, wr / EPS, bad);
    }

    std::vector<double> fh, fc;
    for (size_t i = 0; i < n_feed; ++i)
        if (Rs[i] == nc::NCR_NEAR) { fh.push_back(H[i]); fc.push_back(C[i]); }
    const int MF = (int)fh.size();
    std::printf("[4] THROUGHPUT, feed NEAR subset, best of 5\n");
    std::printf("    %8s  %11s  %10s  %10s   %8s  %8s\n", "batch", "recurrence", "book", "shipped", "book/rec", "shp/rec");
    for (int M : {64, 256, 1024, 4096, MF}) {
        if (M > MF) continue;
        const int R = std::max(20, 4000000 / M);
        std::vector<double> w1(M), w2(M), w3(M); std::vector<int> r1(M), r2(M);
        double b1 = 1e30, b2 = 1e30, b3 = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            double t0 = now_s();
            for (int r = 0; r < R; ++r) nr::near_variance_rec_batch(fh.data(), fc.data(), w1.data(), r1.data(), M);
            double t1 = now_s();
            for (int r = 0; r < R; ++r) nb::near_variance_book_batch(fh.data(), fc.data(), w2.data(), r2.data(), M);
            double t2 = now_s();
            for (int r = 0; r < R; ++r) volfi_annulus::implied_variance_grid_batch(fh.data(), fc.data(), w3.data(), M);
            double t3 = now_s();
            const double nn = (double)M * R;
            b1 = std::min(b1, 1e9 * (t1 - t0) / nn);
            b2 = std::min(b2, 1e9 * (t2 - t1) / nn);
            b3 = std::min(b3, 1e9 * (t3 - t2) / nn);
        }
        std::printf("    %8d  %8.2f ns  %7.2f ns  %7.2f ns   %8.2f  %8.2f\n", M, b1, b2, b3, b2 / b1, b3 / b1);
    }
    if (hexout) {
        FILE* f = std::fopen(hexout, "w");
        if (f) {
            std::fprintf(f, "N %zu\n", N);
            for (size_t i = 0; i < N; ++i) std::fprintf(f, "%zu %a %a %a %d\n", i, H[i], C[i], Wb[i], Rb[i]);
            std::fclose(f);
            std::printf("\n[5] wrote %s (%zu lines); the file must be identical for all three ISAs\n", hexout, N);
        }
    }
    return (mism == 0 && rmism == 0) ? 0 : 1;
}
