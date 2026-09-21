// wb_vec_gate.cpp -- the SIMD twins of the whole-book chart against the scalar chart.
//   [1] bit-identity of the batch driver against implied_variance_wb, lane by lane,
//       over the whole probe population (feed, truth sets, grid, seams, hostile lanes)
//   [2] code agreement (region / fallback) with the scalar wrapper
//   [3] accuracy through the batch path on the whole-book truth set
//   [4] throughput on the FULL feed: whole-book batch vs the shipped batch, at several sizes
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
#include "volfi_wb.hpp"
#include "volfi_wb_vec.hpp"

namespace nc = volfi_near_certified;
namespace nw = volfi_wb;

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
    std::printf("WHOLE-BOOK chart SIMD twin gate: %s\n\n", isa);

    std::vector<double> H, C; std::vector<Rec> tr, trn;
    load_feed("market_feed.csv", H, C);
    const size_t n_feed = H.size();
    if (!load_truth("wb_truth.bin", tr) && !load_truth("../../volfi_test/gate/wb_truth.bin", tr)) {
        std::printf("FATAL: wb_truth.bin not found\n"); return 2;
    }
    const size_t n_tr0 = H.size();
    for (const Rec& r : tr) { H.push_back(r.h); C.push_back(r.c); }
    if (load_truth("near_truth.bin", trn) || load_truth("../../volfi_test/gate/near_truth.bin", trn))
        for (const Rec& r : trn) { H.push_back(r.h); C.push_back(r.c); }
    for (int i = 0; i < 600; ++i) {                    // the whole price box
        double hh = std::exp(std::log(1e-6) + (std::log(16.4) - std::log(1e-6)) * i / 599.0);
        for (int j = 0; j < 600; ++j) {
            double cc = std::exp(std::log(1e-300) + (std::log(0.999) - std::log(1e-300)) * j / 599.0);
            if (cc > 0 && cc < 1) { H.push_back(hh); C.push_back(cc); }
        }
    }
    H.push_back(0.1); C.push_back(0.0);
    H.push_back(0.1); C.push_back(5e-324);
    H.push_back(0.0); C.push_back(0.5);
    H.push_back(0.1); C.push_back(1.0);
    H.push_back(NAN); C.push_back(0.5);
    H.push_back(17.0); C.push_back(0.5);
    const size_t N = H.size();

    std::vector<double> Ws(N, 0.0), Wb(N, 0.0);
    std::vector<int>    Rs(N, -9),  Rb(N, -9);
    long nA = 0, nB = 0, nO = 0;
    for (size_t i = 0; i < N; ++i) {
        Ws[i] = nw::implied_variance_wb(H[i], C[i], &Rs[i]);
        if (Rs[i] == nw::NW_A) ++nA; else if (Rs[i] == nw::NW_B) ++nB; else ++nO;
    }
    nw::implied_variance_wb_batch(H.data(), C.data(), Wb.data(), Rb.data(), (int)N);

    long mism = 0, rmism = 0; size_t at = 0;
    for (size_t i = 0; i < N; ++i) {
        if (Rb[i] != Rs[i]) ++rmism;
        if (!(same_bits(Wb[i], Ws[i]) || (std::isnan(Wb[i]) && std::isnan(Ws[i])))) { if (!mism) at = i; ++mism; }
    }
    std::printf("[1] BIT-IDENTITY batch == scalar over %zu quotes (A %ld, B %ld, shipped fallback %ld) : mismatches = %ld\n",
                N, nA, nB, nO, mism);
    if (mism) std::printf("      first at h=%.17g c=%.17g batch=%a scalar=%a (codes %d %d)\n", H[at], C[at], Wb[at], Ws[at], Rb[at], Rs[at]);
    std::printf("[2] CODE agreement batch == scalar                : mismatches = %ld\n", rmism);
    {
        const double EPS = 2.220446049250313e-16;
        double wr = 0; long used = 0, bad = 0;
        for (size_t k = 0; k < tr.size(); ++k) {
            const size_t i = n_tr0 + k;
            if (Rs[i] == nw::NW_OUT) continue;
            ++used;
            const double e = std::fabs(std::sqrt(Wb[i]) - tr[k].v) / tr[k].v;
            if (e > 1e-15) ++bad;
            wr = std::max(wr, e);
        }
        std::printf("[3] ACCURACY batch path, whole-book truth set (n=%ld): max rel %.3e (%.2f ULP)  pts>1e-15 = %ld\n\n",
                    used, wr, wr / EPS, bad);
    }

    if (n_feed == 0) { std::printf("[4] THROUGHPUT skipped: market_feed.csv not found (not redistributed; see README.md)\n"); return 0; }
    std::vector<double> fh(H.begin(), H.begin() + n_feed), fc(C.begin(), C.begin() + n_feed);
    const int MF = (int)fh.size();
    std::printf("[4] THROUGHPUT, FULL feed (all routes), best of 5\n");
    std::printf("    %8s  %11s  %10s   %8s\n", "batch", "whole-book", "shipped", "shp/wb");
    for (int M : {64, 256, 1024, 4096, MF}) {
        if (M > MF) continue;
        const int R = std::max(20, 4000000 / M);
        std::vector<double> w1(M), w2(M); std::vector<int> r1(M);
        double b1 = 1e30, b2 = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            double t0 = now_s();
            for (int r = 0; r < R; ++r) nw::implied_variance_wb_batch(fh.data(), fc.data(), w1.data(), r1.data(), M);
            double t1 = now_s();
            for (int r = 0; r < R; ++r) volfi_annulus::implied_variance_grid_batch(fh.data(), fc.data(), w2.data(), M);
            double t2 = now_s();
            const double nn = (double)M * R;
            b1 = std::min(b1, 1e9 * (t1 - t0) / nn);
            b2 = std::min(b2, 1e9 * (t2 - t1) / nn);
        }
        std::printf("    %8d  %8.2f ns  %7.2f ns   %8.2f\n", M, b1, b2, b2 / b1);
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
