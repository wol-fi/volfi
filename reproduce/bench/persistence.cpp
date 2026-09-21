// persistence.cpp -- how persistent is a node's membership across consecutive snapshots?
// Reads the licensed vendor file spx2024_otmcall_tradeable.csv (never published), groups quotes by
// node (exdate, K, cp_flag), orders each node's quotes by date, and for every pair of consecutive
// quoting days counts: (a) crossings of a = 2 pi (the whole-book kernel's one branch), (b) changes
// of the whole-book region code (A / B / fallback), (c) changes of the shipped router's chart
// (Near / Far / Wing / Upper).  Also the median |delta a| per day and the distribution of a.
// Answers the referee's question whether a book sorted once by a (or bucketed once by chart) stays
// sorted, and how many quotes per day would have to move.
// Build: g++ -std=c++17 -O3 -w -I<volfi_annulus> -I<volfi_test/src> persistence.cpp -o persistence ; run from bench_run.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
namespace nw = volfi_wb;
struct Q { std::string date; double h, c; };
static int route(double h, double c) {
    using namespace volfi_annulus;
    if (!(c > 0.0) || c >= 1.0 || !(h > 0.0)) return 4;
    return detail::grid_endpoint_route(h, c);   // 1 near, 2 upper, 3 wing, else far (0)
}
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "spx2024_otmcall_tradeable.csv";
    std::ifstream fin(path); if (!fin) { std::printf("cannot open %s\n", path); return 2; }
    std::string line; std::getline(fin, line);
    std::vector<std::string> cols; { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) { if (!c.empty() && c.back() == '\r') c.pop_back(); cols.push_back(c); } }
    int iDate = -1, iEx = -1, iK = -1, iCP = -1, iH = -1, iC = -1;
    for (int i = 0; i < (int)cols.size(); ++i) { if (cols[i] == "date") iDate = i; if (cols[i] == "exdate") iEx = i; if (cols[i] == "K") iK = i; if (cols[i] == "cp_flag") iCP = i; if (cols[i] == "h") iH = i; if (cols[i] == "c") iC = i; }
    if (iDate < 0 || iEx < 0 || iK < 0 || iCP < 0 || iH < 0 || iC < 0) { std::printf("columns not found\n"); return 2; }
    std::map<std::string, std::vector<Q>> nodes; long nq = 0;
    while (std::getline(fin, line)) {
        std::stringstream ss(line); std::string f; std::vector<std::string> v; while (std::getline(ss, f, ',')) v.push_back(f);
        if ((int)v.size() <= std::max({iDate, iEx, iK, iCP, iH, iC})) continue;
        const double h = std::atof(v[iH].c_str()), c = std::atof(v[iC].c_str());
        if (!(h > 1e-4 && h < 16.5) || !(c > 0 && c < 1)) continue;
        nodes[v[iEx] + "|" + v[iK] + "|" + v[iCP]].push_back({v[iDate], h, c}); ++nq;
    }
    long pairs = 0, cross2pi = 0, wbchange = 0, rtchange = 0, nodesN = 0, multi = 0;
    long regA = 0, regB = 0, regO = 0; std::vector<double> dA; std::vector<double> aAll;
    std::map<int, long> rtcount;
    for (auto& kv : nodes) {
        auto& qs = kv.second; ++nodesN; if (qs.size() < 2) continue; ++multi;
        std::sort(qs.begin(), qs.end(), [](const Q& x, const Q& y) { return x.date < y.date; });
        int prevSide = -1, prevWb = -1, prevRt = -1; double prevA = 0;
        for (size_t i = 0; i < qs.size(); ++i) {
            const double E = nw::nw_expm1(qs[i].h); const double a = std::log1p(E / qs[i].c);
            int code; nw::implied_variance_wb(qs[i].h, qs[i].c, &code);
            const int side = a < nw::NW_TWO_PI ? 0 : 1, rt = route(qs[i].h, qs[i].c);
            aAll.push_back(a); if (code == 1) ++regA; else if (code == 2) ++regB; else ++regO; ++rtcount[rt];
            if (i > 0) { ++pairs; if (side != prevSide) ++cross2pi; if (code != prevWb) ++wbchange; if (rt != prevRt) ++rtchange; dA.push_back(std::fabs(a - prevA)); }
            prevSide = side; prevWb = code; prevRt = rt; prevA = a;
        }
    }
    std::sort(dA.begin(), dA.end()); std::sort(aAll.begin(), aAll.end());
    auto q = [&](std::vector<double>& v, double p) { return v.empty() ? 0.0 : v[(size_t)(p * (v.size() - 1))]; };
    std::printf("=== node persistence on %s ===\n", path);
    std::printf("  quotes %ld, nodes (exdate,K,cp) %ld, nodes quoted on >= 2 days %ld, consecutive-day pairs %ld\n", nq, nodesN, multi, pairs);
    std::printf("  whole-book regions over all quotes: A %ld (%.2f%%)  B %ld (%.2f%%)  fallback %ld (%.3f%%)\n", regA, 100.0 * regA / nq, regB, 100.0 * regB / nq, regO, 100.0 * regO / nq);
    std::printf("  shipped router over all quotes: near %ld  far %ld  wing %ld  upper %ld\n", rtcount[1], rtcount[0], rtcount[3], rtcount[2]);
    std::printf("  a: median %.3f, 90%% %.3f, 99%% %.3f, max %.3f\n", q(aAll, 0.5), q(aAll, 0.9), q(aAll, 0.99), aAll.empty() ? 0 : aAll.back());
    std::printf("  |delta a| day to day: median %.4f, 90%% %.4f, 99%% %.4f\n", q(dA, 0.5), q(dA, 0.9), q(dA, 0.99));
    std::printf("  a = 2 pi crossings: %ld of %ld pairs (%.3f%%)   whole-book region changes: %ld (%.3f%%)   shipped chart changes: %ld (%.3f%%)\n",
                cross2pi, pairs, 100.0 * cross2pi / pairs, wbchange, 100.0 * wbchange / pairs, rtchange, 100.0 * rtchange / pairs);
    return 0;
}
