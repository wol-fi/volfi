// bracket_dump.cpp -- for a truth file (int64 n; n x {h, c, v_oracle}) print h, c and the entry point's
// returned total volatility as exact hexadecimal doubles, plus the route code, for bracket_cert.py.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"
int main(int argc, char** argv) {
    FILE* f = std::fopen(argv[1], "rb"); if (!f) return 2;
    int64_t n; if (std::fread(&n, 8, 1, f) != 1) return 2;
    for (int64_t i = 0; i < n; ++i) {
        double b[3]; if (std::fread(b, 8, 3, f) != 3) return 2;
        int code; const double w = volfi_wb::implied_variance_wb(b[0], b[1], &code);
        std::printf("%a %a %a %d\n", b[0], b[1], std::sqrt(w), code);
    }
    return 0;
}
