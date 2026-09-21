// bracket_cert.cpp -- pointwise residual-bracket certificate in Arb ball arithmetic.
// For every point of a truth file (int64 n; n x {h, c, v_oracle}) the entry point's returned double vhat
// is certified through
//     C(h,(1-eta) vhat) < c < C(h,(1+eta) vhat),   eta = eps/(1+eps),
// which by strict monotonicity of C in v proves |vhat - v*|/v* <= eps. The oracle column is not read.
// h, c and vhat enter as exact binary64 values, every sign test is a certain ball test, and the
// precision is doubled until both signs resolve.
// Build (Arb 2): g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -march=native -w -I../include/volfi accuracy/bracket_cert.cpp -o out/bracket_cert -lflint-arb -lflint
// Build (FLINT 3): the same with  -lflint  only.
// Run from data/:  ../out/bracket_cert 1e-15 wb_truth.bin wb_truth_boundary.bin ...
#include <cstdio>
#include <cstdint>
#include <cmath>
#if __has_include(<flint/arb.h>)
#include <flint/arb.h>
#include <flint/arb_hypgeom.h>
#else
#include <arb.h>
#include <arb_hypgeom.h>
#endif
#include "volfi_annulus_all.hpp"
#include "volfi_wb.hpp"

// r = C(h,v) - c,  C(h,v) = Phi(v/2 - h/v) - e^h Phi(-v/2 - h/v),  Phi(x) = erfc(-x/sqrt2)/2
static void residual(arb_t r, const arb_t h, const arb_t v, const arb_t c, slong prec) {
    arb_t x, d, s2, t, u; arb_init(x); arb_init(d); arb_init(s2); arb_init(t); arb_init(u);
    arb_sqrt_ui(s2, 2, prec); arb_div(x, h, v, prec); arb_mul_2exp_si(d, v, -1);
    arb_sub(t, x, d, prec); arb_div(t, t, s2, prec); arb_hypgeom_erfc(t, t, prec);      // erfc(-(v/2-h/v)/sqrt2)
    arb_add(u, x, d, prec); arb_div(u, u, s2, prec); arb_hypgeom_erfc(u, u, prec);      // erfc(-(-v/2-h/v)/sqrt2)
    arb_exp(x, h, prec); arb_mul(u, u, x, prec); arb_sub(t, t, u, prec); arb_mul_2exp_si(t, t, -1);
    arb_sub(r, t, c, prec);
    arb_clear(x); arb_clear(d); arb_clear(s2); arb_clear(t); arb_clear(u);
}
static double lower(const arb_t m) { arf_t a; arf_init(a); arb_get_lbound_arf(a, m, 64); double d = arf_get_d(a, ARF_RND_DOWN); arf_clear(a); return d; }

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: bracket_cert EPS file.bin ...\n"); return 2; }
    long N = 0, OK = 0, FAIL = 0, UND = 0;
    arb_t h, c, v, e, eta, vl, vu, lo, hi, m; arb_init(h); arb_init(c); arb_init(v); arb_init(e); arb_init(eta);
    arb_init(vl); arb_init(vu); arb_init(lo); arb_init(hi); arb_init(m);
    for (int a = 2; a < argc; ++a) {
        FILE* f = std::fopen(argv[a], "rb"); if (!f) { std::printf("cannot open %s\n", argv[a]); return 2; }
        int64_t n; if (std::fread(&n, 8, 1, f) != 1) return 2;
        long ok = 0, fail = 0, und = 0; double mlo = INFINITY, mhi = INFINITY; slong pmax = 0;
        for (int64_t i = 0; i < n; ++i) {
            double b[3]; if (std::fread(b, 8, 3, f) != 3) return 2;
            int code; const double vh = std::sqrt(volfi_wb::implied_variance_wb(b[0], b[1], &code));
            if (!(vh > 0)) { ++und; continue; }
            bool done = false;
            for (slong prec = 256; prec <= 8192 && !done; prec *= 2) {
                arb_set_d(h, b[0]); arb_set_d(c, b[1]); arb_set_d(v, vh);
                arb_set_str(e, argv[1], prec); arb_add_ui(eta, e, 1, prec); arb_div(eta, e, eta, prec);
                arb_sub_ui(vl, eta, 1, prec); arb_neg(vl, vl); arb_mul(vl, vl, v, prec);
                arb_add_ui(vu, eta, 1, prec); arb_mul(vu, vu, v, prec);
                residual(lo, h, vl, c, prec); residual(hi, h, vu, c, prec);
                if (arb_contains_zero(lo) || arb_contains_zero(hi)) continue;
                done = true; if (prec > pmax) pmax = prec;
                if (arb_is_negative(lo) && arb_is_positive(hi)) {
                    ++ok; arb_div(m, lo, c, prec); arb_neg(m, m); double x = lower(m); if (x < mlo) mlo = x;
                    arb_div(m, hi, c, prec); x = lower(m); if (x < mhi) mhi = x;
                } else { if (fail < 5) std::printf("     fail h=%.17g c=%.17g vhat=%.17g code=%d\n", b[0], b[1], vh, code); ++fail; }
            }
            if (!done) ++und;
        }
        std::fclose(f);
        std::printf("%-22s eps=%s n=%6lld certified=%6ld failed=%4ld unresolved=%ld  min margin -F-/c=%.3e F+/c=%.3e  max prec=%ld\n",
                    argv[a], argv[1], (long long)n, ok, fail, und, mlo, mhi, (long)pmax);
        N += n; OK += ok; FAIL += fail; UND += und;
    }
    std::printf("TOTAL eps=%s n=%ld certified=%ld failed=%ld unresolved=%ld\n", argv[1], N, OK, FAIL, UND);
    return (FAIL == 0 && UND == 0) ? 0 : 1;
}
