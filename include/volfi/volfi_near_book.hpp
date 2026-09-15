// volfi_near_book.hpp -- the certified NEAR chart restricted to the book, laid out
// for a GPU.  v0.3.0-test, EXPERIMENTAL.
// =============================================================================
// WHAT.  The same function as volfi_near_certified::near_variance_certified, on the
// tau range the traded book occupies: tau = (h/a)^2 in [0, NB_TAU_BOOK] = [0, 0.05]
// against a feed maximum of 0.033.  Quotes above it (the ceiling region, empty on
// the feed) fall back to the 12-cell chart, unchanged.
//
// WHY.  On an H100 the 12-cell chart measured 0.062 ns per quote against the
// shipped NEAR kernel's 0.037, and the difference is not arithmetic, of which the
// certified chart does less, but layout: graded rows with data-dependent loop
// bounds, indexed through three offset arrays, read at warp-divergent addresses.
// Two facts from the certificate remove that.
//   (1) The tau-radius is exactly 1 everywhere (fixed-rho theorem), so on
//       [0, 0.05] every cell needs about nine rows; the 19-row shallow cells of the
//       full chart exist only for the ceiling at tau = 0.416, which the book never
//       reaches.
//   (2) The a-direction singularities sit on the imaginary axis at height
//       2 pi/(1 + sqrt tau) >= 5.1, independent of a, so with tau capped a SINGLE
//       cell certifies over the whole a-range [0, A_WING]: 10 rows, 152
//       coefficients (gen/certified_book.py).
// One cell means every coefficient index is a compile-time constant.  The row
// profile is emitted as template specialisations NBRow<cell,m>, so g++ and nvcc
// unroll the entire 2-D Clenshaw into straight-line fma whose coefficient operands
// come from the constant bank, nothing is loaded through an index, and nothing
// diverges.  That is the shape of the shipped NEAR kernel, with less arithmetic
// and three divisions instead of seven.
//
// DIVISIONS.  The chart's own divisions are E/c, s = f/(2+f) inside the log, and
// theta = h/a.  The affine maps to [-1,1] use reciprocals the generator rounds
// once from 40 digits (NBCell<>::st, ::sa, ::ba), and the mode-3 log1p forms its
// reduced argument exactly without the Sterbenz correction's division.
//
// SHARED SOURCE.  This header is compiled for the host by g++ and for both host
// and device by nvcc; the Clenshaw is the same text in all three, which is a
// stronger identity claim than a hand-mirrored copy.  Coefficients are read
// through NB_C(i), which resolves to the __constant__ device copy under
// __CUDA_ARCH__ and to the constexpr host array otherwise.
//
// Layout selection: -DNB_LAYOUT=1 (default, one cell), 2 or 3.  The multi-cell
// layouts exist for the measurement; an unbucketed multi-cell chart on a GPU costs
// the SUM of its cells per mixed warp.
// =============================================================================
#ifndef VOLFI_NEAR_BOOK_HPP
#define VOLFI_NEAR_BOOK_HPP

#include <cmath>
#include <cstdint>

#ifndef NB_LAYOUT
#define NB_LAYOUT 1
#endif
#if   NB_LAYOUT == 1
#include "volfi_near_book_tables_L1.hpp"
#elif NB_LAYOUT == 2
#include "volfi_near_book_tables_L2.hpp"
#elif NB_LAYOUT == 3
#include "volfi_near_book_tables_L3.hpp"
#else
#error "NB_LAYOUT must be 1, 2 or 3"
#endif

#if defined(__CUDACC__)
#define NB_HD __host__ __device__
#else
#define NB_HD
#endif

namespace volfi_near_book {

// Coefficient access.  On the device the array lives in __constant__ memory under
// the name d_NB_COEFF, declared by the .cu before it includes this header.
#if defined(__CUDA_ARCH__)
#define NB_C(i)   (d_NB_COEFF[(i)])
#define NB_FMA(a, b, c) fma((a), (b), (c))
#else
#define NB_C(i)   (NB_COEFF[(i)])
#define NB_FMA(a, b, c) std::fma((a), (b), (c))
#endif

// Inner Clenshaw of one tau-row: DEG coefficients at NB_COEFF[OFF .. OFF+DEG-1],
// same operation sequence as clenshaw2_graded in volfi_near_certified.hpp.
template<int DEG, int OFF>
NB_HD inline double nb_row(double xa, double xa2) {
    if constexpr (DEG <= 0) {
        (void)xa; (void)xa2;
        return 0.0;
    } else {
        double e0 = 0.0, e1 = 0.0;
#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__)
#pragma GCC unroll 64
#endif
        for (int j = DEG - 1; j >= 1; --j) {
            const double b = NB_FMA(xa2, e0, NB_C(OFF + j)) - e1;
            e1 = e0; e0 = b;
        }
        return NB_FMA(xa, e0, NB_C(OFF)) - e1;
    }
}

// Outer Clenshaw over rows M .. 1 (row 0 is finished by the caller).
template<int CELL, int M>
NB_HD inline void nb_rows(double xt2, double xa, double xa2, double& g0, double& g1) {
    if constexpr (M >= 1) {
        const double r = nb_row<NBRow<CELL, M>::deg, NBRow<CELL, M>::off>(xa, xa2);
        const double b = NB_FMA(xt2, g0, r) - g1;
        g1 = g0; g0 = b;
        nb_rows<CELL, M - 1>(xt2, xa, xa2, g0, g1);
    } else {
        (void)xt2; (void)xa; (void)xa2; (void)g0; (void)g1;
    }
}

// V(tau, a) on cell CELL from the chart coordinates.  Straight-line after
// instantiation: NBCell<CELL>::nrow rows, sum of NBRow<CELL,m>::deg inner steps.
template<int CELL>
NB_HD inline double nb_V(double tau, double a) {
    const double xt  = NB_FMA(tau, NBCell<CELL>::st, -1.0);
    const double xa  = NB_FMA(a,   NBCell<CELL>::sa, NBCell<CELL>::ba);
    const double xt2 = 2.0 * xt;
    const double xa2 = 2.0 * xa;
    double g0 = 0.0, g1 = 0.0;
    nb_rows<CELL, NBCell<CELL>::nrow - 1>(xt2, xa, xa2, g0, g1);
    const double r = nb_row<NBRow<CELL, 0>::deg, NBRow<CELL, 0>::off>(xa, xa2);
    return NB_FMA(xt, g0, r) - g1;
}

// Cell of a.  With one cell this is a constant; with two or three it is a
// one- or two-compare ladder against the per-cell lower edges, which are
// constexpr scalars and therefore usable on the device as constant expressions.
NB_HD inline int nb_cell_of(double a) {
    int j = 0;
#if NB_LAYOUT >= 2
    if (a >= NBCell<1>::a0) j = 1;
#endif
#if NB_LAYOUT >= 3
    if (a >= NBCell<2>::a0) j = 2;
#endif
    (void)a;
    return j;
}

// w = v*v from the coordinates, dispatching on the cell.  Returns a negative
// value when the quote is above the book's tau range for its cell, so the caller
// can fall back to the full 12-cell chart.
NB_HD inline double nb_variance_from_coords(double theta, double tau, double a) {
    const int cell = nb_cell_of(a);
    double V;
    switch (cell) {
        case 0:
            if (!(tau <= NBCell<0>::t1)) return -1.0;
            V = nb_V<0>(tau, a);
            break;
#if NB_LAYOUT >= 2
        case 1:
            if (!(tau <= NBCell<1>::t1)) return -1.0;
            V = nb_V<1>(tau, a);
            break;
#endif
#if NB_LAYOUT >= 3
        case 2:
            if (!(tau <= NBCell<2>::t1)) return -1.0;
            V = nb_V<2>(tau, a);
            break;
#endif
        default:
            return -1.0;
    }
    const double v = theta * V;
    return v * v;
}

#undef NB_C
#undef NB_FMA

} // namespace volfi_near_book

// ---------------------------------------------------------------------------
// Host wrapper: the book chart with the 12-cell fallback, using the mode-3
// coordinates of volfi_near_certified.  This is the CPU reference the device
// result must match bit for bit.  Host-only code, but declared in both of
// nvcc's passes, since host code is parsed in the device pass as well.
// ---------------------------------------------------------------------------
#include "volfi_near_certified.hpp"
namespace volfi_near_book {

// Returns w.  *used_fallback (optional) is set to 1 when the quote was above the
// book's tau range and the 12-cell chart answered instead.
inline double near_variance_book(double h, double c, int* used_fallback = nullptr) {
    namespace nc = volfi_near_certified;
    const nc::near_coords z = nc::near_coords_of(h, c);
    const double w = nb_variance_from_coords(z.theta, z.tau, z.a);
    if (w >= 0.0) { if (used_fallback) *used_fallback = 0; return w; }
    if (used_fallback) *used_fallback = 1;
    return nc::near_variance_from_coords(z);
}

} // namespace volfi_near_book

#endif // VOLFI_NEAR_BOOK_HPP
