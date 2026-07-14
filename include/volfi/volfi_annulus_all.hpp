// volfi_annulus_all.hpp -- umbrella header for the Phase-7 broad-range volfi
// annulus implied-variance inverter.  Include this one file and you get the whole
// broad-domain inverter (v up to ~8, |x|=h up to ~16) at machine precision with
// bit-identical scalar/batch results.
//
//   #include "volfi_annulus_all.hpp"
//   double w = volfi_annulus::implied_variance_otm(h, c);      // w = sigma^2
//   double s = volfi_annulus::implied_volatility_otm(h, c, T); // sigma (opt.)
//
// Conventions: h = |log(K/F)| >= 0, c = normalized OTM call price in (0,1),
//   c = Phi(-h/v + v/2) - e^h Phi(-h/v - v/2), v = sigma*sqrt(T) = sqrt(w).
//
// Charts (branchless per-context seam prices route between them):
//   WING    resurgent evaluator, true W=h^2/2w >= 3  (GL-40, erf-free)
//   LEFT    matched small-h chart, h < 0.3, v <= 1.70
//   CENTRAL Phase-6 bivariate-Chebyshev main table, h in [0.3,6.65], v <= 2
//   RIGHT   erf-free seed + exact-equation Newton, v > 1.70 / h > 6.65
//
// Build: -std=c++17 -O3 -march=native -ffp-contract=off -fno-fast-math
//        (also -mno-avx512f for AVX2, or -mavx2 -mno-avx512f for no-FMA).
#ifndef VOLFI_ANNULUS_ALL_HPP
#define VOLFI_ANNULUS_ALL_HPP

// wing_variance() is referenced by volfi_annulus.hpp's deep-OTM sub-router, so the
// wing header must be visible first.  (tables + broadrange + paper_volfi are pulled
// in by volfi_annulus.hpp itself.)
#include "volfi_annulus_wing.hpp"
#include "volfi_annulus.hpp"

#endif // VOLFI_ANNULUS_ALL_HPP
