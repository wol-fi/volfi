# reproduce/upper: the one-step UPPER chart and the fast scalar route (v0.3.1)

| file | what it is |
|---|---|
| `gen_upper_v031.py` | 40-digit truth set `upper_truth.bin` (2,465 points on the wing seam, the `v = 1.85` boundary, the `h` cap, `h -> 0`, `v -> 8`, `c = 1/2`) and the `erfcx` piece on `[-0.40, 0]` |
| `gen_upper_one.py` | the seed: `x0 = -Phi^{-1}(exp(-a_U/2)/2)` as a Chebyshev series in `sqrt(a_U)`, and the 13 coefficients of `R(sigma_U, zeta)` fitted in the uniform norm by a linear programme |
| `gen_fastroute.cpp` | the routing seams in the book coordinate, `a_top(h)` and `a_w(h)`, as per-band Chebyshev fits with their margins |
| `upper_truth.bin` | `int64 n`, then `n x {h, c, v_oracle}` |

The generators write research-format tables next to themselves. The shipped headers
`include/volfi/volfi_annulus_upper1_tables.hpp` and `volfi_annulus_fastroute_tables.hpp` hold the
same doubles. `gen_upper_*.py` need `numpy`, `scipy` and `mpmath`; `gen_fastroute.cpp` builds with
`g++ -std=c++17 -O2 -ffp-contract=off -I../../include/volfi gen_fastroute.cpp`.

Gates and results: `../book/seam_gate.cpp`, `../book/results/upper_one_step_gate_2026-09-17_clean.txt`,
`upper_seed_scan_2026-09-17.txt`.

The `UPPER` gate file was written by the development harness that compared the three-step chart of
v0.3.0 with a two-step and the shipped one-step candidate. That harness is not part of the
release. The shipped chart is covered by `make check`, `seam_gate.cpp` and the oracle sets.
