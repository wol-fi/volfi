# Documentation

- **[volfi_v0.2.0_paper.pdf](volfi_v0.2.0_paper.pdf)** — the technical paper for volfi v0.2.0:
  *A Routed, Vectorizable Table Inverter for Machine-Precision Black–Scholes Implied
  Volatility.* Covers the four-chart routing and its branch-point boundaries, the
  degree-selection and accuracy certification against a 40-digit `mpmath` oracle, the
  cross-build bit-identity construction (`-ffp-contract=off`), and the full timing
  methodology behind the paper's table.

The paper is the authoritative technical reference; the top-level
[README](../README.md) is the quick-start, and [`reproduce/`](../reproduce) reproduces the
accuracy and timing results.
