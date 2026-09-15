# Documentation

- **[volfi_v0.3.0_paper.pdf](volfi_v0.3.0_paper.pdf)** — the paper, *Implied Volatility in
  One Straight Line: Machine Precision at Vector-Hardware Throughput.* The intrinsic
  coordinates and the exact rational rows (Proposition 1 with proof), the book kernel built
  from them, the routed four-chart inverter behind it with its branch-point-sized table and
  matched endpoint charts, the accuracy campaigns against a 40-digit `mpmath` oracle (including
  the 20,000-point campaign on every switch of the kernel), the cross-build bit-identity
  construction (`-ffp-contract=off`, `--fmad=false`), and the like-for-like timing methodology
  on the CPU (one binary with Let's Be Rational at its release flags and the PDE method of
  Matić, Radoičić and Stefanica) and on one NVIDIA H100.

The paper is the authoritative technical reference; the top-level
[README](../README.md) is the quick-start, [`reproduce/`](../reproduce) reproduces the v0.2.4
suite and [`reproduce/book/`](../reproduce/book) the v0.3.0 campaign.
