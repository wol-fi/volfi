# Notice

`volfi` is released under the BSD 3-Clause License. See `LICENSE` for the full license terms.

If this software, method, benchmark, or documentation influences research, software, internal
development, or published results, please cite the repository and the accompanying paper
(`docs/volfi_v0.2.0_paper.pdf`).

This software is provided as a research kernel and without warranty. It is not a drop-in
replacement for production pricing, risk-management, execution, or model-validation systems
without independent validation and appropriate integration work.

## Third-party components not included

- **Let's Be Rational** (Peter Jäckel) is used only as the reference solver in the optional
  comparison benchmark (`reproduce/benchmark_vec.cpp`, `reproduce/accuracy_vs_lbr.cpp`). It is
  the author's own copyrighted work and is **not** redistributed here; obtain it separately to
  run those two files. Nothing else in the repository depends on it.
- The market-feed workload in the paper's timing table is derived from a **licensed
  OptionMetrics** end-of-day option feed and is **not** included. `reproduce/README.md`
  documents how to regenerate an equivalent normalized `(h, c)` feed from your own licensed
  data. All other benchmarks and the entire verification suite are synthetic or oracle-based
  and run without it.
