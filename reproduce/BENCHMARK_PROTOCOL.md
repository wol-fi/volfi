# Reproducing the timing results

> **Chart names.** The four charts are `NEAR` (small moneyness, `h < 0.3`), `FAR` (the Chebyshev
> table, `0.3 <= h <= H_BOX`), `WING` (deep OTM, below the wing seam) and `UPPER` (large
> volatility, above the ceiling). Paper and source use the same names.
>
> They were `LEFT`, `CENTRAL`, `RIGHT` and `WING` up to v0.2.3. The pair `LEFT`/`RIGHT` reads as
> two ends of one axis and is not — `LEFT` was a condition on `h`, `RIGHT` a condition on `c` —
> so three of the four were renamed in v0.2.4. Two places still carry the old letters, both
> deliberately:
>
> - the per-point chart tag inside `oracle_*.bin`, which is certification data and is translated
>   for display by `verify_vec` (`C`→`F`, `L`→`N`, `R`→`U`);
> - `br::cwing_price` and `br::c2_price`, which name *seams* of the retired v0.2.2 geometry and
>   are kept only so `old_covered()` can measure what the pre-v0.2.3 architecture covered.
>
> The rename touched ~750 identifiers across the headers, both SIMD twins, the CUDA port and
> every harness. It is **bitwise neutral**, and that was verified rather than assumed: a
> 339,430-line hex dump of every driver's output over 66,000 points spanning all four charts is
> byte-identical before and after, on all three instruction sets. The reference numbers in §6
> and §7 were measured on v0.2.3 and carry over unchanged for the same reason.

This folder is self-contained. It holds the implementation, the golden verification vectors
(`oracle_*.bin`), the market feed (`market_feed.csv`), and every harness needed to reproduce
the accuracy, bit-identity and throughput results reported in the paper.

Accuracy and bit-identity are host-independent: §3 must give exactly the printed values on any
machine. Throughput is host-specific; §4–§6 reproduce the *procedure*, and §7 lists the
reference numbers our host produced so you can judge whether yours is in the same regime.

---

## 1. Requirements

- **CPU**: x86-64. AVX-512 for the eight-wide column, AVX2 for the four-wide one; without
  either, the scalar build still runs and still passes every correctness gate.
- **OS/toolchain**: Linux (native or WSL2), GCC ≥ 11 or Clang ≥ 14, C++17.
- **Reference solver** (only for the comparison columns): *Let's Be Rational*, rev. 1520, from
  <http://www.jaeckel.org/>. Unpack it and point `$LBR` at the directory.
- **GPU section only**: an NVIDIA datacenter GPU with real fp64 throughput (H100/A100 class)
  and CUDA ≥ 12. Consumer cards run fp64 at 1/32–1/64 rate and will not reproduce §7's numbers,
  though they will reproduce its correctness check.

Everything is built by one script, which encodes the flags and the reference-solver source
list. Use it rather than assembling command lines by hand:

```bash
cd bench_run
./build_all.sh "/path/to/LetsBeRational"
```

It produces `smoke_512/256/scl`, `vv_512/256/scl`, `bench_512`, `bench_256` and `bp`.

Two things it gets right that a hand-written command line usually does not. The reference
solver's folder ships Excel, Octave and Python bindings beside the four sources actually
needed, so a `"$LBR"/*.cpp` glob fails on `octave/oct.h` and `Python.h`; only
`lets_be_rational.cpp`, `normaldistribution.cpp`, `rationalcubic.cpp` and `erf_cody.cpp` are
compiled. And every path is quoted, which matters because these directories contain spaces.

The flags it uses are:

```
-std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -DNO_XL_API -w -I..
```

Both floating-point flags are load-bearing. `-ffp-contract=off` makes every fusion an explicit
`fma` in the source, which is what makes results identical across compilers and instruction
sets; `-ffast-math` would break that and must never be used. `-funroll-loops` is a pure-speed
flag (bitwise neutral). Re-run §3 after any compiler change: both are codegen settings, and
the guarantee is claimed for tested configurations, not asserted for all.

## 2. Machine preparation (CPU timings)

Timings need a quiet machine; correctness does not.

- Mains power, no browser/IDE/file-sync running, no other heavy processes.
- Optionally `sudo cpupower frequency-set -g performance`.
- Pin every timed run: `taskset -c 2 nice -n -5 ./binary` (drop `nice` if it needs root).
- Do **not** use a shared or virtualized cloud vCPU. Run-to-run variation there is several
  times the effect being measured.

## 3. Correctness gates (host-independent — must pass before any timing)

```bash
for b in smoke_512 smoke_256 smoke_scl; do ./$b | tail -1; done
for b in vv_512 vv_256 vv_scl; do ./$b; done
```

Each `smoke_*` must print `SMOKE PASS`. Each `vv_*` must print:

- `BIT-IDENTITY scalar==batch:  grid=0  permuted=0  fixed-h=0`
- `ACCURACY sqrt(w) vs oracle:  n=3511  pts>1e-15=0`
- per-chart worst errors `[A] 8.29e-16  [F] 4.49e-16  [N] 8.45e-16  [U] 4.40e-16  [W] 4.16e-16`

All three builds — AVX-512, AVX2, scalar — must print the **same** accuracy numbers and zero
mismatches. That equality across instruction sets is the determinism claim; a difference
anywhere is a real failure, not tolerance.

`bench_512` additionally asserts internally, and must report:

- `GRIDBATCH_VS_SCALAR broad grid ... mismatches=0 (>1ulp=0)`
- `GRIDBATCH_VS_SCALAR market feed (speculative driver): mismatches=0`
- `NEW 4-chart coverage (WING+NEAR+FAR+UPPER) = 120000/120000 = 100.00%`

## 4. Throughput

```bash
for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ./bench_512 > run512_$i.txt; done
for i in 1 2 3 4 5; do taskset -c 2 nice -n -5 ./bench_256 > run256_$i.txt; done
```

Take medians across the five repetitions. Each internal row is itself a median over fifteen
passes of eight repeats. Spread across repetitions should be a few percent or less; if it is
not, the machine is not quiet enough.

**Abort if either binary reports `GRIDBATCH_VS_SCALAR ... mismatches` other than 0.**

Where the output maps in the paper's timing table:

| output line | table row |
|---|---|
| `FAR h=1.00 v in [0.41,1.80]` | central batch |
| `NEAR h=0.20` | small-moneyness batch |
| `UPPER h=1.00 v in [2.10,8.00]` | large-volatility batch |
| `TIMING market D_gridbatch_feed` | market feed |
| `TIMING stream E_warm2_feed` | warm streaming steady state |

Columns: `B_lbr` → reference, `C_scalar` → our scalar entry, `D_batch`/`D_gridbatch` → batch.
The fixed-moneyness rows hoist the shared per-`h` constants (kernel benchmarks, not expiry
slices — at one maturity `h` varies across strikes). The market row is timed all-inside: every
method pays its own per-quote input transform within the timed loop.

## 5. Driver phase breakdown (optional)

Splits the cold market-feed cost into its components. Needs no reference solver.

```bash
g++ $FLAGS -march=native bench_phases.cpp -o bp && taskset -c 2 nice -n -5 ./bp
```

## 6. Reference results

Intel Core i5-1145G7 (Tiger Lake), Ubuntu 22.04 under WSL2, GCC 11.4, quiesced and pinned;
medians of five repetitions, ns per quote.  **v0.2.3, campaign of 2026-07-27** — these are the
numbers in Table 4 of the paper.

| Workload | reference | ours scalar | batch AVX-512 | batch AVX2 |
|---|---|---|---|---|
| Far batch, h=1 | 169 | 69 | **17** | **18** |
| Near batch, h=0.2 | 139 | 141 | **26** | **42** |
| Upper batch, h=1 | 205 | 557 | **74** | 148 |
| Wing subset of the broad grid | 307 | 1225 | **250** | 347 |
| Market feed (2024 SPX) | 227 | 321 | **48** | **86** |
| Market feed, warm 2-step | — | — | **37** | 79 |

Absolute values are hardware-specific. What should reproduce on comparable hardware is the
*structure*: batch beats the reference on every workload except large-volatility AVX2; the
market feed runs ≈4.7× the reference on AVX-512 and ≈2.6× on AVX2; warm streaming is ≈6× the
reference; and the scalar entry is slower than the reference on the endpoint charts, which is
the honest cost of a design whose unit of work is a batch.

### Batch-size sweep (Figure 1), v0.2.3, 2026-07-27

`bench_sweep` → `batchsweep.txt`, medians of 7, ns per quote. The figure plots `lbr/warm2`.

| n | cold | warm2 | reference | lbr/warm2 |
|---|---|---|---|---|
| 1 | 274 | 288 | 264 | **0.92** |
| 4 | 252 | 283 | 211 | 0.75 |
| 8 | 43 | 37.2 | 221 | 5.94 |
| 64 | 60 | 37.0 | 219 | 5.90 |
| 1024 | 48.5 | 37.4 | 220 | 5.90 |
| 16384 | 44.7 | 38.3 | 230 | 5.99 |

The three claims the figure carries all hold: the warm driver is flat (5.81–6.00 over the whole
range n≥8, warm cost 36.8–38.8 ns, 5.5% spread); the step at n=8 is one AVX-512 register filling;
and at n=1 the reference wins (0.92), the single regime where it does. The cold driver's
sort-and-drain only amortizes slowly — `lbr/cold` climbs 2.1 → 5.1 between n=16 and n=8192 —
which is the contrast the streaming paragraph draws.

The n=8 cold cell (43 ns) sits below the n≥1024 plateau because a single full vector needs no
tiling and no deferred drain; the n=16 cell (101 ns) is the first to pay both. This is a real
feature of the driver, not noise, and it is why the cold curve is not plotted.

The generator was written for this campaign. The previous `batchsweep.txt` (2026-07-14) had no
generator in the repo and its small-n rows were unusable — 294 ns at n=4 against 66 ns at n=8,
the reference wandering 173–460 — because the timed region was shorter than the clock's
resolution. `bench_sweep.cpp` calibrates every timed region to ≥20 ms and rotates the slice
offset across repetitions.

### What v0.2.3 changed, and what the campaign settled

Routing changed substantially: the wing seam moved from the iso-W contour W=3 to W=3.8; the
small-moneyness chart gained a third `Binv` regime; and the two tabulated charts now share ONE
ceiling, C(h,1.85), instead of stepping between C(h,1.70) and C(h,2) at h=0.3.  On the 2024
S&P 500 feed the deep-wing share falls from **4.84% to 1.14%** and the central table's share
moves 5.11% -> 6.95%.  Over the full 641,072-quote tradeable population the mix is 91.96% /
6.81% / 1.23% / 0.00% (near / far / wing / upper), from `feed_route_mix.cpp`.

Two effects pulled against each other and were unmeasured beforehand: both seams are cheaper
to evaluate than what they replaced, while `Binv` gained a third Chebyshev chain charged to
~92% of the book.  **The campaign settled it: nothing got slower.**  Small-moneyness is
unchanged to within 0.4%, so the third `Binv` regime is free at measurement resolution, and the
239-cell table costs nothing either.  The market feed improved 12% (AVX-512) and 4% (AVX2)
purely through the collapse of the wing share, and the wing subset itself improved from 288 to
250 ns because its remaining quotes are fewer but the driver is the same.

Comparability caveat: the two FAR fixed-h rows are **not** directly comparable to the
v0.2.2 numbers, because their v-range had to move with the ceiling (from [0.45,1.95] and
[0.85,1.95] to [0.41,1.80] and [0.77,1.80]) to stay chart-pure.  The first v0.2.3 run was taken
before that was noticed and reported a spurious FAR regression; see §8.  All other rows are
like-for-like.  For the record, the v0.2.2 baseline was: central 165 / 72 / **17** / **19**;
large-volatility 205 / 565 / **77** / 152; wing 313 / 1231 / **288** / 381; feed 230 / 367 /
**55** / **90**; warm 2-step **39** / 83.

## 7. GPU

Build and run on the GPU host. Requires `market_feed.csv`, the `gpu/` sources, and the seven
headers from the parent directory.

**Before copying anything, regenerate the device headers on the host machine.** They are
mechanical transforms of the CPU headers, and a stale copy silently desynchronises the port:

```bash
cd gpu
python make_device_constants.py     # constants  -> volfi_constants_cuda.cuh
python make_device_tables.py        # main table -> volfi_annulus_tables_cuda.cuh
```

Then on the GPU host:

```bash
nvcc -O3 -arch=sm_90 -std=c++17 --fmad=false volfi_gpu_book.cu -o volfi_gpu_book
./volfi_gpu_book 5000000 300
```

`--fmad=false` is the device analogue of `-ffp-contract=off` and is required: it keeps the
device kernels on the source's explicit `fma`s so they match the host bit for bit.
Use `-arch=sm_80` for A100.

**Every chart must report `mismatches=0` at `worst_ulp=0`** — each device result is compared
against the CPU reference compiled into the same binary. A nonzero count is a real defect;
the timing is meaningless until it is fixed.

The run prints two timing sections. The first is the **market feed** at its live route mix.
The second is four **chart-pure fixed-h surfaces**, which are what give the large-volatility
chart a number at all: the tradeable feed routes 0% to it, so its feed-mix line is skipped.
Each surface is route-filtered through the same classifier the feed uses, so a row is chart-pure
by construction.

Reference below is **v0.2.2 and superseded**, kept as the comparison baseline. Three v0.2.3
changes affect it. The wing seam moved to W=3.8, so the feed's wing slice shrinks from 4.84% to
1.14% of the book and the central slice grows; `d_binv` gained the third regime; and the four
surface v-ranges moved to mirror the CPU's updated per-chart rows (FAR [0.45,1.95] →
[0.41,1.80] because the shared ceiling dropped to 1.85, WING [0.15,0.40] → [0.15,0.35] because
the seam rose to v=h/√7.6 = 0.363 at h=1). The per-chart surface rows should be near-unchanged;
the **full-book** row is the one to watch, since it is a weighted average over a route mix that
moved substantially.

Note that the surfaces were never at risk of the mislabelling that hit the CPU harness: each is
route-filtered through the live classifier, so a stale v-range discards quotes rather than
mistiming them. Watch the `off-chart` count — with the ranges above it should be zero.

**v0.2.3, campaign of 2026-07-27** (NVIDIA H100 PCIe, driver 570.124.06, CUDA 12.4, 300 passes,
two runs; raw output in `gpu_run_2026-07-27.txt`), ns per quote — these are the numbers in
Table 5 of the paper:

| row | workload | ns/quote | v0.2.2 |
|---|---|---|---|
| Far surface | h=1, v ∈ [0.41,1.80] | 0.140 | 0.141 |
| Near surface | h=0.2, v ∈ [0.30,1.60] | 0.038 | 0.038 |
| Upper surface | h=1, v ∈ [2.10,8.00] | 0.168 | 0.169 |
| Wing surface | h=1, v ∈ [0.15,0.35] | 0.669 | 0.688 |
| **Full book** | market feed, 5.01M quotes | **0.078** | 0.100 |

Feed-mix slices: left 0.057, central 0.153, wing 1.167 (v0.2.2: 0.055 / 0.173 / 0.837). Those
reproduce the measured full book to 2% (0.0763 against 0.0781); the residue is the four kernel
launches. The projection made before the run was 0.073 — the shortfall is the wing, which costs
39% more in situ than it did, its surviving quotes being fewer but deeper.

The two runs agree to three digits on every row except the far surface (0.1385 and 0.1414);
the paper's caption states this rather than claiming three digits throughout. All four charts
reported `mismatches=0 worst_ulp=0` against the CPU reference in both runs, and the route mix
printed by the device binary (91.91 / 6.95 / 1.14) matches the CPU harness exactly.

Occupancy gate: `wing_kernel`, `upper_kernel`, `near_kernel` all at 0 bytes stack frame (72, 55,
36 registers); `far_kernel` 256 bytes / 40 registers, unchanged and still the known reason it
is the slowest chart on the device.

Note the wing is no longer the device's cost centre: at 1.1% of the book it is about a sixth of
the total, against roughly 40% under the v0.2.2 routing. Two thirds is now the small-moneyness
chart. Any further device-side optimization has to act there.

Note the chart ranking inverts relative to CPU: left is the *cheapest* chart on the device and
central the most expensive, because left is closed-form arithmetic while central is a table
lookup and goes memory-bound.

A structural check that needs no GPU, only the compiler:

```bash
nvcc -O3 -arch=sm_90 -std=c++17 --fmad=false -Xptxas -v -c volfi_gpu_book.cu -o /dev/null
```

`wing_kernel`, `left_kernel` and `right_kernel` must report **0 bytes stack frame** (on our host:
72, 36 and 55 registers respectively). A nonzero frame means a coefficient array landed in device
local memory and throughput will be several times worse. `far_kernel` reports a 256-byte frame;
that is known and is the likely reason it is the slowest chart on the device.

*Obtaining a suitable instance:* prefer a **PCIe** H100 over SXM5. SXM parts sit on an NVLink
fabric, and if the host's fabric manager has not registered the GPU, CUDA fails with
`system not yet initialized` (`cudaGetDeviceCount = 0`) even though `nvidia-smi` lists the
card. PCIe parts have no such dependency. Verify CUDA initializes before building anything:

```bash
echo '#include <cstdio>
#include <cuda_runtime.h>
int main(){int n=0;cudaError_t e=cudaGetDeviceCount(&n);printf("devices=%d %s\n",n,cudaGetErrorString(e));}' > t.cu
nvcc -arch=sm_90 t.cu -o t && ./t
```

If the host carries only the driver and no toolkit, build and run inside a CUDA devel
container, mounting the sources:

```bash
sudo docker run --rm --gpus all -v $HOME/volfi:/w nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  bash /w/gpu/run.sh
```

Put the build and run commands in that script rather than passing them inline; several layers
of shell quoting otherwise mangle `$PATH`.

## 8. Notes on the data

`market_feed.csv` is 30,000 (h, c) pairs resampled from the tradeable 2024 S&P 500 population,
where `c` is the actual parity-projected quote price normalized as in the paper — tick rounding
and spread noise included, not prices regenerated from fitted volatilities. It derives from a
licensed OptionMetrics file, so only these transformed pairs are distributed, together with the
filter and projection recipe needed to regenerate them from a licensed copy.

`oracle_*.bin` are mpmath golden vectors at 40 digits: `oracle_vec.bin` (verification),
`oracle_heat.bin` (the error heatmap), `oracle_stressed.bin`, `oracle_edge.bin`.

### The stale-router trap (v0.2.3, 2026-07-27)

`benchmark_vec.cpp` used to carry its own hand-mirrored copy of the routing predicate. It kept
the v0.2.2 seams (`cwing_price` / `c2_price` / `NEAR_UPPER_VSEAM`) straight through the v0.2.3
seam change, so the first campaign timed the correct library but partitioned and labelled it
with the old boundaries: the market route mix printed the v0.2.2 figures verbatim, and the
FAR fixed-h rows carried UPPER quotes above the new 1.85 ceiling, showing a 24%/43%
"regression" that did not exist. **None of the accuracy or bit-identity gates catch this** — the
values were right; only the buckets were wrong.

Two defences are now in place, and neither should be removed:

1. `route()` delegates to `volfi_annulus::detail::grid_endpoint_route`. There is no seam
   arithmetic left in the harness that can drift. `old_covered()` deliberately keeps the frozen
   phase-6 seams, because it measures what the *old* architecture covered.
2. Every fixed-h surface self-checks for chart purity against the live router and prints
   `[MIXED: ...]` when its v-range spans charts. A surface is named after a chart but *defined*
   by a v-range, so any seam move silently contaminates it. Only `FARc`, which is
   deliberately wing-contaminated, should ever carry the tag.

The reusable lesson: when a benchmark reports a partition, the partition must come from the
shipped classifier, never from a copy.
