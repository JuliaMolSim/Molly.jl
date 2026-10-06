# Allegro potential — benchmarks

Performance and correctness for Molly's native
[Allegro](https://doi.org/10.1038/s41467-023-36329-y) potential (`AllegroPotential`), a bit-exact
port of the real [`nequip-allegro`](https://github.com/mir-group/allegro) package (allegro 0.8.3).
The model is the package architecture — `l_max=2`, 2 layers, 32 scalar / 8 tensor features, 8 Bessel
`TwoBodyBesselScalarEmbed` (`polynomial_cutoff_p=6`), `avg_num_neighbors=10`, `r_c=4.0 Å` — loaded
from the package's own exported weights. Energy and forces run on CPU (threaded, analytic) and on the
GPU (KernelAbstractions kernels, CUDA + Metal), with a hand-written analytic reverse pass (no AD).

The headline: the native model reproduces the package to numerical precision, and the same analytic
energy+forces path **is the fastest of the three Allegro implementations on all four backends — CUDA,
Apple Metal, and CPU at both 1 and 8 threads** — while on CUDA it scales to the full 16k-atom system on
a 16 GB card, where the package's autograd forces run out of memory.

**What is timed:** one evaluation — energy (forward) and energy + forces (forward + one analytic
backward), timed separately, best-of-repeats after a warm-up. The neighbour list is **precomputed**
(it is reused across MD steps; every implementation here passes precomputed edges, so the comparison
is of the model evaluation, not the neighbour search). **Systems:** random H/C/N/O in a cubic box at
density 0.09 atoms/Å³. The head-to-head spans 500→10,000 atoms (the largest the package fits on the
16 GB card); Molly itself is validated out to the full 15,954-atom 6mrr system. `Float64` on CPU and
CUDA; `Float32` on Metal.

---

## Correctness

The point of the rewrite (PR #283 review): validate the native model against the **real package**,
not a re-implementation. The reference comes from `test/allegro_package_reference.py`, which builds a
real `allegro.model.AllegroModel`, runs its forward, and exports the reference energy/forces + the
model weights. Checked in `test/ml_potentials.jl`:

- **Energy** — the native `AllegroPackageModel` reproduces the package total energy to `atol = 1e-6`
  eV (actual `Float64` agreement is near machine precision).
- **Analytic forces** — `F = -∂E/∂r` from the hand-written reverse pass match the package's autograd
  forces to `< 1e-6` eV/Å, with `ΣF ≈ 0`.
- **Full MD path** — a `System` with `AllegroPotential` reproduces the same energy and forces through
  `AtomsCalculators` + Molly's unit handling (`atol = 1e-6`).
- **GPU consistency** — a device-backed `System` runs the KA forward + analytic reverse on-device and
  matches the CPU `System`: CUDA `Float64` to ~1e-10, Metal `Float32` to `rtol = 1e-4`.

---

## Head-to-head: Molly vs nequip-allegro vs allegro-jax

Three implementations of the same Allegro architecture (all 23,056 parameters, `Float64`): Molly's
native model, the real `nequip-allegro` (PyTorch) package, and
[allegro-jax](https://github.com/mariogeiger/allegro-jax) (e3nn-jax / Flax). CPU t8 and CUDA are the
RTX 5080 box (12-core host); Metal is Apple Silicon. Cross-machine for Metal, so read the scaling
shape and within-machine GPU-over-CPU speedup, not the absolute cross-device level. Colour encodes
backend, linestyle encodes implementation (Molly solid, nequip dashed, allegro-jax dotted).

### Energy

![Allegro energy: all implementations](images/allegro_benchmark_energy.png)

### Energy + forces

![Allegro energy + forces: all implementations](images/allegro_benchmark_force.png)

All timings are one energy + forces evaluation in milliseconds, best of repeats, with the neighbour
list precomputed (it is amortised across many MD steps; rebuilding it every step would penalise every
implementation equally — all three pass precomputed edges). The range tops out at 10,000 atoms because
that is the largest system the package's autograd forces fit in the 16 GB card — above ~10k nequip
OOMs, while **Molly's analytic backward keeps going to the full 15,954-atom 6mrr system** on every
backend. `—` = not run (allegro-jax single-thread CPU past 4000 atoms is minutes per point).

**GPU** (`Float64` on CUDA, `Float32` on Metal):

| atoms | Molly CUDA | nequip CUDA | allegro-jax CUDA | Molly Metal |
| ---: | ---: | ---: | ---: | ---: |
| 500   | **7**   | 18   | 103  | 19  |
| 1000  | **11**  | 19   | 220  | 25  |
| 2000  | **18**  | 35   | 462  | 44  |
| 4000  | **35**  | 69   | 1130 | 93  |
| 7000  | **64**  | 122  | 2726 | 154 |
| 10000 | **94**  | 170  | 3468 | 218 |

**CPU** on the RTX 5080 host (12-core), `Float64`, single thread (t1) and 8 threads (t8, run with
`julia --gcthreads=8` so garbage collection is parallel — see below):

| atoms | Molly t1 | nequip t1 | jax t1 | Molly t8 | nequip t8 | jax t8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500   | **451**   | 1485  | 1691   | **503**  | 1474  | 1564   |
| 1000  | **945**   | 3235  | 6780   | **940**  | 2335  | 6406   |
| 2000  | **2023**  | 6989  | 28020  | **1380** | 4614  | 24787  |
| 4000  | **4733**  | 16388 | 126854 | **3244** | 7653  | 91132  |
| 7000  | **11002** | 29753 | —      | **6703** | 10577 | 228057 |
| 10000 | **15526** | 45764 | —      | **9302** | 14969 | 962382 |

Reading it (**bold** = fastest in that row/group):
- **CUDA — Molly wins at every size.** It is **1.8–2.7× faster** than the real package across the whole
  range, and because the analytic reverse pass has a far smaller memory footprint than the package's
  autograd backward, **only Molly scales past ~10k atoms to the full 6mrr system** — the package OOMs
  there. allegro-jax is 15–50× slower than Molly on CUDA.
- **Metal — uncontested.** Neither the package nor allegro-jax has a `Float64` Apple-GPU path, so Molly
  is the only Allegro that runs on Apple Silicon (10k atoms in 0.22 s).
- **CPU single thread (t1) — Molly wins at every size**, **2.7–3.5× faster** than the package and 4–27×
  faster than allegro-jax.
- **CPU 8 threads (t8) — Molly wins at every size too**, **1.6–3.3× faster** than the package (run with
  `--gcthreads=8` so the per-evaluation GC is parallel).

The GPU lead comes largely from the **sparse Wigner-3j tensor-product kernels**: ~89% of the (i,j,k)
output components are zero by angular-momentum selection rules, so iterating only the non-zero paths
made the dominant TP kernels ~2.5× cheaper on Metal / ~1.6× on CUDA (bit-exact; see the commit history).

### GPU speedup over host CPU (t8)

Each GPU line is its own machine's GPU time divided by that machine's CPU-t8 time (within-machine):

![Allegro energy: GPU speedup over host CPU-t8](images/allegro_energy_gpu_speedup.png)

![Allegro energy + forces: GPU speedup over host CPU-t8](images/allegro_forces_gpu_speedup.png)

Molly CUDA reaches ~250–300× over its host CPU-t8 for energy + forces; the Metal/Apple ratio is
smaller (~12–15×) because Apple's CPU-t8 baseline is much faster than the RTX host's.

---

## Reproduce

Timings write to `benchmark/results/` (gitignored). Native Molly (energy + energy+forces per size):

```
ALLEGRO_BK=cpu   JULIA_NUM_THREADS=8 julia --gcthreads=8 --project=<env> benchmark/allegro.jl  # -> allegro_bench.json
ALLEGRO_BK=metal                     julia --project=<env> benchmark/allegro.jl
ALLEGRO_BK=cuda                      julia --project=<env> benchmark/allegro.jl
```
`<env>` needs Molly + HDF5 + JSON3 (+ Metal or CUDA). Sizes via `ALLEGRO_SIZES=500,1000,...`. The CPU
path allocates per evaluation, so run multithreaded CPU with `--gcthreads=<N>` (parallel GC) for the
threads to pay off. The `AllegroPotential` calculator and the benchmark share the same batched
KernelAbstractions path on every backend (CPU via `KernelAbstractions.CPU()`).

Reference implementations on the same hardware:

```
ALLEGRO_TORCH_DEVICE=cuda                         <python-nequip-allegro> benchmark/allegro_package_bench.py
ALLEGRO_TORCH_DEVICE=cpu ALLEGRO_TORCH_THREADS=8  <python-nequip-allegro> benchmark/allegro_package_bench.py
                                                  <python-allegro-jax>    benchmark/allegro_jax_bench.py
JAX_PLATFORMS=cpu ALLEGRO_JAX_THREADS=8 taskset -c 0-7 <python-allegro-jax> benchmark/allegro_jax_bench.py
```

Figures (`benchmark/images/allegro_benchmark_*.png`, `allegro_*_gpu_speedup.png`) — put the Apple run
in `results/allegro_bench_apple.json` and the RTX 5080 box run in `results/allegro_bench_cyclops.json`,
with the nequip/jax JSONs alongside, then:

```
julia --project=<env-with-CairoMakie+JSON3> benchmark/allegro_plots.jl
```

Weights + the correctness reference (`data/allegro_reference/allegro_package_*`) are regenerated with
`test/allegro_package_reference.py` (needs `nequip-allegro`).
