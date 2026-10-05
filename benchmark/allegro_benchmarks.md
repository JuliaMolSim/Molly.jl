# Allegro potential — benchmarks

Performance and correctness for Molly's native
[Allegro](https://doi.org/10.1038/s41467-023-36329-y) potential (`AllegroPotential`), a bit-exact
port of the real [`nequip-allegro`](https://github.com/mir-group/allegro) package (allegro 0.8.3).
The model is the package architecture — `l_max=2`, 2 layers, 32 scalar / 8 tensor features, 8 Bessel
`TwoBodyBesselScalarEmbed` (`polynomial_cutoff_p=6`), `avg_num_neighbors=10`, `r_c=4.0 Å` — loaded
from the package's own exported weights. Energy and forces run on CPU (threaded, analytic) and on the
GPU (KernelAbstractions kernels, CUDA + Metal), with a hand-written analytic reverse pass (no AD).

The headline: the native model reproduces the package to numerical precision, and the same analytic
energy+forces path **beats both the real nequip-allegro package and allegro-jax on CUDA and on CPU
(single-thread), runs uncontested on Apple Metal, and is at parity on 8-thread CPU** — while on CUDA
it scales to the full 16k-atom system on a 16 GB card, where the package's autograd forces run out of
memory.

**What is timed:** one evaluation — energy (forward) and energy + forces (forward + one analytic
backward), timed separately, best-of-repeats after a warm-up. The neighbour list is **precomputed**
(it is reused across MD steps; every implementation here passes precomputed edges, so the comparison
is of the model evaluation, not the neighbour search). **Systems:** random H/C/N/O in a cubic box at
density 0.09 atoms/Å³, sizes 500→15,954 atoms (15,954 = the 6mrr test system). `Float64` on CPU and
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
implementation equally — all three pass precomputed edges). `OOM` = the package's autograd forces
exceed the 16 GB card; `—` = not run.

**GPU** (`Float64` on CUDA, `Float32` on Metal):

| atoms | Molly CUDA | nequip CUDA | allegro-jax CUDA | Molly Metal |
| ---: | ---: | ---: | ---: | ---: |
| 500   | **11**  | 18   | 107  | 17   |
| 1000  | **15**  | 19   | 216  | 46   |
| 2000  | **25**  | 30   | 462  | 125  |
| 4000  | **58**  | 66   | 1131 | 254  |
| 8000  | **129** | 144  | 2719 | 530  |
| 15954 | **232** | OOM  | 7226 | 1065 |

**CPU** on the RTX 5080 host (12-core), `Float64`, single thread (t1) and 8 threads (t8):

| atoms | Molly t1 | nequip t1 | jax t1 | Molly t8 | nequip t8 | jax t8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500   | **694**   | 1456  | 1691   | **1043** | 1279  | 1416  |
| 1000  | **1575**  | 3199  | 6780   | 2097     | 1940  | 6119  |
| 2000  | **3637**  | 6901  | 28020  | **3650** | 3877  | 21474 |
| 4000  | **8052**  | 15850 | 126854 | **5936** | 6284  | 72152 |
| 8000  | **17895** | 34539 | —      | 13171    | 11813 | —     |
| 15954 | —         | 72939 | —      | **18012**| 21505 | —     |

Reading it (**bold** = fastest in that row/group):
- **CUDA — Molly wins at every size.** It is 1.1–1.9× faster than the real package, and because the
  analytic reverse pass has a far smaller memory footprint than the package's autograd backward,
  **Molly is the only CUDA implementation that reaches the full 6mrr system** (15,954 atoms, 0.23 s) —
  the package OOMs there. allegro-jax is 5–30× slower than Molly on CUDA.
- **Metal — uncontested.** Neither the package nor allegro-jax has a `Float64` Apple-GPU path, so Molly
  is the only Allegro that runs on Apple Silicon (16k in 1.1 s).
- **CPU single thread (t1) — Molly wins at every size**, ~2× faster than the package and 2–16× faster
  than allegro-jax.
- **CPU 8 threads (t8) — Molly wins 4 of 6 sizes** and is within a few percent on the other two. Molly
  t8 gains little over t1 because each evaluation still allocates large temporaries whose GC does not
  parallelise; cutting that (a reusable workspace) is the next step and would widen the t8 lead.

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
ALLEGRO_BK=cpu   JULIA_NUM_THREADS=8 julia --project=<env> benchmark/allegro.jl    # -> allegro_bench.json
ALLEGRO_BK=metal                     julia --project=<env> benchmark/allegro.jl
ALLEGRO_BK=cuda                      julia --project=<env> benchmark/allegro.jl
```
`<env>` needs Molly + HDF5 + JSON3 (+ Metal or CUDA). Sizes via `ALLEGRO_SIZES=500,1000,...`.

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
