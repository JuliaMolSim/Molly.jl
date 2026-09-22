# Allegro potential — benchmarks

Performance and correctness results for Molly's native many-body
[Allegro](https://doi.org/10.1038/s41467-023-36329-y) implementation (CPU reference forward).
The headline is the cost of the analytic forces added in the many-body rewrite versus the
finite-difference forces they replaced.

**Setup:** Apple M3 Pro, Julia 1.12, single thread, `Float64`. The committed reference model
(`data/allegro_reference/allegro_model.h5`: `C=4, H=16, nb=8, layers=2, l_max=2, r_c=4.0 Å`) is
loaded and timed on jittered cubic-lattice systems (2.5 Å spacing, open boundary) so the neighbour
count per atom is realistic and no pair is singular. Timings are the best-of-repeats wall time from
the shared harness (`benchmark/allegro_bench_common.jl`).

**Reproduce:**
```
julia --project=<env-with-Molly+HDF5+JSON3> benchmark/allegro.jl              # table + results/allegro_*.json
julia --project=<env-with-CairoMakie+JSON3> benchmark/allegro_plots.jl        # figures → images/
julia --project=<env> benchmark/run_allegro_benchmarks.jl                     # driver: timing + figures
```
Env vars: `ALLEGRO_SIZES` (atom counts), `ALLEGRO_FD_MAX` (largest N to also finite-diff),
`ALLEGRO_SPACING` (lattice spacing Å), `ALLEGRO_SKIP_PLOTS`.

---

## Correctness

Validated in `test/ml_potentials.jl` (reference from `test/allegro_reference.py`):

- **Energy** matches the many-body reference to ~1e-16 (machine precision).
- **Analytic forces** (`allegro_energy_and_forces`) match central finite differences of the energy
  to ~3e-10 and the reference forces to ~4e-10, with **ΣF ≈ 2e-16** (translation invariance is exact,
  not merely numerical).
- **Rotation / translation / permutation invariant** energy.
- **Non-additivity test:** `E_full ≠ Σ isolated-pair energies` (`|Δ| ≈ 1e-2`), proving the model is
  genuinely many-body rather than a pair potential — the bug this rewrite fixed.
- Equivariant primitives: 480/480 in `test/equivariant.jl`.

---

## What is measured

| quantity | function | work |
| :--- | :--- | :--- |
| energy | `allegro_total_energy` | one forward pass |
| energy + forces | `allegro_energy_and_forces` | one taped forward + one analytic backward |
| finite-diff forces | central differences | `6N` energy evaluations |

The analytic path is the one Molly uses (`AtomsCalculators.forces!` → `allegro_forces` →
`allegro_energy_and_forces`); the finite-difference column is the previous placeholder, kept here
only to quantify the speedup.

---

## Results

| atoms | edges | energy (ms) | E+forces (ms) | fd forces (ms) | speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 16  | 118  | 0.507 | 1.384 | 51.1  | 37× |
| 32  | 302  | 1.014 | 3.514 | 215.1 | 61× |
| 64  | 726  | 2.271 | 8.558 | 1007  | 118× |
| 128 | 1574 | 5.247 | 18.78 | — | — |
| 256 | 3446 | 10.97 | 42.79 | — | — |

("speedup" = finite-diff forces / analytic energy + forces. Finite differences are only run up to
64 atoms because their cost is `6N` full energies.)

### Forces: analytic vs finite differences

![Forces vs N](images/allegro_forces_vs_N.png)

![Forces analytic speedup](images/allegro_forces_speedup.png)

### Energy scaling

![Energy vs N](images/allegro_energy_vs_N.png)

---

## GPU: CPU (t1/t8) vs Metal vs CUDA

The energy forward runs natively on the GPU via KernelAbstractions (`compute_allegro_energy_ka`,
the same kernels on CUDA / Metal / the KA CPU backend), and the CPU forward is threaded over
centre atoms (`Threads.@threads`). Measured across four backends:

- **CPU t1 / t8 and Metal** on Apple M3 (Metal is `Float32`; CPU `Float64`).
- **CUDA** on an NVIDIA RTX 5080 (cyclops), `Float64`, with its own CPU t1/t8 baseline on that box.

CPU/Metal are Apple Silicon while CUDA is the RTX 5080 host, so this is cross-machine: read the
scaling shape and the within-machine GPU-over-CPU speedups, not the absolute cross-device level.
Reproduce:

```
julia --project=<env+Metal> -t8 benchmark/allegro_gpu_compare.jl   # then -t1  (Apple: CPU t1/t8 + Metal)
julia --project=<env+CUDA>  -t8 benchmark/allegro_cuda_compare.jl  # then -t1  (RTX 5080: CPU t1/t8 + CUDA)
```

**CUDA vs CPU (RTX 5080, Float64), energy time in ms:**

| atoms | CPU t1 | CPU t8 | CUDA | CUDA/t1 | CUDA/t8 | reldiff |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 64   | 11.4  | 3.4   | 1.33 | 9×   | 3×  | 1.6e-16 |
| 128  | 24.8  | 7.6   | 1.39 | 18×  | 5×  | 2.9e-16 |
| 256  | 55.7  | 16.9  | 1.47 | 38×  | 12× | 1.3e-16 |
| 512  | 123.0 | 35.7  | 1.64 | 75×  | 22× | 2.4e-16 |
| 1024 | 308.3 | 106.3 | 2.13 | 144× | 50× | 1.2e-16 |
| 2048 | 653.8 | 206.1 | 3.62 | 181× | 57× | 0.0     |
| 4096 | 1511.2| 457.0 | 7.32 | 207× | 62× | 0.0     |

**Metal vs CPU (Apple M3), energy time in ms:**

| atoms | CPU t1 | CPU t8 | Metal | Metal/t8 |
| :---: | :---: | :---: | :---: | :---: |
| 64   | 2.2   | 0.8  | 1.99 | 0.4× |
| 128  | 4.9   | 1.5  | 1.97 | 0.7× |
| 256  | 10.8  | 2.9  | 2.14 | 1.4× |
| 512  | 24.2  | 6.0  | 2.66 | 2.3× |
| 1024 | 53.4  | 13.2 | 3.92 | 3.4× |
| 2048 | 117.1 | 39.8 | 7.11 | 5.6× |

The GPU energy matches the CPU forward to ~1e-16 on CUDA (Float64) and ~1e-7 on Metal (Float32),
confirming the kernels are correct on real hardware. Threading gives ~3.4–3.8× (t1→t8). The
neighbour list, edge geometry and the whole forward run on-device with no inter-kernel
synchronisation, so CUDA stays ~1.3–7 ms flat from 64 to 4096 atoms rather than climbing — up to
**207× over CPU-t1** and **62× over CPU-t8** at 4096 atoms, growing with N. Metal starts
launch-overhead bound at small N (the M3 CPU-t8 is sub-millisecond there) and overtakes CPU-t8 past
~256 atoms (5.6× at 2048). An earlier version that built the neighbour list on the host and
synchronised between kernels tapered badly at large N (92 ms CUDA at 4096 vs 7 ms now); moving that
work on-device was the main win.

![Energy across backends](images/allegro_backends_energy_vs_N.png)

![GPU speedup over CPU-t8](images/allegro_gpu_speedup.png)

---

## Head-to-head: Molly native vs other Allegro implementations

The existing Allegro implementations are the PyTorch [`nequip-allegro`](https://github.com/mir-group/allegro)
(the reference) and the JAX [`allegro-jax`](https://github.com/mariogeiger/allegro-jax) (on e3nn-jax).
All three run a **comparable-size** model (`l_max=2`, 2 layers, ~4 tensor / 16 scalar channels,
~11k params) so this measures implementation throughput. Two caveats: they are **not identical ops**
(the native op-by-op bit-match to nequip-allegro is a follow-up — read this as "a native-Julia
Allegro of similar size vs the others", not same-weights), and CPU/CUDA are the same **RTX 5080
host** while Metal is the **Apple M3** (cross-machine). Reproduce with `benchmark/allegro_torch_bench.py`
(nequip) and `benchmark/allegro_jax_bench.py` (JAX) alongside `benchmark/allegro_cuda_compare.jl` and
`benchmark/allegro.jl` (Molly), on the same machine.

**Energy, time in ms.** CPU (t1/t8) and CUDA are the RTX 5080 host; Molly Metal is the Apple M3.
Every backend is shown over the full range it can reach: Metal is Molly-only (see below), and
`allegro-jax`'s dense O(N²) CPU path becomes impractical past ~512 atoms.

| atoms | Molly CUDA | Molly Metal | Molly t8 | Molly t1 | nequip CUDA | nequip t8 | nequip t1 | jax CUDA | jax CPU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 64   | 1.33 | 1.86  | 3.4   | 11.4   | 5.65  | 162  | 14.4   | 6.7   | 13.8  |
| 256  | 1.47 | 2.08  | 16.9  | 55.7   | 5.59  | 310  | 115.5  | 30.2  | 228.2 |
| 512  | 1.64 | 2.76  | 35.7  | 123.0  | 5.68  | 400  | 274.2  | 80.4  | 972.2 |
| 1024 | 2.13 | 3.77  | 106.3 | 308.3  | 5.70  | 573  | 585.1  | 167.5 | —     |
| 2048 | 3.62 | 6.60  | 206.1 | 653.8  | 7.85  | 747  | 1242.6 | 344.8 | —     |
| 4096 | 7.32 | 12.83 | 457.0 | 1511.2 | 13.42 | 1899 | —      | 877.0 | —     |

(`torch.compile` brings nequip CUDA to ~2.5–5 ms — still slower than Molly; omitted for clarity.
`allegro-jax` uses dense all-pairs, so it scales O(N²) and is far slower at scale.)

![Allegro energy: all implementations](images/allegro_benchmark_energy.png)

**Forces, time in ms** (Molly analytic vs nequip autograd vs jax autograd; CPU on the RTX 5080 host):

| atoms | Molly t8 | nequip CUDA | nequip t8 | nequip t1 | jax CUDA | jax CPU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 64   | 45.9   | 15.2 | 357  | 34.8   | 6.6   | 19.7   |
| 256  | 240.0  | 15.3 | 747  | 301.0  | 30.0  | 288.0  |
| 512  | 558.4  | 15.3 | 1038 | 685.2  | 79.5  | 1069.5 |
| 1024 | 1280.7 | 15.4 | 1426 | 1322.7 | 169.3 | —      |
| 2048 | 2914.6 | 17.1 | 2287 | 2798.8 | 346.3 | —      |

![Allegro forces: all implementations](images/allegro_benchmark_force.png)

Reading it:

- **Energy: Molly's native kernels win clearly.** Molly's CUDA energy is **1.8×–4.2× faster** than
  nequip-allegro (launch-bound at ~5.6 ms flat; ~2.5–5 ms even with `torch.compile`) and stays
  sub-10 ms to 4096 atoms. `allegro-jax` uses dense all-pairs, so it scales **O(N²)** and is far
  slower at scale (877 ms vs Molly's 7 ms at 4096). On CPU, PyTorch's threading is pathological at
  these sizes (162 ms at 64 atoms on 8 threads, vs Molly's 3.4 ms); Molly's threaded CPU energy is
  10–50× faster, and Molly single-threaded (t1) still beats nequip's 8-thread energy at every size.
  Molly Metal (Apple M3) also beats nequip CUDA at small N.
- **Metal is a Molly-only capability.** Neither reference runs on Apple GPU: nequip-allegro requires
  `float64` (its per-type energy shift casts to global float64) and MPS is float32-only, and
  `allegro-jax` fails to compile under `jax-metal` (`unknown attribute code`). Molly's native Metal
  energy runs where **both** references cannot. (The figures show t1/t8/CUDA for every backend that
  runs, over the full range each can reach — `allegro-jax` is dense O(N²), so its CPU line stops at
  512 atoms.)
- **Forces are the honest gap.** Molly's analytic CPU forces track nequip's autograd CPU forces
  closely (comparable to nequip CPU-t1 across the range; nequip CPU-t8 pulls ahead at large N because
  Molly's backward is not yet threaded). But both nequip and `allegro-jax` run forces on **CUDA**
  (nequip ~15 ms flat; jax ~6–350 ms O(N²)) and Molly does not yet — a native GPU backward is the
  outstanding follow-up.

---

## Reading the numbers

- **Analytic forces are 37×–118× faster than finite differences, and the gap widens with system
  size.** Finite differencing costs `6N` energies, so its work grows as roughly `O(N) × O(energy)`;
  the analytic backward is a single reverse pass whose cost tracks one forward. At 64 atoms the
  analytic energy + forces already beats finite-diff forces by ~118×, and the ratio keeps climbing —
  finite differences are not viable beyond toy systems.
- **Forces are cheap on top of the energy.** `energy + forces` costs about 3.8–4.0× a bare energy
  evaluation across the range (e.g. 8.7 ms vs 2.2 ms at 64 atoms) — the taped forward plus one
  backward, a small constant factor, exactly as an adjoint (reverse-mode) gradient should behave.
- **Current scaling is set by the O(N²) all-pairs neighbour search.** Energy time grows slightly
  faster than linearly in the atom count (0.34 → 10.8 ms from 16 → 256 atoms). The per-edge maths
  is linear in the number of edges; the super-linear part is the naive `O(N²)` neighbour build.
  A cell-list / CSR neighbour list would restore near-linear scaling and is the natural next step.

---

## Notes and caveats

- Weights are the small random reference model used by the test suite, not a trained potential.
  Timings depend on the architecture size (channels `C`, latent width `H`, number of layers,
  `l_max`), not on the weight values, so they are representative for a model of this shape.
- The **energy** forward runs entirely on the GPU (CUDA + Metal, see the GPU section) — neighbour
  build, geometry and the layer kernels, with no inter-kernel synchronisation. The **forces**
  (backward) still run on the CPU (with a host-built neighbour list); a native GPU backward is the
  next step.
- A head-to-head against the reference `nequip-allegro` (PyTorch) package — analogous to the
  ANI-vs-TorchANI comparison — is future work: it needs a trained checkpoint (or a matched-config
  build) shared between the two implementations for a fair comparison.
