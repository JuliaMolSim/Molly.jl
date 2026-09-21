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
| 16  | 118  | 0.343 | 1.313 | 38.06 | 29× |
| 32  | 302  | 0.867 | 3.419 | 196.4 | 57× |
| 64  | 726  | 2.214 | 8.694 | 944.4 | 109× |
| 128 | 1574 | 4.868 | 18.88 | — | — |
| 256 | 3446 | 10.82 | 42.93 | — | — |

("speedup" = finite-diff forces / analytic energy + forces. Finite differences are only run up to
64 atoms because their cost is `6N` full energies.)

### Forces: analytic vs finite differences

![Forces vs N](images/allegro_forces_vs_N.png)

![Forces analytic speedup](images/allegro_forces_speedup.png)

### Energy scaling

![Energy vs N](images/allegro_energy_vs_N.png)

---

## CUDA (RTX 5080)

The energy forward also runs natively on the GPU via KernelAbstractions (`compute_allegro_energy_ka`,
the same kernels on CUDA / Metal / the KA CPU backend). Measured on an NVIDIA RTX 5080 (cyclops,
Julia 1.12, `Float64`); both the CPU and CUDA series are on that same box, so this is a
within-machine GPU-vs-CPU comparison. Reproduce:

```
julia --project=<env-with-Molly+HDF5+CUDA+JSON3> benchmark/allegro_cuda_compare.jl
```

| atoms | edges | CPU (ms) | CUDA (ms) | speedup | reldiff vs CPU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64   | 726   | 11.34  | 2.01 | 5.6×  | 4.9e-16 |
| 128  | 1574  | 24.83  | 2.53 | 9.8×  | 8.8e-16 |
| 256  | 3446  | 55.78  | 3.04 | 18.4× | 1.1e-15 |
| 512  | 7444  | 123.5  | 3.86 | 32.0× | 5.2e-15 |
| 1024 | 15556 | 287.2  | 9.54 | 30.1× | 6.9e-16 |
| 2048 | 32344 | 661.6  | 26.6 | 24.9× | 1.1e-15 |
| 4096 | 66808 | 1487.2 | 92.3 | 16.1× | 3.0e-15 |

The CUDA energy matches the CPU forward to ~1e-15 (Float64 machine precision), confirming the GPU
kernels are correct on real hardware. Speedup peaks around **32× at ~512 atoms**. It tapers at
larger N because the neighbour list is still built on the host (O(N²)) and copied over each call;
a device-side neighbour build would remove that ceiling. The GPU energy also runs on Apple Metal
(validated to ~3.6e-7 in Float32).

![CUDA energy vs N](images/allegro_cuda_energy_vs_N.png)

![CUDA speedup](images/allegro_cuda_speedup.png)

---

## Reading the numbers

- **Analytic forces are 29×–109× faster than finite differences, and the gap widens with system
  size.** Finite differencing costs `6N` energies, so its work grows as roughly `O(N) × O(energy)`;
  the analytic backward is a single reverse pass whose cost tracks one forward. At 64 atoms the
  analytic energy + forces already beats finite-diff forces by ~110×, and the ratio keeps climbing —
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
- The **energy** forward now runs natively on the GPU (CUDA + Metal, see the CUDA section). The
  **forces** (backward) still run on the CPU; a native GPU backward is the next step. The neighbour
  list is built on the host for both, so a device-side neighbour build is the main remaining
  optimisation.
- A head-to-head against the reference `nequip-allegro` (PyTorch) package — analogous to the
  ANI-vs-TorchANI comparison — is future work: it needs a trained checkpoint (or a matched-config
  build) shared between the two implementations for a fair comparison.
