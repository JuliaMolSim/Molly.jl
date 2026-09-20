# Allegro potential — performance

Timings for the native-Julia many-body [Allegro](https://doi.org/10.1038/s41467-023-36329-y)
potential (CPU reference forward) in Molly. The headline result is the cost of the analytic forces
added in the many-body rewrite versus the finite-difference forces they replaced.

Reproduce with:

```
julia --project=<Molly-dev-env-with-HDF5> benchmark/allegro_benchmark.jl
```

The script loads the committed reference model
(`data/allegro_reference/allegro_model.h5`) and times three quantities on jittered cubic-lattice
systems (2.5 Å spacing, open boundary) of increasing size.

## What is measured

| quantity | function | work |
| :--- | :--- | :--- |
| energy | `allegro_total_energy` | one forward pass |
| energy + forces | `allegro_energy_and_forces` | one taped forward + one analytic backward |
| finite-diff forces | central differences | `6N` energy evaluations |

The analytic path is the one Molly uses (`AtomsCalculators.forces!` → `allegro_forces` →
`allegro_energy_and_forces`); the finite-difference column is the previous placeholder, kept here
only to quantify the speedup.

## Results

Model: `C=4, H=16, nb=8, layers=2, l_max=2, r_c=4.0 Å`.
Machine: Apple M3 Pro, Julia 1.12, single thread, `Float64`. Median of 5 runs (3 for finite diff).

|  atoms |    edges |  energy (ms) |  E+forces (ms) |   fd forces (ms) |    speedup |
| :----: |   :----: |       :----: |         :----: |           :----: |     :----: |
|     16 |      118 |        0.420 |          1.427 |            39.57 |        28× |
|     32 |      302 |        0.943 |          3.727 |           199.33 |        53× |
|     64 |      726 |        2.274 |          8.819 |          1081.93 |       123× |
|    128 |     1574 |        5.463 |         20.697 |                — |          — |
|    256 |     3446 |       11.662 |         56.985 |                — |          — |

("speedup" = finite-diff forces / analytic energy + forces. Finite differences are only run up to
64 atoms because their cost is `6N` full energies.)

## Reading the numbers

- **Analytic forces are 28×–123× faster than finite differences, and the gap widens with system
  size.** Finite differencing costs `6N` energies, so its total work grows as roughly `O(N) ×
  O(cost of one energy)`; the analytic backward is a single reverse pass whose cost tracks one
  forward. At 64 atoms the analytic energy + forces already beats finite-diff forces by ~120×, and
  the ratio keeps climbing — finite differences are simply not viable beyond toy systems.
- **Forces are cheap on top of the energy.** `energy + forces` costs about 3.4–4.9× a bare energy
  evaluation across the range (e.g. 8.8 ms vs 2.3 ms at 64 atoms) — the taped forward plus one
  backward, a small constant factor, exactly as an adjoint (reverse-mode) gradient should behave.
- **Current scaling is set by the O(N²) all-pairs neighbour search.** Energy time grows slightly
  faster than linearly in the atom count (0.42 → 11.7 ms from 16 → 256 atoms). The per-edge maths
  is linear in the number of edges; the super-linear part is the naive `O(N²)` neighbour build.
  A cell-list / CSR neighbour list would restore near-linear scaling and is the natural next
  optimisation.

## Notes and caveats

- Weights here are the small random reference model used by the test suite, not a trained
  potential. Timings depend on the architecture size (channels `C`, latent width `H`, number of
  layers, `l_max`), not on the weight values, so they are representative for a model of this shape.
- Everything runs on the CPU. GPU-backed systems currently take a host round-trip; native
  on-device (KernelAbstractions / CUDA) kernels for the many-body forward and backward are a
  planned follow-up and would change these numbers substantially.
- The energy and analytic forces are validated for correctness in `test/ml_potentials.jl`
  (energy vs a many-body reference, analytic forces vs finite differences and vs the reference,
  ΣF ≈ 0, rotation invariance, and a non-additivity test proving the model is genuinely
  many-body rather than a pair potential).
