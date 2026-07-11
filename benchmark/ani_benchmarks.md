# ANI-2x potential — benchmarks

Consolidated performance and correctness results for Molly's native ANI-2x implementation.

**Setup:** Apple Silicon (12 cores: 8P + 4E), Julia 1.12, ANI-2x, Float32 AEVs. Systems are
slices of `data/6mrr_equil.pdb` with a `DistanceNeighborFinder` (cutoff `max(r_c_R,r_c_A)+1 = 6.1 Å`),
single ensemble member (`ensemble_idx=0`) unless noted. "CPU" numbers use `-t8` unless a thread
count is given.

**Reproduce:**
```
julia --project=<env> -t <N> benchmark/ani.jl              # CPU energy/forces, thread sweep, same-species
julia --project=<env> -t 8  benchmark/ani_gpu_compare.jl   # CPU vs Apple Metal energy
julia --project=<env> -t 8  benchmark/ani_trajectory.jl    # 6mrr NVE trajectory (DCD)
python  test/torchani_reference.py --benchmark --device cpu   # TorchANI reference timing
```
Env vars: `ANI_SIZES`, `ANI_FORCES`, `ANI_ENSEMBLE` (`0`|`full`), `ANI_SAMESPECIES`;
`ANI_TRAJ_{N,STEPS,DT_FS,LOG,TEMP}`.

---

## Correctness

- **18/18** ANI tests pass (`test/ml_potentials.jl`) under `-t8` with Enzyme + KernelAbstractions.
- AEV vs TorchANI: ~1e-8 (N₂), 1.2e-7 (H₂O).
- Forces vs TorchANI: N₂ 4.3e-5 eV/Å, H₂O 2.4e-3 eV/Å (Float32 AEV limit).
- Threaded AEV is **bit-identical** to serial (Test 15). GPU kernels match the CPU path for
  Cubic and Triclinic boundaries (Test 18): 0.0 on the CPU backend, ≤1.5e-6 on Metal.
- 6mrr full protein (15,954 atoms): energy within <0.001% of the TorchANI reference (Test 13).

---

## Energy (CPU)

Single ensemble member, `potential_energy`.

### Full 6mrr (15,954 atoms)

| threads | 1       | 8         |
|---------|---------|-----------|
| time    | 1493 ms | **467 ms** |

The per-atom CSR consumption of the finder's neighbours made this **15.5×** faster than the
original all-pairs-rescan baseline (7219 ms serial). (An earlier pre-CSR thread sweep scaled
near-linearly to 8 threads: 1.9×/3.5×/5.3× at 2/4/8 threads.)

### Energy vs system size

| N atoms | 500  | 1000 | 2000 | 5000  | 8000  | 15,954 |
|---------|------|------|------|-------|-------|--------|
| CPU (t8)| 16.8 | 30.5 | 48.6 | 112.7 | 192.5 | 467    | ms

### Full 8-member ensemble

| system      | 1 thread | 8 threads |
|-------------|----------|-----------|
| 6mrr 1000   | 146 ms   | 71 ms     |
| 6mrr 15954  | 2103 ms  | 1051 ms   |

8 members cost only ~1.4× a single member — the AEV is computed once and only the per-element
NN runs 8× (species-batched).

---

## Forces (CPU, Enzyme reverse-mode AD)

- Single member, 1000 atoms: **2643 ms** (species-batched; 1.56× and 2.1× less memory than the
  per-atom baseline).
- Full 8-member ensemble, 1000 atoms — the per-member reverse passes run across threads:

  | threads | 1     | 8       |
  |---------|-------|---------|
  | time    | ~78 s | **6.4 s** |

  Correct at both counts (N₂ 5–7e-7, H₂O <5e-7 eV/Å; the ~1e-7 t1/t8 difference is Float32 BLAS
  non-associativity). Molly GPU forces are not implemented yet (energy only on Metal).

---

## GPU (Apple Metal)

### AEV kernels (AEV only, ms)

| N atoms | neighbour-list | write-reduced | all-pairs |
|---------|----------------|---------------|-----------|
| 1000    | 31.1           | **7.5**       | 61.3      |
| 2000    | 32.0           | 7.9           | 95.7      |
| 4000    | 33.0           | 10.1          | O(N³)     |
| 8000    | 35.4           | 20.5          | O(N³)     |

The neighbour list turns the O(N²)/O(N³) all-pairs cost into O(N·k)/O(N·k²). The write-reduced
kernel (one workgroup per atom, shared-row `@atomic` accumulation, single coalesced write) is a
further 1.8–4.4× on top. Metal supports atomic float-add on threadgroup memory.

### End-to-end energy: CPU vs Metal

Metal times the on-device path (`compute_ani_energy_ka`: GPU AEV + on-device NN):

| N atoms      | CPU (t8) | Metal   | Metal speedup |
|--------------|----------|---------|---------------|
| 500          | 16.8 ms  | 68.5 ms | 0.25×         |
| 1000         | 30.5 ms  | 76.8 ms | 0.40×         |
| 2000         | 48.6 ms  | 82.4 ms | 0.59×         |
| 5000         | 112.7 ms | 86.6 ms | **1.30×**     |
| 8000         | 198.9 ms | 91.0 ms | **2.19×**     |
| 12000        | 283.1 ms | 113.3 ms| **2.50×**     |
| **15,954 (full 6mrr)** | 400.3 ms | **158.1 ms** | **2.53×** |

Below ~4000 atoms Metal is dominated by launch/transfer/per-call param-move overhead (nearly
flat ~68–92 ms), so the threaded CPU wins. Above the ~4000-atom crossover Metal pulls ahead as
the CPU grows linearly: at the full 6mrr system (15,954 atoms) Metal is **2.5× the CPU**
(158 vs 400 ms). Caching device params and running forces on-GPU would widen this further.

---

## Periodic boundaries (minimum image)

The CPU path and all three GPU kernels compute displacements via `Molly.vector(ci, coords[j],
boundary)`, so they apply the minimum-image convention. Both `CubicBoundary` and
`TriclinicBoundary` work (the boundary is only ever touched via `vector`). Verified identical to
the CPU reference on a periodic imaged-pair system (Test 18).

---

## NVE trajectory stability

`benchmark/ani_trajectory.jl` — VelocityVerlet NVE, real element masses, DCD via `TrajectoryWriter`.

| slice | steps | dt     | ms/step | energy drift `|ΔE|/|E0|` |
|-------|-------|--------|---------|--------------------------|
| 60 atoms  | 50   | 0.5 fs | 56.5    | 6.2e-7 |
| 300 atoms | 3000 | 0.5 fs | 502     | **1.5e-7** |

Energy is conserved to ~1e-7 over 3000 steps — the fast path is stable for real dynamics.

---

## Same-species diagnostic

All-carbon vs the real mixed-species system at 1000 atoms: `mixed / all-one-species ≈ 1.0`.
Species branching is essentially free — no need to sort atoms by species.

---

## TorchANI comparison

`test/torchani_reference.py --benchmark --device {cpu,mps} --sizes 500,1000,2000,5000,8000` times
TorchANI energy+forces on the same slices (warmup + min-of-N with device sync), writing
`data/ani_reference/6mrr_timing_torchani_<device>.json` to join against the Molly numbers above.
Needs `pip install torchani==2.2.4 torch ase h5py`. CPU-to-CPU is the definitive comparison;
TorchANI GPU normally means CUDA — on a Mac it runs via PyTorch-MPS (best-effort; some ops may
fall back).

| N atoms | Molly CPU | Molly Metal | TorchANI CPU | TorchANI MPS |
|---------|-----------|-------------|--------------|--------------|
| 1000    | 30.5 ms   | 76.8 ms     | *(run script)* | *(run script)* |
| 8000    | 198.9 ms  | 91.0 ms     | *(run script)* | *(run script)* |
| 15,954  | 400.3 ms  | 158.1 ms    | *(run script)* | *(run script)* |

---

## Scripts

| script | what it measures |
|--------|------------------|
| `benchmark/ani.jl`             | CPU energy/forces, thread sweep, same-species diagnostic |
| `benchmark/ani_gpu_compare.jl` | Molly energy: CPU vs Apple Metal |
| `benchmark/ani_trajectory.jl`  | 6mrr NVE trajectory + energy drift, DCD output |
| `test/torchani_reference.py --benchmark` | TorchANI energy/forces timing |
