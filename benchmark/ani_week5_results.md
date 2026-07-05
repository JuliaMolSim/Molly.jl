# ANI-2x Week 5 performance results

Hardware: Apple Silicon (12 cores), Julia 1.12, Float32 AEVs, ANI-2x.
System: 6mrr protein slices (`data/6mrr_equil.pdb`), `DistanceNeighborFinder` (6.1 Å),
single ensemble member (`ensemble_idx=0`) unless noted.

Reproduce: `julia --project=<env> -t <N> benchmark/ani.jl`
(`ANI_SIZES`, `ANI_FORCES`, `ANI_ENSEMBLE`, `ANI_SAMESPECIES` env vars; see `benchmark/ani.jl`).

## What changed

1. **Threaded AEV** — the central-atom loop in `_compute_aevs_buf!` runs with
   `Threads.@threads :static` over contiguous atom chunks, each with its own
   `AEVScratch` (no shared mutable state). Output rows are disjoint per atom, so the
   threaded result is **bit-identical** to serial (Test 15: 0.0 deviation).
2. **NN batched by species** — `_ani_energy_single` gathers each element's AEV rows
   into one `(aev_len, n_s)` matrix and calls `Lux.apply` once per element instead of
   once per atom. Same batching mirrored in the Enzyme AD energy (`_ani_energy_for_ad`).

## Energy (potential_energy)

| system        | baseline (serial) | A+B, 1 thread | A+B, best  | speedup vs baseline |
|---------------|-------------------|---------------|------------|---------------------|
| 6mrr 1000     | 149.7 ms          | 130.7 ms      | 37.8 ms (t8)  | **3.96×**        |
| 6mrr 15954    | 7219 ms           | 6974 ms       | 1158 ms (t12) | **6.2×**         |

### Thread scaling — full 6mrr (15,954 atoms), energy

| threads | 1     | 2     | 4     | 8     | 12    |
|---------|-------|-------|-------|-------|-------|
| time    | 6974  | 3617  | 1986  | 1326  | 1158  |
| speedup | 1.0×  | 1.93× | 3.51× | 5.26× | 6.02× |

Near-linear to 8 threads; diminishing past 8 (memory-bandwidth bound + efficiency
cores on this CPU).

## Forces (Enzyme reverse-mode AD, 1000 atoms, 1 thread)

| metric  | per-atom (baseline) | batched-by-species | improvement |
|---------|---------------------|--------------------|-------------|
| time    | 4130 ms             | 2643 ms            | **1.56×**   |
| allocs  | 2.35 GB             | 1.11 GB            | **2.1×**    |

(The AD energy uses the allocating serial `compute_aevs`, so forces don't yet benefit
from AEV threading — see "Next".)

## Same-species diagnostic (Joe's question)

Running 1000 atoms all set to carbon vs the real mixed-species system:
`mixed / all-one-species ≈ 0.9–1.05` across runs ⇒ **species branching is essentially
free; no need to sort atoms by species before the kernel.**

## Correctness

All **15/15** ANI tests pass (Tests 1–14 + new Test 15) under `-t8` with Enzyme +
KernelAbstractions loaded. AEV deviations vs TorchANI ~1e-8; Enzyme N₂ force dev
4.3e-5 eV/Å, H₂O 2.4e-3 eV/Å (unchanged by batching).

## GPU: neighbour-list AEV kernel on Metal

New `_aev_kernel_nl!` scans only each atom's neighbours (CSR list built within
`max(r_c_R, r_c_A)`), turning the all-pairs O(N²)/O(N³) cost into O(N·k)/O(N·k²).
Selected via `compute_aevs_ka(...; neighbors=:auto)` (or pass a prebuilt `(off, idx)`).
Validated bit-identical to the all-pairs kernel on the CPU backend (Test 16) and
matching to 3e-6 on Metal (Float32 transcendentals).

AEV-only kernel time (ms), 6mrr slices on Apple Metal:

| N atoms | avg neigh | Metal NL | Metal all-pairs | CPU-KA NL (1 thread) |
|---------|-----------|----------|-----------------|----------------------|
| 1000    | 40.5      | 31.1     | 61.3            | 68.9                 |
| 2000    | 27.7      | 32.0     | 95.7            | 88.7                 |
| 4000    | 21.2      | 33.0     | (O(N³))         | 113.0                |
| 8000    | 25.2      | 35.4     | (O(N³))         | 224.2                |

Metal NL stays ~flat 31→35 ms from 1k→8k atoms (kernel launch + transfer dominate;
GPU is under-occupied at these sizes), already 6.3× the single-thread CPU-KA kernel
at 8000 atoms and ≥2× the all-pairs kernel even at 1000. The neighbour list is what
makes ≥2000 atoms tractable on the GPU at all.

`workgroup` is now a tunable kwarg (default 256).

## Next (not in this round)

- Forces still run AEV serially inside the Enzyme pass; thread the AD AEV or the
  ensemble loop for a forces speedup at MD scale.
- GPU global-write reduction: each thread still does a global read-modify-write per
  AEV term. A shared-memory tile of the AEV row (or neighbour coords) would cut global
  traffic further — left as future work since the NL kernel already scales flat.
- Run the NN forward pass on-GPU (currently CPU/Lux) for an end-to-end GPU energy.
- Build the GPU neighbour list with a cell list (current host build is O(N²)).

---

# ANI-2x Week 6 performance results

Focus (mentor steer): **consume the neighbours passed into `potential_energy`/`forces`**
(`CellListMapNeighborFinder` on CPU, `DistanceNeighborFinder` on GPU) — efficiently —
rather than building a cell list. Same hardware/setup as Week 5.

## A — per-atom CSR from the passed-in NeighborList (CPU)

The previous neighbour path rescanned the whole `NeighborList` for every atom
(O(N × total_pairs)). Now a per-atom CSR adjacency is built once per call (counting sort,
half-pairs symmetrised) and each atom reads only its slice — energies are bit-identical to
all-pairs.

| 6mrr energy, 15,954 atoms | Week 5 (rescan) | Week 6 (CSR) | speedup |
|---------------------------|-----------------|--------------|---------|
| 1 thread                  | 7219 ms         | **1493 ms**  | 4.8×    |
| 8 threads                 | 1326 ms         | **467 ms**   | 2.8×    |

467 ms is **15.5×** the original Week-5 serial baseline (7219 ms). Forces inherit the same
CSR win (they go through `compute_aevs`).

## B — ensemble-parallel Enzyme forces

The 8 per-ensemble-member reverse passes are independent → run across threads (member 1
warmed serially to compile once). Full 8-member forces, 1000 atoms:

| forces (1000 atoms, 8-member ensemble) | 1 thread | 8 threads |
|----------------------------------------|----------|-----------|
| time                                   | ~78 s    | **6.4 s** |

Correct at both thread counts: N₂ 5–7e-7 eV/Å, H₂O <5e-7 vs TorchANI (the ~1e-7 t1/t8
difference is Float32 BLAS non-associativity, not a race).

## Full 8-member ensemble energy (all optimisations)

| system      | 1 thread | 8 threads |
|-------------|----------|-----------|
| 6mrr 1000   | 146 ms   | 71 ms     |
| 6mrr 15954  | 2103 ms  | **1051 ms** |

8 members cost only ~1.4× a single member (2103 vs 1493 ms at t1) — the AEV is computed
once and only the per-element NN runs 8×, so species-batching + shared AEV pay off.

## C — GPU kernel consumes the finder's NeighborList

`compute_aevs_ka(...; neighbors=<NeighborList>)` now builds the CSR from the finder's list
(the `_build_neighbor_csr` O(N²) coords scan is demoted to a `:auto` benchmarking fallback).
Validated: fed a real `DistanceNeighborFinder` list, the kernel matches scalar all-pairs to
9.5e-7 (CPU backend) and 3.1e-6 (Metal).

## E — end-to-end on-device ANI energy (`compute_ani_energy_ka`)

New: after the GPU AEV, the per-element Lux networks run on the **same** device (params
moved via `Lux.gpu_device()`), so a full energy evaluation keeps the AEVs on-GPU. Metal and
KA-CPU agree to 4e-5 eV; both match the CPU `potential_energy` reference to 0.12 eV over 500
atoms (2.4e-4 eV/atom — the Float32-AEV vs Float64-AEV precision gap, not a bug).

## D — GPU global-write reduction (`write_reduce=true`)

`_aev_kernel_wg!`: **one workgroup per atom**. The W threads split the atom's neighbours
(radial) and neighbour pairs (angular) and accumulate into a shared threadgroup row
(`@localmem`, aev_len×4 B = 4 KB, fits Metal's 32 KB) via `@atomic` float-adds; the row is
written to global **once** (coalesced) instead of a global read-modify-write per term. Metal
supports atomic float-add on threadgroup memory (verified). Correct to 3.1e-6 vs scalar.

This also **parallelises each atom's work across the workgroup** — the plain NL kernel used
one thread per atom (serial per atom). Measured on Metal (`workgroup=64`):

| N atoms | NL kernel | write-reduced | speedup |
|---------|-----------|---------------|---------|
| 1000    | 33.0 ms   | **7.5 ms**    | 4.4×    |
| 2000    | 33.9 ms   | **7.9 ms**    | 4.3×    |
| 4000    | 34.5 ms   | **10.1 ms**   | 3.4×    |
| 8000    | 37.4 ms   | **20.5 ms**   | 1.8×    |

(Correcting the Week-6 mid-week note that guessed the kernel was launch-bound: it was
per-atom serialization-bound, so the workgroup rewrite helps substantially.)

## Tests

All **17/17** ANI tests pass (added Test 16 finder-`NeighborList` → KA kernel, Test 17
end-to-end `compute_ani_energy_ka` on the CPU backend). Run with
`julia --project -t8 -e 'using Molly, Lux, HDF5, Enzyme, KernelAbstractions; include("test/ml_potentials.jl")'`.
