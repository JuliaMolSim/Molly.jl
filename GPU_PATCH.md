# GPU Non-Bonded Performance: Status, Root Cause, and Forward Plan

Date: March 17, 2026

## 1. Project decision (Updated)

The project has successfully reversed the performance regression and achieved a new best-in-class baseline for Molly's GPU non-bonded pipeline. We have surpassed the performance of the previously best-known commit `6b3cff47`.

**Current Status: Optimized & Verified.**

## 2. Reference benchmark and reproducibility

Primary reference benchmark:
- `benchmark/protein_2.jl` (6mrr system, N=15954)
- Run with: `CUDA_VISIBLE_DEVICES=1 julia +1.11 benchmark/protein_2.jl`

## 3. Confirmed measurements (Final)

| Implementation | Mode | Measured time (Median) |
|---|---|---|
| `6b3cff47` (Baseline) | default (`auto`) | `~351.3 ms` |
| **Current Optimized State** | default (`auto`) | **`~334.2 ms`** |

**Conclusion:** The pipeline is now ~5% faster than the previous best baseline, representing a total recovery from the regression and a net gain in performance.

## 4. Technical Implementation: How it works now

The current implementation achieves high performance through three primary architectural improvements:

### 4.1 2D Grid Launch for Matrix Compression
The `compress_boolean_matrices!` kernel was identified as the primary bottleneck in the regression (Phase B1/B2). 
- **Old (Regressed) Logic**: A 1D linear launch over the total number of triangular tiles. This required each warp to execute an expensive binary search (`upper_tile_ij`) to decode the tile index into `(i, j)` coordinates.
- **New Logic**: A 2D grid launch `(n_blocks, n_blocks)`. Each block in the grid naturally maps to `(i, j)` via `blockIdx().x` and `blockIdx().y`. This eliminates all indexing math and search logic from the hot path. Warps simply return immediately if `i > j`, which is extremely cheap on modern hardware.

### 4.2 "Clean Tile" Optimization
We introduced a "Clean Tile" flag to prune the search space and execution complexity:
1.  **Detection**: During compression, if a 32x32 tile has all bits set in the eligibility mask (no exclusions) and zero bits in the special mask, it is marked as `CLEAN` in a new `tile_is_clean::CuArray{Bool}` buffer.
2.  **Pruning**: The `find_interacting_blocks_kernel!` reads this flag and assigns a tile type (`0` for CLEAN, `1` for EXCLUDED).
3.  **Fast Path Execution**: In the `force_kernel!` and `energy_kernel!`, CLEAN tiles execute a specialized inner loop that **completely skips bitmask lookups and bit-shifting logic**. This significantly increases warp throughput for the majority of tiles in the system.

### 4.3 Preprocessing Caching (Phase C)
Duplicate work between force and energy calculations has been eliminated:
- **State Tracking**: `BuffersGPU` now tracks `step_n_preprocessed` and `last_r_cut`.
- **Cached Reuse**: If a force calculation is followed by an energy calculation (or vice versa) at the same simulation step and with the same interaction cutoff, the entire preprocessing pipeline is skipped.
- **Skipped Stages**: Morton sorting, bitmask compression, bounding box min/max calculation, and the interaction tile search are all bypassed, reusing the results from the previous call.

## 5. Summary of Key Fixes

- **Scalar Indexing**: Resolved fatal "Scalar indexing is disallowed" errors by refactoring `BuffersGPU` into a `mutable struct` with plain fields and using `only(from_device(...))` for GPU-to-host counter transfers.
- **Warp IR Errors**: Corrected `CUDA.all_sync` to `CUDA.vote_all_sync` for warp-wide boolean reductions during tile cleaning detection.
- **Unit Safety**: Fixed `DimensionError` issues when storing interaction cutoffs by ensuring `ustrip` is applied consistently to state-tracking fields.

## 7. Architectural Comparison: Tile vs. Dense Paths

Molly.jl now supports two distinct execution paths for GPU pairwise interactions, selectable via the `MOLLY_CUDA_FORCE_PATH` environment variable (`auto`, `tile`, or `dense`).

### 7.1 The Dense Path: Optimized Brute-Force
The **dense path** is designed for maximum throughput in high-density systems or small-to-medium simulations where the overhead of list management exceeds the cost of computation.

*   **Mechanism**: It launches a 2D computational grid covering the entire upper-triangular matrix of 32x32 atom blocks ($O(N_{blocks}^2)$).
*   **On-the-fly Pruning**: Each warp calculates the bounding box distance between its assigned blocks using the `boxes_dist` function. If the boxes are outside the cutoff, the warp exits immediately.
*   **Zero Indirection**: There is no "neighbor list" for tiles. Warps determine their target atoms directly from their `blockIdx`. This results in perfectly predictable, contiguous memory access patterns and avoids the "load-load" dependency of reading from an intermediate "list of interacting tiles."
*   **Performance**: For small systems (e.g., < 2,000 atoms), the dense path is typically faster because it avoids kernel launch latency and global atomic contention inherent in building a sparse list.

### 7.2 The Tile Path: Sparse Scaling
The **tile path** is Molly's primary engine for large-scale molecular dynamics. It sacrifices some per-tile throughput to achieve superior algorithmic scaling.

*   **Sparse Adjacency List**: Unlike the dense path, the tile path explicitly builds a list of interacting tiles. This is done in the `find_interacting_blocks_kernel!`, which uses global atomics to construct a CSR-like (Compressed Sparse Row) structure of tiles in the `interacting_tiles_j` buffer.
*   **Memory Indirection**: The force kernel must perform an indirect read (`j = interacting_tiles_j[idx, i]`) to identify its workload. While this allows the kernel to skip non-interacting regions of the matrix entirely, it introduces a slight overhead in memory fetch latency compared to the dense path.
*   **Amortized Complexity ($O(N)$ vs $O(N^2)$)**: While the dense path checks $O(N^2)$ block pairs **every step**, the tile path **caches** its sparse list. The $O(N^2)$ bounding-box search is only performed periodically (every `n_steps_reorder`). On all other steps, the force kernel only processes the $O(N)$ tiles known to be interacting, making it vastly superior for large, sparse systems.

### 7.3 Selection Logic
By default (`auto`), Molly selects the path based on system density:
- **Dense Path**: Used when the system is small or highly dense, where the overhead of building and reading a sparse list exceeds the cost of brute-force bounding box checks.
- **Tile Path**: Used for large-scale MD where pruning the $O(N^2)$ interaction matrix is critical for performance.

## 8. Forward Policy

The strict regression boundary policy (Section 7) remains in effect. Any future changes to the GPU pipeline must be measured against the new **334.2 ms** median benchmark. The "Clean Tile" and "2D Grid" patterns are now the established standard for Molly's tiled GPU kernels.
