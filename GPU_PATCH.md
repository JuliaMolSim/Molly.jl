# GPU Non-Bonded Performance: Status, Root Cause, and Forward Plan

Date: March 16, 2026

## 1. Project decision

We will stop optimizing for "OpenMM pipeline parity" as a primary goal.

The primary goals are:

1. Correctness.
2. End-to-end performance in Molly workloads.
3. No regression versus the best known Molly implementation.

The strict performance boundary is now the best of:

- `66ccb5eaec96cb032c14f71a34ece2d8c39200fd`
- `6b3cff47` (`Recover dense GPU nonbonded paths and profiling`)

At the moment, `6b3cff47` is the best known practical boundary for the reference protein benchmark on this machine.

## 2. Reference benchmark and reproducibility

Primary reference benchmark:

- `/lmb/home/alexandrebg/Documents/GPU/main.jl`
- run exactly with:
  - `julia +1.11 -i /lmb/home/alexandrebg/Documents/GPU/main.jl`

Secondary in-repo benchmark:

- `benchmark/protein_2.jl`

The reference script uses:

- `6mrr_equil.pdb`
- `array_type=CuArray`
- `nonbonded_method=:none` (but GPU pairwise nonbonded kernels are still active)
- `@btime simulate!(sys, sim, 1_000)`

## 3. Confirmed measurements

### 3.1 End-to-end benchmark (`main.jl`, same command)

| Commit | Mode | Measured time |
|---|---|---|
| `66ccb5ea` | default (`auto`) | `~354.9 ms` |
| `6b3cff47` | default (`auto`) | `~351.3 ms` |
| `b8725569` | default (`auto`) | `~464.8 ms` |
| `6b3cff47` | `MOLLY_CUDA_FORCE_PATH=dense` | `~338.5 ms` |
| `b8725569` | `MOLLY_CUDA_FORCE_PATH=dense` | `~467.7 ms` |

Conclusion:

- The regression from `66ccb5ea`/`6b3cff47` to `b8725569` is real and large (`~110-130 ms` on this benchmark).
- It persists even when force path is forced to dense, so this is not only a path-selection issue.

### 3.2 Dense force stage timing breakdown (instrumented)

Measured on the same protein system (`N=15954`, `n_blocks=499`, `n_steps_reorder=25`):

`6b3cff47` refresh step:

- `morton_ms ~ 0.131`
- `compress_ms ~ 1.048`
- `bounds_ms ~ 0.028`
- `kernel_ms ~ 0.226`
- `total_refresh_ms ~ 1.432`

`b8725569` refresh step:

- `morton_ms ~ 0.120`
- `compress_ms ~ 3.919`
- `bounds_ms ~ 0.053`
- `kernel_ms ~ 0.223`
- `total_refresh_ms ~ 4.315`

Steady (non-refresh) step is almost unchanged (`~0.244-0.247 ms`).

Interpretation:

- The regression is dominated by refresh-time preprocessing.
- The main delta is `compress_boolean_matrices!` (`~1.05 -> ~3.92 ms`).
- With reorder every 25 steps, 1000 steps trigger ~40 refreshes, so:
  - `~(3.92 - 1.05) * 40 ~= 115 ms`
- This matches the observed end-to-end slowdown.

## 4. Root cause of the current regression

The slowdown is primarily from the triangular compressed-mask redesign, not from dense pairwise force math.

Key changes associated with the regression:

1. Mask storage changed from rectangular to upper-triangular:
   - `src/force.jl`
   - `compressed_*` moved from `[32, n_blocks, n_blocks]` to `[32, n_upper_tiles]`.
2. Compression kernel launch changed from 2D grid to linear upper-triangle tile count:
   - `@cuda blocks=upper_tile_count(n_blocks) ... compress_boolean_matrices!`
3. Compression kernel now decodes `tile_idx -> (i,j)` inside kernel:
   - `upper_tile_ij(...)` with search logic and warp broadcast.
4. Force/energy kernels read masks through `upper_tile_index(i,j,n_blocks)`.

What did not regress meaningfully:

- Dense force arithmetic kernel body (`kernel_ms`) is approximately unchanged.
- Morton sort and bounds are not the dominant source.

## 5. What works

1. `66ccb5ea` and `6b3cff47` provide acceptable/best known performance on the reference protein workload.
2. Recovered dense force path is useful and should remain available.
3. Split policy (`force=dense`, `energy=tile`) is a practical default for dense protein-like cases.
4. Correctness validation strategy (GPU tests + CPU force/energy comparisons) remains valid.

## 6. What does not work (based on tested changes)

1. Forcing the whole project toward OpenMM-like tile preprocessing does not automatically improve Molly performance.
2. Current triangular mask compression path (`b8725569`) causes unacceptable refresh-time overhead.
3. Earlier speculative optimizations (on-demand compression in tile-finder, naive multi-warp compression, quick warp-vote transplants in energy) did not produce robust gains.

## 7. Strict regression boundary policy

From now on, every optimization patch must satisfy all of the following before it is kept.

### 7.1 Performance gates

Primary gate:

- Reference benchmark (`main.jl`) must be <= best baseline median (`66ccb5ea` / `6b3cff47`) within noise.
- Practical acceptance threshold: no worse than `+2%` median over at least 5 runs.

Secondary gate:

- Forced dense run (`MOLLY_CUDA_FORCE_PATH=dense`) must also be <= best dense baseline within `+2%`.

Stage gate:

- Refresh-time `compress_ms` on the protein case must not regress versus baseline.

### 7.2 Correctness gates

1. `test/gpu_optimizations.jl` must pass.
2. CPU/GPU agreement tests for force and potential energy must pass.
3. No overflow or invalid kernel execution in tile metadata paths.

### 7.3 Process gates

1. Changes are applied in small, isolated steps.
2. Benchmark after every step.
3. Revert immediately if a step violates any gate.

## 8. Plan to improve performance from the strict baseline

The plan starts from the best-performing implementation, not from the current slow state.

## Phase A: Lock the baseline

1. Use `6b3cff47` as the active optimization base for new work.
2. Keep `66ccb5ea` as historical cross-check.
3. Freeze benchmark protocol and environment for all comparisons.

Deliverable:

- A baseline report committed with exact commands and medians.

## Phase B: Redesign preprocessing (highest expected gain)

Target areas:

1. Morton sorting/reorder pipeline.
2. Boolean mask compression and mask layout.

### B1. Mask layout A/B test (first granular step)

Implement and compare two concrete designs:

1. Rectangular layout (revert style): `[32, n_blocks, n_blocks]`.
2. Triangular layout with cheap index mapping (no per-tile search in compression kernel).

Rules:

- Implement only one design change at a time.
- Measure immediately against baseline.

Decision:

- Keep only the design that wins on the reference protein benchmark and does not hurt sparse behavior materially.

### B2. Remove expensive `tile_idx -> (i,j)` decode from hot compression path

If triangular layout is retained:

1. Precompute `tile_i`, `tile_j` mapping buffers once.
2. Compression kernel reads mapping directly.
3. Avoid binary-search-like decode logic in-kernel.

### B3. Cut mask memory traffic

Evaluate:

1. Packing eligible/special together for fewer global transactions.
2. Read/write ordering to maximize coalescing in compression and consumption kernels.
3. Shared-memory staging only where it reduces global reads in measured terms.

## Phase C: Reuse preprocessing across force and energy when safe

1. Add explicit validity flags for Morton order, reordered buffers, bounds, and compressed masks.
2. Reuse preprocessing artifacts when coordinates are unchanged between force/energy requests.
3. Invalidate only when required (coordinate update or neighbor state change).

Goal:

- Eliminate duplicate preprocessing work without correctness risk.

## Phase D: Re-evaluate tile path scope

Given current evidence:

- Dense and near-dense systems dominate our reference workload.

Policy:

1. Keep tile path only if it shows clear value on sparse systems.
2. If sparse advantage is small or unstable, reduce tile-path complexity and maintenance burden.
3. Dense path performance must never be traded away for rare sparse wins.

## 9. Immediate next steps (execution order)

1. Re-baseline from `6b3cff47` and `66ccb5ea` with 5-run medians using `main.jl`.
2. Implement preprocessing redesign step B1 with a single minimal patch.
3. Run correctness + benchmarks.
4. Keep or revert based on strict gates.
5. Continue with B2 only if B1 direction is clearly positive.

## 10. Final status statement

Current status is clear:

1. We have an unacceptable regression from `66ccb5ea`/`6b3cff47` to current triangular-mask state.
2. The dominant source is mask compression preprocessing, not dense force arithmetic.
3. The project should now optimize from the fastest known Molly path with strict regression enforcement.
