# IMPLEMENTATION OF BETTER PERFORMANCE GPU KERNELS

## 0. What the latest measurements changed

  ### Cross-commit benchmark finding:

  - On the benchmarked 4096-atom Float64 `GPUNeighborFinder` case, the force-path
    timings were:
      - `66ccb5ea`: `1.032 ms`
      - `ab051454`: `1.403 ms`
      - `0abbaee5`: `1.402 ms`
      - `c35bef05`: `1.409 ms`
      - current `10a6d080`: `1.407 ms`

  ### Interpretation:

  - The major regression starts at `ab051454`, where the OpenMM-style
    interacting-tile-list redesign was introduced.
  - The later physical reordering work in `c35bef05` is not the primary source of
    the slowdown on the measured path.
  - The later removal of the host sync on `num_interacting_tiles` is still a good
    change, but it is not the fix for the regression versus `66ccb5`.
  - The recent profiling work confirms that the regression is structural in the
    CUDA force path, not a small launch-parameter issue.

  ### Kernel-level reason to plan around:

  - `66ccb5` still did tile pruning inside the main force kernel launch and let
    `BLOCK_Y` warps share the same `i` tile, reduce in shared memory, and emit
    fewer global atomics.
  - `ab051454` introduced a tile-list prepass and lost that grouping. The current
    path has more preprocessing and more direct global atomic accumulation.
  - Therefore, the first performance task must be to recover `66ccb5`-class dense
    performance, not to keep refining the tile-list path in isolation.

## 0.5. Current verified project state

  ### Done:

  - A reusable GPU profiling harness now exists in `benchmark/gpu_profile_utils.jl`.
  - `benchmark/benchmark_gpu_tiles.jl` now reports:
      - dense synthetic case
      - sparse synthetic case
      - stage timings for Morton sort, mask compression, reorder, bounds, tile
        finding, force kernel, reverse reorder, and energy kernel
      - tile counters: total interacting tiles, clean tiles, masked tiles,
        overflow, interacting fraction
  - `benchmark/new_prot_bench.jl` now reports the same stage/tile counters for
    the protein benchmark.
  - `test/gpu_optimizations.jl` now exercises the profiling harness and also
    checks a recovered dense-force-path override against CPU forces and energy.

  ### First dense-path recovery step done:

  - A recovered `66ccb5`-style fused dense force path now exists in
    `ext/MollyCUDAExt.jl`.
  - It is currently behind an explicit override:
      - `MOLLY_CUDA_FORCE_PATH=dense`
  - The current tile-list force path remains the default.
  - This was deliberate: the first goal was to verify correctness and direction
    of improvement without yet changing runtime dispatch policy for all systems.

  ### Verified measurements from the current tree:

  - On the synthetic 1024-atom Float64 benchmark with the current default
    selection logic:
      - dense case selects `force=dense`, `energy=tile`
          - forces: `0.537 ms`
          - potential energy: `0.463 ms`
      - sparse case selects `force=tile`, `energy=tile`
          - forces: `0.390 ms`
          - potential energy: `0.342 ms`
  - On the reduced protein benchmark (`MOLLY_CUDA_BENCH_MIN_STEPS=0`,
    one-sample sanity run), the current default selection logic chooses
    `force=dense`, `energy=tile` for all three launch policies:
      - legacy explicit: `forces=1.670 ms`, `pe=1.655 ms`, `total=3.325 ms`
      - tuned explicit: `forces=1.709 ms`, `pe=1.641 ms`, `total=3.350 ms`
      - auto: `forces=1.677 ms`, `pe=1.685 ms`, `total=3.361 ms`
  - With an explicit `MOLLY_CUDA_FORCE_PATH=dense` override on the same reduced
    protein benchmark, the dense energy path is correct but slower:
      - legacy explicit: `pe=1.730 ms`
      - tuned explicit: `pe=1.747 ms`
      - auto: `pe=1.740 ms`
  - Interpretation:
      - the recovered fused force path is clearly useful on dense workloads
      - the dense energy kernel is now implemented and correct
      - the reduced protein case still prefers the tile-list energy path, so
        energy should not auto-switch to dense yet

  ### Important limitation of the current state:

  - Automatic dispatch is now implemented for the force path only.
  - A recovered dense energy override now exists, but the default `auto`
    selection deliberately keeps energy on the tile path.
  - The profiling output is now aware of this and avoids reporting misleading
    tile-path timings when a dense override/selection is active.
  - The benchmark harness now warms up both force and energy profiling passes so
    the reported stage timings are steady-state rather than first-use numbers.

## 1. Baseline and measurement harness

  ### Patch order:

  - benchmark/new_prot_bench.jl
  - benchmark/benchmark_gpu_tiles.jl
  - test/gpu_optimizations.jl

  ### What to add:

  - A stable benchmark case for:
      - the `66ccb5` dense cutoff-style baseline
      - the current tile-list path on the same dense system
      - one sparse synthetic system where tile-list behavior should help
  - Timing breakdowns for:
      - Morton sort/compress/reorder
      - tile finding
      - force kernel
      - reverse reorder
      - energy kernel
  - Optional debug counters for num_interacting_tiles, clean vs excluded tiles, and average active pairs per tile.

  ### Why first:

  - Every later patch should be judged against the same workload and the same
    counters.
  - The dense benchmark must be a hard gate against regressions from `66ccb5`, not
    just against the current tree.

  ### Verification:

  - Benchmarks run before and after each phase with identical launch config.
  - The dense benchmark reports current / candidate / `66ccb5` side by side.
  - Existing GPU correctness tests still pass.

  ### Status:

  - Done.
  - The harness is now good enough to gate the next kernel changes.
  - The most relevant immediate use is to compare:
      - default tile path
      - `MOLLY_CUDA_FORCE_PATH=dense`
      - later auto-dispatch logic
  - The harness already exposed that `compress_boolean_matrices!` is a large
    contributor on the protein path, but this is not the first thing to fix
    because the primary regression still begins in the force-path redesign.

## 2. Recover the `66ccb5` dense-force path before optimizing further

  ### Patch order:

  - ext/MollyCUDAExt.jl
  - src/force.jl
  - benchmark/new_prot_bench.jl
  - benchmark/benchmark_gpu_tiles.jl
  - test/gpu_tile_lists.jl
  - test/gpu_simple.jl

  ### Concrete change:

  - Reintroduce a fused dense CUDA force/energy path with the essential
    `66ccb5` properties:
      - block-local box pruning inside the hot kernel, or an equivalent scheme
        that does not materialize a global tile list for dense cases
      - `BLOCK_Y` warps grouped on the same `i` tile
      - shared-memory reduction before global force writes
  - Keep the tile-list path available, but do not force all workloads through it.
  - Add a runtime selection strategy:
      - use the fused `66ccb5`-style path for dense or near-dense workloads
      - use the tile-list path only when sparsity is strong enough to pay for the
        prepass and changed accumulation pattern
  - Make the first target explicit: dense-system performance must return to
    `66ccb5`-class numbers before proceeding with more tile-list work.

  ### What is already done:

  - The fused dense force kernel has been restored in the current tree.
  - A matching dense energy kernel now also exists as an explicit override path.
  - Correctness against CPU forces and energy has been checked.
  - Runtime auto-dispatch now uses the dense force path for dense workloads and
    keeps the tile-list force path for sparse ones.
  - Dense synthetic and reduced protein measurements confirm that the force-path
    recovery moved in the right direction.
  - Reduced protein measurements also show that the dense energy path should
    remain opt-in for now.

  ### What is not done yet:

  - Energy auto-dispatch is intentionally not enabled yet.
  - The current default is now a split policy:
      - `force=dense` for dense or near-dense systems
      - `force=tile` for sparse systems
      - `energy=tile` unless an explicit override requests dense
  - Dense-system performance has improved, but we have not yet shown a full
    return to `66ccb5`-class end-to-end behavior.

  ### Main edit sites:

  - ext/MollyCUDAExt.jl: force-path mode selection and dense launch params
  - ext/MollyCUDAExt.jl: `pairwise_forces_loop_gpu!`
  - ext/MollyCUDAExt.jl: `pairwise_forces_loop_gpu_dense!`
  - ext/MollyCUDAExt.jl: `dense_force_kernel!`
  - src/force.jl:165

  ### Verification:

  - Dense benchmark is no slower than `66ccb5` within measurement noise.
  - Sparse benchmark is not catastrophically worse than the current tile-list
    path.
  - Force/energy results unchanged.

  ### Revised next substep:

  - Do not auto-enable the dense energy path yet.
  - The next granular changes should focus on the retained tile-list path,
    especially the energy-side preprocessing costs that still dominate the
    reduced protein profile (`compress_boolean_matrices!`, reorder, and tile
    finding).
  - Any future energy auto-dispatch should be gated by reduced protein PE
    measurements, not just by synthetic dense cases.

## 3. Keep the device-side tile-count consumption, but only as part of the retained tile path

  ### Patch order:

  - ext/MollyCUDAExt.jl
  - src/force.jl if buffer shape/API changes are needed
  - test/gpu_simple.jl
  - test/gpu_tile_lists.jl

  ### Concrete change:

  - Keep the current device-side `num_interacting_tiles` consumption and
    overflow-flag handling in the tile-list path.
  - Do not spend more engineering time on this as a standalone optimization until
    step 2 is done.
  - Ensure the fused dense path does not accidentally reintroduce a CPU round-trip
    if it still needs any metadata from the tile pipeline.

  ### Updated state:

  - This is still the right plan.
  - The newly restored dense force path does not consume the tile list and does
    not reintroduce the old host round-trip.
  - The tile-count machinery remains useful for:
      - the tile-list path
      - profiling and later auto-dispatch heuristics

  ### Main edit sites:

  - ext/MollyCUDAExt.jl:175
  - ext/MollyCUDAExt.jl:277
  - ext/MollyCUDAExt.jl:713
  - ext/MollyCUDAExt.jl:1099

  ### Verification:

  - No CPU round-trip in Nsight Systems between tile finding and force/energy kernels.
  - Dense benchmark does not move backward after step 2.

## 4. Split tile types more aggressively: clean, masked, sparse

  ### Patch order:

  - ext/MollyCUDAExt.jl
  - src/force.jl only if buffer fields change
  - test/gpu_tile_lists.jl
  - test/gpu_optimizations.jl

  ### Concrete change:

  - Extend `find_interacting_blocks_kernel!` to classify tiles into:
      - clean full tile
      - excluded/special full tile
      - sparse tile with only a subset of candidate pairs worth evaluating
  - Add an auxiliary sparse-pair representation per tile, ideally a compact
    bitmask first, not a full pair list yet.
  - In `force_kernel!` and `energy_kernel!`, keep the current clean fast path and
    add a sparse path that skips dead pair slots before evaluating the interaction.

  ### Main edit sites:

  - ext/MollyCUDAExt.jl:646
  - ext/MollyCUDAExt.jl:687
  - ext/MollyCUDAExt.jl:1075

  ### Notes:

  - Do not start with a full OpenMM-like `interactingAtoms` clone.
  - First get a profitable mask-based skip path with minimal buffer growth.
  - This step is for making the retained tile path worth using on sparse systems,
    not for fixing the dense regression.

  ### Verification:

  - Sparse-system benchmark improves.
  - Dense-system benchmark does not regress materially.

## 5. Introduce packed GPU parameter arrays for hot pairwise kernels

  ### Patch order:

  - src/types.jl
  - src/force.jl
  - ext/MollyCUDAExt.jl
  - Potentially src/setup.jl if system initialization is the right place to
    build/cache them
  - test/protein.jl

  ### Concrete change:

  - Add cached GPU-side arrays for the common nonbonded fields used in CUDA kernels, for example:
      - charge
      - sigma
      - epsilon
      - maybe atom type index if needed
  - Stop shuffling/reconstructing full Atom structs in the tile kernels for supported built-in interactions.
  - Keep the generic atom-based path as a fallback.

  ### Main edit sites:

  - src/types.jl:250
  - ext/MollyCUDAExt.jl:736
  - ext/MollyCUDAExt.jl:1146

  ### Verification:

  - Register count drops for force_kernel!.
  - Global memory traffic per interaction drops.
  - Correctness against OpenMM still holds.

## 6. Add a specialized CUDA fast path for dominant built-in pairwise combinations

  ### Patch order:

  - src/kernels.jl
  - src/interactions/lennard_jones.jl
  - src/interactions/coulomb.jl
  - ext/MollyCUDAExt.jl
  - test/protein.jl
  - test/gpu_simple.jl

  ### Concrete change:

  - Keep sum_pairwise_forces and sum_pairwise_potentials for generality.
  - Add dispatch branches in the CUDA extension for:
      - Tuple{LennardJones}
      - Tuple{CoulombReactionField}
      - Tuple{LennardJones,CoulombReactionField} or the most common actual
        system combination
  - Inline the arithmetic directly in the CUDA tile kernel using packed parameter arrays.
  - Route unsupported/custom interaction tuples back to the generic path.

  ### Main edit sites:

  - src/kernels.jl:3
  - ext/MollyCUDAExt.jl:672

  ### Verification:

  - Protein cutoff benchmark improves measurably.
  - No regression for custom interactions because fallback still exists.

## 7. Amortize preprocessing across force and energy evaluations

  ### Patch order:

  - src/force.jl
  - src/energy.jl
  - ext/MollyCUDAExt.jl
  - test/protein.jl

  ### Concrete change:

  - Cache reorder state, compressed masks, and tile metadata for the current coordinate snapshot when safe.
  - Avoid rebuilding the same ordering/compression when force and energy are requested back-to-back without coordinate changes.
  - Keep invalidation simple: coordinate write, neighbor-finder reorder step, or boundary change invalidates the cache.

  ### Why later:

  - Useful, but less rewarding until the hot force path is fixed.

## 8. Only then revisit accumulation strategy

  ### Patch order:

  - ext/MollyCUDAExt.jl
  - test/protein.jl
  - test/gpu_float64.jl

  ### Concrete change:

  - Evaluate whether per-warp or per-block buffered accumulation can reduce floating-point atomic pressure further.
  - If the tile-list path remains, evaluate sorting or grouping tiles by `i` so it
    can recover the old block-local reduction pattern instead of always doing
    per-warp direct global atomics.
  - Do not jump straight to OpenMM-style fixed-point accumulation unless profiling shows atomics are still dominant after the earlier phases.

## Recommended patch sequence for actual work

  1. Instrumentation and benchmark stabilization.
  2. Recover `66ccb5`-class dense performance with a fused or hybrid dense path.
  3. Keep device-side tile-count consumption as part of the retained tile path.
  4. Add sparse-tile skip path.
  5. Add packed parameter arrays.
  6. Add specialized LJ / Coulomb fast path.
  7. Cache preprocessing across force/energy.
  8. Reassess atomics only after re-profiling.

  ### Checkpointing
  After steps 2, 5, and 6, it would be worth generating a commit if the numbers
  are good. If you want, I can start with step 2 and implement the dense-path
  recovery patch set.
