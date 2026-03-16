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

## 0.6. What has been tested after the dense-path recovery

  ### Baseline used for all follow-up experiments:

  - The recovered dense-force / split force-energy state was committed as:
      - `6b3cff47`
      - commit message: `Recover dense GPU nonbonded paths and profiling`
  - Before trying any further optimization, the validation baseline was:
      - `CUDA_VISIBLE_DEVICES=1 julia +1.11 -e 'include("test/gpu_optimizations.jl")'`
      - result: `635 / 635` passing tests
  - The main comparison runs were:
      - `CUDA_VISIBLE_DEVICES=1 MOLLY_CUDA_BENCH_SAMPLES=2 MOLLY_CUDA_TILE_BENCH_ATOMS=1024 julia +1.11 benchmark/benchmark_gpu_tiles.jl`
      - `CUDA_VISIBLE_DEVICES=1 MOLLY_CUDA_DEVICE=0 MOLLY_CUDA_BENCH_SAMPLES=1 MOLLY_CUDA_BENCH_MIN_STEPS=0 julia +1.11 benchmark/new_prot_bench.jl`
  - These were chosen because they cover:
      - one dense synthetic case where the `66ccb5`-style path should win
      - one sparse synthetic case where the tile path should remain competitive
      - one reduced real protein case where preprocessing costs dominate and bad
        “optimizations” show up quickly

  ### Experiment A: compute mask compression only for interacting tiles

  - Why it was worth trying:
      - The reduced protein profile shows `compress_boolean_matrices!` as a
        major cost.
      - The existing tile path compresses the entire upper-triangular tile space
        before box pruning, even though many tiles are later discarded.
      - A natural idea is to push compression into `find_interacting_blocks_kernel!`
        so masks are produced only for tiles that already passed the box test.
  - How it was implemented:
      - `find_interacting_blocks_kernel!` in `ext/MollyCUDAExt.jl` was extended
        to receive `sorted_seq`, `eligible`, and `special`.
      - The kernel then computed per-tile compressed masks on demand and used
        those masks immediately to classify tiles as clean or masked.
      - The old global `compress_boolean_matrices!` launch was removed from the
        tile-list force and energy paths only; dense paths were left unchanged.
  - How it was tested:
      - First by correctness:
          - `CUDA_VISIBLE_DEVICES=1 julia +1.11 -e 'include("test/gpu_optimizations.jl")'`
      - Then by performance on the synthetic benchmark:
          - `CUDA_VISIBLE_DEVICES=1 MOLLY_CUDA_BENCH_SAMPLES=2 MOLLY_CUDA_TILE_BENCH_ATOMS=1024 julia +1.11 benchmark/benchmark_gpu_tiles.jl`
      - Then by re-checking the reduced protein benchmark if the synthetic run
        looked promising.
  - What happened:
      - After fixing one wiring bug around the last partial tile size, the code
        was correct.
      - But the tile-finding stage became much more expensive than the original
        split `compress + tile_find` design.
      - The sparse synthetic case regressed sharply, which means the work was
        not actually being reduced; it was just being moved into a slower part of
        the pipeline.
  - Conclusion:
      - This is the wrong granularity for the current code structure.
      - On-demand compression inside tile finding should not be pursued further
        without a more substantial redesign of the tile-builder kernel.

  ### Experiment B: use warp votes inside the energy kernels to skip inactive work

  - Why it was worth trying:
      - The force kernels already use `CUDA.vote_any_sync` in their clean-tile
        fast paths to avoid needless work when no lane has an active pair.
      - The energy kernels still evaluated the branch structure more directly.
      - Adding the same warp vote seemed like a small, local optimization with
        limited surface area.
  - How it was implemented:
      - `energy_kernel!` and `dense_energy_kernel!` in `ext/MollyCUDAExt.jl`
        were patched to introduce `any_active = CUDA.vote_any_sync(...)` around
        the pairwise potential calculation.
      - The initial patch touched both the uniform-loop and the diagonal /
        partial-tile branches.
  - How it was tested:
      - First with the GPU correctness suite:
          - `CUDA_VISIBLE_DEVICES=1 julia +1.11 -e 'include("test/gpu_optimizations.jl")'`
      - After the first correctness failure, the patch was restricted to the
        uniform-loop branches and tested again with the same command.
  - What happened:
      - The first version produced wrong energies because the diagonal branches
        have lane-dependent loop bounds; a warp-wide vote is not equivalent
        there.
      - The restricted version avoided the obvious numerical error, but the test
        run became unreliable enough that it was not a safe direction to keep.
      - At that point the optimization had already failed the main criterion:
        “small, obviously safe, and clearly heading the right way.”
  - Conclusion:
      - The energy kernels are more sensitive to this kind of warp-synchronous
        change than the force kernels.
      - Any future warp-vote use in the energy path needs a more formal
        case-by-case proof of warp participation, not a quick transplant from
        the force code.

  ### Experiment C: make `compress_boolean_matrices!` multi-warp per block

  - Why it was worth trying:
      - `compress_boolean_matrices!` is still one of the clearest remaining
        hotspots on the reduced protein benchmark.
      - The kernel is embarrassingly parallel across `j` tiles for a fixed `i`
        tile, so mapping multiple warps to a block looked like a plausible way to
        increase throughput without changing the bit-packing logic.
  - How it was implemented:
      - A new launch helper selected a `block_y` for compression.
      - `compress_boolean_matrices!` was rewritten so `threadIdx().y` selected
        which `j` tile a warp handled inside a block.
      - All call sites in `ext/MollyCUDAExt.jl` and the benchmark profiling code
        were updated to launch `(32, block_y)` threads and fewer `y` blocks.
  - How it was tested:
      - First with the GPU correctness suite:
          - `CUDA_VISIBLE_DEVICES=1 julia +1.11 -e 'include("test/gpu_optimizations.jl")'`
      - Then with the synthetic benchmark:
          - `CUDA_VISIBLE_DEVICES=1 MOLLY_CUDA_BENCH_SAMPLES=2 MOLLY_CUDA_TILE_BENCH_ATOMS=1024 julia +1.11 benchmark/benchmark_gpu_tiles.jl`
      - Then with an explicit block-size sanity check:
          - `CUDA_VISIBLE_DEVICES=1 MOLLY_CUDA_COMPRESS_BLOCK_Y=8 MOLLY_CUDA_BENCH_SAMPLES=1 MOLLY_CUDA_TILE_BENCH_ATOMS=1024 julia +1.11 benchmark/benchmark_gpu_tiles.jl`
  - What happened:
      - Correctness remained intact.
      - Performance collapsed across the board, including the profiling stages
        themselves, which started reporting multi-millisecond times for nearly
        every stage on the small synthetic case.
      - The explicit `MOLLY_CUDA_COMPRESS_BLOCK_Y=8` sanity run did not rescue
        the result, so the issue was not just a poor automatic block-size pick.
  - Conclusion:
      - The naive multi-warp mapping is not compatible with the current memory
        access pattern of this kernel.
      - This path should not be revisited without a deeper analysis of memory
        access, occupancy, and how the compressed arrays are laid out.

  ### Overall result of the post-recovery experiments:

  - All three experiments were reverted.
  - This was intentional: none of them met the “granular change moving in the
    right direction” bar.
  - The repository was brought back to the committed recovery state, and the
    baseline validation was re-run:
      - `CUDA_VISIBLE_DEVICES=1 julia +1.11 -e 'include("test/gpu_optimizations.jl")'`
      - result: `635 / 635` passing tests
  - Therefore the current verified project state is still exactly the split
    policy captured in `6b3cff47`:
      - dense force path recovered and auto-selected for dense workloads
      - tile force path retained for sparse workloads
      - dense energy path exists but stays opt-in
      - no additional post-commit optimization has yet been proven beneficial

  ### What this means for the next round of work:

  - The remaining cost in the retained tile path is real, but the last three
    “local” ideas all failed for understandable reasons:
      - moving work into tile finding destroyed sparsity wins
      - warp-vote changes in energy kernels are easy to get wrong
      - more warps per compression block is not automatically better
  - The next optimization should therefore start from a more explicit
    performance model, not from another small speculative kernel tweak.
  - The most defensible next directions are:
      - reduce duplicate preprocessing between force and energy evaluations
      - redesign mask compression with memory layout in mind
      - further specialize the tile path only after proving the access pattern
        and synchronization model are sound

## 0.7. Current iteration: triangular-mask stabilization and compression micro-optimizations

  ### Why this iteration was needed:

  - After switching compressed mask storage to upper-triangular indexing, the
    dense force path failed to compile on GPU due dynamic dispatch around
    `upper_tile_index`.
  - In parallel, the reduced protein profile still showed mask compression as a
    dominant preprocessing cost (`~3.9 ms` in the tile energy path), so small,
    low-risk compression-kernel improvements were tested.

  ### Change 1: fix GPU compile regression in dense force path

  - File: `ext/MollyCUDAExt.jl`
  - `dense_force_kernel!` now casts block-derived tile indices to `Int32` at
    assignment:
      - `i = Int32(blockIdx().x)`
      - `j = Int32((blockIdx().y - a) * BLOCK_Y + threadIdx().y)`
  - Result:
      - The `InvalidIRError` (`unsupported dynamic function invocation` for
        `upper_tile_index`) is gone.

  ### Change 2: strengthen index-helper type stability

  - File: `ext/MollyCUDAExt.jl`
  - Added inline integer-wrapper methods:
      - `upper_tile_row_start(::Integer, ::Integer)`
      - `upper_tile_index(::Integer, ::Integer, ::Integer)`
      - `upper_tile_ij(::Integer, ::Integer)`
  - Also cast `dense_energy_kernel!` tile indices to `Int32` for consistency.
  - Result:
      - Dense force/energy kernels now use a more robust path for tile-index
        helper calls.

  ### Change 3: compression-kernel memory-traffic cleanup

  - File: `ext/MollyCUDAExt.jl`
  - `compress_boolean_matrices!` now stages tile-`j` sorted indices in shared
    memory once per tile (`CuStaticSharedArray(Int32, 32)`), then reuses them
    across lanes.
  - This removes repeated warp-wide reloads of identical
    `sorted_seq[j_0_tile + m]` values.

  ### Change 4: compression-kernel tile decode cleanup

  - File: `ext/MollyCUDAExt.jl`
  - `compress_boolean_matrices!` now performs `tile_idx -> (i,j)` decode only
    on the warp leader and broadcasts via `CUDA.shfl_sync`.
  - This removes redundant per-lane decode work.

  ### Validation commands used in this iteration:

  - Correctness:
      - `CUDA_VISIBLE_DEVICES=1 julia +1.11 -e 'include("test/gpu_optimizations.jl")'`
      - Re-run after each granular step.
  - Synthetic profiling:
      - `CUDA_VISIBLE_DEVICES=1 MOLLY_CUDA_BENCH_SAMPLES=2 MOLLY_CUDA_TILE_BENCH_ATOMS=1024 julia +1.11 benchmark/benchmark_gpu_tiles.jl`
  - Reduced protein profiling:
      - `CUDA_VISIBLE_DEVICES=1 MOLLY_CUDA_DEVICE=0 MOLLY_CUDA_BENCH_SAMPLES=1 MOLLY_CUDA_BENCH_MIN_STEPS=0 julia +1.11 benchmark/new_prot_bench.jl`

  ### Measured outcomes:

  - Correctness:
      - `test/gpu_optimizations.jl`: `635 / 635` passing after each fix set.
  - Synthetic benchmark (1024 atoms):
      - Compression remains low (`~0.017–0.029 ms`) and behavior remains stable.
  - Reduced protein benchmark:
      - Energy-path compression remains the dominant preprocessing stage at
        roughly `3.86–4.13 ms` depending on policy/sample.
      - Net effect of these micro-optimizations on the protein-level
        `compress_ms` is neutral to very small (no robust macro reduction yet).

  ### Interpretation from this iteration:

  - The crucial compile regression from the triangular-mask transition is fixed.
  - The remaining high `compress_ms` appears dominated by core matrix-read work
    in mask construction, not by index decoding or repeated `sorted_seq` loads.
  - Therefore, further gains likely require a more structural preprocessing
    redesign (e.g., reducing the amount of pair-mask materialization), not more
    arithmetic micro-tuning inside the current kernel.

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
