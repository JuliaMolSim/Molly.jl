# IMPLEMENTATION OF BETTER PERFORMANCE GPU KERNELS

## 1. Baseline and measurement harness

  ### Patch order:

  - benchmark/new_prot_bench.jl
  - benchmark/benchmark_gpu_tiles.jl
  - test/gpu_optimizations.jl

  ### What to add:

  - A stable benchmark case for the CUDA tile path on one realistic cutoff
    system and one sparse synthetic system.
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

  ### Verification:

  - Benchmarks run before and after each phase with identical launch config.
  - Existing GPU correctness tests still pass.

## 2. Remove the host synchronization on num_interacting_tiles

  ### Patch order:

  - ext/MollyCUDAExt.jl
  - src/force.jl if buffer shape/API changes are needed
  - test/gpu_simple.jl
  - test/gpu_tile_lists.jl

  ### Concrete change:

  - Replace num_tiles = Array(buffers.num_interacting_tiles)[1] in the force and energy paths with device-side consumption.
  - Launch over length(buffers.interacting_tiles_i) and have each warp/thread block early-return when tile_idx > num_interacting_tiles[1].
  - Keep overflow detection on-device with a separate flag/counter rather than a host read in the hot path.

  ### Main edit sites:

  - ext/MollyCUDAExt.jl:160
  - ext/MollyCUDAExt.jl:263
  - ext/MollyCUDAExt.jl:697
  - ext/MollyCUDAExt.jl:1106

  ### Verification:

  - No CPU round-trip in Nsight Systems between tile finding and force/energy kernels.
  - Force/energy results unchanged.

## 3. Split tile types more aggressively: clean, masked, sparse

  ### Patch order:

  - ext/MollyCUDAExt.jl
  - src/force.jl only if buffer fields change
  - test/gpu_tile_lists.jl
  - test/gpu_optimizations.jl

  ### Concrete change:

  - Extend find_interacting_blocks_kernel! to classify tiles into:
      - clean full tile
      - excluded/special full tile
      - sparse tile with only a subset of candidate pairs worth evaluating
  - Add an auxiliary sparse-pair representation per tile, ideally a compact bitmask first, not a full pair list yet.
  - In force_kernel! and energy_kernel!, keep the current clean fast path and add a sparse path that skips dead pair slots before evaluating the interaction.

  ### Main edit sites:

  - ext/MollyCUDAExt.jl:634
  - ext/MollyCUDAExt.jl:672
  - ext/MollyCUDAExt.jl:1066

  ### Notes:

  - Do not start with a full OpenMM-like interactingAtoms clone.
  - First get a profitable mask-based skip path with minimal buffer growth.

  ### Verification:

  - Sparse-system benchmark improves.
  - Dense-system benchmark does not regress materially.

## 4. Introduce packed GPU parameter arrays for hot pairwise kernels

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

## 5. Add a specialized CUDA fast path for dominant built-in pairwise combinations

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

## 6. Amortize preprocessing across force and energy evaluations

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

## 7. Only then revisit accumulation strategy

  ### Patch order:

  - ext/MollyCUDAExt.jl
  - test/protein.jl
  - test/gpu_float64.jl

  ### Concrete change:

  - Evaluate whether per-warp or per-block buffered accumulation can reduce floating-point atomic pressure further.
  - Do not jump straight to OpenMM-style fixed-point accumulation unless profiling shows atomics are still dominant after the earlier phases.

## Recommended patch sequence for actual work

  1. Instrumentation and benchmark stabilization.
  2. Remove host sync on tile count.
  3. Add sparse-tile skip path.
  4. Add packed parameter arrays.
  5. Add specialized LJ / Coulomb fast path.
  6. Cache preprocessing across force/energy.
  7. Reassess atomics only after re-profiling.

  ### Checkpointing
  After steps 2, 4, and 5, it would be worth generating a commit if the numbers are good. If you want, I can start with step 2 and implement the first patch set.
