# CUDA kernels that use warp-level features
# This file is only loaded when CUDA is imported

"""
    MollyCUDAExt

CUDA extension for Molly.jl. This module provides highly optimized CUDA kernels
for pairwise force and energy calculations, utilizing warp-level primitives,
Morton ordering for spatial locality, and a tiled preprocessing pipeline.

The pipeline generally follows these steps:
1.  **Reordering**: Atoms are periodically reordered based on Morton (Z-order) curves
    to improve cache hits during pairwise interactions.
2.  **Compression**: Interaction matrices (eligibility and special interactions) are
    compressed into bitmasks (32x32 tiles) to save memory and speed up lookups.
3.  **Tile Finding**: A kernel identifies pairs of 32x32 atom blocks (tiles) that are
    within the interaction cutoff, using bounding box checks.
4.  **Execution**: Specialized kernels iterate over the list of interacting tiles,
    using warp shuffles to efficiently compute forces/energies within each tile.
"""
module MollyCUDAExt

using Molly
using Molly: from_device, box_sides, sorted_morton_seq!, sum_pairwise_forces,
             sum_pairwise_potentials, volume
using CUDA
using Atomix
using KernelAbstractions

const WARPSIZE = UInt32(32)
const MAX_BLOCK_Y = 32
const AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES = (1, 2, 4, 8, 16)
const AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES = (1, 2, 4, 8, 16)
const AUTOTUNE_TILE_THREAD_CANDIDATES = ((8, 8), (16, 8), (16, 16), (32, 8), (32, 16))
const AUTOTUNE_WARMUP_RUNS = 1
const AUTOTUNE_MEASURE_RUNS = 3

struct LaunchAutotuneKey
    device_name::String
    capability::String
    sm_count::Int
    coord_type::DataType
    boundary_type::DataType
    force_units_type::DataType
    energy_units_type::DataType
    dim::Int
    n_atoms::Int
    n_blocks::Int
    box_signature::Tuple
    r_cut::Float64
    r_neighbors::Float64
    interaction_types::Tuple
    force_maxregs::Union{Nothing, Int}
end

const CUDA_LAUNCH_AUTOTUNE_CACHE = Dict{LaunchAutotuneKey, Molly.CUDALaunchConfig}()
const CUDA_LAUNCH_AUTOTUNE_LOCK = ReentrantLock()

function __init__()
    empty!(CUDA_LAUNCH_AUTOTUNE_CACHE)
    Molly.CUDA_LAUNCH_AUTOTUNE_CACHE_RESET_HOOK[] = () -> begin
        lock(CUDA_LAUNCH_AUTOTUNE_LOCK) do
            empty!(CUDA_LAUNCH_AUTOTUNE_CACHE)
        end
        return nothing
    end
    return nothing
end

Molly.uses_gpu_neighbor_finder(::Type{<:CuArray}) = true

CUDA.Const(nl::Molly.NoNeighborList) = nl

function env_int(name::AbstractString)
    value = ENV[name]
    parsed = tryparse(Int, value)
    parsed === nothing && error("Invalid integer value for $(name): $(repr(value))")
    return parsed
end

function env_override(name::AbstractString)
    return haskey(ENV, name) ? env_int(name) : nothing
end

prefer_override(primary, secondary) = primary === nothing ? secondary : primary

function validate_block_y(name::AbstractString, block_y::Int)
    1 <= block_y <= MAX_BLOCK_Y || error("$(name) must be in 1:$(MAX_BLOCK_Y), got $(block_y)")
    return block_y
end

function choose_block_y(conf_threads::Int)
    return max(1, min(MAX_BLOCK_Y, fld(conf_threads, Int(WARPSIZE))))
end

function choose_tile_threads(conf_threads::Int)
    threads_x = min(Int(WARPSIZE), conf_threads)
    threads_y = max(1, min(MAX_BLOCK_Y, fld(conf_threads, threads_x)))
    return (threads_x, threads_y)
end

@inline autotune_scalar(x) = round(Float64(ustrip(x)); sigdigits=12)

function autotune_box_signature(boundary, ::Val{D}) where D
    sides = box_sides(boundary)
    return ntuple(i -> autotune_scalar(sides[i]), D)
end

gpu_neighbor_pairwise_inters(sys) = Tuple(filter(use_neighbors, values(sys.pairwise_inters)))

function effective_tile_threads_override(config::Molly.CUDALaunchConfig)
    threads_x_env = env_override("MOLLY_CUDA_TILE_THREADS_X")
    threads_y_env = env_override("MOLLY_CUDA_TILE_THREADS_Y")
    if xor(threads_x_env === nothing, threads_y_env === nothing)
        error("Set both MOLLY_CUDA_TILE_THREADS_X and MOLLY_CUDA_TILE_THREADS_Y together")
    end
    return config.tile_threads === nothing ?
           (threads_x_env === nothing ? nothing : (threads_x_env, threads_y_env)) :
           config.tile_threads
end

effective_force_block_y_override(config::Molly.CUDALaunchConfig) =
    prefer_override(config.force_block_y, env_override("MOLLY_CUDA_FORCE_BLOCK_Y"))

effective_energy_block_y_override(config::Molly.CUDALaunchConfig) =
    prefer_override(config.energy_block_y, env_override("MOLLY_CUDA_ENERGY_BLOCK_Y"))

effective_force_maxregs_override(config::Molly.CUDALaunchConfig) =
    prefer_override(config.force_maxregs, env_override("MOLLY_CUDA_FORCE_MAXREGS"))

function autotune_key(sys::System{D, <:CuArray}, pairwise_inters, force_maxregs_override) where D
    dev = CUDA.device()
    sm_count = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    n_atoms = length(sys.coords)
    return LaunchAutotuneKey(
        CUDA.name(dev),
        string(CUDA.capability(dev)),
        sm_count,
        eltype(eltype(sys.coords)),
        typeof(sys.boundary),
        typeof(sys.force_units),
        typeof(sys.energy_units),
        D,
        n_atoms,
        cld(n_atoms, Int(WARPSIZE)),
        autotune_box_signature(sys.boundary, Val(D)),
        autotune_scalar(sys.neighbor_finder.dist_cutoff),
        autotune_scalar(sys.neighbor_finder.dist_neighbors),
        Tuple(map(typeof, pairwise_inters)),
        force_maxregs_override,
    )
end

function autotune_stage_time_ms!(f::F) where {F}
    CUDA.synchronize()
    start_ns = time_ns()
    f()
    CUDA.synchronize()
    return (time_ns() - start_ns) / 1.0e6
end

function autotune_benchmark_ms!(prepare!::F, run!::G;
                                warmup::Int=AUTOTUNE_WARMUP_RUNS,
                                repeats::Int=AUTOTUNE_MEASURE_RUNS) where {F, G}
    for _ in 1:warmup
        prepare!()
        run!()
        CUDA.synchronize()
    end

    best_ms = Inf
    for _ in 1:repeats
        prepare!()
        best_ms = min(best_ms, autotune_stage_time_ms!(run!))
    end
    return best_ms
end

function autotune_prepare_block_bounds!(buffers, sys::System{D, <:CuArray, T}, N::Int) where {D, T}
    n_blocks = cld(N, Int(WARPSIZE))
    if sys.boundary isa TriclinicBoundary
        H = SMatrix{3, 3, T}(
            sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
            sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
            sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val,
        )
        @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
            buffers.morton_seq,
            buffers.box_mins,
            buffers.box_maxs,
            sys.coords,
            inv(H),
            Val(N),
            sys.boundary,
            Val(D),
        )
    else
        @cuda blocks=n_blocks threads=32 kernel_min_max!(
            buffers.morton_seq,
            buffers.box_mins,
            buffers.box_maxs,
            sys.coords,
            Val(N),
            sys.boundary,
            Val(D),
        )
    end
    CUDA.synchronize()
    return nothing
end

function autotune_prepare_common_state!(buffers, sys::System{D, <:CuArray}, N::Int) where D
    morton_bits = 10
    sides = box_sides(sys.boundary)
    cell_width = sides ./ (2^morton_bits)
    sorted_morton_seq!(buffers, sys.coords, cell_width, morton_bits)
    compress_sparse!(buffers, sys.neighbor_finder, Val(N))
    reorder_system_gpu!(buffers, sys)
    KernelAbstractions.synchronize(get_backend(sys.coords))
    autotune_prepare_block_bounds!(buffers, sys, N)
    return nothing
end

function autotune_tile_kernel(buffers, sys::System{D, <:CuArray}, N::Int) where D
    n_blocks = cld(N, Int(WARPSIZE))
    max_tiles = length(buffers.interacting_tiles_i)
    return @cuda launch=false find_interacting_blocks_kernel!(
        buffers.interacting_tiles_i,
        buffers.interacting_tiles_j,
        buffers.interacting_tiles_type,
        buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow,
        buffers.box_mins,
        buffers.box_maxs,
        sys.boundary,
        sys.neighbor_finder.dist_neighbors,
        Val(n_blocks),
        Val(D),
        max_tiles,
        buffers.compressed_masks,
        buffers.tile_is_clean,
    )
end

function launch_autotune_tile_kernel!(kernel, buffers, sys::System{D, <:CuArray}, N::Int,
                                      threads_xy::NTuple{2, Int}) where D
    n_blocks = cld(N, Int(WARPSIZE))
    max_tiles = length(buffers.interacting_tiles_i)
    kernel(
        buffers.interacting_tiles_i,
        buffers.interacting_tiles_j,
        buffers.interacting_tiles_type,
        buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow,
        buffers.box_mins,
        buffers.box_maxs,
        sys.boundary,
        sys.neighbor_finder.dist_neighbors,
        Val(n_blocks),
        Val(D),
        max_tiles,
        buffers.compressed_masks,
        buffers.tile_is_clean;
        blocks=(cld(n_blocks, threads_xy[1]), cld(n_blocks, threads_xy[2])),
        threads=threads_xy,
    )
    return nothing
end

function autotune_tile_thread_candidates(kernel)
    max_threads = CUDA.maxthreads(kernel)
    candidates = NTuple{2, Int}[]
    for threads_xy in AUTOTUNE_TILE_THREAD_CANDIDATES
        threads_x, threads_y = threads_xy
        if threads_x * threads_y <= max_threads && threads_y <= fld(max_threads, threads_x)
            push!(candidates, threads_xy)
        end
    end
    isempty(candidates) && push!(candidates, tile_launch_params(kernel))
    return candidates
end

function autotune_block_y_candidates(kernel, fallback_block_y::Int, candidates)
    max_block_y = max(1, fld(CUDA.maxthreads(kernel), Int(WARPSIZE)))
    valid_candidates = Int[]
    for block_y in candidates
        block_y <= max_block_y && push!(valid_candidates, block_y)
    end
    isempty(valid_candidates) && push!(valid_candidates, min(fallback_block_y, max_block_y))
    return valid_candidates
end

function autotune_tile_threads!(buffers, sys::System{D, <:CuArray}, N::Int) where D
    kernel = autotune_tile_kernel(buffers, sys, N)
    candidates = autotune_tile_thread_candidates(kernel)
    best_threads = first(candidates)
    best_ms = Inf
    expected_num_tiles = nothing

    for threads_xy in candidates
        ms = autotune_benchmark_ms!(
            () -> reset_interacting_tile_state!(buffers),
            () -> launch_autotune_tile_kernel!(kernel, buffers, sys, N, threads_xy),
        )
        overflow_count = Int(only(from_device(buffers.interacting_tiles_overflow)))
        overflow_count == 0 || continue

        num_tiles = Int(only(from_device(buffers.num_interacting_tiles)))
        if expected_num_tiles === nothing
            expected_num_tiles = num_tiles
        elseif num_tiles != expected_num_tiles
            continue
        end

        if ms < best_ms
            best_ms = ms
            best_threads = threads_xy
        end
    end

    reset_interacting_tile_state!(buffers)
    launch_autotune_tile_kernel!(kernel, buffers, sys, N, best_threads)
    CUDA.synchronize()
    throw_if_interacting_tiles_overflowed(buffers)
    buffers.last_r_cut = ustrip(sys.neighbor_finder.dist_neighbors)
    buffers.num_pairs = Int(only(from_device(buffers.num_interacting_tiles)))
    return best_threads
end

function autotune_force_kernel(buffers, sys::System{D, <:CuArray, T}, pairwise_inters,
                               N::Int, force_maxregs_override) where {D, T}
    if force_maxregs_override === nothing
        return @cuda launch=false always_inline=true force_kernel!(
            buffers.fs_mat_reordered,
            buffers.virial_nounits,
            buffers.coords_reordered,
            buffers.velocities_reordered,
            buffers.atoms_reordered,
            Val(N),
            sys.neighbor_finder.dist_cutoff,
            Val(sys.force_units),
            pairwise_inters,
            sys.boundary,
            0,
            buffers.compressed_masks,
            Val(false),
            Val(T),
            Val(D),
            buffers.interacting_tiles_i,
            buffers.interacting_tiles_j,
            buffers.interacting_tiles_type,
            buffers.num_interacting_tiles,
            buffers.interacting_tiles_overflow,
        )
    end

    return @cuda launch=false maxregs=force_maxregs_override always_inline=true force_kernel!(
        buffers.fs_mat_reordered,
        buffers.virial_nounits,
        buffers.coords_reordered,
        buffers.velocities_reordered,
        buffers.atoms_reordered,
        Val(N),
        sys.neighbor_finder.dist_cutoff,
        Val(sys.force_units),
        pairwise_inters,
        sys.boundary,
        0,
        buffers.compressed_masks,
        Val(false),
        Val(T),
        Val(D),
        buffers.interacting_tiles_i,
        buffers.interacting_tiles_j,
        buffers.interacting_tiles_type,
        buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow,
    )
end

function autotune_force_block_y!(buffers, sys::System{D, <:CuArray, T}, pairwise_inters,
                                 N::Int, force_maxregs_override) where {D, T}
    kernel = autotune_force_kernel(buffers, sys, pairwise_inters, N, force_maxregs_override)
    candidates = autotune_block_y_candidates(kernel, 4, AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES)
    num_pairs = buffers.num_pairs
    num_pairs == 0 && return first(candidates)

    best_block_y = first(candidates)
    best_ms = Inf
    for block_y in candidates
        n_blocks_launch = cld(num_pairs, block_y)
        ms = autotune_benchmark_ms!(
            () -> begin
                fill!(buffers.fs_mat_reordered, zero(T))
                fill!(buffers.virial_nounits, zero(T))
            end,
            () -> kernel(
                buffers.fs_mat_reordered,
                buffers.virial_nounits,
                buffers.coords_reordered,
                buffers.velocities_reordered,
                buffers.atoms_reordered,
                Val(N),
                sys.neighbor_finder.dist_cutoff,
                Val(sys.force_units),
                pairwise_inters,
                sys.boundary,
                0,
                buffers.compressed_masks,
                Val(false),
                Val(T),
                Val(D),
                buffers.interacting_tiles_i,
                buffers.interacting_tiles_j,
                buffers.interacting_tiles_type,
                buffers.num_interacting_tiles,
                buffers.interacting_tiles_overflow;
                threads=(32, block_y),
                blocks=n_blocks_launch,
            ),
        )
        if ms < best_ms
            best_ms = ms
            best_block_y = block_y
        end
    end
    return best_block_y
end

function autotune_energy_block_y!(buffers, sys::System{D, <:CuArray, T}, pairwise_inters,
                                  N::Int) where {D, T}
    kernel = @cuda launch=false always_inline=true energy_kernel!(
        buffers.pe_vec_nounits,
        buffers.coords_reordered,
        buffers.velocities_reordered,
        buffers.atoms_reordered,
        Val(N),
        sys.neighbor_finder.dist_cutoff,
        Val(sys.energy_units),
        pairwise_inters,
        sys.boundary,
        0,
        buffers.compressed_masks,
        Val(T),
        Val(D),
        buffers.interacting_tiles_i,
        buffers.interacting_tiles_j,
        buffers.interacting_tiles_type,
        buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow,
    )
    candidates = autotune_block_y_candidates(kernel, 4, AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES)
    num_pairs = buffers.num_pairs
    num_pairs == 0 && return first(candidates)

    best_block_y = first(candidates)
    best_ms = Inf
    for block_y in candidates
        n_blocks_launch = cld(num_pairs, block_y)
        ms = autotune_benchmark_ms!(
            () -> fill!(buffers.pe_vec_nounits, zero(T)),
            () -> kernel(
                buffers.pe_vec_nounits,
                buffers.coords_reordered,
                buffers.velocities_reordered,
                buffers.atoms_reordered,
                Val(N),
                sys.neighbor_finder.dist_cutoff,
                Val(sys.energy_units),
                pairwise_inters,
                sys.boundary,
                0,
                buffers.compressed_masks,
                Val(T),
                Val(D),
                buffers.interacting_tiles_i,
                buffers.interacting_tiles_j,
                buffers.interacting_tiles_type,
                buffers.num_interacting_tiles,
                buffers.interacting_tiles_overflow;
                blocks=n_blocks_launch,
                threads=(32, block_y),
            ),
        )
        if ms < best_ms
            best_ms = ms
            best_block_y = block_y
        end
    end
    return best_block_y
end

function autotune_cuda_launch_config(sys::System{D, <:CuArray, T}, pairwise_inters,
                                     force_maxregs_override) where {D, T}
    N = length(sys.coords)
    buffers = Molly.init_buffers!(sys, 1, true)
    autotune_prepare_common_state!(buffers, sys, N)
    tile_threads = autotune_tile_threads!(buffers, sys, N)
    force_block_y = autotune_force_block_y!(buffers, sys, pairwise_inters, N, force_maxregs_override)
    energy_block_y = autotune_energy_block_y!(buffers, sys, pairwise_inters, N)
    return Molly.CUDALaunchConfig(
        force_block_y=force_block_y,
        force_maxregs=nothing,
        tile_threads=tile_threads,
        energy_block_y=energy_block_y,
    )
end

function cached_autotune_config(key::LaunchAutotuneKey)
    lock(CUDA_LAUNCH_AUTOTUNE_LOCK) do
        return get(CUDA_LAUNCH_AUTOTUNE_CACHE, key, nothing)
    end
end

function cache_autotune_config!(key::LaunchAutotuneKey, config::Molly.CUDALaunchConfig)
    lock(CUDA_LAUNCH_AUTOTUNE_LOCK) do
        return get!(CUDA_LAUNCH_AUTOTUNE_CACHE, key, config)
    end
end

function Molly.optimize_cuda_launch_config!(sys::System{D, <:CuArray, T}) where {D, T}
    sys.neighbor_finder isa GPUNeighborFinder || return nothing
    pairwise_inters = gpu_neighbor_pairwise_inters(sys)
    isempty(pairwise_inters) && return nothing

    current_config = Molly.cuda_launch_config()
    force_block_y_override = effective_force_block_y_override(current_config)
    energy_block_y_override = effective_energy_block_y_override(current_config)
    tile_threads_override = effective_tile_threads_override(current_config)
    force_maxregs_override = effective_force_maxregs_override(current_config)

    current_config.force_block_y === nothing || validate_block_y("force_block_y", current_config.force_block_y)
    current_config.energy_block_y === nothing || validate_block_y("energy_block_y", current_config.energy_block_y)
    force_maxregs_override === nothing || force_maxregs_override > 0 ||
        error("MOLLY_CUDA_FORCE_MAXREGS must be positive, got $(force_maxregs_override)")

    needs_force = force_block_y_override === nothing
    needs_energy = energy_block_y_override === nothing
    needs_tile = tile_threads_override === nothing

    if !(needs_force || needs_energy || needs_tile)
        return force_block_y_override
    end

    key = autotune_key(sys, pairwise_inters, force_maxregs_override)
    tuned_config = cached_autotune_config(key)
    if tuned_config === nothing
        tuned_config = autotune_cuda_launch_config(sys, pairwise_inters, force_maxregs_override)
        tuned_config = cache_autotune_config!(key, tuned_config)
    end

    merged_config = Molly.CUDALaunchConfig(
        force_block_y = needs_force ? tuned_config.force_block_y : current_config.force_block_y,
        force_maxregs = current_config.force_maxregs,
        tile_threads = needs_tile ? tuned_config.tile_threads : current_config.tile_threads,
        energy_block_y = needs_energy ? tuned_config.energy_block_y : current_config.energy_block_y,
    )
    Molly.set_cuda_launch_config!(merged_config)

    return something(
        effective_force_block_y_override(merged_config),
        tuned_config.force_block_y,
    )
end

# The following functions handle indexing for the upper triangular part of the
# N_blocks x N_blocks tile matrix, stored as a 1D array.

"""
    upper_tile_count(n_blocks::Integer)

Returns the total number of tiles in the upper triangular part of an `n_blocks` x `n_blocks` matrix,
including the diagonal.
"""
upper_tile_count(n_blocks::Integer) = (Int64(n_blocks) * (Int64(n_blocks) + 1)) ÷ 2

"""
    upper_tile_index(i::Integer, j::Integer, n_blocks::Integer)

Maps 2D block indices `(i, j)` to a 1D index in the upper triangular storage.
Assumes `i <= j`.
"""
@inline function upper_tile_index(i::Int32, j::Int32, n_blocks::Int32)
    r = Int64(i - Int32(1))
    n = Int64(n_blocks)
    return Int32((r * (2 * n - r + 1)) ÷ 2 + Int64(j - i + Int32(1)))
end

@inline upper_tile_index(i::Integer, j::Integer, n_blocks::Integer) = upper_tile_index(Int32(i), Int32(j), Int32(n_blocks))

function force_launch_params(kernel)
    config = Molly.cuda_launch_config()
    block_y_override = prefer_override(config.force_block_y, env_override("MOLLY_CUDA_FORCE_BLOCK_Y"))
    maxregs_override = prefer_override(config.force_maxregs, env_override("MOLLY_CUDA_FORCE_MAXREGS"))
    block_y_override === nothing || validate_block_y("MOLLY_CUDA_FORCE_BLOCK_Y", block_y_override)
    maxregs_override === nothing || maxregs_override > 0 || error("MOLLY_CUDA_FORCE_MAXREGS must be positive, got $(maxregs_override)")

    conf = launch_configuration(kernel.fun)
    # Ensure block_y does not exceed physical kernel limits
    max_threads = CUDA.maxthreads(kernel)
    max_block_y = fld(max_threads, Int(WARPSIZE))
    
    block_y = something(block_y_override, min(Int(WARPSIZE), 4, choose_block_y(conf.threads)))
    block_y = min(block_y, max_block_y)
    
    return (block_y, maxregs_override)
end

function energy_launch_params(kernel)
    config = Molly.cuda_launch_config()
    block_y_override = prefer_override(config.energy_block_y, env_override("MOLLY_CUDA_ENERGY_BLOCK_Y"))
    block_y_override === nothing || validate_block_y("MOLLY_CUDA_ENERGY_BLOCK_Y", block_y_override)

    conf = launch_configuration(kernel.fun)
    # Ensure block_y does not exceed physical kernel limits
    max_threads = CUDA.maxthreads(kernel)
    max_block_y = fld(max_threads, Int(WARPSIZE))

    block_y = something(block_y_override, min(Int(WARPSIZE), 4, choose_block_y(conf.threads)))
    block_y = min(block_y, max_block_y)
    
    return block_y
end

function tile_launch_params(kernel)
    config = Molly.cuda_launch_config()
    config_tile_threads = config.tile_threads
    threads_x_override = config_tile_threads === nothing ? env_override("MOLLY_CUDA_TILE_THREADS_X") : config_tile_threads[1]
    threads_y_override = config_tile_threads === nothing ? env_override("MOLLY_CUDA_TILE_THREADS_Y") : config_tile_threads[2]
    if xor(threads_x_override === nothing, threads_y_override === nothing)
        error("Set both MOLLY_CUDA_TILE_THREADS_X and MOLLY_CUDA_TILE_THREADS_Y together")
    end

    max_threads = CUDA.maxthreads(kernel)

    if threads_x_override !== nothing
        threads_x_override > 0 || error("MOLLY_CUDA_TILE_THREADS_X must be positive, got $(threads_x_override)")
        threads_y_override > 0 || error("MOLLY_CUDA_TILE_THREADS_Y must be positive, got $(threads_y_override)")
        
        # Clamp overrides to physical limits. We clamp y to keep x warp-aligned if possible.
        actual_threads_x = min(threads_x_override, max_threads)
        actual_threads_y = min(threads_y_override, fld(max_threads, actual_threads_x))
        
        return (actual_threads_x, actual_threads_y)
    end

    conf = launch_configuration(kernel.fun)
    actual_threads = min(conf.threads, max_threads)
    return choose_tile_threads(actual_threads)
end

function reset_interacting_tile_state!(buffers)
     fill!(buffers.num_interacting_tiles, 0)
     fill!(buffers.interacting_tiles_overflow, 0)
     return nothing
end

function throw_if_interacting_tiles_overflowed(buffers)
    overflow_count = only(from_device(buffers.interacting_tiles_overflow))
    overflow_count == 0 && return nothing

    max_tiles = length(buffers.interacting_tiles_i)
    error("Maximum number of interacting tiles exceeded (> $(max_tiles)); increase buffer size.")
end

macro shfl_multiple_sync(mask, target, width, vars...)
    all_lines = map(vars) do v
        Expr(:(=), v,
            Expr(:call, :shfl_sync,
                mask, v, target, width
            )
        )
    end
    return esc(Expr(:block, all_lines...))
end

CUDA.shfl_recurse(op, x::Quantity) = op(x.val) * unit(x)
CUDA.shfl_recurse(op, x::SVector{1, C}) where C = SVector{1, C}(op(x[1]))
CUDA.shfl_recurse(op, x::SVector{2, C}) where C = SVector{2, C}(op(x[1]), op(x[2]))
CUDA.shfl_recurse(op, x::SVector{3, C}) where C = SVector{3, C}(op(x[1]), op(x[2]), op(x[3]))

function Molly.pairwise_forces_loop_gpu!(buffers, sys::System{D, <:CuArray}, pairwise_inters,
                            nbs::Molly.NoNeighborList, step_n) where D
    kernel = @cuda launch=false pairwise_force_kernel_nonl!(
            buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters, step_n,
            Val(D), Val(sys.force_units))
    conf = launch_configuration(kernel.fun)
    threads_basic = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
    nthreads = min(length(sys.atoms), threads_basic, conf.threads)
    nthreads = cld(nthreads, WARPSIZE) * WARPSIZE
    n_blocks_i = cld(length(sys.atoms), WARPSIZE)
    n_blocks_j = cld(length(sys.atoms), nthreads)
    kernel(buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters,
            step_n, Val(D), Val(sys.force_units); threads=nthreads,
            blocks=(n_blocks_i, n_blocks_j))
    return buffers
end

#=
    pairwise_forces_loop_gpu!(buffers, sys, pairwise_inters, nbs, needs_vir, step_n)

Maintainer entry point for the CUDA tiled pairwise force path.

Pipeline:
1. Rebuild the Morton ordering and compressed tile masks when the
   `GPUNeighborFinder` reorder cadence invalidates them.
2. Reorder coordinates, velocities, and atoms into Morton order.
3. Recompute the compact list of interacting 32x32 tiles when the cached tile
   list is stale for the current `dist_neighbors`.
4. Launch `force_kernel!` over that compact tile list, reverse the reorder, and
   surface overflow errors.

Cache contract:
- `buffers.step_n_preprocessed` gates reuse of reordered coordinates and tile
  search work within a simulation step.
- `buffers.last_r_cut` stores the neighbor cutoff used to build the current
  interacting-tile list.
- `buffers.num_pairs` is the host-side cached interacting-tile count used to
  size the force-kernel launch.
- `sys.neighbor_finder.initialized` only indicates whether the sparse exception
  masks are current. The interacting-tile list still depends on
  `n_steps_reorder` and `dist_neighbors`.
=#
function Molly.pairwise_forces_loop_gpu!(buffers, sys::System{D, <:CuArray, T}, pairwise_inters,
                         nbs::Nothing, ::Val{needs_vir}, step_n) where {D, T, needs_vir}
    N = length(sys.coords)
    n_blocks = cld(N, WARPSIZE)
    r_cut = sys.neighbor_finder.dist_cutoff
    r_neighbors = sys.neighbor_finder.dist_neighbors
    backend = get_backend(sys.coords)

    # Preprocessing Cache Check: Skip if step_n hasn't changed
    if step_n != buffers.step_n_preprocessed
        # Periodic Reordering and Bitmask Compression
        if step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized
            morton_bits = 10
            sides = box_sides(sys.boundary)
            w = sides ./ (2^morton_bits)
            sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
            sys.neighbor_finder.initialized = true
            compress_sparse!(buffers, sys.neighbor_finder, Val(N))
        end

        reorder_system_gpu!(buffers, sys)
        KernelAbstractions.synchronize(backend)

        # Find Interacting Tiles
        if step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized || ustrip(r_neighbors) != buffers.last_r_cut
            # Bounding Box Calculation for Tiles
            if sys.boundary isa TriclinicBoundary
                H = SMatrix{3, 3, T}(
                    sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
                    sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
                    sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val
                )
                Hinv = inv(H)
                @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
                        buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Hinv, Val(N),
                        sys.boundary, Val(D))
            else
                @cuda blocks=n_blocks threads=32 kernel_min_max!(
                        buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords,
                        Val(N), sys.boundary, Val(D))
            end

            max_tiles = length(buffers.interacting_tiles_i)
            reset_interacting_tile_state!(buffers)
            n_blocks_i = cld(N, WARPSIZE)
            tile_kernel = @cuda launch=false find_interacting_blocks_kernel!(
                buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
                buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
                buffers.box_mins, buffers.box_maxs, sys.boundary, r_neighbors,
                Val(n_blocks_i), Val(D), max_tiles,
                buffers.compressed_masks, buffers.tile_is_clean)
            tile_threads_xy = tile_launch_params(tile_kernel)

            tile_kernel(
                buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
                buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
                buffers.box_mins, buffers.box_maxs, sys.boundary, r_neighbors,
                Val(n_blocks_i), Val(D), max_tiles,
                
                buffers.compressed_masks, buffers.tile_is_clean;
                blocks=(cld(n_blocks_i, tile_threads_xy[1]), cld(n_blocks_i, tile_threads_xy[2])),
                threads=tile_threads_xy)
            
            buffers.last_r_cut = ustrip(r_neighbors)
            
            # Sync num_pairs back to CPU only when changed
            num_pairs_gpu = from_device(buffers.num_interacting_tiles)
            buffers.num_pairs = only(num_pairs_gpu)
        end
        # Update cache state
        buffers.step_n_preprocessed = step_n
    end
    
    # Execute Force Kernel over the list of interacting tiles
    auto_kernel = @cuda launch=false always_inline=true force_kernel!(
        buffers.fs_mat_reordered,
        buffers.virial_nounits,
        buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
        Val(N), r_cut, Val(sys.force_units), pairwise_inters,
        sys.boundary, step_n, buffers.compressed_masks,
        Val(needs_vir), Val(T), Val(D),
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow)
    block_y, maxregs = force_launch_params(auto_kernel)
    
    num_pairs = buffers.num_pairs
    n_blocks_launch = num_pairs > 0 ? cld(num_pairs, block_y) : 1

    kernel = if maxregs === nothing
        auto_kernel
    else
        @cuda launch=false maxregs=maxregs always_inline=true force_kernel!(
            buffers.fs_mat_reordered,
            buffers.virial_nounits,
            buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
            Val(N), r_cut, Val(sys.force_units), pairwise_inters,
            sys.boundary, step_n, buffers.compressed_masks,
            Val(needs_vir), Val(T), Val(D),
            buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
            buffers.num_interacting_tiles, buffers.interacting_tiles_overflow)
    end

    if num_pairs > 0
        kernel(
            buffers.fs_mat_reordered,
            buffers.virial_nounits,
            buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
            Val(N), r_cut, Val(sys.force_units), pairwise_inters,
            sys.boundary, step_n, buffers.compressed_masks,
            Val(needs_vir), Val(T), Val(D),
            buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
            buffers.num_interacting_tiles, buffers.interacting_tiles_overflow;
            threads=(32, block_y), blocks=n_blocks_launch
        )
    end

    reverse_reorder_forces_gpu!(buffers, sys)
    KernelAbstractions.synchronize(backend)
    throw_if_interacting_tiles_overflowed(buffers)

    return buffers
end

#=
    pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys, pairwise_inters, nbs, step_n)

Maintainer entry point for the CUDA tiled pairwise energy path.

This follows the same Morton reorder -> sparse mask compression -> tile search
pipeline as `pairwise_forces_loop_gpu!`, but launches `energy_kernel!` instead
of the force kernel. The key difference is that energy evaluation reuses any
preprocessing already performed for the current step so forces and energies can
share the same cached tile metadata.
=#
function Molly.pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys::System{D, <:CuArray, T},
                                      pairwise_inters, nbs::Nothing,
                                      step_n) where {D, T}
    # The ordering is usually recomputed for potential energy, but we can reuse it
    #   if it was already computed for this step
    N = length(sys.coords)
    n_blocks = cld(N, WARPSIZE)
    r_cut = sys.neighbor_finder.dist_cutoff
    r_neighbors = sys.neighbor_finder.dist_neighbors
    backend = get_backend(sys.coords)

    if step_n != buffers.step_n_preprocessed
        if step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized
            morton_bits = 10
            sides = box_sides(sys.boundary)
            w = sides ./ (2^morton_bits)
            sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
            sys.neighbor_finder.initialized = true
            compress_sparse!(buffers, sys.neighbor_finder, Val(N))
        end
        reorder_system_gpu!(buffers, sys)
        KernelAbstractions.synchronize(backend)

        if step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized || ustrip(r_neighbors) != buffers.last_r_cut
            if sys.boundary isa TriclinicBoundary
                H = SMatrix{3, 3, T}(
                    sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
                    sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
                    sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val
                )
                Hinv = inv(H)
                @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
                        buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Hinv, Val(N),
                        sys.boundary, Val(D))
            else
                @cuda blocks=cld(N, WARPSIZE) threads=32 kernel_min_max!(
                        buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords,
                        Val(N), sys.boundary, Val(D))
            end

            max_tiles = length(buffers.interacting_tiles_i)
            reset_interacting_tile_state!(buffers)
            tile_kernel = @cuda launch=false find_interacting_blocks_kernel!(
                buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
                buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
                buffers.box_mins, buffers.box_maxs, sys.boundary, r_neighbors,
                Val(n_blocks), Val(D), max_tiles,
                buffers.compressed_masks, buffers.tile_is_clean)
            tile_threads_xy = tile_launch_params(tile_kernel)
            tile_kernel(
                buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
                buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
                buffers.box_mins, buffers.box_maxs, sys.boundary, r_neighbors,
                Val(n_blocks), Val(D), max_tiles,
                buffers.compressed_masks, buffers.tile_is_clean;
                blocks=(cld(n_blocks, tile_threads_xy[1]), cld(n_blocks, tile_threads_xy[2])),
                threads=tile_threads_xy)
            
            buffers.last_r_cut = ustrip(r_neighbors)
            
            # Sync num_pairs back to CPU only when changed
            num_pairs_gpu = from_device(buffers.num_interacting_tiles)
            buffers.num_pairs = only(num_pairs_gpu)
        end
        buffers.step_n_preprocessed = step_n
    end

    kernel = @cuda launch=false always_inline=true energy_kernel!(
            pe_vec_nounits, buffers.coords_reordered,
            buffers.velocities_reordered, buffers.atoms_reordered, Val(N), r_cut, Val(sys.energy_units), pairwise_inters,
            sys.boundary, step_n, buffers.compressed_masks,
            Val(T), Val(D), buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.num_interacting_tiles, buffers.interacting_tiles_overflow)
    block_y = energy_launch_params(kernel)
    
    num_pairs = buffers.num_pairs
    n_blocks_launch = num_pairs > 0 ? cld(num_pairs, block_y) : 1
    
    if num_pairs > 0
        kernel(
                pe_vec_nounits, buffers.coords_reordered,
                buffers.velocities_reordered, buffers.atoms_reordered, Val(N), r_cut, Val(sys.energy_units), pairwise_inters,
                sys.boundary, step_n, buffers.compressed_masks,
                Val(T), Val(D), buffers.interacting_tiles_i, buffers.interacting_tiles_j,
                buffers.interacting_tiles_type, buffers.num_interacting_tiles, buffers.interacting_tiles_overflow;
                blocks=n_blocks_launch, threads=(32, block_y))
    end
     throw_if_interacting_tiles_overflowed(buffers)
     return pe_vec_nounits
 end

function boxes_dist(r1_min::SVector{D, T}, r1_max::SVector{D, T}, r2_min::SVector{D, T}, r2_max::SVector{D, T}, boundary) where {D, T}
    va = vector(r2_max, r1_min, boundary)
    vb = vector(r1_max, r2_min, boundary)
    a = SVector{D}(ntuple(d -> abs(va[d]), D))
    b = SVector{D}(ntuple(d -> abs(vb[d]), D))

    return SVector(ntuple(d -> r1_min[d] - r2_max[d] <= zero(T) && r2_min[d] - r1_max[d] <= zero(T) ? zero(T) : ifelse(a[d] < b[d], a[d], b[d]), D))
end

# Triclinic boxes case is treated only in 3 dimensions
function boxes_dist(r1_min::SVector{D, T}, r1_max::SVector{D, T}, r2_min::SVector{D, T}, r2_max::SVector{D, T}, boundary::TriclinicBoundary) where {D, T}
    r3 = r2_max - r1_min
    r2 = r3 - boundary.basis_vectors[3] .* round(r3[3] / boundary.basis_vectors[3][3])
    r1 = r2 - boundary.basis_vectors[2] .* round(r2[2] / boundary.basis_vectors[2][2])
    r_a = boundary.basis_vectors[1] .* round(r1[1] / boundary.basis_vectors[1][1])
    a = SVector(abs(r_a[1]), abs(r_a[2]), abs(r_a[3]))
    r3 = r1_max - r2_min
    r2 = r3 - boundary.basis_vectors[3] .* round(r3[3] / boundary.basis_vectors[3][3])
    r1 = r2 - boundary.basis_vectors[2] .* round(r2[2] / boundary.basis_vectors[2][2])
    r_b = boundary.basis_vectors[1] .* round(r1[1] / boundary.basis_vectors[1][1])
    b = SVector(abs(r_b[1]), abs(r_b[2]), abs(r_b[3]))

    return SVector(
        r_a[1] >= zero(T) && r_b[1] >= zero(T) ? zero(T) : ifelse(a[1] < b[1], a[1], b[1]),
        r_a[2] >= zero(T) && r_b[2] >= zero(T) ? zero(T) : ifelse(a[2] < b[2], a[2], b[2]),
        r_a[3] >= zero(T) && r_b[3] >= zero(T) ? zero(T) : ifelse(a[3] < b[3], a[3], b[3])
    )
end

"""
    reorder_system_gpu!(buffers, sys)

Reorders `coords`, `velocities`, and `atoms` into `_reordered` buffers 
according to the Morton sequence to improve spatial cache locality.
"""
function reorder_system_gpu!(buffers, sys)
    N = length(sys)
    backend = get_backend(sys.coords)
    n_threads = 256
    
    # We use KernelAbstractions kernels from Molly
    Molly.reorder_kernel!(backend, n_threads)(buffers.coords_reordered, sys.coords, buffers.morton_seq, ndrange=N)
    Molly.reorder_kernel!(backend, n_threads)(buffers.velocities_reordered, sys.velocities, buffers.morton_seq, ndrange=N)
    Molly.reorder_kernel!(backend, n_threads)(buffers.atoms_reordered, sys.atoms, buffers.morton_seq, ndrange=N)
    
    return nothing
end

"""
    reverse_reorder_forces_gpu!(buffers, sys)

Maps forces from the `fs_mat_reordered` buffer back to the original atom indices 
in `fs_mat`.
"""
function reverse_reorder_forces_gpu!(buffers, sys::System{D, <:CuArray, T}) where {D, T}
    N = length(sys)
    backend = get_backend(sys.coords)
    n_threads = 256
    
    # fs_mat is D x N. We need to reverse reorder each dimension or use a specialized kernel.
    # Let's use a specialized kernel for D x N matrix reverse reordering.
    Molly.reverse_reorder_forces_kernel!(backend, n_threads)(buffers.fs_mat, buffers.fs_mat_reordered, buffers.morton_seq, ndrange=N)
    
    return nothing
end

#=
    kernel_min_max!(...)

Computes the minimum and maximum coordinates for each 32-atom block.
Used for tile-level bounding box intersection tests during tile finding.
=#
function kernel_min_max!(
    sorted_seq,
    mins::AbstractArray{C},
    maxs::AbstractArray{C},
    coords_var,
    ::Val{n},
    boundary,
    ::Val{D}) where {n, C, D}

    D32 = Int32(32)
    a = Int32(1)
    b = Int32(D)
    r = Int32(n % D32)
    i = threadIdx().x + (blockIdx().x - a) * blockDim().x
    local_i = threadIdx().x
    sorted_seq_ro = CUDA.Const(sorted_seq)
    coords = CUDA.Const(coords_var)
    mins_smem = CuStaticSharedArray(C, (D32, b))
    maxs_smem = CuStaticSharedArray(C, (D32, b))
    r_smem = CuStaticSharedArray(C, (r, b))

    if i <= n - r && local_i <= D32
        @inbounds s_i = sorted_seq_ro[i]
        for k in a:b
            @inbounds val = coords[s_i][k]
            @inbounds mins_smem[local_i, k] = val
            @inbounds maxs_smem[local_i, k] = val
        end
    end
    sync_threads()
    if i <= n - r && local_i <= D32
        for p in a:Int32(log2(D32))
            for k in a:b
                @inbounds begin
                    if local_i % Int32(2^p) == Int32(0)
                        if mins_smem[local_i, k] > mins_smem[local_i - Int32(2^(p - 1)), k]
                            mins_smem[local_i, k] = mins_smem[local_i - Int32(2^(p - 1)), k]
                        end
                        if maxs_smem[local_i, k] < maxs_smem[local_i - Int32(2^(p - 1)), k]
                            maxs_smem[local_i, k] = maxs_smem[local_i - Int32(2^(p - 1)), k]
                        end
                    end
                end
            end
        end
        if local_i == D32
            @inbounds for k in a:b
                mins[blockIdx().x, k] = mins_smem[local_i, k]
                maxs[blockIdx().x, k] = maxs_smem[local_i, k]
            end
        end

    end

    # Since the remainder array is low-dimensional, we do the scan
    if i > n - r && i <= n && local_i <= r
        @inbounds s_i = sorted_seq_ro[i]
        for k in a:b
            @inbounds r_smem[local_i, k] = coords[s_i][k]
        end
    end
    xyz_min = CuStaticSharedArray(C, b)
    xyz_max = CuStaticSharedArray(C, b)
    @inbounds for k in a:b
        xyz_min[k] =  10 * box_sides(boundary, k) # very large (arbitrary) value
        xyz_max[k] = -10 * box_sides(boundary, k)
    end
    if local_i == a
        for j in a:r
            @inbounds begin
                for k in a:b
                    if r_smem[j, k] < xyz_min[k]
                        xyz_min[k] = r_smem[j, k]
                    end
                    if r_smem[j, k] > xyz_max[k]
                        xyz_max[k] = r_smem[j, k]
                    end
                end
            end
        end
        if blockIdx().x == ceil(Int32, n/D32) && r != Int32(0)
            @inbounds for k in a:b
                mins[blockIdx().x, k] = xyz_min[k]
                maxs[blockIdx().x, k] = xyz_max[k]
            end
        end
    end

    return nothing
end

function kernel_min_max_triclinic!(
    sorted_seq,
    mins::AbstractArray{C},
    maxs::AbstractArray{C},
    coords_var,
    Hinv,
    ::Val{n},
    boundary,
    ::Val{D}) where {n, C, D}

    D32 = Int32(32)
    a = Int32(1)
    b = Int32(D)
    r = Int32(n % D32)
    i = threadIdx().x + (blockIdx().x - a) * blockDim().x
    local_i = threadIdx().x
    sorted_seq_ro = CUDA.Const(sorted_seq)
    coords = CUDA.Const(coords_var)
    mins_smem = CuStaticSharedArray(C, (D32, b))
    maxs_smem = CuStaticSharedArray(C, (D32, b))
    r_smem = CuStaticSharedArray(C, (r, b))

    if i <= n - r && local_i <= D32
        @inbounds s_i = sorted_seq_ro[i]
        @inbounds r_i = coords[s_i]
        @inbounds for k in a:b
            val = zero(C)
            for j in a:b
                @fastmath val += Hinv[k,j]*r_i[j]
            end
            @inbounds mins_smem[local_i, k] = val
            @inbounds maxs_smem[local_i, k] = val
        end
    end
    sync_threads()
    if i <= n - r && local_i <= D32
        for p in a:Int32(log2(D32))
            for k in a:b
                @inbounds begin
                    if local_i % Int32(2^p) == Int32(0)
                        if mins_smem[local_i, k] > mins_smem[local_i - Int32(2^(p - 1)), k]
                            mins_smem[local_i, k] = mins_smem[local_i - Int32(2^(p - 1)), k]
                        end
                        if maxs_smem[local_i, k] < maxs_smem[local_i - Int32(2^(p - 1)), k]
                            maxs_smem[local_i, k] = maxs_smem[local_i - Int32(2^(p - 1)), k]
                        end
                    end
                end
            end
        end
        if local_i == D32
            @inbounds for k in a:b
                mins[blockIdx().x, k] = mins_smem[local_i, k]
                maxs[blockIdx().x, k] = maxs_smem[local_i, k]
            end
        end

    end

    # Since the remainder array is low-dimensional, we do the scan
    if i > n - r && i <= n && local_i <= r
        @inbounds s_i = sorted_seq_ro[i]
        @inbounds r_i = coords[s_i]
        for k in a:b
            val = zero(C)
            @inbounds for j in a:b
                @fastmath val += Hinv[k,j]*r_i[j]
            end
            @inbounds r_smem[local_i, k] = val # Transform to fractional space: s = Hinv * r
        end
    end
    xyz_min = CuStaticSharedArray(C, b)
    xyz_max = CuStaticSharedArray(C, b)
    @inbounds for k in a:b
        xyz_min[k] =  10 * box_sides(boundary, k) # Very large (arbitrary) value
        xyz_max[k] = -10 * box_sides(boundary, k)
    end
    if local_i == a
        for j in a:r
            @inbounds begin
                for k in a:b
                    if r_smem[j, k] < xyz_min[k]
                        xyz_min[k] = r_smem[j, k]
                    end
                    if r_smem[j, k] > xyz_max[k]
                        xyz_max[k] = r_smem[j, k]
                    end
                end
            end
        end
        if blockIdx().x == ceil(Int32, n/D32) && r != Int32(0)
            @inbounds for k in a:b
                mins[blockIdx().x, k] = xyz_min[k]
                maxs[blockIdx().x, k] = xyz_max[k]
            end
        end
    end

    return nothing
end

function update_inv_morton_kernel!(inv_morton_seq, morton_seq, ::Val{N}) where N
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    morton_seq_ro = CUDA.Const(morton_seq)
    if i <= N
        @inbounds inv_morton_seq[morton_seq_ro[i]] = i
    end
    return nothing
end

function init_compressed_masks_kernel!(compressed_masks, tile_is_clean, ::Val{N}, ::Val{n_upper_tiles}) where {N, n_upper_tiles}
    lane = laneid()
    n_blocks = ceil(Int32, N / 32)
    tile_idx = blockIdx().x

    if tile_idx > n_upper_tiles
        return nothing
    end

    # Binary search for i such that S(i) <= tile_idx < S(i+1)
    # S(i) = ((i - 1) * (2 * n_blocks - i + 2)) / 2 + 1
    low = Int32(1)
    high = Int32(n_blocks)
    i = Int32(1)
    while low <= high
        mid = (low + high) ÷ Int32(2)
        start_idx_mid = ((Int64(mid) - 1) * (2 * Int64(n_blocks) - Int64(mid) + 2)) ÷ 2 + 1
        if start_idx_mid <= tile_idx
            i = mid
            low = mid + Int32(1)
        else
            high = mid - Int32(1)
        end
    end
    
    start_idx_i = ((Int64(i) - 1) * (2 * Int64(n_blocks) - Int64(i) + 2)) ÷ 2 + 1
    j = i + Int32(tile_idx - start_idx_i)

    r = Int32((N - 1) % 32 + 1)

    eligible_bitmask = UInt32(0xFFFFFFFF)
    special_bitmask = UInt32(0x00000000)

    # Boundary Masking
    if j == n_blocks
        mask = ifelse(r == Int32(32), UInt32(0xFFFFFFFF), (UInt32(1) << r) - UInt32(1))
        eligible_bitmask = (mask << (Int32(32) - r))
    end

    # Diagonal Self-Interactions
    if i == j
        eligible_bitmask &= ~(UInt32(1) << (Int32(32) - lane))
    end

    @inbounds compressed_masks[lane, 1, tile_idx] = eligible_bitmask
    @inbounds compressed_masks[lane, 2, tile_idx] = special_bitmask

    if lane == 1
        @inbounds tile_is_clean[tile_idx] = (i < j) && (j < n_blocks)
    end

    return nothing
end

function apply_sparse_exceptions_kernel!(
    excluded_i, excluded_j, special_i, special_j,
    inv_morton_seq, compressed_masks, tile_is_clean,
    ::Val{n_blocks}, ::Val{n_excluded}, ::Val{n_special}
) where {n_blocks, n_excluded, n_special}
    excluded_i_ro = CUDA.Const(excluded_i)
    excluded_j_ro = CUDA.Const(excluded_j)
    special_i_ro = CUDA.Const(special_i)
    special_j_ro = CUDA.Const(special_j)
    inv_morton_seq_ro = CUDA.Const(inv_morton_seq)

    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    
    if idx <= n_excluded
        @inbounds orig_i = excluded_i_ro[idx]
        @inbounds orig_j = excluded_j_ro[idx]
        
        @inbounds p_i = inv_morton_seq_ro[orig_i]
        @inbounds p_j = inv_morton_seq_ro[orig_j]
        
        if p_i > p_j
            p_i, p_j = p_j, p_i
        end
        
        t_i = (p_i - Int32(1)) ÷ Int32(32) + Int32(1)
        t_j = (p_j - Int32(1)) ÷ Int32(32) + Int32(1)
        tile_idx = upper_tile_index(t_i, t_j, Int32(n_blocks))
        
        lane_i = (p_i - Int32(1)) % Int32(32) + Int32(1)
        lane_j = (p_j - Int32(1)) % Int32(32) + Int32(1)
        
        target_bit = UInt32(1) << (Int32(32) - lane_j)
        
        # For exclusions, mask_type = 1
        linear_idx = Int64(lane_i) + Int64(64) * (Int64(tile_idx) - Int64(1))
        
        CUDA.atomic_and!(pointer(compressed_masks, linear_idx), ~target_bit)
        @inbounds tile_is_clean[tile_idx] = false
    end
    
    if idx <= n_special
        @inbounds orig_i = special_i_ro[idx]
        @inbounds orig_j = special_j_ro[idx]
        
        @inbounds p_i = inv_morton_seq_ro[orig_i]
        @inbounds p_j = inv_morton_seq_ro[orig_j]
        
        if p_i > p_j
            p_i, p_j = p_j, p_i
        end
        
        t_i = (p_i - Int32(1)) ÷ Int32(32) + Int32(1)
        t_j = (p_j - Int32(1)) ÷ Int32(32) + Int32(1)
        tile_idx = upper_tile_index(t_i, t_j, Int32(n_blocks))
        
        lane_i = (p_i - Int32(1)) % Int32(32) + Int32(1)
        lane_j = (p_j - Int32(1)) % Int32(32) + Int32(1)
        
        target_bit = UInt32(1) << (Int32(32) - lane_j)
        
        # For special, mask_type = 2
        linear_idx = Int64(lane_i) + Int64(32) + Int64(64) * (Int64(tile_idx) - Int64(1))
        
        CUDA.atomic_or!(pointer(compressed_masks, linear_idx), target_bit)
        @inbounds tile_is_clean[tile_idx] = false
    end
    
    return nothing
end

#=
    compress_sparse!(buffers, nf, ::Val{N})

Convert the sparse exception pairs stored on `GPUNeighborFinder` into the
Morton-ordered tile masks consumed by the tiled CUDA pairwise kernels.

Stages:
1. Build an inverse Morton map from original atom indices into reordered atom
   indices.
2. Optimistically initialize every upper-triangular tile as fully eligible and
   CLEAN.
3. Scatter excluded and special pairs into `buffers.compressed_masks`, clearing
   `tile_is_clean` for any tile that requires per-pair mask lookups.

The sparse exception vectors remain in original atom indexing; this routine is
what realigns them with the current reordered system state.
=#
function compress_sparse!(buffers, nf::GPUNeighborFinder, ::Val{N}) where N
    n_blocks = ceil(Int32, N / 32)
    n_upper_tiles = upper_tile_count(n_blocks)
    
    # Stage A: Inverse Morton Mapping
    @cuda threads=256 blocks=cld(N, 256) update_inv_morton_kernel!(
        buffers.morton_seq_inv, buffers.morton_seq, Val(N))
    
    # Stage B: Optimistic Initialization
    @cuda blocks=n_upper_tiles threads=32 init_compressed_masks_kernel!(
        buffers.compressed_masks, buffers.tile_is_clean, Val(N), Val(n_upper_tiles))
    
    # Stage C: Atomic Scatter
    n_exc = length(nf.excluded_i)
    n_spec = length(nf.special_i)
    n_max = max(n_exc, n_spec)
    if n_max > 0
        @cuda threads=256 blocks=cld(Int32(n_max), Int32(256)) apply_sparse_exceptions_kernel!(
            nf.excluded_i, nf.excluded_j, nf.special_i, nf.special_j,
            buffers.morton_seq_inv, buffers.compressed_masks, buffers.tile_is_clean,
            Val(n_blocks), Val(Int32(n_exc)), Val(Int32(n_spec)))
    end
end

#=
**The No-neighborlist pairwise force summation kernel (algorithm by Eastman, see https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.21413)**:
1. Case j < n_blocks && i < j, i.e., `WARPSIZE`×`WARPSIZE` tiles: For such tiles each row is assiged to a different thread in a warp which calculates the
forces for the entire row in `WARPSIZE` steps. This is done such that some data can be shuffled from `i+1`'th thread to `i`'th thread in each
subsequent iteration of the force calculation in a row. If `a, b, ...` are different atoms and `1, 2, ...` are order in which each thread calculates
the interatomic forces, then we can represent this scenario as (considering `WARPSIZE=8`):
```
    × | i j k l m n o p
    --------------------
    a | 1 2 3 4 5 6 7 8
    b | 8 1 2 3 4 5 6 7
    c | 7 8 1 2 3 4 5 6
    d | 6 7 8 1 2 3 4 5
    e | 5 6 7 8 1 2 3 4
    f | 4 5 6 7 8 1 2 3
    g | 3 4 5 6 7 8 1 2
    h | 2 3 4 5 6 7 8 1
```

2. Cases j == n_blocks && i < n_blocks, i == j && i < n_blocks, i == n_blocks && j == n_blocks: In such cases, it is not possible to shuffle data generally
so there is no need to order calculations for each thread diagonally and it is also a bit more complicated to do so.
That's why the calculations are done in the following order:
```
    × | i j k l m n
    ----------------
    a | 1 2 3 4 5 6
    b | 1 2 3 4 5 6
    c | 1 2 3 4 5 6
    d | 1 2 3 4 5 6
    e | 1 2 3 4 5 6
    f | 1 2 3 4 5 6
    g | 1 2 3 4 5 6
    h | 1 2 3 4 5 6
```
=#


#=
    find_interacting_blocks_kernel!(...)

Scan the upper-triangular matrix of 32x32 Morton-ordered atom tiles and append
only those whose bounding boxes fall within `r_cut`.

For each interacting tile this kernel:
1. Appends the tile coordinates `(i, j)` to the compact 1D tile list.
2. Tags the tile as CLEAN (`0`) or mask-backed (`1`) using `tile_is_clean`.
3. Sets `interacting_tiles_overflow` if the compact list would exceed the
   preallocated capacity.

The downstream force and energy kernels consume this compact list directly, so
this kernel defines both the tile-processing order and the launch size for the
pairwise work on the current step.
=#
function find_interacting_blocks_kernel!(
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
    interacting_tiles_overflow,
    mins::AbstractArray{C}, maxs::AbstractArray{C}, boundary, r_cut, ::Val{N_blocks}, ::Val{D}, max_total_tiles,
    compressed_masks, tile_is_clean
) where {C, N_blocks, D}
    mins_ro = CUDA.Const(mins)
    maxs_ro = CUDA.Const(maxs)
    tile_is_clean_ro = CUDA.Const(tile_is_clean)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= N_blocks && j <= N_blocks && i <= j
        @inbounds r_min_i = SVector{D}(ntuple(d -> mins_ro[i, d], D))
        @inbounds r_max_i = SVector{D}(ntuple(d -> maxs_ro[i, d], D))
        @inbounds r_min_j = SVector{D}(ntuple(d -> mins_ro[j, d], D))
        @inbounds r_max_j = SVector{D}(ntuple(d -> maxs_ro[j, d], D))

        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)
        
        if sum(d_block .* d_block) <= r_cut * r_cut
            is_clean = (i < j) && (j < N_blocks)
            if is_clean
                mask_idx = upper_tile_index(Int32(i), Int32(j), Int32(N_blocks))
                @inbounds is_clean = tile_is_clean_ro[mask_idx]
            end

            idx = CUDA.atomic_add!(pointer(num_interacting_tiles, 1), Int32(1)) + Int32(1)
            if idx <= max_total_tiles
                @inbounds interacting_tiles_i[idx] = Int32(i)
                @inbounds interacting_tiles_j[idx] = Int32(j)
                @inbounds interacting_tiles_type[idx] = is_clean ? UInt8(0) : UInt8(1)
            else
                CUDA.atomic_add!(pointer(interacting_tiles_overflow, 1), Int32(1))
            end
        end
    end
    return nothing
end

#=
    force_kernel!(...)

Compute pairwise forces for the compact list of interacting 32x32 tiles produced
by `find_interacting_blocks_kernel!`.

Execution model:
- `threadIdx().x` spans one warp lane within a tile row.
- `threadIdx().y` selects one tile from the compact list, so a single block can
  process multiple listed tiles at once.
- The kernel keeps the `i`-atom contribution in registers/shared memory and
  atomically scatters the opposite contribution for the `j` atoms.

Tile cases:
1. Full off-diagonal tiles.
2. Boundary-column tiles containing the final partial atom block.
3. Diagonal tiles, where only unique pairs are evaluated.
4. The terminal corner tile, where both axes are partial.

CLEAN tiles skip bitmask loads entirely; mask-backed tiles consult
`compressed_masks` to apply exclusions and special-pair handling.
=#
function force_kernel!(
    fs_mat,
    global_virial,
    coords_var,
    velocities_var,
    atoms_var::AbstractArray{A},
    ::Val{N},
    r_cut,
    ::Val{force_units},
    inters_tuple,
    boundary,
    step_n,
    compressed_masks,
    ::Val{needs_vir},
    ::Val{T},
    ::Val{D},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
    interacting_tiles_overflow) where {N, A, force_units, needs_vir, T, D}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    compressed_masks_ro = CUDA.Const(compressed_masks)
    tiles_i_ro = CUDA.Const(interacting_tiles_i)
    tiles_j_ro = CUDA.Const(interacting_tiles_j)
    tiles_type_ro = CUDA.Const(interacting_tiles_type)
    num_interacting_tiles_ro = CUDA.Const(num_interacting_tiles)
    interacting_tiles_overflow_ro = CUDA.Const(interacting_tiles_overflow)
    
    idx = (blockIdx().x - a) * blockDim().y + threadIdx().y

    @inbounds if interacting_tiles_overflow_ro[1] != 0
        return nothing
    end
    
    @inbounds num_pairs = num_interacting_tiles_ro[1]
    if idx > num_pairs
        return nothing
    end

    lane = threadIdx().x
    warpid = threadIdx().y

    @inbounds i = tiles_i_ro[idx]
    @inbounds j = tiles_j_ro[idx]
    @inbounds type = tiles_type_ro[idx]

    i_0_tile = (i - a) * warpsize()
    index_i = i_0_tile + lane

    opposites_sum = CuStaticSharedArray(T, (32, D, MAX_BLOCK_Y))

    r = Int32((N - 1) % 32 + 1)
    
    force_i_x = zero(T)
    force_i_y = zero(T)
    force_i_z = zero(T)

    @inbounds for k in a:b
        opposites_sum[lane, k, warpid] = zero(T)
    end

    vir_xx = zero(T); vir_yy = zero(T); vir_zz = zero(T)
    vir_xy = zero(T); vir_xz = zero(T); vir_yz = zero(T)

    mask_idx = upper_tile_index(i, j, n_blocks)

    j_0_tile = (j - a) * warpsize()
    index_j = j_0_tile + lane

    # Part 1: Standard non-diagonal tiles
    if j < n_blocks && i < j
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        
        @inbounds coords_j = coords[index_j]
        @inbounds vel_j = velocities[index_j]
        shuffle_idx = lane
        @inbounds atoms_j = atoms[index_j]
        atom_fields = getfield.((atoms_j,), fieldnames(A))
        
        if type == UInt8(0) # CLEAN
            @inbounds for m in a:warpsize()
                sync_warp()
                coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, lane + a, warpsize())
                vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, lane + a, warpsize())
                shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, lane + a, warpsize())
                atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, lane + a, warpsize())
                atoms_j_shuffle = A(atom_fields...)

                dr = vector(coords_i, coords_j, boundary)
                r2 = @fastmath sum(abs2, dr)
                condition = r2 <= r_cut * r_cut
                any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)
                
                if any_active
                    f = condition ? sum_pairwise_forces(
                        inters_tuple, atoms_i, atoms_j_shuffle, Val(force_units),
                        false, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                    ) : zero(SVector{D, T})

                    @fastmath force_i_x += ustrip(f[1])
                    @fastmath opposites_sum[shuffle_idx, 1, warpid] -= ustrip(f[1])
                    if D >= 2
                        @fastmath force_i_y += ustrip(f[2])
                        @fastmath opposites_sum[shuffle_idx, 2, warpid] -= ustrip(f[2])
                    end
                    if D >= 3
                        @fastmath force_i_z += ustrip(f[3])
                        @fastmath opposites_sum[shuffle_idx, 3, warpid] -= ustrip(f[3])
                    end

                    if needs_vir
                        @fastmath vir_xx += ustrip(f[1]) * ustrip(dr[1])
                        if D >= 2
                            @fastmath vir_yy += ustrip(f[2]) * ustrip(dr[2])
                            @fastmath vir_xy += ustrip(f[1]) * ustrip(dr[2])
                        end
                        if D >= 3
                            @fastmath vir_zz += ustrip(f[3]) * ustrip(dr[3])
                            @fastmath vir_xz += ustrip(f[1]) * ustrip(dr[3])
                            @fastmath vir_yz += ustrip(f[2]) * ustrip(dr[3])
                        end
                    end
                end
            end
        else # EXCLUDED
            @inbounds eligible_bitmask = compressed_masks_ro[lane, 1, mask_idx]
            @inbounds special_bitmask = compressed_masks_ro[lane, 2, mask_idx]

            @inbounds for m in a:warpsize()
                sync_warp()
                coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, lane + a, warpsize())
                vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, lane + a, warpsize())
                shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, lane + a, warpsize())
                atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, lane + a, warpsize())
                atoms_j_shuffle = A(atom_fields...)

                dr = vector(coords_i, coords_j, boundary)
                r2 = @fastmath sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - shuffle_idx)) | (eligible_bitmask << shuffle_idx)
                spec = (special_bitmask >> (warpsize() - shuffle_idx)) | (special_bitmask << shuffle_idx)
                
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
                any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)
                
                if any_active
                    f = condition ? sum_pairwise_forces(
                        inters_tuple, atoms_i, atoms_j_shuffle, Val(force_units),
                        (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                    ) : zero(SVector{D, T})

                    @fastmath force_i_x += ustrip(f[1])
                    @fastmath opposites_sum[shuffle_idx, 1, warpid] -= ustrip(f[1])
                    if D >= 2
                        @fastmath force_i_y += ustrip(f[2])
                        @fastmath opposites_sum[shuffle_idx, 2, warpid] -= ustrip(f[2])
                    end
                    if D >= 3
                        @fastmath force_i_z += ustrip(f[3])
                        @fastmath opposites_sum[shuffle_idx, 3, warpid] -= ustrip(f[3])
                    end

                    if needs_vir
                        @fastmath vir_xx += ustrip(f[1]) * ustrip(dr[1])
                        if D >= 2
                            @fastmath vir_yy += ustrip(f[2]) * ustrip(dr[2])
                            @fastmath vir_xy += ustrip(f[1]) * ustrip(dr[2])
                        end
                        if D >= 3
                            @fastmath vir_zz += ustrip(f[3]) * ustrip(dr[3])
                            @fastmath vir_xz += ustrip(f[1]) * ustrip(dr[3])
                            @fastmath vir_yz += ustrip(f[2]) * ustrip(dr[3])
                        end
                    end
                end
            end
        end

        sync_warp()
        if index_j <= N
            @inbounds for k in a:b
                if opposites_sum[lane, k, warpid] != zero(T)
                    CUDA.atomic_add!(pointer(fs_mat, Int64(index_j) * b - (b - k)), -opposites_sum[lane, k, warpid])
                end
                opposites_sum[lane, k, warpid] = zero(T)
            end
        end
    end

    # Part 2: Boundary column tiles
    if j == n_blocks && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        
        @inbounds eligible_bitmask = compressed_masks_ro[lane, 1, mask_idx]
        @inbounds special_bitmask = compressed_masks_ro[lane, 2, mask_idx]

        @inbounds for m in a:r
            idx_j = j_0_tile + m
            @inbounds coords_j = coords[idx_j]
            @inbounds vel_j = velocities[idx_j]
            @inbounds atoms_j = atoms[idx_j]
            
            dr = vector(coords_i, coords_j, boundary)
            r2 = @fastmath sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            
            condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
            any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)

            if any_active
                f = condition ? sum_pairwise_forces(
                    inters_tuple, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : zero(SVector{D, T})

                @fastmath force_i_x += ustrip(f[1])
                if ustrip(f[1]) != zero(T)
                    CUDA.atomic_add!(pointer(fs_mat, Int64(idx_j) * b - (b - 1)), ustrip(f[1]))
                end
                if D >= 2
                    @fastmath force_i_y += ustrip(f[2])
                    if ustrip(f[2]) != zero(T)
                        CUDA.atomic_add!(pointer(fs_mat, Int64(idx_j) * b - (b - 2)), ustrip(f[2]))
                    end
                end
                if D >= 3
                    @fastmath force_i_z += ustrip(f[3])
                    if ustrip(f[3]) != zero(T)
                        CUDA.atomic_add!(pointer(fs_mat, Int64(idx_j) * b - (b - 3)), ustrip(f[3]))
                    end
                end
                
                if needs_vir
                    @fastmath vir_xx += ustrip(f[1]) * ustrip(dr[1])
                    if D >= 2
                        @fastmath vir_yy += ustrip(f[2]) * ustrip(dr[2])
                        @fastmath vir_xy += ustrip(f[1]) * ustrip(dr[2])
                    end
                    if D >= 3
                        @fastmath vir_zz += ustrip(f[3]) * ustrip(dr[3])
                        @fastmath vir_xz += ustrip(f[1]) * ustrip(dr[3])
                        @fastmath vir_yz += ustrip(f[2]) * ustrip(dr[3])
                    end
                end
            end
        end
    end

    # Part 3: Diagonal tiles
    if i == j && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        
        @inbounds eligible_bitmask = compressed_masks_ro[lane, 1, mask_idx]
        @inbounds special_bitmask = compressed_masks_ro[lane, 2, mask_idx]

        @inbounds for m in (lane + a) : warpsize()
            idx_j = j_0_tile + m
            @inbounds coords_j = coords[idx_j]
            @inbounds vel_j = velocities[idx_j]
            @inbounds atoms_j = atoms[idx_j]
            
            dr = vector(coords_i, coords_j, boundary)
            r2 = @fastmath sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
            
            # Divergence-safe execution (no vote_any_sync)
            f = condition ? sum_pairwise_forces(
                inters_tuple, atoms_i, atoms_j, Val(force_units),
                (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
            ) : zero(SVector{D, T})

            @fastmath force_i_x += ustrip(f[1])
            @fastmath opposites_sum[m, 1, warpid] -= ustrip(f[1])
            if D >= 2
                @fastmath force_i_y += ustrip(f[2])
                @fastmath opposites_sum[m, 2, warpid] -= ustrip(f[2])
            end
            if D >= 3
                @fastmath force_i_z += ustrip(f[3])
                @fastmath opposites_sum[m, 3, warpid] -= ustrip(f[3])
            end
            
            if needs_vir
                @fastmath vir_xx += ustrip(f[1]) * ustrip(dr[1])
                if D >= 2
                    @fastmath vir_yy += ustrip(f[2]) * ustrip(dr[2])
                    @fastmath vir_xy += ustrip(f[1]) * ustrip(dr[2])
                end
                if D >= 3
                    @fastmath vir_zz += ustrip(f[3]) * ustrip(dr[3])
                    @fastmath vir_xz += ustrip(f[1]) * ustrip(dr[3])
                    @fastmath vir_yz += ustrip(f[2]) * ustrip(dr[3])
                end
            end
        end

        sync_warp()
        @fastmath force_i_x += opposites_sum[lane, 1, warpid]
        opposites_sum[lane, 1, warpid] = zero(T)
        if D >= 2
            @fastmath force_i_y += opposites_sum[lane, 2, warpid]
            opposites_sum[lane, 2, warpid] = zero(T)
        end
        if D >= 3
            @fastmath force_i_z += opposites_sum[lane, 3, warpid]
            opposites_sum[lane, 3, warpid] = zero(T)
        end
    end

    # Part 4: Terminal corner tile
    if i == n_blocks && j == n_blocks
        if lane <= r
            @inbounds coords_i = coords[index_i]
            @inbounds vel_i = velocities[index_i]
            @inbounds atoms_i = atoms[index_i]
            
            @inbounds eligible_bitmask = compressed_masks_ro[lane, 1, mask_idx]
            @inbounds special_bitmask = compressed_masks_ro[lane, 2, mask_idx]

            @inbounds for m in (lane + a) : r
                idx_j = j_0_tile + m
                @inbounds coords_j = coords[idx_j]
                @inbounds vel_j = velocities[idx_j]
                @inbounds atoms_j = atoms[idx_j]
                
                dr = vector(coords_i, coords_j, boundary)
                r2 = @fastmath sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
                
                # Divergence-safe execution (no vote_any_sync)
                f = condition ? sum_pairwise_forces(
                    inters_tuple, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : zero(SVector{D, T})

                @fastmath force_i_x += ustrip(f[1])
                @fastmath opposites_sum[m, 1, warpid] -= ustrip(f[1])
                if D >= 2
                    @fastmath force_i_y += ustrip(f[2])
                    @fastmath opposites_sum[m, 2, warpid] -= ustrip(f[2])
                end
                if D >= 3
                    @fastmath force_i_z += ustrip(f[3])
                    @fastmath opposites_sum[m, 3, warpid] -= ustrip(f[3])
                end
                
                if needs_vir
                    @fastmath vir_xx += ustrip(f[1]) * ustrip(dr[1])
                    if D >= 2
                        @fastmath vir_yy += ustrip(f[2]) * ustrip(dr[2])
                        @fastmath vir_xy += ustrip(f[1]) * ustrip(dr[2])
                    end
                    if D >= 3
                        @fastmath vir_zz += ustrip(f[3]) * ustrip(dr[3])
                        @fastmath vir_xz += ustrip(f[1]) * ustrip(dr[3])
                        @fastmath vir_yz += ustrip(f[2]) * ustrip(dr[3])
                    end
                end
            end
        end
        
        sync_warp()

        if lane <= r
            @fastmath force_i_x += opposites_sum[lane, 1, warpid]
            opposites_sum[lane, 1, warpid] = zero(T)
            if D >= 2
                @fastmath force_i_y += opposites_sum[lane, 2, warpid]
                opposites_sum[lane, 2, warpid] = zero(T)
            end
            if D >= 3
                @fastmath force_i_z += opposites_sum[lane, 3, warpid]
                opposites_sum[lane, 3, warpid] = zero(T)
            end
        end
    end

    if needs_vir
        offset_val = Int32(16)
        while offset_val > 0
            @fastmath vir_xx += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xx, offset_val)
            if D >= 2
                @fastmath vir_yy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yy, offset_val)
                @fastmath vir_xy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xy, offset_val)
            end
            if D >= 3
                @fastmath vir_zz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_zz, offset_val)
                @fastmath vir_xz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xz, offset_val)
                @fastmath vir_yz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yz, offset_val)
            end
            offset_val ÷= 2
        end

        if lane == 1
            if vir_xx != zero(T)
                CUDA.atomic_add!(pointer(global_virial, 1), vir_xx)
            end
            if D >= 2
                if vir_yy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, D + 2), vir_yy)
                end
                if vir_xy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 2), vir_xy)
                    CUDA.atomic_add!(pointer(global_virial, D + 1), vir_xy)
                end
            end
            if D >= 3
                if vir_zz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 9), vir_zz)
                end
                if vir_xz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 3), vir_xz)
                    CUDA.atomic_add!(pointer(global_virial, 2 * D + 1), vir_xz)
                end
                if vir_yz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 6), vir_yz)
                    CUDA.atomic_add!(pointer(global_virial, 2 * D + 2), vir_yz)
                end
            end
        end
    end

    sync_warp()

    if index_i <= N
        if force_i_x != zero(T)
            CUDA.atomic_add!(pointer(fs_mat, Int64(index_i) * b - (b - 1)), -force_i_x)
        end
        if D >= 2 && force_i_y != zero(T)
            CUDA.atomic_add!(pointer(fs_mat, Int64(index_i) * b - (b - 2)), -force_i_y)
        end
        if D >= 3 && force_i_z != zero(T)
            CUDA.atomic_add!(pointer(fs_mat, Int64(index_i) * b - (b - 3)), -force_i_z)
        end
    end

    return nothing
end

#=
    energy_kernel!(...)

Compute pairwise potential energies for the compact list of interacting 32x32
tiles produced by `find_interacting_blocks_kernel!`.

This mirrors `force_kernel!` structurally: the same compact tile list, the same
four tile-shape cases, and the same CLEAN-vs-mask-backed fast path. The main
difference is the reduction target: each warp accumulates energy into shared
memory and performs a warp reduction before the final atomic add to
`energy_nounits`.
=#
function energy_kernel!(
    energy_nounits,
    coords_var,
    velocities_var,
    atoms_var::AbstractArray{A},
    ::Val{N},
    r_cut,
    ::Val{energy_units},
    inters_tuple,
    boundary,
    step_n,
    compressed_masks,
    ::Val{T},
    ::Val{D},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
    interacting_tiles_overflow) where {N, A, energy_units, T, D}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    compressed_masks_ro = CUDA.Const(compressed_masks)
    tiles_i_ro = CUDA.Const(interacting_tiles_i)
    tiles_j_ro = CUDA.Const(interacting_tiles_j)
    tiles_type_ro = CUDA.Const(interacting_tiles_type)
    num_interacting_tiles_ro = CUDA.Const(num_interacting_tiles)
    interacting_tiles_overflow_ro = CUDA.Const(interacting_tiles_overflow)

    idx = (blockIdx().x - a) * blockDim().y + threadIdx().y

    @inbounds if interacting_tiles_overflow_ro[1] != 0
        return nothing
    end

    @inbounds num_pairs = num_interacting_tiles_ro[1]
    if idx > num_pairs
        return nothing
    end

    lane = threadIdx().x
    warpid = threadIdx().y

    @inbounds i = tiles_i_ro[idx]
    @inbounds j = tiles_j_ro[idx]
    @inbounds type = tiles_type_ro[idx]

    i_0_tile = (i - a) * warpsize()
    index_i = i_0_tile + lane

    E_smem = CuStaticSharedArray(T, (32, MAX_BLOCK_Y))
    @inbounds E_smem[lane, warpid] = zero(T)

    r = Int32((N - 1) % 32 + 1)

    mask_idx = upper_tile_index(i, j, n_blocks)

    j_0_tile = (j - a) * warpsize()
    index_j = j_0_tile + lane

    if j < n_blocks && i < j
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        @inbounds coords_j = coords[index_j]
        @inbounds vel_j = velocities[index_j]
        shuffle_idx = lane
        @inbounds atoms_j = atoms[index_j]
        atom_fields = getfield.((atoms_j,), fieldnames(A))

        if type == UInt8(0) # CLEAN
            @inbounds for m in a:warpsize()
                sync_warp()
                coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, lane + a, warpsize())
                vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, lane + a, warpsize())
                shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, lane + a, warpsize())
                atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, lane + a, warpsize())
                atoms_j_shuffle = A(atom_fields...)

                dr = vector(coords_i, coords_j, boundary)
                r2 = @fastmath sum(abs2, dr)
                condition = r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j_shuffle,
                    Val(energy_units),
                    false,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})

                @fastmath E_smem[lane, warpid] += ustrip(pe[1])
            end
        else # EXCLUDED
            @inbounds eligible_bitmask = compressed_masks_ro[lane, 1, mask_idx]
            @inbounds special_bitmask = compressed_masks_ro[lane, 2, mask_idx]

            @inbounds for m in a:warpsize()
                sync_warp()
                coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, lane + a, warpsize())
                vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, lane + a, warpsize())
                shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, lane + a, warpsize())
                atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, lane + a, warpsize())
                atoms_j_shuffle = A(atom_fields...)

                dr = vector(coords_i, coords_j, boundary)
                r2 = @fastmath sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - shuffle_idx)) | (eligible_bitmask << shuffle_idx)
                spec = (special_bitmask >> (warpsize() - shuffle_idx)) | (special_bitmask << shuffle_idx)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j_shuffle,
                    Val(energy_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})

                @fastmath E_smem[lane, warpid] += ustrip(pe[1])
            end
        end
    elseif j == n_blocks && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        @inbounds eligible_bitmask = compressed_masks_ro[lane, 1, mask_idx]
        @inbounds special_bitmask = compressed_masks_ro[lane, 2, mask_idx]

        @inbounds for m in a:r
            idx_j = j_0_tile + m
            @inbounds coords_j = coords[idx_j]
            @inbounds vel_j = velocities[idx_j]
            @inbounds atoms_j = atoms[idx_j]
            dr = vector(coords_i, coords_j, boundary)
            r2 = @fastmath sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

            pe = condition ? sum_pairwise_potentials(
                inters_tuple,
                atoms_i, atoms_j,
                Val(energy_units),
                (spec & 0x1) == true,
                coords_i, coords_j,
                boundary,
                vel_i, vel_j,
                step_n) : zero(SVector{1, T})
            @fastmath E_smem[lane, warpid] += ustrip(pe[1])
        end
    elseif i == j && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        @inbounds eligible_bitmask = compressed_masks_ro[lane, 1, mask_idx]
        @inbounds special_bitmask = compressed_masks_ro[lane, 2, mask_idx]

        @inbounds for m in (lane + a) : warpsize()
            idx_j = j_0_tile + m
            @inbounds coords_j = coords[idx_j]
            @inbounds vel_j = velocities[idx_j]
            @inbounds atoms_j = atoms[idx_j]
            dr = vector(coords_i, coords_j, boundary)
            r2 = @fastmath sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

            pe = condition ? sum_pairwise_potentials(
                inters_tuple,
                atoms_i, atoms_j,
                Val(energy_units),
                (spec & 0x1) == true,
                coords_i, coords_j,
                boundary,
                vel_i, vel_j,
                step_n) : zero(SVector{1, T})
            @fastmath E_smem[lane, warpid] += ustrip(pe[1])
        end
    elseif i == n_blocks && j == n_blocks
        if lane <= r
            @inbounds coords_i = coords[index_i]
            @inbounds vel_i = velocities[index_i]
            @inbounds atoms_i = atoms[index_i]
            @inbounds eligible_bitmask = compressed_masks_ro[lane, 1, mask_idx]
            @inbounds special_bitmask = compressed_masks_ro[lane, 2, mask_idx]

            @inbounds for m in (lane + a) : r
                idx_j = j_0_tile + m
                @inbounds coords_j = coords[idx_j]
                @inbounds vel_j = velocities[idx_j]
                @inbounds atoms_j = atoms[idx_j]
                dr = vector(coords_i, coords_j, boundary)
                r2 = @fastmath sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})
                @fastmath E_smem[lane, warpid] += ustrip(pe[1])
            end
        end
    end

    # No sync_threads needed since we only sum within warp
    @inbounds sum_E = E_smem[lane, warpid]

    # Warp reduction
    offset = Int32(16)
    while offset > 0
        @fastmath sum_E += CUDA.shfl_down_sync(0xFFFFFFFF, sum_E, offset)
        offset ÷= 2
    end

    if lane == a && sum_E != zero(T)
        CUDA.atomic_add!(pointer(energy_nounits), sum_E)
    end

    return nothing
end

#=
    pairwise_force_kernel_nonl!(...)

Fallback CUDA force kernel used when no neighbor finder is active.

This evaluates every atom pair directly and is therefore `O(N^2)`. It is kept
for very small systems, explicit no-neighbor-list runs, and test coverage. The
tiled `GPUNeighborFinder` path is the production fast path for CUDA systems.
=#
function pairwise_force_kernel_nonl!(forces::AbstractArray{T}, coords_var, velocities_var,
                        atoms_var, boundary, inters, step_n, ::Val{D}, ::Val{F}) where {T, D, F}
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
    atoms = CUDA.Const(atoms_var)
    n_atoms = length(atoms)

    tidx = threadIdx().x
    i_0_tile = (blockIdx().x - 1) * warpsize()
    j_0_block = (blockIdx().y - 1) * blockDim().x
    warpidx = cld(tidx, warpsize())
    j_0_tile = j_0_block + (warpidx - 1) * warpsize()
    i = i_0_tile + laneid()

    forces_shmem = CuStaticSharedArray(T, (3, 1024))
    @inbounds for dim in 1:3
        forces_shmem[dim, tidx] = zero(T)
    end

    if i_0_tile + warpsize() > n_atoms || j_0_tile + warpsize() > n_atoms
        @inbounds if i <= n_atoms
            njs = min(warpsize(), n_atoms - j_0_tile)
            @inbounds atom_i, coord_i, vel_i = atoms[i], coords[i], velocities[i]
            for del_j in 1:njs
                j = j_0_tile + del_j
                if i != j
                    @inbounds atom_j, coord_j, vel_j = atoms[j], coords[j], velocities[j]
                    f = sum_pairwise_forces(inters, atom_i, atom_j, Val(F), false, coord_i,
                                            coord_j, boundary, vel_i, vel_j, step_n)
                    for dim in 1:D
                        forces_shmem[dim, tidx] += -ustrip(f[dim])
                    end
                end
            end

            for dim in 1:D
                Atomix.@atomic :monotonic forces[dim, i] += forces_shmem[dim, tidx]
            end
        end
    else
        j = j_0_tile + laneid()
        tilesteps = warpsize()
        if i_0_tile == j_0_tile  # To not compute i-i forces
            j = j_0_tile + laneid() % warpsize() + 1
            tilesteps -= 1
        end

        @inbounds atom_i, coord_i, vel_i = atoms[i], coords[i], velocities[i]
        @inbounds coord_j, vel_j = coords[j], velocities[j]
        @inbounds for _ in 1:tilesteps
            sync_warp()
            @inbounds atom_j = atoms[j]
            f = sum_pairwise_forces(inters, atom_i, atom_j, Val(F), false, coord_i, coord_j,
                                    boundary, vel_i, vel_j, step_n)
            for dim in 1:D
                forces_shmem[dim, tidx] += -ustrip(f[dim])
            end
            @shfl_multiple_sync(FULL_MASK, laneid() + 1, warpsize(), j, coord_j)
        end

        @inbounds for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += forces_shmem[dim, tidx]
        end
    end

    return nothing
end

end
