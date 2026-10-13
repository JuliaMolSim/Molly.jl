"""
    MollyCUDAExt

CUDA extension for Molly.jl.

The tiled pairwise kernels used with [`GPUNeighborFinder`](@ref) are portable
KernelAbstractions/KernelInterface kernels in Molly itself, see `src/gpu_tiles.jl`.
This extension only holds what is specific to CUDA:
- compiling the pairwise kernels with fast math (for `Float32`), always inlining and
  an optional register limit,
- the autotuner for the launch parameters of the tiled kernels, which is keyed on
  the CUDA device,
- device selection.
"""
module MollyCUDAExt

using Molly
using Molly: from_device, box_sides, sorted_morton_seq!, volume,
             # Re-exported for the tests, which reach them through the extension
             reset_interacting_tile_state!, find_interacting_tiles_all_pairs!,
             find_interacting_tiles_tree!, find_interacting_tiles!, use_tile_tree,
             prune_interacting_tiles!, refresh_tile_exceptions!, reorder_system_gpu!,
             compute_block_bounds!, launch_find_interacting_blocks!, launch_force_tiles!,
             launch_energy_tiles!, force_kernel_localmem, energy_kernel_localmem,
             throw_if_interacting_tiles_overflowed, env_override, prefer_override,
             validate_block_y, MAX_BLOCK_Y, MAX_TILE_LOCALMEM, TILE_WIDTH
using CUDA
using KernelAbstractions

function Molly.get_gpu_devices(::Val{true})
    return collect(CUDA.devices())
end

function Molly.set_gpu_device!(gpu_id, ::Val{true})
    CUDA.device!(gpu_id)
end

const CUDA_CORE = isdefined(CUDA, :CUDACore) ? CUDACore : CUDA

Molly.uses_gpu_neighbor_finder(::Type{<:CuArray}) = true

# Whether to compile the pairwise kernels with fast math
# Restricted to Float32, where the extra error is not observable
pairwise_fastmath(::Type{Float32}) = true
pairwise_fastmath(::Type) = false

function Molly.tile_kernel_config(backend::CUDABackend, ::Type{T}, maxregs) where T
    kernel_backend = CUDABackend(; prefer_blocks=backend.prefer_blocks, always_inline=true,
                                 fastmath=pairwise_fastmath(T))
    options = maxregs === nothing ? (;) : (; maxregs=maxregs)
    return kernel_backend, options
end

## Autotuning of the launch parameters

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
    interaction_signature::Tuple
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

function effective_tile_threads_override(config::Molly.CUDALaunchConfig)
    threads_x_env = env_override("MOLLY_CUDA_TILE_THREADS_X")
    threads_y_env = env_override("MOLLY_CUDA_TILE_THREADS_Y")
    if xor(threads_x_env === nothing, threads_y_env === nothing)
        error("set both MOLLY_CUDA_TILE_THREADS_X and MOLLY_CUDA_TILE_THREADS_Y together")
    end
    config_tile_threads = Molly.cuda_tile_threads(config)
    return config_tile_threads === nothing ?
           (threads_x_env === nothing ? nothing : (threads_x_env, threads_y_env)) :
           config_tile_threads
end

effective_force_block_y_override(config::Molly.CUDALaunchConfig) =
    prefer_override(Molly.cuda_force_block_y(config), env_override("MOLLY_CUDA_FORCE_BLOCK_Y"))

effective_energy_block_y_override(config::Molly.CUDALaunchConfig) =
    prefer_override(Molly.cuda_energy_block_y(config), env_override("MOLLY_CUDA_ENERGY_BLOCK_Y"))

effective_force_maxregs_override(config::Molly.CUDALaunchConfig) =
    prefer_override(Molly.cuda_force_maxregs(config), env_override("MOLLY_CUDA_FORCE_MAXREGS"))

@inline autotune_scalar(x) = round(Float64(ustrip(x)); sigdigits=12)
@inline autotune_interaction_signature(inter) = hash(repr(inter))

function autotune_box_signature(boundary, ::Val{D}) where D
    sides = box_sides(boundary)
    return ntuple(i -> autotune_scalar(sides[i]), D)
end

gpu_neighbor_pairwise_inters(sys) = Tuple(filter(use_neighbors, values(sys.pairwise_inters)))

function autotune_key(sys::System{D, <:CuArray}, pairwise_inters, force_maxregs_override) where D
    dev = CUDA.device()
    sm_count = CUDA.attribute(dev, CUDA_CORE.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
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
        cld(n_atoms, TILE_WIDTH),
        autotune_box_signature(sys.boundary, Val(D)),
        autotune_scalar(sys.neighbor_finder.dist_cutoff),
        Tuple(map(autotune_interaction_signature, pairwise_inters)),
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

"""
    autotune_benchmark_ms!(prepare!, run!; warmup=1, repeats=3)

Benchmark a given kernel execution function `run!` after setting up the state
with `prepare!`. Returns the minimum execution time in milliseconds across the
`repeats` (excluding `warmup` runs).
"""
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

function autotune_prepare_common_state!(buffers, sys::System{D, <:CuArray}, N::Int) where D
    morton_bits = 10
    sides = box_sides(sys.boundary)
    cell_width = sides ./ (2^morton_bits)
    sorted_morton_seq!(buffers, sys.coords, cell_width, morton_bits)
    refresh_tile_exceptions!(buffers, sys.neighbor_finder, Val(N))
    buffers.sparse_pair_generation = sys.neighbor_finder.cache_generation
    reorder_system_gpu!(buffers, sys)
    compute_block_bounds!(buffers, sys, N)
    CUDA.synchronize()
    return nothing
end

function autotune_tile_thread_candidates()
    max_threads = Molly.KI.max_work_group_size(CUDABackend())
    candidates = NTuple{2, Int}[]
    for threads_xy in AUTOTUNE_TILE_THREAD_CANDIDATES
        threads_xy[1] * threads_xy[2] <= max_threads && push!(candidates, threads_xy)
    end
    return candidates
end

# The block_y candidates whose local memory fits the static shared memory limit
function autotune_block_y_candidates(candidates, localmem_bytes)
    valid_candidates = Int[]
    for block_y in candidates
        localmem_bytes(block_y) <= MAX_TILE_LOCALMEM && push!(valid_candidates, block_y)
    end
    isempty(valid_candidates) && push!(valid_candidates, 1)
    return valid_candidates
end

"""
    autotune_tile_threads!(buffers, sys, N)

Benchmark work-group dimensions for the tile finding kernel (`find_interacting_blocks_kernel!`).
Tests candidates from `AUTOTUNE_TILE_THREAD_CANDIDATES` and returns the fastest `threads_xy`
that does not cause a tile buffer overflow.
"""
function autotune_tile_threads!(buffers, sys::System{D, <:CuArray}, N::Int) where D
    n_blocks = cld(N, TILE_WIDTH)
    # Also grows the tile list to fit, so that the candidates below do not overflow it
    find_interacting_tiles!(buffers, sys, n_blocks)
    if use_tile_tree(sys, n_blocks)
        # The tree search has no work-group shape to tune
        return nothing
    end
    candidates = autotune_tile_thread_candidates()
    best_threads = first(candidates)
    best_ms = Inf
    expected_num_tiles = nothing

    for threads_xy in candidates
        ms = autotune_benchmark_ms!(
            () -> reset_interacting_tile_state!(buffers),
            () -> launch_find_interacting_blocks!(buffers, sys, n_blocks, threads_xy),
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
    launch_find_interacting_blocks!(buffers, sys, n_blocks, best_threads)
    CUDA.synchronize()
    throw_if_interacting_tiles_overflowed(buffers)
    buffers.num_pairs = Int(only(from_device(buffers.num_interacting_tiles)))
    return best_threads
end

"""
    autotune_force_block_y!(buffers, sys, pairwise_inters, N, force_maxregs_override)

Benchmark `block_y` sizes, the number of tiles per work-group, for the pairwise
`force_kernel!`. Returns the `block_y` configuration that achieves the minimum
execution time.
"""
function autotune_force_block_y!(buffers, sys::System{D, <:CuArray, T, TH}, pairwise_inters,
                                 N::Int, force_maxregs_override) where {D, T, TH}
    uses_vel = Molly.any_uses_velocity(pairwise_inters)
    candidates = autotune_block_y_candidates(AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES,
        by -> force_kernel_localmem(buffers, Val(D), T, uses_vel, by, pairwise_inters))
    buffers.num_pairs == 0 && return first(candidates)

    best_block_y = first(candidates)
    best_ms = Inf
    for block_y in candidates
        ms = autotune_benchmark_ms!(
            () -> begin
                fill!(buffers.fs_mat_reordered, zero(T))
                fill!(buffers.virial_nounits, zero(TH))
            end,
            () -> launch_force_tiles!(buffers, sys, pairwise_inters, Val(false), 0, block_y,
                                      force_maxregs_override),
        )
        if ms < best_ms
            best_ms = ms
            best_block_y = block_y
        end
    end
    return best_block_y
end

"""
    autotune_energy_block_y!(buffers, sys, pairwise_inters, N)

Benchmark `block_y` sizes for the pairwise `energy_kernel!`.
Returns the `block_y` configuration that achieves the minimum execution time.
"""
function autotune_energy_block_y!(buffers, sys::System{D, <:CuArray, T, TH}, pairwise_inters,
                                  N::Int) where {D, T, TH}
    uses_vel = Molly.any_uses_velocity(pairwise_inters)
    candidates = autotune_block_y_candidates(AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES,
        by -> energy_kernel_localmem(buffers, uses_vel, by, pairwise_inters))
    buffers.num_pairs == 0 && return first(candidates)

    best_block_y = first(candidates)
    best_ms = Inf
    for block_y in candidates
        ms = autotune_benchmark_ms!(
            () -> fill!(buffers.pe_vec_nounits, zero(TH)),
            () -> launch_energy_tiles!(buffers.pe_vec_nounits, buffers, sys, pairwise_inters,
                                       0, block_y),
        )
        if ms < best_ms
            best_ms = ms
            best_block_y = block_y
        end
    end
    return best_block_y
end

"""
    autotune_cuda_launch_config(sys, pairwise_inters, force_maxregs_override)

Perform a full autotuning run for the tiled pairwise kernels on CUDA.
Sets up temporary GPU buffers and Morton/exception states, then individually tunes the tile search
work-group, force `block_y`, and energy `block_y`. Returns a populated `CUDALaunchConfig`.
"""
function autotune_cuda_launch_config(sys::System{D, <:CuArray, T}, pairwise_inters,
                                     force_maxregs_override) where {D, T}
    N = length(sys.coords)
    buffers = Molly.init_buffers!(sys, 1, true)
    autotune_prepare_common_state!(buffers, sys, N)
    tile_threads = autotune_tile_threads!(buffers, sys, N)
    # Time the kernels on the pruned tile list that a simulation evaluates, since an
    #   unpruned list over-represents tiles that are skipped or need exclusion masks
    prune_interacting_tiles!(buffers, sys, N)
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

    current_config = Molly.cuda_launch_config(sys)
    force_block_y_override = effective_force_block_y_override(current_config)
    energy_block_y_override = effective_energy_block_y_override(current_config)
    tile_threads_override = effective_tile_threads_override(current_config)
    force_maxregs_override = effective_force_maxregs_override(current_config)

    current_force_block_y = Molly.cuda_force_block_y(current_config)
    current_energy_block_y = Molly.cuda_energy_block_y(current_config)
    current_force_block_y === nothing || validate_block_y("force_block_y", current_force_block_y)
    current_energy_block_y === nothing || validate_block_y("energy_block_y", current_energy_block_y)
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
        force_block_y = needs_force ? Molly.cuda_force_block_y(tuned_config) : current_force_block_y,
        force_maxregs = Molly.cuda_force_maxregs(current_config),
        tile_threads = needs_tile ? Molly.cuda_tile_threads(tuned_config) : Molly.cuda_tile_threads(current_config),
        energy_block_y = needs_energy ? Molly.cuda_energy_block_y(tuned_config) : current_energy_block_y,
    )
    Molly.set_cuda_launch_config!(sys, merged_config)

    return something(
        effective_force_block_y_override(merged_config),
        Molly.cuda_force_block_y(tuned_config),
    )
end

end
