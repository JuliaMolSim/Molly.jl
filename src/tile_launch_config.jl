# Launch parameters of the tiled pairwise kernels of GPUNeighborFinder, see gpu_tiles.jl:
#   the overrides kept in a system and the autotuner, which run on any backend that
#   supports the kernels

export optimize_tile_launch_config!

#=
    tile_launch_config(system)

Return the stored launch overrides of the tiled pairwise kernels of
[`GPUNeighborFinder`](@ref) for `system`, a `TileLaunchConfig`.

They are consulted whenever the kernels choose launch parameters for that
[`System`](@ref), ahead of the environment variables `MOLLY_TILE_FORCE_BLOCK_Y`,
`MOLLY_TILE_ENERGY_BLOCK_Y`, `MOLLY_TILE_THREADS_X`, `MOLLY_TILE_THREADS_Y` and
`MOLLY_TILE_FORCE_MAXREGS`.
=#
tile_launch_config(sys::System) = sys.launch_config

#=
    set_tile_launch_config!(system, config::TileLaunchConfig)
    set_tile_launch_config!(system; kwargs...)

Set the launch overrides of the tiled pairwise kernels for `system`.

This mutates `system.launch_config`. The new configuration is used by subsequent force,
tile search and energy launches for that system until [`reset_tile_launch_config!`](@ref)
is called.
=#
function set_tile_launch_config!(sys::System, config::TileLaunchConfig)
    sys.launch_config = config
    return config
end

set_tile_launch_config!(sys::System; kwargs...) =
    set_tile_launch_config!(sys, TileLaunchConfig(; kwargs...))

#=
    reset_tile_launch_config!(system)

Clear any explicit launch overrides of the tiled pairwise kernels for `system` and return
to automatic launch selection.
=#
reset_tile_launch_config!(sys::System) = set_tile_launch_config!(sys, TileLaunchConfig())

## Autotuning

const AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES = (1, 2, 4, 8, 16)
const AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES = (1, 2, 4, 8, 16)
const AUTOTUNE_TILE_THREAD_CANDIDATES = ((8, 8), (16, 8), (16, 16), (32, 8), (32, 16))
const AUTOTUNE_WARMUP_RUNS = 1
const AUTOTUNE_MEASURE_RUNS = 3

#=
The tuned configurations are cached in the process, for the backend type and the device,
which identify a device within a process, and the properties of the system that the
timings depend on.
=#
struct LaunchAutotuneKey
    backend_type::Type
    device::Int
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

const TILE_LAUNCH_AUTOTUNE_CACHE = Dict{LaunchAutotuneKey, TileLaunchConfig}()
const TILE_LAUNCH_AUTOTUNE_LOCK = ReentrantLock()

#=
    reset_tile_launch_autotune_cache!()

Clear any cached autotuning results.

This does not modify the active `TileLaunchConfig` of any system, it only clears the
process-local cache used by [`optimize_tile_launch_config!`](@ref).
=#
function reset_tile_launch_autotune_cache!()
    lock(TILE_LAUNCH_AUTOTUNE_LOCK) do
        empty!(TILE_LAUNCH_AUTOTUNE_CACHE)
    end
    return nothing
end

function effective_tile_threads_override(config::TileLaunchConfig)
    threads_x_env = env_override("MOLLY_TILE_THREADS_X")
    threads_y_env = env_override("MOLLY_TILE_THREADS_Y")
    if xor(threads_x_env === nothing, threads_y_env === nothing)
        error("set both MOLLY_TILE_THREADS_X and MOLLY_TILE_THREADS_Y together")
    end
    threads = config_tile_threads(config)
    return threads === nothing ?
           (threads_x_env === nothing ? nothing : (threads_x_env, threads_y_env)) :
           threads
end

effective_force_block_y_override(config::TileLaunchConfig) =
    prefer_override(config_force_block_y(config), env_override("MOLLY_TILE_FORCE_BLOCK_Y"))

effective_energy_block_y_override(config::TileLaunchConfig) =
    prefer_override(config_energy_block_y(config), env_override("MOLLY_TILE_ENERGY_BLOCK_Y"))

effective_force_maxregs_override(config::TileLaunchConfig) =
    prefer_override(config_force_maxregs(config), env_override("MOLLY_TILE_FORCE_MAXREGS"))

@inline autotune_scalar(x) = round(Float64(ustrip(x)); sigdigits=12)
@inline autotune_interaction_signature(inter) = hash(repr(inter))

function autotune_box_signature(boundary, ::Val{D}) where D
    sides = box_sides(boundary)
    return ntuple(i -> autotune_scalar(sides[i]), D)
end

gpu_neighbor_pairwise_inters(sys) = Tuple(filter(use_neighbors, values(sys.pairwise_inters)))

function autotune_key(sys::System{D}, backend, pairwise_inters,
                      force_maxregs_override) where D
    n_atoms = length(sys.coords)
    return LaunchAutotuneKey(
        typeof(backend),
        KI.device(backend),
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

function autotune_stage_time_ms!(f::F, backend) where F
    KernelAbstractions.synchronize(backend)
    start_ns = time_ns()
    f()
    KernelAbstractions.synchronize(backend)
    return (time_ns() - start_ns) / 1.0e6
end

#=
    autotune_benchmark_ms!(prepare!, run!, backend; warmup=1, repeats=3)

Benchmark a given kernel execution function `run!` after setting up the state with
`prepare!`. Returns the minimum execution time in milliseconds across the `repeats`
(excluding `warmup` runs).
=#
function autotune_benchmark_ms!(prepare!::F, run!::G, backend;
                                warmup::Int=AUTOTUNE_WARMUP_RUNS,
                                repeats::Int=AUTOTUNE_MEASURE_RUNS) where {F, G}
    for _ in 1:warmup
        prepare!()
        run!()
        KernelAbstractions.synchronize(backend)
    end

    best_ms = Inf
    for _ in 1:repeats
        prepare!()
        best_ms = min(best_ms, autotune_stage_time_ms!(run!, backend))
    end
    return best_ms
end

function autotune_prepare_common_state!(buffers, sys, N::Int)
    sorted_morton_seq!(buffers, sys.coords, sys.boundary)
    refresh_tile_exceptions!(buffers, sys.neighbor_finder, Val(N))
    buffers.sparse_pair_generation = sys.neighbor_finder.cache_generation
    reorder_system_gpu!(buffers, sys)
    compute_block_bounds!(buffers, sys, N)
    KernelAbstractions.synchronize(get_backend(sys.coords))
    return nothing
end

function autotune_tile_thread_candidates(backend)
    max_threads = KI.max_work_group_size(backend)
    candidates = NTuple{2, Int}[]
    for threads_xy in AUTOTUNE_TILE_THREAD_CANDIDATES
        threads_xy[1] * threads_xy[2] <= max_threads && push!(candidates, threads_xy)
    end
    return candidates
end

# The block_y candidates whose local memory fits in `max_localmem` bytes
function autotune_block_y_candidates(candidates, localmem_bytes, max_localmem)
    valid_candidates = Int[]
    for block_y in candidates
        localmem_bytes(block_y) <= max_localmem && push!(valid_candidates, block_y)
    end
    isempty(valid_candidates) && push!(valid_candidates, 1)
    return valid_candidates
end

#=
    autotune_tile_threads!(buffers, sys, N)

Benchmark work-group dimensions for the tile finding kernel
(`find_interacting_blocks_kernel!`). Tests candidates from
`AUTOTUNE_TILE_THREAD_CANDIDATES` and returns the fastest `threads_xy` that does not cause
a tile buffer overflow.
=#
function autotune_tile_threads!(buffers, sys, N::Int)
    backend = get_backend(sys.coords)
    n_blocks = cld(N, TILE_WIDTH)
    # Also grows the tile list to fit, so that the candidates below do not overflow it
    find_interacting_tiles!(buffers, sys, n_blocks)
    if use_tile_tree(sys, n_blocks, buffers.num_tile_candidates)
        # The tree search has no work-group shape to tune
        return nothing
    end
    candidates = autotune_tile_thread_candidates(backend)
    best_threads = first(candidates)
    best_ms = Inf
    expected_num_tiles = nothing

    for threads_xy in candidates
        ms = autotune_benchmark_ms!(
            () -> reset_interacting_tile_state!(buffers),
            () -> launch_find_interacting_blocks!(buffers, sys, n_blocks, threads_xy),
            backend,
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
    KernelAbstractions.synchronize(backend)
    throw_if_interacting_tiles_overflowed(buffers)
    buffers.num_pairs = Int(only(from_device(buffers.num_interacting_tiles)))
    buffers.num_tile_candidates = buffers.num_pairs
    return best_threads
end

#=
    autotune_force_block_y!(buffers, sys, pairwise_inters, N, force_maxregs_override)

Benchmark `block_y` sizes, the number of tiles per work-group, for the pairwise
`force_kernel!`. Returns the `block_y` configuration that achieves the minimum execution
time.
=#
function autotune_force_block_y!(buffers, sys::System{D, <:Any, T, TH}, pairwise_inters,
                                 N::Int, force_maxregs_override) where {D, T, TH}
    backend = get_backend(sys.coords)
    uses_vel = any_uses_velocity(pairwise_inters)
    candidates = autotune_block_y_candidates(AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES,
        by -> force_kernel_localmem(buffers, Val(D), T, uses_vel, by, pairwise_inters),
        max_tile_localmem(backend))
    buffers.num_pairs == 0 && return first(candidates)

    best_block_y = first(candidates)
    best_ms = Inf
    for block_y in candidates
        ms = autotune_benchmark_ms!(
            () -> begin
                fill!(buffers.fs_mat_reordered, zero(T))
                fill!(buffers.virial_slots, zero(TH))
            end,
            () -> launch_force_tiles!(buffers, sys, pairwise_inters, Val(false), 0, block_y,
                                      force_maxregs_override),
            backend,
        )
        if ms < best_ms
            best_ms = ms
            best_block_y = block_y
        end
    end
    return best_block_y
end

#=
    autotune_energy_block_y!(buffers, sys, pairwise_inters, N)

Benchmark `block_y` sizes for the pairwise `energy_kernel!`. Returns the `block_y`
configuration that achieves the minimum execution time.
=#
function autotune_energy_block_y!(buffers, sys::System{D, <:Any, T, TH}, pairwise_inters,
                                  N::Int) where {D, T, TH}
    backend = get_backend(sys.coords)
    uses_vel = any_uses_velocity(pairwise_inters)
    candidates = autotune_block_y_candidates(AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES,
        by -> energy_kernel_localmem(buffers, uses_vel, by, pairwise_inters),
        max_tile_localmem(backend))
    buffers.num_pairs == 0 && return first(candidates)

    best_block_y = first(candidates)
    best_ms = Inf
    for block_y in candidates
        ms = autotune_benchmark_ms!(
            () -> fill!(buffers.pe_vec_nounits, zero(TH)),
            () -> launch_energy_tiles!(buffers.pe_vec_nounits, buffers, sys, pairwise_inters,
                                       0, block_y),
            backend,
        )
        if ms < best_ms
            best_ms = ms
            best_block_y = block_y
        end
    end
    return best_block_y
end

#=
    autotune_tile_launch_config(sys, pairwise_inters, force_maxregs_override)

Perform a full autotuning run for the tiled pairwise kernels. Sets up temporary buffers
and the Morton and exception state, then tunes the tile search work-group, the force
`block_y` and the energy `block_y` in turn. Returns a populated `TileLaunchConfig`.
=#
function autotune_tile_launch_config(sys, pairwise_inters, force_maxregs_override)
    N = length(sys.coords)
    buffers = init_buffers!(sys, 1, true)
    autotune_prepare_common_state!(buffers, sys, N)
    tile_threads = autotune_tile_threads!(buffers, sys, N)
    # Time the kernels on the pruned tile list that a simulation evaluates, since an
    #   unpruned list over-represents tiles that are skipped or need exclusion masks
    prune_interacting_tiles!(buffers, sys, N)
    force_block_y = autotune_force_block_y!(buffers, sys, pairwise_inters, N,
                                            force_maxregs_override)
    energy_block_y = autotune_energy_block_y!(buffers, sys, pairwise_inters, N)
    return TileLaunchConfig(
        force_block_y=force_block_y,
        force_maxregs=nothing,
        tile_threads=tile_threads,
        energy_block_y=energy_block_y,
    )
end

function cached_autotune_config(key::LaunchAutotuneKey)
    lock(TILE_LAUNCH_AUTOTUNE_LOCK) do
        return get(TILE_LAUNCH_AUTOTUNE_CACHE, key, nothing)
    end
end

function cache_autotune_config!(key::LaunchAutotuneKey, config::TileLaunchConfig)
    lock(TILE_LAUNCH_AUTOTUNE_LOCK) do
        return get!(TILE_LAUNCH_AUTOTUNE_CACHE, key, config)
    end
end

"""
    optimize_tile_launch_config!(system)

Autotune and cache the launch parameters of the tiled pairwise kernels of
[`GPUNeighborFinder`](@ref) for the given system.

For a GPU system with a [`GPUNeighborFinder`](@ref), this benchmarks a small set of
launch candidates for the force, energy and tile search kernels, caches the result for the
device and the system signature, and writes the winning values into
`system.launch_config`, keeping any values already set there or with the `MOLLY_TILE_*`
environment variables. It returns the force kernel `block_y` that is used.
For other systems this does nothing and returns `nothing`.
This is called when setting up a [`System`](@ref) from a file unless
`autotune_launch=false` is given.
"""
function optimize_tile_launch_config!(sys::System{<:Any, <:AbstractGPUArray})
    sys.neighbor_finder isa GPUNeighborFinder || return nothing
    pairwise_inters = gpu_neighbor_pairwise_inters(sys)
    isempty(pairwise_inters) && return nothing

    current_config = tile_launch_config(sys)
    force_block_y_override = effective_force_block_y_override(current_config)
    energy_block_y_override = effective_energy_block_y_override(current_config)
    tile_threads_override = effective_tile_threads_override(current_config)
    force_maxregs_override = effective_force_maxregs_override(current_config)

    current_force_block_y = config_force_block_y(current_config)
    current_energy_block_y = config_energy_block_y(current_config)
    current_force_block_y === nothing || validate_block_y("force_block_y", current_force_block_y)
    current_energy_block_y === nothing || validate_block_y("energy_block_y", current_energy_block_y)
    force_maxregs_override === nothing || force_maxregs_override > 0 ||
        error("MOLLY_TILE_FORCE_MAXREGS must be positive, got $(force_maxregs_override)")

    needs_force = force_block_y_override === nothing
    needs_energy = energy_block_y_override === nothing
    needs_tile = tile_threads_override === nothing

    if !(needs_force || needs_energy || needs_tile)
        return force_block_y_override
    end

    backend = get_backend(sys.coords)
    key = autotune_key(sys, backend, pairwise_inters, force_maxregs_override)
    tuned_config = cached_autotune_config(key)
    if tuned_config === nothing
        tuned_config = autotune_tile_launch_config(sys, pairwise_inters, force_maxregs_override)
        tuned_config = cache_autotune_config!(key, tuned_config)
    end

    merged_config = TileLaunchConfig(
        force_block_y = needs_force ? config_force_block_y(tuned_config) : current_force_block_y,
        force_maxregs = config_force_maxregs(current_config),
        tile_threads = needs_tile ? config_tile_threads(tuned_config) :
                                    config_tile_threads(current_config),
        energy_block_y = needs_energy ? config_energy_block_y(tuned_config) :
                                        current_energy_block_y,
    )
    set_tile_launch_config!(sys, merged_config)

    return something(
        effective_force_block_y_override(merged_config),
        config_force_block_y(tuned_config),
    )
end

optimize_tile_launch_config!(sys) = nothing

@inline function maybe_optimize_tile_launch_config!(sys; enabled::Bool=true)
    enabled || return nothing
    return optimize_tile_launch_config!(sys)
end
