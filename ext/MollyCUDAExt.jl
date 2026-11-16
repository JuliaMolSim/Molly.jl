"""
    MollyCUDAExt

CUDA extension for Molly.jl. This module provides highly optimized CUDA kernels
for pairwise force and energy calculations, utilizing warp-level primitives,
Morton ordering for spatial locality, and a tiled preprocessing pipeline.

The pipeline generally follows these steps:
1.  **Reordering**: Atoms are periodically reordered based on Morton (Z-order) curves
    to improve cache hits during pairwise interactions.
2.  **Exceptions**: The sparse per-atom lists of excluded and special pairs are
    translated into Morton positions, so that the bitmasks of the few 32x32 tiles
    that contain exceptions can be built on the fly without storing a mask per tile.
3.  **Tile Finding**: A kernel identifies pairs of 32x32 atom blocks (tiles) that are
    within the interaction cutoff, using bounding box checks.
4.  **Execution**: Specialized kernels iterate over the list of interacting tiles,
    using warp shuffles to efficiently compute forces/energies within each tile.
"""
module MollyCUDAExt

using Molly
using Molly: from_device, box_sides, sorted_morton_seq!, sum_pairwise_forces_gpu,
             sum_pairwise_forces_nonl, sum_pairwise_potentials_gpu, volume
using CUDA
using Atomix
using KernelAbstractions

function Molly.get_gpu_devices(::Val{true})
    return collect(CUDA.devices())
end

function Molly.set_gpu_device!(gpu_id, ::Val{true})
    CUDA.device!(gpu_id)
end

# At the moment this is needed, since a change on naming for CUDA 6
# - In principle, CUDA does export Const, but it conflicts with another
#   namespace, so annotation is required.
# - shfl_recurse is unexported so annotation is required. A PR has been opened
#   for this, we have to wait until it is available in the stable release.
const CUDA_CORE = isdefined(CUDA, :CUDACore) ? CUDACore : CUDA

# Whether to compile the pairwise kernels with fast math
# Restricted to Float32, where the extra error is not observable
pairwise_fastmath(::Type{Float32}) = true
pairwise_fastmath(::Type) = false

#=
`@cuda` finds the compiled kernel from the device types of the arguments at every launch,
which converts the arguments, builds a tuple type at run time and looks it up under a lock.
For the kernels launched every step that is often longer than the kernel takes to run, and
the simulation step is limited by the host launching work. Those kernels are kept in a
slot for each function, argument types on the host and compilation options instead, from
`kernel_slot`, which is found when the calling code is compiled. The kernel is looked up
again for another context, or once methods have been defined as CUDA.jl would then check
whether it has to be compiled again.
=#
@generated kernel_slot(f, options, args...) = :($(Ref{Any}(nothing)))

struct CachedKernel{K}
    ctx::CuContext
    world::UInt
    kernel::K
end

# The compiled kernel of `f(args...)` with the compilation options of `cufunction` in the
#   named tuple of `Val(options)`
function cached_cufunction(f::F, args::Tuple, ::Val{options}) where {F, options}
    slot = kernel_slot(f, Val(options), args...)
    entry = slot[]
    ctx = CUDA.context()
    world = Base.get_world_counter()
    if !(entry isa CachedKernel) || entry.ctx != ctx || entry.world != world
        tt = Tuple{map(a -> Core.Typeof(cudaconvert(a)), args)...}
        entry = CachedKernel(ctx, world, cufunction(f, tt; options...))
        slot[] = entry
    end
    return entry.kernel
end

#=
Faster launches of KernelAbstractions kernels with a static workgroup size, see
`Molly.launch_kernel!`.
The CUDA backend of KernelAbstractions converts the arguments to find their device types,
looks up the compiled kernel under a lock, then converts the arguments again to launch it,
at every launch. Here the compiled kernel is kept as above, so that the arguments are only
converted for the launch. The launch configuration is found as the backend finds it.
=#
function Molly.launch_kernel!(obj::KernelAbstractions.Kernel{CUDABackend,
                                        <:KernelAbstractions.NDIteration.StaticSize},
                              args...; ndrange)
    KA = KernelAbstractions
    ndrange_, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, nothing)
    ctx = KA.mkcontext(obj, ndrange_, iterspace)
    blocks = length(KA.blocks(iterspace))
    blocks == 0 && return nothing
    threads = length(KA.workitems(iterspace))
    maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    always_inline = KA.backend(obj).always_inline
    # Known from the types, so the options are constant apart from `always_inline`
    fastmath = Molly.kernel_fastmath(obj.f, args)
    kernel = (always_inline ?
        cached_cufunction(obj.f, (ctx, args...),
                          Val((always_inline=true, maxthreads=maxthreads, fastmath=fastmath))) :
        cached_cufunction(obj.f, (ctx, args...),
                          Val((always_inline=false, maxthreads=maxthreads, fastmath=fastmath))))
    kernel(ctx, args...; threads=threads, blocks=blocks)
    return nothing
end

#=
The stream for the work launched by `Molly.run_side_work!`, with the events that order it
with the stream of the task. It has the highest priority, so that the blocks of its kernels
are run as soon as there is room on a multiprocessor rather than after the blocks of the
pairwise force kernel launched before them, which would leave little time to run at once.
Kept per task and device, as each task has its own stream.
=#
struct SideStream
    stream::CuStream
    started::CuEvent
    finished::CuEvent
end

function side_stream()
    key = (:molly_side_stream, CUDA.device())
    return get!(task_local_storage(), key) do
        SideStream(CuStream(; priority=last(CUDA.priority_range())),
                   CuEvent(CUDA.EVENT_DISABLE_TIMING), CuEvent(CUDA.EVENT_DISABLE_TIMING))
    end::SideStream
end

struct SideWork
    main::CuStream
    side::SideStream
end

function Molly.side_work_start(::CUDABackend)
    side = side_stream()
    main = CUDA.stream()
    CUDA.record(side.started, main)
    return SideWork(main, side)
end

#=
CUDA.jl makes the host wait for the stream that an array was last used on when it is used
on another, which here would wait for the pairwise force kernel. The events order the
streams instead, so the shared arrays are used without that and left marked as used on the
main stream. Other arrays used by the side work, such as those of the interactions, are
then marked as used on the side stream.
=#
function Molly.run_side_work!(f, w::SideWork, shared)
    s = w.side
    CUDA.wait(s.started, s.stream)
    synchronizing = map(unsynchronize!, shared)
    try
        CUDA.stream!(f, s.stream)
    finally
        foreach((x, sync) -> resynchronize!(x, sync, w.main), shared, synchronizing)
    end
    CUDA.record(s.finished, s.stream)
    return w
end

function Molly.side_work_finish!(w::SideWork)
    CUDA.wait(w.side.finished, w.main)
    return w
end

function unsynchronize!(x::CuArray)
    managed = x.data[]
    synchronizing = managed.synchronizing
    managed.synchronizing = false
    return synchronizing
end

unsynchronize!(x) = false

function resynchronize!(x::CuArray, synchronizing, stream)
    managed = x.data[]
    managed.synchronizing = synchronizing
    managed.stream = stream
    return x
end

resynchronize!(x, synchronizing, stream) = x

# The gradients of the forces are found without the side stream
Molly.concurrent_gpu_forces(sys::System{<:Any, <:CuArray}) = !sys.grad_safe

#=
The accumulator in shared memory of the forces on the atoms of block j in `force_kernel!`,
indexed as an array of size (32, D, block_y). The components of each atom are held together
as one aligned vector, of four values for three dimensions, so that adding the force of a
pair, which reads and writes every component, is one vector load and store rather than one
of each for every component once the compiler has merged the accesses.
=#
struct JForceAccumulator{A}
    data::A
end

@inline jforce_eltype(::Type{T}, ::Val{D}) where {T, D} = NTuple{(D == 3 ? 4 : D), VecElement{T}}

@inline Base.getindex(acc::JForceAccumulator, slot, k, w) = @inbounds acc.data[slot, w][k].value

@inline function Base.setindex!(acc::JForceAccumulator, val, slot, k, w)
    v = @inbounds acc.data[slot, w]
    @inbounds acc.data[slot, w] = Base.setindex(v, VecElement(val), k)
    return acc
end

# Per-j-atom data staged in shared memory by force_kernel!'s Part 1 inner loop.
# Only the Atom fields the active interactions actually read are staged (`P` is the
# narrow payload tuple type from `atom_shuffle_payload`/`resolve_atom_fields`)
struct JStage{V, P}
    coords::V
    atom_payload::P
end

# Compile-time-only: the Tuple type produced by `atom_shuffle_payload(atom::A, Val(syms))`,
# without needing an atom instance. Must be kept in sync with that function so that
# host-side shared memory sizing and the device-side staged layout agree byte-for-byte.
@inline atom_payload_type(::Type{A}, ::Val{syms}) where {A, syms} = Tuple{map(s -> fieldtype(A, s), syms)...}

const WARPSIZE = UInt32(32)
# Values stored in `interacting_tiles_type`: a tile is either free of exclusions and
# special pairs (0), mask-backed (1), or known to hold no in-cutoff atom pair at all
# and skipped entirely by the force/energy kernels (`TILE_DEAD`)
const TILE_DEAD = UInt8(2)
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

Molly.uses_gpu_neighbor_finder(::Type{<:CuArray}) = true

CUDA_CORE.Const(nl::Molly.NoNeighborList) = nl

function env_int(name::AbstractString)
    value = ENV[name]
    parsed = tryparse(Int, value)
    parsed === nothing && error("invalid integer value for $(name): $(repr(value))")
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
@inline autotune_interaction_signature(inter) = hash(repr(inter))

function autotune_box_signature(boundary, ::Val{D}) where D
    sides = box_sides(boundary)
    return ntuple(i -> autotune_scalar(sides[i]), D)
end

gpu_neighbor_pairwise_inters(sys) = Tuple(filter(use_neighbors, values(sys.pairwise_inters)))

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
        cld(n_atoms, Int(WARPSIZE)),
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

# Arguments
- `prepare!`: A callable that prepares any necessary buffers or state before each run.
- `run!`: A callable that launches the CUDA kernel to be benchmarked.
- `warmup`: Number of initial runs to discard.
- `repeats`: Number of measured runs to take the minimum over.
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

function autotune_prepare_block_bounds!(buffers, sys::System{D, <:CuArray}, N::Int) where D
    n_blocks = cld(N, Int(WARPSIZE))
    @cuda blocks=n_blocks threads=32 kernel_min_max!(
        buffers.morton_seq,
        buffers.box_mins,
        buffers.box_maxs,
        sys.coords,
        Val(N),
        sys.boundary,
        Val(D),
    )
    CUDA.synchronize()
    return nothing
end

function autotune_prepare_common_state!(buffers, sys::System{D, <:CuArray}, N::Int) where D
    sorted_morton_seq!(buffers, sys.coords, sys.boundary)
    refresh_tile_exceptions!(buffers, sys.neighbor_finder, Val(N))
    buffers.sparse_pair_generation = sys.neighbor_finder.cache_generation
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
        Val(sys.neighbor_finder.dist_cutoff_2),
        Val(n_blocks),
        Val(D),
        max_tiles,
        buffers.block_exc_min,
        buffers.block_exc_max,
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
        Val(sys.neighbor_finder.dist_cutoff_2),
        Val(n_blocks),
        Val(D),
        max_tiles,
        buffers.block_exc_min,
        buffers.block_exc_max;
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
    if isempty(candidates)
        conf = launch_configuration(kernel.fun)
        push!(candidates, choose_tile_threads(min(conf.threads, max_threads)))
    end
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

"""
    autotune_tile_threads!(buffers, sys, N)

Benchmark block thread dimensions for the tile finding kernel (`find_interacting_blocks_kernel!`).
Tests candidates from `AUTOTUNE_TILE_THREAD_CANDIDATES` and returns the fastest `threads_xy`
that does not cause a tile buffer overflow.

# Arguments
- `buffers`: Temporary GPU buffers to manage state.
- `sys`: The system being benchmarked.
- `N`: Number of atoms.
"""
function autotune_tile_threads!(buffers, sys::System{D, <:CuArray}, N::Int) where D
    n_blocks = cld(N, Int(WARPSIZE))
    # Also grows the tile list to fit, so that the candidates below do not overflow it
    find_interacting_tiles!(buffers, sys, n_blocks)
    if use_tile_tree(sys, n_blocks, buffers.num_tile_candidates)
        # The tree search has no block shape to tune
        return nothing
    end
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
    buffers.num_pairs = Int(only(from_device(buffers.num_interacting_tiles)))
    buffers.num_tile_candidates = buffers.num_pairs
    return best_threads
end

function autotune_force_kernel(buffers, sys::System{D, <:CuArray, T, TH}, pairwise_inters,
                               N::Int, force_maxregs_override) where {D, T, TH}
    if force_maxregs_override === nothing
        return @cuda launch=false always_inline=true fastmath=pairwise_fastmath(T) force_kernel!(
            buffers.fs_mat_reordered,
            buffers.virial_slots,
            buffers.coords_reordered,
            buffers.velocities_reordered,
            buffers.atoms_reordered,
            Val(N),
            Val(kernel_pair_cutoff_2(sys, pairwise_inters)),
            Val(sys.force_units),
            pairwise_inters,
            sys.boundary,
            0,
            tile_exceptions(buffers, sys.neighbor_finder),
            Val(false),
            Val(T),
            Val(TH),
            Val(D),
            buffers.interacting_tiles_i,
            buffers.interacting_tiles_j,
            buffers.interacting_tiles_type,
            buffers.interacting_tiles_diag,
            buffers.num_interacting_tiles,
            buffers.interacting_tiles_overflow,
            nothing,
        )
    end

    fm = pairwise_fastmath(T)
    return @cuda launch=false maxregs=force_maxregs_override always_inline=true fastmath=fm force_kernel!(
        buffers.fs_mat_reordered,
        buffers.virial_slots,
        buffers.coords_reordered,
        buffers.velocities_reordered,
        buffers.atoms_reordered,
        Val(N),
        Val(kernel_pair_cutoff_2(sys, pairwise_inters)),
        Val(sys.force_units),
        pairwise_inters,
        sys.boundary,
        0,
        tile_exceptions(buffers, sys.neighbor_finder),
        Val(false),
        Val(T),
        Val(TH),
        Val(D),
        buffers.interacting_tiles_i,
        buffers.interacting_tiles_j,
        buffers.interacting_tiles_type,
        buffers.interacting_tiles_diag,
        buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow,
        nothing,
    )
end

# The Val(syms) of Atom fields the active interactions actually read, resolved once
# and shared between host-side shmem sizing and the device kernels so they can never
# drift apart (both call this same function on the same `pairwise_inters`/`A`).
@inline function resolved_atom_shuffle_syms(pairwise_inters, ::Type{A}) where {A}
    return Val(Molly.resolve_atom_fields(Molly.combine_atom_fields(pairwise_inters), A))
end

# Dynamic shared memory (bytes) for force_kernel!: must match the device-side
# CuDynamicSharedArray layout (opposites_sum + staged j coords/atoms/velocities)
function force_kernel_dynamic_shmem(buffers, ::Val{D}, ::Type{T}, uses_vel::Bool,
                                    block_y::Integer, pairwise_inters) where {D, T}
    nslot = 32 * block_y
    A = eltype(buffers.atoms_reordered)
    shuf_syms = resolved_atom_shuffle_syms(pairwise_inters, A)
    P = atom_payload_type(A, shuf_syms)
    JT = JStage{eltype(buffers.coords_reordered), P}
    bytes = nslot * sizeof(jforce_eltype(T, Val(D))) + nslot * sizeof(JT)
    if uses_vel
        bytes += nslot * sizeof(eltype(buffers.velocities_reordered))
    end
    return bytes
end

# Dynamic shared memory (bytes) for energy_kernel!: staged j coords/atoms (and
# velocities when an interaction uses them). Unlike force_kernel! there is no
# opposites_sum, since energy accumulates only the per-warp scalar `sum_E`.
function energy_kernel_dynamic_shmem(buffers, uses_vel::Bool, block_y::Integer, pairwise_inters)
    nslot = 32 * block_y
    A = eltype(buffers.atoms_reordered)
    shuf_syms = resolved_atom_shuffle_syms(pairwise_inters, A)
    P = atom_payload_type(A, shuf_syms)
    JT = JStage{eltype(buffers.coords_reordered), P}
    bytes = nslot * sizeof(JT)
    if uses_vel
        bytes += nslot * sizeof(eltype(buffers.velocities_reordered))
    end
    return bytes
end

# Dynamic shared memory above the default static per-block limit (48 KiB on
# current hardware) is only available if a kernel explicitly opts in; otherwise
# the launch fails with "invalid argument" even though the device supports much
# more (opt-in max is ~100 KiB on Ampere+). The j-atom staging in force_kernel!/
# energy_kernel! routinely needs more than 48 KiB for large `block_y` in Float64,
# so every launch site using dynamic shmem must opt in before calling the kernel.
# The limit is set with a call to the driver, so it is kept for each kernel and only set
#   when it changes
const kernel_dynamic_shmem = IdDict{Any, Int}()
const kernel_dynamic_shmem_lock = ReentrantLock()

function set_max_dynamic_shmem!(kernel, shmem::Integer)
    lock(kernel_dynamic_shmem_lock) do
        if get(kernel_dynamic_shmem, kernel, -1) != shmem
            CUDA.attributes(kernel.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
            kernel_dynamic_shmem[kernel] = shmem
        end
    end
    return nothing
end

# The number of threads per block that gives the most warps on a multiprocessor and the most
#   threads per block of a kernel, which take calls to the driver to find, kept for each
#   kernel
const kernel_thread_limits = IdDict{Any, Tuple{Int, Int}}()
const kernel_thread_limits_lock = ReentrantLock()

function thread_limits(kernel)
    return lock(kernel_thread_limits_lock) do
        get!(kernel_thread_limits, kernel) do
            (launch_configuration(kernel.fun).threads, CUDA.maxthreads(kernel))
        end
    end
end

"""
    autotune_force_block_y!(buffers, sys, pairwise_inters, N, force_maxregs_override)

Benchmark `block_y` sizes for the pairwise `force_kernel!`.
Returns the `block_y` configuration that achieves the minimum execution time.

# Arguments
- `buffers`: Temporary GPU buffers with the active interacting tile list.
- `sys`: The system being benchmarked.
- `pairwise_inters`: Pairwise interactions to calculate.
- `N`: Number of atoms.
- `force_maxregs_override`: Maximum number of registers per thread (or `nothing`).
"""
function autotune_force_block_y!(buffers, sys::System{D, <:CuArray, T, TH}, pairwise_inters,
                                 N::Int, force_maxregs_override) where {D, T, TH}
    kernel = autotune_force_kernel(buffers, sys, pairwise_inters, N, force_maxregs_override)
    candidates = autotune_block_y_candidates(kernel, 4, AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES)
    num_pairs = buffers.num_pairs
    num_pairs == 0 && return first(candidates)

    uses_vel = Molly.any_uses_velocity(pairwise_inters)
    best_block_y = first(candidates)
    best_ms = Inf
    for block_y in candidates
        n_blocks_launch = cld(num_pairs, block_y)
        shmem = force_kernel_dynamic_shmem(buffers, Val(D), T, uses_vel, block_y, pairwise_inters)
        set_max_dynamic_shmem!(kernel, shmem)
        ms = autotune_benchmark_ms!(
            () -> begin
                fill!(buffers.fs_mat_reordered, zero(T))
                fill!(buffers.virial_slots, zero(TH))
            end,
            () -> kernel(
                buffers.fs_mat_reordered,
                buffers.virial_slots,
                buffers.coords_reordered,
                buffers.velocities_reordered,
                buffers.atoms_reordered,
                Val(N),
                Val(kernel_pair_cutoff_2(sys, pairwise_inters)),
                Val(sys.force_units),
                pairwise_inters,
                sys.boundary,
                0,
                tile_exceptions(buffers, sys.neighbor_finder),
                Val(false),
                Val(T),
                Val(TH),
                Val(D),
                buffers.interacting_tiles_i,
                buffers.interacting_tiles_j,
                buffers.interacting_tiles_type,
                buffers.interacting_tiles_diag,
                buffers.num_interacting_tiles,
                buffers.interacting_tiles_overflow,
                nothing;
                threads=(32, block_y),
                blocks=n_blocks_launch,
                shmem=shmem,
            ),
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

# Arguments
- `buffers`: Temporary GPU buffers with the active interacting tile list.
- `sys`: The system being benchmarked.
- `pairwise_inters`: Pairwise interactions to calculate.
- `N`: Number of atoms.
"""
function autotune_energy_block_y!(buffers, sys::System{D, <:CuArray, T, TH}, pairwise_inters,
                                  N::Int) where {D, T, TH}
    kernel = @cuda launch=false always_inline=true fastmath=pairwise_fastmath(T) energy_kernel!(
        buffers.pe_vec_nounits,
        buffers.coords_reordered,
        buffers.velocities_reordered,
        buffers.atoms_reordered,
        Val(N),
        Val(kernel_pair_cutoff_2(sys, pairwise_inters)),
        Val(sys.energy_units),
        pairwise_inters,
        sys.boundary,
        0,
        tile_exceptions(buffers, sys.neighbor_finder),
        Val(T),
        Val(TH),
        Val(D),
        buffers.interacting_tiles_i,
        buffers.interacting_tiles_j,
        buffers.interacting_tiles_type,
        buffers.interacting_tiles_diag,
        buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow,
        nothing,
    )
    candidates = autotune_block_y_candidates(kernel, 4, AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES)
    num_pairs = buffers.num_pairs
    num_pairs == 0 && return first(candidates)

    uses_vel = Molly.any_uses_velocity(pairwise_inters)
    best_block_y = first(candidates)
    best_ms = Inf
    for block_y in candidates
        n_blocks_launch = cld(num_pairs, block_y)
        shmem = energy_kernel_dynamic_shmem(buffers, uses_vel, block_y, pairwise_inters)
        set_max_dynamic_shmem!(kernel, shmem)
        ms = autotune_benchmark_ms!(
            () -> fill!(buffers.pe_vec_nounits, zero(T)),
            () -> kernel(
                buffers.pe_vec_nounits,
                buffers.coords_reordered,
                buffers.velocities_reordered,
                buffers.atoms_reordered,
                Val(N),
                Val(kernel_pair_cutoff_2(sys, pairwise_inters)),
                Val(sys.energy_units),
                pairwise_inters,
                sys.boundary,
                0,
                tile_exceptions(buffers, sys.neighbor_finder),
                Val(T),
                Val(TH),
                Val(D),
                buffers.interacting_tiles_i,
                buffers.interacting_tiles_j,
                buffers.interacting_tiles_type,
                buffers.interacting_tiles_diag,
                buffers.num_interacting_tiles,
                buffers.interacting_tiles_overflow,
                nothing;
                blocks=n_blocks_launch,
                threads=(32, block_y),
                shmem=shmem,
            ),
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

Perform a full autotuning run for the CUDA pairwise kernels.
Sets up temporary GPU buffers and Morton/exception states, then individually tunes the tile search
threads, force `block_y`, and energy `block_y`. Returns a populated `CUDALaunchConfig`.

# Arguments
- `sys`: The system being benchmarked.
- `pairwise_inters`: Tuple of pairwise interaction types.
- `force_maxregs_override`: Maximum registers to use for the force kernel, or `nothing`.
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

function force_launch_params(sys, kernel)
    config = Molly.cuda_launch_config(sys)
    block_y_override = prefer_override(Molly.cuda_force_block_y(config), env_override("MOLLY_CUDA_FORCE_BLOCK_Y"))
    maxregs_override = prefer_override(Molly.cuda_force_maxregs(config), env_override("MOLLY_CUDA_FORCE_MAXREGS"))
    block_y_override === nothing || validate_block_y("MOLLY_CUDA_FORCE_BLOCK_Y", block_y_override)
    maxregs_override === nothing || maxregs_override > 0 || error("MOLLY_CUDA_FORCE_MAXREGS must be positive, got $(maxregs_override)")

    conf_threads, max_threads = thread_limits(kernel)
    # Ensure block_y does not exceed physical kernel limits
    max_block_y = fld(max_threads, Int(WARPSIZE))

    block_y = something(block_y_override, min(Int(WARPSIZE), 4, choose_block_y(conf_threads)))
    block_y = min(block_y, max_block_y)

    return (block_y, maxregs_override)
end

function energy_launch_params(sys, kernel)
    config = Molly.cuda_launch_config(sys)
    block_y_override = prefer_override(Molly.cuda_energy_block_y(config), env_override("MOLLY_CUDA_ENERGY_BLOCK_Y"))
    block_y_override === nothing || validate_block_y("MOLLY_CUDA_ENERGY_BLOCK_Y", block_y_override)

    conf = launch_configuration(kernel.fun)
    # Ensure block_y does not exceed physical kernel limits
    max_threads = CUDA.maxthreads(kernel)
    max_block_y = fld(max_threads, Int(WARPSIZE))

    block_y = something(block_y_override, min(Int(WARPSIZE), 4, choose_block_y(conf.threads)))
    block_y = min(block_y, max_block_y)
    
    return block_y
end

function tile_launch_params(sys, kernel)
    config = Molly.cuda_launch_config(sys)
    config_tile_threads = Molly.cuda_tile_threads(config)
    threads_x_override = config_tile_threads === nothing ? env_override("MOLLY_CUDA_TILE_THREADS_X") : config_tile_threads[1]
    threads_y_override = config_tile_threads === nothing ? env_override("MOLLY_CUDA_TILE_THREADS_Y") : config_tile_threads[2]
    if xor(threads_x_override === nothing, threads_y_override === nothing)
        error("set both MOLLY_CUDA_TILE_THREADS_X and MOLLY_CUDA_TILE_THREADS_Y together")
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

#=
Squared distance past which `force_kernel!`/`energy_kernel!` can skip a pair.
The tile list is built with the neighbor finder's buffered cutoff so that it stays
valid between rebuilds, but a pair only needs evaluating out to the largest cutoff
of the active interactions.
=#
function kernel_pair_cutoff_2(sys, pairwise_inters)
    nf_cutoff_2 = sys.neighbor_finder.dist_cutoff_2
    inter_cutoff_2 = Molly.max_zero_beyond(pairwise_inters)
    isnothing(inter_cutoff_2) && return nf_cutoff_2
    return min(inter_cutoff_2, nf_cutoff_2)
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
    error("maximum number of interacting tiles exceeded (> $(max_tiles)), increase buffer size")
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

CUDA_CORE.shfl_recurse(op, x::Quantity) = op(x.val) * unit(x)
CUDA_CORE.shfl_recurse(op, x::SVector{1, C}) where C = SVector{1, C}(op(x[1]))
CUDA_CORE.shfl_recurse(op, x::SVector{2, C}) where C = SVector{2, C}(op(x[1]), op(x[2]))
CUDA_CORE.shfl_recurse(op, x::SVector{3, C}) where C = SVector{3, C}(op(x[1]), op(x[2]), op(x[3]))

function Molly.pairwise_forces_loop_gpu!(buffers, sys::System{D, <:CuArray, T}, pairwise_inters,
                            nbs::Molly.NoNeighborList, step_n) where {D, T}
    kernel = @cuda launch=false fastmath=pairwise_fastmath(T) pairwise_force_kernel_nonl!(
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
Displacement check

The two largest squared displacements of the atoms since the reference coordinates, see
`Molly.neighbors_invalidated`, found by one kernel. Each block reduces its atoms with warp
shuffles, and the last block to finish reduces the results of all the blocks, decides
whether the atoms have moved further than the allowance and writes the result to pinned
host memory that the device can write to. It can also set a flag on the device that makes
the tiled force and energy kernels return straight away, so that those kernels can be
launched before the host knows the result. The host then only waits once the kernels are
queued, and the device is kept busy while it waits.
=#

const displacement_threads = 256
const displacement_max_blocks = 256

# The field types are concrete given `T`, so that the check is type stable although the
#   tracker holds the scratch without a type
mutable struct CUDADisplacementScratch{T}
    # Pinned host memory mapped into the device address space, holding the two largest
    #   squared displacements and whether they exceed the allowance, then the same for
    #   the inner masks of the tile list, see `InnerTileMasks`
    result::CuArray{T, 1, CUDA_CORE.HostMemory}
    result_host::Vector{T}
    # The results of each block and the count of finished blocks on the device, for the
    #   reference and for the inner masks
    block_maxs_1::CuArray{T, 1, CUDA_CORE.DeviceMemory}
    block_maxs_2::CuArray{T, 1, CUDA_CORE.DeviceMemory}
    inner_block_maxs_1::CuArray{T, 1, CUDA_CORE.DeviceMemory}
    inner_block_maxs_2::CuArray{T, 1, CUDA_CORE.DeviceMemory}
    counter::CuArray{Int32, 1, CUDA_CORE.DeviceMemory}
    event::CuEvent
    context::CuContext
end

function cuda_displacement_scratch(tracker, ::Type{T},
                                   n_blocks::Integer=displacement_max_blocks) where T
    scratch = (isnothing(tracker) ? nothing : tracker.scratch)
    if scratch isa CUDADisplacementScratch{T} && scratch.context == CUDA.context() &&
            length(scratch.block_maxs_1) >= n_blocks
        return scratch
    end
    result = CuArray{T, 1, CUDA_CORE.HostMemory}(undef, 6)
    result_host = unsafe_wrap(Array, result)
    counter = fill!(CuArray{Int32, 1, CUDA_CORE.DeviceMemory}(undef, 1), Int32(0))
    n_blocks_alloc = max(n_blocks, displacement_max_blocks)
    block_maxs = ntuple(_ -> CuArray{T, 1, CUDA_CORE.DeviceMemory}(undef, n_blocks_alloc), 4)
    scratch = CUDADisplacementScratch{T}(result, result_host, block_maxs..., counter,
                    CuEvent(CUDA_CORE.EVENT_DISABLE_TIMING), CUDA.context())
    if !isnothing(tracker)
        tracker.scratch = scratch
    end
    return scratch
end

@inline function warp_reduce_top2(max_1, max_2)
    offset = 16
    while offset > 0
        other_1 = CUDA.shfl_down_sync(0xFFFFFFFF, max_1, offset)
        other_2 = CUDA.shfl_down_sync(0xFFFFFFFF, max_2, offset)
        max_1, max_2 = Molly.merge_top2((max_1, max_2), (other_1, other_2))
        offset >>= 1
    end
    return max_1, max_2
end

# The two largest values over the block, correct in the first thread, called by every
#   thread of the block
# The lowest lane of each warp ends up with the two largest values of the warp, the
#   values that other lanes end up with are not used
@inline function block_reduce_top2(max_1, max_2, warp_maxs_1, warp_maxs_2, ::Val{T}) where T
    tid = threadIdx().x
    lane = (tid - 1) % 32 + 1
    warp = (tid - 1) ÷ 32 + 1
    max_1, max_2 = warp_reduce_top2(max_1, max_2)
    @inbounds if lane == 1
        warp_maxs_1[warp] = max_1
        warp_maxs_2[warp] = max_2
    end
    sync_threads()
    if warp == 1
        n_warps = blockDim().x ÷ 32
        @inbounds max_1 = (lane <= n_warps ? warp_maxs_1[lane] : zero(T))
        @inbounds max_2 = (lane <= n_warps ? warp_maxs_2[lane] : zero(T))
        max_1, max_2 = warp_reduce_top2(max_1, max_2)
    end
    sync_threads()
    return max_1, max_2
end

@inline set_skip_flag!(::Nothing) = nothing
@inline set_skip_flag!(skip_flag) = (@inbounds skip_flag[1] = Int32(1); nothing)

# The two largest values over all the blocks, from those each block wrote, correct in the
#   first thread
@inline function grid_reduce_top2(block_maxs_1, block_maxs_2, warp_maxs_1, warp_maxs_2,
                                  ::Val{T}) where T
    # Atomic reads, which are made in the device-wide cache, see the results of the other
    #   blocks
    max_1, max_2 = zero(T), zero(T)
    b = threadIdx().x
    @inbounds while b <= gridDim().x
        other_1 = CUDA.atomic_add!(pointer(block_maxs_1, b), zero(T))
        other_2 = CUDA.atomic_add!(pointer(block_maxs_2, b), zero(T))
        max_1, max_2 = Molly.merge_top2((max_1, max_2), (other_1, other_2))
        b += blockDim().x
    end
    return block_reduce_top2(max_1, max_2, warp_maxs_1, warp_maxs_2, Val(T))
end

#=
Called by every thread of a block with the two largest values of the block, writes them,
and the last block to finish reduces the values of all the blocks and writes the result.
`inner` is `nothing` or holds the check of the inner masks of the tile list, see
`InnerTileMasks`, made with the squared displacements `inner_d2` of the thread since the
masks were found: whether they exceed its allowance, or the masks have to be found again
whatever they are, is written to the result and to the flag on the device that the force
kernel reads.
=#
@inline function finish_displacement_check!(max_1, max_2, result, block_maxs_1, block_maxs_2,
                                            counter, skip_flag, allowance::T, ::Val{T},
                                            inner=nothing, inner_d2=zero(T)) where T
    tid = threadIdx().x
    block_i = blockIdx().x
    n_blocks = gridDim().x
    warp_maxs_1 = CuStaticSharedArray(T, 32)
    warp_maxs_2 = CuStaticSharedArray(T, 32)
    is_last = CuStaticSharedArray(Bool, 1)
    max_1, max_2 = block_reduce_top2(max_1, max_2, warp_maxs_1, warp_maxs_2, Val(T))
    if !isnothing(inner)
        inner_1, inner_2 = block_reduce_top2(inner_d2, zero(T), warp_maxs_1, warp_maxs_2,
                                             Val(T))
    end
    @inbounds if tid == 1
        block_maxs_1[block_i] = max_1
        block_maxs_2[block_i] = max_2
        if !isnothing(inner)
            inner.block_maxs_1[block_i] = inner_1
            inner.block_maxs_2[block_i] = inner_2
        end
        # The result is visible to every block before this block is counted as finished
        threadfence()
        n_finished = CUDA.atomic_add!(pointer(counter, 1), Int32(1))
        is_last[1] = (n_finished == n_blocks - Int32(1))
    end
    sync_threads()

    @inbounds if is_last[1]
        max_1, max_2 = grid_reduce_top2(block_maxs_1, block_maxs_2, warp_maxs_1, warp_maxs_2,
                                        Val(T))
        if !isnothing(inner)
            inner_1, inner_2 = grid_reduce_top2(inner.block_maxs_1, inner.block_maxs_2,
                                                warp_maxs_1, warp_maxs_2, Val(T))
        end
        if tid == 1
            invalid = !(sqrt(max_1) + sqrt(max_2) <= allowance)
            result[1] = max_1
            result[2] = max_2
            result[3] = (invalid ? one(T) : zero(T))
            if invalid
                set_skip_flag!(skip_flag)
            end
            if !isnothing(inner)
                stale = inner.force || !(sqrt(inner_1) + sqrt(inner_2) <= inner.allowance)
                result[4] = inner_1
                result[5] = inner_2
                result[6] = (stale ? one(T) : zero(T))
                inner.stale[1] = (stale ? Int32(1) : Int32(0))
            end
            # Ready for the next check
            counter[1] = Int32(0)
        end
    end
    return nothing
end

function max_sq_displacements_kernel!(result, block_maxs_1, block_maxs_2, counter,
                                      skip_flag, coords_var, ref_coords_var, boundary, F,
                                      dist_unit, n_atoms::Int32, allowance::T,
                                      ::Val{T}) where T
    coords = CUDA_CORE.Const(coords_var)
    ref_coords = CUDA_CORE.Const(ref_coords_var)
    stride = blockDim().x * gridDim().x
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    max_1, max_2 = zero(T), zero(T)
    @inbounds while i <= n_atoms
        d2 = Molly.sq_displacement(coords[i], ref_coords[i], boundary, F, dist_unit, T)
        max_1, max_2 = Molly.push_top2(max_1, max_2, d2)
        i += stride
    end
    finish_displacement_check!(max_1, max_2, result, block_maxs_1, block_maxs_2, counter,
                               skip_flag, allowance, Val(T))
    return nothing
end

#=
Reorder the coordinates and velocities into the current Morton order, as
`reorder_system_kernel!` does, and check the displacements of the atoms in the same pass,
setting the overflow flag of the tile list if they are too large. The tiled kernels
launched after it then return straight away.
=#
function reorder_check_kernel!(coords_reordered, velocities_reordered, coords_var,
                               velocities_var, seq_var, ref_coords_var, result, block_maxs_1,
                               block_maxs_2, counter, skip_flag, boundary, F, dist_unit,
                               allowance::T, inner, ::Val{N}, ::Val{T}) where {N, T}
    coords = CUDA_CORE.Const(coords_var)
    velocities = CUDA_CORE.Const(velocities_var)
    seq = CUDA_CORE.Const(seq_var)
    ref_coords = CUDA_CORE.Const(ref_coords_var)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    max_1 = zero(T)
    inner_d2 = zero(T)
    @inbounds if i <= N
        original_i = seq[i]
        c = coords[original_i]
        coords_reordered[i] = c
        velocities_reordered[i] = velocities[original_i]
        max_1 = Molly.sq_displacement(c, ref_coords[original_i], boundary, F, dist_unit, T)
        if !isnothing(inner)
            # The inner reference is in the Morton order
            inner_d2 = Molly.sq_displacement(c, inner.coords[i], boundary, inner.F,
                                             dist_unit, T)
        end
    end
    finish_displacement_check!(max_1, zero(T), result, block_maxs_1, block_maxs_2, counter,
                               skip_flag, allowance, Val(T), inner, inner_d2)
    return nothing
end

# `inner` is `nothing` or holds the coordinates the inner masks of the tile list were found
#   from, the box deformation since then and the allowance, the flag on the device and
#   whether the masks have to be found again whatever the displacements are, see
#   `InnerTileMasks`
function launch_reorder_check!(buffers, sys, tracker, ref_coords, F, dist_unit,
                               allowance::T, inner) where T
    N = length(sys.coords)
    n_blocks = max(cld(N, displacement_threads), 1)
    scratch = cuda_displacement_scratch(tracker, T, n_blocks)
    inner_args = (isnothing(inner) ? nothing :
                  (; inner..., block_maxs_1=scratch.inner_block_maxs_1,
                     block_maxs_2=scratch.inner_block_maxs_2))
    args = (buffers.coords_reordered, buffers.velocities_reordered, sys.coords,
            sys.velocities, buffers.morton_seq, ref_coords, scratch.result,
            scratch.block_maxs_1, scratch.block_maxs_2, scratch.counter,
            buffers.interacting_tiles_overflow, sys.boundary, F, dist_unit, allowance,
            inner_args, Val(N), Val(T))
    kernel = cached_cufunction(reorder_check_kernel!, args, Val(NamedTuple()))
    kernel(args...; threads=displacement_threads, blocks=n_blocks)
    CUDA.record(scratch.event, CUDA.stream())
    return scratch
end

#=
Launch the check without waiting for it, returning the scratch to wait on.
`allowance` is the total displacement of the two atoms that have moved furthest that the
list allows, and `skip_flag`, if given, is set on the device when it is exceeded.
=#
function launch_displacement_check!(tracker, coords::CuArray, ref_coords, boundary, F,
                                    dist_unit, ::Type{T}, allowance=T(Inf),
                                    skip_flag=nothing) where T
    n_atoms = length(coords)
    n_blocks = clamp(cld(n_atoms, displacement_threads), 1, displacement_max_blocks)
    scratch = cuda_displacement_scratch(tracker, T)
    @cuda threads=displacement_threads blocks=n_blocks max_sq_displacements_kernel!(
            scratch.result, scratch.block_maxs_1, scratch.block_maxs_2, scratch.counter,
            skip_flag, coords, ref_coords, boundary, F, dist_unit, Int32(n_atoms), T(allowance),
            Val(T))
    CUDA.record(scratch.event, CUDA.stream())
    return scratch
end

#=
Wait for a check, returning the two largest squared displacements and whether they
exceed the allowance.
The host spins on the event rather than calling `CUDA.synchronize`, which after a short
spin hands the wait to another thread and can take hundreds of microseconds to wake up
once the event completes. That is the usual case when the device is behind the host, as
for the neighbor finders that build a list on the GPU, where it made the check cost about
15% of a step. After a second the normal synchronization is used.
=#
function wait_displacement_check(scratch::CUDADisplacementScratch)
    event = scratch.event
    if !CUDA.isdone(event)
        t_start = time_ns()
        while !CUDA.isdone(event)
            ccall(:jl_cpu_pause, Cvoid, ())
            # Lets other threads collect garbage while this one waits
            ccall(:jl_gc_safepoint, Cvoid, ())
            if time_ns() - t_start > 1_000_000_000
                CUDA.synchronize(event)
                break
            end
        end
    end
    result = scratch.result_host
    @inbounds return result[1], result[2], !iszero(result[3])
end

function Molly.max_sq_displacements(coords::CuArray, ref_coords, boundary, F, dist_unit,
                                    ::Type{T}, n_threads::Integer, tracker=nothing) where T
    iszero(length(coords)) && return (zero(T), zero(T))
    scratch = launch_displacement_check!(tracker, coords, ref_coords, boundary, F,
                                         dist_unit, T)
    max_sq_1, max_sq_2, _ = wait_displacement_check(scratch)
    return max_sq_1, max_sq_2
end

#=
Dynamic pruning

The tile list holds the pairs within the cutoff of the neighbor finder at the coordinates
it was built from, which is longer than the cutoff of the interactions so that the list
stays valid for a number of steps. Many of the pairs it holds are beyond the cutoff of the
interactions at a given step. As in GROMACS, the force kernel instead uses masks of the
tiles found at a shorter inner cutoff from more recent coordinates, which hold fewer pairs
and leave some tiles with none, which are skipped. The inner masks stay valid while the
two atoms that have moved furthest since they were found have moved less in total than
the inner cutoff less that of the interactions, as for the list, so the check of the list
checks this too. When they are not valid the force kernel uses the masks of the list and
finds the inner masks again from the pairs it visits, which costs little more than not
pruning.
The masks of the list are kept, so a pair within the cutoff of the interactions is always
in them whilst the list is valid, and the inner masks are found from all of their pairs.
The energy kernel uses the masks of the list.

`masks` are the inner masks of the tiles, found at `coords`, in the Morton order of the
tile list, and `boundary` with the inner cutoff `cutoff`, which is without units. The
check of the list sets `stale` on the device when the force kernel has to find the masks
again. `valid` is whether the masks are for the current tile list, and `refresh_queued`
is whether `stale` was set by the host since the last check, so that the next force
kernel finds the masks.
=#
mutable struct InnerTileMasks{C, B, T}
    masks::CuArray{UInt32, 1, CUDA_CORE.DeviceMemory}
    coords::C
    boundary::B
    cutoff::T
    stale::CuArray{Int32, 1, CUDA_CORE.DeviceMemory}
    valid::Bool
    refresh_queued::Bool
end

#=
The coordinates and boundary that the tile list held in a set of buffers was built from,
kept in the buffers since a neighbor finder can be used with more than one set of
buffers, for example those of a simulation and those made by a call to `forces(sys)`.
`moved_last` and `moved_prev` are the total displacements of the two atoms that had moved
furthest at the last two checks, used to predict when the tile list will be rebuilt.
`pending_check` is a check that has been launched but not waited for, see
`run_tiled_kernel!`. `inner` holds the masks of the dynamic pruning, see `InnerTileMasks`,
or is `nothing` when it is not used.
=#
mutable struct TileReference{C, B, T}
    coords::C
    boundary::B
    moved_last::T
    moved_prev::T
    pending_check::Union{Nothing, CUDADisplacementScratch{T}}
    inner::Union{Nothing, InnerTileMasks{C, B, T}}
end

# The part of the gap between the cutoff of the interactions and that of the tile list
#   that is between the cutoff of the interactions and the inner cutoff. Smaller fractions
#   prune more pairs but the masks are found again more often.
const inner_cutoff_fraction = 0.2

# The inner cutoff of the dynamic pruning without units, or `nothing` if it is not used
function inner_tile_cutoff(sys, pairwise_inters, ::Type{T}) where T
    inter_sqdist = tile_inters_sqdist(sys, pairwise_inters)
    isnothing(inter_sqdist) && return nothing
    dist_unit = Molly.coords_length_unit(sys.coords)
    r_list = sys.neighbor_finder.dist_cutoff
    dimension(inter_sqdist) == dimension(dist_unit^2) || return nothing
    dimension(r_list) == dimension(dist_unit) || return nothing
    r_c = T(ustrip(dist_unit, sqrt(inter_sqdist)))
    r_list = T(ustrip(dist_unit, r_list))
    r_inner = r_c + T(inner_cutoff_fraction) * (r_list - r_c)
    return (r_c < r_inner < r_list ? r_inner : nothing)
end

#=
Set up the inner masks of `ref` for a tile list that has just been built, so that the next
force kernel finds them, reusing the memory of the previous ones if possible.
=#
function reset_inner_masks!(ref::TileReference{C, B, T}, buffers, sys,
                            pairwise_inters) where {C, B, T}
    cutoff = inner_tile_cutoff(sys, pairwise_inters, T)
    if isnothing(cutoff)
        ref.inner = nothing
        return ref
    end
    capacity = length(buffers.interacting_tiles_diag)
    inner = ref.inner
    if isnothing(inner) || length(inner.coords) != length(buffers.coords_reordered)
        inner = InnerTileMasks{C, B, T}(similar(buffers.interacting_tiles_diag, capacity),
                    similar(buffers.coords_reordered), sys.boundary, cutoff,
                    CuArray{Int32, 1, CUDA_CORE.DeviceMemory}(undef, 1), false, false)
        ref.inner = inner
    elseif length(inner.masks) < capacity
        inner.masks = similar(buffers.interacting_tiles_diag, capacity)
    end
    copyto!(inner.coords, buffers.coords_reordered)
    inner.boundary = sys.boundary
    inner.cutoff = cutoff
    fill!(inner.stale, Int32(1))
    inner.valid = false
    inner.refresh_queued = true
    return ref
end

# Record that a force kernel has been launched after the stale flag of the inner masks was
#   set by the host, so that it found them
function record_inner_refresh!(ref)
    isnothing(ref) && return ref
    inner = ref.inner
    if !isnothing(inner) && inner.refresh_queued
        inner.valid = true
        inner.refresh_queued = false
    end
    return ref
end

#=
The inner masks for the force kernel, see `InnerTileMasks`, or `nothing` if they are not
used. `r_cut2` is the squared cutoff of the force kernel, whose type the squared inner
cutoff is given in.
=#
function inner_kernel_args(buffers, sys, r_cut2)
    sys.neighbor_finder.auto_update || return nothing
    ref = tile_reference(buffers, sys)
    (isnothing(ref) || isnothing(ref.inner)) && return nothing
    inner = ref.inner
    dist_unit = Molly.coords_length_unit(sys.coords)
    r_inner = inner.cutoff * dist_unit
    return (masks=inner.masks, stale=inner.stale, r2=convert(typeof(r_cut2), r_inner^2),
            r=convert(typeof(sqrt(r_cut2)), r_inner))
end

#=
Whether the next check is expected to fail, extrapolating the displacements of the last
two checks. The tile list is then rebuilt without checking, which is always safe, rather
than checking and launching the force kernel before the result is known, since waiting
for a check that fails means the device is idle while the rebuild is launched. This only
affects speed.
=#
@inline function tile_check_expected_to_fail(ref::TileReference, allowance)
    predicted = ref.moved_last + max(ref.moved_last - ref.moved_prev, zero(ref.moved_last))
    return predicted > allowance
end

@inline function record_tile_check!(ref::TileReference, max_sq_1, max_sq_2)
    ref.moved_prev = ref.moved_last
    ref.moved_last = sqrt(max_sq_1) + sqrt(max_sq_2)
    return ref
end

# The squared cutoff of the pairwise interactions that use the tile list
tile_inters_sqdist(sys, pairwise_inters) = Molly.neighbor_inters_sqdist(pairwise_inters, (),
                                                Molly.coords_length_unit(sys.coords))

# Whether the tile list has to be rebuilt for reasons other than the atoms moving
@inline function tile_refresh_due(buffers, nf::GPUNeighborFinder)
    return buffers.step_n_preprocessed == -1 ||
           buffers.reorder_generation != nf.reorder_generation ||
           buffers.tile_generation != nf.tile_generation ||
           buffers.sparse_pair_generation != nf.cache_generation || !nf.initialized
end

#=
The reference held in `buffers` if it is for coordinates and a boundary of the types of
those of `sys`, otherwise `nothing`. The buffers hold the reference without a type, so
this makes the check type stable. A reference of another type, which would come from
using the buffers with another system, is replaced when the tile list is rebuilt.
=#
@inline function tile_reference(buffers, sys)
    T = Molly.coords_float_type(sys.coords)
    ref = buffers.neighbor_reference[]
    return (ref isa TileReference{typeof(sys.coords), typeof(sys.boundary), T} ? ref : nothing)
end

#=
Reorder the coordinates and velocities into the current Morton order and check whether the
atoms have moved far enough since the tile list in `buffers` was built that a pair within
the cutoff of `pairwise_inters` could be missing. The check sets the overflow flag of the
tile list if so, which makes the force and energy kernels return without doing anything,
so that they can be launched straight after. Returns the scratch to wait on, or `nothing`
if the tile list has to be rebuilt whatever the displacements are, as when the cutoff is
not known or when the check is expected to fail, in which case nothing is launched.
=#
function start_tile_check!(ref::TileReference, buffers, sys, nf, pairwise_inters,
                           uses_inner::Bool)
    T = Molly.coords_float_type(sys.coords)
    dist_unit = Molly.coords_length_unit(sys.coords)
    F = Molly.box_deformation(ref.boundary, sys.boundary, T)
    allowance = Molly.neighbor_allowance(sys.boundary, F, nf.dist_cutoff,
                                tile_inters_sqdist(sys, pairwise_inters), dist_unit, T)
    if isnothing(allowance) || length(ref.coords) != length(sys.coords)
        return nothing
    end
    # Rebuilt without checking, see above
    tile_check_expected_to_fail(ref, allowance) && return nothing
    nf.tracker.n_checks += 1
    inner = ref.inner
    if uses_inner && !isnothing(inner)
        inner_args = inner_check_args(inner, sys, pairwise_inters, dist_unit, T)
        # The check sets the flag of the inner masks from now on
        inner.refresh_queued = false
        return launch_reorder_check!(buffers, sys, nf.tracker, ref.coords, F, dist_unit,
                                     allowance, inner_args)
    else
        return launch_reorder_check!(buffers, sys, nf.tracker, ref.coords, F, dist_unit,
                                     allowance, nothing)
    end
end

#=
The inputs to the check of the inner masks for `reorder_check_kernel!`. The masks are
found again whatever the displacements are when they are not for the current tile list,
or when the box has changed in a way that is not a deformation.
=#
function inner_check_args(inner::InnerTileMasks, sys, pairwise_inters, dist_unit,
                          ::Type{T}) where T
    F = Molly.box_deformation(inner.boundary, sys.boundary, T)
    allowance = Molly.neighbor_allowance(sys.boundary, F, inner.cutoff * dist_unit,
                            tile_inters_sqdist(sys, pairwise_inters), dist_unit, T)
    force = !inner.valid || isnothing(allowance)
    return (coords=inner.coords, F=(ismissing(F) ? nothing : F),
            allowance=something(allowance, zero(T)), stale=inner.stale, force=force)
end

# Record that the tile list held in `buffers` was built from the current coordinates,
#   reusing the memory of the previous reference `ref` if possible
function record_tile_reference!(ref, buffers, sys, pairwise_inters)
    if !isnothing(ref) && length(ref.coords) == length(sys.coords)
        copyto!(ref.coords, sys.coords)
        ref.boundary = sys.boundary
        ref.moved_last = zero(ref.moved_last)
        ref.moved_prev = zero(ref.moved_prev)
        ref.pending_check = nothing
    else
        T = Molly.coords_float_type(sys.coords)
        ref = TileReference(copy(sys.coords), sys.boundary, zero(T), zero(T), nothing,
                            nothing)
        buffers.neighbor_reference[] = ref
    end
    reset_inner_masks!(ref, buffers, sys, pairwise_inters)
    return buffers
end

function record_gpu_tiles!(buffers, sys, nf::GPUNeighborFinder, pairwise_inters)
    nf.tracker.n_builds += 1
    nf.auto_update || return buffers
    if Molly.neighbors_trackable(sys.boundary, sys.coords, nf.dist_cutoff,
                                 tile_inters_sqdist(sys, pairwise_inters))
        record_tile_reference!(tile_reference(buffers, sys), buffers, sys, pairwise_inters)
    else
        # The tile list has to be rebuilt every time, so no reference is needed
        buffers.neighbor_reference[] = nothing
    end
    return buffers
end

# find_neighbors increases the rebuild counts of the neighbor finder when the Morton order
#   or the tile list is due to be rebuilt, so the cache follows the same rebuild points as
#   the lists of the other neighbor finders
# `moved` is whether the atoms have moved far enough to invalidate the tile list, in which
#   case the tile list and Morton order are rebuilt
@inline function gpu_neighbor_refresh_flags(buffers, nf::GPUNeighborFinder, moved::Bool)
    first_preprocess = (buffers.step_n_preprocessed == -1)
    needs_morton_refresh = (first_preprocess || moved ||
                            buffers.reorder_generation != nf.reorder_generation)
    sparse_changed = (buffers.sparse_pair_generation != nf.cache_generation)
    needs_reorder = true # Coordinates can be changed at any point
    needs_sparse_refresh = (needs_morton_refresh || !nf.initialized || sparse_changed)
    needs_tile_refresh = (needs_sparse_refresh ||
                          buffers.tile_generation != nf.tile_generation)
    return needs_morton_refresh, needs_reorder, needs_sparse_refresh, needs_tile_refresh
end

function record_neighbor_refresh!(buffers, nf::GPUNeighborFinder, step_n)
    buffers.reorder_generation = nf.reorder_generation
    buffers.tile_generation = nf.tile_generation
    buffers.step_n_preprocessed = step_n
    return buffers
end

# Bring the Morton order, the exception data, the reordered arrays and the tile list up to
#   date as required by the flags
function refresh_gpu_tiles!(buffers, sys, nf, pairwise_inters, step_n, flags,
                            sync_reorder::Bool)
    needs_morton_refresh, needs_reorder, needs_sparse_refresh, needs_tile_refresh = flags
    N = length(sys.coords)
    if needs_reorder || needs_sparse_refresh || needs_tile_refresh
        if needs_morton_refresh
            sorted_morton_seq!(buffers, sys.coords, sys.boundary)
        end

        if needs_sparse_refresh
            refresh_tile_exceptions!(buffers, nf, Val(N))
            nf.initialized = true
            buffers.sparse_pair_generation = nf.cache_generation
        end

        if needs_reorder
            reorder_system_gpu!(buffers, sys, Val(needs_morton_refresh))
            sync_reorder && KernelAbstractions.synchronize(get_backend(sys.coords))
        end

        if needs_tile_refresh
            refresh_interacting_tiles!(buffers, sys, N)
            record_gpu_tiles!(buffers, sys, nf, pairwise_inters)
        end
        record_neighbor_refresh!(buffers, nf, step_n)
    end
    return buffers
end

#=
Wait for the check pending on `ref` and record its result, returning whether the atoms
have moved too far for the tile list.
`refreshed_inner` is whether the check was of the inner masks too and the kernel launched
after it found them again if the check says so and the tile list is valid. The
coordinates it found them from are then recorded.
=#
function resolve_tile_check!(ref::TileReference, check::CUDADisplacementScratch, buffers,
                             sys, refreshed_inner::Bool)
    ref.pending_check = nothing
    max_sq_1, max_sq_2, moved = wait_displacement_check(check)
    record_tile_check!(ref, max_sq_1, max_sq_2)
    inner = ref.inner
    if refreshed_inner && !isnothing(inner) && !moved && !iszero(@inbounds check.result_host[6])
        copyto!(inner.coords, buffers.coords_reordered)
        inner.boundary = sys.boundary
        inner.valid = true
    end
    return moved
end

#=
Run `launch_kernel!()`, which launches the tiled force or energy kernel, with a tile list
that is up to date.
With `auto_update`, when no rebuild is due for other reasons, the displacement check is
made by the kernel that reorders the coordinates, and the kernel is launched speculatively
with the current tile list. If the check finds that the atoms have moved too far the
kernel returns straight away, and the tile list is rebuilt and the kernel launched again.
Otherwise the host only waits for the check once the kernel is queued, so the device does
not wait for the host. When the last checks show that the next one is likely to fail, the
tile list is rebuilt straight away instead.
With `defer`, the host does not wait for the check here but leaves it pending in the
reference, and `finish_tile_check!` must be called before the results of the kernel are
used. A check left pending, for example by an error before it was finished, is finished
at the next call.
`uses_inner` is whether the kernel uses the inner masks of the dynamic pruning, see
`InnerTileMasks`, in which case the check is of the inner masks too, and `refreshes_inner`
is whether it finds them again when they are stale, as the force kernel does and the
energy kernel does not.
=#
# Inlined so that the closure is not allocated
@inline function run_tiled_kernel!(launch_kernel!::K, buffers, sys, nf::GPUNeighborFinder,
                                   pairwise_inters, step_n, sync_reorder::Bool,
                                   uses_inner::Bool, refreshes_inner::Bool,
                                   defer::Bool=false) where K
    # Whether the atoms have moved too far for the tile list, the only reason to rebuild it
    #   that is not recorded by the generation counts
    moved = false
    ref = tile_reference(buffers, sys)
    if !isnothing(ref)
        stale_check = ref.pending_check
        if !isnothing(stale_check)
            moved = resolve_tile_check!(ref, stale_check, buffers, sys, false)
            # Whether the kernel found the inner masks again is not known
            isnothing(ref.inner) || (ref.inner.valid = false)
        end
    end
    if nf.auto_update && !moved && !tile_refresh_due(buffers, nf)
        if isnothing(ref)
            # There is no reference to check against
            moved = true
        else
            check = start_tile_check!(ref, buffers, sys, nf, pairwise_inters, uses_inner)
            if isnothing(check)
                # The check could not be made or is expected to fail
                moved = true
            else
                # The coordinates were reordered by the check
                sync_reorder && KernelAbstractions.synchronize(get_backend(sys.coords))
                launch_kernel!()
                if defer
                    ref.pending_check = check
                    return buffers
                end
                moved = resolve_tile_check!(ref, check, buffers, sys, refreshes_inner)
                # Otherwise the kernel did nothing, so it is run again below once the tile
                #   list is rebuilt, which also clears the overflow flag
                moved || return buffers
            end
        end
    end
    refresh_gpu_tiles!(buffers, sys, nf, pairwise_inters, step_n,
                       gpu_neighbor_refresh_flags(buffers, nf, moved), sync_reorder)
    launch_kernel!()
    (refreshes_inner && nf.auto_update) && record_inner_refresh!(tile_reference(buffers, sys))
    return buffers
end

#=
Wait for a check left pending by `run_tiled_kernel!` with `defer`. If the atoms have moved
too far the kernel launched before the check did nothing, so the tile list is rebuilt and
`launch_kernel!()` called again. Returns whether this was done.
=#
@inline function finish_tile_check!(launch_kernel!::K, buffers, sys, nf::GPUNeighborFinder,
                                    pairwise_inters, step_n, sync_reorder::Bool,
                                    uses_inner::Bool) where K
    ref = tile_reference(buffers, sys)
    isnothing(ref) && return false
    check = ref.pending_check
    isnothing(check) && return false
    moved = resolve_tile_check!(ref, check, buffers, sys, uses_inner)
    if moved
        refresh_gpu_tiles!(buffers, sys, nf, pairwise_inters, step_n,
                           gpu_neighbor_refresh_flags(buffers, nf, true), sync_reorder)
        launch_kernel!()
        uses_inner && record_inner_refresh!(tile_reference(buffers, sys))
    end
    return moved
end

function refresh_interacting_tiles!(buffers, sys::System{D, <:CuArray}, N::Int) where D
    n_blocks = cld(N, WARPSIZE)
    @cuda blocks=n_blocks threads=32 kernel_min_max!(
        buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords,
        Val(N), sys.boundary, Val(D))

    find_interacting_tiles!(buffers, sys, n_blocks)
    prune_interacting_tiles!(buffers, sys, N)
    return nothing
end

# Refine the bounding-box candidates with an exact atom-pair test, which also decides which
#   tiles hold exclusions or special pairs, and remove the tiles without an in-range pair
function prune_interacting_tiles!(buffers, sys, N::Int)
    n_blocks = cld(N, WARPSIZE)
    if buffers.num_pairs > 0
        prune_y = 4
        @cuda threads=(32, prune_y) blocks=cld(buffers.num_pairs, prune_y) always_inline=true prune_interacting_tiles_kernel!(
            buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.interacting_tiles_diag,
            buffers.num_interacting_tiles,
            buffers.coords_reordered, Val(N), Val(sys.neighbor_finder.dist_cutoff_2),
            Val(n_blocks), sys.boundary, tile_exceptions(buffers, sys.neighbor_finder))
        compact_interacting_tiles!(buffers)
    end
    return buffers
end

#=
Remove the tiles marked `TILE_DEAD` by `prune_interacting_tiles_kernel!` from the list,
keeping the order of the others, and set the tile counts to the number left.
The force and energy kernels run more than one tile in a thread block, which keeps its
place on a multiprocessor until its last warp finishes. The warp of a dead tile returns
straight away but is not replaced until the other tiles of its block are done, so with
dead tiles in the list fewer warps run at once. About a third of the candidate tiles of a
solvated protein are dead.
This runs when the list is rebuilt, and waits for the device to get the number of tiles.
=#
function compact_interacting_tiles!(buffers)
    n_tiles = buffers.num_pairs
    n_tiles == 0 && return buffers
    # The tiles are copied so that the compacted list can be written over the original
    tiles_i = buffers.interacting_tiles_i[1:n_tiles]
    tiles_j = buffers.interacting_tiles_j[1:n_tiles]
    tiles_type = buffers.interacting_tiles_type[1:n_tiles]
    tiles_diag = buffers.interacting_tiles_diag[1:n_tiles]
    positions = similar(buffers.interacting_tiles_i, Int32, n_tiles)
    n_threads = 256
    n_launch_blocks = cld(n_tiles, n_threads)
    @cuda threads=n_threads blocks=n_launch_blocks live_tile_flags_kernel!(
            positions, tiles_type, Int32(n_tiles))
    # The inclusive prefix sum of the flags is the place of each live tile in the list
    Molly.AcceleratedKernels.accumulate!(+, positions, get_backend(positions);
                                         init=Int32(0))
    @cuda threads=n_threads blocks=n_launch_blocks compact_tiles_kernel!(
            buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.interacting_tiles_diag,
            buffers.num_interacting_tiles, tiles_i, tiles_j, tiles_type, tiles_diag,
            positions, Int32(n_tiles))
    buffers.num_pairs = Int(only(from_device(view(positions, n_tiles:n_tiles))))
    foreach(CUDA.unsafe_free!, (tiles_i, tiles_j, tiles_type, tiles_diag, positions))
    return buffers
end

function live_tile_flags_kernel!(flags, tiles_type, n_tiles)
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds if t <= n_tiles
        flags[t] = (tiles_type[t] == TILE_DEAD ? Int32(0) : Int32(1))
    end
    return nothing
end

function compact_tiles_kernel!(tiles_i_out, tiles_j_out, tiles_type_out, tiles_diag_out,
                               num_interacting_tiles, tiles_i, tiles_j, tiles_type,
                               tiles_diag, positions, n_tiles)
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds if t <= n_tiles
        type = tiles_type[t]
        if type != TILE_DEAD
            p = positions[t]
            tiles_i_out[p] = tiles_i[t]
            tiles_j_out[p] = tiles_j[t]
            tiles_type_out[p] = type
            tiles_diag_out[p] = tiles_diag[t]
        end
        if t == n_tiles
            num_interacting_tiles[1] = positions[t]
        end
    end
    return nothing
end

#=
Build the list of interacting tiles and set `buffers.num_pairs` to its length.
The tile counter is incremented for every tile found, including those that do not fit,
so when the list overflows its vectors are grown to the number found and the search is
repeated. The vectors start at a size that is enough for typical systems, see
`init_buffers!`, and are kept for later searches.
=#
function find_interacting_tiles!(buffers, sys, n_blocks)
    use_tree = use_tile_tree(sys, n_blocks, buffers.num_tile_candidates)
    search! = use_tree ? find_interacting_tiles_tree! : find_interacting_tiles_all_pairs!
    reset_interacting_tile_state!(buffers)
    search!(buffers, sys, n_blocks)
    n_tiles = Int(only(from_device(buffers.num_interacting_tiles)))
    if n_tiles > length(buffers.interacting_tiles_i)
        grow_interacting_tiles!(buffers, n_tiles)
        reset_interacting_tile_state!(buffers)
        search!(buffers, sys, n_blocks)
        n_tiles = Int(only(from_device(buffers.num_interacting_tiles)))
        throw_if_interacting_tiles_overflowed(buffers)
    end
    buffers.num_pairs = n_tiles
    buffers.num_tile_candidates = n_tiles
    use_tree && sort_interacting_tiles!(buffers, n_tiles)
    return buffers
end

# Replace the interacting tile vectors with ones that hold at least `n_tiles` tiles,
#   with some headroom so that the list does not have to grow again straight away
function grow_interacting_tiles!(buffers, n_tiles)
    capacity = cld(5 * n_tiles, 4)
    buffers.interacting_tiles_i = similar(buffers.interacting_tiles_i, capacity)
    buffers.interacting_tiles_j = similar(buffers.interacting_tiles_j, capacity)
    buffers.interacting_tiles_type = similar(buffers.interacting_tiles_type, capacity)
    buffers.interacting_tiles_diag = similar(buffers.interacting_tiles_diag, capacity)
    return buffers
end

# Search every pair of blocks for interacting tiles
function find_interacting_tiles_all_pairs!(buffers, sys::System{D}, n_blocks) where D
    max_tiles = length(buffers.interacting_tiles_i)
    tile_kernel = @cuda launch=false find_interacting_blocks_kernel!(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, sys.boundary, Val(sys.neighbor_finder.dist_cutoff_2),
        Val(n_blocks), Val(D), max_tiles,
        buffers.block_exc_min, buffers.block_exc_max)
    tile_threads_xy = tile_launch_params(sys, tile_kernel)

    tile_kernel(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, sys.boundary, Val(sys.neighbor_finder.dist_cutoff_2),
        Val(n_blocks), Val(D), max_tiles,
        buffers.block_exc_min, buffers.block_exc_max;
        blocks=(cld(n_blocks, tile_threads_xy[1]), cld(n_blocks, tile_threads_xy[2])),
        threads=tile_threads_xy)
    return buffers
end

# Search for interacting tiles by walking a tree of block bounding boxes
function find_interacting_tiles_tree!(buffers, sys::System{D}, n_blocks) where D
    top, offsets = build_tile_tree!(buffers, n_blocks, Val(D))
    n_threads = 128
    @cuda threads=n_threads blocks=cld(n_blocks, n_threads) find_interacting_blocks_tree_kernel!(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, buffers.tree_mins, buffers.tree_maxs, offsets,
        Val(top), sys.boundary, Val(sys.neighbor_finder.dist_cutoff_2), Val(n_blocks), Val(D),
        length(buffers.interacting_tiles_i), buffers.block_exc_min, buffers.block_exc_max)
    return buffers
end

#=
The tree search appends tiles in the order that its threads find them, so tiles next to
each other in the list rarely share a block. `force_kernel!` runs consecutive tiles on
the warps of a thread block, and with that order it is 10-20% slower than with the tiles
of the all-pairs search, which appends the tiles of each 32x32 group of blocks column by
column. Sorting the tiles into that order gives the speed back. The sort key holds the
whole tile, so the tiles are read back from the sorted keys.
=#
function sort_interacting_tiles!(buffers, n_tiles)
    n_tiles == 0 && return buffers
    keys = similar(buffers.interacting_tiles_i, UInt64, n_tiles)
    temp = similar(keys)
    n_threads = 256
    n_launch_blocks = cld(n_tiles, n_threads)
    @cuda threads=n_threads blocks=n_launch_blocks tile_sort_keys_kernel!(
        keys, buffers.interacting_tiles_i, buffers.interacting_tiles_j,
        buffers.interacting_tiles_type, Int32(n_tiles))
    Molly.AcceleratedKernels.sort!(keys; temp=temp, block_size=256)
    @cuda threads=n_threads blocks=n_launch_blocks tiles_from_sort_keys_kernel!(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j,
        buffers.interacting_tiles_type, keys, Int32(n_tiles))
    CUDA.unsafe_free!(keys)
    CUDA.unsafe_free!(temp)
    return buffers
end

# Tile `(i, j)` ordered by the 32-block group of `j`, the 32-block group of `i`, `j` and
#   `i`, with the tile type, 0 or 1 from `emit_interacting_tile!`, in the lowest bit
# The groups take 26 bits each, enough for the Int32 block indices
@inline function tile_sort_key(i::Int32, j::Int32, type::UInt8)
    i0 = UInt64(i - Int32(1))
    j0 = UInt64(j - Int32(1))
    return ((j0 >> 5) << 37) | ((i0 >> 5) << 11) | ((j0 & 0x1f) << 6) | ((i0 & 0x1f) << 1) |
           UInt64(type & 0x01)
end

function tile_sort_keys_kernel!(keys, tiles_i, tiles_j, tiles_type, n_tiles)
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if t <= n_tiles
        @inbounds keys[t] = tile_sort_key(tiles_i[t], tiles_j[t], tiles_type[t])
    end
    return nothing
end

function tiles_from_sort_keys_kernel!(tiles_i, tiles_j, tiles_type, keys, n_tiles)
    t = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if t <= n_tiles
        @inbounds key = keys[t]
        i0 = (((key >> 11) & 0x03ffffff) << 5) | ((key >> 1) & 0x1f)
        j0 = ((key >> 37) << 5) | ((key >> 6) & 0x1f)
        @inbounds tiles_i[t] = (i0 % Int32) + Int32(1)
        @inbounds tiles_j[t] = (j0 % Int32) + Int32(1)
        @inbounds tiles_type[t] = (key & 0x01) % UInt8
    end
    return nothing
end

"""
    pairwise_forces_loop_gpu!(buffers, sys, pairwise_inters, nbs::Nothing, needs_vir, step_n)

Maintainer entry point for the CUDA tiled pairwise force path.

Pipeline:
1. Rebuild the Morton ordering and the Morton-ordered exception data when the
   atoms have moved far enough since the last rebuild to invalidate the tile list
   with `auto_update=true`, or when [`find_neighbors`](@ref) has marked them as due
   every `n_steps` steps of the [`GPUNeighborFinder`](@ref) with `auto_update=false`.
2. Reorder coordinates, velocities, and atoms into Morton order.
3. Recompute the compact list of interacting 32x32 tiles when the order changed
   or [`find_neighbors`](@ref) forced a rebuild, growing it if it overflows.
4. Launch `force_kernel!` over that compact tile list and reverse the reorder.

Cache contract:
- The Morton order and tile list are rebuilt when [`find_neighbors`](@ref) has
  increased the rebuild counts of the [`GPUNeighborFinder`](@ref) since they were
  last built, when `buffers.step_n_preprocessed` is `-1` for new buffers, or with
  `auto_update=true` when the displacement check against `buffers.neighbor_reference`
  fails. The force kernel is then launched before the result of the check is known,
  and the check sets `buffers.interacting_tiles_overflow` when it fails so that the
  kernel returns straight away, see `run_tiled_kernel!`.
- `gpu_forces!` calls `start_pairwise_forces_gpu!` instead, which does not wait for
  the check, and `finish_pairwise_forces_gpu!` once the specific forces are launched,
  which waits for it, calculates the forces again if it failed and reverses the reorder.
- `buffers.num_pairs` is the host-side cached interacting-tile count used to
  size the force-kernel launch.
- `sys.neighbor_finder.initialized` only indicates whether the Morton-ordered
  exception data is current.
"""
function Molly.pairwise_forces_loop_gpu!(buffers, sys::System{D, <:CuArray, T, TH}, pairwise_inters,
                         nbs::Nothing, ::Val{needs_vir}, step_n) where {D, T, TH, needs_vir}
    return tiled_pairwise_forces!(buffers, sys, pairwise_inters, Val(needs_vir), step_n,
                                  false)
end

# As above, but the host does not wait for the check of the tile list, which is left to
#   `finish_pairwise_forces_gpu!` once `gpu_forces!` has launched the specific forces.
#   The forces are added to fs_mat by `finish_pairwise_forces_gpu!`, so that the specific
#   forces can be added to it in the meantime, see `Molly.gpu_forces_concurrent!`.
function Molly.start_pairwise_forces_gpu!(buffers, sys::System{D, <:CuArray, T, TH},
                                          pairwise_inters, nbs::Nothing, ::Val{needs_vir},
                                          step_n) where {D, T, TH, needs_vir}
    return tiled_pairwise_forces!(buffers, sys, pairwise_inters, Val(needs_vir), step_n,
                                  true)
end

# If the check failed the kernel launched before it did nothing, and it is launched again
function Molly.finish_pairwise_forces_gpu!(buffers, sys::System{D, <:CuArray, T, TH},
                                           pairwise_inters, nbs::Nothing, ::Val{needs_vir},
                                           step_n) where {D, T, TH, needs_vir}
    r_cut2 = kernel_pair_cutoff_2(sys, pairwise_inters)
    launch_kernel! = () -> launch_force_kernel!(buffers, sys, pairwise_inters, r_cut2,
                                                Val(needs_vir), step_n)
    finish_tile_check!(launch_kernel!, buffers, sys, sys.neighbor_finder, pairwise_inters,
                       step_n, false, true)
    reverse_reorder_forces_gpu!(buffers, sys)
    return buffers
end

# With `defer` the forces are left in the reordered buffer, see above
function tiled_pairwise_forces!(buffers, sys, pairwise_inters, ::Val{needs_vir}, step_n,
                                defer::Bool) where needs_vir
    r_cut2 = kernel_pair_cutoff_2(sys, pairwise_inters)
    nf = sys.neighbor_finder
    launch_kernel! = () -> launch_force_kernel!(buffers, sys, pairwise_inters, r_cut2,
                                                Val(needs_vir), step_n)
    run_tiled_kernel!(launch_kernel!, buffers, sys, nf, pairwise_inters, step_n, false, true,
                      true, defer)
    defer || reverse_reorder_forces_gpu!(buffers, sys)
    return buffers
end

# Launch the force kernel over the list of interacting tiles
# `r_cut2` is a value rather than a `Val` so that the call to this function is not a
#   dynamic dispatch, as the launch inside it is anyway
function launch_force_kernel!(buffers, sys, pairwise_inters, r_cut2, ::Val{needs_vir},
                              step_n) where needs_vir
    inner = inner_kernel_args(buffers, sys, r_cut2)
    if isnothing(inner)
        return launch_force_kernel!(buffers, sys, pairwise_inters, r_cut2, Val(needs_vir),
                                    step_n, nothing)
    else
        return launch_force_kernel!(buffers, sys, pairwise_inters, r_cut2, Val(needs_vir),
                                    step_n, inner)
    end
end

function launch_force_kernel!(buffers, sys::System{D, <:CuArray, T, TH}, pairwise_inters,
                              r_cut2, ::Val{needs_vir}, step_n,
                              inner) where {D, T, TH, needs_vir}
    N = length(sys.coords)
    args = (buffers.fs_mat_reordered, buffers.virial_slots, buffers.coords_reordered,
            buffers.velocities_reordered, buffers.atoms_reordered, Val(N), Val(r_cut2),
            Val(sys.force_units), pairwise_inters, sys.boundary, step_n,
            tile_exceptions(buffers, sys.neighbor_finder), Val(needs_vir), Val(T), Val(TH),
            Val(D), buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.interacting_tiles_diag,
            buffers.num_interacting_tiles, buffers.interacting_tiles_overflow, inner)
    num_pairs = buffers.num_pairs
    num_pairs > 0 || return buffers
    auto_kernel = cached_cufunction(force_kernel!, args,
                                    Val((always_inline=true, fastmath=pairwise_fastmath(T))))
    block_y, maxregs = force_launch_params(sys, auto_kernel)
    n_blocks_launch = cld(num_pairs, block_y)

    kernel = if maxregs === nothing
        auto_kernel
    else
        @cuda launch=false maxregs=maxregs always_inline=true fastmath=pairwise_fastmath(T) force_kernel!(args...)
    end

    uses_vel = Molly.any_uses_velocity(pairwise_inters)
    shmem = force_kernel_dynamic_shmem(buffers, Val(D), T, uses_vel, block_y, pairwise_inters)
    set_max_dynamic_shmem!(kernel, shmem)
    kernel(args...; threads=(32, block_y), blocks=n_blocks_launch, shmem=shmem)
    return buffers
end

"""
    pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys, pairwise_inters, nbs::Nothing, step_n)

Maintainer entry point for the CUDA tiled pairwise energy path.

This follows the same Morton reorder -> exception translation -> tile search
pipeline as `pairwise_forces_loop_gpu!`, including the displacement check, but launches
`energy_kernel!` instead of the force kernel. Energy evaluation reuses any preprocessing
already performed since the last rebuild, so forces and energies share the same cached
tile metadata.
"""
function Molly.pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys::System{D, <:CuArray, T, TH},
                                      pairwise_inters, nbs::Nothing,
                                      step_n) where {D, T, TH}
    # The ordering and tiles are reused if they are current
    r_cut2 = kernel_pair_cutoff_2(sys, pairwise_inters)
    nf = sys.neighbor_finder
    launch_kernel! = () -> launch_energy_kernel!(pe_vec_nounits, buffers, sys, pairwise_inters,
                                                 r_cut2, step_n)
    # The energy kernel uses the inner masks when they are current but does not find them
    #   again when they are not
    run_tiled_kernel!(launch_kernel!, buffers, sys, nf, pairwise_inters, step_n, true, true,
                      false)
    return pe_vec_nounits
end

# Launch the energy kernel over the list of interacting tiles
function launch_energy_kernel!(pe_vec_nounits, buffers, sys::System{D, <:CuArray, T, TH},
                               pairwise_inters, r_cut2, step_n) where {D, T, TH}
    N = length(sys.coords)
    inner = inner_kernel_args(buffers, sys, r_cut2)
    kernel = @cuda launch=false always_inline=true fastmath=pairwise_fastmath(T) energy_kernel!(
            pe_vec_nounits, buffers.coords_reordered,
            buffers.velocities_reordered, buffers.atoms_reordered, Val(N), Val(r_cut2), Val(sys.energy_units), pairwise_inters,
            sys.boundary, step_n, tile_exceptions(buffers, sys.neighbor_finder),
            Val(T), Val(TH), Val(D), buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.interacting_tiles_diag,
            buffers.num_interacting_tiles, buffers.interacting_tiles_overflow, inner)
    block_y = energy_launch_params(sys, kernel)

    uses_vel = Molly.any_uses_velocity(pairwise_inters)
    shmem = energy_kernel_dynamic_shmem(buffers, uses_vel, block_y, pairwise_inters)
    set_max_dynamic_shmem!(kernel, shmem)

    num_pairs = buffers.num_pairs
    n_blocks_launch = num_pairs > 0 ? cld(num_pairs, block_y) : 1

    if num_pairs > 0
        kernel(
                pe_vec_nounits, buffers.coords_reordered,
                buffers.velocities_reordered, buffers.atoms_reordered, Val(N), Val(r_cut2), Val(sys.energy_units), pairwise_inters,
                sys.boundary, step_n, tile_exceptions(buffers, sys.neighbor_finder),
                Val(T), Val(TH), Val(D), buffers.interacting_tiles_i, buffers.interacting_tiles_j,
                buffers.interacting_tiles_type, buffers.interacting_tiles_diag,
                buffers.num_interacting_tiles, buffers.interacting_tiles_overflow, inner;
                blocks=n_blocks_launch, threads=(32, block_y), shmem=shmem)
    end
    return pe_vec_nounits
end

function boxes_dist(r1_min::SVector{D, T}, r1_max::SVector{D, T}, r2_min::SVector{D, T}, r2_max::SVector{D, T}, boundary) where {D, T}
    va = vector(r2_max, r1_min, boundary)
    vb = vector(r1_max, r2_min, boundary)
    a = SVector{D}(ntuple(d -> abs(va[d]), D))
    b = SVector{D}(ntuple(d -> abs(vb[d]), D))

    return SVector(ntuple(d -> r1_min[d] - r2_max[d] <= zero(T) && r2_min[d] - r1_max[d] <= zero(T) ? zero(T) : ifelse(a[d] < b[d], a[d], b[d]), D))
end

# Whether some periodic image of the bounding box of one block of atoms is within the cutoff
#   of the bounding box of another
@inline function boxes_within_cutoff(r1_min, r1_max, r2_min, r2_max, boundary, r_cut2)
    d_block = boxes_dist(r1_min, r1_max, r2_min, r2_max, boundary)
    return sum(d_block .* d_block) <= r_cut2
end

#=
The block bounds of a triclinic box are in the brick coordinates of `Molly.morton_coord`.
A displacement from a point in box 1 to a point in box 2 lies in the box
[r2_min - r1_max, r2_max - r1_min], and its periodic images add n1 * a + n2 * b + n3 * c for
integers n1, n2 and n3. As a has no y or z component and b has no z component, the z range of
an image only depends on n3 and the y range only depends on n2 and n3. So the images are
visited from z to y, over the integers that bring each range within the cutoff of zero,
skipping those that are already further away than the cutoff, and the closest image along
x is found directly. This is exact for the boxes, and since the distance between two boxes
only gets smaller as a box grows, it can also be used for the tree search.
The image counts are floats, so that the loops are empty rather than erroring for NaN.
=#
function boxes_within_cutoff(r1_min, r1_max, r2_min, r2_max, boundary::TriclinicBoundary,
                             r_cut2)
    a, b, c = boundary.basis_vectors
    lo = r2_min - r1_max
    hi = r2_max - r1_min
    r_cut = sqrt(r_cut2)
    n3 = ceil((-r_cut - hi[3]) / c[3])
    n3_max = floor((r_cut - lo[3]) / c[3])
    while n3 <= n3_max
        dz2 = interval_dist2(lo[3] + n3 * c[3], hi[3] + n3 * c[3])
        y_lo = lo[2] + n3 * c[2]
        y_hi = hi[2] + n3 * c[2]
        n2 = ceil((-r_cut - y_hi) / b[2])
        n2_max = floor((r_cut - y_lo) / b[2])
        while n2 <= n2_max
            dyz2 = dz2 + interval_dist2(y_lo + n2 * b[2], y_hi + n2 * b[2])
            if dyz2 <= r_cut2
                x_shift = n3 * c[1] + n2 * b[1]
                dx2 = periodic_interval_dist2(lo[1] + x_shift, hi[1] + x_shift, a[1])
                if dyz2 + dx2 <= r_cut2
                    return true
                end
            end
            n2 += one(n2)
        end
        n3 += one(n3)
    end
    return false
end

# The squared distance from zero to the interval [lo, hi]
@inline function interval_dist2(lo, hi)
    d = max(lo, -hi, zero(lo))
    return d * d
end

# The squared distance from zero to the closest of the intervals [lo, hi] + n * side
@inline function periodic_interval_dist2(lo, hi, side)
    lo_wrap = lo - side * floor(lo / side)
    hi_wrap = lo_wrap + (hi - lo)
    d = ifelse(hi_wrap >= side, zero(lo), min(lo_wrap, side - hi_wrap))
    return d * d
end

"""
    reorder_system_gpu!(buffers, sys)

Reorders `coords`, `velocities`, and `atoms` into `_reordered` buffers 
according to the Morton sequence to improve spatial cache locality.
"""
function reorder_system_gpu!(buffers, sys)
    return reorder_system_gpu!(buffers, sys, Val(true))
end

function reorder_system_gpu!(buffers, sys, ::Val{reorder_atoms}) where reorder_atoms
    N = length(sys)
    n_threads = 256
    n_blocks = cld(N, n_threads)

    @cuda threads=n_threads blocks=n_blocks reorder_system_kernel!(
        buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
        sys.coords, sys.velocities, sys.atoms, buffers.morton_seq, Val(N), Val(reorder_atoms))

    return nothing
end

function reorder_system_kernel!(coords_reordered, velocities_reordered, atoms_reordered,
                                coords_var, velocities_var, atoms_var, seq_var,
                                ::Val{N}, ::Val{reorder_atoms}) where {N, reorder_atoms}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    coords = CUDA_CORE.Const(coords_var)
    velocities = CUDA_CORE.Const(velocities_var)
    atoms = CUDA_CORE.Const(atoms_var)
    seq = CUDA_CORE.Const(seq_var)

    @inbounds if i <= N
        original_i = seq[i]
        coords_reordered[i] = coords[original_i]
        velocities_reordered[i] = velocities[original_i]
        if reorder_atoms
            atoms_reordered[i] = atoms[original_i]
        end
    end
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
    Molly.launch_kernel!(Molly.reverse_reorder_forces_kernel!(backend, n_threads),
        buffers.fs_mat, buffers.fs_mat_reordered, buffers.morton_seq, Val(D); ndrange=N)
    
    return nothing
end

"""
    kernel_min_max!(sorted_seq, mins, maxs, coords, n_atoms, boundary, D)

Compute the minimum and maximum coordinates for each 32-atom block.
These bounds are used for fast bounding-box intersection tests during tile finding
to skip non-interacting tile pairs.
For a triclinic boundary the coordinates are first wrapped into the brick of the
diagonal components of the basis vectors, see `Molly.morton_coord`.
"""
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
    sorted_seq_ro = CUDA_CORE.Const(sorted_seq)
    coords = CUDA_CORE.Const(coords_var)
    mins_smem = CuStaticSharedArray(C, (D32, b))
    maxs_smem = CuStaticSharedArray(C, (D32, b))
    r_smem = CuStaticSharedArray(C, (r, b))

    if i <= n - r && local_i <= D32
        @inbounds s_i = sorted_seq_ro[i]
        @inbounds c_i = Molly.morton_coord(coords[s_i], boundary)
        for k in a:b
            @inbounds val = c_i[k]
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
            # Level p reads slots written by other lanes at level p - 1. Lanes of a
            # warp are not guaranteed to run in lockstep (independent thread
            # scheduling since Volta), so a warp barrier is required between levels.
            # The enclosing branch is uniform across the block (n - r is a multiple
            # of 32 and threads=32), so all lanes reach this barrier.
            sync_warp()
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
        @inbounds c_i = Molly.morton_coord(coords[s_i], boundary)
        for k in a:b
            @inbounds r_smem[local_i, k] = c_i[k]
        end
    end
    xyz_min = CuStaticSharedArray(C, b)
    xyz_max = CuStaticSharedArray(C, b)
    @inbounds for k in a:b
        xyz_min[k] =  10 * box_sides(boundary, k) # very large (arbitrary) value
        xyz_max[k] = -10 * box_sides(boundary, k)
    end
    # Publish the r_smem writes from lanes 1:r and the xyz_min/xyz_max initialisation
    # by all lanes before lane 1 reads and updates them.
    sync_threads()
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

"""
    update_inv_morton_kernel!(inv_morton_seq, morton_seq, N)

Generate the inverse Morton mapping.
Scatters the dense `1:N` original atom indices into their new Morton-ordered
positions in `inv_morton_seq`.
"""
function update_inv_morton_kernel!(inv_morton_seq, morton_seq, ::Val{N}) where N
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    morton_seq_ro = CUDA_CORE.Const(morton_seq)
    if i <= N
        @inbounds inv_morton_seq[morton_seq_ro[i]] = i
    end
    return nothing
end

# Mask state of tile `(i, j)` before any exclusions or special pairs are applied:
# everything eligible apart from diagonal self-interactions and out-of-bounds
# boundary masking, and nothing special.
# Row `lane` of a tile describes atom `lane` of block `i` and bit `32 - s` of the row
# describes atom `s` of block `j`.
@inline function pristine_tile_masks(i, j, lane, n_blocks, r)
    eligible_bitmask = UInt32(0xFFFFFFFF)

    # Boundary Masking
    if j == n_blocks
        mask = ifelse(r == Int32(32), UInt32(0xFFFFFFFF), (UInt32(1) << r) - UInt32(1))
        eligible_bitmask = (mask << (Int32(32) - r))
    end

    # Diagonal Self-Interactions
    if i == j
        eligible_bitmask &= ~(UInt32(1) << (Int32(32) - lane))
    end

    return eligible_bitmask, UInt32(0x00000000)
end

#=
The exclusions and special pairs read by the tiled kernels.
The per-atom lists of `nf.eligible` and `nf.special`, which are SparsePairMatrix values
on the device, are indexed by original atom index. `buffers.excluded_pos` and
`buffers.special_pos` hold, for each list entry, the Morton position of the partner atom,
refreshed by `refresh_tile_exceptions!` whenever the Morton order changes. This takes
memory proportional to the number of exceptions, where a mask per tile takes memory
proportional to `n_atoms^2`.
The NamedTuple is converted to device arrays when a kernel is launched.
=#
function tile_exceptions(buffers, nf::GPUNeighborFinder)
    return (
        morton_seq=buffers.morton_seq,
        excluded_starts=nf.eligible.starts,
        excluded_pos=buffers.excluded_pos,
        special_starts=nf.special.starts,
        special_pos=buffers.special_pos,
    )
end

"""
    refresh_tile_exceptions!(buffers, nf, N)

Translate the sparse exception lists stored on [`GPUNeighborFinder`](@ref) into the
Morton-ordered form read by the tiled CUDA pairwise kernels.

This needs work proportional to the number of atoms and exceptions: the inverse Morton
map is updated, the partner of each exception is converted to its Morton position and
the range of blocks holding the partners of the atoms of each 32-atom block is found,
which lets the tile search mark most tiles as free of exceptions without looking at
the atoms.
"""
function refresh_tile_exceptions!(buffers, nf::GPUNeighborFinder, ::Val{N}) where N
    n_blocks = cld(N, 32)
    @cuda threads=256 blocks=cld(N, 256) update_inv_morton_kernel!(
        buffers.morton_seq_inv, buffers.morton_seq, Val(N))

    # The number of exceptions changes when pairs are added to the neighbor finder
    if length(buffers.excluded_pos) != length(nf.eligible.partners)
        buffers.excluded_pos = similar(buffers.excluded_pos, length(nf.eligible.partners))
    end
    if length(buffers.special_pos) != length(nf.special.partners)
        buffers.special_pos = similar(buffers.special_pos, length(nf.special.partners))
    end
    if length(buffers.block_exc_min) != n_blocks
        buffers.block_exc_min = similar(buffers.block_exc_min, n_blocks)
        buffers.block_exc_max = similar(buffers.block_exc_max, n_blocks)
    end

    n_threads = 256
    @cuda threads=n_threads blocks=cld(n_blocks * 32, n_threads) tile_exceptions_kernel!(
        buffers.excluded_pos, buffers.special_pos, buffers.block_exc_min,
        buffers.block_exc_max, nf.eligible.starts, nf.eligible.partners,
        nf.special.starts, nf.special.partners, buffers.morton_seq,
        buffers.morton_seq_inv, Val(N), Val(n_blocks))
    return buffers
end

# Convert the partners of one atom's exception list to Morton positions, returning the
#   lowest and highest block they fall in
@inline function translate_exception_list!(pos, starts, partners, morton_seq_inv, atom_i,
                                           min_block, max_block)
    @inbounds k_start, k_end = starts[atom_i], starts[atom_i + Int32(1)]
    k = k_start
    @inbounds while k < k_end
        q = morton_seq_inv[partners[k]]
        pos[k] = q
        block_q = ((q - Int32(1)) >> 5) + Int32(1)
        min_block = min(min_block, block_q)
        max_block = max(max_block, block_q)
        k += Int32(1)
    end
    return min_block, max_block
end

#=
One thread per Morton position `p`, so that each warp covers one 32-atom block.
The thread converts the exception partners of the atom at `p` to Morton positions,
writing each list entry once since each atom belongs to one position. The warp then
reduces the lowest and highest block holding a partner of any atom in the block, which
is `typemax(Int32)` and `0` for a block without exceptions.
=#
function tile_exceptions_kernel!(excluded_pos, special_pos, block_exc_min, block_exc_max,
                                 excluded_starts, excluded_partners, special_starts,
                                 special_partners, morton_seq, morton_seq_inv,
                                 ::Val{N}, ::Val{n_blocks}) where {N, n_blocks}
    p = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    min_block = typemax(Int32)
    max_block = Int32(0)
    if p <= N
        @inbounds atom_i = morton_seq[p]
        min_block, max_block = translate_exception_list!(excluded_pos, excluded_starts,
                    excluded_partners, morton_seq_inv, atom_i, min_block, max_block)
        min_block, max_block = translate_exception_list!(special_pos, special_starts,
                    special_partners, morton_seq_inv, atom_i, min_block, max_block)
    end

    # Every thread of the warp takes part, including those past the last atom
    offset = Int32(16)
    while offset > Int32(0)
        min_block = min(min_block, CUDA.shfl_xor_sync(0xFFFFFFFF, min_block, offset))
        max_block = max(max_block, CUDA.shfl_xor_sync(0xFFFFFFFF, max_block, offset))
        offset >>= Int32(1)
    end

    block_i = ((p - Int32(1)) >> 5) + Int32(1)
    if laneid() == Int32(1) && block_i <= n_blocks
        @inbounds block_exc_min[block_i] = min_block
        @inbounds block_exc_max[block_i] = max_block
    end
    return nothing
end

# Whether the atom at Morton position `p` has an exclusion or special pair with an atom
#   in block `block_i`
@inline function atom_has_partner_in_block(tile_exceptions, p, block_i)
    morton_seq = CUDA_CORE.Const(tile_exceptions.morton_seq)
    excluded_starts = CUDA_CORE.Const(tile_exceptions.excluded_starts)
    excluded_pos = CUDA_CORE.Const(tile_exceptions.excluded_pos)
    special_starts = CUDA_CORE.Const(tile_exceptions.special_starts)
    special_pos = CUDA_CORE.Const(tile_exceptions.special_pos)

    @inbounds atom_i = morton_seq[p]
    found = false
    @inbounds for k in excluded_starts[atom_i]:(excluded_starts[atom_i + Int32(1)] - Int32(1))
        found |= (((excluded_pos[k] - Int32(1)) >> 5) + Int32(1) == block_i)
    end
    @inbounds for k in special_starts[atom_i]:(special_starts[atom_i + Int32(1)] - Int32(1))
        found |= (((special_pos[k] - Int32(1)) >> 5) + Int32(1) == block_i)
    end
    return found
end

#=
The eligible and special bitmask rows of tile `(i, j)` for atom `lane` of block `i`,
built from the exceptions of that atom.
This replaces a stored mask per tile. It is only called for the tiles that can contain
an exception, i.e. the few tiles that `prune_interacting_tiles_kernel!` marks as masked
and the diagonal and boundary tiles, and each atom only has a few exceptions.
Bits for pairs that a kernel does not evaluate, such as the pairs below the diagonal of
a diagonal tile, may be set as the exception lists hold each pair under both atoms.
=#
@inline function tile_exception_masks(tile_exceptions, i, j, lane, n_blocks, r,
                                      ::Val{N}) where N
    eligible_bitmask, special_bitmask = pristine_tile_masks(i, j, lane, n_blocks, r)
    p = (i - Int32(1)) * Int32(32) + lane
    if p <= N
        morton_seq = CUDA_CORE.Const(tile_exceptions.morton_seq)
        excluded_starts = CUDA_CORE.Const(tile_exceptions.excluded_starts)
        excluded_pos = CUDA_CORE.Const(tile_exceptions.excluded_pos)
        special_starts = CUDA_CORE.Const(tile_exceptions.special_starts)
        special_pos = CUDA_CORE.Const(tile_exceptions.special_pos)

        @inbounds atom_i = morton_seq[p]
        j_0 = (j - Int32(1)) * Int32(32)
        @inbounds for k in excluded_starts[atom_i]:(excluded_starts[atom_i + Int32(1)] - Int32(1))
            slot = excluded_pos[k] - j_0
            if Int32(1) <= slot <= Int32(32)
                eligible_bitmask &= ~(UInt32(1) << (Int32(32) - slot))
            end
        end
        @inbounds for k in special_starts[atom_i]:(special_starts[atom_i + Int32(1)] - Int32(1))
            slot = special_pos[k] - j_0
            if Int32(1) <= slot <= Int32(32)
                special_bitmask |= UInt32(1) << (Int32(32) - slot)
            end
        end
    end
    return eligible_bitmask, special_bitmask
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


"""
    find_interacting_blocks_kernel!(interacting_tiles_i, interacting_tiles_j,
                                    interacting_tiles_type, num_interacting_tiles,
                                    interacting_tiles_overflow, mins, maxs, boundary,
                                    r_cut2, N_blocks, D, max_total_tiles,
                                    block_exc_min, block_exc_max)

Scan the upper-triangular matrix of 32x32 Morton-ordered atom tiles and append
only those whose bounding boxes fall within `r_cut`.

A full off-diagonal tile is marked clean, i.e. free of exclusions and special pairs,
unless each block lies in the range of blocks holding exception partners of the
other, see `refresh_tile_exceptions!`. That test is conservative and
`prune_interacting_tiles_kernel!` marks the tiles that pass it but hold no exception
as clean.
"""
function find_interacting_blocks_kernel!(
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
    interacting_tiles_overflow,
    mins::AbstractArray{C}, maxs::AbstractArray{C}, boundary, ::Val{r_cut2}, ::Val{N_blocks}, ::Val{D}, max_total_tiles,
    block_exc_min, block_exc_max
) where {C, r_cut2, N_blocks, D}
    mins_ro = CUDA_CORE.Const(mins)
    maxs_ro = CUDA_CORE.Const(maxs)
    block_exc_min_ro = CUDA_CORE.Const(block_exc_min)
    block_exc_max_ro = CUDA_CORE.Const(block_exc_max)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= N_blocks && j <= N_blocks && i <= j
        @inbounds r_min_i = SVector{D}(ntuple(d -> mins_ro[i, d], D))
        @inbounds r_max_i = SVector{D}(ntuple(d -> maxs_ro[i, d], D))
        @inbounds r_min_j = SVector{D}(ntuple(d -> mins_ro[j, d], D))
        @inbounds r_max_j = SVector{D}(ntuple(d -> maxs_ro[j, d], D))

        if boxes_within_cutoff(r_min_i, r_max_i, r_min_j, r_max_j, boundary, r_cut2)
            emit_interacting_tile!(interacting_tiles_i, interacting_tiles_j,
                                   interacting_tiles_type, num_interacting_tiles,
                                   interacting_tiles_overflow, max_total_tiles,
                                   block_exc_min_ro, block_exc_max_ro, Int32(i), Int32(j),
                                   Int32(N_blocks))
        end
    end
    return nothing
end

# Append tile `(i, j)`, `i <= j`, to the interacting tile list, marking it clean if it is
#   a full off-diagonal tile and the block ranges of `refresh_tile_exceptions!` rule out
#   an exception in it
@inline function emit_interacting_tile!(interacting_tiles_i, interacting_tiles_j,
                                        interacting_tiles_type, num_interacting_tiles,
                                        interacting_tiles_overflow, max_total_tiles,
                                        block_exc_min, block_exc_max, i, j, N_blocks)
    is_clean = (i < j) && (j < N_blocks)
    if is_clean
        @inbounds may_have_exception = (block_exc_min[i] <= j <= block_exc_max[i]) &&
                                       (block_exc_min[j] <= i <= block_exc_max[j])
        is_clean = !may_have_exception
    end

    idx = CUDA.atomic_add!(pointer(num_interacting_tiles, 1), Int32(1)) + Int32(1)
    if idx <= max_total_tiles
        @inbounds interacting_tiles_i[idx] = i
        @inbounds interacting_tiles_j[idx] = j
        @inbounds interacting_tiles_type[idx] = is_clean ? UInt8(0) : UInt8(1)
    else
        CUDA.atomic_add!(pointer(interacting_tiles_overflow, 1), Int32(1))
    end
    return nothing
end

#=
Tree search for the interacting tiles.

The brute-force search above tests all `n_blocks^2 / 2` pairs of blocks, which dominates
the run time from about a million atoms. Since the blocks are consecutive runs of
Morton-ordered atoms, runs of consecutive blocks are spatially compact, so the bounding
boxes of the blocks can be merged pairwise into an implicit binary tree: level 0 holds
the blocks and node `n` of level `k` holds the union of the boxes of blocks
`(n - 1) * 2^k + 1` to `n * 2^k`. One thread per block `i` then walks the tree from the
root, skipping every subtree whose box is further than the cutoff from block `i` or
that only holds blocks `j < i`, which takes time proportional to the number of
interacting tiles times the depth of the tree.

The box distance only gets smaller as a box grows, so a subtree that is skipped holds
no block that the brute-force search would have accepted and the two searches give the
same tiles. This holds for triclinic boundaries too, see `boxes_within_cutoff`. The
tiles are then sorted into the order of the brute-force search, see
`sort_interacting_tiles!`.
=#

#=
Whether to use the tree search rather than the all-pairs search, given the number of
tiles found by the previous search, or 0 before the first search.
The all-pairs search takes time in proportion to the square of the number of blocks.
Below about a million atoms there are too few blocks to keep the GPU busy with one thread
walking the tree per block, so the time of the tree search and of the sort of its tiles
depends less on the number of blocks than on the tiles per block, which set the length of
each walk. The two take the same time at a number of blocks that grows with the square
root of the tiles per block. Measured as the time per step on an RTX A6000 for liquid
argon, water and protein in water with cutoffs of 0.7-2.0 nm, giving 40-260 tiles per
block, it is 4,600-9,700 blocks for orthorhombic boxes and 3,200-5,400 blocks for
truncated octahedra, where the box test costs more, see `boxes_within_cutoff`. The
expressions below are within 10% of most of these and at most 35% above the rest, which
costs at most 2% of the time per step since the searches take similar times there.
Before the first search, the tiles per block of a solvated protein with a cutoff of
1.2 nm are assumed.
=#
const TILE_TREE_BLOCKS_PER_SQRT_TILE = 750
const TILE_TREE_BLOCKS_PER_SQRT_TILE_TRICLINIC = 470
const TILE_TREE_DEFAULT_TILES_PER_BLOCK = 100

function use_tile_tree(sys, n_blocks, n_tiles_previous)
    min_blocks = env_override("MOLLY_CUDA_TILE_TREE_MIN_BLOCKS")
    if min_blocks === nothing
        tiles_per_block = (n_tiles_previous > 0 ? n_tiles_previous / n_blocks :
                                                  TILE_TREE_DEFAULT_TILES_PER_BLOCK)
        blocks_per_sqrt_tile = (sys.boundary isa TriclinicBoundary ?
                                TILE_TREE_BLOCKS_PER_SQRT_TILE_TRICLINIC :
                                TILE_TREE_BLOCKS_PER_SQRT_TILE)
        min_blocks = blocks_per_sqrt_tile * sqrt(tiles_per_block)
    end
    return n_blocks >= min_blocks
end

# The number of levels above the blocks and, for each of those levels `k`, the offset
#   of its nodes in the tree arrays
function tile_tree_levels(n_blocks::Integer)
    top = 0
    while (1 << top) < n_blocks
        top += 1
    end
    offsets = zeros(Int32, 32)
    offset = 0
    for k in 1:top
        offsets[k] = offset
        offset += cld(n_blocks, 1 << k)
    end
    return top, NTuple{32, Int32}(offsets)
end

# Fill node `n` of a tree level with the union of the boxes of its two children
function build_tile_tree_level_kernel!(tree_mins, tree_maxs, src_mins, src_maxs, src_offset,
                                       dst_offset, n_src, n_dst, ::Val{D}) where D
    n = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if n <= n_dst
        c1 = Int32(2) * n - Int32(1)
        c2 = Int32(2) * n
        @inbounds for d in 1:D
            lo = src_mins[src_offset + c1, d]
            hi = src_maxs[src_offset + c1, d]
            if c2 <= n_src
                lo = min(lo, src_mins[src_offset + c2, d])
                hi = max(hi, src_maxs[src_offset + c2, d])
            end
            tree_mins[dst_offset + n, d] = lo
            tree_maxs[dst_offset + n, d] = hi
        end
    end
    return nothing
end

function build_tile_tree!(buffers, n_blocks, ::Val{D}) where D
    top, offsets = tile_tree_levels(n_blocks)
    n_threads = 256
    for k in 1:top
        n_src = cld(n_blocks, 1 << (k - 1))
        n_dst = cld(n_blocks, 1 << k)
        if k == 1
            src_mins, src_maxs, src_offset = buffers.box_mins, buffers.box_maxs, Int32(0)
        else
            src_mins, src_maxs, src_offset = buffers.tree_mins, buffers.tree_maxs, offsets[k - 1]
        end
        @cuda threads=n_threads blocks=cld(n_dst, n_threads) build_tile_tree_level_kernel!(
            buffers.tree_mins, buffers.tree_maxs, src_mins, src_maxs, src_offset,
            offsets[k], Int32(n_src), Int32(n_dst), Val(D))
    end
    return top, offsets
end

# The bounding box stored in row `idx` of a pair of (n, D) min and max arrays
@inline function stored_box(mins, maxs, idx, ::Val{D}) where D
    r_min = SVector{D}(ntuple(d -> @inbounds(mins[idx, d]), Val(D)))
    r_max = SVector{D}(ntuple(d -> @inbounds(maxs[idx, d]), Val(D)))
    return r_min, r_max
end

function find_interacting_blocks_tree_kernel!(
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
    interacting_tiles_overflow, mins::AbstractArray{C}, maxs::AbstractArray{C},
    tree_mins, tree_maxs, tree_offsets, ::Val{top}, boundary, ::Val{r_cut2},
    ::Val{N_blocks}, ::Val{D}, max_total_tiles, block_exc_min, block_exc_max,
) where {C, top, r_cut2, N_blocks, D}
    mins_ro = CUDA_CORE.Const(mins)
    maxs_ro = CUDA_CORE.Const(maxs)
    tree_mins_ro = CUDA_CORE.Const(tree_mins)
    tree_maxs_ro = CUDA_CORE.Const(tree_maxs)
    block_exc_min_ro = CUDA_CORE.Const(block_exc_min)
    block_exc_max_ro = CUDA_CORE.Const(block_exc_max)

    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i > N_blocks
        return nothing
    end
    r_min_i, r_max_i = stored_box(mins_ro, maxs_ro, i, Val(D))

    # Depth-first walk without a stack: after a node is done, move to its right sibling,
    #   going up while the node is a right child, and stop on returning to the root
    k = Int32(top)
    n = Int32(1)
    while true
        n_nodes_k = Int32((Int64(N_blocks) + (Int64(1) << k) - 1) >> k)
        # The node holds blocks up to min(n * 2^k, N_blocks), which has to reach i
        visit = (n <= n_nodes_k) && ((Int64(n) << k) >= i)
        if visit
            if k == Int32(0)
                r_min_j, r_max_j = stored_box(mins_ro, maxs_ro, n, Val(D))
            else
                @inbounds node = tree_offsets[k] + n
                r_min_j, r_max_j = stored_box(tree_mins_ro, tree_maxs_ro, node, Val(D))
            end
            visit = boxes_within_cutoff(r_min_i, r_max_i, r_min_j, r_max_j, boundary, r_cut2)
        end
        if visit
            if k == Int32(0)
                emit_interacting_tile!(interacting_tiles_i, interacting_tiles_j,
                                       interacting_tiles_type, num_interacting_tiles,
                                       interacting_tiles_overflow, max_total_tiles,
                                       block_exc_min_ro, block_exc_max_ro, Int32(i), n,
                                       Int32(N_blocks))
            else
                k -= Int32(1)
                n = Int32(2) * n - Int32(1)
                continue
            end
        end
        while iszero(n & Int32(1)) && k < Int32(top)
            n >>= Int32(1)
            k += Int32(1)
        end
        k == Int32(top) && break
        n += Int32(1)
    end
    return nothing
end

#=
Second pruning pass over the candidate tiles emitted by
`find_interacting_blocks_kernel!`.

The bounding-box test in that kernel is loose: with 32 atoms per block the boxes are
comparable in size to the cutoff, so most tiles it emits are largely empty. This pass
runs one warp per candidate tile and performs the exact 32x32 atom-pair distance test,
recording which of the 32 inner-loop iterations of `force_kernel!`/`energy_kernel!`
contain at least one in-range pair. Iteration `m` of those kernels evaluates the pairs
`(lane, slot)` with `slot - lane == m (mod 32)`, so a hit between the `l`-th atom of
block `i` and the `s`-th atom of block `j` sets bit `(s - l) & 31`. Tiles with no
in-range pair at all get an empty mask and are marked `TILE_DEAD`, and are then removed
from the list by `compact_interacting_tiles!`.

This only runs on a neighbor list rebuild, so its cost is amortised over the `n_steps`
steps of the neighbor finder. Only full off-diagonal tiles (`i < j < n_blocks`) are
analysed; the diagonal and the final partial block make up a vanishing fraction of the
list and keep a fully populated mask.

Tiles that the block ranges in `find_interacting_blocks_kernel!` could not rule out as
holding an exclusion or special pair are checked exactly here, and marked clean if
none of their atom pairs is an exception.
=#
function prune_interacting_tiles_kernel!(
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
    interacting_tiles_diag, num_interacting_tiles, coords_var, ::Val{N}, ::Val{r_cut2},
    ::Val{n_blocks}, boundary, tile_exceptions) where {N, r_cut2, n_blocks}
    coords = CUDA_CORE.Const(coords_var)
    tiles_i_ro = CUDA_CORE.Const(interacting_tiles_i)
    tiles_j_ro = CUDA_CORE.Const(interacting_tiles_j)
    num_interacting_tiles_ro = CUDA_CORE.Const(num_interacting_tiles)

    a = Int32(1)
    idx = (blockIdx().x - a) * blockDim().y + threadIdx().y

    @inbounds num_pairs = num_interacting_tiles_ro[1]
    if idx > num_pairs
        return nothing
    end

    lane = threadIdx().x
    @inbounds i = tiles_i_ro[idx]
    @inbounds j = tiles_j_ro[idx]
    if i >= j || j >= Int32(n_blocks)
        if lane == a
            @inbounds interacting_tiles_diag[idx] = typemax(UInt32)
        end
        return nothing
    end

    # Read before the shuffles below, which every lane passes before lane 1 writes the
    # type, so the value and the branch on it are the same for the whole warp
    @inbounds tile_type = interacting_tiles_type[idx]
    @inbounds coords_j = coords[(j - a) * warpsize() + lane]
    i_0_tile = (i - a) * warpsize()

    # The displacements are found from the image of atom j closest to the first atom of
    #   block i when that gives the minimum image of every pair within the cutoff, see
    #   `tile_single_image`
    @inbounds single_image, coords_j_image = tile_single_image(coords_j,
                    coords[i_0_tile + lane], coords[i_0_tile + a], boundary, Val(r_cut2))
    lane_mask = (single_image ?
        prune_lane_mask(coords, i_0_tile, coords_j, coords_j_image, lane, boundary,
                        Val(r_cut2), Val(true)) :
        prune_lane_mask(coords, i_0_tile, coords_j, coords_j_image, lane, boundary,
                        Val(r_cut2), Val(false)))

    offset = Int32(16)
    while offset > 0
        lane_mask |= CUDA.shfl_xor_sync(0xFFFFFFFF, lane_mask, offset)
        offset ÷= Int32(2)
    end

    # Exact test for an exception between the blocks, from the atoms of block j
    has_exception = true
    if tile_type == UInt8(1) && lane_mask != UInt32(0)
        lane_has_exception = atom_has_partner_in_block(tile_exceptions,
                                                       (j - a) * warpsize() + lane, i)
        has_exception = CUDA.vote_any_sync(0xFFFFFFFF, lane_has_exception)
    end

    if lane == a
        @inbounds interacting_tiles_diag[idx] = lane_mask
        if lane_mask == UInt32(0)
            @inbounds interacting_tiles_type[idx] = TILE_DEAD
        elseif !has_exception
            @inbounds interacting_tiles_type[idx] = UInt8(0)
        end
    end
    return nothing
end

#=
For the boundaries with a rectangular box, the displacements of the pairs of a tile can
be found by subtracting the coordinates of atom i from those of atom j once the image of
atom i closest to a reference point is taken, rather than by finding the minimum image of
each pair, which takes several times as many instructions as the subtraction. With the
reference point the first atom of block j, this gives the minimum image of every pair
within the cutoff if the other atoms of block j are within `d` of it, with `d` plus the
cutoff less than half the shortest side of the box. The nearer image of a pair beyond the
cutoff can be missed, but the image used is then beyond the cutoff too.
Returns whether this holds for the warp, `coords_j` being the atom of block j of this lane
and `ref_j` the first one, and the image of atom i to use if so.
The other boundaries use the minimum image of each pair.
=#
@inline function tile_single_image(coords_i, coords_j, ref_j,
                                   boundary::Union{CubicBoundary, RectangularBoundary},
                                   ::Val{r_cut2}) where r_cut2
    return tile_single_image(coords_i, coords_j, ref_j, boundary, sqrt(r_cut2))
end

# As above for the cutoff `r_cut`
@inline function tile_single_image(coords_i, coords_j, ref_j,
                                   boundary::Union{CubicBoundary, RectangularBoundary}, r_cut)
    d2 = ustrip(sum(abs2, coords_j - ref_j))
    offset = Int32(16)
    while offset > Int32(0)
        d2 = max(d2, CUDA.shfl_xor_sync(0xFFFFFFFF, d2, offset))
        offset ÷= Int32(2)
    end
    # A plain comparison rather than `minimum`, which handles NaN and signed zeros with
    #   several instructions per element on every tile
    min_side = reduce((a, b) -> ifelse(a < b, a, b), boundary.side_lengths)
    max_d = min_side / 2 - r_cut
    valid = max_d > zero(max_d) && d2 < ustrip(max_d^2)
    return valid, image_closest(coords_i, ref_j, boundary)
end

@inline tile_single_image(coords_i, coords_j, ref_j, boundary, r_cut) = (false, coords_i)

# The image of `c` closest to `p` in a rectangular box, which is `c` unless they are more
#   than half a box side apart in a dimension
@inline function image_closest(c, p, boundary::Union{CubicBoundary, RectangularBoundary})
    return image_closest_1D.(c, p, boundary.side_lengths)
end

@inline function image_closest_1D(c, p, side_length)
    v = c - p
    half_sl = side_length / 2
    return ifelse(v > half_sl, c - side_length, ifelse(v < -half_sl, c + side_length, c))
end

#=
The forces of the pairs of a full tile, from the iterations in `diag_mask` (see
`prune_interacting_tiles_kernel!`), added to the accumulators `acc` of atom i and the
virial, which are returned, and to `opposites_sum` for the atoms of block j.
Only iterations marked as holding an in-range pair are walked. Bit b covers the pairs with
`slot - lane == b (mod 32)`, so it selects the same slot permutation as `m == b`. Testing
all 32 bits with a branch instead costs a loop header and a divergent branch per skipped
iteration, which the SASS shows dominating the loop.
With `has_exceptions` the bitmasks of the tile give the excluded and special pairs, and
otherwise there are none. With `single_image` the displacements are found from
`coords_i_image`, see `tile_single_image`.
With `refresh_inner` the inner masks of the dynamic pruning are found too, from the pairs
within the squared inner cutoff `r_inner2`, see `InnerTileMasks`, and returned with the
accumulators.
=#
@inline function tile_pair_forces(acc, opposites_sum, sh_stage, sh_vel, diag_mask,
                        eligible_bitmask, special_bitmask, ::Val{has_exceptions},
                        ::Val{single_image}, lane, warpid, coords_i, coords_i_image, atoms_i,
                        vel_i, ::Type{A}, shuf_syms, uses_vel, inters_tuple, boundary, step_n,
                        ::Val{r_cut2}, refresh_inner, r_inner2, ::Val{force_units},
                        ::Val{needs_vir}, ::Val{D}) where {has_exceptions, single_image, A,
                                                           r_cut2, force_units, needs_vir, D}
    a = Int32(1)
    (force_i_x, force_i_y, force_i_z, vir_xx, vir_yy, vir_zz, vir_xy, vir_xz, vir_yz,
     vir_yx, vir_zx, vir_zy) = acc
    inner_mask = UInt32(0)
    active = diag_mask
    @inbounds while active != UInt32(0)
        m = Int32(trailing_zeros(active))
        active &= active - UInt32(1)
        slot = ((lane - a + m) & Int32(31)) + a
        js = sh_stage[slot, warpid]
        coords_j = js.coords
        atoms_j_stage = Molly.rebuild_shuffled_atom(A, atoms_i, js.atom_payload, shuf_syms)
        vel_j = uses_vel ? sh_vel[slot, warpid] : vel_i

        dr = (single_image ? coords_j - coords_i_image : vector(coords_i, coords_j, boundary))
        r2 = sum(abs2, dr)
        if has_exceptions
            excl = (eligible_bitmask >> (warpsize() - slot)) | (eligible_bitmask << slot)
            spec = (special_bitmask >> (warpsize() - slot)) | (special_bitmask << slot)
            eligible = (excl & 0x1) == true
            condition = eligible && r2 <= r_cut2
            special = (spec & 0x1) == true
        else
            eligible = true
            condition = r2 <= r_cut2
            special = false
        end
        if refresh_inner && CUDA.vote_any_sync(0xFFFFFFFF, eligible && r2 <= r_inner2)
            inner_mask |= UInt32(1) << m
        end
        any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)

        if any_active
            f = condition ? sum_pairwise_forces_gpu(
                inters_tuple, dr, atoms_i, atoms_j_stage, Val(force_units),
                special, coords_i, coords_j, boundary, vel_i, vel_j, step_n
            ) : Molly.zero_pairwise_force(dr, force_units)

            force_i_x += ustrip(f[1])
            opposites_sum[slot, 1, warpid] -= ustrip(f[1])
            if D >= 2
                force_i_y += ustrip(f[2])
                opposites_sum[slot, 2, warpid] -= ustrip(f[2])
            end
            if D >= 3
                force_i_z += ustrip(f[3])
                opposites_sum[slot, 3, warpid] -= ustrip(f[3])
            end

            if needs_vir
                vir_xx += ustrip(dr[1]) * ustrip(f[1])
                if D >= 2
                    vir_yy += ustrip(dr[2]) * ustrip(f[2])
                    vir_xy += ustrip(dr[1]) * ustrip(f[2])
                    vir_yx += ustrip(dr[2]) * ustrip(f[1])
                end
                if D >= 3
                    vir_zz += ustrip(dr[3]) * ustrip(f[3])
                    vir_xz += ustrip(dr[1]) * ustrip(f[3])
                    vir_zx += ustrip(dr[3]) * ustrip(f[1])
                    vir_yz += ustrip(dr[2]) * ustrip(f[3])
                    vir_zy += ustrip(dr[3]) * ustrip(f[2])
                end
            end
        end
    end
    return (force_i_x, force_i_y, force_i_z, vir_xx, vir_yy, vir_zz, vir_xy, vir_xz, vir_yz,
            vir_yx, vir_zx, vir_zy), inner_mask
end

# The cutoff for `tile_single_image` in the force kernel, and the squared inner cutoff and
#   the store of an inner mask, see `InnerTileMasks`
@inline single_image_cutoff(::Nothing, refresh_inner, ::Val{r_cut2}) where {r_cut2} = sqrt(r_cut2)
@inline function single_image_cutoff(inner, refresh_inner, ::Val{r_cut2}) where r_cut2
    return ifelse(refresh_inner, inner.r, sqrt(r_cut2))
end
@inline inner_sq_cutoff(::Nothing, ::Val{r_cut2}) where {r_cut2} = r_cut2
@inline inner_sq_cutoff(inner, ::Val{r_cut2}) where {r_cut2} = inner.r2
@inline store_inner_mask!(::Nothing, idx, mask) = nothing
@inline store_inner_mask!(inner, idx, mask) = (@inbounds inner.masks[idx] = mask; nothing)

# This lane holds the `lane`-th atom of block j, so a hit against the m-th atom of block i
#   lands on inner-loop iteration `(lane - m) & 31`
@inline function prune_lane_mask(coords, i_0_tile, coords_j, coords_j_image, lane, boundary,
                                 ::Val{r_cut2}, ::Val{single_image}) where {r_cut2, single_image}
    a = Int32(1)
    lane_mask = UInt32(0)
    @inbounds for m in a:warpsize()
        coords_i = coords[i_0_tile + m]
        dr = (single_image ? coords_j_image - coords_i : vector(coords_i, coords_j, boundary))
        r2 = sum(abs2, dr)
        if r2 <= r_cut2
            lane_mask |= UInt32(1) << ((lane - m) & Int32(31))
        end
    end
    return lane_mask
end

"""
    force_kernel!(fs_mat, global_virial, coords, velocities, atoms, N, r_cut2, force_units,
                  inters_tuple, boundary, step_n, tile_exceptions, needs_vir, T, D,
                  interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
                  interacting_tiles_diag, num_interacting_tiles)

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

CLEAN tiles skip the bitmasks entirely; mask-backed tiles build them from the
sparse exception lists in `tile_exceptions` to apply exclusions and special-pair
handling.
"""
function force_kernel!(
    fs_mat,
    global_virial,
    coords_var,
    velocities_var,
    atoms_var::AbstractArray{A},
    ::Val{N},
    ::Val{r_cut2},
    ::Val{force_units},
    inters_tuple,
    boundary,
    step_n,
    tile_exceptions,
    ::Val{needs_vir},
    ::Val{T},
    ::Val{TH},
    ::Val{D},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
    interacting_tiles_diag, num_interacting_tiles,
    interacting_tiles_overflow, inner) where {N, r_cut2, A, force_units, needs_vir, T, TH, D}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)
    coords = CUDA_CORE.Const(coords_var)
    velocities = CUDA_CORE.Const(velocities_var)
    atoms = CUDA_CORE.Const(atoms_var)
    tiles_i_ro = CUDA_CORE.Const(interacting_tiles_i)
    tiles_j_ro = CUDA_CORE.Const(interacting_tiles_j)
    tiles_type_ro = CUDA_CORE.Const(interacting_tiles_type)
    tiles_diag_ro = CUDA_CORE.Const(interacting_tiles_diag)
    num_interacting_tiles_ro = CUDA_CORE.Const(num_interacting_tiles)
    interacting_tiles_overflow_ro = CUDA_CORE.Const(interacting_tiles_overflow)
    uses_vel = Molly.any_uses_velocity(inters_tuple)
    
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

    # `prune_interacting_tiles_kernel!` proved this tile holds no in-cutoff pair
    if type == TILE_DEAD
        return nothing
    end
    @inbounds diag_mask = tiles_diag_ro[idx]

    # The inner masks of the dynamic pruning are used for the full tiles, the only ones
    #   with masks, unless they have to be found again, see `InnerTileMasks`
    refresh_inner = false
    if !isnothing(inner) && j < n_blocks && i < j
        refresh_inner = (@inbounds inner.stale[1] != Int32(0))
        if !refresh_inner
            @inbounds diag_mask = inner.masks[idx]
            if diag_mask == UInt32(0)
                return nothing
            end
        end
    end

    i_0_tile = (i - a) * warpsize()
    index_i = i_0_tile + lane

    # Dynamic shared memory sized to the actual block_y (not MAX_BLOCK_Y): the
    # j-force accumulator plus staged j-atom data (coords/atoms/velocities). The
    # Part 1 inner loop indexes this shared data by slot instead of rotating it
    # around the warp with serial shuffles. Host must pass a matching `shmem`.
    # @inbounds elides CuDynamicSharedArray's size check: the host allocates a
    # matching `shmem`, and sh_vel (unallocated when no interaction uses velocity)
    # is never dereferenced in that case
    # Only the Atom fields the active interactions actually read are staged (not the
    # full Atom), matching what the old warp-shuffle path sent per lane.
    shuf_syms = resolved_atom_shuffle_syms(inters_tuple, A)
    P = atom_payload_type(A, shuf_syms)
    by = Int(blockDim().y)
    JT = JStage{eltype(coords_var), P}
    opposites_sum = JForceAccumulator(
                        @inbounds(CuDynamicSharedArray(jforce_eltype(T, Val(D)), (32, by))))
    stage_off = 32 * by * sizeof(jforce_eltype(T, Val(D)))
    sh_stage = @inbounds CuDynamicSharedArray(JT, (32, by), stage_off)
    stage_off_v = stage_off + 32 * by * sizeof(JT)
    sh_vel = @inbounds CuDynamicSharedArray(eltype(velocities_var), (32, by), stage_off_v)

    r = Int32((N - 1) % 32 + 1)
    
    force_i_x = zero(T)
    force_i_y = zero(T)
    force_i_z = zero(T)

    # One warp handles exactly one tile, so the accumulator only needs clearing here
    @inbounds for k in a:b
        opposites_sum[lane, k, warpid] = zero(T)
    end

    # Virial W_αβ = Σ dr_α f_β with f the force on j, matching the CPU path. All
    #   components are kept since the tensor is not symmetric for non-central forces.
    vir_xx = zero(T); vir_yy = zero(T); vir_zz = zero(T)
    vir_xy = zero(T); vir_xz = zero(T); vir_yz = zero(T)
    vir_yx = zero(T); vir_zx = zero(T); vir_zy = zero(T)

    j_0_tile = (j - a) * warpsize()
    index_j = j_0_tile + lane

    # Part 1: Standard non-diagonal tiles
    if j < n_blocks && i < j
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        
        # Stage this tile's j-atom data into shared memory once, then index it by
        # slot each iteration. Replaces the per-iteration serial warp shuffles; the
        # staged reads are independent so the latency pipelines. Only the Atom fields
        # the active interactions read are staged (`shuf_syms`/`P` above), matching
        # the old shuffle path's payload rather than the full 32 B Atom. sync_warp()
        # also publishes the opposites_sum zeroing above.
        @inbounds coords_j_lane = coords[index_j]
        @inbounds sh_stage[lane, warpid] = JStage(coords_j_lane, Molly.atom_shuffle_payload(atoms[index_j], shuf_syms))
        if uses_vel
            @inbounds sh_vel[lane, warpid] = velocities[index_j]
        end
        sync_warp()

        # The displacements are found from the image of atom i closest to the first atom
        #   of block j when that gives the minimum image of every pair within the cutoff,
        #   or the inner cutoff when the inner masks are found, see `tile_single_image`
        single_image, coords_i_image = tile_single_image(coords_i, coords_j_lane,
                                    @inbounds(sh_stage[a, warpid].coords), boundary,
                                    single_image_cutoff(inner, refresh_inner, Val(r_cut2)))

        acc = (force_i_x, force_i_y, force_i_z, vir_xx, vir_yy, vir_zz, vir_xy, vir_xz,
               vir_yz, vir_yx, vir_zx, vir_zy)
        r_inner2 = inner_sq_cutoff(inner, Val(r_cut2))
        if type == UInt8(0) # CLEAN
            if single_image
                acc, inner_mask = tile_pair_forces(acc, opposites_sum, sh_stage, sh_vel,
                            diag_mask, UInt32(0), UInt32(0), Val(false), Val(true), lane,
                            warpid, coords_i, coords_i_image, atoms_i, vel_i, A, shuf_syms,
                            uses_vel, inters_tuple, boundary, step_n, Val(r_cut2),
                            refresh_inner, r_inner2, Val(force_units), Val(needs_vir), Val(D))
            else
                acc, inner_mask = tile_pair_forces(acc, opposites_sum, sh_stage, sh_vel,
                            diag_mask, UInt32(0), UInt32(0), Val(false), Val(false), lane,
                            warpid, coords_i, coords_i_image, atoms_i, vel_i, A, shuf_syms,
                            uses_vel, inters_tuple, boundary, step_n, Val(r_cut2),
                            refresh_inner, r_inner2, Val(force_units), Val(needs_vir), Val(D))
            end
        else # EXCLUDED
            # These tiles are few enough that the minimum image of each pair is used
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))
            acc, inner_mask = tile_pair_forces(acc, opposites_sum, sh_stage, sh_vel,
                        diag_mask, eligible_bitmask, special_bitmask, Val(true), Val(false),
                        lane, warpid, coords_i, coords_i_image, atoms_i, vel_i, A, shuf_syms,
                        uses_vel, inters_tuple, boundary, step_n, Val(r_cut2),
                        refresh_inner, r_inner2, Val(force_units), Val(needs_vir), Val(D))
        end
        (force_i_x, force_i_y, force_i_z, vir_xx, vir_yy, vir_zz, vir_xy, vir_xz, vir_yz,
         vir_yx, vir_zx, vir_zy) = acc
        if refresh_inner && lane == a
            store_inner_mask!(inner, idx, inner_mask)
        end

        sync_warp()
        if index_j <= N
            @inbounds for k in a:b
                if opposites_sum[lane, k, warpid] != zero(T)
                    CUDA.atomic_add!(
                        pointer(fs_mat, Int64(index_j) * b - (b - k)),
                        -opposites_sum[lane, k, warpid],
                    )
                end
            end
        end
    end

    # Part 2: Boundary column tiles
    if j == n_blocks && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        
        eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                    lane, n_blocks, r, Val(N))

        @inbounds for m in a:r
            idx_j = j_0_tile + m
            @inbounds coords_j = coords[idx_j]
            @inbounds vel_j = velocities[idx_j]
            @inbounds atoms_j = atoms[idx_j]
            
            dr = vector(coords_i, coords_j, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            
            condition = (excl & 0x1) == true && r2 <= r_cut2
            any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)

            if any_active
                f = condition ? sum_pairwise_forces_gpu(
                    inters_tuple, dr, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : Molly.zero_pairwise_force(dr, force_units)

                force_i_x += ustrip(f[1])
                if ustrip(f[1]) != zero(T)
                    CUDA.atomic_add!(pointer(fs_mat, Int64(idx_j) * b - (b - 1)), ustrip(f[1]))
                end
                if D >= 2
                    force_i_y += ustrip(f[2])
                    if ustrip(f[2]) != zero(T)
                        CUDA.atomic_add!(pointer(fs_mat, Int64(idx_j) * b - (b - 2)), ustrip(f[2]))
                    end
                end
                if D >= 3
                    force_i_z += ustrip(f[3])
                    if ustrip(f[3]) != zero(T)
                        CUDA.atomic_add!(pointer(fs_mat, Int64(idx_j) * b - (b - 3)), ustrip(f[3]))
                    end
                end
                
                if needs_vir
                    vir_xx += ustrip(dr[1]) * ustrip(f[1])
                    if D >= 2
                        vir_yy += ustrip(dr[2]) * ustrip(f[2])
                        vir_xy += ustrip(dr[1]) * ustrip(f[2])
                        vir_yx += ustrip(dr[2]) * ustrip(f[1])
                    end
                    if D >= 3
                        vir_zz += ustrip(dr[3]) * ustrip(f[3])
                        vir_xz += ustrip(dr[1]) * ustrip(f[3])
                        vir_zx += ustrip(dr[3]) * ustrip(f[1])
                        vir_yz += ustrip(dr[2]) * ustrip(f[3])
                        vir_zy += ustrip(dr[3]) * ustrip(f[2])
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
        
        eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                    lane, n_blocks, r, Val(N))

        @inbounds for m in (lane + a) : warpsize()
            idx_j = j_0_tile + m
            @inbounds coords_j = coords[idx_j]
            @inbounds vel_j = velocities[idx_j]
            @inbounds atoms_j = atoms[idx_j]
            
            dr = vector(coords_i, coords_j, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut2
            
            # Divergence-safe execution (no vote_any_sync)
            f = condition ? sum_pairwise_forces_gpu(
                inters_tuple, dr, atoms_i, atoms_j, Val(force_units),
                (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
            ) : Molly.zero_pairwise_force(dr, force_units)

            force_i_x += ustrip(f[1])
            opposites_sum[m, 1, warpid] -= ustrip(f[1])
            if D >= 2
                force_i_y += ustrip(f[2])
                opposites_sum[m, 2, warpid] -= ustrip(f[2])
            end
            if D >= 3
                force_i_z += ustrip(f[3])
                opposites_sum[m, 3, warpid] -= ustrip(f[3])
            end
            
            if needs_vir
                vir_xx += ustrip(dr[1]) * ustrip(f[1])
                if D >= 2
                    vir_yy += ustrip(dr[2]) * ustrip(f[2])
                    vir_xy += ustrip(dr[1]) * ustrip(f[2])
                    vir_yx += ustrip(dr[2]) * ustrip(f[1])
                end
                if D >= 3
                    vir_zz += ustrip(dr[3]) * ustrip(f[3])
                    vir_xz += ustrip(dr[1]) * ustrip(f[3])
                    vir_zx += ustrip(dr[3]) * ustrip(f[1])
                    vir_yz += ustrip(dr[2]) * ustrip(f[3])
                    vir_zy += ustrip(dr[3]) * ustrip(f[2])
                end
            end
        end

        sync_warp()
        force_i_x += opposites_sum[lane, 1, warpid]
        if D >= 2
            force_i_y += opposites_sum[lane, 2, warpid]
        end
        if D >= 3
            force_i_z += opposites_sum[lane, 3, warpid]
        end
    end

    # Part 4: Terminal corner tile
    if i == n_blocks && j == n_blocks
        if lane <= r
            @inbounds coords_i = coords[index_i]
            @inbounds vel_i = velocities[index_i]
            @inbounds atoms_i = atoms[index_i]
            
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))

            @inbounds for m in (lane + a) : r
                idx_j = j_0_tile + m
                @inbounds coords_j = coords[idx_j]
                @inbounds vel_j = velocities[idx_j]
                @inbounds atoms_j = atoms[idx_j]
                
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut2
                
                # Divergence-safe execution (no vote_any_sync)
                f = condition ? sum_pairwise_forces_gpu(
                    inters_tuple, dr, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : Molly.zero_pairwise_force(dr, force_units)

                force_i_x += ustrip(f[1])
                opposites_sum[m, 1, warpid] -= ustrip(f[1])
                if D >= 2
                    force_i_y += ustrip(f[2])
                    opposites_sum[m, 2, warpid] -= ustrip(f[2])
                end
                if D >= 3
                    force_i_z += ustrip(f[3])
                    opposites_sum[m, 3, warpid] -= ustrip(f[3])
                end
                
                if needs_vir
                    vir_xx += ustrip(dr[1]) * ustrip(f[1])
                    if D >= 2
                        vir_yy += ustrip(dr[2]) * ustrip(f[2])
                        vir_xy += ustrip(dr[1]) * ustrip(f[2])
                        vir_yx += ustrip(dr[2]) * ustrip(f[1])
                    end
                    if D >= 3
                        vir_zz += ustrip(dr[3]) * ustrip(f[3])
                        vir_xz += ustrip(dr[1]) * ustrip(f[3])
                        vir_zx += ustrip(dr[3]) * ustrip(f[1])
                        vir_yz += ustrip(dr[2]) * ustrip(f[3])
                        vir_zy += ustrip(dr[3]) * ustrip(f[2])
                    end
                end
            end
        end
        
        sync_warp()

        if lane <= r
            force_i_x += opposites_sum[lane, 1, warpid]
            if D >= 2
                force_i_y += opposites_sum[lane, 2, warpid]
            end
            if D >= 3
                force_i_z += opposites_sum[lane, 3, warpid]
            end
        end
    end

    if needs_vir
        offset_val = Int32(16)
        while offset_val > 0
            vir_xx += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xx, offset_val)
            if D >= 2
                vir_yy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yy, offset_val)
                vir_xy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xy, offset_val)
                vir_yx += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yx, offset_val)
            end
            if D >= 3
                vir_zz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_zz, offset_val)
                vir_xz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xz, offset_val)
                vir_zx += CUDA.shfl_down_sync(0xFFFFFFFF, vir_zx, offset_val)
                vir_yz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yz, offset_val)
                vir_zy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_zy, offset_val)
            end
            offset_val ÷= 2
        end

        # The virial is column-major, element (α, β) is at α + (β - 1) * D in the slot of
        #   the tile, see `Molly.virial_slot`
        if lane == 1
            vir_off = (Molly.virial_slot(idx, size(global_virial, 3)) - 1) * D * D
            if vir_xx != zero(T)
                CUDA.atomic_add!(pointer(global_virial, vir_off + 1), TH(vir_xx))
            end
            if D >= 2
                if vir_yy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, vir_off + D + 2), TH(vir_yy))
                end
                if vir_xy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, vir_off + D + 1), TH(vir_xy))
                end
                if vir_yx != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, vir_off + 2), TH(vir_yx))
                end
            end
            if D >= 3
                if vir_zz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, vir_off + 2 * D + 3), TH(vir_zz))
                end
                if vir_xz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, vir_off + 2 * D + 1), TH(vir_xz))
                end
                if vir_zx != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, vir_off + 3), TH(vir_zx))
                end
                if vir_yz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, vir_off + 2 * D + 2), TH(vir_yz))
                end
                if vir_zy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, vir_off + D + 3), TH(vir_zy))
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

"""
    energy_kernel!(energy_nounits, coords, velocities, atoms, N, r_cut2, energy_units,
                   inters_tuple, boundary, step_n, tile_exceptions, T, D,
                   interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
                   interacting_tiles_diag, num_interacting_tiles, interacting_tiles_overflow,
                   inner)

Compute pairwise potential energies for the compact list of interacting 32x32
tiles produced by `find_interacting_blocks_kernel!`.

This mirrors `force_kernel!` structurally: the same compact tile list, the same
four tile-shape cases, and the same CLEAN-vs-mask-backed fast path. The main
difference is the reduction target: each warp accumulates energy into shared
memory and performs a warp reduction before the final atomic add to
`energy_nounits`.

`inner` is `nothing` or the inner masks of the dynamic pruning, see `InnerTileMasks`,
which are used for the full tiles when they are current. Unlike `force_kernel!` the
masks are not found again when they are stale, the masks of the list are used instead.
"""
function energy_kernel!(
    energy_nounits,
    coords_var,
    velocities_var,
    atoms_var::AbstractArray{A},
    ::Val{N},
    ::Val{r_cut2},
    ::Val{energy_units},
    inters_tuple,
    boundary,
    step_n,
    tile_exceptions,
    ::Val{T},
    ::Val{TH},
    ::Val{D},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
    interacting_tiles_diag, num_interacting_tiles,
    interacting_tiles_overflow, inner) where {N, r_cut2, A, energy_units, T, TH, D}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)
    coords = CUDA_CORE.Const(coords_var)
    velocities = CUDA_CORE.Const(velocities_var)
    atoms = CUDA_CORE.Const(atoms_var)
    tiles_i_ro = CUDA_CORE.Const(interacting_tiles_i)
    tiles_j_ro = CUDA_CORE.Const(interacting_tiles_j)
    tiles_type_ro = CUDA_CORE.Const(interacting_tiles_type)
    tiles_diag_ro = CUDA_CORE.Const(interacting_tiles_diag)
    num_interacting_tiles_ro = CUDA_CORE.Const(num_interacting_tiles)
    interacting_tiles_overflow_ro = CUDA_CORE.Const(interacting_tiles_overflow)
    uses_vel = Molly.any_uses_velocity(inters_tuple)

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

    # `prune_interacting_tiles_kernel!` proved this tile holds no in-cutoff pair
    if type == TILE_DEAD
        return nothing
    end
    @inbounds diag_mask = tiles_diag_ro[idx]

    # The inner masks of the dynamic pruning are used for the full tiles, the only ones
    #   with masks, when they are current, see `InnerTileMasks`
    if !isnothing(inner) && j < n_blocks && i < j
        if (@inbounds inner.stale[1]) == Int32(0)
            @inbounds diag_mask = inner.masks[idx]
            if diag_mask == UInt32(0)
                return nothing
            end
        end
    end

    i_0_tile = (i - a) * warpsize()
    index_i = i_0_tile + lane

    sum_E = zero(T)

    # Dynamic shared memory for staging this tile's j-atom data (coords/atoms and,
    # when an interaction uses them, velocities), mirroring force_kernel!'s Part 1.
    # Only the Atom fields the active interactions read are staged (`shuf_syms`/`P`).
    # Energy needs no opposites_sum accumulator. @inbounds elides the size check;
    # the host passes a matching `shmem`, and sh_vel (unallocated when no
    # interaction uses velocity) is never dereferenced in that case.
    shuf_syms = resolved_atom_shuffle_syms(inters_tuple, A)
    P = atom_payload_type(A, shuf_syms)
    by = Int(blockDim().y)
    JT = JStage{eltype(coords_var), P}
    sh_stage = @inbounds CuDynamicSharedArray(JT, (32, by))
    stage_off_v = 32 * by * sizeof(JT)
    sh_vel = @inbounds CuDynamicSharedArray(eltype(velocities_var), (32, by), stage_off_v)

    r = Int32((N - 1) % 32 + 1)

    j_0_tile = (j - a) * warpsize()
    index_j = j_0_tile + lane

    if j < n_blocks && i < j
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        # Stage this tile's j-atom data into shared memory once, then index it by
        # slot each iteration instead of rotating it around the warp with serial
        # shuffles. The staged reads are independent so their latency pipelines.
        # Only the Atom fields the active interactions read are staged (`shuf_syms`),
        # matching the old shuffle path's payload rather than the full Atom.
        @inbounds sh_stage[lane, warpid] = JStage(coords[index_j], Molly.atom_shuffle_payload(atoms[index_j], shuf_syms))
        if uses_vel
            @inbounds sh_vel[lane, warpid] = velocities[index_j]
        end
        sync_warp()

        if type == UInt8(0) # CLEAN
            # Walk only the iterations `prune_interacting_tiles_kernel!` marked as holding
            # an in-range pair. Bit b covers the pairs with `slot - lane == b (mod 32)`, so
            # it selects the same slot permutation as `m == b`. Testing all 32 bits with a
            # branch instead costs a loop header and a divergent branch per skipped
            # iteration, which the SASS shows dominating the loop.
            active = diag_mask
            @inbounds while active != UInt32(0)
                m = Int32(trailing_zeros(active))
                active &= active - UInt32(1)
                slot = ((lane - a + m) & Int32(31)) + a
                js = sh_stage[slot, warpid]
                coords_j = js.coords
                atoms_j_stage = Molly.rebuild_shuffled_atom(A, atoms_i, js.atom_payload, shuf_syms)
                vel_j = uses_vel ? sh_vel[slot, warpid] : vel_i

                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                condition = r2 <= r_cut2
                any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)

                if any_active
                    pe = condition ? sum_pairwise_potentials_gpu(
                        inters_tuple,
                        dr,
                        atoms_i, atoms_j_stage,
                        Val(energy_units),
                        false,
                        coords_i, coords_j,
                        boundary,
                        vel_i, vel_j,
                        step_n) : SVector(Molly.zero_pairwise_energy(dr, energy_units))

                    sum_E += ustrip(pe[1])
                end
            end
        else # EXCLUDED
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))

            active = diag_mask
            @inbounds while active != UInt32(0)
                m = Int32(trailing_zeros(active))
                active &= active - UInt32(1)
                slot = ((lane - a + m) & Int32(31)) + a
                js = sh_stage[slot, warpid]
                coords_j = js.coords
                atoms_j_stage = Molly.rebuild_shuffled_atom(A, atoms_i, js.atom_payload, shuf_syms)
                vel_j = uses_vel ? sh_vel[slot, warpid] : vel_i

                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - slot)) | (eligible_bitmask << slot)
                spec = (special_bitmask >> (warpsize() - slot)) | (special_bitmask << slot)
                condition = (excl & 0x1) == true && r2 <= r_cut2
                any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)

                if any_active
                    pe = condition ? sum_pairwise_potentials_gpu(
                        inters_tuple,
                        dr,
                        atoms_i, atoms_j_stage,
                        Val(energy_units),
                        (spec & 0x1) == true,
                        coords_i, coords_j,
                        boundary,
                        vel_i, vel_j,
                        step_n) : SVector(Molly.zero_pairwise_energy(dr, energy_units))

                    sum_E += ustrip(pe[1])
                end
            end
        end
    elseif j == n_blocks && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                    lane, n_blocks, r, Val(N))

        @inbounds for m in a:r
            idx_j = j_0_tile + m
            @inbounds coords_j = coords[idx_j]
            @inbounds vel_j = velocities[idx_j]
            @inbounds atoms_j = atoms[idx_j]
            dr = vector(coords_i, coords_j, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut2

            pe = condition ? sum_pairwise_potentials_gpu(
                inters_tuple,
                dr,
                atoms_i, atoms_j,
                Val(energy_units),
                (spec & 0x1) == true,
                coords_i, coords_j,
                boundary,
                vel_i, vel_j,
                step_n) : SVector(Molly.zero_pairwise_energy(dr, energy_units))
            sum_E += ustrip(pe[1])
        end
    elseif i == j && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                    lane, n_blocks, r, Val(N))

        @inbounds for m in (lane + a) : warpsize()
            idx_j = j_0_tile + m
            @inbounds coords_j = coords[idx_j]
            @inbounds vel_j = velocities[idx_j]
            @inbounds atoms_j = atoms[idx_j]
            dr = vector(coords_i, coords_j, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut2

            pe = condition ? sum_pairwise_potentials_gpu(
                inters_tuple,
                dr,
                atoms_i, atoms_j,
                Val(energy_units),
                (spec & 0x1) == true,
                coords_i, coords_j,
                boundary,
                vel_i, vel_j,
                step_n) : SVector(Molly.zero_pairwise_energy(dr, energy_units))
            sum_E += ustrip(pe[1])
        end
    elseif i == n_blocks && j == n_blocks
        if lane <= r
            @inbounds coords_i = coords[index_i]
            @inbounds vel_i = velocities[index_i]
            @inbounds atoms_i = atoms[index_i]
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))

            @inbounds for m in (lane + a) : r
                idx_j = j_0_tile + m
                @inbounds coords_j = coords[idx_j]
                @inbounds vel_j = velocities[idx_j]
                @inbounds atoms_j = atoms[idx_j]
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut2

                pe = condition ? sum_pairwise_potentials_gpu(
                    inters_tuple,
                    dr,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : SVector(Molly.zero_pairwise_energy(dr, energy_units))
                sum_E += ustrip(pe[1])
            end
        end
    end


    # Warp reduction
    offset = Int32(16)
    while offset > 0
        sum_E += CUDA.shfl_down_sync(0xFFFFFFFF, sum_E, offset)
        offset ÷= 2
    end

    if lane == a && sum_E != zero(T)
        CUDA.atomic_add!(pointer(energy_nounits, Molly.energy_slot(idx, length(energy_nounits))),
                         TH(sum_E))
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
    coords = CUDA_CORE.Const(coords_var)
    velocities = CUDA_CORE.Const(velocities_var)
    atoms = CUDA_CORE.Const(atoms_var)
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
                    f = sum_pairwise_forces_nonl(inters, atom_i, atom_j, Val(F), false, coord_i,
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
            f = sum_pairwise_forces_nonl(inters, atom_i, atom_j, Val(F), false, coord_i, coord_j,
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

#=
Two buffers for the center of mass momentum, used in turn by `remove_CM_motion!` so that it
does not allocate and zero one each time it is called, which is every step by default.
The kernel that removes the center of mass velocity zeroes the buffer that the next call
accumulates into. They are kept per task, since each task has its own stream.
=#
mutable struct CMMomentumScratch{T}
    buffers::NTuple{2, CuArray{T, 1, CUDA_CORE.DeviceMemory}}
    current::Int
    context::CuContext
end

function cm_momentum_scratch(::Type{T}) where T
    tls = task_local_storage()
    key = (:molly_cm_momentum_scratch, T)
    scratch = get(tls, key, nothing)
    if !(scratch isa CMMomentumScratch{T}) || scratch.context != CUDA.context()
        buffers = (fill!(CuArray{T, 1, CUDA_CORE.DeviceMemory}(undef, 3), zero(T)),
                   fill!(CuArray{T, 1, CUDA_CORE.DeviceMemory}(undef, 3), zero(T)))
        scratch = CMMomentumScratch{T}(buffers, 1, CUDA.context())
        tls[key] = scratch
    end
    return scratch
end

function Molly.remove_CM_motion!(sys::System{3, <:CuArray, T}) where T
    M = unit(zero(eltype(eltype(sys.velocities))) * zero(sys.total_mass))
    n_atoms = length(sys)
    scratch = cm_momentum_scratch(T)
    cm_momentum = scratch.buffers[scratch.current]
    next_cm_momentum = scratch.buffers[3 - scratch.current]
    scratch.current = 3 - scratch.current
    n_threads = 256
    n_blocks = cld(n_atoms, n_threads)
    shmem = 3 * n_threads * sizeof(T)

    args_momentum = (cm_momentum, sys.velocities, masses(sys), Val(n_atoms))
    kernel_momentum = cached_cufunction(cm_momentum_kernel_3d!, args_momentum,
                                        Val(NamedTuple()))
    kernel_momentum(args_momentum...; threads=n_threads, blocks=n_blocks, shmem=shmem)
    args_remove = (sys.velocities, cm_momentum, next_cm_momentum, sys.virtual_site_flags,
                   sys.total_mass, Val(M), Val(n_atoms), Val(!isempty(sys.virtual_sites)))
    kernel_remove = cached_cufunction(remove_cm_velocity_kernel_3d!, args_remove,
                                      Val(NamedTuple()))
    kernel_remove(args_remove...; threads=n_threads, blocks=n_blocks)
    return sys
end

function cm_momentum_kernel_3d!(cm_momentum::CuDeviceVector{T}, velocities, atom_masses,
                                ::Val{n_atoms}) where {T, n_atoms}
    tid = threadIdx().x
    block_size = blockDim().x
    idx = (blockIdx().x - 1) * block_size + tid
    stride = block_size * gridDim().x
    shmem = CuDynamicSharedArray(T, 3 * block_size)

    mx = zero(T)
    my = zero(T)
    mz = zero(T)
    @inbounds while idx <= n_atoms
        p = velocities[idx] * atom_masses[idx]
        mx += ustrip(p[1])
        my += ustrip(p[2])
        mz += ustrip(p[3])
        idx += stride
    end

    @inbounds begin
        shmem[tid] = mx
        shmem[block_size + tid] = my
        shmem[2 * block_size + tid] = mz
    end
    sync_threads()

    offset = block_size >>> 1
    while offset > 0
        if tid <= offset
            @inbounds begin
                shmem[tid] += shmem[tid + offset]
                shmem[block_size + tid] += shmem[block_size + tid + offset]
                shmem[2 * block_size + tid] += shmem[2 * block_size + tid + offset]
            end
        end
        sync_threads()
        offset >>>= 1
    end

    if tid == 1
        CUDA.atomic_add!(pointer(cm_momentum, 1), shmem[1])
        CUDA.atomic_add!(pointer(cm_momentum, 2), shmem[block_size + 1])
        CUDA.atomic_add!(pointer(cm_momentum, 3), shmem[2 * block_size + 1])
    end
    return nothing
end

# Also zeroes `next_cm_momentum`, the buffer the next call accumulates into
function remove_cm_velocity_kernel_3d!(velocities, cm_momentum::CuDeviceVector{T},
                            next_cm_momentum, virtual_site_flags, total_mass,
                            ::Val{momentum_unit}, ::Val{n_atoms},
                            ::Val{has_vs}) where {T, momentum_unit, n_atoms, has_vs}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if idx <= 3
        next_cm_momentum[idx] = zero(T)
    end
    @inbounds if idx <= n_atoms
        cm_velocity = SVector(
            cm_momentum[1] * momentum_unit / total_mass,
            cm_momentum[2] * momentum_unit / total_mass,
            cm_momentum[3] * momentum_unit / total_mass,
        )
        v = velocities[idx]
        if has_vs && virtual_site_flags[idx]
            velocities[idx] = zero(v)
        else
            velocities[idx] = v - cm_velocity
        end
    end
    return nothing
end

end
