# Portable tiled pairwise kernels used with GPUNeighborFinder
#
# These kernels use the sub-group (warp) operations of KernelInterface, so they run on any
#   KernelAbstractions backend that supports sub-groups of 32 work-items with shuffles,
#   such as CUDA and the POCL-based CPU backend of KernelAbstractions.
#
# The pipeline generally follows these steps:
# 1.  **Reordering**: Atoms are periodically reordered based on Morton (Z-order) curves
#     to improve cache hits during pairwise interactions.
# 2.  **Exceptions**: The sparse per-atom lists of excluded and special pairs are
#     translated into Morton positions, so that the bitmasks of the few 32x32 tiles
#     that contain exceptions can be built on the fly without storing a mask per tile.
# 3.  **Tile Finding**: A kernel identifies pairs of 32x32 atom blocks (tiles) that are
#     within the interaction cutoff, using bounding box checks.
# 4.  **Execution**: Specialized kernels iterate over the list of interacting tiles,
#     with one sub-group of 32 work-items per tile.
#
# A tile is processed by one sub-group, and its atoms are indexed by the lane
#   `KI.get_sub_group_local_id()` and the sub-group `KI.get_sub_group_id()`, never by the
#   local index of the work-item, since KernelInterface does not specify how the
#   work-items of a work-group form sub-groups.
# The kernels assume that a 1-D work-group of `32 * k` work-items consists of `k` full
#   sub-groups of 32 work-items, which is checked once per backend, see
#   `check_tiled_backend`.

# Atoms per block, which is also the sub-group width the kernels need, since a lane
#   holds one atom of a block and the tile masks have a bit per lane
const TILE_WIDTH = 32
# Values stored in `interacting_tiles_type`: a tile is either free of exclusions and
#   special pairs (0), mask-backed (1), or known to hold no in-cutoff atom pair at all
#   and skipped entirely by the force/energy kernels (`TILE_DEAD`)
const TILE_DEAD = UInt8(2)
const MAX_BLOCK_Y = 32
const DEFAULT_BLOCK_Y = 4
const DEFAULT_TILE_THREADS = (32, 8)
# Number of blocks from which the tree search is used
# Below this the all-pairs search is faster, since there are too few blocks to keep
#   the GPU busy with one work-item walking the tree per block
const TILE_TREE_MIN_BLOCKS = 6000

## Backend support

const TILED_BACKEND_PROBES = Dict{Any, Bool}()
const TILED_BACKEND_LOCK = ReentrantLock()

# Records the sub-group of every work-item, to check how a backend forms sub-groups
function sub_group_probe_kernel!(out)
    l = KI.get_local_id(Int32).x
    out[1, l] = KI.get_sub_group_id(Int32)
    out[2, l] = KI.get_sub_group_local_id(Int32)
    out[3, l] = KI.get_sub_group_size(Int32)
    out[4, l] = KI.get_num_sub_groups(Int32)
    return nothing
end

# Whether 1-D work-groups of `n_items` work-items, a multiple of 32, consist of full
#   sub-groups of 32 work-items
# KernelInterface only guarantees this for a work-group of 32 work-items, the tiled
#   kernels with several tiles per work-group rely on it for larger work-groups
function probe_full_sub_groups(backend, n_items::Integer)
    key = (backend, Int(n_items))
    lock(TILED_BACKEND_LOCK) do
        get!(TILED_BACKEND_PROBES, key) do
            out = KernelAbstractions.zeros(backend, Int32, 4, n_items)
            KI.@launch backend numgroups=1 workgroupsize=n_items sub_group_probe_kernel!(out)
            res = from_device(out)
            n_sg = n_items ÷ TILE_WIDTH
            all(==(TILE_WIDTH), res[3, :]) && all(==(n_sg), res[4, :]) &&
                length(unique(zip(res[1, :], res[2, :]))) == n_items &&
                all(sg -> 1 <= sg <= n_sg, res[1, :])
        end
    end
end

# Shuffles in a branch that only every other sub-group of the work-group takes
function divergent_sub_group_probe_kernel!(out)
    sg = KI.get_sub_group_id(Int32)
    lane = KI.get_sub_group_local_id(Int32)
    if isodd(sg)
        val = KI.shfl_xor(lane, Int32(1))
        if lane == Int32(1)
            out[sg] = val
        end
    end
    return nothing
end

#=
Whether the sub-groups of a work-group of `n_items` work-items can take different branches
around sub-group operations, as KernelInterface specifies.
The CPU backend of KernelAbstractions (POCL) implements sub-group operations with
work-group barriers, so a sub-group operation in a branch that only some sub-groups of
the work-group take runs on all of them. The tiled kernels then use work-groups of a single
sub-group, where the branches of a sub-group are those of its work-group.
=#
function probe_divergent_sub_groups(backend, n_items::Integer)
    key = (backend, Int(n_items), :divergent)
    lock(TILED_BACKEND_LOCK) do
        get!(TILED_BACKEND_PROBES, key) do
            n_sg = n_items ÷ TILE_WIDTH
            out = KernelAbstractions.zeros(backend, Int32, n_sg)
            KI.@launch backend numgroups=1 workgroupsize=n_items divergent_sub_group_probe_kernel!(out)
            res = from_device(out)
            all(sg -> res[sg] == (isodd(sg) ? 2 : 0), 1:n_sg)
        end
    end
end

# The number of tiles, i.e. sub-groups, per work-group to use for a requested `block_y`
function tiles_per_group(backend, block_y::Integer)
    block_y > 1 || return 1
    if !probe_full_sub_groups(backend, TILE_WIDTH * block_y) ||
            !probe_divergent_sub_groups(backend, TILE_WIDTH * block_y)
        return 1
    end
    return Int(block_y)
end

"""
    supports_tiled_kernels(backend)

Whether Molly's tiled pairwise kernels, used with [`GPUNeighborFinder`](@ref), can run on
a KernelAbstractions backend: it has to support sub-groups of 32 work-items with shuffles.
"""
function supports_tiled_kernels(backend)
    KI.supports_subgroups(backend) || return false
    KI.sub_group_size(backend) == TILE_WIDTH || return false
    return all(T -> KI.supports_shuffle(backend, T), (Int32, UInt32, Float32))
end

function check_tiled_backend(backend, ::Type{T}, n_items::Integer=TILE_WIDTH) where T
    if !KI.supports_subgroups(backend)
        throw(ArgumentError("GPUNeighborFinder needs a backend that supports sub-groups, " *
                            "$backend does not, use DistanceNeighborFinder instead"))
    end
    width = KI.sub_group_size(backend)
    if width != TILE_WIDTH
        # The tiles hold a bit per lane in UInt32 masks, so 64-wide sub-groups (e.g. AMD
        #   wavefronts) would need UInt64 masks and 64-atom blocks
        throw(ArgumentError("GPUNeighborFinder needs sub-groups of $TILE_WIDTH work-items, " *
                            "$backend has sub-groups of $width work-items, use " *
                            "DistanceNeighborFinder instead"))
    end
    for S in (Int32, UInt32, T)
        if !KI.supports_shuffle(backend, S)
            throw(ArgumentError("GPUNeighborFinder needs a backend that supports shuffles " *
                                "of $S, $backend does not, use DistanceNeighborFinder instead"))
        end
    end
    if n_items > TILE_WIDTH && !probe_full_sub_groups(backend, n_items)
        throw(ArgumentError("GPUNeighborFinder needs work-groups of $n_items work-items to " *
                            "consist of full sub-groups of $TILE_WIDTH work-items, which " *
                            "$backend does not do"))
    end
    return backend
end

#=
    tile_kernel_config(backend, T, maxregs)

The backend and compiler options to compile the tiled pairwise force and energy kernels
with, for coordinates of element type `T`. Backend extensions can return a backend with
different compiler settings, e.g. with fast math, and backend-specific compiler options,
which are passed to `KernelInterface.kernel_function`. `maxregs` is the
`force_maxregs` override, or `nothing`.
=#
tile_kernel_config(backend, ::Type{T}, maxregs) where {T} = (backend, (;))

# Compile a KernelInterface kernel with the given compiler options and launch it
function launch_ki_kernel!(backend, f::F, args...; numgroups, workgroupsize,
                           options=(;)) where F
    tt = KI.argument_types(backend, args)
    kernel = KI.kernel_function(backend, f, tt; options...)
    kernel(args...; numgroups=numgroups, workgroupsize=workgroupsize)
    return nothing
end

## Launch parameters

function env_int(name::AbstractString)
    value = ENV[name]
    parsed = tryparse(Int, value)
    parsed === nothing && error("invalid integer value for $(name): $(repr(value))")
    return parsed
end

env_override(name::AbstractString) = haskey(ENV, name) ? env_int(name) : nothing

prefer_override(primary, secondary) = primary === nothing ? secondary : primary

function validate_block_y(name::AbstractString, block_y::Int)
    1 <= block_y <= MAX_BLOCK_Y || error("$(name) must be in 1:$(MAX_BLOCK_Y), got $(block_y)")
    return block_y
end

# Per-j-atom data staged in local memory by force_kernel!'s Part 1 inner loop.
# Only the Atom fields the active interactions actually read are staged (`P` is the
# narrow payload tuple type from `atom_shuffle_payload`/`resolve_atom_fields`)
struct JStage{V, P}
    coords::V
    atom_payload::P
end

# Compile-time-only: the Tuple type produced by `atom_shuffle_payload(atom::A, Val(syms))`,
# without needing an atom instance. Must be kept in sync with that function so that
# host-side local memory sizing and the device-side staged layout agree byte-for-byte.
@inline atom_payload_type(::Type{A}, ::Val{syms}) where {A, syms} = Tuple{map(s -> fieldtype(A, s), syms)...}

# The Val(syms) of Atom fields the active interactions actually read, resolved once
# and shared between host-side local memory sizing and the device kernels
@inline function resolved_atom_shuffle_syms(pairwise_inters, ::Type{A}) where {A}
    return Val(resolve_atom_fields(combine_atom_fields(pairwise_inters), A))
end

# Local memory (bytes) of force_kernel!: opposites_sum and the staged j coords/atoms/velocities
function force_kernel_localmem(buffers, ::Val{D}, ::Type{T}, uses_vel::Bool,
                               block_y::Integer, pairwise_inters) where {D, T}
    nslot = TILE_WIDTH * block_y
    A = eltype(buffers.atoms_reordered)
    P = atom_payload_type(A, resolved_atom_shuffle_syms(pairwise_inters, A))
    JT = JStage{eltype(buffers.coords_reordered), P}
    bytes = nslot * D * sizeof(T) + nslot * sizeof(JT)
    if uses_vel
        bytes += nslot * sizeof(eltype(buffers.velocities_reordered))
    end
    return bytes
end

# Local memory (bytes) of energy_kernel!: the staged j coords/atoms (and velocities)
function energy_kernel_localmem(buffers, uses_vel::Bool, block_y::Integer, pairwise_inters)
    nslot = TILE_WIDTH * block_y
    A = eltype(buffers.atoms_reordered)
    P = atom_payload_type(A, resolved_atom_shuffle_syms(pairwise_inters, A))
    JT = JStage{eltype(buffers.coords_reordered), P}
    bytes = nslot * sizeof(JT)
    if uses_vel
        bytes += nslot * sizeof(eltype(buffers.velocities_reordered))
    end
    return bytes
end

#=
The local memory of the tiled kernels is sized at compile time, since KernelInterface has
no dynamically sized local memory. Static local memory is limited to 48 KiB on CUDA,
which is also a portable lower bound on other GPUs.
=#
const MAX_TILE_LOCALMEM = 48 * 1024

# The largest block_y not above `block_y` for which the local memory fits
function clamp_block_y(block_y::Integer, localmem_bytes)
    while block_y > 1 && localmem_bytes(block_y) > MAX_TILE_LOCALMEM
        block_y = prevpow(2, block_y - 1)
    end
    return block_y
end

function force_block_y(sys, buffers, pairwise_inters)
    config = cuda_launch_config(sys)
    block_y_override = prefer_override(cuda_force_block_y(config),
                                       env_override("MOLLY_CUDA_FORCE_BLOCK_Y"))
    block_y_override === nothing || validate_block_y("MOLLY_CUDA_FORCE_BLOCK_Y", block_y_override)
    block_y = something(block_y_override, DEFAULT_BLOCK_Y)
    D, T = length(eltype(sys.coords)), float_type(sys)
    uses_vel = any_uses_velocity(pairwise_inters)
    return clamp_block_y(block_y, by -> force_kernel_localmem(buffers, Val(D), T, uses_vel,
                                                              by, pairwise_inters))
end

function force_maxregs(sys)
    config = cuda_launch_config(sys)
    maxregs = prefer_override(cuda_force_maxregs(config), env_override("MOLLY_CUDA_FORCE_MAXREGS"))
    maxregs === nothing || maxregs > 0 ||
        error("MOLLY_CUDA_FORCE_MAXREGS must be positive, got $(maxregs)")
    return maxregs
end

function energy_block_y(sys, buffers, pairwise_inters)
    config = cuda_launch_config(sys)
    block_y_override = prefer_override(cuda_energy_block_y(config),
                                       env_override("MOLLY_CUDA_ENERGY_BLOCK_Y"))
    block_y_override === nothing || validate_block_y("MOLLY_CUDA_ENERGY_BLOCK_Y", block_y_override)
    block_y = something(block_y_override, DEFAULT_BLOCK_Y)
    uses_vel = any_uses_velocity(pairwise_inters)
    return clamp_block_y(block_y, by -> energy_kernel_localmem(buffers, uses_vel, by,
                                                               pairwise_inters))
end

function tile_threads(sys)
    config = cuda_launch_config(sys)
    config_tile_threads = cuda_tile_threads(config)
    threads_x = config_tile_threads === nothing ? env_override("MOLLY_CUDA_TILE_THREADS_X") : config_tile_threads[1]
    threads_y = config_tile_threads === nothing ? env_override("MOLLY_CUDA_TILE_THREADS_Y") : config_tile_threads[2]
    if xor(threads_x === nothing, threads_y === nothing)
        error("set both MOLLY_CUDA_TILE_THREADS_X and MOLLY_CUDA_TILE_THREADS_Y together")
    end
    threads_x === nothing && return DEFAULT_TILE_THREADS
    threads_x > 0 || error("MOLLY_CUDA_TILE_THREADS_X must be positive, got $(threads_x)")
    threads_y > 0 || error("MOLLY_CUDA_TILE_THREADS_Y must be positive, got $(threads_y)")
    # Clamp overrides to the work-group size limit, keeping x if possible
    max_threads = KI.max_work_group_size(get_backend(sys.coords))
    actual_x = min(threads_x, max_threads)
    actual_y = min(threads_y, fld(max_threads, actual_x))
    return (actual_x, actual_y)
end

#=
Squared distance past which `force_kernel!`/`energy_kernel!` can skip a pair.
The tile list is built with the neighbor finder's buffered cutoff so that it stays
valid between rebuilds, but a pair only needs evaluating out to the largest cutoff
of the active interactions.
=#
function kernel_pair_cutoff_2(sys, pairwise_inters)
    nf_cutoff_2 = sys.neighbor_finder.dist_cutoff_2
    inter_cutoff_2 = max_zero_beyond(pairwise_inters)
    isnothing(inter_cutoff_2) && return nf_cutoff_2
    return min(inter_cutoff_2, nf_cutoff_2)
end

@inline function triclinic_boundary_matrix(boundary::TriclinicBoundary, ::Type{T}) where T
    return SMatrix{3, 3, T}(
        ustrip(boundary.basis_vectors[1][1]), ustrip(boundary.basis_vectors[2][1]), ustrip(boundary.basis_vectors[3][1]),
        ustrip(boundary.basis_vectors[1][2]), ustrip(boundary.basis_vectors[2][2]), ustrip(boundary.basis_vectors[3][2]),
        ustrip(boundary.basis_vectors[1][3]), ustrip(boundary.basis_vectors[2][3]), ustrip(boundary.basis_vectors[3][3]),
    )
end

## Host-side pipeline

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

@inline function gpu_neighbor_refresh_flags(buffers, nf::GPUNeighborFinder, step_n)
    first_preprocess = (buffers.step_n_preprocessed == -1)
    step_changed = (step_n != buffers.step_n_preprocessed)
    needs_morton_refresh = (first_preprocess || (step_changed && step_n % nf.n_steps == 0))
    sparse_changed = (buffers.sparse_pair_generation != nf.cache_generation)
    needs_reorder = true # Coordinates can be changed at any point
    needs_sparse_refresh = (needs_morton_refresh || !nf.initialized || sparse_changed)
    needs_tile_refresh = (needs_morton_refresh || !nf.initialized || sparse_changed)
    return needs_morton_refresh, needs_reorder, needs_sparse_refresh, needs_tile_refresh
end

#=
    preprocess_tiles!(buffers, sys, step_n)

Run the Morton reorder -> exception translation -> tile search pipeline as far as the
cached state in `buffers` and the [`GPUNeighborFinder`](@ref) require for `step_n`.

Cache contract:
- `buffers.step_n_preprocessed` gates reuse of reordered coordinates and tile
  search work within a simulation step.
- `buffers.num_pairs` is the host-side cached interacting-tile count used to
  size the force/energy kernel launch.
- `sys.neighbor_finder.initialized` only indicates whether the Morton-ordered
  exception data is current. The interacting-tile list still depends on
  the neighbor finder's `n_steps` and `dist_cutoff`.
=#
function preprocess_tiles!(buffers, sys, step_n)
    N = length(sys.coords)
    nf = sys.neighbor_finder
    backend = get_backend(sys.coords)
    needs_morton_refresh, needs_reorder, needs_sparse_refresh, needs_tile_refresh =
        gpu_neighbor_refresh_flags(buffers, nf, step_n)
    if needs_reorder || needs_sparse_refresh || needs_tile_refresh
        if needs_morton_refresh
            morton_bits = 10
            sides = box_sides(sys.boundary)
            w = sides ./ (2^morton_bits)
            sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
        end

        if needs_sparse_refresh
            refresh_tile_exceptions!(buffers, nf, Val(N))
            nf.initialized = true
            buffers.sparse_pair_generation = nf.cache_generation
        end

        if needs_reorder
            reorder_system_gpu!(buffers, sys, Val(needs_morton_refresh))
            KernelAbstractions.synchronize(backend)
        end

        if needs_tile_refresh
            refresh_interacting_tiles!(buffers, sys, N)
        end
        buffers.step_n_preprocessed = step_n
    end
    return buffers
end

"""
    tiled_pairwise_forces!(buffers, sys, pairwise_inters, needs_vir, step_n)

Maintainer entry point for the tiled pairwise force path of [`GPUNeighborFinder`](@ref).

Pipeline:
1. Rebuild the Morton ordering and the Morton-ordered exception data when the
   [`GPUNeighborFinder`](@ref) reorder cadence invalidates them.
2. Reorder coordinates, velocities, and atoms into Morton order.
3. Recompute the compact list of interacting 32x32 tiles when the cached tile
   list is stale for the current `dist_cutoff`, growing it if it overflows.
4. Launch `force_kernel!` over that compact tile list and reverse the reorder.

This works on any backend that `supports_tiled_kernels`, for any array type, which
lets the GPU code path be tested with `Array`s on the CPU backend of KernelAbstractions.
"""
function tiled_pairwise_forces!(buffers, sys::System{D, <:Any, T}, pairwise_inters,
                                ::Val{needs_vir}, step_n) where {D, T, needs_vir}
    backend = get_backend(sys.coords)
    check_tiled_backend(backend, T)
    preprocess_tiles!(buffers, sys, step_n)
    block_y = force_block_y(sys, buffers, pairwise_inters)
    launch_force_tiles!(buffers, sys, pairwise_inters, Val(needs_vir), step_n, block_y,
                        force_maxregs(sys))
    reverse_reorder_forces_gpu!(buffers, sys)
    return buffers
end

"""
    tiled_pairwise_pe!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n)

Maintainer entry point for the tiled pairwise energy path of [`GPUNeighborFinder`](@ref).

This follows the same Morton reorder -> exception translation -> tile search
pipeline as `tiled_pairwise_forces!`, but launches `energy_kernel!` instead
of the force kernel. Energy evaluation reuses any preprocessing already performed for
the current step so forces and energies can share the same cached tile metadata.
"""
function tiled_pairwise_pe!(pe_vec_nounits, buffers, sys::System{D, <:Any, T}, pairwise_inters,
                            step_n) where {D, T}
    backend = get_backend(sys.coords)
    check_tiled_backend(backend, T)
    preprocess_tiles!(buffers, sys, step_n)
    block_y = energy_block_y(sys, buffers, pairwise_inters)
    launch_energy_tiles!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n, block_y)
    return pe_vec_nounits
end

function pairwise_forces_loop_gpu!(buffers, sys::System{D, <:AbstractGPUArray}, pairwise_inters,
                                   ::Nothing, ::Val{needs_vir}, step_n) where {D, needs_vir}
    return tiled_pairwise_forces!(buffers, sys, pairwise_inters, Val(needs_vir), step_n)
end

function pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys::System{<:Any, <:AbstractGPUArray},
                               pairwise_inters, ::Nothing, step_n)
    return tiled_pairwise_pe!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n)
end

function launch_force_tiles!(buffers, sys::System{D, <:Any, T, TH}, pairwise_inters,
                             ::Val{needs_vir}, step_n, block_y::Integer,
                             maxregs=nothing) where {D, T, TH, needs_vir}
    num_pairs = buffers.num_pairs
    num_pairs > 0 || return buffers
    N = length(sys.coords)
    backend = get_backend(sys.coords)
    block_y = tiles_per_group(backend, block_y)
    n_items = TILE_WIDTH * block_y
    check_tiled_backend(backend, T, n_items)
    kernel_backend, options = tile_kernel_config(backend, T, maxregs)
    uses_vel = any_uses_velocity(pairwise_inters)
    launch_ki_kernel!(kernel_backend, force_kernel!,
        buffers.fs_mat_reordered,
        buffers.virial_nounits,
        buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
        Val(N), Val(kernel_pair_cutoff_2(sys, pairwise_inters)), Val(sys.force_units),
        pairwise_inters, sys.boundary, step_n, tile_exceptions(buffers, sys.neighbor_finder),
        Val(needs_vir), Val(T), Val(TH), Val(D), Val(Int(block_y)), Val(uses_vel),
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.interacting_tiles_diag, buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow;
        numgroups=cld(num_pairs, block_y), workgroupsize=n_items, options=options,
    )
    return buffers
end

function launch_energy_tiles!(pe_vec_nounits, buffers, sys::System{D, <:Any, T, TH},
                              pairwise_inters, step_n, block_y::Integer) where {D, T, TH}
    num_pairs = buffers.num_pairs
    num_pairs > 0 || return pe_vec_nounits
    N = length(sys.coords)
    backend = get_backend(sys.coords)
    block_y = tiles_per_group(backend, block_y)
    n_items = TILE_WIDTH * block_y
    check_tiled_backend(backend, T, n_items)
    kernel_backend, options = tile_kernel_config(backend, T, nothing)
    uses_vel = any_uses_velocity(pairwise_inters)
    launch_ki_kernel!(kernel_backend, energy_kernel!,
        pe_vec_nounits, buffers.coords_reordered,
        buffers.velocities_reordered, buffers.atoms_reordered, Val(N),
        Val(kernel_pair_cutoff_2(sys, pairwise_inters)), Val(sys.energy_units), pairwise_inters,
        sys.boundary, step_n, tile_exceptions(buffers, sys.neighbor_finder),
        Val(T), Val(TH), Val(D), Val(Int(block_y)), Val(uses_vel),
        buffers.interacting_tiles_i, buffers.interacting_tiles_j,
        buffers.interacting_tiles_type, buffers.interacting_tiles_diag,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow;
        numgroups=cld(num_pairs, block_y), workgroupsize=n_items, options=options,
    )
    return pe_vec_nounits
end

# The bounding boxes of the 32-atom blocks
function compute_block_bounds!(buffers, sys::System{D, <:Any, T}, N::Integer) where {D, T}
    n_blocks = cld(N, TILE_WIDTH)
    backend = get_backend(sys.coords)
    Hinv = sys.boundary isa TriclinicBoundary ?
           inv(triclinic_boundary_matrix(sys.boundary, T)) : nothing
    kernel_min_max!(backend, TILE_WIDTH)(
        buffers.box_mins, buffers.box_maxs, buffers.morton_seq, sys.coords, Hinv, Val(N),
        sys.boundary, Val(D); ndrange=n_blocks * TILE_WIDTH)
    return buffers
end

function refresh_interacting_tiles!(buffers, sys, N::Int)
    n_blocks = cld(N, TILE_WIDTH)
    compute_block_bounds!(buffers, sys, N)
    find_interacting_tiles!(buffers, sys, n_blocks)
    prune_interacting_tiles!(buffers, sys, N)
    return nothing
end

# Refine the bounding-box candidates with an exact atom-pair test, which also decides which
#   tiles hold exclusions or special pairs
function prune_interacting_tiles!(buffers, sys, N::Int)
    n_blocks = cld(N, TILE_WIDTH)
    if buffers.num_pairs > 0
        backend = get_backend(sys.coords)
        prune_y = tiles_per_group(backend, 4)
        n_items = TILE_WIDTH * prune_y
        check_tiled_backend(backend, UInt32, n_items)
        prune_interacting_tiles_kernel!(backend, n_items)(
            buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.interacting_tiles_diag,
            buffers.num_interacting_tiles,
            buffers.coords_reordered, Val(N), Val(sys.neighbor_finder.dist_cutoff_2),
            Val(n_blocks), sys.boundary, tile_exceptions(buffers, sys.neighbor_finder),
            Val(prune_y); ndrange=cld(buffers.num_pairs, prune_y) * n_items)
    end
    return buffers
end

#=
Build the list of interacting tiles and set `buffers.num_pairs` to its length.
The tile counter is incremented for every tile found, including those that do not fit,
so when the list overflows its vectors are grown to the number found and the search is
repeated. The vectors start at a size that is enough for typical systems, see
`init_buffers!`, and are kept for later searches.
=#
function find_interacting_tiles!(buffers, sys, n_blocks)
    search! = use_tile_tree(sys, n_blocks) ? find_interacting_tiles_tree! :
                                             find_interacting_tiles_all_pairs!
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

function launch_find_interacting_blocks!(buffers, sys::System{D}, n_blocks,
                                         threads_xy::NTuple{2, Int}) where D
    backend = get_backend(sys.coords)
    find_interacting_blocks_kernel!(backend)(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, sys.boundary, Val(sys.neighbor_finder.dist_cutoff_2),
        Val(n_blocks), Val(D), length(buffers.interacting_tiles_i),
        buffers.block_exc_min, buffers.block_exc_max;
        ndrange=(n_blocks, n_blocks), workgroupsize=threads_xy)
    return buffers
end

# Search every pair of blocks for interacting tiles
function find_interacting_tiles_all_pairs!(buffers, sys, n_blocks)
    return launch_find_interacting_blocks!(buffers, sys, n_blocks, tile_threads(sys))
end

# Search for interacting tiles by walking a tree of block bounding boxes
function find_interacting_tiles_tree!(buffers, sys::System{D}, n_blocks) where D
    top, offsets = build_tile_tree!(buffers, sys, n_blocks, Val(D))
    backend = get_backend(sys.coords)
    find_interacting_blocks_tree_kernel!(backend, 128)(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, buffers.tree_mins, buffers.tree_maxs, offsets,
        Val(top), sys.boundary, Val(sys.neighbor_finder.dist_cutoff_2), Val(n_blocks), Val(D),
        length(buffers.interacting_tiles_i), buffers.block_exc_min, buffers.block_exc_max;
        ndrange=n_blocks)
    return buffers
end

function use_tile_tree(sys, n_blocks)
    sys.boundary isa TriclinicBoundary && return false
    min_blocks = something(env_override("MOLLY_CUDA_TILE_TREE_MIN_BLOCKS"), TILE_TREE_MIN_BLOCKS)
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

function build_tile_tree!(buffers, sys, n_blocks, ::Val{D}) where D
    top, offsets = tile_tree_levels(n_blocks)
    backend = get_backend(sys.coords)
    for k in 1:top
        n_src = cld(n_blocks, 1 << (k - 1))
        n_dst = cld(n_blocks, 1 << k)
        if k == 1
            src_mins, src_maxs, src_offset = buffers.box_mins, buffers.box_maxs, Int32(0)
        else
            src_mins, src_maxs, src_offset = buffers.tree_mins, buffers.tree_maxs, offsets[k - 1]
        end
        build_tile_tree_level_kernel!(backend, 256)(
            buffers.tree_mins, buffers.tree_maxs, src_mins, src_maxs, src_offset,
            offsets[k], Int32(n_src), Val(D); ndrange=n_dst)
    end
    return top, offsets
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
    backend = get_backend(sys.coords)
    reorder_system_kernel!(backend, 256)(
        buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
        sys.coords, sys.velocities, sys.atoms, buffers.morton_seq, Val(reorder_atoms);
        ndrange=N)
    return nothing
end

"""
    reverse_reorder_forces_gpu!(buffers, sys)

Maps forces from the `fs_mat_reordered` buffer back to the original atom indices
in `fs_mat`.
"""
function reverse_reorder_forces_gpu!(buffers, sys::System{D}) where D
    N = length(sys)
    backend = get_backend(sys.coords)
    reverse_reorder_forces_kernel!(backend, 256)(
        buffers.fs_mat, buffers.fs_mat_reordered, buffers.morton_seq, Val(D); ndrange=N)
    return nothing
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
Morton-ordered form read by the tiled pairwise kernels.

This needs work proportional to the number of atoms and exceptions: the inverse Morton
map is updated, the partner of each exception is converted to its Morton position and
the range of blocks holding the partners of the atoms of each 32-atom block is found,
which lets the tile search mark most tiles as free of exceptions without looking at
the atoms.
"""
function refresh_tile_exceptions!(buffers, nf::GPUNeighborFinder, ::Val{N}) where N
    n_blocks = cld(N, TILE_WIDTH)
    backend = get_backend(buffers.morton_seq)
    update_inv_morton_kernel!(backend, 256)(buffers.morton_seq_inv, buffers.morton_seq;
                                            ndrange=N)

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

    n_items = 256
    check_tiled_backend(backend, Float32, n_items)
    tile_exceptions_kernel!(backend, n_items)(
        buffers.excluded_pos, buffers.special_pos, buffers.block_exc_min,
        buffers.block_exc_max, nf.eligible.starts, nf.eligible.partners,
        nf.special.starts, nf.special.partners, buffers.morton_seq,
        buffers.morton_seq_inv, Val(N), Val(n_blocks); ndrange=cld(N, n_items) * n_items)
    return buffers
end

## Device-side helpers

@inline function boxes_dist(r1_min::SVector{D, T}, r1_max::SVector{D, T}, r2_min::SVector{D, T},
                            r2_max::SVector{D, T}, boundary) where {D, T}
    va = vector(r2_max, r1_min, boundary)
    vb = vector(r1_max, r2_min, boundary)
    a = SVector{D}(ntuple(d -> abs(va[d]), D))
    b = SVector{D}(ntuple(d -> abs(vb[d]), D))

    return SVector(ntuple(d -> r1_min[d] - r2_max[d] <= zero(T) && r2_min[d] - r1_max[d] <= zero(T) ? zero(T) : ifelse(a[d] < b[d], a[d], b[d]), D))
end

# Triclinic boxes case is treated only in 3 dimensions
@inline function boxes_dist(r1_min::SVector{D, T}, r1_max::SVector{D, T}, r2_min::SVector{D, T},
                            r2_max::SVector{D, T}, boundary::TriclinicBoundary) where {D, T}
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

# The coordinates the bounding boxes of the blocks are computed in: the coordinates
#   themselves, or fractional coordinates `Hinv * r` for a triclinic boundary
@inline block_bound_coords(::Nothing, r) = r

@inline function block_bound_coords(Hinv::SMatrix{D, D}, r::SVector{D, C}) where {D, C}
    return SVector{D, C}(ntuple(Val(D)) do k
        val = zero(C)
        for j in 1:D
            val += Hinv[k, j] * r[j]
        end
        val
    end)
end

@inline min_max_op(a, b) = (min.(a[1], b[1]), max.(a[2], b[2]))

"""
    kernel_min_max!(mins, maxs, sorted_seq, coords, Hinv, n_atoms, boundary, D)

Compute the minimum and maximum coordinates for each 32-atom block, with one work-group
of 32 work-items, a single sub-group, per block. With a triclinic boundary the bounds are
computed in fractional coordinates using `Hinv`.
These bounds are used for fast bounding-box intersection tests during tile finding
to skip non-interacting tile pairs.
"""
@kernel unsafe_indices=true inbounds=true function kernel_min_max!(
                    mins::AbstractArray{C}, maxs, @Const(sorted_seq), @Const(coords), Hinv,
                    ::Val{N}, boundary, ::Val{D}) where {C, N, D}
    block = @index(Group, Linear)
    lane = KI.get_sub_group_local_id(Int32)
    i = (block - 1) * TILE_WIDTH + lane
    # Very large (arbitrary) bounds, which the lanes past the last atom contribute
    big = SVector{D, C}(ntuple(k -> 10 * box_sides(boundary, k), Val(D)))
    if i <= N
        r = block_bound_coords(Hinv, coords[sorted_seq[i]])
        lo, hi = r, r
    else
        lo, hi = big, -big
    end
    lo, hi = @subgroupreduce(min_max_op, (lo, hi), (big, -big))
    if lane == 1
        for k in 1:D
            mins[block, k] = lo[k]
            maxs[block, k] = hi[k]
        end
    end
end

"""
    update_inv_morton_kernel!(inv_morton_seq, morton_seq)

Generate the inverse Morton mapping.
Scatters the dense `1:N` original atom indices into their new Morton-ordered
positions in `inv_morton_seq`.
"""
@kernel inbounds=true function update_inv_morton_kernel!(inv_morton_seq, @Const(morton_seq))
    i = @index(Global, Linear)
    inv_morton_seq[morton_seq[i]] = i
end

@kernel inbounds=true function reorder_system_kernel!(coords_reordered, velocities_reordered,
                        atoms_reordered, @Const(coords), @Const(velocities), @Const(atoms),
                        @Const(seq), ::Val{reorder_atoms}) where reorder_atoms
    i = @index(Global, Linear)
    original_i = seq[i]
    coords_reordered[i] = coords[original_i]
    velocities_reordered[i] = velocities[original_i]
    if reorder_atoms
        atoms_reordered[i] = atoms[original_i]
    end
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

@inline min_max_block_op(a, b) = (min(a[1], b[1]), max(a[2], b[2]))

#=
One work-item per Morton position `p`, so that each sub-group covers one 32-atom block.
The work-item converts the exception partners of the atom at `p` to Morton positions,
writing each list entry once since each atom belongs to one position. The sub-group then
reduces the lowest and highest block holding a partner of any atom in the block, which
is `typemax(Int32)` and `0` for a block without exceptions.
=#
@kernel unsafe_indices=true inbounds=true function tile_exceptions_kernel!(excluded_pos,
                    special_pos, block_exc_min, block_exc_max, @Const(excluded_starts),
                    @Const(excluded_partners), @Const(special_starts),
                    @Const(special_partners), @Const(morton_seq), @Const(morton_seq_inv),
                    ::Val{N}, ::Val{n_blocks}) where {N, n_blocks}
    lane = KI.get_sub_group_local_id(Int32)
    sub_groups_per_group = KI.get_local_size(Int32).x ÷ Int32(TILE_WIDTH)
    block_i = (@index(Group, Linear) - Int32(1)) * sub_groups_per_group +
              KI.get_sub_group_id(Int32)
    p = (block_i - Int32(1)) * Int32(TILE_WIDTH) + lane
    min_block = typemax(Int32)
    max_block = Int32(0)
    if p <= N
        atom_i = morton_seq[p]
        min_block, max_block = translate_exception_list!(excluded_pos, excluded_starts,
                    excluded_partners, morton_seq_inv, atom_i, min_block, max_block)
        min_block, max_block = translate_exception_list!(special_pos, special_starts,
                    special_partners, morton_seq_inv, atom_i, min_block, max_block)
    end

    # Every work-item of the sub-group takes part, including those past the last atom
    min_block, max_block = @subgroupreduce(min_max_block_op, (min_block, max_block),
                                           (typemax(Int32), Int32(0)))

    if lane == Int32(1) && block_i <= n_blocks
        block_exc_min[block_i] = min_block
        block_exc_max[block_i] = max_block
    end
end

# Whether the atom at Morton position `p` has an exclusion or special pair with an atom
#   in block `block_i`
@inline function atom_has_partner_in_block(tile_exceptions, p, block_i)
    morton_seq = KernelAbstractions.constify(tile_exceptions.morton_seq)
    excluded_starts = KernelAbstractions.constify(tile_exceptions.excluded_starts)
    excluded_pos = KernelAbstractions.constify(tile_exceptions.excluded_pos)
    special_starts = KernelAbstractions.constify(tile_exceptions.special_starts)
    special_pos = KernelAbstractions.constify(tile_exceptions.special_pos)

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
        morton_seq = KernelAbstractions.constify(tile_exceptions.morton_seq)
        excluded_starts = KernelAbstractions.constify(tile_exceptions.excluded_starts)
        excluded_pos = KernelAbstractions.constify(tile_exceptions.excluded_pos)
        special_starts = KernelAbstractions.constify(tile_exceptions.special_starts)
        special_pos = KernelAbstractions.constify(tile_exceptions.special_pos)

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
@kernel inbounds=true function find_interacting_blocks_kernel!(
            interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
            num_interacting_tiles, interacting_tiles_overflow, @Const(mins), @Const(maxs),
            boundary, ::Val{r_cut2}, ::Val{N_blocks}, ::Val{D}, max_total_tiles,
            @Const(block_exc_min), @Const(block_exc_max)) where {r_cut2, N_blocks, D}
    i, j = @index(Global, NTuple)

    if i <= j
        r_min_i, r_max_i = stored_box(mins, maxs, i, Val(D))
        r_min_j, r_max_j = stored_box(mins, maxs, j, Val(D))
        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)

        if sum(d_block .* d_block) <= r_cut2
            emit_interacting_tile!(interacting_tiles_i, interacting_tiles_j,
                                   interacting_tiles_type, num_interacting_tiles,
                                   interacting_tiles_overflow, max_total_tiles,
                                   block_exc_min, block_exc_max, Int32(i), Int32(j),
                                   Int32(N_blocks))
        end
    end
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

    idx = (Atomix.@atomic :monotonic num_interacting_tiles[1] + Int32(1)).first + Int32(1)
    if idx <= max_total_tiles
        @inbounds interacting_tiles_i[idx] = i
        @inbounds interacting_tiles_j[idx] = j
        @inbounds interacting_tiles_type[idx] = is_clean ? UInt8(0) : UInt8(1)
    else
        Atomix.@atomic :monotonic interacting_tiles_overflow[1] += Int32(1)
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
`(n - 1) * 2^k + 1` to `n * 2^k`. One work-item per block `i` then walks the tree from the
root, skipping every subtree whose box is further than the cutoff from block `i` or
that only holds blocks `j < i`, which takes time proportional to the number of
interacting tiles times the depth of the tree.

The box distance only gets smaller as a box grows, so a subtree that is skipped holds
no block that the brute-force search would have accepted and the two searches give the
same tiles. Only orthorhombic boundaries use the tree, see `use_tile_tree`.
=#

# Fill node `n` of a tree level with the union of the boxes of its two children
@kernel inbounds=true function build_tile_tree_level_kernel!(tree_mins, tree_maxs,
                    src_mins, src_maxs, src_offset, dst_offset, n_src,
                    ::Val{D}) where D
    n = Int32(@index(Global, Linear))
    c1 = Int32(2) * n - Int32(1)
    c2 = Int32(2) * n
    for d in 1:D
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

# The bounding box stored in row `idx` of a pair of (n, D) min and max arrays
@inline function stored_box(mins, maxs, idx, ::Val{D}) where D
    r_min = SVector{D}(ntuple(d -> @inbounds(mins[idx, d]), Val(D)))
    r_max = SVector{D}(ntuple(d -> @inbounds(maxs[idx, d]), Val(D)))
    return r_min, r_max
end

@kernel inbounds=true function find_interacting_blocks_tree_kernel!(
            interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
            num_interacting_tiles, interacting_tiles_overflow, @Const(mins), @Const(maxs),
            @Const(tree_mins), @Const(tree_maxs), tree_offsets, ::Val{top}, boundary,
            ::Val{r_cut2}, ::Val{N_blocks}, ::Val{D}, max_total_tiles,
            @Const(block_exc_min), @Const(block_exc_max)) where {top, r_cut2, N_blocks, D}
    i = @index(Global, Linear)
    r_min_i, r_max_i = stored_box(mins, maxs, i, Val(D))

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
                r_min_j, r_max_j = stored_box(mins, maxs, n, Val(D))
            else
                node = tree_offsets[k] + n
                r_min_j, r_max_j = stored_box(tree_mins, tree_maxs, node, Val(D))
            end
            d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)
            visit = sum(d_block .* d_block) <= r_cut2
        end
        if visit
            if k == Int32(0)
                emit_interacting_tile!(interacting_tiles_i, interacting_tiles_j,
                                       interacting_tiles_type, num_interacting_tiles,
                                       interacting_tiles_overflow, max_total_tiles,
                                       block_exc_min, block_exc_max, Int32(i), n,
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
end

#=
Second pruning pass over the candidate tiles emitted by
`find_interacting_blocks_kernel!`.

The bounding-box test in that kernel is loose: with 32 atoms per block the boxes are
comparable in size to the cutoff, so most tiles it emits are largely empty. This pass
runs one sub-group per candidate tile and performs the exact 32x32 atom-pair distance
test, recording which of the 32 inner-loop iterations of `force_kernel!`/`energy_kernel!`
contain at least one in-range pair. Iteration `m` of those kernels evaluates the pairs
`(lane, slot)` with `slot - lane == m (mod 32)`, so a hit between the `l`-th atom of
block `i` and the `s`-th atom of block `j` sets bit `(s - l) & 31`. Tiles with no
in-range pair at all get an empty mask and are marked `TILE_DEAD` so the kernels drop
them immediately.

This only runs on a neighbor list rebuild, so its cost is amortised over the `n_steps`
steps of the neighbor finder. Only full off-diagonal tiles (`i < j < n_blocks`) are
analysed; the diagonal and the final partial block make up a vanishing fraction of the
list and keep a fully populated mask.

Tiles that the block ranges in `find_interacting_blocks_kernel!` could not rule out as
holding an exclusion or special pair are checked exactly here, and marked clean if
none of their atom pairs is an exception.
=#
@kernel unsafe_indices=true inbounds=true function prune_interacting_tiles_kernel!(
                    @Const(interacting_tiles_i), @Const(interacting_tiles_j),
                    interacting_tiles_type, interacting_tiles_diag,
                    @Const(num_interacting_tiles), @Const(coords), ::Val{N}, ::Val{r_cut2},
                    ::Val{n_blocks}, boundary, tile_exceptions,
                    ::Val{block_y}) where {N, r_cut2, n_blocks, block_y}
    a = Int32(1)
    lane = KI.get_sub_group_local_id(Int32)
    idx = (Int32(@index(Group, Linear)) - a) * Int32(block_y) + KI.get_sub_group_id(Int32)

    # The tile is uniform across the sub-group, so it returns as a whole
    num_pairs = num_interacting_tiles[1]
    if idx > num_pairs
        return nothing
    end

    i = interacting_tiles_i[idx]
    j = interacting_tiles_j[idx]
    if i >= j || j >= Int32(n_blocks)
        if lane == a
            interacting_tiles_diag[idx] = typemax(UInt32)
        end
        return nothing
    end

    # Read before the shuffles below, which every lane passes before lane 1 writes the
    # type, so the value and the branch on it are the same for the whole sub-group
    tile_type = interacting_tiles_type[idx]
    coords_j = coords[(j - a) * Int32(32) + lane]
    i_0_tile = (i - a) * Int32(32)

    # This lane holds the `lane`-th atom of block j, so a hit against the m-th atom of
    # block i lands on inner-loop iteration `(lane - m) & 31`
    lane_mask = UInt32(0)
    for m in a:Int32(32)
        coords_i = coords[i_0_tile + m]
        dr = vector(coords_i, coords_j, boundary)
        r2 = sum(abs2, dr)
        if r2 <= r_cut2
            lane_mask |= UInt32(1) << ((lane - m) & Int32(31))
        end
    end

    # Butterfly OR-reduction, so that every lane gets the mask of the tile
    offset = Int32(16)
    while offset > 0
        lane_mask |= KI.shfl_xor(lane_mask, offset)
        offset >>= Int32(1)
    end

    # Exact test for an exception between the blocks, from the atoms of block j
    has_exception = true
    if tile_type == UInt8(1) && lane_mask != UInt32(0)
        lane_has_exception = atom_has_partner_in_block(tile_exceptions,
                                                       (j - a) * Int32(32) + lane, i)
        has_exception = KI.sub_group_any(lane_has_exception)
    end

    if lane == a
        interacting_tiles_diag[idx] = lane_mask
        if lane_mask == UInt32(0)
            interacting_tiles_type[idx] = TILE_DEAD
        elseif !has_exception
            interacting_tiles_type[idx] = UInt8(0)
        end
    end
end

# The force on atom i of a pair, without units
@inline stripped_force(f, ::Val{D}, ::Type{T}) where {D, T} =
    SVector{D, T}(ntuple(k -> ustrip(f[k]), Val(D)))

# Add the virial contribution of a pair to (xx, yy, zz, xy, xz, yz)
@inline function add_pair_virial(vir::SVector{6, T}, f, dr, ::Val{D}) where {T, D}
    vir_xx, vir_yy, vir_zz, vir_xy, vir_xz, vir_yz = vir
    vir_xx += ustrip(f[1]) * ustrip(dr[1])
    if D >= 2
        vir_yy += ustrip(f[2]) * ustrip(dr[2])
        vir_xy += ustrip(f[1]) * ustrip(dr[2])
    end
    if D >= 3
        vir_zz += ustrip(f[3]) * ustrip(dr[3])
        vir_xz += ustrip(f[1]) * ustrip(dr[3])
        vir_yz += ustrip(f[2]) * ustrip(dr[3])
    end
    return SVector{6, T}(vir_xx, vir_yy, vir_zz, vir_xy, vir_xz, vir_yz)
end

# Subtract the force on atom i of a pair from the local accumulator of atom `slot`
@inline function sub_opposite!(opposites_sum, fi::SVector{D}, slot, warpid) where D
    for k in 1:D
        @inbounds opposites_sum[slot, k, warpid] -= fi[k]
    end
    return nothing
end

# Whether the pair of `lane` with atom `slot` of block j of a tile is evaluated, and whether
#   it is special: for a CLEAN tile (`masked = false`) only the distance decides
@inline pair_flags(::Val{false}, r2, r_cut2, eligible_bitmask, special_bitmask, slot) =
    (r2 <= r_cut2, false)

@inline function pair_flags(::Val{true}, r2, r_cut2, eligible_bitmask, special_bitmask, slot)
    w = Int32(TILE_WIDTH)
    excl = (eligible_bitmask >> (w - slot)) | (eligible_bitmask << slot)
    spec = (special_bitmask >> (w - slot)) | (special_bitmask << slot)
    return ((excl & 0x1) == true && r2 <= r_cut2, (spec & 0x1) == true)
end

#=
The forces of a full off-diagonal tile, Part 1 of `force_kernel!`, from the j atoms staged
in local memory.
Walk only the iterations `prune_interacting_tiles_kernel!` marked as holding an in-range
pair. Bit b covers the pairs with `slot - lane == b (mod 32)`, so it selects the same slot
permutation as `m == b`. Testing all 32 bits with a branch instead costs a loop header and
a divergent branch per skipped iteration. The mask is uniform across the sub-group, so all
lanes run the same iterations.
=#
@inline function full_tile_forces(::Val{masked}, force_i, vir, opposites_sum, sh_stage,
                    sh_vel, warpid, lane, diag_mask, eligible_bitmask, special_bitmask,
                    coords_i, vel_i, atoms_i, inters_tuple, boundary, step_n, ::Val{r_cut2},
                    ::Val{force_units}, ::Val{needs_vir}, ::Val{uses_vel}, ::Type{A},
                    shuf_syms, ::Val{D}, ::Type{T}) where {masked, r_cut2, force_units,
                                                           needs_vir, uses_vel, A, D, T}
    a = Int32(1)
    active = diag_mask
    @inbounds while active != UInt32(0)
        m = Int32(trailing_zeros(active))
        active &= active - UInt32(1)
        slot = ((lane - a + m) & Int32(31)) + a
        js = sh_stage[slot, warpid]
        coords_j = js.coords
        atoms_j_stage = rebuild_shuffled_atom(A, atoms_i, js.atom_payload, shuf_syms)
        vel_j = uses_vel ? sh_vel[slot, warpid] : vel_i

        dr = vector(coords_i, coords_j, boundary)
        r2 = sum(abs2, dr)
        condition, special = pair_flags(Val(masked), r2, r_cut2, eligible_bitmask,
                                        special_bitmask, slot)
        any_active = KI.sub_group_any(condition)

        if any_active
            f = condition ? sum_pairwise_forces_gpu(
                inters_tuple, dr, atoms_i, atoms_j_stage, Val(force_units),
                special, coords_i, coords_j, boundary, vel_i, vel_j, step_n
            ) : zero_pairwise_force(dr, force_units)

            fi = stripped_force(f, Val(D), T)
            force_i += fi
            # The slots of the lanes differ within an iteration
            sub_opposite!(opposites_sum, fi, slot, warpid)
            if needs_vir
                vir = add_pair_virial(vir, f, dr, Val(D))
            end
            # The next iteration updates the slot of another lane
            KI.sub_group_barrier()
        end
    end
    return force_i, vir
end

# The energy of a full off-diagonal tile, see `full_tile_forces`
@inline function full_tile_energy(::Val{masked}, sum_E, sh_stage, sh_vel, warpid, lane,
                    diag_mask, eligible_bitmask, special_bitmask, coords_i, vel_i, atoms_i,
                    inters_tuple, boundary, step_n, ::Val{r_cut2}, ::Val{energy_units},
                    ::Val{uses_vel}, ::Type{A}, shuf_syms) where {masked, r_cut2,
                                                                  energy_units, uses_vel, A}
    a = Int32(1)
    active = diag_mask
    @inbounds while active != UInt32(0)
        m = Int32(trailing_zeros(active))
        active &= active - UInt32(1)
        slot = ((lane - a + m) & Int32(31)) + a
        js = sh_stage[slot, warpid]
        coords_j = js.coords
        atoms_j_stage = rebuild_shuffled_atom(A, atoms_i, js.atom_payload, shuf_syms)
        vel_j = uses_vel ? sh_vel[slot, warpid] : vel_i

        dr = vector(coords_i, coords_j, boundary)
        r2 = sum(abs2, dr)
        condition, special = pair_flags(Val(masked), r2, r_cut2, eligible_bitmask,
                                        special_bitmask, slot)
        any_active = KI.sub_group_any(condition)

        if any_active
            pe = condition ? sum_pairwise_potentials_gpu(
                inters_tuple, dr, atoms_i, atoms_j_stage, Val(energy_units), special,
                coords_i, coords_j, boundary, vel_i, vel_j, step_n,
            ) : SVector(zero_pairwise_energy(dr, energy_units))

            sum_E += convert(typeof(sum_E), ustrip(pe[1]))
        end
    end
    return sum_E
end

"""
    force_kernel!(fs_mat, global_virial, coords, velocities, atoms, N, r_cut2, force_units,
                  inters_tuple, boundary, step_n, tile_exceptions, needs_vir, T, TH, D,
                  block_y, uses_vel, interacting_tiles_i, interacting_tiles_j,
                  interacting_tiles_type, interacting_tiles_diag, num_interacting_tiles,
                  interacting_tiles_overflow)

Compute pairwise forces for the compact list of interacting 32x32 tiles produced
by `find_interacting_blocks_kernel!`. This is a KernelInterface kernel, launched with
1-D work-groups of `32 * block_y` work-items.

Execution model:
- One sub-group of 32 work-items processes one tile of the compact list, so a
  work-group processes `block_y` tiles at once. The lane
  `KI.get_sub_group_local_id()` spans the atoms of a tile row, and the sub-group
  `KI.get_sub_group_id()` selects the tile.
- The kernel keeps the `i`-atom contribution in registers/local memory and
  atomically scatters the opposite contribution for the `j` atoms.

Tile cases:
1. Full off-diagonal tiles.
2. Boundary-column tiles containing the final partial atom block.
3. Diagonal tiles, where only unique pairs are evaluated.
4. The terminal corner tile, where both axes are partial.

CLEAN tiles skip the bitmasks entirely; mask-backed tiles build them from the
sparse exception lists in `tile_exceptions` to apply exclusions and special-pair
handling.

The lanes of a sub-group are not assumed to execute in lockstep: the local memory
accumulator `opposites_sum` is only updated by a single lane per slot between two
`KI.sub_group_barrier`s.
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
    ::Val{block_y},
    ::Val{uses_vel},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
    interacting_tiles_diag, num_interacting_tiles,
    interacting_tiles_overflow) where {N, r_cut2, A, force_units, needs_vir, T, TH, D,
                                       block_y, uses_vel}

    a = Int32(1)
    b = Int32(D)
    w = Int32(TILE_WIDTH)
    n_blocks = Int32(cld(N, TILE_WIDTH))
    coords = KernelAbstractions.constify(coords_var)
    velocities = KernelAbstractions.constify(velocities_var)
    atoms = KernelAbstractions.constify(atoms_var)
    tiles_i_ro = KernelAbstractions.constify(interacting_tiles_i)
    tiles_j_ro = KernelAbstractions.constify(interacting_tiles_j)
    tiles_type_ro = KernelAbstractions.constify(interacting_tiles_type)
    tiles_diag_ro = KernelAbstractions.constify(interacting_tiles_diag)
    num_interacting_tiles_ro = KernelAbstractions.constify(num_interacting_tiles)
    interacting_tiles_overflow_ro = KernelAbstractions.constify(interacting_tiles_overflow)

    lane = KI.get_sub_group_local_id(Int32)
    warpid = KI.get_sub_group_id(Int32)
    idx = (KI.get_group_id(Int32).x - a) * Int32(block_y) + warpid

    # Local memory for the j-force accumulator plus staged j-atom data
    # (coords/atoms/velocities). The Part 1 inner loop indexes the staged data by slot
    # instead of rotating it around the sub-group with serial shuffles.
    # Only the Atom fields the active interactions actually read are staged (not the
    # full Atom), matching what the old shuffle path sent per lane.
    shuf_syms = resolved_atom_shuffle_syms(inters_tuple, A)
    P = atom_payload_type(A, shuf_syms)
    JT = JStage{eltype(coords_var), P}
    opposites_sum = KI.localmemory(T, Val((TILE_WIDTH, D, block_y)))
    sh_stage = KI.localmemory(JT, Val((TILE_WIDTH, block_y)))
    sh_vel = uses_vel ? KI.localmemory(eltype(velocities_var), Val((TILE_WIDTH, block_y))) :
                        nothing

    # The conditions of these returns are uniform across the sub-group
    @inbounds if interacting_tiles_overflow_ro[1] != 0
        return nothing
    end

    @inbounds num_pairs = num_interacting_tiles_ro[1]
    if idx > num_pairs
        return nothing
    end

    @inbounds i = tiles_i_ro[idx]
    @inbounds j = tiles_j_ro[idx]
    @inbounds type = tiles_type_ro[idx]

    # `prune_interacting_tiles_kernel!` proved this tile holds no in-cutoff pair
    if type == TILE_DEAD
        return nothing
    end
    @inbounds diag_mask = tiles_diag_ro[idx]

    i_0_tile = (i - a) * w
    index_i = i_0_tile + lane

    r = Int32((N - 1) % TILE_WIDTH + 1)

    force_i = zero(SVector{D, T})

    # One sub-group handles exactly one tile, so the accumulator only needs clearing here
    @inbounds for k in a:b
        opposites_sum[lane, k, warpid] = zero(T)
    end
    # Publish the clearing before other lanes accumulate into these slots
    KI.sub_group_barrier()

    vir = zero(SVector{6, T})

    j_0_tile = (j - a) * w
    index_j = j_0_tile + lane

    # Part 1: Standard non-diagonal tiles
    if j < n_blocks && i < j
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]

        # Stage this tile's j-atom data into local memory once, then index it by slot
        # each iteration
        @inbounds sh_stage[lane, warpid] = JStage(coords[index_j],
                                    atom_shuffle_payload(atoms[index_j], shuf_syms))
        if uses_vel
            @inbounds sh_vel[lane, warpid] = velocities[index_j]
        end
        KI.sub_group_barrier()

        # Separate loops for CLEAN and EXCLUDED tiles, so that the loop over CLEAN tiles
        #   does not handle the masks
        if type == UInt8(0) # CLEAN
            force_i, vir = full_tile_forces(Val(false), force_i, vir, opposites_sum, sh_stage,
                    sh_vel, warpid, lane, diag_mask, typemax(UInt32), zero(UInt32), coords_i,
                    vel_i, atoms_i, inters_tuple, boundary, step_n, Val(r_cut2),
                    Val(force_units), Val(needs_vir), Val(uses_vel), A, shuf_syms, Val(D), T)
        else # EXCLUDED
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))
            force_i, vir = full_tile_forces(Val(true), force_i, vir, opposites_sum, sh_stage,
                    sh_vel, warpid, lane, diag_mask, eligible_bitmask, special_bitmask,
                    coords_i, vel_i, atoms_i, inters_tuple, boundary, step_n, Val(r_cut2),
                    Val(force_units), Val(needs_vir), Val(uses_vel), A, shuf_syms, Val(D), T)
        end

        if index_j <= N
            @inbounds for k in a:b
                if opposites_sum[lane, k, warpid] != zero(T)
                    Atomix.@atomic :monotonic fs_mat[k, index_j] += -opposites_sum[lane, k, warpid]
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
            excl = (eligible_bitmask >> (w - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (w - m)) | (special_bitmask << m)

            condition = (excl & 0x1) == true && r2 <= r_cut2
            any_active = KI.sub_group_any(condition)

            if any_active
                f = condition ? sum_pairwise_forces_gpu(
                    inters_tuple, dr, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : zero_pairwise_force(dr, force_units)

                fi = stripped_force(f, Val(D), T)
                force_i += fi
                for k in 1:D
                    if fi[k] != zero(T)
                        Atomix.@atomic :monotonic fs_mat[k, idx_j] += fi[k]
                    end
                end

                if needs_vir
                    vir = add_pair_virial(vir, f, dr, Val(D))
                end
            end
        end
    end

    # Part 3: Diagonal tiles, and Part 4: the terminal corner tile
    # Lane `lane` evaluates the pairs with the atoms `m > lane` of the tile, up to `r` for
    #   the terminal corner tile. The loop runs the same 31 iterations on all lanes, so
    #   that a barrier separates the updates of `opposites_sum[m]` by different lanes.
    if i == j
        n_tile = (i == n_blocks) ? r : w
        in_tile = lane <= n_tile
        if in_tile
            @inbounds coords_i = coords[index_i]
            @inbounds vel_i = velocities[index_i]
            @inbounds atoms_i = atoms[index_i]
        else
            # Placeholders for the lanes past the last atom, which evaluate no pair
            @inbounds coords_i = coords[i_0_tile + a]
            @inbounds vel_i = velocities[i_0_tile + a]
            @inbounds atoms_i = atoms[i_0_tile + a]
        end

        eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                    lane, n_blocks, r, Val(N))

        @inbounds for t in a:(w - a)
            m = lane + t
            if in_tile && m <= n_tile
                idx_j = j_0_tile + m
                coords_j = coords[idx_j]
                vel_j = velocities[idx_j]
                atoms_j = atoms[idx_j]

                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (w - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (w - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut2

                f = condition ? sum_pairwise_forces_gpu(
                    inters_tuple, dr, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : zero_pairwise_force(dr, force_units)

                fi = stripped_force(f, Val(D), T)
                force_i += fi
                sub_opposite!(opposites_sum, fi, m, warpid)

                if needs_vir
                    vir = add_pair_virial(vir, f, dr, Val(D))
                end
            end
            KI.sub_group_barrier()
        end

        if in_tile
            force_i += SVector{D, T}(ntuple(k -> @inbounds(opposites_sum[lane, k, warpid]),
                                            Val(D)))
        end
    end

    if needs_vir
        vir = @subgroupreduce(+, vir, zero(SVector{6, T}))
        vir_xx, vir_yy, vir_zz, vir_xy, vir_xz, vir_yz = vir

        if lane == 1
            if vir_xx != zero(T)
                Atomix.@atomic :monotonic global_virial[1] += TH(vir_xx)
            end
            if D >= 2
                if vir_yy != zero(T)
                    Atomix.@atomic :monotonic global_virial[D + 2] += TH(vir_yy)
                end
                if vir_xy != zero(T)
                    Atomix.@atomic :monotonic global_virial[2] += TH(vir_xy)
                    Atomix.@atomic :monotonic global_virial[D + 1] += TH(vir_xy)
                end
            end
            if D >= 3
                if vir_zz != zero(T)
                    Atomix.@atomic :monotonic global_virial[9] += TH(vir_zz)
                end
                if vir_xz != zero(T)
                    Atomix.@atomic :monotonic global_virial[3] += TH(vir_xz)
                    Atomix.@atomic :monotonic global_virial[2 * D + 1] += TH(vir_xz)
                end
                if vir_yz != zero(T)
                    Atomix.@atomic :monotonic global_virial[6] += TH(vir_yz)
                    Atomix.@atomic :monotonic global_virial[2 * D + 2] += TH(vir_yz)
                end
            end
        end
    end

    if index_i <= N
        for k in 1:D
            if force_i[k] != zero(T)
                Atomix.@atomic :monotonic fs_mat[k, index_i] += -force_i[k]
            end
        end
    end

    return nothing
end

"""
    energy_kernel!(energy_nounits, coords, velocities, atoms, N, r_cut2, energy_units,
                   inters_tuple, boundary, step_n, tile_exceptions, T, TH, D, block_y,
                   uses_vel, interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
                   interacting_tiles_diag, num_interacting_tiles, interacting_tiles_overflow)

Compute pairwise potential energies for the compact list of interacting 32x32
tiles produced by `find_interacting_blocks_kernel!`.

This mirrors `force_kernel!` structurally: the same compact tile list, the same
four tile-shape cases, and the same CLEAN-vs-mask-backed fast path. The main
difference is the reduction target: each sub-group reduces its energy with
`@subgroupreduce` before the final atomic add to `energy_nounits`.
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
    ::Val{block_y},
    ::Val{uses_vel},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
    interacting_tiles_diag, num_interacting_tiles,
    interacting_tiles_overflow) where {N, r_cut2, A, energy_units, T, TH, D, block_y,
                                       uses_vel}

    a = Int32(1)
    w = Int32(TILE_WIDTH)
    n_blocks = Int32(cld(N, TILE_WIDTH))
    coords = KernelAbstractions.constify(coords_var)
    velocities = KernelAbstractions.constify(velocities_var)
    atoms = KernelAbstractions.constify(atoms_var)
    tiles_i_ro = KernelAbstractions.constify(interacting_tiles_i)
    tiles_j_ro = KernelAbstractions.constify(interacting_tiles_j)
    tiles_type_ro = KernelAbstractions.constify(interacting_tiles_type)
    tiles_diag_ro = KernelAbstractions.constify(interacting_tiles_diag)
    num_interacting_tiles_ro = KernelAbstractions.constify(num_interacting_tiles)
    interacting_tiles_overflow_ro = KernelAbstractions.constify(interacting_tiles_overflow)

    lane = KI.get_sub_group_local_id(Int32)
    warpid = KI.get_sub_group_id(Int32)
    idx = (KI.get_group_id(Int32).x - a) * Int32(block_y) + warpid

    # Local memory for staging this tile's j-atom data, mirroring force_kernel!'s Part 1
    shuf_syms = resolved_atom_shuffle_syms(inters_tuple, A)
    P = atom_payload_type(A, shuf_syms)
    JT = JStage{eltype(coords_var), P}
    sh_stage = KI.localmemory(JT, Val((TILE_WIDTH, block_y)))
    sh_vel = uses_vel ? KI.localmemory(eltype(velocities_var), Val((TILE_WIDTH, block_y))) :
                        nothing

    @inbounds if interacting_tiles_overflow_ro[1] != 0
        return nothing
    end

    @inbounds num_pairs = num_interacting_tiles_ro[1]
    if idx > num_pairs
        return nothing
    end

    @inbounds i = tiles_i_ro[idx]
    @inbounds j = tiles_j_ro[idx]
    @inbounds type = tiles_type_ro[idx]

    # `prune_interacting_tiles_kernel!` proved this tile holds no in-cutoff pair
    if type == TILE_DEAD
        return nothing
    end
    @inbounds diag_mask = tiles_diag_ro[idx]

    i_0_tile = (i - a) * w
    index_i = i_0_tile + lane

    sum_E = zero(T)

    r = Int32((N - 1) % TILE_WIDTH + 1)

    j_0_tile = (j - a) * w
    index_j = j_0_tile + lane

    if j < n_blocks && i < j
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        @inbounds sh_stage[lane, warpid] = JStage(coords[index_j],
                                    atom_shuffle_payload(atoms[index_j], shuf_syms))
        if uses_vel
            @inbounds sh_vel[lane, warpid] = velocities[index_j]
        end
        KI.sub_group_barrier()

        if type == UInt8(0) # CLEAN
            sum_E = full_tile_energy(Val(false), sum_E, sh_stage, sh_vel, warpid, lane,
                    diag_mask, typemax(UInt32), zero(UInt32), coords_i, vel_i, atoms_i,
                    inters_tuple, boundary, step_n, Val(r_cut2), Val(energy_units),
                    Val(uses_vel), A, shuf_syms)
        else # EXCLUDED
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))
            sum_E = full_tile_energy(Val(true), sum_E, sh_stage, sh_vel, warpid, lane,
                    diag_mask, eligible_bitmask, special_bitmask, coords_i, vel_i, atoms_i,
                    inters_tuple, boundary, step_n, Val(r_cut2), Val(energy_units),
                    Val(uses_vel), A, shuf_syms)
        end
    elseif j == n_blocks && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]
        eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                    lane, n_blocks, r, Val(N))

        @inbounds for m in a:r
            idx_j = j_0_tile + m
            coords_j = coords[idx_j]
            vel_j = velocities[idx_j]
            atoms_j = atoms[idx_j]
            dr = vector(coords_i, coords_j, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (w - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (w - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut2

            pe = condition ? sum_pairwise_potentials_gpu(
                inters_tuple, dr, atoms_i, atoms_j, Val(energy_units), (spec & 0x1) == true,
                coords_i, coords_j, boundary, vel_i, vel_j, step_n,
            ) : SVector(zero_pairwise_energy(dr, energy_units))
            sum_E += convert(typeof(sum_E), ustrip(pe[1]))
        end
    elseif i == j
        # Diagonal tiles and the terminal corner tile, see force_kernel!
        n_tile = (i == n_blocks) ? r : w
        if lane <= n_tile
            @inbounds coords_i = coords[index_i]
            @inbounds vel_i = velocities[index_i]
            @inbounds atoms_i = atoms[index_i]
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))

            @inbounds for m in (lane + a):n_tile
                idx_j = j_0_tile + m
                coords_j = coords[idx_j]
                vel_j = velocities[idx_j]
                atoms_j = atoms[idx_j]
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (w - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (w - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut2

                pe = condition ? sum_pairwise_potentials_gpu(
                    inters_tuple, dr, atoms_i, atoms_j, Val(energy_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n,
                ) : SVector(zero_pairwise_energy(dr, energy_units))
                sum_E += convert(typeof(sum_E), ustrip(pe[1]))
            end
        end
    end

    # Sub-group reduction, the result is on the first lane
    sum_E = @subgroupreduce(+, sum_E, zero(T))

    if lane == a && sum_E != zero(T)
        Atomix.@atomic :monotonic energy_nounits[1] += TH(sum_E)
    end

    return nothing
end

## No neighbor list

#=
**The No-neighborlist pairwise force summation kernel (algorithm by Eastman, see https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.21413)**:
1. Case j < n_blocks && i < j, i.e., `32`×`32` tiles: For such tiles each row is assiged to a different lane in a sub-group which calculates the
forces for the entire row in `32` steps. This is done such that some data can be shuffled from `i+1`'th lane to `i`'th lane in each
subsequent iteration of the force calculation in a row. If `a, b, ...` are different atoms and `1, 2, ...` are order in which each lane calculates
the interatomic forces, then we can represent this scenario as (considering a width of 8):
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
so there is no need to order calculations for each lane diagonally and it is also a bit more complicated to do so.
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

This evaluates every atom pair twice and is therefore `O(N^2)`, without a virial.
It is a KernelInterface kernel, launched with `n_blocks_i * n_blocks_j` 1-D work-groups.
=#
function pairwise_force_kernel_nonl!(forces::AbstractArray{T}, coords_var, velocities_var,
                                     atoms_var, boundary, inters, step_n, ::Val{D}, ::Val{F},
                                     ::Val{n_blocks_i}) where {T, D, F, n_blocks_i}
    coords = KernelAbstractions.constify(coords_var)
    velocities = KernelAbstractions.constify(velocities_var)
    atoms = KernelAbstractions.constify(atoms_var)
    n_atoms = Int32(length(atoms))
    w = Int32(TILE_WIDTH)

    group = KI.get_group_id(Int32).x - Int32(1)
    block_i = group % Int32(n_blocks_i)
    block_j = group ÷ Int32(n_blocks_i)
    lane = KI.get_sub_group_local_id(Int32)
    warpidx = KI.get_sub_group_id(Int32)
    i_0_tile = block_i * w
    j_0_block = block_j * KI.get_local_size(Int32).x
    j_0_tile = j_0_block + (warpidx - Int32(1)) * w
    i = i_0_tile + lane

    force_i = zero(SVector{D, T})

    # Uniform across the sub-group
    if i_0_tile + w > n_atoms || j_0_tile + w > n_atoms
        @inbounds if i <= n_atoms
            njs = min(w, n_atoms - j_0_tile)
            atom_i, coord_i, vel_i = atoms[i], coords[i], velocities[i]
            for del_j in Int32(1):njs
                j = j_0_tile + del_j
                if i != j
                    atom_j, coord_j, vel_j = atoms[j], coords[j], velocities[j]
                    f = sum_pairwise_forces_nonl(inters, atom_i, atom_j, Val(F), false, coord_i,
                                                 coord_j, boundary, vel_i, vel_j, step_n)
                    force_i += -stripped_force(f, Val(D), T)
                end
            end

            for dim in 1:D
                Atomix.@atomic :monotonic forces[dim, i] += force_i[dim]
            end
        end
    else
        j = j_0_tile + lane
        tilesteps = w
        if i_0_tile == j_0_tile  # To not compute i-i forces
            j = j_0_tile + lane % w + Int32(1)
            tilesteps -= Int32(1)
        end

        # The lane holding the next j atom
        src_lane = lane % w + Int32(1)
        @inbounds atom_i, coord_i, vel_i = atoms[i], coords[i], velocities[i]
        @inbounds coord_j, vel_j = coords[j], velocities[j]
        @inbounds for _ in 1:tilesteps
            atom_j = atoms[j]
            f = sum_pairwise_forces_nonl(inters, atom_i, atom_j, Val(F), false, coord_i, coord_j,
                                         boundary, vel_i, vel_j, step_n)
            force_i += -stripped_force(f, Val(D), T)
            j = KI.shfl(j, src_lane)
            coord_j = KI.shfl(coord_j, src_lane)
            vel_j = KI.shfl(vel_j, src_lane)
        end

        @inbounds for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += force_i[dim]
        end
    end

    return nothing
end

"""
    nonl_pairwise_forces!(buffers, sys, pairwise_inters, step_n)

Calculate the pairwise forces of the interactions that do not use a neighbor list with
the sub-group kernel `pairwise_force_kernel_nonl!`, without a virial.
"""
function nonl_pairwise_forces!(buffers, sys::System{D, <:Any, T}, pairwise_inters,
                               step_n) where {D, T}
    N = length(sys.atoms)
    backend = get_backend(sys.coords)
    threads_basic = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
    n_threads = min(N, threads_basic, KI.max_work_group_size(backend))
    n_threads = TILE_WIDTH * tiles_per_group(backend, cld(n_threads, TILE_WIDTH))
    check_tiled_backend(backend, T, n_threads)
    n_blocks_i = cld(N, TILE_WIDTH)
    n_blocks_j = cld(N, n_threads)
    kernel_backend, options = tile_kernel_config(backend, T, nothing)
    launch_ki_kernel!(kernel_backend, pairwise_force_kernel_nonl!,
        buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters,
        step_n, Val(D), Val(sys.force_units), Val(n_blocks_i);
        numgroups=n_blocks_i * n_blocks_j, workgroupsize=n_threads, options=options)
    return buffers
end

# The sub-group kernel computes no virial, so it is only used when the virial is not needed
function pairwise_forces_loop_gpu!(buffers, sys::System{D, <:AbstractGPUArray, T},
                                   pairwise_inters, nbs::NoNeighborList, ::Val{false},
                                   step_n) where {D, T}
    backend = get_backend(sys.coords)
    if supports_tiled_kernels(backend) && KI.supports_shuffle(backend, T)
        return nonl_pairwise_forces!(buffers, sys, pairwise_inters, step_n)
    end
    return invoke(pairwise_forces_loop_gpu!,
                  Tuple{Any, System{D, <:AbstractGPUArray}, Any, Any, Val{false}, Any},
                  buffers, sys, pairwise_inters, nbs, Val(false), step_n)
end

## Center of mass motion

@kernel inbounds=true function cm_momentum_kernel!(cm_momentum::AbstractVector{T},
                        @Const(velocities), @Const(atom_masses),
                        ::Val{D}) where {T, D}
    i = @index(Global, Linear)
    p = ustrip.(velocities[i] * atom_masses[i])
    p_group = @groupreduce(+, SVector{D, T}(p), zero(SVector{D, T}))
    if @index(Local, Linear) == 1
        for k in 1:D
            Atomix.@atomic :monotonic cm_momentum[k] += p_group[k]
        end
    end
end

@kernel inbounds=true function remove_cm_velocity_kernel!(velocities, @Const(cm_momentum),
                        @Const(virtual_site_flags), total_mass, ::Val{momentum_unit},
                        ::Val{has_vs}, ::Val{D}) where {momentum_unit, has_vs, D}
    i = @index(Global, Linear)
    cm_velocity = SVector{D}(ntuple(k -> cm_momentum[k] * momentum_unit / total_mass, Val(D)))
    v = velocities[i]
    if has_vs && virtual_site_flags[i]
        velocities[i] = zero(v)
    else
        velocities[i] = v - cm_velocity
    end
end

function remove_CM_motion!(sys::System{D, <:AbstractGPUArray, T}) where {D, T}
    M = unit(zero(eltype(eltype(sys.velocities))) * zero(sys.total_mass))
    backend = get_backend(sys.velocities)
    cm_momentum = KernelAbstractions.zeros(backend, T, D)
    n_threads = 256
    cm_momentum_kernel!(backend, n_threads)(cm_momentum, sys.velocities, masses(sys), Val(D);
                                            ndrange=length(sys))
    remove_cm_velocity_kernel!(backend, n_threads)(
        sys.velocities, cm_momentum, sys.virtual_site_flags, sys.total_mass, Val(M),
        Val(!isempty(sys.virtual_sites)), Val(D); ndrange=length(sys))
    return sys
end
