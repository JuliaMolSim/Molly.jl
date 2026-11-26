# Portable tiled pairwise kernels used with GPUNeighborFinder
#
# These kernels use the sub-group (warp, wavefront or SIMD-group) operations of
#   KernelInterface, so they run on any KernelAbstractions backend that supports atomics and
#   sub-groups of 32 or 64 work-items with shuffles and forms them from consecutive
#   work-items, see `supports_tiled_kernels`. That includes the CUDA, AMDGPU, oneAPI and
#   Metal backends, on devices where they report that, and the POCL-based CPU backend of
#   KernelAbstractions, which runs them without a GPU.
#
# The pipeline generally follows these steps:
# 1.  **Reordering**: Atoms are periodically reordered along a space-filling curve (Morton
#     or Hilbert, see `GPUNeighborFinder`)
#     to improve cache hits during pairwise interactions.
# 2.  **Exceptions**: The sparse per-atom lists of excluded and special pairs are
#     translated into reordered positions, so that the bitmasks of the few 32x32 tiles
#     that contain exceptions can be built on the fly without storing a mask per tile.
# 3.  **Tile Finding**: A kernel identifies pairs of 32x32 atom blocks (tiles) that are
#     within the interaction cutoff, using bounding box checks, and an exact atom-pair
#     test then removes the tiles without a pair within the cutoff.
# 4.  **Execution**: Specialized kernels iterate over the list of interacting tiles,
#     with one segment of 32 work-items of a sub-group per tile.
# 5.  **Tracking**: With `auto_update`, the displacements of the atoms since the tile list
#     was built are checked when the coordinates are reordered, and the list is only
#     rebuilt when a pair within the cutoff could be missing from it.
#
# A tile is processed by a segment of 32 consecutive lanes of a sub-group, and its atoms are
#   indexed by the lane in the segment `tile_lane` and the tile of the work-group by
#   `group_tile`, see "Segments of sub-groups" below. With sub-groups of 32 work-items, as on
#   NVIDIA, Intel and Apple GPUs and AMD RDNA GPUs, a segment is a sub-group. A sub-group of
#   64 work-items, as the wavefronts of other AMD GPUs, holds two segments, which process
#   different tiles.
# The kernels need a 1-D work-group of `32 * k` work-items to consist of `k` segments,
#   which `KI.supports_linear_subgroups` guarantees. Work-groups hold several tiles only on
#   backends whose sub-groups can take different branches,
#   `KI.supports_independent_subgroups`, otherwise they hold one.
#
# No work-item reads and writes local memory that another work-item of its segment writes
#   in the same phase: the j atoms of a tile are staged in local memory once, published with
#   `tile_barrier` and only read afterwards, and the forces on the j atoms are passed between
#   lanes with shuffles. So the kernels do not rely on the work-items of a segment executing
#   in lockstep, which is not guaranteed on GPUs with independent thread scheduling (NVIDIA
#   from Volta on).

# Atoms per block, which is also the width of the segments of the sub-groups that process
#   the tiles, since a lane holds one atom of a block and the tile masks have a bit per lane
const TILE_WIDTH = 32
# Values stored in `interacting_tiles_type`: a tile is either free of exclusions and
#   special pairs (0), mask-backed (1), or known to hold no in-cutoff atom pair at all
#   and removed from the list by `compact_interacting_tiles!` (`TILE_DEAD`)
const TILE_DEAD = UInt8(2)
const MAX_BLOCK_Y = 32
const DEFAULT_BLOCK_Y = 4
const DEFAULT_TILE_THREADS = (32, 8)

## Backend support

#=
    supports_tiled_kernels(backend)

Whether Molly's tiled pairwise kernels, used with [`GPUNeighborFinder`](@ref), can run on
a KernelAbstractions backend: it has to support atomics and sub-groups formed from
consecutive work-items with shuffles, which have to be 32 work-items wide, or 64
work-items wide and able to diverge in segments of 32 work-items, see
`supports_tile_sub_group_width`. This is what the backend reports for the active device.
=#
function supports_tiled_kernels(backend)
    KI.supports_atomics(backend) || return false
    KI.supports_subgroups(backend) || return false
    KI.supports_linear_subgroups(backend) || return false
    supports_tile_sub_group_width(backend) || return false
    return all(T -> KI.supports_shuffle(backend, T), (Int32, UInt32, Float32))
end

#=
Whether the sub-groups of `backend` can process the tiles: either they are `TILE_WIDTH`
wide, or twice as wide and can be partitioned into segments of `TILE_WIDTH` lanes that
take different branches, `KI.supports_partitionable_subgroups`, so that a sub-group
processes two tiles. Wider sub-groups are not supported, the votes of KernelInterface
within segments need sub-groups of at most 64 work-items.
=#
function supports_tile_sub_group_width(backend)
    width = KI.sub_group_size(backend)
    width == TILE_WIDTH && return true
    return width == 2 * TILE_WIDTH && KI.supports_partitionable_subgroups(backend)
end

# Throw an error if the tiled kernels can not run on `backend` with coordinates of float
#   type `T`, called when the buffers of a GPUNeighborFinder system are made
function check_tiled_backend(backend, ::Type{T}) where T
    if !KI.supports_atomics(backend)
        throw(ArgumentError("GPUNeighborFinder needs a backend that supports atomics, " *
                            "$backend does not, use GPUCellListNeighborFinder or " *
                            "DistanceNeighborFinder instead"))
    end
    if !KI.supports_subgroups(backend)
        throw(ArgumentError("GPUNeighborFinder needs a backend that supports sub-groups, " *
                            "$backend does not, use GPUCellListNeighborFinder or " *
                            "DistanceNeighborFinder instead"))
    end
    if !supports_tile_sub_group_width(backend)
        # The tiles hold a bit per lane in UInt32 masks, so a sub-group processes one tile,
        #   or two with a backend that lets the halves of 64-wide sub-groups diverge
        width = KI.sub_group_size(backend)
        throw(ArgumentError("GPUNeighborFinder needs sub-groups of $TILE_WIDTH work-items, " *
                            "or of $(2 * TILE_WIDTH) work-items whose segments of " *
                            "$TILE_WIDTH work-items can diverge " *
                            "(KernelInterface.supports_partitionable_subgroups), $backend " *
                            "has sub-groups of $width work-items" *
                            (width == 2 * TILE_WIDTH ? " that can not" : "") * ", use " *
                            "GPUCellListNeighborFinder or DistanceNeighborFinder instead"))
    end
    if !KI.supports_linear_subgroups(backend)
        throw(ArgumentError("GPUNeighborFinder needs a backend that forms sub-groups from " *
                            "consecutive work-items, $backend does not guarantee that, use " *
                            "GPUCellListNeighborFinder or DistanceNeighborFinder instead"))
    end
    for S in (Int32, UInt32, T)
        if !KI.supports_shuffle(backend, S)
            throw(ArgumentError("GPUNeighborFinder needs a backend that supports shuffles " *
                                "of $S, $backend does not, use GPUCellListNeighborFinder " *
                                "or DistanceNeighborFinder instead"))
        end
    end
    return backend
end

#=
The number of tiles, i.e. sub-groups, per work-group to use for a requested `block_y`.
The sub-groups of a work-group process different tiles, so they take different branches
and return at different points, which needs independent sub-groups. Otherwise, for example
on the CPU backend of KernelAbstractions (POCL), which executes sub-group operations as
work-group barriers, a work-group holds one tile.
=#
@inline function tiles_per_group(backend, block_y::Integer)
    return KI.supports_independent_subgroups(backend) ? Int(block_y) : 1
end

## Segments of sub-groups

#=
The kernels process a tile with a segment of `TILE_WIDTH` consecutive lanes of a sub-group.
They take the sub-group width `W` of the backend as `Val(W)`, see `tile_sub_group_width`,
and use the functions below for the lane of a work-item in its segment, the tile of the
work-group that it processes and the sub-group operations within a segment.
With `W == TILE_WIDTH` a segment is a sub-group and these are the sub-group operations of
KernelInterface, so the kernels for 32-wide sub-groups are the same as without segments.
Otherwise they are the operations with a `width`, which only need the work-items of a
segment to execute them together on backends with `KI.supports_partitionable_subgroups`,
so that the segments of a sub-group can process tiles that take different branches.
=#

#=
The sub-group width that the tiled kernels are compiled for when it is not 0, rather than
the width of the backend. The kernels for 64-wide sub-groups are also valid with 32-wide
sub-groups, see `group_tile`, so this lets them be tested on any backend. Only for testing.
=#
const TILE_SUB_GROUP_WIDTH_OVERRIDE = Ref(0)

#=
`Val(W)` for the width `W` of the sub-groups of the tiled kernels on `backend`, which is
`TILE_WIDTH` or twice that, see `supports_tile_sub_group_width`.
=#
@inline function tile_sub_group_width(backend)
    width = TILE_SUB_GROUP_WIDTH_OVERRIDE[]
    if width == 0
        width = KI.sub_group_size(backend)
    end
    return width == TILE_WIDTH ? Val(TILE_WIDTH) : Val(2 * TILE_WIDTH)
end

# The lane of the work-item in its segment, from 1 to `TILE_WIDTH`, as the shuffles within a
#   segment number them
@inline tile_lane(::Val{TILE_WIDTH}) = KI.get_sub_group_local_id(Int32)
@inline tile_lane(::Val{W}) where {W} =
    ((KI.get_sub_group_local_id(Int32) - Int32(1)) & Int32(TILE_WIDTH - 1)) + Int32(1)

# The segment of the work-item in its work-group, from 1 to the number of segments
# As the sub-groups are formed from consecutive work-items of the 1-D work-group, this is
#   the position of the work-item in the work-group divided by TILE_WIDTH, which also holds
#   for sub-groups of TILE_WIDTH work-items, so that the kernels for wider sub-groups can be
#   tested on GPUs with 32-wide sub-groups
@inline group_tile(::Val{TILE_WIDTH}) = KI.get_sub_group_id(Int32)
@inline group_tile(::Val{W}) where {W} =
    ((KI.get_local_id(Int32).x - Int32(1)) >> 5) + Int32(1)

# The sub-group operations within a segment, see `KI.shfl`, `KI.shfl_xor`,
#   `KI.sub_group_any`, `KI.sub_group_reduce` and `KI.sub_group_barrier`
@inline tile_shfl(val, lane, ::Val{TILE_WIDTH}) = KI.shfl(val, lane)
@inline tile_shfl(val, lane, ::Val{W}) where {W} = KI.shfl(val, lane, TILE_WIDTH)

@inline tile_shfl_xor(val, mask, ::Val{TILE_WIDTH}) = KI.shfl_xor(val, mask)
@inline tile_shfl_xor(val, mask, ::Val{W}) where {W} = KI.shfl_xor(val, mask, TILE_WIDTH)

@inline tile_any(pred, ::Val{TILE_WIDTH}) = KI.sub_group_any(pred)
@inline tile_any(pred, ::Val{W}) where {W} = KI.sub_group_any(pred, TILE_WIDTH)

@inline tile_reduce(op, val, ::Val{TILE_WIDTH}) = KI.sub_group_reduce(op, val)
@inline tile_reduce(op, val, ::Val{W}) where {W} = KI.sub_group_reduce(op, val, TILE_WIDTH)

@inline tile_barrier(::Val{TILE_WIDTH}) = KI.sub_group_barrier()
@inline tile_barrier(::Val{W}) where {W} = KI.sub_group_barrier(TILE_WIDTH)

# The sum over the lanes of a full segment, in every lane
@inline function tile_sum(x, ::Val{W}) where W
    offset = Int32(TILE_WIDTH ÷ 2)
    while offset > Int32(0)
        x += tile_shfl_xor(x, offset, Val(W))
        offset >>= Int32(1)
    end
    return x
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

#=
The compiled KernelInterface kernel `f` for the host-side arguments `args` on `backend`,
with the compiler options `options`. The kernels without options, which are the ones
launched every step, are kept, see `cached_kernel_function`.
=#
function tile_kernel(backend, f::F, args::Tuple, options::NamedTuple) where F
    return KI.kernel_function(backend, f, KI.argument_types(backend, args); options...)
end

function tile_kernel(backend, f::F, args::Tuple, options::NamedTuple{()}) where F
    return cached_kernel_function(backend, f, args, Val(options))
end

#=
Finding the compiled kernel for a launch converts the arguments, builds a tuple type at run
time and looks it up in the compilation cache under a lock. For the kernels launched every
step that is often longer than the kernel takes to run, and the simulation step is limited
by the host launching work. Those kernels are kept in a slot for each function, argument
types on the host and compilation options instead, from `kernel_slot`, which is found
when the calling code is compiled. The kernel is looked up again for another backend or
context, see `kernel_cache_context`, or once methods have been defined, as the backend
would then check whether it has to be compiled again.
=#
@generated kernel_slot(f, options, args...) = :($(Ref{Any}(nothing)))

struct CachedKernel{B, C, K}
    context::C
    world::UInt
    backend::B
    kernel::K
end

#=
What the compiled kernels on `backend` belong to, which is the active device by default.
Backend extensions can return something that also changes when the device is reset, such
as the CUDA context.
=#
kernel_cache_context(backend) = KI.device(backend)

# The compiled kernel of `f(args...)` for `backend` with the compilation options `options`,
#   which are given as the type of `Val(options)` to be part of the key of the slot
# The context and the backend of the entry, whose types are not known here, are compared
#   with `===`, which unlike `==` needs no dynamic dispatch at every launch
function cached_kernel_function(backend, f::F, args::Tuple,
                                ::Val{options}) where {F, options}
    slot = kernel_slot(f, Val(options), args...)
    entry = slot[]
    context = kernel_cache_context(backend)
    world = Base.get_world_counter()
    if !(entry isa CachedKernel) || entry.context !== context || entry.world != world ||
            entry.backend !== backend
        tt = KI.argument_types(backend, args)
        entry = CachedKernel(context, world, backend,
                             KI.kernel_function(backend, f, tt; options...))
        slot[] = entry
    end
    return entry.kernel
end

# Compile a KernelInterface kernel and launch it with `numgroups` work-groups of
#   `workgroupsize` work-items
function launch_tile_kernel!(backend, f::F, args...; numgroups, workgroupsize,
                             options=(;)) where F
    kernel = tile_kernel(backend, f, args, options)
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

# Per-j-atom data staged in local memory by the inner loop of the full tiles.
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

# Local memory (bytes) of force_kernel!: the staged j coords/atoms/velocities, the forces
#   on the j atoms are passed with shuffles
function force_kernel_localmem(buffers, ::Val{D}, ::Type{T}, uses_vel::Bool,
                               block_y::Integer, pairwise_inters) where {D, T}
    return energy_kernel_localmem(buffers, uses_vel, block_y, pairwise_inters)
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
no dynamically sized local memory. `max_tile_localmem(backend)` is the most that they use,
32 KiB by default, which Apple GPUs have and other GPUs exceed. Backend extensions can
allow more, e.g. the 48 KiB of static shared memory of CUDA. The default `block_y` stays
far below either, but a larger `block_y` is only kept within the limit.
=#
max_tile_localmem(backend) = 32 * 1024

# The largest block_y not above `block_y` for which the local memory fits in
#   `max_localmem` bytes and work-groups of `TILE_WIDTH * block_y` work-items are allowed,
#   `max_items` being the largest work-group of the kernel
function clamp_block_y(block_y::Integer, localmem_bytes, max_localmem::Integer,
                       max_items::Integer=typemax(Int))
    while block_y > 1 && (localmem_bytes(block_y) > max_localmem ||
                          TILE_WIDTH * block_y > max_items)
        block_y = prevpow(2, block_y - 1)
    end
    return block_y
end

function force_block_y(sys, buffers, pairwise_inters)
    config = tile_launch_config(sys)
    block_y_override = prefer_override(config_force_block_y(config),
                                       env_override("MOLLY_TILE_FORCE_BLOCK_Y"))
    block_y_override === nothing || validate_block_y("MOLLY_TILE_FORCE_BLOCK_Y", block_y_override)
    block_y = something(block_y_override, DEFAULT_BLOCK_Y)
    D, T = length(eltype(sys.coords)), float_type(sys)
    uses_vel = any_uses_velocity(pairwise_inters)
    return clamp_block_y(block_y, by -> force_kernel_localmem(buffers, Val(D), T, uses_vel,
                                                              by, pairwise_inters),
                         max_tile_localmem(get_backend(sys.coords)))
end

function force_maxregs(sys)
    config = tile_launch_config(sys)
    maxregs = prefer_override(config_force_maxregs(config),
                              env_override("MOLLY_TILE_FORCE_MAXREGS"))
    maxregs === nothing || maxregs > 0 ||
        error("MOLLY_TILE_FORCE_MAXREGS must be positive, got $(maxregs)")
    return maxregs
end

function energy_block_y(sys, buffers, pairwise_inters)
    config = tile_launch_config(sys)
    block_y_override = prefer_override(config_energy_block_y(config),
                                       env_override("MOLLY_TILE_ENERGY_BLOCK_Y"))
    block_y_override === nothing || validate_block_y("MOLLY_TILE_ENERGY_BLOCK_Y", block_y_override)
    block_y = something(block_y_override, DEFAULT_BLOCK_Y)
    uses_vel = any_uses_velocity(pairwise_inters)
    return clamp_block_y(block_y, by -> energy_kernel_localmem(buffers, uses_vel, by,
                                                               pairwise_inters),
                         max_tile_localmem(get_backend(sys.coords)))
end

function tile_threads(sys)
    config = tile_launch_config(sys)
    config_threads = config_tile_threads(config)
    threads_x = config_threads === nothing ? env_override("MOLLY_TILE_THREADS_X") : config_threads[1]
    threads_y = config_threads === nothing ? env_override("MOLLY_TILE_THREADS_Y") : config_threads[2]
    if xor(threads_x === nothing, threads_y === nothing)
        error("set both MOLLY_TILE_THREADS_X and MOLLY_TILE_THREADS_Y together")
    end
    threads_x === nothing && return DEFAULT_TILE_THREADS
    threads_x > 0 || error("MOLLY_TILE_THREADS_X must be positive, got $(threads_x)")
    threads_y > 0 || error("MOLLY_TILE_THREADS_Y must be positive, got $(threads_y)")
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

## Scratch memory

#=
Scratch memory of the tiled path kept with a set of buffers, see `tile_scratch`:
`reorder_sort` is the workspace of the sort of the curve codes, which is reused at every
reorder since the number of atoms does not change.
=#
mutable struct TileScratch
    reorder_sort::Any
end

function tile_scratch(buffers)
    scratch = buffers.tile_scratch[]
    if !(scratch isa TileScratch)
        scratch = TileScratch(nothing)
        buffers.tile_scratch[] = scratch
    end
    return scratch::TileScratch
end

# The algorithm of the sort of the curve codes, see `sort_reorder_seq!`
# On the CPU AcceleratedKernels chooses a threaded sort itself, on GPUs its merge sort is used
#   with blocks of 512 elements
reorder_sort_alg(backend) = AcceleratedKernels.MergeSort(block_size=512)
reorder_sort_alg(::KernelAbstractions.CPU) = AcceleratedKernels.Auto()

#=
The workspace of the sort of the curve codes in `buffers`, made the first time so that the
sort does not allocate scratch memory at every reorder.
=#
function reorder_sort_workspace(buffers, backend)
    scratch = tile_scratch(buffers)
    ws = scratch.reorder_sort
    if isnothing(ws)
        ws = AcceleratedKernels.workspace(AcceleratedKernels.sortperm!, buffers.reorder_seq,
                                          buffers.reorder_codes;
                                          alg=reorder_sort_alg(backend))
        scratch.reorder_sort = ws
    end
    return ws
end

## Displacement tracking of the tile list

#=
The tile list holds the pairs within the cutoff of the neighbor finder at the coordinates it
was built from. With `auto_update`, the coordinates are checked against those coordinates
whenever they are reordered, which a force or energy calculation does first. The check is
made on the device by the kernel that reorders the coordinates, and sets the overflow flag
of the tile list when the atoms have moved too far, which makes the force and energy
kernels return straight away. So the kernels can be launched before the host knows the
result, and the host only waits once they are queued.

Each work-group reduces the two largest squared displacements of its atoms. With
`fused_tile_check(backend)`, the last work-group to finish then reduces the results of all
the work-groups, which needs atomic operations with acquire-release ordering across
work-groups. Otherwise a second kernel with one work-group does that.

The result is written to `result` and read on the host from `result_host`: the two largest
squared displacements and whether they exceed the allowance, then the same for the inner
masks of the tile list, see `InnerTileMasks`. Backend extensions can give memory that the
device writes to and the host reads directly, and an event to wait on, see
`check_result_arrays`.
=#

# Work-items per work-group of the displacement checks, and the largest number of
#   work-groups of a check without reordering, which loops over the atoms
const tile_check_group_size = 256
const tile_check_max_n_groups = 256

# Whether the backend supports the reduction of a check by its last work-group, see above
fused_tile_check(backend) = false
fused_tile_check(::KernelAbstractions.POCLBackend) = true

#=
The arrays that the check kernels write the result of length `n` to and that the host reads
it from after `wait_check_result!`, and an event to record after the check is launched or
`nothing`. Backend extensions can define methods that return memory that the device can
write to directly, and an event, see `record_check!` and `wait_check_result!`.
=#
check_result_arrays(backend, ::Type{T}, n::Integer) where {T} =
    (KernelAbstractions.zeros(backend, T, n), zeros(T, n), nothing)

# Called after the check kernels are launched
record_check!(backend, event, scratch) = scratch

# Wait for the check and make its result available in `scratch.result_host`, the default
#   copies it, which waits for the device to finish all the work queued so far
function wait_check_result!(backend, event, scratch)
    copyto!(scratch.result_host, scratch.result)
    return scratch.result_host
end

# Scratch memory of a check, see above, kept in a `TileReference` or the tracker of a
#   neighbor finder
mutable struct TileCheckScratch{T, B, R, H, A, I, E}
    backend::B
    device::Int
    result::R
    result_host::H
    block_maxs_1::A
    block_maxs_2::A
    inner_block_maxs_1::A
    inner_block_maxs_2::A
    counter::I
    event::E
end

function TileCheckScratch(backend, ::Type{T}, n_groups::Integer) where T
    result, result_host, event = check_result_arrays(backend, T, 6)
    n_alloc = max(Int(n_groups), tile_check_max_n_groups)
    block_maxs = ntuple(_ -> KernelAbstractions.zeros(backend, T, n_alloc), 4)
    counter = KernelAbstractions.zeros(backend, Int32, 1)
    return TileCheckScratch{T, typeof(backend), typeof(result), typeof(result_host),
                            typeof(block_maxs[1]), typeof(counter), typeof(event)}(
                backend, KI.device(backend), result, result_host, block_maxs..., counter,
                event)
end

#=
A copy of the fields would have a host array that is a copy of the results rather than the
memory the device writes to, so every check with it would read the result of the last check
before the copy, and it would share the event. A copy of a system that has made checks,
which holds the scratch in the tracker of its neighbor finder, would then never rebuild its
list, or rebuild it every time. The copy is a new scratch instead, with a result that says
the atoms have moved in case a check was pending.
=#
function Base.deepcopy_internal(scratch::TileCheckScratch{T}, dict::IdDict) where T
    haskey(dict, scratch) && return dict[scratch]::TileCheckScratch{T}
    scratch_copy = TileCheckScratch(scratch.backend, T, length(scratch.block_maxs_1))
    scratch_copy.result_host .= (zero(T), zero(T), one(T), zero(T), zero(T), one(T))
    dict[scratch] = scratch_copy
    return scratch_copy
end

# The scratch held in `holder.scratch` if it can be used for a check with `n_groups`
#   work-groups on `backend`, otherwise a new one
function tile_check_scratch(holder, backend, ::Type{T}, n_groups::Integer) where T
    scratch = (isnothing(holder) ? nothing : holder.scratch)
    if scratch isa TileCheckScratch{T, typeof(backend)} && scratch.backend == backend &&
            scratch.device == KI.device(backend) && length(scratch.block_maxs_1) >= n_groups
        return scratch
    end
    scratch = TileCheckScratch(backend, T, n_groups)
    if !isnothing(holder)
        holder.scratch = scratch
    end
    return scratch
end

# The two largest values of a pair over the work-group, in every work-item, from the pairs of
#   the work-items, `sh` is local memory for a pair per work-item, which is free afterwards
# Called by every work-item of the work-group
@inline function group_top2(val::Tuple{T, T}, sh) where T
    lane = KI.get_sub_group_local_id(Int32)
    sg = KI.get_sub_group_id(Int32)
    n_sg = KI.get_num_sub_groups(Int32)
    val = KI.sub_group_reduce(merge_top2, val)
    if lane == Int32(1)
        @inbounds sh[sg] = val
    end
    KI.barrier()
    # Every sub-group reduces the results of the sub-groups, so that the sub-group
    #   operations are uniform over the work-group
    other = (lane <= n_sg ? (@inbounds sh[lane]) : (zero(T), zero(T)))
    val = KI.sub_group_reduce(merge_top2, other)
    KI.barrier()
    return val
end

# Reduce the results of the work-groups of a check, write the result and set the flags,
#   called by every work-item of one work-group
# The results of the other work-groups are read with atomic loads, which read the
#   device-wide cache
@inline function finish_tile_check!(result, block_maxs_1, block_maxs_2, n_groups, skip_flag,
                                    allowance::T, inner, sh) where T
    l = KI.get_local_id(Int32).x
    n_items = KI.get_local_size(Int32).x
    top = (zero(T), zero(T))
    inner_top = (zero(T), zero(T))
    b = l
    while b <= n_groups
        top = merge_top2(top, (Atomix.@atomic(:monotonic, block_maxs_1[b]),
                               Atomix.@atomic(:monotonic, block_maxs_2[b])))
        if !isnothing(inner)
            inner_top = merge_top2(inner_top,
                                   (Atomix.@atomic(:monotonic, inner.block_maxs_1[b]),
                                    Atomix.@atomic(:monotonic, inner.block_maxs_2[b])))
        end
        b += n_items
    end
    max_1, max_2 = group_top2(top, sh)
    if !isnothing(inner)
        inner_1, inner_2 = group_top2(inner_top, sh)
    end
    @inbounds if l == Int32(1)
        invalid = !(sqrt(max_1) + sqrt(max_2) <= allowance)
        result[1] = max_1
        result[2] = max_2
        result[3] = (invalid ? one(T) : zero(T))
        if invalid && !isnothing(skip_flag)
            skip_flag[1] = Int32(1)
        end
        if !isnothing(inner)
            stale = inner.force || !(sqrt(inner_1) + sqrt(inner_2) <= inner.allowance)
            result[4] = inner_1
            result[5] = inner_2
            result[6] = (stale ? one(T) : zero(T))
            inner.stale[1] = (stale ? Int32(1) : Int32(0))
        end
    end
    return nothing
end

#=
Store the results of the work-group of a check, the two largest squared displacements of
its work-items `max_1` and `max_2`, and `inner_d2` for the inner masks. With `fused`, the
last work-group to finish then finishes the check.
Called by every work-item of the work-group.
=#
@inline function store_group_check!(max_1::T, max_2::T, inner_d2::T, result, block_maxs_1,
                                    block_maxs_2, counter, skip_flag, allowance::T, inner,
                                    sh, is_last, ::Val{fused}) where {T, fused}
    l = KI.get_local_id(Int32).x
    g = KI.get_group_id(Int32).x
    n_groups = KI.get_num_groups(Int32).x
    max_1, max_2 = group_top2((max_1, max_2), sh)
    if !isnothing(inner)
        inner_1, inner_2 = group_top2((inner_d2, zero(T)), sh)
    end
    @inbounds if l == Int32(1)
        block_maxs_1[g] = max_1
        block_maxs_2[g] = max_2
        if !isnothing(inner)
            inner.block_maxs_1[g] = inner_1
            inner.block_maxs_2[g] = inner_2
        end
        if fused
            # The release makes the results above visible to the work-group that sees the
            #   count, and the acquire those of the work-groups counted before
            n_finished = Atomix.@atomic :acquire_release counter[1] += Int32(1)
            is_last[1] = (n_finished == n_groups)
        end
    end
    if fused
        KI.barrier()
        if @inbounds is_last[1]
            finish_tile_check!(result, block_maxs_1, block_maxs_2, n_groups, skip_flag,
                               allowance, inner, sh)
            if l == Int32(1)
                # Ready for the next check
                @inbounds counter[1] = Int32(0)
            end
        end
    end
    return nothing
end

#=
Reorder the coordinates and velocities into the current atom order, as
`reorder_system_kernel!` does, and check the displacements of the atoms since the reference
coordinates in the same pass, setting `skip_flag` if they are too large. `inner` is `nothing`
or holds the check of the inner masks of the tile list, see `InnerTileMasks`, whose reference
coordinates are in the atom order.
=#
function reorder_check_kernel!(coords_reordered, velocities_reordered, coords_var,
                               velocities_var, seq_var, ref_coords_var, result, block_maxs_1,
                               block_maxs_2, counter, skip_flag, boundary, F, dist_unit,
                               allowance::T, inner, ::Val{N}, ::Val{T},
                               ::Val{fused}) where {N, T, fused}
    coords = KernelAbstractions.constify(coords_var)
    velocities = KernelAbstractions.constify(velocities_var)
    seq = KernelAbstractions.constify(seq_var)
    ref_coords = KernelAbstractions.constify(ref_coords_var)
    sh = KI.localmemory(Tuple{T, T}, Val(tile_check_group_size))
    is_last = KI.localmemory(Bool, Val(1))
    i = KI.get_global_id(Int32).x
    max_1 = zero(T)
    inner_d2 = zero(T)
    @inbounds if i <= N
        original_i = seq[i]
        c = coords[original_i]
        coords_reordered[i] = c
        velocities_reordered[i] = velocities[original_i]
        max_1 = sq_displacement(c, ref_coords[original_i], boundary, F, dist_unit, T)
        if !isnothing(inner)
            inner_d2 = sq_displacement(c, inner.coords[i], boundary, inner.F, dist_unit, T)
        end
    end
    store_group_check!(max_1, zero(T), inner_d2, result, block_maxs_1, block_maxs_2, counter,
                       skip_flag, allowance, inner, sh, is_last, Val(fused))
    return nothing
end

# The check of the displacements without reordering, looping over the atoms
function displacement_check_kernel!(result, block_maxs_1, block_maxs_2, counter, skip_flag,
                                    coords_var, ref_coords_var, boundary, F, dist_unit,
                                    n_atoms::Int32, allowance::T, ::Val{T},
                                    ::Val{fused}) where {T, fused}
    coords = KernelAbstractions.constify(coords_var)
    ref_coords = KernelAbstractions.constify(ref_coords_var)
    sh = KI.localmemory(Tuple{T, T}, Val(tile_check_group_size))
    is_last = KI.localmemory(Bool, Val(1))
    stride = KI.get_local_size(Int32).x * KI.get_num_groups(Int32).x
    i = KI.get_global_id(Int32).x
    max_1, max_2 = zero(T), zero(T)
    @inbounds while i <= n_atoms
        d2 = sq_displacement(coords[i], ref_coords[i], boundary, F, dist_unit, T)
        max_1, max_2 = push_top2(max_1, max_2, d2)
        i += stride
    end
    store_group_check!(max_1, max_2, zero(T), result, block_maxs_1, block_maxs_2, counter,
                       skip_flag, allowance, nothing, sh, is_last, Val(fused))
    return nothing
end

# Finish a check made without `fused_tile_check`, launched with one work-group
function finish_check_kernel!(result, block_maxs_1, block_maxs_2, n_groups::Int32, skip_flag,
                              allowance::T, inner, ::Val{T}) where T
    sh = KI.localmemory(Tuple{T, T}, Val(tile_check_group_size))
    finish_tile_check!(result, block_maxs_1, block_maxs_2, n_groups, skip_flag, allowance,
                       inner, sh)
    return nothing
end

# The inputs of the inner check with the work-group results of the scratch
function inner_check_kernel_args(inner, scratch)
    isnothing(inner) && return nothing
    return (; inner..., block_maxs_1=scratch.inner_block_maxs_1,
            block_maxs_2=scratch.inner_block_maxs_2)
end

# Launch the second kernel of a check made without `fused_tile_check`
function launch_finish_check!(backend, scratch, n_groups, skip_flag, allowance::T,
                              inner_args) where T
    launch_tile_kernel!(backend, finish_check_kernel!, scratch.result, scratch.block_maxs_1,
                        scratch.block_maxs_2, Int32(n_groups), skip_flag, allowance,
                        inner_args, Val(T); numgroups=1, workgroupsize=tile_check_group_size)
    return scratch
end

#=
Reorder the coordinates and velocities of `sys` into the atom order of `buffers` and check
the displacements since `ref_coords`, without waiting for the check, returning the scratch
to wait on. See above.
=#
function launch_reorder_check!(buffers, sys, scratch::TileCheckScratch{T}, ref_coords, F,
                               dist_unit, allowance::T, inner) where T
    N = length(sys.coords)
    backend = get_backend(sys.coords)
    n_groups = max(cld(N, tile_check_group_size), 1)
    fused = fused_tile_check(backend)
    inner_args = inner_check_kernel_args(inner, scratch)
    launch_tile_kernel!(backend, reorder_check_kernel!, buffers.coords_reordered,
                        buffers.velocities_reordered, sys.coords, sys.velocities,
                        buffers.reorder_seq, ref_coords, scratch.result, scratch.block_maxs_1,
                        scratch.block_maxs_2, scratch.counter,
                        buffers.interacting_tiles_overflow, sys.boundary, F, dist_unit,
                        allowance, inner_args, Val(N), Val(T), Val(fused);
                        numgroups=n_groups, workgroupsize=tile_check_group_size)
    if !fused
        launch_finish_check!(backend, scratch, n_groups, buffers.interacting_tiles_overflow,
                             allowance, inner_args)
    end
    record_check!(backend, scratch.event, scratch)
    return scratch
end

# Launch the check of `coords` against `ref_coords` without waiting for it, returning the
#   scratch to wait on, `allowance` is the total displacement of the two atoms that have
#   moved furthest that the list allows
function launch_displacement_check!(scratch::TileCheckScratch{T}, coords, ref_coords,
                                    boundary, F, dist_unit,
                                    allowance::T=T(Inf)) where T
    n_atoms = length(coords)
    backend = get_backend(coords)
    n_groups = clamp(cld(n_atoms, tile_check_group_size), 1, tile_check_max_n_groups)
    fused = fused_tile_check(backend)
    launch_tile_kernel!(backend, displacement_check_kernel!, scratch.result,
                        scratch.block_maxs_1, scratch.block_maxs_2, scratch.counter, nothing,
                        coords, ref_coords, boundary, F, dist_unit, Int32(n_atoms), allowance,
                        Val(T), Val(fused);
                        numgroups=n_groups, workgroupsize=tile_check_group_size)
    if !fused
        launch_finish_check!(backend, scratch, n_groups, nothing, allowance, nothing)
    end
    record_check!(backend, scratch.event, scratch)
    return scratch
end

# Wait for a check, returning the two largest squared displacements and whether they exceed
#   the allowance
function wait_displacement_check(scratch::TileCheckScratch)
    result = wait_check_result!(scratch.backend, scratch.event, scratch)
    @inbounds return result[1], result[2], !iszero(result[3])
end

#=
`max_sq_displacements` on a backend that supports `fused_tile_check`, as one kernel that
reduces the result on the device. Backends without it use the version in neighbors.jl,
which reduces the results of the work-groups on the host.
=#
function fused_max_sq_displacements(coords, ref_coords, boundary, F, dist_unit, ::Type{T},
                                    tracker) where T
    iszero(length(coords)) && return (zero(T), zero(T))
    n_groups = clamp(cld(length(coords), tile_check_group_size), 1, tile_check_max_n_groups)
    scratch = tile_check_scratch(tracker, get_backend(coords), T, n_groups)
    return fused_max_sq_displacements(scratch, coords, ref_coords, boundary, F, dist_unit)
end

function fused_max_sq_displacements(scratch::TileCheckScratch, coords, ref_coords, boundary,
                                    F, dist_unit)
    launch_displacement_check!(scratch, coords, ref_coords, boundary, F, dist_unit)
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
The energy kernel uses the inner masks when they are current, and the masks of the list
otherwise.

`masks` are the inner masks of the tiles, found at `coords`, in the atom order of the
tile list, and `boundary` with the inner cutoff `cutoff`, which is without units. The
check of the list sets `stale` on the device when the force kernel has to find the masks
again. `valid` is whether the masks are for the current tile list, and `refresh_queued`
is whether `stale` was set by the host since the last check, so that the next force
kernel finds the masks.
=#
mutable struct InnerTileMasks{C, B, T, M, I}
    masks::M
    coords::C
    boundary::B
    cutoff::T
    stale::I
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
`run_tiled_kernel!`, and `scratch` the scratch memory of the checks. `inner` holds the masks
of the dynamic pruning, see `InnerTileMasks`, or is `nothing` when it is not used.
=#
mutable struct TileReference{C, B, T, M, I}
    coords::C
    boundary::B
    moved_last::T
    moved_prev::T
    pending_check::Any
    scratch::Any
    inner::Union{Nothing, InnerTileMasks{C, B, T, M, I}}
end

# The part of the gap between the cutoff of the interactions and that of the tile list
#   that is between the cutoff of the interactions and the inner cutoff. Smaller fractions
#   prune more pairs but the masks are found again more often.
const inner_cutoff_fraction = 0.2

# The squared cutoff of the pairwise interactions that use the tile list
tile_inters_sqdist(sys, pairwise_inters) = neighbor_inters_sqdist(pairwise_inters, (),
                                                coords_length_unit(sys.coords))

# The inner cutoff of the dynamic pruning without units, or `nothing` if it is not used
function inner_tile_cutoff(sys, pairwise_inters, ::Type{T}) where T
    inter_sqdist = tile_inters_sqdist(sys, pairwise_inters)
    isnothing(inter_sqdist) && return nothing
    dist_unit = coords_length_unit(sys.coords)
    r_list = sys.neighbor_finder.dist_cutoff
    dimension(inter_sqdist) == dimension(dist_unit^2) || return nothing
    dimension(r_list) == dimension(dist_unit) || return nothing
    r_c = T(ustrip(dist_unit, sqrt(inter_sqdist)))
    r_list = T(ustrip(dist_unit, r_list))
    r_inner = r_c + T(inner_cutoff_fraction) * (r_list - r_c)
    return (r_c < r_inner < r_list ? r_inner : nothing)
end

#=
The reference held in `buffers` if it is for coordinates and a boundary of the types of
those of `sys`, otherwise `nothing`. The buffers hold the reference without a type, so
this makes the check type stable. A reference of another type, which would come from
using the buffers with another system, is replaced when the tile list is rebuilt.
=#
@inline function tile_reference(buffers, sys)
    T = coords_float_type(sys.coords)
    ref = buffers.neighbor_reference[]
    R = TileReference{typeof(sys.coords), typeof(sys.boundary), T,
                      typeof(buffers.interacting_tiles_diag),
                      typeof(buffers.num_interacting_tiles)}
    return (ref isa R ? ref : nothing)
end

#=
Set up the inner masks of `ref` for a tile list that has just been built, so that the next
force kernel finds them, reusing the memory of the previous ones if possible.
=#
function reset_inner_masks!(ref::TileReference{C, B, T, M, I}, buffers, sys,
                            pairwise_inters) where {C, B, T, M, I}
    cutoff = inner_tile_cutoff(sys, pairwise_inters, T)
    if isnothing(cutoff)
        ref.inner = nothing
        return ref
    end
    capacity = length(buffers.interacting_tiles_diag)
    inner = ref.inner
    if isnothing(inner) || length(inner.coords) != length(buffers.coords_reordered)
        inner = InnerTileMasks{C, B, T, M, I}(similar(buffers.interacting_tiles_diag, capacity),
                    similar(buffers.coords_reordered), sys.boundary, cutoff,
                    similar(buffers.num_interacting_tiles, 1), false, false)
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
The inner masks for the force and energy kernels, see `InnerTileMasks`, or `nothing` if they
are not used. `r_cut2` is the squared cutoff of the force kernel, whose type the squared
inner cutoff is given in.
=#
function inner_kernel_args(buffers, sys, r_cut2)
    sys.neighbor_finder.auto_update || return nothing
    ref = tile_reference(buffers, sys)
    (isnothing(ref) || isnothing(ref.inner)) && return nothing
    inner = ref.inner
    dist_unit = coords_length_unit(sys.coords)
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

# Whether the tile list has to be rebuilt for reasons other than the atoms moving
@inline function tile_refresh_due(buffers, nf::GPUNeighborFinder)
    return buffers.step_n_preprocessed == -1 ||
           buffers.reorder_generation != nf.reorder_generation ||
           buffers.tile_generation != nf.tile_generation ||
           buffers.sparse_pair_generation != nf.cache_generation || !nf.initialized
end

#=
Reorder the coordinates and velocities into the current atom order and check whether the
atoms have moved far enough since the tile list in `buffers` was built that a pair within
the cutoff of `pairwise_inters` could be missing. The check sets the overflow flag of the
tile list if so, which makes the force and energy kernels return without doing anything,
so that they can be launched straight after. Returns the scratch to wait on, or `nothing`
if the tile list has to be rebuilt whatever the displacements are, as when the cutoff is
not known or when the check is expected to fail, in which case nothing is launched.
=#
function start_tile_check!(ref::TileReference, buffers, sys, nf, pairwise_inters,
                           uses_inner::Bool)
    T = coords_float_type(sys.coords)
    dist_unit = coords_length_unit(sys.coords)
    F = box_deformation(ref.boundary, sys.boundary, T)
    allowance = neighbor_allowance(sys.boundary, F, nf.dist_cutoff,
                                   tile_inters_sqdist(sys, pairwise_inters), dist_unit, T)
    if isnothing(allowance) || length(ref.coords) != length(sys.coords)
        return nothing
    end
    # Rebuilt without checking, see above
    tile_check_expected_to_fail(ref, allowance) && return nothing
    nf.tracker.n_checks += 1
    N = length(sys.coords)
    scratch = tile_check_scratch(ref, get_backend(sys.coords), T,
                                 max(cld(N, tile_check_group_size), 1))
    inner = ref.inner
    if uses_inner && !isnothing(inner)
        inner_args = inner_check_args(inner, sys, pairwise_inters, dist_unit, T)
        # The check sets the flag of the inner masks from now on
        inner.refresh_queued = false
        return launch_reorder_check!(buffers, sys, scratch, ref.coords, F, dist_unit,
                                     allowance, inner_args)
    else
        return launch_reorder_check!(buffers, sys, scratch, ref.coords, F, dist_unit,
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
    F = box_deformation(inner.boundary, sys.boundary, T)
    allowance = neighbor_allowance(sys.boundary, F, inner.cutoff * dist_unit,
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
        T = coords_float_type(sys.coords)
        ref = TileReference{typeof(sys.coords), typeof(sys.boundary), T,
                            typeof(buffers.interacting_tiles_diag),
                            typeof(buffers.num_interacting_tiles)}(
                    copy(sys.coords), sys.boundary, zero(T), zero(T), nothing, nothing,
                    nothing)
        buffers.neighbor_reference[] = ref
    end
    reset_inner_masks!(ref, buffers, sys, pairwise_inters)
    return buffers
end

function record_gpu_tiles!(buffers, sys, nf::GPUNeighborFinder, pairwise_inters)
    nf.tracker.n_builds += 1
    nf.auto_update || return buffers
    if neighbors_trackable(sys.boundary, sys.coords, nf.dist_cutoff,
                           tile_inters_sqdist(sys, pairwise_inters))
        record_tile_reference!(tile_reference(buffers, sys), buffers, sys, pairwise_inters)
    else
        # The tile list has to be rebuilt every time, so no reference is needed
        buffers.neighbor_reference[] = nothing
    end
    return buffers
end

# find_neighbors increases the rebuild counts of the neighbor finder when the atom order
#   or the tile list is due to be rebuilt, so the cache follows the same rebuild points as
#   the lists of the other neighbor finders
# `moved` is whether the atoms have moved far enough to invalidate the tile list, in which
#   case the tile list and atom order are rebuilt
@inline function gpu_neighbor_refresh_flags(buffers, nf::GPUNeighborFinder, moved::Bool)
    first_preprocess = (buffers.step_n_preprocessed == -1)
    needs_order_refresh = (first_preprocess || moved ||
                            buffers.reorder_generation != nf.reorder_generation)
    sparse_changed = (buffers.sparse_pair_generation != nf.cache_generation)
    needs_reorder = true # Coordinates can be changed at any point
    needs_sparse_refresh = (needs_order_refresh || !nf.initialized || sparse_changed)
    needs_tile_refresh = (needs_sparse_refresh ||
                          buffers.tile_generation != nf.tile_generation)
    return needs_order_refresh, needs_reorder, needs_sparse_refresh, needs_tile_refresh
end

function record_neighbor_refresh!(buffers, nf::GPUNeighborFinder, step_n)
    buffers.reorder_generation = nf.reorder_generation
    buffers.tile_generation = nf.tile_generation
    buffers.step_n_preprocessed = step_n
    return buffers
end

#=
Bring the atom order, the exception data, the reordered arrays and the tile list up to
date as required by the flags.
Every step is ordered on the device by the queue it is launched on, so the host does not
wait for the reorder. It does wait for the device when the tile list is rebuilt, to find the
number of tiles.
=#
function refresh_gpu_tiles!(buffers, sys, nf, pairwise_inters, step_n, flags)
    needs_order_refresh, needs_reorder, needs_sparse_refresh, needs_tile_refresh = flags
    N = length(sys.coords)
    if needs_reorder || needs_sparse_refresh || needs_tile_refresh
        if needs_order_refresh
            sort_reorder_seq!(buffers, sys.coords, sys.boundary; curve=nf.curve)
        end

        if needs_sparse_refresh
            refresh_tile_exceptions!(buffers, nf, Val(N))
            nf.initialized = true
            buffers.sparse_pair_generation = nf.cache_generation
        end

        if needs_reorder
            reorder_system_gpu!(buffers, sys, Val(needs_order_refresh))
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
function resolve_tile_check!(ref::TileReference, check::TileCheckScratch, buffers, sys,
                             refreshed_inner::Bool)
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
                                   pairwise_inters, step_n, uses_inner::Bool,
                                   refreshes_inner::Bool, defer::Bool=false) where K
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
                       gpu_neighbor_refresh_flags(buffers, nf, moved))
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
                                    pairwise_inters, step_n, uses_inner::Bool) where K
    ref = tile_reference(buffers, sys)
    isnothing(ref) && return false
    check = ref.pending_check
    isnothing(check) && return false
    moved = resolve_tile_check!(ref, check, buffers, sys, uses_inner)
    if moved
        refresh_gpu_tiles!(buffers, sys, nf, pairwise_inters, step_n,
                           gpu_neighbor_refresh_flags(buffers, nf, true))
        launch_kernel!()
        uses_inner && record_inner_refresh!(tile_reference(buffers, sys))
    end
    return moved
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

# The neighbor finder of a system that uses the tiled kernels, or an error for a system
#   that has none but calls the pairwise kernels without a neighbor list
@inline function tiled_neighbor_finder(sys)
    nf = sys.neighbor_finder
    if !(nf isa GPUNeighborFinder)
        error("an interaction uses the neighbor list but neighbors is nothing, pairwise " *
              "interactions used without a neighbor finder need use_neighbors=false and " *
              "GPUNeighborFinder is only available on GPUs that support the tiled kernels, " *
              "on other GPUs use GPUCellListNeighborFinder or DistanceNeighborFinder instead")
    end
    return nf
end

#=
    tiled_pairwise_forces!(buffers, sys, pairwise_inters, needs_vir, step_n, defer=false)

Maintainer entry point for the tiled pairwise force path of [`GPUNeighborFinder`](@ref).

Pipeline:
1. Rebuild the atom order and the reordered exception data when the
   atoms have moved far enough since the last rebuild to invalidate the tile list
   with `auto_update=true`, or when [`find_neighbors`](@ref) has marked them as due
   every `n_steps` steps of the [`GPUNeighborFinder`](@ref) with `auto_update=false`.
2. Reorder coordinates, velocities, and atoms into the atom order.
3. Recompute the compact list of interacting 32x32 tiles when the order changed
   or [`find_neighbors`](@ref) forced a rebuild, growing it if it overflows.
4. Launch `force_kernel!` over that compact tile list and reverse the reorder.

Cache contract:
- The atom order and tile list are rebuilt when [`find_neighbors`](@ref) has
  increased the rebuild counts of the [`GPUNeighborFinder`](@ref) since they were
  last built, when `buffers.step_n_preprocessed` is `-1` for new buffers, or with
  `auto_update=true` when the displacement check against `buffers.neighbor_reference`
  fails. The force kernel is then launched before the result of the check is known,
  and the check sets `buffers.interacting_tiles_overflow` when it fails so that the
  kernel returns straight away, see `run_tiled_kernel!`.
- With `defer`, as `gpu_forces!` calls it through `start_pairwise_forces_gpu!`, the host
  does not wait for the check and the forces are left in `buffers.fs_mat_reordered`.
  `finish_pairwise_forces_gpu!`, called once the specific forces are launched, waits for
  it, calculates the forces again if it failed and reverses the reorder.
- `buffers.num_pairs` is the host-side cached interacting-tile count used to
  size the force-kernel launch.
- `sys.neighbor_finder.initialized` only indicates whether the reordered
  exception data is current.

This works on any backend that `supports_tiled_kernels`, for any array type, which
lets the GPU code path be tested with `Array`s on the CPU backend of KernelAbstractions.
=#
function tiled_pairwise_forces!(buffers, sys, pairwise_inters, ::Val{needs_vir}, step_n,
                                defer::Bool=false) where needs_vir
    nf = tiled_neighbor_finder(sys)
    r_cut2 = kernel_pair_cutoff_2(sys, pairwise_inters)
    launch_kernel! = () -> launch_force_kernel!(buffers, sys, pairwise_inters, r_cut2,
                                                Val(needs_vir), step_n)
    run_tiled_kernel!(launch_kernel!, buffers, sys, nf, pairwise_inters, step_n, true, true,
                      defer)
    defer || reverse_reorder_forces_gpu!(buffers, sys)
    return buffers
end

# If the check failed the kernel launched before it did nothing, and it is launched again
function finish_tiled_pairwise_forces!(buffers, sys, pairwise_inters, ::Val{needs_vir},
                                       step_n) where needs_vir
    nf = tiled_neighbor_finder(sys)
    r_cut2 = kernel_pair_cutoff_2(sys, pairwise_inters)
    launch_kernel! = () -> launch_force_kernel!(buffers, sys, pairwise_inters, r_cut2,
                                                Val(needs_vir), step_n)
    finish_tile_check!(launch_kernel!, buffers, sys, nf, pairwise_inters, step_n, true)
    reverse_reorder_forces_gpu!(buffers, sys)
    return buffers
end

#=
    tiled_pairwise_pe!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n)

Maintainer entry point for the tiled pairwise energy path of [`GPUNeighborFinder`](@ref).

This follows the same reorder -> exception translation -> tile search
pipeline as `tiled_pairwise_forces!`, including the displacement check, but launches
`energy_kernel!` instead of the force kernel. Energy evaluation reuses any preprocessing
already performed since the last rebuild, so forces and energies share the same cached
tile metadata.
=#
function tiled_pairwise_pe!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n)
    nf = tiled_neighbor_finder(sys)
    r_cut2 = kernel_pair_cutoff_2(sys, pairwise_inters)
    launch_kernel! = () -> launch_energy_kernel!(pe_vec_nounits, buffers, sys,
                                                 pairwise_inters, r_cut2, step_n)
    # The energy kernel uses the inner masks when they are current but does not find them
    #   again when they are not
    run_tiled_kernel!(launch_kernel!, buffers, sys, nf, pairwise_inters, step_n, true, false)
    return pe_vec_nounits
end

function pairwise_forces_loop_gpu!(buffers, sys::System{D, <:AbstractGPUArray}, pairwise_inters,
                                   ::Nothing, ::Val{needs_vir}, step_n) where {D, needs_vir}
    return tiled_pairwise_forces!(buffers, sys, pairwise_inters, Val(needs_vir), step_n)
end

#=
As above, but the host does not wait for the check of the tile list, which is left to
`finish_pairwise_forces_gpu!` once `gpu_forces!` has launched the specific forces. The
forces are added to `fs_mat` by `finish_pairwise_forces_gpu!`, so that the specific forces
can be added to it in the meantime, see `gpu_forces_concurrent!`.
=#
function start_pairwise_forces_gpu!(buffers, sys::System{D, <:AbstractGPUArray},
                                    pairwise_inters, ::Nothing, ::Val{needs_vir},
                                    step_n) where {D, needs_vir}
    return tiled_pairwise_forces!(buffers, sys, pairwise_inters, Val(needs_vir), step_n,
                                  true)
end

function finish_pairwise_forces_gpu!(buffers, sys::System{D, <:AbstractGPUArray},
                                     pairwise_inters, ::Nothing, ::Val{needs_vir},
                                     step_n) where {D, needs_vir}
    return finish_tiled_pairwise_forces!(buffers, sys, pairwise_inters, Val(needs_vir),
                                         step_n)
end

function pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys::System{<:Any, <:AbstractGPUArray},
                               pairwise_inters, ::Nothing, step_n)
    return tiled_pairwise_pe!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n)
end

# Launch the force kernel over the list of interacting tiles
# `r_cut2` is a value rather than a `Val` so that the call to this function is not a
#   dynamic dispatch
function launch_force_kernel!(buffers, sys, pairwise_inters, r_cut2, ::Val{needs_vir},
                              step_n) where needs_vir
    block_y = force_block_y(sys, buffers, pairwise_inters)
    maxregs = force_maxregs(sys)
    inner = inner_kernel_args(buffers, sys, r_cut2)
    if isnothing(inner)
        return launch_force_tiles!(buffers, sys, pairwise_inters, Val(needs_vir), step_n,
                                   block_y, maxregs, nothing, Val(r_cut2))
    else
        return launch_force_tiles!(buffers, sys, pairwise_inters, Val(needs_vir), step_n,
                                   block_y, maxregs, inner, Val(r_cut2))
    end
end

# Launch the energy kernel over the list of interacting tiles
function launch_energy_kernel!(pe_vec_nounits, buffers, sys, pairwise_inters, r_cut2,
                               step_n)
    block_y = energy_block_y(sys, buffers, pairwise_inters)
    inner = inner_kernel_args(buffers, sys, r_cut2)
    if isnothing(inner)
        return launch_energy_tiles!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n,
                                    block_y, nothing, Val(r_cut2))
    else
        return launch_energy_tiles!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n,
                                    block_y, inner, Val(r_cut2))
    end
end

#=
Launch `force_kernel!` with `block_y` tiles per work-group, or fewer if the backend or the
compiled kernel does not allow that many, see `tiles_per_group` and `clamp_block_y`.
`inner` is `nothing` or the inner masks of the dynamic pruning, see `inner_kernel_args`.
=#
function launch_force_tiles!(buffers, sys::System{D, <:Any, T, TH}, pairwise_inters,
                             ::Val{needs_vir}, step_n, block_y::Integer,
                             maxregs=nothing, inner=nothing,
                             r_cut2_val=Val(kernel_pair_cutoff_2(sys, pairwise_inters))) where {D, T, TH, needs_vir}
    num_pairs = buffers.num_pairs
    num_pairs > 0 || return buffers
    backend = get_backend(sys.coords)
    block_y = tiles_per_group(backend, block_y)
    kernel_backend, options = tile_kernel_config(backend, T, maxregs)
    uses_vel = any_uses_velocity(pairwise_inters)
    launch_force_tiles!(kernel_backend, options, buffers, sys, pairwise_inters, Val(needs_vir),
                        step_n, Val(Int(block_y)), tile_sub_group_width(backend), Val(uses_vel),
                        inner, r_cut2_val)
    return buffers
end

function launch_force_tiles!(kernel_backend, options, buffers, sys::System{D, <:Any, T, TH},
                             pairwise_inters, ::Val{needs_vir}, step_n, ::Val{block_y},
                             ::Val{W}, ::Val{uses_vel}, inner,
                             ::Val{r_cut2}) where {D, T, TH, needs_vir, block_y, W, uses_vel,
                                                   r_cut2}
    N = length(sys.coords)
    args = (buffers.fs_mat_reordered, buffers.virial_slots, buffers.coords_reordered,
            buffers.velocities_reordered, buffers.atoms_reordered, Val(N), Val(r_cut2),
            Val(sys.force_units), pairwise_inters, sys.boundary, step_n,
            tile_exceptions(buffers, sys.neighbor_finder), Val(needs_vir), Val(T), Val(TH),
            Val(D), Val(block_y), Val(W), Val(uses_vel), buffers.interacting_tiles_i,
            buffers.interacting_tiles_j, buffers.interacting_tiles_type,
            buffers.interacting_tiles_diag, Int32(buffers.num_pairs),
            buffers.interacting_tiles_overflow, inner)
    kernel = tile_kernel(kernel_backend, force_kernel!, args, options)
    # A kernel with many registers can allow fewer work-items than the requested tiles
    if TILE_WIDTH * block_y > KI.max_work_group_size(kernel)
        by = clamp_block_y(block_y, _ -> 0, typemax(Int), KI.max_work_group_size(kernel))
        return launch_force_tiles!(kernel_backend, options, buffers, sys, pairwise_inters,
                                   Val(needs_vir), step_n, Val(Int(by)), Val(W), Val(uses_vel),
                                   inner, Val(r_cut2))
    end
    kernel(args...; numgroups=cld(buffers.num_pairs, block_y),
           workgroupsize=TILE_WIDTH * block_y)
    return buffers
end

function launch_energy_tiles!(pe_vec_nounits, buffers, sys::System{D, <:Any, T, TH},
                              pairwise_inters, step_n, block_y::Integer, inner=nothing,
                              r_cut2_val=Val(kernel_pair_cutoff_2(sys, pairwise_inters))) where {D, T, TH}
    num_pairs = buffers.num_pairs
    num_pairs > 0 || return pe_vec_nounits
    backend = get_backend(sys.coords)
    block_y = tiles_per_group(backend, block_y)
    kernel_backend, options = tile_kernel_config(backend, T, nothing)
    uses_vel = any_uses_velocity(pairwise_inters)
    launch_energy_tiles!(kernel_backend, options, pe_vec_nounits, buffers, sys,
                         pairwise_inters, step_n, Val(Int(block_y)),
                         tile_sub_group_width(backend), Val(uses_vel), inner, r_cut2_val)
    return pe_vec_nounits
end

function launch_energy_tiles!(kernel_backend, options, pe_vec_nounits, buffers,
                              sys::System{D, <:Any, T, TH}, pairwise_inters, step_n,
                              ::Val{block_y}, ::Val{W}, ::Val{uses_vel}, inner,
                              ::Val{r_cut2}) where {D, T, TH, block_y, W, uses_vel, r_cut2}
    N = length(sys.coords)
    args = (pe_vec_nounits, buffers.coords_reordered, buffers.velocities_reordered,
            buffers.atoms_reordered, Val(N), Val(r_cut2), Val(sys.energy_units),
            pairwise_inters, sys.boundary, step_n,
            tile_exceptions(buffers, sys.neighbor_finder), Val(T), Val(TH), Val(D),
            Val(block_y), Val(W), Val(uses_vel), buffers.interacting_tiles_i,
            buffers.interacting_tiles_j, buffers.interacting_tiles_type,
            buffers.interacting_tiles_diag, Int32(buffers.num_pairs),
            buffers.interacting_tiles_overflow, inner)
    kernel = tile_kernel(kernel_backend, energy_kernel!, args, options)
    if TILE_WIDTH * block_y > KI.max_work_group_size(kernel)
        by = clamp_block_y(block_y, _ -> 0, typemax(Int), KI.max_work_group_size(kernel))
        return launch_energy_tiles!(kernel_backend, options, pe_vec_nounits, buffers, sys,
                                    pairwise_inters, step_n, Val(Int(by)), Val(W),
                                    Val(uses_vel), inner, Val(r_cut2))
    end
    kernel(args...; numgroups=cld(buffers.num_pairs, block_y),
           workgroupsize=TILE_WIDTH * block_y)
    return pe_vec_nounits
end

# The bounding boxes of the 32-atom blocks
function compute_block_bounds!(buffers, sys::System{D}, N::Integer) where D
    n_blocks = cld(N, TILE_WIDTH)
    backend = get_backend(sys.coords)
    launch_kernel!(kernel_min_max!(backend, TILE_WIDTH), buffers.box_mins, buffers.box_maxs,
                   buffers.reorder_seq, sys.coords, Val(N), sys.boundary, Val(D);
                   ndrange=n_blocks * TILE_WIDTH)
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
#   tiles hold exclusions or special pairs, and remove the tiles without an in-range pair
function prune_interacting_tiles!(buffers, sys, N::Int)
    n_blocks = cld(N, TILE_WIDTH)
    if buffers.num_pairs > 0
        backend = get_backend(sys.coords)
        prune_y = tiles_per_group(backend, 4)
        launch_tile_kernel!(backend, prune_interacting_tiles_kernel!,
            buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.interacting_tiles_diag,
            Int32(buffers.num_pairs),
            buffers.coords_reordered, Val(N), Val(sys.neighbor_finder.dist_cutoff_2),
            Val(n_blocks), sys.boundary, tile_exceptions(buffers, sys.neighbor_finder),
            Val(prune_y), tile_sub_group_width(backend);
            numgroups=cld(buffers.num_pairs, prune_y), workgroupsize=TILE_WIDTH * prune_y)
        compact_interacting_tiles!(buffers)
    end
    return buffers
end

#=
Remove the tiles marked `TILE_DEAD` by `prune_interacting_tiles_kernel!` from the list,
keeping the order of the others, and set the tile counts to the number left.
The force and energy kernels run more than one tile in a work-group, which keeps its
place on a multiprocessor until its last sub-group finishes. The sub-group of a dead tile
returns straight away but is not replaced until the other tiles of its work-group are done,
so with dead tiles in the list fewer sub-groups run at once. About a third of the candidate
tiles of a solvated protein are dead.
This runs when the list is rebuilt, and waits for the device to get the number of tiles.
=#
function compact_interacting_tiles!(buffers)
    n_tiles = buffers.num_pairs
    n_tiles == 0 && return buffers
    backend = get_backend(buffers.interacting_tiles_i)
    # The tiles are copied so that the compacted list can be written over the original
    tiles_i = buffers.interacting_tiles_i[1:n_tiles]
    tiles_j = buffers.interacting_tiles_j[1:n_tiles]
    tiles_type = buffers.interacting_tiles_type[1:n_tiles]
    tiles_diag = buffers.interacting_tiles_diag[1:n_tiles]
    positions = similar(buffers.interacting_tiles_i, Int32, n_tiles)
    n_threads = 256
    launch_kernel!(live_tile_flags_kernel!(backend, n_threads), positions, tiles_type,
                   Int32(n_tiles); ndrange=n_tiles)
    # The inclusive prefix sum of the flags is the place of each live tile in the list
    AcceleratedKernels.accumulate!(+, positions; init=Int32(0))
    launch_kernel!(compact_tiles_kernel!(backend, n_threads), buffers.interacting_tiles_i,
                   buffers.interacting_tiles_j, buffers.interacting_tiles_type,
                   buffers.interacting_tiles_diag, buffers.num_interacting_tiles, tiles_i,
                   tiles_j, tiles_type, tiles_diag, positions, Int32(n_tiles); ndrange=n_tiles)
    buffers.num_pairs = Int(only(from_device(view(positions, n_tiles:n_tiles))))
    foreach(KI.unsafe_free!, (tiles_i, tiles_j, tiles_type, tiles_diag, positions))
    return buffers
end

@kernel inbounds=true function live_tile_flags_kernel!(flags, @Const(tiles_type), n_tiles)
    t = @index(Global, Linear)
    if t <= n_tiles
        flags[t] = (tiles_type[t] == TILE_DEAD ? Int32(0) : Int32(1))
    end
end

@kernel inbounds=true function compact_tiles_kernel!(tiles_i_out, tiles_j_out, tiles_type_out,
                        tiles_diag_out, num_interacting_tiles, @Const(tiles_i),
                        @Const(tiles_j), @Const(tiles_type), @Const(tiles_diag),
                        @Const(positions), n_tiles)
    t = @index(Global, Linear)
    if t <= n_tiles
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
    launch_kernel!(find_interacting_blocks_tree_kernel!(backend, 128),
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, buffers.tree_mins, buffers.tree_maxs, offsets,
        Val(top), sys.boundary, Val(sys.neighbor_finder.dist_cutoff_2), Val(n_blocks), Val(D),
        length(buffers.interacting_tiles_i), buffers.block_exc_min, buffers.block_exc_max;
        ndrange=n_blocks)
    return buffers
end

#=
Whether to use the tree search rather than the all-pairs search, given the number of
tiles found by the previous search, or 0 before the first search.
The all-pairs search takes time in proportion to the square of the number of blocks.
Below about a million atoms there are too few blocks to keep the GPU busy with one work-item
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
    min_blocks = env_override("MOLLY_TILE_TREE_MIN_BLOCKS")
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
        launch_kernel!(build_tile_tree_level_kernel!(backend, 256),
            buffers.tree_mins, buffers.tree_maxs, src_mins, src_maxs, src_offset,
            offsets[k], Int32(n_src), Int32(n_dst), Val(D); ndrange=n_dst)
    end
    return top, offsets
end

#=
The tree search appends tiles in the order that its work-items find them, so tiles next to
each other in the list rarely share a block. `force_kernel!` runs consecutive tiles on
the sub-groups of a work-group, and with that order it is 10-20% slower than with the tiles
of the all-pairs search, which appends the tiles of each 32x32 group of blocks column by
column. Sorting the tiles into that order gives the speed back. The sort key holds the
whole tile, so the tiles are read back from the sorted keys.
=#
function sort_interacting_tiles!(buffers, n_tiles)
    n_tiles == 0 && return buffers
    backend = get_backend(buffers.interacting_tiles_i)
    keys = similar(buffers.interacting_tiles_i, UInt64, n_tiles)
    n_threads = 256
    launch_kernel!(tile_sort_keys_kernel!(backend, n_threads), keys,
                   buffers.interacting_tiles_i, buffers.interacting_tiles_j,
                   buffers.interacting_tiles_type, Int32(n_tiles); ndrange=n_tiles)
    AcceleratedKernels.sort!(keys)
    launch_kernel!(tiles_from_sort_keys_kernel!(backend, n_threads),
                   buffers.interacting_tiles_i, buffers.interacting_tiles_j,
                   buffers.interacting_tiles_type, keys, Int32(n_tiles); ndrange=n_tiles)
    KI.unsafe_free!(keys)
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

@kernel inbounds=true function tile_sort_keys_kernel!(keys, @Const(tiles_i), @Const(tiles_j),
                                                      @Const(tiles_type), n_tiles)
    t = @index(Global, Linear)
    if t <= n_tiles
        keys[t] = tile_sort_key(tiles_i[t], tiles_j[t], tiles_type[t])
    end
end

@kernel inbounds=true function tiles_from_sort_keys_kernel!(tiles_i, tiles_j, tiles_type,
                                                            @Const(keys), n_tiles)
    t = @index(Global, Linear)
    if t <= n_tiles
        key = keys[t]
        i0 = (((key >> 11) & 0x03ffffff) << 5) | ((key >> 1) & 0x1f)
        j0 = ((key >> 37) << 5) | ((key >> 6) & 0x1f)
        tiles_i[t] = (i0 % Int32) + Int32(1)
        tiles_j[t] = (j0 % Int32) + Int32(1)
        tiles_type[t] = (key & 0x01) % UInt8
    end
end

#=
    reorder_system_gpu!(buffers, sys)

Reorders `coords`, `velocities`, and `atoms` into `_reordered` buffers
according to the reorder sequence to improve spatial cache locality.
=#
function reorder_system_gpu!(buffers, sys)
    return reorder_system_gpu!(buffers, sys, Val(true))
end

function reorder_system_gpu!(buffers, sys, ::Val{reorder_atoms}) where reorder_atoms
    N = length(sys)
    backend = get_backend(sys.coords)
    launch_kernel!(reorder_system_kernel!(backend, 256),
        buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
        sys.coords, sys.velocities, sys.atoms, buffers.reorder_seq, Val(reorder_atoms);
        ndrange=N)
    return nothing
end

#=
    reverse_reorder_forces_gpu!(buffers, sys)

Maps forces from the `fs_mat_reordered` buffer back to the original atom indices
in `fs_mat`.
=#
function reverse_reorder_forces_gpu!(buffers, sys::System{D}) where D
    N = length(sys)
    backend = get_backend(sys.coords)
    launch_kernel!(reverse_reorder_forces_kernel!(backend, 256),
        buffers.fs_mat, buffers.fs_mat_reordered, buffers.reorder_seq, Val(D); ndrange=N)
    return nothing
end

#=
The exclusions and special pairs read by the tiled kernels.
The per-atom lists of `nf.eligible` and `nf.special`, which are SparsePairMatrix values
on the device, are indexed by original atom index. `buffers.excluded_pos` and
`buffers.special_pos` hold, for each list entry, the reordered position of the partner atom,
refreshed by `refresh_tile_exceptions!` whenever the atom order changes. This takes
memory proportional to the number of exceptions, where a mask per tile takes memory
proportional to `n_atoms^2`.
The NamedTuple is converted to device arrays when a kernel is launched.
=#
function tile_exceptions(buffers, nf::GPUNeighborFinder)
    return (
        reorder_seq=buffers.reorder_seq,
        excluded_starts=nf.eligible.starts,
        excluded_pos=buffers.excluded_pos,
        special_starts=nf.special.starts,
        special_pos=buffers.special_pos,
    )
end

#=
    refresh_tile_exceptions!(buffers, nf, N)

Translate the sparse exception lists stored on [`GPUNeighborFinder`](@ref) into the
reordered form read by the tiled pairwise kernels.

This needs work proportional to the number of atoms and exceptions: the inverse reorder
map is updated, the partner of each exception is converted to its reordered position and
the range of blocks holding the partners of the atoms of each 32-atom block is found,
which lets the tile search mark most tiles as free of exceptions without looking at
the atoms.
=#
function refresh_tile_exceptions!(buffers, nf::GPUNeighborFinder, ::Val{N}) where N
    n_blocks = cld(N, TILE_WIDTH)
    backend = get_backend(buffers.reorder_seq)
    launch_kernel!(update_reorder_seq_inv_kernel!(backend, 256), buffers.reorder_seq_inv,
                   buffers.reorder_seq; ndrange=N)

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
    launch_kernel!(tile_exceptions_kernel!(backend, n_items),
        buffers.excluded_pos, buffers.special_pos, buffers.block_exc_min,
        buffers.block_exc_max, nf.eligible.starts, nf.eligible.partners,
        nf.special.starts, nf.special.partners, buffers.reorder_seq,
        buffers.reorder_seq_inv, Val(N), Val(n_blocks), tile_sub_group_width(backend);
        ndrange=cld(N, n_items) * n_items)
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

# Whether some periodic image of the bounding box of one block of atoms is within the cutoff
#   of the bounding box of another
@inline function boxes_within_cutoff(r1_min, r1_max, r2_min, r2_max, boundary, r_cut2)
    d_block = boxes_dist(r1_min, r1_max, r2_min, r2_max, boundary)
    return sum(d_block .* d_block) <= r_cut2
end

#=
The block bounds of a triclinic box are in the brick coordinates of `aligned_coord`.
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
@inline function boxes_within_cutoff(r1_min, r1_max, r2_min, r2_max,
                                     boundary::TriclinicBoundary, r_cut2)
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

@inline min_max_op(a, b) = (min.(a[1], b[1]), max.(a[2], b[2]))

#=
    kernel_min_max!(mins, maxs, sorted_seq, coords, n_atoms, boundary, D)

Compute the minimum and maximum coordinates for each 32-atom block, with one work-group
of 32 work-items, a single sub-group, per block. The bounds are found with a sub-group
reduction, which needs no local memory and no barrier. With 64-wide sub-groups the
sub-group of a work-group is partial, which the reduction takes into account.
For a triclinic boundary the coordinates are first wrapped into the brick of the
diagonal components of the basis vectors, see `aligned_coord`.
These bounds are used for fast bounding-box intersection tests during tile finding
to skip non-interacting tile pairs.
=#
@kernel unsafe_indices=true inbounds=true function kernel_min_max!(
                    mins::AbstractArray{C}, maxs, @Const(sorted_seq), @Const(coords),
                    ::Val{N}, boundary, ::Val{D}) where {C, N, D}
    block = @index(Group, Linear)
    lane = KI.get_sub_group_local_id(Int32)
    i = (block - 1) * TILE_WIDTH + lane
    # Very large (arbitrary) bounds, which the lanes past the last atom contribute
    big = SVector{D, C}(ntuple(k -> 10 * box_sides(boundary, k), Val(D)))
    if i <= N
        r = aligned_coord(coords[sorted_seq[i]], boundary)
        lo, hi = r, r
    else
        lo, hi = big, -big
    end
    # Every lane of the sub-group takes part
    lo, hi = KI.sub_group_reduce(min_max_op, (lo, hi))
    if lane == 1
        for k in 1:D
            mins[block, k] = lo[k]
            maxs[block, k] = hi[k]
        end
    end
end

#=
    update_reorder_seq_inv_kernel!(inv_reorder_seq, reorder_seq)

Generate the inverse reorder mapping.
Scatters the dense `1:N` original atom indices into their new reordered
positions in `inv_reorder_seq`.
=#
@kernel inbounds=true function update_reorder_seq_inv_kernel!(inv_reorder_seq,
                                                               @Const(reorder_seq))
    i = @index(Global, Linear)
    inv_reorder_seq[reorder_seq[i]] = i
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

# Convert the partners of one atom's exception list to reordered positions, returning the
#   lowest and highest block they fall in
@inline function translate_exception_list!(pos, starts, partners, reorder_seq_inv, atom_i,
                                           min_block, max_block)
    @inbounds k_start, k_end = starts[atom_i], starts[atom_i + Int32(1)]
    k = k_start
    @inbounds while k < k_end
        q = reorder_seq_inv[partners[k]]
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
One work-item per reordered position `p`, so that each segment of a sub-group covers one
32-atom block, see `tile_lane`. The work-item converts the exception partners of the atom
at `p` to reordered positions, writing each list entry once since each atom belongs to one
position. The segment then reduces the lowest and highest block holding a partner of any
atom in the block, which is `typemax(Int32)` and `0` for a block without exceptions.
=#
@kernel unsafe_indices=true inbounds=true function tile_exceptions_kernel!(excluded_pos,
                    special_pos, block_exc_min, block_exc_max, @Const(excluded_starts),
                    @Const(excluded_partners), @Const(special_starts),
                    @Const(special_partners), @Const(reorder_seq), @Const(reorder_seq_inv),
                    ::Val{N}, ::Val{n_blocks}, ::Val{W}) where {N, n_blocks, W}
    lane = tile_lane(Val(W))
    tiles_per_group = KI.get_local_size(Int32).x ÷ Int32(TILE_WIDTH)
    block_i = (@index(Group, Linear) - Int32(1)) * tiles_per_group + group_tile(Val(W))
    p = (block_i - Int32(1)) * Int32(TILE_WIDTH) + lane
    min_block = typemax(Int32)
    max_block = Int32(0)
    if p <= N
        atom_i = reorder_seq[p]
        min_block, max_block = translate_exception_list!(excluded_pos, excluded_starts,
                    excluded_partners, reorder_seq_inv, atom_i, min_block, max_block)
        min_block, max_block = translate_exception_list!(special_pos, special_starts,
                    special_partners, reorder_seq_inv, atom_i, min_block, max_block)
    end

    # Every work-item of the segment takes part, including those past the last atom
    min_block, max_block = tile_reduce(min_max_block_op, (min_block, max_block), Val(W))

    if lane == Int32(1) && block_i <= n_blocks
        block_exc_min[block_i] = min_block
        block_exc_max[block_i] = max_block
    end
end

# Whether the atom at reordered position `p` has an exclusion or special pair with an atom
#   in block `block_i`
@inline function atom_has_partner_in_block(tile_exceptions, p, block_i)
    reorder_seq = KernelAbstractions.constify(tile_exceptions.reorder_seq)
    excluded_starts = KernelAbstractions.constify(tile_exceptions.excluded_starts)
    excluded_pos = KernelAbstractions.constify(tile_exceptions.excluded_pos)
    special_starts = KernelAbstractions.constify(tile_exceptions.special_starts)
    special_pos = KernelAbstractions.constify(tile_exceptions.special_pos)

    @inbounds atom_i = reorder_seq[p]
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
        reorder_seq = KernelAbstractions.constify(tile_exceptions.reorder_seq)
        excluded_starts = KernelAbstractions.constify(tile_exceptions.excluded_starts)
        excluded_pos = KernelAbstractions.constify(tile_exceptions.excluded_pos)
        special_starts = KernelAbstractions.constify(tile_exceptions.special_starts)
        special_pos = KernelAbstractions.constify(tile_exceptions.special_pos)

        @inbounds atom_i = reorder_seq[p]
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
    find_interacting_blocks_kernel!(interacting_tiles_i, interacting_tiles_j,
                                    interacting_tiles_type, num_interacting_tiles,
                                    interacting_tiles_overflow, mins, maxs, boundary,
                                    r_cut2, N_blocks, D, max_total_tiles,
                                    block_exc_min, block_exc_max)

Scan the upper-triangular matrix of 32x32 reordered atom tiles and append
only those whose bounding boxes fall within `r_cut`.

A full off-diagonal tile is marked clean, i.e. free of exclusions and special pairs,
unless each block lies in the range of blocks holding exception partners of the
other, see `refresh_tile_exceptions!`. That test is conservative and
`prune_interacting_tiles_kernel!` marks the tiles that pass it but hold no exception
as clean.
=#
@kernel inbounds=true function find_interacting_blocks_kernel!(
            interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
            num_interacting_tiles, interacting_tiles_overflow, @Const(mins), @Const(maxs),
            boundary, ::Val{r_cut2}, ::Val{N_blocks}, ::Val{D}, max_total_tiles,
            @Const(block_exc_min), @Const(block_exc_max)) where {r_cut2, N_blocks, D}
    i, j = @index(Global, NTuple)

    if i <= j
        r_min_i, r_max_i = stored_box(mins, maxs, i, Val(D))
        r_min_j, r_max_j = stored_box(mins, maxs, j, Val(D))

        if boxes_within_cutoff(r_min_i, r_max_i, r_min_j, r_max_j, boundary, r_cut2)
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
reordered atoms, runs of consecutive blocks are spatially compact, so the bounding
boxes of the blocks can be merged pairwise into an implicit binary tree: level 0 holds
the blocks and node `n` of level `k` holds the union of the boxes of blocks
`(n - 1) * 2^k + 1` to `n * 2^k`. One work-item per block `i` then walks the tree from the
root, skipping every subtree whose box is further than the cutoff from block `i` or
that only holds blocks `j < i`, which takes time proportional to the number of
interacting tiles times the depth of the tree.

The box distance only gets smaller as a box grows, so a subtree that is skipped holds
no block that the brute-force search would have accepted and the two searches give the
same tiles. This holds for triclinic boundaries too, see `boxes_within_cutoff`. The
tiles are then sorted into the order of the brute-force search, see
`sort_interacting_tiles!`.
=#

# Fill node `n` of a tree level with the union of the boxes of its two children
@kernel inbounds=true function build_tile_tree_level_kernel!(tree_mins, tree_maxs,
                    src_mins, src_maxs, src_offset, dst_offset, n_src, n_dst,
                    ::Val{D}) where D
    n = Int32(@index(Global, Linear))
    if n <= n_dst
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
            visit = boxes_within_cutoff(r_min_i, r_max_i, r_min_j, r_max_j, boundary, r_cut2)
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
For the boundaries with a rectangular box, the displacements of the pairs of a tile can
be found by subtracting the coordinates of atom i from those of atom j once the image of
atom i closest to a reference point is taken, rather than by finding the minimum image of
each pair, which takes several times as many instructions as the subtraction. With the
reference point the first atom of block j, this gives the minimum image of every pair
within the cutoff if the other atoms of block j are within `d` of it, with `d` plus the
cutoff less than half the shortest side of the box. The nearer image of a pair beyond the
cutoff can be missed, but the image used is then beyond the cutoff too.
Returns whether this holds for the segment, `coords_j` being the atom of block j of this
lane and `ref_j` the first one, and the image of atom i to use if so.
The other boundaries use the minimum image of each pair.
Called by every work-item of the segment, of a sub-group of width `W`.
=#
@inline function tile_single_image(coords_i, coords_j, ref_j,
                                   boundary::Union{CubicBoundary, RectangularBoundary},
                                   ::Val{r_cut2}, ::Val{W}) where {r_cut2, W}
    return tile_single_image(coords_i, coords_j, ref_j, boundary, sqrt(r_cut2), Val(W))
end

# As above for the cutoff `r_cut`
@inline function tile_single_image(coords_i, coords_j, ref_j,
                                   boundary::Union{CubicBoundary, RectangularBoundary}, r_cut,
                                   ::Val{W}) where W
    d2 = ustrip(sum(abs2, coords_j - ref_j))
    offset = Int32(TILE_WIDTH ÷ 2)
    while offset > Int32(0)
        d2 = max(d2, tile_shfl_xor(d2, offset, Val(W)))
        offset >>= Int32(1)
    end
    # A plain comparison rather than `minimum`, which handles NaN and signed zeros with
    #   several instructions per element on every tile
    min_side = reduce((a, b) -> ifelse(a < b, a, b), boundary.side_lengths)
    max_d = min_side / 2 - r_cut
    valid = max_d > zero(max_d) && d2 < ustrip(max_d^2)
    return valid, image_closest(coords_i, ref_j, boundary)
end

@inline tile_single_image(coords_i, coords_j, ref_j, boundary, r_cut, ::Val) = (false, coords_i)

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

# This lane holds the `lane`-th atom of block j, so a hit against the m-th atom of block i
#   lands on inner-loop iteration `(lane - m) & 31`
@inline function prune_lane_mask(coords, i_0_tile, coords_j, coords_j_image, lane, boundary,
                                 ::Val{r_cut2}, ::Val{single_image}) where {r_cut2, single_image}
    a = Int32(1)
    lane_mask = UInt32(0)
    @inbounds for m in a:Int32(TILE_WIDTH)
        coords_i = coords[i_0_tile + m]
        dr = (single_image ? coords_j_image - coords_i : vector(coords_i, coords_j, boundary))
        r2 = sum(abs2, dr)
        if r2 <= r_cut2
            lane_mask |= UInt32(1) << ((lane - m) & Int32(31))
        end
    end
    return lane_mask
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
in-range pair at all get an empty mask and are marked `TILE_DEAD`, and are then removed
from the list by `compact_interacting_tiles!`.

This only runs on a neighbor list rebuild, so its cost is amortised over the steps
between rebuilds. Only full off-diagonal tiles (`i < j < n_blocks`) are analysed; the
diagonal and the final partial block make up a vanishing fraction of the list and keep a
fully populated mask. A segment of a sub-group of width `W` processes a tile, see
`tile_lane`.

Tiles that the block ranges in `find_interacting_blocks_kernel!` could not rule out as
holding an exclusion or special pair are checked exactly here, and marked clean if
none of their atom pairs is an exception.
=#
function prune_interacting_tiles_kernel!(interacting_tiles_i, interacting_tiles_j,
                    interacting_tiles_type, interacting_tiles_diag, num_pairs::Int32,
                    coords_var, ::Val{N}, ::Val{r_cut2}, ::Val{n_blocks}, boundary,
                    tile_exceptions, ::Val{block_y},
                    ::Val{W}) where {N, r_cut2, n_blocks, block_y, W}
    coords = KernelAbstractions.constify(coords_var)
    tiles_i_ro = KernelAbstractions.constify(interacting_tiles_i)
    tiles_j_ro = KernelAbstractions.constify(interacting_tiles_j)

    a = Int32(1)
    lane = tile_lane(Val(W))
    idx = (KI.get_group_id(Int32).x - a) * Int32(block_y) + group_tile(Val(W))

    # The tile is uniform across the segment, so it returns as a whole
    if idx > num_pairs
        return nothing
    end

    @inbounds i = tiles_i_ro[idx]
    @inbounds j = tiles_j_ro[idx]
    if i >= j || j >= Int32(n_blocks)
        if lane == a
            @inbounds interacting_tiles_diag[idx] = typemax(UInt32)
        end
        return nothing
    end

    # Read before the shuffles below, which every lane passes before lane 1 writes the
    # type, so the value and the branch on it are the same for the whole segment
    @inbounds tile_type = interacting_tiles_type[idx]
    @inbounds coords_j = coords[(j - a) * Int32(32) + lane]
    i_0_tile = (i - a) * Int32(32)

    # The displacements are found from the image of atom j closest to the first atom of
    #   block i when that gives the minimum image of every pair within the cutoff, see
    #   `tile_single_image`
    @inbounds single_image, coords_j_image = tile_single_image(coords_j,
                    coords[i_0_tile + lane], coords[i_0_tile + a], boundary, Val(r_cut2),
                    Val(W))
    lane_mask = (single_image ?
        prune_lane_mask(coords, i_0_tile, coords_j, coords_j_image, lane, boundary,
                        Val(r_cut2), Val(true)) :
        prune_lane_mask(coords, i_0_tile, coords_j, coords_j_image, lane, boundary,
                        Val(r_cut2), Val(false)))

    # Butterfly OR-reduction, so that every lane gets the mask of the tile
    offset = Int32(TILE_WIDTH ÷ 2)
    while offset > Int32(0)
        lane_mask |= tile_shfl_xor(lane_mask, offset, Val(W))
        offset >>= Int32(1)
    end

    # Exact test for an exception between the blocks, from the atoms of block j
    has_exception = true
    if tile_type == UInt8(1) && lane_mask != UInt32(0)
        lane_has_exception = atom_has_partner_in_block(tile_exceptions,
                                                       (j - a) * Int32(32) + lane, i)
        has_exception = tile_any(lane_has_exception, Val(W))
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

# The force on atom j of a pair, without units, in the float type of the kernel
@inline stripped_force(f, ::Val{D}, ::Type{T}) where {D, T} =
    SVector{D, T}(ntuple(k -> T(ustrip(f[k])), Val(D)))

# Add the virial contribution of a pair, W_αβ = dr_α f_β with `f` the force on atom j and
#   `dr` the vector from atom i to atom j, matching the CPU path. All components are kept
#   since the tensor is not symmetric for non-central forces.
@inline function add_pair_virial(vir::SMatrix{D, D, T}, f::SVector{D, T}, dr) where {D, T}
    dr_nounits = SVector{D, T}(ntuple(k -> T(ustrip(dr[k])), Val(D)))
    return vir + dr_nounits * transpose(f)
end

# Add the components of `v` to column `idx` of `fs_mat` with atomics, unrolled so that `v`
#   stays in registers
@inline function atomic_add_columns!(fs_mat, v::SVector{D, T}, idx) where {D, T}
    ntuple(Val(D)) do k
        @inbounds if v[k] != zero(T)
            Atomix.@atomic :monotonic fs_mat[k, idx] += v[k]
        end
        nothing
    end
    return nothing
end

# Add the virial of a tile to its slot of the virial, column-major, see `virial_slot`
@inline function atomic_add_virial!(global_virial, vir::SMatrix{D, D, T}, idx,
                                    ::Type{TH}) where {D, T, TH}
    slot = virial_slot(idx, size(global_virial, 3))
    ntuple(Val(D * D)) do k
        @inbounds if vir[k] != zero(T)
            α = (k - 1) % D + 1
            β = (k - 1) ÷ D + 1
            Atomix.@atomic :monotonic global_virial[α, β, slot] += TH(vir[k])
        end
        nothing
    end
    return nothing
end

#=
The forces of the pairs of a full tile, from the iterations in `diag_mask` (see
`prune_interacting_tiles_kernel!`), added to the accumulators of atom i, the virial and
the force on the j atom of the lane, which are returned.
Only iterations marked as holding an in-range pair are walked. Bit b covers the pairs with
`slot - lane == b (mod 32)`, so it selects the same slot permutation as `m == b`. Testing
all 32 bits with a branch instead costs a loop header and a divergent branch per skipped
iteration. The mask is uniform across the segment, so all lanes run the same iterations.
The force on the j atom held by lane `slot` is calculated by lane `slot - m`, and passed to
it with a shuffle, so that the lanes do not update a shared accumulator. The order in which
the forces on a j atom are summed is the same as that of the iterations. The lanes are
those of a segment of a sub-group of width `W`, see `tile_lane`.
With `has_exceptions` the bitmasks of the tile give the excluded and special pairs, and
otherwise there are none. With `single_image` the displacements are found from
`coords_i_image`, see `tile_single_image`.
With `refresh_inner` the inner masks of the dynamic pruning are found too, from the pairs
within the squared inner cutoff `r_inner2`, see `InnerTileMasks`, and returned with the
accumulators.
=#
@inline function tile_pair_forces(::Val{refresh_inner}, ::Val{single_image}, force_i, vir,
                        fj, sh_stage, sh_vel, diag_mask, eligible_bitmask, special_bitmask,
                        ::Val{has_exceptions}, lane, segment, coords_i, coords_i_image,
                        atoms_i, vel_i, ::Type{A}, shuf_syms, ::Val{uses_vel}, inters_tuple,
                        boundary, step_n, ::Val{r_cut2}, r_inner2, ::Val{force_units},
                        ::Val{needs_vir}, ::Val{D}, ::Type{T},
                        ::Val{W}) where {refresh_inner, has_exceptions, single_image, A,
                                         uses_vel, r_cut2, force_units, needs_vir, D, T, W}
    a = Int32(1)
    w = Int32(TILE_WIDTH)
    inner_mask = UInt32(0)
    active = diag_mask
    @inbounds while active != UInt32(0)
        m = Int32(trailing_zeros(active))
        active &= active - UInt32(1)
        slot = ((lane - a + m) & Int32(31)) + a
        js = sh_stage[slot, segment]
        coords_j = js.coords
        atoms_j = rebuild_shuffled_atom(A, atoms_i, js.atom_payload, shuf_syms)
        vel_j = uses_vel ? sh_vel[slot, segment] : vel_i

        dr = (single_image ? coords_j - coords_i_image : vector(coords_i, coords_j, boundary))
        r2 = sum(abs2, dr)
        if has_exceptions
            excl = (eligible_bitmask >> (w - slot)) | (eligible_bitmask << slot)
            spec = (special_bitmask >> (w - slot)) | (special_bitmask << slot)
            eligible = (excl & 0x1) == true
            condition = eligible && r2 <= r_cut2
            special = (spec & 0x1) == true
        else
            eligible = true
            condition = r2 <= r_cut2
            special = false
        end
        if refresh_inner && tile_any(eligible && r2 <= r_inner2, Val(W))
            inner_mask |= UInt32(1) << m
        end

        if tile_any(condition, Val(W))
            f = condition ? sum_pairwise_forces_gpu(
                inters_tuple, dr, atoms_i, atoms_j, Val(force_units),
                special, coords_i, coords_j, boundary, vel_i, vel_j, step_n
            ) : zero_pairwise_force(dr, force_units)

            fi = stripped_force(f, Val(D), T)
            force_i += fi
            # Lane `src` evaluated the pair with the j atom of this lane
            src = ((lane - a - m) & Int32(31)) + a
            fj += tile_shfl(fi, src, Val(W))
            if needs_vir
                vir = add_pair_virial(vir, fi, dr)
            end
        end
    end
    return force_i, vir, fj, inner_mask
end

# The loop is compiled for each value of `refresh_inner` and `single_image`, which are
#   tested here once per tile rather than in every iteration: the compiler does not move the
#   tests out of loops with sub-group operations such as `tile_any` and `tile_shfl` itself.
#   Both have to be the same for the work-items of a segment.
@inline function tile_pair_forces(refresh_inner::Bool, single_image::Bool,
                                  args::Vararg{Any, N}) where N
    if single_image
        return refresh_inner ? tile_pair_forces(Val(true), Val(true), args...) :
                               tile_pair_forces(Val(false), Val(true), args...)
    else
        return refresh_inner ? tile_pair_forces(Val(true), Val(false), args...) :
                               tile_pair_forces(Val(false), Val(false), args...)
    end
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

#=
    force_kernel!(fs_mat, global_virial, coords, velocities, atoms, N, r_cut2, force_units,
                  inters_tuple, boundary, step_n, tile_exceptions, needs_vir, T, TH, D,
                  block_y, W, uses_vel, interacting_tiles_i, interacting_tiles_j,
                  interacting_tiles_type, interacting_tiles_diag, num_pairs,
                  interacting_tiles_overflow, inner)

Compute pairwise forces for the compact list of interacting 32x32 tiles produced
by `find_interacting_blocks_kernel!`. This is a KernelInterface kernel, launched with
1-D work-groups of `32 * block_y` work-items.

Execution model:
- One segment of 32 work-items of a sub-group of width `W` processes one tile of the
  compact list, so a work-group processes `block_y` tiles at once, see `tile_lane`. The
  lane `tile_lane(Val(W))` spans the atoms of a tile row, and the segment
  `group_tile(Val(W))` selects the tile.
- The kernel keeps the `i`-atom contribution in registers and passes the `j`-atom
  contributions between lanes with shuffles, then atomically adds both to `fs_mat`.

Tile cases:
1. Full off-diagonal tiles.
2. Boundary-column tiles containing the final partial atom block.
3. Diagonal tiles, where only unique pairs are evaluated.
4. The terminal corner tile, where both axes are partial.

CLEAN tiles skip the bitmasks entirely; mask-backed tiles build them from the
sparse exception lists in `tile_exceptions` to apply exclusions and special-pair
handling.

`inner` is `nothing` or the inner masks of the dynamic pruning, see `InnerTileMasks`,
which are used for the full tiles, or found again from the masks of the list when they are
stale.

The lanes of a segment are not assumed to execute in lockstep: local memory is only
written before a `tile_barrier` and read after it, and the forces on the j atoms are passed
with shuffles in control flow that is uniform over the segment.
=#
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
    ::Val{W},
    ::Val{uses_vel},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
    interacting_tiles_diag, num_pairs,
    interacting_tiles_overflow, inner) where {N, r_cut2, A, force_units, needs_vir, T, TH, D,
                                              block_y, W, uses_vel}

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
    interacting_tiles_overflow_ro = KernelAbstractions.constify(interacting_tiles_overflow)

    lane = tile_lane(Val(W))
    segment = group_tile(Val(W))
    idx = (KI.get_group_id(Int32).x - a) * Int32(block_y) + segment

    # Local memory for the staged j-atom data (coords/atoms/velocities) of the full tiles.
    # The inner loop indexes the staged data by slot instead of rotating it around the
    # segment with serial shuffles.
    # Only the Atom fields the active interactions actually read are staged (not the
    # full Atom).
    shuf_syms = resolved_atom_shuffle_syms(inters_tuple, A)
    P = atom_payload_type(A, shuf_syms)
    JT = JStage{eltype(coords_var), P}
    sh_stage = KI.localmemory(JT, Val((TILE_WIDTH, block_y)))
    sh_vel = uses_vel ? KI.localmemory(eltype(velocities_var), Val((TILE_WIDTH, block_y))) :
                        nothing

    if idx > num_pairs
        return nothing
    end

    # Loaded together, so that the tile waits for memory once instead of once per check.
    @inbounds overflow = interacting_tiles_overflow_ro[1]
    @inbounds i = tiles_i_ro[idx]
    @inbounds j = tiles_j_ro[idx]
    @inbounds type = tiles_type_ro[idx]
    @inbounds diag_mask = tiles_diag_ro[idx]
    if overflow != 0
        return nothing
    end

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

    i_0_tile = (i - a) * w
    index_i = i_0_tile + lane

    r = Int32((N - 1) % TILE_WIDTH + 1)

    force_i = zero(SVector{D, T})
    vir = zero(SMatrix{D, D, T})

    j_0_tile = (j - a) * w
    index_j = j_0_tile + lane

    # Part 1: Standard non-diagonal tiles
    if j < n_blocks && i < j
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]

        # Stage this tile's j-atom data into local memory once, then index it by slot
        # each iteration
        @inbounds coords_j_lane = coords[index_j]
        @inbounds sh_stage[lane, segment] = JStage(coords_j_lane,
                                    atom_shuffle_payload(atoms[index_j], shuf_syms))
        if uses_vel
            @inbounds sh_vel[lane, segment] = velocities[index_j]
        end
        tile_barrier(Val(W))

        # The displacements are found from the image of atom i closest to the first atom
        #   of block j when that gives the minimum image of every pair within the cutoff,
        #   or the inner cutoff when the inner masks are found, see `tile_single_image`
        single_image, coords_i_image = tile_single_image(coords_i, coords_j_lane,
                                    @inbounds(sh_stage[a, segment].coords), boundary,
                                    single_image_cutoff(inner, refresh_inner, Val(r_cut2)),
                                    Val(W))

        fj = zero(SVector{D, T})
        r_inner2 = inner_sq_cutoff(inner, Val(r_cut2))
        # Separate loops for CLEAN and EXCLUDED tiles, so that the loop over CLEAN tiles
        #   does not handle the masks
        if type == UInt8(0) # CLEAN
            force_i, vir, fj, inner_mask = tile_pair_forces(refresh_inner, single_image,
                        force_i, vir, fj, sh_stage, sh_vel, diag_mask, UInt32(0), UInt32(0),
                        Val(false), lane, segment, coords_i, coords_i_image, atoms_i, vel_i,
                        A, shuf_syms, Val(uses_vel), inters_tuple, boundary, step_n,
                        Val(r_cut2), r_inner2, Val(force_units), Val(needs_vir), Val(D), T,
                        Val(W))
        else # EXCLUDED
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))
            force_i, vir, fj, inner_mask = tile_pair_forces(refresh_inner, single_image,
                        force_i, vir, fj, sh_stage, sh_vel, diag_mask, eligible_bitmask,
                        special_bitmask, Val(true), lane, segment, coords_i, coords_i_image,
                        atoms_i, vel_i, A, shuf_syms, Val(uses_vel), inters_tuple, boundary,
                        step_n, Val(r_cut2), r_inner2, Val(force_units), Val(needs_vir),
                        Val(D), T, Val(W))
        end
        if refresh_inner && lane == a
            store_inner_mask!(inner, idx, inner_mask)
        end

        atomic_add_columns!(fs_mat, fj, index_j)
    end

    # Part 2: Boundary column tiles
    if j == n_blocks && i < n_blocks
        @inbounds coords_i = coords[index_i]
        @inbounds vel_i = velocities[index_i]
        @inbounds atoms_i = atoms[index_i]

        eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                    lane, n_blocks, r, Val(N))

        # The trip count is uniform across the segment
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

            if tile_any(condition, Val(W))
                f = condition ? sum_pairwise_forces_gpu(
                    inters_tuple, dr, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : zero_pairwise_force(dr, force_units)

                fi = stripped_force(f, Val(D), T)
                force_i += fi
                # The force on atom j from the whole segment, added once
                fj = tile_sum(fi, Val(W))
                if lane == a
                    atomic_add_columns!(fs_mat, fj, idx_j)
                end

                if needs_vir
                    vir = add_pair_virial(vir, fi, dr)
                end
            end
        end
    end

    # Part 3: Diagonal tiles, and Part 4: the terminal corner tile
    # Lane `lane` evaluates the pairs with the atoms `m > lane` of the tile, up to `r` for
    #   the terminal corner tile. The loop runs the same 31 iterations on all lanes, so
    #   that the force on atom `m` can be passed to lane `m` with a shuffle: at iteration
    #   `t`, lane `m` receives it from lane `m - t`.
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

        fj = zero(SVector{D, T})
        @inbounds for t in a:(w - a)
            m = lane + t
            fi = zero(SVector{D, T})
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

                if needs_vir
                    vir = add_pair_virial(vir, fi, dr)
                end
            end
            # Lanes `lane <= t` have no lane `lane - t`, they get the value of another lane
            #   which is not used
            f_recv = tile_shfl(fi, ((lane - a - t) & Int32(31)) + a, Val(W))
            if lane > t
                fj += f_recv
            end
        end

        force_i -= fj
    end

    if needs_vir
        vir = tile_sum(vir, Val(W))
        if lane == a
            atomic_add_virial!(global_virial, vir, idx, TH)
        end
    end

    if index_i <= N
        atomic_add_columns!(fs_mat, -force_i, index_i)
    end

    return nothing
end

# The energy of the pairs of a full tile, see `tile_pair_forces`
@inline function tile_pair_energy(sum_E, sh_stage, sh_vel, diag_mask, eligible_bitmask,
                        special_bitmask, ::Val{has_exceptions}, lane, segment, coords_i,
                        atoms_i, vel_i, ::Type{A}, shuf_syms, ::Val{uses_vel}, inters_tuple,
                        boundary, step_n, ::Val{r_cut2}, ::Val{energy_units},
                        ::Val{W}) where {has_exceptions, A, uses_vel, r_cut2, energy_units, W}
    a = Int32(1)
    w = Int32(TILE_WIDTH)
    active = diag_mask
    @inbounds while active != UInt32(0)
        m = Int32(trailing_zeros(active))
        active &= active - UInt32(1)
        slot = ((lane - a + m) & Int32(31)) + a
        js = sh_stage[slot, segment]
        coords_j = js.coords
        atoms_j = rebuild_shuffled_atom(A, atoms_i, js.atom_payload, shuf_syms)
        vel_j = uses_vel ? sh_vel[slot, segment] : vel_i

        dr = vector(coords_i, coords_j, boundary)
        r2 = sum(abs2, dr)
        if has_exceptions
            excl = (eligible_bitmask >> (w - slot)) | (eligible_bitmask << slot)
            spec = (special_bitmask >> (w - slot)) | (special_bitmask << slot)
            condition = (excl & 0x1) == true && r2 <= r_cut2
            special = (spec & 0x1) == true
        else
            condition = r2 <= r_cut2
            special = false
        end

        if tile_any(condition, Val(W))
            pe = condition ? sum_pairwise_potentials_gpu(
                inters_tuple, dr, atoms_i, atoms_j, Val(energy_units), special,
                coords_i, coords_j, boundary, vel_i, vel_j, step_n,
            ) : SVector(zero_pairwise_energy(dr, energy_units))

            sum_E += convert(typeof(sum_E), ustrip(pe[1]))
        end
    end
    return sum_E
end

#=
    energy_kernel!(energy_nounits, coords, velocities, atoms, N, r_cut2, energy_units,
                   inters_tuple, boundary, step_n, tile_exceptions, T, TH, D, block_y, W,
                   uses_vel, interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
                   interacting_tiles_diag, num_pairs, interacting_tiles_overflow,
                   inner)

Compute pairwise potential energies for the compact list of interacting 32x32
tiles produced by `find_interacting_blocks_kernel!`.

This mirrors `force_kernel!` structurally: the same compact tile list, the same
four tile-shape cases, and the same CLEAN-vs-mask-backed fast path. The main
difference is the reduction target: each segment sums its energy with shuffles
before the final atomic add to the slot of its tile in `energy_nounits`, see
`energy_slot`.

`inner` is `nothing` or the inner masks of the dynamic pruning, see `InnerTileMasks`,
which are used for the full tiles when they are current. Unlike `force_kernel!` the
masks are not found again when they are stale, the masks of the list are used instead.
=#
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
    ::Val{W},
    ::Val{uses_vel},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
    interacting_tiles_diag, num_pairs,
    interacting_tiles_overflow, inner) where {N, r_cut2, A, energy_units, T, TH, D, block_y,
                                              W, uses_vel}

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
    interacting_tiles_overflow_ro = KernelAbstractions.constify(interacting_tiles_overflow)

    lane = tile_lane(Val(W))
    segment = group_tile(Val(W))
    idx = (KI.get_group_id(Int32).x - a) * Int32(block_y) + segment

    # Local memory for staging this tile's j-atom data, mirroring force_kernel!'s Part 1
    shuf_syms = resolved_atom_shuffle_syms(inters_tuple, A)
    P = atom_payload_type(A, shuf_syms)
    JT = JStage{eltype(coords_var), P}
    sh_stage = KI.localmemory(JT, Val((TILE_WIDTH, block_y)))
    sh_vel = uses_vel ? KI.localmemory(eltype(velocities_var), Val((TILE_WIDTH, block_y))) :
                        nothing

    if idx > num_pairs
        return nothing
    end

    @inbounds overflow = interacting_tiles_overflow_ro[1]
    @inbounds i = tiles_i_ro[idx]
    @inbounds j = tiles_j_ro[idx]
    @inbounds type = tiles_type_ro[idx]
    @inbounds diag_mask = tiles_diag_ro[idx]
    if overflow != 0
        return nothing
    end

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
        @inbounds sh_stage[lane, segment] = JStage(coords[index_j],
                                    atom_shuffle_payload(atoms[index_j], shuf_syms))
        if uses_vel
            @inbounds sh_vel[lane, segment] = velocities[index_j]
        end
        tile_barrier(Val(W))

        if type == UInt8(0) # CLEAN
            sum_E = tile_pair_energy(sum_E, sh_stage, sh_vel, diag_mask, typemax(UInt32),
                        zero(UInt32), Val(false), lane, segment, coords_i, atoms_i, vel_i, A,
                        shuf_syms, Val(uses_vel), inters_tuple, boundary, step_n,
                        Val(r_cut2), Val(energy_units), Val(W))
        else # EXCLUDED
            eligible_bitmask, special_bitmask = tile_exception_masks(tile_exceptions, i, j,
                                                        lane, n_blocks, r, Val(N))
            sum_E = tile_pair_energy(sum_E, sh_stage, sh_vel, diag_mask, eligible_bitmask,
                        special_bitmask, Val(true), lane, segment, coords_i, atoms_i, vel_i,
                        A, shuf_syms, Val(uses_vel), inters_tuple, boundary, step_n,
                        Val(r_cut2), Val(energy_units), Val(W))
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
        # No lane passes values to another lane here, so the lanes loop separately
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

    # Segment reduction, every lane gets the sum
    sum_E = tile_sum(sum_E, Val(W))

    if lane == a && sum_E != zero(T)
        Atomix.@atomic :monotonic energy_nounits[energy_slot(idx, length(energy_nounits))] += TH(sum_E)
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
It is a KernelInterface kernel, launched with `n_blocks_i * n_blocks_j` 1-D work-groups,
with a segment of a sub-group of width `W` per tile, see `tile_lane`.
=#
function pairwise_force_kernel_nonl!(forces::AbstractArray{T}, coords_var, velocities_var,
                                     atoms_var, boundary, inters, step_n, ::Val{D}, ::Val{F},
                                     ::Val{n_blocks_i},
                                     ::Val{W}) where {T, D, F, n_blocks_i, W}
    coords = KernelAbstractions.constify(coords_var)
    velocities = KernelAbstractions.constify(velocities_var)
    atoms = KernelAbstractions.constify(atoms_var)
    n_atoms = Int32(length(atoms))
    w = Int32(TILE_WIDTH)

    group = KI.get_group_id(Int32).x - Int32(1)
    block_i = group % Int32(n_blocks_i)
    block_j = group ÷ Int32(n_blocks_i)
    lane = tile_lane(Val(W))
    segment = group_tile(Val(W))
    i_0_tile = block_i * w
    j_0_block = block_j * KI.get_local_size(Int32).x
    j_0_tile = j_0_block + (segment - Int32(1)) * w
    i = i_0_tile + lane

    force_i = zero(SVector{D, T})

    # Uniform across the segment
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

            atomic_add_columns!(forces, force_i, i)
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
            j = tile_shfl(j, src_lane, Val(W))
            coord_j = tile_shfl(coord_j, src_lane, Val(W))
            vel_j = tile_shfl(vel_j, src_lane, Val(W))
        end

        atomic_add_columns!(forces, force_i, i)
    end

    return nothing
end

#=
    nonl_pairwise_forces!(buffers, sys, pairwise_inters, step_n)

Calculate the pairwise forces of the interactions that do not use a neighbor list with
the sub-group kernel `pairwise_force_kernel_nonl!`, without a virial.
=#
function nonl_pairwise_forces!(buffers, sys::System{D, <:Any, T}, pairwise_inters,
                               step_n) where {D, T}
    N = length(sys.atoms)
    backend = get_backend(sys.coords)
    threads_basic = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
    n_threads = min(N, threads_basic, KI.max_work_group_size(backend))
    n_threads = TILE_WIDTH * tiles_per_group(backend, cld(n_threads, TILE_WIDTH))
    n_blocks_i = cld(N, TILE_WIDTH)
    n_blocks_j = cld(N, n_threads)
    kernel_backend, options = tile_kernel_config(backend, T, nothing)
    launch_tile_kernel!(kernel_backend, pairwise_force_kernel_nonl!,
        buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters,
        step_n, Val(D), Val(sys.force_units), Val(n_blocks_i), tile_sub_group_width(backend);
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

# Also zeroes `next_cm_momentum`, the buffer that the next call accumulates into
@kernel inbounds=true function remove_cm_velocity_kernel!(velocities, @Const(cm_momentum),
                        next_cm_momentum, @Const(virtual_site_flags), total_mass,
                        ::Val{momentum_unit}, ::Val{has_vs},
                        ::Val{D}) where {momentum_unit, has_vs, D}
    i = @index(Global, Linear)
    if i <= D
        next_cm_momentum[i] = zero(eltype(next_cm_momentum))
    end
    if i <= length(velocities)
        cm_velocity = SVector{D}(ntuple(k -> cm_momentum[k] * momentum_unit / total_mass,
                                        Val(D)))
        v = velocities[i]
        if has_vs && virtual_site_flags[i]
            velocities[i] = zero(v)
        else
            velocities[i] = v - cm_velocity
        end
    end
end

#=
Two buffers for the center of mass momentum, used in turn by `remove_CM_motion!` so that it
does not allocate and zero one each time it is called, which is every step by default.
The kernel that removes the center of mass velocity zeroes the buffer that the next call
accumulates into. They are kept per task, since each task has its own queue.
=#
mutable struct CMMomentumScratch{A}
    buffers::NTuple{2, A}
    current::Int
    device::Int
end

function cm_momentum_scratch(backend, ::Type{T}, ::Val{D}) where {T, D}
    tls = task_local_storage()
    key = (:molly_cm_momentum_scratch, backend, T, D)
    scratch = get(tls, key, nothing)
    device = KI.device(backend)
    A = typeof(KernelAbstractions.zeros(backend, T, 0))
    if !(scratch isa CMMomentumScratch{A}) || scratch.device != device
        buffers = (KernelAbstractions.zeros(backend, T, D),
                   KernelAbstractions.zeros(backend, T, D))
        scratch = CMMomentumScratch{A}(buffers, 1, device)
        tls[key] = scratch
    end
    return scratch::CMMomentumScratch{A}
end

# The kernels need atomics, otherwise the reduction of GPUArrays is used
function remove_CM_motion!(sys::System{D, <:AbstractGPUArray, T}) where {D, T}
    backend = get_backend(sys.velocities)
    if !KI.supports_atomics(backend)
        remove_CM_motion!(sys.velocities, sys)
        return sys
    end
    M = unit(zero(eltype(eltype(sys.velocities))) * zero(sys.total_mass))
    scratch = cm_momentum_scratch(backend, T, Val(D))
    cm_momentum = scratch.buffers[scratch.current]
    next_cm_momentum = scratch.buffers[3 - scratch.current]
    scratch.current = 3 - scratch.current
    n_threads = 256
    launch_kernel!(cm_momentum_kernel!(backend, n_threads), cm_momentum, sys.velocities,
                   masses(sys), Val(D); ndrange=length(sys))
    # At least D work-items, which zero the next buffer
    launch_kernel!(remove_cm_velocity_kernel!(backend, n_threads), sys.velocities,
                   cm_momentum, next_cm_momentum, sys.virtual_site_flags, sys.total_mass,
                   Val(M), Val(!isempty(sys.virtual_sites)), Val(D);
                   ndrange=max(length(sys), D))
    return sys
end
