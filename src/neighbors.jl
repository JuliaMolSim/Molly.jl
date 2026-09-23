# Neighbor finders

export
    use_neighbors,
    NoNeighborFinder,
    find_neighbors,
    GPUNeighborFinder,
    GPUCellListNeighborFinder,
    DistanceNeighborFinder,
    TreeNeighborFinder,
    CellListMapNeighborFinder

"""
    use_neighbors(inter)

Whether a pairwise interaction uses the neighbor list, default `false`.

Custom pairwise interactions can define a method for this function.
For built-in interactions such as [`LennardJones`](@ref) this function accesses
the `use_neighbors` field of the struct.
"""
use_neighbors(inter) = false

function check_neighbor_matrices(eligible, special)
    if !isnothing(eligible) && !isnothing(special) && size(eligible) != size(special)
        throw(ArgumentError("size of the eligible matrix $(size(eligible)) must be " *
                            "the same as the size of the special matrix $(size(special))"))
    end
    if size(eligible, 1) != size(eligible, 2)
        throw(ArgumentError("eligible and special matrices must be square, " *
                            "found size $(size(eligible))"))
    end
    if !isnothing(eligible) && !issymmetric(eligible)
        throw(ArgumentError("eligible matrix is not symmetric"))
    end
    if !isnothing(special) && !issymmetric(special)
        throw(ArgumentError("special matrix is not symmetric"))
    end
end

"""
    NoNeighborFinder()

Placeholder neighbor finder that returns no neighbors.

When using this neighbor finder, ensure that [`use_neighbors`](@ref) for the interactions
returns `false`.
"""
struct NoNeighborFinder end

"""
    find_neighbors(system; n_threads=Threads.nthreads())
    find_neighbors(system, neighbor_finder, current_neighbors=nothing, step_n=0,
                   force_recompute=false; n_threads=Threads.nthreads())

Obtain a list of close atoms in a [`System`](@ref).

Custom neighbor finders should implement this function.

Returns a [`NeighborList`](@ref) for the classical neighbor finders and a
[`GPUCellListNeighborList`](@ref) for [`GPUCellListNeighborFinder`](@ref).

For [`GPUNeighborFinder`](@ref), this returns `nothing`: the CUDA pairwise force
and energy kernels build and cache their interacting tile list internally from
the neighbor-finder metadata.

A neighbor finder may reuse the device buffers behind `current_neighbors` for the
list it returns, as [`GPUCellListNeighborFinder`](@ref) does, so the list passed in
should not be used afterwards.
"""
find_neighbors(sys::System; kwargs...) = find_neighbors(sys, sys.neighbor_finder; kwargs...)

find_neighbors(sys::System, nf::NoNeighborFinder, args...; kwargs...) = nothing

#=
    uses_gpu_neighbor_finder(AT)

Indicate whether an array type `AT` is compatible with [`GPUNeighborFinder`](@ref).

Custom GPU array types should define a method for this function that returns `true`
if they are supported. By default, this returns `false`.
=#
uses_gpu_neighbor_finder(AT) = false

"""
    GPUNeighborFinder(; n_atoms, dist_cutoff,
                      excluded_pairs=(), special_pairs=(), n_steps_reorder=10,
                      initialized=false, device_vector_type)
    GPUNeighborFinder(; eligible, dist_cutoff,
                      special=nothing, n_steps_reorder=10,
                      initialized=false, device_vector_type=nothing)

Neighbor finder for CUDA systems that uses Molly's tiled pairwise kernels.

`GPUNeighborFinder` does not materialize a conventional per-atom neighbor list.
Instead, the CUDA pairwise force and energy paths reorder atoms on a Morton
curve, convert sparse exclusions and special pairs into per-tile bitmasks, and
build a compact list of interacting 32x32 tiles directly on the device. For
that reason, [`find_neighbors`](@ref) returns `nothing` for this neighbor
finder.

This is the recommended neighbor finder for `CuArray` systems. On other GPUs use
[`GPUCellListNeighborFinder`](@ref).

# Keyword arguments
- `n_atoms`: number of atoms when constructing directly from sparse exception
  pairs.
- `eligible`, `special`: compatibility inputs for dense boolean masks. These
  are converted once at construction into sparse exception lists; the dense
  masks are not retained.
- `dist_cutoff`: the neighbor search distance used by the pairwise kernels.
  This should be the interaction cutoff distance plus a buffer distance, since
  the tile list is only refreshed every `n_steps_reorder` steps. The buffer
  should be larger than the distance an atom can move in that time.
- `excluded_pairs`: iterable of `(i, j)` pairs that should be excluded from the
  normal nonbonded interaction path.
- `special_pairs`: iterable of `(i, j)` pairs that should use the "special"
  interaction path.
- `n_steps_reorder`: number of simulation steps between Morton reorder and
  tile-list refresh passes.
- `initialized`: whether the current sparse-mask preprocessing state can be
  reused.
- `device_vector_type`: concrete `AbstractVector{Int32}` storage type for
  sparse exception indices. It is required when constructing from `n_atoms`,
  and inferred from `eligible` when dense masks are provided.

# Notes
- Sparse exceptions are stored as four device vectors:
  `excluded_i`, `excluded_j`, `special_i`, and `special_j`.
- Updating the exception pairs should reset `initialized` so the tile masks are
  rebuilt on the next GPU force or energy evaluation.
"""
mutable struct GPUNeighborFinder{B, D, D2, E}
    n_atoms::B
    dist_cutoff::D
    dist_cutoff_2::D2
    n_steps_reorder::Int
    initialized::Bool
    cache_generation::UInt64
    excluded_i::E
    excluded_j::E
    special_i::E
    special_j::E
end

#=
    copy_to_bitmatrix(x)

Convert a given matrix `x` to a `BitMatrix`, copying if it is already one.

This is an internal utility used to ensure dense masks are efficiently handled
on the CPU before converting to sparse exceptions.
=#
copy_to_bitmatrix(x::BitMatrix) = copy(x)
copy_to_bitmatrix(x) = BitMatrix(Array(x))

#=
    to_bitmatrix(x)

Convert a given matrix `x` to a `BitMatrix`, returning it unchanged if it is already one.

Unlike `copy_to_bitmatrix` this does not copy, so the result should not be modified.
=#
to_bitmatrix(x::BitMatrix) = x
to_bitmatrix(x) = BitMatrix(Array(x))

#=
    gpu_exception_vector_type(eligible, device_vector_type)

Determine or validate the 1D `Int32` array type for storing sparse GPU exceptions.

This function ensures that the selected `device_vector_type` is compatible with the
provided `eligible` matrix (if on the GPU) or explicitly supplied.

# Arguments
- `eligible`: the dense boolean mask of eligible interactions.
- `device_vector_type`: an explicitly requested vector type or `nothing` to infer it.
=#
function gpu_exception_vector_type(eligible, device_vector_type)
    if eligible isa AbstractGPUArray
        return typeof(eligible).name.wrapper{Int32, 1}
    end
    return validate_device_vector_type(device_vector_type;
                                       missing_message="eligible must be on the GPU or device_vector_type must be provided")
end

function validate_device_vector_type(device_vector_type; missing_message::AbstractString="device_vector_type must be provided")
    if isnothing(device_vector_type)
        throw(ArgumentError(missing_message))
    end
    if !(device_vector_type isa Type && device_vector_type <: AbstractVector{Int32})
        throw(ArgumentError("device_vector_type must be a 1D Int32 array type, got $device_vector_type"))
    end
    return device_vector_type
end

#=
    normalize_pairs(pairs; allow_diagonal=false)

Normalize an iterable of pairs into a sorted, unique list of `(Int32, Int32)` tuples.

Each pair `(i, j)` is sorted such that `i < j` (unless `i == j`). If `allow_diagonal`
is `false`, any pairs where `i == j` are excluded. The resulting sequence contains
no duplicate pairs.

# Arguments
- `pairs`: an iterable of tuple pairs or 2-element arrays representing atom index pairs.
- `allow_diagonal::Bool=false`: whether to include `(i, i)` self-interactions.
=#
function normalize_pairs(pairs; allow_diagonal::Bool=false, n_atoms=nothing)
    normalized = Tuple{Int32, Int32}[]
    seen = Set{Tuple{Int32, Int32}}()
    n_atoms_32 = isnothing(n_atoms) ? nothing : Int32(n_atoms)
    for (i, j) in pairs
        i32 = Int32(i)
        j32 = Int32(j)
        if !isnothing(n_atoms_32) && !(Int32(1) <= i32 <= n_atoms_32 && Int32(1) <= j32 <= n_atoms_32)
            throw(ArgumentError("pair ($(Int(i32)), $(Int(j32))) is out of bounds for $n_atoms atoms"))
        end
        if j32 < i32
            i32, j32 = j32, i32
        end
        if i32 == j32 && !allow_diagonal
            continue
        end
        pair = (i32, j32)
        if pair ∉ seen
            push!(seen, pair)
            push!(normalized, pair)
        end
    end
    sort!(normalized)
    return normalized
end

#= 
    pair_list_vectors(pairs, ET)

Convert an iterable of pairs into two separate GPU device vectors of type `ET`.

Extracts the first and second elements of each pair into distinct arrays
and transfers them to the device using `Molly.to_device`.

# Arguments
- `pairs`: an iterable of `(i, j)` tuple pairs.
- `ET`: the target 1D `Int32` device vector type.
=#
function pair_list_vectors(pairs, ET)
    is = Int32[first(pair) for pair in pairs]
    js = Int32[last(pair) for pair in pairs]
    return Molly.to_device(is, ET), Molly.to_device(js, ET)
end

#=
    dense_masks_to_pair_lists(eligible_cpu, special_cpu)

Convert dense boolean matrices into sparse lists of `(i, j)` exclusion and special pairs.

This function identifies the entries where `eligible_cpu` is `false` to form excluded
pairs, and where `special_cpu` is `true` to form special interaction pairs. It only
retains the upper triangle indices (`i < j`).

# Arguments
- `eligible_cpu`: a dense boolean mask matrix indicating allowed standard interactions.
- `special_cpu`: a dense boolean mask matrix indicating special interactions.
=#
function dense_masks_to_pair_lists(eligible_cpu, special_cpu)
    all_exc = findall(.!eligible_cpu)
    excluded_pairs = Tuple{Int32, Int32}[]
    for idx in all_exc
        if idx[1] < idx[2]
            push!(excluded_pairs, (Int32(idx[1]), Int32(idx[2])))
        end
    end

    all_spec = findall(special_cpu)
    special_pairs = Tuple{Int32, Int32}[]
    for idx in all_spec
        if idx[1] < idx[2]
            push!(special_pairs, (Int32(idx[1]), Int32(idx[2])))
        end
    end
    return excluded_pairs, special_pairs
end

function neighbor_finder_masks(nf::GPUNeighborFinder)
    eligible = trues(nf.n_atoms, nf.n_atoms)
    special = falses(nf.n_atoms, nf.n_atoms)
    for i in 1:nf.n_atoms
        eligible[i, i] = false
    end
    for (i, j) in zip(from_device(nf.excluded_i), from_device(nf.excluded_j))
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, j) in zip(from_device(nf.special_i), from_device(nf.special_j))
        special[i, j] = true
        special[j, i] = true
    end
    return eligible, special
end

neighbor_finder_masks(nf, ::Integer) = neighbor_finder_masks(nf)

function neighbor_finder_masks(::NoNeighborFinder, n_atoms::Integer)
    eligible = trues(n_atoms, n_atoms)
    special = falses(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    return eligible, special
end

#=
    update_sparse_pairs!(nf, excluded_pairs, special_pairs)

Replace the existing exception lists in a [`GPUNeighborFinder`](@ref) with new ones.

This normalizes the `excluded_pairs` and `special_pairs`, uploads them to the
device, and resets the neighbor finder's `initialized` flag so that internal
GPU interaction masks are rebuilt.

# Arguments
- `nf::GPUNeighborFinder`: the neighbor finder instance to update.
- `excluded_pairs`: an iterable of `(i, j)` pairs that should be excluded.
- `special_pairs`: an iterable of `(i, j)` pairs that should use the special interaction path.
=#
function update_sparse_pairs!(nf::GPUNeighborFinder, excluded_pairs, special_pairs)
    ET = typeof(nf.excluded_i)
    excluded_pairs = normalize_pairs(excluded_pairs; n_atoms=nf.n_atoms)
    special_pairs = normalize_pairs(special_pairs; n_atoms=nf.n_atoms)
    nf.excluded_i, nf.excluded_j = pair_list_vectors(excluded_pairs, ET)
    nf.special_i, nf.special_j = pair_list_vectors(special_pairs, ET)
    nf.initialized = false
    nf.cache_generation += 0x0000000000000001
    return nf
end

#=
    append_excluded_pairs!(nf, pairs)

Append new excluded pairs to the existing exclusions in a [`GPUNeighborFinder`](@ref).

This downloads the current sparse lists from the device, concatenates them with
the new `pairs`, and then calls `update_sparse_pairs!` to normalize and upload
everything back to the GPU. The `initialized` state will be reset.

# Arguments
- `nf::GPUNeighborFinder`: the neighbor finder instance to update.
- `pairs`: an iterable of `(i, j)` pairs to add to the exclusions list.
=#
function append_excluded_pairs!(nf::GPUNeighborFinder, pairs)
    existing_pairs = collect(zip(from_device(nf.excluded_i), from_device(nf.excluded_j)))
    update_sparse_pairs!(nf, vcat(existing_pairs, collect(pairs)),
                         collect(zip(from_device(nf.special_i), from_device(nf.special_j))))
    return nf
end

function GPUNeighborFinder(;
                            n_atoms=nothing,
                            eligible=nothing,
                            dist_cutoff,
                            excluded_pairs=(),
                            special_pairs=(),
                            special=nothing,
                            n_steps_reorder=10,
                            initialized=false,
                            device_vector_type=nothing)
    if !isnothing(n_atoms)
        ET = validate_device_vector_type(device_vector_type)
        excluded_pairs_norm = normalize_pairs(excluded_pairs; n_atoms=n_atoms)
        special_pairs_norm = normalize_pairs(special_pairs; n_atoms=n_atoms)
        excluded_i, excluded_j = pair_list_vectors(excluded_pairs_norm, ET)
        special_i, special_j = pair_list_vectors(special_pairs_norm, ET)
        dist_cutoff_2 = dist_cutoff^2
        return GPUNeighborFinder{Int, typeof(dist_cutoff), typeof(dist_cutoff_2), typeof(excluded_i)}(
                    Int(n_atoms), dist_cutoff, dist_cutoff_2, n_steps_reorder, initialized, 0,
                    excluded_i, excluded_j, special_i, special_j)
    end

    check_neighbor_matrices(eligible, special)
    isnothing(eligible) && throw(ArgumentError("either n_atoms or eligible must be provided"))
    ET = gpu_exception_vector_type(eligible, device_vector_type)
    if isnothing(special)
        special = zero(eligible)
    end
    eligible_cpu = copy_to_bitmatrix(eligible)
    special_cpu = copy_to_bitmatrix(special)
    if !(size(eligible_cpu) == size(special_cpu))
        throw(ArgumentError("eligible and special must have the same size"))
    end
    excluded_pairs_cpu, special_pairs_cpu = dense_masks_to_pair_lists(eligible_cpu, special_cpu)
    return GPUNeighborFinder(
        n_atoms=size(eligible_cpu, 1),
        eligible=nothing,
        dist_cutoff=dist_cutoff,
        excluded_pairs=excluded_pairs_cpu,
        special_pairs=special_pairs_cpu,
        n_steps_reorder=n_steps_reorder,
        initialized=initialized,
        device_vector_type=ET,
    )
end

# The interacting tile list is constructed within the CUDA pairwise kernels.
find_neighbors(sys::System, nf::GPUNeighborFinder, args...; kwargs...) = nothing

# Mark neighbor data cached in `buffers` as stale so that it is rebuilt on the next force
#   or energy evaluation
# Neighbor finders that return a neighbor list from `find_neighbors` do not cache
#   anything in the buffers, so this does nothing for them
invalidate_cached_neighbors!(buffers, neighbor_finder) = buffers

function invalidate_cached_neighbors!(buffers::BuffersGPU, nf::GPUNeighborFinder)
    buffers.step_n_preprocessed = -1
    return buffers
end

"""
    GPUCellListNeighborFinder(; dist_cutoff, n_steps=10, max_neighbors=nothing, output=:molly_pairs,
                              eligible=nothing, special=nothing, n_atoms=nothing, excluded_pairs=(),
                              special_pairs=(), device_vector_type=nothing)

Neighbor finder for GPU systems that bins atoms into a uniform grid of cells and
searches the 3x3x3 stencil of cells around each atom.

Returns a [`GPUCellListNeighborList`](@ref). It runs on any GPU backend and is
the recommended neighbor finder for non-CUDA GPUs, where [`GPUNeighborFinder`](@ref)
is not available. Three-dimensional [`CubicBoundary`](@ref) and
[`TriclinicBoundary`](@ref) systems are supported, provided the grid has at least
three cells along every box axis, i.e. opposite box faces are at least three times
`dist_cutoff` apart. For a [`TriclinicBoundary`](@ref) that distance is smaller than
the length of the corresponding basis vector.

`dist_cutoff` is the neighbor search distance, which should be the interaction cutoff
distance plus a buffer distance since the list is only updated every `n_steps` steps.

Like [`GPUNeighborFinder`](@ref), dense `eligible` and `special` masks are converted
once at construction into sparse per-atom exception lists and are not retained.
The exceptions can also be given directly as `excluded_pairs` and
`special_pairs` alongside `n_atoms` and `device_vector_type`, which avoids building
the dense masks at all.

# Output modes
- `:molly_pairs`: half-pair list with excluded pairs removed and the special flag set
  for special pairs. This is the only mode that can be used for the pairwise
  interactions of a [`System`](@ref), since it is the only one that applies
  exclusions.
- `:geometric_pairs`: half-pair list of every pair within `dist_cutoff`, with all
  special flags false. The exceptions are ignored, so pairs that a bonded topology
  excludes are still returned and using this mode for a system with bonded exclusions
  gives wrong forces.
- `:ragged`: per-atom padded neighbor matrix only, with no flat pair list. The
  exceptions are ignored. This mode is for querying neighbors directly, it can not be
  used for the pairwise interactions of a [`System`](@ref) but can be useful for
  machine learning potentials.

`max_neighbors` is the initial per-atom capacity of the neighbor matrix. When it is
`nothing` the capacity is estimated from the global number density and `dist_cutoff`
with a safety factor of 1.5, rounded up to a multiple of 32. An explicit positive
integer overrides this estimate. Either way the capacity is grown automatically, and
kept for later calls, if an atom turns out to have more neighbors than it can hold.

The device buffers behind the returned list are owned by that list and are reused,
and hence overwritten, when it is passed back in as `current_neighbors`. A list that
is passed to [`find_neighbors`](@ref) should therefore be treated as consumed, since
any other reference to it, for example one kept by a simulator to compare against a
trial list, would see the new neighbors with a stale pair count.
"""
mutable struct GPUCellListNeighborFinder{D, E}
    dist_cutoff::D
    n_steps::Int
    max_neighbors::Union{Nothing, Int}
    output::Symbol
    n_atoms::Int
    # Compressed per-atom exception lists, see atom_pair_csr
    excluded_starts::E
    excluded_js::E
    special_starts::E
    special_js::E
end

#=
Compress a list of `(i, j)` pairs with `i < j` into per-atom lists.

Only pairs with `atom_i > atom_j` are looked at by the pair kernels, so each pair is
stored once under the larger of its two indices. `starts[atom_i]` to
`starts[atom_i + 1] - 1` is the range of `js` holding the partners of `atom_i`. These
lists are a few entries per atom, so they stay in cache where the dense masks they
replace do not.
=#
function atom_pair_csr(pairs, n_atoms::Integer, ET)
    starts = ones(Int32, n_atoms + 1)
    for (_, j) in pairs
        starts[j + 1] += Int32(1)
    end
    for atom_i in 1:n_atoms
        starts[atom_i + 1] += starts[atom_i] - Int32(1)
    end

    js = Vector{Int32}(undef, length(pairs))
    next = copy(starts)
    for (i, j) in pairs
        js[next[j]] = i
        next[j] += Int32(1)
    end

    return to_device(starts, ET), to_device(js, ET)
end

# Rebuild the exception lists of a neighbor finder from pairs of atom indices
function update_sparse_pairs!(nf::GPUCellListNeighborFinder, excluded_pairs, special_pairs)
    ET = typeof(nf.excluded_js)
    nf.excluded_starts, nf.excluded_js = atom_pair_csr(
                normalize_pairs(excluded_pairs; n_atoms=nf.n_atoms), nf.n_atoms, ET)
    nf.special_starts, nf.special_js = atom_pair_csr(
                normalize_pairs(special_pairs; n_atoms=nf.n_atoms), nf.n_atoms, ET)
    return nf
end

# The pairs of atom indices behind the compressed lists of a neighbor finder
function sparse_pairs(starts, js)
    starts_cpu, js_cpu = from_device(starts), from_device(js)
    return [(js_cpu[k], Int32(atom_i))
            for atom_i in 1:(length(starts_cpu) - 1)
            for k in starts_cpu[atom_i]:(starts_cpu[atom_i + 1] - Int32(1))]
end

function append_excluded_pairs!(nf::GPUCellListNeighborFinder, pairs)
    update_sparse_pairs!(nf, vcat(sparse_pairs(nf.excluded_starts, nf.excluded_js),
                                  collect(pairs)),
                         sparse_pairs(nf.special_starts, nf.special_js))
    return nf
end

function GPUCellListNeighborFinder(;
    dist_cutoff,
    n_steps=10,
    max_neighbors=nothing,
    output::Symbol=:molly_pairs,
    eligible=nothing,
    special=nothing,
    n_atoms=nothing,
    excluded_pairs=(),
    special_pairs=(),
    device_vector_type=nothing,
)
    n_steps_int = Int(n_steps)
    max_neighbors_int = isnothing(max_neighbors) ? nothing : Int(max_neighbors)

    dist_cutoff > zero(dist_cutoff) || throw(
        ArgumentError("dist_cutoff must be positive, got $dist_cutoff"),
    )

    isfinite(ustrip(dist_cutoff)) || throw(
        ArgumentError("dist_cutoff must be finite, got $dist_cutoff"),
    )

    n_steps_int > 0 || throw(
        ArgumentError("n_steps must be positive, got $n_steps"),
    )

    if !isnothing(max_neighbors_int)
        max_neighbors_int > 0 || throw(
            ArgumentError(
                "max_neighbors must be positive, got $max_neighbors",
            ),
        )
    end

    output in (:ragged, :geometric_pairs, :molly_pairs) || throw(
        ArgumentError(
            "output must be :ragged, :geometric_pairs or :molly_pairs, " *
            "got $output",
        ),
    )

    if output !== :molly_pairs
        # The other modes ignore the exceptions, so nothing is stored for them
        return GPUCellListNeighborFinder{typeof(dist_cutoff), Nothing}(
            dist_cutoff, n_steps_int, max_neighbors_int, output,
            (isnothing(n_atoms) ? 0 : Int(n_atoms)), nothing, nothing, nothing, nothing,
        )
    end

    if isnothing(n_atoms)
        if isnothing(eligible)
            throw(ArgumentError(":molly_pairs requires either eligible or n_atoms"))
        end
        isnothing(special) && throw(ArgumentError(":molly_pairs requires special"))
        check_neighbor_matrices(eligible, special)

        # The exception lists are read inside the pair kernels, so they have to end up
        #   on the device
        ET = gpu_exception_vector_type(eligible, device_vector_type)
        excluded_pairs, special_pairs = dense_masks_to_pair_lists(
                    copy_to_bitmatrix(eligible), copy_to_bitmatrix(special))

        return GPUCellListNeighborFinder(;
            dist_cutoff=dist_cutoff,
            n_steps=n_steps_int,
            max_neighbors=max_neighbors_int,
            output=output,
            n_atoms=size(eligible, 1),
            excluded_pairs=excluded_pairs,
            special_pairs=special_pairs,
            device_vector_type=ET,
        )
    end

    ET = validate_device_vector_type(device_vector_type)
    n_atoms_int = Int(n_atoms)
    excluded_starts, excluded_js = atom_pair_csr(
                normalize_pairs(excluded_pairs; n_atoms=n_atoms_int), n_atoms_int, ET)
    special_starts, special_js = atom_pair_csr(
                normalize_pairs(special_pairs; n_atoms=n_atoms_int), n_atoms_int, ET)

    return GPUCellListNeighborFinder{typeof(dist_cutoff), typeof(excluded_starts)}(
        dist_cutoff, n_steps_int, max_neighbors_int, output, n_atoms_int,
        excluded_starts, excluded_js, special_starts, special_js,
    )
end

#=
GPU cell-list neighbor finder.

Atoms are binned into a uniform grid of cells with side at least the neighbor cutoff,
so every neighbor of an atom lies in the 3x3x3 stencil of cells around its own cell.
One group handles one tile of up to `CELL_BLOCK_SIZE` atoms from a single cell and
walks the stencil, staging candidate atoms in local memory.

The device buffers live in a `GPUCellListState` that is attached to the returned
`GPUCellListNeighborList` and reused when that list is passed back in as
`current_neighbors`. Only the cell grid depends on the box, so a box change, for
example from a barostat, updates the grid in place and keeps the per-atom buffers.
=#

# Threads per group in the neighbor search kernel, also the number of candidate atoms
#   staged in local memory at a time
const CELL_BLOCK_SIZE = 32

# Slots in the device counter array, read back to the host once per rebuild
const COUNTER_HOST_TILES = Int32(1)      # number of scheduled (cell, tile) pairs
const COUNTER_N_OVERFLOW = Int32(2)      # number of atoms with too many neighbors
const COUNTER_N_PAIRS = Int32(3)         # number of half pairs written
const N_CELL_LIST_COUNTERS = 3

gpu_threads_cell_list(n_items) = gpu_threads_env("MOLLY_GPUNTHREADS_CELLLIST", 256)

#=
Device buffers for the GPU cell list.

The per-atom and output buffers only depend on the number of atoms and the neighbor
capacity, the cell buffers only on the cell grid, so the struct is mutable and grown
in place rather than reallocated when the box changes.
=#
@kwdef mutable struct GPUCellListState{T, V, I, M, P}
    # Coordinates split into components and wrapped into the box
    x::V
    y::V
    z::V
    # Coordinates gathered into cell order, indexed like cell_particles
    cell_x::V
    cell_y::V
    cell_z::V
    cell_ids::I
    cell_particles::I
    neighbor_counts::I
    neighbors::M
    host_tile_cells::I
    host_tile_starts::I
    pair_counts::I
    pair_inclusive_counts::I
    pair_offsets::I
    pair_list::P
    cell_counts::I
    inclusive_counts::I
    cell_offsets::I
    cell_write_counts::I
    cell_tile_counts::I
    cell_tile_inclusive_counts::I
    cell_tile_offsets::I
    counters::I
    counters_host::Vector{Int32}
    n_atoms::Int32
    n_cells::Int32
    cell_capacity::Int
    max_host_tiles::Int
    n_host_tiles::Int
    max_neighbors::Int32
    pair_capacity::Int
    num_cell_x::Int32
    num_cell_y::Int32
    num_cell_z::Int32
    # Columns are the box basis vectors, so that a triclinic box is handled the same
    #   way as a cubic one, along with the inverse that gives fractional coordinates
    box::SMatrix{3, 3, T, 9}
    box_inv::SMatrix{3, 3, T, 9}
    cutoff2::T
end

#=
Allocate an uninitialized device buffer like `template`.

None of the cell-list buffers need to be zeroed when they are allocated: the ones that
are accumulated into are zeroed by `gpu_cell_list_reset_kernel!` at the start of every
rebuild, and the rest are fully written before they are read.
=#
cell_list_buffer(template, ::Type{T}, dims...) where {T} = similar(template, T, dims...)

function estimate_gpu_cell_list_max_neighbors(n_atoms, box_volume::T, cutoff::T) where {T}
    density = T(n_atoms) / box_volume
    expected_neighbors = T(1.5) * density * T(4π / 3) * cutoff^3
    return max(CELL_BLOCK_SIZE,
               cld(ceil(Int, expected_neighbors), CELL_BLOCK_SIZE) * CELL_BLOCK_SIZE)
end

#=
Distance between the opposite faces of the box along each basis vector.

This is the box side for a `CubicBoundary` and less than the side for a skewed
`TriclinicBoundary`. Cells have to be at least the cutoff wide by this measure for the
3x3x3 stencil to contain every neighbor.
=#
cell_list_box_widths(boundary::CubicBoundary{3}) = box_sides(boundary)

function cell_list_box_widths(boundary::TriclinicBoundary)
    bv = boundary.basis_vectors
    V = volume(boundary)
    return SVector(V / norm(cross(bv[2], bv[3])), V / norm(cross(bv[1], bv[3])),
                   V / norm(cross(bv[1], bv[2])))
end

# The box basis vectors as the columns of a matrix, stripped of units
function cell_list_box_matrix(boundary::CubicBoundary{3}, ::Type{T}, dist_unit) where {T}
    return SMatrix{3, 3, T}(Diagonal(T.(ustrip.(dist_unit, box_sides(boundary)))))
end

function cell_list_box_matrix(boundary::TriclinicBoundary, ::Type{T}, dist_unit) where {T}
    bv = boundary.basis_vectors
    return SMatrix{3, 3, T}(T.(ustrip.(dist_unit, hcat(bv[1], bv[2], bv[3]))))
end

# Cell grid for a box, with cells at least `cutoff` wide so that the 3x3x3 stencil
#   around a cell covers every neighbor exactly once
function gpu_cell_list_grid(widths::SVector{3, T}, cutoff::T) where {T}
    num_cell_x = floor(Int32, widths[1] / cutoff)
    num_cell_y = floor(Int32, widths[2] / cutoff)
    num_cell_z = floor(Int32, widths[3] / cutoff)

    minimum((num_cell_x, num_cell_y, num_cell_z)) >= 3 || throw(ArgumentError(
        "GPUCellListNeighborFinder requires at least three cells along every box " *
        "axis; the box is $(widths[1]) by $(widths[2]) by $(widths[3]) wide between " *
        "opposite faces and the cutoff is $cutoff"))

    # Widened before multiplying since the product can overflow Int32
    n_cells = Int(num_cell_x) * Int(num_cell_y) * Int(num_cell_z)

    n_cells <= typemax(Int32) || throw(ArgumentError(
        "GPUCellListNeighborFinder cell grid has $n_cells cells, which does not fit " *
        "in Int32; use a larger cutoff or a smaller box"))

    return num_cell_x, num_cell_y, num_cell_z, n_cells
end

# Upper bound on the number of (cell, tile) pairs, used to size the schedule and to
#   launch the search kernel without reading the exact count back to the host
function max_gpu_cell_list_host_tiles(n_atoms::Integer, n_cells::Integer)
    return min(Int(n_atoms), Int(n_cells)) + Int(n_atoms) ÷ CELL_BLOCK_SIZE
end

function gpu_cell_list_pair_capacity(n_atoms::Integer, max_neighbors::Integer)
    pair_capacity = cld(Int(n_atoms) * Int(max_neighbors), 2)

    pair_capacity <= typemax(Int32) || throw(ArgumentError(
        "GPU cell-list pair capacity $pair_capacity does not fit in Int32; reduce " *
        "max_neighbors or the number of atoms"))

    return pair_capacity
end

function allocate_gpu_cell_list_state(coords, ::Type{T}, box::SMatrix{3, 3, T},
                                      widths::SVector{3, T}, cutoff::T;
                                      max_neighbors=Int32(128),
                                      allocate_pairs=false) where {T}
    n_atoms = length(coords)
    num_cell_x, num_cell_y, num_cell_z, n_cells = gpu_cell_list_grid(widths, cutoff)
    max_host_tiles = max_gpu_cell_list_host_tiles(n_atoms, n_cells)
    pair_capacity = (allocate_pairs ? gpu_cell_list_pair_capacity(n_atoms, max_neighbors) : 0)
    pair_buffer() = cell_list_buffer(coords, Int32, allocate_pairs ? n_atoms : 0)

    state = GPUCellListState(;
        x=cell_list_buffer(coords, T, n_atoms),
        y=cell_list_buffer(coords, T, n_atoms),
        z=cell_list_buffer(coords, T, n_atoms),
        cell_x=cell_list_buffer(coords, T, n_atoms),
        cell_y=cell_list_buffer(coords, T, n_atoms),
        cell_z=cell_list_buffer(coords, T, n_atoms),
        cell_ids=cell_list_buffer(coords, Int32, n_atoms),
        cell_particles=cell_list_buffer(coords, Int32, n_atoms),
        neighbor_counts=cell_list_buffer(coords, Int32, n_atoms),
        neighbors=cell_list_buffer(coords, Int32, Int(max_neighbors), n_atoms),
        host_tile_cells=cell_list_buffer(coords, Int32, max_host_tiles),
        host_tile_starts=cell_list_buffer(coords, Int32, max_host_tiles),
        pair_counts=pair_buffer(),
        pair_inclusive_counts=pair_buffer(),
        pair_offsets=pair_buffer(),
        pair_list=(allocate_pairs ?
                   cell_list_buffer(coords, Tuple{Int32, Int32, Bool}, pair_capacity) :
                   nothing),
        cell_counts=cell_list_buffer(coords, Int32, n_cells),
        inclusive_counts=cell_list_buffer(coords, Int32, n_cells),
        cell_offsets=cell_list_buffer(coords, Int32, n_cells),
        cell_write_counts=cell_list_buffer(coords, Int32, n_cells),
        cell_tile_counts=cell_list_buffer(coords, Int32, n_cells),
        cell_tile_inclusive_counts=cell_list_buffer(coords, Int32, n_cells),
        cell_tile_offsets=cell_list_buffer(coords, Int32, n_cells),
        counters=cell_list_buffer(coords, Int32, N_CELL_LIST_COUNTERS),
        counters_host=zeros(Int32, N_CELL_LIST_COUNTERS),
        n_atoms=Int32(n_atoms),
        n_cells=Int32(n_cells),
        cell_capacity=n_cells,
        max_host_tiles=max_host_tiles,
        n_host_tiles=0,
        max_neighbors=Int32(max_neighbors),
        pair_capacity=pair_capacity,
        num_cell_x=num_cell_x,
        num_cell_y=num_cell_y,
        num_cell_z=num_cell_z,
        box=box,
        box_inv=inv(box),
        cutoff2=cutoff * cutoff,
    )

    split_gpu_cell_list_coordinates!(state, coords)

    return state
end

#=
Point an existing state at a new box, cutoff or neighbor capacity.

Only the cell buffers depend on the box, and only through the number of cells, so a
box change usually reuses every buffer. The per-atom and output buffers, which are by
far the largest, are never reallocated here.
=#
function update_gpu_cell_list_state!(state::GPUCellListState{T}, box::SMatrix{3, 3, T},
                                     widths::SVector{3, T}, cutoff::T,
                                     max_neighbors::Integer) where {T}
    num_cell_x, num_cell_y, num_cell_z, n_cells = gpu_cell_list_grid(widths, cutoff)

    if n_cells > state.cell_capacity
        state.cell_counts = cell_list_buffer(state.cell_counts, Int32, n_cells)
        state.inclusive_counts = cell_list_buffer(state.cell_counts, Int32, n_cells)
        state.cell_offsets = cell_list_buffer(state.cell_counts, Int32, n_cells)
        state.cell_write_counts = cell_list_buffer(state.cell_counts, Int32, n_cells)
        state.cell_tile_counts = cell_list_buffer(state.cell_counts, Int32, n_cells)
        state.cell_tile_inclusive_counts = cell_list_buffer(state.cell_counts, Int32, n_cells)
        state.cell_tile_offsets = cell_list_buffer(state.cell_counts, Int32, n_cells)
        state.cell_capacity = n_cells
    end

    max_host_tiles = max_gpu_cell_list_host_tiles(state.n_atoms, n_cells)

    if max_host_tiles > state.max_host_tiles
        state.host_tile_cells = cell_list_buffer(state.cell_counts, Int32, max_host_tiles)
        state.host_tile_starts = cell_list_buffer(state.cell_counts, Int32, max_host_tiles)
        state.max_host_tiles = max_host_tiles
    end

    state.n_cells = Int32(n_cells)
    state.num_cell_x = num_cell_x
    state.num_cell_y = num_cell_y
    state.num_cell_z = num_cell_z
    state.box = box
    state.box_inv = inv(box)
    state.cutoff2 = cutoff * cutoff

    # The capacity is only ever grown, since shrinking it would force a rebuild the
    #   next time the density goes back up
    if max_neighbors > state.max_neighbors
        grow_gpu_cell_list_neighbors!(state, max_neighbors)
    end

    return state
end

function grow_gpu_cell_list_neighbors!(state::GPUCellListState, max_neighbors::Integer)
    new_max = Int32(cld(Int(max_neighbors), CELL_BLOCK_SIZE) * CELL_BLOCK_SIZE)
    n_atoms = Int(state.n_atoms)

    state.neighbors = cell_list_buffer(state.neighbors, Int32, Int(new_max), n_atoms)
    state.max_neighbors = new_max

    if !isnothing(state.pair_list)
        pair_capacity = gpu_cell_list_pair_capacity(n_atoms, new_max)
        if pair_capacity > state.pair_capacity
            grow_gpu_cell_list_pairs!(state, pair_capacity)
        end
    end

    return state
end

function grow_gpu_cell_list_pairs!(state::GPUCellListState, pair_capacity::Integer)
    state.pair_list = cell_list_buffer(state.pair_list, Tuple{Int32, Int32, Bool},
                                       Int(pair_capacity))
    state.pair_capacity = Int(pair_capacity)
    return state
end

# Coordinates in units of the box basis vectors, which are in [0, 1) inside the box
@inline cell_list_fractional(box_inv::SMatrix{3, 3, T}, coord::SVector{3, T}) where {T} =
                        box_inv * coord

@kernel inbounds=true function gpu_cell_list_split_coords_kernel!(x, y, z, @Const(coords),
                                    n_atoms, box::SMatrix{3, 3, T},
                                    box_inv::SMatrix{3, 3, T}) where {T}
    atom_i = @index(Global, Linear) % Int32

    if atom_i <= n_atoms
        c = ustrip_vec(coords[atom_i])
        coord = SVector{3, T}(T(c[1]), T(c[2]), T(c[3]))
        # Coordinates are wrapped here rather than in the cell ID kernel so that atoms
        #   from an unwrapped structure land in the right cell. Subtracting whole box
        #   images leaves a coordinate that is already inside the box untouched
        wrapped = coord - box * floor.(cell_list_fractional(box_inv, coord))
        x[atom_i] = wrapped[1]
        y[atom_i] = wrapped[2]
        z[atom_i] = wrapped[3]
    end
end

function split_gpu_cell_list_coordinates!(state::GPUCellListState, coords)
    n_atoms = Int(state.n_atoms)
    backend = get_backend(coords)
    kernel! = gpu_cell_list_split_coords_kernel!(backend, gpu_threads_cell_list(n_atoms))
    kernel!(state.x, state.y, state.z, coords, state.n_atoms, state.box, state.box_inv;
            ndrange=n_atoms)
    return state
end

# Zero the buffers that the next rebuild accumulates into, in one launch
@kernel inbounds=true function gpu_cell_list_reset_kernel!(cell_counts, cell_write_counts,
                                                           counters, n_counters)
    cell = @index(Global, Linear) % Int32

    if cell <= length(cell_counts)
        cell_counts[cell] = Int32(0)
        cell_write_counts[cell] = Int32(0)
    end
    if cell <= n_counters
        counters[cell] = Int32(0)
    end
end

@kernel inbounds=true function gpu_cell_list_cell_ids_kernel!(cell_ids, cell_counts,
                                    @Const(x), @Const(y), @Const(z), n_atoms, num_cell_x,
                                    num_cell_y, num_cell_z, box_inv::SMatrix{3, 3, T}) where {T}
    atom_i = @index(Global, Linear) % Int32

    if atom_i <= n_atoms
        # The grid is uniform in fractional coordinates, which makes the cells
        #   parallelepipeds that follow a triclinic box
        s = cell_list_fractional(box_inv, SVector{3, T}(x[atom_i], y[atom_i], z[atom_i]))
        # The coordinates are wrapped, so the clamp only guards against floating point
        #   landing one cell outside the grid. Clamping rather than wrapping keeps the
        #   cell consistent with the coordinate that the distances are computed from
        cell_id_x = min(max(floor(Int32, s[1] * num_cell_x), Int32(0)),
                        num_cell_x - Int32(1))
        cell_id_y = min(max(floor(Int32, s[2] * num_cell_y), Int32(0)),
                        num_cell_y - Int32(1))
        cell_id_z = min(max(floor(Int32, s[3] * num_cell_z), Int32(0)),
                        num_cell_z - Int32(1))
        cell = Int32(1) + cell_id_x + num_cell_x * (cell_id_y + num_cell_y * cell_id_z)
        cell_ids[atom_i] = cell
        Atomix.@atomic cell_counts[cell] += Int32(1)
    end
end

# Each atom takes the next free slot of its cell and its coordinates are gathered into
#   cell order at the same time, so that the search kernel reads them contiguously
@kernel inbounds=true function gpu_cell_list_cell_particles_kernel!(cell_particles, cell_x,
                                    cell_y, cell_z, cell_write_counts, @Const(cell_ids),
                                    @Const(cell_offsets), @Const(x), @Const(y), @Const(z),
                                    n_atoms)
    atom_i = @index(Global, Linear) % Int32

    if atom_i <= n_atoms
        cell = cell_ids[atom_i]
        # The atomic returns the new value, so the slot taken is one less
        slot = (Atomix.@atomic cell_write_counts[cell] += Int32(1)) - Int32(1)
        cell_index = cell_offsets[cell] + slot
        cell_particles[cell_index] = atom_i
        cell_x[cell_index] = x[atom_i]
        cell_y[cell_index] = y[atom_i]
        cell_z[cell_index] = z[atom_i]
    end
end

#=
Turn an inclusive prefix sum of counts into the one-based start index of each entry.

The last entry also holds the total, which the thread that reads it copies into a
counter slot so that the host does not need a separate transfer for it.
=#
@kernel inbounds=true function gpu_cell_list_offsets_kernel!(offsets, counters,
                                            @Const(inclusive_counts), n_entries, counter_slot)
    entry = @index(Global, Linear) % Int32

    if entry <= n_entries
        offsets[entry] = (entry == Int32(1) ? Int32(1) :
                          inclusive_counts[entry - Int32(1)] + Int32(1))
        if entry == n_entries && counter_slot > Int32(0)
            counters[counter_slot] = inclusive_counts[n_entries]
        end
    end
end

@kernel inbounds=true function gpu_cell_list_tile_counts_kernel!(cell_tile_counts,
                                            @Const(cell_counts), n_cells)
    cell = @index(Global, Linear) % Int32

    if cell <= n_cells
        cell_tile_counts[cell] = cld(cell_counts[cell], Int32(CELL_BLOCK_SIZE))
    end
end

# One group handles one tile of a cell, so the tiles of every cell are listed out
@kernel inbounds=true function gpu_cell_list_tile_schedule_kernel!(host_tile_cells,
                                    host_tile_starts, @Const(cell_tile_counts),
                                    @Const(cell_tile_offsets), n_cells)
    cell = @index(Global, Linear) % Int32

    if cell <= n_cells
        n_tiles = cell_tile_counts[cell]
        first_tile = cell_tile_offsets[cell]
        for local_tile in Int32(0):(n_tiles - Int32(1))
            host_tile_cells[first_tile + local_tile] = cell
            host_tile_starts[first_tile + local_tile] = local_tile * Int32(CELL_BLOCK_SIZE)
        end
    end
end

# Number of exceptions of an atom that are held in registers, see AtomExceptions
const N_CACHED_EXCEPTIONS = 4

#=
The exceptions of one atom, i.e. the atoms before it in the list that it is either
excluded from interacting with or has a special interaction with.

The first few are read into registers when a thread starts on an atom. The list is
looked at once per candidate pair, so reading it from memory there instead would add
a load that depends on the candidate to a loop that is already latency-bound, and
measures slower than the dense mask it replaces. Almost every atom has at most
`N_CACHED_EXCEPTIONS` exceptions and the rest fall back to reading the tail.
=#
struct AtomExceptions{J, N}
    js::J
    cached::NTuple{N, Int32}
    first_k::Int32
    last_k::Int32
end

@inline function atom_exceptions(starts, js, atom_i)
    first_k = starts[atom_i]
    last_k = starts[atom_i + Int32(1)] - Int32(1)
    # Val unrolls this, and zero never matches an atom index so it pads a list that
    #   is shorter than the cache
    cached = ntuple(Val(N_CACHED_EXCEPTIONS)) do n
        k = first_k + Int32(n - 1)
        k <= last_k ? js[k] : Int32(0)
    end
    return AtomExceptions(js, cached, first_k, last_k)
end

@inline atom_exceptions(::Val{:geometric}, starts, js, atom_i) = nothing
@inline atom_exceptions(::Val{:molly}, starts, js, atom_i) =
                        atom_exceptions(starts, js, atom_i)

@inline in_exceptions(::Nothing, atom_j) = false

@inline function in_exceptions(exceptions::AtomExceptions{<:Any, N}, atom_j) where {N}
    found = reduce(|, ntuple(n -> exceptions.cached[n] == atom_j, Val(N)))
    # The tail of a list longer than the cache, which almost no atom has
    if exceptions.last_k - exceptions.first_k >= Int32(N)
        for k in (exceptions.first_k + Int32(N)):exceptions.last_k
            found |= (exceptions.js[k] == atom_j)
        end
    end
    return found
end

@inline include_half_pair(excluded, atom_i, atom_j) = atom_i > atom_j &&
                                                      !in_exceptions(excluded, atom_j)

@kernel inbounds=true function gpu_cell_list_pair_counts_kernel!(pair_counts,
                                    @Const(neighbor_counts), @Const(neighbors),
                                    excluded_starts, excluded_js, mode, n_atoms,
                                    max_neighbors)
    atom_i = @index(Global, Linear) % Int32

    if atom_i <= n_atoms
        count = Int32(0)
        # Clamped since an overflowing count is reported but not stored
        n_neighbors = min(neighbor_counts[atom_i], max_neighbors)
        excluded = atom_exceptions(mode, excluded_starts, excluded_js, atom_i)

        for slot in Int32(1):n_neighbors
            if include_half_pair(excluded, atom_i, neighbors[slot, atom_i])
                count += Int32(1)
            end
        end

        pair_counts[atom_i] = count
    end
end

@kernel inbounds=true function gpu_cell_list_pair_write_kernel!(pair_list,
                                    @Const(pair_offsets), @Const(neighbor_counts),
                                    @Const(neighbors), excluded_starts, excluded_js,
                                    special_starts, special_js, mode, n_atoms,
                                    max_neighbors, pair_capacity)
    atom_i = @index(Global, Linear) % Int32

    if atom_i <= n_atoms
        write_position = pair_offsets[atom_i]
        n_neighbors = min(neighbor_counts[atom_i], max_neighbors)
        excluded = atom_exceptions(mode, excluded_starts, excluded_js, atom_i)
        specials = atom_exceptions(mode, special_starts, special_js, atom_i)

        for slot in Int32(1):n_neighbors
            atom_j = neighbors[slot, atom_i]
            if include_half_pair(excluded, atom_i, atom_j)
                # The total is checked on the host afterwards, this keeps an overflowing
                #   write inside the buffer until then
                if write_position <= pair_capacity
                    pair_list[write_position] = (atom_i, atom_j,
                                                 in_exceptions(specials, atom_j))
                end
                write_position += Int32(1)
            end
        end
    end
end

# Wrap a stencil cell index into the grid, along with the number of box images the
#   wrap moved by, which gives the shift that brings a candidate into the same image
@inline function wrapped_cell_index(cell, num_cell)
    if cell < Int32(0)
        return cell + num_cell, -one(Int32)
    elseif cell >= num_cell
        return cell - num_cell, one(Int32)
    else
        return cell, zero(Int32)
    end
end

#=
Search the 3x3x3 stencil of cells around each atom of one tile.

Note that this kernel can not run on the KernelAbstractions CPU backend, which splits
a kernel at each `@synchronize` and so does not carry the per-thread state here across
the barriers. The finder only dispatches on GPU array types, where the barrier is a
group synchronization and the state is kept.
=#
@kernel inbounds=true function gpu_cell_list_search_kernel!(neighbor_counts, neighbors,
                                    counters, @Const(cell_counts), @Const(cell_offsets),
                                    @Const(cell_particles), @Const(host_tile_cells),
                                    @Const(host_tile_starts), @Const(cell_x), @Const(cell_y),
                                    @Const(cell_z), num_cell_x, num_cell_y, num_cell_z,
                                    box::SMatrix{3, 3, T}, cutoff2::T, max_neighbors,
                                    ::Val{block_size}) where {T, block_size}
    host_tile = @index(Group, Linear) % Int32
    lane = @index(Local, Linear) % Int32

    shared_x = @localmem T (block_size,)
    shared_y = @localmem T (block_size,)
    shared_z = @localmem T (block_size,)
    shared_ids = @localmem Int32 (block_size,)

    # Groups are launched up to an upper bound on the tile count so that the exact count
    #   does not have to be read back to the host, the rest exit here
    if host_tile <= counters[COUNTER_HOST_TILES]
        host_cell = host_tile_cells[host_tile]
        host_start = cell_offsets[host_cell]
        host_local = host_tile_starts[host_tile] + lane - Int32(1)
        host_active = host_local < cell_counts[host_cell]

        atom_i = Int32(0)
        x_i, y_i, z_i = zero(T), zero(T), zero(T)
        count = Int32(0)

        if host_active
            host_index = host_start + host_local
            atom_i = cell_particles[host_index]
            x_i = cell_x[host_index]
            y_i = cell_y[host_index]
            z_i = cell_z[host_index]
        end

        cell0 = host_cell - Int32(1)
        cx = cell0 % num_cell_x
        tmp = cell0 ÷ num_cell_x
        cy = tmp % num_cell_y
        cz = tmp ÷ num_cell_y

        # The stencil cell fixes which periodic image of a candidate is the closest one,
        #   so a shift per cell replaces recomputing the minimum image for every pair
        for dz in Int32(-1):Int32(1)
            nz, image_z = wrapped_cell_index(cz + dz, num_cell_z)
            shift_c = T(image_z) * box[:, 3]
            for dy in Int32(-1):Int32(1)
                ny, image_y = wrapped_cell_index(cy + dy, num_cell_y)
                shift_bc = shift_c + T(image_y) * box[:, 2]
                for dx in Int32(-1):Int32(1)
                    nx, image_x = wrapped_cell_index(cx + dx, num_cell_x)
                    shift = shift_bc + T(image_x) * box[:, 1]
                    candidate_cell = Int32(1) + nx + num_cell_x * (ny + num_cell_y * nz)
                    candidate_start = cell_offsets[candidate_cell]
                    n_candidates = cell_counts[candidate_cell]
                    candidate_tile_start = Int32(0)

                    # The loop bound is the same for every thread in the group, so the
                    #   barriers below are reached by all of them
                    while candidate_tile_start < n_candidates
                        candidate_local = candidate_tile_start + lane - Int32(1)
                        tile_count = min(Int32(block_size),
                                         n_candidates - candidate_tile_start)

                        if candidate_local < n_candidates
                            candidate_index = candidate_start + candidate_local
                            shared_ids[lane] = cell_particles[candidate_index]
                            shared_x[lane] = cell_x[candidate_index]
                            shared_y[lane] = cell_y[candidate_index]
                            shared_z[lane] = cell_z[candidate_index]
                        end

                        @synchronize

                        if host_active
                            for candidate_lane in Int32(1):tile_count
                                atom_j = shared_ids[candidate_lane]
                                if atom_j != atom_i
                                    dx_ij = (shared_x[candidate_lane] - x_i) + shift[1]
                                    dy_ij = (shared_y[candidate_lane] - y_i) + shift[2]
                                    dz_ij = (shared_z[candidate_lane] - z_i) + shift[3]
                                    r2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij
                                    if r2 <= cutoff2
                                        count += Int32(1)
                                        if count <= max_neighbors
                                            neighbors[count, atom_i] = atom_j
                                        end
                                    end
                                end
                            end
                        end

                        @synchronize

                        candidate_tile_start += Int32(block_size)
                    end
                end
            end
        end

        if host_active
            neighbor_counts[atom_i] = count
            # The host reads the counter and grows the buffer if it is not zero
            if count > max_neighbors
                Atomix.@atomic counters[COUNTER_N_OVERFLOW] += Int32(1)
            end
        end
    end
end

function build_gpu_cell_list!(state::GPUCellListState)
    n_atoms, n_cells = Int(state.n_atoms), Int(state.n_cells)
    backend = get_backend(state.x)
    n_threads_atoms = gpu_threads_cell_list(n_atoms)
    n_threads_cells = gpu_threads_cell_list(n_cells)

    reset_kernel! = gpu_cell_list_reset_kernel!(backend, n_threads_cells)
    reset_kernel!(state.cell_counts, state.cell_write_counts, state.counters,
                  Int32(N_CELL_LIST_COUNTERS);
                  ndrange=max(length(state.cell_counts), N_CELL_LIST_COUNTERS))

    cell_ids_kernel! = gpu_cell_list_cell_ids_kernel!(backend, n_threads_atoms)
    cell_ids_kernel!(state.cell_ids, state.cell_counts, state.x, state.y, state.z,
                     state.n_atoms, state.num_cell_x, state.num_cell_y, state.num_cell_z,
                     state.box_inv; ndrange=n_atoms)

    # The buffers can be longer than the cell grid after the box has shrunk, but the
    #   counts past n_cells are zero so the scan is still correct over the grid
    AcceleratedKernels.accumulate!(+, state.inclusive_counts, state.cell_counts, backend;
                                   init=Int32(0))

    offsets_kernel! = gpu_cell_list_offsets_kernel!(backend, n_threads_cells)
    offsets_kernel!(state.cell_offsets, state.counters, state.inclusive_counts,
                    state.n_cells, Int32(0); ndrange=n_cells)

    tile_counts_kernel! = gpu_cell_list_tile_counts_kernel!(backend, n_threads_cells)
    tile_counts_kernel!(state.cell_tile_counts, state.cell_counts, state.n_cells;
                        ndrange=n_cells)

    AcceleratedKernels.accumulate!(+, state.cell_tile_inclusive_counts,
                                   state.cell_tile_counts, backend; init=Int32(0))

    offsets_kernel!(state.cell_tile_offsets, state.counters,
                    state.cell_tile_inclusive_counts, state.n_cells, COUNTER_HOST_TILES;
                    ndrange=n_cells)

    tile_schedule_kernel! = gpu_cell_list_tile_schedule_kernel!(backend, n_threads_cells)
    tile_schedule_kernel!(state.host_tile_cells, state.host_tile_starts,
                          state.cell_tile_counts, state.cell_tile_offsets, state.n_cells;
                          ndrange=n_cells)

    cell_particles_kernel! = gpu_cell_list_cell_particles_kernel!(backend, n_threads_atoms)
    cell_particles_kernel!(state.cell_particles, state.cell_x, state.cell_y, state.cell_z,
                           state.cell_write_counts, state.cell_ids, state.cell_offsets,
                           state.x, state.y, state.z, state.n_atoms; ndrange=n_atoms)

    return state
end

function query_gpu_cell_list!(state::GPUCellListState)
    backend = get_backend(state.x)
    # Every atom lies in exactly one scheduled tile, so every count is written. The
    #   ndrange is a whole number of groups, which the barriers in the kernel need
    search_kernel! = gpu_cell_list_search_kernel!(backend, CELL_BLOCK_SIZE)
    search_kernel!(state.neighbor_counts, state.neighbors, state.counters, state.cell_counts,
                   state.cell_offsets, state.cell_particles, state.host_tile_cells,
                   state.host_tile_starts, state.cell_x, state.cell_y, state.cell_z,
                   state.num_cell_x, state.num_cell_y, state.num_cell_z, state.box,
                   state.cutoff2, state.max_neighbors, Val(CELL_BLOCK_SIZE);
                   ndrange=(state.max_host_tiles * CELL_BLOCK_SIZE))
    return state
end

function count_pairs_gpu_cell_list!(state::GPUCellListState, mode, nf)
    n_atoms = Int(state.n_atoms)
    backend = get_backend(state.x)
    n_threads_gpu = gpu_threads_cell_list(n_atoms)

    pair_counts_kernel! = gpu_cell_list_pair_counts_kernel!(backend, n_threads_gpu)
    pair_counts_kernel!(state.pair_counts, state.neighbor_counts, state.neighbors,
                        nf.excluded_starts, nf.excluded_js, mode, state.n_atoms,
                        state.max_neighbors; ndrange=n_atoms)

    AcceleratedKernels.accumulate!(+, state.pair_inclusive_counts, state.pair_counts, backend;
                                   init=Int32(0))

    offsets_kernel! = gpu_cell_list_offsets_kernel!(backend, n_threads_gpu)
    offsets_kernel!(state.pair_offsets, state.counters, state.pair_inclusive_counts,
                    state.n_atoms, COUNTER_N_PAIRS; ndrange=n_atoms)

    return state
end

function write_pairs_gpu_cell_list!(state::GPUCellListState, mode, nf)
    n_atoms = Int(state.n_atoms)
    backend = get_backend(state.x)

    pair_write_kernel! = gpu_cell_list_pair_write_kernel!(backend,
                                                          gpu_threads_cell_list(n_atoms))
    pair_write_kernel!(state.pair_list, state.pair_offsets, state.neighbor_counts,
                       state.neighbors, nf.excluded_starts, nf.excluded_js,
                       nf.special_starts, nf.special_js, mode, state.n_atoms,
                       state.max_neighbors, Int32(state.pair_capacity); ndrange=n_atoms)

    return state
end

function build_pair_list!(state::GPUCellListState, mode, nf)
    isnothing(state.pair_list) && error(
                "pair buffers were not allocated for this GPU cell-list state")
    count_pairs_gpu_cell_list!(state, mode, nf)
    write_pairs_gpu_cell_list!(state, mode, nf)
    return state
end

# Read the counters written during the rebuild, the only device to host transfer in
#   find_neighbors
function read_gpu_cell_list_counters!(state::GPUCellListState)
    copyto!(state.counters_host, state.counters)
    state.n_host_tiles = Int(state.counters_host[COUNTER_HOST_TILES])
    return state.counters_host
end

#=
Run one full rebuild, growing the buffers and repeating if they turned out to be too
small. Everything up to the counter read is asynchronous.
=#
function rebuild_gpu_cell_list!(state::GPUCellListState, mode, nf, build_pairs)
    build_gpu_cell_list!(state)
    query_gpu_cell_list!(state)
    build_pairs && build_pair_list!(state, mode, nf)
    counters = read_gpu_cell_list_counters!(state)

    if counters[COUNTER_N_OVERFLOW] > Int32(0)
        # Only when the capacity was too small, since this is a second device to host
        #   transfer
        required = Int(maximum(state.neighbor_counts))

        # Grown past what was needed so that a slowly densifying system does not have to
        #   grow again on the next rebuild
        grow_gpu_cell_list_neighbors!(state, required + required ÷ 8)

        build_gpu_cell_list!(state)
        query_gpu_cell_list!(state)
        build_pairs && build_pair_list!(state, mode, nf)
        counters = read_gpu_cell_list_counters!(state)

        iszero(counters[COUNTER_N_OVERFLOW]) || error(
            "GPU cell-list neighbor capacity of $(state.max_neighbors) was still " *
            "exceeded by $(counters[COUNTER_N_OVERFLOW]) atoms after growing the buffer")
    end

    build_pairs || return 0

    n_pairs = Int(counters[COUNTER_N_PAIRS])

    if n_pairs > state.pair_capacity
        grow_gpu_cell_list_pairs!(state, n_pairs)
        write_pairs_gpu_cell_list!(state, mode, nf)
    end

    return n_pairs
end

# The state is only reused when it describes the same atoms in the same float type,
#   everything that depends on the box is updated in place
function reusable_gpu_cell_list_state(current_neighbors, n_atoms::Integer, ::Type{T},
                                      build_pairs::Bool) where {T}
    current_neighbors isa GPUCellListNeighborList || return nothing
    state = current_neighbors.state
    state isa GPUCellListState{T} || return nothing
    state.n_atoms == n_atoms || return nothing
    (!build_pairs || !isnothing(state.pair_list)) || return nothing
    return state
end

function find_neighbors(sys::System{3, AT},
                        nf::GPUCellListNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        kwargs...) where {AT <: AbstractGPUArray}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    (sys.boundary isa CubicBoundary{3} || sys.boundary isa TriclinicBoundary) || throw(
        ArgumentError("GPUCellListNeighborFinder currently supports only " *
                      "three-dimensional CubicBoundary and TriclinicBoundary systems, " *
                      "got $(typeof(sys.boundary))"))

    has_infinite_boundary(sys.boundary) && throw(ArgumentError(
        "GPUCellListNeighborFinder does not support infinite boundaries"))

    dist_unit = unit(zero(eltype(eltype(sys.coords))))
    T = float_type(sys)
    box = cell_list_box_matrix(sys.boundary, T, dist_unit)
    widths = SVector{3, T}(T.(ustrip.(dist_unit, cell_list_box_widths(sys.boundary))))
    cutoff = T(ustrip(dist_unit, nf.dist_cutoff))

    n_atoms = length(sys)
    build_pairs = nf.output !== :ragged
    pair_mode = (nf.output === :molly_pairs ? Val(:molly) : Val(:geometric))

    max_neighbors = if isnothing(nf.max_neighbors)
        estimate_gpu_cell_list_max_neighbors(n_atoms, T(ustrip(dist_unit^3,
                                                              volume(sys.boundary))), cutoff)
    else
        nf.max_neighbors
    end

    if iszero(n_atoms)
        # Still validated so that the box is reported the same way as for a system that
        #   does have atoms
        gpu_cell_list_grid(widths, cutoff)
        return GPUCellListNeighborList(
            similar(sys.coords, Int32, 0),
            similar(sys.coords, Int32, max_neighbors, 0),
            0,
            (build_pairs ? similar(sys.coords, Tuple{Int32, Int32, Bool}, 0) : nothing),
            nothing,
        )
    end

    state = reusable_gpu_cell_list_state(current_neighbors, n_atoms, T, build_pairs)

    if isnothing(state)
        state = allocate_gpu_cell_list_state(sys.coords, T, box, widths, cutoff;
                                             max_neighbors=Int32(max_neighbors),
                                             allocate_pairs=build_pairs)
    else
        update_gpu_cell_list_state!(state, box, widths, cutoff, max_neighbors)
        split_gpu_cell_list_coordinates!(state, sys.coords)
    end

    n_pairs = rebuild_gpu_cell_list!(state, pair_mode, nf, build_pairs)

    return GPUCellListNeighborList(state.neighbor_counts, state.neighbors, n_pairs,
                                   (build_pairs ? state.pair_list : nothing), state)
end

#=
Whether a box can be used with GPUCellListNeighborFinder.

The 3x3x3 cell stencil needs at least three cells along every box axis, so opposite
box faces have to be at least three times the neighbor search distance apart.
=#
function gpu_cell_list_suitable(boundary, dist_cutoff)
    (boundary isa CubicBoundary{3} || boundary isa TriclinicBoundary) || return false
    has_infinite_boundary(boundary) && return false
    return minimum(cell_list_box_widths(boundary)) >= 3 * dist_cutoff
end

# Defined so that unsupported systems get an explanation rather than a MethodError
function find_neighbors(sys::System,
                        nf::GPUCellListNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        kwargs...)
    throw(ArgumentError("GPUCellListNeighborFinder requires a three-dimensional GPU " *
                        "system with a CubicBoundary or TriclinicBoundary, got a " *
                        "$(AtomsBase.n_dimensions(sys.boundary))D $(array_type(sys.coords)) " *
                        "system with a $(typeof(sys.boundary)); use " *
                        "DistanceNeighborFinder or CellListMapNeighborFinder instead"))
end

function neighbor_finder_masks(nf::GPUCellListNeighborFinder, n_atoms::Integer)
    eligible = trues(n_atoms, n_atoms)
    special = falses(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    # :ragged and :geometric_pairs ignore the exceptions, so all pairs are eligible
    isnothing(nf.excluded_js) && return eligible, special
    for (i, j) in sparse_pairs(nf.excluded_starts, nf.excluded_js)
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, j) in sparse_pairs(nf.special_starts, nf.special_js)
        special[i, j] = true
        special[j, i] = true
    end
    return eligible, special
end

function Base.show(io::IO, neighbor_finder::GPUCellListNeighborFinder)
    println(io, typeof(neighbor_finder))
    println(io, "  output = ", neighbor_finder.output)
    if !isnothing(neighbor_finder.excluded_js)
        println(io, "  n_atoms = ", neighbor_finder.n_atoms)
        println(io, "  n_excluded = ", length(neighbor_finder.excluded_js))
        println(io, "  n_special = ", length(neighbor_finder.special_js))
    end
    println(io, "  n_steps = ", neighbor_finder.n_steps)
    print(  io, "  dist_cutoff = ", neighbor_finder.dist_cutoff)
end

"""
    DistanceNeighborFinder(; eligible, dist_cutoff, special, n_steps)

Find close atoms by distance.

This is the recommended neighbor finder on non-NVIDIA GPUs.

`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.
"""
struct DistanceNeighborFinder{B, D}
    eligible::B
    dist_cutoff::D
    special::B
    n_steps::Int
end

function DistanceNeighborFinder(;
                                eligible,
                                dist_cutoff,
                                special=zero(eligible),
                                n_steps=10)
    check_neighbor_matrices(eligible, special)
    return DistanceNeighborFinder{typeof(eligible), typeof(dist_cutoff)}(
                eligible, dist_cutoff, special, n_steps)
end

#=
The neighbor search for `DistanceNeighborFinder` is done in two passes on both CPU and
    GPU, which avoids growing intermediate lists and lets the output be written exactly
    once into an array of the right size.
The first pass records whether each candidate pair is a neighbor as one bit of a mask
    word and counts the bits set in that word, the counts are turned into write offsets
    with a prefix sum, and the second pass expands the mask words into the neighbor list.
Splitting the distance test off from the list building also lets the distance loop be
    vectorized on the CPU, which it cannot be when it contains a `push!`.
=#

# Number of pair flags packed into one CPU mask word
const n_pairs_per_mask_cpu = 64

# Multiplier that gathers the low bit of each of 8 bytes into the top byte of the product
const byte_flags_to_bits = 0x0102040810204080

#=
Record in `flags[j]` whether atom j is within the cutoff of the atom at `ci` and eligible
Written as a separate loop from the list building so that it can be vectorized
=#
@inline function neighbor_flags!(flags, coords_dims, ci, boundary, sqdist_cutoff, eligible_i,
                                 n_j, ::Val{D}) where D
    @inbounds for j in 1:n_j
        cj = SVector{D}(ntuple(d -> @inbounds(coords_dims[d][j]), Val(D)))
        r2 = sum(abs2, vector(ci, cj, boundary))
        flags[j] = (r2 <= sqdist_cutoff) & eligible_i[j]
    end
    return flags
end

#=
Pack the first `n_j` 0/1 bytes in `flags` into mask words starting at `mask_offset`
`flag_words` is `flags` viewed 8 bytes at a time, which makes the packing 8 times cheaper
Any bits of the last word past `n_j` are zeroed, so the flags there do not have to be
Returns the number of bits set, i.e. the number of neighbors found for the atom
=#
@inline function pack_neighbor_masks!(masks, flag_words, mask_offset, n_j)
    n_words = cld(n_j, n_pairs_per_mask_cpu)
    n_set = 0
    @inbounds for w in 1:n_words
        word_start = (w - 1) << 3
        mask = zero(UInt64)
        for b in 0:7
            byte_flags = flag_words[word_start + b + 1]
            mask |= UInt64((byte_flags * byte_flags_to_bits) >> 56) << (b << 3)
        end
        n_bits = n_j - ((w - 1) << 6)
        if n_bits < n_pairs_per_mask_cpu
            mask &= (one(UInt64) << n_bits) - one(UInt64)
        end
        masks[mask_offset + w] = mask
        n_set += count_ones(mask)
    end
    return n_set
end

# Expand the mask words for atom i into `neighbors_list`, starting after index `list_offset`
@inline function expand_neighbor_masks!(neighbors_list, masks, mask_offset, n_j, i,
                                        special_i, list_offset)
    ni = list_offset
    @inbounds for w in 1:cld(n_j, n_pairs_per_mask_cpu)
        mask = masks[mask_offset + w]
        while !iszero(mask)
            j = ((w - 1) << 6) + trailing_zeros(mask) + 1
            mask &= mask - one(mask)
            ni += 1
            neighbors_list[ni] = (Int32(i), Int32(j), special_i[j])
        end
    end
    return ni
end

function find_neighbors(sys::System{D},
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where D
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    n_atoms = length(sys)
    sqdist_cutoff = nf.dist_cutoff ^ 2
    # The coordinates are copied to one array per dimension since the strided reads of an
    #   array of static vectors stop the distance loop being vectorized
    coords_dims = ntuple(d -> [c[d] for c in sys.coords], Val(D))

    # Mask words are packed by atom, with the words for atom i starting at mask_starts[i]
    mask_starts = Vector{Int}(undef, n_atoms)
    n_mask_words = 0
    @inbounds for i in 1:n_atoms
        mask_starts[i] = n_mask_words
        n_mask_words += cld(i - 1, n_pairs_per_mask_cpu)
    end
    masks = Vector{UInt64}(undef, n_mask_words)
    n_neighbors_atom = Vector{Int}(undef, n_atoms)
    # Rounded up to a whole number of mask words so that the flags can be viewed as words
    n_flags = cld(n_atoms, n_pairs_per_mask_cpu) * n_pairs_per_mask_cpu
    flags_threads = [zeros(UInt8, n_flags) for _ in 1:n_threads]

    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        flags = flags_threads[chunk_i]
        flag_words = reinterpret(UInt64, flags)
        @inbounds for i in chunk_i:n_threads:n_atoms
            neighbor_flags!(flags, coords_dims, sys.coords[i], sys.boundary, sqdist_cutoff,
                            (@view nf.eligible[:, i]), i - 1, Val(D))
            n_neighbors_atom[i] = pack_neighbor_masks!(masks, flag_words, mask_starts[i], i - 1)
        end
    end

    # Exclusive prefix sum of the per-atom neighbor counts gives the write offsets
    list_starts = Vector{Int}(undef, n_atoms)
    n_neighbors = 0
    @inbounds for i in 1:n_atoms
        list_starts[i] = n_neighbors
        n_neighbors += n_neighbors_atom[i]
    end
    neighbors_list = Vector{Tuple{Int32, Int32, Bool}}(undef, n_neighbors)

    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        @inbounds for i in chunk_i:n_threads:n_atoms
            expand_neighbor_masks!(neighbors_list, masks, mask_starts[i], i - 1, i,
                                   (@view nf.special[:, i]), list_starts[i])
        end
    end

    return NeighborList(n_neighbors, neighbors_list)
end

function gpu_threads_dnf(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_DISTANCENF", "512"))
    return n_threads_gpu
end

const n_pairs_per_mask_gpu = 32 # n pair flags packed into one GPU mask word
const n_masks_per_group_gpu = 32 # n mask words filled by one group of consecutive GPU threads
const n_pairs_per_group_gpu = n_pairs_per_mask_gpu * n_masks_per_group_gpu

#=
Map a one-based index over the n_atoms * (n_atoms - 1) / 2 pairs to the pair (i, j), i < j,
    running down the columns of the pair triangle so that consecutive indices give
    consecutive i for a fixed j.
This is the transpose of `pair_index` and is used by the GPU neighbor finder kernels, where
    it makes the `eligible[i, j]` and `coords[i]` reads of neighboring threads coalesced.
As in `pair_index` the square root is taken in Float32 since Metal GPUs do not support
    Float64, so the initial estimate of j is corrected below using exact integer arithmetic.
=#
@inline function pair_index_col(n_atoms::Integer, ind::Integer)
    n, kz = promote(n_atoms, ind - one(ind))
    T = typeof(n)
    # Column j holds j - 1 pairs, so jz = j - 1 is the largest value with
    #   jz * (jz - 1) / 2 <= kz
    jz = unsafe_trunc(T, (sqrt(Float32(8 * kz + 1)) + 1.0f0) / 2)
    jz = min(max(jz, one(T)), n - one(T))
    while jz > one(T) && (jz * (jz - one(T))) ÷ T(2) > kz
        jz -= one(T)
    end
    while jz < n - one(T) && ((jz + one(T)) * jz) ÷ T(2) <= kz
        jz += one(T)
    end
    i = kz - (jz * (jz - one(T))) ÷ T(2) + one(T)
    j = jz + one(T)
    return i, j
end

#=
Map bit `bit_i` (zero-based) of mask word `word_i` to the index of the pair it records.
Consecutive threads in a group take consecutive pairs at each step so that the coordinate
    reads are coalesced, meaning that consecutive bits of a mask word are
    `n_masks_per_group_gpu` pairs apart.
=#
@inline function mask_bit_pair_index(word_i, bit_i)
    group_i, word_in_group = divrem(word_i - 1, n_masks_per_group_gpu)
    return group_i * n_pairs_per_group_gpu + bit_i * n_masks_per_group_gpu + word_in_group + 1
end

@kernel inbounds=true function distance_neighbor_finder_mask_kernel!(masks, counts, @Const(coords),
                                        @Const(eligible), boundary, sq_dist_cutoff, n_inters)
    n_atoms = length(coords)
    word_i = @index(Global, Linear)

    if word_i <= length(masks)
        mask = zero(UInt32)
        for bit_i in 0:(n_pairs_per_mask_gpu - 1)
            inter_i = mask_bit_pair_index(word_i, bit_i)
            if inter_i <= n_inters
                i, j = pair_index_col(n_atoms, inter_i)
                if eligible[i, j]
                    dr = vector(coords[i], coords[j], boundary)
                    r2 = sum(abs2, dr)
                    if r2 <= sq_dist_cutoff
                        mask |= (one(UInt32) << bit_i)
                    end
                end
            end
        end
        masks[word_i] = mask
        counts[word_i] = Int32(count_ones(mask))
    end
end

@kernel inbounds=true function distance_neighbor_finder_fill_kernel!(neighbors_list, @Const(masks),
                                        @Const(list_ends), @Const(special), n_atoms)
    word_i = @index(Global, Linear)

    if word_i <= length(masks)
        mask = masks[word_i]
        if !iszero(mask)
            ni = list_ends[word_i] - Int32(count_ones(mask))
            while !iszero(mask)
                bit_i = trailing_zeros(mask)
                mask &= mask - one(mask)
                i, j = pair_index_col(n_atoms, mask_bit_pair_index(word_i, bit_i))
                ni += Int32(1)
                neighbors_list[ni] = (Int32(j), Int32(i), special[j, i])
            end
        end
    end
end

function find_neighbors(sys::System{D, AT},
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        kwargs...) where {D, AT <: AbstractGPUArray}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    n_inters = n_atoms_to_n_pairs(length(sys))
    if iszero(n_inters)
        return NeighborList(0, similar(sys.coords, Tuple{Int32, Int32, Bool}, 0))
    end
    n_threads_gpu = gpu_threads_dnf(n_inters)
    backend = get_backend(sys.coords)

    # Rounded up to a whole number of groups so that every pair is covered by a mask bit
    n_masks = cld(n_inters, n_pairs_per_group_gpu) * n_masks_per_group_gpu
    masks = similar(sys.coords, UInt32, n_masks)
    counts = similar(sys.coords, Int32, n_masks)

    mask_kernel! = distance_neighbor_finder_mask_kernel!(backend, n_threads_gpu)
    mask_kernel!(masks, counts, sys.coords, nf.eligible, sys.boundary, nf.dist_cutoff^2,
                 n_inters; ndrange=n_masks)

    # The inclusive prefix sum of the per-word neighbor counts gives the index one past the
    #   last neighbor written by each mask word, from which the fill kernel subtracts its
    #   own count to get its write offset
    AcceleratedKernels.accumulate!(+, counts, backend; init=Int32(0))
    n_neighbors = Int(only(Array(@view counts[n_masks:n_masks])))
    neighbors_list = similar(sys.coords, Tuple{Int32, Int32, Bool}, n_neighbors)

    if n_neighbors > 0
        fill_kernel! = distance_neighbor_finder_fill_kernel!(backend, n_threads_gpu)
        fill_kernel!(neighbors_list, masks, counts, nf.special, length(sys); ndrange=n_masks)
    end

    return NeighborList(n_neighbors, neighbors_list)
end

"""
    TreeNeighborFinder(; eligible, dist_cutoff, special, n_steps)

Find close atoms by distance using a tree search.

`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.

Can not be used if one or more dimensions has infinite boundaries.
Can not be used with [`TriclinicBoundary`](@ref).
"""
struct TreeNeighborFinder{D}
    eligible::BitArray{2}
    dist_cutoff::D
    special::BitArray{2}
    n_steps::Int
end

function TreeNeighborFinder(;
                            eligible,
                            dist_cutoff,
                            special=zero(eligible),
                            n_steps=10)
    check_neighbor_matrices(eligible, special)
    return TreeNeighborFinder(eligible, dist_cutoff, special, n_steps)
end

function find_neighbors(sys::System{<:Any, AT},
                        nf::TreeNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where AT
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    dist_unit = unit(first(first(sys.coords)))
    bv = ustrip.(dist_unit, sys.boundary)
    btree = BallTree(ustrip_vec.(sys.coords), PeriodicEuclidean(bv))
    dist_cutoff = ustrip(dist_unit, nf.dist_cutoff)
    nl_threads = [Tuple{Int32, Int32, Bool}[] for i in 1:n_threads]

    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        for i in chunk_i:n_threads:length(sys)
            ci = ustrip.(sys.coords[i])
            nbi = @view nf.eligible[:, i]
            speci = @view nf.special[:, i]
            idxs = inrange(btree, ci, dist_cutoff, true)
            for j in idxs
                if nbi[j] && i > j
                    push!(nl_threads[chunk_i], (Int32(i), Int32(j), speci[j]))
                end
            end
        end
    end

    neighbors_list = Tuple{Int32, Int32, Bool}[]
    for nl in nl_threads
        append!(neighbors_list, nl)
    end

    return NeighborList(length(neighbors_list), to_device(neighbors_list, AT))
end

"""
    CellListMapNeighborFinder(; eligible, dist_cutoff, boundary,
                                special, n_steps, x0, number_of_batches)

Find close atoms by distance using a cell list algorithm from CellListMap.jl.

This is the recommended neighbor finder on CPU.
`dist_cutoff` is the neighbor search distance, which should be the interaction
cutoff distance plus a buffer distance since the list is only updated every
`n_steps` steps.
`x0` are optional initial coordinates that improve the
first approximation of the cell list structure.
The number of dimensions `dims` is inferred from the boundary or `x0`, or assumed
to be 3 otherwise.

The `boundary` parameter is required, and must be an `AbstractBoundary`. 
Infinite boundaries are only accepted in all dimensions.

CellListMap.jl chooses how many batches to split the work over from the number of Julia
threads, so the `n_threads` argument to [`find_neighbors`](@ref) only selects between a
serial (`n_threads=1`) and a parallel run. `number_of_batches` can be given as a tuple of
the number of batches used to build the cell lists and to map over the pairs, which caps
the number of tasks used, with `(0, 0)`, the default, meaning to use the CellListMap.jl
heuristics.
"""
mutable struct CellListMapNeighborFinder{N, T, S}
    eligible::BitArray{2}
    dist_cutoff::T
    special::BitArray{2}
    n_steps::Int
    # The CellListMap.ParticleSystem object
    clm_particlesystem::S
end

function clm_unitcell_arg(b::Union{CubicBoundary, RectangularBoundary}) 
    uc = b.side_lengths
    D = size(uc, 1)
    if any(isinf.(uc))
        if all(isinf.(uc))
            return nothing, D
        else
            throw(ArgumentError("cannot use infinite boundaries in some, but not all, " *
                                "dimensions with CellListMapNeighborFinder"))
        end
    end
    return uc, D
end

function clm_unitcell_arg(b::TriclinicBoundary) 
    uc = hcat(b.basis_vectors...)
    D = size(uc, 1)
    return uc, D
end

# This function sets up the ParticleSystem structure for CellListMap. 
function CellListMapNeighborFinder(;
                                   eligible,
                                   dist_cutoff::T,
                                   boundary::AbstractBoundary,
                                   special=zero(eligible),
                                   n_steps=10,
                                   x0=nothing,
                                   number_of_batches=(0, 0)) where T
    check_neighbor_matrices(eligible, special)
    # Obtain unit cell from boundary: If all boundaries are infinite, use `nothing`
    uc, D = clm_unitcell_arg(boundary)

    clm_system = CellListMap.ParticleSystem(;
        positions=isnothing(x0) ? SVector{D,T}[] : x0,
        cutoff=dist_cutoff,
        unitcell=uc,
        nbatches=number_of_batches,
        parallel=true,
        output=NeighborList(),
        output_name=:neighbors,
    )

    return CellListMapNeighborFinder{D, T, typeof(clm_system)}(eligible, dist_cutoff, special, n_steps, clm_system)
end

# Add a pair to the pair list
# If the buffer size is large enough update the element, otherwise push a new element
#   to `neighbors.list`
@inline function push_pair!(pair, neighbors::NeighborList, eligible, special)
    i, j = pair.i, pair.j
    @inbounds if eligible[i, j]
        n = neighbors.n + 1
        neighbors.n = n
        list = neighbors.list
        element = (i, j, special[i, j])
        if n > length(list)
            push!(list, element)
        else
            @inbounds list[n] = element
        end
    end
    return neighbors
end

# Parallelization interface for custom output of CellListMap
CellListMap.copy_output(nl::NeighborList) = NeighborList(nl.n, copy(nl.list)) 
CellListMap.reset_output!(nl::NeighborList) = empty!(nl)
CellListMap.reducer(nl1::NeighborList, nl2::NeighborList) = append!(nl1, nl2)

function CellListMap.reduce_output!(output::NeighborList, output_threaded::Vector{<:NeighborList})
    n_start = output.n
    n_tot = n_start
    for nb in output_threaded
        n_tot += nb.n
    end
    if length(output.list) < n_tot
        resize!(output.list, n_tot)
    end

    if (n_tot - n_start) > 100_000 && length(output_threaded) > 1 && Threads.nthreads() > 1
        Threads.@threads for i in eachindex(output_threaded)
            chunk_offset = n_start
            @inbounds for jb in 1:(i - 1)
                chunk_offset += output_threaded[jb].n
            end
            nb = output_threaded[i]
            if nb.n > 0
                copyto!(output.list, chunk_offset + 1, nb.list, 1, nb.n)
            end
        end
    else
        offset = n_start
        for nb in output_threaded
            if nb.n > 0
                copyto!(output.list, offset + 1, nb.list, 1, nb.n)
                offset += nb.n
            end
        end
    end

    output.n = n_tot
    return output
end

function find_neighbors(sys::System{D, AT},
                        nf::CellListMapNeighborFinder,
                        current_neighbors=sys.neighbor_finder.clm_particlesystem.neighbors, 
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where {D, AT}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    # Update the CellListMap.ParticleSystem
    positions = from_device(sys.coords)
    unitcell = first(clm_unitcell_arg(sys.boundary))
    parallel = (n_threads > 1)
    if isnothing(unitcell) # Avoid small Union dispatch
        CellListMap.update!(nf.clm_particlesystem; positions=positions, unitcell=nothing,
                            parallel=parallel)
    else
        CellListMap.update!(nf.clm_particlesystem; positions=positions, unitcell=unitcell,
                            parallel=parallel)
    end

    # Update the neighbor list
    neighbors = CellListMap.pairwise!(
        (p, neighbors) -> push_pair!(p, neighbors, nf.eligible, nf.special), 
        nf.clm_particlesystem,
    )

    if AT <: AbstractGPUArray
        return NeighborList(neighbors.n, to_device(neighbors.list, AT))
    else
        return neighbors
    end
end

function neighbor_finder_masks(nf::Union{DistanceNeighborFinder, TreeNeighborFinder, CellListMapNeighborFinder})
    return copy_to_bitmatrix(from_device(nf.eligible)), copy_to_bitmatrix(from_device(nf.special))
end

function Base.show(io::IO, neighbor_finder::Union{DistanceNeighborFinder,
                                TreeNeighborFinder, CellListMapNeighborFinder})
    println(io, typeof(neighbor_finder))
    println(io, "  Size of eligible matrix = " , size(neighbor_finder.eligible))
    println(io, "  n_steps = " , neighbor_finder.n_steps)
    print(  io, "  dist_cutoff = ", neighbor_finder.dist_cutoff)
end

function Base.show(io::IO, neighbor_finder::GPUNeighborFinder)
    println(io, typeof(neighbor_finder))
    println(io, "  n_atoms = " , neighbor_finder.n_atoms)
    println(io, "  n_excluded = " , length(neighbor_finder.excluded_i))
    println(io, "  n_special = " , length(neighbor_finder.special_i))
    println(io, "  n_steps_reorder = " , neighbor_finder.n_steps_reorder)
    print(  io, "  dist_cutoff = ", neighbor_finder.dist_cutoff)
end
