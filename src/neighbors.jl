# Neighbor finders

export
    use_neighbors,
    NoNeighborFinder,
    find_neighbors,
    GPUNeighborFinder,
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

For [`GPUNeighborFinder`](@ref), this returns `nothing`: the CUDA pairwise force
and energy kernels build and cache their interacting tile list internally from
the neighbor-finder metadata.
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
                      excluded_pairs=(), special_pairs=(), n_steps_reorder=25,
                      initialized=false, device_vector_type)
    GPUNeighborFinder(; eligible, dist_cutoff,
                      special=nothing, n_steps_reorder=25,
                      initialized=false, device_vector_type=nothing)

Neighbor finder for CUDA systems that uses Molly's tiled pairwise kernels.

`GPUNeighborFinder` does not materialize a conventional per-atom neighbor list.
Instead, the CUDA pairwise force and energy paths reorder atoms on a Morton
curve, convert sparse exclusions and special pairs into per-tile bitmasks, and
build a compact list of interacting 32x32 tiles directly on the device. For
that reason, [`find_neighbors`](@ref) returns `nothing` for this neighbor
finder.

This is the recommended neighbor finder for `CuArray` systems.

# Keyword arguments
- `n_atoms`: number of atoms when constructing directly from sparse exception
  pairs.
- `eligible`, `special`: compatibility inputs for dense boolean masks. These
  are converted once at construction into sparse exception lists; the dense
  masks are not retained.
- `dist_cutoff`: interaction cutoff used by the pairwise kernels.
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

gpu_sparse_pairs(nf::GPUNeighborFinder) = (
    collect(zip(from_device(nf.excluded_i), from_device(nf.excluded_j))),
    collect(zip(from_device(nf.special_i), from_device(nf.special_j))),
)

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
    existing_pairs, special_pairs = gpu_sparse_pairs(nf)
    update_sparse_pairs!(nf, vcat(existing_pairs, collect(pairs)), special_pairs)
    return nf
end

function GPUNeighborFinder(;
                            n_atoms=nothing,
                            eligible=nothing,
                            dist_cutoff,
                            excluded_pairs=(),
                            special_pairs=(),
                            special=nothing,
                            n_steps_reorder=25,
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

    isnothing(eligible) && throw(ArgumentError("either n_atoms or eligible must be provided"))
    ET = gpu_exception_vector_type(eligible, device_vector_type)
    if isnothing(special)
        special = zero(eligible)
    end
    eligible_cpu = copy_to_bitmatrix(eligible)
    special_cpu = copy_to_bitmatrix(special)
    size(eligible_cpu) == size(special_cpu) || throw(ArgumentError("eligible and special must have the same size"))
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

"""
    DistanceNeighborFinder(; eligible, dist_cutoff, special, n_steps)

Find close atoms by distance.

This is the recommended neighbor finder on non-NVIDIA GPUs.
"""
struct DistanceNeighborFinder{B, D}
    eligible::B
    dist_cutoff::D
    special::B
    n_steps::Int
    neighbors::B # Used internally during neighbor calculation on the GPU
end

function DistanceNeighborFinder(;
                                eligible,
                                dist_cutoff,
                                special=zero(eligible),
                                n_steps=10)
    return DistanceNeighborFinder{typeof(eligible), typeof(dist_cutoff)}(
                eligible, dist_cutoff, special, n_steps, zero(eligible))
end

function find_neighbors(sys::System,
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads())
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    sqdist_cutoff = nf.dist_cutoff ^ 2
    nl_threads = [Tuple{Int32, Int32, Bool}[] for i in 1:n_threads]

    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        for i in chunk_i:n_threads:length(sys)
            ci = sys.coords[i]
            nbi = @view nf.eligible[:, i]
            speci = @view nf.special[:, i]
            for j in 1:(i - 1)
                r2 = sum(abs2, vector(ci, sys.coords[j], sys.boundary))
                if r2 <= sqdist_cutoff && nbi[j]
                    push!(nl_threads[chunk_i], (Int32(i), Int32(j), speci[j]))
                end
            end
        end
    end

    neighbors_list = Tuple{Int32, Int32, Bool}[]
    for nl in nl_threads
        append!(neighbors_list, nl)
    end

    return NeighborList(length(neighbors_list), neighbors_list)
end

function gpu_threads_dnf(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_DISTANCENF", "512"))
    return n_threads_gpu
end

@kernel function distance_neighbor_finder_kernel!(neighbors, @Const(coords),
                                                  @Const(eligible), boundary, sq_dist_cutoff)
    n_atoms = length(coords)
    n_inters = n_atoms_to_n_pairs(n_atoms)
    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= n_inters
        i, j = pair_index(n_atoms, inter_i)
        if eligible[i, j]
            dr = vector(coords[i], coords[j], boundary)
            r2 = sum(abs2, dr)
            if r2 <= sq_dist_cutoff
                neighbors[j, i] = true
            end
        end
    end
end

lists_to_tuple_list(i, j, w) = (Int32(i), Int32(j), w)

function find_neighbors(sys::System{D, AT},
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        kwargs...) where {D, AT <: AbstractGPUArray}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    nf.neighbors .= false
    n_inters = n_atoms_to_n_pairs(length(sys))
    n_threads_gpu = gpu_threads_dnf(n_inters)

    backend = get_backend(sys.coords)
    kernel! = distance_neighbor_finder_kernel!(backend, n_threads_gpu)
    kernel!(nf.neighbors, sys.coords, nf.eligible, sys.boundary,
            nf.dist_cutoff^2, ndrange=n_inters)

    pairs = findall(nf.neighbors)
    nbsi, nbsj = getindex.(pairs, 1), getindex.(pairs, 2)
    special = nf.special[pairs]
    nl = lists_to_tuple_list.(nbsi, nbsj, special)
    return NeighborList(length(nl), nl)
end

"""
    TreeNeighborFinder(; eligible, dist_cutoff, special, n_steps)

Find close atoms by distance using a tree search.

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
    CellListMapNeighborFinder(; eligible, dist_cutoff, special, n_steps, x0,
                              unit_cell, dims)

Find close atoms by distance using a cell list algorithm from CellListMap.jl.

This is the recommended neighbor finder on CPU.
`x0` and `unit_cell` are optional initial coordinates and system unit cell that improve the
first approximation of the cell list structure.
The number of dimensions `dims` is inferred from `unit_cell` or `x0`, or assumed
to be 3 otherwise.

Can not be used if one or more dimensions has infinite boundaries.
"""
mutable struct CellListMapNeighborFinder{N, T}
    eligible::BitArray{2}
    dist_cutoff::T
    special::BitArray{2}
    n_steps::Int
    # Auxiliary arrays for multi-threaded in-place updating of the lists
    cl::CellListMap.CellList{N, T}
    aux::CellListMap.AuxThreaded{N, T}
    neighbors_threaded::Vector{NeighborList}
end

clm_box_arg(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths
clm_box_arg(b::TriclinicBoundary) = hcat(b.basis_vectors...)

# This function sets up the box structure for CellListMap. It uses the unit cell
# if it is given, or guesses a box size from the number of particles, assuming
# that the atomic density is similar to that of liquid water at ambient conditions.
function CellListMapNeighborFinder(;
                                   eligible,
                                   dist_cutoff::T,
                                   special=zero(eligible),
                                   n_steps=10,
                                   x0=nothing,
                                   unit_cell=nothing,
                                   dims=nothing,
                                   number_of_batches=(0, 0)) where T
    n_atoms = size(eligible, 1)
    if !isnothing(dims)
        D = dims
    elseif !isnothing(unit_cell)
        D = n_dimensions(unit_cell)
    elseif !isnothing(x0)
        D = size(eltype(x0))[1]
    else
        D = 3
    end

    if isnothing(unit_cell)
        twice_cutoff = nextfloat(2 * dist_cutoff)
        if unit(dist_cutoff) == NoUnits
            side = max(twice_cutoff, T((n_atoms * 0.01) ^ (1 / 3)))
        else
            side = max(
                twice_cutoff,
                uconvert(unit(dist_cutoff), T((n_atoms * 0.01u"nm^3") ^ (1 / 3))),
            )
        end
        sides = SVector(fill(side, D)...)
        box = CellListMap.Box(sides, dist_cutoff)
    else
        box = CellListMap.Box(clm_box_arg(unit_cell), dist_cutoff)
    end
    if isnothing(x0)
        x = [ustrip.(diag(box.input_unit_cell.matrix)) .* rand(SVector{D, T}) for _ in 1:n_atoms]
    else
        x = x0
    end

    # Construct the cell list for the first time, to allocate
    cl = CellList(x, box; parallel=true, nbatches=number_of_batches)
    return CellListMapNeighborFinder{D, T}(
        eligible, dist_cutoff, special, n_steps,
        cl, CellListMap.AuxThreaded(cl),
        [NeighborList(0, [(Int32(0), Int32(0), false)]) for _ in 1:CellListMap.nbatches(cl)],
    )
end

# Add a pair to the pair list
# If the buffer size is large enough update the element, otherwise push a new element
#   to `neighbor.list`
function push_pair!(neighbors::NeighborList, i::Integer, j::Integer, eligible, special)
    if eligible[i, j]
        push!(neighbors, (Int32(i), Int32(j), special[i, j]))
    end
    return neighbors
end

# This is only called in the parallel case
function reduce_pairs(neighbors::NeighborList, neighbors_threaded::Vector{NeighborList})
    neighbors.n = 0
    for i in eachindex(neighbors_threaded)
        append!(neighbors, neighbors_threaded[i])
    end
    return neighbors
end

function find_neighbors(sys::System{D, AT},
                        nf::CellListMapNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where {D, AT}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    if isnothing(current_neighbors)
        neighbors = NeighborList()
    elseif AT <: AbstractGPUArray
        neighbors = NeighborList(current_neighbors.n, from_device(current_neighbors.list))
    else
        neighbors = current_neighbors
    end
    aux = nf.aux
    cl = nf.cl
    neighbors.n = 0
    neighbors_threaded = nf.neighbors_threaded
    if n_threads > 1
        for i in eachindex(neighbors_threaded)
            neighbors_threaded[i].n = 0
        end
    else
        neighbors_threaded[1].n = 0
    end

    box = CellListMap.Box(clm_box_arg(sys.boundary), nf.dist_cutoff; lcell=1)
    parallel = n_threads > 1
    cl = UpdateCellList!(from_device(sys.coords), box, cl, aux; parallel=parallel)

    map_pairwise!(
        (x, y, i, j, d2, pairs) -> push_pair!(pairs, i, j, nf.eligible, nf.special),
        neighbors,
        box,
        cl;
        reduce=reduce_pairs,
        output_threaded=neighbors_threaded,
        parallel=parallel,
    )

    nf.cl = cl
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

Unitful.ustrip(nf::NoNeighborFinder) = nf

function Unitful.ustrip(nf::GPUNeighborFinder)
    excluded_pairs, special_pairs = gpu_sparse_pairs(nf)
    return GPUNeighborFinder(
        n_atoms = nf.n_atoms,
        dist_cutoff = ustrip(nf.dist_cutoff),
        excluded_pairs = excluded_pairs,
        special_pairs = special_pairs,
        n_steps_reorder = nf.n_steps_reorder,
        initialized = nf.initialized,
        device_vector_type = typeof(nf.excluded_i),
    )
end

Unitful.ustrip(nf::DistanceNeighborFinder) = DistanceNeighborFinder(
    eligible = nf.eligible,
    dist_cutoff = ustrip(nf.dist_cutoff),
    special = nf.special,
    n_steps = nf.n_steps
)

Unitful.ustrip(nf::TreeNeighborFinder) = TreeNeighborFinder(
    eligible = nf.eligible,
    dist_cutoff = ustrip(nf.dist_cutoff),
    special = nf.special,
    n_steps = nf.n_steps
)

# For CellListMap, we must extract the dimension N from the type signature 
# to properly rebuild the internal CellList buffers without needing dummy coordinates.
Unitful.ustrip(nf::CellListMapNeighborFinder{N, T}) where {N, T} = CellListMapNeighborFinder(
    eligible = nf.eligible,
    dist_cutoff = ustrip(nf.dist_cutoff),
    special = nf.special,
    n_steps = nf.n_steps,
    dims = N 
)
