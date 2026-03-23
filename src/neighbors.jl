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

# Indicates whether an array type is compatible with GPUNeighborFinder
uses_gpu_neighbor_finder(AT) = false

"""
    GPUNeighborFinder(; n_atoms, dist_cutoff, dist_neighbors=dist_cutoff,
                      excluded_pairs=(), special_pairs=(), n_steps_reorder=25,
                      initialized=false, device_vector_type)
    GPUNeighborFinder(; eligible, dist_cutoff, dist_neighbors=dist_cutoff,
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
- `dist_neighbors`: cutoff used during tile search. Defaults to
  `dist_cutoff`.
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
mutable struct GPUNeighborFinder{B, D, E}
    n_atoms::B
    dist_cutoff::D
    dist_neighbors::D
    n_steps_reorder::Int
    initialized::Bool
    excluded_i::E
    excluded_j::E
    special_i::E
    special_j::E
end

copy_to_bitmatrix(x::BitMatrix) = copy(x)
copy_to_bitmatrix(x) = BitMatrix(Array(x))

function gpu_exception_vector_type(eligible, device_vector_type)
    if eligible isa AbstractGPUArray
        return typeof(eligible).name.wrapper{Int32, 1}
    end
    if isnothing(device_vector_type)
        throw(ArgumentError("eligible must be on the GPU or device_vector_type must be provided"))
    end
    if !(device_vector_type <: AbstractVector{Int32})
        throw(ArgumentError("device_vector_type must be a 1D Int32 array type, got $device_vector_type"))
    end
    return device_vector_type
end

function normalize_pairs(pairs; allow_diagonal::Bool=false)
    normalized = Tuple{Int32, Int32}[]
    seen = Set{Tuple{Int32, Int32}}()
    for (i, j) in pairs
        i32 = Int32(i)
        j32 = Int32(j)
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

function pair_list_vectors(pairs, ET)
    is = Int32[first(pair) for pair in pairs]
    js = Int32[last(pair) for pair in pairs]
    return Molly.to_device(is, ET), Molly.to_device(js, ET)
end

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

function update_sparse_pairs!(nf::GPUNeighborFinder, excluded_pairs, special_pairs)
    ET = typeof(nf.excluded_i)
    excluded_pairs = normalize_pairs(excluded_pairs)
    special_pairs = normalize_pairs(special_pairs)
    nf.excluded_i, nf.excluded_j = pair_list_vectors(excluded_pairs, ET)
    nf.special_i, nf.special_j = pair_list_vectors(special_pairs, ET)
    nf.initialized = false
    return nf
end

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
                            dist_neighbors=dist_cutoff,
                            excluded_pairs=(),
                            special_pairs=(),
                            special=nothing,
                            n_steps_reorder=25,
                            initialized=false,
                            device_vector_type=nothing)
    if !isnothing(n_atoms)
        if !(device_vector_type <: AbstractVector{Int32})
            throw(ArgumentError("device_vector_type must be a 1D Int32 array type, got $device_vector_type"))
        end
        excluded_pairs_norm = normalize_pairs(excluded_pairs)
        special_pairs_norm = normalize_pairs(special_pairs)
        excluded_i, excluded_j = pair_list_vectors(excluded_pairs_norm, device_vector_type)
        special_i, special_j = pair_list_vectors(special_pairs_norm, device_vector_type)
        return GPUNeighborFinder{Int, typeof(dist_cutoff), typeof(excluded_i)}(
                    Int(n_atoms), dist_cutoff, dist_neighbors, n_steps_reorder, initialized,
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
        dist_neighbors=dist_neighbors,
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
