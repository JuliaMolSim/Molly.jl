# Neighbor finders

export
    use_neighbors,
    NoNeighborFinder,
    find_neighbors,
    DistanceNeighborFinder,
    TreeNeighborFinder,
    CellListMapNeighborFinder,
    GPUNeighborFinder

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
"""
find_neighbors(sys::System; kwargs...) = find_neighbors(sys, sys.neighbor_finder; kwargs...)

find_neighbors(sys::System, nf::NoNeighborFinder, args...; kwargs...) = nothing

"""
    DistanceNeighborFinder(; eligible, dist_cutoff, special, n_steps)

Find close atoms by distance.
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

mutable struct GPUNeighborFinder{B, D}
    eligible::B
    dist_cutoff::D
    special::B
    n_steps_reorder::Int
    initialized::Bool
end

function GPUNeighborFinder(;
                            eligible,
                            dist_cutoff,
                            special=zero(eligible),
                            n_steps_reorder=10,
                            initialized)
    return GPUNeighborFinder{typeof(eligible), typeof(dist_cutoff)}(
                eligible, dist_cutoff, special, n_steps_reorder, false)
end

find_neighbors(sys::System{D, true}, nf::GPUNeighborFinder, current_neighbors=nothing, step_n::Integer=0, initialized::Bool=false; kwargs...) where D = nothing

function find_neighbors(sys::System{D, false},
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where D
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    sqdist_cutoff = nf.dist_cutoff ^ 2

    @floop ThreadedEx(basesize = length(sys) ÷ n_threads) for i in eachindex(sys)
        ci = sys.coords[i]
        nbi = @view nf.eligible[:, i]
        speci = @view nf.special[:, i]
        for j in 1:(i - 1)
            r2 = sum(abs2, vector(ci, sys.coords[j], sys.boundary))
            if r2 <= sqdist_cutoff && nbi[j]
                nn = (Int32(j), Int32(i), speci[j])
                @reduce(neighbors_list = append!(Tuple{Int32, Int32, Bool}[], (nn,)))
            end
        end
    end

    return NeighborList(length(neighbors_list), neighbors_list)
end

function cuda_threads_blocks_dnf(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_DISTANCENF", "512"))
    n_blocks = cld(n_inters, n_threads_gpu)
    return n_threads_gpu, n_blocks
end

function distance_neighbor_finder_kernel!(neighbors, coords_var, eligible_var,
                                          boundary, sq_dist_neighbors)
    coords    = CUDA.Const(coords_var)
    eligible = CUDA.Const(eligible_var)

    n_atoms = length(coords)
    n_inters = n_atoms_to_n_pairs(n_atoms)
    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= n_inters
        i, j = pair_index(n_atoms, inter_i)
        if eligible[i, j]
            dr = vector(coords[i], coords[j], boundary)
            r2 = sum(abs2, dr)
            if r2 <= sq_dist_neighbors
                neighbors[j, i] = true
            end
        end
    end
    return nothing
end

lists_to_tuple_list(i, j, w) = (Int32(i), Int32(j), w)

function find_neighbors(sys::System{D, true},
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        kwargs...) where D
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    nf.neighbors .= false
    n_inters = n_atoms_to_n_pairs(length(sys))
    n_threads_gpu, n_blocks = cuda_threads_blocks_dnf(n_inters)

    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks distance_neighbor_finder_kernel!(
        nf.neighbors, sys.coords, nf.eligible, sys.boundary, nf.dist_cutoff^2,
    )

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
    return TreeNeighborFinder{typeof(dist_cutoff)}(eligible, dist_cutoff, special, n_steps)
end

function find_neighbors(sys::System,
                        nf::TreeNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads())
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    dist_unit = unit(first(first(sys.coords)))
    bv = ustrip.(dist_unit, sys.boundary)
    btree = BallTree(ustrip_vec.(sys.coords), PeriodicEuclidean(bv))
    dist_cutoff = ustrip(dist_unit, nf.dist_cutoff)

    @floop ThreadedEx(basesize = length(sys) ÷ n_threads) for i in eachindex(sys)
        ci = ustrip.(sys.coords[i])
        nbi = @view nf.eligible[:, i]
        speci = @view nf.special[:, i]
        idxs = inrange(btree, ci, dist_cutoff, true)
        for j in idxs
            if nbi[j] && i > j
                nn = (Int32(j), Int32(i), speci[j])
                @reduce(neighbors_list = append!(Tuple{Int32, Int32, Bool}[], (nn,)))
            end
        end
    end

    return NeighborList(length(neighbors_list), move_array(neighbors_list, sys))
end

"""
    CellListMapNeighborFinder(; eligible, dist_cutoff, special, n_steps, x0, unit_cell)

Find close atoms by distance using a cell list algorithm from CellListMap.jl.

`x0` and `unit_cell` are optional initial coordinates and system unit cell that improve the
first approximation of the cell list structure.
Can not be used if one or more dimensions has infinite boundaries.

### Example

```julia-repl
julia> coords
15954-element Vector{SVector{3, Quantity{Float64, 𝐋, Unitful.FreeUnits{(nm,), 𝐋, nothing}}}}:
 [2.5193063341012127 nm, 3.907448346081021 nm, 4.694954671434135 nm]
 [2.4173958848835233 nm, 3.916034913604175 nm, 4.699661024574953 nm]
 ⋮
 [1.818842280373283 nm, 5.592152965227421 nm, 4.992100424805031 nm]
 [1.7261366568663976 nm, 5.610326185704369 nm, 5.084523386833478 nm]

julia> boundary
CubicBoundary{Quantity{Float64, 𝐋, Unitful.FreeUnits{(nm,), 𝐋, nothing}}}(Quantity{Float64, 𝐋, Unitful.FreeUnits{(nm,), 𝐋, nothing}}[5.676 nm, 5.6627 nm, 6.2963 nm])

julia> neighbor_finder = CellListMapNeighborFinder(
           eligible=s.neighbor_finder.eligible, dist_cutoff=1.2u"nm",
           special=s.neighbor_finder.special, n_steps=10,
           x0=coords, unit_cell=boundary,
       )
CellListMapNeighborFinder{Quantity{Float64, 𝐋, Unitful.FreeUnits{(nm,), 𝐋, nothing}}, 3, Float64}
  Size of eligible matrix = (15954, 15954)
  n_steps = 10
  dist_cutoff = 1.2 nm

```
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
                                   number_of_batches=(0, 0)) where T
    np = size(eligible, 1)
    if isnothing(unit_cell)
        twice_cutoff = nextfloat(2 * dist_cutoff)
        if unit(dist_cutoff) == NoUnits
            side = max(twice_cutoff, T((np * 0.01) ^ (1 / 3)))
        else
            side = max(twice_cutoff, uconvert(unit(dist_cutoff), T((np * 0.01u"nm^3") ^ (1 / 3))))
        end
        sides = SVector(side, side, side)
        box = CellListMap.Box(sides, dist_cutoff)
    else
        box = CellListMap.Box(clm_box_arg(unit_cell), dist_cutoff)
    end
    if isnothing(x0)
        x = [ustrip.(diag(box.input_unit_cell.matrix)) .* rand(SVector{3, T}) for _ in 1:np]
    else
        x = x0
    end
    # Construct the cell list for the first time, to allocate
    cl = CellList(x, box; parallel=true, nbatches=number_of_batches)
    return CellListMapNeighborFinder{3, T}(
        eligible, dist_cutoff, special, n_steps,
        cl, CellListMap.AuxThreaded(cl),
        [NeighborList(0, [(Int32(0), Int32(0), false)]) for _ in 1:CellListMap.nbatches(cl)],
    )
end

# Add a pair to the pair list
# If the buffer size is large enough update the element, otherwise push a new element
#   to `neighbor.list`.
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

function find_neighbors(sys::System{D, G},
                        nf::CellListMapNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0,
                        force_recompute::Bool=false;
                        n_threads::Integer=Threads.nthreads()) where {D, G}
    if !force_recompute && !iszero(step_n % nf.n_steps)
        return current_neighbors
    end

    if isnothing(current_neighbors)
        neighbors = NeighborList()
    elseif G
        neighbors = NeighborList(current_neighbors.n, Array(current_neighbors.list))
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
    cl = UpdateCellList!(Array(sys.coords), box, cl, aux; parallel=parallel)

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
    if G
        return NeighborList(neighbors.n, CuArray(neighbors.list))
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

function Base.show(io::IO, neighbor_finder::Union{GPUNeighborFinder})
    println(io, typeof(neighbor_finder))
    println(io, "  Size of eligible matrix = " , size(neighbor_finder.eligible))
    println(io, "  n_steps_reorder = " , neighbor_finder.n_steps_reorder)
    print(  io, "  dist_cutoff = ", neighbor_finder.dist_cutoff)
end
