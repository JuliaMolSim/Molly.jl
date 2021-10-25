# Neighbor finders

export
    NoNeighborFinder,
    find_neighbors!,
    DistanceNeighborFinder,
    TreeNeighborFinder,
    CellListMapNeighborFinder

"""
    NoNeighborFinder()

Placeholder neighbor finder that returns no neighbors.
When using this neighbor finder, ensure that `nl_only` for the interactions is
set to `false`.
"""
struct NoNeighborFinder <: NeighborFinder end

"""
    find_neighbors!(simulation, neighbor_finder, step_n; parallel=true)

Obtain a list of close atoms in a system.
Custom neighbor finders should implement this function.
"""
function find_neighbors!(s::Simulation,
                         ::NoNeighborFinder,
                         ::Integer;
                         kwargs...)
    return
end

"""
    DistanceNeighborFinder(; nb_matrix, matrix_14, n_steps, dist_cutoff)

Find close atoms by distance.
"""
struct DistanceNeighborFinder{D} <: NeighborFinder
    nb_matrix::BitArray{2}
    matrix_14::BitArray{2}
    n_steps::Int
    dist_cutoff::D
end

function DistanceNeighborFinder(;
                                nb_matrix,
                                matrix_14=falses(size(nb_matrix)),
                                n_steps=10,
                                dist_cutoff)
    return DistanceNeighborFinder{typeof(dist_cutoff)}(nb_matrix, matrix_14, n_steps, dist_cutoff)
end

function find_neighbors!(s::Simulation,
                         nf::DistanceNeighborFinder,
                         step_n::Integer;
                         parallel::Bool=true)
    !iszero(step_n % nf.n_steps) && return

    neighbors = s.neighbors
    empty!(neighbors)

    sqdist_cutoff = nf.dist_cutoff ^ 2

    if parallel && nthreads() > 1
        nl_threads = [Tuple{Int, Int, Bool}[] for i in 1:nthreads()]

        @threads for i in 1:length(s.coords)
            nl = nl_threads[threadid()]
            ci = s.coords[i]
            nbi = @view nf.nb_matrix[:, i]
            w14i = @view nf.matrix_14[:, i]
            for j in 1:(i - 1)
                r2 = sum(abs2, vector(ci, s.coords[j], s.box_size))
                if r2 <= sqdist_cutoff && nbi[j]
                    push!(nl, (i, j, w14i[j]))
                end
            end
        end

        for nl in nl_threads
            append!(neighbors, nl)
        end
    else
        for i in 1:length(s.coords)
            ci = s.coords[i]
            nbi = @view nf.nb_matrix[:, i]
            w14i = @view nf.matrix_14[:, i]
            for j in 1:(i - 1)
                r2 = sum(abs2, vector(ci, s.coords[j], s.box_size))
                if r2 <= sqdist_cutoff && nbi[j]
                    push!(neighbors, (i, j, w14i[j]))
                end
            end
        end
    end
end

"""
    TreeNeighborFinder(; nb_matrix, matrix_14, n_steps, dist_cutoff)

Find close atoms by distance using a tree search.
"""
struct TreeNeighborFinder{D} <: NeighborFinder
    nb_matrix::BitArray{2}
    matrix_14::BitArray{2}
    n_steps::Int
    dist_cutoff::D
end

function TreeNeighborFinder(;
                            nb_matrix,
                            matrix_14=falses(size(nb_matrix)),
                            n_steps=10,
                            dist_cutoff)
    return TreeNeighborFinder{typeof(dist_cutoff)}(nb_matrix, matrix_14, n_steps, dist_cutoff)
end

function find_neighbors!(s::Simulation,
                         nf::TreeNeighborFinder,
                         step_n::Integer;
                         parallel::Bool=true)
    !iszero(step_n % nf.n_steps) && return

    neighbors = s.neighbors
    empty!(neighbors)

    dist_unit = unit(first(first(s.coords)))
    bv = ustrip.(dist_unit, s.box_size)
    btree = BallTree(ustripvec.(s.coords), PeriodicEuclidean(bv))
    dist_cutoff = ustrip(dist_unit, nf.dist_cutoff)

    if parallel && nthreads() > 1
        nl_threads = [Tuple{Int, Int, Bool}[] for i in 1:nthreads()]

        @threads for i in 1:length(s.coords)
            nl = nl_threads[threadid()]
            ci = ustrip.(s.coords[i])
            nbi = @view nf.nb_matrix[:, i]
            w14i = @view nf.matrix_14[:, i]
            idxs = inrange(btree, ci, dist_cutoff, true)
            for j in idxs
                if nbi[j] && i > j
                    push!(nl, (i, j, w14i[j]))
                end
            end
        end

        for nl in nl_threads
            append!(neighbors, nl)
        end
    else
        for i in 1:length(s.coords)
            ci = ustrip.(s.coords[i])
            nbi = @view nf.nb_matrix[:, i]
            w14i = @view nf.matrix_14[:, i]
            idxs = inrange(btree, ci, dist_cutoff, true)
            for j in idxs
                if nbi[j] && i > j
                    push!(neighbors, (i, j, w14i[j]))
                end
            end
        end
    end
end

#
# Find neighbor lists using CellListMap.jl
#
"""
    CellListMapNeighborFinder(; nb_matrix, matrix_14, n_steps, dist_cutoff)

Find close atoms by distance, and store auxiliary arrays for in-place threading.
"""
mutable struct CellListMapNeighborFinder{D,N,T} <: NeighborFinder
    nb_matrix::BitArray{2}
    matrix_14::BitArray{2}
    n_steps::Int
    dist_cutoff::D
    # auxiliary arrays for multi-threaded in-place updating of the lists
    cl::CellListMap.CellList{N,T}
    aux::CellListMap.AuxThreaded{N,T}
    neighbors_threaded::Vector{NeighborList}
end
function Base.show(io::IO,::MIME"text/plain",neighbor_finder::NeighborFinder)
    println(io,typeof(neighbor_finder))
    println(io,"  Size of nb_matrix = " , size(neighbor_finder.nb_matrix))
    println(io,"  n_steps = " , neighbor_finder.n_steps)
    print(io,"  dist_cutoff = ", neighbor_finder.dist_cutoff)
end

function CellListMapNeighborFinder(;
    nb_matrix,
    matrix_14=falses(size(nb_matrix)),
    n_steps=10,
    dist_cutoff::D) where D
    cutoff = ustrip(dist_cutoff)
    T = typeof(cutoff)
    # Ideally one would want the coordinates of the particles and box size of the
    # initial system to properly initalize the cell lists here. 
    # Number of particles
    np = size(nb_matrix,1)
    # guess box side from number of particles, just to provide a reasonable estimate 
    # of the side of the cell lists: water has roughly 100 atoms per nm^3
    side = (T(ustrip(uconvert(unit(dist_cutoff^3),np * 0.01u"nm^3"))))^(1/3)
    # If very few particles are used, use the minimum box side that is accepted
    # for this cutoff
    side = max(2*cutoff,side)
    cl = CellList(
        [ side*rand(SVector{3,T}) for _ in 1:np ],
        Box(side*ones(SVector{3,T}),cutoff;T=T),
        parallel=true
    ) # will be overwritten
    return CellListMapNeighborFinder{D,3,T}(
        nb_matrix, matrix_14, n_steps, dist_cutoff,
        cl, CellListMap.AuxThreaded(cl), 
        [ NeighborList(0,[(0,0,false)]) for _ in 1:nthreads() ] 
    )
end

"""

```
push_pair!(neighbor::NeighborList, i, j, nb_matrix, matrix_14)
```

Add pair to pair list. If the buffer size is large enough, update element, otherwise
push new element to `neighbor.list`

"""
function push_pair!(neighbors::NeighborList, i, j, nb_matrix, matrix_14)
    if nb_matrix[i, j]
        push!(neighbors,(i, j, matrix_14[i, j]))
    end
    return neighbors
end

# This is only called in the parallel case
function reduce_pairs(neighbors::NeighborList, neighbors_threaded::Vector{NeighborList})
    neighbors.n = 0
    for i in 1:nthreads()
        append!(neighbors,neighbors_threaded[i])
    end
    return neighbors
end

# Add method to strip_value from cell list map to pass the 
# coordinates with units without having to reallocate the vector 
# requires CellListMap >= 0.54 
CellListMap.strip_value(x::Unitful.Quantity) = Unitful.ustrip(x)

"""
```
Molly.find_neighbors!(s::Simulation,
                      nf::CellListMapNeighborFinder,
                      step_n::Integer;
                      parallel::Bool=true)
```

Find neighbors using `CellListMap`, without in-place updating. Should be called only
the first time the cell lists are built. Modifies the *mutable* `nf` structure.

"""
function Molly.find_neighbors!(s::Simulation,
                               nf::CellListMapNeighborFinder,
                               step_n::Integer;
                               parallel::Bool=true)
    !iszero(step_n % nf.n_steps) && return

    aux = nf.aux
    cl = nf.cl

    neighbors = s.neighbors
    neighbors.n = 0
    neighbors_threaded = nf.neighbors_threaded
    for i in 1:nthreads()
        neighbors_threaded[i].n = 0
    end

    dist_unit = unit(first(first(s.coords)))
    box_size_conv = ustrip.(dist_unit, s.box_size)
    dist_cutoff_conv = ustrip(dist_unit, nf.dist_cutoff)

    box = CellListMap.Box(box_size_conv, dist_cutoff_conv; T=typeof(dist_cutoff_conv), lcell=1)
    cl = UpdateCellList!(s.coords, box, cl, aux; parallel=parallel)

    map_pairwise!(
        (x, y, i, j, d2, pairs) -> push_pair!(pairs, i, j, nf.nb_matrix, nf.matrix_14),
        neighbors, box, cl;
        reduce=reduce_pairs,
        output_threaded=neighbors_threaded,
        parallel=parallel
    )

    nf.cl = cl
    return neighbors
end
