# Neighbor finders

export
    NoNeighborFinder,
    find_neighbors!,
    DistanceNeighborFinder,
    TreeNeighborFinder,
    CellListNeighborFinder

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

"""
    CellListNeighborFinder(; nb_matrix, matrix_14, n_steps, dist_cutoff)

Find close atoms using a cell list provided by CellListMap.jl.
"""
struct CellListNeighborFinder{D} <: NeighborFinder
    nb_matrix::BitArray{2}
    matrix_14::BitArray{2}
    n_steps::Int
    dist_cutoff::D
end

function CellListNeighborFinder(;
                                nb_matrix,
                                matrix_14=falses(size(nb_matrix)),
                                n_steps=10,
                                dist_cutoff)
    return CellListNeighborFinder{typeof(dist_cutoff)}(nb_matrix, matrix_14, n_steps, dist_cutoff)
end

function push_pair!(pairs, i, j, nb_matrix, matrix_14)
    if nb_matrix[i, j]
        push!(pairs, (i, j, matrix_14[i, j]))
    end
    return pairs
end

# This is only called in the parallel case
function reduce_pairs(pairs, pairs_threaded)
    for i in 1:nthreads()
        append!(pairs, pairs_threaded[i])
    end
    return pairs
end

function Molly.find_neighbors!(s::Simulation,
                                nf::CellListNeighborFinder,
                                step_n::Integer;
                                parallel::Bool=true)
    !iszero(step_n % nf.n_steps) && return

    neighbors = s.neighbors
    empty!(neighbors)

    dist_unit = unit(first(first(s.coords)))
    box_size_conv = ustrip.(dist_unit, s.box_size)
    dist_cutoff_conv = ustrip(dist_unit, nf.dist_cutoff)

    box = CellListMap.Box(box_size_conv, dist_cutoff_conv; T=typeof(dist_cutoff_conv), lcell=1)
    cl = CellList(ustripvec.(s.coords), box; parallel=parallel)

    neighbors = map_pairwise!(
        (x, y, i, j, d2, pairs) -> push_pair!(pairs, i, j, nf.nb_matrix, nf.matrix_14),
        neighbors, box, cl;
        reduce=reduce_pairs,
        parallel=parallel,
    )
end
