# Neighbor finders

export
    NoNeighborFinder,
    find_neighbors!,
    DistanceNeighborFinder,
    TreeNeighborFinder

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
    DistanceNeighborFinder(nb_matrix, n_steps, dist_cutoff)
    DistanceNeighborFinder(nb_matrix, n_steps)

Find close atoms by distance.
"""
struct DistanceNeighborFinder{T} <: NeighborFinder
    nb_matrix::BitArray{2}
    n_steps::Int
    dist_cutoff::T
end

function DistanceNeighborFinder(nb_matrix::BitArray{2},
                                 n_steps::Integer)
    return DistanceNeighborFinder(nb_matrix, n_steps, 1.2)
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
        nl_threads = [Tuple{Int, Int}[] for i in 1:nthreads()]

        @threads for i in 1:length(s.coords)
            nl = nl_threads[threadid()]
            ci = s.coords[i]
            nbi = nf.nb_matrix[:, i]
            for j in 1:(i - 1)
                r2 = sum(abs2, vector(ci, s.coords[j], s.box_size))
                if r2 <= sqdist_cutoff && nbi[j]
                    push!(nl, (i, j))
                end
            end
        end

        for nl in nl_threads
            append!(neighbors, nl)
        end
    else
        for i in 1:length(s.coords)
            ci = s.coords[i]
            nbi = nf.nb_matrix[:, i]
            for j in 1:(i - 1)
                r2 = sum(abs2, vector(ci, s.coords[j], s.box_size))
                if r2 <= sqdist_cutoff && nbi[j]
                    push!(neighbors, (i, j))
                end
            end
        end
    end
end


"""
    TreeNeighborFinder(nb_matrix, n_steps, dist_cutoff)
    TreeNeighborFinder(nb_matrix, n_steps)

Find close atoms by distance (using a tree search).
"""
struct TreeNeighborFinder{T} <: NeighborFinder
    nb_matrix::BitArray{2}
    n_steps::Int
    dist_cutoff::T
end

function TreeNeighborFinder(nb_matrix::BitArray{2},
                                 n_steps::Integer)
    return TreeNeighborFinder(nb_matrix, n_steps, 1.2)
end

function find_neighbors!(s::Simulation,
                          nf::TreeNeighborFinder,
                          step_n::Integer;
                          parallel::Bool=true)
    !iszero(step_n % nf.n_steps) && return

    neighbors = s.neighbors
    empty!(neighbors)

    bv = SVector{3}(s.box_size, s.box_size, s.box_size)
    btree = BallTree(s.coords, PeriodicEuclidean(bv))

    if parallel && nthreads() > 1
        nl_threads = [Tuple{Int, Int}[] for i in 1:nthreads()]

        @threads for i in 1:length(s.coords)
            nl = nl_threads[threadid()]
            ci = s.coords[i]
            nbi = nf.nb_matrix[:, i]
            idxs = inrange(btree, ci, nf.dist_cutoff, true)
            for j in idxs
                if nbi[j] && i > j
                    push!(nl, (i, j))
                end
            end
        end

        for nl in nl_threads
            append!(neighbors, nl)
        end
    else
        for i in 1:length(s.coords)
            ci = s.coords[i]
            nbi = nf.nb_matrix[:, i]
            idxs = inrange(btree, ci, nf.dist_cutoff, true)
            for j in idxs
                if nbi[j] && i > j
                    push!(neighbors, (i, j))
                end
            end
        end
    end
end
