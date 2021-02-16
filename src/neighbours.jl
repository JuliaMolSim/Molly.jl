# Neighbour finders

export
    NoNeighbourFinder,
    find_neighbours!,
    DistanceNeighbourFinder,
    DistanceNeighbourFinderTree

"""
    NoNeighbourFinder()

Placeholder neighbour finder that returns no neighbours.
When using this neighbour finder, ensure that `nl_only` for the interactions is
set to `false`.
"""
struct NoNeighbourFinder <: NeighbourFinder end

"""
    find_neighbours!(simulation, neighbour_finder, step_n; parallel=true)

Obtain a list of close atoms in a system.
Custom neighbour finders should implement this function.
"""
function find_neighbours!(s::Simulation,
                            ::NoNeighbourFinder,
                            ::Integer;
                            kwargs...)
    return
end

"""
    DistanceNeighbourFinder(nb_matrix, n_steps, dist_cutoff)
    DistanceNeighbourFinder(nb_matrix, n_steps)

Find close atoms by distance.
"""
struct DistanceNeighbourFinder{T} <: NeighbourFinder
    nb_matrix::BitArray{2}
    n_steps::Int
    dist_cutoff::T
end

function DistanceNeighbourFinder(nb_matrix::BitArray{2},
                                 n_steps::Integer)
    return DistanceNeighbourFinder(nb_matrix, n_steps, 1.2)
end

function find_neighbours!(s::Simulation,
                            nf::DistanceNeighbourFinder,
                            step_n::Integer;
                            parallel::Bool=true)
    !iszero(step_n % nf.n_steps) && return

    neighbours = s.neighbours
    empty!(neighbours)
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
            append!(neighbours, nl)
        end
    else
        for i in 1:length(s.coords)
            ci = s.coords[i]
            nbi = nf.nb_matrix[:, i]
            for j in 1:(i - 1)
                r2 = sum(abs2, vector(ci, s.coords[j], s.box_size))
                if r2 <= sqdist_cutoff && nbi[j]
                    push!(neighbours, (i, j))
                end
            end
        end
    end
end


"""
    DistanceNeighbourFinderTree(nb_matrix, n_steps, dist_cutoff)
    DistanceNeighbourFinderTree(nb_matrix, n_steps)

Find close atoms by distance (using a tree search).
"""
struct DistanceNeighbourFinderTree{T} <: NeighbourFinder
    nb_matrix::BitArray{2}
    n_steps::Int
    dist_cutoff::T
end

function DistanceNeighbourFinderTree(nb_matrix::BitArray{2},
                                 n_steps::Integer)
    return DistanceNeighbourFinderTree(nb_matrix, n_steps, 1.2)
end

function find_neighbours!(s::Simulation,
                          nf::DistanceNeighbourFinderTree,
                          step_n::Integer;
                          parallel::Bool=true)
    !iszero(step_n % nf.n_steps) && return

    neighbours = s.neighbours
    empty!(neighbours)

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
            append!(neighbours, nl)
        end
    else
        for i in 1:length(s.coords)
            ci = s.coords[i]
            nbi = nf.nb_matrix[:, i]
            idxs = inrange(btree, ci, nf.dist_cutoff, true)
            for j in idxs
                if nbi[j] && i > j
                    push!(neighbours, (i, j))
                end
            end
        end
    end
end