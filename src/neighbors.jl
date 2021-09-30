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

Find close atoms by distance.
"""
struct DistanceNeighborFinder{W, D} <: NeighborFinder
    nb_matrix::Array{W, 2}
    n_steps::Int
    dist_cutoff::D
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
        nl_threads = [Tuple{Int, Int, eltype(nf.nb_matrix)}[] for i in 1:nthreads()]

        @threads for i in 1:length(s.coords)
            nl = nl_threads[threadid()]
            ci = s.coords[i]
            nbi = @view nf.nb_matrix[:, i]
            for j in 1:(i - 1)
                r2 = sum(abs2, vector(ci, s.coords[j], s.box_size))
                if r2 <= sqdist_cutoff && !iszero(nbi[j])
                    push!(nl, (i, j, nbi[j]))
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
            for j in 1:(i - 1)
                r2 = sum(abs2, vector(ci, s.coords[j], s.box_size))
                if r2 <= sqdist_cutoff && !iszero(nbi[j])
                    push!(neighbors, (i, j, nbi[j]))
                end
            end
        end
    end
end

"""
    TreeNeighborFinder(nb_matrix, n_steps, dist_cutoff)

Find close atoms by distance using a tree search.
"""
struct TreeNeighborFinder{W, D} <: NeighborFinder
    nb_matrix::Array{W, 2}
    n_steps::Int
    dist_cutoff::D
end

function find_neighbors!(s::Simulation,
                          nf::TreeNeighborFinder,
                          step_n::Integer;
                          parallel::Bool=true)
    !iszero(step_n % nf.n_steps) && return

    neighbors = s.neighbors
    empty!(neighbors)

    dist_unit = unit(first(first(s.coords)))
    box_size = ustrip(dist_unit, s.box_size)
    bv = SVector{3}(box_size, box_size, box_size)
    btree = BallTree(ustripvec.(s.coords), PeriodicEuclidean(bv))
    dist_cutoff = ustrip(dist_unit, nf.dist_cutoff)

    if parallel && nthreads() > 1
        nl_threads = [Tuple{Int, Int, eltype(nf.nb_matrix)}[] for i in 1:nthreads()]

        @threads for i in 1:length(s.coords)
            nl = nl_threads[threadid()]
            ci = ustrip.(s.coords[i])
            nbi = @view nf.nb_matrix[:, i]
            idxs = inrange(btree, ci, dist_cutoff, true)
            for j in idxs
                if !iszero(nbi[j]) && i > j
                    push!(nl, (i, j, nbi[j]))
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
            idxs = inrange(btree, ci, dist_cutoff, true)
            for j in idxs
                if !iszero(nbi[j]) && i > j
                    push!(neighbors, (i, j, nbi[j]))
                end
            end
        end
    end
end
