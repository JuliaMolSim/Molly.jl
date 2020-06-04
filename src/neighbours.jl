# Neighbour finders

export
    NoNeighbourFinder,
    find_neighbours,
    DistanceNeighbourFinder

"Placeholder neighbour finder that returns no neighbours."
struct NoNeighbourFinder <: NeighbourFinder end

function find_neighbours(s::Simulation,
                            current_neighbours,
                            ::NoNeighbourFinder,
                            ::Integer;
                            kwargs...)
    return Tuple{Int, Int}[]
end

"Find close atoms by distance."
struct DistanceNeighbourFinder{T} <: NeighbourFinder
    nb_matrix::BitArray{2}
    n_steps::Int
    dist_cutoff::T
end

function DistanceNeighbourFinder(nb_matrix::BitArray{2},
                                n_steps::Integer)
    return DistanceNeighbourFinder(nb_matrix, n_steps, 1.2)
end

"Update list of close atoms between which non-bonded forces are calculated."
function find_neighbours(s::Simulation,
                            current_neighbours,
                            nf::DistanceNeighbourFinder,
                            step_n::Integer;
                            parallel::Bool=true)
    if step_n % nf.n_steps == 0
        neighbours = Tuple{Int, Int}[]
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
        return neighbours
    else
        return current_neighbours
    end
end
