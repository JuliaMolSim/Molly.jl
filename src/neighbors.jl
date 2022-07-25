# Neighbor finders

export
    NoNeighborFinder,
    find_neighbors,
    DistanceNeighborFinder,
    DistanceVecNeighborFinder,
    TreeNeighborFinder,
    CellListMapNeighborFinder

"""
    NoNeighborFinder()

Placeholder neighbor finder that returns no neighbors.
When using this neighbor finder, ensure that `nl_only` for the interactions is
set to `false`.
"""
struct NoNeighborFinder <: AbstractNeighborFinder end

"""
    find_neighbors(system; n_threads = Threads.nthreads())
    find_neighbors(system, neighbor_finder, current_neighbors=nothing,
                    step_n=0; n_threads = Threads.nthreads())

Obtain a list of close atoms in a [`System`](@ref).
Custom neighbor finders should implement this function.
"""
find_neighbors(s::System; kwargs...) = find_neighbors(s, s.neighbor_finder; kwargs...)

function find_neighbors(s::System{D, false},
                        nf::NoNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0;
                        kwargs...) where D
    return nothing
end

function find_neighbors(s::System{D, true},
                        nf::NoNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0;
                        kwargs...) where D
    step_n > 0 && return current_neighbors
    all_pairs = all_neighbors(length(s))
    return NeighborListVec(NeighborsVec(), all_pairs)
end

Base.show(io::IO, neighbor_finder::NoNeighborFinder) = print(io, typeof(neighbor_finder))

"""
    DistanceNeighborFinder(; nb_matrix, matrix_14, n_steps, dist_cutoff)

Find close atoms by distance.
"""
struct DistanceNeighborFinder{D} <: AbstractNeighborFinder
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

function find_neighbors(s::System,
                        nf::DistanceNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0;
                        n_threads::Integer=Threads.nthreads())
    !iszero(step_n % nf.n_steps) && return current_neighbors

    sqdist_cutoff = nf.dist_cutoff ^ 2

    @floop ThreadedEx(basesize = length(s) ÷ n_threads) for i in 1:length(s)
        ci = s.coords[i]
        nbi = @view nf.nb_matrix[:, i]
        w14i = @view nf.matrix_14[:, i]
        for j in 1:(i - 1)
            r2 = sum(abs2, vector(ci, s.coords[j], s.boundary))
            if r2 <= sqdist_cutoff && nbi[j]
                nn = (i, j, w14i[j])
                @reduce(neighbors_list = append!(Tuple{Int, Int, Bool}[], (nn,)))
            end
        end
    end

    return NeighborList(length(neighbors_list), neighbors_list)
end

"""
    DistanceVecNeighborFinder(; nb_matrix, matrix_14, n_steps, dist_cutoff)

Find close atoms by distance in a GPU and Zygote compatible manner.
"""
struct DistanceVecNeighborFinder{D, B, I} <: AbstractNeighborFinder
    nb_matrix::B
    matrix_14::B
    n_steps::Int
    dist_cutoff::D
    is::I
    js::I
end

function DistanceVecNeighborFinder(;
                                nb_matrix,
                                matrix_14=falses(size(nb_matrix)),
                                n_steps=10,
                                dist_cutoff)
    n_atoms = size(nb_matrix, 1)
    if isa(nb_matrix, CuArray)
        is = CuArray(hcat([collect(1:n_atoms) for i in 1:n_atoms]...))
        js = CuArray(permutedims(is, (2, 1)))
        m14 = CuArray(matrix_14)
    else
        is = hcat([collect(1:n_atoms) for i in 1:n_atoms]...)
        js = permutedims(is, (2, 1))
        m14 = matrix_14
    end
    return DistanceVecNeighborFinder{typeof(dist_cutoff), typeof(nb_matrix), typeof(is)}(
            nb_matrix, m14, n_steps, dist_cutoff, is, js)
end

# Find the boundaries of an ordered list of integers
function find_boundaries(nbs_ord, n_atoms)
    inds = zeros(Int, n_atoms)
    atom_i = 1
    for (nb_i, nb_ai) in enumerate(nbs_ord)
        while atom_i < nb_ai
            inds[atom_i] = nb_i
            atom_i += 1
        end
    end
    while atom_i < (n_atoms + 1)
        inds[atom_i] = length(nbs_ord) + 1
        atom_i += 1
    end
    return inds
end

function find_neighbors(s::System,
                        nf::DistanceVecNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0;
                        kwargs...)
    !iszero(step_n % nf.n_steps) && return current_neighbors

    n_atoms = length(s)
    if any(inter -> inter.nl_only, values(s.pairwise_inters))
        sqdist_cutoff = nf.dist_cutoff ^ 2
        sqdists = square_distance.(nf.is, nf.js, (s.coords,), (s.boundary,))

        close = sqdists .< sqdist_cutoff
        close_nb = close .* nf.nb_matrix
        eligible = tril(close_nb, -1)

        fa = Array(findall(!iszero, eligible))
        nbsi, nbsj = getindex.(fa, 1), getindex.(fa, 2)
        order_i = sortperm(nbsi)
        weights_14 = @view nf.matrix_14[fa]

        nbsi_ordi, nbsj_ordi = nbsi[order_i], nbsj[order_i]
        sortperm_j = sortperm(nbsj_ordi)
        weights_14_ordi = @view weights_14[order_i]
        atom_bounds_i = find_boundaries(nbsi_ordi, n_atoms)
        atom_bounds_j = find_boundaries(view(nbsj_ordi, sortperm_j), n_atoms)

        close_pairs = NeighborsVec{typeof(weights_14_ordi)}(nbsi_ordi, nbsj_ordi,
                        atom_bounds_i, atom_bounds_j, sortperm_j, weights_14_ordi)
    else
        close_pairs = NeighborsVec()
    end

    if any(inter -> !inter.nl_only, values(s.pairwise_inters))
        all_pairs = all_neighbors(n_atoms)
    else
        all_pairs = NeighborsVec()
    end

    return NeighborListVec(close_pairs, all_pairs)
end

function all_neighbors(n_atoms)
    nbs_all = findall(!iszero, tril(ones(Bool, n_atoms, n_atoms), -1))
    nbsi_all, nbsj_all = getindex.(nbs_all, 1), getindex.(nbs_all, 2)
    order_i_all = sortperm(nbsi_all)

    nbsi_ordi_all, nbsj_ordi_all = nbsi_all[order_i_all], nbsj_all[order_i_all]
    sortperm_j_all = sortperm(nbsj_ordi_all)
    atom_bounds_i_all = find_boundaries(nbsi_ordi_all, n_atoms)
    atom_bounds_j_all = find_boundaries(view(nbsj_ordi_all, sortperm_j_all), n_atoms)

    all_pairs = NeighborsVec{Nothing}(nbsi_ordi_all, nbsj_ordi_all, atom_bounds_i_all,
                    atom_bounds_j_all, sortperm_j_all, nothing)
end

"""
    TreeNeighborFinder(; nb_matrix, matrix_14, n_steps, dist_cutoff)

Find close atoms by distance using a tree search.
Can not be used if one or more dimensions has infinite boundaries.
"""
struct TreeNeighborFinder{D} <: AbstractNeighborFinder
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

function find_neighbors(s::System,
                        nf::TreeNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0;
                        n_threads::Integer=Threads.nthreads())
    !iszero(step_n % nf.n_steps) && return current_neighbors

    dist_unit = unit(first(first(s.coords)))
    bv = ustrip.(dist_unit, s.boundary)
    btree = BallTree(ustrip_vec.(s.coords), PeriodicEuclidean(bv))
    dist_cutoff = ustrip(dist_unit, nf.dist_cutoff)

    @floop ThreadedEx(basesize = length(s) ÷ n_threads) for i in 1:length(s)
        ci = ustrip.(s.coords[i])
        nbi = @view nf.nb_matrix[:, i]
        w14i = @view nf.matrix_14[:, i]
        idxs = inrange(btree, ci, dist_cutoff, true)
        for j in idxs
            if nbi[j] && i > j
                nn = (i, j, w14i[j])
                @reduce(neighbors_list = append!(Tuple{Int, Int, Bool}[], (nn,)))
            end
        end
    end

    return NeighborList(length(neighbors_list), neighbors_list)
end

"""
    CellListMapNeighborFinder(; nb_matrix, matrix_14, n_steps, dist_cutoff, x0, unit_cell)

Find close atoms by distance and store auxiliary arrays for in-place threading. `x0` and `unit_cell` 
are optional initial coordinates and system unit cell that improve the first approximation of the
cell list structure. The unit cell can be provided as a three-component vector of box sides on each
direction, in which case the unit cell is considered `OrthorhombicCell`, or as a unit cell matrix,
in which case the cell is considered a general `TriclinicCell` by the cell list algorithm.
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
           nb_matrix=s.neighbor_finder.nb_matrix, matrix_14=s.neighbor_finder.matrix_14, 
           n_steps=10, dist_cutoff=1.2u"nm",
           x0=coords, unit_cell=boundary,
       )
CellListMapNeighborFinder{Quantity{Float64, 𝐋, Unitful.FreeUnits{(nm,), 𝐋, nothing}}, 3, Float64}
  Size of nb_matrix = (15954, 15954)
  n_steps = 10
  dist_cutoff = 1.2 nm

```
"""
mutable struct CellListMapNeighborFinder{N, T} <: AbstractNeighborFinder
    nb_matrix::BitArray{2}
    matrix_14::BitArray{2}
    n_steps::Int
    dist_cutoff::T
    # Auxiliary arrays for multi-threaded in-place updating of the lists
    cl::CellListMap.CellList{N, T}
    aux::CellListMap.AuxThreaded{N, T}
    neighbors_threaded::Vector{NeighborList}
end

# This function sets up the box structure for CellListMap. It uses the unit cell
# if it is given, or guesses a box size from the number of particles, assuming 
# that the atomic density is similar to that of liquid water at ambient conditions.
function CellListMapNeighborFinder(;
                                   nb_matrix,
                                   matrix_14=falses(size(nb_matrix)),
                                   n_steps=10,
                                   x0=nothing,
                                   unit_cell=nothing,
                                   number_of_batches=(0, 0), # (0, 0): use default heuristic
                                   dist_cutoff::T) where T
    np = size(nb_matrix, 1)
    if isnothing(unit_cell)
        if unit(dist_cutoff) == NoUnits
            side = max(2 * dist_cutoff, (np * 0.01) ^ (1 / 3))
        else
            side = max(2 * dist_cutoff, uconvert(unit(dist_cutoff), (np * 0.01u"nm^3") ^ (1 / 3)))
        end
        sides = SVector(side, side, side)
        box = CellListMap.Box(sides, dist_cutoff)
    else
        box = CellListMap.Box(unit_cell.side_lengths, dist_cutoff)
    end
    if isnothing(x0)
        x = [ustrip.(box.unit_cell_max) .* rand(SVector{3, T}) for _ in 1:np]
    else
        x = x0
    end
    # Construct the cell list for the first time, to allocate 
    cl = CellList(x, box; parallel=true, nbatches=number_of_batches)
    return CellListMapNeighborFinder{3, T}(
        nb_matrix, matrix_14, n_steps, dist_cutoff,
        cl, CellListMap.AuxThreaded(cl), 
        [NeighborList(0, [(0, 0, false)]) for _ in 1:CellListMap.nbatches(cl)],
    )
end

"""
    push_pair!(neighbor::NeighborList, i, j, nb_matrix, matrix_14)

Add pair to pair list. If the buffer size is large enough, update element, otherwise
push new element to `neighbor.list`.
"""
function push_pair!(neighbors::NeighborList, i, j, nb_matrix, matrix_14)
    if nb_matrix[i, j]
        push!(neighbors, (Int(i), Int(j), matrix_14[i, j]))
    end
    return neighbors
end

# This is only called in the parallel case
function reduce_pairs(neighbors::NeighborList, neighbors_threaded::Vector{NeighborList})
    neighbors.n = 0
    for i in 1:length(neighbors_threaded)
        append!(neighbors, neighbors_threaded[i])
    end
    return neighbors
end

function find_neighbors(s::System,
                        nf::CellListMapNeighborFinder,
                        current_neighbors=nothing,
                        step_n::Integer=0;
                        n_threads=Threads.nthreads())
    !iszero(step_n % nf.n_steps) && return current_neighbors

    if isnothing(current_neighbors)
        neighbors = NeighborList()
    else
        neighbors = current_neighbors
    end
    aux = nf.aux
    cl = nf.cl
    neighbors.n = 0
    neighbors_threaded = nf.neighbors_threaded
    if n_threads > 1
        for i in 1:length(neighbors_threaded)
            neighbors_threaded[i].n = 0
        end
    else
        neighbors_threaded[1].n = 0
    end

    box = CellListMap.Box(s.boundary.side_lengths, nf.dist_cutoff; lcell=1)
    parallel = n_threads > 1
    cl = UpdateCellList!(s.coords, box, cl, aux; parallel=parallel)

    map_pairwise!(
        (x, y, i, j, d2, pairs) -> push_pair!(pairs, i, j, nf.nb_matrix, nf.matrix_14),
        neighbors, box, cl;
        reduce=reduce_pairs,
        output_threaded=neighbors_threaded,
        parallel=parallel,
    )

    nf.cl = cl
    return neighbors
end

function Base.show(io::IO, neighbor_finder::AbstractNeighborFinder)
    println(io, typeof(neighbor_finder))
    println(io, "  Size of nb_matrix = " , size(neighbor_finder.nb_matrix))
    println(io, "  n_steps = " , neighbor_finder.n_steps)
    print(  io, "  dist_cutoff = ", neighbor_finder.dist_cutoff)
end
