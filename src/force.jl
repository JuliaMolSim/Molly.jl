# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See OpenMM documentation and Gromacs manual for other aspects of forces

export
    ustrip_vec,
    accelerations,
    force,
    SpecificForce1Atoms,
    SpecificForce2Atoms,
    SpecificForce3Atoms,
    SpecificForce4Atoms,
    forces

"""
    ustrip_vec(x)

Broadcasted form of `ustrip` from Unitful.jl, allowing e.g. `ustrip_vec.(coords)`.
"""
ustrip_vec(x...) = ustrip.(x...)

function check_force_units(fdr, force_units)
    if unit(first(fdr)) != force_units
        error("System force units are ", force_units, " but encountered force units ",
                unit(first(fdr)))
    end
end

"""
    accelerations(system, neighbors=nothing; n_threads=Threads.nthreads())

Calculate the accelerations of all atoms using the pairwise, specific and
general interactions and Newton's second law of motion.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.
"""
function accelerations(s, neighbors=nothing; n_threads::Integer=Threads.nthreads())
    return forces(s, neighbors; n_threads=n_threads) ./ masses(s)
end

"""
    force(inter::PairwiseInteraction, vec_ij, coord_i, coord_j,
          atom_i, atom_j, boundary)
    force(inter::PairwiseInteraction, vec_ij, coord_i, coord_j,
          atom_i, atom_j, boundary, weight_14)
    force(inter::SpecificInteraction, coord_i, coord_j,
          boundary)
    force(inter::SpecificInteraction, coord_i, coord_j,
          coord_k, boundary)
    force(inter::SpecificInteraction, coord_i, coord_j,
          coord_k, coord_l, boundary)

Calculate the force between atoms due to a given interation type.
For [`PairwiseInteraction`](@ref)s returns a single force vector and for
[`SpecificInteraction`](@ref)s returns a type such as [`SpecificForce2Atoms`](@ref).
Custom pairwise and specific interaction types should implement this function.
"""
function force(inter, dr, coord_i, coord_j, atom_i, atom_j, boundary, weight_14)
    # Fallback for interactions where the 1-4 weighting is not relevant
    return force(inter, dr, coord_i, coord_j, atom_i, atom_j, boundary)
end

@inline @inbounds function force!(fs, inter, s::System, i::Integer, j::Integer, force_units, weight_14::Bool=false)
    dr = vector(s.coords[i], s.coords[j], s.boundary)
    fdr = force(inter, dr, s.coords[i], s.coords[j], s.atoms[i], s.atoms[j], s.boundary, weight_14)
    check_force_units(fdr, force_units)
    fdr_ustrip = ustrip.(fdr)
    fs[i] -= fdr_ustrip
    fs[j] += fdr_ustrip
    return nothing
end

"""
    SpecificForce1Atoms(f1)

Force on one atom arising from an interaction such as a position restraint.
"""
struct SpecificForce1Atoms{D, T}
    f1::SVector{D, T}
end

"""
    SpecificForce2Atoms(f1, f2)

Forces on two atoms arising from an interaction such as a bond potential.
"""
struct SpecificForce2Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
end

"""
    SpecificForce3Atoms(f1, f2, f3)

Forces on three atoms arising from an interaction such as a bond angle potential.
"""
struct SpecificForce3Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
end

"""
    SpecificForce4Atoms(f1, f2, f3, f4)

Forces on four atoms arising from an interaction such as a torsion potential.
"""
struct SpecificForce4Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
    f4::SVector{D, T}
end

function SpecificForce1Atoms(f1::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce1Atoms{D, T}(f1)
end

function SpecificForce2Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce2Atoms{D, T}(f1, f2)
end

function SpecificForce3Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                            f3::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce3Atoms{D, T}(f1, f2, f3)
end

function SpecificForce4Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                            f3::StaticArray{Tuple{D}, T}, f4::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce4Atoms{D, T}(f1, f2, f3, f4)
end

Base.:+(x::SpecificForce1Atoms, y::SpecificForce1Atoms) = SpecificForce1Atoms(x.f1 + y.f1)
Base.:+(x::SpecificForce2Atoms, y::SpecificForce2Atoms) = SpecificForce2Atoms(x.f1 + y.f1, x.f2 + y.f2)
Base.:+(x::SpecificForce3Atoms, y::SpecificForce3Atoms) = SpecificForce3Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3)
Base.:+(x::SpecificForce4Atoms, y::SpecificForce4Atoms) = SpecificForce4Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3, x.f4 + y.f4)

get_f1(x) = x.f1
get_f2(x) = x.f2
get_f3(x) = x.f3
get_f4(x) = x.f4

@views function specific_force(inter_list::InteractionList1Atoms, coords, boundary, force_units, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], (boundary,)))
    fis = get_f1.(sparse_fs)
    check_force_units(first(first(fis)), force_units)
    return sparsevec(inter_list.is, fis, n_atoms)
end

@views function specific_force(inter_list::InteractionList2Atoms, coords, boundary, force_units, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js], (boundary,)))
    fis, fjs = get_f1.(sparse_fs), get_f2.(sparse_fs)
    check_force_units(first(first(fis)), force_units)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms)
end

@views function specific_force(inter_list::InteractionList3Atoms, coords, boundary, force_units, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                coords[inter_list.ks], (boundary,)))
    fis, fjs, fks = get_f1.(sparse_fs), get_f2.(sparse_fs), get_f3.(sparse_fs)
    check_force_units(first(first(fis)), force_units)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms) +
           sparsevec(inter_list.ks, fks, n_atoms)
end

@views function specific_force(inter_list::InteractionList4Atoms, coords, boundary, force_units, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                coords[inter_list.ks], coords[inter_list.ls], (boundary,)))
    fis, fjs, fks, fls = get_f1.(sparse_fs), get_f2.(sparse_fs), get_f3.(sparse_fs), get_f4.(sparse_fs)
    check_force_units(first(first(fis)), force_units)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms) +
           sparsevec(inter_list.ks, fks, n_atoms) + sparsevec(inter_list.ls, fls, n_atoms)
end

"""
    forces(system, neighbors=nothing; n_threads=Threads.nthreads())

Calculate the forces on all atoms in the system using the pairwise, specific and
general interactions.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.

    forces(inter, system, neighbors=nothing)

Calculate the forces on all atoms in the system arising from a general
interaction.
If the interaction uses neighbor lists, the neighbors should be computed
first and passed to the function.
Custom general interaction types should implement this function.
"""
function forces(s::System{D, false}, neighbors=nothing;
                n_threads::Integer=Threads.nthreads()) where D
    n_atoms = length(s)

    if n_threads > 1
        fs_chunks = [ustrip_vec.(zero(s.coords)) for i in 1:n_threads]
        for inter in values(s.pairwise_inters)
            if inter.nl_only
                if isnothing(neighbors)
                    error("An interaction uses the neighbor list but neighbors is nothing")
                end
                basesize = max(1, Int(ceil(neighbors.n / n_threads)))
                chunks = [i:min(i + basesize - 1, neighbors.n) for i in 1:basesize:neighbors.n]
                Threads.@threads for (id, rng) in collect(enumerate(chunks))
                    for ni in rng
                        i, j, w = neighbors.list[ni]
                        force!(fs_chunks[id], inter, s, i, j, s.force_units, w)
                    end
                end
            else
                basesize = max(1, Int(ceil(n_atoms / n_threads)))
                chunks = [i:min(i + basesize - 1, n_atoms) for i in 1:basesize:n_atoms]
                Threads.@threads for (id, rng) in collect(enumerate(chunks))
                    for i in rng
                        for j in 1:(i - 1)
                            force!(fs_chunks[id], inter, s, i, j, s.force_units)
                        end
                    end
                end
            end
        end
        fs = sum(fs_chunks)
    else
        fs = ustrip_vec.(zero(s.coords))
        for inter in values(s.pairwise_inters)
            if inter.nl_only
                if isnothing(neighbors)
                    error("An interaction uses the neighbor list but neighbors is nothing")
                end
                for ni in 1:neighbors.n
                    i, j, w = neighbors.list[ni]
                    force!(fs, inter, s, i, j, s.force_units, w)
                end
            else
                for i in 1:n_atoms
                    for j in 1:(i - 1)
                        force!(fs, inter, s, i, j, s.force_units)
                    end
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        sparse_vec = specific_force(inter_list, s.coords, s.boundary, s.force_units, n_atoms)
        fs += ustrip_vec.(Array(sparse_vec))
    end

    for inter in values(s.general_inters)
        # Force type not checked
        fs += ustrip_vec.(forces(inter, s, neighbors))
    end

    return fs * s.force_units
end

function forces(s::System{D, true, T}, neighbors=nothing; n_threads::Integer=Threads.nthreads()) where {D, T}
    fs = ustrip_vec.(zero(s.coords))
    n_atoms = length(s)
    fs_mat = CuArray(zeros(T, 3, n_atoms))
    virial = CuArray(zeros(T, 1))

    pairwise_inters_nonl = filter(inter -> !inter.nl_only, values(s.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        CUDA.@sync @cuda threads=256 blocks=4000 pairwise_force_kernel!(
                fs_mat, virial, s.coords, s.atoms, s.boundary, pairwise_inters_nonl,
                NoNeighborList(n_atoms), Val(s.force_units), Val(2000))
    end

    pairwise_inters_nl = filter(inter -> inter.nl_only, values(s.pairwise_inters))
    if length(pairwise_inters_nl) > 0 && neighbors.n > 0
        CUDA.@sync @cuda threads=256 blocks=1600 pairwise_force_kernel!(
                fs_mat, virial, s.coords, s.atoms, s.boundary, pairwise_inters_nl,
                neighbors.list[1:neighbors.n], Val(s.force_units), Val(2000))
    end

    for inter_list in values(s.specific_inter_lists)
        specific_force_kernel!(fs_mat, inter_list, s.coords, s.boundary, Val(s.force_units))
    end

    for inter in values(s.general_inters)
        # Force type not checked
        fs += ustrip_vec.(forces(inter, s, neighbors))
    end

    fs += CuArray(SVector{3, T}.(eachcol(Array(fs_mat))))
    return fs * s.force_units
end
