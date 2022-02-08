# See https://udel.edu/~arthij/MD.pdf for information on forces
# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See OpenMM documentation and Gromacs manual for other aspects of forces

export
    ustrip_vec,
    force,
    accelerations,
    SpecificForce2Atoms,
    SpecificForce3Atoms,
    SpecificForce4Atoms,
    forces

"""
    ustrip_vec(x)

Broadcasted form of `ustrip` from Unitful.jl, allowing e.g. `ustrip_vec.(coords)`.
"""
ustrip_vec(x) = ustrip.(x)

function check_force_units(fdr, force_units)
    if unit(first(fdr)) != force_units
        error("System force units are ", force_units, " but encountered force units ", unit(first(fdr)))
    end
end

"""
    force(inter::PairwiseInteraction, vec_ij, coord_i, coord_j,
          atom_i, atom_j, box_size)
    force(inter::SpecificInteraction, coord_i, coord_j,
          box_size)
    force(inter::SpecificInteraction, coord_i, coord_j,
          coord_k, box_size)
    force(inter::SpecificInteraction, coord_i, coord_j,
          coord_k, coord_l, box_size)

Calculate the force between atoms due to a given interation type.
For `PairwiseInteraction`s returns a single force vector and for
`SpecificInteraction`s returns a type such as `SpecificForce2Atoms`.
Custom pairwise and specific interaction types should implement
this function.
"""
function force end

@inline @inbounds function force!(fs, inter, s::System, i::Integer, j::Integer, force_units, weight_14::Bool=false)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    if weight_14
        fdr = force(inter, dr, s.coords[i], s.coords[j], s.atoms[i], s.atoms[j], s.box_size, true)
    else
        fdr = force(inter, dr, s.coords[i], s.coords[j], s.atoms[i], s.atoms[j], s.box_size)
    end
    check_force_units(fdr, force_units)
    fdr_ustrip = ustrip.(fdr)
    fs[i] -= fdr_ustrip
    fs[j] += fdr_ustrip
    return nothing
end

@inline @inbounds function force_nounit(inters, coord_i, coord_j, atom_i, atom_j,
                                        box_size, force_units, weight_14::Bool=false)
    dr = vector(coord_i, coord_j, box_size)
    sum(inters) do inter
        if weight_14
            fdr = force(inter, dr, coord_i, coord_j, atom_i, atom_j, box_size, true)
        else
            fdr = force(inter, dr, coord_i, coord_j, atom_i, atom_j, box_size)
        end
        check_force_units(fdr, force_units)
        return ustrip.(fdr)
    end
end

accumulateadd(x) = accumulate(+, x)

# Accumulate values in an array based on the ordered boundaries in bounds
# Used to speed up views with repeated indices on the GPU when you have the bounds
@views @inbounds function accumulate_bounds(arr, bounds)
    zf = zero(arr[1:1])
    accum_pad = vcat(zf, accumulateadd(arr))
    accum_bounds = accum_pad[bounds]
    accum_bounds_offset = vcat(zf, accum_bounds[1:(end - 1)])
    return accum_bounds .- accum_bounds_offset
end

# Uses the Zygote path which gives wrong gradients on the GPU for repeated indices
# Hence only used when we know the indices don't contain repeats
# See https://github.com/FluxML/Zygote.jl/pull/1131
unsafe_getindex(arr, inds) = @view arr[inds]

# Sum forces on neighboring atom pairs to get forces on each atom
# Neighbor forces are accumulated and then atom forces extracted by subtraction
#   at the atom boundaries
# The forces are re-arranged for the other atom index and the process is repeated
# This isn't pretty but it works on the GPU
@views @inbounds function sum_forces(nb_forces, neighbors)
    fs_accum_i = accumulate_bounds(nb_forces, neighbors.atom_bounds_i)
    fs_accum_j = accumulate_bounds(unsafe_getindex(nb_forces, neighbors.sortperm_j), neighbors.atom_bounds_j)
    return fs_accum_j .- fs_accum_i
end

"""
    accelerations(system, neighbors=nothing; parallel=true)

Calculate the accelerations of all atoms using the pairwise and specific
interactions and Newton's second law.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.
"""
function accelerations(s::System, neighbors=nothing; parallel::Bool=true)
    return forces(s, neighbors; parallel=parallel) ./ mass.(s.atoms)
end

function accelerations(s::System, coords, atoms, neighbors=nothing, neighbors_all=nothing)
    return forces(s, coords, atoms, neighbors, neighbors_all) ./ mass.(s.atoms)
end

# Functions defined to allow us to write rrules
getindices_i(arr, neighbors) = @view arr[neighbors.nbsi]
getindices_j(arr, neighbors) = @view arr[neighbors.nbsj]

@views function forces_inters(inters, coords, atoms, neighbors, box_size, force_units, weights_14)
    coords_i, atoms_i = getindices_i(coords, neighbors), getindices_i(atoms, neighbors)
    coords_j, atoms_j = getindices_j(coords, neighbors), getindices_j(atoms, neighbors)
    @inbounds nb_forces = force_nounit.((inters,), coords_i, coords_j, atoms_i, atoms_j,
                                        (box_size,), force_units, weights_14)
    return sum_forces(nb_forces, neighbors)
end

"""
    SpecificForce2Atoms(f1, f2)

Forces on two atoms arising from an interaction.
"""
struct SpecificForce2Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
end

"""
    SpecificForce3Atoms(f1, f2, f3)

Forces on three atoms arising from an interaction.
"""
struct SpecificForce3Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
end

"""
    SpecificForce4Atoms(f1, f2, f3, f4)

Forces on four atoms arising from an interaction.
"""
struct SpecificForce4Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
    f4::SVector{D, T}
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

Base.:+(x::SpecificForce2Atoms, y::SpecificForce2Atoms) = SpecificForce2Atoms(x.f1 + y.f1, x.f2 + y.f2)
Base.:+(x::SpecificForce3Atoms, y::SpecificForce3Atoms) = SpecificForce3Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3)
Base.:+(x::SpecificForce4Atoms, y::SpecificForce4Atoms) = SpecificForce4Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3, x.f4 + y.f4)

getf1(x) = x.f1
getf2(x) = x.f2
getf3(x) = x.f3
getf4(x) = x.f4

@views function specific_force(inter_list::InteractionList2Atoms, coords, box_size, force_units, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js], (box_size,)))
    fis, fjs = getf1.(sparse_fs), getf2.(sparse_fs)
    check_force_units(first(first(fis)), force_units)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms)
end

@views function specific_force(inter_list::InteractionList3Atoms, coords, box_size, force_units, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                coords[inter_list.ks], (box_size,)))
    fis, fjs, fks = getf1.(sparse_fs), getf2.(sparse_fs), getf3.(sparse_fs)
    check_force_units(first(first(fis)), force_units)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms) + sparsevec(inter_list.ks, fks, n_atoms)
end

@views function specific_force(inter_list::InteractionList4Atoms, coords, box_size, force_units, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                coords[inter_list.ks], coords[inter_list.ls], (box_size,)))
    fis, fjs, fks, fls = getf1.(sparse_fs), getf2.(sparse_fs), getf3.(sparse_fs), getf4.(sparse_fs)
    check_force_units(first(first(fis)), force_units)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms) + sparsevec(inter_list.ks, fks, n_atoms) + sparsevec(inter_list.ls, fls, n_atoms)
end

"""
    forces(system, neighbors=nothing; parallel=true)
    forces(system, coords, atoms, neighbors=nothing, neighbors_all=nothing)

Calculate the forces on all atoms in the system.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.

    forces(inter, system, neighbors=nothing)

Calculate the forces on all atoms in the system arising from a general
interaction.
If the interaction uses neighbor lists, the neighbors should be computed
first and passed to the function.
Custom general interaction types should implement this function.
"""
function forces(s::System, neighbors=nothing; parallel::Bool=true)
    n_atoms = length(s)

    if parallel && nthreads() > 1 && n_atoms >= 100
        fs_threads = [ustrip_vec.(zero(s.coords)) for i in 1:nthreads()]

        # Loop over interactions and calculate the acceleration due to each
        for inter in values(s.pairwise_inters)
            if inter.nl_only
                if isnothing(neighbors)
                    error("An interaction uses the neighbor list but neighbors is nothing")
                end
                @threads for ni in 1:neighbors.n
                    i, j, w = neighbors.list[ni]
                    force!(fs_threads[threadid()], inter, s, i, j, s.force_units, w)
                end
            else
                @threads for i in 1:n_atoms
                    for j in 1:(i - 1)
                        force!(fs_threads[threadid()], inter, s, i, j, s.force_units)
                    end
                end
            end
        end

        fs = sum(fs_threads)
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
        sparse_vec = specific_force(inter_list, s.coords, s.box_size, s.force_units, n_atoms)
        fs += ustrip_vec.(Array(sparse_vec))
    end

    for inter in values(s.general_inters)
        # Force type not checked
        fs += ustrip_vec.(forces(inter, s, neighbors))
    end

    return fs * s.force_units
end

function forces(s::System, coords, atoms, neighbors=nothing, neighbors_all=nothing)
    n_atoms = length(s)
    fs = ustrip_vec.(zero(coords))

    pairwise_inters_nonl = filter(inter -> !inter.nl_only, values(s.pairwise_inters))
    @views if length(pairwise_inters_nonl) > 0
        fs += Zygote.checkpointed(forces_inters, pairwise_inters_nonl, coords, atoms, neighbors_all,
                                    s.box_size, s.force_units, false)
    end

    pairwise_inters_nl = filter(inter -> inter.nl_only, values(s.pairwise_inters))
    if length(pairwise_inters_nl) > 0 && length(neighbors.nbsi) > 0
        fs += Zygote.checkpointed(forces_inters, pairwise_inters_nl, coords, atoms, neighbors,
                                    s.box_size, s.force_units, neighbors.weights_14)
    end

    for inter_list in values(s.specific_inter_lists)
        sparse_vec = specific_force(inter_list, coords, s.box_size, s.force_units, n_atoms)
        # Move back to GPU if required
        fs += convert(typeof(fs), ustrip_vec.(Array(sparse_vec)))
    end

    for inter in values(s.general_inters)
        # Force type not checked
        fs += ustrip_vec.(forces(inter, s, neighbors))
    end

    return fs * s.force_units
end
