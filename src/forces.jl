# See https://udel.edu/~arthij/MD.pdf for information on forces
# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See OpenMM documentation and Gromacs manual for other aspects of forces

export
    ustripvec,
    force,
    accelerations,
    SpecificForce2Atoms,
    SpecificForce3Atoms,
    SpecificForce4Atoms,
    forces,
    LennardJones,
    SoftSphere,
    Mie,
    Coulomb,
    CoulombReactionField,
    Gravity,
    HarmonicBond,
    HarmonicAngle,
    PeriodicTorsion,
    RBTorsion

"""
    ustripvec(x)

Broadcasted form of `ustrip` from Unitful.jl, allowing e.g. `ustripvec.(coords)`.
"""
ustripvec(x) = ustrip.(x)

function checkforcetype(fdr, force_unit)
    if unit(first(fdr)) != force_unit
        error("System force type is ", force_unit, " but encountered force type ", unit(first(fdr)))
    end
end

"""
    force(inter, coord_i, coord_j, atom_i, atom_j, box_size)

Calculate the force between a pair of atoms due to a given interation type.
Custom interaction types should implement this function.
"""
function force end

@inline @inbounds function force!(fs, inter, s::System, i::Integer, j::Integer, force_unit, weight_14::Bool=false)
    if weight_14
        fdr = force(inter, s.coords[i], s.coords[j], s.atoms[i], s.atoms[j], s.box_size, true)
    else
        fdr = force(inter, s.coords[i], s.coords[j], s.atoms[i], s.atoms[j], s.box_size)
    end
    checkforcetype(fdr, force_unit)
    fdr_ustrip = ustrip.(fdr)
    fs[i] -= fdr_ustrip
    fs[j] += fdr_ustrip
    return nothing
end

@inline @inbounds function force_nounit(inter, coord_i, coord_j, atom_i, atom_j,
                                        box_size, force_unit, weight_14::Bool=false)
    if weight_14
        fdr = force(inter, coord_i, coord_j, atom_i, atom_j, box_size, true)
    else
        fdr = force(inter, coord_i, coord_j, atom_i, atom_j, box_size)
    end
    checkforcetype(fdr, force_unit)
    return ustrip.(fdr)
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
@views @inbounds function sumforces(nb_forces, neighbors)
    fs_accum_i = accumulate_bounds(nb_forces, neighbors.atom_bounds_i)
    fs_accum_j = accumulate_bounds(unsafe_getindex(nb_forces, neighbors.sortperm_j), neighbors.atom_bounds_j)
    return fs_accum_j .- fs_accum_i
end

"""
    accelerations(system, neighbors=nothing; parallel=true)

Calculate the accelerations of all atoms using the general and specific
interactions and Newton's second law.
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

@views function specific_force(inter_list::InteractionList2Atoms, coords, box_size, force_unit, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js], (box_size,)))
    fis, fjs = getf1.(sparse_fs), getf2.(sparse_fs)
    checkforcetype(first(first(fis)), force_unit)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms)
end

@views function specific_force(inter_list::InteractionList3Atoms, coords, box_size, force_unit, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                coords[inter_list.ks], (box_size,)))
    fis, fjs, fks = getf1.(sparse_fs), getf2.(sparse_fs), getf3.(sparse_fs)
    checkforcetype(first(first(fis)), force_unit)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms) + sparsevec(inter_list.ks, fks, n_atoms)
end

@views function specific_force(inter_list::InteractionList4Atoms, coords, box_size, force_unit, n_atoms)
    sparse_fs = Array(force.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                coords[inter_list.ks], coords[inter_list.ls], (box_size,)))
    fis, fjs, fks, fls = getf1.(sparse_fs), getf2.(sparse_fs), getf3.(sparse_fs), getf4.(sparse_fs)
    checkforcetype(first(first(fis)), force_unit)
    return sparsevec(inter_list.is, fis, n_atoms) + sparsevec(inter_list.js, fjs, n_atoms) + sparsevec(inter_list.ks, fks, n_atoms) + sparsevec(inter_list.ls, fls, n_atoms)
end

"""
    forces(system, neighbors=nothing; parallel=true)

Calculate the forces on all atoms using the general and specific interactions.
"""
function forces(s::System, neighbors=nothing; parallel::Bool=true)
    n_atoms = length(s)

    if parallel && nthreads() > 1 && n_atoms >= 100
        fs_threads = [ustripvec.(zero(s.coords)) for i in 1:nthreads()]

        # Loop over interactions and calculate the acceleration due to each
        for inter in values(s.general_inters)
            if inter.nl_only
                @threads for ni in 1:neighbors.n
                    i, j, w = neighbors.list[ni]
                    force!(fs_threads[threadid()], inter, s, i, j, s.force_unit, w)
                end
            else
                @threads for i in 1:n_atoms
                    for j in 1:(i - 1)
                        force!(fs_threads[threadid()], inter, s, i, j, s.force_unit)
                    end
                end
            end
        end

        fs = sum(fs_threads)
    else
        fs = ustripvec.(zero(s.coords))

        for inter in values(s.general_inters)
            if inter.nl_only
                for ni in 1:neighbors.n
                    i, j, w = neighbors.list[ni]
                    force!(fs, inter, s, i, j, s.force_unit, w)
                end
            else
                for i in 1:n_atoms
                    for j in 1:(i - 1)
                        force!(fs, inter, s, i, j, s.force_unit)
                    end
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        sparse_vec = specific_force(inter_list, s.coords, s.box_size, s.force_unit, n_atoms)
        fs += ustripvec.(Array(sparse_vec))
    end

    return fs * s.force_unit
end

function forces(s::System, coords, atoms, neighbors=nothing, neighbors_all=nothing)
    n_atoms = length(s)
    fs = ustripvec.(zero(coords))

    general_inters_nonl = [inter for inter in values(s.general_inters) if !inter.nl_only]
    @views if length(general_inters_nonl) > 0
        coords_i, atoms_i = getindices_i(coords, neighbors_all), getindices_i(atoms, neighbors_all)
        coords_j, atoms_j = getindices_j(coords, neighbors_all), getindices_j(atoms, neighbors_all)
        for inter in general_inters_nonl
            @inbounds nb_forces = force_nounit.((inter,), coords_i, coords_j, atoms_i, atoms_j,
                                                (s.box_size,), s.force_unit, false)
            fs += sumforces(nb_forces, neighbors_all)
        end
    end

    general_inters_nl = [inter for inter in values(s.general_inters) if inter.nl_only]
    @views if length(general_inters_nl) > 0
        coords_i, atoms_i = getindices_i(coords, neighbors), getindices_i(atoms, neighbors)
        coords_j, atoms_j = getindices_j(coords, neighbors), getindices_j(atoms, neighbors)
        if length(neighbors.nbsi) > 0
            @inbounds nb_forces = force_nounit.((first(general_inters_nl),), coords_i, coords_j,
                    atoms_i, atoms_j, (s.box_size,), s.force_unit, neighbors.weights_14)
            # Add all atom pair forces before summation
            for inter in general_inters_nl[2:end]
                @inbounds nb_forces += force_nounit.((inter,), coords_i, coords_j,
                    atoms_i, atoms_j, (s.box_size,), s.force_unit, neighbors.weights_14)
            end
            fs += sumforces(nb_forces, neighbors)
        end
    end

    for inter_list in values(s.specific_inter_lists)
        sparse_vec = specific_force(inter_list, coords, s.box_size, s.force_unit, n_atoms)
        # Move back to GPU if required
        fs += convert(typeof(fs), ustripvec.(Array(sparse_vec)))
    end

    return fs * s.force_unit
end

include("interactions/lennard_jones.jl")
include("interactions/soft_sphere.jl")
include("interactions/mie.jl")
include("interactions/coulomb.jl")
include("interactions/coulomb_reaction_field.jl")
include("interactions/gravity.jl")
include("interactions/harmonic_bond.jl")
include("interactions/harmonic_angle.jl")
include("interactions/periodic_torsion.jl")
include("interactions/rb_torsion.jl")
