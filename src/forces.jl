# See https://udel.edu/~arthij/MD.pdf for information on forces
# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See OpenMM documentation and Gromacs manual for other aspects of forces

export
    mass,
    ustripvec,
    force,
    accelerations,
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
    mass(atom)

The mass of an atom.
"""
mass(atom::Atom) = atom.mass

"""
    ustripvec(x)

Broadcasted form of `ustrip` from Unitful.jl, allowing e.g. `ustripvec.(coords)`.
"""
ustripvec(x) = ustrip.(x)

function checkforcetype(fdr, force_type)
    if unit(first(fdr)) != force_type
        error("Simulation force type is ", force_type,
                " but encountered force type ", unit(first(fdr)))
    end
end

"""
    force(inter, coord_i, coord_j, atom_i, atom_j, box_size)

Calculate the force between a pair of atoms due to a given interation type.
Custom interaction types should implement this function.
"""
function force end

@inline @inbounds function force(inter, coord_i, coord_j, atom_i, atom_j, box_size, force_type)
    fdr = force(inter, coord_i, coord_j, atom_i, atom_j, box_size)
    checkforcetype(fdr, force_type)
    return ustrip.(fdr)
end

@inline @inbounds function force!(forces, inter, s::Simulation, i::Integer, j::Integer, force_type, weight_14::Bool=false)
    fdr = force(inter, s.coords[i], s.coords[j], s.atoms[i], s.atoms[j], s.box_size)
    checkforcetype(fdr, force_type)
    if weight_14
        fdr_ustrip = ustrip.(fdr) * inter.weight_14
    else
        fdr_ustrip = ustrip.(fdr)
    end
    forces[i] -= fdr_ustrip
    forces[j] += fdr_ustrip
    return nothing
end

"""
    accelerations(simulation; parallel=true)

Calculate the accelerations of all atoms using the general and specific
interactions and Newton's second law.
"""
function accelerations(s::Simulation; parallel::Bool=true)
    n_atoms = length(s.coords)

    if parallel && nthreads() > 1 && n_atoms >= 100
        forces_threads = [ustripvec.(zero(s.coords)) for i in 1:nthreads()]

        # Loop over interactions and calculate the acceleration due to each
        for inter in values(s.general_inters)
            if inter.nl_only
                neighbors = s.neighbors
                @threads for ni in 1:neighbors.n
                    i, j, w = neighbors.list[ni]
                    force!(forces_threads[threadid()], inter, s, i, j, s.force_unit, w)
                end
            else
                @threads for i in 1:n_atoms
                    for j in 1:(i - 1)
                        force!(forces_threads[threadid()], inter, s, i, j, s.force_unit)
                    end
                end
            end
        end

        forces = sum(forces_threads)
    else
        forces = ustripvec.(zero(s.coords))

        for inter in values(s.general_inters)
            if inter.nl_only
                neighbors = s.neighbors
                for ni in 1:neighbors.n
                    i, j, w = neighbors.list[ni]
                    force!(forces, inter, s, i, j, s.force_unit, w)
                end
            else
                for i in 1:n_atoms
                    for j in 1:(i - 1)
                        force!(forces, inter, s, i, j, s.force_unit)
                    end
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        sparse_forces = force.(inter_list, (s.coords,), (s,))
        ge1, ge2 = getindex.(sparse_forces, 1), getindex.(sparse_forces, 2)
        checkforcetype(first(first(ge2)), s.force_unit)
        sparse_vec = sparsevec(reduce(vcat, ge1), reduce(vcat, ge2), n_atoms)
        forces += ustripvec.(Array(sparse_vec))
    end

    return (forces * s.force_unit) ./ mass.(s.atoms)
end

function accelerations(s::Simulation, coords, atoms, nbsi, nbsj, nb_inds)
    n_atoms = length(coords)
    forces = ustripvec.(zero(coords))

    @views for inter in values(s.general_inters)
        fs = force.((inter,), coords[nbsi], coords[nbsj], atoms[nbsi], atoms[nbsj], (s.box_size,), s.force_unit)
        zf = zero(fs[1:1])
        afs_ip = accumulate!(+, fs, fs)
        afs = vcat(zf, afs_ip)
        afs_inds = afs[nb_inds]
        afs_inds_i1 = vcat(zf, afs_inds[1:(end - 1)])
        forces -= afs_inds .- afs_inds_i1
    end

    for inter_list in values(s.specific_inter_lists)
        # Take coords off the GPU if they are on there
        coords_cpu = Array(coords)
        sparse_forces = force.(inter_list, (coords_cpu,), (s,))
        ge1, ge2 = getindex.(sparse_forces, 1), getindex.(sparse_forces, 2)
        checkforcetype(first(first(ge2)), s.force_unit)
        sparse_vec = sparsevec(reduce(vcat, ge1), reduce(vcat, ge2), n_atoms)
        # Move back to GPU if required
        forces += convert(typeof(forces), ustripvec.(Array(sparse_vec)))
    end

    return (forces * s.force_unit) ./ mass.(s.atoms)
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
