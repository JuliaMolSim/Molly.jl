# See https://udel.edu/~arthij/MD.pdf for information on forces
# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See Gromacs manual for other aspects of forces

export
    force!,
    accelerations,
    LennardJones,
    SoftSphere,
    Mie,
    Coulomb,
    Gravity,
    HarmonicBond,
    HarmonicAngle,
    Torsion

"""
    force!(forces, interaction, simulation, atom_i, atom_j)

Update the force for an atom pair in response to a given interation type.
Custom interaction types should implement this function.
"""
function force! end

@inline @inbounds function force!(forces, inter, s::Simulation, i::Integer, j::Integer)
    fdr = force(inter, s.coords[i], s.coords[j], s.atoms[i], s.atoms[j], s.box_size)
    forces[i] -= fdr
    forces[j] += fdr
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
        forces_threads = [zero(s.coords) for i in 1:nthreads()]

        # Loop over interactions and calculate the acceleration due to each
        for inter in values(s.general_inters)
            if inter.nl_only
                neighbours = s.neighbours
                @threads for ni in 1:length(neighbours)
                    i, j = neighbours[ni]
                    force!(forces_threads[threadid()], inter, s, i, j)
                end
            else
                @threads for i in 1:n_atoms
                    for j in 1:i
                        force!(forces_threads[threadid()], inter, s, i, j)
                    end
                end
            end
        end

        forces = sum(forces_threads)
    else
        forces = zero(s.coords)

        for inter in values(s.general_inters)
            if inter.nl_only
                neighbours = s.neighbours
                for ni in 1:length(neighbours)
                    i, j = neighbours[ni]
                    force!(forces, inter, s, i, j)
                end
            else
                for i in 1:n_atoms
                    for j in 1:i
                        force!(forces, inter, s, i, j)
                    end
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        for inter in inter_list
            force!(forces, inter, s)
        end
    end

    for i in 1:n_atoms
        forces[i] /= s.atoms[i].mass
    end

    return forces
end

function accelerations(s::Simulation, coords, coords_is, coords_js, atoms_is, atoms_js)
    n_atoms = length(coords)
    forces = zero(coords)

    for inter in values(s.general_inters)
        if inter.nl_only
            forces -= reshape(sum(force.((inter,), coords_is, coords_js, atoms_is, atoms_js,
                                            s.box_size), dims=2), n_atoms)
        else
            forces -= reshape(sum(force.((inter,), coords_is, coords_js, atoms_is, atoms_js,
                                            s.box_size), dims=2), n_atoms)
        end
    end

    for inter_list in values(s.specific_inter_lists)
        sparse_forces = force.((coords,), inter_list, (s,))
        sparse_vecs = SparseVector.(n_atoms, getindex.(sparse_forces, 1),
                                    getindex.(sparse_forces, 2))
        forces += Array(sum(sparse_vecs))
    end

    return forces #./ getproperty.(s.atoms, :mass) # TODO doesn't work on GPU
end

include("interactions/lennard_jones.jl")
include("interactions/coulomb.jl")
include("interactions/gravity.jl")
include("interactions/soft_sphere.jl")
include("interactions/harmonic_bond.jl")
include("interactions/harmonic_angle.jl")
include("interactions/torsion.jl")
