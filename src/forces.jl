# See https://udel.edu/~arthij/MD.pdf for information on forces
# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See Gromacs manual for other aspects of forces

export
    force,
    accelerations,
    LennardJones,
    SoftSphere,
    Mie,
    Coulomb,
    Gravity,
    HarmonicBond,
    HarmonicAngle,
    Torsion

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
        sparse_forces = force.(inter_list, (s.coords,), (s,))
        ge1, ge2 = getindex.(sparse_forces, 1), getindex.(sparse_forces, 2)
        sparse_vec = SparseVector(n_atoms, reduce(vcat, ge1), reduce(vcat, ge2))
        forces += Array(sparse_vec)
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
        # Take coords off the GPU if they are on there
        coords_cpu = Array(coords)
        sparse_forces = force.(inter_list, (coords_cpu,), (s,))
        ge1, ge2 = getindex.(sparse_forces, 1), getindex.(sparse_forces, 2)
        sparse_vec = SparseVector(n_atoms, reduce(vcat, ge1), reduce(vcat, ge2))
        # Move back to GPU if required
        forces += convert(typeof(coords), Array(sparse_vec))
    end

    mass_i = findfirst(x -> x == :mass, fieldnames(eltype(atoms_is)))
    return forces ./ getfield.(s.atoms, mass_i)
end

"""
    force(inter, coord_i, coord_j, atom_i, atom_j, box_size)

Calculate the force between a pair of atoms due to a given interation type.
Custom interaction types should implement this function.
"""
function force end

include("interactions/lennard_jones.jl")
include("interactions/coulomb.jl")
include("interactions/gravity.jl")
include("interactions/soft_sphere.jl")
include("interactions/harmonic_bond.jl")
include("interactions/harmonic_angle.jl")
include("interactions/torsion.jl")
