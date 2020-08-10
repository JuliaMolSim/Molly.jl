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

"""
    Mie(m, n, nl_only)

The Mie generalized interaction.
When `m` equals 6 and `n` equals 12 this is equivalent to the Lennard Jones interaction.
"""
struct Mie{T} <: GeneralInteraction
    m::T
    n::T
    nl_only::Bool
    mn_fac::T
end

Mie(m, n, nl_only) = Mie(m, n, nl_only, convert(typeof(m), (n / (n - m)) * (n / m) ^ (m / (n - m))))
Mie(m, n) = Mie(m, n, false)

@fastmath @inbounds function force!(forces,
                                    inter::Mie,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ) || i == j
        return
    end
    m = inter.m
    n = inter.n
    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r = norm(dr)
    abs2(r) > sqdist_cutoff_nb && return
    # Derivative obtained via wolfram
    const_mn = inter.mn_fac * ϵ / r
    σ_r = σ / r
    f = m * σ_r ^ m - n * σ_r ^ n
    # Limit this to 100 as a fudge to stop it exploding
    f = min(-f * const_mn / r, 100)
    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end

include("interactions/lennard_jones.jl")
include("interactions/coulomb.jl")
include("interactions/gravity.jl")
include("interactions/soft_sphere.jl")
include("interactions/harmonic_bond.jl")
include("interactions/harmonic_angle.jl")
include("interactions/torsion.jl")
