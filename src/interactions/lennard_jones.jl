"""
    LennardJones(nl_only)

The Lennard-Jones 6-12 interaction.
"""
struct LennardJones{C, T} <: GeneralInteraction
    cutoff::C
    nl_only::Bool
    sqdist_cutoff_nb::T
    inv_sqdist_cutoff::T
end

LennardJones() = LennardJones(ShiftCutoff(true),
                              false,
                              1.0,
                              1.0
)

LennardJones(nl_only) = LennardJones(ShiftCutoff(true),
                              nl_only,
                              1.0,
                              1.0
)

@fastmath @inbounds function force!(forces,
                                    inter::LennardJones{ShiftCutoff},
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    i == j && return
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    sqdist_cutoff_nb = inter.sqdist_cutoff_nb
    r2 > sqdist_cutoff_nb && return

    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return
    end
    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)

    invr2 = inv(r2)
    six_term = (σ ^ 2 * invr2) ^ 3
    f = (24ϵ * invr2) * (2 * six_term ^ 2 - six_term)
    if inter.cutoff.limit_force
        # Limit this to 100 as a fudge to stop it exploding
        f = min(f, 100)
    end
    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end

@inbounds function potential_energy(inter::LennardJones,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    T = eltype(s.coords[i]) # this is not Unitful compatible
    sqdist_cutoff_nb = inter.sqdist_cutoff_nb
    i == j && return zero(T)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    r2 > sqdist_cutoff_nb && return zero(T)

    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return zero(T)
    end
    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)

    invr2 = inv(r2)
    six_term = (σ ^ 2 * invr2) ^ 3

    return 4ϵ * (2 * six_term ^ 2 - six_term) - cutoff_energy(inter, σ, ϵ)
end

function cutoff_energy(inter::LennardJones, σ, ϵ)
    invr2 = inter.inv_sqdist_cutoff
    six_term = (σ ^ 2 * invr2) ^ 3
    
    4ϵ * (2 * six_term ^ 2 - six_term)
end
