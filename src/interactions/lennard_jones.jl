"""
    LennardJones(nl_only)

The Lennard-Jones 6-12 interaction.
"""
struct LennardJones{T, C} <: GeneralInteraction
    cutoff::C
    nl_only::Bool
    sqdist_cutoff_coeff::T
    inv_sqdist_cutoff::T
end

LennardJones() = LennardJones(ShiftCutoff(true),
                              false,
                              9.0,
                              1/9.0
)

LennardJones(nl_only) = LennardJones(ShiftCutoff(true),
                              nl_only,
                              9.0,
                              1/9.0
)

@inline @inbounds function force!(forces,
                                    inter::LennardJones{T, ShiftCutoff},
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer) where T
    i == j && return
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)

    σ2 = σ^2
    sqdist_cutoff = inter.sqdist_cutoff_coeff * σ2
    r2 > sqdist_cutoff && return
    
    invr2 = inv(r2)
    six_term = (σ2 * invr2) ^ 3
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
    i == j && return zero(T)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    
    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return zero(T)
    end
    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)
    
    σ2 = σ^2
    sqdist_cutoff = inter.sqdist_cutoff_coeff * σ2
    r2 > sqdist_cutoff && return zero(T)
    
    invr2 = inv(r2)
    six_term = (σ2 * invr2) ^ 3

    return 4ϵ * (six_term ^ 2 - six_term) - cutoff_energy(inter, σ, ϵ)
end

function cutoff_energy(inter::LennardJones, σ, ϵ)
    invr2 = inter.inv_sqdist_cutoff
    six_term = (σ ^ 2 * invr2) ^ 3
    
    4ϵ * (six_term ^ 2 - six_term)
end
