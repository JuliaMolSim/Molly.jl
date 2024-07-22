export SoftSphere

@doc raw"""
    SoftSphere(; cutoff, use_neighbors, shortcut, σ_mixing, ϵ_mixing)

The soft-sphere potential.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}
```
"""
@kwdef struct SoftSphere{C} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::Function = lj_zero_shortcut
    σ_mixing::Function = lorentz_σ_mixing
    ϵ_mixing::Function = geometric_ϵ_mixing
end

use_neighbors(inter::SoftSphere) = inter.use_neighbors

function Base.zero(ss::SoftSphere)
    return SoftSphere(ss.cutoff, ss.use_neighbors, ss.shortcut, ss.σ_mixing, ss.ϵ_mixing)
end

function Base.:+(s1::SoftSphere, ::SoftSphere)
    return SoftSphere(s1.cutoff, s1.use_neighbors, s1.shortcut, s1.σ_mixing, s1.ϵ_mixing)
end

@inline function force(inter::SoftSphere,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       args...)
    if inter.shortcut(atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
    return f * dr
end

function force_divr(::SoftSphere, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3
    return (24ϵ * invr2) * 2 * six_term ^ 2
end

function potential_energy(inter::SoftSphere,
                          dr,
                          atom_i,
                          atom_j,
                          energy_units=u"kJ * mol^-1",
                          args...)
    if inter.shortcut(atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    return potential_with_cutoff(inter, r2, params, cutoff, energy_units)
end

function potential(::SoftSphere, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3
    return 4ϵ * (six_term ^ 2)
end
