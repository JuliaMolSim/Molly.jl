export SoftSphere

@doc raw"""
    SoftSphere(; cutoff, use_neighbors, shortcut, σ_mixing, ϵ_mixing)

The soft-sphere potential.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}
```
"""
@kwdef struct SoftSphere{C, H, S, E}
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
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
    r = norm(dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    f = force_cutoff(cutoff, inter, r, params, force_units)
    return (f / r) * dr
end

function pairwise_force(::SoftSphere, r, (σ2, ϵ))
    six_term = (σ2 / r^2) ^ 3
    return (24ϵ / r) * 2 * six_term ^ 2
end

@inline function potential_energy(inter::SoftSphere,
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
    r = norm(dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    return pe_cutoff(cutoff, inter, r, params, energy_units)
end

function pairwise_pe(::SoftSphere, r, (σ2, ϵ))
    six_term = (σ2 / r^2) ^ 3
    return 4ϵ * (six_term ^ 2)
end
