export SoftSphere

@doc raw"""
    SoftSphere(; cutoff, use_neighbors, lorentz_mixing, skip_shortcut)

The soft-sphere potential.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}
```
"""
struct SoftSphere{S, C} <: PairwiseInteraction
    cutoff::C
    use_neighbors::Bool
    lorentz_mixing::Bool
end

function SoftSphere(;
                    cutoff=NoCutoff(),
                    use_neighbors=false,
                    lorentz_mixing=true,
                    skip_shortcut=false)
    return SoftSphere{skip_shortcut, typeof(cutoff)}(
        cutoff, use_neighbors, lorentz_mixing)
end

use_neighbors(inter::SoftSphere) = inter.use_neighbors

@inline function force(inter::SoftSphere{S, C},
                       dr,
                       atom_i,
                       atom_j,
                       force_units,
                       args...) where {S, C}
    if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
              iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
        return ustrip.(zero(coord_i)) * force_units
    end

    # Lorentz-Berthelot mixing rules use the arithmetic average for σ
    # Otherwise use the geometric average
    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

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

function potential_energy(inter::SoftSphere{S, C},
                          dr,
                          atom_i,
                          atom_j,
                          energy_units,
                          args...) where {S, C}
    if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
              iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
        return ustrip(zero(coord_i[1])) * energy_units
    end

    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

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
