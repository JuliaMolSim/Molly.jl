export Mie

@doc raw"""
    Mie(; m, n, cutoff, use_neighbors, lorentz_mixing, skip_shortcut)

The Mie generalized interaction between two atoms.

When `m` equals 6 and `n` equals 12 this is equivalent to the Lennard-Jones interaction.
The potential energy is defined as
```math
V(r_{ij}) = C \varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^n - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^m\right]
```
where
```math
C = \frac{n}{n - m} \left( \frac{n}{m} \right) ^\frac{m}{n - m}
```
"""
struct Mie{S, C, T} <: PairwiseInteraction
    m::T
    n::T
    cutoff::C
    use_neighbors::Bool
    lorentz_mixing::Bool
    mn_fac::T
end

function Mie(;
                m,
                n,
                cutoff=NoCutoff(),
                use_neighbors=false,
                lorentz_mixing=true,
                skip_shortcut=false)
    m_p, n_p, mn_fac = promote(m, n, (n / (n - m)) * (n / m) ^ (m / (n - m)))
    return Mie{skip_shortcut, typeof(cutoff), typeof(m_p)}(
        m_p, n_p, cutoff, use_neighbors, lorentz_mixing, mn_fac)
end

use_neighbors(inter::Mie) = inter.use_neighbors

function force(inter::Mie{S, C, T},
               dr,
               atom_i,
               atom_j,
               force_units=u"kJ * mol^-1 * nm^-1",
               args...) where {S, C, T}
    if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
              iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
        return ustrip.(zero(dr)) * force_units
    end

    # Lorentz-Berthelot mixing rules use the arithmetic average for σ
    # Otherwise use the geometric average
    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    r = √r2
    m = inter.m
    n = inter.n
    const_mn = inter.mn_fac * ϵ
    σ_r = σ / r
    params = (m, n, σ_r, const_mn)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
    return f * dr
end

function force_divr(::Mie, r2, invr2, (m, n, σ_r, const_mn))
    return -const_mn / r2 * (m * σ_r ^ m - n * σ_r ^ n)
end

@inline function potential_energy(inter::Mie{S, C, T},
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  args...) where {S, C, T}
    if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
              iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
        return ustrip(zero(dr[1])) * energy_units
    end

    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    r = √r2
    m = inter.m
    n = inter.n
    const_mn = inter.mn_fac * ϵ
    σ_r = σ / r
    params = (m, n, σ_r, const_mn)

    return potential_with_cutoff(inter, r2, params, cutoff, energy_units)
end

function potential(::Mie, r2, invr2, (m, n, σ_r, const_mn))
    return const_mn * (σ_r ^ n - σ_r ^ m)
end
