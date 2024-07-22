export
    LennardJones,
    LennardJonesSoftCore

function lj_zero_shortcut(atom_i, atom_j)
    return iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
           iszero_value(atom_i.σ) || iszero_value(atom_j.σ)
end

no_shortcut(atom_i, atom_j) = false

function lorentz_σ_mixing(atom_i, atom_j)
    return (atom_i.σ + atom_j.σ) / 2
end

function geometric_σ_mixing(atom_i, atom_j)
    return sqrt(atom_i.σ * atom_j.σ)
end

function geometric_ϵ_mixing(atom_i, atom_j)
    return sqrt(atom_i.ϵ * atom_j.ϵ)
end

@doc raw"""
    LennardJones(; cutoff, use_neighbors, shortcut, σ_mixing, ϵ_mixing, weight_special)

The Lennard-Jones 6-12 interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]
```
and the force on each atom by
```math
\begin{aligned}
\vec{F}_i &= 24\varepsilon_{ij} \left(2\frac{\sigma_{ij}^{12}}{r_{ij}^{13}} - \frac{\sigma_{ij}^6}{r_{ij}^{7}}\right) \frac{\vec{r}_{ij}}{r_{ij}} \\
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}^{6}}{r_{ij}^{6}}\right)^2 -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \vec{r}_{ij}
\end{aligned}
```
"""
@kwdef struct LennardJones{C, H, S, E, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
    weight_special::W = 1
end

use_neighbors(inter::LennardJones) = inter.use_neighbors

function Base.zero(lj::LennardJones{C, H, S, E, W}) where {C, H, S, E, W}
    return LennardJones(
        lj.cutoff,
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        zero(W),
    )
end

function Base.:+(l1::LennardJones, l2::LennardJones)
    return LennardJones(
        l1.cutoff,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.weight_special + l2.weight_special,
    )
end

@inline function force(inter::LennardJones,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
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
    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr(::LennardJones, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3
    return (24ϵ * invr2) * (2 * six_term ^ 2 - six_term)
end

@inline function potential_energy(inter::LennardJones,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
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

    pe = potential_with_cutoff(inter, r2, params, cutoff, energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::LennardJones, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3
    return 4ϵ * (six_term ^ 2 - six_term)
end

@doc raw"""
    LennardJonesSoftCore(; cutoff, α, λ, p, use_neighbors, shortcut, σ_mixing, ϵ_mixing,
                         weight_special)

The Lennard-Jones 6-12 interaction between two atoms with a soft core.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}^{\text{sc}}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}^{\text{sc}}}\right)^{6}\right]
```
and the force on each atom by
```math
\vec{F}_i = 24\varepsilon_{ij} \left(2\frac{\sigma_{ij}^{12}}{(r_{ij}^{\text{sc}})^{13}} - \frac{\sigma_{ij}^6}{(r_{ij}^{\text{sc}})^{7}}\right) \left(\frac{r_{ij}}{r_{ij}^{\text{sc}}}\right)^5 \frac{\vec{r}_{ij}}{r_{ij}}
```
where
```math
r_{ij}^{\text{sc}} = \left(r_{ij}^6 + \alpha \sigma_{ij}^6 \lambda^p \right)^{1/6}
```
If ``\alpha`` or ``\lambda`` are zero this gives the standard [`LennardJones`](@ref) potential.
"""
@kwdef struct LennardJonesSoftCore{C, A, L, P, H, S, E, W, R} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    p::P = 2
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
    weight_special::W = 1
    σ6_fac::R = α * λ^p
end

use_neighbors(inter::LennardJonesSoftCore) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCore{C, A, L, P, H, S, E, W, R}) where {C, A, L, P, H, S, E, W, R}
    return LennardJonesSoftCore(
        lj.cutoff,
        zero(A),
        zero(L),
        zero(P),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        zero(W),
        zero(R),
    )
end

function Base.:+(l1::LennardJonesSoftCore, l2::LennardJonesSoftCore)
    return LennardJonesSoftCore(
        l1.cutoff,
        l1.α + l2.α,
        l1.λ + l2.λ,
        l1.p + l2.p,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.weight_special + l2.weight_special,
        l1.σ6_fac + l2.σ6_fac,
    )
end

@inline function force(inter::LennardJonesSoftCore,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    if inter.shortcut(atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    σ2 = σ^2
    params = (σ2, ϵ, inter.σ6_fac)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr(::LennardJonesSoftCore, r2, invr2, (σ2, ϵ, σ6_fac))
    inv_rsc6 = inv(r2^3 + σ2^3 * σ6_fac) # rsc = (r2^3 + α * σ2^3 * λ^p)^(1/6)
    inv_rsc  = √cbrt(inv_rsc6)
    six_term = σ2^3 * inv_rsc6
    ff  = (24ϵ * inv_rsc) * (2 * six_term^2 - six_term) * (√r2 * inv_rsc)^5
    return ff * √invr2
end

@inline function potential_energy(inter::LennardJonesSoftCore,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    if inter.shortcut(atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    σ2 = σ^2
    params = (σ2, ϵ, inter.σ6_fac)

    pe = potential_with_cutoff(inter, r2, params, cutoff, energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::LennardJonesSoftCore, r2, invr2, (σ2, ϵ, σ6_fac))
    six_term = σ2^3 * inv(r2^3 + σ2^3 * σ6_fac)
    return 4ϵ * (six_term ^ 2 - six_term)
end
