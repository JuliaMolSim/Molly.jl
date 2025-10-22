import Molly: pairwise_pe, potential_energy
using Molly: lj_zero_shortcut, pe_cutoff, lorentz_σ_mixing, geometric_ϵ_mixing

export
    LennardJonesSoftCoreStandard

@doc raw"""
    LennardJonesSoftCoreStandard(; cutoff, α, λ, p, use_neighbors, shortcut, σ_mixing, ϵ_mixing,
                         weight_special)

The Lennard-Jones 6-12 interaction between two atoms with a soft core, used for appearing and disappearing of atoms.

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
If ``\lambda`` are 1.0 this gives the standard [`LennardJones`](@ref) potential and means atom is fully turned on. ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct LennardJonesSoftCoreStandard{C, A, L, P, H, S, E, W, R} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    p::P = 2
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
    weight_special::W = 1
    σ6_fac::R = α * (1-λ)^p
end

use_neighbors(inter::LennardJonesSoftCoreStandard) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreStandard{C, A, L, P, H, S, E, W, R}) where {C, A, L, P, H, S, E, W, R}
    return LennardJonesSoftCoreStandard(
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

function Base.:+(l1::LennardJonesSoftCoreStandard, l2::LennardJonesSoftCoreStandard)
    return LennardJonesSoftCoreStandard(
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

@inline function force(inter::LennardJonesSoftCoreStandard,
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
    r = norm(dr)
    σ2 = σ^2
    params = (σ2, ϵ, inter.σ6_fac)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::LennardJonesSoftCoreStandard, r, (σ2, ϵ, σ6_fac))
    inv_rsc6 = inv(r^6 + σ2^3 * σ6_fac) # rsc = (r^6 + α * σ2^3 * λ^p)^(1/6)
    inv_rsc  = sqrt(cbrt(inv_rsc6))
    six_term = σ2^3 * inv_rsc6
    return (24ϵ * inv_rsc) * (2 * six_term^2 - six_term) * (r * inv_rsc)^5
end

@inline function potential_energy(inter::LennardJonesSoftCoreStandard,
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
    r = norm(dr)
    params = (σ, ϵ, inter.σ6_fac)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::LennardJonesSoftCoreStandard, r, (σ, ϵ, σ6_fac))
    C12 = 4*ϵ*(σ^12)
    C6 = 4*ϵ*(σ^6)
    S = (C12/C6)^(1/6)
    r_A = ((σ6_fac*(S^6))+r^6)^(1/6)
    return ((C12/(r_A^12)) - (C6/(r_A^6)))
end

