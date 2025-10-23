export
    LennardJonesSoftCoreBeutler,
    LennardJonesSoftCoreGapsys

@doc raw"""
    LennardJonesSoftCoreBeutler(; cutoff, α, λ, p, use_neighbors, shortcut, σ_mixing, ϵ_mixing,
                         weight_special)

The Lennard-Jones 6-12 interaction between two atoms with a soft core, used for appearing and disappearing of atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{C^{(12)}}{r_{LJ}^{12}} - \frac{C^{(6)}}{r_{LJ}^{6}}
```
and the force on each atom by
```math
\vec{F}_i = ((\frac{12C^{(12)}}{r_{LJ}^{13}} - \frac{6C^{(6)}}{r_{LJ}^7})(\frac{r_{ij}}{r_{LJ}})^5) \frac{\vec{r_{ij}}}{r_{ij}}
```
where
```math
r_{LJ} = (\frac{\alpha(1-\lambda)C^{(12)}}{C^{(6)}}+r^6)^{1/6}
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```
If ``\lambda`` is 1.0, this gives the standard [`LennardJones`](@ref) potential and means atom is fully turned on. ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct LennardJonesSoftCoreBeutler{C, A, L, P, H, S, E, W, R} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    p::P = 2
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
    weight_special::W = 1
    σ6_fac::R = α * (1-λ)
end

use_neighbors(inter::LennardJonesSoftCoreBeutler) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreBeutler{C, A, L, P, H, S, E, W, R}) where {C, A, L, P, H, S, E, W, R}
    return LennardJonesSoftCoreBeutler(
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

function Base.:+(l1::LennardJonesSoftCoreBeutler, l2::LennardJonesSoftCoreBeutler)
    return LennardJonesSoftCoreBeutler(
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

@inline function force(inter::LennardJonesSoftCoreBeutler,
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
    params = (dr, σ, ϵ, inter.σ6_fac)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        # return fdr * inter.weight_special
        return f
    else
        return fdr
    end
end

function pairwise_force(::LennardJonesSoftCoreBeutler, r, (dr, σ, ϵ, σ6_fac))
    C12 = 4*ϵ*(σ^12)
    C6 = 4*ϵ*(σ^6)
    S = (C12/C6)^(1/6)
    R = ((σ6_fac*(S^6))+r^6)^(1/6)
    return (((12*C12)/(R^13)) - ((6*C6)/(R^7)))*((r/R)^5)
end

@inline function potential_energy(inter::LennardJonesSoftCoreBeutler,
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

function pairwise_pe(::LennardJonesSoftCoreBeutler, r, (σ, ϵ, σ6_fac))
    C12 = 4*ϵ*(σ^12)
    C6 = 4*ϵ*(σ^6)
    S = (C12/C6)^(1/6)
    R = ((σ6_fac*(S^6))+r^6)^(1/6)
    return ((C12/(R^12)) - (C6/(R^6)))
end

@doc raw"""
    LennardJonesSoftCoreGapsys(; cutoff, α, λ, p, use_neighbors, shortcut, σ_mixing, ϵ_mixing,
                         weight_special)

The Lennard-Jones 6-12 interaction between two atoms with a soft core potential based on the Gapsys et al. 2012 (JCTC) paper, used for appearing and disappearing of atoms

The potential energy is defined as
```math
V(r_{ij}) = \left\{ \begin{array}{cl}
\frac{C^{(12)}}{r_{ij}^{12}} - \frac{C^{(6)}}{r_{ij}^{6}}, & \text{if} & r \ge r_{LJ} \\
(\frac{78C^{(12)}}{r_{LJ}^{14}}-\frac{21C^{(6)}}{r_{LJ}^{8}})r_{ij}^2 - (\frac{168C^{(12)}}{r_{LJ}^{13}}-\frac{48C^{(6)}}{r_{LJ}^{7}})r_{ij} + \frac{91C^{(12)}}{r_{LJ}^{12}}-\frac{28C^{(6)}}{r_{LJ}^{6}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
and the force on each atom by
```math
\vec{F}_i = \left\{ \begin{array}{cl}
(\frac{12C^{(12)}}{r_{ij}^{13}} - \frac{6C^{(6)}}{r_{ij}^{7}})\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
((\frac{-156C^{(12)}}{r_{LJ}^{14}}+\frac{48C^{(6)}}{r_{LJ}^{8}})r_{ij} - (\frac{168C^{(12)}}{r_{LJ}^{13}}-\frac{48C^{(6)}}{r_{LJ}^{7}}))\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
where
```math
r_{LJ} = \alpha(\frac{26C^{(12)}(1-\lambda)}{7C^{(6)}})^{\frac{1}{6}}
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```
If ``\lambda`` are 1.0 this gives the standard [`LennardJones`](@ref) potential and means atom is fully turned on. ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct LennardJonesSoftCoreGapsys{C, A, L, H, S, E, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreGapsys) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreGapsys{C, A, L, H, S, E, W}) where {C, A, L, H, S, E, W}
    return LennardJonesSoftCoreGapsys(
        lj.cutoff,
        zero(A),
        zero(L),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        zero(W),
    )
end

function Base.:+(l1::LennardJonesSoftCoreGapsys, l2::LennardJonesSoftCoreGapsys)
    return LennardJonesSoftCoreGapsys(
        l1.cutoff,
        l1.α + l2.α,
        l1.λ + l2.λ,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.weight_special + l2.weight_special,
        l1.σ6_fac + l2.σ6_fac,
    )
end

@inline function force(inter::LennardJonesSoftCoreGapsys,
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
    params = (dr, σ, ϵ)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        # return fdr * inter.weight_special
        return f
    else
        return fdr
    end
end

function pairwise_force(inter::LennardJonesSoftCoreGapsys, r, (dr, σ, ϵ))
    C12 = 4*ϵ*(σ^12)
    C6 = 4*ϵ*(σ^6)
    R = (inter.α*((26/7)*(C12/C6)*(1-inter.λ))^(1/6))
    invR = 1/R
    if r >= R
        return (((12*C12)/r^13)-((6*C6)/r^7))
    elseif r < R
        return (((-156*C12*(invR^14)) + (42*C6*(invR^8)))*r + (168*C12*(invR^13)) - (48*C6*(invR^7)))
    end
end

@inline function potential_energy(inter::LennardJonesSoftCoreGapsys,
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
    params = (σ, ϵ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(inter::LennardJonesSoftCoreGapsys, r, (σ, ϵ))
    C12 = 4*ϵ*(σ^12)
    C6 = 4*ϵ*(σ^6)
    R = (inter.α*((26/7)*(C12/C6)*(1-inter.λ))^(1/6))
    invR = 1/R
    if r >= R
        return (C12/(r^12))-(C6/(r^6))
    elseif r < R
        return ((78*C12*(invR^14)) - (21*C6*(invR^8)))*(r^2) - ((168*C12*(invR^13)) - (48*C6*(invR^7)))*r + (91*C12*(invR^12)) - (28*C6*(invR^6))
    end
end
