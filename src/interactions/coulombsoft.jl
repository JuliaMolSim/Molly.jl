export
    CoulombSoftCoreBeutler,
    CoulombSoftCoreGapsys

const coulomb_const = 138.93545764u"kJ * mol^-1 * nm" # 1 / 4πϵ0

@doc raw"""
    CoulombSoftCoreBeutler(; cutoff, α, λ, p, use_neighbors, σ_mixing, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core, used for appearing and disappearing of atoms based on the potential described in Beutler et al. 1994 (Chem. Phys. Lett.).

The potential energy is defined as
```math
V(r_{ij}) = \frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_Q}
```
and the force on each atom by
```math
\vec{F}_i = \frac{1}{4\pi\epsilon_0} \frac{q_iq_jr_{ij}^5}{r_Q}\frac{\vec{r_{ij}}}{r_{ij}}
```
where
```math
r_{Q} = (\frac{\alpha(1-\lambda)C^{(12)}}{C^{(6)}})+r_{ij}^6)^{1/6}
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```
If ``\lambda`` is 1.0, this gives the standard [`Coulomb`](@ref) potential and means atom is fully turned on. ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct CoulombSoftCoreBeutler{C, A, L, S, E, W, T, R} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    use_neighbors::Bool = false
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
    weight_special::W = 1
    coulomb_const::T = coulomb_const
    σ6_fac::R = α * (1-λ)
end

use_neighbors(inter::CoulombSoftCoreBeutler) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreBeutler{C, A, L, S, E, W, T, R}) where {C, A, L, S, E, W, T, R}
    return CoulombSoftCoreBeutler(
        coul.cutoff,
        zero(A),
        zero(L),
        coul.use_neighbors,
        coul.σ_mixing,
        coul.ϵ_mixing,
        zero(W),
        zero(T),
        zero(R),
    )
end

function Base.:+(c1::CoulombSoftCoreBeutler, c2::CoulombSoftCoreBeutler)
    return CoulombSoftCoreBeutler(
        c1.cutoff,
        c1.α + c2.α,
        c1.λ + c2.λ,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.ϵ_mixing,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.σ6_fac + c2.σ6_fac,
    )
end

@inline function force(inter::CoulombSoftCoreBeutler,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    params = (ke, dr, qi, qj, σ, ϵ, inter.σ6_fac)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        # return fdr * inter.weight_special
        return f
    else
        return fdr
    end
end

function pairwise_force(::CoulombSoftCoreBeutler, r, (ke, dr, qi, qj, σ, ϵ, σ6_fac))
    C12 = 4*ϵ*(σ^12)
    C6 = 4*ϵ*(σ^6)
    S = (C12/C6)^(1/6)
    R = ((σ6_fac*(S^6))+r^6)^(7/6)
    return ke * ((qi*qj)/R) * (r^5)
end

@inline function potential_energy(inter::CoulombSoftCoreBeutler,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    params = (ke, qi, qj, σ, ϵ, inter.σ6_fac)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::CoulombSoftCoreBeutler, r, (ke, qi, qj, σ, ϵ, σ6_fac))
    C12 = 4*ϵ*(σ^12)
    C6 = 4*ϵ*(σ^6)
    S = (C12/C6)^(1/6)
    R = ((σ6_fac*(S^6))+r^6)^(1/6)
    return ke * ((qi * qj)/R)
end

@doc raw"""
    CoulombSoftCoreGapsys; cutoff, α, λ, p, σQ, use_neighbors, σ_mixing, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core, used for appearing and disappearing of atoms.

The potential energy is defined as
```math
V(r_{ij}) = \left\{ \begin{array}{cl}
\frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
\frac{1}{4\pi\epsilon_0} (\frac{q_iq_j}{r_{Q}^3}r_{ij}^2-\frac{3q_iq_j}{r_{Q}^2}r_{ij}+\frac{3q_iq_j}{r_{Q}}), & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
and the force on each atom by
```math
\vec{F}_i = \left\{ \begin{array}{cl}
\frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_{ij}^2}\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
\frac{1}{4\pi\epsilon_0} (\frac{-2q_iq_j}{r_{Q}^3}r_{ij}-\frac{3q_iq_j}{r_{Q}^2})\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
where
```math
r_{Q} = \alpha(1-\lambda)(1+σ_Q|qi*qj|)
```

If ``\lambda`` is 1.0, this gives the standard [`Coulomb`](@ref) potential and means atom is fully turned on. ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct CoulombSoftCoreGapsys{C, A, L, S, W, T, R} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    σQ::S = 1.0
    use_neighbors::Bool = false
    weight_special::W = 1
    coulomb_const::T = coulomb_const
    σ6_fac::R = α * (1-λ)^(1/6)
end

use_neighbors(inter::CoulombSoftCoreGapsys) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreGapsys{C, A, L, S, W, T, R}) where {C, A, L, S, W, T, R}
    return CoulombSoftCoreGapsys(
        coul.cutoff,
        zero(A),
        zero(L),
        zero(Q),
        coul.use_neighbors,
        zero(W),
        zero(T),
        zero(R),
    )
end

function Base.:+(c1::CoulombSoftCoreGapsys, c2::CoulombSoftCoreGapsys)
    return CoulombSoftCoreGapsys(
        c1.cutoff,
        c1.α + c2.α,
        c1.λ + c2.λ,
        c1.σQ + c2.σQ,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.σ6_fac + c2.σ6_fac,
    )
end

@inline function force(inter::CoulombSoftCoreGapsys,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, dr, qi, qj, inter.σQ, inter.σ6_fac)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        # return fdr * inter.weight_special
        return f
    else
        return fdr
    end
end

function pairwise_force(::CoulombSoftCoreGapsys, r, (ke, dr, qi, qj, σQ, σ6_fac))
    R = σ6_fac*(1+(σQ*abs(qi*qj)))
    if r >= R
        return ke * ((qi*qj)/(r^2))
    elseif r< R
        return ke * (-(((2*qi*qj)/(R^3)) * r) + ((3*qi*qj)/(R^2))) 
    end
end

@inline function potential_energy(inter::CoulombSoftCoreGapsys,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj, inter.σQ, inter.σ6_fac)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::CoulombSoftCoreGapsys, r, (ke, qi, qj, σQ, σ6_fac))
    R = σ6_fac*(1+(σQ*abs(qi*qj)))
    if r>= R
        return ke * ((qi*qj)/r)
    elseif r<R
        return ke * ((((qi*qj)/(R^3))*(r^2))-(((3*qi*qj)/(R^2))*r)+((3*qi*qj)/R))
    end
end


