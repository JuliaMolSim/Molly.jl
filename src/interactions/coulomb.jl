export
    Coulomb,
    CoulombSoftCore,
    CoulombReactionField

@doc raw"""
    Coulomb(; cutoff, use_neighbors, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}}
```
"""
struct Coulomb{C, W, T} <: PairwiseInteraction
    cutoff::C
    use_neighbors::Bool
    weight_special::W
    coulomb_const::T
end

const coulomb_const = 138.93545764u"kJ * mol^-1 * nm" # 1 / 4πϵ0

function Coulomb(;
                    cutoff=NoCutoff(),
                    use_neighbors=false,
                    weight_special=1,
                    coulomb_const=coulomb_const)
    return Coulomb(cutoff, use_neighbors, weight_special, coulomb_const)
end

use_neighbors(inter::Coulomb) = inter.use_neighbors

function Base.zero(coul::Coulomb{C, W, T}) where {C, W, T}
    return Coulomb(coul.cutoff, coul.use_neighbors, zero(W), zero(T))
end

function Base.:+(c1::Coulomb, c2::Coulomb)
    return Coulomb(
        c1.cutoff,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

@inline function force(inter::Coulomb{C},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where C
    r2 = sum(abs2, dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr(::Coulomb, r2, invr2, (ke, qi, qj))
    return (ke * qi * qj) / √(r2 ^ 3)
end

@inline function potential_energy(inter::Coulomb{C},
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where C
    r2 = sum(abs2, dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    pe = potential_with_cutoff(inter, r2, params, cutoff, energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::Coulomb, r2, invr2, (ke, qi, qj))
    return (ke * qi * qj) * √invr2
end

@doc raw"""
    CoulombSoftCore(; cutoff, α, λ, p, use_neighbors, σ_mixing, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 (r_{ij}^6 + \alpha  \sigma_{ij}^6  \lambda^p)^{\frac{1}{6}}}
```
If ``\alpha`` or ``\lambda`` are zero this gives the standard [`Coulomb`](@ref) potential.
"""
struct CoulombSoftCore{C, A, L, P, W, T, R} <: PairwiseInteraction
    cutoff::C
    α::A
    λ::L
    p::P
    use_neighbors::Bool
    σ_mixing::Function
    weight_special::W
    coulomb_const::T
    σ6_fac::R
end

function CoulombSoftCore(;
                    cutoff=NoCutoff(),
                    α=1,
                    λ=0,
                    p=2,
                    use_neighbors=false,
                    σ_mixing=lorentz_σ_mixing,
                    weight_special=1,
                    coulomb_const=coulomb_const)
    σ6_fac = α * λ^p
    return CoulombSoftCore(cutoff, α, λ, p, use_neighbors, σ_mixing, weight_special,
                           coulomb_const, σ6_fac)
end

use_neighbors(inter::CoulombSoftCore) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCore{C, A, L, P, W, T, R}) where {C, A, L, P, W, T, R}
    return CoulombSoftCore(
        coul.cutoff,
        zero(A),
        zero(L),
        zero(P),
        coul.use_neighbors,
        coul.σ_mixing,
        zero(W),
        zero(T),
        zero(R),
    )
end

function Base.:+(c1::CoulombSoftCore, c2::CoulombSoftCore)
    return CoulombSoftCore(
        c1.cutoff,
        c1.α + c2.α,
        c1.λ + c2.λ,
        c1.p + c2.p,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.σ6_fac + c2.σ6_fac,
    )
end

@inline function force(inter::CoulombSoftCore,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r2 = sum(abs2, dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    σ = inter.σ_mixing(atom_i, atom_j)
    params = (ke, qi, qj, σ, inter.σ6_fac)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr(::CoulombSoftCore, r2, invr2, (ke, qi, qj, σ, σ6_fac))
    inv_rsc6 = inv(r2^3 + σ6_fac * σ^6)
    inv_rsc2 = cbrt(inv_rsc6)
    inv_rsc3 = sqrt(inv_rsc6)
    ff = (ke * qi * qj) * inv_rsc2 * sqrt(r2)^5 * inv_rsc2 * inv_rsc3
    return ff * √invr2
end

@inline function potential_energy(inter::CoulombSoftCore,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r2 = sum(abs2, dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    σ = inter.σ_mixing(atom_i, atom_j)
    params = (ke, qi, qj, σ, inter.σ6_fac)

    pe = potential_with_cutoff(inter, r2, params, cutoff, energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::CoulombSoftCore, r2, invr2, (ke, qi, qj, σ, σ6_fac))
    inv_rsc6 = inv(r2^3 + σ6_fac * σ^6)
    return (ke * qi * qj) * √cbrt(inv_rsc6)
end

"""
    CoulombReactionField(; dist_cutoff, solvent_dielectric, use_neighbors, weight_special,
                            coulomb_const)

The Coulomb electrostatic interaction modified using the reaction field approximation
between two atoms.
"""
struct CoulombReactionField{D, S, W, T} <: PairwiseInteraction
    dist_cutoff::D
    solvent_dielectric::S
    use_neighbors::Bool
    weight_special::W
    coulomb_const::T
end

const crf_solvent_dielectric = 78.3

function CoulombReactionField(;
                    dist_cutoff,
                    solvent_dielectric=crf_solvent_dielectric,
                    use_neighbors=false,
                    weight_special=1,
                    coulomb_const=coulomb_const)
    return CoulombReactionField(dist_cutoff, solvent_dielectric, use_neighbors,
                                weight_special, coulomb_const)
end

use_neighbors(inter::CoulombReactionField) = inter.use_neighbors

function Base.zero(coul::CoulombReactionField{D, S, W, T}) where {D, S, W, T}
    return CoulombReactionField{D, S, W, T}(
        zero(D),
        zero(S),
        coul.use_neighbors,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombReactionField, c2::CoulombReactionField)
    return CoulombReactionField(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

@inline function force(inter::CoulombReactionField,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r2 = sum(abs2, dr)
    if r2 > (inter.dist_cutoff ^ 2)
        return ustrip.(zero(dr)) * force_units
    end

    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = √r2
    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = (1 / (inter.dist_cutoff ^ 3)) * 0
    else
        # These values could be pre-computed but this way is easier for AD
        krf = (1 / (inter.dist_cutoff ^ 3)) * ((inter.solvent_dielectric - 1) /
              (2 * inter.solvent_dielectric + 1))
    end

    f = (ke * qi * qj) * (inv(r) - 2 * krf * r2) * inv(r2)

    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

@inline function potential_energy(inter::CoulombReactionField,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r2 = sum(abs2, dr)
    if r2 > (inter.dist_cutoff ^ 2)
        return ustrip(zero(dr[1])) * energy_units
    end

    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = √r2
    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = (1 / (inter.dist_cutoff ^ 3)) * 0
        crf = (1 /  inter.dist_cutoff     ) * 0
    else
        krf = (1 / (inter.dist_cutoff ^ 3)) * ((inter.solvent_dielectric - 1) /
              (2 * inter.solvent_dielectric + 1))
        crf = (1 /  inter.dist_cutoff     ) * ((3 * inter.solvent_dielectric) /
              (2 * inter.solvent_dielectric + 1))
    end

    pe = (ke * qi * qj) * (inv(r) + krf * r2 - crf)

    if special
        return pe * inter.weight_special
    else
        return pe
    end
end
