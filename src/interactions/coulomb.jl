export
    Coulomb,
    CoulombSoftCore,
    CoulombReactionField

@doc raw"""
    Coulomb(; cutoff, nl_only, weight_14, coulomb_const, force_units, energy_units)

The Coulomb electrostatic interaction between two atoms.
The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}}
```
"""
struct Coulomb{C, W, T, F, E} <: PairwiseInteraction
    cutoff::C
    nl_only::Bool
    weight_14::W
    coulomb_const::T
    force_units::F
    energy_units::E
end

const coulombconst = 138.93545764u"kJ * mol^-1 * nm" # 1 / 4πϵ0

function Coulomb(;
                    cutoff=NoCutoff(),
                    nl_only=false,
                    weight_14=1,
                    coulomb_const=coulombconst,
                    force_units=u"kJ * mol^-1 * nm^-1",
                    energy_units=u"kJ * mol^-1")
    return Coulomb{typeof(cutoff), typeof(weight_14), typeof(coulomb_const), typeof(force_units), typeof(energy_units)}(
        cutoff, nl_only, weight_14, coulomb_const, force_units, energy_units)
end

@inline @inbounds function force(inter::Coulomb{C},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    boundary,
                                    weight_14::Bool=false) where C
    r2 = sum(abs2, dr)

    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge

    params = (coulomb_const, qi, qj)

    if cutoff_points(C) == 0
        f = force_divr_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        f = force_divr_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        if r2 < cutoff.sqdist_activation
            f = force_divr_nocutoff(inter, r2, inv(r2), params)
        else
            f = force_divr_cutoff(cutoff, r2, inter, params)
        end
    end

    if weight_14
        return f * dr * inter.weight_14
    else
        return f * dr
    end
end

@fastmath function force_divr_nocutoff(::Coulomb, r2, invr2, (coulomb_const, qi, qj))
    (coulomb_const * qi * qj) / √(r2 ^ 3)
end

@inline @inbounds function potential_energy(inter::Coulomb{C},
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            boundary,
                                            weight_14::Bool=false) where C
    r2 = sum(abs2, dr)

    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (coulomb_const, qi, qj)

    if cutoff_points(C) == 0
        pe = potential(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(coord_i[1])) * inter.energy_units

        pe = potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(coord_i[1])) * inter.energy_units

        if r2 < cutoff.sqdist_activation
            pe = potential(inter, r2, inv(r2), params)
        else
            pe = potential_cutoff(cutoff, r2, inter, params)
        end
    end

    if weight_14
        return pe * inter.weight_14
    else
        return pe
    end
end

@fastmath function potential(::Coulomb, r2, invr2, (coulomb_const, qi, qj))
    (coulomb_const * qi * qj) * √invr2
end

@doc raw"""
    CoulombSoftCore(; cutoff, α, λ, p, nl_only, lorentz_mixing, weight_14,
                    coulomb_const, force_units, energy_units)

The Coulomb electrostatic interaction between two atoms with a soft core.
The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 (r_{ij}^6 + \alpha  \sigma_{ij}^6  \lambda^p)^{\frac{1}{6}}}
```

Here, ``\alpha``, ``\lambda``, and ``p`` adjust the functional form of the soft core of the potential. For `α=0` or 
`λ=0` we get the standard Coulomb potential.
"""
struct CoulombSoftCore{C, A, L, P, R, W, T, F, E} <: PairwiseInteraction
    cutoff::C
    α::A
    λ::L
    p::P
    σ6_fac::R
    nl_only::Bool
    lorentz_mixing::Bool
    weight_14::W
    coulomb_const::T
    force_units::F
    energy_units::E
end

function CoulombSoftCore(;
                    cutoff=NoCutoff(),
                    α=1,
                    λ=0,
                    p=2,
                    nl_only=false,
                    lorentz_mixing=true,
                    weight_14=1,
                    coulomb_const=coulombconst,
                    force_units=u"kJ * mol^-1 * nm^-1",
                    energy_units=u"kJ * mol^-1")
    σ6_fac = α*λ^p
    return CoulombSoftCore{typeof(cutoff), typeof(α), typeof(λ), typeof(p), typeof(σ6_fac),
                           typeof(weight_14), typeof(coulomb_const), typeof(force_units), typeof(energy_units)}(
        cutoff, α, λ, p, σ6_fac, nl_only, lorentz_mixing, weight_14, coulomb_const, force_units, energy_units)
end

@inline @inbounds function force(inter::CoulombSoftCore{C},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    boundary,
                                    weight_14::Bool=false) where C
    r2 = sum(abs2, dr)

    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)

    params = (coulomb_const, qi, qj, σ, inter.σ6_fac)

    if cutoff_points(C) == 0
        f = force_divr_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        f = force_divr_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        if r2 < cutoff.sqdist_activation
            f = force_divr_nocutoff(inter, r2, inv(r2), params)
        else
            f = force_divr_cutoff(cutoff, r2, inter, params)
        end
    end

    if weight_14
        return f * dr * inter.weight_14
    else
        return f * dr
    end
end

@fastmath function force_divr_nocutoff(::CoulombSoftCore, r2, invr2, (coulomb_const, qi, qj, σ, σ6_fac))
    inv_rsc6 = inv(r2^3 + σ6_fac * σ^6)
    inv_rsc2 = cbrt(inv_rsc6)
    inv_rsc3 = sqrt(inv_rsc6)

    ff = (coulomb_const * qi * qj) * inv_rsc2 * sqrt(r2)^5 * inv_rsc2 * inv_rsc3
    # √invr2 is for normalizing dr
    return ff * √invr2
end

@inline @inbounds function potential_energy(inter::CoulombSoftCore{C},
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            boundary,
                                            weight_14::Bool=false) where C
    r2 = sum(abs2, dr)

    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)

    params = (coulomb_const, qi, qj, σ, inter.σ6_fac)

    if cutoff_points(C) == 0
        pe = potential(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(coord_i[1])) * inter.energy_units

        pe = potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(coord_i[1])) * inter.energy_units

        if r2 < cutoff.sqdist_activation
            pe = potential(inter, r2, inv(r2), params)
        else
            pe = potential_cutoff(cutoff, r2, inter, params)
        end
    end

    if weight_14
        return pe * inter.weight_14
    else
        return pe
    end
end

@fastmath function potential(::CoulombSoftCore, r2, invr2, (coulomb_const, qi, qj, σ, σ6_fac))
    inv_rsc6 = inv(r2^3 + σ6_fac * σ^6)
    return (coulomb_const * qi * qj) * √cbrt(inv_rsc6)
end

"""
    CoulombReactionField(; dist_cutoff, solvent_dielectric, nl_only, weight_14,
                            coulomb_const, force_units, energy_units)

The Coulomb electrostatic interaction modified using the reaction field approximation
between two atoms.
"""
struct CoulombReactionField{D, S, W, T, F, E} <: PairwiseInteraction
    dist_cutoff::D
    solvent_dielectric::S
    nl_only::Bool
    weight_14::W
    coulomb_const::T
    force_units::F
    energy_units::E
end

const crf_solvent_dielectric = 78.3

function CoulombReactionField(;
                    dist_cutoff,
                    solvent_dielectric=crf_solvent_dielectric,
                    nl_only=false,
                    weight_14=1,
                    coulomb_const=coulombconst,
                    force_units=u"kJ * mol^-1 * nm^-1",
                    energy_units=u"kJ * mol^-1")
    return CoulombReactionField{typeof(dist_cutoff), typeof(solvent_dielectric), typeof(weight_14),
                                typeof(coulomb_const), typeof(force_units), typeof(energy_units)}(
        dist_cutoff, solvent_dielectric, nl_only, weight_14,
        coulomb_const, force_units, energy_units)
end

@inline @inbounds function force(inter::CoulombReactionField,
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    boundary,
                                    weight_14::Bool=false)
    r2 = sum(abs2, dr)

    if r2 > (inter.dist_cutoff ^ 2)
        return ustrip.(zero(coord_i)) * inter.force_units
    end

    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = √r2
    i, j = atom_i.index, atom_j.index
    if weight_14
        # 1-4 interactions do not use the reaction field approximation
        krf = (1 / (inter.dist_cutoff ^ 3)) * 0
    else
        # These values could be pre-computed but this way is easier for AD
        krf = (1 / (inter.dist_cutoff ^ 3)) * ((inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1))
    end

    f = (coulomb_const * qi * qj) * (inv(r) - 2 * krf * r2) * inv(r2)

    if weight_14
        return f * dr * inter.weight_14
    else
        return f * dr
    end
end

@inline @inbounds function potential_energy(inter::CoulombReactionField,
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            boundary,
                                            weight_14::Bool=false)
    r2 = sum(abs2, dr)

    if r2 > (inter.dist_cutoff ^ 2)
        return ustrip(zero(coord_i[1])) * inter.energy_units
    end

    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = √r2
    if weight_14
        # 1-4 interactions do not use the reaction field approximation
        krf = (1 / (inter.dist_cutoff ^ 3)) * 0
        crf = (1 /  inter.dist_cutoff     ) * 0
    else
        krf = (1 / (inter.dist_cutoff ^ 3)) * ((inter.solvent_dielectric - 1) / (2 * inter.solvent_dielectric + 1))
        crf = (1 /  inter.dist_cutoff     ) * ((3 * inter.solvent_dielectric) / (2 * inter.solvent_dielectric + 1))
    end

    pe = (coulomb_const * qi * qj) * (inv(r) + krf * r2 - crf)

    if weight_14
        return pe * inter.weight_14
    else
        return pe
    end
end
