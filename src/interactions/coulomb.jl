export Coulomb

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
