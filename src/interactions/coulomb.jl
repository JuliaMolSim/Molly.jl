"""
    Coulomb(; cutoff, nl_only, weight_14, coulomb_const, force_unit, energy_unit)

The Coulomb electrostatic interaction.
"""
struct Coulomb{C, W, T, F, E} <: GeneralInteraction
    cutoff::C
    nl_only::Bool
    weight_14::W
    coulomb_const::T
    force_unit::F
    energy_unit::E
end

const coulombconst = 138.93545764u"kJ * mol^-1 * nm" # 1 / 4πϵ0

function Coulomb(;
                    cutoff=NoCutoff(),
                    nl_only=false,
                    weight_14=1.0,
                    coulomb_const=coulombconst,
                    force_unit=u"kJ * mol^-1 * nm^-1",
                    energy_unit=u"kJ * mol^-1")
    return Coulomb{typeof(cutoff), typeof(weight_14), typeof(coulomb_const), typeof(force_unit), typeof(energy_unit)}(
        cutoff, nl_only, weight_14, coulomb_const, force_unit, energy_unit)
end

@inline @inbounds function force(inter::Coulomb{C},
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    box_size,
                                    weight_14::Bool=false) where C
    dr = vector(coord_i, coord_j, box_size)
    r2 = sum(abs2, dr)

    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge

    params = (coulomb_const, qi, qj)

    if cutoff_points(C) == 0
        f = force_divr_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_unit

        f = force_divr_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_unit

        if r2 < cutoff.activation_dist
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
                                    s::System,
                                    i::Integer,
                                    j::Integer,
                                    weight_14::Bool=false) where C
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = s.atoms[i].charge, s.atoms[j].charge
    params = (coulomb_const, qi, qj)

    if cutoff_points(C) == 0
        pe = potential(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(s.box_size[1])) * inter.energy_unit

        pe = potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(s.box_size[1])) * inter.energy_unit

        if r2 < cutoff.activation_dist
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
