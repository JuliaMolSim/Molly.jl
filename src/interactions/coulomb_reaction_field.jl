export CoulombReactionField

"""
    CoulombReactionField(; dist_cutoff, solvent_dielectric, nl_only, weight_14,
                            coulomb_const, force_units, energy_units)

The Coulomb electrostatic interaction modified using the reaction field approximation.
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
                                    box_size,
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
                                            box_size,
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
