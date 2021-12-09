"""
    CoulombReactionField(; dist_cutoff, solvent_dielectric, nl_only, weight_14,
                            coulomb_const, force_unit, energy_unit)

The Coulomb electrostatic interaction modified using the reaction field approximation.
"""
struct CoulombReactionField{D, S, W, T, F, E} <: GeneralInteraction
    dist_cutoff::D
    solvent_dielectric::S
    nl_only::Bool
    weight_14::W
    coulomb_const::T
    force_unit::F
    energy_unit::E
end

const solventdielectric = 78.3

function CoulombReactionField(;
                    dist_cutoff,
                    solvent_dielectric=solventdielectric,
                    nl_only=false,
                    weight_14=1.0,
                    coulomb_const=coulombconst,
                    force_unit=u"kJ * mol^-1 * nm^-1",
                    energy_unit=u"kJ * mol^-1")
    return CoulombReactionField{typeof(dist_cutoff), typeof(solvent_dielectric), typeof(weight_14),
                                typeof(coulomb_const), typeof(force_unit), typeof(energy_unit)}(
        dist_cutoff, solvent_dielectric, nl_only, weight_14,
        coulomb_const, force_unit, energy_unit)
end

@inline @inbounds function force(inter::CoulombReactionField,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    box_size,
                                    weight_14::Bool=false)
    dr = vector(coord_i, coord_j, box_size)
    r2 = sum(abs2, dr)

    if r2 > (inter.dist_cutoff ^ 2)
        return ustrip.(zero(coord_i)) * inter.force_unit
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
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer,
                                    weight_14::Bool=false)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    if r2 > (inter.dist_cutoff ^ 2)
        return ustrip(zero(s.timestep)) * inter.energy_unit
    end

    coulomb_const = inter.coulomb_const
    qi, qj = s.atoms[i].charge, s.atoms[j].charge
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
