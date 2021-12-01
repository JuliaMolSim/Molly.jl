"""
    CoulombReactionField(; cutoff_dist, solvent_dielectric, nl_only, weight_14,
                            coulomb_const, force_unit, energy_unit)

The Coulomb electrostatic interaction modified using the reaction field approximation.
"""
struct CoulombReactionField{D, S, W, T, F, E, D2, K, R} <: GeneralInteraction
    cutoff_dist::D
    solvent_dielectric::S
    nl_only::Bool
    weight_14::W
    coulomb_const::T
    force_unit::F
    energy_unit::E
    sqdist_cutoff::D2
    krf::K
    crf::R
    krf_14::K
    crf_14::R
end

const solventdielectric = 78.3

# Defined as a function since this is reused elsewhere
function coulombproperties(cutoff_dist, solvent_dielectric)
    sqdist_cutoff = cutoff_dist ^ 2
    krf = (1 / (cutoff_dist ^ 3)) * ((solvent_dielectric - 1) / (2 * solvent_dielectric + 1))
    crf = (1 /  cutoff_dist     ) * ((3 * solvent_dielectric) / (2 * solvent_dielectric + 1))
    krf_14 = (1 / (cutoff_dist ^ 3)) * 0
    crf_14 = (1 /  cutoff_dist     ) * 0
    return 
end

function CoulombReactionField(;
                    cutoff_dist,
                    solvent_dielectric=solventdielectric,
                    nl_only=false,
                    weight_14=1.0,
                    coulomb_const=coulombconst,
                    force_unit=u"kJ * mol^-1 * nm^-1",
                    energy_unit=u"kJ * mol^-1")
    sqdist_cutoff, krf, crf, krf_14, crf_14 = coulombproperties(cutoff_dist, solvent_dielectric)
    return CoulombReactionField{typeof(cutoff_dist), typeof(solvent_dielectric), typeof(weight_14),
                                typeof(coulomb_const), typeof(force_unit), typeof(energy_unit),
                                typeof(sqdist_cutoff), typeof(krf), typeof(crf)}(
        cutoff_dist, solvent_dielectric, nl_only, weight_14, coulomb_const,
        force_unit, energy_unit, sqdist_cutoff, krf, crf, krf_14, crf_14)
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

    if r2 > inter.sqdist_cutoff
        return ustrip.(zero(coord_i)) * inter.force_unit
    end

    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = √r2
    i, j = atom_i.index, atom_j.index
    if weight_14
        # 1-4 interactions do not use the reaction field approximation
        krf = inter.krf_14
    else
        krf = inter.krf
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

    if r2 > inter.sqdist_cutoff
        return ustrip(zero(s.timestep)) * inter.energy_unit
    end

    coulomb_const = inter.coulomb_const
    qi, qj = s.atoms[i].charge, s.atoms[j].charge
    r = √r2
    if weight_14
        # 1-4 interactions do not use the reaction field approximation
        krf = inter.krf_14
        crf = inter.crf_14
    else
        krf = inter.krf
        crf = inter.crf
    end

    pe = (coulomb_const * qi * qj) * (inv(r) + krf * r2 - crf)

    if weight_14
        return pe * inter.weight_14
    else
        return pe
    end
end
