"""
    SoftSphere(; cutoff, nl_only, force_unit, energy_unit)

The soft-sphere potential.
"""
struct SoftSphere{S, C, F, E} <: GeneralInteraction
    cutoff::C
    nl_only::Bool
    force_unit::F
    energy_unit::E
end

SoftSphere{S}(cutoff, nl_only, force_unit, energy_unit) where S =
    SoftSphere{S, typeof(cutoff), typeof(force_unit), typeof(energy_unit)}(
        cutoff, nl_only, force_unit, energy_unit)

function SoftSphere(;
                    cutoff=NoCutoff(),
                    nl_only=false,
                    force_unit=u"kJ * mol^-1 * nm^-1",
                    energy_unit=u"kJ * mol^-1")
    return SoftSphere{false, typeof(cutoff), typeof(force_unit), typeof(energy_unit)}(
        cutoff, nl_only, force_unit, energy_unit)
end

@inline @inbounds function force(inter::SoftSphere{S, C},
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    box_size) where {S, C}
    dr = vector(coord_i, coord_j, box_size)
    r2 = sum(abs2, dr)

    if !S && iszero(atom_i.σ) || iszero(atom_j.σ)
        return ustrip.(zero(coord_i)) * inter.force_unit
    end

    σ = sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

    cutoff = inter.cutoff
    σ2 = σ^2
    params = (σ2, ϵ)

    if cutoff_points(C) == 0
        f = force_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        sqdist_cutoff = cutoff.sqdist_cutoff * σ2
        r2 > sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_unit

        f = force_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        sqdist_cutoff = cutoff.sqdist_cutoff * σ2
        activation_dist = cutoff.activation_dist * σ2

        r2 > sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_unit

        if r2 < activation_dist
            f = force_nocutoff(inter, r2, inv(r2), params)
        else
            f = force_cutoff(cutoff, r2, inter, params)
        end
    end

    return f * dr
end

@fastmath function force_nocutoff(::SoftSphere, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3

    return (24ϵ * invr2) * 2 * six_term ^ 2
end

@inbounds function potential_energy(inter::SoftSphere{S, C},
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer) where {S, C}
    zero_energy = ustrip(zero(s.timestep)) * inter.energy_unit
    i == j && return zero_energy

    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    if !S && iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return zero_energy
    end

    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)

    cutoff = inter.cutoff
    σ2 = σ^2
    params = (σ2, ϵ)

    if cutoff_points(C) == 0
        potential(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        sqdist_cutoff = cutoff.sqdist_cutoff * σ2
        r2 > sqdist_cutoff && return zero_energy

        potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > sqdist_cutoff && return zero_energy

        if r2 < activation_dist
            potential(inter, r2, inv(r2), params)
        else
            potential_cutoff(cutoff, r2, inter, params)
        end
    end
end

@fastmath function potential(::SoftSphere, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3

    return 4ϵ * (six_term ^ 2)
end
