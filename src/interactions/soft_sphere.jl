export SoftSphere

"""
    SoftSphere(; cutoff, nl_only, lorentz_mixing, force_units, energy_units, skip_shortcut)

The soft-sphere potential.
"""
struct SoftSphere{S, C, F, E} <: PairwiseInteraction
    cutoff::C
    nl_only::Bool
    lorentz_mixing::Bool
    force_units::F
    energy_units::E
end

function SoftSphere(;
                    cutoff=NoCutoff(),
                    nl_only=false,
                    lorentz_mixing=true,
                    force_units=u"kJ * mol^-1 * nm^-1",
                    energy_units=u"kJ * mol^-1",
                    skip_shortcut=false)
    return SoftSphere{skip_shortcut, typeof(cutoff), typeof(force_units), typeof(energy_units)}(
        cutoff, nl_only, lorentz_mixing, force_units, energy_units)
end

@inline @inbounds function force(inter::SoftSphere{S, C},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    box_size) where {S, C}
    r2 = sum(abs2, dr)

    if !S && (iszero(atom_i.ϵ) || iszero(atom_j.ϵ) || iszero(atom_i.σ) || iszero(atom_j.σ))
        return ustrip.(zero(coord_i)) * inter.force_units
    end

    # Lorentz-Berthelot mixing rules use the arithmetic average for σ
    # Otherwise use the geometric average
    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

    cutoff = inter.cutoff
    σ2 = σ^2
    params = (σ2, ϵ)

    if cutoff_points(C) == 0
        f = force_divr_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        f = force_divr_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        if r2 < cutoff.activation_dist
            f = force_divr_nocutoff(inter, r2, inv(r2), params)
        else
            f = force_divr_cutoff(cutoff, r2, inter, params)
        end
    end

    return f * dr
end

@fastmath function force_divr_nocutoff(::SoftSphere, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3

    return (24ϵ * invr2) * 2 * six_term ^ 2
end

@inbounds function potential_energy(inter::SoftSphere{S, C},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    box_size) where {S, C}
    r2 = sum(abs2, dr)

    if !S && (iszero(atom_i.ϵ) || iszero(atom_j.ϵ) || iszero(atom_i.σ) || iszero(atom_j.σ))
        return ustrip(zero(box_size[1])) * inter.energy_units
    end

    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

    cutoff = inter.cutoff
    σ2 = σ^2
    params = (σ2, ϵ)

    if cutoff_points(C) == 0
        potential(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(box_size[1])) * inter.energy_units

        potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(box_size[1])) * inter.energy_units

        if r2 < cutoff.activation_dist
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
