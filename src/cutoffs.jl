# Cutoff strategies for long-range interactions

export
    NoCutoff,
    ShiftedPotentialCutoff,
    ShiftedForceCutoff

"""
    NoCutoff()

Placeholder cutoff that does not alter forces or potentials.
"""
struct NoCutoff <: AbstractCutoff end

cutoff_points(::Type{NoCutoff}) = 0

"""
    ShiftedPotentialCutoff(cutoff_dist)

Cutoff that shifts the potential to be continuous at a specified cutoff point.
"""
struct ShiftedPotentialCutoff{D, S, I} <: AbstractCutoff
    cutoff_dist::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedPotentialCutoff(cutoff_dist)
    return ShiftedPotentialCutoff(cutoff_dist, cutoff_dist ^ 2, inv(cutoff_dist ^ 2))
end

cutoff_points(::Type{ShiftedPotentialCutoff{D, S, I}}) where {D, S, I} = 1

"""
    ShiftedForceCutoff(cutoff_dist)

Cutoff that shifts the force to be continuous at a specified cutoff point.
"""
struct ShiftedForceCutoff{D, S, I} <: AbstractCutoff
    cutoff_dist::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedForceCutoff(cutoff_dist)
    return ShiftedForceCutoff(cutoff_dist, cutoff_dist ^ 2, inv(cutoff_dist ^ 2))
end

cutoff_points(::Type{ShiftedForceCutoff{D, S, I}}) where {D, S, I} = 1

force_divr_cutoff(::NoCutoff, r2, inter, params) = force_divr_nocutoff(inter, r2, inv(r2), params)
potential_cutoff(::NoCutoff, r2, inter, params) = potential(inter, r2, inv(r2), params)

force_divr_cutoff(::ShiftedPotentialCutoff, r2, inter, params) = force_divr_nocutoff(inter, r2, inv(r2), params)

function potential_cutoff(cutoff::ShiftedPotentialCutoff, r2, inter, params)
    potential(inter, r2, inv(r2), params) - potential(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end

function force_divr_cutoff(cutoff::ShiftedForceCutoff, r2, inter, params)
    return force_divr_nocutoff(inter, r2, inv(r2), params) - force_divr_nocutoff(
                                inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end

@fastmath function potential_cutoff(cutoff::ShiftedForceCutoff, r2, inter, params)
    invr2 = inv(r2)
    r = √r2
    rc = cutoff.cutoff_dist
    fc = force_divr_nocutoff(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params) * r

    potential(inter, r2, invr2, params) - (r - rc) * fc -
        potential(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end
