# Cutoff strategies for long-range interactions

export
    NoCutoff,
    DistanceCutoff,
    ShiftedPotentialCutoff,
    ShiftedForceCutoff

"""
    NoCutoff()

Placeholder cutoff that does not alter forces or potentials.
"""
struct NoCutoff <: AbstractCutoff end

cutoff_points(::Type{NoCutoff}) = 0

"""
    DistanceCutoff(dist_cutoff)

Cutoff that sets the potential and force to be zero past a specified cutoff point.
"""
struct DistanceCutoff{D, S, I} <: AbstractCutoff
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function DistanceCutoff(dist_cutoff)
    return DistanceCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{DistanceCutoff{D, S, I}}) where {D, S, I} = 1

"""
    ShiftedPotentialCutoff(dist_cutoff)

Cutoff that shifts the potential to be continuous at a specified cutoff point.
"""
struct ShiftedPotentialCutoff{D, S, I} <: AbstractCutoff
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedPotentialCutoff(dist_cutoff)
    return ShiftedPotentialCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{ShiftedPotentialCutoff{D, S, I}}) where {D, S, I} = 1

"""
    ShiftedForceCutoff(dist_cutoff)

Cutoff that shifts the force to be continuous at a specified cutoff point.
"""
struct ShiftedForceCutoff{D, S, I} <: AbstractCutoff
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedForceCutoff(dist_cutoff)
    return ShiftedForceCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{ShiftedForceCutoff{D, S, I}}) where {D, S, I} = 1

force_divr_cutoff(::NoCutoff, r2, inter, params) = force_divr_nocutoff(inter, r2, inv(r2), params)
potential_cutoff(::NoCutoff, r2, inter, params) = potential(inter, r2, inv(r2), params)

force_divr_cutoff(::DistanceCutoff, r2, inter, params) = force_divr_nocutoff(inter, r2, inv(r2), params)
potential_cutoff(::DistanceCutoff, r2, inter, params) = potential(inter, r2, inv(r2), params)

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
    rc = cutoff.dist_cutoff
    fc = force_divr_nocutoff(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params) * r

    potential(inter, r2, invr2, params) - (r - rc) * fc -
        potential(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end
