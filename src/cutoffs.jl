# Cutoff strategies for long-range interactions

export
    NoCutoff,
    DistanceCutoff,
    ShiftedPotentialCutoff,
    ShiftedForceCutoff,
    CubicSplineCutoff

"""
    NoCutoff()

Placeholder cutoff that does not alter forces or potentials.
"""
struct NoCutoff end

cutoff_points(::Type{NoCutoff}) = 0

"""
    DistanceCutoff(dist_cutoff)

Cutoff that sets the potential and force to be zero past a specified cutoff point.
"""
struct DistanceCutoff{D, S}
    dist_cutoff::D
    sqdist_cutoff::S
end

function DistanceCutoff(dist_cutoff)
    return DistanceCutoff(dist_cutoff, dist_cutoff ^ 2)
end

cutoff_points(::Type{DistanceCutoff{D, S}}) where {D, S} = 1

force_apply_cutoff(::DistanceCutoff, inter, r2, params) = pairwise_force(inter, r2, params)
pe_apply_cutoff(::DistanceCutoff, inter, r2, params) = pairwise_pe(inter, r2, params)

"""
    ShiftedPotentialCutoff(dist_cutoff)

Cutoff that shifts the potential to be continuous at a specified cutoff point.
"""
struct ShiftedPotentialCutoff{D, S}
    dist_cutoff::D
    sqdist_cutoff::S
end

function ShiftedPotentialCutoff(dist_cutoff)
    return ShiftedPotentialCutoff(dist_cutoff, dist_cutoff ^ 2)
end

cutoff_points(::Type{ShiftedPotentialCutoff{D, S}}) where {D, S} = 1

function force_apply_cutoff(::ShiftedPotentialCutoff, inter, r2, params)
    return pairwise_force(inter, r2, params)
end

function pe_apply_cutoff(cutoff::ShiftedPotentialCutoff, inter, r2, params)
    pe_r = pairwise_pe(inter, r2, params)
    pe_cut = pairwise_pe(inter, cutoff.sqdist_cutoff, params)
    return pe_r - pe_cut
end

"""
    ShiftedForceCutoff(dist_cutoff)

Cutoff that shifts the force to be continuous at a specified cutoff point.
"""
struct ShiftedForceCutoff{D, S}
    dist_cutoff::D
    sqdist_cutoff::S
end

function ShiftedForceCutoff(dist_cutoff)
    return ShiftedForceCutoff(dist_cutoff, dist_cutoff ^ 2)
end

cutoff_points(::Type{ShiftedForceCutoff{D, S}}) where {D, S} = 1

function force_apply_cutoff(cutoff::ShiftedForceCutoff, inter, r2, params)
    f_r = pairwise_force(inter, r2, params)
    f_cut = pairwise_force(inter, cutoff.sqdist_cutoff, params)
    return f_r - f_cut
end

function pe_apply_cutoff(cutoff::ShiftedForceCutoff, inter, r2, params)
    r = sqrt(r2)
    f_cut = pairwise_force(inter, cutoff.sqdist_cutoff, params)
    pe_r = pairwise_pe(inter, r2, params)
    pe_cut = pairwise_pe(inter, cutoff.sqdist_cutoff, params)
    return pe_r + (r - cutoff.dist_cutoff) * f_cut - pe_cut
end

"""
    CubicSplineCutoff(dist_activation, dist_cutoff)

Cutoff that interpolates the true potential and zero between an activation point
and a cutoff point, using a cubic Hermite spline.
"""
struct CubicSplineCutoff{D, S}
    dist_activation::D
    dist_cutoff::D
    sqdist_activation::S
    sqdist_cutoff::S
end

function CubicSplineCutoff(dist_activation, dist_cutoff)
    if dist_cutoff <= dist_activation
        error("the cutoff radius must be strictly larger than the activation radius")
    end
    return CubicSplineCutoff(dist_activation, dist_cutoff, dist_activation^2, dist_cutoff^2)
end

cutoff_points(::Type{CubicSplineCutoff{D, S}}) where {D, S} = 2

function force_apply_cutoff(cutoff::CubicSplineCutoff, inter, r2, params)
    r = sqrt(r2)
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff - cutoff.dist_activation)
    Va = pairwise_pe(inter, cutoff.sqdist_activation, params)
    dVa = -pairwise_force(inter, cutoff.sqdist_activation, params)
    return -(6t^2 - 6t) * Va / (cutoff.dist_cutoff - cutoff.dist_activation) - (3t^2 - 4t + 1) * dVa
end

function pe_apply_cutoff(cutoff::CubicSplineCutoff, inter, r2, params)
    r = sqrt(r2)
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff - cutoff.dist_activation)
    Va = pairwise_pe(inter, cutoff.sqdist_activation, params)
    dVa = -pairwise_force(inter, cutoff.sqdist_activation, params)
    return (2t^3 - 3t^2 + 1) * Va + (t^3 - 2t^2 + t) *
           (cutoff.dist_cutoff - cutoff.dist_activation) * dVa
end

Base.:+(c1::T, ::T) where {T <: Union{NoCutoff, DistanceCutoff, ShiftedPotentialCutoff,
                                      ShiftedForceCutoff, CubicSplineCutoff}} = c1

function force_cutoff(cutoff::C, inter, r2, params, force_units) where C
    if cutoff_points(C) == 0
        return pairwise_force(inter, r2, params)
    elseif cutoff_points(C) == 1
        return force_apply_cutoff(cutoff, inter, r2, params) * (r2 <= cutoff.sqdist_cutoff)
    elseif cutoff_points(C) == 2
        return ifelse(
            r2 < cutoff.sqdist_activation,
            pairwise_force(inter, r2, params),
            force_apply_cutoff(cutoff, inter, r2, params) * (r2 <= cutoff.sqdist_cutoff),
        )
    end
end

function pe_cutoff(cutoff::C, inter, r2, params, energy_units) where C
    if cutoff_points(C) == 0
        return pairwise_pe(inter, r2, params)
    elseif cutoff_points(C) == 1
        return pe_apply_cutoff(cutoff, inter, r2, params) * (r2 <= cutoff.sqdist_cutoff)
    elseif cutoff_points(C) == 2
        return ifelse(
            r2 < cutoff.sqdist_activation,
            pairwise_pe(inter, r2, params),
            pe_apply_cutoff(cutoff, inter, r2, params) * (r2 <= cutoff.sqdist_cutoff),
        )
    end
end
