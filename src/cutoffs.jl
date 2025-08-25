# Cutoff strategies for long-range interactions

export
    NoCutoff,
    DistanceCutoff,
    ShiftedPotentialCutoff,
    ShiftedForceCutoff,
    CubicSplineCutoff

"""
    NoCutoff()

Placeholder cutoff that does not alter the potential or force.
"""
struct NoCutoff end

cutoff_points(::Type{NoCutoff}) = 0

"""
    DistanceCutoff(dist_cutoff)

Cutoff that sets the potential and force to be zero past a specified cutoff distance.
"""
struct DistanceCutoff{D}
    dist_cutoff::D
end

cutoff_points(::Type{DistanceCutoff{D}}) where D = 1

pe_apply_cutoff(::DistanceCutoff, inter, r, params) = pairwise_pe(inter, r, params)
force_apply_cutoff(::DistanceCutoff, inter, r, params) = pairwise_force(inter, r, params)

"""
    ShiftedPotentialCutoff(dist_cutoff)

Cutoff that shifts the potential to be continuous at a specified cutoff distance.
"""
struct ShiftedPotentialCutoff{D}
    dist_cutoff::D
end

cutoff_points(::Type{ShiftedPotentialCutoff{D}}) where D = 1

function pe_apply_cutoff(cutoff::ShiftedPotentialCutoff, inter, r, params)
    pe_r = pairwise_pe(inter, r, params)
    pe_cut = pairwise_pe(inter, cutoff.dist_cutoff, params)
    return pe_r - pe_cut
end

function force_apply_cutoff(::ShiftedPotentialCutoff, inter, r, params)
    return pairwise_force(inter, r, params)
end

"""
    ShiftedForceCutoff(dist_cutoff)

Cutoff that shifts the force to be continuous at a specified cutoff distance.
"""
struct ShiftedForceCutoff{D}
    dist_cutoff::D
end

cutoff_points(::Type{ShiftedForceCutoff{D}}) where D = 1

function pe_apply_cutoff(cutoff::ShiftedForceCutoff, inter, r, params)
    pe_r = pairwise_pe(inter, r, params)
    pe_cut = pairwise_pe(inter, cutoff.dist_cutoff, params)
    f_cut = pairwise_force(inter, cutoff.dist_cutoff, params)
    return pe_r + (r - cutoff.dist_cutoff) * f_cut - pe_cut
end

function force_apply_cutoff(cutoff::ShiftedForceCutoff, inter, r, params)
    f_r = pairwise_force(inter, r, params)
    f_cut = pairwise_force(inter, cutoff.dist_cutoff, params)
    return f_r - f_cut
end

"""
    CubicSplineCutoff(dist_activation, dist_cutoff)

Cutoff that interpolates between the true potential at an activation distance
and zero at a cutoff distance using a cubic Hermite spline.
"""
struct CubicSplineCutoff{D}
    dist_activation::D
    dist_cutoff::D
end

function CubicSplineCutoff(dist_activation, dist_cutoff)
    if dist_cutoff <= dist_activation
        throw(ArgumentError("the cutoff radius $dist_cutoff must be larger " *
                            "than the activation radius $dist_activation"))
    end
    return CubicSplineCutoff(dist_activation, dist_cutoff)
end

cutoff_points(::Type{CubicSplineCutoff{D}}) where D = 2

function pe_apply_cutoff(cutoff::CubicSplineCutoff, inter, r, params)
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff - cutoff.dist_activation)
    Va = pairwise_pe(inter, cutoff.dist_activation, params)
    dVa = -pairwise_force(inter, cutoff.dist_activation, params)
    return (2t^3 - 3t^2 + 1) * Va + (t^3 - 2t^2 + t) *
           (cutoff.dist_cutoff - cutoff.dist_activation) * dVa
end

function force_apply_cutoff(cutoff::CubicSplineCutoff, inter, r, params)
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff - cutoff.dist_activation)
    Va = pairwise_pe(inter, cutoff.dist_activation, params)
    dVa = -pairwise_force(inter, cutoff.dist_activation, params)
    return -(6t^2 - 6t) * Va / (cutoff.dist_cutoff - cutoff.dist_activation) -
                    (3t^2 - 4t + 1) * dVa
end

Base.:+(c1::T, ::T) where {T <: Union{NoCutoff, DistanceCutoff, ShiftedPotentialCutoff,
                                      ShiftedForceCutoff, CubicSplineCutoff}} = c1

function pe_cutoff(cutoff::C, inter, r, params, energy_units) where C
    if cutoff_points(C) == 0
        return pairwise_pe(inter, r, params)
    elseif cutoff_points(C) == 1
        return pe_apply_cutoff(cutoff, inter, r, params) * (r <= cutoff.dist_cutoff)
    elseif cutoff_points(C) == 2
        return ifelse(
            r <= cutoff.dist_activation,
            pairwise_pe(inter, r, params),
            pe_apply_cutoff(cutoff, inter, r, params) * (r <= cutoff.dist_cutoff),
        )
    end
end

function force_cutoff(cutoff::C, inter, r, params, force_units) where C
    if cutoff_points(C) == 0
        return pairwise_force(inter, r, params)
    elseif cutoff_points(C) == 1
        return force_apply_cutoff(cutoff, inter, r, params) * (r <= cutoff.dist_cutoff)
    elseif cutoff_points(C) == 2
        return ifelse(
            r <= cutoff.dist_activation,
            pairwise_force(inter, r, params),
            force_apply_cutoff(cutoff, inter, r, params) * (r <= cutoff.dist_cutoff),
        )
    end
end
