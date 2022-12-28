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

force_divr_cutoff(::NoCutoff, r2, inter, params) = force_divr_nocutoff(inter, r2, inv(r2), params)
potential_cutoff(::NoCutoff, r2, inter, params) = potential(inter, r2, inv(r2), params)

"""
    DistanceCutoff(dist_cutoff)

Cutoff that sets the potential and force to be zero past a specified cutoff point.
"""
struct DistanceCutoff{D, S, I}
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function DistanceCutoff(dist_cutoff)
    return DistanceCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{DistanceCutoff{D, S, I}}) where {D, S, I} = 1

force_divr_cutoff(::DistanceCutoff, r2, inter, params) = force_divr_nocutoff(inter, r2, inv(r2), params)
potential_cutoff(::DistanceCutoff, r2, inter, params) = potential(inter, r2, inv(r2), params)

"""
    ShiftedPotentialCutoff(dist_cutoff)

Cutoff that shifts the potential to be continuous at a specified cutoff point.
"""
struct ShiftedPotentialCutoff{D, S, I}
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedPotentialCutoff(dist_cutoff)
    return ShiftedPotentialCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{ShiftedPotentialCutoff{D, S, I}}) where {D, S, I} = 1

force_divr_cutoff(::ShiftedPotentialCutoff, r2, inter, params) = force_divr_nocutoff(inter, r2, inv(r2), params)

function potential_cutoff(cutoff::ShiftedPotentialCutoff, r2, inter, params)
    potential(inter, r2, inv(r2), params) - potential(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end

"""
    ShiftedForceCutoff(dist_cutoff)

Cutoff that shifts the force to be continuous at a specified cutoff point.
"""
struct ShiftedForceCutoff{D, S, I}
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedForceCutoff(dist_cutoff)
    return ShiftedForceCutoff(dist_cutoff, dist_cutoff ^ 2, inv(dist_cutoff ^ 2))
end

cutoff_points(::Type{ShiftedForceCutoff{D, S, I}}) where {D, S, I} = 1

function force_divr_cutoff(cutoff::ShiftedForceCutoff, r2, inter, params)
    return force_divr_nocutoff(inter, r2, inv(r2), params) - force_divr_nocutoff(
                                inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end

@fastmath function potential_cutoff(cutoff::ShiftedForceCutoff, r2, inter, params)
    invr2 = inv(r2)
    r = √r2
    rc = cutoff.dist_cutoff
    fc = force_divr_nocutoff(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params) * r

    potential(inter, r2, invr2, params) + (r - rc) * fc -
        potential(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end

"""
    CubicSplineCutoff(dist_activation, dist_cutoff)

Cutoff that interpolates the true potential and zero between an activation point
and a cutoff point, using a cubic Hermite spline.
"""
struct CubicSplineCutoff{D, S, I}
    dist_cutoff::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
    dist_activation::D
    sqdist_activation::S
    inv_sqdist_activation::I
end

function CubicSplineCutoff(dist_activation, dist_cutoff)
    if dist_cutoff <= dist_activation
        error("The cutoff radius must be strictly larger than the activation radius")
    end

    D, S, I = typeof.([dist_cutoff, dist_cutoff^2, inv(dist_cutoff^2)])

    return CubicSplineCutoff{D, S, I}(dist_cutoff, dist_cutoff^2, inv(dist_cutoff^2), dist_activation,
                                        dist_activation^2, inv(dist_activation^2))
end

cutoff_points(::Type{CubicSplineCutoff{D, S, I}}) where {D, S, I} = 2

@fastmath function force_divr_cutoff(cutoff::CubicSplineCutoff, r2, inter, params)
    r = √r2
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff-cutoff.dist_activation)

    Va = potential(inter, cutoff.sqdist_activation, cutoff.inv_sqdist_activation, params)
    dVa = -force_divr_nocutoff(inter, cutoff.sqdist_activation, cutoff.inv_sqdist_activation, params) * cutoff.dist_activation
    
    return -((6t^2 - 6t) * Va / (cutoff.dist_cutoff-cutoff.dist_activation) + (3t^2 - 4t + 1) * dVa)/r
end

@fastmath function potential_cutoff(cutoff::CubicSplineCutoff, r2, inter, params)
    r = √r2
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff-cutoff.dist_activation)

    Va = potential(inter, cutoff.sqdist_activation, cutoff.inv_sqdist_activation, params)
    dVa = -force_divr_nocutoff(inter, cutoff.sqdist_activation, cutoff.inv_sqdist_activation, params) * cutoff.dist_activation
    
    return (2t^3 - 3t^2 + 1) * Va + (t^3 - 2t^2 + t) * (cutoff.dist_cutoff-cutoff.dist_activation) * dVa
end
