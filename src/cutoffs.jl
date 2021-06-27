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
    ShiftedPotentialCutoff(; cutoff_dist, max_force)

Cutoff that shifts the potential to be continuous at a specified cutoff point.
"""
struct ShiftedPotentialCutoff{F, D, S, I} <: AbstractCutoff
    max_force::F
    cutoff_dist::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedPotentialCutoff(; cutoff_dist, max_force=nothing)
    return ShiftedPotentialCutoff(max_force, cutoff_dist, cutoff_dist ^ 2, inv(cutoff_dist ^ 2))
end

cutoff_points(::Type{ShiftedPotentialCutoff{F, D, S, I}}) where {F, D, S, I} = 1

"""
    ShiftedForceCutoff(; cutoff_dist, max_force)

Cutoff that shifts the force to be continuous at a specified cutoff point.
"""
struct ShiftedForceCutoff{F, D, S, I} <: AbstractCutoff
    max_force::F
    cutoff_dist::D
    sqdist_cutoff::S
    inv_sqdist_cutoff::I
end

function ShiftedForceCutoff(; cutoff_dist, max_force=nothing)
    return ShiftedForceCutoff(max_force, cutoff_dist, cutoff_dist ^ 2, inv(cutoff_dist ^ 2))
end

cutoff_points(::Type{ShiftedForceCutoff{F, D, S, I}}) where {F, D, S, I} = 1

force_cutoff(::NoCutoff, r2, inter, params) = force_nocutoff(inter, r2, inv(r2), params)
potential_cutoff(::NoCutoff, r2, inter, params) = potential(inter, r2, inv(r2), params)

function force_cutoff(cutoff::ShiftedPotentialCutoff{F}, r2, inter, params) where F
    f = force_nocutoff(inter, r2, inv(r2), params)

    if @generated
        quote
            if !(F === Nothing)
                f = min(f, cutoff.max_force)
            end
        end
    else
        if !(F === Nothing)
            f = min(f, cutoff.max_force)
        end
    end

    return f
end

function potential_cutoff(cutoff::ShiftedPotentialCutoff, r2, inter, params)
    potential(inter, r2, inv(r2), params) - potential(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end

function force_cutoff(cutoff::ShiftedForceCutoff{F}, r2, inter, params) where F
    invr2 = inv(r2)
    f = force_nocutoff(inter, r2, invr2, params) - force_nocutoff(
                                inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)

    if @generated
        quote
            if !(F === Nothing)
                f = min(f, cutoff.max_force)
            end
        end
    else
        if !(F === Nothing)
            f = min(f, cutoff.max_force)
        end
    end

    return f
end

@fastmath function potential_cutoff(cutoff::ShiftedForceCutoff, r2, inter, params)
    invr2 = inv(r2)
    r = √r2
    rc = cutoff.cutoff_dist
    fc = force_nocutoff(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params) * r

    potential(inter, r2, invr2, params) - (r - rc) * fc -
        potential(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end
