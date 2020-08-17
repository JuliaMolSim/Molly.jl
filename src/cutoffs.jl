export ShiftedPotentialCutoff, ShiftedForceCutoff, NoCutoff

struct NoCutoff <: AbstractCutoff end

cutoff_points(::Type{NoCutoff}) = 0

struct ShiftedPotentialCutoff{F, T} <: AbstractCutoff
    max_force::F
    cutoff_dist::T
    sqdist_cutoff::T
    inv_sqdist_cutoff::T
end

ShiftedPotentialCutoff(cutoff_dist) = ShiftedPotentialCutoff(
    100,
    cutoff_dist,
    cutoff_dist^2,
    inv(cutoff_dist),
)

cutoff_points(::Type{ShiftedPotentialCutoff{F, T}}) where {F, T} = 1

struct ShiftedForceCutoff{F, T} <: AbstractCutoff
    max_force::F
    cutoff_dist::T
    sqdist_cutoff::T
    inv_sqdist_cutoff::T
end

ShiftedForceCutoff(cutoff_dist) = ShiftedForceCutoff(
    100,
    cutoff_dist,
    cutoff_dist^2,
    inv(cutoff_dist),
)

cutoff_points(::Type{ShiftedForceCutoff{F, T}}) where {F, T} = 1

force_cutoff(::NoCutoff, r2, inter, params) = force(inter, r2, inv(r2), params)
potential_cutoff(::NoCutoff, r2, inter, params) = potential(inter, r2, inv(r2), params)

function force_cutoff(cutoff::ShiftedPotentialCutoff{F}, r2, inter, params) where F
    f = force(inter, r2, inv(r2), params)

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
    f = force(inter, r2, invr2, params) - force(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)

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
    fc = force(inter, cutoff.inv_sqdist_cutoff, params)

    potential(inter, r2, invr2, params) - (r - rc) * fc -
        potential(inter, cutoff.sqdist_cutoff, cutoff.inv_sqdist_cutoff, params)
end
