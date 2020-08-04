export ShiftedPotentialCutoff, ShiftedForceCutoff, NoCutoff

struct NoCutoff <: AbstractCutoff end

cutoff_points(::Type{NoCutoff}) = 0

struct ShiftedPotentialCutoff{T} <: AbstractCutoff
    min_force::T
end

cutoff_points(::Type{ShiftedPotentialCutoff{T}}) where T = 1

struct ShiftedForceCutoff{L, T} <: AbstractCutoff
    min_force::T
end

cutoff_points(::Type{ShiftedForceCutoff}) = 1

function force_cutoff(cutoff::ShiftedPotentialCutoff{T}, r2, inter, params) where T
    invr2 = inv(r2)
    f = force(inter, invr2, params)

    if @generated
        quote
            if !(T === Nothing)
                f = min(f, cutoff.min_force)
            end
        end
    else
        if !(T === Nothing)
            f = min(f, cutoff.min_force)
        end
    end

    return f
end

function potential_cutoff(::ShiftedPotentialCutoff, r2, inter, params)
    invr2 = inv(r2)
    potential(inter, invr2, params) - potential(inter, inter.inv_sqdist_cutoff, params)
end

function force_cutoff(::ShiftedForceCutoff{T}, r2, inter, params) where T
    invr2 = inv(r2)
    f = force(inter, invr2, params) - force(inter, inter.inv_sqdist_cutoff, params)

    if @generated
        quote
            if !(T === Nothing)
                f = min(f, cutoff.min_force)
            end
        end
    else
        if !(T === Nothing)
            f = min(f, cutoff.min_force)
        end
    end

    return f
end

function potential_cutoff(::ShiftedForceCutoff{false}, r2, inter, params)
    # implement generic fallback
end

force_cutoff(::NoCutoff, r2, inter, params) = force(inter, inv(r2), params)
potential_cutoff(::NoCutoff, r2, inter, params) = potential(inter, inv(r2), params)