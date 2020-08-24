"""
    SoftSphere(nl_only)

The soft-sphere potential.
"""
struct SoftSphere{S, C} <: GeneralInteraction
    cutoff::C
    nl_only::Bool
end

SoftSphere{S}(cutoff, nl_only) where S =
    SoftSphere{S, typeof(cutoff)}(cutoff, nl_only)

SoftSphere(cutoff, nl_only) =
    SoftSphere{false, typeof(cutoff)}(cutoff, nl_only)

SoftSphere(nl_only=false) =
    SoftSphere(ShiftedPotentialCutoff(3.0), nl_only)

@inline @inbounds function force!(forces,
                                    inter::SoftSphere,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    i == j && return
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    if !S && iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return
    end

    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)

    cutoff = inter.cutoff
    σ2 = σ^2
    params = (σ2, ϵ)

    if cutoff_points(C) == 0
        f = force_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        sqdist_cutoff = cutoff.sqdist_cutoff * σ2
        r2 > sqdist_cutoff && return

        f = force_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        sqdist_cutoff = cutoff.sqdist_cutoff * σ2
        activation_dist = cutoff.activation_dist * σ2

        r2 > sqdist_cutoff && return

        if r2 < activation_dist
            f = force_nocutoff(inter, r2, inv(r2), params)
        else
            f = force_cutoff(cutoff, r2, inter, params)
        end
    end

    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end

@fastmath function force_nocutoff(::SoftSphere, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3

    return (24ϵ * invr2) * 2 * six_term ^ 2
end

@inbounds function potential_energy(inter::SoftSphere,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    U = eltype(s.coords[i]) # this is not Unitful compatible
    i == j && return zero(T)

    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    if !S && iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return zero(U)
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
        r2 > sqdist_cutoff && return zero(U)

        potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > sqdist_cutoff && return zero(U)

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
