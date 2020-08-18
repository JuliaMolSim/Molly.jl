"""
    Mie(m, n, nl_only)

The Mie generalized interaction.
When `m` equals 6 and `n` equals 12 this is equivalent to the Lennard Jones interaction.
"""
struct Mie{T} <: GeneralInteraction
    m::T
    n::T
    nl_only::Bool
    mn_fac::T
end

Mie(m, n, nl_only) = Mie(m, n, nl_only, convert(typeof(m), (n / (n - m)) * (n / m) ^ (m / (n - m))))
Mie(m, n) = Mie(m, n, false)

@fastmath @inbounds function force!(forces,
                                    inter::Mie,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    i == j && return    # TODO: get rid of this check
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    r = √r2

    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return
    end

    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)
    
    cutoff = inter.cutoff
    m = inter.m
    n = inter.n
    # Derivative obtained via wolfram
    const_mn = inter.mn_fac * ϵ
    σ_r = σ / r
    params = (m, n, σ_r, const_mn)

    if cutoff_points(C) == 0
        f = force(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        sqdist_cutoff = cutoff.sqdist_cutoff * σ2
        r2 > sqdist_cutoff && return

        f = force_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        sqdist_cutoff = cutoff.sqdist_cutoff * σ2
        activation_dist = cutoff.activation_dist * σ2

        r2 > sqdist_cutoff && return
        
        if r2 < activation_dist
            f = force(inter, r2, inv(r2), params)
        else
            f = force_cutoff(cutoff, r2, inter, params)
        end
    end

    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end

@fastmath function force(::Mie, r2, invr2, (m, n, σ_r, const_mn))
    return -const_mn / r2 * (m * σ_r ^ m - n * σ_r ^ n)
end

@inline @inbounds function potential_energy(inter::Mie,
                                            s::Simulation,
                                            i::Integer,
                                            j::Integer)
    U = eltype(s.coords[i]) # this is not Unitful compatible
    i == j && return zero(T)

    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return zero(U)
    end

    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)
    
    cutoff = inter.cutoff
    m = inter.m
    n = inter.n
    const_mn = inter.mn_fac * ϵ
    σ_r = σ / r
    params = (m, n, σ_r, const_mn)

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

@fastmath function potential(::Mie, r2, invr2, (m, n, σ_r, const_mn))
    return const_mn * (σ_r ^ m - σ_r ^ m)
end
