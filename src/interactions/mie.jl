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
    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ) || i == j
        return
    end
    m = inter.m
    n = inter.n
    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r = norm(dr)
    abs2(r) > sqdist_cutoff_nb && return
    # Derivative obtained via wolfram
    const_mn = inter.mn_fac * ϵ / r
    σ_r = σ / r
    f = m * σ_r ^ m - n * σ_r ^ n
    # Limit this to 100 as a fudge to stop it exploding
    f = min(-f * const_mn / r, 100)
    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end
