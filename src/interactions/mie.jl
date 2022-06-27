export Mie

"""
    Mie(; m, n, cutoff, nl_only, lorentz_mixing, force_units, energy_units, skip_shortcut)

The Mie generalized interaction.
When `m` equals 6 and `n` equals 12 this is equivalent to the Lennard-Jones interaction.
"""
struct Mie{S, C, T, F, E} <: PairwiseInteraction
    m::T
    n::T
    cutoff::C
    nl_only::Bool
    lorentz_mixing::Bool
    force_units::F
    energy_units::E
    mn_fac::T
end

function Mie(;
                m,
                n,
                cutoff=NoCutoff(),
                nl_only=false,
                lorentz_mixing=true,
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
                skip_shortcut=false)
    mn_fac = convert(typeof(m), (n / (n - m)) * (n / m) ^ (m / (n - m)))
    return Mie{skip_shortcut, typeof(cutoff), typeof(m), typeof(force_units), typeof(energy_units)}(
        m, n, cutoff, nl_only, lorentz_mixing, force_units, energy_units, mn_fac)
end

@fastmath @inbounds function force(inter::Mie{S, C, T},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    boundary) where {S, C, T}
    r2 = sum(abs2, dr)
    r = √r2

    if !S && (iszero(atom_i.ϵ) || iszero(atom_j.ϵ) || iszero(atom_i.σ) || iszero(atom_j.σ))
        return ustrip.(zero(coord_i)) * inter.force_units
    end

    # Lorentz-Berthelot mixing rules use the arithmetic average for σ
    # Otherwise use the geometric average
    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

    cutoff = inter.cutoff
    m = inter.m
    n = inter.n
    # Derivative obtained via wolfram
    const_mn = inter.mn_fac * ϵ
    σ_r = σ / r
    params = (m, n, σ_r, const_mn)

    if cutoff_points(C) == 0
        f = force_divr_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        f = force_divr_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        if r2 < cutoff.sqdist_activation
            f = force_divr_nocutoff(inter, r2, inv(r2), params)
        else
            f = force_divr_cutoff(cutoff, r2, inter, params)
        end
    end

    return f * dr
end

@fastmath function force_divr_nocutoff(::Mie, r2, invr2, (m, n, σ_r, const_mn))
    return -const_mn / r2 * (m * σ_r ^ m - n * σ_r ^ n)
end

@inline @inbounds function potential_energy(inter::Mie{S, C, T},
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            boundary) where {S, C, T}
    r2 = sum(abs2, dr)
    r = √r2

    if !S && (iszero(atom_i.ϵ) || iszero(atom_j.ϵ) || iszero(atom_i.σ) || iszero(atom_j.σ))
        return ustrip(zero(coord_i[1])) * inter.energy_units
    end

    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)

    cutoff = inter.cutoff
    m = inter.m
    n = inter.n
    const_mn = inter.mn_fac * ϵ
    σ_r = σ / r
    params = (m, n, σ_r, const_mn)

    if cutoff_points(C) == 0
        potential(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(coord_i[1])) * inter.energy_units

        potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(coord_i[1])) * inter.energy_units

        if r2 < cutoff.sqdist_activation
            potential(inter, r2, inv(r2), params)
        else
            potential_cutoff(cutoff, r2, inter, params)
        end
    end
end

@fastmath function potential(::Mie, r2, invr2, (m, n, σ_r, const_mn))
    return const_mn * (σ_r ^ n - σ_r ^ m)
end
