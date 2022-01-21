export LennardJones

@doc raw"""
    LennardJones(; cutoff, nl_only, lorentz_mixing, weight_14, weight_solute_solvent,
                 force_units, energy_units, skip_shortcut)

The Lennard-Jones 6-12 interaction. The potential is given by
```math
V_{ij}(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]
```
and the force on each atom by
```math
\begin{aligned}
\vec{F}_i &= 24\varepsilon_{ij} \left(2\frac{\sigma_{ij}^{12}}{r_{ij}^{13}} - \frac{\sigma_{ij}^6}{r_{ij}^{7}}\right) \frac{\vec{r}_{ij}}{r_{ij}} \\
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}^{6}}{r_{ij}^{6}}\right)^2 -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \vec{r}_{ij}
\end{aligned}
```
"""
struct LennardJones{S, C, W, WS, F, E} <: GeneralInteraction
    cutoff::C
    nl_only::Bool
    lorentz_mixing::Bool
    weight_14::W
    weight_solute_solvent::WS
    force_units::F
    energy_units::E
end

function LennardJones(;
                        cutoff=NoCutoff(),
                        nl_only=false,
                        lorentz_mixing=true,
                        weight_14=1,
                        weight_solute_solvent=1,
                        force_units=u"kJ * mol^-1 * nm^-1",
                        energy_units=u"kJ * mol^-1",
                        skip_shortcut=false)
    return LennardJones{skip_shortcut, typeof(cutoff), typeof(weight_14), typeof(weight_solute_solvent),
                        typeof(force_units), typeof(energy_units)}(
        cutoff, nl_only, lorentz_mixing, weight_14, weight_solute_solvent, force_units, energy_units)
end

is_solute(at::Atom) = at.solute
is_solute(at) = false

@inline @inbounds function force(inter::LennardJones{S, C},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    box_size,
                                    weight_14::Bool=false) where {S, C}
    r2 = sum(abs2, dr)

    if !S && (iszero(atom_i.ϵ) || iszero(atom_j.ϵ) || iszero(atom_i.σ) || iszero(atom_j.σ))
        return ustrip.(zero(coord_i)) * inter.force_units
    end

    # Lorentz-Berthelot mixing rules use the arithmetic average for σ
    # Otherwise use the geometric average
    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    if (is_solute(atom_i) && !is_solute(atom_j)) || (is_solute(atom_j) && !is_solute(atom_i))
        ϵ = inter.weight_solute_solvent * sqrt(atom_i.ϵ * atom_j.ϵ)
    else
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
    end

    cutoff = inter.cutoff
    σ2 = σ^2
    params = (σ2, ϵ)

    if cutoff_points(C) == 0
        f = force_divr_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        f = force_divr_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        if r2 < cutoff.activation_dist
            f = force_divr_nocutoff(inter, r2, inv(r2), params)
        else
            f = force_divr_cutoff(cutoff, r2, inter, params)
        end
    end

    if weight_14
        return f * dr * inter.weight_14
    else
        return f * dr
    end
end

@fastmath function force_divr_nocutoff(::LennardJones, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3

    return (24ϵ * invr2) * (2 * six_term ^ 2 - six_term)
end

@inline @inbounds function potential_energy(inter::LennardJones{S, C},
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            box_size,
                                            weight_14::Bool=false) where {S, C}
    r2 = sum(abs2, dr)

    if !S && (iszero(atom_i.ϵ) || iszero(atom_j.ϵ) || iszero(atom_i.σ) || iszero(atom_j.σ))
        return ustrip(zero(box_size[1])) * inter.energy_units
    end

    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    if (is_solute(atom_i) && !is_solute(atom_j)) || (is_solute(atom_j) && !is_solute(atom_i))
        ϵ = inter.weight_solute_solvent * sqrt(atom_i.ϵ * atom_j.ϵ)
    else
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
    end

    cutoff = inter.cutoff
    σ2 = σ^2
    params = (σ2, ϵ)

    if cutoff_points(C) == 0
        pe = potential(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(box_size[1])) * inter.energy_units

        pe = potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(box_size[1])) * inter.energy_units

        if r2 < cutoff.activation_dist
            pe = potential(inter, r2, inv(r2), params)
        else
            pe = potential_cutoff(cutoff, r2, inter, params)
        end
    end

    if weight_14
        return pe * inter.weight_14
    else
        return pe
    end
end

@fastmath function potential(::LennardJones, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3

    return 4ϵ * (six_term ^ 2 - six_term)
end
