export
    LennardJones,
    LennardJonesSoftCore

@doc raw"""
    LennardJones(; cutoff, use_neighbors, lorentz_mixing, weight_special, weight_solute_solvent,
                 force_units, energy_units, skip_shortcut)

The Lennard-Jones 6-12 interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]
```
and the force on each atom by
```math
\begin{aligned}
\vec{F}_i &= 24\varepsilon_{ij} \left(2\frac{\sigma_{ij}^{12}}{r_{ij}^{13}} - \frac{\sigma_{ij}^6}{r_{ij}^{7}}\right) \frac{\vec{r}_{ij}}{r_{ij}} \\
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}^{6}}{r_{ij}^{6}}\right)^2 -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \vec{r}_{ij}
\end{aligned}
```
"""
struct LennardJones{S, C, W, WS, F, E} <: PairwiseInteraction
    cutoff::C
    use_neighbors::Bool
    lorentz_mixing::Bool
    weight_special::W
    weight_solute_solvent::WS
    force_units::F
    energy_units::E
end

function LennardJones(;
                        cutoff=NoCutoff(),
                        use_neighbors=false,
                        lorentz_mixing=true,
                        weight_special=1,
                        weight_solute_solvent=1,
                        force_units=u"kJ * mol^-1 * nm^-1",
                        energy_units=u"kJ * mol^-1",
                        skip_shortcut=false)
    return LennardJones{skip_shortcut, typeof(cutoff), typeof(weight_special),
                        typeof(weight_solute_solvent), typeof(force_units), typeof(energy_units)}(
        cutoff, use_neighbors, lorentz_mixing, weight_special, weight_solute_solvent,
        force_units, energy_units)
end

use_neighbors(inter::LennardJones) = inter.use_neighbors

is_solute(at::Atom) = at.solute
is_solute(at) = false

function Base.zero(lj::LennardJones{S, C, W, WS, F, E}) where {S, C, W, WS, F, E}
    return LennardJones{S, C, W, WS, F, E}(
        lj.cutoff,
        false,
        false,
        zero(W),
        zero(WS),
        lj.force_units,
        lj.energy_units,
    )
end

function Base.:+(l1::LennardJones{S, C, W, WS, F, E},
                 l2::LennardJones{S, C, W, WS, F, E}) where {S, C, W, WS, F, E}
    return LennardJones{S, C, W, WS, F, E}(
        l1.cutoff,
        l1.use_neighbors,
        l1.lorentz_mixing,
        l1.weight_special + l2.weight_special,
        l1.weight_solute_solvent + l2.weight_solute_solvent,
        l1.force_units,
        l1.energy_units,
    )
end

@inline function force(inter::LennardJones{S, C},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    boundary,
                                    special::Bool=false) where {S, C}
    if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
              iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
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
    r2 = sum(abs2, dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, inter.force_units)
    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr(::LennardJones, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3
    return (24ϵ * invr2) * (2 * six_term ^ 2 - six_term)
end

@inline function potential_energy(inter::LennardJones{S, C},
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            boundary,
                                            special::Bool=false) where {S, C}
    if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
              iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
        return ustrip(zero(coord_i[1])) * inter.energy_units
    end

    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    if (is_solute(atom_i) && !is_solute(atom_j)) || (is_solute(atom_j) && !is_solute(atom_i))
        ϵ = inter.weight_solute_solvent * sqrt(atom_i.ϵ * atom_j.ϵ)
    else
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
    end

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    pe = potential_with_cutoff(inter, r2, params, cutoff, coord_i, inter.energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::LennardJones, r2, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3
    return 4ϵ * (six_term ^ 2 - six_term)
end

@doc raw"""
    LennardJonesSoftCore(; cutoff, α, λ, p, use_neighbors, lorentz_mixing, weight_special,
                         weight_solute_solvent, force_units, energy_units, skip_shortcut)

The Lennard-Jones 6-12 interaction between two atoms with a soft core.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}^{\text{sc}}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}^{\text{sc}}}\right)^{6}\right]
```
and the force on each atom by
```math
\vec{F}_i = 24\varepsilon_{ij} \left(2\frac{\sigma_{ij}^{12}}{(r_{ij}^{\text{sc}})^{13}} - \frac{\sigma_{ij}^6}{(r_{ij}^{\text{sc}})^{7}}\right) \left(\frac{r_{ij}}{r_{ij}^{\text{sc}}}\right)^5 \frac{\vec{r}_{ij}}{r_{ij}}
```
where
```math
r_{ij}^{\text{sc}} = \left(r_{ij}^6 + \alpha \sigma_{ij}^6 \lambda^p \right)^{1/6}
```
If ``\alpha`` or ``\lambda`` are zero this gives the standard [`LennardJones`](@ref) potential.
"""
struct LennardJonesSoftCore{S, C, A, L, P, R, W, WS, F, E} <: PairwiseInteraction
    cutoff::C
    α::A
    λ::L
    p::P
    σ6_fac::R
    use_neighbors::Bool
    lorentz_mixing::Bool
    weight_special::W
    weight_solute_solvent::WS
    force_units::F
    energy_units::E
end

function LennardJonesSoftCore(;
                        cutoff=NoCutoff(),
                        α=1,
                        λ=0,
                        p=2,
                        use_neighbors=false,
                        lorentz_mixing=true,
                        weight_special=1,
                        weight_solute_solvent=1,
                        force_units=u"kJ * mol^-1 * nm^-1",
                        energy_units=u"kJ * mol^-1",
                        skip_shortcut=false)
    σ6_fac = α * λ^p
    return LennardJonesSoftCore{skip_shortcut, typeof(cutoff), typeof(α), typeof(λ), typeof(p),
                                typeof(σ6_fac), typeof(weight_special), typeof(weight_solute_solvent),
                                typeof(force_units), typeof(energy_units)}(
        cutoff, α, λ, p, σ6_fac, use_neighbors, lorentz_mixing, weight_special,
        weight_solute_solvent, force_units, energy_units)
end

use_neighbors(inter::LennardJonesSoftCore) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCore{S, C, A, L, P, R, W, WS, F, E}) where {S, C, A, L, P, R, W, WS, F, E}
    return LennardJonesSoftCore{S, C, A, L, P, R, W, WS, F, E}(
        lj.cutoff,
        zero(A),
        zero(L),
        zero(P),
        zero(R),
        false,
        false,
        zero(W),
        zero(WS),
        lj.force_units,
        lj.energy_units,
    )
end

function Base.:+(l1::LennardJonesSoftCore{S, C, A, L, P, R, W, WS, F, E},
                 l2::LennardJonesSoftCore{S, C, A, L, P, R, W, WS, F, E}) where {S, C, A, L, P, R, W, WS, F, E}
    return LennardJonesSoftCore{S, C, A, L, P, R, W, WS, F, E}(
        l1.cutoff,
        l1.α + l2.α,
        l1.λ + l2.λ,
        l1.p + l2.p,
        l1.σ6_fac + l2.σ6_fac,
        l1.use_neighbors,
        l1.lorentz_mixing,
        l1.weight_special + l2.weight_special,
        l1.weight_solute_solvent + l2.weight_solute_solvent,
        l1.force_units,
        l1.energy_units,
    )
end

@inline function force(inter::LennardJonesSoftCore{S, C},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    boundary,
                                    special::Bool=false) where {S, C}
    if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
              iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
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
    r2 = sum(abs2, dr)
    σ2 = σ^2
    params = (σ2, ϵ, inter.σ6_fac)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, inter.force_units)
    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr(::LennardJonesSoftCore, r2, invr2, (σ2, ϵ, σ6_fac))
    inv_rsc6 = inv(r2^3 + σ2^3 * σ6_fac) # rsc = (r2^3 + α * σ2^3 * λ^p)^(1/6)
    inv_rsc  = √cbrt(inv_rsc6)
    six_term = σ2^3 * inv_rsc6
    ff  = (24ϵ * inv_rsc) * (2 * six_term^2 - six_term) * (√r2 * inv_rsc)^5
    return ff * √invr2
end

@inline function potential_energy(inter::LennardJonesSoftCore{S, C},
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            boundary,
                                            special::Bool=false) where {S, C}
    if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
              iszero_value(atom_i.σ) ||iszero_value(atom_j.σ))
        return ustrip(zero(coord_i[1])) * inter.energy_units
    end

    σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
    if (is_solute(atom_i) && !is_solute(atom_j)) || (is_solute(atom_j) && !is_solute(atom_i))
        ϵ = inter.weight_solute_solvent * sqrt(atom_i.ϵ * atom_j.ϵ)
    else
        ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
    end

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    σ2 = σ^2
    params = (σ2, ϵ, inter.σ6_fac)

    pe = potential_with_cutoff(inter, r2, params, cutoff, coord_i, inter.energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::LennardJonesSoftCore, r2, invr2, (σ2, ϵ, σ6_fac))
    six_term = σ2^3 * inv(r2^3 + σ2^3 * σ6_fac)
    return 4ϵ * (six_term ^ 2 - six_term)
end
