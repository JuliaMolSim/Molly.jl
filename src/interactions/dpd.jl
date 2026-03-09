export DPDInteraction

@doc raw"""
    DPDInteraction(; a, γ, σ, r_c, dt, use_neighbors)

The dissipative particle dynamics (DPD) interaction between two particles.

Combines conservative, dissipative, and random pairwise forces as described in
[Groot and Warren 1997](https://doi.org/10.1063/1.474784).

The total pairwise force is
```math
\vec{F}_{ij} = \vec{F}_{ij}^C + \vec{F}_{ij}^D + \vec{F}_{ij}^R
```
where the conservative force is
```math
\vec{F}_{ij}^C = a \left(1 - \frac{r_{ij}}{r_c}\right) \hat{r}_{ij}
```
the dissipative force is
```math
\vec{F}_{ij}^D = -\gamma \left(1 - \frac{r_{ij}}{r_c}\right)^2
    (\hat{r}_{ij} \cdot \vec{v}_{ij}) \hat{r}_{ij}
```
and the random force is
```math
\vec{F}_{ij}^R = \sigma \left(1 - \frac{r_{ij}}{r_c}\right)
    \xi_{ij} \Delta t^{-1/2} \hat{r}_{ij}
```

All forces are zero for ``r_{ij} \geq r_c``.
The weight functions satisfy ``w^D(r) = [w^R(r)]^2`` with ``w^R(r) = 1 - r/r_c``.

The fluctuation-dissipation relation ``\sigma^2 = 2 \gamma k_B T`` must be
satisfied by the user's choice of parameters to correctly thermostat at
temperature ``T``.

The conservative potential energy is
```math
V(r_{ij}) = \frac{a}{2} r_c \left(1 - \frac{r_{ij}}{r_c}\right)^2
```

Deterministic pairwise random numbers are generated from a hash of the particle
indices and the step number, ensuring momentum conservation and reproducibility.

When using a neighbor list, set `dist_cutoff` to at least `1.5 * r_c` to provide a
skin distance that accounts for particle movement between neighbor list rebuilds.

# Arguments
- `a`: the conservative force strength parameter.
- `γ`: the dissipative force strength parameter.
- `σ`: the random force strength parameter (must satisfy σ² = 2γkBT).
- `r_c`: the cutoff distance beyond which all forces are zero.
- `dt`: the simulation timestep (needed for random force scaling as Δt⁻¹ᐟ²).
- `use_neighbors::Bool=true`: whether to use the neighbor list.
"""
@kwdef struct DPDInteraction{A, G, S, R, D} <: PairwiseInteraction
    a::A = 25.0
    γ::G = 4.5
    σ::S = 3.0
    r_c::R = 1.0
    dt::D = 0.01
    use_neighbors::Bool = true
end

use_neighbors(inter::DPDInteraction) = inter.use_neighbors

function Base.zero(d::DPDInteraction)
    return DPDInteraction(d.a, d.γ, d.σ, d.r_c, d.dt, d.use_neighbors)
end

function Base.:+(d1::DPDInteraction, ::DPDInteraction)
    return DPDInteraction(d1.a, d1.γ, d1.σ, d1.r_c, d1.dt, d1.use_neighbors)
end

# Deterministic per-pair Gaussian noise via hash-based Box-Muller transform.
# Symmetric in (i, j) to ensure momentum conservation.
@inline function dpd_gaussian(idx_i::Integer, idx_j::Integer, step_n::Integer)
    a, b = minmax(idx_i, idx_j)
    h = hash((a, b, step_n))
    u1 = (h & 0xFFFFFFFF) / 4294967296.0
    u2 = ((h >> 32) & 0xFFFFFFFF) / 4294967296.0
    return sqrt(-2 * log(u1 + 1e-30)) * cos(2π * u2)
end

@inline function force(inter::DPDInteraction,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       coord_i=nothing,
                       coord_j=nothing,
                       boundary=nothing,
                       velocity_i=nothing,
                       velocity_j=nothing,
                       step_n=0)
    r = norm(dr)
    r_c = inter.r_c

    if r >= r_c || iszero(r)
        return ustrip.(zero(dr)) * force_units
    end

    w_R = 1 - r / r_c
    w_D = w_R * w_R
    inv_r = inv(r)

    # Conservative: repulsive soft potential
    f_C = inter.a * w_R * inv_r

    # Dissipative: velocity-dependent drag along line of centers
    v_rel = velocity_i - velocity_j
    rdotv = dot(dr, v_rel) * inv_r * inv_r
    f_D = inter.γ * w_D * rdotv

    # Random: pairwise-correlated stochastic force
    ξ = dpd_gaussian(atom_i.index, atom_j.index, step_n)
    f_R = inter.σ * w_R * ξ * inv(sqrt(inter.dt)) * inv_r

    return (f_C + f_D + f_R) * dr
end

@inline function potential_energy(inter::DPDInteraction,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  args...)
    r = norm(dr)
    r_c = inter.r_c

    if r >= r_c || iszero(r)
        return ustrip(zero(dr[1])) * energy_units
    end

    # Only the conservative part has a well-defined potential energy
    w = 1 - r / r_c
    return (inter.a / 2) * r_c * w * w
end
