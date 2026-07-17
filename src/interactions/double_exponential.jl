export 
    DoubleExponential,
    DoubleExponentialSoftCore

@doc raw"""
    DoubleExponential(; cutoff, use_neighbors, α, β, shortcut, σ_mixing, ϵ_mixing,
                      weight_special)

The double exponential interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \varepsilon_{ij} \left[
    \frac{\beta e^\alpha}{\alpha - \beta} e^{-\alpha r_{ij} / r_m}
    - \frac{\alpha e^\beta}{\alpha - \beta} e^{-\beta r_{ij} / r_m}
\right]
```
and the force on each atom by
```math
\vec{F}_i = \frac{\varepsilon_{ij}}{r_m} \left[
    \frac{\alpha\beta e^\alpha}{\alpha - \beta} e^{-\alpha r_{ij} / r_m}
    - \frac{\alpha\beta e^\beta}{\alpha - \beta} e^{-\beta r_{ij} / r_m}
\right] \frac{\vec{r}_{ij}}{r_{ij}}
```
where
```math
r_m = 2^{1/6}\sigma_{ij}
```

The default ``\alpha`` and ``\beta`` values are those predicted by Garnet.
"""
@kwdef struct DoubleExponential{C, T, SC, S, E, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    α::T = 12.159626 # These are the parameters predicted by Garnet
    β::T = 4.326311
    shortcut::SC = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    weight_special::W = 1
end

use_neighbors(inter::DoubleExponential) = inter.use_neighbors

function Base.zero(i::DoubleExponential{C, T, SC, S, E, W}) where {C, T, SC, S, E, W}
    return DoubleExponential(
        i.cutoff,
        false,
        zero(T),
        zero(T),
        i.shortcut,
        i.σ_mixing,
        i.ϵ_mixing,
        zero(W)
    )
end

function Base.:+(i1::DoubleExponential, i2::DoubleExponential)
    return DoubleExponential(
        i1.cutoff,
        false,
        i1.α + i2.α,
        i1.β + i2.β,
        i1.shortcut,
        i1.σ_mixing,
        i1.ϵ_mixing,
        i1.weight_special + i2.weight_special
    )
end

function inject_interaction(inter::DoubleExponential, params::AbstractVector,
                            idx_α::Int, idx_β::Int, idx_weight::Int)
    α = idx_α > 0 ? typeof(inter.α)(params[idx_α]) : inter.α
    β = idx_β > 0 ? typeof(inter.β)(params[idx_β]) : inter.β
    weight = idx_weight > 0 ? typeof(inter.weight_special)(params[idx_weight]) :
                              inter.weight_special

    return DoubleExponential(
        inter.cutoff, inter.use_neighbors, α, β, inter.shortcut, inter.σ_mixing,
        inter.ϵ_mixing, weight,
    )
end

function extract_parameter_indices!(buf::ParamBuffer, inter::DoubleExponential)
    key_prefix = "inter_DEXP_"
    return (
        _push_param!(buf, key_prefix * "alpha", inter.α),
        _push_param!(buf, key_prefix * "beta", inter.β),
        _push_param!(buf, key_prefix * "weight_14", inter.weight_special),
    )
end

@inline function potential_energy(
    inter::DoubleExponential{C, T},
    dr, 
    atom_i,
    atom_j,
    energy_units=u"kJ * mol^-1", 
    special=false,
    args...
) where {C, T}

    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return ustrip(zero(dr[1])) * energy_units
    end
    r = sqrt(sum(abs2, dr))
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    rm = σ * T(2^(1/6))
    α, β = inter.α, inter.β
    params = (α, β, rm, ϵ)
    pe = pe_cutoff(inter.cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

@inline function pairwise_pe(::DoubleExponential, r, (α, β, rm, ϵ))
    x = 1 - r / rm
    return ϵ * (β * exp(α * x) - α * exp(β * x)) / (α - β)
end

@inline function force(
    inter::DoubleExponential{C, T},
    dr,
    atom_i,
    atom_j,
    force_units=u"kJ * mol^-1 * nm^-1",
    special=false,
    args...
) where {C, T}

    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return zero_pairwise_force(dr, force_units)
    end
    r = sqrt(sum(abs2, dr))
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    rm = σ * T(2^(1/6))
    α, β = inter.α, inter.β
    params = (α, β, rm, ϵ)
    f = force_cutoff(inter.cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

@inline function pairwise_force(::DoubleExponential, r, (α, β, rm, ϵ))
    x = 1 - r / rm
    return ϵ * α * β * (exp(α * x) - exp(β * x)) / (rm * (α - β))
end

@doc raw"""
    DoubleExponentialSoftCore(; cutoff, use_neighbors, α, β, shortcut, σ_mixing,
                              ϵ_mixing, λ_mixing, scheduler, weight_special)

The double exponential interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

The potential energy is defined as
```math
V(r_{ij}) = \lambda\varepsilon_{ij} \left[
    \frac{\beta_s e^{\alpha_s}}{\alpha_s - \beta_s} e^{-\alpha_s r_{ij} / r_m}
    - \frac{\alpha_s e^{\beta_s}}{\alpha_s - \beta_s} e^{-\beta_s r_{ij} / r_m}
\right]
```
and the force on each atom by
```math
\vec{F}_i = \frac{\lambda\varepsilon_{ij}}{r_m} \left[
    \frac{\alpha_s\beta_s e^{\alpha_s}}{\alpha_s - \beta_s} e^{-\alpha_s r_{ij} / r_m}
    - \frac{\alpha_s\beta_s e^{\beta_s}}{\alpha_s - \beta_s} e^{-\beta_s r_{ij} / r_m}
\right] \frac{\vec{r}_{ij}}{r_{ij}}
```
where
```math
r_m = 2^{1/6}\sigma_{ij}
```
```math
\alpha_s = 1.1 + \lambda(\alpha - 1.1)
```
```math
\beta_s = 1 + \lambda(\beta - 1)
```

If ``\lambda`` is 1.0, this gives the standard [`DoubleExponential`](@ref) potential and
means the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` and ``\beta`` determine the shape of the double exponential potential.
"""
@kwdef struct DoubleExponentialSoftCore{C, T, SC, S, E, LM, SCH, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    α::T = 12.159626 # These are the parameters predicted by Garnet
    β::T = 4.326311
    shortcut::SC = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    weight_special::W = 1
end

use_neighbors(inter::DoubleExponentialSoftCore) = inter.use_neighbors

function Base.zero(i::DoubleExponentialSoftCore{C, T, SC, S, E, LM, SCH, W}) where {C, T, SC, S, E, LM, SCH, W}
    return DoubleExponentialSoftCore(
        i.cutoff,
        false,
        zero(T),
        zero(T),
        i.shortcut,
        i.σ_mixing,
        i.ϵ_mixing,
        i.λ_mixing,
        i.scheduler,
        zero(W)
    )
end

function Base.:+(i1::DoubleExponentialSoftCore, i2::DoubleExponentialSoftCore)
    return DoubleExponentialSoftCore(
        i1.cutoff,
        false,
        i1.α + i2.α,
        i1.β + i2.β,
        i1.shortcut,
        i1.σ_mixing,
        i1.ϵ_mixing,
        i1.λ_mixing,
        i1.scheduler,
        i1.weight_special + i2.weight_special
    )
end

function inject_interaction(inter::DoubleExponentialSoftCore, params::AbstractVector,
                            idx_α::Int, idx_β::Int, idx_weight::Int)
    α = idx_α > 0 ? typeof(inter.α)(params[idx_α]) : inter.α
    β = idx_β > 0 ? typeof(inter.β)(params[idx_β]) : inter.β
    weight = idx_weight > 0 ? typeof(inter.weight_special)(params[idx_weight]) :
                              inter.weight_special

    return DoubleExponentialSoftCore(
        inter.cutoff, inter.use_neighbors, α, β, inter.shortcut, inter.σ_mixing,
        inter.ϵ_mixing, inter.λ_mixing, inter.scheduler, weight,
    )
end

function extract_parameter_indices!(buf::ParamBuffer, inter::DoubleExponentialSoftCore)
    key_prefix = "inter_DEXPSC_"
    return (
        _push_param!(buf, key_prefix * "alpha", inter.α),
        _push_param!(buf, key_prefix * "beta", inter.β),
        _push_param!(buf, key_prefix * "weight_14", inter.weight_special),
    )
end

@inline function potential_energy(
    inter::DoubleExponentialSoftCore{C, T},
    dr,
    atom_i,
    atom_j,
    energy_units=u"kJ * mol^-1",
    special=false,
    args...
) where {C, T}

    # Mix Lambda
    λ_glob = T(λ_mixing(inter.λ_mixing, atom_i, atom_j))
    λ = T(sterics_lambda(inter.scheduler, atom_i, atom_j, λ_glob))
    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return ustrip(zero(dr[1])) * energy_units
    end
    r = sqrt(sum(abs2, dr))
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    rm = σ * T(2^(1/6))
    # Following  https://doi.org/10.1039/d3dd00070b
    # αs = (1.1 + λ(α − 1.1)) and βs = (1 + λ(β − 1))
    α_s = T(1.1 + λ * (inter.α - 1.1))
    β_s = T(1 + λ * (inter.β - 1))
    params = (α_s, β_s, rm, ϵ)
    pe = pe_cutoff(inter.cutoff, inter, r, params)
    if special
        return λ * pe * inter.weight_special
    else
        return λ * pe
    end
end

@inline function pairwise_pe(::DoubleExponentialSoftCore, r, (α, β, rm, ϵ))
    x = 1 - r / rm
    return ϵ * (β * exp(α * x) - α * exp(β * x)) / (α - β)
end

@inline function force(
    inter::DoubleExponentialSoftCore{C, T},
    dr,
    atom_i,
    atom_j,
    force_units=u"kJ * mol^-1 * nm^-1",
    special=false,
    args...
) where {C, T}

    λ_glob = T(λ_mixing(inter.λ_mixing, atom_i, atom_j))
    λ = T(sterics_lambda(inter.scheduler, atom_i, atom_j, λ_glob))
    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return zero_pairwise_force(dr, force_units)
    end
    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    rm = σ * T(2^(1/6))
    # Following  https://doi.org/10.1039/d3dd00070b
    # αs = (1.1 + λ(α − 1.1)) and βs = (1 + λ(β − 1))
    α_s = T(1.1 + λ * (inter.α - 1.1))
    β_s = T(1 + λ * (inter.β - 1))
    params = (α_s, β_s, rm, ϵ)
    f = force_cutoff(inter.cutoff, inter, r, params)
    fdr = (λ * f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

@inline function pairwise_force(::DoubleExponentialSoftCore, r, (α, β, rm, ϵ))
    x = 1 - r / rm
    return ϵ * α * β * (exp(α * x) - exp(β * x)) / (rm * (α - β))
end
