export 
    DoubleExponential,
    DoubleExponentialSoftCore,
    DExpDispersionCorrection

@public DExpDispersionCorrectionλ

@doc raw"""
    DoubleExponential(; cutoff, use_neighbors, α, β, shortcut, σ_mixing, ϵ_mixing,
                      weight_special)

The double exponential interaction between two atoms.

See [Wu and Brooks 2019](https://doi.org/10.1063/1.5107505).
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
and ``\sigma_{ij}`` and ``\varepsilon_{ij}`` are mixed with `σ_mixing` and `ϵ_mixing`.
"""
struct DoubleExponential{C, T, SC, S, E, W} <: PairwiseInteraction
    cutoff::C
    use_neighbors::Bool
    α::T
    β::T
    shortcut::SC
    σ_mixing::S
    ϵ_mixing::E
    weight_special::W
end

function DoubleExponential(; cutoff=NoCutoff(), use_neighbors=true, α, β,
                           shortcut=LJZeroShortcut(), σ_mixing=LorentzMixing(),
                           ϵ_mixing=GeometricMixing(), weight_special=1)
    return DoubleExponential(cutoff, use_neighbors, α, β, shortcut, σ_mixing, ϵ_mixing,
                             weight_special)
end

use_neighbors(inter::DoubleExponential) = inter.use_neighbors

required_atom_fields(inter::DoubleExponential) = (:σ, :ϵ, :λ,
            mixing_atom_fields(inter.σ_mixing)..., mixing_atom_fields(inter.ϵ_mixing)...)

function Base.zero(i::DoubleExponential{C, T, SC, S, E, W}) where {C, T, SC, S, E, W}
    return DoubleExponential(
        i.cutoff,
        i.use_neighbors,
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
        i1.use_neighbors,
        i1.α + i2.α,
        i1.β + i2.β,
        i1.shortcut,
        i1.σ_mixing,
        i1.ϵ_mixing,
        i1.weight_special + i2.weight_special
    )
end

parameter_prefix(::DoubleExponential) = "inter_DEXP_"
parameter_fields(::Type{<:DoubleExponential}) =
    ((:α, "alpha"), (:β, "beta"), (:weight_special, "weight_14"))


const two_power_sixth = 2 ^ (1 / 6)

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
    rm = σ * T(two_power_sixth)
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
        return zero_pairwise_energy(dr, energy_units)
    end
    r = sqrt(sum(abs2, dr))
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    rm = σ * T(two_power_sixth)
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

@doc raw"""
    DoubleExponentialSoftCore(; cutoff, use_neighbors, α, β, shortcut, σ_mixing,
                              ϵ_mixing, λ_mixing, scheduler, weight_special)

The double exponential interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Horton et al. 2023](https://doi.org/10.1039/d3dd00070b).
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

``\lambda`` is the steric coupling of the pair, which `scheduler`, a
[`Molly.LambdaScheduler`](@ref), gives from the `λ` of the atoms mixed with `λ_mixing` and
their alchemical roles.
If ``\lambda`` is 1 this gives the standard [`DoubleExponential`](@ref) potential, and if
it is 0 the interaction is turned off.
``\alpha`` and ``\beta`` determine the shape of the double exponential potential.
"""
struct DoubleExponentialSoftCore{C, T, SC, S, E, LM, SCH, W} <: PairwiseInteraction
    cutoff::C
    use_neighbors::Bool
    # The scheduler follows `use_neighbors`, as in the other λ interactions, so that their
    #   Bools fill one word; placed after α and β, Enzyme dropped the gradients of
    #   `weight_special` and of every later interaction in the tuple once the interaction
    #   was read back out of a System
    λ_mixing::LM
    scheduler::SCH
    α::T
    β::T
    shortcut::SC
    σ_mixing::S
    ϵ_mixing::E
    weight_special::W
end

function DoubleExponentialSoftCore(; cutoff=NoCutoff(), use_neighbors=true, α, β,
                                   shortcut=LJZeroShortcut(), σ_mixing=LorentzMixing(),
                                   ϵ_mixing=GeometricMixing(), λ_mixing=MinimumMixing(),
                                   scheduler=DefaultLambdaScheduler(), weight_special=1)
    return DoubleExponentialSoftCore(cutoff, use_neighbors, λ_mixing, scheduler, α, β, shortcut,
                                     σ_mixing, ϵ_mixing, weight_special)
end

use_neighbors(inter::DoubleExponentialSoftCore) = inter.use_neighbors

required_atom_fields(inter::DoubleExponentialSoftCore) = (:σ, :ϵ, :λ, :alch_role,
            mixing_atom_fields(inter.σ_mixing)..., mixing_atom_fields(inter.ϵ_mixing)...,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(i::DoubleExponentialSoftCore{C, T, SC, S, E, LM, SCH, W}) where {C, T, SC, S, E, LM, SCH, W}
    return DoubleExponentialSoftCore(
        i.cutoff,
        i.use_neighbors,
        i.λ_mixing,
        i.scheduler,
        zero(T),
        zero(T),
        i.shortcut,
        i.σ_mixing,
        i.ϵ_mixing,
        zero(W)
    )
end

function Base.:+(i1::DoubleExponentialSoftCore, i2::DoubleExponentialSoftCore)
    return DoubleExponentialSoftCore(
        i1.cutoff,
        i1.use_neighbors,
        i1.λ_mixing,
        i1.scheduler,
        i1.α + i2.α,
        i1.β + i2.β,
        i1.shortcut,
        i1.σ_mixing,
        i1.ϵ_mixing,
        i1.weight_special + i2.weight_special
    )
end

parameter_prefix(::DoubleExponentialSoftCore) = "inter_DEXP_"
parameter_fields(::Type{<:DoubleExponentialSoftCore}) =
    ((:α, "alpha"), (:β, "beta"), (:weight_special, "weight_14"))


# The double exponential has one soft core, so the soft core chosen for Lennard-Jones does not
#   apply to it
function to_lambda_function(inter::DoubleExponential; λ_mixing=MinimumMixing(),
                            scheduler=DefaultLambdaScheduler())
    return DoubleExponentialSoftCore(cutoff=inter.cutoff, use_neighbors=inter.use_neighbors,
                                     α=inter.α, β=inter.β, shortcut=inter.shortcut,
                                     σ_mixing=inter.σ_mixing, ϵ_mixing=inter.ϵ_mixing,
                                     λ_mixing=λ_mixing, scheduler=scheduler,
                                     weight_special=inter.weight_special)
end

# Energy prefactor, soft-core coupling and mixed σ/ϵ of a pair, from the scheduler as for the
# Lennard-Jones soft cores
@inline function dexp_lambda_params(inter::DoubleExponentialSoftCore{C, T}, atom_i, atom_j,
                                    special) where {C, T}
    λ, λR, _, σ, ϵ = λ_params_function(inter.scheduler, inter.λ_mixing, inter.σ_mixing,
                                       inter.ϵ_mixing, atom_i, atom_j, special)
    return T(λ), T(λ * λR), σ, ϵ
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

    λ, λ_soft, σ, ϵ = dexp_lambda_params(inter, atom_i, atom_j, special)
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

    rm = σ * T(two_power_sixth)
    # Following  https://doi.org/10.1039/d3dd00070b
    # αs = (1.1 + λ(α − 1.1)) and βs = (1 + λ(β − 1))
    α_s = T(1.1 + λ_soft * (inter.α - 1.1))
    β_s = T(1 + λ_soft * (inter.β - 1))
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

@inline function potential_energy(
    inter::DoubleExponentialSoftCore{C, T},
    dr,
    atom_i,
    atom_j,
    energy_units=u"kJ * mol^-1",
    special=false,
    args...
) where {C, T}

    λ, λ_soft, σ, ϵ = dexp_lambda_params(inter, atom_i, atom_j, special)
    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return zero_pairwise_energy(dr, energy_units)
    end
    r = sqrt(sum(abs2, dr))
    rm = σ * T(two_power_sixth)
    # Following  https://doi.org/10.1039/d3dd00070b
    # αs = (1.1 + λ(α − 1.1)) and βs = (1 + λ(β − 1))
    α_s = T(1.1 + λ_soft * (inter.α - 1.1))
    β_s = T(1 + λ_soft * (inter.β - 1))
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

@doc raw"""
    DExpDispersionCorrection(atoms, cutoff, α, β, σ_mixing=LorentzMixing(),
                             ϵ_mixing=GeometricMixing())

The long-range dispersion correction for the [`DoubleExponential`](@ref) interaction, the
analogue of [`LJDispersionCorrection`](@ref).

Use it with the same `α`, `β`, mixing rules and cutoff as the pairwise interaction.
`cutoff` can be a [`DistanceCutoff`](@ref) (or a distance), a [`ShiftedPotentialCutoff`](@ref)
or a [`PolynomialCutoff`](@ref).
Assuming a uniform density beyond the cutoff, the potential energy for a
[`DistanceCutoff`](@ref) is
```math
E = \frac{2 \pi N^2}{V} \left< \int_{r_c}^\infty r^2 V_{ij}(r) \, dr \right>
  = \frac{2 \pi N^2}{V} \left< \frac{\varepsilon_{ij} r_m^3}{\alpha - \beta} \left[
    \beta G_\alpha\left(\frac{r_c}{r_m}\right) - \alpha G_\beta\left(\frac{r_c}{r_m}\right)
  \right] \right>
```
where
```math
G_k(x) = e^{k(1 - x)} \left( \frac{x^2}{k} + \frac{2x}{k^2} + \frac{2}{k^3} \right)
```
``V_{ij}`` is the pair potential, ``V`` the volume and ``r_c`` the cutoff distance. The
averages run over all pairs, including excluded pairs and self pairs.
The other cutoffs also include the switched-out or shifted part of the potential, as for
[`LJDispersionCorrection`](@ref).
The forces are zero, with an isotropic virial contribution.

The number of atoms is assumed not to change after setup. [`inject_gradients`](@ref)
recomputes the correction from injected σ, ϵ, `α` and `β`, which share their keys with the
pairwise interaction, including its gradients.
Only compatible with 3D systems.
"""
struct DExpDispersionCorrection{F, EC, C, T, S, E, P}
    factor::F
    factor_vir::F
    energy_const::EC
    cutoff::C
    α::T
    β::T
    σ_mix::S
    ϵ_mix::E
    # As for `LJDispersionCorrection`
    group_pairs::P
end

dexp_pe(r, α, β, rm, ϵ) = ϵ * (β * exp(α * (1 - r / rm)) - α * exp(β * (1 - r / rm))) / (α - β)

# The integral of r^2 e^(k(1 - r / r_m)) from r_c to ∞ divided by r_m^3, with x = r_c / r_m
dexp_tail_term(k, x) = exp(k * (1 - x)) * (x^2 / k + 2x / k^2 + 2 / k^3)

# The integral of r^2 V(r) from r_c to ∞ for one pair
function dexp_tail_integral(rc, α, β, rm, ϵ)
    x = rc / rm
    return ϵ * rm^3 * (β * dexp_tail_term(α, x) - α * dexp_tail_term(β, x)) / (α - β)
end

# The integral of r^2 (1 - S) V(r) from the activation distance to the cutoff for one pair, the
#   part that the switching function of PolynomialCutoff leaves out, by the composite
#   Gauss-Legendre quadrature of `switched_out_integrals` with the nodes and weights given
function dexp_switched_out_integral(ra, rc, α, β, rm, ϵ, (xs, ws))
    # One interval is exact to machine precision for usual activation distances, but an
    #   activation distance within the repulsive core, where e^(α(1 - r / r_m)) grows steeply,
    #   needs about 8; 16 keeps a margin at negligible cost
    n_intervals = 16
    h = (rc - ra) / n_intervals
    j = 0.0
    for m in 1:n_intervals, (xg, wg) in zip(xs, ws)
        r = ra + (m - 1 + (xg + 1) / 2) * h
        t = (r - ra) / (rc - ra)
        j += wg * h / 2 * t^3 * (10 - 15t + 6t^2) * r^2 * dexp_pe(r, α, β, rm, ϵ)
    end
    return j
end

#=
For one pair, without units, the parts of the energy and of each diagonal component of the
    virial that are divided by the volume, both divided by 2πN^2, and the potential at the
    cutoff that gives the part of the energy independent of the volume.
Beyond the cutoff the virial component is -(1/3)∫r^3 V'(r), which integrates by parts to
    ∫r^2 V(r) + r_c^3 V(r_c) / 3.
The shifted potential leaves out V(r_c) for the pairs within the cutoff as well, which adds
    r_c^3 V(r_c) / 3 to the energy but not to the virial, since the shift has no force.
When the potential is continuous at the cutoff, as for the shifted and switched potentials, the
    virial is therefore the energy, as for `disp_corr_virial_weights`.
=#
function dexp_disp_corr_pair(c::DistanceCutoff, α, β, rm, ϵ, quad)
    rc = c.dist_cutoff
    tail = dexp_tail_integral(rc, α, β, rm, ϵ)
    return tail, tail + rc^3 * dexp_pe(rc, α, β, rm, ϵ) / 3, 0.0
end

function dexp_disp_corr_pair(c::ShiftedPotentialCutoff, α, β, rm, ϵ, quad)
    rc = c.dist_cutoff
    V_rc = dexp_pe(rc, α, β, rm, ϵ)
    e = dexp_tail_integral(rc, α, β, rm, ϵ) + rc^3 * V_rc / 3
    return e, e, V_rc
end

function dexp_disp_corr_pair(c::PolynomialCutoff, α, β, rm, ϵ, quad)
    ra, rc = c.dist_activation, c.dist_cutoff
    e = dexp_tail_integral(rc, α, β, rm, ϵ) + dexp_switched_out_integral(ra, rc, α, β, rm, ϵ, quad)
    return e, e, 0.0
end

dexp_float_cutoff(c::DistanceCutoff, L) = DistanceCutoff(Float64(ustrip(L, c.dist_cutoff)))

function dexp_float_cutoff(c::ShiftedPotentialCutoff, L)
    return ShiftedPotentialCutoff(Float64(ustrip(L, c.dist_cutoff)))
end

function dexp_float_cutoff(c::PolynomialCutoff, L)
    return PolynomialCutoff(Float64(ustrip(L, c.dist_activation)),
                            Float64(ustrip(L, c.dist_cutoff)))
end

#=
The factors of the correction from the atoms, summing over the pairs of groups of atoms in
    `group_pairs`, with `n` atoms. `pair_params(atom_i, atom_j)` gives the coupling λ, the shape
    parameters α and β, σ and ϵ of a pair.
The sums are taken in Float64 and the factors are given in the types F and EC. This is also how
    injected parameters are taken up, so it has to stay differentiable.
The shifted potential subtracts V(r_c) from each pair within the cutoff, and as in GROMACS and
    `disp_corr_energy_const` each atom is taken to have one fewer partner within the cutoff than
    the uniform density gives.
=#
function dexp_disp_corr_factors(cutoff, group_pairs, atoms, n, pair_params, F, EC)
    at = first(atoms)
    L, EU = unit(first(at.σ)), unit(first(at.ϵ))
    cutoff_f = dexp_float_cutoff(cutoff, L)
    e_sum, w_sum, V_rc_sum = 0.0, 0.0, 0.0
    for (i, j, n_pairs_group) in group_pairs
        λ, α, β, σ, ϵ = pair_params(atoms[i], atoms[j])
        rm = Float64(ustrip(L, σ)) * two_power_sixth
        ϵ_f = Float64(ustrip(EU, ϵ))
        # Pairs that the shortcut of the pairwise interaction turns off
        (iszero(rm) || iszero(ϵ_f) || iszero(λ)) && continue
        e, w, V_rc = dexp_disp_corr_pair(cutoff_f, Float64(α), Float64(β), rm, ϵ_f,
                                         gauss_legendre_20)
        e_sum    += n_pairs_group * λ * e
        w_sum    += n_pairs_group * λ * w
        V_rc_sum += n_pairs_group * λ * V_rc
    end
    n_pairs = n * (n + 1) / 2
    # Every atom decoupled leaves no pairs, where the correction is zero rather than 0/0
    mean(x) = (iszero(n_pairs) ? zero(x) : x / n_pairs)
    factor     = convert(F, 2π * n^2 * mean(e_sum) * EU * L^3)
    factor_vir = convert(F, 2π * n^2 * mean(w_sum) * EU * L^3)
    energy_const = convert(EC, -n / 2 * mean(V_rc_sum) * EU)
    return factor, factor_vir, energy_const
end

# The types of the factors of a double exponential dispersion correction for an atom
dexp_disp_corr_types(at) = typeof(first(at.ϵ) * first(at.σ)^3), typeof(first(at.ϵ))

function DExpDispersionCorrection(atoms::AbstractArray,
                                  cutoff,
                                  α,
                                  β,
                                  σ_mix=LorentzMixing(),
                                  ϵ_mix=GeometricMixing())
    cutoff_used = disp_corr_cutoff(cutoff)
    atoms_cpu = from_device(atoms)
    F, EC = dexp_disp_corr_types(first(atoms_cpu))
    dc = DExpDispersionCorrection(zero(F), zero(F), zero(EC), cutoff_used, α, β, σ_mix, ϵ_mix,
                                  identical_group_pairs(atoms_cpu))
    return update_cached_parameters(dc, atoms_cpu)
end

# Recompute the factors from the atoms and α and β, which may have injected values
function update_cached_parameters(dc::DExpDispersionCorrection, atoms)
    pair_params(atom_i, atom_j) = (1, dc.α, dc.β, σ_mixing(dc.σ_mix, atom_i, atom_j),
                                   ϵ_mixing(dc.ϵ_mix, atom_i, atom_j))
    factor, factor_vir, energy_const = dexp_disp_corr_factors(dc.cutoff,
            from_device(dc.group_pairs), atoms, Float64(length(atoms)), pair_params,
            typeof(dc.factor), typeof(dc.energy_const))
    return DExpDispersionCorrection(factor, factor_vir, energy_const, dc.cutoff, dc.α, dc.β,
                                    dc.σ_mix, dc.ϵ_mix, dc.group_pairs)
end

function Base.zero(dc::DExpDispersionCorrection)
    return DExpDispersionCorrection(zero(dc.factor), zero(dc.factor_vir), zero(dc.energy_const),
                                    dc.cutoff, dc.α, dc.β, dc.σ_mix, dc.ϵ_mix, dc.group_pairs)
end

function Base.:+(dc1::DExpDispersionCorrection, dc2::DExpDispersionCorrection)
    return DExpDispersionCorrection(
        dc1.factor + dc2.factor,
        dc1.factor_vir + dc2.factor_vir,
        dc1.energy_const + dc2.energy_const,
        dc1.cutoff,
        dc1.α,
        dc1.β,
        dc1.σ_mix,
        dc1.ϵ_mix,
        dc1.group_pairs,
    )
end

function Unitful.ustrip(dc::DExpDispersionCorrection)
    return DExpDispersionCorrection(ustrip(dc.factor), ustrip(dc.factor_vir),
                                    ustrip(dc.energy_const), ustrip_cutoff(dc.cutoff), dc.α,
                                    dc.β, dc.σ_mix, dc.ϵ_mix, dc.group_pairs)
end

@doc raw"""
    DExpDispersionCorrectionλ(atoms, cutoff, α, β, λ_mix, σ_mix, ϵ_mix; scheduler)

The long-range dispersion correction for the [`DoubleExponentialSoftCore`](@ref) interaction,
the λ version of [`DExpDispersionCorrection`](@ref).

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each
[`DExpDispersionCorrection`](@ref) to this type.
For a [`DistanceCutoff`](@ref) the potential energy is
```math
E = \frac{2 \pi N^2}{V} \left< \lambda_{ij} \int_{r_c}^\infty r^2 V_{ij}(r; \alpha_s, \beta_s) \, dr \right>
```
where ``V_{ij}(r; \alpha_s, \beta_s)`` is the pair potential with the soft-core shape
parameters of [`DoubleExponentialSoftCore`](@ref).
Unlike for Lennard-Jones, the soft core changes the potential at every distance, so the
coupling enters through ``\alpha_s`` and ``\beta_s`` as well as ``\lambda_{ij}``.
The averages, ``\lambda_{ij}`` and the effective number of atoms ``N`` are as for
[`Molly.LJDispersionCorrectionλ`](@ref), and the other cutoffs as for
[`DExpDispersionCorrection`](@ref).
The forces are zero, with an isotropic virial contribution.

The factors are computed from the `λ` of `atoms` at setup; [`Molly.set_lambda!`](@ref)
recomputes them.
The number of atoms is otherwise assumed not to change after setup, and
[`inject_gradients`](@ref) takes up injected σ, ϵ, `α` and `β` as for
[`DExpDispersionCorrection`](@ref).
Only compatible with 3D systems.
"""
struct DExpDispersionCorrectionλ{F, EC, C, T, L, S, E, SCH, P}
    factor::F
    factor_vir::F
    energy_const::EC
    cutoff::C
    α::T
    β::T
    λ_mix::L
    σ_mix::S
    ϵ_mix::E
    # Kept so that `set_lambda!` can rebuild the factors, which are precomputed from the coupling
    scheduler::SCH
    # As for `LJDispersionCorrectionλ`
    class_pairs::P
    n_eff::Float64
end

# α and β share their keys with the pairwise interaction, so they are injected with it
parameter_prefix(::Union{DExpDispersionCorrection, DExpDispersionCorrectionλ}) = "inter_DEXP_"
parameter_fields(::Type{<:Union{DExpDispersionCorrection, DExpDispersionCorrectionλ}}) =
    ((:α, "alpha"), (:β, "beta"))

function DExpDispersionCorrectionλ(atoms, cutoff, α, β, λ_mix, σ_mix, ϵ_mix; scheduler)
    cutoff_used = disp_corr_cutoff(cutoff)
    atoms_cpu = from_device(atoms)
    F, EC = dexp_disp_corr_types(first(atoms_cpu))
    class_pairs, n_eff = lambda_class_pairs(atoms_cpu, scheduler)
    dc = DExpDispersionCorrectionλ(zero(F), zero(F), zero(EC), cutoff_used, α, β, λ_mix, σ_mix,
                                   ϵ_mix, scheduler, class_pairs, n_eff)
    return update_cached_parameters(dc, atoms_cpu)
end

function update_cached_parameters(dc::DExpDispersionCorrectionλ, atoms)
    function pair_params(atom_i, atom_j)
        λ, λR, _, σ, ϵ = λ_params_function(dc.scheduler, dc.λ_mix, dc.σ_mix, dc.ϵ_mix,
                                           lambda_disp_corr_atom(dc.scheduler, atom_i),
                                           lambda_disp_corr_atom(dc.scheduler, atom_j), false)
        # The soft-core shape parameters, as in `DoubleExponentialSoftCore`
        λ_soft = λ * λR
        return λ, 1.1 + λ_soft * (dc.α - 1.1), 1 + λ_soft * (dc.β - 1), σ, ϵ
    end
    factor, factor_vir, energy_const = dexp_disp_corr_factors(dc.cutoff,
            from_device(dc.class_pairs), atoms, dc.n_eff, pair_params, typeof(dc.factor),
            typeof(dc.energy_const))
    return DExpDispersionCorrectionλ(factor, factor_vir, energy_const, dc.cutoff, dc.α, dc.β,
                                     dc.λ_mix, dc.σ_mix, dc.ϵ_mix, dc.scheduler,
                                     dc.class_pairs, dc.n_eff)
end

function dexp_disp_corr_virial!(buffers, sys, inter)
    w = inter.factor_vir / volume(sys)
    for d in axes(buffers.virial, 1)
        buffers.virial[d, d] += w
    end
    return buffers
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                    inter::DExpDispersionCorrection; kwargs...)
    return inter.factor / volume(sys) + inter.energy_const
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces!(
        fs, sys, inter::DExpDispersionCorrection;
        buffers=nothing,
        needs_vir=false,
        kwargs...)
    if needs_vir
        dexp_disp_corr_virial!(buffers, sys, inter)
    end
    return fs
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                    inter::DExpDispersionCorrectionλ; kwargs...)
    return inter.factor / volume(sys) + inter.energy_const
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces!(
        fs, sys, inter::DExpDispersionCorrectionλ;
        buffers=nothing,
        needs_vir=false,
        kwargs...)
    if needs_vir
        dexp_disp_corr_virial!(buffers, sys, inter)
    end
    return fs
end
