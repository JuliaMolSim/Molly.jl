export
    Coulomb,
    SetupCoulomb,
    CoulombSoftCoreBeutler,
    CoulombSoftCoreGapsys,
    CoulombReactionField,
    SetupCoulombReactionField,
    CoulombReactionFieldExclusion,
    CoulombSoftCoreBeutlerReactionField,
    CoulombSoftCoreGapsysReactionField,
    CoulombEwald,
    CoulombSoftCoreBeutlerEwald,
    CoulombSoftCoreGapsysEwald,
    Yukawa

@public coulomb_const,
    CoulombScaled,
    CoulombReactionFieldScaled,
    CoulombReactionFieldExclusionData,
    reaction_field_exclusions,
    CoulombEwaldScaled

"The Coulomb constant, 1 / 4πϵ0, where ϵ0 is the vacuum permittivity."
const coulomb_const = 138.93545764u"kJ * mol^-1 * nm" # 1 / 4πϵ0

@doc raw"""
    Coulomb(; cutoff, use_neighbors, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}}
```
where `coulomb_const` is ``1 / 4 \pi \varepsilon_0``.
The interaction between special pairs is scaled by `weight_special`.
"""
@kwdef struct Coulomb{C, W, T} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = true
    weight_special::W = 1
    coulomb_const::T = coulomb_const
end

use_neighbors(inter::Coulomb) = inter.use_neighbors

required_atom_fields(::Coulomb) = (:charge,)

function Base.zero(coul::Coulomb{C, W, T}) where {C, W, T}
    return Coulomb(coul.cutoff, coul.use_neighbors, zero(W), zero(T))
end

function Base.:+(c1::Coulomb, c2::Coulomb)
    return Coulomb(
        c1.cutoff,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

parameter_prefix(::Coulomb) = "inter_CO_"
parameter_fields(::Type{<:Coulomb}) =
    ((:weight_special, "weight_14"), (:coulomb_const, "coulomb_const"))



@inline function force(inter::Coulomb{C},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where C
    r = sqrt(sum(abs2, dr))
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = charge(atom_i), charge(atom_j)
    params = (ke, qi, qj)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::Coulomb, r, (ke, qi, qj))
    return (ke * qi * qj) / r^2
end

@inline function potential_energy(inter::Coulomb{C},
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where C
    r = sqrt(sum(abs2, dr))
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = charge(atom_i), charge(atom_j)
    params = (ke, qi, qj)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::Coulomb, r, (ke, qi, qj))
    return (ke * qi * qj) * inv(r)
end

"""
    SetupCoulomb(; cutoff, coulomb_const=coulomb_const)

Set up the [`Coulomb`](@ref) electrostatic interaction with the cutoff `cutoff`, for example
[`DistanceCutoff`](@ref), and the Coulomb constant `coulomb_const`.

Passed to the [`System`](@ref) constructor from files, where it creates a
[`Coulomb`](@ref) pairwise interaction.
Passing a cutoff as the non-bonded method is the same as passing `SetupCoulomb` with that
cutoff and the default Coulomb constant.
"""
@kwdef struct SetupCoulomb{CU, C}
    cutoff::CU
    coulomb_const::C = coulomb_const
end

function setup_coulomb_pairwise(sc::SetupCoulomb, dist_cutoff, weight_special, use_neighbors,
                                units, T)
    return Coulomb(
        cutoff=sc.cutoff,
        use_neighbors=use_neighbors,
        weight_special=weight_special,
        coulomb_const=convert_setup_quantity(sc.coulomb_const, units, T),
    )
end

@doc raw"""
    CoulombScaled(; cutoff, use_neighbors, λ_mixing, scheduler, weight_special,
                  coulomb_const)

The Coulomb electrostatic interaction between two atoms scaled by λ without a soft core,
the λ version of [`Coulomb`](@ref).

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each [`Coulomb`](@ref)
to this type with `Csoftcore=:scaled`.
The potential energy is defined as
```math
V(r_{ij}) = \lambda \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}}
```
where ``\lambda`` is the electrostatic coupling of the pair, which `scheduler`, a
[`Molly.LambdaScheduler`](@ref), gives from the `λ` of the atoms mixed with `λ_mixing`
and their alchemical roles.
In single topology the charges are interpolated between the end states.
"""
@kwdef struct CoulombScaled{C, LM, SCH, W, T} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = true
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    weight_special::W = 1
    coulomb_const::T = coulomb_const
end

use_neighbors(inter::CoulombScaled) = inter.use_neighbors

required_atom_fields(::CoulombScaled) = (:charge, :λ, :alch_role)

function Base.zero(coul::CoulombScaled{C, LM, SCH, W, T}) where {C, LM, SCH, W, T}
    return CoulombScaled(
        coul.cutoff,
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombScaled, c2::CoulombScaled)
    return CoulombScaled(
        c1.cutoff,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function to_lambda_function(inter::Coulomb, ::ScaledSoftCore; λ_mixing=MinimumMixing(), 
    scheduler=DefaultLambdaScheduler(), float_type=Float32, args...)
    return CoulombScaled(cutoff=inter.cutoff,use_neighbors=inter.use_neighbors, 
                            λ_mixing=λ_mixing, scheduler=scheduler,
                            weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end


parameter_prefix(::CoulombScaled) = "inter_CO_"
parameter_fields(::Type{<:CoulombScaled}) =
    ((:weight_special, "weight_14"), (:coulomb_const, "coulomb_const"))


@inline function force(inter::CoulombScaled{C},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where C
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if iszero_value(qij)
        return zero_pairwise_force(dr, force_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    cutoff = inter.cutoff
    params = (ke, qij)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = λ * (f / r) * dr
    return special ? fdr * inter.weight_special : fdr
end

function pairwise_force(::CoulombScaled, r, (ke, qij))
    return (ke * qij) / r^2
end

@inline function potential_energy(inter::CoulombScaled{C},
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where C
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end

    if iszero_value(qij)
        return zero_pairwise_energy(dr, energy_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_energy(dr, energy_units)
    end

    cutoff = inter.cutoff
    params = (ke, qij)

    pe = λ * pe_cutoff(cutoff, inter, r, params)
    return special ? pe * inter.weight_special : pe
end

function pairwise_pe(::CoulombScaled, r, (ke, qij))
    return (ke * qij) * inv(r)
end

# At exact overlap, use the finite soft-core limit whenever a nonzero
# regularization scale exists. If softening is effectively disabled, fall back
# to zero to avoid singular endpoint evaluations in the soft-core wrappers.
@inline function overlap_pe_coulomb_softcore_beutler(dr, energy_units, ke, qij, λ, σ6_fac)
    if iszero_value(σ6_fac)
        return zero_pairwise_energy(dr, energy_units)
    end
    return λ * ke * (qij / sqrt(cbrt(σ6_fac)))
end

@inline function overlap_pe_coulomb_softcore_gapsys(dr, energy_units, ke, qij, λ, R)
    if iszero_value(R)
        return zero_pairwise_energy(dr, energy_units)
    end
    return λ * ke * ((3 * qij) / R)
end

@inline function overlap_pe_coulomb_softcore_beutler_rf(dr,
                                                        energy_units,
                                                        ke,
                                                        qij,
                                                        λ,
                                                        σ6_fac,
                                                        krf,
                                                        rc,
                                                        special)
    if iszero_value(σ6_fac)
        return zero_pairwise_energy(dr, energy_units)
    end

    R_eff = sqrt(cbrt(σ6_fac))
    if special
        return λ * ke * (qij / R_eff)
    end

    crf_λ = inv(sqrt(cbrt(σ6_fac + rc^6))) + krf * rc^2
    return λ * ke * qij * (inv(R_eff) - crf_λ)
end

@inline function overlap_pe_coulomb_softcore_gapsys_rf(dr,
                                                       energy_units,
                                                       ke,
                                                       qij,
                                                       λ,
                                                       R,
                                                       crf)
    if iszero_value(R)
        return zero_pairwise_energy(dr, energy_units)
    end
    return λ * ke * qij * (3 * inv(R) - crf)
end

@doc raw"""
    CoulombSoftCoreBeutler(; cutoff, α, use_neighbors, σ_mixing, ϵ_mixing, λ_mixing,
                           scheduler, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1).
[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each [`Coulomb`](@ref)
to this type with `Csoftcore=:beutler`.
The potential energy is defined as
```math
V(r_{ij}) = \lambda \frac{1}{4\pi\varepsilon_0} \frac{q_iq_j}{r_Q^{1/6}}
```
and the force on each atom by
```math
\vec{F}_i = \lambda \frac{1}{4\pi\varepsilon_0} \frac{q_iq_jr_{ij}^5}{r_Q^{7/6}}\frac{\vec{r}_{ij}}{r_{ij}}
```
where
```math
r_Q = \alpha(1-\lambda)\sigma_{ij}^6 + r_{ij}^6
```
and ``\sigma_{ij}`` is mixed with `σ_mixing`.
``\lambda`` is the electrostatic coupling of the pair, which `scheduler`, a
[`Molly.LambdaScheduler`](@ref), gives from the `λ` of the atoms mixed with `λ_mixing`
and their alchemical roles.
If ``\lambda`` is 1 this gives the standard [`Coulomb`](@ref) potential, and if it is 0
the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
struct CoulombSoftCoreBeutler{C, A, S, E, LM, SCH, W, T} <: PairwiseInteraction
    cutoff::C
    α::A
    use_neighbors::Bool 
    σ_mixing::S 
    ϵ_mixing::E 
    λ_mixing::LM 
    scheduler::SCH 
    weight_special::W 
    coulomb_const::T
end

function CoulombSoftCoreBeutler(; cutoff=NoCutoff(), α=0.3, use_neighbors=true,
                                    σ_mixing=LorentzMixing(), ϵ_mixing=GeometricMixing(), 
                                    λ_mixing=MinimumMixing(),
                                    scheduler=DefaultLambdaScheduler(), weight_special=1,
                                    coulomb_const=coulomb_const)
    T = typeof(ustrip(α))
    return CoulombSoftCoreBeutler(
        cutoff,
        α,
        use_neighbors,
        σ_mixing,
        ϵ_mixing,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
    )
end

use_neighbors(inter::CoulombSoftCoreBeutler) = inter.use_neighbors

required_atom_fields(inter::CoulombSoftCoreBeutler) = (:charge, :σ, :ϵ, :λ, :alch_role,
            mixing_atom_fields(inter.σ_mixing)..., mixing_atom_fields(inter.ϵ_mixing)...,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(coul::CoulombSoftCoreBeutler{C, A, S, E, LM, SCH, W, T}) where {C, A, S, E, LM, SCH, W, T}
    return CoulombSoftCoreBeutler(
        coul.cutoff,
        zero(A),
        coul.use_neighbors,
        coul.σ_mixing,
        coul.ϵ_mixing,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombSoftCoreBeutler, c2::CoulombSoftCoreBeutler)
    return CoulombSoftCoreBeutler(
        c1.cutoff,
        c1.α + c2.α,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.ϵ_mixing,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function to_lambda_function(inter::Coulomb, ::BeutlerSoftCore; α=0.3, λ_mixing=MinimumMixing(), 
    scheduler=DefaultLambdaScheduler(), float_type=Float32, σ_mixing=LorentzMixing(), ϵ_mixing=GeometricMixing(), args...)
    return CoulombSoftCoreBeutler(cutoff=inter.cutoff, α=float_type(α), use_neighbors=inter.use_neighbors, 
                                    σ_mixing=σ_mixing, ϵ_mixing=ϵ_mixing,
                                    λ_mixing=λ_mixing, scheduler=scheduler,
                                    weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end

parameter_prefix(::CoulombSoftCoreBeutler) = "inter_CO_"
parameter_fields(::Type{<:CoulombSoftCoreBeutler}) =
    ((:α, "α"), (:weight_special, "weight_14"), (:coulomb_const, "coulomb_const"))


@inline function force(inter::CoulombSoftCoreBeutler,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    # 1. Type stability
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)
    
    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if iszero_value(qij)
        return zero_pairwise_force(dr, force_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end
    cutoff = inter.cutoff

    # 2. Fast Path: Standard Coulomb (λ >= 1.0)
    # Use tuple padding (Nothing) to match length 5 of the alchemical path
    if λ >= 1 && λR >= 1
        params = (ke, qij, nothing, nothing)
        f = force_cutoff(cutoff, inter, r, params)
        fdr = radial_force_vector(f, r, dr, force_units)
        return special ? fdr * inter.weight_special : fdr
    end

    # 3. Alchemical Path
    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j, λ_params, pair_role)^6
    σ6_fac = inter.α * (1 - (λ*λR)) * σ6

    params = (ke, qij, σ6_fac, λ)
    f = force_cutoff(cutoff, inter, r, params)
    fdr = radial_force_vector(f, r, dr, force_units)
    return special ? fdr * inter.weight_special : fdr
end

# Dispatch 1: Standard Coulomb Logic (Matches Tuple length 5 with Nothings)
@inline function pairwise_force(::CoulombSoftCoreBeutler, r, (ke, qij, _, _)::Tuple{Any, Any, Any, Nothing})
    return (ke * qij) / r^2
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 5 with Real/Quantities)
@inline function pairwise_force(::CoulombSoftCoreBeutler, r, (ke, qij, σ6_fac, λ)::Tuple{Any, Any, Any, Any})
    r3 = r^3
    term = σ6_fac + (r3 * r3)
    R = term * sqrt(cbrt(term))
    return λ * ke * ((qij) / R) * (r3 * r * r)
end

@inline function potential_energy(inter::CoulombSoftCoreBeutler,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    # 1. Type stability
    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end

    if iszero_value(qij)
        return zero_pairwise_energy(dr, energy_units)
    end

    r = sqrt(sum(abs2, dr))
    cutoff = inter.cutoff

    # 2. Fast Path: Standard Coulomb (λ >= 1.0)
    if λ >= 1 && λR >= 1
        if iszero_value(r)
            return zero_pairwise_energy(dr, energy_units)
        end
        params = (ke, qij, nothing, nothing)
        pe = pe_cutoff(cutoff, inter, r, params)
        return special ? pe * inter.weight_special : pe
    end

    # 3. Alchemical Path
    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j, λ_params, pair_role)^6
    σ6_fac = inter.α * (1 - (λ*λR)) * σ6

    if iszero_value(r)
        pe = overlap_pe_coulomb_softcore_beutler(dr, energy_units, ke, qij, λ, σ6_fac)
        return special ? pe * inter.weight_special : pe
    end

    params = (ke, qij, σ6_fac, λ)
    pe = pe_cutoff(cutoff, inter, r, params)
    return special ? pe * inter.weight_special : pe
end

# Dispatch 1: Standard Coulomb PE
@inline function pairwise_pe(::CoulombSoftCoreBeutler, r, (ke, qij, _, _)::Tuple{Any, Any, Nothing, Nothing})
    return (ke * qij) * inv(r)
end

# Dispatch 2: Soft Core PE
@inline function pairwise_pe(::CoulombSoftCoreBeutler, r, (ke, qij, σ6_fac, λ)::Tuple{Any, Any, Any, Any})
    R = sqrt(cbrt(σ6_fac + r^6))
    return λ * ke * ((qij) / R)
end

@doc raw"""
    CoulombSoftCoreGapsys(; cutoff, α, σQ, use_neighbors, λ_mixing, scheduler,
                          weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Gapsys et al. 2012](https://doi.org/10.1021/ct300220p).
[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each [`Coulomb`](@ref)
to this type with `Csoftcore=:gapsys`.
The potential energy is defined as
```math
V(r_{ij}) = \left\{ \begin{array}{cl}
\lambda \frac{1}{4\pi\varepsilon_0} \frac{q_iq_j}{r_{ij}}, & \text{if} & r_{ij} \ge r_Q \\
\lambda \frac{1}{4\pi\varepsilon_0} (\frac{q_iq_j}{r_Q^3}r_{ij}^2-\frac{3q_iq_j}{r_Q^2}r_{ij}+\frac{3q_iq_j}{r_Q}), & \text{if} & r_{ij} \lt r_Q \\
\end{array} \right.
```
and the force on each atom by
```math
\vec{F}_i = \left\{ \begin{array}{cl}
\lambda \frac{1}{4\pi\varepsilon_0} \frac{q_iq_j}{r_{ij}^2}\frac{\vec{r}_{ij}}{r_{ij}}, & \text{if} & r_{ij} \ge r_Q \\
\lambda \frac{1}{4\pi\varepsilon_0} (\frac{-2q_iq_j}{r_Q^3}r_{ij}+\frac{3q_iq_j}{r_Q^2})\frac{\vec{r}_{ij}}{r_{ij}}, & \text{if} & r_{ij} \lt r_Q \\
\end{array} \right.
```
where
```math
r_Q = \alpha(1-\lambda)^{1/6}(1+\sigma_Q|q_i q_j|)
```
and ``\sigma_Q`` is `σQ`.
``\lambda`` is the electrostatic coupling of the pair, which `scheduler`, a
[`Molly.LambdaScheduler`](@ref), gives from the `λ` of the atoms mixed with `λ_mixing`
and their alchemical roles.
If ``\lambda`` is 1 this gives the standard [`Coulomb`](@ref) potential, and if it is 0
the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
struct CoulombSoftCoreGapsys{C, A, S, LM, SCH, W, T} <: PairwiseInteraction
    cutoff::C
    α::A
    σQ::S
    use_neighbors::Bool
    λ_mixing::LM
    scheduler::SCH
    weight_special::W
    coulomb_const::T
end

function CoulombSoftCoreGapsys(; cutoff=NoCutoff(), α=0.3, σQ=1.0u"nm", use_neighbors=true,
                                    λ_mixing=MinimumMixing(),
                                    scheduler=DefaultLambdaScheduler(), weight_special=1,
                                    coulomb_const=coulomb_const)
    return CoulombSoftCoreGapsys(
        cutoff,
        α,
        σQ,
        use_neighbors,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
    )
end

use_neighbors(inter::CoulombSoftCoreGapsys) = inter.use_neighbors

required_atom_fields(inter::CoulombSoftCoreGapsys) = (:charge, :λ, :alch_role,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(coul::CoulombSoftCoreGapsys{C, A, S, LM, SCH, W, T}) where {C, A, S, LM, SCH, W, T}
    return CoulombSoftCoreGapsys(
        coul.cutoff,
        zero(A),
        coul.σQ,
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombSoftCoreGapsys, c2::CoulombSoftCoreGapsys)
    return CoulombSoftCoreGapsys(
        c1.cutoff,
        c1.α + c2.α,
        c1.σQ + c2.σQ,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

parameter_prefix(::CoulombSoftCoreGapsys) = "inter_CO_"
parameter_fields(::Type{<:CoulombSoftCoreGapsys}) =
    ((:α, "α"), (:σQ, "σQ"), (:weight_special, "weight_14"),
     (:coulomb_const, "coulomb_const"))


# σQ of the Gapsys soft core, in nm, with units only if the interaction it replaces has them
function softcore_σQ(σQ, coulomb_const, float_type)
    σQ_nm = float_type(σQ isa Unitful.Quantity ? ustrip(u"nm", σQ) : σQ)
    return unit(coulomb_const) == NoUnits ? σQ_nm : σQ_nm * u"nm"
end

function to_lambda_function(inter::Coulomb, ::GapsysSoftCore; α=0.3, σQ=1.0u"nm", λ_mixing=MinimumMixing(), 
                                scheduler=DefaultLambdaScheduler(), float_type=Float32, args...)
    return CoulombSoftCoreGapsys(cutoff=inter.cutoff, α=float_type(α), σQ=softcore_σQ(σQ, inter.coulomb_const, float_type), 
                                    use_neighbors=inter.use_neighbors, λ_mixing=λ_mixing, scheduler=scheduler, 
                                    weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end

@inline function force(inter::CoulombSoftCoreGapsys, 
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)

    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)
    
    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if iszero_value(qij)
        return zero_pairwise_force(dr, force_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end
    cutoff = inter.cutoff

    # Fast Path: Standard Coulomb (Length 4)
    if λ >= 1
        params = (ke, qij, nothing, nothing)
        f = force_cutoff(cutoff, inter, r, params)
        fdr = radial_force_vector(f, r, dr, force_units)
        return special ? fdr * inter.weight_special : fdr
    end

    # Alchemical Path
    # R is precomputed here, saving absolute value and unit math inside the inner loop
    σ6_fac = inter.α * sqrt(cbrt(1 - (λ*λR)))
    R = σ6_fac * (oneunit(r) + (inter.σQ * abs(qij)))
    params = (ke, qij, λ, R)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = radial_force_vector(f, r, dr, force_units)
    return special ? fdr * inter.weight_special : fdr
end

@inline function pairwise_force(::CoulombSoftCoreGapsys, r, (ke, qij, _, _)::Tuple{Any, Any, Nothing, Nothing})
    return (ke * qij) / r^2
end

@inline function pairwise_force(::CoulombSoftCoreGapsys, r, (ke, qij, λ, R)::Tuple{Any, Any, Any, Any})
    if !(r < R)
        return λ * ke * (qij / (r^2))
    else
        return λ * ke * (-(((2 * qij) / (R^3)) * r) + ((3 * qij) / (R^2)))
    end
end

@inline function potential_energy(inter::CoulombSoftCoreGapsys,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)

    ke = inter.coulomb_const
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end

    if iszero_value(qij)
        return zero_pairwise_energy(dr, energy_units)
    end

    r = sqrt(sum(abs2, dr))
    cutoff = inter.cutoff

    if λ >= 1
        if iszero_value(r)
            return zero_pairwise_energy(dr, energy_units)
        end
        params = (ke, qij, nothing, nothing)
        pe = pe_cutoff(cutoff, inter, r, params)
        return special ? pe * inter.weight_special : pe
    end

    # Precompute R
    σ6_fac = inter.α * sqrt(cbrt(1 - (λ*λR)))
    R = σ6_fac * (oneunit(r) + (inter.σQ * abs(qij)))

    if iszero_value(r)
        pe = overlap_pe_coulomb_softcore_gapsys(dr, energy_units, ke, qij, λ, R)
        return special ? pe * inter.weight_special : pe
    end

    params = (ke, qij, λ, R)

    pe = pe_cutoff(cutoff, inter, r, params)
    return special ? pe * inter.weight_special : pe
end

@inline function pairwise_pe(::CoulombSoftCoreGapsys, r, (ke, qij, _, _)::Tuple{Any, Any, Nothing, Nothing})
    return (ke * qij) * inv(r)
end

@inline function pairwise_pe(::CoulombSoftCoreGapsys, r, (ke, qij, λ, R)::Tuple{Any, Any, Any, Any})
    if !(r < R)
        return λ * (ke * (qij/r))
    else
        return λ * (ke * (((qij/(R^3))*(r^2))-(((3*qij)/(R^2))*r)+((3*qij)/R)))
    end
end

const crf_solvent_dielectric = 78.3

function check_reaction_field_dielectrics(solvent_dielectric, solute_dielectric)
    if !(solute_dielectric > 0) || isinf(solute_dielectric)
        throw(ArgumentError("solute_dielectric ($solute_dielectric) must be positive and finite"))
    end
    if !(solvent_dielectric > 0)
        throw(ArgumentError("solvent_dielectric ($solvent_dielectric) must be positive, use " *
                            "Inf for conducting boundary conditions"))
    end
    if solvent_dielectric < solute_dielectric
        throw(ArgumentError("solvent_dielectric ($solvent_dielectric) must not be less than " *
                            "solute_dielectric ($solute_dielectric)"))
    end
    return nothing
end

# The reaction field constants are not stored in the interactions as recomputing them is
#   easier for AD
@inline function reaction_field_constants(dist_cutoff, solvent_dielectric, solute_dielectric)
    if isinf(solvent_dielectric) # Conducting boundary conditions
        krf = inv(2 * dist_cutoff^3)
        crf = 3 * inv(2 * dist_cutoff)
    else
        krf = inv(dist_cutoff^3) * (solvent_dielectric - solute_dielectric) /
                    (2 * solvent_dielectric + solute_dielectric)
        crf = inv(dist_cutoff) * (3 * solvent_dielectric) /
                    (2 * solvent_dielectric + solute_dielectric)
    end
    return krf, crf
end

abstract type AbstractReactionField <: PairwiseInteraction end

@doc raw"""
    CoulombReactionField(; dist_cutoff, solvent_dielectric, solute_dielectric, use_neighbors,
                         weight_special, coulomb_const)

The Coulomb electrostatic interaction modified using the reaction field approximation
between two atoms.

See [Tironi et al. 1995](https://doi.org/10.1063/1.469273).
The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 \varepsilon_\mathrm{r}} \left( \frac{1}{r_{ij}} + k_\mathrm{rf} r_{ij}^2 - c_\mathrm{rf} \right)
```
where
```math
k_\mathrm{rf} = \frac{1}{r_\mathrm{c}^3} \frac{\varepsilon_\mathrm{rf} - \varepsilon_\mathrm{r}}{2\varepsilon_\mathrm{rf} + \varepsilon_\mathrm{r}}, \quad
c_\mathrm{rf} = \frac{1}{r_\mathrm{c}} + k_\mathrm{rf} r_\mathrm{c}^2 = \frac{1}{r_\mathrm{c}} \frac{3\varepsilon_\mathrm{rf}}{2\varepsilon_\mathrm{rf} + \varepsilon_\mathrm{r}}
```
`solvent_dielectric` corresponds to ``\varepsilon_\mathrm{rf}``, the relative permittivity of
the continuum beyond the cutoff, and `solute_dielectric` to ``\varepsilon_\mathrm{r}``, the
relative permittivity within the cutoff.
`coulomb_const` is divided by `solute_dielectric`, so it should not include
``\varepsilon_\mathrm{r}`` itself.
Setting `solvent_dielectric=Inf` gives conducting boundary conditions
(``k_\mathrm{rf} = 1/(2r_\mathrm{c}^3)``, ``c_\mathrm{rf} = 3/(2r_\mathrm{c})``).
`solvent_dielectric` must not be less than `solute_dielectric`.
`solute_dielectric` defaults to one, which gives the reaction field of OpenMM.

Special (1-4) pairs use the plain Coulomb interaction without the reaction field, scaled by
`weight_special` and, like all pairs, divided by `solute_dielectric`.

[`SetupCoulombReactionField`](@ref) provides reaction field parameters when setting up a
system from a file.
"""
struct CoulombReactionField{D, S, E, W, T} <: AbstractReactionField
    dist_cutoff::D
    solvent_dielectric::S
    solute_dielectric::E
    use_neighbors::Bool
    weight_special::W
    coulomb_const::T
end

function CoulombReactionField(; dist_cutoff,
                              solvent_dielectric=float(typeof(ustrip(dist_cutoff)))(crf_solvent_dielectric),
                              solute_dielectric=one(solvent_dielectric),
                              use_neighbors=true, weight_special=1, coulomb_const=coulomb_const)
    check_reaction_field_dielectrics(solvent_dielectric, solute_dielectric)
    return CoulombReactionField(dist_cutoff, solvent_dielectric, solute_dielectric,
                                use_neighbors, weight_special, coulomb_const)
end

use_neighbors(inter::CoulombReactionField) = inter.use_neighbors

required_atom_fields(::CoulombReactionField) = (:charge,)

function Base.zero(coul::CoulombReactionField{D, S, E, W, T}) where {D, S, E, W, T}
    return CoulombReactionField{D, S, E, W, T}(
        zero(D),
        zero(S),
        zero(E),
        coul.use_neighbors,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombReactionField, c2::CoulombReactionField)
    return CoulombReactionField(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.solute_dielectric + c2.solute_dielectric,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

parameter_prefix(::CoulombReactionField) = "inter_CO_"
parameter_fields(::Type{<:CoulombReactionField}) =
    ((:dist_cutoff, "dist_cutoff"), (:solvent_dielectric, "solvent_dielectric"),
     (:solute_dielectric, "solute_dielectric"), (:weight_special, "weight_14"),
     (:coulomb_const, "coulomb_const"))


@inline function force(inter::CoulombReactionField,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r2 = sum(abs2, dr)
    ke = inter.coulomb_const / inter.solute_dielectric
    qi, qj = charge(atom_i), charge(atom_j)
    r = sqrt(r2)

    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = inv(inter.dist_cutoff^3) * 0
    else
        krf, _ = reaction_field_constants(inter.dist_cutoff, inter.solvent_dielectric,
                                          inter.solute_dielectric)
    end

    f = (ke * qi * qj) * (inv(r) - 2 * krf * r2) * inv(r2)

    if special
        return f * dr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return f * dr * (r <= inter.dist_cutoff)
    end
end

@inline function potential_energy(inter::CoulombReactionField,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r2 = sum(abs2, dr)
    ke = inter.coulomb_const / inter.solute_dielectric
    qi, qj = charge(atom_i), charge(atom_j)
    r = sqrt(r2)

    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = inv(inter.dist_cutoff^3) * 0
        crf = inv(inter.dist_cutoff) * 0
    else
        krf, crf = reaction_field_constants(inter.dist_cutoff, inter.solvent_dielectric,
                                            inter.solute_dielectric)
    end

    pe = (ke * qi * qj) * (inv(r) + krf * r2 - crf)

    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * (r <= inter.dist_cutoff)
    end
end

"""
    SetupCoulombReactionField(; solvent_dielectric, solute_dielectric, coulomb_const,
                              exclusion_correction=false)

Set up the Coulomb electrostatic interaction modified using the reaction field approximation
between two atoms.

Passed to the [`System`](@ref) constructor from files, where it creates a
[`CoulombReactionField`](@ref) pairwise interaction.
If `exclusion_correction` is `true`, the reaction field is also applied to the excluded and
special pairs and to each atom with itself, using the [`CoulombReactionFieldExclusion`](@ref)
specific interaction, which requires a neighbor finder.
"""
struct SetupCoulombReactionField{T, E, C}
    solvent_dielectric::T
    solute_dielectric::E
    coulomb_const::C
    exclusion_correction::Bool
end

function SetupCoulombReactionField(; solvent_dielectric=crf_solvent_dielectric,
                                   solute_dielectric=one(crf_solvent_dielectric),
                                   coulomb_const=coulomb_const,
                                   exclusion_correction::Bool=false)
    check_reaction_field_dielectrics(solvent_dielectric, solute_dielectric)
    return SetupCoulombReactionField(solvent_dielectric, solute_dielectric, coulomb_const,
                                     exclusion_correction)
end

function setup_coulomb_pairwise(scrf::SetupCoulombReactionField, dist_cutoff,
                                weight_special, use_neighbors, units, T)
    return CoulombReactionField(
        dist_cutoff=T(dist_cutoff),
        solvent_dielectric=T(scrf.solvent_dielectric),
        solute_dielectric=T(scrf.solute_dielectric),
        use_neighbors=use_neighbors,
        weight_special=weight_special,
        coulomb_const=convert_setup_quantity(scrf.coulomb_const, units, T),
    )
end

@doc raw"""
    CoulombReactionFieldScaled(; dist_cutoff, solvent_dielectric, solute_dielectric,
                               use_neighbors, λ_mixing, scheduler, weight_special,
                               coulomb_const)

The reaction field Coulomb interaction between two atoms scaled by λ without a soft core,
the λ version of [`CoulombReactionField`](@ref).

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each
[`CoulombReactionField`](@ref) to this type with `Csoftcore=:scaled`.
The potential energy is that of [`CoulombReactionField`](@ref) multiplied by ``\lambda``,
the electrostatic coupling of the pair, which `scheduler`, a
[`Molly.LambdaScheduler`](@ref), gives from the `λ` of the atoms mixed with `λ_mixing`
and their alchemical roles.
In single topology the charges are interpolated between the end states.
"""
struct CoulombReactionFieldScaled{D, S, E, LM, SCH, W, T} <: AbstractReactionField
    dist_cutoff::D
    solvent_dielectric::S
    solute_dielectric::E
    use_neighbors::Bool
    λ_mixing::LM
    scheduler::SCH
    weight_special::W
    coulomb_const::T
end

function CoulombReactionFieldScaled(; dist_cutoff,
                                    solvent_dielectric=float(typeof(ustrip(dist_cutoff)))(crf_solvent_dielectric),
                                    solute_dielectric=one(solvent_dielectric),
                                    use_neighbors=true, λ_mixing=MinimumMixing(),
                                    scheduler=DefaultLambdaScheduler(), weight_special=1,
                                    coulomb_const=coulomb_const)
    check_reaction_field_dielectrics(solvent_dielectric, solute_dielectric)
    return CoulombReactionFieldScaled(dist_cutoff, solvent_dielectric, solute_dielectric,
                                      use_neighbors, λ_mixing, scheduler, weight_special,
                                      coulomb_const)
end

use_neighbors(inter::CoulombReactionFieldScaled) = inter.use_neighbors

required_atom_fields(::CoulombReactionFieldScaled) = (:charge, :λ, :alch_role)

function Base.zero(coul::CoulombReactionFieldScaled{D, S, E, LM, SCH, W, T}) where {D, S, E, LM, SCH, W, T}
    return CoulombReactionFieldScaled{D, S, E, LM, SCH, W, T}(
        zero(D),
        zero(S),
        zero(E),
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombReactionFieldScaled, c2::CoulombReactionFieldScaled)
    return CoulombReactionFieldScaled(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.solute_dielectric + c2.solute_dielectric,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

parameter_prefix(::CoulombReactionFieldScaled) = "inter_CO_"
parameter_fields(::Type{<:CoulombReactionFieldScaled}) =
    ((:dist_cutoff, "dist_cutoff"), (:solvent_dielectric, "solvent_dielectric"),
     (:solute_dielectric, "solute_dielectric"), (:weight_special, "weight_14"),
     (:coulomb_const, "coulomb_const"))


function to_lambda_function(inter::CoulombReactionField, ::ScaledSoftCore; λ_mixing=MinimumMixing(), 
                                scheduler=DefaultLambdaScheduler(), float_type=Float32, args...)
    return CoulombReactionFieldScaled(dist_cutoff=inter.dist_cutoff, solvent_dielectric=inter.solvent_dielectric,
                                    solute_dielectric=inter.solute_dielectric,
                                    use_neighbors=inter.use_neighbors, λ_mixing=λ_mixing, scheduler=scheduler, 
                                    weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end


@inline function force(inter::CoulombReactionFieldScaled,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    ke = inter.coulomb_const / inter.solute_dielectric
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if iszero_value(qij)
        return zero_pairwise_force(dr, force_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    if special
        krf = inv(inter.dist_cutoff^3) * 0
    else
        krf, _ = reaction_field_constants(inter.dist_cutoff, inter.solvent_dielectric,
                                          inter.solute_dielectric)
    end

    f = λ * (ke * qij) * (inv(r) - 2 * krf * r2) * inv(r2)
    return special ? f * dr * inter.weight_special * (r <= inter.dist_cutoff) :
                     f * dr * (r <= inter.dist_cutoff)
end

@inline function potential_energy(inter::CoulombReactionFieldScaled,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    ke = inter.coulomb_const / inter.solute_dielectric
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end

    if iszero_value(qij)
        return zero_pairwise_energy(dr, energy_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    if iszero_value(r)
        return zero_pairwise_energy(dr, energy_units)
    end

    if special
        krf = inv(inter.dist_cutoff^3) * 0
        crf = inv(inter.dist_cutoff) * 0
    else
        krf, crf = reaction_field_constants(inter.dist_cutoff, inter.solvent_dielectric,
                                            inter.solute_dielectric)
    end

    pe = λ * (ke * qij) * (inv(r) + krf * r2 - crf)
    return special ? pe * inter.weight_special * (r <= inter.dist_cutoff) :
                     pe * (r <= inter.dist_cutoff)
end

@doc raw"""
    CoulombSoftCoreBeutlerReactionField(; dist_cutoff, solvent_dielectric, solute_dielectric,
                                         α, use_neighbors, σ_mixing, ϵ_mixing, λ_mixing,
                                         scheduler, weight_special, coulomb_const)

The Coulomb electrostatic interaction with Beutler soft core and reaction field correction,
used for the appearing and disappearing of atoms in alchemical simulations.

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each
[`CoulombReactionField`](@ref) to this type with `Csoftcore=:beutler`.
The soft core and ``\lambda``, the electrostatic coupling of the pair, are as for
[`CoulombSoftCoreBeutler`](@ref).
At ``\lambda = 1`` this reduces to [`CoulombReactionField`](@ref).
At ``\lambda = 0`` the interaction is zero.
Special (1-4) pairs disable the reaction field correction and use plain soft-core Coulomb
scaled by `weight_special` and divided by `solute_dielectric`.
"""
struct CoulombSoftCoreBeutlerReactionField{D, S, E, A, SM, EM, LM, SCH, W, T} <: AbstractReactionField
    dist_cutoff::D
    solvent_dielectric::S 
    solute_dielectric::E
    α::A 
    use_neighbors::Bool 
    σ_mixing::SM 
    ϵ_mixing::EM 
    λ_mixing::LM 
    scheduler::SCH 
    weight_special::W 
    coulomb_const::T
end

function CoulombSoftCoreBeutlerReactionField(; dist_cutoff,
                                              solvent_dielectric=float(typeof(ustrip(dist_cutoff)))(crf_solvent_dielectric),
                                              solute_dielectric=one(solvent_dielectric),
                                              α=0.3, use_neighbors=true, σ_mixing=LorentzMixing(),
                                              ϵ_mixing=GeometricMixing(), λ_mixing=MinimumMixing(),
                                              scheduler=DefaultLambdaScheduler(), weight_special=1,
                                              coulomb_const=coulomb_const)
    check_reaction_field_dielectrics(solvent_dielectric, solute_dielectric)
    return CoulombSoftCoreBeutlerReactionField(
        dist_cutoff,
        solvent_dielectric,
        solute_dielectric,
        α,
        use_neighbors,
        σ_mixing,
        ϵ_mixing,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
    )
end

use_neighbors(inter::CoulombSoftCoreBeutlerReactionField) = inter.use_neighbors

required_atom_fields(inter::CoulombSoftCoreBeutlerReactionField) = (:charge, :σ, :ϵ, :λ, :alch_role,
            mixing_atom_fields(inter.σ_mixing)..., mixing_atom_fields(inter.ϵ_mixing)...,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(coul::CoulombSoftCoreBeutlerReactionField{D, S, E, A, SM, EM, LM, SCH, W, T}) where {D, S, E, A, SM, EM, LM, SCH, W, T}
    return CoulombSoftCoreBeutlerReactionField(
        zero(D),
        zero(S),
        zero(E),
        zero(A),
        coul.use_neighbors,
        coul.σ_mixing,
        coul.ϵ_mixing,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombSoftCoreBeutlerReactionField, c2::CoulombSoftCoreBeutlerReactionField)
    return CoulombSoftCoreBeutlerReactionField(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.solute_dielectric + c2.solute_dielectric,
        c1.α + c2.α,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.ϵ_mixing,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

parameter_prefix(::CoulombSoftCoreBeutlerReactionField) = "inter_CO_"
parameter_fields(::Type{<:CoulombSoftCoreBeutlerReactionField}) =
    ((:dist_cutoff, "dist_cutoff"), (:solvent_dielectric, "solvent_dielectric"),
     (:solute_dielectric, "solute_dielectric"), (:α, "α"), (:weight_special, "weight_14"),
     (:coulomb_const, "coulomb_const"))


function to_lambda_function(inter::CoulombReactionField, ::BeutlerSoftCore; α=0.3, λ_mixing=MinimumMixing(), 
                            scheduler=DefaultLambdaScheduler(), float_type=Float32,
                            σ_mixing=LorentzMixing(), ϵ_mixing=GeometricMixing(), args...)
    return CoulombSoftCoreBeutlerReactionField(dist_cutoff=inter.dist_cutoff, 
                                                solvent_dielectric=inter.solvent_dielectric,
                                                solute_dielectric=inter.solute_dielectric, α=float_type(α), 
                                                use_neighbors=inter.use_neighbors, σ_mixing=σ_mixing, 
                                                ϵ_mixing=ϵ_mixing, λ_mixing=λ_mixing, 
                                                scheduler=scheduler, weight_special=inter.weight_special, 
                                                coulomb_const=inter.coulomb_const)
end

@inline function force(inter::CoulombSoftCoreBeutlerReactionField,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    ke = inter.coulomb_const / inter.solute_dielectric
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if iszero_value(qij)
        return zero_pairwise_force(dr, force_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end
    rc = inter.dist_cutoff

    if special
        krf = inv(rc^3) * zero(T)
    else
        krf, _ = reaction_field_constants(rc, inter.solvent_dielectric, inter.solute_dielectric)
    end

    if λ >= 1 && λR >= 1
        f = (ke * qij) * (inv(r) - 2 * krf * r2) * inv(r2)
        if special
            return f * dr * inter.weight_special * (r <= rc)
        else
            return f * dr * (r <= rc)
        end
    end

    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j, λ_params, pair_role)^6
    term = inter.α * (1 - (λ*λR)) * σ6 + r2^3
    f = λ * (ke * qij) * (r2^2 / (term * sqrt(cbrt(term))) - 2 * krf)

    if special
        return f * dr * inter.weight_special * (r <= rc)
    else
        return f * dr * (r <= rc)
    end
end

@inline function potential_energy(inter::CoulombSoftCoreBeutlerReactionField,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    ke = inter.coulomb_const / inter.solute_dielectric
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end

    if iszero_value(qij)
        return zero_pairwise_energy(dr, energy_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    rc = inter.dist_cutoff

    if λ >= 1 && λR >= 1
        if iszero_value(r)
            return zero_pairwise_energy(dr, energy_units)
        end
        if special
            krf = inv(rc^3) * zero(T)
            crf = inv(rc) * zero(T)
        else
            krf, crf = reaction_field_constants(rc, inter.solvent_dielectric,
                                                inter.solute_dielectric)
        end
        pe = (ke * qij) * (inv(r) + krf * r2 - crf)
        if special
            return pe * inter.weight_special * (r <= rc)
        else
            return pe * (r <= rc)
        end
    end

    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j, λ_params, pair_role)^6
    σ6_fac = inter.α * (1 - (λ*λR)) * σ6

    if iszero_value(r)
        krf = special ? inv(rc^3) * zero(T) :
              first(reaction_field_constants(rc, inter.solvent_dielectric,
                                             inter.solute_dielectric))
        pe = overlap_pe_coulomb_softcore_beutler_rf(
            dr,
            energy_units,
            ke,
            qij,
            λ,
            σ6_fac,
            krf,
            rc,
            special,
        )
        return special ? pe * inter.weight_special * (r <= rc) : pe * (r <= rc)
    end

    term = σ6_fac + r2^3
    R_eff = sqrt(cbrt(term))

    if special
        pe = λ * (ke * qij) * inv(R_eff)
        return pe * inter.weight_special * (r <= rc)
    else
        krf, _ = reaction_field_constants(rc, inter.solvent_dielectric, inter.solute_dielectric)
        crf_λ = inv(sqrt(cbrt(inter.α * (1 - λ*λR) * σ6 + rc^6))) + krf * rc^2
        pe = λ * (ke * qij) * (inv(R_eff) + krf * r2 - crf_λ)
        return pe * (r <= rc)
    end
end

@doc raw"""
    CoulombSoftCoreGapsysReactionField(; dist_cutoff, solvent_dielectric, solute_dielectric,
                                        α, σQ, use_neighbors, λ_mixing, scheduler,
                                        weight_special, coulomb_const)

The Coulomb electrostatic interaction with Gapsys soft core and reaction field correction,
used for the appearing and disappearing of atoms in alchemical simulations.

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each
[`CoulombReactionField`](@ref) to this type with `Csoftcore=:gapsys`.
The soft core and ``\lambda``, the electrostatic coupling of the pair, are as for
[`CoulombSoftCoreGapsys`](@ref).
At ``\lambda = 1`` this reduces to [`CoulombReactionField`](@ref).
At ``\lambda = 0`` the interaction is zero.
Special (1-4) pairs disable the reaction field correction and use the standard Gapsys
polynomial scaled by `weight_special` and divided by `solute_dielectric`.
"""
struct CoulombSoftCoreGapsysReactionField{D, S, E, A, SQ, LM, SCH, W, T} <: AbstractReactionField
    dist_cutoff::D
    solvent_dielectric::S 
    solute_dielectric::E
    α::A 
    σQ::SQ 
    use_neighbors::Bool 
    λ_mixing::LM 
    scheduler::SCH 
    weight_special::W 
    coulomb_const::T 
end

function CoulombSoftCoreGapsysReactionField(; dist_cutoff,
                                              solvent_dielectric=float(typeof(ustrip(dist_cutoff)))(crf_solvent_dielectric),
                                              solute_dielectric=one(solvent_dielectric),
                                              α=0.3, σQ=1.0u"nm", use_neighbors=true,
                                              λ_mixing=MinimumMixing(),
                                              scheduler=DefaultLambdaScheduler(), weight_special=1,
                                              coulomb_const=coulomb_const)
    check_reaction_field_dielectrics(solvent_dielectric, solute_dielectric)
    return CoulombSoftCoreGapsysReactionField(
        dist_cutoff,
        solvent_dielectric,
        solute_dielectric,
        α,
        σQ,
        use_neighbors,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
    )
end

use_neighbors(inter::CoulombSoftCoreGapsysReactionField) = inter.use_neighbors

required_atom_fields(inter::CoulombSoftCoreGapsysReactionField) = (:charge, :λ, :alch_role,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(coul::CoulombSoftCoreGapsysReactionField{D, S, E, A, SQ, LM, SCH, W, T}) where {D, S, E, A, SQ, LM, SCH, W, T}
    return CoulombSoftCoreGapsysReactionField(
        zero(D),
        zero(S),
        zero(E),
        zero(A),
        coul.σQ,
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombSoftCoreGapsysReactionField, c2::CoulombSoftCoreGapsysReactionField)
    return CoulombSoftCoreGapsysReactionField(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.solute_dielectric + c2.solute_dielectric,
        c1.α + c2.α,
        c1.σQ + c2.σQ,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

parameter_prefix(::CoulombSoftCoreGapsysReactionField) = "inter_CO_"
parameter_fields(::Type{<:CoulombSoftCoreGapsysReactionField}) =
    ((:dist_cutoff, "dist_cutoff"), (:solvent_dielectric, "solvent_dielectric"),
     (:solute_dielectric, "solute_dielectric"), (:α, "α"), (:σQ, "σQ"),
     (:weight_special, "weight_14"), (:coulomb_const, "coulomb_const"))


function to_lambda_function(inter::CoulombReactionField, ::GapsysSoftCore; α=0.3, σQ=1.0u"nm", 
                                λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler(), 
                                float_type=Float32, args...)
    return CoulombSoftCoreGapsysReactionField(dist_cutoff=inter.dist_cutoff, 
                                                solvent_dielectric=inter.solvent_dielectric,
                                                solute_dielectric=inter.solute_dielectric, α=float_type(α), 
                                                σQ=softcore_σQ(σQ, inter.coulomb_const, float_type), use_neighbors=inter.use_neighbors, 
                                                λ_mixing=λ_mixing, scheduler=scheduler, 
                                                weight_special=inter.weight_special, coulomb_const=inter.coulomb_const)
end

@inline function force(inter::CoulombSoftCoreGapsysReactionField,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    ke = inter.coulomb_const / inter.solute_dielectric
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if iszero_value(qij)
        return zero_pairwise_force(dr, force_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    rc = inter.dist_cutoff

    if special
        krf = inv(rc^3) * zero(T)
    else
        krf, _ = reaction_field_constants(rc, inter.solvent_dielectric, inter.solute_dielectric)
    end

    if λ >= 1 && λR >= 1
        f = (ke * qij) * (inv(r) - 2 * krf * r2) * inv(r2)
        if special
            return f * dr * inter.weight_special * (r <= rc)
        else
            return f * dr * (r <= rc)
        end
    end

    R = inter.α * sqrt(cbrt(1 - (λ*λR))) * (oneunit(r) + inter.σQ * abs(qij))
    if !(r < R)
        f = λ * (ke * qij) * (inv(r) - 2 * krf * r2) * inv(r2)
    else
        f = λ * ke * (-2 * qij * (inv(R^3) + krf) + 3 * qij * inv(R^2) * inv(r))
    end

    if special
        return f * dr * inter.weight_special * (r <= rc)
    else
        return f * dr * (r <= rc)
    end
end

@inline function potential_energy(inter::CoulombSoftCoreGapsysReactionField,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    ke = inter.coulomb_const / inter.solute_dielectric
    T = typeof(ustrip(ke))
    pair_role, λ, λR, λ_params, qij = softcore_pair_elec_lambda(inter, atom_i, atom_j, special)

    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end

    if iszero_value(qij)
        return zero_pairwise_energy(dr, energy_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    rc = inter.dist_cutoff

    if special
        krf = inv(rc^3) * zero(T)
        crf = inv(rc) * zero(T)
    else
        krf, crf = reaction_field_constants(rc, inter.solvent_dielectric, inter.solute_dielectric)
    end

    if λ >= 1 && λR >= 1
        if iszero_value(r)
            return zero_pairwise_energy(dr, energy_units)
        end
        pe = (ke * qij) * (inv(r) + krf * r2 - crf)
        if special
            return pe * inter.weight_special * (r <= rc)
        else
            return pe * (r <= rc)
        end
    end

    R = inter.α * sqrt(cbrt(1 - (λ*λR))) * (oneunit(r) + inter.σQ * abs(qij))
    if iszero_value(r)
        pe = overlap_pe_coulomb_softcore_gapsys_rf(
            dr,
            energy_units,
            ke,
            qij,
            λ,
            R,
            crf,
        )
    elseif !(r < R)
        pe = λ * (ke * qij) * (inv(r) + krf * r2 - crf)
    else
        A = qij * (inv(R^3) + krf)
        B = -3 * qij * inv(R^2)
        C = qij * (3 * inv(R) - crf)
        pe = λ * ke * (A * r2 + B * r + C)
    end

    if special
        return pe * inter.weight_special * (r <= rc)
    else
        return pe * (r <= rc)
    end
end

@doc raw"""
    CoulombReactionFieldExclusion(; self=false)

The reaction field correction for an excluded atom pair, or for an atom with itself if
`self` is `true`.

The dielectric continuum beyond the cutoff responds to all charges within the cutoff,
including those of excluded pairs, so the reaction field is applied to excluded pairs and to
each atom with itself, without the direct Coulomb term.
For an excluded pair within the cutoff the potential energy is
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 \varepsilon_\mathrm{r}} \left( k_\mathrm{rf} r_{ij}^2 - c_\mathrm{rf} \right)
```
and for each atom it is
```math
V_\mathrm{self} = -\frac{q_i^2}{4 \pi \varepsilon_0 \varepsilon_\mathrm{r}} \frac{c_\mathrm{rf}}{2}
```
which gives no force.
``k_\mathrm{rf}`` and ``c_\mathrm{rf}`` are as for [`CoulombReactionField`](@ref).
Excluded pairs beyond the cutoff contribute nothing, so the energy of an excluded pair jumps
by ``q_i q_j / (4 \pi \varepsilon_0 \varepsilon_\mathrm{r} r_\mathrm{c})`` at the cutoff.
This matches GROMACS, whereas OpenMM does not apply the correction.

Since the properties of the interaction are the same for all pairs, they are stored once
in the `data` field of the [`InteractionList2Atoms`](@ref) with
[`Molly.CoulombReactionFieldExclusionData`](@ref).
[`Molly.reaction_field_exclusions`](@ref) sets up the list of excluded pairs and atoms.
"""
@kwdef struct CoulombReactionFieldExclusion
    self::Bool = false
end

Base.zero(e::CoulombReactionFieldExclusion) = e
Base.:+(e1::CoulombReactionFieldExclusion, ::CoulombReactionFieldExclusion) = e1

"""
    CoulombReactionFieldExclusionData(; dist_cutoff, solvent_dielectric, solute_dielectric,
                                      coulomb_const, scheduler)
    CoulombReactionFieldExclusionData(inter; scheduler)

The parameters of the [`CoulombReactionFieldExclusion`](@ref) interactions, stored in the
`data` field of their [`InteractionList2Atoms`](@ref).

The parameters should match the reaction field pairwise interaction, `inter`, from which they
can be taken.
`scheduler`, a [`Molly.LambdaScheduler`](@ref), gives the charges of alchemical atoms as for
[`EwaldExclusion`](@ref) and defaults to that of `inter`, if any.
"""
struct CoulombReactionFieldExclusionData{D, S, E, C, SCH}
    dist_cutoff::D
    solvent_dielectric::S
    solute_dielectric::E
    coulomb_const::C
    scheduler::SCH
end

function CoulombReactionFieldExclusionData(; dist_cutoff,
                                           solvent_dielectric=float(typeof(ustrip(dist_cutoff)))(crf_solvent_dielectric),
                                           solute_dielectric=one(solvent_dielectric),
                                           coulomb_const=coulomb_const,
                                           scheduler=DefaultLambdaScheduler())
    check_reaction_field_dielectrics(solvent_dielectric, solute_dielectric)
    return CoulombReactionFieldExclusionData(dist_cutoff, solvent_dielectric, solute_dielectric,
                                             coulomb_const, scheduler)
end

function CoulombReactionFieldExclusionData(inter::AbstractReactionField;
                                           scheduler=(hasproperty(inter, :scheduler) ?
                                                inter.scheduler : DefaultLambdaScheduler()))
    return CoulombReactionFieldExclusionData(
        dist_cutoff=inter.dist_cutoff,
        solvent_dielectric=inter.solvent_dielectric,
        solute_dielectric=inter.solute_dielectric,
        coulomb_const=inter.coulomb_const,
        scheduler=scheduler,
    )
end

function Base.zero(d::CoulombReactionFieldExclusionData{D, S, E, C}) where {D, S, E, C}
    return CoulombReactionFieldExclusionData(zero(D), zero(S), zero(E), zero(C), d.scheduler)
end

function Base.:+(d1::CoulombReactionFieldExclusionData, d2::CoulombReactionFieldExclusionData)
    return CoulombReactionFieldExclusionData(
        d1.dist_cutoff + d2.dist_cutoff,
        d1.solvent_dielectric + d2.solvent_dielectric,
        d1.solute_dielectric + d2.solute_dielectric,
        d1.coulomb_const + d2.coulomb_const,
        d1.scheduler,
    )
end

to_device(x::CoulombReactionFieldExclusionData, ::Type{AT}) where AT = x

# The same charge product as the Ewald exclusions
@inline function reaction_field_exclusion_qq(data, atom_i, atom_j)
    T = float(typeof(ustrip(data.dist_cutoff)))
    return ewald_pair_qq(data.scheduler, atom_i, atom_j, Val(T))
end

@inline function force(inter::CoulombReactionFieldExclusion, coord_i, coord_j, boundary,
                       atom_i, atom_j, force_units, velocities_i, velocities_j, step_n,
                       data::CoulombReactionFieldExclusionData)
    vec_ij = vector(coord_i, coord_j, boundary)
    ke = data.coulomb_const / data.solute_dielectric
    if inter.self
        F0 = zero(vec_ij) * (ke / data.dist_cutoff^3)
        return SpecificForce2Atoms(F0, F0)
    end
    r2 = sum(abs2, vec_ij)
    krf, _ = reaction_field_constants(data.dist_cutoff, data.solvent_dielectric,
                                      data.solute_dielectric)
    qq = reaction_field_exclusion_qq(data, atom_i, atom_j)
    F = 2 * ke * qq * krf * (r2 <= data.dist_cutoff^2) * vec_ij
    return SpecificForce2Atoms(F, -F)
end

@inline function potential_energy(inter::CoulombReactionFieldExclusion, coord_i, coord_j,
                                  boundary, atom_i, atom_j, energy_units, velocities_i,
                                  velocities_j, step_n, data::CoulombReactionFieldExclusionData)
    vec_ij = vector(coord_i, coord_j, boundary)
    r2 = sum(abs2, vec_ij)
    ke = data.coulomb_const / data.solute_dielectric
    krf, crf = reaction_field_constants(data.dist_cutoff, data.solvent_dielectric,
                                        data.solute_dielectric)
    qq = reaction_field_exclusion_qq(data, atom_i, atom_j)
    pe = ke * qq * (krf * r2 - crf) * (r2 <= data.dist_cutoff^2)
    return inter.self ? pe / 2 : pe
end

"""
    reaction_field_exclusions(inter, eligible, special=nothing; self=true, scheduler,
                              array_type=Array)
    reaction_field_exclusions(inter, n_atoms, excluded_pairs; self=true, scheduler,
                              array_type=Array)

Set up an [`InteractionList2Atoms`](@ref) of [`CoulombReactionFieldExclusion`](@ref)
interactions that applies the reaction field correction to excluded pairs and,
if `self` is `true`, to each atom with itself.

`inter` is the reaction field pairwise interaction of the system, from which the parameters
are taken with [`Molly.CoulombReactionFieldExclusionData`](@ref).
It must have `use_neighbors=true`, since otherwise the excluded pairs are not removed from
the pairwise interaction.
The excluded pairs are either given as `excluded_pairs`, a list of atom index pairs, or are
the pairs that are not `eligible` or are `special`, the matrices given to the neighbor
finder.
Special (1-4) pairs get the correction with their full charges, whereas
`weight_special` only scales their direct Coulomb interaction.
"""
function reaction_field_exclusions(inter::AbstractReactionField, eligible, special=nothing;
                                   kwargs...)
    return reaction_field_exclusions(inter, size(eligible, 1),
                                     find_excluded_pairs(eligible, special); kwargs...)
end

function reaction_field_exclusions(inter::AbstractReactionField, n_atoms::Integer,
                                   excluded_pairs; self::Bool=true, array_type=Array,
                                   kwargs...)
    use_neighbors(inter) || throw(ArgumentError("the reaction field interaction must use " *
                    "the neighbor list, otherwise excluded pairs are counted twice"))
    data = CoulombReactionFieldExclusionData(inter; kwargs...)
    return reaction_field_exclusions(data, n_atoms, excluded_pairs, self, array_type)
end

function reaction_field_exclusions(data::CoulombReactionFieldExclusionData, n_atoms,
                                   excluded_pairs, self, array_type)
    pairs = normalize_pairs(excluded_pairs; n_atoms=n_atoms)
    self_inds = (self ? (1:n_atoms) : (1:0))
    is = Int32[[p[1] for p in pairs]; self_inds]
    js = Int32[[p[2] for p in pairs]; self_inds]
    inters = [fill(CoulombReactionFieldExclusion(false), length(pairs));
              fill(CoulombReactionFieldExclusion(true), length(self_inds))]
    return InteractionList2Atoms(
        to_device(is, array_type),
        to_device(js, array_type),
        to_device(inters, array_type),
        fill("", length(is)),
        data,
    )
end

has_self_exclusions(inter_list) = any(inter -> inter.self, from_device(inter_list.inters))

# The exclusion data hold a copy of the parameters, which must agree with the pairwise
#   reaction field
function check_reaction_field_parameters(pairwise_inters, specific_inter_lists)
    for il in values(specific_inter_lists)
        il isa InteractionList2Atoms && il.data isa CoulombReactionFieldExclusionData || continue
        for inter in values(pairwise_inters)
            inter isa AbstractReactionField || continue
            for field in (:dist_cutoff, :solvent_dielectric, :solute_dielectric, :coulomb_const)
                if !isapprox(getfield(il.data, field), getfield(inter, field))
                    throw(ArgumentError("the $field of the reaction field exclusions, " *
                            "$(getfield(il.data, field)), differs from that of the pairwise " *
                            "reaction field, $(getfield(inter, field))"))
                end
            end
        end
    end
    return nothing
end

@inline function inject_list_data(il::InteractionList2Atoms{<:Any, <:Any,
                                                           <:CoulombReactionFieldExclusionData},
                                  pis)
    rfs = filter(inter -> inter isa AbstractReactionField, pis)
    isempty(rfs) && return il
    inter = first(rfs)
    data = CoulombReactionFieldExclusionData(inter.dist_cutoff, inter.solvent_dielectric,
                                             inter.solute_dielectric, inter.coulomb_const,
                                             il.data.scheduler)
    return InteractionList2Atoms(il.is, il.js, il.inters, il.types, data)
end

@doc raw"""
    CoulombEwald(; dist_cutoff, error_tol=0.0005, ϵr=1, use_neighbors=true, weight_special=1,
                 coulomb_const=coulomb_const, approximate_erfc=true)

The short range Ewald electrostatic interaction between two atoms.

Should be used alongside the [`Ewald`](@ref) or [`PME`](@ref) general interaction,
which provide the long-range term, and the [`EwaldExclusion`](@ref) specific
interaction, which provides the exclusions for bonded atoms.
`dist_cutoff`, `error_tol`, `ϵr` and `coulomb_const` should match these interactions.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 \varepsilon_r} \frac{\mathrm{erfc}(\alpha r_{ij})}{r_{ij}}, \quad
\alpha = \frac{\sqrt{-\ln(2 \epsilon_{\mathrm{tol}})}}{r_c}
```
for ``r_{ij} \le r_c``, where ``r_c`` is `dist_cutoff`, the cutoff distance for short range
interactions, ``\epsilon_{\mathrm{tol}}`` is `error_tol` and ``\varepsilon_r`` is `ϵr`, the
relative permittivity.
Special pairs use the undamped Coulomb interaction scaled by `weight_special`.
`approximate_erfc` determines whether to use a fast approximation to the erfc function.
"""
struct CoulombEwald{T, D, E, W, C, A} <: PairwiseInteraction
    dist_cutoff::D
    error_tol::T
    ϵr::E
    use_neighbors::Bool
    weight_special::W
    coulomb_const::C
    α_ewald::A
    approximate_erfc::Bool
end

function CoulombEwald(; dist_cutoff, error_tol=0.0005, ϵr=1, use_neighbors=true,
                      weight_special=1, coulomb_const=coulomb_const, approximate_erfc=true)
    α_ewald = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    ϵr_T = typeof(ustrip(dist_cutoff))(ϵr)
    return CoulombEwald(dist_cutoff, error_tol, ϵr_T, use_neighbors, weight_special,
                        coulomb_const, α_ewald, approximate_erfc)
end

use_neighbors(inter::CoulombEwald) = inter.use_neighbors

required_atom_fields(::CoulombEwald) = (:charge,)

function Base.zero(coul::CoulombEwald{T, D, E, W, C, A}) where {T, D, E, W, C, A}
    return CoulombEwald(
        zero(D),
        zero(T),
        zero(E),
        coul.use_neighbors,
        zero(W),
        zero(C),
        zero(A),
        coul.approximate_erfc,
    )
end

function Base.:+(c1::CoulombEwald, c2::CoulombEwald)
    return CoulombEwald(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.error_tol + c2.error_tol,
        c1.ϵr + c2.ϵr,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.α_ewald + c2.α_ewald,
        c1.approximate_erfc,
    )
end

parameter_prefix(::CoulombEwald) = "inter_CO_"
parameter_fields(::Type{<:CoulombEwald}) =
    ((:dist_cutoff, "dist_cutoff"), (:weight_special, "weight_14"),
     (:coulomb_const, "coulomb_const"))

function calc_erfc(αr::T, exp_mαr2, approximate_erfc) where T
    if approximate_erfc
        # See the OpenMM source code, Abramowitz and Stegun 1964, and Hastings 1995
        t = inv(one(T) + T(0.3275911) * αr)
        return (T(0.254829592)+(T(-0.284496736)+(T(1.421413741)+
                                    (T(-1.453152027)+T(1.061405429)*t)*t)*t)*t)*t*exp_mαr2
    else
        return erfc(αr)
    end
end

@inline function force(inter::CoulombEwald{T},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where T
    r2 = sum(abs2, dr)
    ke, α_ewald = inter.coulomb_const / inter.ϵr, inter.α_ewald
    qi, qj = charge(atom_i), charge(atom_j)
    r = sqrt(r2)
    inv_r = inv(r)
    αr = α_ewald * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    f = ke * qi * qj * inv_r^3
    if special
        # Special interactions excluded from reciprocal calculation
        # so have a standard interaction
        return f * dr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return f * dr * (erfc_αr + 2 * αr * exp_mαr2 / sqrt(T(π))) * (r <= inter.dist_cutoff)
    end
end

@inline function potential_energy(inter::CoulombEwald,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r2 = sum(abs2, dr)
    ke, α_ewald = inter.coulomb_const / inter.ϵr, inter.α_ewald
    qi, qj = charge(atom_i), charge(atom_j)
    r = sqrt(r2)
    inv_r = inv(r)
    αr = α_ewald * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    pe = ke * qi * qj * inv_r
    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * erfc_αr * (r <= inter.dist_cutoff)
    end
end

@doc raw"""
    CoulombEwaldScaled(; dist_cutoff, error_tol=0.0005, use_neighbors=true,
                       λ_mixing=MinimumMixing(), scheduler=Molly.DefaultLambdaScheduler(),
                       weight_special=1, coulomb_const=coulomb_const, approximate_erfc=true)

The short-range Ewald electrostatic interaction between two atoms scaled by λ without a
soft core, the λ version of [`CoulombEwald`](@ref).

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each
[`CoulombEwald`](@ref) to this type with `Csoftcore=:scaled`.
The potential energy is that of [`CoulombEwald`](@ref) multiplied by ``\lambda``, the
electrostatic coupling of the pair, which `scheduler`, a [`Molly.LambdaScheduler`](@ref),
gives from the `λ` of the atoms mixed with `λ_mixing` and their alchemical roles.
Should be used alongside [`Ewald`](@ref) or [`PME`](@ref) configured with the same
electrostatic scheduler.
"""
struct CoulombEwaldScaled{T, D, LM, SCH, W, C, A} <: PairwiseInteraction
    dist_cutoff::D
    error_tol::T
    use_neighbors::Bool
    λ_mixing::LM
    scheduler::SCH
    weight_special::W
    coulomb_const::C
    α_ewald::A
    approximate_erfc::Bool
end

function CoulombEwaldScaled(; dist_cutoff, error_tol=0.0005, use_neighbors=true, λ_mixing=MinimumMixing(),
                            scheduler=DefaultLambdaScheduler(), weight_special=1,
                            coulomb_const=coulomb_const, approximate_erfc=true)
    α_ewald = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    return CoulombEwaldScaled(dist_cutoff, error_tol, use_neighbors, λ_mixing, scheduler,
                              weight_special, coulomb_const, α_ewald, approximate_erfc)
end

use_neighbors(inter::CoulombEwaldScaled) = inter.use_neighbors

required_atom_fields(::CoulombEwaldScaled) = (:charge, :λ, :alch_role)

function Base.zero(coul::CoulombEwaldScaled{T, D, LM, SCH, W, C, A}) where {T, D, LM, SCH, W, C, A}
    return CoulombEwaldScaled(
        zero(D),
        zero(T),
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(C),
        zero(A),
        coul.approximate_erfc,
    )
end

function Base.:+(c1::CoulombEwaldScaled, c2::CoulombEwaldScaled)
    return CoulombEwaldScaled(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.error_tol + c2.error_tol,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.α_ewald + c2.α_ewald,
        c1.approximate_erfc,
    )
end

parameter_prefix(::CoulombEwaldScaled) = "inter_CO_"
parameter_fields(::Type{<:CoulombEwaldScaled}) =
    ((:dist_cutoff, "dist_cutoff"), (:weight_special, "weight_14"),
     (:coulomb_const, "coulomb_const"))


# The λ versions of CoulombEwald have no ϵr, so it is folded into the Coulomb constant
function to_lambda_function(inter::CoulombEwald, ::ScaledSoftCore; α=0.3,
                                λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler(), 
                                float_type=Float32, args...)
    return CoulombEwaldScaled(;dist_cutoff=inter.dist_cutoff, error_tol=inter.error_tol, 
                                use_neighbors=inter.use_neighbors, λ_mixing=λ_mixing, 
                                scheduler=scheduler, weight_special=inter.weight_special, 
                                coulomb_const=inter.coulomb_const / inter.ϵr,
                                approximate_erfc=inter.approximate_erfc)
end

@inline function force(inter::CoulombEwaldScaled{T},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where T
    ke, α_ewald = inter.coulomb_const, inter.α_ewald
    qq = ewald_pair_qq(inter.scheduler, atom_i, atom_j, Val(T); special=special)

    if iszero_value(qq)
        return zero_pairwise_force(dr, force_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    inv_r = inv(r)
    αr = α_ewald * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    f = ke * qq * inv_r^3
    if special
        return f * dr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return f * dr * (erfc_αr + 2 * αr * exp_mαr2 / sqrt(T(π))) * (r <= inter.dist_cutoff)
    end
end

@inline function potential_energy(inter::CoulombEwaldScaled{T},
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where T
    ke, α_ewald = inter.coulomb_const, inter.α_ewald
    qq = ewald_pair_qq(inter.scheduler, atom_i, atom_j, Val(T); special=special)

    if iszero_value(qq)
        return zero_pairwise_energy(dr, energy_units)
    end

    r2 = sum(abs2, dr)
    r = sqrt(r2)
    if iszero_value(r)
        return zero_pairwise_energy(dr, energy_units)
    end

    inv_r = inv(r)
    αr = α_ewald * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    pe = ke * qq * inv_r
    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * erfc_αr * (r <= inter.dist_cutoff)
    end
end

@inline function lambda_pair(inter, atom_i, atom_j)
    T = typeof(ustrip(inter.coulomb_const))
    λ_glob = T(λ_mixing(inter.λ_mixing, (atom_i, atom_j)))
    pair_role = mix_roles(inter.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λR, λ_params = scale_elec_dual(inter.scheduler, λ_glob, pair_role)
    return pair_role, λ, λR, λ_params
end

@inline function softcore_pair_elec_lambda(inter, atom_i, atom_j, special)
    T = typeof(ustrip(inter.coulomb_const))
    pair_role, λ, λR, λ_params = lambda_pair(inter, atom_i, atom_j)
    if !(atom_i.charge isa Tuple)
        qij = atom_i.charge * atom_j.charge
    elseif !(inter.scheduler.Cindividual) || (!(inter.scheduler.Cspecial) && special)
        qij = atom_i.charge .* atom_j.charge
        qij = params_mixing(λ_params, qij)
    else
        qij = effective_charge(inter.scheduler, atom_i, Val(T)) *
              effective_charge(inter.scheduler, atom_j, Val(T))
    end
    return pair_role, λ, λR, λ_params, qij
end

@inline function softcore_ewald_screen(inter, r)
    T = typeof(ustrip(inter.coulomb_const))
    αr = inter.α_ewald * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    force_screen = 2 * inter.α_ewald * exp_mαr2 / sqrt(T(π))
    return erfc_αr, force_screen
end

@doc raw"""
    CoulombSoftCoreBeutlerEwald(; dist_cutoff, error_tol=0.0005, α=0.3,
                                use_neighbors=true, σ_mixing=LorentzMixing(),
                                ϵ_mixing=GeometricMixing(), λ_mixing=MinimumMixing(),
                                scheduler=Molly.DefaultLambdaScheduler(), weight_special=1,
                                coulomb_const=coulomb_const, approximate_erfc=true)

The short-range Ewald electrostatic interaction between two atoms with the
Beutler soft-core distance modification.

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each
[`CoulombEwald`](@ref) to this type with `Csoftcore=:beutler`.
The soft core and ``\lambda``, the electrostatic coupling of the pair, are as for
[`CoulombSoftCoreBeutler`](@ref).
Should be used alongside [`Ewald`](@ref) or [`PME`](@ref) configured with the same
electrostatic scheduler. Special pairs are excluded from the reciprocal-space
calculation, so their interaction uses the undamped soft-core form.
"""
struct CoulombSoftCoreBeutlerEwald{ET, D, A, SM, EM, LM, SCH, W, C, EA} <: PairwiseInteraction
    dist_cutoff::D
    error_tol::ET
    α::A
    use_neighbors::Bool
    σ_mixing::SM
    ϵ_mixing::EM
    λ_mixing::LM
    scheduler::SCH
    weight_special::W
    coulomb_const::C
    α_ewald::EA
    approximate_erfc::Bool
end

function CoulombSoftCoreBeutlerEwald(; dist_cutoff, error_tol=0.0005, α=0.3,
                                     use_neighbors=true, σ_mixing=LorentzMixing(),
                                     ϵ_mixing=GeometricMixing(), λ_mixing=MinimumMixing(),
                                     scheduler=DefaultLambdaScheduler(), weight_special=1,
                                     coulomb_const=coulomb_const, approximate_erfc=true)
    T = typeof(ustrip(dist_cutoff))
    α_ewald = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    return CoulombSoftCoreBeutlerEwald(
        dist_cutoff,
        error_tol,
        α,
        use_neighbors,
        σ_mixing,
        ϵ_mixing,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
        α_ewald,
        approximate_erfc,
    )
end

use_neighbors(inter::CoulombSoftCoreBeutlerEwald) = inter.use_neighbors

required_atom_fields(inter::CoulombSoftCoreBeutlerEwald) = (:charge, :σ, :ϵ, :λ, :alch_role,
            mixing_atom_fields(inter.σ_mixing)..., mixing_atom_fields(inter.ϵ_mixing)...,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(coul::CoulombSoftCoreBeutlerEwald{ET, D, A, SM, EM, LM, SCH, W, C, EA}) where {ET, D, A, SM, EM, LM, SCH, W, C, EA}
    return CoulombSoftCoreBeutlerEwald(
        zero(D),
        zero(ET),
        zero(A),
        coul.use_neighbors,
        coul.σ_mixing,
        coul.ϵ_mixing,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(C),
        zero(EA),
        coul.approximate_erfc,
    )
end

function Base.:+(c1::CoulombSoftCoreBeutlerEwald, c2::CoulombSoftCoreBeutlerEwald)
    return CoulombSoftCoreBeutlerEwald(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.error_tol + c2.error_tol,
        c1.α + c2.α,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.ϵ_mixing,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.α_ewald + c2.α_ewald,
        c1.approximate_erfc,
    )
end

parameter_prefix(::CoulombSoftCoreBeutlerEwald) = "inter_CO_"
parameter_fields(::Type{<:CoulombSoftCoreBeutlerEwald}) =
    ((:dist_cutoff, "dist_cutoff"), (:α, "α"), (:weight_special, "weight_14"),
     (:coulomb_const, "coulomb_const"))


function to_lambda_function(inter::CoulombEwald, ::BeutlerSoftCore; α=0.3, σ_mixing=LorentzMixing(),
                                     ϵ_mixing=GeometricMixing(), λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler(), float_type=Float32, args...)
    return CoulombSoftCoreBeutlerEwald(dist_cutoff=inter.dist_cutoff, error_tol=inter.error_tol, α=float_type(α), 
                                            use_neighbors=inter.use_neighbors, σ_mixing=σ_mixing, 
                                            ϵ_mixing=ϵ_mixing, λ_mixing=λ_mixing, scheduler=scheduler, 
                                            weight_special=inter.weight_special, coulomb_const=inter.coulomb_const / inter.ϵr,
                                            approximate_erfc=inter.approximate_erfc)
end

@inline function force(inter::CoulombSoftCoreBeutlerEwald,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    pair_role, λ, λR, λ_params = lambda_pair(inter, atom_i, atom_j)
    qq = ewald_pair_qq(inter.scheduler, atom_i, atom_j,
                       Val(typeof(ustrip(inter.coulomb_const))); special=special)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if iszero_value(qq)
        return zero_pairwise_force(dr, force_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    if λ >= 1 && λR >= 1
        pe_soft = inter.coulomb_const * (qq / r)
        f_soft = inter.coulomb_const * (qq / r^2)
    else
        term = inter.α * (1 - (λ*λR)) * σ_mixing(inter.σ_mixing, atom_i, atom_j, λ_params, pair_role)^6 + r^6
        pe_soft = inter.coulomb_const * (qq / sqrt(cbrt(term)))
        f_soft = inter.coulomb_const * (qq / (term * sqrt(cbrt(term)))) * r^5
    end

    if special
        return radial_force_vector(f_soft, r, dr, force_units) * inter.weight_special * (r <= inter.dist_cutoff)
    end

    erfc_αr, force_screen = softcore_ewald_screen(inter, r)
    f = (f_soft * erfc_αr) + (pe_soft * force_screen)
    return radial_force_vector(f, r, dr, force_units) * (r <= inter.dist_cutoff)
end

@inline function potential_energy(inter::CoulombSoftCoreBeutlerEwald,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    pair_role, λ, λR, λ_params = lambda_pair(inter, atom_i, atom_j)
    qq = ewald_pair_qq(inter.scheduler, atom_i, atom_j,
                       Val(typeof(ustrip(inter.coulomb_const))); special=special)

    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end

    if iszero_value(qq)
        return zero_pairwise_energy(dr, energy_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        if λ >= 1 && λR >= 1
            pe_soft = zero_pairwise_energy(dr, energy_units)
        else
            σ6_fac = inter.α * (1 - (λ*λR)) * σ_mixing(inter.σ_mixing, atom_i, atom_j, λ_params, pair_role)^6
            pe_soft = overlap_pe_coulomb_softcore_beutler(
                dr,
                energy_units,
                inter.coulomb_const,
                qq,
                one(λ),
                σ6_fac,
            )
        end
    elseif λ >= 1 && λR >= 1
        pe_soft = inter.coulomb_const * (qq / r)
    else
        σ6_fac = inter.α * (1 - (λ*λR)) * σ_mixing(inter.σ_mixing, atom_i, atom_j, λ_params, pair_role)^6
        pe_soft = inter.coulomb_const * (qq / sqrt(cbrt(σ6_fac + r^6)))
    end

    if special
        return pe_soft * inter.weight_special * (r <= inter.dist_cutoff)
    end

    erfc_αr, _ = softcore_ewald_screen(inter, r)
    return pe_soft * erfc_αr * (r <= inter.dist_cutoff)
end

@doc raw"""
    CoulombSoftCoreGapsysEwald(; dist_cutoff, error_tol=0.0005, α=0.3, σQ=1.0u"nm",
                               use_neighbors=true, λ_mixing=MinimumMixing(),
                               scheduler=Molly.DefaultLambdaScheduler(), weight_special=1,
                               coulomb_const=coulomb_const, approximate_erfc=true)

The short-range Ewald electrostatic interaction between two atoms with the
Gapsys soft-core polynomial near the origin.

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each
[`CoulombEwald`](@ref) to this type with `Csoftcore=:gapsys`.
The soft core and ``\lambda``, the electrostatic coupling of the pair, are as for
[`CoulombSoftCoreGapsys`](@ref).
Should be used alongside [`Ewald`](@ref) or [`PME`](@ref) configured with the same
electrostatic scheduler. Special pairs are excluded from the reciprocal-space
calculation, so their interaction uses the undamped soft-core form.
"""
struct CoulombSoftCoreGapsysEwald{ET, D, A, SQ, LM, SCH, W, C, EA} <: PairwiseInteraction
    dist_cutoff::D
    error_tol::ET
    α::A
    σQ::SQ
    use_neighbors::Bool
    λ_mixing::LM
    scheduler::SCH
    weight_special::W
    coulomb_const::C
    α_ewald::EA
    approximate_erfc::Bool
end

function CoulombSoftCoreGapsysEwald(; dist_cutoff, error_tol=0.0005, α=0.3,
                                    σQ=1.0u"nm", use_neighbors=true,
                                    λ_mixing=MinimumMixing(),
                                    scheduler=DefaultLambdaScheduler(),
                                    weight_special=1, coulomb_const=coulomb_const,
                                    approximate_erfc=true)
    T = typeof(ustrip(dist_cutoff))
    α_ewald = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    return CoulombSoftCoreGapsysEwald(
        dist_cutoff,
        error_tol,
        α,
        σQ,
        use_neighbors,
        λ_mixing,
        scheduler,
        weight_special,
        coulomb_const,
        α_ewald,
        approximate_erfc,
    )
end

use_neighbors(inter::CoulombSoftCoreGapsysEwald) = inter.use_neighbors

required_atom_fields(inter::CoulombSoftCoreGapsysEwald) = (:charge, :λ, :alch_role,
            mixing_atom_fields(inter.λ_mixing)...)

function Base.zero(coul::CoulombSoftCoreGapsysEwald{ET, D, A, SQ, LM, SCH, W, C, EA}) where {ET, D, A, SQ, LM, SCH, W, C, EA}
    return CoulombSoftCoreGapsysEwald(
        zero(D),
        zero(ET),
        zero(A),
        coul.σQ,
        coul.use_neighbors,
        coul.λ_mixing,
        coul.scheduler,
        zero(W),
        zero(C),
        zero(EA),
        coul.approximate_erfc,
    )
end

function Base.:+(c1::CoulombSoftCoreGapsysEwald, c2::CoulombSoftCoreGapsysEwald)
    return CoulombSoftCoreGapsysEwald(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.error_tol + c2.error_tol,
        c1.α + c2.α,
        c1.σQ + c2.σQ,
        c1.use_neighbors,
        c1.λ_mixing,
        c1.scheduler,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.α_ewald + c2.α_ewald,
        c1.approximate_erfc,
    )
end

parameter_prefix(::CoulombSoftCoreGapsysEwald) = "inter_CO_"
parameter_fields(::Type{<:CoulombSoftCoreGapsysEwald}) =
    ((:dist_cutoff, "dist_cutoff"), (:α, "α"), (:σQ, "σQ"), (:weight_special, "weight_14"),
     (:coulomb_const, "coulomb_const"))


function to_lambda_function(inter::CoulombEwald, ::GapsysSoftCore; α=0.3, σQ=1.0u"nm", λ_mixing=MinimumMixing(), 
                                scheduler=DefaultLambdaScheduler(), float_type=Float32, args...)
    return CoulombSoftCoreGapsysEwald(dist_cutoff=inter.dist_cutoff, error_tol=inter.error_tol, α=float_type(α), 
                                        σQ=softcore_σQ(σQ, inter.coulomb_const, float_type), use_neighbors=inter.use_neighbors, 
                                        λ_mixing=λ_mixing, scheduler=scheduler, 
                                        weight_special=inter.weight_special, coulomb_const=inter.coulomb_const / inter.ϵr,
                                        approximate_erfc=inter.approximate_erfc)
end

@inline function force(inter::CoulombSoftCoreGapsysEwald,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    # λ and λR still shape the soft-core distance, but the charge magnitude has to follow
    # the reciprocal sum that the scheduler selects
    pair_role, λ, λR, λ_params = lambda_pair(inter, atom_i, atom_j)
    qq = ewald_pair_qq(inter.scheduler, atom_i, atom_j,
                       Val(typeof(ustrip(inter.coulomb_const))); special=special)

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if iszero_value(qq)
        return zero_pairwise_force(dr, force_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    if λ >= 1 && λR >= 1
        pe_soft = inter.coulomb_const * (qq / r)
        f_soft = inter.coulomb_const * (qq / r^2)
    else
        R = inter.α * sqrt(cbrt(1 - (λ*λR))) * (oneunit(r) + inter.σQ * abs(qq))
        if !(r < R)
            pe_soft = inter.coulomb_const * (qq / r)
            f_soft = inter.coulomb_const * (qq / r^2)
        else
            pe_soft = inter.coulomb_const * (((qq / R^3) * r^2) - (((3 * qq) / R^2) * r) +
                      ((3 * qq) / R))
            f_soft = inter.coulomb_const * (-(((2 * qq) / R^3) * r) + ((3 * qq) / R^2))
        end
    end

    if special
        return radial_force_vector(f_soft, r, dr, force_units) * inter.weight_special * (r <= inter.dist_cutoff)
    end

    erfc_αr, force_screen = softcore_ewald_screen(inter, r)
    f = (f_soft * erfc_αr) + (pe_soft * force_screen)
    return radial_force_vector(f, r, dr, force_units) * (r <= inter.dist_cutoff)
end

@inline function potential_energy(inter::CoulombSoftCoreGapsysEwald,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    # λ and λR still shape the soft-core distance, but the charge magnitude has to follow
    # the reciprocal sum that the scheduler selects
    pair_role, λ, λR, λ_params = lambda_pair(inter, atom_i, atom_j)
    qq = ewald_pair_qq(inter.scheduler, atom_i, atom_j,
                       Val(typeof(ustrip(inter.coulomb_const))); special=special)

    if λ <= 0
        return zero_pairwise_energy(dr, energy_units)
    end

    if iszero_value(qq)
        return zero_pairwise_energy(dr, energy_units)
    end

    r = sqrt(sum(abs2, dr))
    if iszero_value(r)
        if λ >= 1 && λR >= 1
            pe_soft = zero_pairwise_energy(dr, energy_units)
        else
            R = inter.α * sqrt(cbrt(1 - (λ*λR))) * (oneunit(r) + inter.σQ * abs(qq))
            pe_soft = overlap_pe_coulomb_softcore_gapsys(
                dr,
                energy_units,
                inter.coulomb_const,
                qq,
                one(λ),
                R,
            )
        end
    elseif λ >= 1 && λR >= 1
        pe_soft = inter.coulomb_const * (qq / r)
    else
        R = inter.α * sqrt(cbrt(1 - (λ*λR))) * (oneunit(r) + inter.σQ * abs(qq))
        if !(r < R)
            pe_soft = inter.coulomb_const * (qq / r)
        else
            pe_soft = inter.coulomb_const * (((qq / R^3) * r^2) - (((3 * qq) / R^2) * r) +
                      ((3 * qq) / R))
        end
    end

    if special
        return pe_soft * inter.weight_special * (r <= inter.dist_cutoff)
    end

    erfc_αr, _ = softcore_ewald_screen(inter, r)
    return pe_soft * erfc_αr * (r <= inter.dist_cutoff)
end

@doc raw"""
    Yukawa(; cutoff, use_neighbors, weight_special, coulomb_const, kappa)

The Yukawa electrostatic interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}} \exp(-\kappa r_{ij})
```
and the force on each atom by
```math
\vec{F}_i = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}^2} \exp(-\kappa r_{ij})\left(\kappa r_{ij} + 1\right) \frac{\vec{r}_{ij}}{r_{ij}}
```
where ``\kappa`` is `kappa`, the inverse of the screening length.
"""
@kwdef struct Yukawa{C, W, T, K} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = true
    weight_special::W = 1
    coulomb_const::T = coulomb_const
    kappa::K = 1.0*u"nm^-1"
end

use_neighbors(inter::Yukawa) = inter.use_neighbors

required_atom_fields(::Yukawa) = (:charge,)

function Base.zero(yukawa::Yukawa{C, W, T, K}) where {C, W, T, K}
    return Yukawa(
        yukawa.cutoff,
        yukawa.use_neighbors,
        zero(W),
        zero(T),
        zero(K),
    )
end

function Base.:+(c1::Yukawa, c2::Yukawa)
    return Yukawa(
        c1.cutoff,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.kappa + c2.kappa,
    )
end

parameter_prefix(::Yukawa) = "inter_YU_"
parameter_fields(::Type{<:Yukawa}) =
    ((:weight_special, "weight_14"), (:coulomb_const, "coulomb_const"), (:kappa, "kappa"))

@inline function force(inter::Yukawa,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r = sqrt(sum(abs2, dr))
    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = charge(atom_i), charge(atom_j)
    kappa = inter.kappa
    params = (coulomb_const, qi, qj, kappa)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::Yukawa, r, (coulomb_const, qi, qj, kappa))
    return (coulomb_const * qi * qj) * exp(-kappa * r) * (kappa * r + 1) / r^2
end

@inline function potential_energy(inter::Yukawa,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special::Bool=false,
                                  args...)
    r = sqrt(sum(abs2, dr))
    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = charge(atom_i), charge(atom_j)
    params = (coulomb_const, qi, qj, inter.kappa)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::Yukawa, r, (coulomb_const, qi, qj, kappa))
    return (coulomb_const * qi * qj) * inv(r) * exp(-kappa * r)
end
