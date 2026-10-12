export HarmonicAngle

@public HarmonicAngleλ

@doc raw"""
    HarmonicAngle(; k, θ0)

A harmonic bond angle between three atoms.

`θ0` is in radians.
The second atom is the middle atom.
The potential energy is defined as
```math
V(\theta) = \frac{1}{2} k (\theta - \theta_0)^2
```
"""
@kwdef struct HarmonicAngle{K, D}
    k::K
    θ0::D
end

Base.zero(::Type{HarmonicAngle{K, D}}) where {K, D} = HarmonicAngle(k=zero(K), θ0=zero(D))
Base.zero(a::HarmonicAngle) = zero(typeof(a))

Base.:+(a1::HarmonicAngle, a2::HarmonicAngle) = HarmonicAngle(k=(a1.k + a2.k), θ0=(a1.θ0 + a2.θ0))

parameter_prefix(::HarmonicAngle, inter_type) = "inter_HA_$(inter_type)_"
parameter_fields(::Type{<:HarmonicAngle}) = ((:k, "k"), (:θ0, "θ0"))


@inline function force(a::HarmonicAngle, coords_i, coords_j, coords_k, boundary, args...)
    # In 2D we use then eliminate the cross product
    ba = vector_pad3D(coords_j, coords_i, boundary)
    bc = vector_pad3D(coords_j, coords_k, boundary)
    cross_ba_bc = ba × bc
    if iszero_value(cross_ba_bc)
        zf = zero(a.k ./ trim3D(ba, boundary))
        return SpecificForce3Atoms(zf, zf, zf)
    end
    pa = normalize(trim3D( ba × cross_ba_bc, boundary))
    pc = normalize(trim3D(-bc × cross_ba_bc, boundary))
    angle_term = -a.k * (acos_bound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.θ0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return SpecificForce3Atoms(fa, fb, fc)
end

@inline function potential_energy(a::HarmonicAngle, coords_i, coords_j,
                                  coords_k, boundary, args...)
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)
    return (a.k / 2) * (θ - a.θ0) ^ 2
end

@doc raw"""
    HarmonicAngleλ(; k, θ0, λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())

A harmonic bond angle between three atoms in an alchemical system, the λ version of
[`HarmonicAngle`](@ref).

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each [`HarmonicAngle`](@ref)
to this type.
The potential energy is defined as
```math
V(\theta) = \frac{w}{2} k (\theta - \theta_0)^2
```
where `θ0` and the atom order are as for [`HarmonicAngle`](@ref), and the weight ``w`` and
the parameters depend on the alchemical roles of the atoms and on ``\lambda``, the `λ` of the
atoms mixed with `λ_mixing`.
`scheduler` is the [`LambdaScheduler`](@ref), which selects dual or single topology.

In dual topology `k` and `θ0` are single values.
``w = 1`` if the angle involves an atom unique to one state or only environment atoms,
otherwise ``w = 1 - \lambda`` if it involves a core atom with the parameters of state A and
``w = \lambda`` if it involves a core atom with the parameters of state B.

In single topology ``w = 1`` and `k` and `θ0` are tuples of the values in states A and B.
Each parameter ``p`` is ``(1 - \lambda) p^A + \lambda p^B`` if the angle involves a core or
unique atom and ``p^A`` otherwise.
"""
@kwdef struct HarmonicAngleλ{K, D, LM, SCH} <: AlchemicalBondedInteraction
    k::K
    θ0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

Base.zero(a::HarmonicAngleλ) = HarmonicAngleλ(k=zero.(a.k), θ0=zero.(a.θ0), λ_mixing=a.λ_mixing,
                                              scheduler=a.scheduler)

Base.:+(a1::HarmonicAngleλ, a2::HarmonicAngleλ) = HarmonicAngleλ(k=(a1.k .+ a2.k), θ0=(a1.θ0 .+ a2.θ0),
                                                                 λ_mixing=a1.λ_mixing, scheduler=a1.scheduler)

function Base.show(io::IO, x::HarmonicAngleλ)
    println(io, "HarmonicAngleλ: (k: $(x.k)) - θ0: $(x.θ0) - λ_mixing: $(x.λ_mixing) - scheduler: $(x.scheduler)")
end

function to_lambda_function(inter::HarmonicAngle; λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return HarmonicAngleλ(k=inter.k, θ0=inter.θ0, λ_mixing=λ_mixing, scheduler=scheduler)
end

function to_lambda_function_single(interA::HarmonicAngle, interB::Nothing; 
                                   λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    k_A  = interA.k
    k_B  = interA.k 
    θ0_A = interA.θ0
    θ0_B = interA.θ0
    
    return HarmonicAngleλ(k=(k_A, k_B), θ0=(θ0_A, θ0_B), λ_mixing=λ_mixing, scheduler=scheduler)
end

function to_lambda_function_single(interA::Nothing, interB::HarmonicAngle; 
                                   λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    k_A  = interB.k 
    k_B  = interB.k
    θ0_A = interB.θ0 
    θ0_B = interB.θ0
    
    return HarmonicAngleλ(k=(k_A, k_B), θ0=(θ0_A, θ0_B), λ_mixing=λ_mixing, scheduler=scheduler)
end

function update_lambda_function(existing_lambda::HarmonicAngleλ, interB::HarmonicAngle)
    return HarmonicAngleλ(k=(existing_lambda.k[1], interB.k), 
                          θ0=(existing_lambda.θ0[1], interB.θ0), 
                          λ_mixing=existing_lambda.λ_mixing, 
                          scheduler=existing_lambda.scheduler)
end

plain_interaction(a::HarmonicAngleλ, λ_params) = HarmonicAngle(k=params_mixing(λ_params, a.k), θ0=params_mixing(λ_params, a.θ0))

@inline function force(a::HarmonicAngleλ, coords_i, coords_j, coords_k, boundary, atom_i, atom_j,
                       atom_k, args...)
    λ, λ_params = bonded_lambda(a, (atom_i, atom_j, atom_k))
    return λ * force(plain_interaction(a, λ_params), coords_i, coords_j, coords_k, boundary)
end

@inline function potential_energy(a::HarmonicAngleλ, coords_i, coords_j, coords_k, boundary, atom_i,
                                  atom_j, atom_k, args...)
    λ, λ_params = bonded_lambda(a, (atom_i, atom_j, atom_k))
    return λ * potential_energy(plain_interaction(a, λ_params), coords_i, coords_j, coords_k,
                                boundary)
end