export CosineAngle

@doc raw"""
    CosineAngle(; k, θ0)

A cosine bond angle between three atoms.

`θ0` is in radians.
The potential energy is defined as
```math
V(\theta) = k(1 + \cos(\theta - \theta_0))
```
"""
@kwdef struct CosineAngle{K, D}
    k::K
    θ0::D
end

Base.zero(::Type{CosineAngle{K, D}}) where {K, D} = CosineAngle(k=zero(K), θ0=zero(D))
Base.zero(a::CosineAngle) = zero(typeof(a))

Base.:+(a1::CosineAngle, a2::CosineAngle) = CosineAngle(k=(a1.k + a2.k), θ0=(a1.θ0 + a2.θ0))

parameter_prefix(::CosineAngle, inter_type) = "inter_CA_$(inter_type)_"
parameter_fields(::Type{<:CosineAngle}) = ((:k, "k"), (:θ0, "θ0"))


@inline function force(a::CosineAngle, coords_i, coords_j, coords_k, boundary, args...)
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
    θ = bond_angle(ba, bc)
    angle_term = a.k * sin(θ - a.θ0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return SpecificForce3Atoms(fa, fb, fc)
end

@inline function potential_energy(a::CosineAngle, coords_i, coords_j,
                                  coords_k, boundary, args...)
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)
    return a.k * (1 + cos(θ - a.θ0))
end

# λ version of `CosineAngle` for alchemical systems, built by `to_lambda_function`.
@kwdef struct CosineAngleλ{K, D, LM, SCH} <: AlchemicalBondedInteraction
    k::K
    θ0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

Base.zero(a::CosineAngleλ) = CosineAngleλ(k=zero.(a.k), θ0=zero.(a.θ0), λ_mixing=a.λ_mixing,
                                          scheduler=a.scheduler)

Base.:+(a1::CosineAngleλ, a2::CosineAngleλ) = CosineAngleλ(k=(a1.k .+ a2.k), θ0=(a1.θ0 .+ a2.θ0),
                                                           λ_mixing=a1.λ_mixing,
                                                           scheduler=a1.scheduler)

function to_lambda_function(inter::CosineAngle; λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return CosineAngleλ(k=inter.k, θ0=inter.θ0, λ_mixing=λ_mixing, scheduler=scheduler)
end

plain_interaction(a::CosineAngleλ, λ_params) = CosineAngle(k=params_mixing(λ_params, a.k), θ0=params_mixing(λ_params, a.θ0))

@inline function force(a::CosineAngleλ, coords_i, coords_j, coords_k, boundary, atom_i, atom_j,
                       atom_k, args...)
    λ, λ_params = bonded_lambda(a, (atom_i, atom_j, atom_k))
    return λ * force(plain_interaction(a, λ_params), coords_i, coords_j, coords_k, boundary)
end

@inline function potential_energy(a::CosineAngleλ, coords_i, coords_j, coords_k, boundary, atom_i,
                                  atom_j, atom_k, args...)
    λ, λ_params = bonded_lambda(a, (atom_i, atom_j, atom_k))
    return λ * potential_energy(plain_interaction(a, λ_params), coords_i, coords_j, coords_k,
                                boundary)
end
