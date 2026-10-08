export HarmonicBond

@doc raw"""
    HarmonicBond(; k, r0)

A harmonic bond between two atoms.

The potential energy is defined as
```math
V(r) = \frac{1}{2} k (r - r_0)^2
```
"""
@kwdef struct HarmonicBond{K, D}
    k::K
    r0::D
end

Base.zero(::Type{HarmonicBond{K, D}}) where {K, D} = HarmonicBond(k=zero(K), r0=zero(D))
Base.zero(b::HarmonicBond) = zero(typeof(b))

Base.:+(b1::HarmonicBond, b2::HarmonicBond) = HarmonicBond(k=(b1.k + b2.k), r0=(b1.r0 + b2.r0))

parameter_prefix(::HarmonicBond, inter_type) = "inter_HB_$(inter_type)_"
parameter_fields(::Type{<:HarmonicBond}) = ((:k, "k"), (:r0, "r0"))


@inline function force(b::HarmonicBond, coord_i, coord_j, boundary, args...)
    ab = vector(coord_i, coord_j, boundary)
    c = b.k * (norm(ab) - b.r0)
    f = c * normalize(ab)
    return SpecificForce2Atoms(f, -f)
end

@inline function potential_energy(b::HarmonicBond, coord_i, coord_j, boundary, args...)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    return (b.k / 2) * (r - b.r0) ^ 2
end

# λ version of `HarmonicBond` for alchemical systems, built by `to_lambda_function`.
@kwdef struct HarmonicBondλ{K, D, LM, SCH} <: AlchemicalBondedInteraction
    k::K
    r0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

Base.zero(b::HarmonicBondλ) = HarmonicBondλ(k=zero.(b.k), r0=zero.(b.r0), λ_mixing=b.λ_mixing,
                                            scheduler=b.scheduler)

Base.:+(b1::HarmonicBondλ, b2::HarmonicBondλ) = HarmonicBondλ(k=(b1.k .+ b2.k), r0=(b1.r0 .+ b2.r0),
                                                              λ_mixing=b1.λ_mixing, scheduler=b1.scheduler)

function to_lambda_function(inter::HarmonicBond; λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return HarmonicBondλ(k=inter.k, r0=inter.r0, λ_mixing=λ_mixing, scheduler=scheduler)
end

function to_lambda_function_single(interA::HarmonicBond, interB::Nothing; 
                                   λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    k_A  = interA.k
    k_B  = interA.k
    r0_A = interA.r0
    r0_B = interA.r0
    
    return HarmonicBondλ(k=(k_A, k_B), r0=(r0_A, r0_B), λ_mixing=λ_mixing, scheduler=scheduler)
end


function to_lambda_function_single(interA::Nothing, interB::HarmonicBond; 
                                   λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    k_A  = interB.k
    k_B  = interB.k
    r0_A = interB.r0
    r0_B = interB.r0
    
    return HarmonicBondλ(k=(k_A, k_B), r0=(r0_A, r0_B), λ_mixing=λ_mixing, scheduler=scheduler)
end


function update_lambda_function(existing_lambda::HarmonicBondλ, interB::HarmonicBond)
    return HarmonicBondλ(k=(existing_lambda.k[1], interB.k), 
                         r0=(existing_lambda.r0[1], interB.r0), 
                         λ_mixing=existing_lambda.λ_mixing, 
                         scheduler=existing_lambda.scheduler)
end

plain_interaction(b::HarmonicBondλ, λ_params) = HarmonicBond(k=params_mixing(λ_params, b.k), r0=params_mixing(λ_params, b.r0))

@inline function force(b::HarmonicBondλ, coord_i, coord_j, boundary, atom_i, atom_j, args...)
    λ, λ_params = bonded_lambda(b, (atom_i, atom_j))
    return λ * force(plain_interaction(b, λ_params), coord_i, coord_j, boundary)
end

@inline function potential_energy(b::HarmonicBondλ, coord_i, coord_j, boundary, atom_i, atom_j,
                                  args...)
    λ, λ_params = bonded_lambda(b, (atom_i, atom_j))
    return λ * potential_energy(plain_interaction(b, λ_params), coord_i, coord_j, boundary)
end