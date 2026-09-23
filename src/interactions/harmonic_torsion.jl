export HarmonicTorsion

@doc raw"""
    HarmonicTorsion(; k, θ0)

A harmonic torsion angle between four atoms, often used for improper torsions.

`θ0` is in radians.
The potential energy is defined as
```math
V(\theta) = k (\theta - \theta_0)^2
```
where `θ` is the angle between the planes defined by atoms (i, j, k) and (j, k, l).

Only compatible with 3D systems.
"""
@kwdef struct HarmonicTorsion{K, D}
    k::K
    θ0::D
end

Base.zero(::Type{HarmonicTorsion{K, D}}) where {K, D} = HarmonicTorsion(k=zero(K), θ0=zero(D))
Base.zero(t::HarmonicTorsion) = zero(typeof(t))

Base.:+(t1::HarmonicTorsion, t2::HarmonicTorsion) = HarmonicTorsion(k=(t1.k + t2.k),
                                                                        θ0=(t1.θ0 + t2.θ0))

parameter_prefix(::HarmonicTorsion, inter_type) = "inter_HT_$(inter_type)_"
parameter_fields(::Type{<:HarmonicTorsion}) = ((:k, "k"), (:θ0, "θ0"))


@inline function force(d::HarmonicTorsion, coords_i, coords_j, coords_k, coords_l,
                       boundary, args...)
    ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ = torsion_vectors(
                                    coords_i, coords_j, coords_k, coords_l, boundary)
    dEdθ = d.k * (θ - d.θ0) + d.k * (θ - d.θ0)
    fi =  dEdθ * bc_norm * cross_ab_bc / dot(cross_ab_bc, cross_ab_bc)
    fl = -dEdθ * bc_norm * cross_bc_cd / dot(cross_bc_cd, cross_bc_cd)
    v = (dot(-ab, bc) / bc_norm^2) * fi - (dot(-cd, bc) / bc_norm^2) * fl
    fj =  v - fi
    fk = -v - fl
    return SpecificForce4Atoms(fi, fj, fk, fl)
end

@inline function potential_energy(d::HarmonicTorsion, coords_i, coords_j, coords_k,
                                  coords_l, boundary, args...)
    θ = torsion_angle(coords_i, coords_j, coords_k, coords_l, boundary)
    return d.k * (θ - d.θ0)^2
end

# λ version of `HarmonicTorsion` for alchemical systems, built by `to_lambda_function`.
@kwdef struct HarmonicTorsionλ{K, D, LM, SCH} <: AlchemicalBondedInteraction
    k::K
    θ0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

is_torsion(::HarmonicTorsionλ) = true

Base.zero(t::HarmonicTorsionλ) = HarmonicTorsionλ(k=zero.(t.k), θ0=zero.(t.θ0), λ_mixing=t.λ_mixing,
                                                  scheduler=t.scheduler)

Base.:+(t1::HarmonicTorsionλ, t2::HarmonicTorsionλ) = HarmonicTorsionλ(k=(t1.k .+ t2.k),
                                                                        θ0=(t1.θ0 .+ t2.θ0),
                                                                        λ_mixing=t1.λ_mixing,
                                                                        scheduler=t1.scheduler)

function to_lambda_function(inter::HarmonicTorsion; λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return HarmonicTorsionλ(k=inter.k, θ0=inter.θ0, λ_mixing=λ_mixing, scheduler=scheduler)
end

function to_lambda_function_single(interA::HarmonicTorsion, interB::Nothing;
                                   λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return HarmonicTorsionλ(k=(interA.k, interA.k), θ0=(interA.θ0, interA.θ0), λ_mixing=λ_mixing,
                            scheduler=scheduler)
end

function to_lambda_function_single(interA::Nothing, interB::HarmonicTorsion;
                                   λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return HarmonicTorsionλ(k=(interB.k, interB.k), θ0=(interB.θ0, interB.θ0), λ_mixing=λ_mixing,
                            scheduler=scheduler)
end

function update_lambda_function(existing_lambda::HarmonicTorsionλ, interB::HarmonicTorsion)
    return HarmonicTorsionλ(k=(existing_lambda.k[1], interB.k), θ0=(existing_lambda.θ0[1], interB.θ0),
                            λ_mixing=existing_lambda.λ_mixing, scheduler=existing_lambda.scheduler)
end

plain_interaction(d::HarmonicTorsionλ, λ_params) =
    HarmonicTorsion(k=params_mixing(λ_params, d.k), θ0=params_mixing(λ_params, d.θ0))

@inline function force(d::HarmonicTorsionλ, coords_i, coords_j, coords_k, coords_l,
                       boundary, atom_i, atom_j, atom_k, atom_l, args...)
    λ, λ_params = bonded_lambda(d, (atom_i, atom_j, atom_k, atom_l))
    return λ * force(plain_interaction(d, λ_params), coords_i, coords_j, coords_k, coords_l,
                     boundary)
end

@inline function potential_energy(d::HarmonicTorsionλ, coords_i, coords_j, coords_k,
                                  coords_l, boundary, atom_i, atom_j, atom_k, atom_l, args...)
    λ, λ_params = bonded_lambda(d, (atom_i, atom_j, atom_k, atom_l))
    return λ * potential_energy(plain_interaction(d, λ_params), coords_i, coords_j, coords_k,
                                coords_l, boundary)
end