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
@kwdef struct HarmonicTorsionλ{K, D, LM, SCH}
    k::K
    θ0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

Base.zero(::HarmonicTorsionλ{K, D, LM, SCH}) where {K, D, LM, SCH} = HarmonicTorsionλ(k=zero(K), θ0=zero(D))

Base.:+(t1::HarmonicTorsionλ, t2::HarmonicTorsionλ) = HarmonicTorsionλ(k=(t1.k + t2.k),
                                                                        θ0=(t1.θ0 + t2.θ0))

function to_lambda_function(inter::HarmonicTorsion; λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return HarmonicTorsionλ(k=inter.k, θ0=inter.θ0, λ_mixing=λ_mixing, scheduler=scheduler)
end

@inline function force(d::HarmonicTorsionλ, coords_i, coords_j, coords_k, coords_l,
                       boundary, atom_i, atom_j, atom_k, atom_l, args...)
    T = typeof(ustrip(atom_i.λ))
    ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ = torsion_vectors(
                                    coords_i, coords_j, coords_k, coords_l, boundary)

    λ_glob = T(λ_mixing(d.λ_mixing, (atom_i.λ, atom_j.λ, atom_k.λ, atom_l.λ)))    
    pair_role = mix_roles(d.scheduler, (atom_i.alch_role, atom_j.alch_role, atom_k.alch_role, atom_l.alch_role))
    λ, λ_params = scale_dual(d.scheduler, λ_glob, pair_role)

    dEdθ = d.k * (θ - d.θ0) + d.k * (θ - d.θ0)
    fi =  dEdθ * bc_norm * cross_ab_bc / dot(cross_ab_bc, cross_ab_bc)
    fl = -dEdθ * bc_norm * cross_bc_cd / dot(cross_bc_cd, cross_bc_cd)
    v = (dot(-ab, bc) / bc_norm^2) * fi - (dot(-cd, bc) / bc_norm^2) * fl
    fj =  v - fi
    fk = -v - fl
    return SpecificForce4Atoms(λ*fi, λ*fj, λ*fk, λ*fl)
end

@inline function potential_energy(d::HarmonicTorsionλ, coords_i, coords_j, coords_k,
                                  coords_l, boundary, atom_i, atom_j, atom_k, atom_l, args...)
    T = typeof(ustrip(atom_i.λ))
    θ = torsion_angle(coords_i, coords_j, coords_k, coords_l, boundary)
    λ_glob = T(λ_mixing(d.λ_mixing, (atom_i.λ, atom_j.λ, atom_k.λ, atom_l.λ)))    
    pair_role = mix_roles(d.scheduler, (atom_i.alch_role, atom_j.alch_role, atom_k.alch_role, atom_l.alch_role))
    λ, λ_params = scale_dual(d.scheduler, λ_glob, pair_role)
    return λ * d.k * (θ - d.θ0)^2
end

@inline function force_λ(d::HarmonicTorsionλ, coords_i, coords_j, coords_k, coords_l,
                       boundary, atoms_i, atoms_j, atoms_k, atoms_l, F, args...)
    @warn("HarmonicTorsions not implemented yet for force with respect to λ")
    return SpecificForce4Atoms(zero_pairwise_force(coords_i, F), zero_pairwise_force(coords_i, F), zero_pairwise_force(coords_i, F), zero_pairwise_force(coords_i, F))
end