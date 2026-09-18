export HarmonicAngle

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

@inline function force_λ(a::HarmonicAngle, coords_i, coords_j, coords_k, boundary, atoms_i,
                        atoms_j, atoms_k, F, args...)
    dr = vector_pad3D(coords_j, coords_i, boundary)
    return SpecificForce3Atoms(zero_pairwise_force(dr, F), zero_pairwise_force(dr, F), zero_pairwise_force(dr, F))
end

# λ version of `HarmonicAngle` for alchemical systems, built by `to_lambda_function`.
@kwdef struct HarmonicAngleλ{K, D, LM, SCH}
    k::K
    θ0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

Base.zero(::HarmonicAngleλ{K, D}) where {K, D} = HarmonicAngleλ(k=zero(K), θ0=zero(D))

Base.:+(a1::HarmonicAngleλ, a2::HarmonicAngleλ) = HarmonicAngleλ(k=(a1.k + a2.k), θ0=(a1.θ0 + a2.θ0))

function Base.show(io::IO, x::HarmonicAngleλ)
    println(io, "HarmonicAngleλ: (k: $(x.k)) - θ0: $(x.θ0) - λ_mixing: $(x.λ_mixing) - scheduler: $(x.scheduler)")
end

function extract_parameters!(params_dic,
                             inter::InteractionList3Atoms{<:Any, <:AbstractVector{<:HarmonicAngleλ}},
                             ff)
    for (angle_type, ang) in zip(inter.types, from_device(inter.inters))
        key_prefix = "inter_HA_$(angle_type)_"
        if !haskey(params_dic, key_prefix * "k")
            params_dic[key_prefix * "k" ] = ang.k
            params_dic[key_prefix * "θ0"] = ang.θ0
        end
    end
    return params_dic
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

@inline function force(a::HarmonicAngleλ, coords_i, coords_j, coords_k, boundary, 
                        atom_i, atom_j, atom_k, args...)
    T = typeof(ustrip(atom_i.λ))
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

    λ_glob = T(λ_mixing(a.λ_mixing, (atom_i.λ, atom_j.λ, atom_k.λ)))    
    pair_role = mix_roles(a.scheduler, (atom_i.alch_role, atom_j.alch_role, atom_k.alch_role))
    λ, λ_params = scale_dual(a.scheduler, λ_glob, pair_role)
    k = params_mixing(λ_params, a.k)
    θ0 = params_mixing(λ_params, a.θ0)

    angle_term = -k * (acos_bound(dot(ba, bc) / (norm(ba) * norm(bc))) - θ0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return SpecificForce3Atoms(λ*fa, λ*fb, λ*fc)
end

@inline function potential_energy(a::HarmonicAngleλ, coords_i, coords_j,
                                  coords_k, boundary, atom_i,
                                  atom_j, atom_k, args...)
    T = typeof(ustrip(atom_i.λ))
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)
    λ_glob = T(λ_mixing(a.λ_mixing, (atom_i.λ, atom_j.λ, atom_k.λ)))    
    pair_role = mix_roles(a.scheduler, (atom_i.alch_role, atom_j.alch_role, atom_k.alch_role))
    λ, λ_params = scale_dual(a.scheduler, λ_glob, pair_role)
    k = params_mixing(λ_params, a.k)
    θ0 = params_mixing(λ_params, a.θ0)
    return λ * (k / 2) * (θ - θ0) ^ 2
end

@inline function force_λ(a::HarmonicAngleλ, coords_i, coords_j, coords_k, boundary, atom_i,
                        atom_j, atom_k, F, args...)
    dr = vector_pad3D(coords_j, coords_i, boundary)
    return SpecificForce3Atoms(zero_pairwise_force(dr, F), zero_pairwise_force(dr, F), zero_pairwise_force(dr, F))
end