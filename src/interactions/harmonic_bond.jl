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

Base.zero(::HarmonicBond{K, D}) where {K, D} = HarmonicBond(k=zero(K), r0=zero(D))

Base.:+(b1::HarmonicBond, b2::HarmonicBond) = HarmonicBond(k=(b1.k + b2.k), r0=(b1.r0 + b2.r0))

function inject_interaction(inter::HarmonicBond, inter_type, params_dic)
    key_prefix = "inter_HB_$(inter_type)_"
    return HarmonicBond(
        dict_get(params_dic, key_prefix * "k" , inter.k ),
        dict_get(params_dic, key_prefix * "r0", inter.r0),
    )
end

function extract_parameters!(params_dic,
                             inter::InteractionList2Atoms{<:Any, <:AbstractVector{<:HarmonicBond}},
                             ff)
    for (bond_type, bond) in zip(inter.types, from_device(inter.inters))
        key_prefix = "inter_HB_$(bond_type)_"
        if !haskey(params_dic, key_prefix * "k")
            params_dic[key_prefix * "k" ] = bond.k
            params_dic[key_prefix * "r0"] = bond.r0
        end
    end
    return params_dic
end

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

@inline function force_λ(b::HarmonicBond, coord_i, coord_j, boundary, atoms_i, atoms_j, F, args...)
    dr = vector(coord_i, coord_j, boundary)
    return SpecificForce2Atoms(zero_pairwise_force(dr, F), zero_pairwise_force(dr, F))
end

@doc raw"""
    HarmonicBondλ(; k, r0)

A harmonic bond between two atoms that is scaled by λ for core atoms.

The potential energy is defined as
```math
V(r) = \frac{1}{2} k (r - r_0)^2
k = (1-λ)*k^a + λ*k^b
r_0 = (1-λ)*r_0^a + λ*r_0^b
```
"""
@kwdef struct HarmonicBondλ{K, D, LM, SCH}
    k::K
    r0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

Base.zero(::HarmonicBondλ{K, D, LM, SCH}) where {K, D, LM, SCH} = HarmonicBondλ(k=zero(K), r0=zero(D))

Base.:+(b1::HarmonicBondλ, b2::HarmonicBondλ) = HarmonicBondλ(k=(b1.k + b2.k), r0=(b1.r0 + b2.r0))

function inject_interaction(inter::HarmonicBondλ, inter_type, params_dic)
    key_prefix = "inter_HB_$(inter_type)_"
    return HarmonicBondλ(
        dict_get(params_dic, key_prefix * "k" , inter.k ),
        dict_get(params_dic, key_prefix * "r0", inter.r0),
    )
end

function extract_parameters!(params_dic,
                             inter::InteractionList2Atoms{<:Any, <:AbstractVector{<:HarmonicBondλ}},
                             ff)
    for (bond_type, bond) in zip(inter.types, from_device(inter.inters))
        key_prefix = "inter_HB_$(bond_type)_"
        if !haskey(params_dic, key_prefix * "k")
            params_dic[key_prefix * "k" ] = bond.k
            params_dic[key_prefix * "r0"] = bond.r0
        end
    end
    return params_dic
end

function to_lambda_function(inter::HarmonicBond; λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return HarmonicBondλ(k=inter.k, r0=inter.r0, λ_mixing=λ_mixing, scheduler=scheduler)
end

function to_lambda_function_single(interA::Union{HarmonicBond, Nothing}, interB::Union{HarmonicBond, Nothing}; 
                                   λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    ref = isnothing(interA) ? interB : interA
    
    k_A  = isnothing(interA) ? ref.k  : interA.k
    k_B  = isnothing(interB) ? ref.k  : interB.k
    r0_A = isnothing(interA) ? ref.r0 : interA.r0
    r0_B = isnothing(interB) ? ref.r0 : interB.r0
    
    return HarmonicBondλ(k=(k_A, k_B), r0=(r0_A, r0_B), λ_mixing=λ_mixing, scheduler=scheduler)
end

function update_lambda_function(existing_lambda::HarmonicBondλ, interB::HarmonicBond)
    return HarmonicBondλ(k=(existing_lambda.k[1], interB.k), 
                         r0=(existing_lambda.r0[1], interB.r0), 
                         λ_mixing=existing_lambda.λ_mixing, 
                         scheduler=existing_lambda.scheduler)
end

@inline function force(b::HarmonicBondλ{K, D, LM, SCH},coord_i, coord_j, 
                                    boundary, atom_i, atom_j, args...) where {K, D, LM, SCH}
    T = typeof(ustrip(atom_i.λ))
    ab = vector(coord_i, coord_j, boundary)
    λ_glob = T(λ_mixing(b.λ_mixing, (atom_i.λ, atom_j.λ)))    
    pair_role = mix_roles(b.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_dual(b.scheduler, λ_glob, pair_role)
    k = params_mixing(λ_params, b.k)
    r0 = params_mixing(λ_params, b.r0)
    c = k * (norm(ab) - r0)
    f = c * normalize(ab)
    return SpecificForce2Atoms(λ*f, λ*-f)
end

@inline function potential_energy(b::HarmonicBondλ{K, D, LM, SCH}, coord_i, coord_j, 
                                    boundary, atom_i, atom_j, args...) where {K, D, LM, SCH}
    T = typeof(ustrip(atom_i.λ))
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    λ_glob = T(λ_mixing(b.λ_mixing, (atom_i.λ, atom_j.λ)))  
    pair_role = mix_roles(b.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_dual(b.scheduler, λ_glob, pair_role)
    k = params_mixing(λ_params, b.k)

    r0 = params_mixing(λ_params, b.r0)
    E = λ * (k / 2) * (r - r0) ^ 2
    return E
end

@inline function force_λ(b::HarmonicBondλ, coord_i, coord_j, boundary, atoms_i, atoms_j, F, args...)
    dr = vector(coord_i, coord_j, boundary)
    return SpecificForce2Atoms(zero_pairwise_force(dr, F), zero_pairwise_force(dr, F))
end