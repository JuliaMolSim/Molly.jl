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

function inject_interaction(inter::HarmonicTorsion, inter_type, params_dic)
    key_prefix = "inter_HT_$(inter_type)_"
    return HarmonicTorsion(
        dict_get(params_dic, key_prefix * "k" , inter.k ),
        dict_get(params_dic, key_prefix * "θ0", inter.θ0),
    )
end

function extract_parameters!(params_dic,
                             inter::InteractionList4Atoms{<:Any, <:AbstractVector{<:HarmonicTorsion}},
                             ff)
    for (torsion_type, torsion) in zip(inter.types, from_device(inter.inters))
        key_prefix = "inter_HT_$(torsion_type)_"
        if !haskey(params_dic, key_prefix * "k")
            params_dic[key_prefix * "k" ] = torsion.k
            params_dic[key_prefix * "θ0"] = torsion.θ0
        end
    end
    return params_dic
end

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
