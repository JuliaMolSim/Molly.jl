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
