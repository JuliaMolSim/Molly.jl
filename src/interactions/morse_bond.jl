export MorseBond

@doc raw"""
    MorseBond(; D, a, r0)

A Morse potential bond between two atoms.

The potential energy is defined as
```math
V(r) = D(1 - e^{-a(r - r_0)})^2
```
"""
@kwdef struct MorseBond{T, A, R}
    D::T
    a::A
    r0::R
end

Base.zero(::MorseBond{T, A, R}) where {T, A, R} = MorseBond(D=zero(T), a=zero(A), r0=zero(R))

Base.:+(b1::MorseBond, b2::MorseBond) = MorseBond(D=(b1.D + b2.D), a=(b1.a + b2.a),
                                                  r0=(b1.r0 + b2.r0))

function inject_interaction(inter::MorseBond, inter_type, params_dic)
    key_prefix = "inter_MB_$(inter_type)_"
    return MorseBond(
        dict_get(params_dic, key_prefix * "D" , inter.D ),
        dict_get(params_dic, key_prefix * "a" , inter.a ),
        dict_get(params_dic, key_prefix * "r0", inter.r0),
    )
end

function extract_parameters!(params_dic,
                             inter::InteractionList2Atoms{<:Any, <:AbstractVector{<:MorseBond}},
                             ff)
    for (bond_type, bond) in zip(inter.types, from_device(inter.inters))
        key_prefix = "inter_MB_$(bond_type)_"
        if !haskey(params_dic, key_prefix * "D")
            params_dic[key_prefix * "D" ] = bond.D
            params_dic[key_prefix * "a" ] = bond.a
            params_dic[key_prefix * "r0"] = bond.r0
        end
    end
    return params_dic
end

@inline function force(b::MorseBond, coord_i, coord_j, boundary, args...)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    ralp = exp(-b.a * (r - b.r0))
    c = 2 * b.D * b.a * (1 - ralp) * ralp
    f = c * normalize(dr)
    return SpecificForce2Atoms(f, -f)
end

@inline function potential_energy(b::MorseBond, coord_i, coord_j, boundary, args...)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    ralp = exp(-b.a * (r - b.r0))
    return b.D * (1 - ralp)^2
end
