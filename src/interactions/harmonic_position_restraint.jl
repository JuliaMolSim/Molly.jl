export HarmonicPositionRestraint

@doc raw"""
    HarmonicPositionRestraint(; k, x0)

A harmonic position restraint on an atom to coordinate `x0`.

The potential energy is defined as
```math
V(\boldsymbol{x}) = \frac{1}{2} k |\boldsymbol{x} - \boldsymbol{x}_0|^2
```

Does not contribute to the virial.
"""
@kwdef struct HarmonicPositionRestraint{K, C}
    k::K
    x0::C
end

function Base.zero(::Type{HarmonicPositionRestraint{K, C}}) where {K, C}
    return HarmonicPositionRestraint(k=zero(K), x0=zero(C))
end

Base.zero(r::HarmonicPositionRestraint) = zero(typeof(r))

function Base.:+(r1::HarmonicPositionRestraint, r2::HarmonicPositionRestraint)
    return HarmonicPositionRestraint(k=(r1.k + r2.k), x0=(r1.x0 + r2.x0))
end

function inject_interaction(inter::HarmonicPositionRestraint, inter_type, params_dic)
    key_prefix = "inter_HPR_$(inter_type)_"
    return HarmonicPositionRestraint(
        dict_get(params_dic, key_prefix * "k", inter.k),
        inter.x0,
    )
end

function extract_parameters!(params_dic,
                             inter::InteractionList1Atoms{<:Any, <:AbstractVector{<:HarmonicPositionRestraint}},
                             ff)
    for (restraint_type, restraint) in zip(inter.types, from_device(inter.inters))
        key_prefix = "inter_HPR_$(restraint_type)_"
        if !haskey(params_dic, key_prefix * "k")
            params_dic[key_prefix * "k"] = restraint.k
        end
    end
    return params_dic
end

@inline function force(pr::HarmonicPositionRestraint, coord_i, boundary, args...)
    ab = vector(coord_i, pr.x0, boundary)
    c = pr.k * norm(ab)
    if iszero_value(c)
        f = c * ustrip.(ab)
        return SpecificForce1Atoms(f)
    end
    f = c * normalize(ab)
    return SpecificForce1Atoms(f)
end

@inline function potential_energy(pr::HarmonicPositionRestraint, coord_i, boundary, args...)
    dr = vector(coord_i, pr.x0, boundary)
    return (pr.k / 2) * dot(dr, dr)
end
