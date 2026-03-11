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

Unitful.ustrip(b::HarmonicBond) = HarmonicBond(
    k = ustrip(b.k),
    r0 = ustrip(b.r0)
)

function inject_interaction(inter::HarmonicBond, params::AbstractVector, idx_k::Int, idx_r0::Int)
    new_k  = idx_k > 0  ? typeof(inter.k)(params[idx_k])   : inter.k
    new_r0 = idx_r0 > 0 ? typeof(inter.r0)(params[idx_r0]) : inter.r0
    return HarmonicBond(new_k, new_r0)
end

function extract_parameter_indices!(buf::ParamBuffer,
                                    inter::InteractionList2Atoms{<:Any, <:AbstractVector{<:HarmonicBond}})
    bonds = from_device(inter.inters)
    types = from_device(inter.types)
    idx_k = Vector{Int}(undef, length(bonds))
    idx_r0 = Vector{Int}(undef, length(bonds))

    for i in eachindex(bonds)
        key_prefix = "inter_HB_$(types[i])_"
        idx_k[i] = _push_param!(buf, key_prefix * "k", bonds[i].k)
        idx_r0[i] = _push_param!(buf, key_prefix * "r0", bonds[i].r0)
    end

    return (idx_k, idx_r0)
end
