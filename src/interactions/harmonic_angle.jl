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

Base.zero(::HarmonicAngle{K, D}) where {K, D} = HarmonicAngle(k=zero(K), θ0=zero(D))

Base.:+(a1::HarmonicAngle, a2::HarmonicAngle) = HarmonicAngle(k=(a1.k + a2.k), θ0=(a1.θ0 + a2.θ0))

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
    angle_term = -a.k * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.θ0)
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

Unitful.ustrip(a::HarmonicAngle) = HarmonicAngle(
    k = ustrip(a.k),
    θ0 = ustrip(a.θ0)
)

function inject_interaction(inter::HarmonicAngle, params::AbstractVector, idx_k::Int, idx_θ0::Int)
    new_k  = idx_k > 0  ? typeof(inter.k)(params[idx_k])   : inter.k
    new_θ0 = idx_θ0 > 0 ? typeof(inter.θ0)(params[idx_θ0]) : inter.θ0
    return HarmonicAngle(new_k, new_θ0)
end

function extract_parameter_indices!(buf::ParamBuffer,
                                    inter::InteractionList3Atoms{<:Any, <:AbstractVector{<:HarmonicAngle}})
    angles = from_device(inter.inters)
    types = from_device(inter.types)
    idx_k = Vector{Int}(undef, length(angles))
    idx_θ0 = Vector{Int}(undef, length(angles))

    for i in eachindex(angles)
        key_prefix = "inter_HA_$(types[i])_"
        idx_k[i] = _push_param!(buf, key_prefix * "k", angles[i].k)
        idx_θ0[i] = _push_param!(buf, key_prefix * "θ0", angles[i].θ0)
    end

    return (idx_k, idx_θ0)
end
