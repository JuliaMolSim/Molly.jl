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

Base.zero(::Type{HarmonicBond{K, D}}) where {K, D} = HarmonicBond(k=zero(K), r0=zero(D))
Base.zero(b::HarmonicBond) = zero(typeof(b))

Base.:+(b1::HarmonicBond, b2::HarmonicBond) = HarmonicBond(k=(b1.k + b2.k), r0=(b1.r0 + b2.r0))

parameter_prefix(::HarmonicBond, inter_type) = "inter_HB_$(inter_type)_"
parameter_fields(::Type{<:HarmonicBond}) = ((:k, "k"), (:r0, "r0"))


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
