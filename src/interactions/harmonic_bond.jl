export HarmonicBond

@doc raw"""
    HarmonicBond(; k, r0)

A harmonic bond between two atoms.
The potential energy is defined as
```math
V(r) = \frac{1}{2} k (r - r_0)^2
```
"""
struct HarmonicBond{K, D} <: SpecificInteraction
    k::K
    r0::D
end

HarmonicBond(; k, r0) = HarmonicBond{typeof(k), typeof(r0)}(k, r0)

@inline @inbounds function force(b::HarmonicBond, coord_i, coord_j, boundary)
    ab = vector(coord_i, coord_j, boundary)
    c = b.k * (norm(ab) - b.r0)
    f = c * normalize(ab)
    return SpecificForce2Atoms(f, -f)
end

@inline @inbounds function potential_energy(b::HarmonicBond, coord_i, coord_j, boundary)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    return (b.k / 2) * (r - b.r0) ^ 2
end
