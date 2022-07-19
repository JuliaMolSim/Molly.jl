export HarmonicBond

@doc raw"""
    HarmonicBond(; b0, kb)

A harmonic bond between two atoms.
The potential energy is defined as
```math
V(r) = \frac{1}{2}k(r - r_0)^2
```
"""
struct HarmonicBond{D, K} <: SpecificInteraction
    b0::D
    kb::K
end

HarmonicBond(; b0, kb) = HarmonicBond{typeof(b0), typeof(kb)}(b0, kb)

@inline @inbounds function force(b::HarmonicBond, coord_i, coord_j, boundary)
    ab = vector(coord_i, coord_j, boundary)
    c = b.kb * (norm(ab) - b.b0)
    f = c * normalize(ab)
    return SpecificForce2Atoms(f, -f)
end

@inline @inbounds function potential_energy(b::HarmonicBond, coord_i, coord_j, boundary)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    return (b.kb / 2) * (r - b.b0) ^ 2
end
