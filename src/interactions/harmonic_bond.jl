"""
    HarmonicBond(; b0, kb)

A harmonic bond between two atoms.
"""
struct HarmonicBond{D, K} <: SpecificInteraction
    b0::D
    kb::K
end

HarmonicBond(; b0, kb) = HarmonicBond{typeof(b0), typeof(kb)}(b0, kb)

@inline @inbounds function force(b::HarmonicBond, coord_i, coord_j, box_size)
    ab = vector(coord_i, coord_j, box_size)
    c = b.kb * (norm(ab) - b.b0)
    f = c * normalize(ab)
    return SpecificForce2Atom(f, -f)
end

@inline @inbounds function potential_energy(b::HarmonicBond, coord_i, coord_j, box_size)
    dr = vector(coord_i, coord_j, box_size)
    r = norm(dr)
    return (b.kb / 2) * (r - b.b0) ^ 2
end
