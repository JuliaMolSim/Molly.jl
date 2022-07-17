export HarmonicPositionRestraint

"""
    HarmonicPositionRestraint(; x0, kb)

A harmonic position restraint on an atom to coordinate `x0`.
"""
struct HarmonicPositionRestraint{C, K} <: SpecificInteraction
    x0::C
    kb::K
end

HarmonicPositionRestraint(; x0, kb) = HarmonicPositionRestraint{typeof(x0), typeof(kb)}(x0, kb)

@inline @inbounds function force(pr::HarmonicPositionRestraint, coord_i, boundary)
    ab = vector(coord_i, pr.x0, boundary)
    c = pr.kb * norm(ab)
    f = c * normalize(ab)
    return SpecificForce1Atoms(f)
end

@inline @inbounds function potential_energy(pr::HarmonicPositionRestraint, coord_i, boundary)
    dr = vector(coord_i, pr.x0, boundary)
    r = norm(dr)
    return (pr.kb / 2) * r ^ 2
end
