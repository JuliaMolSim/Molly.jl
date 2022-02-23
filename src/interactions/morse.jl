export Morse

"""
    MorseBond(; D, α, r0)

Morse potential for bonds between two atoms
"""

struct MorseBond{T, A, R} <: SpecificInteraction
    D::T
    α::A
    r0::R
end

MorseBond(; D, α, r0) = MorseBond{typeof(D), typeof(α), typeof(r0)}(D, α, r0)

@inline @inbounds function force(b::MorseBond, coord_i, coord_j, box_size)
    dr = vector(coord_i, coord_j, box_size)
    r = norm(dr)
    ralp = exp(-b.α*(r-b.r0))
    c = -2.0*b.D*b.α*(1-ralp)*ralp/r
    f = c * normalize(dr)
    return SpecificForce2Atoms(f, -f)
end

@inline @inbounds function potential_energy(b::MorseBond, coord_i, coord_j, box_size)
    dr = vector(coord_i, coord_j, box_size)
    r = norm(dr)
    ralp = exp(-b.α*(r - b.r0))
    return b.D * (1 - ralp)^2
end

