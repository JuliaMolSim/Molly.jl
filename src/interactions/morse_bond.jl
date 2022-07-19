export MorseBond

@doc raw"""
    MorseBond(; D, a, r0)

A Morse potential bond between two atoms.
The potential energy is defined as
```math
V(r) = D(1 - e^{-a(r - r_0)}))^2
```
"""
struct MorseBond{T, A, R} <: SpecificInteraction
    D::T
    a::A
    r0::R
end

MorseBond(; D, a, r0) = MorseBond{typeof(D), typeof(a), typeof(r0)}(D, a, r0)

@inline @inbounds function force(b::MorseBond, coord_i, coord_j, boundary)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    ralp = exp(-b.a * (r - b.r0))
    c = 2 * b.D * b.a * (1 - ralp) * ralp
    f = c * normalize(dr)
    return SpecificForce2Atoms(f, -f)
end

@inline @inbounds function potential_energy(b::MorseBond, coord_i, coord_j, boundary)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    ralp = exp(-b.a * (r - b.r0))
    return b.D * (1 - ralp)^2
end
