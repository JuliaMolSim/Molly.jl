export MorseBond

@doc raw"""
    MorseBond(; D, a, r0)

A Morse potential bond between two atoms.

The potential energy is defined as
```math
V(r) = D(1 - e^{-a(r - r_0)})^2
```
"""
struct MorseBond{T, A, R}
    D::T
    a::A
    r0::R
end

MorseBond(; D, a, r0) = MorseBond(D, a, r0)

Base.zero(::MorseBond{T, A, R}) where {T, A, R} = MorseBond(D=zero(T), a=zero(A), r0=zero(R))

Base.:+(b1::MorseBond, b2::MorseBond) = MorseBond(D=(b1.D + b2.D), a=(b1.a + b2.a),
                                                  r0=(b1.r0 + b2.r0))

@inline function force(b::MorseBond, coord_i, coord_j, boundary, args...)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    ralp = exp(-b.a * (r - b.r0))
    c = 2 * b.D * b.a * (1 - ralp) * ralp
    f = c * normalize(dr)
    return SpecificForce2Atoms(f, -f)
end

@inline function potential_energy(b::MorseBond, coord_i, coord_j, boundary, args...)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    ralp = exp(-b.a * (r - b.r0))
    return b.D * (1 - ralp)^2
end
