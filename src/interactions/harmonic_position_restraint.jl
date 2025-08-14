export HarmonicPositionRestraint

@doc raw"""
    HarmonicPositionRestraint(; k, x0)

A harmonic position restraint on an atom to coordinate `x0`.

The potential energy is defined as
```math
V(\boldsymbol{x}) = \frac{1}{2} k |\boldsymbol{x} - \boldsymbol{x}_0|^2
```
"""
@kwdef struct HarmonicPositionRestraint{K, C}
    k::K
    x0::C
end

@inline function force(pr::HarmonicPositionRestraint, coord_i, boundary, args...)
    ab = vector(coord_i, pr.x0, boundary)
    c = pr.k * norm(ab)
    if iszero_value(c)
        f = c * ustrip.(ab)
        return SpecificForce1Atoms(f)
    end
    f = c * normalize(ab)
    return SpecificForce1Atoms(f)
end

@inline function potential_energy(pr::HarmonicPositionRestraint, coord_i, boundary, args...)
    dr = vector(coord_i, pr.x0, boundary)
    return (pr.k / 2) * dot(dr, dr)
end
