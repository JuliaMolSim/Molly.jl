export CosineAngle

@doc raw"""
    CosineAngle(; k, θ0)

A cosine bond angle between three atoms.

`θ0` is in radians.
The potential energy is defined as
```math
V(\theta) = k(1 + \cos(\theta - \theta_0))
```
"""
struct CosineAngle{K, D}
    k::K
    θ0::D
end

CosineAngle(; k, θ0) = CosineAngle{typeof(k), typeof(θ0)}(k, θ0)

@inline function force(a::CosineAngle, coords_i, coords_j, coords_k, boundary, args...)
    # In 2D we use then eliminate the cross product
    ba = vector_pad3D(coords_j, coords_i, boundary)
    bc = vector_pad3D(coords_j, coords_k, boundary)
    cross_ba_bc = ba × bc
    if iszero_value(cross_ba_bc)
        zf = zero(a.k ./ ba)
        return SpecificForce3Atoms(zf, zf, zf)
    end
    pa = normalize(trim3D( ba × cross_ba_bc, boundary))
    pc = normalize(trim3D(-bc × cross_ba_bc, boundary))
    θ = bond_angle(ba, bc)
    angle_term = a.k * sin(θ - a.θ0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return SpecificForce3Atoms(fa, fb, fc)
end

@inline function potential_energy(a::CosineAngle, coords_i, coords_j,
                                  coords_k, boundary, args...)
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)
    return a.k * (1 + cos(θ - a.θ0))
end
