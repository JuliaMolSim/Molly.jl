export HarmonicAngle

@doc raw"""
    HarmonicAngle(; k, θ0)

A harmonic bond angle between three atoms.
The potential energy is defined as
```math
V(\theta) = \frac{1}{2} k (\theta - \theta_0)^2
```
"""
struct HarmonicAngle{K, D} <: SpecificInteraction
    k::K
    θ0::D
end

HarmonicAngle(; k, θ0) = HarmonicAngle{typeof(k), typeof(θ0)}(k, θ0)

@inline @inbounds function force(a::HarmonicAngle, coords_i, coords_j, coords_k, boundary)
    # In 2D we use then eliminate the cross product
    ba = vector_pad3D(coords_j, coords_i, boundary)
    bc = vector_pad3D(coords_j, coords_k, boundary)
    cross_ba_bc = ba × bc
    if iszero_value(cross_ba_bc)
        zf = zero(a.k ./ trim3D(ba, boundary))
        return SpecificForce3Atoms(zf, zf, zf)
    end
    pa = normalize(trim3D( ba × cross_ba_bc, boundary))
    pc = normalize(trim3D(-bc × cross_ba_bc, boundary))
    angle_term = -a.k * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.θ0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return SpecificForce3Atoms(fa, fb, fc)
end

@inline @inbounds function potential_energy(a::HarmonicAngle, coords_i, coords_j,
                                            coords_k, boundary)
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)
    return (a.k / 2) * (θ - a.θ0) ^ 2
end
