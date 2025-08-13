export UreyBradley

@doc raw"""
    UreyBradley(; kangle, θ0, kbond, r0)

An interaction between three atoms consisting of a harmonic bond angle
and a harmonic bond between the outer atoms.

`θ0` is in radians.
The second atom is the middle atom.
The potential energy is defined as
```math
V(\theta, r) = \frac{1}{2} k_a (\theta - \theta_0)^2 + \frac{1}{2} k_b (r - r_0)^2
```
"""
@kwdef struct UreyBradley{KA, A, KB, D}
    kangle::KA
    θ0::A
    kbond::KB
    r0::D
end

function Base.zero(::UreyBradley{KA, A, KB, D}) where {KA, A, KB, D}
    return UreyBradley(kangle=zero(KA), θ0=zero(A), kbond=zero(KB), r0=zero(D))
end

function Base.:+(a1::UreyBradley, a2::UreyBradley)
    return UreyBradley(kangle=(a1.kangle + a2.kangle), θ0=(a1.θ0 + a2.θ0),
                       kbond=(a1.kbond + a2.kbond), r0=(a1.r0 + a2.r0))
end

@inline function force(a::UreyBradley, coords_i, coords_j, coords_k, boundary, args...)
    # In 2D we use then eliminate the cross product
    ba = vector_pad3D(coords_j, coords_i, boundary)
    bc = vector_pad3D(coords_j, coords_k, boundary)
    cross_ba_bc = ba × bc
    if iszero_value(cross_ba_bc)
        zf = zero(a.kangle ./ trim3D(ba, boundary))
        fa, fb, fc = zf, zf, zf
    else
        pa = normalize(trim3D( ba × cross_ba_bc, boundary))
        pc = normalize(trim3D(-bc × cross_ba_bc, boundary))
        angle_term = -a.kangle * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.θ0)
        fa = (angle_term / norm(ba)) * pa
        fc = (angle_term / norm(bc)) * pc
        fb = -fa - fc
    end
    vec_ik = vector(coords_i, coords_k, boundary)
    c = a.kbond * (norm(vec_ik) - a.r0)
    f = c * normalize(vec_ik)
    fa += f
    fc -= f
    return SpecificForce3Atoms(fa, fb, fc)
end

@inline function potential_energy(a::UreyBradley, coords_i, coords_j,
                                  coords_k, boundary, args...)
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)
    rik = norm(vector(coords_i, coords_k, boundary))
    return (a.kangle / 2) * (θ - a.θ0) ^ 2 + (a.kbond / 2) * (rik - a.r0) ^ 2
end
