export RBTorsion

@doc raw"""
    RBTorsion(; c0, c1, c2, c3, c4, c5)

A Ryckaert-Bellemans torsion angle between four atoms.

The potential energy is defined as
```math
V(\phi) = \sum_{n=0}^{5} C_n \cos^n(\psi)
```
where ``\psi = \phi - 180^{\circ}`` and ``\phi`` is the angle between the planes
defined by atoms (i, j, k) and (j, k, l).
Coefficients using the other sign convention, i.e. powers of ``\cos(\phi)``
rather than ``\cos(\psi)``, can be converted by multiplying ``C_n`` by
``(-1)^n``.

Only compatible with 3D systems.
"""
struct RBTorsion{T}
    c0::T
    c1::T
    c2::T
    c3::T
    c4::T
    c5::T
end

RBTorsion(; c0, c1, c2, c3, c4, c5) = RBTorsion{typeof(c0)}(c0, c1, c2, c3, c4, c5)

function Base.zero(::Type{RBTorsion{T}}) where T
    z = zero(T)
    return RBTorsion(z, z, z, z, z, z)
end

Base.zero(t::RBTorsion) = zero(typeof(t))

function Base.:+(t1::RBTorsion, t2::RBTorsion)
    return RBTorsion(t1.c0 + t2.c0, t1.c1 + t2.c1, t1.c2 + t2.c2, t1.c3 + t2.c3,
                     t1.c4 + t2.c4, t1.c5 + t2.c5)
end

parameter_prefix(::RBTorsion, inter_type) = "inter_RB_$(inter_type)_"
parameter_fields(::Type{<:RBTorsion}) =
    ((:c0, "c0"), (:c1, "c1"), (:c2, "c2"), (:c3, "c3"), (:c4, "c4"), (:c5, "c5"))

@inline function force(d::RBTorsion, coords_i, coords_j, coords_k, coords_l, boundary, args...)
    ab, bc, cd, cross_ab_bc, cross_bc_cd, bc_norm, θ = torsion_vectors(
                                    coords_i, coords_j, coords_k, coords_l, boundary)
    # ψ = θ - π, so cos(ψ) = -cos(θ) and sin(ψ) = -sin(θ)
    cos_ψ = -cos(θ)
    # dV/dθ = dV/dψ = -sin(ψ) * dV/dcos(ψ)
    dEdθ = sin(θ) * (d.c1 + cos_ψ * (2 * d.c2 + cos_ψ * (3 * d.c3 +
                        cos_ψ * (4 * d.c4 + cos_ψ * 5 * d.c5))))
    fi =  dEdθ * bc_norm * cross_ab_bc / dot(cross_ab_bc, cross_ab_bc)
    fl = -dEdθ * bc_norm * cross_bc_cd / dot(cross_bc_cd, cross_bc_cd)
    v = (dot(-ab, bc) / bc_norm^2) * fi - (dot(-cd, bc) / bc_norm^2) * fl
    fj =  v - fi
    fk = -v - fl
    return SpecificForce4Atoms(fi, fj, fk, fl)
end

@inline function potential_energy(d::RBTorsion, coords_i, coords_j, coords_k,
                                  coords_l, boundary, args...)
    θ = torsion_angle(coords_i, coords_j, coords_k, coords_l, boundary)
    cos_ψ = -cos(θ)
    return d.c0 + cos_ψ * (d.c1 + cos_ψ * (d.c2 + cos_ψ * (d.c3 +
                               cos_ψ * (d.c4 + cos_ψ * d.c5))))
end
