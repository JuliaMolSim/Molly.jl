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

# λ version of `RBTorsion` for alchemical systems, built by `to_lambda_function`. The torsion
# itself is evaluated by `RBTorsion` and scaled by the λ prefactor of the four atoms.
@kwdef struct RBTorsionλ{T, LM, SCH}
    c0::T
    c1::T
    c2::T
    c3::T
    c4::T
    c5::T
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

function to_lambda_function(inter::RBTorsion; λ_mixing=MinimumMixing(),
                            scheduler=DefaultLambdaScheduler())
    return RBTorsionλ(c0=inter.c0, c1=inter.c1, c2=inter.c2, c3=inter.c3, c4=inter.c4,
                      c5=inter.c5, λ_mixing=λ_mixing, scheduler=scheduler)
end

rb_torsion(d::RBTorsionλ) = RBTorsion(d.c0, d.c1, d.c2, d.c3, d.c4, d.c5)

@inline function rb_torsion_λ(d::RBTorsionλ, atom_i, atom_j, atom_k, atom_l)
    T = typeof(ustrip(atom_i.λ))
    λ_glob = T(λ_mixing(d.λ_mixing, (atom_i.λ, atom_j.λ, atom_k.λ, atom_l.λ)))
    pair_role = mix_roles(d.scheduler, (atom_i.alch_role, atom_j.alch_role, atom_k.alch_role,
                                        atom_l.alch_role))
    λ, λ_params = scale_dual(d.scheduler, λ_glob, pair_role)
    return λ
end

@inline function force(d::RBTorsionλ, coords_i, coords_j, coords_k, coords_l, boundary,
                       atom_i, atom_j, atom_k, atom_l, args...)
    fs = force(rb_torsion(d), coords_i, coords_j, coords_k, coords_l, boundary)
    λ = rb_torsion_λ(d, atom_i, atom_j, atom_k, atom_l)
    return SpecificForce4Atoms(λ * fs.f1, λ * fs.f2, λ * fs.f3, λ * fs.f4)
end

@inline function potential_energy(d::RBTorsionλ, coords_i, coords_j, coords_k,
                                  coords_l, boundary, atom_i, atom_j, atom_k, atom_l, args...)
    pe = potential_energy(rb_torsion(d), coords_i, coords_j, coords_k, coords_l, boundary)
    return rb_torsion_λ(d, atom_i, atom_j, atom_k, atom_l) * pe
end
