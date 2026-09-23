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

function Base.zero(::Type{UreyBradley{KA, A, KB, D}}) where {KA, A, KB, D}
    return UreyBradley(kangle=zero(KA), θ0=zero(A), kbond=zero(KB), r0=zero(D))
end

Base.zero(a::UreyBradley) = zero(typeof(a))

function Base.:+(a1::UreyBradley, a2::UreyBradley)
    return UreyBradley(kangle=(a1.kangle + a2.kangle), θ0=(a1.θ0 + a2.θ0),
                       kbond=(a1.kbond + a2.kbond), r0=(a1.r0 + a2.r0))
end

parameter_prefix(::UreyBradley, inter_type) = "inter_UB_$(inter_type)_"
parameter_fields(::Type{<:UreyBradley}) =
    ((:kangle, "kangle"), (:θ0, "θ0"), (:kbond, "kbond"), (:r0, "r0"))


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
        angle_term = -a.kangle * (acos_bound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.θ0)
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

# λ version of `UreyBradley` for alchemical systems, built by `to_lambda_function`.
@kwdef struct UreyBradleyλ{KA, A, KB, D, LM, SCH} <: AlchemicalBondedInteraction
    kangle::KA
    θ0::A
    kbond::KB
    r0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

function Base.zero(a::UreyBradleyλ)
    return UreyBradleyλ(kangle=zero.(a.kangle), θ0=zero.(a.θ0), kbond=zero.(a.kbond), r0=zero.(a.r0),
                        λ_mixing=a.λ_mixing, scheduler=a.scheduler)
end

function Base.:+(a1::UreyBradleyλ, a2::UreyBradleyλ)
    return UreyBradleyλ(kangle=(a1.kangle .+ a2.kangle), θ0=(a1.θ0 .+ a2.θ0),
                        kbond=(a1.kbond .+ a2.kbond), r0=(a1.r0 .+ a2.r0),
                        λ_mixing=a1.λ_mixing, scheduler=a1.scheduler)
end

function to_lambda_function(inter::UreyBradley; λ_mixing=MinimumMixing(),
                            scheduler=DefaultLambdaScheduler())
    return UreyBradleyλ(kangle=inter.kangle, θ0=inter.θ0, kbond=inter.kbond, r0=inter.r0,
                        λ_mixing=λ_mixing, scheduler=scheduler)
end

plain_interaction(a::UreyBradleyλ, λ_params) = UreyBradley(
    kangle=params_mixing(λ_params, a.kangle), θ0=params_mixing(λ_params, a.θ0),
    kbond=params_mixing(λ_params, a.kbond), r0=params_mixing(λ_params, a.r0))

@inline function force(a::UreyBradleyλ, coords_i, coords_j, coords_k, boundary, atom_i, atom_j,
                       atom_k, args...)
    λ, λ_params = bonded_lambda(a, (atom_i, atom_j, atom_k))
    return λ * force(plain_interaction(a, λ_params), coords_i, coords_j, coords_k, boundary)
end

@inline function potential_energy(a::UreyBradleyλ, coords_i, coords_j, coords_k, boundary, atom_i,
                                  atom_j, atom_k, args...)
    λ, λ_params = bonded_lambda(a, (atom_i, atom_j, atom_k))
    return λ * potential_energy(plain_interaction(a, λ_params), coords_i, coords_j, coords_k,
                                boundary)
end
