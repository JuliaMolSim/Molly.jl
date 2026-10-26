export UreyBradley

@public UreyBradleyλ

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

@doc raw"""
    UreyBradleyλ(; kangle, θ0, kbond, r0, λ_mixing=MinimumMixing(),
                 scheduler=Molly.DefaultLambdaScheduler())

An interaction between three atoms in an alchemical system consisting of a harmonic bond
angle and a harmonic bond between the outer atoms, the λ version of [`UreyBradley`](@ref).

[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref) convert each [`UreyBradley`](@ref)
to this type.
The potential energy is defined as
```math
V(\theta, r) = w \left( \frac{1}{2} k_a (\theta - \theta_0)^2 + \frac{1}{2} k_b (r - r_0)^2 \right)
```
where `θ0` and the atom order are as for [`UreyBradley`](@ref), and the weight ``w`` and the
parameters depend on the alchemical roles of the atoms and on ``\lambda``, the `λ` of the
atoms mixed with `λ_mixing`.
`scheduler` is a [`Molly.LambdaScheduler`](@ref), which selects dual or single topology.

In dual topology `kangle`, `θ0`, `kbond` and `r0` are single values.
``w = 1`` if the interaction involves an atom unique to one state or only environment atoms,
otherwise ``w = 1 - \lambda`` if it involves a core atom with the parameters of state A and
``w = \lambda`` if it involves a core atom with the parameters of state B.

In single topology ``w = 1`` and `kangle`, `θ0`, `kbond` and `r0` are tuples of the values in
states A and B.
Each parameter ``p`` is ``(1 - \lambda) p^A + \lambda p^B`` if the interaction involves a core
or unique atom and ``p^A`` otherwise.
"""
@kwdef struct UreyBradleyλ{KA, A, KB, D, LM, SCH} <: AlchemicalBondedInteraction
    kangle::KA
    θ0::A
    kbond::KB
    r0::D
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

parameter_prefix(::UreyBradleyλ, inter_type) = "inter_UB_$(inter_type)_"
parameter_fields(::Type{<:UreyBradleyλ}) = parameter_fields(UreyBradley)

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

# Single topology: each of the four parameters is interpolated between the end states, the angle as
#   GROMACS and OpenFE do for angles and the 1-3 distance as they do for bonds
function to_lambda_function_single(interA::UreyBradley, interB::Nothing;
                                  λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return UreyBradleyλ(kangle=(interA.kangle, interA.kangle), θ0=(interA.θ0, interA.θ0),
                        kbond=(interA.kbond, interA.kbond), r0=(interA.r0, interA.r0),
                        λ_mixing=λ_mixing, scheduler=scheduler)
end

function to_lambda_function_single(interA::Nothing, interB::UreyBradley;
                                  λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return UreyBradleyλ(kangle=(interB.kangle, interB.kangle), θ0=(interB.θ0, interB.θ0),
                        kbond=(interB.kbond, interB.kbond), r0=(interB.r0, interB.r0),
                        λ_mixing=λ_mixing, scheduler=scheduler)
end

function update_lambda_function(existing_lambda::UreyBradleyλ, interB::UreyBradley)
    return UreyBradleyλ(kangle=(existing_lambda.kangle[1], interB.kangle),
                        θ0=(existing_lambda.θ0[1], interB.θ0),
                        kbond=(existing_lambda.kbond[1], interB.kbond),
                        r0=(existing_lambda.r0[1], interB.r0),
                        λ_mixing=existing_lambda.λ_mixing, scheduler=existing_lambda.scheduler)
end

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
