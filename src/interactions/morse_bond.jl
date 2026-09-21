export MorseBond

@doc raw"""
    MorseBond(; D, a, r0)

A Morse potential bond between two atoms.

The potential energy is defined as
```math
V(r) = D(1 - e^{-a(r - r_0)})^2
```
"""
@kwdef struct MorseBond{T, A, R}
    D::T
    a::A
    r0::R
end

Base.zero(::Type{MorseBond{T, A, R}}) where {T, A, R} = MorseBond(D=zero(T), a=zero(A), r0=zero(R))
Base.zero(b::MorseBond) = zero(typeof(b))

Base.:+(b1::MorseBond, b2::MorseBond) = MorseBond(D=(b1.D + b2.D), a=(b1.a + b2.a),
                                                  r0=(b1.r0 + b2.r0))

parameter_prefix(::MorseBond, inter_type) = "inter_MB_$(inter_type)_"
parameter_fields(::Type{<:MorseBond}) = ((:D, "D"), (:a, "a"), (:r0, "r0"))


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

# λ version of `MorseBond` for alchemical systems, built by `to_lambda_function`.
@kwdef struct MorseBondλ{T, A, R, LM, SCH}
    D::T
    a::A
    r0::R
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
end

Base.zero(::MorseBondλ{T, A, R, LM, SCH}) where {T, A, R, LM, SCH} = MorseBondλ(D=zero(T), a=zero(A), r0=zero(R))

Base.:+(b1::MorseBondλ, b2::MorseBondλ) = MorseBondλ(D=(b1.D + b2.D), a=(b1.a + b2.a),
                                                  r0=(b1.r0 + b2.r0))


function to_lambda_function(inter::MorseBond; λ_mixing=MinimumMixing(), scheduler=DefaultLambdaScheduler())
    return MorseBondλ(D=inter.D, a=inter.a, r0=inter.r0, λ_mixing=λ_mixing, scheduler=scheduler)
end

@inline function force(b::MorseBondλ, coord_i, coord_j, boundary, atom_i, atom_j, args...)
    T = typeof(ustrip(atom_i.λ))
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    λ_glob = T(λ_mixing(b.λ_mixing, (atom_i.λ, atom_j.λ)))
    pair_role = mix_roles(b.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_dual(b.scheduler, λ_glob, pair_role)
    D = params_mixing(λ_params, b.D)
    a = params_mixing(λ_params, b.a)
    r0 = params_mixing(λ_params, b.r0)
    ralp = exp(-a * (r - r0))
    c = 2 * D * a * (1 - ralp) * ralp
    f = c * normalize(dr)
    return SpecificForce2Atoms(λ*f, λ*-f)
end

@inline function potential_energy(b::MorseBondλ, coord_i, coord_j, boundary, atom_i, atom_j,
                                  args...)
    T = typeof(ustrip(atom_i.λ))
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    λ_glob = T(λ_mixing(b.λ_mixing, (atom_i.λ, atom_j.λ)))
    pair_role = mix_roles(b.scheduler, (atom_i.alch_role, atom_j.alch_role))
    λ, λ_params = scale_dual(b.scheduler, λ_glob, pair_role)
    D = params_mixing(λ_params, b.D)
    a = params_mixing(λ_params, b.a)
    r0 = params_mixing(λ_params, b.r0)
    ralp = exp(-a * (r - r0))
    return λ * (D * (1 - ralp)^2)
end