# Radial edge features: a Bessel basis with a smooth polynomial envelope, plus analytic
# derivatives (needed for analytic forces). Matches the nequip/DimeNet conventions so trained
# Allegro radial weights transfer. Pure maths, no Lux/HDF5 — core Molly. Internal (unexported).

"""
    bessel_basis(r, r_c, n_basis) -> SVector

The (sine) Bessel radial basis `B_n(r) = √(2/r_c)·sin(nπ r/r_c)/r`, for `n = 1:n_basis`, as used
by nequip/DimeNet. `r` and `r_c` are in the same length unit.
"""
@inline function bessel_basis(r::T, r_c::T, ::Val{N}) where {T,N}
    pref = sqrt(T(2) / r_c)
    invr = inv(r)
    k = T(π) / r_c
    return SVector{N,T}(ntuple(n -> pref * sin(n * k * r) * invr, Val(N)))
end

"""
    bessel_basis_grad(r, r_c, Val(N)) -> (B, dB)

Bessel basis `B` and its derivative `dB_n/dr`. `dB_n/dr = √(2/r_c)·[(nπ/r_c)cos(nπ r/r_c)/r −
sin(nπ r/r_c)/r²]`.
"""
@inline function bessel_basis_grad(r::T, r_c::T, ::Val{N}) where {T,N}
    pref = sqrt(T(2) / r_c)
    invr = inv(r)
    invr2 = invr * invr
    k = T(π) / r_c
    B = SVector{N,T}(ntuple(n -> pref * sin(n * k * r) * invr, Val(N)))
    dB = SVector{N,T}(ntuple(n -> pref * (n * k * cos(n * k * r) * invr - sin(n * k * r) * invr2), Val(N)))
    return B, dB
end

"""
    poly_envelope(r, r_c, p=6)

DimeNet polynomial cutoff envelope `u(r)`, smooth (C²) at `r = r_c` and zero beyond:
`u(d) = 1 − ((p+1)(p+2)/2)dᵖ + p(p+2)dᵖ⁺¹ − (p(p+1)/2)dᵖ⁺²`, with `d = r/r_c`. Returns 0 for
`r ≥ r_c`.
"""
@inline function poly_envelope(r::T, r_c::T, p::Int=6) where T
    r < r_c || return zero(T)
    d = r / r_c
    a = T((p + 1) * (p + 2) ÷ 2)
    b = T(p * (p + 2))
    c = T(p * (p + 1) ÷ 2)
    return one(T) - a * d^p + b * d^(p + 1) - c * d^(p + 2)
end

"""
    poly_envelope_grad(r, r_c, p=6) -> (u, du)

Envelope `u(r)` and its derivative `du/dr`. Both are zero for `r ≥ r_c`.
"""
@inline function poly_envelope_grad(r::T, r_c::T, p::Int=6) where T
    r < r_c || return (zero(T), zero(T))
    d = r / r_c
    a = T((p + 1) * (p + 2) ÷ 2)
    b = T(p * (p + 2))
    c = T(p * (p + 1) ÷ 2)
    u = one(T) - a * d^p + b * d^(p + 1) - c * d^(p + 2)
    du = (-a * p * d^(p - 1) + b * (p + 1) * d^p - c * (p + 2) * d^(p + 1)) / r_c
    return u, du
end

"""
    radial_embedding(r, r_c, Val(N), p=6) -> (R, dR)

Enveloped Bessel radial embedding `R_n(r) = B_n(r)·u(r)` and its derivative `dR_n/dr`
(product rule), the actual radial feature fed to the Allegro latent MLP.
"""
@inline function radial_embedding(r::T, r_c::T, ::Val{N}, p::Int=6) where {T,N}
    B, dB = bessel_basis_grad(r, r_c, Val(N))
    u, du = poly_envelope_grad(r, r_c, p)
    R = B .* u
    dR = dB .* u .+ B .* du
    return R, dR
end
