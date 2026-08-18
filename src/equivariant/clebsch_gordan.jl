# Real Clebsch-Gordan (Wigner-3j-like) coupling coefficients for the equivariant tensor product.
#
# Strategy for guaranteed self-consistency with the real spherical harmonics in
# spherical_harmonics.jl: we build the *complex* SH (component-normalized) and the standard
# *complex* Clebsch-Gordan coefficients (both textbook, mutually consistent), then derive, once at
# setup in Float64, the unitary change-of-basis U_l that maps our complex SH to our real SH
# (`real Y = U_l · complex Y`), and transform the complex CG into the real basis:
#
#   CGʳ_{M1,M2,M3} = Σ_{m1,m2,m3} U3_{M3,m3} · conj(U1_{M1,m1}) · conj(U2_{M2,m2}) · CGᶜ_{m1,m2,m3}
#
# Because both the real SH and the real CG come from the *same* U_l, the tensor product built from
# CGʳ is exactly equivariant with respect to those real SH. Only nonzero coefficients are stored.
#
# Pure maths (StaticArrays/LinearAlgebra), no Lux/HDF5 — core Molly. Internal (unexported).

# ---- complex spherical harmonics (component-normalized), l = 0,1,2 --------------------------------

# Component normalization multiplies the orthonormal complex SH by √(4π), giving Σ_m |Y|² = 2l+1
# and making U_l unitary. Written for a unit vector (x, y, z) with the Condon-Shortley phase.
function _complex_sph_harm(l::Int, x::Float64, y::Float64, z::Float64)
    if l == 0
        return ComplexF64[1.0]
    elseif l == 1
        a = sqrt(1.5)  # √(3/2)
        return ComplexF64[ a * (x - im * y),          # m = -1:  +√(3/2)(x−iy)
                           sqrt(3.0) * z,              # m =  0
                          -a * (x + im * y) ]          # m = +1: −√(3/2)(x+iy)
    elseif l == 2
        b1 = sqrt(15.0 / 8.0)  # for m=±2
        b2 = sqrt(15.0 / 2.0)  # for m=±1
        c0 = 0.5 * sqrt(5.0)
        xy2p = (x + im * y)^2
        xy2m = (x - im * y)^2
        return ComplexF64[  b1 * xy2m,                 # m = -2
                            b2 * (x - im * y) * z,      # m = -1
                            c0 * (2z*z - x*x - y*y),    # m =  0  (= √5/2·(3z²−1) on unit sphere)
                           -b2 * (x + im * y) * z,      # m = +1
                            b1 * xy2p ]                 # m = +2
    else
        throw(ArgumentError("complex SH implemented only for l ≤ 2"))
    end
end

# The matching real SH (component-normalized), evaluated at a unit vector — mirrors
# spherical_harmonics.jl but in plain Float64 for the setup-time U_l fit.
function _real_sph_harm(l::Int, x::Float64, y::Float64, z::Float64)
    if l == 0
        return [1.0]
    elseif l == 1
        return [C1 * x, C1 * y, C1 * z]
    elseif l == 2
        return [C2A * x * y, C2A * y * z, C2B * (2z*z - x*x - y*y), C2A * x * z, C2C * (x*x - y*y)]
    else
        throw(ArgumentError("real SH implemented only for l ≤ 2"))
    end
end

# Unitary change of basis U_l with real Y = U_l · complex Y, recovered by least squares over a
# set of sample directions (exact in Float64 since the two bases are related by a fixed unitary).
function _real_transform(l::Int)
    d = 2l + 1
    # Deterministic, well-spread sample directions (≥ d of them).
    dirs = NTuple{3,Float64}[]
    for i in 1:(2d + 3)
        # golden-spiral points on the sphere (no RNG — reproducible)
        t = (i - 0.5) / (2d + 3)
        zc = 1 - 2t
        rho = sqrt(max(0.0, 1 - zc^2))
        phi = 2π * i * 0.6180339887498949
        push!(dirs, (rho * cos(phi), rho * sin(phi), zc))
    end
    K = length(dirs)
    R = Matrix{ComplexF64}(undef, d, K)  # real SH samples
    C = Matrix{ComplexF64}(undef, d, K)  # complex SH samples
    for (k, (x, y, z)) in enumerate(dirs)
        R[:, k] = _real_sph_harm(l, x, y, z)
        C[:, k] = _complex_sph_harm(l, x, y, z)
    end
    # real = U * complex  ⇒  U = R * pinv(C)
    U = R * pinv(C)
    return U
end

# ---- complex Clebsch-Gordan closed form ----------------------------------------------------------

_lfact(n::Int) = n < 0 ? Inf : Float64(factorial(big(n)))

# <j1 m1 j2 m2 | j3 m3>, standard Condon-Shortley convention.
function _cg_complex(j1, m1, j2, m2, j3, m3)
    (m1 + m2 == m3) || return 0.0
    (abs(j1 - j2) <= j3 <= j1 + j2) || return 0.0
    (abs(m1) <= j1 && abs(m2) <= j2 && abs(m3) <= j3) || return 0.0

    pref = sqrt((2j3 + 1) *
                _lfact(j3 + j1 - j2) * _lfact(j3 - j1 + j2) * _lfact(j1 + j2 - j3) /
                _lfact(j1 + j2 + j3 + 1))
    pref *= sqrt(_lfact(j3 + m3) * _lfact(j3 - m3) *
                 _lfact(j1 - m1) * _lfact(j1 + m1) *
                 _lfact(j2 - m2) * _lfact(j2 + m2))
    s = 0.0
    for k in 0:(j1 + j2 + j3)
        a1 = j1 + j2 - j3 - k
        a2 = j1 - m1 - k
        a3 = j2 + m2 - k
        a4 = j3 - j2 + m1 + k
        a5 = j3 - j1 - m2 + k
        (k >= 0 && a1 >= 0 && a2 >= 0 && a3 >= 0 && a4 >= 0 && a5 >= 0) || continue
        s += (-1)^k / (_lfact(k) * _lfact(a1) * _lfact(a2) * _lfact(a3) * _lfact(a4) * _lfact(a5))
    end
    return pref * s
end

# ---- sparse real CG for one (l1, l2, l3) coupling ------------------------------------------------

"""
    SparseCG

Nonzero real Clebsch-Gordan coefficients for a set of tensor-product paths, stored flat for
kernel-friendly iteration. For path `p`, the coefficients occupy `range(p) = (poff[p]+1):poff[p+1]`
of the `(m1, m2, m3, val)` arrays. Indices `m1, m2, m3` are 1-based component indices into the
`2l+1`-dim blocks of `in1`, `in2`, `out` respectively. `val` is real (Float64, cast at use).
"""
struct SparseCG{T}
    m1::Vector{Int32}
    m2::Vector{Int32}
    m3::Vector{Int32}
    val::Vector{T}
    poff::Vector{Int}   # length n_paths+1; path p spans poff[p]+1 : poff[p+1]
end

# Real CG for one coupling, as a dense (d1, d2, d3) array (component indices 1-based).
function _real_cg_dense(l1::Int, l2::Int, l3::Int)
    d1, d2, d3 = 2l1 + 1, 2l2 + 1, 2l3 + 1
    U1, U2, U3 = _real_transform(l1), _real_transform(l2), _real_transform(l3)
    # complex CG tensor in component-index form
    Cc = zeros(ComplexF64, d1, d2, d3)
    for (i1, mm1) in enumerate(-l1:l1), (i2, mm2) in enumerate(-l2:l2), (i3, mm3) in enumerate(-l3:l3)
        Cc[i1, i2, i3] = _cg_complex(l1, mm1, l2, mm2, l3, mm3)
    end
    Cr = zeros(ComplexF64, d1, d2, d3)
    for M1 in 1:d1, M2 in 1:d2, M3 in 1:d3
        acc = 0.0 + 0.0im
        for m1 in 1:d1, m2 in 1:d2, m3 in 1:d3
            v = Cc[m1, m2, m3]
            v == 0 && continue
            acc += U3[M3, m3] * conj(U1[M1, m1]) * conj(U2[M2, m2]) * v
        end
        Cr[M1, M2, M3] = acc
    end
    # The naive complex→real transform is purely imaginary when l1+l2+l3 is odd; the standard
    # i^(l1+l2+l3) phase makes the real Wigner-3j real in every case. A global real rescaling of a
    # path's coefficients preserves equivariance (it is absorbed by the layer weights).
    Cr .*= im^(l1 + l2 + l3)
    maximum(abs.(imag.(Cr))) < 1e-9 ||
        error("real CG has nonzero imaginary part (max $(maximum(abs.(imag.(Cr))))) — convention bug")
    return real.(Cr)
end

"""
    build_sparse_cg(paths; T=Float32, tol=1e-10)

Build the [`SparseCG`](@ref) table for the couplings in a `TensorProductPaths`, keeping only
coefficients with `|val| > tol`. Coefficients are computed in Float64 and cast to `T`.
"""
function build_sparse_cg(paths::TensorProductPaths; T::Type=Float32, tol::Float64=1e-10)
    m1s, m2s, m3s = Int32[], Int32[], Int32[]
    vals = T[]
    poff = Int[0]
    # cache dense CG per (l1,l2,l3)
    cache = Dict{NTuple{3,Int},Array{Float64,3}}()
    for p in eachindex(paths.l)
        l1, l2, l3 = paths.l[p]
        dense = get!(cache, (l1, l2, l3)) do
            _real_cg_dense(l1, l2, l3)
        end
        d1, d2, d3 = size(dense)
        for i1 in 1:d1, i2 in 1:d2, i3 in 1:d3
            v = dense[i1, i2, i3]
            abs(v) > tol || continue
            push!(m1s, i1); push!(m2s, i2); push!(m3s, i3); push!(vals, T(v))
        end
        push!(poff, length(vals))
    end
    return SparseCG{T}(m1s, m2s, m3s, vals, poff)
end
