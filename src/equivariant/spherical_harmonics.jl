# Real spherical harmonics up to l_max = 2 for equivariant potentials.
#
# We use the "component"-normalized real spherical harmonics: on the unit sphere
# Σ_m Y_lm(r̂)^2 = 2l+1. These equal the real regular solid harmonics (homogeneous degree-l
# polynomials in the unit vector) with fixed per-l constants, and are written directly as
# polynomials so their gradients are analytic — needed for analytic forces.
#
# Axis/normalization convention: matches e3nn's real spherical harmonics
# (`o3.spherical_harmonics(l, x, normalize=true, normalization="component")`, verified against
# e3nn 0.6.0 to ~1e-15). l=1 is (x, y, z); l=2 is
# [√15·xz, √15·xy, (√5/2)(2y²−x²−z²), √15·yz, (√15/2)(z²−x²)]. This is internally consistent with
# the Clebsch-Gordan coefficients in clebsch_gordan.jl (both derive from the same real transform),
# so the tensor product is exactly equivariant, and the bit-match to e3nn lets trained weights load.
#
# Pure StaticArrays maths, no Lux/HDF5 — lives in core Molly. Internal (unexported).

const SH_MAX_L = 2  # highest supported l in this first implementation

"Length of the concatenated real-SH vector for orders l = 0:lmax, i.e. Σ (2l+1) = (lmax+1)^2."
sph_harm_length(lmax::Integer) = (lmax + 1)^2

# Component-normalization constants (verified numerically against Σ_m Y_lm^2 = 2l+1).
const C1 = sqrt(3.0)          # l=1 prefactor
const C2A = sqrt(15.0)        # l=2 off-diagonal (xy, yz, xz)
const C2B = 0.5 * sqrt(5.0)   # l=2 m=0
const C2C = 0.5 * sqrt(15.0)  # l=2 m=±2

# The homogeneous degree-l polynomials P_l^m(v) with Y_lm(r̂) = P_l^m(r̂). Written for a general
# vector v = (x, y, z); on the unit sphere they give the component-normalized real SH.

@inline function poly_l1(x::T, y::T, z::T) where T
    c = T(C1)
    return (c * x, c * y, c * z)                       # m = -1, 0, +1  ↔  x, y, z
end

# e3nn l=2 basis (component-normalized): [√15·xz, √15·xy, (√5/2)(2y²−x²−z²), √15·yz, (√15/2)(z²−x²)].
@inline function poly_l2(x::T, y::T, z::T) where T
    ca, cb, cc = T(C2A), T(C2B), T(C2C)
    return (ca * x * z,                                # [0]  √15·xz
            ca * x * y,                                # [1]  √15·xy
            cb * (2y * y - x * x - z * z),             # [2]  (√5/2)(2y²−x²−z²)
            ca * y * z,                                # [3]  √15·yz
            cc * (z * z - x * x))                      # [4]  (√15/2)(z²−x²)
end

"""
    real_sph_harm(lmax, r)

Component-normalized real spherical harmonics for orders `l = 0:lmax` evaluated at the direction
`r/‖r‖`, returned as a length-`(lmax+1)^2` `SVector` (blocks concatenated in l order, `m = -l:l`
within each block). `r` is an `SVector{3}`; only its direction matters. Supports `lmax ≤ 2`.
"""
@inline function real_sph_harm(lmax::Integer, r::SVector{3,T}) where T
    lmax <= SH_MAX_L || throw(ArgumentError("real_sph_harm supports lmax ≤ $SH_MAX_L, got $lmax"))
    d = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
    invd = inv(d)
    x, y, z = r[1] * invd, r[2] * invd, r[3] * invd
    if lmax == 0
        return SVector{1,T}(one(T))
    elseif lmax == 1
        p1 = poly_l1(x, y, z)
        return SVector{4,T}(one(T), p1[1], p1[2], p1[3])
    else
        p1 = poly_l1(x, y, z)
        p2 = poly_l2(x, y, z)
        return SVector{9,T}(one(T),
                            p1[1], p1[2], p1[3],
                            p2[1], p2[2], p2[3], p2[4], p2[5])
    end
end

# Concrete `lmax = 2` real-SH value + Jacobian, written out element by element (no closures, no
# runtime-`lmax` union return) so it const-folds and runs inside GPU kernels on CUDA and Metal. The
# rows are `∂Y_i/∂r = (∇P_i(r̂) − l·Y_i·r̂)/d`; `J` is assembled column-major (∂/∂x, ∂/∂y, ∂/∂z).
@inline function _real_sph_harm_grad2(r::SVector{3,T}) where T
    d = sqrt(r[1]^2 + r[2]^2 + r[3]^2); invd = inv(d)
    x = r[1] * invd; y = r[2] * invd; z = r[3] * invd
    c1 = T(C1); ca = T(C2A); cb = T(C2B); cc = T(C2C); c5 = T(sqrt(5.0))  # c5 = 2cb, ca = 2cc
    Y2 = c1 * x;           Y3 = c1 * y;           Y4 = c1 * z
    Y5 = ca * x * z;       Y6 = ca * x * y;       Y7 = cb * (2y * y - x * x - z * z)
    Y8 = ca * y * z;       Y9 = cc * (z * z - x * x)
    Y = SVector{9,T}(one(T), Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9)
    r2x = (c1 - Y2 * x) * invd; r2y = (   - Y2 * y) * invd; r2z = (   - Y2 * z) * invd
    r3x = (   - Y3 * x) * invd; r3y = (c1 - Y3 * y) * invd; r3z = (   - Y3 * z) * invd
    r4x = (   - Y4 * x) * invd; r4y = (   - Y4 * y) * invd; r4z = (c1 - Y4 * z) * invd
    r5x = (ca * z - 2Y5 * x) * invd; r5y = (       - 2Y5 * y) * invd; r5z = (ca * x - 2Y5 * z) * invd
    r6x = (ca * y - 2Y6 * x) * invd; r6y = (ca * x - 2Y6 * y) * invd; r6z = (       - 2Y6 * z) * invd
    r7x = (-c5 * x - 2Y7 * x) * invd; r7y = (2c5 * y - 2Y7 * y) * invd; r7z = (-c5 * z - 2Y7 * z) * invd
    r8x = (       - 2Y8 * x) * invd; r8y = (ca * z - 2Y8 * y) * invd; r8z = (ca * y - 2Y8 * z) * invd
    r9x = (-ca * x - 2Y9 * x) * invd; r9y = (       - 2Y9 * y) * invd; r9z = (ca * z - 2Y9 * z) * invd
    J = SMatrix{9,3,T}(zero(T), r2x, r3x, r4x, r5x, r6x, r7x, r8x, r9x,
                       zero(T), r2y, r3y, r4y, r5y, r6y, r7y, r8y, r9y,
                       zero(T), r2z, r3z, r4z, r5z, r6z, r7z, r8z, r9z)
    return Y, J
end

"""
    real_sph_harm_grad(lmax, r) -> (Y, J)

Real spherical harmonics `Y` (as in [`real_sph_harm`](@ref)) together with the Jacobian
`J[i, b] = ∂Y[i]/∂r_b` with respect to the raw (unnormalized) vector `r`. Using the homogeneous
polynomials `P` and `d = ‖r‖`, `∂Y_l/∂r = (1/d)·(∇P_l(r̂) − l·Y_l(r̂)·r̂)`. Returns `Y::SVector`
and `J::SMatrix{(lmax+1)^2, 3}`. Supports `lmax ≤ 2`.
"""
@inline function real_sph_harm_grad(lmax::Integer, r::SVector{3,T}) where T
    lmax <= SH_MAX_L || throw(ArgumentError("real_sph_harm_grad supports lmax ≤ $SH_MAX_L, got $lmax"))
    d = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
    invd = inv(d)
    rh = r * invd                       # unit vector
    x, y, z = rh[1], rh[2], rh[3]

    if lmax == 0
        Y = SVector{1,T}(one(T))
        J = @SMatrix zeros(T, 1, 3)
        return Y, J
    end

    # l=1: P = c1·(x,y,z). ∂Y_l/∂r = (∇P_l(r̂) − l·Y_l·r̂)/d; the l=0 row is zero. Built column-major
    # and element by element (no closures) so it const-folds and runs inside GPU kernels.
    c1 = T(C1)
    p1 = poly_l1(x, y, z)

    if lmax == 1
        Y2 = p1[1]; Y3 = p1[2]; Y4 = p1[3]
        Y = SVector{4,T}(one(T), Y2, Y3, Y4)
        r2x = (c1 - Y2 * x) * invd; r2y = (   - Y2 * y) * invd; r2z = (   - Y2 * z) * invd
        r3x = (   - Y3 * x) * invd; r3y = (c1 - Y3 * y) * invd; r3z = (   - Y3 * z) * invd
        r4x = (   - Y4 * x) * invd; r4y = (   - Y4 * y) * invd; r4z = (c1 - Y4 * z) * invd
        J = SMatrix{4,3,T}(zero(T), r2x, r3x, r4x,
                           zero(T), r2y, r3y, r4y,
                           zero(T), r2z, r3z, r4z)
        return Y, J
    end

    # l=2 — the concrete, GPU-safe path (also the one the analytic forces use).
    return _real_sph_harm_grad2(r)
end
