# Real spherical harmonics up to l_max = 2 for equivariant potentials.
#
# We use the "component"-normalized real spherical harmonics: on the unit sphere
# Σ_m Y_lm(r̂)^2 = 2l+1. These equal the real regular solid harmonics (homogeneous degree-l
# polynomials in the unit vector) with fixed per-l constants, and are written directly as
# polynomials so their gradients are analytic — needed for analytic forces.
#
# Axis/normalization convention: components are ordered by m = -l, …, +l in the standard real-SH
# xyz convention (l=1 → (x, y, z)). This is internally self-consistent with the Clebsch-Gordan
# coefficients derived in clebsch_gordan.jl (both are built from the same real transform), so the
# tensor product is exactly equivariant. Matching e3nn's exact convention (its l=1 → (y, z, x)
# axis order) is verified separately by a reference bit-match test and is a prerequisite for
# loading e3nn-trained weights; see test/allegro_reference.py.
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

@inline function _P1(x::T, y::T, z::T) where T
    c = T(C1)
    return (c * x, c * y, c * z)                       # m = -1, 0, +1  ↔  x, y, z
end

@inline function _P2(x::T, y::T, z::T) where T
    ca, cb, cc = T(C2A), T(C2B), T(C2C)
    return (ca * x * y,                                # m = -2  ↔  xy
            ca * y * z,                                # m = -1  ↔  yz
            cb * (2z * z - x * x - y * y),             # m =  0  ↔  (3z²−r²)/… homogeneous form
            ca * x * z,                                # m = +1  ↔  xz
            cc * (x * x - y * y))                      # m = +2  ↔  x²−y²
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
        p1 = _P1(x, y, z)
        return SVector{4,T}(one(T), p1[1], p1[2], p1[3])
    else
        p1 = _P1(x, y, z)
        p2 = _P2(x, y, z)
        return SVector{9,T}(one(T),
                            p1[1], p1[2], p1[3],
                            p2[1], p2[2], p2[3], p2[4], p2[5])
    end
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

    # Gradients of the homogeneous polynomials w.r.t. the (unit) components, ∇P_l(r̂).
    c1 = T(C1)
    # l=1: P = c1*(x,y,z) ⇒ ∇ = c1*I
    gP1 = (SVector{3,T}(c1, 0, 0), SVector{3,T}(0, c1, 0), SVector{3,T}(0, 0, c1))
    p1 = _P1(x, y, z)

    if lmax == 1
        Y = SVector{4,T}(one(T), p1[1], p1[2], p1[3])
        # ∂Y_l/∂r = (1/d)(∇P - l Y r̂); l=0 row is zero, l=1 rows use l=1.
        rows = ntuple(4) do i
            if i == 1
                SVector{3,T}(0, 0, 0)
            else
                g = gP1[i - 1]
                (g - one(T) * Y[i] * rh) * invd
            end
        end
        J = SMatrix{4,3,T}(vcat((r' for r in rows)...))
        return Y, J
    end

    # l=2
    ca, cb, cc = T(C2A), T(C2B), T(C2C)
    p2 = _P2(x, y, z)
    # ∇P2 for each m (degree-1 polynomials in x,y,z):
    gP2 = (
        SVector{3,T}(ca * y, ca * x, 0),                       # xy
        SVector{3,T}(0, ca * z, ca * y),                       # yz
        SVector{3,T}(cb * (-2x), cb * (-2y), cb * (4z)),       # 2z²−x²−y²
        SVector{3,T}(ca * z, 0, ca * x),                       # xz
        SVector{3,T}(cc * (2x), cc * (-2y), 0),                # x²−y²
    )
    Y = SVector{9,T}(one(T), p1[1], p1[2], p1[3], p2[1], p2[2], p2[3], p2[4], p2[5])
    rows = ntuple(9) do i
        if i == 1
            SVector{3,T}(0, 0, 0)
        elseif i <= 4
            g = gP1[i - 1]
            (g - one(T) * Y[i] * rh) * invd
        else
            g = gP2[i - 4]
            (g - T(2) * Y[i] * rh) * invd
        end
    end
    J = SMatrix{9,3,T}(vcat((r' for r in rows)...))
    return Y, J
end
