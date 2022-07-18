export HarmonicAngle

"""
    HarmonicAngle(; th0, cth)

A harmonic bond angle between three atoms.
"""
struct HarmonicAngle{D, K} <: SpecificInteraction
    th0::D
    cth::K
end

HarmonicAngle(; th0, cth) = HarmonicAngle{typeof(th0), typeof(cth)}(th0, cth)

@inline @inbounds function force(a::HarmonicAngle, coords_i, coords_j, coords_k, boundary)
    # In 2D we use then eliminate the cross product
    ba = vector_pad3D(coords_j, coords_i, boundary)
    bc = vector_pad3D(coords_j, coords_k, boundary)
    cross_ba_bc = ba × bc
    if iszero(cross_ba_bc)
        zf = zero(a.cth ./ trim3D(ba))
        return SpecificForce3Atoms(zf, zf, zf)
    end
    pa = normalize(trim3D( ba × cross_ba_bc, boundary))
    pc = normalize(trim3D(-bc × cross_ba_bc, boundary))
    angle_term = -a.cth * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.th0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return SpecificForce3Atoms(fa, fb, fc)
end

@inline @inbounds function potential_energy(a::HarmonicAngle, coords_i, coords_j,
                                            coords_k, boundary)
    θ = bond_angle(coords_i, coords_j, coords_k, boundary)
    return (a.cth / 2) * (θ - a.th0) ^ 2
end
