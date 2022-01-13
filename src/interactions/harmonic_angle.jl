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

@inline @inbounds function force(a::HarmonicAngle, coords_i, coords_j, coords_k, box_size)
    # In 2D we use then eliminate the cross product
    ba = vector_pad3D(coords_j, coords_i, box_size)
    bc = vector_pad3D(coords_j, coords_k, box_size)
    cross_ba_bc = ba × bc
    pa = normalize(trim3D( ba × cross_ba_bc, box_size))
    pc = normalize(trim3D(-bc × cross_ba_bc, box_size))
    angle_term = -a.cth * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.th0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return SpecificForce3Atoms(fa, fb, fc)
end

@inline @inbounds function potential_energy(a::HarmonicAngle, coords_i, coords_j,
                                            coords_k, box_size)
    θ = bond_angle(coords_i, coords_j, coords_k, box_size)
    return (a.cth / 2) * (θ - a.th0) ^ 2
end
