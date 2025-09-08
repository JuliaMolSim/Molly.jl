export RBTorsion

"""
    RBTorsion(; f1, f2, f3, f4)

A Ryckaert-Bellemans torsion angle between four atoms.

Only compatible with 3D systems.
"""
struct RBTorsion{T}
    f1::T
    f2::T
    f3::T
    f4::T
end

RBTorsion(; f1, f2, f3, f4) = RBTorsion{typeof(f1)}(f1, f2, f3, f4)

@inline function force(d::RBTorsion, coords_i, coords_j, coords_k, coords_l, boundary, args...)
    ab = vector(coords_i, coords_j, boundary)
    bc = vector(coords_j, coords_k, boundary)
    cd = vector(coords_k, coords_l, boundary)
    cross_ab_bc = ab × bc
    cross_bc_cd = bc × cd
    bc_norm = norm(bc)
    θ = atan(
        ustrip(dot(cross_ab_bc × cross_bc_cd, bc / bc_norm)),
        ustrip(dot(cross_ab_bc, cross_bc_cd)),
    )
    dEdθ = (d.f1*sin(θ) - 2*d.f2*sin(2*θ) + 3*d.f3*sin(3*θ)) / 2
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
    return (d.f1 * (1 + cos(θ)) + d.f2 * (1 - cos(2θ)) + d.f3 * (1 + cos(3θ)) + d.f4) / 2
end
