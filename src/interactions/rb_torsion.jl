"""
    RBTorsion(; i, j, k, l, f1, f2, f3, f4)

A Ryckaert-Bellemans torsion angle between four atoms.
"""
struct RBTorsion{T} <: SpecificInteraction
    i::Int
    j::Int
    k::Int
    l::Int
    f1::T
    f2::T
    f3::T
    f4::T
end

RBTorsion(; i, j, k, l, f1, f2, f3, f4) = RBTorsion{typeof(f1)}(i, j, k, l, f1, f2, f3, f4)

@inline @inbounds function force(d::RBTorsion,
                                  coords,
                                  s::System)
    ab = vector(coords[d.i], coords[d.j], s.box_size)
    bc = vector(coords[d.j], coords[d.k], s.box_size)
    cd = vector(coords[d.k], coords[d.l], s.box_size)
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
    return [d.i, d.j, d.k, d.l], [fi, fj, fk, fl]
end

@inline @inbounds function potential_energy(d::RBTorsion,
                                            s::System)
    ab = vector(s.coords[d.i], s.coords[d.j], s.box_size)
    bc = vector(s.coords[d.j], s.coords[d.k], s.box_size)
    cd = vector(s.coords[d.k], s.coords[d.l], s.box_size)
    cross_ab_bc = ab × bc
    cross_bc_cd = bc × cd
    θ = atan(
        ustrip(dot(cross_ab_bc × cross_bc_cd, normalize(bc))),
        ustrip(dot(cross_ab_bc, cross_bc_cd)),
    )
    return (d.f1 * (1 + cos(θ)) + d.f2 * (1 - cos(2θ)) + d.f3 * (1 + cos(3θ)) + d.f4) / 2
end
