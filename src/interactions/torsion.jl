"""
    Torsion(i, j, k, l, f1, f2, f3, f4)

A dihedral torsion angle between four atoms.
"""
struct Torsion{T} <: SpecificInteraction
    i::Int
    j::Int
    k::Int
    l::Int
    f1::T
    f2::T
    f3::T
    f4::T
end

@inline @inbounds function force(d::Torsion,
                                  coords,
                                  s::Simulation)
    ba = vector(coords[d.j], coords[d.i], s.box_size)
    bc = vector(coords[d.j], coords[d.k], s.box_size)
    dc = vector(coords[d.l], coords[d.k], s.box_size)
    p1 = normalize(ba × bc)
    p2 = normalize(-dc × -bc)
    θ = atan(dot((-ba × bc) × (bc × -dc), normalize(bc)), dot(-ba × bc, bc × -dc))
    angle_term = (d.f1*sin(θ) - 2*d.f2*sin(2*θ) + 3*d.f3*sin(3*θ)) / 2
    fa = (angle_term / (norm(ba) * sin(acosbound(dot(ba, bc) / (norm(ba) * norm(bc)))))) * p1
    # fd clashes with a function name
    f_d = (angle_term / (norm(dc) * sin(acosbound(dot(bc, dc) / (norm(bc) * norm(dc)))))) * p2
    oc = bc / 2
    tc = -(oc × f_d + (-dc × f_d) / 2 + (ba × fa) / 2)
    fc = (1 / dot(oc, oc)) * (tc × oc)
    fb = -fa - fc - f_d
    return [d.i, d.j, d.k, d.l], [fa, fb, fc, fd]
end

@inline @inbounds function potential_energy(d::Torsion,
                                            s::Simulation)
    ba = vector(s.coords[d.j], s.coords[d.i], s.box_size)
    bc = vector(s.coords[d.j], s.coords[d.k], s.box_size)
    dc = vector(s.coords[d.l], s.coords[d.k], s.box_size)

    θ = atan(dot((-ba × bc) × (bc × -dc), normalize(bc)), dot(-ba × bc, bc × -dc))

    (d.f1 * (1 + cos(θ)) + d.f2 * (1 - cos(2θ)) + d.f3 * (1 + cos(3θ)) + d.f4) / 2
end
