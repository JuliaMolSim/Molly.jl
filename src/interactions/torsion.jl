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

function force!(forces,
                d::Torsion,
                s::Simulation)
    ba = vector(s.coords[d.j], s.coords[d.i], s.box_size)
    bc = vector(s.coords[d.j], s.coords[d.k], s.box_size)
    dc = vector(s.coords[d.l], s.coords[d.k], s.box_size)
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
    forces[d.i] += fa
    forces[d.j] += fb
    forces[d.k] += fc
    forces[d.l] += f_d
    return nothing
end
