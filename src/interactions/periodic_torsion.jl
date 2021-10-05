"""
    PeriodicTorsion(; periodicities, phases, ks)

A periodic torsion angle between four atoms.
"""
struct PeriodicTorsion{T, E} <: SpecificInteraction
    i::Int
    j::Int
    k::Int
    l::Int
    periodicities::Vector{Int}
    phases::Vector{T}
    ks::Vector{E}
end

PeriodicTorsion(; i, j, k, l, periodicities, phases, ks) = PeriodicTorsion{eltype(phases), eltype(ks)}(
    i, j, k, l, periodicities, phases, ks)

@inline @inbounds function force(d::PeriodicTorsion,
                                  coords,
                                  s::Simulation)
    ab = vector(coords[d.i], coords[d.j], s.box_size)
    bc = vector(coords[d.j], coords[d.k], s.box_size)
    cd = vector(coords[d.k], coords[d.l], s.box_size)
    cross_ab_bc = ab × bc
    cross_bc_cd = bc × cd
    θ = atan(
        ustrip(dot(cross_ab_bc × cross_bc_cd, normalize(bc))),
        ustrip(dot(cross_ab_bc, cross_bc_cd)),
    )
    fs = sum(zip(d.periodicities, d.phases, d.ks)) do (periodicity, phase, k)
        angle_term = -k * periodicity * sin((periodicity * θ) - phase)
        fa = angle_term * normalize(-cross_ab_bc) / norm(ab)
        # fd clashes with a function name
        f_d = angle_term * normalize(cross_bc_cd) / norm(cd)
        # Forces on the middle atoms have to keep the sum of torques null
        # Forces taken from http://www.softberry.com/freedownloadhelp/moldyn/description.html
        bc_norm = norm(bc)
        fb = (((ab .* -bc) / (bc_norm ^ 2)) .- 1) .* fa .- ((cd .* -bc) / (bc_norm ^ 2)) .* f_d
        fc = -fa - fb - f_d
        return [fa, fb, fc, f_d]
    end
    return [d.i, d.j, d.k, d.l], fs
end

@inline @inbounds function potential_energy(d::PeriodicTorsion,
                                            s::Simulation)
    ab = vector(s.coords[d.i], s.coords[d.j], s.box_size)
    bc = vector(s.coords[d.j], s.coords[d.k], s.box_size)
    cd = vector(s.coords[d.k], s.coords[d.l], s.box_size)
    cross_ab_bc = ab × bc
    cross_bc_cd = bc × cd
    θ = atan(
        ustrip(dot(cross_ab_bc × cross_bc_cd, normalize(bc))),
        ustrip(dot(cross_ab_bc, cross_bc_cd)),
    )
    E = sum(zip(d.periodicities, d.phases, d.ks)) do (periodicity, phase, k)
        k + k * cos((periodicity * θ) - phase)
    end
    return E
end
