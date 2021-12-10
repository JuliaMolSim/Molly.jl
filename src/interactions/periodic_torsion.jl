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
    fs = sum(zip(d.periodicities, d.phases, d.ks)) do (periodicity, phase, k)
        dEdθ = -k * periodicity * sin((periodicity * θ) - phase)
        fi =  dEdθ * bc_norm * cross_ab_bc / dot(cross_ab_bc, cross_ab_bc)
        fl = -dEdθ * bc_norm * cross_bc_cd / dot(cross_bc_cd, cross_bc_cd)
        v = (dot(-ab, bc) / bc_norm^2) * fi - (dot(-cd, bc) / bc_norm^2) * fl
        fj =  v - fi
        fk = -v - fl
        return [fi, fj, fk, fl]
    end
    return [d.i, d.j, d.k, d.l], fs
end

@inline @inbounds function potential_energy(d::PeriodicTorsion,
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
    E = sum(zip(d.periodicities, d.phases, d.ks)) do (periodicity, phase, k)
        k + k * cos((periodicity * θ) - phase)
    end
    return E
end
