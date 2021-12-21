"""
    PeriodicTorsion(; periodicities, phases, ks)

A periodic torsion angle between four atoms.
"""
struct PeriodicTorsion{T, E} <: SpecificInteraction
    periodicities::Vector{Int}
    phases::Vector{T}
    ks::Vector{E}
end

PeriodicTorsion(; periodicities, phases, ks) = PeriodicTorsion{eltype(phases), eltype(ks)}(
                                                        periodicities, phases, ks)

@inline @inbounds function force(d::PeriodicTorsion, coords_i, coords_j, coords_k,
                                    coords_l, box_size)
    ab = vector(coords_i, coords_j, box_size)
    bc = vector(coords_j, coords_k, box_size)
    cd = vector(coords_k, coords_l, box_size)
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
    return SpecificForce4Atom(fs...)
end

@inline @inbounds function potential_energy(d::PeriodicTorsion, coords_i, coords_j, coords_k,
                                            coords_l, box_size)
    ab = vector(coords_i, coords_j, box_size)
    bc = vector(coords_j, coords_k, box_size)
    cd = vector(coords_k, coords_l, box_size)
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
