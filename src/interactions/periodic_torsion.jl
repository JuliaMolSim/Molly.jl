"""
    PeriodicTorsion(; periodicities, phases, ks)

A periodic torsion angle between four atoms.
"""
struct PeriodicTorsion{N, T, E} <: SpecificInteraction
    periodicities::NTuple{N, Int}
    phases::NTuple{N, T}
    ks::NTuple{N, E}
end

function PeriodicTorsion(; periodicities, phases, ks, n_terms=length(periodicities))
    T, E = eltype(phases), eltype(ks)
    if n_terms > length(periodicities)
        n_to_add = n_terms - length(periodicities)
        periodicities_pad = vcat(collect(periodicities), ones(Int, n_to_add))
        phases_pad = vcat(collect(phases), zeros(T, n_to_add))
        ks_pad = vcat(collect(ks), zeros(E, n_to_add))
    else
        periodicities_pad, phases_pad, ks_pad = periodicities, phases, ks
    end
    PeriodicTorsion{n_terms, T, E}(tuple(periodicities_pad...), tuple(phases_pad...),
                                    tuple(ks_pad...))
end

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
        return SpecificForce4Atoms(fi, fj, fk, fl)
    end
    return fs
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
        return k + k * cos((periodicity * θ) - phase)
    end
    return E
end
