export PeriodicTorsion

@doc raw"""
    PeriodicTorsion(; periodicities, phases, ks, proper)

A periodic torsion angle between four atoms.
The potential energy is defined as
```math
V(\phi) = \sum_{n=1}^N k_n (1 + \cos(n \phi - \phi_{s,n}))
```
"""
struct PeriodicTorsion{N, T, E} <: SpecificInteraction
    periodicities::NTuple{N, Int}
    phases::NTuple{N, T}
    ks::NTuple{N, E}
    proper::Bool
end

function PeriodicTorsion(; periodicities, phases, ks, proper::Bool=true,
                            n_terms=length(periodicities))
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
                                    tuple(ks_pad...), proper)
end

function Base.zero(::PeriodicTorsion{N, T, E}) where {N, T, E}
    return PeriodicTorsion{N, T, E}(
        ntuple(_ -> 0      , N),
        ntuple(_ -> zero(T), N),
        ntuple(_ -> zero(E), N),
        false,
    )
end

function Base.:+(p1::PeriodicTorsion{N, T, E}, p2::PeriodicTorsion{N, T, E}) where {N, T, E}
    return PeriodicTorsion{N, T, E}(
        p1.periodicities,
        p1.phases .+ p2.phases,
        p1.ks .+ p2.ks,
        p1.proper,
    )
end

@inline @inbounds function force(d::PeriodicTorsion, coords_i, coords_j, coords_k,
                                    coords_l, boundary)
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
                                            coords_l, boundary)
    θ = torsion_angle(coords_i, coords_j, coords_k, coords_l, boundary)
    E = sum(zip(d.periodicities, d.phases, d.ks)) do (periodicity, phase, k)
        return k + k * cos((periodicity * θ) - phase)
    end
    return E
end
