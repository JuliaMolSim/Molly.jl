"""
    Gravity(nl_only, G)

The gravitational interaction.
"""
struct Gravity{T} <: GeneralInteraction
    nl_only::Bool
    G::T
end

@inline @inbounds function force!(forces,
                                 inter::Gravity,
                                 s::Simulation,
                                 i::Integer,
                                 j::Integer)
    i == j && return
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    mi, mj = s.atoms[i].mass, s.atoms[j].mass
    params = (inter.G, mi, mj)

    f = force_nocutoff(inter, r2, inv(r2), params)
    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end

@fastmath function force_nocutoff(::Gravity, r2, invr2, (G, mi, mj))
    (G * mi * mj) / √(r2 ^ 3)
end

@inline @inbounds function potential_energy(inter::Gravity,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    U = eltype(s.coords[i]) # this is not Unitful compatible
    i == j && return zero(U)

    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    mi, mj = s.atoms[i].mass, s.atoms[j].mass
    params = (inter.G, mi, mj)

    potential(inter, r2, inv(r2), params)
end


@fastmath function potential(::Gravity, r2, invr2, (G, mi, mj))
    (G * mi * mj) * √invr2
end
