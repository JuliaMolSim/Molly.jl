"""
    Gravity(nl_only, G)

The gravitational interaction.
"""
struct Gravity{T} <: GeneralInteraction
    nl_only::Bool
    G::T
end

function force!(forces,
                inter::Gravity,
                s::Simulation,
                i::Integer,
                j::Integer)
    i == j && return
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    f = -inter.G * s.atoms[i].mass * s.atoms[j].mass * inv(sum(abs2, dr))
    fdr = f * normalize(dr)
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end
