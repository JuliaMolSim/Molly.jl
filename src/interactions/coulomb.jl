"""
    Coulomb(nl_only)

The Coulomb electrostatic interaction.
"""
struct Coulomb{T} <: GeneralInteraction
    nl_only::Bool
    coulomb_const::T
end

Coulomb() = Coulomb(false,
                    138.935458 / 70.0 # Treat ϵr as 70 for now
)

@fastmath @inbounds function force!(forces,
                                    inter::Coulomb,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    i == j && return
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    r2 > sqdist_cutoff_nb && return
    T = typeof(r2)
    f = (T(coulomb_const) * s.atoms[i].charge * s.atoms[j].charge) / sqrt(r2 ^ 3)
    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end
