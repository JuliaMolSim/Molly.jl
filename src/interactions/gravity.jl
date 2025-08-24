export Gravity

@doc raw"""
    Gravity(; cutoff, G, use_neighbors)

The gravitational interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = -\frac{G m_i m_j}{r_{ij}}
```
"""
@kwdef struct Gravity{T, C}
    cutoff::C = NoCutoff()
    G::T = Unitful.G
    use_neighbors::Bool = false
end

use_neighbors(inter::Gravity) = inter.use_neighbors

function Base.zero(gr::Gravity{T}) where T
    return Gravity(gr.cutoff, zero(T), gr.use_neighbors)
end

Base.:+(g1::Gravity, g2::Gravity) = Gravity(g1.cutoff, g1.G + g2.G, g1.use_neighbors)

@inline function force(inter::Gravity,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       args...)
    r2 = sum(abs2, dr)
    params = (inter.G, mass(atom_i), mass(atom_j))
    f = force_cutoff(inter.cutoff, inter, r2, params, force_units)
    return f * normalize(dr)
end

function pairwise_force(::Gravity, r2, (G, mi, mj))
    return (-G * mi * mj) / r2
end

@inline function potential_energy(inter::Gravity,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  args...)
    r2 = sum(abs2, dr)
    params = (inter.G, mass(atom_i), mass(atom_j))
    return pe_cutoff(inter.cutoff, inter, r2, params, energy_units)
end

function pairwise_pe(::Gravity, r2, (G, mi, mj))
    return (-G * mi * mj) * inv(sqrt(r2))
end
