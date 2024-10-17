export Gravity

@doc raw"""
    Gravity(; G, use_neighbors)

The gravitational interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = -\frac{G m_i m_j}{r_{ij}}
```
"""
@kwdef struct Gravity{T}
    G::T = Unitful.G
    use_neighbors::Bool = false
end

use_neighbors(inter::Gravity) = inter.use_neighbors

function Base.zero(gr::Gravity{T}) where T
    return Gravity(zero(T), gr.use_neighbors)
end

Base.:+(g1::Gravity, g2::Gravity) = Gravity(g1.G + g2.G, g1.use_neighbors)

@inline function force(inter::Gravity,
                       dr,
                       atom_i,
                       atom_j,
                       args...)
    r2 = sum(abs2, dr)
    params = (inter.G, mass(atom_i), mass(atom_j))
    f = force_divr(inter, r2, inv(r2), params)
    return f * dr
end

function force_divr(::Gravity, r2, invr2, (G, mi, mj))
    return (-G * mi * mj) / √(r2 ^ 3)
end

@inline function potential_energy(inter::Gravity,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  args...)
    r2 = sum(abs2, dr)
    params = (inter.G, mass(atom_i), mass(atom_j))
    potential(inter, r2, inv(r2), params)
end

function potential(::Gravity, r2, invr2, (G, mi, mj))
    return (-G * mi * mj) * √invr2
end
