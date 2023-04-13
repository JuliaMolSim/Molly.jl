export Gravity

@doc raw"""
    Gravity(; G, use_neighbors)

The gravitational interaction between two atoms.
The potential energy is defined as
```math
V(r_{ij}) = -\frac{G m_i m_j}{r_{ij}}
```
"""
struct Gravity{T} <: PairwiseInteraction
    G::T
    use_neighbors::Bool
end

Gravity(; G=Unitful.G, use_neighbors=false) = Gravity{typeof(G)}(G, use_neighbors)

@inline @inbounds function force(inter::Gravity,
                                 dr,
                                 coord_i,
                                 coord_j,
                                 atom_i,
                                 atom_j,
                                 boundary)
    r2 = sum(abs2, dr)

    params = (inter.G, mass(atom_i), mass(atom_j))

    f = force_divr_nocutoff(inter, r2, inv(r2), params)
    return f * dr
end

function force_divr_nocutoff(::Gravity, r2, invr2, (G, mi, mj))
    (-G * mi * mj) / √(r2 ^ 3)
end

@inline @inbounds function potential_energy(inter::Gravity,
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            boundary)
    r2 = sum(abs2, dr)

    params = (inter.G, mass(atom_i), mass(atom_j))

    potential(inter, r2, inv(r2), params)
end

function potential(::Gravity, r2, invr2, (G, mi, mj))
    (-G * mi * mj) * √invr2
end
