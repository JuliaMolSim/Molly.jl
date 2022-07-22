export Gravity

@doc raw"""
    Gravity(; G, nl_only)

The gravitational interaction between two atoms.
The potential energy is defined as
```math
V(r_{ij}) = -\frac{G m_i m_j}{r_{ij}}
```
"""
struct Gravity{T} <: PairwiseInteraction
    G::T
    nl_only::Bool
end

Gravity(; G=Unitful.G, nl_only=false) = Gravity{typeof(G)}(G, nl_only)

@inline @inbounds function force(inter::Gravity,
                                 dr,
                                 coord_i,
                                 coord_j,
                                 atom_i,
                                 atom_j,
                                 boundary)
    r2 = sum(abs2, dr)

    mi, mj = atom_i.mass, atom_j.mass
    params = (inter.G, mi, mj)

    f = force_divr_nocutoff(inter, r2, inv(r2), params)
    return f * dr
end

@fastmath function force_divr_nocutoff(::Gravity, r2, invr2, (G, mi, mj))
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

    mi, mj = atom_i.mass, atom_j.mass
    params = (inter.G, mi, mj)

    potential(inter, r2, inv(r2), params)
end


@fastmath function potential(::Gravity, r2, invr2, (G, mi, mj))
    (-G * mi * mj) * √invr2
end
