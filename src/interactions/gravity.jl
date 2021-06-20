"""
    Gravity(; G, nl_only, force_units, energy_units)

The gravitational interaction.
"""
struct Gravity{T, F, E} <: GeneralInteraction
    G::T
    nl_only::Bool
    force_units::F
    energy_units::E
end

function Gravity(;
                    G=Unitful.G,
                    nl_only=false,
                    force_units=u"kg * m * s^-2",
                    energy_units=u"kg * m^2 * s^-2")
    return Gravity{typeof(G), typeof(force_units), typeof(energy_units)}(
        G, nl_only, force_units, energy_units)
end

@inline @inbounds function force(inter::Gravity,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    box_size)
    dr = vector(coord_i, coord_j, box_size)
    r2 = sum(abs2, dr)

    mi, mj = atom_i.mass, atom_j.mass
    params = (inter.G, mi, mj)

    f = force_nocutoff(inter, r2, inv(r2), params)
    return f * dr
end

@fastmath function force_nocutoff(::Gravity, r2, invr2, (G, mi, mj))
    (G * mi * mj) / √(r2 ^ 3)
end

@inline @inbounds function potential_energy(inter::Gravity,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    i == j && return ustrip(zero(s.timestep)) * inter.energy_units

    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    mi, mj = s.atoms[i].mass, s.atoms[j].mass
    params = (inter.G, mi, mj)

    potential(inter, r2, inv(r2), params)
end


@fastmath function potential(::Gravity, r2, invr2, (G, mi, mj))
    (G * mi * mj) * √invr2
end
