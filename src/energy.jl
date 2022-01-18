# Energy calculation

export total_energy,
    kinetic_energy,
    temperature,
    potential_energy

"""
    total_energy(s, neighbors=nothing)

Compute the total energy of the system.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.
"""
total_energy(s, neighbors=nothing) = kinetic_energy(s) + potential_energy(s, neighbors)

kinetic_energy_noconvert(s) = sum(mass.(s.atoms) .* sum.(abs2, s.velocities)) / 2

"""
    kinetic_energy(s)

Compute the kinetic energy of the system.
"""
function kinetic_energy(s)
    ke = kinetic_energy_noconvert(s)
    # Convert energy to per mol if required
    T = typeof(ustrip(ke))
    if dimension(s.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return T(uconvert(s.energy_units, ke * Unitful.Na))
    else
        return T(uconvert(s.energy_units, ke))
    end
end

const temp_conversion_factor = ustrip(u"nm^2 * u * K^-1 * ps^-2", Unitful.k)

"""
    temperature(system)

Calculate the temperature of a system from the kinetic energy of the atoms.
"""
function temperature(s)
    ke = kinetic_energy_noconvert(s)
    df = 3 * length(s) - 3
    T = typeof(ustrip(ke))
    if unit(ke) == NoUnits
        k = T(temp_conversion_factor)
    else
        k = T(uconvert(u"K^-1" * unit(ke), Unitful.k))
    end
    return 2 * ke / (df * k)
end

"""
    potential_energy(s, neighbors=nothing)

Compute the potential energy of the system.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.

    potential_energy(inter, vec_ij, coord_i, coord_j, atom_i, atom_j, box_size)

Calculate the potential energy between a pair of atoms due to a given
interation type.
Custom interaction types should implement this function.
"""
function potential_energy(s, neighbors=nothing)
    n_atoms = length(s)
    potential = zero(ustrip(s.box_size[1])) * s.energy_units

    for inter in values(s.general_inters)
        if inter.nl_only
            if isnothing(neighbors)
                error("An interaction uses the neighbor list but neighbors is nothing")
            end
            @inbounds for ni in 1:neighbors.n
                i, j, weight_14 = neighbors.list[ni]
                dr = vector(s.coords[i], s.coords[j], s.box_size)
                if weight_14
                    potential += potential_energy(inter, dr, s.coords[i], s.coords[j], s.atoms[i],
                                                    s.atoms[j], s.box_size, true)
                else
                    potential += potential_energy(inter, dr, s.coords[i], s.coords[j], s.atoms[i],
                                                    s.atoms[j], s.box_size)
                end
            end
        else
            for i in 1:n_atoms
                for j in (i + 1):n_atoms
                    dr = vector(s.coords[i], s.coords[j], s.box_size)
                    potential += potential_energy(inter, dr, s.coords[i], s.coords[j], s.atoms[i],
                                                    s.atoms[j], s.box_size)
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        potential += potential_energy(inter_list, s.coords, s.box_size)
    end

    return uconvert(s.energy_units, potential)
end

@views function potential_energy(inter_list::InteractionList2Atoms, coords, box_size)
    return sum(potential_energy.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                    (box_size,)))
end

@views function potential_energy(inter_list::InteractionList3Atoms, coords, box_size)
    return sum(potential_energy.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                    coords[inter_list.ks], (box_size,)))
end

@views function potential_energy(inter_list::InteractionList4Atoms, coords, box_size)
    return sum(potential_energy.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                    coords[inter_list.ks], coords[inter_list.ls], (box_size,)))
end
