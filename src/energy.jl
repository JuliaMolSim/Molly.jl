# Energy calculation

export total_energy,
    kinetic_energy,
    potential_energy

"""
    total_energy(s, neighbors=nothing)

Compute the total energy of the system.
"""
total_energy(s, neighbors=nothing) = kinetic_energy(s) + potential_energy(s, neighbors)

"""
    kinetic_energy(s)

Compute the kinetic energy of the system.
"""
function kinetic_energy(s)
    ke = sum(i -> s.atoms[i].mass * dot(s.velocities[i], s.velocities[i]) / 2, axes(s.atoms, 1))
    # Convert energy to per mol if required
    if dimension(s.energy_unit) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        T = typeof(ustrip(ke))
        return uconvert(s.energy_unit, ke * T(Unitful.Na))
    else
        return uconvert(s.energy_unit, ke)
    end
end

"""
    potential_energy(s, neighbors=nothing)

Compute the potential energy of the system.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.
"""
function potential_energy(s, neighbors=nothing)
    n_atoms = length(s)
    potential = zero(ustrip(s.box_size[1])) * s.energy_unit

    for inter in values(s.general_inters)
        if inter.nl_only
            @inbounds for ni in 1:neighbors.n
                i, j, weight_14 = neighbors.list[ni]
                if weight_14
                    potential += potential_energy(inter, s.coords[i], s.coords[j], s.atoms[i],
                                                    s.atoms[j], s.box_size, true)
                else
                    potential += potential_energy(inter, s.coords[i], s.coords[j], s.atoms[i],
                                                    s.atoms[j], s.box_size)
                end
            end
        else
            for i in 1:n_atoms
                for j in (i + 1):n_atoms
                    potential += potential_energy(inter, s.coords[i], s.coords[j], s.atoms[i],
                                                    s.atoms[j], s.box_size)
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        potential += potential_energy(inter_list, s.coords, s.box_size)
    end

    return uconvert(s.energy_unit, potential)
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
