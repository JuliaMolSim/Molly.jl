# Energy calculation

export
    total_energy,
    kinetic_energy,
    temperature,
    potential_energy

"""
    total_energy(s, neighbors=nothing)

Calculate the total energy of the system.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.
Not currently compatible with automatic differentiation using Zygote.
"""
total_energy(s, neighbors=nothing) = kinetic_energy(s) + potential_energy(s, neighbors)

kinetic_energy_noconvert(s) = sum(masses(s) .* sum.(abs2, s.velocities)) / 2

"""
    kinetic_energy(s)

Calculate the kinetic energy of the system.
"""
function kinetic_energy(s::System{D, G, T}) where {D, G, T}
    ke = kinetic_energy_noconvert(s)
    # Convert energy to per mol if required
    if dimension(s.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return T(uconvert(s.energy_units, ke * Unitful.Na))
    else
        return T(uconvert(s.energy_units, ke))
    end
end

"""
    temperature(system)

Calculate the temperature of a system from the kinetic energy of the atoms.
"""
function temperature(s)
    ke = kinetic_energy_noconvert(s)
    df = 3 * length(s) - 3
    temp = 2 * ke / (df * s.k)
    if s.energy_units == NoUnits
        return temp
    else
        return uconvert(u"K", temp)
    end
end

function check_energy_units(E, energy_units)
    if unit(E) != energy_units
        error("System energy units are ", energy_units, " but encountered energy units ",
                unit(E))
    end
end

@inline @inbounds function potential_energy_nounit(inters, coord_i, coord_j, atom_i, atom_j,
                                        boundary, energy_units, weight_14::Bool=false)
    dr = vector(coord_i, coord_j, boundary)
    sum(inters) do inter
        E = potential_energy(inter, dr, coord_i, coord_j, atom_i, atom_j, boundary, weight_14)
        check_energy_units(E, energy_units)
        return ustrip(E)
    end
end

@views function potential_energy_inters(inters, coords, atoms, neighbors, boundary,
                                        energy_units, weights_14)
    coords_i, atoms_i = getindices_i(coords, neighbors), getindices_i(atoms, neighbors)
    coords_j, atoms_j = getindices_j(coords, neighbors), getindices_j(atoms, neighbors)
    @inbounds energies = potential_energy_nounit.((inters,), coords_i, coords_j,
                                atoms_i, atoms_j, (boundary,), energy_units, weights_14)
    return sum(energies) * energy_units
end

"""
    potential_energy(s, neighbors=nothing)

Calculate the potential energy of the system using the pairwise, specific and
general interactions.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.
Not currently compatible with automatic differentiation using Zygote.

    potential_energy(inter::PairwiseInteraction, vec_ij, coord_i, coord_j,
                     atom_i, atom_j, boundary)
    potential_energy(inter::SpecificInteraction, coords_i, coords_j,
                     boundary)
    potential_energy(inter::SpecificInteraction, coords_i, coords_j,
                     coords_k, boundary)
    potential_energy(inter::SpecificInteraction, coords_i, coords_j,
                     coords_k, coords_l, boundary)
    potential_energy(inter, system, neighbors=nothing)

Calculate the potential energy due to a given interation type.
Custom interaction types should implement this function.
"""
function potential_energy(s::System{D, false, T}, neighbors=nothing) where {D, T}
    n_atoms = length(s)
    potential = zero(T) * s.energy_units

    for inter in values(s.pairwise_inters)
        if inter.nl_only
            if isnothing(neighbors)
                error("An interaction uses the neighbor list but neighbors is nothing")
            end
            @inbounds for ni in 1:neighbors.n
                i, j, weight_14 = neighbors.list[ni]
                dr = vector(s.coords[i], s.coords[j], s.boundary)
                potential += potential_energy(inter, dr, s.coords[i], s.coords[j], s.atoms[i],
                                                s.atoms[j], s.boundary, weight_14)
            end
        else
            for i in 1:n_atoms
                for j in (i + 1):n_atoms
                    dr = vector(s.coords[i], s.coords[j], s.boundary)
                    potential += potential_energy(inter, dr, s.coords[i], s.coords[j], s.atoms[i],
                                                    s.atoms[j], s.boundary)
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        potential += potential_energy(inter_list, s.coords, s.boundary)
    end

    for inter in values(s.general_inters)
        potential += potential_energy(inter, s, neighbors)
    end

    return uconvert(s.energy_units, potential)
end

function potential_energy(s::System{D, true, T}, neighbors=nothing) where {D, T}
    potential = zero(T) * s.energy_units

    pairwise_inters_nonl = filter(inter -> !inter.nl_only, values(s.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        potential += potential_energy_inters(pairwise_inters_nonl, s.coords, s.atoms,
                        neighbors.all, s.boundary, s.energy_units, false)
    end

    pairwise_inters_nl = filter(inter -> inter.nl_only, values(s.pairwise_inters))
    if length(pairwise_inters_nl) > 0 && length(neighbors.close.nbsi) > 0
        potential += potential_energy_inters(pairwise_inters_nl, s.coords, s.atoms,
                        neighbors.close, s.boundary, s.energy_units, neighbors.close.weights_14)
    end

    for inter_list in values(s.specific_inter_lists)
        potential += potential_energy(inter_list, s.coords, s.boundary)
    end

    for inter in values(s.general_inters)
        potential += potential_energy(inter, s, neighbors)
    end

    return uconvert(s.energy_units, potential)
end

@views function potential_energy(inter_list::InteractionList2Atoms, coords, boundary)
    return sum(potential_energy.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                    (boundary,)))
end

@views function potential_energy(inter_list::InteractionList3Atoms, coords, boundary)
    return sum(potential_energy.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                    coords[inter_list.ks], (boundary,)))
end

@views function potential_energy(inter_list::InteractionList4Atoms, coords, boundary)
    return sum(potential_energy.(inter_list.inters, coords[inter_list.is], coords[inter_list.js],
                                    coords[inter_list.ks], coords[inter_list.ls], (boundary,)))
end

function potential_energy(inter, dr, coord_i, coord_j, atom_i, atom_j, boundary, weight_14)
    # Fallback for interactions where the 1-4 weighting is not relevant
    return potential_energy(inter, dr, coord_i, coord_j, atom_i, atom_j, boundary)
end
