function potential_energy(s::Simulation)
    n_atoms = length(s.coords)
    potential = zero(eltype(eltype(s.coords)))

    for inter in values(s.general_inters)
        if inter.nl_only
            neighbours = s.neighbours
            @inbounds for ni in 1:length(neighbours)
                i, j = neighbours[ni]
                potential += potential_energy(inter, s, i, j)
            end
        else
            for i in 1:n_atoms
                for j in 1:i
                    potential += potential_energy(inter, s, i, j)
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        for inter in inter_list
            potential += potential_energy(inter, s)
        end
    end

    return potential
end

function kinetic_energy(s::Simulation)
    sum(i->s.atoms[i].mass*s.velocities[i]^2, axes(s.atoms, 1))
end

energy(s) = kinetic_energy(s) + potential_energy(s)