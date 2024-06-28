# Energy calculation

export
    total_energy,
    kinetic_energy,
    temperature,
    potential_energy

"""
    total_energy(system, neighbors=find_neighbors(sys); n_threads=Threads.nthreads())

Calculate the total energy of a system as the sum of the [`kinetic_energy`](@ref)
and the [`potential_energy`](@ref).
"""
function total_energy(sys; n_threads::Integer=Threads.nthreads())
    return total_energy(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

function total_energy(sys, neighbors; n_threads::Integer=Threads.nthreads())
    return kinetic_energy(sys) + potential_energy(sys, neighbors; n_threads=n_threads)
end

kinetic_energy_noconvert(sys) = sum(masses(sys) .* sum.(abs2, sys.velocities)) / 2

"""
    kinetic_energy(system)

Calculate the kinetic energy of a system.
"""
function kinetic_energy(sys::System{D, G, T}) where {D, G, T}
    ke = kinetic_energy_noconvert(sys)
    return uconvert(sys.energy_units, ke)
end

"""
    temperature(system)

Calculate the temperature of a system from the kinetic energy of the atoms.
"""
function temperature(sys)
    ke = kinetic_energy_noconvert(sys)
    temp = 2 * ke / (sys.df * sys.k)
    if sys.energy_units == NoUnits
        return temp
    else
        return uconvert(u"K", temp)
    end
end

"""
    potential_energy(system, neighbors=find_neighbors(sys); n_threads=Threads.nthreads())

Calculate the potential energy of a system using the pairwise, specific and
general interactions.

    potential_energy(inter::PairwiseInteraction, vec_ij, atom_i, atom_j, energy_units, special,
                     coord_i, coord_j, boundary, velocity_i, velocity_j, step_n)
    potential_energy(inter::SpecificInteraction, coords_i, coords_j,
                     boundary)
    potential_energy(inter::SpecificInteraction, coords_i, coords_j,
                     coords_k, boundary)
    potential_energy(inter::SpecificInteraction, coords_i, coords_j,
                     coords_k, coords_l, boundary)

Calculate the potential energy due to a given interaction type.

Custom interaction types should implement this function.
"""
function potential_energy(sys; n_threads::Integer=Threads.nthreads())
    return potential_energy(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

# Allow GPU-specific potential energy functions to be defined if required
potential_energy_gpu(inter::PairwiseInteraction, dr, ai, aj, eu, sp, ci, cj, bnd, vi, vj, sn) = potential_energy(inter, dr, ai, aj, eu, sp, ci, cj, bnd, vi, vj, sn)
potential_energy_gpu(inter::SpecificInteraction, ci, bnd)             = potential_energy(inter, ci, bnd)
potential_energy_gpu(inter::SpecificInteraction, ci, cj, bnd)         = potential_energy(inter, ci, cj, bnd)
potential_energy_gpu(inter::SpecificInteraction, ci, cj, ck, bnd)     = potential_energy(inter, ci, cj, ck, bnd)
potential_energy_gpu(inter::SpecificInteraction, ci, cj, ck, cl, bnd) = potential_energy(inter, ci, cj, ck, cl, bnd)

function potential_energy(sys::System{D, false}, neighbors;
                          n_threads::Integer=Threads.nthreads()) where D
    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    pairwise_inters_nl   = filter( use_neighbors, values(sys.pairwise_inters))
    sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(sys.specific_inter_lists))
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))

    ft = typeof(ustrip(sys.coords[1][1])) # Allow types like those from Measurements.jl
    pe = potential_energy_pair_spec(sys.coords, sys.velocities, sys.atoms, pairwise_inters_nonl, pairwise_inters_nl,
                            sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms, sys.boundary,
                            sys.energy_units, neighbors, n_threads, Val(ft))

    for inter in values(sys.general_inters)
        pe += AtomsCalculators.potential_energy(sys, inter; neighbors=neighbors, n_threads=n_threads)
    end

    return pe
end

function potential_energy_pair_spec(coords, velocities, atoms, pairwise_inters_nonl, pairwise_inters_nl,
                                    sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms,
                                    boundary, energy_units, neighbors, n_threads,
                                    val_ft::Val{T}) where T
    pe_vec = zeros(T, 1)
    potential_energy_pair_spec!(pe_vec, coords, velocities, atoms, pairwise_inters_nonl, pairwise_inters_nl,
                                sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms, boundary,
                                energy_units, neighbors, n_threads, val_ft)
    return pe_vec[1] * energy_units
end

function potential_energy_pair_spec!(pe_vec, coords, velocities, atoms, pairwise_inters_nonl,
                        pairwise_inters_nl, sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms,
                        boundary, energy_units, neighbors, n_threads, ::Val{T}) where T
    pe_sum = zero(T)

    @inbounds if n_threads > 1
        pe_sum_chunks = [zero(T) for _ in 1:n_threads]

        if length(pairwise_inters_nonl) > 0
            n_atoms = length(coords)
            Threads.@threads for chunk_i in 1:n_threads
                for i in chunk_i:n_threads:n_atoms
                    for j in (i + 1):n_atoms
                        dr = vector(coords[i], coords[j], boundary)
                        pe = potential_energy(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], energy_units, false,
                                    coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
                        for inter in pairwise_inters_nonl[2:end]
                            pe += potential_energy(inter, dr, atoms[i], atoms[j], energy_units, false,
                                    coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
                        end
                        check_energy_units(pe, energy_units)
                        pe_sum_chunks[chunk_i] += ustrip(pe)
                    end
                end
            end
        end

        if length(pairwise_inters_nl) > 0
            if isnothing(neighbors)
                error("an interaction uses the neighbor list but neighbors is nothing")
            end
            Threads.@threads for chunk_i in 1:n_threads
                for ni in chunk_i:n_threads:length(neighbors)
                    i, j, special = neighbors[ni]
                    dr = vector(coords[i], coords[j], boundary)
                    pe = potential_energy(pairwise_inters_nl[1], dr, atoms[i], atoms[j], energy_units, special,
                                    coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
                    for inter in pairwise_inters_nl[2:end]
                        pe += potential_energy(inter, dr, atoms[i], atoms[j], energy_units, special,
                                    coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
                    end
                    check_energy_units(pe, energy_units)
                    pe_sum_chunks[chunk_i] += ustrip(pe)
                end
            end
        end

        pe_sum += sum(pe_sum_chunks)
    else
        if length(pairwise_inters_nonl) > 0
            n_atoms = length(coords)
            for i in 1:n_atoms
                for j in (i + 1):n_atoms
                    dr = vector(coords[i], coords[j], boundary)
                    pe = potential_energy(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], energy_units, false,
                                coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
                    for inter in pairwise_inters_nonl[2:end]
                        pe += potential_energy(inter, dr, atoms[i], atoms[j], energy_units, false,
                                    coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
                    end
                    check_energy_units(pe, energy_units)
                    pe_sum += ustrip(pe)
                end
            end
        end

        if length(pairwise_inters_nl) > 0
            if isnothing(neighbors)
                error("an interaction uses the neighbor list but neighbors is nothing")
            end
            for ni in eachindex(neighbors)
                i, j, special = neighbors[ni]
                dr = vector(coords[i], coords[j], boundary)
                pe = potential_energy(pairwise_inters_nl[1], dr, atoms[i], atoms[j], energy_units, special,
                                coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
                for inter in pairwise_inters_nl[2:end]
                    pe += potential_energy(inter, dr, atoms[i], atoms[j], energy_units, special,
                                coords[i], coords[j], boundary, velocities[i], velocities[j], 0)
                end
                check_energy_units(pe, energy_units)
                pe_sum += ustrip(pe)
            end
        end
    end

    @inbounds for inter_list in sils_1_atoms
        for (i, inter) in zip(inter_list.is, inter_list.inters)
            pe = potential_energy(inter, coords[i], boundary)
            check_energy_units(pe, energy_units)
            pe_sum += ustrip(pe)
        end
    end

    @inbounds for inter_list in sils_2_atoms
        for (i, j, inter) in zip(inter_list.is, inter_list.js, inter_list.inters)
            pe = potential_energy(inter, coords[i], coords[j], boundary)
            check_energy_units(pe, energy_units)
            pe_sum += ustrip(pe)
        end
    end

    @inbounds for inter_list in sils_3_atoms
        for (i, j, k, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.inters)
            pe = potential_energy(inter, coords[i], coords[j], coords[k], boundary)
            check_energy_units(pe, energy_units)
            pe_sum += ustrip(pe)
        end
    end

    @inbounds for inter_list in sils_4_atoms
        for (i, j, k, l, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                                       inter_list.inters)
            pe = potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary)
            check_energy_units(pe, energy_units)
            pe_sum += ustrip(pe)
        end
    end

    pe_vec[1] = pe_sum
    return nothing
end

function potential_energy(sys::System{D, true, T}, neighbors;
                          n_threads::Integer=Threads.nthreads()) where {D, T}
    n_atoms = length(sys)
    val_ft = Val(T)
    pe_vec = CUDA.zeros(T, 1)

    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        nbs = NoNeighborList(n_atoms)
        pe_vec += pairwise_pe_gpu(sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters_nonl,
                                  nbs, sys.energy_units, val_ft)
    end

    pairwise_inters_nl = filter(use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        if length(neighbors) > 0
            nbs = @view neighbors.list[1:neighbors.n]
            pe_vec += pairwise_pe_gpu(sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters_nl,
                                      nbs, sys.energy_units, val_ft)
        end
    end

    for inter_list in values(sys.specific_inter_lists)
        pe_vec += specific_pe_gpu(inter_list, sys.coords, sys.boundary, sys.energy_units, val_ft)
    end

    pe = Array(pe_vec)[1]

    for inter in values(sys.general_inters)
        pe += ustrip(
            sys.energy_units,
            AtomsCalculators.potential_energy(sys, inter; neighbors=neighbors, n_threads=n_threads),
        )
    end

    return pe * sys.energy_units
end
