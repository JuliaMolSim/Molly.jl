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

@doc raw"""
    kinetic_energy(system)

Calculate the kinetic energy of a system.

The kinetic energy is defined as
```math
E_k = \frac{1}{2} \sum_{i} m_i v_i^2
```
where ``m_i`` is the mass and ``v_i`` is the velocity of atom ``i``.
"""
function kinetic_energy(sys::System)
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
    potential_energy(system, neighbors=find_neighbors(sys), step_n=0; n_threads=Threads.nthreads())

Calculate the potential energy of a system using the pairwise, specific and
general interactions.

    potential_energy(inter, vec_ij, atom_i, atom_j, energy_units, special, coord_i, coord_j,
                     boundary, velocity_i, velocity_j, step_n)
    potential_energy(inter, coord_i, boundary, atom_i, energy_units, velocity_i, step_n)
    potential_energy(inter, coord_i, coord_j, boundary, atom_i, atom_j, energy_units,
                     velocity_i, velocity_j, step_n)
    potential_energy(inter, coord_i, coord_j, coord_k, boundary, atom_i, atom_j, atom_k,
                     energy_units, velocity_i, velocity_j, velocity_k, step_n)
    potential_energy(inter, coord_i, coord_j, coord_k, coord_l, boundary, atom_i, atom_j,
                     atom_k, atom_l, energy_units, velocity_i, velocity_j, velocity_k,
                     velocity_l, step_n)

Calculate the potential energy due to a given interaction type.

Custom interaction types should implement this function.
"""
function potential_energy(sys; n_threads::Integer=Threads.nthreads())
    return potential_energy(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

function potential_energy(sys::System, neighbors, step_n::Integer=0;
                          n_threads::Integer=Threads.nthreads())
    # Allow types like those from Measurements.jl, T from System is different
    T = typeof(ustrip(zero(eltype(eltype(sys.coords)))))
    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    pairwise_inters_nl   = filter( use_neighbors, values(sys.pairwise_inters))
    sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(sys.specific_inter_lists))
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))

    if length(sys.pairwise_inters) > 0
        if n_threads > 1
            pe = pairwise_pe_threads(sys.atoms, sys.coords, sys.velocities, sys.boundary,
                                     neighbors, sys.energy_units, length(sys), pairwise_inters_nonl,
                                     pairwise_inters_nl, Val(T), n_threads, step_n)
        else
            pe = pairwise_pe(sys.atoms, sys.coords, sys.velocities, sys.boundary, neighbors,
                             sys.energy_units, length(sys), pairwise_inters_nonl,
                             pairwise_inters_nl, Val(T), step_n)
        end
    else
        pe = zero(T) * sys.energy_units
    end

    if length(sys.specific_inter_lists) > 0
        pe += specific_pe(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.energy_units,
                          sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms, Val(T), step_n)
    end

    for inter in values(sys.general_inters)
        pe += uconvert(
            sys.energy_units,
            AtomsCalculators.potential_energy(sys, inter; neighbors=neighbors,
                                              step_n=step_n, n_threads=n_threads),
        )
    end

    return pe
end

function pairwise_pe(atoms, coords, velocities, boundary, neighbors, energy_units,
                     n_atoms, pairwise_inters_nonl, pairwise_inters_nl, ::Val{T}, step_n=0) where T
    pe = zero(T) * energy_units

    @inbounds if length(pairwise_inters_nonl) > 0
        n_atoms = length(coords)
        for i in 1:n_atoms
            for j in (i + 1):n_atoms
                dr = vector(coords[i], coords[j], boundary)
                pe_sum = potential_energy(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], energy_units, false,
                            coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                for inter in pairwise_inters_nonl[2:end]
                    pe_sum += potential_energy(inter, dr, atoms[i], atoms[j], energy_units, false,
                                coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                end
                check_energy_units(pe_sum, energy_units)
                pe += pe_sum
            end
        end
    end

    @inbounds if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        for ni in eachindex(neighbors)
            i, j, special = neighbors[ni]
            dr = vector(coords[i], coords[j], boundary)
            pe_sum = potential_energy(pairwise_inters_nl[1], dr, atoms[i], atoms[j], energy_units, special,
                            coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
            for inter in pairwise_inters_nl[2:end]
                pe_sum += potential_energy(inter, dr, atoms[i], atoms[j], energy_units, special,
                            coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
            end
            check_energy_units(pe_sum, energy_units)
            pe += pe_sum
        end
    end

    return pe
end

function pairwise_pe_threads(atoms, coords, velocities, boundary, neighbors, energy_units, n_atoms,
                             pairwise_inters_nonl, pairwise_inters_nl, ::Val{T}, n_threads,
                             step_n=0) where T
    pe_chunks_nounits = zeros(T, n_threads)

    if length(pairwise_inters_nonl) > 0
        n_atoms = length(coords)
        Threads.@threads for chunk_i in 1:n_threads
            for i in chunk_i:n_threads:n_atoms
                for j in (i + 1):n_atoms
                    dr = vector(coords[i], coords[j], boundary)
                    pe_sum = potential_energy(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], energy_units, false,
                                coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                    for inter in pairwise_inters_nonl[2:end]
                        pe_sum += potential_energy(inter, dr, atoms[i], atoms[j], energy_units, false,
                                coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                    end
                    check_energy_units(pe_sum, energy_units)
                    pe_chunks_nounits[chunk_i] += ustrip(pe_sum)
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
                pe_sum = potential_energy(pairwise_inters_nl[1], dr, atoms[i], atoms[j], energy_units, special,
                                coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                for inter in pairwise_inters_nl[2:end]
                    pe_sum += potential_energy(inter, dr, atoms[i], atoms[j], energy_units, special,
                                coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                end
                check_energy_units(pe_sum, energy_units)
                pe_chunks_nounits[chunk_i] += ustrip(pe_sum)
            end
        end
    end

    return sum(pe_chunks_nounits) * energy_units
end

function specific_pe(atoms, coords, velocities, boundary, energy_units, sils_1_atoms,
                     sils_2_atoms, sils_3_atoms, sils_4_atoms, ::Val{T}, step_n=0) where T
    pe = zero(T) * energy_units

    @inbounds for inter_list in sils_1_atoms
        for (i, inter) in zip(inter_list.is, inter_list.inters)
            pe_inter = potential_energy(inter, coords[i], boundary, atoms[i], energy_units,
                                  velocities[i], step_n)
            check_energy_units(pe_inter, energy_units)
            pe += pe_inter
        end
    end

    @inbounds for inter_list in sils_2_atoms
        for (i, j, inter) in zip(inter_list.is, inter_list.js, inter_list.inters)
            pe_inter = potential_energy(inter, coords[i], coords[j], boundary, atoms[i], atoms[j],
                                  energy_units, velocities[i], velocities[j], step_n)
            check_energy_units(pe_inter, energy_units)
            pe += pe_inter
        end
    end

    @inbounds for inter_list in sils_3_atoms
        for (i, j, k, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.inters)
            pe_inter = potential_energy(inter, coords[i], coords[j], coords[k], boundary, atoms[i],
                                  atoms[j], atoms[k], energy_units, velocities[i], velocities[j],
                                  velocities[k], step_n)
            check_energy_units(pe_inter, energy_units)
            pe += pe_inter
        end
    end

    @inbounds for inter_list in sils_4_atoms
        for (i, j, k, l, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                                       inter_list.inters)
            pe_inter = potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary,
                                  atoms[i], atoms[j], atoms[k], atoms[l], energy_units,
                                  velocities[i], velocities[j], velocities[k], velocities[l],
                                  step_n)
            check_energy_units(pe_inter, energy_units)
            pe += pe_inter
        end
    end

    return pe
end

function potential_energy(sys::System{D, AT, T}, neighbors, step_n::Integer=0;
                          n_threads::Integer=Threads.nthreads()) where {D, AT <: AbstractGPUArray, T}
    val_ft = Val(T)
    pe_vec_nounits = KernelAbstractions.zeros(get_backend(sys.coords), T, 1)
    buffers = init_forces_buffer!(sys, ustrip_vec.(zero(sys.coords)), 1, true)

    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        nbs = NoNeighborList(length(sys))
        pairwise_pe_gpu!(pe_vec_nounits, buffers, sys, pairwise_inters_nonl, nbs, step_n)
    end

    pairwise_inters_nl = filter(use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nl) > 0
        pairwise_pe_gpu!(pe_vec_nounits, buffers, sys, pairwise_inters_nl, neighbors, step_n)   
    end

    for inter_list in values(sys.specific_inter_lists)
        specific_pe_gpu!(pe_vec_nounits, inter_list, sys.coords, sys.velocities, sys.atoms,
                         sys.boundary, step_n, sys.energy_units, val_ft)
    end

    pe = only(from_device(pe_vec_nounits)) * sys.energy_units

    for inter in values(sys.general_inters)
        pe += uconvert(
            sys.energy_units,
            AtomsCalculators.potential_energy(sys, inter; neighbors=neighbors,
                                              step_n=step_n, n_threads=n_threads),
        )
    end

    return pe
end

# Allow GPU-specific potential energy functions to be defined if required
potential_energy_gpu(inter, dr, ai, aj, eu, sp, ci, cj, bnd, vi, vj, sn) = potential_energy(inter, dr, ai, aj, eu, sp, ci, cj, bnd, vi, vj, sn)
potential_energy_gpu(inter, ci, bnd, ai, eu, vi, sn) = potential_energy(inter, ci, bnd, ai, eu, vi, sn)
potential_energy_gpu(inter, ci, cj, bnd, ai, aj, eu, vi, vj, sn) = potential_energy(inter, ci, cj, bnd, ai, aj, eu, vi, vj, sn)
potential_energy_gpu(inter, ci, cj, ck, bnd, ai, aj, ak, eu, vi, vj, vk, sn) = potential_energy(inter, ci, cj, ck, bnd, ai, aj, ak, eu, vi, vj, vk, sn)
potential_energy_gpu(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, eu, vi, vj, vk, vl, sn) = potential_energy(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, eu, vi, vj, vk, vl, sn)
