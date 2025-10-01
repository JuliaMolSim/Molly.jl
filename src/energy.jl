# Energy calculation

export
    total_energy,
    kinetic_energy,
    virial,
    scalar_virial,
    temperature,
    potential_energy,
    pairwise_pe

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

raw"""
    kinetic_energy_tensor!(system, kin_tensor)

Calculate the kinetic energy of a system in its tensorial form.

The kinetic energy tensor is defined as
```math
bf{K} = \frac{1}{2} \sum_{i} m_i \bf{v_i} \otimes \bf{v_i}
```
where ``m_i`` is the mass and ``\bf{v_i}`` is the velocity vector of atom ``i``.
"""
function kinetic_energy_tensor!(sys::System{D, AT, T}, kin_tensor) where {D, AT, T}
    fill!(kin_tensor, zero(T)*sys.energy_units)
    half = T(0.5)
    for (m, v) in zip(from_device(sys.masses), from_device(sys.velocities))
        # m: mass per particle, v: velocity with units
        kin_tensor .+= half .* uconvert.(sys.energy_units, m .* (v * transpose(v)))  # units: energy
    end
end

@doc raw"""
    kinetic_energy(system; kin_tensor = nothing)

Retrieve the kinetic energy of a system.

The scalar kinetic energy is defined as
```math
K = \rm{Tr}\left[ \bf{K} \right]
```

where ``\bf{K}`` is the kinetic energy tensor:
```math
\bf{K} = \frac{1}{2} \sum_{i} m_i \bf{v_i} \otimes \bf{v_i}
```
"""
function kinetic_energy(sys::System{D, AT, T}; kin_tensor = nothing) where {D, AT, T}
    if kin_tensor === nothing
        CT = typeof(ustrip(oneunit(eltype(eltype(sys.coords))))) # Allows propagation of uncertainties to tensors
        kin_tensor = zeros(CT, D, D) * sys.energy_units
    end
    kinetic_energy_tensor!(sys, kin_tensor)
    ke = tr(kin_tensor)
    return ke
end

@doc raw"""
    virial(system)

Forces a recomputation of the forces acting on the system
and computes the virial tensor. The virial, in its most general
form, is defined as:

```math
\bf{W} = \sum_i \bf{r_i} \otimes \bf{f_i}
```
where ``\bf{r_i}`` and ``\bf{f_i}`` are the position and force vectors,
respectively, acting on the i``^{th}`` atom. In Molly.jl, we implement
the [virial definition used in LAMMPS](https://docs.lammps.org/compute_stress_atom.html),
and take into account pairwise and specific interactions, as well as the K-space
contribution of the [`Ewald`](@ref) and [`PME`](@ref) methods, computed as indicated
in the [original paper](https://doi.org/10.1063/1.470117).
"""
function virial(sys)
    _, virial = forces(sys; needs_virial = true) # Force recomputation
    return virial
end


@doc """
    scalar_virial(sys)

Retrieves the virial of the system as a scalar instead of as a tensor. Needs
to recompute the forces.
"""
function scalar_virial(sys)
    _, virial = forces(sys; needs_virial = true)
    return tr(virial) 
end

"""
    temperature(system; kin_tensor = nothing, recompute = true)

Calculate the temperature of a system from the kinetic energy of the atoms.
"""
function temperature(sys::System{D, AT, T}; kin_tensor = nothing, recompute = true) where {D, AT, T}
    if kin_tensor === nothing
        CT = typeof(ustrip(oneunit(eltype(eltype(sys.coords))))) # Allows propagation of uncertainties to tensors
        kin_tensor = zeros(CT, D, D) * sys.energy_units
    end
    if recompute
        ke = kinetic_energy(sys; kin_tensor = kin_tensor)
    else
        ke = tr(kin_tensor)
    end
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
        pe = pairwise_pe_loop(sys.atoms, sys.coords, sys.velocities, sys.boundary,
                              neighbors, sys.energy_units, length(sys), pairwise_inters_nonl,
                              pairwise_inters_nl, Val(T), Val(n_threads), step_n)
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

function pairwise_pe_loop(atoms, coords, velocities, boundary, neighbors, energy_units,
                          n_atoms, pairwise_inters_nonl, pairwise_inters_nl, ::Val{T},
                          ::Val{1}, step_n=0) where T
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

function pairwise_pe_loop(atoms, coords, velocities, boundary, neighbors, energy_units,
                          n_atoms, pairwise_inters_nonl, pairwise_inters_nl, ::Val{T},
                          ::Val{n_threads}, step_n=0) where {T, n_threads}
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
    pe_vec_nounits = KernelAbstractions.zeros(get_backend(sys.coords), T, 1)
    buffers = init_buffers!(sys, 1, true)

    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        nbs = NoNeighborList(length(sys))
        pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys, pairwise_inters_nonl, nbs, step_n)
    end

    pairwise_inters_nl = filter(use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nl) > 0
        pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys, pairwise_inters_nl, neighbors, step_n)
    end

    for inter_list in values(sys.specific_inter_lists)
        specific_pe_gpu!(pe_vec_nounits, inter_list, sys.coords, sys.velocities, sys.atoms,
                         sys.boundary, step_n, sys.energy_units, Val(T))
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

"""
    pairwise_pe(inter, r, params)

Calculate the potential energy between two atoms separated by distance `r` due to a
pairwise interaction.

This function is used in [`potential_energy`](@ref) to apply cutoff strategies by calculating
the potential energy at different values of `r`.
Consequently, the parameters `params` should not include terms that depend on distance.
"""
function pairwise_pe end
