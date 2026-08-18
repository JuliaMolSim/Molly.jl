# Energy calculation

export
    total_energy,
    kinetic_energy_tensor,
    kinetic_energy,
    virial,
    scalar_virial,
    temperature,
    potential_energy,
    pairwise_pe

"""
    total_energy(system, neighbors=find_neighbors(sys), step_n=0, buffers=nothing;
                 n_threads=Threads.nthreads(), pairwise_inters=system.pairwise_inters,
                 specific_inter_lists=system.specific_inter_lists,
                 general_inters=system.general_inters)

Calculate the total energy of a system as the sum of the [`kinetic_energy`](@ref)
and the [`potential_energy`](@ref).
"""
function total_energy(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return total_energy(sys, find_neighbors(sys; n_threads=n_threads);
                        n_threads=n_threads, kwargs...)
end

function total_energy(sys, neighbors, step_n::Integer=0, buffers=nothing; kwargs...)
    ke = kinetic_energy(sys)
    pe = potential_energy(sys, neighbors, step_n, buffers; kwargs...)
    return ke + pe
end

@doc raw"""
    kinetic_energy_tensor(system; kin_tensor=nothing)

Calculate the kinetic energy of a system in its tensorial form.

The kinetic energy tensor is defined as
```math
bf{K} = \frac{1}{2} \sum_{i} m_i \bf{v_i} \otimes \bf{v_i}
```
where ``m_i`` is the mass and ``\bf{v_i}`` is the velocity vector of atom ``i``.
"""
function kinetic_energy_tensor(sys::System{D, <:Any, <:Any, TH}; kin_tensor=nothing) where {D, TH}
    if isnothing(kin_tensor)
        kin_tensor_used = zeros(TH, D, D) * sys.energy_units
    else
        kin_tensor_used = kin_tensor
    end
    kinetic_energy_tensor!(kin_tensor_used, sys)
    return kin_tensor_used
end

function kinetic_energy_tensor!(kin_tensor, sys::System{D, <:Any, <:Any, TH}) where {D, TH}
    fill!(kin_tensor, zero(TH) * sys.energy_units)
    masses_cpu = from_device(sys.masses)
    velocities_cpu = from_device(sys.velocities)
    @inbounds for i in eachindex(sys)
        m_half = masses_cpu[i] / 2
        v = velocities_cpu[i]
        for col in 1:D
            for row in 1:D
                kin_tensor[row, col] += uconvert(sys.energy_units, m_half * v[row] * v[col])
            end
        end
    end
    return kin_tensor
end

@doc raw"""
    kinetic_energy(system; kin_tensor=nothing)

Calculate the kinetic energy of a system.

The scalar kinetic energy is defined as
```math
K = \rm{Tr}\left[ \bf{K} \right]
```
where ``\bf{K}`` is the kinetic energy tensor:
```math
\bf{K} = \frac{1}{2} \sum_{i} m_i \bf{v_i} \otimes \bf{v_i}
```
"""
function kinetic_energy(sys; kin_tensor=nothing)
    kin_tensor_used = kinetic_energy_tensor(sys; kin_tensor=kin_tensor)
    return tr(kin_tensor_used)
end

@doc raw"""
    virial(system, neighbors=find_neighbors(system), step_n=0;
           n_threads=Threads.nthreads(), pairwise_inters=system.pairwise_inters,
           specific_inter_lists=system.specific_inter_lists,
           general_inters=system.general_inters)

Calculate the virial tensor of the system.

The virial, in its most general form, is defined as:
```math
\bf{W} = \sum_i \bf{r_i} \otimes \bf{f_i}
```
where ``\bf{r_i}`` and ``\bf{f_i}`` are the position and force vectors,
respectively, acting on atom ``i``.
The [virial definition from LAMMPS](https://docs.lammps.org/compute_stress_atom.html)
is used, taking into account pairwise interactions, specific interactions, and the
[`Ewald`](@ref) and [`PME`](@ref) methods computed as indicated in
[Essmann et al. 1995](https://doi.org/10.1063/1.470117).
Contributions from implicit solvent methods and bias potentials are ignored.
For constrained systems, constraint contributions are approximated using a
deterministic small-step constraint preview.
Compatible with virtual sites apart from [`OutOfPlaneSite`](@ref).

To calculate the scalar virial, see [`scalar_virial`](@ref).
"""
function virial(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return virial(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads, kwargs...)
end

function virial(sys, neighbors, step_n::Integer=0;
                n_threads::Integer=Threads.nthreads(), kwargs...)
    if length(sys.constraints) > 0
        buffers = init_buffers!(sys, n_threads)
        compute_initial_total_virial!(buffers, sys, neighbors, step_n;
                                      n_threads=n_threads, kwargs...)
        return buffers.virial
    else
        _, v = forces_virial(sys, neighbors, step_n; n_threads=n_threads, kwargs...)
        return v
    end
end

"""
    scalar_virial(system, neighbors=find_neighbors(system), step_n=0;
                  n_threads=Threads.nthreads(), pairwise_inters=system.pairwise_inters,
                  specific_inter_lists=system.specific_inter_lists,
                  general_inters=system.general_inters)

Calculate the virial of the system as a scalar.

This is the trace of the [`virial`](@ref) tensor.
"""
function scalar_virial(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return scalar_virial(sys, find_neighbors(sys; n_threads=n_threads);
                         n_threads=n_threads, kwargs...)
end

function scalar_virial(sys, neighbors, step_n::Integer=0;
                       n_threads::Integer=Threads.nthreads(), kwargs...)
    return tr(virial(sys, neighbors, step_n; n_threads=n_threads, kwargs...))
end

@doc raw"""
    temperature(system; kin_tensor=nothing, n_dof=system.df,
                k=system.k, recompute=true)

Calculate the temperature of a system from the kinetic energy of the atoms.

The temperature is defined as
```math
T = \frac{2 E_\mathrm{kin}}{N_\mathrm{df} k}
```
where ``E_\mathrm{kin}`` is the kinetic energy, ``N_\mathrm{df}`` is the number of
degrees of freedom in the system (`n_dof`) and ``k`` is the Boltzmann constant (`k`).
"""
function temperature(sys::System{D, <:Any, <:Any, TH}; kin_tensor=nothing, n_dof=sys.df,
                     k=sys.k, recompute=true) where {D, TH}
    if isnothing(kin_tensor)
        kin_tensor_used = zeros(TH, D, D) * sys.energy_units
    else
        kin_tensor_used = kin_tensor
    end
    if recompute
        ke = kinetic_energy(sys; kin_tensor=kin_tensor_used)
    else
        ke = tr(kin_tensor_used)
    end
    temp = 2 * ke / (n_dof * k)
    if sys.energy_units == NoUnits
        return temp
    else
        return uconvert(u"K", temp)
    end
end

@inline function sum_pairwise_pe(inters::Tuple{T}, dr, atom_i, atom_j, energy_units,
                                 special, coord_i, coord_j, boundary, vel_i, vel_j,
                                 step_n) where {T}
    return potential_energy(inters[1], dr, atom_i, atom_j, energy_units, special, coord_i, coord_j,
                            boundary, vel_i, vel_j, step_n)
end

@inline function sum_pairwise_pe(inters::Tuple, dr, atom_i, atom_j, energy_units,
                                 special, coord_i, coord_j, boundary, vel_i, vel_j, step_n)
    return potential_energy(first(inters), dr, atom_i, atom_j, energy_units, special, coord_i, coord_j,
                            boundary, vel_i, vel_j, step_n) +
           sum_pairwise_pe(Base.tail(inters), dr, atom_i, atom_j, energy_units, special,
                           coord_i, coord_j, boundary, vel_i, vel_j, step_n)
end

"""
    potential_energy(system, neighbors=find_neighbors(system), step_n=0, buffers=nothing;
                     n_threads=Threads.nthreads(), pairwise_inters=system.pairwise_inters,
                     specific_inter_lists=system.specific_inter_lists,
                     general_inters=system.general_inters)

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
    potential_energy(bias_pot, cv; kwargs...)

Calculate the potential energy due to a given interaction type.

Custom interaction types should implement this function.
"""
function potential_energy(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return potential_energy(sys, find_neighbors(sys; n_threads=n_threads);
                            n_threads=n_threads, kwargs...)
end

function potential_energy(sys::System{<:Any, <:Any, <:Any, TH},
                          neighbors,
                          step_n::Integer=0,
                          buffers=nothing;
                          n_threads::Integer=Threads.nthreads(),
                          pairwise_inters=sys.pairwise_inters,
                          specific_inter_lists=sys.specific_inter_lists,
                          general_inters=sys.general_inters) where TH
    if length(pairwise_inters) > 0
        use_vel = any_uses_velocity(pairwise_inters)
        pe = with_pairwise_partition(values(pairwise_inters)) do pis_nonl, pis_nl
            pairwise_pe_loop(sys.atoms, sys.coords, sys.velocities, sys.boundary,
                             neighbors, sys.energy_units, length(sys), pis_nonl,
                             pis_nl, step_n, Val(TH), Val(n_threads), Val(use_vel))
        end
    else
        pe = zero(TH) * sys.energy_units
    end

    if length(specific_inter_lists) > 0
        sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(specific_inter_lists))
        sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(specific_inter_lists))
        sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(specific_inter_lists))
        sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(specific_inter_lists))
        sils_5_atoms = filter(il -> il isa InteractionList5Atoms, values(specific_inter_lists))
        pe += specific_pe(sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.energy_units,
                          sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms, sils_5_atoms,
                          Val(TH), step_n)
    end

    for inter in values(general_inters)
        pe += uconvert(
            sys.energy_units,
            AtomsCalculators.potential_energy(sys, inter; neighbors=neighbors,
                                              step_n=step_n, n_threads=n_threads),
        )
    end

    return pe
end

function pairwise_pe_loop(atoms, coords, velocities, boundary, neighbors, energy_units,
                          n_atoms, pairwise_inters_nonl, pairwise_inters_nl, step_n, ::Val{TH},
                          ::Val{1}, ::Val{use_vel}) where {TH, use_vel}
    pe = zero(TH) * energy_units

    @inbounds if length(pairwise_inters_nonl) > 0
        for i in 1:n_atoms
            coord_i = coords[i]
            atom_i = atoms[i]
            vel_i = maybe_velocity(velocities, i, Val(use_vel))
            for j in (i + 1):n_atoms
                coord_j = coords[j]
                dr = vector(coord_i, coord_j, boundary)
                atom_j = atoms[j]
                vel_j = maybe_velocity(velocities, j, Val(use_vel))
                pe_sum = sum_pairwise_pe(pairwise_inters_nonl, dr, atom_i, atom_j, energy_units,
                                         false, coord_i, coord_j, boundary, vel_i, vel_j, step_n)
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
            coord_i = coords[i]
            coord_j = coords[j]
            dr = vector(coord_i, coord_j, boundary)
            atom_i = atoms[i]
            atom_j = atoms[j]
            vel_i = maybe_velocity(velocities, i, Val(use_vel))
            vel_j = maybe_velocity(velocities, j, Val(use_vel))
            pe_sum = sum_pairwise_pe(pairwise_inters_nl, dr, atom_i, atom_j, energy_units,
                                     special, coord_i, coord_j, boundary, vel_i, vel_j, step_n)
            check_energy_units(pe_sum, energy_units)
            pe += pe_sum
        end
    end

    return pe
end

function pairwise_pe_loop(atoms, coords, velocities, boundary, neighbors, energy_units,
                          n_atoms, pairwise_inters_nonl, pairwise_inters_nl, step_n, ::Val{TH},
                          ::Val{n_threads}, ::Val{use_vel}) where {TH, n_threads, use_vel}
    pe_chunks_nounits = zeros(TH, n_threads)

    @inbounds if length(pairwise_inters_nonl) > 0
        Threads.@threads for chunk_i in 1:n_threads
            pe_chunks_nounits[chunk_i] = pairwise_pe_nonl_range(atoms, coords, velocities, boundary,
                            energy_units, pairwise_inters_nonl, step_n, chunk_i, n_threads, n_atoms,
                            Val(TH), Val(use_vel))
        end
    end

    @inbounds if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        n_neighbors = length(neighbors)
        block_size = 512
        next_block_start = Threads.Atomic{Int}(1)
        @sync for chunk_i in 1:n_threads
            Threads.@spawn begin
                pe_chunk = zero(TH)
                while true
                    block_start = Threads.atomic_add!(next_block_start, block_size)
                    block_start > n_neighbors && break
                    block_stop = min(block_start + block_size - 1, n_neighbors)
                    pe_chunk += pairwise_pe_nl_block(atoms, coords, velocities, boundary, neighbors,
                                    energy_units, pairwise_inters_nl, step_n, block_start, block_stop,
                                    Val(TH), Val(use_vel))
                end
                pe_chunks_nounits[chunk_i] += pe_chunk
            end
        end
    end

    return sum(pe_chunks_nounits) * energy_units
end

@noinline function pairwise_pe_nonl_range(atoms, coords, velocities, boundary, energy_units,
                                          pairwise_inters_nonl, step_n, chunk_i, n_threads, n_atoms,
                                          ::Val{TH}, ::Val{use_vel}) where {TH, use_vel}
    pe_chunk = zero(TH)
    @inbounds for i in chunk_i:n_threads:n_atoms
        coord_i = coords[i]
        atom_i = atoms[i]
        vel_i = maybe_velocity(velocities, i, Val(use_vel))
        for j in (i + 1):n_atoms
            coord_j = coords[j]
            dr = vector(coord_i, coord_j, boundary)
            atom_j = atoms[j]
            vel_j = maybe_velocity(velocities, j, Val(use_vel))
            pe_sum = sum_pairwise_pe(pairwise_inters_nonl, dr, atom_i, atom_j, energy_units,
                                     false, coord_i, coord_j, boundary, vel_i, vel_j, step_n)
            check_energy_units(pe_sum, energy_units)
            pe_chunk += ustrip(pe_sum)
        end
    end
    return pe_chunk
end

@noinline function pairwise_pe_nl_block(atoms, coords, velocities, boundary, neighbors, energy_units,
                                      pairwise_inters_nl, step_n, block_start, block_stop,
                                      ::Val{TH}, ::Val{use_vel}) where {TH, use_vel}
    pe_chunk = zero(TH)
    @inbounds for ni in block_start:block_stop
        i, j, special = neighbors[ni]
        coord_i = coords[i]
        coord_j = coords[j]
        dr = vector(coord_i, coord_j, boundary)
        atom_i = atoms[i]
        atom_j = atoms[j]
        vel_i = maybe_velocity(velocities, i, Val(use_vel))
        vel_j = maybe_velocity(velocities, j, Val(use_vel))
        pe_sum = sum_pairwise_pe(pairwise_inters_nl, dr, atom_i, atom_j, energy_units,
                                 special, coord_i, coord_j, boundary, vel_i, vel_j, step_n)
        check_energy_units(pe_sum, energy_units)
        pe_chunk += ustrip(pe_sum)
    end
    return pe_chunk
end

function specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units, step_n,
                                inter_list::InteractionList1Atoms)
    @inbounds for (i, inter) in zip(inter_list.is, inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], boundary, atoms[i], energy_units,
                              velocities[i], step_n, inter_list.data)
        check_energy_units(pe_inter, energy_units)
        pe += pe_inter
    end
    return pe
end

function specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units, step_n,
                                inter_list::InteractionList2Atoms)
    @inbounds for (i, j, inter) in zip(inter_list.is, inter_list.js, inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], coords[j], boundary, atoms[i], atoms[j],
                              energy_units, velocities[i], velocities[j], step_n,
                              inter_list.data)
        check_energy_units(pe_inter, energy_units)
        pe += pe_inter
    end
    return pe
end

function specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units, step_n,
                                inter_list::InteractionList3Atoms)
    @inbounds for (i, j, k, inter) in zip(inter_list.is, inter_list.js, inter_list.ks,
                                          inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], coords[j], coords[k], boundary, atoms[i],
                              atoms[j], atoms[k], energy_units, velocities[i], velocities[j],
                              velocities[k], step_n, inter_list.data)
        check_energy_units(pe_inter, energy_units)
        pe += pe_inter
    end
    return pe
end

function specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units, step_n,
                                inter_list::InteractionList4Atoms)
    @inbounds for (i, j, k, l, inter) in zip(inter_list.is, inter_list.js, inter_list.ks,
                                             inter_list.ls, inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary,
                              atoms[i], atoms[j], atoms[k], atoms[l], energy_units,
                              velocities[i], velocities[j], velocities[k], velocities[l],
                              step_n, inter_list.data)
        check_energy_units(pe_inter, energy_units)
        pe += pe_inter
    end
    return pe
end

function specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units, step_n,
                                inter_list::InteractionList5Atoms)
    @inbounds for (i, j, k, l, m, inter) in zip(inter_list.is, inter_list.js, inter_list.ks,
                                                inter_list.ls, inter_list.ms, inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], coords[j], coords[k], coords[l],
                              coords[m], boundary, atoms[i], atoms[j], atoms[k], atoms[l],
                              atoms[m], energy_units, velocities[i], velocities[j],
                              velocities[k], velocities[l], velocities[m], step_n,
                              inter_list.data)
        check_energy_units(pe_inter, energy_units)
        pe += pe_inter
    end
    return pe
end

function specific_pe(atoms, coords, velocities, boundary, energy_units, sils_1_atoms,
                     sils_2_atoms, sils_3_atoms, sils_4_atoms, sils_5_atoms, ::Val{TH},
                     step_n=0) where TH
    pe = zero(TH) * energy_units

    for inter_list in sils_1_atoms
        pe = specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units,
                                    step_n, inter_list)
    end
    for inter_list in sils_2_atoms
        pe = specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units,
                                    step_n, inter_list)
    end
    for inter_list in sils_3_atoms
        pe = specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units,
                                    step_n, inter_list)
    end
    for inter_list in sils_4_atoms
        pe = specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units,
                                    step_n, inter_list)
    end
    for inter_list in sils_5_atoms
        pe = specific_pe_inter_list(pe, atoms, coords, velocities, boundary, energy_units,
                                    step_n, inter_list)
    end

    return pe
end

function potential_energy(sys::System{<:Any, <:AbstractGPUArray}, neighbors,
                          step_n::Integer=0, buffers_empty::Nothing=nothing; kwargs...)
    buffers = init_buffers!(sys, 1, true)
    return potential_energy(sys, neighbors, step_n, buffers; kwargs...)
end

function potential_energy(sys::System{<:Any, <:AbstractGPUArray, <:Any, TH},
                          neighbors,
                          step_n::Integer,
                          buffers::BuffersGPU;
                          n_threads::Integer=Threads.nthreads(),
                          pairwise_inters=sys.pairwise_inters,
                          specific_inter_lists=sys.specific_inter_lists,
                          general_inters=sys.general_inters) where TH
    fill!(buffers.pe_vec_nounits, zero(TH))

    with_pairwise_partition(values(pairwise_inters)) do pis_nonl, pis_nl
        if length(pis_nonl) > 0
            nbs = NoNeighborList(length(sys))
            pairwise_pe_loop_gpu!(buffers.pe_vec_nounits, buffers, sys, pis_nonl, nbs, step_n)
        end
        if length(pis_nl) > 0
            pairwise_pe_loop_gpu!(buffers.pe_vec_nounits, buffers, sys, pis_nl,
                                  neighbors, step_n)
        end
        return nothing
    end

    for inter_list in values(specific_inter_lists)
        specific_pe_gpu!(buffers.pe_vec_nounits, inter_list, sys.coords, sys.velocities, sys.atoms,
                         sys.boundary, step_n, sys.energy_units, Val(TH))
    end

    pe = only(from_device(buffers.pe_vec_nounits)) * sys.energy_units

    for inter in values(general_inters)
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
potential_energy_gpu(inter, ci, bnd, ai, eu, vi, sn, data) = potential_energy(inter, ci, bnd, ai, eu, vi, sn, data)
potential_energy_gpu(inter, ci, cj, bnd, ai, aj, eu, vi, vj, sn, data) = potential_energy(inter, ci, cj, bnd, ai, aj, eu, vi, vj, sn, data)
potential_energy_gpu(inter, ci, cj, ck, bnd, ai, aj, ak, eu, vi, vj, vk, sn, data) = potential_energy(inter, ci, cj, ck, bnd, ai, aj, ak, eu, vi, vj, vk, sn, data)
potential_energy_gpu(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, eu, vi, vj, vk, vl, sn, data) = potential_energy(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, eu, vi, vj, vk, vl, sn, data)
potential_energy_gpu(inter, ci, cj, ck, cl, cm, bnd, ai, aj, ak, al, am, eu, vi, vj, vk, vl, vm, sn, data) = potential_energy(inter, ci, cj, ck, cl, cm, bnd, ai, aj, ak, al, am, eu, vi, vj, vk, vl, vm, sn, data)

@inline zero_pairwise_energy(dr, energy_units) = ustrip(zero(dr[1])) * energy_units

"""
    pairwise_pe(inter, r, params)

Calculate the potential energy between two atoms separated by distance `r` due to a
pairwise interaction.

This function is used in [`potential_energy`](@ref) to apply cutoff strategies by calculating
the potential energy at different values of `r`.
Consequently, the parameters `params` should not include terms that depend on distance.
"""
function pairwise_pe end
