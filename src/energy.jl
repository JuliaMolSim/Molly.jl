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

@doc raw"""
    kinetic_energy_tensor(system; kin_tensor=nothing)

Calculate the kinetic energy of a system in its tensorial form.

The kinetic energy tensor is defined as
```math
bf{K} = \frac{1}{2} \sum_{i} m_i \bf{v_i} \otimes \bf{v_i}
```
where ``m_i`` is the mass and ``\bf{v_i}`` is the velocity vector of atom ``i``.
"""
function kinetic_energy_tensor(sys::System{D}; kin_tensor=nothing) where D
    if isnothing(kin_tensor)
        # Allows propagation of uncertainties to tensors
        CT = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
        kin_tensor_used = zeros(CT, D, D) * sys.energy_units
    else
        kin_tensor_used = kin_tensor
    end
    kinetic_energy_tensor!(kin_tensor_used, sys)
    return kin_tensor_used
end

function kinetic_energy_tensor!(kin_tensor, sys::System{<:Any, <:Any, T}) where T
    fill!(kin_tensor, zero(T) * sys.energy_units)
    for (m, v) in zip(from_device(sys.masses), from_device(sys.velocities))
        kin_tensor .+= uconvert.(sys.energy_units, m .* (v * transpose(v))) ./ 2
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
           n_threads=Threads.nthreads())

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
Contributions from constraints, implicit solvent methods and bias potentials are ignored.
Compatible with virtual sites apart from [`OutOfPlaneSite`](@ref).

To calculate the scalar virial, see [`scalar_virial`](@ref).
"""
function virial(sys; n_threads::Integer=Threads.nthreads())
    return virial(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

function virial(sys, neighbors, step_n::Integer=0; n_threads::Integer=Threads.nthreads())
    _, v = forces_virial(sys, neighbors, step_n; n_threads=n_threads)
    return v
end

"""
    scalar_virial(system, neighbors=find_neighbors(system), step_n=0;
                  n_threads=Threads.nthreads())

Calculate the virial of the system as a scalar.

This is the trace of the [`virial`](@ref) tensor.
"""
function scalar_virial(sys; n_threads::Integer=Threads.nthreads())
    return scalar_virial(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

function scalar_virial(sys, neighbors, step_n::Integer=0; n_threads::Integer=Threads.nthreads())
    _, v = forces_virial(sys, neighbors, step_n; n_threads=n_threads)
    return tr(v)
end

"""
    temperature(system; kin_tensor=nothing, recompute=true)

Calculate the temperature of a system from the kinetic energy of the atoms.
"""
function temperature(sys::System{D}; kin_tensor=nothing, recompute=true) where D
    if isnothing(kin_tensor)
        # Allows propagation of uncertainties to tensors
        CT = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
        kin_tensor = zeros(CT, D, D) * sys.energy_units
    end
    if recompute
        ke = kinetic_energy(sys; kin_tensor=kin_tensor)
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
    potential_energy(system, neighbors=find_neighbors(system), step_n=0;
                     n_threads=Threads.nthreads())

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
function potential_energy(sys; n_threads::Integer=Threads.nthreads())
    return potential_energy(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

@inline has_nonl_inters(inters::Tuple{}) = false
@inline has_nonl_inters(inters::Tuple) = !use_neighbors(first(inters)) || has_nonl_inters(Base.tail(inters))

@inline has_nl_inters(inters::Tuple{}) = false
@inline has_nl_inters(inters::Tuple) = use_neighbors(first(inters)) || has_nl_inters(Base.tail(inters))

@inline eval_pe_nonl(inters::Tuple{}, dr, atom_i, atom_j, eu, ci, cj, bnd, vi, vj, step_n, ::Val{T}) where T = zero(T) * eu
@inline function eval_pe_nonl(inters::Tuple, dr, atom_i, atom_j, eu, ci, cj, bnd, vi, vj, step_n, ::Val{T}) where T
    pe_current = if !use_neighbors(first(inters))
        potential_energy(first(inters), dr, atom_i, atom_j, eu, false, ci, cj, bnd, vi, vj, step_n)
    else
        zero(T) * eu
    end
    return pe_current + eval_pe_nonl(Base.tail(inters), dr, atom_i, atom_j, eu, ci, cj, bnd, vi, vj, step_n, Val(T))
end

@inline eval_pe_nl(inters::Tuple{}, dr, atom_i, atom_j, eu, special, ci, cj, bnd, vi, vj, step_n, ::Val{T}) where T = zero(T) * eu
@inline function eval_pe_nl(inters::Tuple, dr, atom_i, atom_j, eu, special, ci, cj, bnd, vi, vj, step_n, ::Val{T}) where T
    pe_current = if use_neighbors(first(inters))
        potential_energy(first(inters), dr, atom_i, atom_j, eu, special, ci, cj, bnd, vi, vj, step_n)
    else
        zero(T) * eu
    end
    return pe_current + eval_pe_nl(Base.tail(inters), dr, atom_i, atom_j, eu, special, ci, cj, bnd, vi, vj, step_n, Val(T))
end

@inline eval_specific(inters::Tuple{}, atoms, coords, velocities, boundary, energy_units, step_n, ::Val{T}) where T = zero(T) * energy_units
@inline eval_specific(inters::Tuple, atoms, coords, velocities, boundary, energy_units, step_n, ::Val{T}) where T = 
    calc_pe_list(first(inters), atoms, coords, velocities, boundary, energy_units, step_n, Val(T)) + 
    eval_specific(Base.tail(inters), atoms, coords, velocities, boundary, energy_units, step_n, Val(T))

@inline eval_general(inters::Tuple{}, sys, neighbors, step_n, n_threads, ::Val{T}) where T = zero(T) * sys.energy_units
@inline function eval_general(inters::Tuple, sys, neighbors, step_n, n_threads, ::Val{T}) where T
    pe_current = uconvert(sys.energy_units, AtomsCalculators.potential_energy(sys, first(inters); neighbors=neighbors, step_n=step_n, n_threads=n_threads))
    return pe_current + eval_general(Base.tail(inters), sys, neighbors, step_n, n_threads, Val(T))
end

function potential_energy(sys::System, neighbors, buffers=nothing, step_n::Integer=0;
                          n_threads::Integer=Threads.nthreads())
    T = typeof(ustrip(zero(eltype(eltype(sys.coords)))))

    pe = pairwise_pe_loop(sys.atoms, sys.coords, sys.velocities, sys.boundary,
                          neighbors, sys.energy_units, length(sys.coords), sys.pairwise_inters,
                          Val(T), Int(n_threads), step_n)

    pe += eval_specific(sys.specific_inter_lists, sys.atoms, sys.coords, sys.velocities, sys.boundary, sys.energy_units, step_n, Val(T))

    pe += eval_general(sys.general_inters, sys, neighbors, step_n, Int(n_threads), Val(T))

    return pe
end

function pairwise_pe_loop(atoms, coords, velocities, boundary, neighbors, energy_units,
                          n_atoms, pairwise_inters, ::Val{T},
                          n_threads::Int, step_n=0) where T
    if n_threads == 1
        return pairwise_pe_loop_single(atoms, coords, velocities, boundary, neighbors, energy_units, n_atoms, pairwise_inters, Val(T), step_n)
    else
        return pairwise_pe_loop_threaded(atoms, coords, velocities, boundary, neighbors, energy_units, n_atoms, pairwise_inters, Val(T), n_threads, step_n)
    end
end

function pairwise_pe_loop_single(atoms, coords, velocities, boundary, neighbors, energy_units,
                                 n_atoms, pairwise_inters, ::Val{T}, step_n) where T
    pe = zero(T) * energy_units

    @inbounds if has_nonl_inters(pairwise_inters)
        for i in 1:n_atoms
            for j in (i + 1):n_atoms
                dr = vector(coords[i], coords[j], boundary)
                pe_sum = eval_pe_nonl(pairwise_inters, dr, atoms[i], atoms[j], energy_units, coords[i], coords[j], boundary, velocities[i], velocities[j], step_n, Val(T))
                check_energy_units(pe_sum, energy_units)
                pe += pe_sum
            end
        end
    end

    @inbounds if has_nl_inters(pairwise_inters)
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        for ni in eachindex(neighbors)
            i, j, special = neighbors[ni]
            dr = vector(coords[i], coords[j], boundary)
            pe_sum = eval_pe_nl(pairwise_inters, dr, atoms[i], atoms[j], energy_units, special, coords[i], coords[j], boundary, velocities[i], velocities[j], step_n, Val(T))
            check_energy_units(pe_sum, energy_units)
            pe += pe_sum
        end
    end

    return pe
end

function pairwise_pe_loop_threaded(atoms, coords, velocities, boundary, neighbors, energy_units,
                                   n_atoms, pairwise_inters, ::Val{T}, n_threads::Int, step_n) where T
    pe_chunks_nounits = zeros(T, n_threads)

    if has_nonl_inters(pairwise_inters)
        Threads.@threads for chunk_i in 1:n_threads
            for i in chunk_i:n_threads:n_atoms
                for j in (i + 1):n_atoms
                    dr = vector(coords[i], coords[j], boundary)
                    pe_sum = eval_pe_nonl(pairwise_inters, dr, atoms[i], atoms[j], energy_units, coords[i], coords[j], boundary, velocities[i], velocities[j], step_n, Val(T))
                    check_energy_units(pe_sum, energy_units)
                    pe_chunks_nounits[chunk_i] += ustrip(pe_sum)
                end
            end
        end
    end

    if has_nl_inters(pairwise_inters)
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        Threads.@threads for chunk_i in 1:n_threads
            for ni in chunk_i:n_threads:length(neighbors)
                i, j, special = neighbors[ni]
                dr = vector(coords[i], coords[j], boundary)
                pe_sum = eval_pe_nl(pairwise_inters, dr, atoms[i], atoms[j], energy_units, special, coords[i], coords[j], boundary, velocities[i], velocities[j], step_n, Val(T))
                check_energy_units(pe_sum, energy_units)
                pe_chunks_nounits[chunk_i] += ustrip(pe_sum)
            end
        end
    end

    return sum(pe_chunks_nounits) * energy_units
end

function specific_pe(atoms, coords, velocities, boundary, energy_units, specific_inter_lists, ::Val{T}, step_n=0) where T
    return eval_specific(specific_inter_lists, atoms, coords, velocities, boundary, energy_units, step_n, Val(T))
end

function calc_pe_list(inter_list::InteractionList1Atoms, atoms, coords, velocities, boundary, energy_units, step_n, ::Val{T}) where T
    pe_il = zero(T) * energy_units
    @inbounds for (i, inter) in zip(inter_list.is, inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], boundary, atoms[i], energy_units, velocities[i], step_n)
        check_energy_units(pe_inter, energy_units)
        pe_il += pe_inter
    end
    return pe_il
end

function calc_pe_list(inter_list::InteractionList2Atoms, atoms, coords, velocities, boundary, energy_units, step_n, ::Val{T}) where T
    pe_il = zero(T) * energy_units
    @inbounds for (i, j, inter) in zip(inter_list.is, inter_list.js, inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], coords[j], boundary, atoms[i], atoms[j], energy_units, velocities[i], velocities[j], step_n)
        check_energy_units(pe_inter, energy_units)
        pe_il += pe_inter
    end
    return pe_il
end

function calc_pe_list(inter_list::InteractionList3Atoms, atoms, coords, velocities, boundary, energy_units, step_n, ::Val{T}) where T
    pe_il = zero(T) * energy_units
    @inbounds for (i, j, k, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], coords[j], coords[k], boundary, atoms[i], atoms[j], atoms[k], energy_units, velocities[i], velocities[j], velocities[k], step_n)
        check_energy_units(pe_inter, energy_units)
        pe_il += pe_inter
    end
    return pe_il
end

function calc_pe_list(inter_list::InteractionList4Atoms, atoms, coords, velocities, boundary, energy_units, step_n, ::Val{T}) where T
    pe_il = zero(T) * energy_units
    @inbounds for (i, j, k, l, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.ls, inter_list.inters)
        pe_inter = potential_energy(inter, coords[i], coords[j], coords[k], coords[l], boundary, atoms[i], atoms[j], atoms[k], atoms[l], energy_units, velocities[i], velocities[j], velocities[k], velocities[l], step_n)
        check_energy_units(pe_inter, energy_units)
        pe_il += pe_inter
    end
    return pe_il
end

function potential_energy(sys::System{<:Any, <:AbstractGPUArray}, neighbors,
                          buffers=nothing, step_n::Integer=0;
                          n_threads::Integer=Threads.nthreads())
    buffers = init_buffers!(sys, 1, true)
    return potential_energy(sys, neighbors, buffers, step_n; n_threads=n_threads)
end

function potential_energy(sys::System{<:Any, <:AbstractGPUArray, T}, neighbors,
                          buffers::BuffersGPU, step_n::Integer=0;
                          n_threads::Integer=Threads.nthreads()) where T
    fill!(buffers.pe_vec_nounits, zero(T))

    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        nbs = NoNeighborList(length(sys))
        pairwise_pe_loop_gpu!(buffers.pe_vec_nounits, buffers, sys, pairwise_inters_nonl, nbs, step_n)
    end

    pairwise_inters_nl = filter(use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nl) > 0
        pairwise_pe_loop_gpu!(buffers.pe_vec_nounits, buffers, sys, pairwise_inters_nl, neighbors, step_n)
    end

    for inter_list in values(sys.specific_inter_lists)
        specific_pe_gpu!(buffers.pe_vec_nounits, inter_list, sys.coords, sys.velocities, sys.atoms,
                         sys.boundary, step_n, sys.energy_units, Val(T))
    end

    pe = only(from_device(buffers.pe_vec_nounits)) * sys.energy_units

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
