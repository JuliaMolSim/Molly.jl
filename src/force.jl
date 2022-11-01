# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See OpenMM documentation and Gromacs manual for other aspects of forces

export
    ustrip_vec,
    accelerations,
    force,
    SpecificForce1Atoms,
    SpecificForce2Atoms,
    SpecificForce3Atoms,
    SpecificForce4Atoms,
    forces

"""
    ustrip_vec(x)

Broadcasted form of `ustrip` from Unitful.jl, allowing e.g. `ustrip_vec.(coords)`.
"""
ustrip_vec(x...) = ustrip.(x...)

function check_force_units(fdr, force_units)
    if unit(first(fdr)) != force_units
        error("System force units are ", force_units, " but encountered force units ",
                unit(first(fdr)))
    end
end

"""
    accelerations(system, neighbors=nothing; n_threads=Threads.nthreads())

Calculate the accelerations of all atoms using the pairwise, specific and
general interactions and Newton's second law of motion.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.
"""
function accelerations(s, neighbors=nothing; n_threads::Integer=Threads.nthreads())
    return forces(s, neighbors; n_threads=n_threads) ./ masses(s)
end

"""
    force(inter::PairwiseInteraction, vec_ij, coord_i, coord_j,
          atom_i, atom_j, boundary)
    force(inter::PairwiseInteraction, vec_ij, coord_i, coord_j,
          atom_i, atom_j, boundary, weight_14)
    force(inter::SpecificInteraction, coord_i, coord_j,
          boundary)
    force(inter::SpecificInteraction, coord_i, coord_j,
          coord_k, boundary)
    force(inter::SpecificInteraction, coord_i, coord_j,
          coord_k, coord_l, boundary)

Calculate the force between atoms due to a given interation type.
For [`PairwiseInteraction`](@ref)s returns a single force vector and for
[`SpecificInteraction`](@ref)s returns a type such as [`SpecificForce2Atoms`](@ref).
Custom pairwise and specific interaction types should implement this function.
"""
function force(inter, dr, coord_i, coord_j, atom_i, atom_j, boundary, weight_14)
    # Fallback for interactions where the 1-4 weighting is not relevant
    return force(inter, dr, coord_i, coord_j, atom_i, atom_j, boundary)
end

"""
    SpecificForce1Atoms(f1)

Force on one atom arising from an interaction such as a position restraint.
"""
struct SpecificForce1Atoms{D, T}
    f1::SVector{D, T}
end

"""
    SpecificForce2Atoms(f1, f2)

Forces on two atoms arising from an interaction such as a bond potential.
"""
struct SpecificForce2Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
end

"""
    SpecificForce3Atoms(f1, f2, f3)

Forces on three atoms arising from an interaction such as a bond angle potential.
"""
struct SpecificForce3Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
end

"""
    SpecificForce4Atoms(f1, f2, f3, f4)

Forces on four atoms arising from an interaction such as a torsion potential.
"""
struct SpecificForce4Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
    f4::SVector{D, T}
end

function SpecificForce1Atoms(f1::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce1Atoms{D, T}(f1)
end

function SpecificForce2Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce2Atoms{D, T}(f1, f2)
end

function SpecificForce3Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                            f3::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce3Atoms{D, T}(f1, f2, f3)
end

function SpecificForce4Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                            f3::StaticArray{Tuple{D}, T}, f4::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce4Atoms{D, T}(f1, f2, f3, f4)
end

Base.:+(x::SpecificForce1Atoms, y::SpecificForce1Atoms) = SpecificForce1Atoms(x.f1 + y.f1)
Base.:+(x::SpecificForce2Atoms, y::SpecificForce2Atoms) = SpecificForce2Atoms(x.f1 + y.f1, x.f2 + y.f2)
Base.:+(x::SpecificForce3Atoms, y::SpecificForce3Atoms) = SpecificForce3Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3)
Base.:+(x::SpecificForce4Atoms, y::SpecificForce4Atoms) = SpecificForce4Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3, x.f4 + y.f4)

"""
    forces(system, neighbors=nothing; n_threads=Threads.nthreads())

Calculate the forces on all atoms in the system using the pairwise, specific and
general interactions.
If the interactions use neighbor lists, the neighbors should be computed
first and passed to the function.

    forces(inter, system, neighbors=nothing; n_threads=Threads.nthreads())

Calculate the forces on all atoms in the system arising from a general
interaction.
If the interaction uses neighbor lists, the neighbors should be computed
first and passed to the function.
Custom general interaction types should implement this function.
"""
function forces(s::System{D, false}, neighbors=nothing;
                n_threads::Integer=Threads.nthreads()) where D
    fs = forces_pair_spec(s, neighbors, n_threads)

    for inter in values(s.general_inters)
        fs += forces(inter, s, neighbors; n_threads=n_threads)
    end

    return fs
end

function forces_pair_spec(s, neighbors, n_threads)
    fs = ustrip_vec.(zero(s.coords))
    pairwise_inters_nonl = filter(inter -> !inter.nl_only, values(s.pairwise_inters))
    pairwise_inters_nl   = filter(inter ->  inter.nl_only, values(s.pairwise_inters))
    sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(s.specific_inter_lists))
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(s.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(s.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(s.specific_inter_lists))
    forces_pair_spec!(fs, s.coords, s.atoms, pairwise_inters_nonl, pairwise_inters_nl,
                      sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms, s.boundary,
                      s.force_units, neighbors, n_threads)
    return fs * s.force_units
end

@inbounds function forces_pair_spec!(fs, coords, atoms, pairwise_inters_nonl, pairwise_inters_nl,
                                     sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms,
                                     boundary, force_units, neighbors, n_threads)
    n_atoms = length(coords)
    if n_threads > 1
        fs_chunks = [zero(fs) for _ in 1:n_threads]

        if length(pairwise_inters_nonl) > 0
            Threads.@threads for thread_id in 1:n_threads
                for i in thread_id:n_threads:n_atoms
                    for j in (i + 1):n_atoms
                        dr = vector(coords[i], coords[j], boundary)
                        f = force(pairwise_inters_nonl[1], dr, coords[i], coords[j], atoms[i],
                                  atoms[j], boundary)
                        for inter in pairwise_inters_nonl[2:end]
                            f += force(inter, dr, coords[i], coords[j], atoms[i], atoms[j], boundary)
                        end
                        check_force_units(f, force_units)
                        f_ustrip = ustrip.(f)
                        fs_chunks[thread_id][i] -= f_ustrip
                        fs_chunks[thread_id][j] += f_ustrip
                    end
                end
            end
        end

        if length(pairwise_inters_nl) > 0
            if isnothing(neighbors)
                error("An interaction uses the neighbor list but neighbors is nothing")
            end
            Threads.@threads for thread_id in 1:n_threads
                for ni in thread_id:n_threads:length(neighbors)
                    i, j, weight_14 = neighbors.list[ni]
                    dr = vector(coords[i], coords[j], boundary)
                    f = force(pairwise_inters_nl[1], dr, coords[i], coords[j], atoms[i],
                              atoms[j], boundary, weight_14)
                    for inter in pairwise_inters_nl[2:end]
                        f += force(inter, dr, coords[i], coords[j], atoms[i], atoms[j], boundary,
                                   weight_14)
                    end
                    check_force_units(f, force_units)
                    f_ustrip = ustrip.(f)
                    fs_chunks[thread_id][i] -= f_ustrip
                    fs_chunks[thread_id][j] += f_ustrip
                end
            end
        end

        fs .+= sum(fs_chunks)
    else
        if length(pairwise_inters_nonl) > 0
            for i in 1:n_atoms
                for j in (i + 1):n_atoms
                    dr = vector(coords[i], coords[j], boundary)
                    f = force(pairwise_inters_nonl[1], dr, coords[i], coords[j], atoms[i],
                              atoms[j], boundary)
                    for inter in pairwise_inters_nonl[2:end]
                        f += force(inter, dr, coords[i], coords[j], atoms[i], atoms[j], boundary)
                    end
                    check_force_units(f, force_units)
                    f_ustrip = ustrip.(f)
                    fs[i] -= f_ustrip
                    fs[j] += f_ustrip
                end
            end
        end

        if length(pairwise_inters_nl) > 0
            if isnothing(neighbors)
                error("An interaction uses the neighbor list but neighbors is nothing")
            end
            for ni in 1:length(neighbors)
                i, j, weight_14 = neighbors.list[ni]
                dr = vector(coords[i], coords[j], boundary)
                f = force(pairwise_inters_nl[1], dr, coords[i], coords[j], atoms[i],
                          atoms[j], boundary, weight_14)
                for inter in pairwise_inters_nl[2:end]
                    f += force(inter, dr, coords[i], coords[j], atoms[i], atoms[j], boundary,
                               weight_14)
                end
                check_force_units(f, force_units)
                f_ustrip = ustrip.(f)
                fs[i] -= f_ustrip
                fs[j] += f_ustrip
            end
        end
    end

    for inter_list in sils_1_atoms
        for (i, inter) in zip(inter_list.is, inter_list.inters)
            sf = force(inter, coords[i], boundary)
            check_force_units(sf.f1, force_units)
            fs[i] += ustrip.(sf.f1)
        end
    end

    for inter_list in sils_2_atoms
        for (i, j, inter) in zip(inter_list.is, inter_list.js, inter_list.inters)
            sf = force(inter, coords[i], coords[j], boundary)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            fs[i] += ustrip.(sf.f1)
            fs[j] += ustrip.(sf.f2)
        end
    end

    for inter_list in sils_3_atoms
        for (i, j, k, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.inters)
            sf = force(inter, coords[i], coords[j], coords[k], boundary)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            check_force_units(sf.f3, force_units)
            fs[i] += ustrip.(sf.f1)
            fs[j] += ustrip.(sf.f2)
            fs[k] += ustrip.(sf.f3)
        end
    end

    for inter_list in sils_4_atoms
        for (i, j, k, l, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                                       inter_list.inters)
            sf = force(inter, coords[i], coords[j], coords[k], coords[l], boundary)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            check_force_units(sf.f3, force_units)
            check_force_units(sf.f4, force_units)
            fs[i] += ustrip.(sf.f1)
            fs[j] += ustrip.(sf.f2)
            fs[k] += ustrip.(sf.f3)
            fs[l] += ustrip.(sf.f4)
        end
    end

    return nothing
end

function forces(s::System{D, true, T}, neighbors=nothing;
                n_threads::Integer=Threads.nthreads()) where {D, T}
    n_atoms = length(s)
    fs_mat = CUDA.zeros(T, D, n_atoms)
    virial = CUDA.zeros(T, 1)

    pairwise_inters_nonl = filter(inter -> !inter.nl_only, values(s.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        nbs = NoNeighborList(n_atoms)
        fs_mat += pairwise_force_gpu(virial, s.coords, s.atoms, s.boundary,
                                     pairwise_inters_nonl, nbs, s.force_units, Val(T))
    end

    pairwise_inters_nl = filter(inter -> inter.nl_only, values(s.pairwise_inters))
    if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("An interaction uses the neighbor list but neighbors is nothing")
        end
        if length(neighbors) > 0
            nbs = @view neighbors.list[1:neighbors.n]
            fs_mat += pairwise_force_gpu(virial, s.coords, s.atoms, s.boundary,
                                         pairwise_inters_nl, nbs, s.force_units, Val(T))
        end
    end

    for inter_list in values(s.specific_inter_lists)
        fs_mat += specific_force_gpu(inter_list, s.coords, s.boundary, s.force_units, Val(T))
    end

    fs = reinterpret(SVector{D, T}, vec(fs_mat))

    for inter in values(s.general_inters)
        # Force type not checked
        fs += ustrip_vec.(forces(inter, s, neighbors; n_threads=n_threads))
    end

    return fs * s.force_units
end
