# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See OpenMM documentation and Gromacs manual for other aspects of forces

export
    accelerations,
    force,
    pairwise_force,
    SpecificForce1Atoms,
    SpecificForce2Atoms,
    SpecificForce3Atoms,
    SpecificForce4Atoms,
    forces

"""
    accelerations(system, neighbors=find_neighbors(sys), step_n=0; n_threads=Threads.nthreads())

Calculate the accelerations of all atoms in a system using the pairwise,
specific and general interactions and Newton's second law of motion.
"""
function accelerations(sys; n_threads::Integer=Threads.nthreads())
    return accelerations(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

function accelerations(sys, neighbors, step_n::Integer=0; n_threads::Integer=Threads.nthreads())
    return forces(sys, neighbors, step_n; n_threads=n_threads) ./ masses(sys)
end

"""
    force(inter, vec_ij, atom_i, atom_j, force_units, special, coord_i, coord_j,
          boundary, velocity_i, velocity_j, step_n)
    force(inter, coord_i, boundary, atom_i, force_units, velocity_i, step_n)
    force(inter, coord_i, coord_j, boundary, atom_i, atom_j, force_units, velocity_i,
          velocity_j, step_n)
    force(inter, coord_i, coord_j, coord_k, boundary, atom_i, atom_j, atom_k,
          force_units, velocity_i, velocity_j, velocity_k, step_n)
    force(inter, coord_i, coord_j, coord_k, coord_l, boundary, atom_i, atom_j, atom_k,
          atom_l, force_units, velocity_i, velocity_j, velocity_k, velocity_l, step_n)

Calculate the force between atoms due to a given interaction type.

For pairwise interactions returns a single force vector and for specific interactions
returns a type such as [`SpecificForce2Atoms`](@ref).
Custom pairwise and specific interaction types should implement this function.
"""
function force end

# Allow GPU-specific force functions to be defined if required
force_gpu(inter, dr, ai, aj, fu, sp, ci, cj, bnd, vi, vj, sn) = force(inter, dr, ai, aj, fu, sp, ci, cj, bnd, vi, vj, sn)
force_gpu(inter, ci, bnd, ai, fu, vi, sn) = force(inter, ci, bnd, ai, fu, vi, sn)
force_gpu(inter, ci, cj, bnd, ai, aj, fu, vi, vj, sn) = force(inter, ci, cj, bnd, ai, aj, fu, vi, vj, sn)
force_gpu(inter, ci, cj, ck, bnd, ai, aj, ak, fu, vi, vj, vk, sn) = force(inter, ci, cj, ck, bnd, ai, aj, ak, fu, vi, vj, vk, sn)
force_gpu(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, fu, vi, vj, vk, vl, sn) = force(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, fu, vi, vj, vk, vl, sn)

"""
    pairwise_force(inter, r, params)

Calculate the force magnitude between two atoms separated by distance `r` due to a
pairwise interaction.

This function is used in [`force`](@ref) to apply cutoff strategies by calculating
the force at different values of `r`.
Consequently, the parameters `params` should not include terms that depend on distance.
"""
function pairwise_force end

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

struct BuffersCPU{F, A, V, VN, VC, KT, PT}
    fs_nounits::F
    fs_chunks::A
    virial::V
    vir_nounits::VN
    vir_chunks::VC
    kin_tensor::KT
    pres_tensor::PT
end

function init_buffers!(sys::System{D, AT, T}, n_threads) where {D, AT, T}
    # Allows propagation of uncertainties to tensors
    CT = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
    fs_nounits  = ustrip_vec.(zero(sys.coords))
    vir         = zeros(CT, D, D) .* sys.energy_units
    vir_nounits = ustrip_vec.(zero(vir))
    kin         = zero(vir)
    pres        = zero(vir_nounits) .* (sys.energy_units == NoUnits ? NoUnits : u"bar")
    # Enzyme errors with nothing when n_threads is 1
    n_copies = (n_threads == 1 ? 0 : n_threads)
    fs_chunks  = [similar(fs_nounits) for _ in 1:n_copies]
    vir_chunks = [similar(vir_nounits) for _ in 1:n_copies] 
    return BuffersCPU(fs_nounits, fs_chunks, vir, vir_nounits, vir_chunks, kin, pres)
end

struct BuffersGPU{F, V, VN, VR, KT, PT, C, M, R}
    fs_mat::F
    virial::V # Main virial buffer
    virial_nounits::VN # For KernelAbstractions
    virial_row_1::VR # For pairwise GPU CUDA kernel
    virial_row_2::VR
    virial_row_3::VR
    kin_tensor::KT
    pres_tensor::PT
    box_mins::C
    box_maxs::C
    morton_seq::M
    morton_seq_buffer_1::M
    morton_seq_buffer_2::M
    compressed_eligible::R
    compressed_special::R
end

function init_buffers!(sys::System{D, AT, T}, n_threads,
                             for_pe::Bool=false) where {D, AT <: AbstractGPUArray, T}
    N = length(sys)
    C = eltype(eltype(sys.coords))
    # Allows propagation of uncertainties to tensors
    CT = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
    n_blocks = cld(N, 32)
    backend = get_backend(sys.coords)
    fs_mat       = KernelAbstractions.zeros(backend, T, D, N)
    virial       = zeros(CT, D, D) .* sys.energy_units
    virial_nu    = KernelAbstractions.zeros(backend, T, D, D)
    virial_row_1 = KernelAbstractions.zeros(backend, T, D, N)
    virial_row_2 = KernelAbstractions.zeros(backend, T, D, N)
    virial_row_3 = KernelAbstractions.zeros(backend, T, D, N)
    kin          = zero(virial)
    pres         = ustrip_vec.(zero(virial)) * (sys.energy_units == NoUnits ? NoUnits : u"bar")
    box_mins = KernelAbstractions.zeros(backend, C, n_blocks, D)
    box_maxs = KernelAbstractions.zeros(backend, C, n_blocks, D)
    morton_seq = KernelAbstractions.zeros(backend, Int32, N)
    morton_seq_buffer_1 = KernelAbstractions.zeros(backend, Int32, N)
    morton_seq_buffer_2 = KernelAbstractions.zeros(backend, Int32, N)
    compressed_eligible = KernelAbstractions.zeros(backend, UInt32, 32, n_blocks, n_blocks)
    compressed_special = KernelAbstractions.zeros(backend, UInt32, 32, n_blocks, n_blocks)
    if !for_pe && sys.neighbor_finder isa GPUNeighborFinder
        sys.neighbor_finder.initialized = false
    end
    return BuffersGPU(fs_mat, 
                      virial, virial_nu, virial_row_1, virial_row_2, virial_row_3, 
                      kin, pres, 
                      box_mins, box_maxs, 
                      morton_seq, morton_seq_buffer_1, morton_seq_buffer_2, 
                      compressed_eligible, compressed_special)
end

zero_forces(sys) = ustrip_vec.(zero(sys.coords)) .* sys.force_units

"""
    forces(system, neighbors=find_neighbors(sys), step_n=0; needs_virial=false,
           n_threads=Threads.nthreads())

Calculate the forces on all atoms in a system using the pairwise, specific and
general interactions. This call also populates the [`virial`](@ref) tensor of
the system.
"""
function forces(sys; needs_virial::Bool = false, n_threads::Integer=Threads.nthreads())
    return forces(sys, find_neighbors(sys; n_threads=n_threads); needs_virial=needs_virial,
                  n_threads=n_threads)
end

function forces(sys, neighbors, step_n::Integer=0; needs_virial::Bool=false,
                n_threads::Integer=Threads.nthreads())
    buffers = init_buffers!(sys, n_threads)
    fs = zero_forces(sys)
    forces!(fs, sys, neighbors, buffers, Val(needs_virial), step_n; n_threads=n_threads)
    if needs_virial
        return fs, buffers.virial
    else
        return fs
    end
end

function forces!(fs, sys::System{D, AT, T}, neighbors, buffers::BuffersCPU,
                 ::Val{needs_virial}, step_n::Integer=0;
                 n_threads::Integer=Threads.nthreads()) where {D, AT, T, needs_virial}
    if needs_virial
        fill!(buffers.virial, zero(T) * sys.energy_units)
    end
    fill!(buffers.kin_tensor,  zero(T) * sys.energy_units)
    fill!(buffers.pres_tensor, zero(T) * (sys.energy_units == NoUnits ? NoUnits : u"bar"))

    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    pairwise_inters_nl   = filter( use_neighbors, values(sys.pairwise_inters))
    sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(sys.specific_inter_lists))
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))

    if length(sys.pairwise_inters) > 0
        pairwise_forces_loop!(buffers.fs_nounits, buffers.fs_chunks, buffers.vir_nounits,
                buffers.vir_chunks, sys.atoms, sys.coords, sys.velocities, sys.boundary, neighbors,
                sys.force_units, length(sys), pairwise_inters_nonl, pairwise_inters_nl,
                Val(n_threads), Val(needs_virial), step_n)
    else
        fill!(buffers.fs_nounits, zero(eltype(buffers.fs_nounits)))
    end

    if length(sys.specific_inter_lists) > 0
        specific_forces!(buffers.fs_nounits, buffers.vir_nounits, sys.atoms, sys.coords,
                         sys.velocities, sys.boundary, sys.force_units, sils_1_atoms, sils_2_atoms,
                         sils_3_atoms, sils_4_atoms, Val(needs_virial), step_n)
    end

    fs .= buffers.fs_nounits .* sys.force_units
    if needs_virial
        buffers.virial .= buffers.vir_nounits .* sys.energy_units
    end

    for inter in values(sys.general_inters)
        AtomsCalculators.forces!(fs, sys, inter; buffers=buffers, needs_virial=needs_virial,
                                 neighbors=neighbors, step_n=step_n, n_threads=n_threads)
    end

    return fs, buffers
end

function pairwise_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords,
                               velocities, boundary, neighbors, force_units, n_atoms,
                               pairwise_inters_nonl, pairwise_inters_nl, ::Val{1},
                               ::Val{needs_virial}, step_n=0) where needs_virial
    fill!(fs_nounits, zero(eltype(fs_nounits)))
    if needs_virial
        fill!(vir_nounits, zero(eltype(vir_nounits)))
    end

    @inbounds if length(pairwise_inters_nonl) > 0
        for i in 1:n_atoms
            for j in (i + 1):n_atoms
                dr = vector(coords[i], coords[j], boundary)
                f = force(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], force_units, false,
                          coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                for inter in pairwise_inters_nonl[2:end]
                    f += force(inter, dr, atoms[i], atoms[j], force_units, false, coords[i],
                               coords[j], boundary, velocities[i], velocities[j], step_n)
                end
                check_force_units(f, force_units)
                f_ustrip = ustrip.(f)
                fs_nounits[i] -= f_ustrip
                fs_nounits[j] += f_ustrip

                if needs_virial
                    # Kronecker product of vector along which force acts and force itself
                    v = dr * transpose(f)
                    vir_nounits .+= ustrip.(v)
                end
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
            f = force(pairwise_inters_nl[1], dr, atoms[i], atoms[j], force_units, special,
                      coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
            for inter in pairwise_inters_nl[2:end]
                f += force(inter, dr, atoms[i], atoms[j], force_units, special, coords[i],
                           coords[j], boundary, velocities[i], velocities[j], step_n)
            end
            check_force_units(f, force_units)
            f_ustrip = ustrip.(f)
            fs_nounits[i] -= f_ustrip
            fs_nounits[j] += f_ustrip

            if needs_virial
                v = dr * transpose(f)
                vir_nounits .+= ustrip.(v)
            end

        end
    end

    return fs_nounits
end

function pairwise_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords,
                               velocities, boundary, neighbors, force_units, n_atoms,
                               pairwise_inters_nonl, pairwise_inters_nl, ::Val{n_threads},
                               ::Val{needs_virial}, step_n=0) where {n_threads, needs_virial}
    if isnothing(fs_chunks) || (needs_virial && isnothing(vir_chunks))
        throw(ArgumentError("fs_chunks / vir_chunks is not set but n_threads is > 1"))
    end
    if (length(fs_chunks) != n_threads) || (needs_virial && length(vir_chunks) != n_threads)
        throw(ArgumentError("length of fs_chunks ($(length(fs_chunks))) or vir_chunks " *
                            "($(length(vir_chunks))) does not match n_threads ($n_threads)"))
    end
    FT = eltype(fs_nounits)
    @inbounds for chunk_i in 1:n_threads
        fill!(fs_chunks[chunk_i], zero(FT))
    end
    if needs_virial
        FTv = eltype(vir_nounits)
        @inbounds for chunk_i in 1:n_threads
            fill!(vir_chunks[chunk_i], zero(FTv))
        end
    end

    @inbounds if length(pairwise_inters_nonl) > 0
        Threads.@threads for chunk_i in 1:n_threads
            for i in chunk_i:n_threads:n_atoms
                for j in (i + 1):n_atoms
                    dr = vector(coords[i], coords[j], boundary)
                    f = force(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], force_units,
                              false, coords[i], coords[j], boundary, velocities[i],
                              velocities[j], step_n)
                    for inter in pairwise_inters_nonl[2:end]
                        f += force(inter, dr, atoms[i], atoms[j], force_units, false, coords[i],
                                   coords[j], boundary, velocities[i], velocities[j], step_n)
                    end
                    check_force_units(f, force_units)
                    f_ustrip = ustrip.(f)
                    fs_chunks[chunk_i][i] -= f_ustrip
                    fs_chunks[chunk_i][j] += f_ustrip

                    if needs_virial
                        v = dr * transpose(f)
                        vir_chunks[chunk_i] .+= ustrip.(v)
                    end
                end
            end
        end
    end

    @inbounds if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        Threads.@threads for chunk_i in 1:n_threads
            for ni in chunk_i:n_threads:length(neighbors)
                i, j, special = neighbors[ni]
                dr = vector(coords[i], coords[j], boundary)
                f = force(pairwise_inters_nl[1], dr, atoms[i], atoms[j], force_units, special,
                          coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                for inter in pairwise_inters_nl[2:end]
                    f += force(inter, dr, atoms[i], atoms[j], force_units, special, coords[i],
                               coords[j], boundary, velocities[i], velocities[j], step_n)
                end
                check_force_units(f, force_units)
                f_ustrip = ustrip.(f)
                fs_chunks[chunk_i][i] -= f_ustrip
                fs_chunks[chunk_i][j] += f_ustrip

                if needs_virial
                    v = dr * transpose(f)
                    vir_chunks[chunk_i] .+= ustrip.(v)
                end
            end
        end
    end

    @inbounds fs_nounits .= fs_chunks[1]
    if needs_virial
        @inbounds vir_nounits .= vir_chunks[1]
    end
    @inbounds for chunk_i in 2:n_threads
        fs_nounits  .+= fs_chunks[chunk_i]
        if needs_virial
            vir_nounits .+= vir_chunks[chunk_i]
        end
    end

    return fs_nounits
end

function specific_forces!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary, force_units,
                          sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms,
                          ::Val{needs_virial}, step_n=0) where needs_virial
    @inbounds for inter_list in sils_1_atoms
        for (i, inter) in zip(inter_list.is, inter_list.inters)
            sf = force(inter, coords[i], boundary, atoms[i], force_units, velocities[i], step_n)
            check_force_units(sf.f1, force_units)
            fs_nounits[i] += ustrip.(sf.f1)

            if needs_virial
                r_i = coords[i]
                v = r_i * transpose(sf.f1)
                vir_nounits .+= ustrip.(v)
            end
        end
    end

    @inbounds for inter_list in sils_2_atoms
        for (i, j, inter) in zip(inter_list.is, inter_list.js, inter_list.inters)
            sf = force(inter, coords[i], coords[j], boundary, atoms[i], atoms[j], force_units,
                       velocities[i], velocities[j], step_n)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            fs_nounits[i] += ustrip.(sf.f1)
            fs_nounits[j] += ustrip.(sf.f2)

            if needs_virial
                r_ji = vector(coords[j], coords[i], boundary) # Second atom is the reference
                v = r_ji * transpose(sf.f1)
                vir_nounits .+= ustrip.(v)
            end

        end
    end

    @inbounds for inter_list in sils_3_atoms
        for (i, j, k, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.inters)
            sf = force(inter, coords[i], coords[j], coords[k], boundary, atoms[i], atoms[j],
                       atoms[k], force_units, velocities[i], velocities[j], velocities[k], step_n)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            check_force_units(sf.f3, force_units)
            fs_nounits[i] += ustrip.(sf.f1)
            fs_nounits[j] += ustrip.(sf.f2)
            fs_nounits[k] += ustrip.(sf.f3)

            if needs_virial
                r_ji = vector(coords[j], coords[i], boundary) # r_i - r_j (second atom is the reference, MIC)
                r_jk = vector(coords[j], coords[k], boundary) # r_k - r_j (second atom is the reference)
                vir_nounits .+= ustrip.(r_ji * transpose(sf.f1) + r_jk * transpose(sf.f3))
            end

        end
    end

    @inbounds for inter_list in sils_4_atoms
        for (i, j, k, l, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                                       inter_list.inters)
            sf = force(inter, coords[i], coords[j], coords[k], coords[l], boundary, atoms[i],
                       atoms[j], atoms[k], atoms[l], force_units, velocities[i], velocities[j],
                       velocities[k], velocities[l], step_n)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            check_force_units(sf.f3, force_units)
            check_force_units(sf.f4, force_units)
            fs_nounits[i] += ustrip.(sf.f1)
            fs_nounits[j] += ustrip.(sf.f2)
            fs_nounits[k] += ustrip.(sf.f3)
            fs_nounits[l] += ustrip.(sf.f4)

            if needs_virial
                r_ji = vector(coords[j], coords[i], boundary) # r_i - r_j
                r_jk = vector(coords[j], coords[k], boundary) # r_k - r_j
                r_jl = vector(coords[j], coords[l], boundary) # r_l - r_j (direct MIC, not sum)
                vir_nounits .+= ustrip.(r_ji * transpose(sf.f1) +
                                        r_jk * transpose(sf.f3) +
                                        r_jl * transpose(sf.f4) )
            end
        end
    end

    return fs_nounits
end

function forces!(fs, sys::System{D, AT, T}, neighbors, buffers::BuffersGPU, ::Val{needs_virial},
                 step_n::Integer=0; n_threads::Integer=Threads.nthreads()) where {D,
                                                        AT <: AbstractGPUArray, T, needs_virial}
    if needs_virial
        fill!(buffers.virial, zero(T) * sys.energy_units)
        fill!(buffers.virial_row_1, zero(T))
        fill!(buffers.virial_row_2, zero(T))
        fill!(buffers.virial_row_3, zero(T))
        fill!(buffers.virial_nounits, zero(T))
    end
    fill!(buffers.kin_tensor, zero(T) * sys.energy_units)
    fill!(buffers.pres_tensor, zero(T) * (sys.energy_units == NoUnits ? NoUnits : u"bar"))
    fill!(buffers.fs_mat, zero(T))

    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        n = length(sys)
        nbs = NoNeighborList(n)
        pairwise_forces_loop_gpu!(buffers, sys, pairwise_inters_nonl, nbs, Val(needs_virial), step_n)
    end

    pairwise_inters_nl = filter(use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nl) > 0
        pairwise_forces_loop_gpu!(buffers, sys, pairwise_inters_nl, neighbors, Val(needs_virial), step_n)
    end

    for inter_list in values(sys.specific_inter_lists)
        specific_forces_gpu!(buffers.fs_mat, buffers.virial_nounits,
                            inter_list, sys.coords, sys.velocities, sys.atoms,
                            sys.boundary, Val(needs_virial), step_n, sys.force_units, Val(T))
    end

    fs .= reinterpret(SVector{D, T}, vec(buffers.fs_mat)) .* sys.force_units

    if needs_virial
        buffers.virial .+= from_device(buffers.virial_nounits) .* sys.energy_units
    end

    for inter in values(sys.general_inters)
        AtomsCalculators.forces!(fs, sys, inter; buffers = buffers, needs_virial = needs_virial, neighbors = neighbors, step_n = step_n,
                                 n_threads=n_threads)
    end

    return fs, buffers
end
