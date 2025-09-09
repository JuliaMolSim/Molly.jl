# KernelAbstractions.jl kernels, CUDA kernels are in extension

@inline function sum_pairwise_forces(inters, atom_i, atom_j, ::Val{F}, special, coord_i, coord_j,
                                     boundary, vel_i, vel_j, step_n) where F
    dr = vector(coord_i, coord_j, boundary)
    f_tuple = ntuple(length(inters)) do inter_type_i
        force_gpu(inters[inter_type_i], dr, atom_i, atom_j, F, special, coord_i, coord_j, boundary,
                  vel_i, vel_j, step_n)
    end
    f = sum(f_tuple)
    if unit(f[1]) != F
        # This triggers an error but it isn't printed
        # See https://discourse.julialang.org/t/error-handling-in-cuda-kernels/79692
        #   for how to throw a more meaningful error
        error("wrong force unit returned, was expecting $F but got $(unit(f[1]))")
    end
    return f
end

@inline function sum_pairwise_potentials(inters, atom_i, atom_j, ::Val{E}, special, coord_i, coord_j,
                                         boundary, vel_i, vel_j, step_n) where E
    dr = vector(coord_i, coord_j, boundary)
    pe_tuple = ntuple(length(inters)) do inter_type_i
        # SVector was required to avoid a GPU error occurring with scalars
        SVector(potential_energy_gpu(inters[inter_type_i], dr, atom_i, atom_j, E, special,
                            coord_i, coord_j, boundary, vel_i, vel_j, step_n))
    end
    pe = sum(pe_tuple)
    if unit(pe[1]) != E
        error("wrong force unit returned, was expecting $E but got $(unit(pe[1]))")
    end
    return pe
end

function gpu_threads_pairwise(n_neighbors)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
    return n_threads_gpu
end

function gpu_threads_specific(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_SPECIFIC", "32"))
    return n_threads_gpu
end

function pairwise_forces_loop_gpu!(buffers, sys::System{D, AT, T},
                    pairwise_inters, neighbors, ::Val{Virial}, step_n) where {D, AT <: AbstractGPUArray, T, Virial}
    if isnothing(neighbors)
        error("neighbors is nothing, if you are using GPUNeighborFinder on a non-NVIDIA GPU you " *
              "should use DistanceNeighborFinder instead")
    end
    if typeof(neighbors) == NoNeighborList
        nbs = neighbors
    else
        nbs = @view neighbors.list[1:neighbors.n]
    end
    if length(neighbors) > 0
        backend = get_backend(sys.coords)
        n_threads_gpu = gpu_threads_pairwise(length(nbs))
        kernel! = pairwise_force_kernel_nl!(backend, n_threads_gpu)
        kernel!(buffers.fs_mat, buffers.virial_specific, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters,
                nbs, step_n, Val(Virial), Val(D), Val(sys.force_units); ndrange=length(nbs))
    end
    return buffers
end

@kernel inbounds=true function pairwise_force_kernel_nl!(forces, virial, @Const(coords),
                                           @Const(velocities), @Const(atoms),
                                           boundary, inters,
                                           @Const(neighbors), step_n, ::Val{Virial}, ::Val{D},
                                           ::Val{F}) where {Virial, D, F}
    inter_i = @index(Global, Linear)
    FT = eltype(forces)

    if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        f = sum_pairwise_forces(inters, atoms[i], atoms[j], Val(F), special, coords[i], coords[j],
                                boundary, velocities[i], velocities[j], step_n)
        for dim in 1:D
            fval = ustrip(f[dim])
            Atomix.@atomic forces[dim, i] += -fval
            Atomix.@atomic forces[dim, j] +=  fval
            if Virial
                Atomix.@atomic virial[dim, 1] += ustrip(dr[1]) * ustrip(fs.f1[dim])
                Atomix.@atomic virial[dim, 2] += ustrip(dr[2]) * ustrip(fs.f1[dim])
                Atomix.@atomic virial[dim, 3] += ustrip(dr[3]) * ustrip(fs.f1[dim])
            end
        end
    end
end

function specific_forces_gpu!(fs_mat, virial, inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                              velocities, atoms, boundary, ::Val{Virial}, step_n, force_units, ::Val{T}) where {D, C, Virial, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_1_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, virial, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.inters, Val(Virial), Val(D), Val(force_units);
            ndrange=length(inter_list))
    return fs_mat
end

function specific_forces_gpu!(fs_mat, virial, inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                              velocities, atoms, boundary, ::Val{Virial}, step_n, force_units, ::Val{T}) where {D, C, T, Virial}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_2_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, virial, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.inters, Val(Virial), Val(D), Val(force_units);
            ndrange=length(inter_list))
    return fs_mat
end

function specific_forces_gpu!(fs_mat, virial, inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                              velocities, atoms, boundary, ::Val{Virial}, step_n, force_units, ::Val{T}) where {D, C, Virial, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_3_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, virial, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.inters, Val(Virial),
            Val(D), Val(force_units); ndrange=length(inter_list))
    return fs_mat
end

function specific_forces_gpu!(fs_mat, virial, inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                              velocities, atoms, boundary, ::Val{Virial}, step_n, force_units, ::Val{T}) where {D, C, Virial, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_4_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, virial, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.inters,
            Val(Virial), Val(D), Val(force_units); ndrange=length(inter_list))
    return fs_mat
end

@kernel inbounds=true function specific_force_1_atoms_kernel!(forces, virial, @Const(coords),
                                                @Const(velocities),
                                                @Const(atoms), boundary,
                                                step_n, @Const(is),
                                                @Const(inters), ::Val{Virial}, ::Val{D},
                                                ::Val{F}) where {Virial, D, F}
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i = is[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], boundary, atoms[i], F, velocities[i], step_n)
        if unit(fs.f1[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            
            Atomix.@atomic forces[dim, i] += ustrip(fs.f1[dim])

            if Virial
                Atomix.@atomic virial[dim, 1] += ustrip(coords[i][1]) * ustrip(fs.f1[dim])
                Atomix.@atomic virial[dim, 2] += ustrip(coords[i][2]) * ustrip(fs.f1[dim])
                Atomix.@atomic virial[dim, 3] += ustrip(coords[i][3]) * ustrip(fs.f1[dim])
            end

        end
    end
end

@kernel inbounds=true function specific_force_2_atoms_kernel!(forces, virial, @Const(coords),
                                                @Const(velocities),
                                                @Const(atoms), boundary,
                                                step_n, @Const(is), @Const(js),
                                                @Const(inters), ::Val{Virial}, ::Val{D},
                                                ::Val{F}) where {Virial, D, F}
    inter_i = @index(Global, Linear)
    FT = eltype(forces)

    if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        dr = vector(coords[j], coords[i], boundary)
        fs = force_gpu(inters[inter_i], coords[i], coords[j], boundary, atoms[i], atoms[j], F,
                       velocities[i], velocities[j], step_n)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic forces[dim, j] += ustrip(fs.f2[dim])

            if Virial
                Atomix.@atomic virial[dim, 1] += ustrip(dr[1]) * ustrip(fs.f1[dim])
                Atomix.@atomic virial[dim, 2] += ustrip(dr[2]) * ustrip(fs.f1[dim])
                Atomix.@atomic virial[dim, 3] += ustrip(dr[3]) * ustrip(fs.f1[dim])
            end

        end
    end
end

@kernel inbounds=true function specific_force_3_atoms_kernel!(forces, virial, @Const(coords),
                                                @Const(velocities),
                                                @Const(atoms), boundary,
                                                step_n, @Const(is),
                                                @Const(js), @Const(ks),
                                                @Const(inters), ::Val{Virial}, ::Val{D},
                                                ::Val{F}) where {Virial, D, F}
    inter_i = @index(Global, Linear)
    FT = eltype(forces)

    if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        r_ik = vector(coords[k], coords[i], boundary)
        r_jk = vector(coords[k], coords[j], boundary)
        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary, atoms[i],
                       atoms[j], atoms[k], F, velocities[i], velocities[j], velocities[k], step_n)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic forces[dim, j] += ustrip(fs.f2[dim])
            Atomix.@atomic forces[dim, k] += ustrip(fs.f3[dim])

            if Virial
                Atomix.@atomic virial[dim, 1] += (ustrip(r_ik[1]) * ustrip(fs.f1[dim]) + ustrip(r_jk[1]) * ustrip(fs.f2[dim]))
                Atomix.@atomic virial[dim, 2] += (ustrip(r_ik[2]) * ustrip(fs.f1[dim]) + ustrip(r_jk[2]) * ustrip(fs.f2[dim]))
                Atomix.@atomic virial[dim, 3] += (ustrip(r_ik[3]) * ustrip(fs.f1[dim]) + ustrip(r_jk[3]) * ustrip(fs.f2[dim]))
            end

        end
    end
end

@kernel inbounds=true function specific_force_4_atoms_kernel!(forces, virial, @Const(coords),
                                                @Const(velocities),
                                                @Const(atoms), boundary,
                                                step_n, @Const(is),
                                                @Const(js), @Const(ks),
                                                @Const(ls),
                                                @Const(inters), ::Val{Virial}, ::Val{D},
                                                ::Val{F}) where {Virial, D, F}
    inter_i = @index(Global, Linear)
    FT = eltype(forces)

    if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        r_il = vector(coords[l], coords[i], boundary)
        r_jl = vector(coords[l], coords[j], boundary)
        r_kl = vector(coords[l], coords[k], boundary)
        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary,
                       atoms[i], atoms[j], atoms[k], atoms[l], F, velocities[i], velocities[j],
                       velocities[k], velocities[l], step_n)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F || unit(fs.f4[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic forces[dim, j] += ustrip(fs.f2[dim])
            Atomix.@atomic forces[dim, k] += ustrip(fs.f3[dim])
            Atomix.@atomic forces[dim, l] += ustrip(fs.f4[dim])

            if Virial
                Atomix.@atomic virial[dim, 1] += (ustrip(r_il[1]) * ustrip(fs.f1[dim]) + ustrip(r_jl[1]) * ustrip(fs.f2[dim]) + ustrip(r_kl[1]) * ustrip(fs.f3[dim]))
                Atomix.@atomic virial[dim, 2] += (ustrip(r_il[2]) * ustrip(fs.f1[dim]) + ustrip(r_jl[2]) * ustrip(fs.f2[dim]) + ustrip(r_kl[2]) * ustrip(fs.f3[dim]))
                Atomix.@atomic virial[dim, 3] += (ustrip(r_il[3]) * ustrip(fs.f1[dim]) + ustrip(r_jl[3]) * ustrip(fs.f2[dim]) + ustrip(r_kl[3]) * ustrip(fs.f3[dim]))
            end

        end
    end
end

function pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys::System{D, AT},
                               pairwise_inters, neighbors,
                               step_n) where {D, AT <: AbstractGPUArray}
    if isnothing(neighbors)
        error("neighbors is nothing, if you are using GPUNeighborFinder on a non-NVIDIA GPU you " *
              "should use DistanceNeighborFinder instead")
    end
    if typeof(neighbors) == NoNeighborList
        nbs = neighbors
    else
        nbs = @view neighbors.list[1:neighbors.n]
    end
    if length(neighbors) > 0
        backend = get_backend(sys.coords)
        n_threads_gpu = gpu_threads_pairwise(length(nbs))
        kernel! = pairwise_pe_kernel!(backend, n_threads_gpu)
        kernel!(pe_vec_nounits, sys.coords, sys.velocities, sys.atoms, sys.boundary,
                pairwise_inters, nbs, step_n, Val(sys.energy_units); ndrange=length(nbs))
    end
    return pe_vec_nounits
end

@kernel inbounds=true function pairwise_pe_kernel!(energy, @Const(coords), @Const(velocities),
                                     @Const(atoms), boundary, inters,
                                     @Const(neighbors), step_n, ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        pe = sum_pairwise_potentials(inters, atoms[i], atoms[j], Val(E), special, coords[i],
                                     coords[j], boundary, velocities[i], velocities[j], step_n)[1]
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_1_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.inters, Val(energy_units); ndrange=length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_2_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.inters, Val(energy_units); ndrange=length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_3_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.inters, Val(energy_units);
            ndrange=length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_4_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.inters, Val(energy_units);
            ndrange=length(inter_list))
    return pe_vec_nounits
end

@kernel inbounds=true function specific_pe_1_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(inters), ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i = is[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], boundary, atoms[i], E,
                                  velocities[i], step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel inbounds=true function specific_pe_2_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(js), @Const(inters),
                    ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], boundary, atoms[i],
                                  atoms[j], E, velocities[i], velocities[j], step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel inbounds=true function specific_pe_3_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(js), @Const(ks),
                    @Const(inters), ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary,
                                  atoms[i], atoms[j], atoms[k], E, velocities[i], velocities[j],
                                  velocities[k], step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel inbounds=true function specific_pe_4_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(js), @Const(ks),
                    @Const(ls), @Const(inters), ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l],
                                  boundary, atoms[i], atoms[j], atoms[k], atoms[l], E,
                                  velocities[i], velocities[j], velocities[k], velocities[l],
                                  step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

function sorted_morton_seq!(buffers, coords, w, morton_bits)
    backend = get_backend(buffers.morton_seq)
    n_threads_gpu = 32
    kernel! = sorted_morton_seq_kernel!(backend, n_threads_gpu)
    kernel!(buffers.morton_seq_buffer_1, coords, w, morton_bits; ndrange=length(coords))
    AcceleratedKernels.sortperm!(buffers.morton_seq, buffers.morton_seq_buffer_1;
                                 temp=buffers.morton_seq_buffer_2, block_size=512)
    return buffers
end

@kernel function sorted_morton_seq_kernel!(morton_seq,
                                           @Const(coords::AbstractVector{SVector{D, C}}),
                                           w,
                                           bits::Integer) where {D, C}
    i = @index(Global, Linear)
    @inbounds if i <= length(coords)
        scaled_coord = floor.(Int32, coords[i] ./ w)
        morton_seq[i] = generalized_morton_code(scaled_coord, bits, D)
    end
end

function generalized_morton_code(indices, bits::Integer, D::Integer)
    code = 0
    for bit in 0:(bits-1)
        for d in 1:D
            code |= ((indices[d] >> bit) & 1) << (D * bit + (d - 1))
        end
    end
    return Int32(code)
end
