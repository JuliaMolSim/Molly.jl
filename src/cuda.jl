# CUDA.jl kernels

function cuda_threads_blocks_pairwise(n_neighbors)
    n_threads = 256
    n_blocks = cld(n_neighbors, n_threads)
    return n_threads, n_blocks
end

function cuda_threads_blocks_specific(n_inters)
    n_threads = 256
    n_blocks = cld(n_inters, n_threads)
    return n_threads, n_blocks
end

function pairwise_force_gpu(virial, coords::AbstractArray{SVector{D, C}}, atoms, boundary,
                            pairwise_inters, nbs, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(atoms))
    n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks pairwise_force_kernel!(
            fs_mat, virial, coords, atoms, boundary, pairwise_inters, nbs, Val(force_units))
    return fs_mat
end

function pairwise_force_kernel!(forces::CuDeviceMatrix{T}, virial, coords_var, atoms_var, boundary,
                                inters, neighbors_var, ::Val{F}) where {T, F}
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(neighbors)
        i, j, weight_14 = neighbors[inter_i]
        coord_i, coord_j = coords[i], coords[j]
        dr = vector(coord_i, coord_j, boundary)
        f = force(inters[1], dr, coord_i, coord_j, atoms[i], atoms[j], boundary, weight_14)
        for inter in inters[2:end]
            f += force(inter, dr, coord_i, coord_j, atoms[i], atoms[j], boundary, weight_14)
        end
        if unit(f[1]) != F
            # This triggers an error but it isn't printed
            # See https://discourse.julialang.org/t/error-handling-in-cuda-kernels/79692
            #   for how to throw a more meaningful error
            error("Wrong force unit returned, was expecting $F but got $(unit(f[1]))")
        end
        dx, dy, dz = ustrip(f[1]), ustrip(f[2]), ustrip(f[3])
        if !iszero(dx)
            Atomix.@atomic :monotonic forces[1, i] += -dx
            Atomix.@atomic :monotonic forces[1, j] +=  dx
        end
        if !iszero(dy)
            Atomix.@atomic :monotonic forces[2, i] += -dy
            Atomix.@atomic :monotonic forces[2, j] +=  dy
        end
        if !iszero(dz)
            Atomix.@atomic :monotonic forces[3, i] += -dz
            Atomix.@atomic :monotonic forces[3, j] +=  dz
        end
        rij_fij = ustrip(norm(dr) * norm(f))
        virial[] += rij_fij
    end
    return nothing
end

function specific_force_gpu(inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(coords))
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_1_atoms_kernel!(fs_mat,
            coords, boundary, inter_list.is, inter_list.inters, Val(force_units))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(coords))
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_2_atoms_kernel!(fs_mat,
            coords, boundary, inter_list.is, inter_list.js, inter_list.inters, Val(force_units))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(coords))
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_3_atoms_kernel!(fs_mat,
            coords, boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.inters,
            Val(force_units))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(coords))
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_4_atoms_kernel!(fs_mat,
            coords, boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
            inter_list.inters, Val(force_units))
    return fs_mat
end

function specific_force_1_atoms_kernel!(forces::CuDeviceMatrix{T}, coords_var, boundary, is_var,
                                        inters_var, ::Val{F}) where {T, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(is)
        i = is[inter_i]
        fs = force(inters[inter_i], coords[i], boundary)
        if unit(fs.f1[1]) != F
            error("Wrong force unit returned, was expecting $F")
        end
        Atomix.@atomic :monotonic forces[1, i] += ustrip(fs.f1[1])
        Atomix.@atomic :monotonic forces[2, i] += ustrip(fs.f1[2])
        Atomix.@atomic :monotonic forces[3, i] += ustrip(fs.f1[3])
    end
    return nothing
end

function specific_force_2_atoms_kernel!(forces::CuDeviceMatrix{T}, coords_var, boundary, is_var,
                                        js_var, inters_var, ::Val{F}) where {T, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        fs = force(inters[inter_i], coords[i], coords[j], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F
            error("Wrong force unit returned, was expecting $F")
        end
        Atomix.@atomic :monotonic forces[1, i] += ustrip(fs.f1[1])
        Atomix.@atomic :monotonic forces[2, i] += ustrip(fs.f1[2])
        Atomix.@atomic :monotonic forces[3, i] += ustrip(fs.f1[3])
        Atomix.@atomic :monotonic forces[1, j] += ustrip(fs.f2[1])
        Atomix.@atomic :monotonic forces[2, j] += ustrip(fs.f2[2])
        Atomix.@atomic :monotonic forces[3, j] += ustrip(fs.f2[3])
    end
    return nothing
end

function specific_force_3_atoms_kernel!(forces::CuDeviceMatrix{T}, coords_var, boundary, is_var,
                                        js_var, ks_var, inters_var, ::Val{F}) where {T, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        fs = force(inters[inter_i], coords[i], coords[j], coords[k], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F
            error("Wrong force unit returned, was expecting $F")
        end
        Atomix.@atomic :monotonic forces[1, i] += ustrip(fs.f1[1])
        Atomix.@atomic :monotonic forces[2, i] += ustrip(fs.f1[2])
        Atomix.@atomic :monotonic forces[3, i] += ustrip(fs.f1[3])
        Atomix.@atomic :monotonic forces[1, j] += ustrip(fs.f2[1])
        Atomix.@atomic :monotonic forces[2, j] += ustrip(fs.f2[2])
        Atomix.@atomic :monotonic forces[3, j] += ustrip(fs.f2[3])
        Atomix.@atomic :monotonic forces[1, k] += ustrip(fs.f3[1])
        Atomix.@atomic :monotonic forces[2, k] += ustrip(fs.f3[2])
        Atomix.@atomic :monotonic forces[3, k] += ustrip(fs.f3[3])
    end
    return nothing
end

function specific_force_4_atoms_kernel!(forces::CuDeviceMatrix{T}, coords_var, boundary, is_var,
                                        js_var, ks_var, ls_var, inters_var, ::Val{F}) where {T, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    ls = CUDA.Const(ls_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        fs = force(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F || unit(fs.f4[1]) != F
            error("Wrong force unit returned, was expecting $F")
        end
        Atomix.@atomic :monotonic forces[1, i] += ustrip(fs.f1[1])
        Atomix.@atomic :monotonic forces[2, i] += ustrip(fs.f1[2])
        Atomix.@atomic :monotonic forces[3, i] += ustrip(fs.f1[3])
        Atomix.@atomic :monotonic forces[1, j] += ustrip(fs.f2[1])
        Atomix.@atomic :monotonic forces[2, j] += ustrip(fs.f2[2])
        Atomix.@atomic :monotonic forces[3, j] += ustrip(fs.f2[3])
        Atomix.@atomic :monotonic forces[1, k] += ustrip(fs.f3[1])
        Atomix.@atomic :monotonic forces[2, k] += ustrip(fs.f3[2])
        Atomix.@atomic :monotonic forces[3, k] += ustrip(fs.f3[3])
        Atomix.@atomic :monotonic forces[1, l] += ustrip(fs.f4[1])
        Atomix.@atomic :monotonic forces[2, l] += ustrip(fs.f4[2])
        Atomix.@atomic :monotonic forces[3, l] += ustrip(fs.f4[3])
    end
    return nothing
end

function pairwise_pe_gpu(coords::AbstractArray{SVector{D, C}}, atoms, boundary,
                         pairwise_inters, nbs, energy_units, ::Val{T}) where {D, C, T}
    pe_vec = CUDA.zeros(T, 1)
    n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks pairwise_pe_kernel!(
            pe_vec, coords, atoms, boundary, pairwise_inters,
            nbs, Val(energy_units), Val(n_threads_gpu))
    return pe_vec
end

function pairwise_pe_kernel!(energy::CuDeviceVector{T}, coords_var, atoms_var, boundary, inters,
                             neighbors_var, ::Val{E}, ::Val{M},
                             shared_pes_arg=nothing) where {T, E, M}
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx
    shared_pes = isnothing(shared_pes_arg) ? CuStaticSharedArray(T, M) : shared_pes_arg
    shared_flags = CuStaticSharedArray(Bool, M)

    if tidx == 1
        for si in 1:M
            shared_flags[si] = false
        end
    end
    sync_threads()

    if inter_i <= length(neighbors)
        i, j, weight_14 = neighbors[inter_i]
        coord_i, coord_j = coords[i], coords[j]
        dr = vector(coord_i, coord_j, boundary)
        pe = potential_energy(inters[1], dr, coord_i, coord_j, atoms[i], atoms[j],
                              boundary, weight_14)
        for inter in inters[2:end]
            pe += potential_energy(inter, dr, coord_i, coord_j, atoms[i], atoms[j],
                                   boundary, weight_14)
        end
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        shared_pes[tidx] = ustrip(pe)
        shared_flags[tidx] = true
    end
    sync_threads()

    n_threads_sum = 16
    n_mem_to_sum = M ÷ n_threads_sum # Should be exact, not currently checked
    if tidx <= n_threads_sum
        pe_sum = zero(T)
        for si in ((tidx - 1) * n_mem_to_sum + 1):(tidx * n_mem_to_sum)
            if !shared_flags[si]
                break
            end
            pe_sum += shared_pes[si]
        end
        Atomix.@atomic :monotonic energy[] += pe_sum
    end
    return nothing
end

function specific_pe_gpu(inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                         boundary, energy_units, ::Val{T}) where {D, C, T}
    pe_vec = CUDA.zeros(T, 1)
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_pe_1_atoms_kernel!(pe_vec,
            coords, boundary, inter_list.is, inter_list.inters, Val(energy_units))
    return pe_vec
end

function specific_pe_gpu(inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                         boundary, energy_units, ::Val{T}) where {D, C, T}
    pe_vec = CUDA.zeros(T, 1)
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_pe_2_atoms_kernel!(pe_vec,
            coords, boundary, inter_list.is, inter_list.js, inter_list.inters, Val(energy_units))
    return pe_vec
end

function specific_pe_gpu(inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                         boundary, energy_units, ::Val{T}) where {D, C, T}
    pe_vec = CUDA.zeros(T, 1)
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_pe_3_atoms_kernel!(pe_vec,
            coords, boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.inters,
            Val(energy_units))
    return pe_vec
end

function specific_pe_gpu(inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                         boundary, energy_units, ::Val{T}) where {D, C, T}
    pe_vec = CUDA.zeros(T, 1)
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_pe_4_atoms_kernel!(pe_vec,
            coords, boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
            inter_list.inters, Val(energy_units))
    return pe_vec
end

function specific_pe_1_atoms_kernel!(energy::CuDeviceVector{T}, coords_var, boundary, is_var,
                                     inters_var, ::Val{E}) where {T, E}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(is)
        i = is[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], boundary)
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[] += ustrip(pe)
    end
    return nothing
end

function specific_pe_2_atoms_kernel!(energy::CuDeviceVector{T}, coords_var, boundary, is_var,
                                     js_var, inters_var, ::Val{E}) where {T, E}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], boundary)
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[] += ustrip(pe)
    end
    return nothing
end

function specific_pe_3_atoms_kernel!(energy::CuDeviceVector{T}, coords_var, boundary, is_var,
                                     js_var, ks_var, inters_var, ::Val{E}) where {T, E}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], coords[k], boundary)
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[] += ustrip(pe)
    end
    return nothing
end

function specific_pe_4_atoms_kernel!(energy::CuDeviceVector{T}, coords_var, boundary, is_var,
                                     js_var, ks_var, ls_var, inters_var, ::Val{E}) where {T, E}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    ls = CUDA.Const(ls_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_i = (blockIdx().x - 1) * blockDim().x + tidx

    if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary)
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[] += ustrip(pe)
    end
    return nothing
end
