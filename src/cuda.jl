# CUDA.jl kernels

function cuda_threads_blocks(n_neighbors)
    n_threads = 256
    n_blocks = cld(n_neighbors, n_threads)
    return n_threads, n_blocks
end

function pairwise_force_gpu(virial, coords::AbstractArray{SVector{D, T}}, atoms, boundary,
                            pairwise_inters, nbs, force_units) where {D, T}
    fs_mat = CUDA.zeros(T, D, length(atoms))
    n_threads_gpu, n_blocks = cuda_threads_blocks(length(nbs))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks pairwise_force_kernel!(
            fs_mat, virial, coords, atoms, boundary, pairwise_inters, nbs,
            Val(force_units), Val(n_threads_gpu))
    return fs_mat
end

function pairwise_force_kernel!(forces::CuDeviceMatrix{T}, virial, coords_var, atoms_var, boundary,
                                inters, neighbors_var, ::Val{F}, ::Val{M},
                                shared_fs_arg=nothing) where {T, F, M}
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    tidx = threadIdx().x
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x
    shared_fs = isnothing(shared_fs_arg) ? CuStaticSharedArray(T, (3, M)) : shared_fs_arg
    shared_vs = CuStaticSharedArray(T, M)
    shared_is = CuStaticSharedArray(Int32, M)
    shared_js = CuStaticSharedArray(Int32, M)

    if tidx == 1
        for si in 1:M
            shared_is[si] = zero(Int32)
        end
    end
    sync_threads()

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(neighbors))
        si = (thread_i - 1) * blockDim().x + tidx
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
        shared_fs[1, si] = ustrip(f[1])
        shared_fs[2, si] = ustrip(f[2])
        shared_fs[3, si] = ustrip(f[3])
        rij_fij = ustrip(norm(dr) * norm(f))
        shared_vs[si] = rij_fij
        shared_is[si] = i
        shared_js[si] = j
    end
    sync_threads()

    n_threads_sum = 8
    n_mem_to_sum = M ÷ n_threads_sum # Should be exact, not currently checked
    if tidx <= n_threads_sum
        virial_sum = zero(T)
        for si in ((tidx - 1) * n_mem_to_sum + 1):(tidx * n_mem_to_sum)
            i = shared_is[si]
            if iszero(i)
                break
            end
            j = shared_js[si]
            dx, dy, dz = shared_fs[1, si], shared_fs[2, si], shared_fs[3, si]
            if !iszero(dx)
                CUDA.atomic_add!(pointer(forces, 3i - 2), -dx)
                CUDA.atomic_add!(pointer(forces, 3j - 2),  dx)
            end
            if !iszero(dy)
                CUDA.atomic_add!(pointer(forces, 3i - 1), -dy)
                CUDA.atomic_add!(pointer(forces, 3j - 1),  dy)
            end
            if !iszero(dz)
                CUDA.atomic_add!(pointer(forces, 3i    ), -dz)
                CUDA.atomic_add!(pointer(forces, 3j    ),  dz)
            end
            virial_sum += shared_vs[si]
        end
        CUDA.atomic_add!(pointer(virial), virial_sum)
    end
    return nothing
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList1Atoms, coords, boundary,
                             force_units)
    CUDA.@sync @cuda threads=256 blocks=64 specific_force_1_atoms_kernel!(fs_mat, coords,
            boundary, inter_list.is, inter_list.inters, Val(force_units))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList2Atoms, coords, boundary,
                             force_units)
    CUDA.@sync @cuda threads=256 blocks=64 specific_force_2_atoms_kernel!(fs_mat, coords,
            boundary, inter_list.is, inter_list.js, inter_list.inters, Val(force_units))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList3Atoms, coords, boundary,
                             force_units)
    CUDA.@sync @cuda threads=256 blocks=64 specific_force_3_atoms_kernel!(fs_mat, coords,
            boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.inters,
            Val(force_units))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList4Atoms, coords, boundary,
                             force_units)
    CUDA.@sync @cuda threads=256 blocks=64 specific_force_4_atoms_kernel!(fs_mat, coords,
            boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
            inter_list.inters, Val(force_units))
    return fs_mat
end

function specific_force_1_atoms_kernel!(forces::CuDeviceMatrix{T}, coords_var, boundary, is_var,
                                        inters_var, ::Val{F}) where {T, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(is))
        si = (thread_i - 1) * blockDim().x + tidx
        i = is[inter_i]
        fs = force(inters[inter_i], coords[i], boundary)
        if unit(fs.f1[1]) != F
            error("Wrong force unit returned, was expecting $F")
        end
        CUDA.atomic_add!(pointer(forces, 3i - 2), ustrip(fs.f1[1]))
        CUDA.atomic_add!(pointer(forces, 3i - 1), ustrip(fs.f1[2]))
        CUDA.atomic_add!(pointer(forces, 3i    ), ustrip(fs.f1[3]))
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
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(is))
        si = (thread_i - 1) * blockDim().x + tidx
        i, j = is[inter_i], js[inter_i]
        fs = force(inters[inter_i], coords[i], coords[j], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F
            error("Wrong force unit returned, was expecting $F")
        end
        CUDA.atomic_add!(pointer(forces, 3i - 2), ustrip(fs.f1[1]))
        CUDA.atomic_add!(pointer(forces, 3i - 1), ustrip(fs.f1[2]))
        CUDA.atomic_add!(pointer(forces, 3i    ), ustrip(fs.f1[3]))
        CUDA.atomic_add!(pointer(forces, 3j - 2), ustrip(fs.f2[1]))
        CUDA.atomic_add!(pointer(forces, 3j - 1), ustrip(fs.f2[2]))
        CUDA.atomic_add!(pointer(forces, 3j    ), ustrip(fs.f2[3]))
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
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(is))
        si = (thread_i - 1) * blockDim().x + tidx
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        fs = force(inters[inter_i], coords[i], coords[j], coords[k], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F
            error("Wrong force unit returned, was expecting $F")
        end
        CUDA.atomic_add!(pointer(forces, 3i - 2), ustrip(fs.f1[1]))
        CUDA.atomic_add!(pointer(forces, 3i - 1), ustrip(fs.f1[2]))
        CUDA.atomic_add!(pointer(forces, 3i    ), ustrip(fs.f1[3]))
        CUDA.atomic_add!(pointer(forces, 3j - 2), ustrip(fs.f2[1]))
        CUDA.atomic_add!(pointer(forces, 3j - 1), ustrip(fs.f2[2]))
        CUDA.atomic_add!(pointer(forces, 3j    ), ustrip(fs.f2[3]))
        CUDA.atomic_add!(pointer(forces, 3k - 2), ustrip(fs.f3[1]))
        CUDA.atomic_add!(pointer(forces, 3k - 1), ustrip(fs.f3[2]))
        CUDA.atomic_add!(pointer(forces, 3k    ), ustrip(fs.f3[3]))
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
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(is))
        si = (thread_i - 1) * blockDim().x + tidx
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        fs = force(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F || unit(fs.f4[1]) != F
            error("Wrong force unit returned, was expecting $F")
        end
        CUDA.atomic_add!(pointer(forces, 3i - 2), ustrip(fs.f1[1]))
        CUDA.atomic_add!(pointer(forces, 3i - 1), ustrip(fs.f1[2]))
        CUDA.atomic_add!(pointer(forces, 3i    ), ustrip(fs.f1[3]))
        CUDA.atomic_add!(pointer(forces, 3j - 2), ustrip(fs.f2[1]))
        CUDA.atomic_add!(pointer(forces, 3j - 1), ustrip(fs.f2[2]))
        CUDA.atomic_add!(pointer(forces, 3j    ), ustrip(fs.f2[3]))
        CUDA.atomic_add!(pointer(forces, 3k - 2), ustrip(fs.f3[1]))
        CUDA.atomic_add!(pointer(forces, 3k - 1), ustrip(fs.f3[2]))
        CUDA.atomic_add!(pointer(forces, 3k    ), ustrip(fs.f3[3]))
        CUDA.atomic_add!(pointer(forces, 3l - 2), ustrip(fs.f4[1]))
        CUDA.atomic_add!(pointer(forces, 3l - 1), ustrip(fs.f4[2]))
        CUDA.atomic_add!(pointer(forces, 3l    ), ustrip(fs.f4[3]))
    end
    return nothing
end

function pairwise_pe_kernel!(energy::CuDeviceVector{T}, coords_var, atoms_var, boundary, inters,
                             neighbors_var, ::Val{E}, ::Val{M}) where {T, E, M}
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    tidx = threadIdx().x
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x
    shared_pes = CuStaticSharedArray(T, M)
    shared_flags = CuStaticSharedArray(Bool, M)

    if tidx == 1
        for si in 1:M
            shared_flags[si] = false
        end
    end
    sync_threads()

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(neighbors))
        si = (thread_i - 1) * blockDim().x + tidx
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
        shared_pes[si] = ustrip(pe)
        shared_flags[si] = true
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
        CUDA.atomic_add!(pointer(energy), pe_sum)
    end
    return nothing
end

function specific_pe_gpu!(pe, inter_list::InteractionList1Atoms, coords, boundary, pe_units)
    CUDA.@sync @cuda threads=256 blocks=64 specific_pe_1_atoms_kernel!(pe, coords,
            boundary, inter_list.is, inter_list.inters, Val(pe_units))
    return pe
end

function specific_pe_gpu!(pe, inter_list::InteractionList2Atoms, coords, boundary, pe_units)
    CUDA.@sync @cuda threads=256 blocks=64 specific_pe_2_atoms_kernel!(pe, coords,
            boundary, inter_list.is, inter_list.js, inter_list.inters, Val(pe_units))
    return pe
end

function specific_pe_gpu!(pe, inter_list::InteractionList3Atoms, coords, boundary, pe_units)
    CUDA.@sync @cuda threads=256 blocks=64 specific_pe_3_atoms_kernel!(pe, coords,
            boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.inters, Val(pe_units))
    return pe
end

function specific_pe_gpu!(pe, inter_list::InteractionList4Atoms, coords, boundary, pe_units)
    CUDA.@sync @cuda threads=256 blocks=64 specific_pe_4_atoms_kernel!(pe, coords,
            boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
            inter_list.inters, Val(pe_units))
    return pe
end

function specific_pe_1_atoms_kernel!(energy::CuDeviceVector{T}, coords_var, boundary, is_var,
                                     inters_var, ::Val{E}) where {T, E}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    inters = CUDA.Const(inters_var)

    tidx = threadIdx().x
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(is))
        si = (thread_i - 1) * blockDim().x + tidx
        i = is[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], boundary)
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        CUDA.atomic_add!(pointer(energy), pe)
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
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(is))
        si = (thread_i - 1) * blockDim().x + tidx
        i, j = is[inter_i], js[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], boundary)
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        CUDA.atomic_add!(pointer(energy), ustrip(pe))
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
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(is))
        si = (thread_i - 1) * blockDim().x + tidx
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], coords[k], boundary)
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        CUDA.atomic_add!(pointer(energy), ustrip(pe))
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
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(is))
        si = (thread_i - 1) * blockDim().x + tidx
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary)
        if unit(pe) != E
            error("Wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        CUDA.atomic_add!(pointer(energy), ustrip(pe))
    end
    return nothing
end
