# CUDA.jl kernels

function cuda_threads_blocks_pairwise(n_neighbors)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
    n_blocks = cld(n_neighbors, n_threads_gpu)
    return n_threads_gpu, n_blocks
end

function cuda_threads_blocks_specific(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_SPECIFIC", "128"))
    n_blocks = cld(n_inters, n_threads_gpu)
    return n_threads_gpu, n_blocks
end

function pairwise_force_gpu(coords::AbstractArray{SVector{D, C}}, atoms, boundary,
                            pairwise_inters, nbs, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(atoms))
    n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks pairwise_force_kernel!(
            fs_mat, coords, atoms, boundary, pairwise_inters, nbs, Val(D), Val(force_units))
    return fs_mat
end

function pairwise_force_kernel!(forces, coords_var, atoms_var, boundary, inters,
                                neighbors_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        coord_i, coord_j = coords[i], coords[j]
        atom_i, atom_j = atoms[i], atoms[j]
        dr = vector(coord_i, coord_j, boundary)
        f_tuple = ntuple(length(inters)) do inter_type_i
            force_gpu(inters[inter_type_i], dr, coord_i, coord_j, atom_i, atom_j, boundary, special)
        end
        f = sum(f_tuple)
        if unit(f[1]) != F
            # This triggers an error but it isn't printed
            # See https://discourse.julialang.org/t/error-handling-in-cuda-kernels/79692
            #   for how to throw a more meaningful error
            error("wrong force unit returned, was expecting $F but got $(unit(f[1]))")
        end
        for dim in 1:D
            fval = ustrip(f[dim])
            Atomix.@atomic :monotonic forces[dim, i] += -fval
            Atomix.@atomic :monotonic forces[dim, j] +=  fval
        end
    end
    return nothing
end

function specific_force_gpu(inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(coords))
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_1_atoms_kernel!(fs_mat,
            coords, boundary, inter_list.is, inter_list.inters, Val(D), Val(force_units))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(coords))
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_2_atoms_kernel!(fs_mat,
            coords, boundary, inter_list.is, inter_list.js, inter_list.inters, Val(D),
            Val(force_units))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(coords))
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_3_atoms_kernel!(fs_mat,
            coords, boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.inters,
            Val(D), Val(force_units))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    fs_mat = CUDA.zeros(T, D, length(coords))
    n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks specific_force_4_atoms_kernel!(fs_mat,
            coords, boundary, inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
            inter_list.inters, Val(D), Val(force_units))
    return fs_mat
end

function specific_force_1_atoms_kernel!(forces, coords_var, boundary, is_var,
                                        inters_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i = is[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], boundary)
        if unit(fs.f1[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += ustrip(fs.f1[dim])
        end
    end
    return nothing
end

function specific_force_2_atoms_kernel!(forces, coords_var, boundary, is_var, js_var,
                                        inters_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic :monotonic forces[dim, j] += ustrip(fs.f2[dim])
        end
    end
    return nothing
end

function specific_force_3_atoms_kernel!(forces, coords_var, boundary, is_var, js_var, ks_var,
                                        inters_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic :monotonic forces[dim, j] += ustrip(fs.f2[dim])
            Atomix.@atomic :monotonic forces[dim, k] += ustrip(fs.f3[dim])
        end
    end
    return nothing
end

function specific_force_4_atoms_kernel!(forces, coords_var, boundary, is_var, js_var, ks_var, ls_var,
                                        inters_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    ls = CUDA.Const(ls_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F || unit(fs.f4[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic :monotonic forces[dim, j] += ustrip(fs.f2[dim])
            Atomix.@atomic :monotonic forces[dim, k] += ustrip(fs.f3[dim])
            Atomix.@atomic :monotonic forces[dim, l] += ustrip(fs.f4[dim])
        end
    end
    return nothing
end

function pairwise_pe_gpu(coords::AbstractArray{SVector{D, C}}, atoms, boundary,
                         pairwise_inters, nbs, energy_units, ::Val{T}) where {D, C, T}
    pe_vec = CUDA.zeros(T, 1)
    n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))
    CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks pairwise_pe_kernel!(
            pe_vec, coords, atoms, boundary, pairwise_inters, nbs, Val(energy_units))
    return pe_vec
end

function pairwise_pe_kernel!(energy, coords_var, atoms_var, boundary, inters, neighbors_var,
                             ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        coord_i, coord_j = coords[i], coords[j]
        dr = vector(coord_i, coord_j, boundary)
        pe = potential_energy(inters[1], dr, coord_i, coord_j, atoms[i], atoms[j],
                              boundary, special)
        for inter in inters[2:end]
            pe += potential_energy(inter, dr, coord_i, coord_j, atoms[i], atoms[j],
                                   boundary, special)
        end
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
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

function specific_pe_1_atoms_kernel!(energy, coords_var, boundary, is_var,
                                     inters_var, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i = is[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end

function specific_pe_2_atoms_kernel!(energy, coords_var, boundary, is_var, js_var,
                                     inters_var, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end

function specific_pe_3_atoms_kernel!(energy, coords_var, boundary, is_var, js_var, ks_var,
                                     inters_var, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], coords[k], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end

function specific_pe_4_atoms_kernel!(energy, coords_var, boundary, is_var, js_var, ks_var, ls_var,
                                     inters_var, ::Val{E}) where E
    coords = CUDA.Const(coords_var)
    is = CUDA.Const(is_var)
    js = CUDA.Const(js_var)
    ks = CUDA.Const(ks_var)
    ls = CUDA.Const(ls_var)
    inters = CUDA.Const(inters_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        pe = potential_energy(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end
