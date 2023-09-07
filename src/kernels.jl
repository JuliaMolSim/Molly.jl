# KernelAbstractions.jl kernels

function get_array_type(a::AT) where AT <: AbstractArray
    return AT.name.wrapper
end

function gpu_threads_blocks_pairwise(n_neighbors)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
    return n_threads_gpu
end

function gpu_threads_blocks_specific(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_SPECIFIC", "128"))
    return n_threads_gpu
end

function pairwise_force_gpu(coords::AbstractArray{SVector{D, C}}, atoms, boundary,
                            pairwise_inters, nbs, force_units, ::Val{T}) where {D, C, T}
    fs_mat = zeros(typeof(coords), D, length(atoms))
    n_threads_gpu, n_blocks = gpu_threads_blocks_pairwise(length(nbs))
    backend = get_backend(coords)
    kernel! = pairwise_force_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, atoms, boundary, pairwise_inters, nbs,
            Val(D), Val(force_units), ndrange = length(nbs))
    return fs_mat
end

@kernel function pairwise_force_kernel!(forces, @Const(coords_var),
                                        @Const(atoms_var), boundary, inters,
                                        @Const(neighbors_var), ::Val{D},
                                        ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

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
end

function specific_force_gpu(inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    fs_mat = zeros(backend, T, D, length(coords))
    n_threads_gpu = gpu_threads_blocks_specific(length(inter_list))
    kernel! = specific_force_1_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, boundary, inter_list.is, inter_list.inters,
            Val(D), Val(force_units), ndrange = length(inter_list))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    fs_mat = zeros(backend, typeof(coords), T, D, length(coords))
    n_threads_gpu = gpu_threads_blocks_specific(length(inter_list))
    kernel! = specific_force_2_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, boundary, inter_list.is, inter_list.js,
            inter_list.inters, Val(D), Val(force_units),
            ndrange = length(inter_list))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    fs_mat = zeros(backend, T, D, length(coords))
    n_threads_gpu = gpu_threads_blocks_specific(length(inter_list))
    kernel! = specific_force_3_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, boundary, inter_list.is, inter_list.js,
            inter_list.ks, inter_list.inters, Val(D), Val(force_units),
            ndrange = length(inter_list))
    return fs_mat
end

function specific_force_gpu(inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                            boundary, force_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    fs_mat = zeros(backend, T, D, length(coords))
    n_threads_gpu = gpu_threads_blocks_specific(length(inter_list))
    kernel! = specific_force_4_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, boundary, inter_list.is, inter_list.js,
            inter_list.ks, inter_list.ls, inter_list.inters, Val(D),
            Val(force_units), ndrange = length(inter_list))
    return fs_mat
end

@kernel function specific_force_1_atoms_kernel!(forces, @Const(coords_var),
                                                boundary, @Const(is_var),
                                                @Const(inters_var), ::Val{D},
                                                ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

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
end

@kernel function specific_force_2_atoms_kernel!(forces, @Const(coords_var),
                                                boundary, @Const(is_var),
                                                @Const(js_var),
                                                @Const(inters_var), ::Val{D},
                                                ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

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
end

@kernel function specific_force_3_atoms_kernel!(forces, @Const(coords_var),
                                                boundary, @Const(is_var),
                                                @Const(js_var), @Const(ks_var),
                                                @Const(inters_var), ::Val{D},
                                                ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

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
end

@kernel function specific_force_4_atoms_kernel!(forces, @Const(coords_var),
                                                boundary, @Const(is_var),
                                                @Const(js_var), @Const(ks_var),
                                                @Const(ls_var),
                                                @Const(inters_var), ::Val{D},
                                                ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

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
end

function pairwise_pe_gpu(coords::AbstractArray{SVector{D, C}}, atoms, boundary,
                         pairwise_inters, nbs, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    pe_vec = zeros(backend, T, 1)
    n_threads_gpu = gpu_threads_blocks_pairwise(length(nbs))
    kernel! = pairwise_pe_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec, coords, atoms, boundary, pairwise_inters, nbs,
            Val(energy_units), ndrange = length(nbs))
    return pe_vec
end

@kernel function pairwise_pe_kernel!(energy, @Const(coords_var),
                                     @Const(atoms_var), boundary, inters,
                                     @Const(neighbors_var), ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        coord_i, coord_j = coords[i], coords[j]
        dr = vector(coord_i, coord_j, boundary)
        pe = potential_energy_gpu(inters[1], dr, coord_i, coord_j, atoms[i], atoms[j],
                                  boundary, special)
        for inter in inters[2:end]
            pe += potential_energy_gpu(inter, dr, coord_i, coord_j, atoms[i], atoms[j],
                                       boundary, special)
        end
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
end

function specific_pe_gpu(inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                         boundary, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    pe_vec = zeros(backend, T, 1)
    n_threads_gpu = gpu_threads_blocks_specific(length(inter_list))
    kernel! = specific_pe_1_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec, coords, boundary, inter_list.is, inter_list.inters,
            Val(energy_units), ndrange = length(inter_list))
    return pe_vec
end

function specific_pe_gpu(inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                         boundary, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    pe_vec = zeros(backend, T, 1)
    n_threads_gpu = gpu_threads_blocks_specific(length(inter_list))
    kernel! = specific_pe_2_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec, coords, boundary, inter_list.is, inter_list.js,
            inter_list.inters, Val(energy_units), ndrange = length(inter_list))
    return pe_vec
end

function specific_pe_gpu(inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                         boundary, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    pe_vec = zeros(backend, T, 1)
    n_threads_gpu = gpu_threads_blocks_specific(length(inter_list))
    kernel! = specific_pe_3_atoms_kernel!(backend, n_hreads_gpu)
    kernel!(pe_vec, coords, boundary, inter_list.is, inter_list.js,
            inter_list.ks, inter_list.inters, Val(energy_units),
            ndrange = length(inter_list))
    return pe_vec
end

function specific_pe_gpu(inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                         boundary, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    pe_vec = zeros(backend, T, 1)
    n_threads_gpu = gpu_threads_blocks_specific(length(inter_list))
    kernel! = specific_pe_4_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec, coords, boundary, inter_list.is, inter_list.js,
            inter_list.ks, inter_list.ls, inter_list.inters,
            Val(energy_units), ndrange = length(inter_list))
    return pe_vec
end

@kernel function specific_pe_1_atoms_kernel!(energy, @Const(coords_var),
                                             boundary, @Const(is_var),
                                             @Const(inters_var),
                                             ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i = is[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
end

@kernel function specific_pe_2_atoms_kernel!(energy, @Const(coords_var),
                                             boundary, @Const(is_var),
                                             @Const(js_var),
                                             @Const(inters_var),
                                             ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
end

@kernel function specific_pe_3_atoms_kernel!(energy, @Const(coords_var),
                                             boundary, @Const(is_var),
                                             @Const(js_var), @Const(ks_var),
                                             @Const(inters_var),
                                             ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
end

@kernel function specific_pe_4_atoms_kernel!(energy, @Const(coords_var),
                                             boundary, @Const(is_var),
                                             @Const(js_var), @Const(ks_var),
                                             @Const(ls_var),
                                             @Const(inters_var),
                                             ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
end
