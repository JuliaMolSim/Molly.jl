# KernelAbstractions.jl kernels

function get_array_type(a::AT) where AT <: AbstractArray
    return AT.name.wrapper
end

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

function gpu_threads_pairwise(n_neighbors)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
    return n_threads_gpu
end

function gpu_threads_specific(n_inters)
    n_threads_gpu = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_SPECIFIC", "128"))
    return n_threads_gpu
end

function pairwise_force_gpu!(buffers, sys::System{D, AT, T}, 
                    pairwise_inters, nbs, step_n) where {D, AT <: AbstractGPUArray, T}
    backend = get_backend(coords)
    if typeof(nbs) == NoNeighborList
        n_threads_gpu = gpu_threads_pairwise(length(atoms))
        kernel! = pairwise_force_kernel_nonl!(backend, n_threads_gpu)
        kernel!(buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters, step_n, Val(D), Val(force_units); ndrange = length(atoms))
    else
        n_threads_gpu = gpu_threads_pairwise(length(nbs))
        kernel! = pairwise_force_kernel_nl!(backend, n_threads_gpu)
        kernel!(buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters,
                nbs, step_n, Val(D), Val(force_units); ndrange = length(nbs))
    end
    return fs_mat
end

@kernel function pairwise_force_kernel_nl!(forces, @Const(coords),
                                           @Const(velocities), @Const(atoms),
                                           boundary, inters,
                                           @Const(neighbors), step_n, ::Val{D},
                                           ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        f = sum_pairwise_forces(inters, atoms[i], atoms[j], Val(F), special, coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
        for dim in 1:D
            fval = ustrip(f[dim])
            Atomix.@atomic forces[dim, i] = forces[dim, i] - fval
            Atomix.@atomic forces[dim, j] = forces[dim, j] + fval
        end
    end
end

@kernel function pairwise_force_kernel_nonl!(forces, @Const(coords),
                                             @Const(velocities), @Const(atoms),
                                             boundary, inters,
                                             step_n, ::Val{D},
                                             ::Val{F}) where {D, F}

    i = @index(Global, Linear)

    @inbounds for j = 1:i
        if i != j
            f = sum_pairwise_forces(inters, atoms[i], atoms[j], Val(F), false, coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
            for dim in 1:D
                fval = ustrip(f[dim])
                Atomix.@atomic forces[dim, i] = forces[dim, i] - fval
                Atomix.@atomic forces[dim, j] = forces[dim, j] + fval
            end
        end
    end
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList1Atoms, coords::AbstractArray{SVector{D, C}},
                            velocities, atoms, boundary, step_n, force_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_1_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.inters, Val(D), Val(force_units);
            ndrange = length(inter_list))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                            velocities, atoms, boundary, step_n, force_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_2_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.inters, Val(D), Val(force_units);
            ndrange = length(inter_list))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                            velocities, atoms, boundary, step_n, force_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_3_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.inters, Val(D),
            Val(force_units); ndrange = length(inter_list))
    return fs_mat
end

function specific_force_gpu!(fs_mat, inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                            velocities, atoms, boundary, step_n, force_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_4_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.inters,
            Val(D), Val(force_units); ndrange = length(inter_list))
    return fs_mat
end

@kernel function specific_force_1_atoms_kernel!(forces, @Const(coords),
                                                @Const(velocities),
                                                @Const(atoms), boundary,
                                                step_n, @Const(is),
                                                @Const(inters), ::Val{D},
                                                ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i = is[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], boundary, atoms[i], F, velocities[i], step_n)
        if unit(fs.f1[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic forces[dim, i] += ustrip(fs.f1[dim])
        end
    end
end

@kernel function specific_force_2_atoms_kernel!(forces, @Const(coords),
                                                @Const(velocities),
                                                @Const(atoms), boundary,
                                                step_n, @Const(is), @Const(js),
                                                @Const(inters), ::Val{D},
                                                ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], boundary, atoms[i], atoms[j], F,
                       velocities[i], velocities[j], step_n)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic forces[dim, j] += ustrip(fs.f2[dim])
        end
    end
end

@kernel function specific_force_3_atoms_kernel!(forces, @Const(coords),
                                                @Const(velocities),
                                                @Const(atoms), boundary,
                                                step_n, @Const(is),
                                                @Const(js), @Const(ks),
                                                @Const(inters), ::Val{D},
                                                ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary, atoms[i],
                       atoms[j], atoms[k], F, velocities[i], velocities[j], velocities[k], step_n)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            Atomix.@atomic forces[dim, i] += ustrip(fs.f1[dim])
            Atomix.@atomic forces[dim, j] += ustrip(fs.f2[dim])
            Atomix.@atomic forces[dim, k] += ustrip(fs.f3[dim])
        end
    end
end

@kernel function specific_force_4_atoms_kernel!(forces, @Const(coords),
                                                @Const(velocities),
                                                @Const(atoms), boundary,
                                                step_n, @Const(is),
                                                @Const(js), @Const(ks),
                                                @Const(ls),
                                                @Const(inters), ::Val{D},
                                                ::Val{F}) where {D, F}

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
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
        end
    end
end

function pairwise_pe_gpu!(pe_vec_nounits, buffers, sys::System{D, AT, T},
                         pairwise_inters, nbs, step_n) where {D, AT <: AbstractGPUArray, T}
    backend = get_backend(sys.coords)
    n_threads_gpu = gpu_threads_pairwise(length(nbs))
    kernel! = pairwise_pe_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters, nbs, step_n, Val(energy_units); ndrange = length(nbs))
    return pe_vec_nounits
end

@kernel function pairwise_pe_kernel!(energy, @Const(coords), @Const(velocities),
                                     @Const(atoms), boundary, inters,
                                     @Const(neighbors), step_n, ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        coord_i, coord_j, vel_i, vel_j = coords[i], coords[j], velocities[i], velocities[j]
        dr = vector(coord_i, coord_j, boundary)
        pe = potential_energy_gpu(inters[1], dr, atoms[i], atoms[j], E, special, coord_i, coord_j,
                                  boundary, vel_i, vel_j, step_n)
        for inter in inters[2:end]
            pe += potential_energy_gpu(inter, dr, atoms[i], atoms[j], E, special, coord_i, coord_j,
                                       boundary, vel_i, vel_j, step_n)
        end
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
            inter_list.inters, Val(energy_units); ndrange = length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}

    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_2_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.inters, Val(energy_units); ndrange = length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_3_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.inters, Val(energy_units);
            ndrange = length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_4_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.inters, Val(energy_units);
            ndrange = length(inter_list))
    return pe_vec_nounits
end

@kernel function specific_pe_1_atoms_kernel!(energy, @Const(coords), @Const(velocities), @Const(atoms), boundary,
                    step_n, @Const(is), @Const(inters), ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i = is[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], boundary, atoms[i], E,
                                  velocities[i], step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel function specific_pe_2_atoms_kernel!(energy, @Const(coords), @Const(velocities), @Const(atoms), boundary,
                    step_n, @Const(is), @Const(js), @Const(inters), ::Val{E}) where E


    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], boundary, atoms[i],
                                  atoms[j], E, velocities[i], velocities[j], step_n)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel function specific_pe_3_atoms_kernel!(energy, @Const(coords), @Const(velocities), @Const(atoms), boundary,
                    step_n, @Const(is), @Const(js), @Const(ks), @Const(inters), ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
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

@kernel function specific_pe_4_atoms_kernel!(energy, @Const(coords), @Const(velocities), @Const(atoms), boundary,
                    step_n, @Const(is), @Const(js), @Const(ks), @Const(ls), @Const(inters), ::Val{E}) where E

    inter_i = @index(Global, Linear)

    @inbounds if inter_i <= length(is)
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
