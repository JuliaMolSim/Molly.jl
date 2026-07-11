# KernelAbstractions.jl kernels, CUDA kernels are in an extension

@inline function sum_pairwise_forces_gpu(inters::Tuple{T}, dr, atom_i, atom_j, ::Val{F},
                                         special, coord_i, coord_j, boundary, vel_i, vel_j,
                                         step_n) where {T, F}
    return force_gpu(inters[1], dr, atom_i, atom_j, F, special, coord_i, coord_j, boundary,
                     vel_i, vel_j, step_n)
end

@inline function sum_pairwise_forces_gpu(inters::Tuple, dr, atom_i, atom_j, ::Val{F},
                                         special, coord_i, coord_j, boundary, vel_i, vel_j,
                                         step_n) where F
    return force_gpu(first(inters), dr, atom_i, atom_j, F, special, coord_i, coord_j, boundary,
                     vel_i, vel_j, step_n) +
           sum_pairwise_forces_gpu(Base.tail(inters), dr, atom_i, atom_j, Val(F), special,
                                   coord_i, coord_j, boundary, vel_i, vel_j, step_n)
end

@inline function sum_pairwise_forces_nonl(inters::Tuple, atom_i, atom_j, ::Val{F}, special,
                                          coord_i, coord_j, boundary, vel_i, vel_j, step_n) where F
    dr = vector(coord_i, coord_j, boundary)
    f = sum_pairwise_forces_gpu(inters, dr, atom_i, atom_j, Val(F), special, coord_i, coord_j,
                                boundary, vel_i, vel_j, step_n)
    if unit(f[1]) != F
        # This triggers an error but it isn't printed
        # See https://discourse.julialang.org/t/error-handling-in-cuda-kernels/79692
        #   for how to throw a more meaningful error
        error("wrong force unit returned, was expecting $F but got $(unit(f[1]))")
    end
    return f
end

@inline function sum_pairwise_potentials_gpu(inters::Tuple{T}, dr, atom_i, atom_j, ::Val{E},
                                             special, coord_i, coord_j, boundary, vel_i, vel_j,
                                             step_n) where {T, E}
    # SVector is required to avoid a GPU error occurring with scalars
    return SVector(potential_energy_gpu(inters[1], dr, atom_i, atom_j, E, special,
                                        coord_i, coord_j, boundary, vel_i, vel_j, step_n))
end

@inline function sum_pairwise_potentials_gpu(inters::Tuple, dr, atom_i, atom_j, ::Val{E},
                                             special, coord_i, coord_j, boundary, vel_i, vel_j,
                                             step_n) where E
    return SVector(potential_energy_gpu(first(inters), dr, atom_i, atom_j, E, special,
                                        coord_i, coord_j, boundary, vel_i, vel_j, step_n)) +
           sum_pairwise_potentials_gpu(Base.tail(inters), dr, atom_i, atom_j, Val(E), special,
                                       coord_i, coord_j, boundary, vel_i, vel_j, step_n)
end

@inline function sum_pairwise_potentials_nonl(inters, atom_i, atom_j, ::Val{E}, special, coord_i,
                                              coord_j, boundary, vel_i, vel_j, step_n) where E
    dr = vector(coord_i, coord_j, boundary)
    pe = sum_pairwise_potentials_gpu(inters, dr, atom_i, atom_j, Val(E), special, coord_i,
                                     coord_j, boundary, vel_i, vel_j, step_n)
    if unit(pe[1]) != E
        error("wrong force unit returned, was expecting $E but got $(unit(pe[1]))")
    end
    return pe
end

function gpu_threads_env(name, default)
    return haskey(ENV, name) ? parse(Int, ENV[name]) : default
end

gpu_threads_pairwise(n_neighbors) = gpu_threads_env("MOLLY_GPUNTHREADS_PAIRWISE", 512)
gpu_threads_specific(n_inters) = gpu_threads_env("MOLLY_GPUNTHREADS_SPECIFIC", 32)
gpu_threads_copy(n_items) = gpu_threads_env("MOLLY_GPUNTHREADS_COPY", 256)

@inline apply_force_units_gpu(f, ::Val{force_units}) where {force_units} = f .* force_units
@inline apply_force_units_gpu(f, ::Val{NoUnits}) = f

function apply_force_units_gpu!(fs::AbstractGPUArray, fs_mat, force_units,
                                ::Val{D}, ::Val{T}) where {D, T}
    backend = get_backend(fs)
    n_threads_gpu = gpu_threads_copy(length(fs))
    kernel! = apply_force_units_kernel!(backend, n_threads_gpu)
    kernel!(fs, fs_mat, Val(force_units), Val(D), Val(T); ndrange=length(fs))
    return fs
end

@kernel inbounds=true function apply_force_units_kernel!(fs, @Const(fs_mat),
                                         ::Val{force_units}, ::Val{D},
                                         ::Val{T}) where {force_units, D, T}
    atom_i = @index(Global, Linear)
    if atom_i <= length(fs)
        f = SVector{D, T}(ntuple(dim -> fs_mat[dim, atom_i], Val(D)))
        fs[atom_i] = apply_force_units_gpu(f, Val(force_units))
    end
end

function pairwise_forces_loop_gpu!(buffers, sys::System{D, <:AbstractGPUArray},
                    pairwise_inters, neighbors, ::Val{needs_vir},
                    step_n) where {D, needs_vir}
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
        kernel!(buffers.fs_mat, buffers.virial_nounits, sys.coords, sys.velocities, sys.atoms,
                sys.boundary, pairwise_inters, nbs, step_n, Val(needs_vir), Val(D),
                Val(sys.force_units); ndrange=length(nbs))
    end
    return buffers
end

@kernel inbounds=true function pairwise_force_kernel_nl!(fs_mat, vir, @Const(coords),
                                           @Const(velocities), @Const(atoms),
                                           boundary, inters, @Const(neighbors), step_n,
                                           ::Val{needs_vir}, ::Val{D},
                                           ::Val{F}) where {needs_vir, D, F}
    inter_i = @index(Global, Linear)

    if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        coord_i = coords[i]
        coord_j = coords[j]
        dr = vector(coord_i, coord_j, boundary)
        f = sum_pairwise_forces_gpu(inters, dr, atoms[i], atoms[j], Val(F), special,
                                    coord_i, coord_j, boundary, velocities[i], velocities[j],
                                    step_n)
        for dim in 1:D
            fval = ustrip(f[dim])
            Atomix.@atomic fs_mat[dim, i] += -fval
            Atomix.@atomic fs_mat[dim, j] +=  fval
            if needs_vir
                @inbounds for alpha in 1:D
                    Atomix.@atomic vir[dim, alpha] += ustrip(dr[alpha]) * fval
                end
            end
        end
    end
end

function specific_forces_gpu!(fs_mat, vir, inter_list::InteractionList1Atoms,
                              coords::AbstractArray{SVector{D, C}}, velocities, atoms, boundary,
                              ::Val{needs_vir}, step_n, force_units,
                              ::Val{T}) where {D, C, needs_vir, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_1_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, vir, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.inters, inter_list.data, Val(needs_vir), Val(D), Val(force_units);
            ndrange=length(inter_list))
    return fs_mat
end

function specific_forces_gpu!(fs_mat, vir, inter_list::InteractionList2Atoms,
                              coords::AbstractArray{SVector{D, C}}, velocities, atoms, boundary,
                              ::Val{needs_vir}, step_n, force_units,
                              ::Val{T}) where {D, C, T, needs_vir}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_2_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, vir, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.inters, inter_list.data, Val(needs_vir), Val(D),
            Val(force_units); ndrange=length(inter_list))
    return fs_mat
end

function specific_forces_gpu!(fs_mat, vir, inter_list::InteractionList3Atoms,
                              coords::AbstractArray{SVector{D, C}}, velocities, atoms, boundary,
                              ::Val{needs_vir}, step_n, force_units,
                              ::Val{T}) where {D, C, needs_vir, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_3_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, vir, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.inters, inter_list.data, Val(needs_vir),
            Val(D), Val(force_units); ndrange=length(inter_list))
    return fs_mat
end

function specific_forces_gpu!(fs_mat, vir, inter_list::InteractionList4Atoms,
                              coords::AbstractArray{SVector{D, C}}, velocities, atoms, boundary,
                              ::Val{needs_vir}, step_n, force_units,
                              ::Val{T}) where {D, C, needs_vir, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_4_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, vir, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.inters, inter_list.data,
            Val(needs_vir), Val(D), Val(force_units); ndrange=length(inter_list))
    return fs_mat
end

function specific_forces_gpu!(fs_mat, vir, inter_list::InteractionList5Atoms,
                              coords::AbstractArray{SVector{D, C}}, velocities, atoms, boundary,
                              ::Val{needs_vir}, step_n, force_units,
                              ::Val{T}) where {D, C, needs_vir, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_force_5_atoms_kernel!(backend, n_threads_gpu)
    kernel!(fs_mat, vir, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.ms, inter_list.inters,
            inter_list.data, Val(needs_vir), Val(D), Val(force_units); ndrange=length(inter_list))
    return fs_mat
end

@kernel inbounds=true function specific_force_1_atoms_kernel!(fs_mat, vir, @Const(coords),
                                        @Const(velocities), @Const(atoms), boundary, step_n,
                                        @Const(is), @Const(inters), @Const(data), ::Val{needs_vir},
                                        ::Val{D}, ::Val{F}) where {needs_vir, D, F}
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i = is[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], boundary, atoms[i], F, velocities[i],
                       step_n, data)
        if unit(fs.f1[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            fval = ustrip(fs.f1[dim])
            Atomix.@atomic fs_mat[dim, i] += fval
            if needs_vir
                λ = λ_mixing(MinimumMixing(), atoms[i], atoms[i])
                @inbounds for alpha in 1:D
                    Atomix.@atomic vir[alpha, dim] += λ * ustrip(coords[i][alpha]) * fval
                end
            end
        end
    end
end

@kernel inbounds=true function specific_force_2_atoms_kernel!(fs_mat, vir, @Const(coords),
                                        @Const(velocities), @Const(atoms), boundary, step_n,
                                        @Const(is), @Const(js), @Const(inters), @Const(data),
                                        ::Val{needs_vir}, ::Val{D},
                                        ::Val{F}) where {needs_vir, D, F}
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], boundary, atoms[i], atoms[j], F,
                       velocities[i], velocities[j], step_n, data)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            f1val = ustrip(fs.f1[dim])
            f2val = ustrip(fs.f2[dim])
            Atomix.@atomic fs_mat[dim, i] += f1val
            Atomix.@atomic fs_mat[dim, j] += f2val
            if needs_vir
                r_ji = vector(coords[j], coords[i], boundary) # Second atom is the reference
                # Ewald exclusions are already lambda-weighted through charge scaling
                λ = inters[inter_i] isa EwaldExclusion ? 1 : λ_mixing(MinimumMixing(), atoms[i], atoms[j])
                @inbounds for alpha in 1:D
                    Atomix.@atomic vir[alpha, dim] += λ * ustrip(r_ji[alpha]) * f1val
                end
            end
        end
    end
end

@kernel inbounds=true function specific_force_3_atoms_kernel!(fs_mat, vir, @Const(coords),
                                        @Const(velocities), @Const(atoms), boundary, step_n,
                                        @Const(is), @Const(js), @Const(ks), @Const(inters),
                                        @Const(data), ::Val{needs_vir}, ::Val{D},
                                        ::Val{F}) where {needs_vir, D, F}
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary, atoms[i],
                       atoms[j], atoms[k], F, velocities[i], velocities[j], velocities[k],
                       step_n, data)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            f1val = ustrip(fs.f1[dim])
            f2val = ustrip(fs.f2[dim])
            f3val = ustrip(fs.f3[dim])
            Atomix.@atomic fs_mat[dim, i] += f1val
            Atomix.@atomic fs_mat[dim, j] += f2val
            Atomix.@atomic fs_mat[dim, k] += f3val
            if needs_vir
                r_ji = vector(coords[j], coords[i], boundary) # r_i - r_j (second atom is the reference, MIC)
                r_jk = vector(coords[j], coords[k], boundary) # r_k - r_j (second atom is the reference)
                λ_ji = λ_mixing(MinimumMixing(), atoms[j], atoms[i])
                λ_jk = λ_mixing(MinimumMixing(), atoms[j], atoms[k])
                λ = minimum((λ_ji, λ_jk))
                @inbounds for alpha in 1:D
                    Atomix.@atomic vir[alpha, dim] += (λ * ustrip(r_ji[alpha]) * f1val +
                                                       λ * ustrip(r_jk[alpha]) * f3val)
                end
            end
        end
    end
end

@kernel inbounds=true function specific_force_4_atoms_kernel!(fs_mat, vir, @Const(coords),
                                        @Const(velocities), @Const(atoms), boundary, step_n,
                                        @Const(is), @Const(js), @Const(ks), @Const(ls),
                                        @Const(inters), @Const(data), ::Val{needs_vir}, ::Val{D},
                                        ::Val{F}) where {needs_vir, D, F}
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]

        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary,
                       atoms[i], atoms[j], atoms[k], atoms[l], F, velocities[i], velocities[j],
                       velocities[k], velocities[l], step_n, data)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F || unit(fs.f4[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            f1val = ustrip(fs.f1[dim])
            f2val = ustrip(fs.f2[dim])
            f3val = ustrip(fs.f3[dim])
            f4val = ustrip(fs.f4[dim])
            Atomix.@atomic fs_mat[dim, i] += f1val
            Atomix.@atomic fs_mat[dim, j] += f2val
            Atomix.@atomic fs_mat[dim, k] += f3val
            Atomix.@atomic fs_mat[dim, l] += f4val
            if needs_vir
                r_ji = vector(coords[j], coords[i], boundary) # r_i - r_j
                r_jk = vector(coords[j], coords[k], boundary) # r_k - r_j
                r_jl = vector(coords[j], coords[l], boundary) # r_l - r_j
                λ_ji = λ_mixing(MinimumMixing(), atoms[j], atoms[i])
                λ_jk = λ_mixing(MinimumMixing(), atoms[j], atoms[k])
                λ_jl = λ_mixing(MinimumMixing(), atoms[j], atoms[l])
                λ = minimum((λ_ji, λ_jk, λ_jl))
                @inbounds for alpha in 1:D
                    Atomix.@atomic vir[alpha, dim] += (λ * ustrip(r_ji[alpha]) * f1val +
                                                       λ * ustrip(r_jk[alpha]) * f3val +
                                                       λ * ustrip(r_jl[alpha]) * f4val)
                end
            end
        end
    end
end

@kernel inbounds=true function specific_force_5_atoms_kernel!(fs_mat, vir, @Const(coords),
                                        @Const(velocities), @Const(atoms), boundary, step_n,
                                        @Const(is), @Const(js), @Const(ks), @Const(ls), @Const(ms),
                                        @Const(inters), @Const(data), ::Val{needs_vir}, ::Val{D},
                                        ::Val{F}) where {needs_vir, D, F}
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j, k, l, m = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i], ms[inter_i]

        fs = force_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l], coords[m],
                       boundary, atoms[i], atoms[j], atoms[k], atoms[l], atoms[m], F, velocities[i],
                       velocities[j], velocities[k], velocities[l], velocities[m], step_n, data)
        if unit(fs.f1[1]) != F || unit(fs.f2[1]) != F || unit(fs.f3[1]) != F ||
                        unit(fs.f4[1]) != F || unit(fs.f5[1]) != F
            error("wrong force unit returned, was expecting $F")
        end
        for dim in 1:D
            f1val = ustrip(fs.f1[dim])
            f2val = ustrip(fs.f2[dim])
            f3val = ustrip(fs.f3[dim])
            f4val = ustrip(fs.f4[dim])
            f5val = ustrip(fs.f5[dim])
            Atomix.@atomic fs_mat[dim, i] += f1val
            Atomix.@atomic fs_mat[dim, j] += f2val
            Atomix.@atomic fs_mat[dim, k] += f3val
            Atomix.@atomic fs_mat[dim, l] += f4val
            Atomix.@atomic fs_mat[dim, m] += f5val
            if needs_vir
                r_ji = vector(coords[j], coords[i], boundary) # r_i - r_j
                r_jk = vector(coords[j], coords[k], boundary) # r_k - r_j
                r_jl = vector(coords[j], coords[l], boundary) # r_l - r_j
                r_jm = vector(coords[j], coords[m], boundary) # r_m - r_j
                λ_ji = λ_mixing(MinimumMixing(), atoms[j], atoms[i])
                λ_jk = λ_mixing(MinimumMixing(), atoms[j], atoms[k])
                λ_jl = λ_mixing(MinimumMixing(), atoms[j], atoms[l])
                λ_jm = λ_mixing(MinimumMixing(), atoms[j], atoms[m])
                λ = minimum((λ_ji, λ_jk, λ_jl, λ_jm))
                @inbounds for alpha in 1:D
                    Atomix.@atomic vir[alpha, dim] += (λ * ustrip(r_ji[alpha]) * f1val +
                                                       λ * ustrip(r_jk[alpha]) * f3val +
                                                       λ * ustrip(r_jl[alpha]) * f4val +
                                                       λ * ustrip(r_jm[alpha]) * f5val)
                end
            end
        end
    end
end

function pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys::System{<:Any, <:AbstractGPUArray},
                               pairwise_inters, neighbors, step_n)
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
        pe = sum_pairwise_potentials_nonl(inters, atoms[i], atoms[j], Val(E), special, coords[i],
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
            inter_list.inters, inter_list.data, Val(energy_units); ndrange=length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList2Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_2_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.inters, inter_list.data, Val(energy_units);
            ndrange=length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList3Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_3_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.inters, inter_list.data, Val(energy_units);
            ndrange=length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList4Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_4_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.inters, inter_list.data,
            Val(energy_units); ndrange=length(inter_list))
    return pe_vec_nounits
end

function specific_pe_gpu!(pe_vec_nounits, inter_list::InteractionList5Atoms, coords::AbstractArray{SVector{D, C}},
                          velocities, atoms, boundary, step_n, energy_units, ::Val{T}) where {D, C, T}
    backend = get_backend(coords)
    n_threads_gpu = gpu_threads_specific(length(inter_list))
    kernel! = specific_pe_5_atoms_kernel!(backend, n_threads_gpu)
    kernel!(pe_vec_nounits, coords, velocities, atoms, boundary, step_n, inter_list.is,
            inter_list.js, inter_list.ks, inter_list.ls, inter_list.ms, inter_list.inters,
            inter_list.data, Val(energy_units); ndrange=length(inter_list))
    return pe_vec_nounits
end

@kernel inbounds=true function specific_pe_1_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(inters), @Const(data),
                    ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i = is[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], boundary, atoms[i], E,
                                  velocities[i], step_n, data)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel inbounds=true function specific_pe_2_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(js), @Const(inters),
                    @Const(data), ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j = is[inter_i], js[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], boundary, atoms[i],
                                  atoms[j], E, velocities[i], velocities[j], step_n, data)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel inbounds=true function specific_pe_3_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(js), @Const(ks),
                    @Const(inters), @Const(data), ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j, k = is[inter_i], js[inter_i], ks[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary,
                                  atoms[i], atoms[j], atoms[k], E, velocities[i], velocities[j],
                                  velocities[k], step_n, data)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel inbounds=true function specific_pe_4_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(js), @Const(ks),
                    @Const(ls), @Const(inters), @Const(data), ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j, k, l = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l],
                                  boundary, atoms[i], atoms[j], atoms[k], atoms[l], E,
                                  velocities[i], velocities[j], velocities[k], velocities[l],
                                  step_n, data)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

@kernel inbounds=true function specific_pe_5_atoms_kernel!(energy, @Const(coords), @Const(velocities),
                    @Const(atoms), boundary, step_n, @Const(is), @Const(js), @Const(ks),
                    @Const(ls), @Const(ms), @Const(inters), @Const(data), ::Val{E}) where E
    inter_i = @index(Global, Linear)

    if inter_i <= length(is)
        i, j, k, l, m = is[inter_i], js[inter_i], ks[inter_i], ls[inter_i], ms[inter_i]
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l],
                                  coords[m], boundary, atoms[i], atoms[j], atoms[k], atoms[l],
                                  atoms[m], E, velocities[i], velocities[j], velocities[k],
                                  velocities[l], velocities[m], step_n, data)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic energy[1] += ustrip(pe)
    end
end

#=
    sorted_morton_seq!(buffers, coords, w, morton_bits)

Compute a Morton (Z-order) sequence for a set of coordinates.
The indices are stored in `buffers.morton_seq`.
=#
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

# See https://en.wikipedia.org/wiki/Z-order_curve for Morton/Z-order indexing,
# and https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN for
# the bit-interleaving pattern used in the specialized 2D/3D fast paths.
function generalized_morton_code(indices, bits::Integer, D::Integer)
    if D == 3 && bits <= 10
        x = UInt32(indices[1]) & 0x000003ff
        y = UInt32(indices[2]) & 0x000003ff
        z = UInt32(indices[3]) & 0x000003ff
        
        x = (x | (x << 16)) & 0x030000ff
        x = (x | (x << 8))  & 0x0300f00f
        x = (x | (x << 4))  & 0x030c30c3
        x = (x | (x << 2))  & 0x09249249

        y = (y | (y << 16)) & 0x030000ff
        y = (y | (y << 8))  & 0x0300f00f
        y = (y | (y << 4))  & 0x030c30c3
        y = (y | (y << 2))  & 0x09249249

        z = (z | (z << 16)) & 0x030000ff
        z = (z | (z << 8))  & 0x0300f00f
        z = (z | (z << 4))  & 0x030c30c3
        z = (z | (z << 2))  & 0x09249249

        return Int32(x | (y << 1) | (z << 2))
    elseif D == 2 && bits <= 15
        x = UInt32(indices[1]) & 0x00007fff
        y = UInt32(indices[2]) & 0x00007fff

        x = (x | (x << 8)) & 0x00ff00ff
        x = (x | (x << 4)) & 0x0f0f0f0f
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555

        y = (y | (y << 8)) & 0x00ff00ff
        y = (y | (y << 4)) & 0x0f0f0f0f
        y = (y | (y << 2)) & 0x33333333
        y = (y | (y << 1)) & 0x55555555

        return Int32(x | (y << 1))
    else
        code = 0
        for bit in 0:(bits-1)
            for d in 1:D
                code |= ((indices[d] >> bit) & 1) << (D * bit + (d - 1))
            end
        end
        return Int32(code)
    end
end

@kernel function reorder_kernel!(reordered, @Const(original), @Const(seq))
    i = @index(Global, Linear)
    @inbounds if i <= length(original)
        reordered[i] = original[seq[i]]
    end
end

@kernel function reverse_reorder_forces_kernel!(fs_mat, @Const(fs_reordered), @Const(seq),
                                                ::Val{D}) where D
    i = @index(Global, Linear)
    @inbounds if i <= length(seq)
        orig_idx = seq[i]
        for d in 1:D
            fs_mat[d, orig_idx] += fs_reordered[d, i]
        end
    end
end

# `D` standard normals as an SVector of float type `FT`, from the Philox counter-based RNG.
# Using a generated function to unroll for arbitrary dimension D. `Float32` gets a dedicated
# method; any other `AbstractFloat` (including `Float64`) falls back to
# drawing in `Float64` and converting.
# After calling this advance ctr0 by num_philox_calls*natoms
@generated function randn_svec(::Type{SVector{D, Float32}}, ctr0::UInt64, ctr1::UInt64,
                               key::UInt64, natoms::UInt64) where {D}
    num_philox_calls = cld(D, 4)
    calls = Expr[
        :(($(Symbol(:c_, 1+4*j)), $(Symbol(:c_, 2+4*j)), $(Symbol(:c_, 3+4*j)), $(Symbol(:c_, 4+4*j)))
          = randn_f32(ctr0, ctr1, key); ctr0 += natoms;)
        for j in 0:(num_philox_calls - 1)
    ]
    args  = [:($(Symbol(:c_, d))) for d in 1:D]
    return quote
        $(calls...)
        SVector{D, Float32}($(args...))
    end
end
@generated function randn_svec(::Type{SVector{D, FT}}, ctr0::UInt64, ctr1::UInt64,
                               key::UInt64, natoms::UInt64) where {D, FT <: AbstractFloat}
    num_philox_calls = cld(D, 2)
    calls = Expr[
        :(($(Symbol(:c_, 1+2*j)), $(Symbol(:c_, 2+2*j)))
          = randn_f64(ctr0, ctr1, key); ctr0 += natoms;)
        for j in 0:(num_philox_calls - 1)
    ]
    args  = [:($(Symbol(:c_, d))) for d in 1:D]
    return quote
        $(calls...)
        SVector{D, FT}($(args...))
    end
end

# units is kT
@kernel function random_velocities_kernel!(
        vels::AbstractVector{SVector{D, C}}, @Const(masses::AbstractVector),
        units, @Const(virtual_sites), ctr1::UInt64, key::UInt64, ::Val{FT}
    ) where {D, C, FT}
    i = @index(Global, Linear)
    natoms = length(vels)%UInt64
    ctr0 = i%UInt64
    @inbounds if i <= length(vels)
        if !virtual_sites[i]
            scale = C(Base.FastMath.sqrt_fast(units / masses[i]))
            vels[i] = @inline(randn_svec(SVector{D, FT}, ctr0, ctr1, key, natoms)) * scale
        else
            vels[i] = zero(SVector{D, C})
        end
    end
end

@kernel function apply_Andersen_coupling_kernel!(
        vels::AbstractVector{SVector{D, C}}, @Const(masses::AbstractVector),
        units, prob_val_u64::UInt64, @Const(virtual_sites), ctr1::UInt64, key::UInt64, ::Val{FT}
    ) where {D, C, FT}
    i = @index(Global, Linear)
    natoms = length(vels)%UInt64
    ctr0 = i%UInt64
    @inbounds if i<= length(vels) && !virtual_sites[i]
        u0, u1 = philox4x32_10(ctr0, ctr1, key)
        rand_u64 = (UInt64(u0) | UInt64(u1)<<Int32(32))
        if rand_u64 < prob_val_u64
            ctr0 += natoms # advance the rng natoms
            scale = C(Base.FastMath.sqrt_fast(units/masses[i]))
            vels[i] = @inline(randn_svec(SVector{D, FT}, ctr0, ctr1, key, natoms)) * scale
        end
    end
end

# Fused inner part of a Langevin step
@kernel function langevin_o_step_kernel!(
        vels::AbstractVector{SVector{D, C}},
        @Const(vel_scales::AbstractVector),
        @Const(noise_scales::AbstractVector),
        philox_ctr1::UInt64,
        philox_key::UInt64,
        ::Val{FT},
    ) where {D, C, FT}
    i = @index(Global, Linear)
    natoms = length(vels)%UInt64
    philox_ctr0 = i%UInt64
    @inbounds if i<= length(vels)
        noise = @inline(randn_svec(SVector{D, FT}, philox_ctr0, philox_ctr1, philox_key, natoms))
        vels[i] = muladd(vel_scales[i], vels[i], noise*noise_scales[i])
    end
end
# host
function langevin_o_step!(
        vels::AbstractVector{SVector{D, C}},
        vel_scales::AbstractVector,
        noise_scales::AbstractVector,
        philox_ctr1::UInt64,
        philox_key::UInt64,
        ::Type{FT},
    ) where {D, C, FT}
    @assert eachindex(noise_scales) == eachindex(vels)
    @assert eachindex(noise_scales) == eachindex(vel_scales)
    natoms = UInt64(length(vels))
    @inbounds @simd ivdep for i in eachindex(vels)
        philox_ctr0 = i%UInt64
        noise = @inline(randn_svec(SVector{D, FT}, philox_ctr0, philox_ctr1, philox_key, natoms))
        vels[i] = muladd(vel_scales[i], vels[i], noise*noise_scales[i])
    end
    nothing
end
# device
function langevin_o_step!(
            vels::AbstractGPUArray,
            vel_scales::AbstractVector,
            noise_scales::AbstractVector,
            philox_ctr1::UInt64,
            philox_key::UInt64,
            ::Type{FT},
        ) where {FT}
    @assert eachindex(noise_scales) == eachindex(vels)
    @assert eachindex(noise_scales) == eachindex(vel_scales)
    backend = get_backend(vels)
    kernel! = langevin_o_step_kernel!(backend)
    kernel!(vels, vel_scales, noise_scales, philox_ctr1, philox_key,
            Val{FT}(); ndrange=length(vels))
    nothing
end
