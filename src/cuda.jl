# CUDA.jl kernels

function pairwise_force_kernel!(forces::CuDeviceMatrix{T}, virial, coords_var, atoms_var, boundary,
                                inters, neighbors_i_var, neighbors_j_var, ::Val{F},
                                ::Val{M}) where {T, F, M}
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    neighbors_i = CUDA.Const(neighbors_i_var)
    neighbors_j = CUDA.Const(neighbors_j_var)

    tidx = threadIdx().x
    inter_ig = (blockIdx().x - 1) * blockDim().x + tidx
    stride = gridDim().x * blockDim().x
    shared_fs = CuStaticSharedArray(T, (3, M))
    shared_vs = CuStaticSharedArray(T, M)
    shared_is = CuStaticSharedArray(Int32, M)
    shared_js = CuStaticSharedArray(Int32, M)

    if tidx == 1
        for si in 1:M
            shared_is[si] = zero(Int32)
        end
    end
    sync_threads()

    for (thread_i, inter_i) in enumerate(inter_ig:stride:length(neighbors_i))
        si = (thread_i - 1) * blockDim().x + tidx
        i, j = neighbors_i[inter_i], neighbors_j[inter_i]
        coord_i, coord_j = coords[i], coords[j]
        dr = vector(coord_i, coord_j, boundary)
        f = force(inters[1], dr, coord_i, coord_j, atoms[i], atoms[j], boundary)
        for inter in inters[2:end]
            f += force(inter, dr, coord_i, coord_j, atoms[i], atoms[j], boundary)
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
        for si in (1 + (tidx - 1) * n_mem_to_sum):(tidx * n_mem_to_sum)
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
    return
end
