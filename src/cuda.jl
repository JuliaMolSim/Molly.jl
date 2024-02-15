# CUDA.jl kernels
const WARPSIZE = UInt32(32)

macro shfl_multiple_sync(mask, target, width, vars...)
    all_lines = map(vars) do v
        Expr(:(=), v,
            Expr(:call, :shfl_sync,
                mask, v, target, width
            )
        )
    end
    return esc(Expr(:block, all_lines...))
end

CUDA.shfl_recurse(op, x::Quantity) = op(x.val) * unit(x)
CUDA.shfl_recurse(op, x::SVector{1, C}) where C = SVector{1, C}(op(x[1]))
CUDA.shfl_recurse(op, x::SVector{2, C}) where C = SVector{2, C}(op(x[1]), op(x[2]))
CUDA.shfl_recurse(op, x::SVector{3, C}) where C = SVector{3, C}(op(x[1]), op(x[2]), op(x[3]))

function cuda_threads_blocks_pairwise(n_neighbors)
    n_threads_gpu = min(n_neighbors, parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512")))
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

    if typeof(nbs) == NoNeighborList
        kernel = @cuda launch=false pairwise_force_kernel_nonl!(
            fs_mat, coords, atoms, boundary, pairwise_inters, Val(D), Val(force_units))
        conf = launch_configuration(kernel.fun)
        threads_basic = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
        nthreads = min(length(atoms), threads_basic, conf.threads)
        nthreads = cld(nthreads, WARPSIZE) * WARPSIZE
        n_blocks_i = cld(length(atoms), WARPSIZE)
        n_blocks_j = cld(length(atoms), nthreads)
        kernel(fs_mat, coords, atoms, boundary, pairwise_inters, Val(D), Val(force_units);
            threads=nthreads, blocks=(n_blocks_i, n_blocks_j))
    else
        n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))
        CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks pairwise_force_kernel_nl!(
                fs_mat, coords, atoms, boundary, pairwise_inters, nbs, Val(D), Val(force_units))
    end
    return fs_mat
end

function pairwise_force_kernel_nl!(forces, coords_var, atoms_var, boundary, inters,
                                neighbors_var, ::Val{D}, ::Val{F}) where {D, F}
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    neighbors = CUDA.Const(neighbors_var)

    inter_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        f = sum_pairwise_forces(inters, coords[i], coords[j], atoms[i], atoms[j], boundary, special, Val(F))
        for dim in 1:D
            fval = ustrip(f[dim])
            Atomix.@atomic :monotonic forces[dim, i] += -fval
            Atomix.@atomic :monotonic forces[dim, j] +=  fval
        end
    end
    return nothing
end

"""
**The No-neighborlist pairwise force summation kernel**: This kernel calculates all the pairwise forces in the system of
`n_atoms` atoms, this is done by dividing the complete matrix of `n_atoms`×`n_atoms` interactions into small tiles. Most 
of the tiles are of size `WARPSIZE`×`WARPSIZE`, but when `n_atoms` is not divisible by `WARPSIZE`, some tiles on the 
edges are of a different size are dealt as a separate case. The force summation for the tiles are done in the following 
way:
1. `WARPSIZE`×`WARPSIZE` tiles: For such tiles each row is assiged to a different tread in a warp which calculates the 
forces for the entire row in `WARPSIZE` steps (or `WARPSIZE - 1` steps for tiles on the diagonal of `n_atoms`×`n_atoms`
matrix of interactions). This is done such that some data can be shuffled from `i+1`'th thread to `i`'th thread in each 
subsequent iteration of the force calculation in a row. If `a, b, ...` are different atoms and `1, 2, ...` are order in 
which each thread calculates the interatomic forces, then we can represent this scenario as (considering `WARPSIZE=8`):
    ```

    × | i j k l m n o p
    --------------------
    a | 1 2 3 4 5 6 7 8
    b | 8 1 2 3 4 5 6 7
    c | 7 8 1 2 3 4 5 6
    d | 6 7 8 1 2 3 4 5
    e | 5 6 7 8 1 2 3 4
    f | 4 5 6 7 8 1 2 3
    g | 3 4 5 6 7 8 1 2
    h | 2 3 4 5 6 7 8 1


    ```

2. Edge tiles when `n_atoms` is not divisible by warpsize: In such cases, it is not possible to shuffle data generally
so there is no need to order calculations for each thread diagonally and it is also a bit more complicated to do so.
That's why the calculations are done in the following order:
    ```

    × | i j k l m n  
    ----------------
    a | 1 2 3 4 5 6
    b | 1 2 3 4 5 6
    c | 1 2 3 4 5 6
    d | 1 2 3 4 5 6
    e | 1 2 3 4 5 6
    f | 1 2 3 4 5 6
    g | 1 2 3 4 5 6
    h | 1 2 3 4 5 6

    ```
"""
function pairwise_force_kernel_nonl!(forces::AbstractArray{T}, coords_var, atoms_var, boundary, inters,
                                     ::Val{D}, ::Val{F}) where {T, D, F}
    coords = CUDA.Const(coords_var)
    atoms = CUDA.Const(atoms_var)
    n_atoms = length(atoms)

    tidx = threadIdx().x
    i_0_tile = (blockIdx().x - 1) * warpsize()
    j_0_block = (blockIdx().y - 1) * blockDim().x
    warpidx = cld(tidx, warpsize())
    j_0_tile = j_0_block + (warpidx - 1) * warpsize()
    i = i_0_tile + laneid()

    forces_shmem = CuStaticSharedArray(T, (3, 1024))
    @inbounds for dim in 1:3
        forces_shmem[dim, tidx] = zero(T)
    end

    if i_0_tile + warpsize() > n_atoms || j_0_tile + warpsize() > n_atoms
        @inbounds if i <= n_atoms
            njs = min(warpsize(), n_atoms - j_0_tile)
            atom_i, coord_i = atoms[i], coords[i]
            for del_j in 1:njs
                j = j_0_tile + del_j
                if i != j
                    atom_j, coord_j = atoms[j], coords[j]
                    f = sum_pairwise_forces(inters, coord_i, coord_j, atom_i, atom_j, boundary, false, Val(F))
                    for dim in 1:D
                        forces_shmem[dim, tidx] += -ustrip(f[dim])
                    end
                end
            end

            for dim in 1:D
                Atomix.@atomic :monotonic forces[dim, i] += forces_shmem[dim, tidx]
            end
        end
    else
        j = j_0_tile + laneid()
        tilesteps = warpsize()
        if i_0_tile == j_0_tile  # To not compute i-i forces
            j = j_0_tile + laneid() % warpsize() + 1
            tilesteps -= 1
        end

        atom_i, coord_i = atoms[i], coords[i]
        coord_j = coords[j]
        @inbounds for _ in 1:tilesteps  
            sync_warp()
            atom_j = atoms[j]
            f = sum_pairwise_forces(inters, coord_i, coord_j, atom_i, atom_j, boundary, false, Val(F))
            for dim in 1:D
                forces_shmem[dim, tidx] += -ustrip(f[dim])
            end
            @shfl_multiple_sync(FULL_MASK, laneid() + 1, warpsize(), j, coord_j)
        end

        @inbounds for dim in 1:D
            Atomix.@atomic :monotonic forces[dim, i] += forces_shmem[dim, tidx]
        end
    end

    return nothing
end

@inline function sum_pairwise_forces(inters, coord_i, coord_j, atom_i, atom_j,
                                    boundary, special, ::Val{F}) where F
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
    return f
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
        pe = potential_energy_gpu(inters[inter_i], coords[i], boundary)
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
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], boundary)
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
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], boundary)
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
        pe = potential_energy_gpu(inters[inter_i], coords[i], coords[j], coords[k], coords[l], boundary)
        if unit(pe) != E
            error("wrong energy unit returned, was expecting $E but got $(unit(pe))")
        end
        Atomix.@atomic :monotonic energy[1] += ustrip(pe)
    end
    return nothing
end
