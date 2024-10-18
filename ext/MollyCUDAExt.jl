module MollyCUDAExt

using Molly
using CUDA
using ChainRulesCore
using Atomix

CUDA.Const(nl::Molly.NoNeighborList) = nl

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

function pairwise_force_gpu(coords::CuArray{SVector{D, C}}, atoms, boundary,
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

#=
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
=#
function pairwise_force_kernel_nonl!(forces::CuArray{T}, coords_var, atoms_var, boundary, inters,
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

# CUDA specific calls for Molly
@non_differentiable CUDA.zeros(args...)
@non_differentiable cuda_threads_blocks_pairwise(args...)
@non_differentiable cuda_threads_blocks_specific(args...)

end
