# CUDA kernels that use warp-level features
# This file is only loaded when CUDA is imported

module MollyCUDAExt

using Molly
using Molly: from_device, box_sides, sorted_morton_seq!, sum_pairwise_forces,
             sum_pairwise_potentials
using CUDA
using Atomix
using KernelAbstractions

const WARPSIZE = UInt32(32)

Molly.uses_gpu_neighbor_finder(::Type{AT}) where {AT <: CuArray} = true

CUDA.Const(nl::Molly.NoNeighborList) = nl

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

function Molly.pairwise_force_gpu!(buffers, sys::System{D, AT, T}, pairwise_inters,
                                   nbs::Molly.NoNeighborList, step_n) where {D, AT <: CuArray, T}
    kernel = @cuda launch=false pairwise_force_kernel_nonl!(
            buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters, step_n,
            Val(D), Val(sys.force_units))
    conf = launch_configuration(kernel.fun)
    threads_basic = parse(Int, get(ENV, "MOLLY_GPUNTHREADS_PAIRWISE", "512"))
    nthreads = min(length(sys.atoms), threads_basic, conf.threads)
    nthreads = cld(nthreads, WARPSIZE) * WARPSIZE
    n_blocks_i = cld(length(sys.atoms), WARPSIZE)
    n_blocks_j = cld(length(sys.atoms), nthreads)
    kernel(buffers.fs_mat, sys.coords, sys.velocities, sys.atoms, sys.boundary, pairwise_inters,
            step_n, Val(D), Val(sys.force_units); threads=nthreads,
            blocks=(n_blocks_i, n_blocks_j))
    return buffers
end

function Molly.pairwise_force_gpu!(buffers, sys::System{D, AT, T}, pairwise_inters, nbs::Nothing,
                                   step_n) where {D, AT <: CuArray, T}
    N = length(sys.coords)
    n_blocks = cld(N, WARPSIZE)
    r_cut = sys.neighbor_finder.dist_cutoff
    if step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized
        morton_bits = 4
        w = r_cut - typeof(ustrip(r_cut))(0.1) * unit(r_cut)
        sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
        sys.neighbor_finder.initialized = true
        CUDA.@sync @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true compress_boolean_matrices!(
                buffers.morton_seq, sys.neighbor_finder.eligible, sys.neighbor_finder.special,
                buffers.compressed_eligible, buffers.compressed_special, Val(N))
    end
    if sys.boundary isa TriclinicBoundary
    #if sys.boundary isa CubicBoundary
        H = SMatrix{D, D, T}(
           (sys.boundary.basis_vectors[j][i].val for j in 1:D, i in 1:D)...
       )
        Hinv = inv(H) 
        CUDA.@sync @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
                buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, H, Hinv, Val(N),
                sys.boundary, Val(D))
        #=if step_n % sys.neighbor_finder.n_steps_reorder == 0
            println("It's a TriclinicBoundary!")
        end=#
        #=CUDA.@sync @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true force_kernel!(
            buffers.morton_seq, buffers.fs_mat, buffers.box_mins, buffers.box_maxs, sys.coords,
            sys.velocities, sys.atoms, Val(N), r_cut, Val(sys.force_units), pairwise_inters,
            sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
            Val(T), Val(D))=#
    else
        CUDA.@sync @cuda blocks=n_blocks threads=32 kernel_min_max!(
                    buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Val(N),
                    sys.boundary, Val(D))
    end
    CUDA.@sync @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true force_kernel!(
            buffers.morton_seq, buffers.fs_mat, buffers.box_mins, buffers.box_maxs, sys.coords,
            sys.velocities, sys.atoms, Val(N), r_cut, Val(sys.force_units), pairwise_inters,
            sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
            Val(T), Val(D))

    return buffers
end

function Molly.pairwise_pe_gpu!(pe_vec_nounits, buffers, sys::System{D, AT, T}, pairwise_inters,
                                nbs::Nothing, step_n) where {D, AT <: CuArray, T}
    # The ordering is always recomputed for potential energy
    # Different buffers are used to the forces case, so sys.neighbor_finder.initialized
    #   is not updated
    N = length(sys.coords)
    n_blocks = cld(N, WARPSIZE)
    r_cut = sys.neighbor_finder.dist_cutoff
    morton_bits = 4
    w = r_cut - typeof(ustrip(r_cut))(0.1) * unit(r_cut)
    sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
    if sys.boundary isa TriclinicBoundary
    #if sys.boundary isa CubicBoundary
        H = SMatrix{D, D, T}(
            (sys.boundary.basis_vectors[j][i].val for j in 1:D, i in 1:D)...
        )
        Hinv = inv(H) 
        CUDA.@sync @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
                buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, H, Hinv, Val(N),
                sys.boundary, Val(D))
        CUDA.@sync @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true energy_kernel!(
            buffers.morton_seq, pe_vec_nounits, buffers.box_mins, buffers.box_maxs, sys.coords,
            sys.velocities, sys.atoms, Val(N), r_cut, Val(sys.energy_units), pairwise_inters,
            sys.boundary, step_n, sys.neighbor_finder.special, sys.neighbor_finder.eligible,
            Val(T), Val(D))
    else
        CUDA.@sync @cuda blocks=cld(N, WARPSIZE) threads=32 kernel_min_max!(
                buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords,
                Val(N), sys.boundary, Val(D))
        CUDA.@sync @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true energy_kernel!(
                buffers.morton_seq, pe_vec_nounits, buffers.box_mins, buffers.box_maxs, sys.coords,
                sys.velocities, sys.atoms, Val(N), r_cut, Val(sys.energy_units), pairwise_inters,
                sys.boundary, step_n, sys.neighbor_finder.special, sys.neighbor_finder.eligible,
                Val(T), Val(D))
    end
    return pe_vec_nounits
end

function boxes_dist(r1_min::SVector{D, T}, r1_max::SVector{D, T}, r2_min::SVector{D, T}, r2_max::SVector{D, T}, boundary) where {D, T}
    va = vector(r2_max, r1_min, boundary)
    vb = vector(r1_max, r2_min, boundary)
    a = SVector(abs(va[1]), abs(va[2]), abs(va[3]))
    b = SVector(abs(vb[1]), abs(vb[2]), abs(vb[3]))

    return SVector(
        r1_min[1] - r2_max[1] <= zero(T) && r2_min[1] - r1_max[1] <= zero(T) ? zero(T) : ifelse(a[1] < b[1], a[1], b[1]),
        r1_min[2] - r2_max[2] <= zero(T) && r2_min[2] - r1_max[2] <= zero(T) ? zero(T) : ifelse(a[2] < b[2], a[2], b[2]),
        r1_min[3] - r2_max[3] <= zero(T) && r2_min[3] - r1_max[3] <= zero(T) ? zero(T) : ifelse(a[3] < b[3], a[3], b[3])
    )
end

function boxes_dist(r1_min::SVector{D, T}, r1_max::SVector{D, T}, r2_min::SVector{D, T}, r2_max::SVector{D, T}, boundary::TriclinicBoundary) where {D, T}
    r3 = r2_max - r1_min
    r2 = r3 - boundary.basis_vectors[3] .* round(r3[3] / boundary.basis_vectors[3][3])
    r1 = r2 - boundary.basis_vectors[2] .* round(r2[2] / boundary.basis_vectors[2][2])
    r_a = boundary.basis_vectors[1] .* round(r1[1] / boundary.basis_vectors[1][1])
    a = SVector(abs(r_a[1]), abs(r_a[2]), abs(r_a[3]))
    r3 = r1_max - r2_min
    r2 = r3 - boundary.basis_vectors[3] .* round(r3[3] / boundary.basis_vectors[3][3])
    r1 = r2 - boundary.basis_vectors[2] .* round(r2[2] / boundary.basis_vectors[2][2])
    r_b = boundary.basis_vectors[1] .* round(r1[1] / boundary.basis_vectors[1][1])
    b = SVector(abs(r_b[1]), abs(r_b[2]), abs(r_b[3]))

    return SVector(
        r_a[1] >= zero(T) && r_b[1] >= zero(T) ? zero(T) : ifelse(a[1] < b[1], a[1], b[1]),
        r_a[2] >= zero(T) && r_b[2] >= zero(T) ? zero(T) : ifelse(a[2] < b[2], a[2], b[2]),
        r_a[3] >= zero(T) && r_b[3] >= zero(T) ? zero(T) : ifelse(a[3] < b[3], a[3], b[3])
    )
end

function kernel_min_max!(
    sorted_seq,
    mins::AbstractArray{C}, 
    maxs::AbstractArray{C}, 
    coords, 
    ::Val{n}, 
    boundary,
    ::Val{D}) where {n, C, D}

    D32 = Int32(32)
    a = Int32(1)
    b = Int32(D)
    r = Int32(n % D32)
    i = threadIdx().x + (blockIdx().x - a) * blockDim().x
    local_i = threadIdx().x
    mins_smem = CuStaticSharedArray(C, (D32, b))
    maxs_smem = CuStaticSharedArray(C, (D32, b))
    r_smem = CuStaticSharedArray(C, (r, b))
 
    if i <= n - r && local_i <= D32
        s_i = sorted_seq[i]
        for k in a:b
            mins_smem[local_i, k] = coords[s_i][k] 
            maxs_smem[local_i, k] = coords[s_i][k]
        end
    end
    sync_threads() 
    if i <= n - r && local_i <= D32
        for p in a:Int32(log2(D32))
            for k in a:b
                @inbounds begin
                    if local_i % Int32(2^p) == Int32(0)
                        if mins_smem[local_i, k] > mins_smem[local_i - Int32(2^(p - 1)), k] 
                            mins_smem[local_i, k] = mins_smem[local_i - Int32(2^(p - 1)), k]
                        end
                        if maxs_smem[local_i, k] < maxs_smem[local_i - Int32(2^(p - 1)), k] 
                            maxs_smem[local_i, k] = maxs_smem[local_i - Int32(2^(p - 1)), k]
                        end
                    end
                end
            end
        end 
        if local_i == D32 
            for k in a:b
                mins[blockIdx().x, k] = mins_smem[local_i, k]
                maxs[blockIdx().x, k] = maxs_smem[local_i, k]
            end
        end

    end 

    # Since the remainder array is low-dimensional, we do the scan
    if i > n - r && i <= n && local_i <= r
        for k in a:b
            r_smem[local_i, k] = coords[sorted_seq[i]][k]
        end
    end
    xyz_min = CuStaticSharedArray(C, b)
    xyz_max = CuStaticSharedArray(C, b)
    for k in a:b
        xyz_min[k] =  10 * box_sides(boundary, k) # very large (arbitrary) value
        xyz_max[k] = -10 * box_sides(boundary, k)
    end
    if local_i == a
        for j in a:r
            @inbounds begin
                for k in a:b
                    if r_smem[j, k] < xyz_min[k] 
                        xyz_min[k] = r_smem[j, k]
                    end
                    if r_smem[j, k] > xyz_max[k] 
                        xyz_max[k] = r_smem[j, k]
                    end
                end
            end
        end
        if blockIdx().x == ceil(Int32, n/D32) && r != Int32(0)
            for k in a:b
                mins[blockIdx().x, k] = xyz_min[k] 
                maxs[blockIdx().x, k] = xyz_max[k]
            end
        end
    end

    return nothing
end

function kernel_min_max_triclinic!(
    sorted_seq,
    mins::AbstractArray{C}, 
    maxs::AbstractArray{C}, 
    coords, 
    H,
    Hinv, 
    ::Val{n}, 
    boundary,
    ::Val{D}) where {n, C, D}

    D32 = Int32(32)
    a = Int32(1)
    b = Int32(D)
    r = Int32(n % D32)
    i = threadIdx().x + (blockIdx().x - a) * blockDim().x
    local_i = threadIdx().x
    mins_smem = CuStaticSharedArray(C, (D32, b))
    maxs_smem = CuStaticSharedArray(C, (D32, b))
    r_smem = CuStaticSharedArray(C, (r, b))
 
    if i <= n - r && local_i <= D32
        s_i = sorted_seq[i]
        r_i = coords[s_i]
        @inbounds for k in a:b
            mins_smem[local_i, k] = Hinv[k,1]*r_i[1] + Hinv[k,2]*r_i[2] + Hinv[k,3]*r_i[3] # Transform to fractional space: s = Hinv * r
            maxs_smem[local_i, k] = Hinv[k,1]*r_i[1] + Hinv[k,2]*r_i[2] + Hinv[k,3]*r_i[3]
        end
    end
    sync_threads() 
    if i <= n - r && local_i <= D32
        for p in a:Int32(log2(D32))
            for k in a:b
                @inbounds begin
                    if local_i % Int32(2^p) == Int32(0)
                        if mins_smem[local_i, k] > mins_smem[local_i - Int32(2^(p - 1)), k] 
                            mins_smem[local_i, k] = mins_smem[local_i - Int32(2^(p - 1)), k]
                        end
                        if maxs_smem[local_i, k] < maxs_smem[local_i - Int32(2^(p - 1)), k] 
                            maxs_smem[local_i, k] = maxs_smem[local_i - Int32(2^(p - 1)), k]
                        end
                    end
                end
            end
        end 
        if local_i == D32 
            for k in a:b
                mins[blockIdx().x, k] = mins_smem[local_i, k]
                maxs[blockIdx().x, k] = maxs_smem[local_i, k]
            end
        end

    end 

    # Since the remainder array is low-dimensional, we do the scan
    if i > n - r && i <= n && local_i <= r
        r_i = coords[sorted_seq[i]]
        for k in a:b
            r_smem[local_i, k] = Hinv[k,1]*r_i[1] + Hinv[k,2]*r_i[2] + Hinv[k,3]*r_i[3] # Transform to fractional space: s = Hinv * r
        end
    end
    xyz_min = CuStaticSharedArray(C, b)
    xyz_max = CuStaticSharedArray(C, b)
    for k in a:b
        xyz_min[k] =  10 * box_sides(boundary, k) # very large (arbitrary) value
        xyz_max[k] = -10 * box_sides(boundary, k)
    end
    if local_i == a
        for j in a:r
            @inbounds begin
                for k in a:b
                    if r_smem[j, k] < xyz_min[k] 
                        xyz_min[k] = r_smem[j, k]
                    end
                    if r_smem[j, k] > xyz_max[k] 
                        xyz_max[k] = r_smem[j, k]
                    end
                end
            end
        end
        if blockIdx().x == ceil(Int32, n/D32) && r != Int32(0)
            for k in a:b
                mins[blockIdx().x, k] = xyz_min[k] 
                maxs[blockIdx().x, k] = xyz_max[k]
            end
        end
    end

    sync_threads()
    if local_i == a
        for k in a:b
            mins[blockIdx().x, k] = H[k, 1]*mins[blockIdx().x, 1] + 
                                    H[k, 2]*mins[blockIdx().x, 2] + 
                                    H[k, 3]*mins[blockIdx().x, 3]
            maxs[blockIdx().x, k] = H[k, 1]*maxs[blockIdx().x, 1] + 
                                    H[k, 2]*maxs[blockIdx().x, 2] + 
                                    H[k, 3]*maxs[blockIdx().x, 3] 
        end
    end

    return nothing
end

function compress_boolean_matrices!(sorted_seq, eligible_matrix, special_matrix,
                                    compressed_eligible, compressed_special, ::Val{N}) where N
    a = Int32(1)
    n_blocks = ceil(Int32, N / 32)
    r = Int32((N - 1) % 32 + 1)
    i = blockIdx().x
    j = blockIdx().y
    i_0_tile = (i - a) * warpsize()
    j_0_tile = (j - a) * warpsize()
    index_i = i_0_tile + laneid()
    index_j = j_0_tile + laneid()

    if j < n_blocks && i <= j 
        s_idx_i = sorted_seq[index_i]
        eligible_bitmask = UInt32(0)
        special_bitmask = UInt32(0)
        for m in a:warpsize()
            s_idx_j = sorted_seq[j_0_tile + m]
            eligible_bitmask = (eligible_bitmask << 1) | UInt32(eligible_matrix[s_idx_i, s_idx_j])
            special_bitmask = (special_bitmask << 1) | UInt32(special_matrix[s_idx_i, s_idx_j])
        end
        compressed_eligible[laneid(), i, j] = eligible_bitmask
        compressed_special[laneid(), i, j] = special_bitmask
    end

    if j == n_blocks && i < j
        s_idx_i = sorted_seq[index_i]
        eligible_bitmask = UInt32(0)
        special_bitmask = UInt32(0)
        for m in a:r
            s_idx_j = sorted_seq[j_0_tile + m]
            eligible_bitmask = (eligible_bitmask << 1) | UInt32(eligible_matrix[s_idx_i, s_idx_j])
            special_bitmask = (special_bitmask << 1) | UInt32(special_matrix[s_idx_i, s_idx_j])
        end
        eligible_bitmask = (eligible_bitmask >> r) | (eligible_bitmask << (warpsize() - r))
        special_bitmask = (special_bitmask >> r) | (special_bitmask << (warpsize() - r))
        compressed_eligible[laneid(), i, j] = eligible_bitmask
        compressed_special[laneid(), i, j] = special_bitmask
    end

    if j == n_blocks && i == j && laneid() <= r
        s_idx_i = sorted_seq[index_i]
        eligible_bitmask = UInt32(0)
        special_bitmask = UInt32(0)
        for m in a:r
            s_idx_j = sorted_seq[j_0_tile + m]
            eligible_bitmask = (eligible_bitmask << 1) | UInt32(eligible_matrix[s_idx_i, s_idx_j])
            special_bitmask = (special_bitmask << 1) | UInt32(special_matrix[s_idx_i, s_idx_j])
        end
        eligible_bitmask = (eligible_bitmask >> r) | (eligible_bitmask << (warpsize() - r))
        special_bitmask = (special_bitmask >> r) | (special_bitmask << (warpsize() - r))
        compressed_eligible[laneid(), i, j] = eligible_bitmask
        compressed_special[laneid(), i, j] = special_bitmask
    end
    return nothing
end

#=
**The No-neighborlist pairwise force summation kernel (algorithm by Eastman, see https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.21413)**: 
1. Case j < n_blocks && i < j, i.e., `WARPSIZE`×`WARPSIZE` tiles: For such tiles each row is assiged to a different thread in a warp which calculates the
forces for the entire row in `WARPSIZE` steps. This is done such that some data can be shuffled from `i+1`'th thread to `i`'th thread in each
subsequent iteration of the force calculation in a row. If `a, b, ...` are different atoms and `1, 2, ...` are order in which each thread calculates
the interatomic forces, then we can represent this scenario as (considering `WARPSIZE=8`):
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

2. Cases j == n_blocks && i < n_blocks, i == j && i < n_blocks, i == n_blocks && j == n_blocks: In such cases, it is not possible to shuffle data generally
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

function force_kernel!( 
    sorted_seq,
    fs_mat, 
    mins::AbstractArray{C}, 
    maxs::AbstractArray{C},
    coords, 
    velocities,
    atoms,
    ::Val{N}, 
    r_cut, 
    ::Val{force_units},
    inters_tuple,
    boundary,
    step_n,
    special_compressed,
    eligible_compressed,
    ::Val{T},
    ::Val{D}) where {N, C, force_units, T, D}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)
    i = blockIdx().x
    j = blockIdx().y
    i_0_tile = (i - a) * warpsize()
    j_0_tile = (j - a) * warpsize()
    index_i = i_0_tile + laneid()
    index_j = j_0_tile + laneid()
    force_smem = CuStaticSharedArray(T, (32, 3))
    opposites_sum = CuStaticSharedArray(T, (32, 3))
    r = Int32((N - 1) % 32 + 1)
    @inbounds for k in a:b
        force_smem[laneid(), k] = zero(T)
        opposites_sum[laneid(), k] = zero(T)
    end

    # The code is organised in 4 mutually excluding parts
    if j < n_blocks && i < j
        dist_block = zero(C) * zero(C)
        r_max_i = SVector(maxs[i, 1], maxs[i, 2], maxs[i, 3])
        r_min_i = SVector(mins[i, 1], mins[i, 2], mins[i, 3])
        r_max_j = SVector(maxs[j, 1], maxs[j, 2], maxs[j, 3])
        r_min_j = SVector(mins[j, 1], mins[j, 2], mins[j, 3])
        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)
        dist_block = sum(abs2, d_block)
        if dist_block <= r_cut * r_cut
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i] 
            vel_i = velocities[s_idx_i] 
            atoms_i = atoms[s_idx_i]
            d_pb = boxes_dist(coords_i, coords_i, r_min_j, r_max_j, boundary)
            dist_pb = sum(abs2, d_pb)

            Bool_excl = dist_pb <= r_cut * r_cut
            s_idx_j = sorted_seq[index_j]
            coords_j = coords[s_idx_j]
            vel_j = velocities[s_idx_j] 
            shuffle_idx = laneid()
            atoms_j = atoms[s_idx_j]
            atype_j = atoms_j.atom_type
            aindex_j = atoms_j.index
            amass_j = atoms_j.mass
            acharge_j = atoms_j.charge
            aσ_j = atoms_j.σ
            aϵ_j = atoms_j.ϵ
            eligible_bitmask = UInt32(0)
            special_bitmask = UInt32(0)
            eligible_bitmask = eligible_compressed[laneid(), i, j]
            special_bitmask = special_compressed[laneid(), i, j]

            # Shuffle
            for m in a:warpsize()
                sync_warp()
                coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, laneid() + a, warpsize())
                vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, laneid() + a, warpsize())
                shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, laneid() + a, warpsize())
                atype_j = CUDA.shfl_sync(0xFFFFFFFF, atype_j, laneid() + a, warpsize())
                aindex_j = CUDA.shfl_sync(0xFFFFFFFF, aindex_j, laneid() + a, warpsize())
                amass_j = CUDA.shfl_sync(0xFFFFFFFF, amass_j, laneid() + a, warpsize())
                acharge_j = CUDA.shfl_sync(0xFFFFFFFF, acharge_j, laneid() + a, warpsize())
                aσ_j = CUDA.shfl_sync(0xFFFFFFFF, aσ_j, laneid() + a, warpsize())
                aϵ_j = CUDA.shfl_sync(0xFFFFFFFF, aϵ_j, laneid() + a, warpsize())
                
                atoms_j_shuffle = Atom(atype_j, aindex_j, amass_j, acharge_j, aσ_j, aϵ_j)
                dr = vector(coords_j, coords_i, boundary)
                r2 = sum(abs2, dr)
                dr = 
                excl = (eligible_bitmask >> (warpsize() - shuffle_idx)) | (eligible_bitmask << shuffle_idx)
                spec = (special_bitmask >> (warpsize() - shuffle_idx)) | (special_bitmask << shuffle_idx)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                f = condition ? sum_pairwise_forces(
                    inters_tuple,
                    atoms_i, atoms_j_shuffle,
                    Val(force_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{D, T})

                @inbounds for k in a:b
                    force_smem[laneid(), k] += ustrip(f[k])
                    opposites_sum[shuffle_idx, k] -= ustrip(f[k])
                end
            end

            sync_threads()
            @inbounds for k in a:b
                CUDA.atomic_add!(
                    pointer(fs_mat, s_idx_i * b - (b - k)), 
                    -force_smem[laneid(), k]
                ) 
                CUDA.atomic_add!(
                    pointer(fs_mat, s_idx_j * b - (b - k)), 
                    -opposites_sum[laneid(), k]
                ) 
            end
        end
    end

    if j == n_blocks && i < n_blocks
        r_max_i = SVector(maxs[i, 1], maxs[i, 2], maxs[i, 3])
        r_min_i = SVector(mins[i, 1], mins[i, 2], mins[i, 3])
        r_max_j = SVector(maxs[j, 1], maxs[j, 2], maxs[j, 3])
        r_min_j = SVector(mins[j, 1], mins[j, 2], mins[j, 3])
        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)
        dist_block = sum(abs2, d_block)

        if dist_block <= r_cut * r_cut 
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i]
            vel_i = velocities[s_idx_i]
            atoms_i = atoms[s_idx_i]
            d_pb = boxes_dist(coords_i, coords_i, r_min_j, r_max_j, boundary)
            dist_pb = sum(abs2, d_pb)

            Bool_excl = dist_pb <= r_cut * r_cut
            eligible_bitmask = UInt32(0)
            special_bitmask = UInt32(0)
            eligible_bitmask = eligible_compressed[laneid(), i, j]
            special_bitmask = special_compressed[laneid(), i, j]
            
            for m in a:r
                s_idx_j = sorted_seq[j_0_tile + m]
                coords_j = coords[s_idx_j]
                vel_j = velocities[s_idx_j]
                atoms_j = atoms[s_idx_j]
                dr = vector(coords_j, coords_i, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                f = condition ? sum_pairwise_forces(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(force_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{D, T})

                @inbounds for k in a:b
                    force_smem[laneid(), k] += ustrip(f[k])
                    CUDA.atomic_add!(
                        pointer(fs_mat, s_idx_j * b - (b - k)), 
                        ustrip(f[k])
                    )
                end
            end

            # Sum contributions of the r-block to the other standard blocks
            @inbounds for k in a:b
                CUDA.atomic_add!(
                    pointer(fs_mat, s_idx_i * b - (b - k)), 
                    -force_smem[laneid(), k]
                ) 
            end
        end
    end

    if i == j && i < n_blocks
        s_idx_i = sorted_seq[index_i]
        coords_i = coords[s_idx_i]
        vel_i = velocities[s_idx_i]
        atoms_i = atoms[s_idx_i]
        eligible_bitmask = UInt32(0)
        special_bitmask = UInt32(0)
        eligible_bitmask = eligible_compressed[laneid(), i, j]
        special_bitmask = special_compressed[laneid(), i, j]

        for m in (laneid() + a) : warpsize()
            s_idx_j = sorted_seq[j_0_tile + m]
            coords_j = coords[s_idx_j]
            vel_j = velocities[s_idx_j]
            atoms_j = atoms[s_idx_j]
            dr = vector(coords_j, coords_i, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

            f = condition ? sum_pairwise_forces(
                inters_tuple,
                atoms_i, atoms_j,
                Val(force_units),
                (spec & 0x1) == true,
                coords_i, coords_j,
                boundary,
                vel_i, vel_j,
                step_n) : zero(SVector{D, T})
            
            @inbounds for k in a:b
                force_smem[laneid(), k] += ustrip(f[k])
                opposites_sum[m, k] -= ustrip(f[k])
            end
        end	

        sync_threads()
        @inbounds for k in a:b
            # In this case i == j, so we can call atomic_add! only once
            CUDA.atomic_add!(
                pointer(fs_mat, s_idx_i * b - (b - k)), 
                -force_smem[laneid(), k] - opposites_sum[laneid(), k]
            ) 
        end
    end

    if i == n_blocks && j == n_blocks
        if laneid() <= r
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i]
            vel_i = velocities[s_idx_i]
            atoms_i = atoms[s_idx_i]
            eligible_bitmask = UInt32(0)
            special_bitmask = UInt32(0)
            eligible_bitmask = eligible_compressed[laneid(), i, j]
            special_bitmask = special_compressed[laneid(), i, j]

            for m in (laneid() + a) : r
                s_idx_j = sorted_seq[j_0_tile + m]
                coords_j = coords[s_idx_j]
                vel_j = velocities[s_idx_j]
                atoms_j = atoms[s_idx_j]
                dr = vector(coords_j, coords_i, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
                
                f = condition ? sum_pairwise_forces(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(force_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{D, T})

                @inbounds for k in a:b
                    force_smem[laneid(), k] += ustrip(f[k])
                    opposites_sum[m, k] -= ustrip(f[k])
                end
            end

            sync_threads()
            @inbounds for k in a:b
                CUDA.atomic_add!(
                    pointer(fs_mat, s_idx_i * b - (b - k)), 
                    -force_smem[laneid(), k] - opposites_sum[laneid(), k]
                ) 
            end
        end
    end

    return nothing
end

function energy_kernel!( 
    sorted_seq,
    energy_nounits, 
    mins::AbstractArray{C}, 
    maxs::AbstractArray{C}, 
    coords, 
    velocities,
    atoms,
    ::Val{N}, 
    r_cut, 
    ::Val{energy_units},
    inters_tuple,
    boundary,
    step_n, 
    special_matrix,
    eligible_matrix,
    ::Val{T},
    ::Val{D}) where {N, C, energy_units, T, D}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)
    r = Int32((N - 1) % 32 + 1)
    i = blockIdx().x
    j = blockIdx().y
    i_0_tile = (i - 1) * warpsize()
    j_0_tile = (j - 1) * warpsize()
    index_i = i_0_tile + laneid()
    index_j = j_0_tile + laneid()
    E_smem = CuStaticSharedArray(T, 32)
    E_smem[laneid()] = zero(T)
    eligible = CuStaticSharedArray(Bool, (32, 32))
    special = CuStaticSharedArray(Bool, (32, 32))

    # The code is organised in 4 mutually excluding parts
    if j < n_blocks && i < j
        r_max_i = SVector(maxs[i, 1], maxs[i, 2], maxs[i, 3])
        r_min_i = SVector(mins[i, 1], mins[i, 2], mins[i, 3])
        r_max_j = SVector(maxs[j, 1], maxs[j, 2], maxs[j, 3])
        r_min_j = SVector(mins[j, 1], mins[j, 2], mins[j, 3])
        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)
        dist_block = sum(abs2, d_block)

        if dist_block <= r_cut * r_cut
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i] 
            vel_i = velocities[s_idx_i]
            atoms_i = atoms[s_idx_i]
            d_pb = boxes_dist(coords_i, coords_i, r_min_j, r_max_j, boundary)
            dist_pb = sum(abs2, d_pb)

            Bool_excl = dist_pb <= r_cut * r_cut
            s_idx_j = sorted_seq[index_j]
            coords_j = coords[s_idx_j]
            vel_j = velocities[s_idx_j]
            shuffle_idx = laneid()
            atoms_j = atoms[s_idx_j]
            atype_j = atoms_j.atom_type
            aindex_j = atoms_j.index
            amass_j = atoms_j.mass
            acharge_j = atoms_j.charge
            aσ_j = atoms_j.σ
            aϵ_j = atoms_j.ϵ
            @inbounds for m in a:warpsize()
                eligible[laneid(), m] = eligible_matrix[s_idx_i, sorted_seq[j_0_tile + m]]
                special[laneid(), m] = special_matrix[s_idx_i, sorted_seq[j_0_tile + m]]
            end

            # Shuffle
            for m in a:warpsize()
                sync_warp()
                coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, laneid() + a, warpsize())
                vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, laneid() + a, warpsize())
                s_idx_j = CUDA.shfl_sync(0xFFFFFFFF, s_idx_j, laneid() + a, warpsize())
                shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, laneid() + a, warpsize())
                atype_j = CUDA.shfl_sync(0xFFFFFFFF, atype_j, laneid() + a, warpsize())
                aindex_j = CUDA.shfl_sync(0xFFFFFFFF, aindex_j, laneid() + a, warpsize())
                amass_j = CUDA.shfl_sync(0xFFFFFFFF, amass_j, laneid() + a, warpsize())
                acharge_j = CUDA.shfl_sync(0xFFFFFFFF, acharge_j, laneid() + a, warpsize())
                aσ_j = CUDA.shfl_sync(0xFFFFFFFF, aσ_j, laneid() + a, warpsize())
                aϵ_j = CUDA.shfl_sync(0xFFFFFFFF, aϵ_j, laneid() + a, warpsize())
                
                atoms_j_shuffle = Atom(atype_j, aindex_j, amass_j, acharge_j, aσ_j, aϵ_j)
                dr = vector(coords_j, coords_i, boundary)
                r2 = sum(abs2, dr)
                condition = eligible[laneid(), shuffle_idx] && Bool_excl && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j_shuffle,
                    Val(energy_units),
                    special[laneid(), shuffle_idx],
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})

                E_smem[laneid()] += ustrip(pe[1])
            end
        end
    end

    if j == n_blocks && i < n_blocks
        r_max_i = SVector(maxs[i, 1], maxs[i, 2], maxs[i, 3])
        r_min_i = SVector(mins[i, 1], mins[i, 2], mins[i, 3])
        r_max_j = SVector(maxs[j, 1], maxs[j, 2], maxs[j, 3])
        r_min_j = SVector(mins[j, 1], mins[j, 2], mins[j, 3])
        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)
        dist_block = sum(abs2, d_block)

        if dist_block <= r_cut * r_cut 
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i]
            vel_i = velocities[s_idx_i]
            atoms_i = atoms[s_idx_i]
            d_pb = boxes_dist(coords_i, coords_i, r_min_j, r_max_j, boundary)
            dist_pb = sum(abs2, d_pb)

            Bool_excl = dist_pb <= r_cut * r_cut
            @inbounds for m in a:r
                s_idx_j = sorted_seq[j_0_tile + m]
                eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
                special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
            end
            
            for m in a:r
                s_idx_j = sorted_seq[j_0_tile + m]
                coords_j = coords[s_idx_j]
                vel_j = velocities[s_idx_j]
                atoms_j = atoms[s_idx_j]
                dr = vector(coords_j, coords_i, boundary)
                r2 = sum(abs2, dr)
                condition = eligible[laneid(), m] && Bool_excl && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    special[laneid(), m],
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})

                E_smem[laneid()] += ustrip(pe[1])
            end
        end
    end

    if i == j && i < n_blocks
        s_idx_i = sorted_seq[index_i]
        coords_i = coords[s_idx_i]
        vel_i = velocities[s_idx_i]
        atoms_i = atoms[s_idx_i]
        @inbounds for m in a:warpsize()
            s_idx_j = sorted_seq[j_0_tile + m]
            eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
            special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
        end
        @inbounds for m in (laneid() + a) : warpsize()
            s_idx_j = sorted_seq[j_0_tile + m]
            coords_j = coords[s_idx_j]
            vel_j = velocities[s_idx_j]
            atoms_j = atoms[s_idx_j]
            dr = vector(coords_j, coords_i, boundary)
            r2 = sum(abs2, dr)
            condition = eligible[laneid(), m] && r2 <= r_cut * r_cut

            pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    special[laneid(), m],
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})

            E_smem[laneid()] += ustrip(pe[1])
        end	
    end

    if i == n_blocks && j == n_blocks
        if laneid() <= r
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i]
            vel_i = velocities[s_idx_i]
            atoms_i = atoms[s_idx_i]
            @inbounds for m in a:r
                s_idx_j = sorted_seq[j_0_tile + m]
                eligible[laneid(), m] = eligible_matrix[s_idx_i, s_idx_j]
                special[laneid(), m] = special_matrix[s_idx_i, s_idx_j]
            end

            @inbounds for m in (laneid() + a) : r
                s_idx_j = sorted_seq[j_0_tile + m]
                coords_j = coords[s_idx_j]
                vel_j = velocities[s_idx_j]
                atoms_j = atoms[s_idx_j]
                dr = vector(coords_j, coords_i, boundary)
                r2 = sum(abs2, dr)
                condition = eligible[laneid(), m] && r2 <= r_cut * r_cut
                
                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    special[laneid(), m],
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})

                E_smem[laneid()] += ustrip(pe[1])
            end
        end
    end

    if threadIdx().x == a
        sum_E = zero(T)
        for k in a:warpsize()
            sum_E += E_smem[k]
        end
        CUDA.atomic_add!(pointer(energy_nounits), sum_E)
    end
    return nothing
end

function pairwise_force_kernel_nonl!(forces::AbstractArray{T}, coords_var, velocities_var,
                        atoms_var, boundary, inters, step_n, ::Val{D}, ::Val{F}) where {T, D, F}
    coords = CUDA.Const(coords_var)
    velocities = CUDA.Const(velocities_var)
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
            atom_i, coord_i, vel_i = atoms[i], coords[i], velocities[i]
            for del_j in 1:njs
                j = j_0_tile + del_j
                if i != j
                    atom_j, coord_j, vel_j = atoms[j], coords[j], velocities[j]
                    f = sum_pairwise_forces(inters, atom_i, atom_j, Val(F), false, coord_i,
                                            coord_j, boundary, vel_i, vel_j, step_n)
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

        atom_i, coord_i, vel_i = atoms[i], coords[i], velocities[i]
        coord_j, vel_j = coords[j], velocities[j]
        @inbounds for _ in 1:tilesteps
            sync_warp()
            atom_j = atoms[j]
            f = sum_pairwise_forces(inters, atom_i, atom_j, Val(F), false, coord_i, coord_j,
                                    boundary, vel_i, vel_j, step_n)
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

end
