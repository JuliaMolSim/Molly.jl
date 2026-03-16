# CUDA kernels that use warp-level features
# This file is only loaded when CUDA is imported

module MollyCUDAExt

using Molly
using Molly: from_device, box_sides, sorted_morton_seq!, sum_pairwise_forces,
             sum_pairwise_potentials, volume
using CUDA
using Atomix
using KernelAbstractions

const WARPSIZE = UInt32(32)
const MAX_BLOCK_Y = 32

Molly.uses_gpu_neighbor_finder(::Type{<:CuArray}) = true

CUDA.Const(nl::Molly.NoNeighborList) = nl

function env_int(name::AbstractString)
    value = ENV[name]
    parsed = tryparse(Int, value)
    parsed === nothing && error("Invalid integer value for $(name): $(repr(value))")
    return parsed
end

function env_override(name::AbstractString)
    return haskey(ENV, name) ? env_int(name) : nothing
end

prefer_override(primary, secondary) = primary === nothing ? secondary : primary

function validate_block_y(name::AbstractString, block_y::Int)
    1 <= block_y <= MAX_BLOCK_Y || error("$(name) must be in 1:$(MAX_BLOCK_Y), got $(block_y)")
    return block_y
end

function choose_block_y(conf_threads::Int)
    return max(1, min(MAX_BLOCK_Y, fld(conf_threads, Int(WARPSIZE))))
end

function choose_tile_threads(conf_threads::Int)
    threads_x = min(Int(WARPSIZE), conf_threads)
    threads_y = max(1, min(MAX_BLOCK_Y, fld(conf_threads, threads_x)))
    return (threads_x, threads_y)
end

function force_path_mode()
    mode = get(ENV, "MOLLY_CUDA_FORCE_PATH", "auto")
    mode in ("auto", "tile", "dense") || error("MOLLY_CUDA_FORCE_PATH must be one of auto, tile, dense; got $(repr(mode))")
    return Symbol(mode)
end

function dense_force_threshold()
    return parse(Float64, get(ENV, "MOLLY_CUDA_DENSE_FORCE_THRESHOLD", "4.0"))
end

function dense_force_metric(sys::System{D}) where {D}
    return Float64(ustrip(length(sys) * sys.neighbor_finder.dist_cutoff^D / volume(sys.boundary)))
end

function selected_force_path(sys::System)
    mode = force_path_mode()
    mode === :auto || return mode
    return dense_force_metric(sys) >= dense_force_threshold() ? :dense : :tile
end

function selected_energy_path(sys::System)
    mode = force_path_mode()
    mode === :auto && return :tile
    return mode
end

function force_launch_params(kernel)
    config = Molly.cuda_launch_config()
    block_y_override = prefer_override(config.force_block_y, env_override("MOLLY_CUDA_FORCE_BLOCK_Y"))
    maxregs_override = prefer_override(config.force_maxregs, env_override("MOLLY_CUDA_FORCE_MAXREGS"))
    block_y_override === nothing || validate_block_y("MOLLY_CUDA_FORCE_BLOCK_Y", block_y_override)
    maxregs_override === nothing || maxregs_override > 0 || error("MOLLY_CUDA_FORCE_MAXREGS must be positive, got $(maxregs_override)")

    conf = launch_configuration(kernel.fun)
    block_y = something(block_y_override, choose_block_y(conf.threads))
    return (block_y, maxregs_override)
end

function energy_launch_params(kernel)
    config = Molly.cuda_launch_config()
    block_y_override = prefer_override(config.energy_block_y, env_override("MOLLY_CUDA_ENERGY_BLOCK_Y"))
    block_y_override === nothing || validate_block_y("MOLLY_CUDA_ENERGY_BLOCK_Y", block_y_override)

    conf = launch_configuration(kernel.fun)
    block_y = something(block_y_override, choose_block_y(conf.threads))
    return block_y
end

function tile_launch_params(kernel)
    config = Molly.cuda_launch_config()
    config_tile_threads = config.tile_threads
    threads_x_override = config_tile_threads === nothing ? env_override("MOLLY_CUDA_TILE_THREADS_X") : config_tile_threads[1]
    threads_y_override = config_tile_threads === nothing ? env_override("MOLLY_CUDA_TILE_THREADS_Y") : config_tile_threads[2]
    if xor(threads_x_override === nothing, threads_y_override === nothing)
        error("Set both MOLLY_CUDA_TILE_THREADS_X and MOLLY_CUDA_TILE_THREADS_Y together")
    end

    if threads_x_override !== nothing
        threads_x_override > 0 || error("MOLLY_CUDA_TILE_THREADS_X must be positive, got $(threads_x_override)")
        threads_y_override > 0 || error("MOLLY_CUDA_TILE_THREADS_Y must be positive, got $(threads_y_override)")
        threads_x_override * threads_y_override <= 1024 || error("MOLLY_CUDA_TILE_THREADS_X * MOLLY_CUDA_TILE_THREADS_Y must be <= 1024")
        return (threads_x_override, threads_y_override)
    end

    conf = launch_configuration(kernel.fun)
    return choose_tile_threads(conf.threads)
end

function dense_force_launch_params()
    config = Molly.cuda_launch_config()
    block_y_override = prefer_override(config.force_block_y, env_override("MOLLY_CUDA_FORCE_BLOCK_Y"))
    maxregs_override = prefer_override(config.force_maxregs, env_override("MOLLY_CUDA_FORCE_MAXREGS"))
    block_y = something(block_y_override, 8)
    maxregs = something(maxregs_override, 64)
    validate_block_y("MOLLY_CUDA_FORCE_BLOCK_Y", block_y)
    maxregs > 0 || error("MOLLY_CUDA_FORCE_MAXREGS must be positive, got $(maxregs)")
    return (block_y, maxregs)
end

function reset_interacting_tile_state!(buffers)
    fill!(buffers.num_interacting_tiles, 0)
    fill!(buffers.interacting_tiles_overflow, 0)
    return nothing
end

function throw_if_interacting_tiles_overflowed(buffers)
    overflow_count = only(from_device(buffers.interacting_tiles_overflow))
    overflow_count == 0 && return nothing

    max_tiles = length(buffers.interacting_tiles_i)
    num_tiles = only(from_device(buffers.num_interacting_tiles))
    error("Maximum number of interacting tiles exceeded ($(num_tiles) > $(max_tiles)); increase buffer size.")
end

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

function Molly.pairwise_forces_loop_gpu!(buffers, sys::System{D, <:CuArray}, pairwise_inters,
                            nbs::Molly.NoNeighborList, step_n) where D
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

function Molly.pairwise_forces_loop_gpu!(buffers, sys::System{D, <:CuArray, T}, pairwise_inters,
                        nbs::Nothing, ::Val{needs_vir}, step_n) where {D, T, needs_vir}
    if selected_force_path(sys) == :dense
        return pairwise_forces_loop_gpu_dense!(buffers, sys, pairwise_inters, Val(needs_vir), step_n)
    end

    N = length(sys.coords)
    n_blocks = cld(N, WARPSIZE)
    r_cut = sys.neighbor_finder.dist_cutoff
    backend = get_backend(sys.coords)

    if step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized
        morton_bits = 10
        sides = box_sides(sys.boundary)
        w = sides ./ (2^morton_bits)
        sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
        sys.neighbor_finder.initialized = true
        @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true compress_boolean_matrices!(
                buffers.morton_seq, sys.neighbor_finder.eligible, sys.neighbor_finder.special,
                buffers.compressed_eligible, buffers.compressed_special, Val(N))
    end

    reorder_system_gpu!(buffers, sys)
    KernelAbstractions.synchronize(backend)

    if sys.boundary isa TriclinicBoundary
        H = SMatrix{3, 3, T}(
            sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
            sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
            sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val
        )
        Hinv = inv(H)
        @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
                buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Hinv, Val(N),
                sys.boundary, Val(D))
    else
        @cuda blocks=n_blocks threads=32 kernel_min_max!(
                    buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Val(N),
                    sys.boundary, Val(D))
    end

    max_tiles = length(buffers.interacting_tiles_i)
    reset_interacting_tile_state!(buffers)
    n_blocks_i = cld(N, WARPSIZE)
    tile_kernel = @cuda launch=false find_interacting_blocks_kernel!(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, sys.boundary, r_cut,
        Val(n_blocks_i), Val(D), max_tiles,
        buffers.compressed_eligible, buffers.compressed_special)
    tile_threads_xy = tile_launch_params(tile_kernel)
    
    tile_kernel(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, sys.boundary, r_cut,
        Val(n_blocks_i), Val(D), max_tiles,
        buffers.compressed_eligible, buffers.compressed_special;
        blocks=(cld(n_blocks_i, tile_threads_xy[1]), cld(n_blocks_i, tile_threads_xy[2])),
        threads=tile_threads_xy)

    auto_kernel = @cuda launch=false always_inline=true force_kernel!(
        buffers.morton_seq,
        buffers.fs_mat_reordered,
        buffers.virial_nounits,
        buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
        Val(N), r_cut, Val(sys.force_units), pairwise_inters,
        sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
        Val(needs_vir), Val(T), Val(D),
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow)
    block_y, maxregs = force_launch_params(auto_kernel)
    n_blocks_launch = cld(max_tiles, block_y)

    kernel = if maxregs === nothing
        auto_kernel
    else
        @cuda launch=false maxregs=maxregs always_inline=true force_kernel!(
            buffers.morton_seq,
            buffers.fs_mat_reordered,
            buffers.virial_nounits,
            buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
            Val(N), r_cut, Val(sys.force_units), pairwise_inters,
            sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
            Val(needs_vir), Val(T), Val(D),
            buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
            buffers.num_interacting_tiles, buffers.interacting_tiles_overflow)
    end

    kernel(
        buffers.morton_seq,
        buffers.fs_mat_reordered,
        buffers.virial_nounits,
        buffers.coords_reordered, buffers.velocities_reordered, buffers.atoms_reordered,
        Val(N), r_cut, Val(sys.force_units), pairwise_inters,
        sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
        Val(needs_vir), Val(T), Val(D),
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow;
        threads=(32, block_y), blocks=n_blocks_launch
    )

    reverse_reorder_forces_gpu!(buffers, sys)
    KernelAbstractions.synchronize(backend)
    throw_if_interacting_tiles_overflowed(buffers)

    return buffers
end

function pairwise_forces_loop_gpu_dense!(buffers, sys::System{D, <:CuArray, T}, pairwise_inters,
                                         ::Val{needs_vir}, step_n) where {D, T, needs_vir}
    N = length(sys.coords)
    n_blocks = cld(N, WARPSIZE)
    r_cut = sys.neighbor_finder.dist_cutoff

    if step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized
        morton_bits = 10
        sides = box_sides(sys.boundary)
        w = sides ./ (2^morton_bits)
        sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
        sys.neighbor_finder.initialized = true
        @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true compress_boolean_matrices!(
            buffers.morton_seq, sys.neighbor_finder.eligible, sys.neighbor_finder.special,
            buffers.compressed_eligible, buffers.compressed_special, Val(N))
    end

    if sys.boundary isa TriclinicBoundary
        H = SMatrix{3, 3, T}(
            sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
            sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
            sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val
        )
        Hinv = inv(H)
        @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
            buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Hinv, Val(N),
            sys.boundary, Val(D))
    else
        @cuda blocks=n_blocks threads=32 kernel_min_max!(
            buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Val(N),
            sys.boundary, Val(D))
    end

    block_y, maxregs = dense_force_launch_params()
    n_blocks_i = cld(N, WARPSIZE)
    n_blocks_j = cld(n_blocks_i, block_y)

    kernel = @cuda launch=false maxregs=maxregs always_inline=true dense_force_kernel!(
        buffers.morton_seq,
        buffers.fs_mat,
        buffers.virial_nounits,
        buffers.box_mins, buffers.box_maxs,
        sys.coords, sys.velocities, sys.atoms,
        Val(N), r_cut, Val(sys.force_units), pairwise_inters,
        sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
        Val(needs_vir), Val(T), Val(D), Val(block_y))

    kernel(
        buffers.morton_seq,
        buffers.fs_mat,
        buffers.virial_nounits,
        buffers.box_mins, buffers.box_maxs,
        sys.coords, sys.velocities, sys.atoms,
        Val(N), r_cut, Val(sys.force_units), pairwise_inters,
        sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
        Val(needs_vir), Val(T), Val(D), Val(block_y);
        threads=(32, block_y), blocks=(n_blocks_i, n_blocks_j)
    )

    return buffers
end

function Molly.pairwise_pe_loop_gpu!(pe_vec_nounits, buffers, sys::System{D, <:CuArray, T},
                                     pairwise_inters, nbs::Nothing,
                                     step_n) where {D, T}
    if selected_energy_path(sys) == :dense
        return pairwise_pe_loop_gpu_dense!(pe_vec_nounits, buffers, sys, pairwise_inters, step_n)
    end

    # The ordering is always recomputed for potential energy
    # Different buffers are used to the forces case, so sys.neighbor_finder.initialized
    #   is not updated
    N = length(sys.coords)
    n_blocks = cld(N, WARPSIZE)
    r_cut = sys.neighbor_finder.dist_cutoff
    backend = get_backend(sys.coords)
    morton_bits = 10
    sides = box_sides(sys.boundary)
    w = sides ./ (2^morton_bits)
    sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
    reorder_system_gpu!(buffers, sys)
    KernelAbstractions.synchronize(backend)
    @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true compress_boolean_matrices!(
                buffers.morton_seq, sys.neighbor_finder.eligible, sys.neighbor_finder.special,
                buffers.compressed_eligible, buffers.compressed_special, Val(N))
    if sys.boundary isa TriclinicBoundary
        H = SMatrix{3, 3, T}(
            sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
            sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
            sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val
        )
        Hinv = inv(H)
        @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
                buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Hinv, Val(N),
                sys.boundary, Val(D))
    else
        @cuda blocks=cld(N, WARPSIZE) threads=32 kernel_min_max!(
                buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords,
                Val(N), sys.boundary, Val(D))
    end
    max_tiles = length(buffers.interacting_tiles_i)
    reset_interacting_tile_state!(buffers)
    tile_kernel = @cuda launch=false find_interacting_blocks_kernel!(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, sys.boundary, r_cut,
        Val(n_blocks), Val(D), max_tiles,
        buffers.compressed_eligible, buffers.compressed_special)
    tile_threads_xy = tile_launch_params(tile_kernel)
    tile_kernel(
        buffers.interacting_tiles_i, buffers.interacting_tiles_j, buffers.interacting_tiles_type,
        buffers.num_interacting_tiles, buffers.interacting_tiles_overflow,
        buffers.box_mins, buffers.box_maxs, sys.boundary, r_cut,
        Val(n_blocks), Val(D), max_tiles,
        buffers.compressed_eligible, buffers.compressed_special;
        blocks=(cld(n_blocks, tile_threads_xy[1]), cld(n_blocks, tile_threads_xy[2])),
        threads=tile_threads_xy)

    kernel = @cuda launch=false always_inline=true energy_kernel!(
            buffers.morton_seq, pe_vec_nounits, buffers.coords_reordered,
            buffers.velocities_reordered, buffers.atoms_reordered, Val(N), r_cut, Val(sys.energy_units), pairwise_inters,
            sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
            Val(T), Val(D), buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.num_interacting_tiles, buffers.interacting_tiles_overflow)
    block_y = energy_launch_params(kernel)
    kernel(
            buffers.morton_seq, pe_vec_nounits, buffers.coords_reordered,
            buffers.velocities_reordered, buffers.atoms_reordered, Val(N), r_cut, Val(sys.energy_units), pairwise_inters,
            sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
            Val(T), Val(D), buffers.interacting_tiles_i, buffers.interacting_tiles_j,
            buffers.interacting_tiles_type, buffers.num_interacting_tiles, buffers.interacting_tiles_overflow;
            blocks=cld(max_tiles, block_y), threads=(32, block_y))
    throw_if_interacting_tiles_overflowed(buffers)
    return pe_vec_nounits
end

function pairwise_pe_loop_gpu_dense!(pe_vec_nounits, buffers, sys::System{D, <:CuArray, T},
                                     pairwise_inters, step_n) where {D, T}
    N = length(sys.coords)
    n_blocks = cld(N, WARPSIZE)
    r_cut = sys.neighbor_finder.dist_cutoff
    morton_bits = 10
    sides = box_sides(sys.boundary)
    w = sides ./ (2^morton_bits)
    sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
    @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true compress_boolean_matrices!(
        buffers.morton_seq, sys.neighbor_finder.eligible, sys.neighbor_finder.special,
        buffers.compressed_eligible, buffers.compressed_special, Val(N))
    if sys.boundary isa TriclinicBoundary
        H = SMatrix{3, 3, T}(
            sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
            sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
            sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val
        )
        Hinv = inv(H)
        @cuda blocks=n_blocks threads=32 kernel_min_max_triclinic!(
            buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords, Hinv, Val(N),
            sys.boundary, Val(D))
    else
        @cuda blocks=cld(N, WARPSIZE) threads=32 kernel_min_max!(
            buffers.morton_seq, buffers.box_mins, buffers.box_maxs, sys.coords,
            Val(N), sys.boundary, Val(D))
    end
    @cuda blocks=(n_blocks, n_blocks) threads=(32, 1) always_inline=true dense_energy_kernel!(
        buffers.morton_seq, pe_vec_nounits, buffers.box_mins, buffers.box_maxs, sys.coords,
        sys.velocities, sys.atoms, Val(N), r_cut, Val(sys.energy_units), pairwise_inters,
        sys.boundary, step_n, buffers.compressed_special, buffers.compressed_eligible,
        Val(T), Val(D))
    return pe_vec_nounits
end

function boxes_dist(r1_min::SVector{D, T}, r1_max::SVector{D, T}, r2_min::SVector{D, T}, r2_max::SVector{D, T}, boundary) where {D, T}
    va = vector(r2_max, r1_min, boundary)
    vb = vector(r1_max, r2_min, boundary)
    a = SVector{D}(ntuple(d -> abs(va[d]), D))
    b = SVector{D}(ntuple(d -> abs(vb[d]), D))

    return SVector(ntuple(d -> r1_min[d] - r2_max[d] <= zero(T) && r2_min[d] - r1_max[d] <= zero(T) ? zero(T) : ifelse(a[d] < b[d], a[d], b[d]), D))
end

# Triclinic boxes case is treated only in 3 dimensions
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

function reorder_system_gpu!(buffers, sys)
    N = length(sys)
    backend = get_backend(sys.coords)
    n_threads = 256
    
    # We use KernelAbstractions kernels from Molly
    Molly.reorder_kernel!(backend, n_threads)(buffers.coords_reordered, sys.coords, buffers.morton_seq, ndrange=N)
    Molly.reorder_kernel!(backend, n_threads)(buffers.velocities_reordered, sys.velocities, buffers.morton_seq, ndrange=N)
    Molly.reorder_kernel!(backend, n_threads)(buffers.atoms_reordered, sys.atoms, buffers.morton_seq, ndrange=N)
    
    return nothing
end

function reverse_reorder_forces_gpu!(buffers, sys::System{D, <:CuArray, T}) where {D, T}
    N = length(sys)
    backend = get_backend(sys.coords)
    n_threads = 256
    
    # fs_mat is D x N. We need to reverse reorder each dimension or use a specialized kernel.
    # Let's use a specialized kernel for D x N matrix reverse reordering.
    Molly.reverse_reorder_forces_kernel!(backend, n_threads)(buffers.fs_mat, buffers.fs_mat_reordered, buffers.morton_seq, ndrange=N)
    
    return nothing
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
            val = zero(C)
            for j in a:b
                val += Hinv[k,j]*r_i[j]
            end
            mins_smem[local_i, k] = val
            maxs_smem[local_i, k] = val
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
            val = zero(C)
            for j in a:b
                val += Hinv[k,j]*r_i[j]
            end
            r_smem[local_i, k] = val # Transform to fractional space: s = Hinv * r
        end
    end
    xyz_min = CuStaticSharedArray(C, b)
    xyz_max = CuStaticSharedArray(C, b)
    for k in a:b
        xyz_min[k] =  10 * box_sides(boundary, k) # Very large (arbitrary) value
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


function find_interacting_blocks_kernel!(
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
    interacting_tiles_overflow,
    mins::AbstractArray{C}, maxs::AbstractArray{C}, boundary, r_cut, ::Val{N_blocks}, ::Val{D}, max_tiles,
    compressed_eligible, compressed_special
) where {C, N_blocks, D}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= N_blocks && j <= N_blocks && i <= j
        r_min_i = SVector{D}(ntuple(d -> mins[i, d], D))
        r_max_i = SVector{D}(ntuple(d -> maxs[i, d], D))
        r_min_j = SVector{D}(ntuple(d -> mins[j, d], D))
        r_max_j = SVector{D}(ntuple(d -> maxs[j, d], D))

        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)
        
        if sum(abs2, d_block) <= r_cut * r_cut
            is_clean = (i < j) && (j < N_blocks)
            if is_clean
                for k in 1:32
                    if compressed_eligible[k, i, j] != 0xFFFFFFFF || compressed_special[k, i, j] != 0
                        is_clean = false
                        break
                    end
                end
            end

            idx = CUDA.atomic_add!(pointer(num_interacting_tiles, 1), Int32(1)) + Int32(1)
            if idx <= max_tiles
                interacting_tiles_i[idx] = Int32(i)
                interacting_tiles_j[idx] = Int32(j)
                interacting_tiles_type[idx] = is_clean ? UInt8(0) : UInt8(1)
            else
                CUDA.atomic_add!(pointer(interacting_tiles_overflow, 1), Int32(1))
            end
        end
    end
    return nothing
end

function force_kernel!(
    sorted_seq,
    fs_mat,
    global_virial,
    coords,
    velocities,
    atoms::AbstractArray{A},
    ::Val{N},
    r_cut,
    ::Val{force_units},
    inters_tuple,
    boundary,
    step_n,
    special_compressed,
    eligible_compressed,
    ::Val{needs_vir},
    ::Val{T},
    ::Val{D},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
    interacting_tiles_overflow) where {N, A, force_units, needs_vir, T, D}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)
    active_warps = Int32(blockDim().y)

    if interacting_tiles_overflow[1] != 0
        return nothing
    end
    
    tile_idx = (blockIdx().x - a) * active_warps + threadIdx().y
    if tile_idx > num_interacting_tiles[1]
        return nothing
    end

    lane = threadIdx().x
    warpid = threadIdx().y
    i = interacting_tiles_i[tile_idx]
    j = interacting_tiles_j[tile_idx]
    type = interacting_tiles_type[tile_idx]

    i_0_tile = (i - a) * warpsize()
    j_0_tile = (j - a) * warpsize()
    index_i = i_0_tile + lane
    index_j = j_0_tile + lane

    force_smem = CuStaticSharedArray(T, (32, D, MAX_BLOCK_Y))
    opposites_sum = CuStaticSharedArray(T, (32, D, MAX_BLOCK_Y))
    
    r = Int32((N - 1) % 32 + 1)
    
    @inbounds for k in a:b
        force_smem[lane, k, warpid] = zero(T)
        opposites_sum[lane, k, warpid] = zero(T)
    end

    vir_xx = zero(T); vir_yy = zero(T); vir_zz = zero(T)
    vir_xy = zero(T); vir_xz = zero(T); vir_yz = zero(T)

    # Part 1: Standard non-diagonal tiles
    if j < n_blocks && i < j
        if true
            coords_i = coords[index_i]
            vel_i = velocities[index_i]
            atoms_i = atoms[index_i]
            
            coords_j = coords[index_j]
            vel_j = velocities[index_j]
            shuffle_idx = lane
            atoms_j = atoms[index_j]
            atom_fields = getfield.((atoms_j,), fieldnames(A))
            
            if type == UInt8(0) # CLEAN
                for m in a:warpsize()
                    sync_warp()
                    coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, lane + a, warpsize())
                    vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, lane + a, warpsize())
                    shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, lane + a, warpsize())
                    atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, lane + a, warpsize())
                    atoms_j_shuffle = A(atom_fields...)

                    dr = vector(coords_i, coords_j, boundary)
                    r2 = sum(abs2, dr)
                    condition = r2 <= r_cut * r_cut
                    any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)
                    
                    if any_active
                        f = condition ? sum_pairwise_forces(
                            inters_tuple, atoms_i, atoms_j_shuffle, Val(force_units),
                            false, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                        ) : zero(SVector{D, T})

                        @inbounds for k in a:b
                            force_smem[lane, k, warpid]           += ustrip(f[k])
                            opposites_sum[shuffle_idx, k, warpid] -= ustrip(f[k])
                        end

                        if needs_vir
                            vir_xx += ustrip(f[1]) * ustrip(dr[1])
                            if D >= 2
                                vir_yy += ustrip(f[2]) * ustrip(dr[2])
                                vir_xy += ustrip(f[1]) * ustrip(dr[2])
                            end
                            if D >= 3
                                vir_zz += ustrip(f[3]) * ustrip(dr[3])
                                vir_xz += ustrip(f[1]) * ustrip(dr[3])
                                vir_yz += ustrip(f[2]) * ustrip(dr[3])
                            end
                        end
                    end
                end
            else # EXCLUDED
                eligible_bitmask = eligible_compressed[lane, i, j]
                special_bitmask = special_compressed[lane, i, j]

                for m in a:warpsize()
                    sync_warp()
                    coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, lane + a, warpsize())
                    vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, lane + a, warpsize())
                    shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, lane + a, warpsize())
                    atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, lane + a, warpsize())
                    atoms_j_shuffle = A(atom_fields...)

                    dr = vector(coords_i, coords_j, boundary)
                    r2 = sum(abs2, dr)
                    excl = (eligible_bitmask >> (warpsize() - shuffle_idx)) | (eligible_bitmask << shuffle_idx)
                    spec = (special_bitmask >> (warpsize() - shuffle_idx)) | (special_bitmask << shuffle_idx)
                    
                    condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
                    any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)
                    
                    if any_active
                        f = condition ? sum_pairwise_forces(
                            inters_tuple, atoms_i, atoms_j_shuffle, Val(force_units),
                            (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                        ) : zero(SVector{D, T})

                        @inbounds for k in a:b
                            force_smem[lane, k, warpid]           += ustrip(f[k])
                            opposites_sum[shuffle_idx, k, warpid] -= ustrip(f[k])
                        end

                        if needs_vir
                            vir_xx += ustrip(f[1]) * ustrip(dr[1])
                            if D >= 2
                                vir_yy += ustrip(f[2]) * ustrip(dr[2])
                                vir_xy += ustrip(f[1]) * ustrip(dr[2])
                            end
                            if D >= 3
                                vir_zz += ustrip(f[3]) * ustrip(dr[3])
                                vir_xz += ustrip(f[1]) * ustrip(dr[3])
                                vir_yz += ustrip(f[2]) * ustrip(dr[3])
                            end
                        end
                    end
                end
            end

            sync_warp()
            @inbounds for k in a:b
                if opposites_sum[lane, k, warpid] != zero(T)
                    CUDA.atomic_add!(pointer(fs_mat, index_j * b - (b - k)), -opposites_sum[lane, k, warpid])
                end
            end
        end
    end

    # Part 2: Boundary column tiles
    if j == n_blocks && i < n_blocks
        if true
            coords_i = coords[index_i]
            vel_i = velocities[index_i]
            atoms_i = atoms[index_i]
            
            eligible_bitmask = eligible_compressed[lane, i, j]
            special_bitmask = special_compressed[lane, i, j]

            for m in a:r
                idx_j = j_0_tile + m
                coords_j = coords[idx_j]
                vel_j = velocities[idx_j]
                atoms_j = atoms[idx_j]
                
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
                any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)

                if any_active
                    f = condition ? sum_pairwise_forces(
                        inters_tuple, atoms_i, atoms_j, Val(force_units),
                        (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                    ) : zero(SVector{D, T})

                    @inbounds for k in a:b
                        force_smem[lane, k, warpid] += ustrip(f[k])
                        if ustrip(f[k]) != zero(T)
                            CUDA.atomic_add!(pointer(fs_mat, idx_j * b - (b - k)), ustrip(f[k]))
                        end
                    end
                    
                    if needs_vir
                        vir_xx += ustrip(f[1]) * ustrip(dr[1])
                        if D >= 2
                            vir_yy += ustrip(f[2]) * ustrip(dr[2])
                            vir_xy += ustrip(f[1]) * ustrip(dr[2])
                        end
                        if D >= 3
                            vir_zz += ustrip(f[3]) * ustrip(dr[3])
                            vir_xz += ustrip(f[1]) * ustrip(dr[3])
                            vir_yz += ustrip(f[2]) * ustrip(dr[3])
                        end
                    end
                end
            end
        end
    end

    # Part 3: Diagonal tiles
    if i == j && i < n_blocks
        coords_i = coords[index_i]
        vel_i = velocities[index_i]
        atoms_i = atoms[index_i]
        
        eligible_bitmask = eligible_compressed[lane, i, j]
        special_bitmask = special_compressed[lane, i, j]

        for m in (lane + a) : warpsize()
            idx_j = j_0_tile + m
            coords_j = coords[idx_j]
            vel_j = velocities[idx_j]
            atoms_j = atoms[idx_j]
            
            dr = vector(coords_i, coords_j, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
            
            # Divergence-safe execution (no vote_any_sync)
            f = condition ? sum_pairwise_forces(
                inters_tuple, atoms_i, atoms_j, Val(force_units),
                (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
            ) : zero(SVector{D, T})

            @inbounds for k in a:b
                force_smem[lane, k, warpid] += ustrip(f[k])
                opposites_sum[m, k, warpid] -= ustrip(f[k])
            end
            
            if needs_vir
                vir_xx += ustrip(f[1]) * ustrip(dr[1])
                if D >= 2
                    vir_yy += ustrip(f[2]) * ustrip(dr[2])
                    vir_xy += ustrip(f[1]) * ustrip(dr[2])
                end
                if D >= 3
                    vir_zz += ustrip(f[3]) * ustrip(dr[3])
                    vir_xz += ustrip(f[1]) * ustrip(dr[3])
                    vir_yz += ustrip(f[2]) * ustrip(dr[3])
                end
            end
        end

        sync_warp()
        @inbounds for k in a:b
            force_smem[lane, k, warpid] += opposites_sum[lane, k, warpid]
        end
    end

    # Part 4: Terminal corner tile
    if i == n_blocks && j == n_blocks
        if lane <= r
            coords_i = coords[index_i]
            vel_i = velocities[index_i]
            atoms_i = atoms[index_i]
            
            eligible_bitmask = eligible_compressed[lane, i, j]
            special_bitmask = special_compressed[lane, i, j]

            for m in (lane + a) : r
                idx_j = j_0_tile + m
                coords_j = coords[idx_j]
                vel_j = velocities[idx_j]
                atoms_j = atoms[idx_j]
                
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
                
                # Divergence-safe execution (no vote_any_sync)
                f = condition ? sum_pairwise_forces(
                    inters_tuple, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : zero(SVector{D, T})

                @inbounds for k in a:b
                    force_smem[lane, k, warpid] += ustrip(f[k])
                    opposites_sum[m, k, warpid] -= ustrip(f[k])
                end
                
                if needs_vir
                    vir_xx += ustrip(f[1]) * ustrip(dr[1])
                    if D >= 2
                        vir_yy += ustrip(f[2]) * ustrip(dr[2])
                        vir_xy += ustrip(f[1]) * ustrip(dr[2])
                    end
                    if D >= 3
                        vir_zz += ustrip(f[3]) * ustrip(dr[3])
                        vir_xz += ustrip(f[1]) * ustrip(dr[3])
                        vir_yz += ustrip(f[2]) * ustrip(dr[3])
                    end
                end
            end
        end
        
        sync_warp()

        if lane <= r
            @inbounds for k in a:b
                force_smem[lane, k, warpid] += opposites_sum[lane, k, warpid]
            end
        end
    end

    # Each warp adds its force_smem to global memory directly because `i` is different for each warp
    if index_i <= N
        @inbounds for k in a:b
            f_i = force_smem[lane, k, warpid]
            if f_i != zero(T)
                CUDA.atomic_add!(pointer(fs_mat, index_i * b - (b - k)), -f_i)
            end
        end
    end

    if needs_vir
        offset = Int32(16)
        while offset > 0
            vir_xx += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xx, offset)
            if D >= 2
                vir_yy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yy, offset)
                vir_xy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xy, offset)
            end
            if D >= 3
                vir_zz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_zz, offset)
                vir_xz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xz, offset)
                vir_yz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yz, offset)
            end
            offset ÷= 2
        end

        if lane == 1
            if vir_xx != zero(T)
                CUDA.atomic_add!(pointer(global_virial, 1), vir_xx)
            end
            if D >= 2
                if vir_xy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 2), vir_xy)
                    CUDA.atomic_add!(pointer(global_virial, D + 1), vir_xy)
                end
                if vir_yy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, D + 2), vir_yy)
                end
            end
            if D >= 3
                if vir_xz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 3), vir_xz)
                    CUDA.atomic_add!(pointer(global_virial, 2*D + 1), vir_xz)
                end
                if vir_yz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 6), vir_yz) 
                    CUDA.atomic_add!(pointer(global_virial, 2*D + 2), vir_yz) 
                end
                if vir_zz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 9), vir_zz)
                end
            end
        end
    end

    return nothing
end

function dense_force_kernel!(
    sorted_seq,
    fs_mat,
    global_virial,
    mins::AbstractArray{C},
    maxs::AbstractArray{C},
    coords,
    velocities,
    atoms::AbstractArray{A},
    ::Val{N},
    r_cut,
    ::Val{force_units},
    inters_tuple,
    boundary,
    step_n,
    special_compressed,
    eligible_compressed,
    ::Val{needs_vir},
    ::Val{T},
    ::Val{D},
    ::Val{BLOCK_Y}) where {N, C, A, force_units, needs_vir, T, D, BLOCK_Y}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)

    i = blockIdx().x
    j = (blockIdx().y - a) * BLOCK_Y + threadIdx().y

    lane = threadIdx().x
    warpid = threadIdx().y

    i_0_tile = (i - a) * warpsize()
    j_0_tile = (j - a) * warpsize()
    index_i = i_0_tile + lane
    index_j = j_0_tile + lane

    is_active_j = j <= n_blocks

    force_smem = CuStaticSharedArray(T, (32, D, BLOCK_Y))
    opposites_sum = CuStaticSharedArray(T, (32, D, BLOCK_Y))

    if needs_vir
        warp_virial = CuStaticSharedArray(T, (6, BLOCK_Y))
    end

    r = Int32((N - 1) % 32 + 1)

    @inbounds for k in a:b
        force_smem[lane, k, warpid] = zero(T)
        opposites_sum[lane, k, warpid] = zero(T)
    end

    vir_xx = zero(T); vir_yy = zero(T); vir_zz = zero(T)
    vir_xy = zero(T); vir_xz = zero(T); vir_yz = zero(T)

    if is_active_j
        if j < n_blocks && i < j
            r_max_i = SVector{D}(ntuple(d -> maxs[i, d], D))
            r_min_i = SVector{D}(ntuple(d -> mins[i, d], D))
            r_max_j = SVector{D}(ntuple(d -> maxs[j, d], D))
            r_min_j = SVector{D}(ntuple(d -> mins[j, d], D))
            d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)

            if sum(abs2, d_block) <= r_cut * r_cut
                s_idx_i = sorted_seq[index_i]
                coords_i = coords[s_idx_i]
                vel_i = velocities[s_idx_i]
                atoms_i = atoms[s_idx_i]

                s_idx_j = sorted_seq[index_j]
                coords_j = coords[s_idx_j]
                vel_j = velocities[s_idx_j]
                shuffle_idx = lane
                atoms_j = atoms[s_idx_j]
                atom_fields = getfield.((atoms_j,), fieldnames(A))

                eligible_bitmask = eligible_compressed[lane, i, j]
                special_bitmask = special_compressed[lane, i, j]

                for m in a:warpsize()
                    sync_warp()
                    coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, lane + a, warpsize())
                    vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, lane + a, warpsize())
                    shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, lane + a, warpsize())
                    atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, lane + a, warpsize())
                    atoms_j_shuffle = A(atom_fields...)

                    dr = vector(coords_i, coords_j, boundary)
                    r2 = sum(abs2, dr)
                    excl = (eligible_bitmask >> (warpsize() - shuffle_idx)) | (eligible_bitmask << shuffle_idx)
                    spec = (special_bitmask >> (warpsize() - shuffle_idx)) | (special_bitmask << shuffle_idx)

                    condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
                    any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)

                    if any_active
                        f = condition ? sum_pairwise_forces(
                            inters_tuple, atoms_i, atoms_j_shuffle, Val(force_units),
                            (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                        ) : zero(SVector{D, T})

                        @inbounds for k in a:b
                            force_smem[lane, k, warpid] += ustrip(f[k])
                            opposites_sum[shuffle_idx, k, warpid] -= ustrip(f[k])
                        end

                        if needs_vir
                            vir_xx += ustrip(f[1]) * ustrip(dr[1])
                            if D >= 2
                                vir_yy += ustrip(f[2]) * ustrip(dr[2])
                                vir_xy += ustrip(f[1]) * ustrip(dr[2])
                            end
                            if D >= 3
                                vir_zz += ustrip(f[3]) * ustrip(dr[3])
                                vir_xz += ustrip(f[1]) * ustrip(dr[3])
                                vir_yz += ustrip(f[2]) * ustrip(dr[3])
                            end
                        end
                    end
                end

                sync_warp()
                @inbounds for k in a:b
                    if opposites_sum[lane, k, warpid] != zero(T)
                        CUDA.atomic_add!(pointer(fs_mat, s_idx_j * b - (b - k)), -opposites_sum[lane, k, warpid])
                    end
                end
            end
        end

        if j == n_blocks && i < n_blocks
            r_max_i = SVector{D}(ntuple(d -> maxs[i, d], D))
            r_min_i = SVector{D}(ntuple(d -> mins[i, d], D))
            r_max_j = SVector{D}(ntuple(d -> maxs[j, d], D))
            r_min_j = SVector{D}(ntuple(d -> mins[j, d], D))
            d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)

            if sum(abs2, d_block) <= r_cut * r_cut
                s_idx_i = sorted_seq[index_i]
                coords_i = coords[s_idx_i]
                vel_i = velocities[s_idx_i]
                atoms_i = atoms[s_idx_i]

                eligible_bitmask = eligible_compressed[lane, i, j]
                special_bitmask = special_compressed[lane, i, j]

                for m in a:r
                    s_idx_j = sorted_seq[j_0_tile + m]
                    coords_j = coords[s_idx_j]
                    vel_j = velocities[s_idx_j]
                    atoms_j = atoms[s_idx_j]

                    dr = vector(coords_i, coords_j, boundary)
                    r2 = sum(abs2, dr)
                    excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                    spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)

                    condition = (excl & 0x1) == true && r2 <= r_cut * r_cut
                    any_active = CUDA.vote_any_sync(0xFFFFFFFF, condition)

                    if any_active
                        f = condition ? sum_pairwise_forces(
                            inters_tuple, atoms_i, atoms_j, Val(force_units),
                            (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                        ) : zero(SVector{D, T})

                        @inbounds for k in a:b
                            force_smem[lane, k, warpid] += ustrip(f[k])
                            if ustrip(f[k]) != zero(T)
                                CUDA.atomic_add!(pointer(fs_mat, s_idx_j * b - (b - k)), ustrip(f[k]))
                            end
                        end

                        if needs_vir
                            vir_xx += ustrip(f[1]) * ustrip(dr[1])
                            if D >= 2
                                vir_yy += ustrip(f[2]) * ustrip(dr[2])
                                vir_xy += ustrip(f[1]) * ustrip(dr[2])
                            end
                            if D >= 3
                                vir_zz += ustrip(f[3]) * ustrip(dr[3])
                                vir_xz += ustrip(f[1]) * ustrip(dr[3])
                                vir_yz += ustrip(f[2]) * ustrip(dr[3])
                            end
                        end
                    end
                end
            end
        end

        if i == j && i < n_blocks
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i]
            vel_i = velocities[s_idx_i]
            atoms_i = atoms[s_idx_i]

            eligible_bitmask = eligible_compressed[lane, i, j]
            special_bitmask = special_compressed[lane, i, j]

            for m in (lane + a):warpsize()
                s_idx_j = sorted_seq[j_0_tile + m]
                coords_j = coords[s_idx_j]
                vel_j = velocities[s_idx_j]
                atoms_j = atoms[s_idx_j]

                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                f = condition ? sum_pairwise_forces(
                    inters_tuple, atoms_i, atoms_j, Val(force_units),
                    (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                ) : zero(SVector{D, T})

                @inbounds for k in a:b
                    force_smem[lane, k, warpid] += ustrip(f[k])
                    opposites_sum[m, k, warpid] -= ustrip(f[k])
                end

                if needs_vir
                    vir_xx += ustrip(f[1]) * ustrip(dr[1])
                    if D >= 2
                        vir_yy += ustrip(f[2]) * ustrip(dr[2])
                        vir_xy += ustrip(f[1]) * ustrip(dr[2])
                    end
                    if D >= 3
                        vir_zz += ustrip(f[3]) * ustrip(dr[3])
                        vir_xz += ustrip(f[1]) * ustrip(dr[3])
                        vir_yz += ustrip(f[2]) * ustrip(dr[3])
                    end
                end
            end

            sync_warp()
            @inbounds for k in a:b
                force_smem[lane, k, warpid] += opposites_sum[lane, k, warpid]
            end
        end

        if i == n_blocks && j == n_blocks
            if lane <= r
                s_idx_i = sorted_seq[index_i]
                coords_i = coords[s_idx_i]
                vel_i = velocities[s_idx_i]
                atoms_i = atoms[s_idx_i]

                eligible_bitmask = eligible_compressed[lane, i, j]
                special_bitmask = special_compressed[lane, i, j]

                for m in (lane + a):r
                    s_idx_j = sorted_seq[j_0_tile + m]
                    coords_j = coords[s_idx_j]
                    vel_j = velocities[s_idx_j]
                    atoms_j = atoms[s_idx_j]

                    dr = vector(coords_i, coords_j, boundary)
                    r2 = sum(abs2, dr)
                    excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                    spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                    condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                    f = condition ? sum_pairwise_forces(
                        inters_tuple, atoms_i, atoms_j, Val(force_units),
                        (spec & 0x1) == true, coords_i, coords_j, boundary, vel_i, vel_j, step_n
                    ) : zero(SVector{D, T})

                    @inbounds for k in a:b
                        force_smem[lane, k, warpid] += ustrip(f[k])
                        opposites_sum[m, k, warpid] -= ustrip(f[k])
                    end

                    if needs_vir
                        vir_xx += ustrip(f[1]) * ustrip(dr[1])
                        if D >= 2
                            vir_yy += ustrip(f[2]) * ustrip(dr[2])
                            vir_xy += ustrip(f[1]) * ustrip(dr[2])
                        end
                        if D >= 3
                            vir_zz += ustrip(f[3]) * ustrip(dr[3])
                            vir_xz += ustrip(f[1]) * ustrip(dr[3])
                            vir_yz += ustrip(f[2]) * ustrip(dr[3])
                        end
                    end
                end
            end

            sync_warp()

            if lane <= r
                @inbounds for k in a:b
                    force_smem[lane, k, warpid] += opposites_sum[lane, k, warpid]
                end
            end
        end
    end

    if needs_vir
        offset = Int32(16)
        while offset > 0
            vir_xx += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xx, offset)
            if D >= 2
                vir_yy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yy, offset)
                vir_xy += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xy, offset)
            end
            if D >= 3
                vir_zz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_zz, offset)
                vir_xz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_xz, offset)
                vir_yz += CUDA.shfl_down_sync(0xFFFFFFFF, vir_yz, offset)
            end
            offset ÷= 2
        end

        if lane == 1
            warp_virial[1, warpid] = vir_xx
            if D >= 2
                warp_virial[2, warpid] = vir_yy
                warp_virial[4, warpid] = vir_xy
            end
            if D >= 3
                warp_virial[3, warpid] = vir_zz
                warp_virial[5, warpid] = vir_xz
                warp_virial[6, warpid] = vir_yz
            end
        end
    end

    sync_threads()

    if warpid == a
        if index_i <= N
            s_idx_i = sorted_seq[index_i]
            @inbounds for k in a:b
                reduced_force_i = zero(T)
                for w in a:BLOCK_Y
                    reduced_force_i += force_smem[lane, k, w]
                end

                if reduced_force_i != zero(T)
                    CUDA.atomic_add!(pointer(fs_mat, s_idx_i * b - (b - k)), -reduced_force_i)
                end
            end
        end

        if needs_vir && lane == a
            sum_xx = zero(T); sum_yy = zero(T); sum_zz = zero(T)
            sum_xy = zero(T); sum_xz = zero(T); sum_yz = zero(T)

            for w in a:BLOCK_Y
                sum_xx += warp_virial[1, w]
                if D >= 2
                    sum_yy += warp_virial[2, w]
                    sum_xy += warp_virial[4, w]
                end
                if D >= 3
                    sum_zz += warp_virial[3, w]
                    sum_xz += warp_virial[5, w]
                    sum_yz += warp_virial[6, w]
                end
            end

            if sum_xx != zero(T)
                CUDA.atomic_add!(pointer(global_virial, 1), sum_xx)
            end
            if D >= 2
                if sum_xy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 2), sum_xy)
                    CUDA.atomic_add!(pointer(global_virial, D + 1), sum_xy)
                end
                if sum_yy != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, D + 2), sum_yy)
                end
            end
            if D >= 3
                if sum_xz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 3), sum_xz)
                    CUDA.atomic_add!(pointer(global_virial, 2 * D + 1), sum_xz)
                end
                if sum_yz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 6), sum_yz)
                    CUDA.atomic_add!(pointer(global_virial, 2 * D + 2), sum_yz)
                end
                if sum_zz != zero(T)
                    CUDA.atomic_add!(pointer(global_virial, 9), sum_zz)
                end
            end
        end
    end

    return nothing
end

function energy_kernel!(
    sorted_seq,
    energy_nounits,
    coords,
    velocities,
    atoms::AbstractArray{A},
    ::Val{N},
    r_cut,
    ::Val{energy_units},
    inters_tuple,
    boundary,
    step_n,
    special_compressed,
    eligible_compressed,
    ::Val{T},
    ::Val{D},
    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
    interacting_tiles_overflow) where {N, A, energy_units, T, D}

    a = Int32(1)
    b = Int32(D)
    n_blocks = ceil(Int32, N / 32)
    active_warps = Int32(blockDim().y)

    if interacting_tiles_overflow[1] != 0
        return nothing
    end
    
    tile_idx = (blockIdx().x - a) * active_warps + threadIdx().y
    num_tiles = num_interacting_tiles[1]
    is_active_tile = tile_idx <= num_tiles

    i = is_active_tile ? interacting_tiles_i[tile_idx] : Int32(1)
    j = is_active_tile ? interacting_tiles_j[tile_idx] : Int32(1)
    
    i_0_tile = (i - a) * warpsize()
    j_0_tile = (j - a) * warpsize()
    index_i = i_0_tile + laneid()
    index_j = j_0_tile + laneid()
    E_smem = CuStaticSharedArray(T, (32, MAX_BLOCK_Y))
    warpid = threadIdx().y
    E_smem[laneid(), warpid] = zero(T)
    r = Int32((N - 1) % 32 + 1)

    if is_active_tile
    # The code is organised in 4 mutually excluding parts
    if j < n_blocks && i < j
        if true
            coords_i = coords[index_i]
            vel_i = velocities[index_i]
            atoms_i = atoms[index_i]
            coords_j = coords[index_j]
            vel_j = velocities[index_j]
            shuffle_idx = laneid()
            atoms_j = atoms[index_j]
            atom_fields = getfield.((atoms_j,), fieldnames(A))
            
            tile_type = interacting_tiles_type[tile_idx]

            if tile_type == UInt8(0) # CLEAN
                for m in a:warpsize()
                    sync_warp()
                    coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, laneid() + a, warpsize())
                    vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, laneid() + a, warpsize())
                    shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, laneid() + a, warpsize())
                    atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, laneid() + a, warpsize())
                    atoms_j_shuffle = A(atom_fields...)

                    dr = vector(coords_i, coords_j, boundary)
                    r2 = sum(abs2, dr)
                    condition = r2 <= r_cut * r_cut

                    pe = condition ? sum_pairwise_potentials(
                        inters_tuple,
                        atoms_i, atoms_j_shuffle,
                        Val(energy_units),
                        false,
                        coords_i, coords_j,
                        boundary,
                        vel_i, vel_j,
                        step_n) : zero(SVector{1, T})

                    E_smem[laneid(), warpid] += ustrip(pe[1])
                end
            else # EXCLUDED
                eligible_bitmask = eligible_compressed[laneid(), i, j]
                special_bitmask = special_compressed[laneid(), i, j]

                # Shuffle
                for m in a:warpsize()
                    sync_warp()
                    coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, laneid() + a, warpsize())
                    vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, laneid() + a, warpsize())
                    shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, laneid() + a, warpsize())
                    atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, laneid() + a, warpsize())
                    atoms_j_shuffle = A(atom_fields...)

                    dr = vector(coords_i, coords_j, boundary)
                    r2 = sum(abs2, dr)
                    excl = (eligible_bitmask >> (warpsize() - shuffle_idx)) | (eligible_bitmask << shuffle_idx)
                    spec = (special_bitmask >> (warpsize() - shuffle_idx)) | (special_bitmask << shuffle_idx)
                    condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                    pe = condition ? sum_pairwise_potentials(
                        inters_tuple,
                        atoms_i, atoms_j_shuffle,
                        Val(energy_units),
                        (spec & 0x1) == true,
                        coords_i, coords_j,
                        boundary,
                        vel_i, vel_j,
                        step_n) : zero(SVector{1, T})

                    E_smem[laneid(), warpid] += ustrip(pe[1])
                end
            end
        end
    end

    if j == n_blocks && i < n_blocks
        if true
            coords_i = coords[index_i]
            vel_i = velocities[index_i]
            atoms_i = atoms[index_i]
            eligible_bitmask = UInt32(0)
            special_bitmask = UInt32(0)
            eligible_bitmask = eligible_compressed[laneid(), i, j]
            special_bitmask = special_compressed[laneid(), i, j]

            for m in a:r
                idx_j = j_0_tile + m
                coords_j = coords[idx_j]
                vel_j = velocities[idx_j]
                atoms_j = atoms[idx_j]
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})

                E_smem[laneid(), warpid] += ustrip(pe[1])
            end
        end
    end

    if i == j && i < n_blocks
        coords_i = coords[index_i]
        vel_i = velocities[index_i]
        atoms_i = atoms[index_i]
        eligible_bitmask = UInt32(0)
        special_bitmask = UInt32(0)
        eligible_bitmask = eligible_compressed[laneid(), i, j]
        special_bitmask = special_compressed[laneid(), i, j]

        for m in (laneid() + a) : warpsize()
            idx_j = j_0_tile + m
            coords_j = coords[idx_j]
            vel_j = velocities[idx_j]
            atoms_j = atoms[idx_j]
            dr = vector(coords_i, coords_j, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

            pe = condition ? sum_pairwise_potentials(
                inters_tuple,
                atoms_i, atoms_j,
                Val(energy_units),
                (spec & 0x1) == true,
                coords_i, coords_j,
                boundary,
                vel_i, vel_j,
                step_n) : zero(SVector{1, T})
            E_smem[laneid(), warpid] += ustrip(pe[1])
        end	
    end

    if i == n_blocks && j == n_blocks
        if laneid() <= r
            coords_i = coords[index_i]
            vel_i = velocities[index_i]
            atoms_i = atoms[index_i]
            eligible_bitmask = UInt32(0)
            special_bitmask = UInt32(0)
            eligible_bitmask = eligible_compressed[laneid(), i, j]
            special_bitmask = special_compressed[laneid(), i, j]

            for m in (laneid() + a) : r
                idx_j = j_0_tile + m
                coords_j = coords[idx_j]
                vel_j = velocities[idx_j]
                atoms_j = atoms[idx_j]
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})
                E_smem[laneid(), warpid] += ustrip(pe[1])
            end
        end
    end

    end # is_active_tile
    sync_warp()
    if threadIdx().x == a && is_active_tile
        sum_E = zero(T)
        for k in a:warpsize()
            sum_E += E_smem[k, warpid]
        end
        CUDA.atomic_add!(pointer(energy_nounits), sum_E)
    end

    return nothing
end

function dense_energy_kernel!(
    sorted_seq,
    energy_nounits,
    mins::AbstractArray{C},
    maxs::AbstractArray{C},
    coords,
    velocities,
    atoms::AbstractArray{A},
    ::Val{N},
    r_cut,
    ::Val{energy_units},
    inters_tuple,
    boundary,
    step_n,
    special_compressed,
    eligible_compressed,
    ::Val{T},
    ::Val{D}) where {N, C, A, energy_units, T, D}

    a = Int32(1)
    n_blocks = ceil(Int32, N / 32)
    i = blockIdx().x
    j = blockIdx().y
    i_0_tile = (i - a) * warpsize()
    j_0_tile = (j - a) * warpsize()
    index_i = i_0_tile + laneid()
    index_j = j_0_tile + laneid()
    E_smem = CuStaticSharedArray(T, 32)
    E_smem[laneid()] = zero(T)
    r = Int32((N - 1) % 32 + 1)

    if j < n_blocks && i < j
        r_max_i = SVector{D}(ntuple(d -> maxs[i, d], D))
        r_min_i = SVector{D}(ntuple(d -> mins[i, d], D))
        r_max_j = SVector{D}(ntuple(d -> maxs[j, d], D))
        r_min_j = SVector{D}(ntuple(d -> mins[j, d], D))
        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)
        if sum(abs2, d_block) <= r_cut * r_cut
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i]
            vel_i = velocities[s_idx_i]
            atoms_i = atoms[s_idx_i]
            s_idx_j = sorted_seq[index_j]
            coords_j = coords[s_idx_j]
            vel_j = velocities[s_idx_j]
            shuffle_idx = laneid()
            atoms_j = atoms[s_idx_j]
            atom_fields = getfield.((atoms_j,), fieldnames(A))
            eligible_bitmask = eligible_compressed[laneid(), i, j]
            special_bitmask = special_compressed[laneid(), i, j]

            for m in a:warpsize()
                sync_warp()
                coords_j = CUDA.shfl_sync(0xFFFFFFFF, coords_j, laneid() + a, warpsize())
                vel_j = CUDA.shfl_sync(0xFFFFFFFF, vel_j, laneid() + a, warpsize())
                shuffle_idx = CUDA.shfl_sync(0xFFFFFFFF, shuffle_idx, laneid() + a, warpsize())
                atom_fields = CUDA.shfl_sync.(0xFFFFFFFF, atom_fields, laneid() + a, warpsize())
                atoms_j_shuffle = A(atom_fields...)

                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - shuffle_idx)) | (eligible_bitmask << shuffle_idx)
                spec = (special_bitmask >> (warpsize() - shuffle_idx)) | (special_bitmask << shuffle_idx)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j_shuffle,
                    Val(energy_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})

                E_smem[laneid()] += ustrip(pe[1])
            end
        end
    end

    if j == n_blocks && i < n_blocks
        r_max_i = SVector{D}(ntuple(d -> maxs[i, d], D))
        r_min_i = SVector{D}(ntuple(d -> mins[i, d], D))
        r_max_j = SVector{D}(ntuple(d -> maxs[j, d], D))
        r_min_j = SVector{D}(ntuple(d -> mins[j, d], D))
        d_block = boxes_dist(r_min_i, r_max_i, r_min_j, r_max_j, boundary)

        if sum(abs2, d_block) <= r_cut * r_cut
            s_idx_i = sorted_seq[index_i]
            coords_i = coords[s_idx_i]
            vel_i = velocities[s_idx_i]
            atoms_i = atoms[s_idx_i]
            eligible_bitmask = eligible_compressed[laneid(), i, j]
            special_bitmask = special_compressed[laneid(), i, j]

            for m in a:r
                s_idx_j = sorted_seq[j_0_tile + m]
                coords_j = coords[s_idx_j]
                vel_j = velocities[s_idx_j]
                atoms_j = atoms[s_idx_j]
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    (spec & 0x1) == true,
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
        eligible_bitmask = eligible_compressed[laneid(), i, j]
        special_bitmask = special_compressed[laneid(), i, j]

        for m in (laneid() + a):warpsize()
            s_idx_j = sorted_seq[j_0_tile + m]
            coords_j = coords[s_idx_j]
            vel_j = velocities[s_idx_j]
            atoms_j = atoms[s_idx_j]
            dr = vector(coords_i, coords_j, boundary)
            r2 = sum(abs2, dr)
            excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
            spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
            condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

            pe = condition ? sum_pairwise_potentials(
                inters_tuple,
                atoms_i, atoms_j,
                Val(energy_units),
                (spec & 0x1) == true,
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
            eligible_bitmask = eligible_compressed[laneid(), i, j]
            special_bitmask = special_compressed[laneid(), i, j]

            for m in (laneid() + a):r
                s_idx_j = sorted_seq[j_0_tile + m]
                coords_j = coords[s_idx_j]
                vel_j = velocities[s_idx_j]
                atoms_j = atoms[s_idx_j]
                dr = vector(coords_i, coords_j, boundary)
                r2 = sum(abs2, dr)
                excl = (eligible_bitmask >> (warpsize() - m)) | (eligible_bitmask << m)
                spec = (special_bitmask >> (warpsize() - m)) | (special_bitmask << m)
                condition = (excl & 0x1) == true && r2 <= r_cut * r_cut

                pe = condition ? sum_pairwise_potentials(
                    inters_tuple,
                    atoms_i, atoms_j,
                    Val(energy_units),
                    (spec & 0x1) == true,
                    coords_i, coords_j,
                    boundary,
                    vel_i, vel_j,
                    step_n) : zero(SVector{1, T})
                E_smem[laneid()] += ustrip(pe[1])
            end
        end
    end

    sync_threads()
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
