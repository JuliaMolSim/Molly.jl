using Molly
using Molly: sorted_morton_seq!
using CUDA
using StaticArrays

# The stages of the tiled pairwise path of GPUNeighborFinder are timed one at a time, each
#   followed by a synchronization, so the times include the launch overhead of each stage

function gpu_stage_time_ms!(f::F) where {F}
    CUDA.synchronize()
    start_ns = time_ns()
    f()
    CUDA.synchronize()
    return (time_ns() - start_ns) / 1.0e6
end

function gpu_tile_stats(buffers, n_atoms::Integer)
    num_tiles = buffers.num_pairs
    overflow_count = Int(only(Array(buffers.interacting_tiles_overflow)))
    tile_types = Array(buffers.interacting_tiles_type[1:num_tiles])
    clean_tiles = count(==(UInt8(0)), tile_types)
    masked_tiles = num_tiles - clean_tiles
    n_blocks = cld(n_atoms, 32)
    total_possible_tiles = n_blocks * (n_blocks + 1) ÷ 2
    interacting_fraction = total_possible_tiles == 0 ? 0.0 : num_tiles / total_possible_tiles
    return (
        num_tiles = num_tiles,
        num_candidates = buffers.num_tile_candidates,
        clean_tiles = clean_tiles,
        masked_tiles = masked_tiles,
        overflow_count = overflow_count,
        interacting_fraction = interacting_fraction,
    )
end

function neighbor_pairwise_inters(sys)
    pairwise_inters = Tuple(filter(use_neighbors, values(sys.pairwise_inters)))
    isempty(pairwise_inters) && error("no neighbor-list pairwise interactions found to profile")
    return pairwise_inters
end

# Rebuild the Morton order, the exception data and the tile list of `buffers` from the
#   coordinates of `sys`, timing each stage
function profile_gpu_tile_rebuild!(sys::System{D, <:CuArray}, buffers) where D
    N = length(sys.coords)
    nf = sys.neighbor_finder
    morton_sort_ms = gpu_stage_time_ms!() do
        sorted_morton_seq!(buffers, sys.coords, sys.boundary)
    end
    compress_ms = gpu_stage_time_ms!() do
        Molly.refresh_tile_exceptions!(buffers, nf, Val(N))
        nf.initialized = true
        buffers.sparse_pair_generation = nf.cache_generation
    end
    reorder_ms = gpu_stage_time_ms!() do
        Molly.reorder_system_gpu!(buffers, sys)
    end
    bounds_ms = gpu_stage_time_ms!() do
        Molly.compute_block_bounds!(buffers, sys, N)
    end
    tile_find_ms = gpu_stage_time_ms!() do
        Molly.find_interacting_tiles!(buffers, sys, cld(N, 32))
    end
    prune_ms = gpu_stage_time_ms!() do
        Molly.prune_interacting_tiles!(buffers, sys, N)
    end
    Molly.throw_if_interacting_tiles_overflowed(buffers)
    return (
        morton_sort_ms = morton_sort_ms,
        compress_ms = compress_ms,
        reorder_ms = reorder_ms,
        bounds_ms = bounds_ms,
        tile_find_ms = tile_find_ms,
        prune_ms = prune_ms,
    )
end

function profile_gpu_force_path!(sys::System{D, <:CuArray, T};
                                 step_n::Integer=0,
                                 needs_vir::Bool=false,
                                 buffers=Molly.init_buffers!(sys, 1)) where {D, T}
    pairwise_inters = neighbor_pairwise_inters(sys)
    N = length(sys.coords)
    TH = Molly.float_type_high(sys)

    fill!(buffers.fs_mat, zero(T))
    fill!(buffers.fs_mat_reordered, zero(T))
    fill!(buffers.virial_slots, zero(TH))

    rebuild_times = profile_gpu_tile_rebuild!(sys, buffers)
    block_y = Molly.force_block_y(sys, buffers, pairwise_inters)
    maxregs = Molly.force_maxregs(sys)
    force_kernel_ms = gpu_stage_time_ms!() do
        Molly.launch_force_tiles!(buffers, sys, pairwise_inters, Val(needs_vir), step_n,
                                  block_y, maxregs)
    end
    reverse_reorder_ms = gpu_stage_time_ms!() do
        Molly.reverse_reorder_forces_gpu!(buffers, sys)
    end

    return (
        times = (
            rebuild_times...,
            force_kernel_ms = force_kernel_ms,
            reverse_reorder_ms = reverse_reorder_ms,
        ),
        tile_stats = gpu_tile_stats(buffers, N),
        launch = (
            force_block_y = block_y,
            force_maxregs = maxregs,
        ),
        buffers = buffers,
    )
end

function profile_gpu_energy_path!(sys::System{D, <:CuArray, T};
                                  step_n::Integer=0,
                                  buffers=Molly.init_buffers!(sys, 1, true)) where {D, T}
    pairwise_inters = neighbor_pairwise_inters(sys)
    N = length(sys.coords)
    TH = Molly.float_type_high(sys)

    fill!(buffers.pe_vec_nounits, zero(TH))

    rebuild_times = profile_gpu_tile_rebuild!(sys, buffers)
    block_y = Molly.energy_block_y(sys, buffers, pairwise_inters)
    energy_kernel_ms = gpu_stage_time_ms!() do
        Molly.launch_energy_tiles!(buffers.pe_vec_nounits, buffers, sys, pairwise_inters,
                                   step_n, block_y)
    end

    return (
        times = (
            rebuild_times...,
            energy_kernel_ms = energy_kernel_ms,
        ),
        tile_stats = gpu_tile_stats(buffers, N),
        launch = (
            energy_block_y = block_y,
        ),
        buffers = buffers,
    )
end
