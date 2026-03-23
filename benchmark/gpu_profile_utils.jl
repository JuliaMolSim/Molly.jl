using Molly
using Molly: box_sides, sorted_morton_seq!
using CUDA
using StaticArrays

function gpu_cuda_ext()
    ext = Base.get_extension(Molly, :MollyCUDAExt)
    ext === nothing && error("MollyCUDAExt is not loaded; import CUDA before profiling GPU kernels.")
    return ext
end

function gpu_stage_time_ms!(f::F) where {F}
    CUDA.synchronize()
    start_ns = time_ns()
    f()
    CUDA.synchronize()
    return (time_ns() - start_ns) / 1.0e6
end

function gpu_tile_stats(buffers, n_atoms::Integer)
    num_tiles = Int(only(Array(buffers.num_interacting_tiles)))
    overflow_count = Int(only(Array(buffers.interacting_tiles_overflow)))
    tile_types = Array(buffers.interacting_tiles_type[1:num_tiles])
    clean_tiles = count(==(UInt8(0)), tile_types)
    masked_tiles = num_tiles - clean_tiles
    n_blocks = cld(n_atoms, 32)
    total_possible_tiles = n_blocks * (n_blocks + 1) ÷ 2
    interacting_fraction = total_possible_tiles == 0 ? 0.0 : num_tiles / total_possible_tiles
    return (
        num_tiles = num_tiles,
        clean_tiles = clean_tiles,
        masked_tiles = masked_tiles,
        overflow_count = overflow_count,
        interacting_fraction = interacting_fraction,
    )
end

function profile_gpu_force_path!(sys::System{D, <:CuArray, T};
                                 step_n::Integer=0,
                                 needs_vir::Bool=false,
                                 buffers=Molly.init_buffers!(sys, 1)) where {D, T}
    ext = gpu_cuda_ext()
    pairwise_inters = Tuple(filter(use_neighbors, values(sys.pairwise_inters)))
    isempty(pairwise_inters) && error("No neighbor-list pairwise interactions found to profile.")

    N = length(sys.coords)
    n_blocks = cld(N, 32)
    r_cut = sys.neighbor_finder.dist_cutoff
    r_neighbors = sys.neighbor_finder.dist_neighbors

    fill!(buffers.fs_mat, zero(T))
    fill!(buffers.fs_mat_reordered, zero(T))
    fill!(buffers.virial_nounits, zero(T))

    morton_sort_ms = 0.0
    compress_ms = 0.0
    reorder_needed = step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized
    if reorder_needed
        morton_bits = 10
        sides = box_sides(sys.boundary)
        cell_width = sides ./ (2^morton_bits)
        morton_sort_ms = gpu_stage_time_ms!() do
            sorted_morton_seq!(buffers, sys.coords, cell_width, morton_bits)
            sys.neighbor_finder.initialized = true
        end
        compress_ms = gpu_stage_time_ms!() do
            ext.compress_sparse!(buffers, sys.neighbor_finder, Val(N))
        end
    end

    reorder_ms = gpu_stage_time_ms!() do
        ext.reorder_system_gpu!(buffers, sys)
    end

    bounds_ms = gpu_stage_time_ms!() do
        if sys.boundary isa TriclinicBoundary
            H = SMatrix{3, 3, T}(
                sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
                sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
                sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val,
            )
            @cuda blocks=n_blocks threads=32 ext.kernel_min_max_triclinic!(
                buffers.morton_seq,
                buffers.box_mins,
                buffers.box_maxs,
                sys.coords,
                inv(H),
                Val(N),
                sys.boundary,
                Val(D),
            )
        else
            @cuda blocks=n_blocks threads=32 ext.kernel_min_max!(
                buffers.morton_seq,
                buffers.box_mins,
                buffers.box_maxs,
                sys.coords,
                Val(N),
                sys.boundary,
                Val(D),
            )
        end
    end

    max_tiles = length(buffers.interacting_tiles_i)
    tile_find_ms = gpu_stage_time_ms!() do
        ext.reset_interacting_tile_state!(buffers)
        tile_kernel = @cuda launch=false ext.find_interacting_blocks_kernel!(
            buffers.interacting_tiles_i,
            buffers.interacting_tiles_j,
            buffers.interacting_tiles_type,
            buffers.num_interacting_tiles,
            buffers.interacting_tiles_overflow,
            buffers.box_mins,
            buffers.box_maxs,
            sys.boundary,
            r_neighbors,
            Val(n_blocks),
            Val(D),
            max_tiles,
            buffers.compressed_masks,
            buffers.tile_is_clean,
        )
        tile_threads_xy = ext.tile_launch_params(tile_kernel)
        tile_kernel(
            buffers.interacting_tiles_i,
            buffers.interacting_tiles_j,
            buffers.interacting_tiles_type,
            buffers.num_interacting_tiles,
            buffers.interacting_tiles_overflow,
            buffers.box_mins,
            buffers.box_maxs,
            sys.boundary,
            r_neighbors,
            Val(n_blocks),
            Val(D),
            max_tiles,
            buffers.compressed_masks,
            buffers.tile_is_clean;
            blocks=(cld(n_blocks, tile_threads_xy[1]), cld(n_blocks, tile_threads_xy[2])),
            threads=tile_threads_xy,
        )
    end

    num_tiles = Int(only(Array(buffers.num_interacting_tiles)))

    auto_kernel = @cuda launch=false always_inline=true ext.force_kernel!(
        buffers.fs_mat_reordered,
        buffers.virial_nounits,
        buffers.coords_reordered,
        buffers.velocities_reordered,
        buffers.atoms_reordered,
        Val(N),
        r_cut,
        Val(sys.force_units),
        pairwise_inters,
        sys.boundary,
        step_n,
        buffers.compressed_masks,
        Val(needs_vir),
        Val(T),
        Val(D),
        buffers.interacting_tiles_i,
        buffers.interacting_tiles_j,
        buffers.interacting_tiles_type,
        buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow,
    )
    block_y, maxregs = ext.force_launch_params(auto_kernel)
    n_blocks_launch = num_tiles > 0 ? cld(num_tiles, block_y) : 1
    force_kernel_ms = gpu_stage_time_ms!() do
        kernel = if maxregs === nothing
            auto_kernel
        else
            @cuda launch=false maxregs=maxregs always_inline=true ext.force_kernel!(
                buffers.fs_mat_reordered,
                buffers.virial_nounits,
                buffers.coords_reordered,
                buffers.velocities_reordered,
                buffers.atoms_reordered,
                Val(N),
                r_cut,
                Val(sys.force_units),
                pairwise_inters,
                sys.boundary,
                step_n,
                buffers.compressed_masks,
                Val(needs_vir),
                Val(T),
                Val(D),
                buffers.interacting_tiles_i,
                buffers.interacting_tiles_j,
                buffers.interacting_tiles_type,
                buffers.num_interacting_tiles,
                buffers.interacting_tiles_overflow,
            )
        end
        kernel(
            buffers.fs_mat_reordered,
            buffers.virial_nounits,
            buffers.coords_reordered,
            buffers.velocities_reordered,
            buffers.atoms_reordered,
            Val(N),
            r_cut,
            Val(sys.force_units),
            pairwise_inters,
            sys.boundary,
            step_n,
            buffers.compressed_masks,
            Val(needs_vir),
            Val(T),
            Val(D),
            buffers.interacting_tiles_i,
            buffers.interacting_tiles_j,
            buffers.interacting_tiles_type,
            buffers.num_interacting_tiles,
            buffers.interacting_tiles_overflow;
            threads=(32, block_y),
            blocks=n_blocks_launch,
        )
    end

    reverse_reorder_ms = gpu_stage_time_ms!() do
        ext.reverse_reorder_forces_gpu!(buffers, sys)
    end
    ext.throw_if_interacting_tiles_overflowed(buffers)

    return (
        times = (
            morton_sort_ms = morton_sort_ms,
            compress_ms = compress_ms,
            reorder_ms = reorder_ms,
            bounds_ms = bounds_ms,
            tile_find_ms = tile_find_ms,
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
    ext = gpu_cuda_ext()
    pairwise_inters = Tuple(filter(use_neighbors, values(sys.pairwise_inters)))
    isempty(pairwise_inters) && error("No neighbor-list pairwise interactions found to profile.")

    N = length(sys.coords)
    n_blocks = cld(N, 32)
    r_cut = sys.neighbor_finder.dist_cutoff
    r_neighbors = sys.neighbor_finder.dist_neighbors

    fill!(buffers.pe_vec_nounits, zero(T))

    morton_sort_ms = 0.0
    compress_ms = 0.0
    reorder_needed = step_n % sys.neighbor_finder.n_steps_reorder == 0 || !sys.neighbor_finder.initialized
    if reorder_needed
        morton_bits = 10
        sides = box_sides(sys.boundary)
        cell_width = sides ./ (2^morton_bits)
        morton_sort_ms = gpu_stage_time_ms!() do
            sorted_morton_seq!(buffers, sys.coords, cell_width, morton_bits)
            sys.neighbor_finder.initialized = true
        end
        compress_ms = gpu_stage_time_ms!() do
            ext.compress_sparse!(buffers, sys.neighbor_finder, Val(N))
        end
    end

    reorder_ms = gpu_stage_time_ms!() do
        ext.reorder_system_gpu!(buffers, sys)
    end

    bounds_ms = gpu_stage_time_ms!() do
        if sys.boundary isa TriclinicBoundary
            H = SMatrix{3, 3, T}(
                sys.boundary.basis_vectors[1][1].val, sys.boundary.basis_vectors[2][1].val, sys.boundary.basis_vectors[3][1].val,
                sys.boundary.basis_vectors[1][2].val, sys.boundary.basis_vectors[2][2].val, sys.boundary.basis_vectors[3][2].val,
                sys.boundary.basis_vectors[1][3].val, sys.boundary.basis_vectors[2][3].val, sys.boundary.basis_vectors[3][3].val,
            )
            @cuda blocks=n_blocks threads=32 ext.kernel_min_max_triclinic!(
                buffers.morton_seq,
                buffers.box_mins,
                buffers.box_maxs,
                sys.coords,
                inv(H),
                Val(N),
                sys.boundary,
                Val(D),
            )
        else
            @cuda blocks=n_blocks threads=32 ext.kernel_min_max!(
                buffers.morton_seq,
                buffers.box_mins,
                buffers.box_maxs,
                sys.coords,
                Val(N),
                sys.boundary,
                Val(D),
            )
        end
    end

    max_tiles = length(buffers.interacting_tiles_i)
    tile_find_ms = gpu_stage_time_ms!() do
        ext.reset_interacting_tile_state!(buffers)
        tile_kernel = @cuda launch=false ext.find_interacting_blocks_kernel!(
            buffers.interacting_tiles_i,
            buffers.interacting_tiles_j,
            buffers.interacting_tiles_type,
            buffers.num_interacting_tiles,
            buffers.interacting_tiles_overflow,
            buffers.box_mins,
            buffers.box_maxs,
            sys.boundary,
            r_neighbors,
            Val(n_blocks),
            Val(D),
            max_tiles,
            buffers.compressed_masks,
            buffers.tile_is_clean,
        )
        tile_threads_xy = ext.tile_launch_params(tile_kernel)
        tile_kernel(
            buffers.interacting_tiles_i,
            buffers.interacting_tiles_j,
            buffers.interacting_tiles_type,
            buffers.num_interacting_tiles,
            buffers.interacting_tiles_overflow,
            buffers.box_mins,
            buffers.box_maxs,
            sys.boundary,
            r_neighbors,
            Val(n_blocks),
            Val(D),
            max_tiles,
            buffers.compressed_masks,
            buffers.tile_is_clean;
            blocks=(cld(n_blocks, tile_threads_xy[1]), cld(n_blocks, tile_threads_xy[2])),
            threads=tile_threads_xy,
        )
    end

    num_tiles = Int(only(Array(buffers.num_interacting_tiles)))

    auto_kernel = @cuda launch=false always_inline=true ext.energy_kernel!(
        buffers.pe_vec_nounits,
        buffers.coords_reordered,
        buffers.velocities_reordered,
        buffers.atoms_reordered,
        Val(N),
        r_cut,
        Val(sys.energy_units),
        pairwise_inters,
        sys.boundary,
        step_n,
        buffers.compressed_masks,
        Val(T),
        Val(D),
        buffers.interacting_tiles_i,
        buffers.interacting_tiles_j,
        buffers.interacting_tiles_type,
        buffers.num_interacting_tiles,
        buffers.interacting_tiles_overflow,
    )
    block_y = ext.energy_launch_params(auto_kernel)
    n_blocks_launch = num_tiles > 0 ? cld(num_tiles, block_y) : 1
    energy_kernel_ms = gpu_stage_time_ms!() do
        auto_kernel(
            buffers.pe_vec_nounits,
            buffers.coords_reordered,
            buffers.velocities_reordered,
            buffers.atoms_reordered,
            Val(N),
            r_cut,
            Val(sys.energy_units),
            pairwise_inters,
            sys.boundary,
            step_n,
            buffers.compressed_masks,
            Val(T),
            Val(D),
            buffers.interacting_tiles_i,
            buffers.interacting_tiles_j,
            buffers.interacting_tiles_type,
            buffers.num_interacting_tiles,
            buffers.interacting_tiles_overflow;
            blocks=n_blocks_launch,
            threads=(32, block_y),
        )
    end
    ext.throw_if_interacting_tiles_overflowed(buffers)

    return (
        times = (
            morton_sort_ms = morton_sort_ms,
            compress_ms = compress_ms,
            reorder_ms = reorder_ms,
            bounds_ms = bounds_ms,
            tile_find_ms = tile_find_ms,
            energy_kernel_ms = energy_kernel_ms,
        ),
        tile_stats = gpu_tile_stats(buffers, N),
        launch = (
            energy_block_y = block_y,
        ),
        buffers = buffers,
    )
end
