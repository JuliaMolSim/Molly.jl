using Molly
using Molly: from_device, to_device, sorted_morton_seq!, init_buffers!, box_sides, GPUNeighborFinder
using CUDA
using Test
using LinearAlgebra
using StaticArrays

include(joinpath(dirname(@__DIR__), "benchmark", "gpu_profile_utils.jl"))

@testset "GPU Optimizations" begin
    if CUDA.functional()
        n_atoms = 100
        D = 3
        T = Float64
        coords = [SVector{D, T}(0.1 * i, 0.1 * i, 0.1 * i) for i in 1:n_atoms]
        boundary = CubicBoundary(T(20.0), T(20.0), T(20.0))
        atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0)) for i in 1:n_atoms]
        
        sys = System(
            atoms=CuArray(atoms),
            coords=CuArray(coords),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=GPUNeighborFinder(
                n_atoms=n_atoms,
                dist_cutoff=T(5.0),
                device_vector_type=CuArray{Int32, 1},
            ),
            force_units=NoUnits,
            energy_units=NoUnits
        )
        
        n_threads = 256
        buffers = init_buffers!(sys, n_threads)

        @testset "CUDA Launch Config API" begin
            Molly.reset_cuda_launch_config!()
            cfg_auto = Molly.cuda_launch_config()
            @test cfg_auto.force_block_y === nothing
            @test cfg_auto.force_maxregs === nothing
            @test cfg_auto.tile_threads === nothing
            @test cfg_auto.energy_block_y === nothing

            cfg_explicit = Molly.set_cuda_launch_config!(
                force_block_y=T == Float64 ? 8 : 12,
                force_maxregs=56,
                tile_threads=(16, 8),
                energy_block_y=4,
            )
            cfg_current = Molly.cuda_launch_config()
            @test cfg_current.force_block_y == cfg_explicit.force_block_y
            @test cfg_current.force_maxregs == cfg_explicit.force_maxregs
            @test cfg_current.tile_threads == cfg_explicit.tile_threads
            @test cfg_current.energy_block_y == cfg_explicit.energy_block_y

            fs_api = forces(sys, find_neighbors(sys))
            @test length(fs_api) == n_atoms

            Molly.reset_cuda_launch_config!()
            cfg_reset = Molly.cuda_launch_config()
            @test cfg_reset.force_block_y === nothing
            @test cfg_reset.force_maxregs === nothing
            @test cfg_reset.tile_threads === nothing
            @test cfg_reset.energy_block_y === nothing
        end

        @testset "CUDA Launch Autotuner" begin
            ext = Base.get_extension(Molly, :MollyCUDAExt)
            @test ext !== nothing

            Molly.reset_cuda_launch_config!()
            Molly.reset_cuda_launch_autotune_cache!()
            @test isempty(ext.CUDA_LAUNCH_AUTOTUNE_CACHE)

            chosen_block_y = Molly.optimize_cuda_launch_config!(sys)
            tuned_cfg = Molly.cuda_launch_config()
            @test chosen_block_y == tuned_cfg.force_block_y
            @test tuned_cfg.force_block_y in ext.AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES
            @test tuned_cfg.energy_block_y in ext.AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES
            @test tuned_cfg.tile_threads in ext.AUTOTUNE_TILE_THREAD_CANDIDATES
            @test length(ext.CUDA_LAUNCH_AUTOTUNE_CACHE) == 1

            # Test environment variable overrides
            ENV["MOLLY_CUDA_FORCE_BLOCK_Y"] = "1"
            ENV["MOLLY_CUDA_ENERGY_BLOCK_Y"] = "1"
            ENV["MOLLY_CUDA_TILE_THREADS_X"] = "8"
            ENV["MOLLY_CUDA_TILE_THREADS_Y"] = "8"
            
            Molly.reset_cuda_launch_config!()
            # optimize_cuda_launch_config! returns the effective force_block_y
            eff_block_y = Molly.optimize_cuda_launch_config!(sys)
            @test eff_block_y == 1
            
            # The global config should be updated with what was tuned (if not overridden in config),
            # but env vars are not stored in the config struct.
            # We can verify that the config has the tuned values for others if we didn't override them in config.
            
            delete!(ENV, "MOLLY_CUDA_FORCE_BLOCK_Y")
            delete!(ENV, "MOLLY_CUDA_ENERGY_BLOCK_Y")
            delete!(ENV, "MOLLY_CUDA_TILE_THREADS_X")
            delete!(ENV, "MOLLY_CUDA_TILE_THREADS_Y")

            Molly.reset_cuda_launch_config!()
            cached_block_y = Molly.optimize_cuda_launch_config!(sys)
            cached_cfg = Molly.cuda_launch_config()
            @test cached_block_y == tuned_cfg.force_block_y
            @test cached_cfg == tuned_cfg
            @test length(ext.CUDA_LAUNCH_AUTOTUNE_CACHE) == 1

            Molly.reset_cuda_launch_config!()
            Molly.set_cuda_launch_config!(force_block_y=8)
            Molly.optimize_cuda_launch_config!(sys)
            merged_cfg = Molly.cuda_launch_config()
            @test merged_cfg.force_block_y == 8
            @test merged_cfg.energy_block_y !== nothing
            @test merged_cfg.tile_threads !== nothing
            @test length(ext.CUDA_LAUNCH_AUTOTUNE_CACHE) == 1

            cfg_before_reset = Molly.cuda_launch_config()
            Molly.reset_cuda_launch_autotune_cache!()
            @test isempty(ext.CUDA_LAUNCH_AUTOTUNE_CACHE)
            @test Molly.cuda_launch_config() == cfg_before_reset

            Molly.reset_cuda_launch_config!()
        end

        @testset "Morton Code Granularity (Phase 1)" begin
            morton_bits = 10
            sides = box_sides(sys.boundary)
            w = sides ./ (2^morton_bits)
            sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
            
            morton_codes = Array(buffers.morton_seq_buffer_1)
            @test length(unique(morton_codes)) == n_atoms
        end
        
        @testset "Physical Data Reordering (Phase 2)" begin
            morton_bits = 10
            sides = box_sides(sys.boundary)
            w = sides ./ (2^morton_bits)
            sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
            
            backend = Molly.get_backend(sys.coords)
            Molly.reorder_kernel!(backend, n_threads)(buffers.coords_reordered, sys.coords, buffers.morton_seq, ndrange=n_atoms)
            
            reordered_coords = Array(buffers.coords_reordered)
            orig_coords = Array(sys.coords)
            morton_seq = Array(buffers.morton_seq)
            
            for i in 1:n_atoms
                @test reordered_coords[i] ≈ orig_coords[morton_seq[i]]
            end
            
            fill!(buffers.fs_mat, 0.0)
            fill!(buffers.fs_mat_reordered, 1.0)
            Molly.reverse_reorder_forces_kernel!(backend, n_threads)(buffers.fs_mat, buffers.fs_mat_reordered, buffers.morton_seq, ndrange=n_atoms)
            
            fs_mat = Array(buffers.fs_mat)
            for i in 1:n_atoms
                orig_idx = morton_seq[i]
                @test fs_mat[1, orig_idx] ≈ 1.0
                @test fs_mat[2, orig_idx] ≈ 1.0
                @test fs_mat[3, orig_idx] ≈ 1.0
            end
        end

        @testset "GPU Profiling Harness" begin
            Molly.reset_cuda_launch_config!()
            profile_force = profile_gpu_force_path!(sys)
            profile_energy = profile_gpu_energy_path!(sys)

            stats = profile_force.tile_stats
            @test stats.num_tiles > 0
            @test stats.clean_tiles + stats.masked_tiles == stats.num_tiles
            @test stats.overflow_count == 0
            @test 0.0 < stats.interacting_fraction <= 1.0

            force_times = profile_force.times
            @test force_times.reorder_ms >= 0.0
            @test force_times.bounds_ms >= 0.0
            @test force_times.tile_find_ms >= 0.0
            @test force_times.force_kernel_ms > 0.0
            @test force_times.reverse_reorder_ms >= 0.0

            energy_times = profile_energy.times
            @test energy_times.morton_sort_ms > 0.0
            @test energy_times.tile_find_ms >= 0.0
            @test energy_times.energy_kernel_ms > 0.0

            @test profile_energy.tile_stats.num_tiles == stats.num_tiles
        end

    else
        @warn "CUDA not functional, skipping GPU optimization tests"
    end
end
