using Molly
using Molly: from_device, to_device, sorted_morton_seq!, init_buffers!, box_sides, GPUNeighborFinder
using CUDA
using Test
using LinearAlgebra
using StaticArrays

@testset "GPU Optimizations" begin
    if CUDA.functional()
        n_atoms = 100
        D = 3
        T = Float64
        # Use better spaced coordinates to avoid massive LJ forces
        coords = [SVector{D, T}(0.1 * i, 0.1 * i, 0.1 * i) for i in 1:n_atoms]
        boundary = CubicBoundary(T(20.0), T(20.0), T(20.0))
        # Atom parameters σ and ϵ must be in internal units (no units here)
        atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0)) for i in 1:n_atoms]
        
        # Use GPUNeighborFinder to trigger the optimized GPU path
        sys = System(
            atoms=CuArray(atoms),
            coords=CuArray(coords),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=GPUNeighborFinder(
                eligible=CuArray(trues(n_atoms, n_atoms)),
                dist_cutoff=T(5.0),
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
        
        @testset "Morton Code Granularity (Phase 1)" begin
            # Phase 1 uses 10 bits per dimension (30 bits total)
            morton_bits = 10
            sides = box_sides(sys.boundary)
            w = sides ./ (2^morton_bits)
            sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
            
            morton_codes = Array(buffers.morton_seq_buffer_1)
            # High precision should mean unique codes for well-spaced atoms
            @test length(unique(morton_codes)) == n_atoms
        end
        
        @testset "Physical Data Reordering (Phase 2)" begin
            morton_bits = 10
            sides = box_sides(sys.boundary)
            w = sides ./ (2^morton_bits)
            sorted_morton_seq!(buffers, sys.coords, w, morton_bits)
            
            # Reorder
            backend = Molly.get_backend(sys.coords)
            Molly.reorder_kernel!(backend, n_threads)(buffers.coords_reordered, sys.coords, buffers.morton_seq, ndrange=n_atoms)
            
            reordered_coords = Array(buffers.coords_reordered)
            orig_coords = Array(sys.coords)
            morton_seq = Array(buffers.morton_seq)
            
            for i in 1:n_atoms
                @test reordered_coords[i] ≈ orig_coords[morton_seq[i]]
            end
            
            # Reverse Reorder Forces
            fill!(buffers.fs_mat, 0.0)
            fill!(buffers.fs_mat_reordered, 1.0) # Dummy forces
            Molly.reverse_reorder_forces_kernel!(backend, n_threads)(buffers.fs_mat, buffers.fs_mat_reordered, buffers.morton_seq, ndrange=n_atoms)
            
            fs_mat = Array(buffers.fs_mat)
            for i in 1:n_atoms
                orig_idx = morton_seq[i]
                @test fs_mat[1, orig_idx] ≈ 1.0
                @test fs_mat[2, orig_idx] ≈ 1.0
                @test fs_mat[3, orig_idx] ≈ 1.0
            end
        end

        @testset "Total Force Consistency" begin
            cpu_sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(T(5.0))),),
                neighbor_finder=DistanceNeighborFinder(
                    eligible=trues(n_atoms, n_atoms),
                    dist_cutoff=T(5.0),
                ),
                force_units=NoUnits,
                energy_units=NoUnits
            )
            
            # Ensure neighbor list is built for both
            neighbors_cpu = find_neighbors(cpu_sys)
            neighbors_gpu = find_neighbors(sys) # This is Nothing for GPU if using tile algorithm
            
            fs_cpu = forces(cpu_sys, neighbors_cpu)
            fs_gpu = forces(sys, neighbors_gpu)
            
            fs_gpu_cpu = Array(fs_gpu)
            for i in 1:n_atoms
                @test isapprox(fs_gpu_cpu[i], fs_cpu[i], rtol=1e-8, atol=1e-10)
            end
            
            pe_cpu = potential_energy(cpu_sys, neighbors_cpu)
            pe_gpu = potential_energy(sys, neighbors_gpu)
            @test isapprox(pe_gpu, pe_cpu, rtol=1e-8, atol=1e-10)
        end
    else
        @warn "CUDA not functional, skipping GPU optimization tests"
    end
end
