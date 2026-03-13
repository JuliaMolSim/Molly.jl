using Molly
using Molly: from_device, to_device, sorted_morton_seq!, init_buffers!, box_sides, GPUNeighborFinder, get_backend
using CUDA
using Test
using LinearAlgebra
using StaticArrays

@testset "GPU Optimizations - Float64 Well-Posed" begin
    if CUDA.functional()
        n_atoms = 100
        D = 3
        T = Float64  # Using Float64 for high precision
        
        # Place atoms on a 3D grid to avoid overlap and simulate a realistic liquid/solid spacing
        coords = SVector{D, T}[]
        n_side = ceil(Int, n_atoms^(1/3))
        spacing = T(1.5) # σ = 1.0, so 1.5 is well outside the harsh repulsive core
        for i in 1:n_side, j in 1:n_side, k in 1:n_side
            if length(coords) < n_atoms
                push!(coords, SVector{D, T}(i * spacing, j * spacing, k * spacing))
            end
        end
        
        box_size = T((n_side + 2) * spacing)
        boundary = CubicBoundary(box_size, box_size, box_size)
        atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(1.0), ϵ=T(1.0)) for i in 1:n_atoms]
        
        r_cut = T(4.0)
        
        sys = System(
            atoms=CuArray(atoms),
            coords=CuArray(coords),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut)),),
            neighbor_finder=GPUNeighborFinder(
                eligible=CuArray(trues(n_atoms, n_atoms)),
                dist_cutoff=r_cut,
            ),
            force_units=NoUnits,
            energy_units=NoUnits
        )
        
        cpu_sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut)),),
            neighbor_finder=DistanceNeighborFinder(
                eligible=trues(n_atoms, n_atoms),
                dist_cutoff=r_cut,
            ),
            force_units=NoUnits,
            energy_units=NoUnits
        )
        
        # Run CPU forces
        neighbors_cpu = find_neighbors(cpu_sys)
        fs_cpu = forces(cpu_sys, neighbors_cpu)
        pe_cpu = potential_energy(cpu_sys, neighbors_cpu)
        
        # Run GPU forces (this triggers Phase 1, Phase 2, and Phase 3 optimizations)
        neighbors_gpu = find_neighbors(sys) # returns nothing for tiles
        fs_gpu = forces(sys, neighbors_gpu)
        pe_gpu = potential_energy(sys, neighbors_gpu)
        
        fs_gpu_cpu = Array(fs_gpu)
        
        # Compare
        for i in 1:n_atoms
            # Use isapprox with atol since some forces might be exactly zero or very close
            @test isapprox(fs_gpu_cpu[i], fs_cpu[i], rtol=1e-8, atol=1e-10)
        end
        
        @test isapprox(pe_gpu, pe_cpu, rtol=1e-8, atol=1e-10)
    end
end
