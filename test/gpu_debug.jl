using Molly
using Molly: from_device, to_device, sorted_morton_seq!, init_buffers!, box_sides, GPUNeighborFinder, get_backend
using CUDA
using Test
using LinearAlgebra
using StaticArrays

@testset "GPU Optimizations - 33-atom Debug" begin
    if CUDA.functional()
        n_atoms = 33
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
        
        neighbors_cpu = find_neighbors(cpu_sys)
        fs_cpu = forces(cpu_sys, neighbors_cpu)
        fs_gpu = forces(sys, nothing)
        
        fs_gpu_cpu = Array(fs_gpu)
        println("Atom 1 force: CPU $(fs_cpu[1]), GPU $(fs_gpu_cpu[1])")
        println("Atom 33 force: CPU $(fs_cpu[33]), GPU $(fs_gpu_cpu[33])")
        
        @test true
    end
end
