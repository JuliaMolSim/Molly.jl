using Molly
using Molly: from_device, to_device, sorted_morton_seq!, init_buffers!, box_sides, GPUNeighborFinder, get_backend
using CUDA
using Test
using LinearAlgebra
using StaticArrays

@testset "GPU Optimizations - 33-atom (No Cancellation)" begin
    if CUDA.functional()
        n_atoms = 33
        D = 3
        T = Float64
        coords = [SVector{D, T}(0.5 * i, 0.5 * i, 0.5 * i) for i in 1:n_atoms]
        boundary = CubicBoundary(T(20.0), T(20.0), T(20.0))
        atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0)) for i in 1:n_atoms]
        
        sys = System(
            atoms=CuArray(atoms),
            coords=CuArray(coords),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(T(5.0))),),
            neighbor_finder=GPUNeighborFinder(
                eligible=CuArray(trues(n_atoms, n_atoms)),
                dist_cutoff=T(5.0),
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
        for i in 1:n_atoms
            @test isapprox(fs_gpu_cpu[i], fs_cpu[i], rtol=1e-8, atol=1e-10)
        end
        
        pe_cpu = potential_energy(cpu_sys, neighbors_cpu)
        pe_gpu = potential_energy(sys, nothing)
        @test isapprox(pe_gpu, pe_cpu, rtol=1e-8, atol=1e-10)
    end
end
