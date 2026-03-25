using Molly
using Molly: from_device, to_device, sorted_morton_seq!, init_buffers!, box_sides, GPUNeighborFinder, get_backend
using CUDA
using Test
using LinearAlgebra
using StaticArrays
using Random

@testset "GPU Consistency" begin
    if CUDA.functional()
        @testset "33-atom (No Cancellation)" begin
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
            for i in 1:n_atoms
                @test isapprox(fs_gpu_cpu[i], fs_cpu[i], rtol=1e-8, atol=1e-10)
            end
            
            pe_cpu = potential_energy(cpu_sys, neighbors_cpu)
            pe_gpu = potential_energy(sys, nothing)
            @test isapprox(pe_gpu, pe_cpu, rtol=1e-8, atol=1e-10)
        end

        @testset "Float64 Well-Posed" begin
            n_atoms = 100
            D = 3
            T = Float64
            
            coords = SVector{D, T}[]
            n_side = ceil(Int, n_atoms^(1/3))
            spacing = T(1.5)
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
                    n_atoms=n_atoms,
                    dist_cutoff=r_cut,
                    device_vector_type=CuArray{Int32, 1},
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
            
            neighbors_cpu = find_neighbors(cpu_sys)
            fs_cpu = forces(cpu_sys, neighbors_cpu)
            pe_cpu = potential_energy(cpu_sys, neighbors_cpu)
            
            neighbors_gpu = find_neighbors(sys)
            fs_gpu = forces(sys, neighbors_gpu)
            pe_gpu = potential_energy(sys, neighbors_gpu)
            
            fs_gpu_cpu = Array(fs_gpu)
            
            for i in 1:n_atoms
                @test isapprox(fs_gpu_cpu[i], fs_cpu[i], rtol=1e-8, atol=1e-10)
            end
            
            @test isapprox(pe_gpu, pe_cpu, rtol=1e-8, atol=1e-10)
        end

        @testset "GPU tile lists (Units & Overflow)" begin
            n_atoms = 100
            atom_mass = 10.0u"g/mol"
            σ = 0.3u"nm"
            ϵ = 1.0u"kJ * mol^-1"
            boundary = CubicBoundary(10.0u"nm")
            
            Random.seed!(42)
            coords = [SVector(rand(), rand(), rand()) * 10.0u"nm" for _ in 1:n_atoms]
            velocities = [zero(SVector{3, typeof(1.0u"nm/ps")}) for _ in 1:n_atoms]
            atoms = [Atom(index=i, mass=atom_mass, charge=0.0, σ=σ, ϵ=ϵ) for i in 1:n_atoms]
            
            pairwise_inters_cpu = (LennardJones(
                cutoff=DistanceCutoff(3.0u"nm"),
                use_neighbors=false,
            ),)

            pairwise_inters_gpu = (LennardJones(
                cutoff=DistanceCutoff(3.0u"nm"),
                use_neighbors=true,
            ),)
            
            sys_cpu = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                pairwise_inters=pairwise_inters_cpu,
                neighbor_finder=Molly.NoNeighborFinder(),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )
            
            sys_gpu = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                velocities=CuArray(velocities),
                boundary=boundary,
                pairwise_inters=pairwise_inters_gpu,
                neighbor_finder=Molly.GPUNeighborFinder(
                    n_atoms=n_atoms,
                    dist_cutoff=3.0u"nm",
                    device_vector_type=CuArray{Int32, 1},
                ),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )
            
            f_cpu = forces(sys_cpu)
            f_gpu = forces(sys_gpu, nothing)
            f_gpu_host = Array(f_gpu)
            
            for i in 1:n_atoms
                @test isapprox(f_cpu[i], f_gpu_host[i], atol=1e-10u"kJ * mol^-1 * nm^-1")
            end

            pe_cpu = potential_energy(sys_cpu)
            pe_gpu = potential_energy(sys_gpu, nothing)
            
            @test isapprox(pe_cpu, pe_gpu, atol=1e-10u"kJ * mol^-1")

            # Overflow test
            pairwise_inters_gpu_overflow = (LennardJones(
                cutoff=DistanceCutoff(20.0u"nm"),
                use_neighbors=true,
            ),)

            sys_gpu_overflow = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                velocities=CuArray(velocities),
                boundary=boundary,
                pairwise_inters=pairwise_inters_gpu_overflow,
                neighbor_finder=Molly.GPUNeighborFinder(
                    n_atoms=n_atoms,
                    dist_cutoff=20.0u"nm",
                    device_vector_type=CuArray{Int32, 1},
                ),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )

            function with_tiny_tile_capacity(buffers)
                tiny_capacity = 1
                return Molly.BuffersGPU(
                    buffers.fs_mat,
                    buffers.pe_vec_nounits,
                    buffers.virial,
                    buffers.virial_nounits,
                    buffers.kin_tensor,
                    buffers.pres_tensor,
                    buffers.box_mins,
                    buffers.box_maxs,
                    buffers.morton_seq,
                    buffers.morton_seq_buffer_1,
                    buffers.morton_seq_buffer_2,
                    buffers.morton_seq_inv,
                    buffers.compressed_masks,
                    buffers.tile_is_clean,
                    CUDA.zeros(Int32, tiny_capacity),
                    CUDA.zeros(Int32, tiny_capacity),
                    CUDA.zeros(UInt8, tiny_capacity),
                    CUDA.zeros(Int32, 1),
                    CUDA.zeros(Int32, 1),
                    buffers.coords_reordered,
                    buffers.velocities_reordered,
                    buffers.atoms_reordered,
                    buffers.fs_mat_reordered,
                    -1,
                    0,
                )
            end

            overflow_force_buffers = with_tiny_tile_capacity(Molly.init_buffers!(sys_gpu_overflow, 1))
            overflow_energy_buffers = with_tiny_tile_capacity(Molly.init_buffers!(sys_gpu_overflow, 1, true))

            @test_throws ErrorException Molly.forces!(
                Molly.zero_forces(sys_gpu_overflow),
                sys_gpu_overflow,
                nothing,
                overflow_force_buffers,
                Val(false),
                0,
            )
            @test_throws ErrorException potential_energy(sys_gpu_overflow, nothing, overflow_energy_buffers, 0)
        end

        @testset "Triclinic Boundary" begin
            n_atoms = 50
            D = 3
            T = Float64
            # Define a triclinic box
            boundary = TriclinicBoundary(SVector(2.0, 0.0, 0.0), SVector(0.1, 2.0, 0.0), SVector(0.2, 0.3, 2.0))
            
            Random.seed!(42)
            coords = [SVector{D, T}(rand()*1.5, rand()*1.5, rand()*1.5) for _ in 1:n_atoms]
            atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0)) for i in 1:n_atoms]
            r_cut = T(0.8)

            sys = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut)),),
                neighbor_finder=GPUNeighborFinder(
                    n_atoms=n_atoms,
                    dist_cutoff=r_cut,
                    device_vector_type=CuArray{Int32, 1},
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
            
            fs_cpu = forces(cpu_sys, find_neighbors(cpu_sys))
            pe_cpu = potential_energy(cpu_sys, find_neighbors(cpu_sys))
            
            fs_gpu = forces(sys, nothing)
            pe_gpu = potential_energy(sys, nothing)
            
            fs_gpu_cpu = Array(fs_gpu)
            for i in 1:n_atoms
                @test isapprox(fs_gpu_cpu[i], fs_cpu[i], rtol=1e-8, atol=1e-10)
            end
            @test isapprox(pe_gpu, pe_cpu, rtol=1e-8, atol=1e-10)
        end

        @testset "Exclusions and Special Pairs" begin
            n_atoms = 10
            D = 3
            T = Float64
            boundary = CubicBoundary(T(10.0))
            coords = [SVector{D, T}(0.5 * i, 0.5 * i, 0.5 * i) for i in 1:n_atoms]
            atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0)) for i in 1:n_atoms]
            r_cut = T(5.0)

            # Exclude 1-2 and 2-3, Special for 1-3
            excluded_pairs = [(1, 2), (2, 3)]
            special_pairs = [(1, 3)]

            sys = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut)),),
                neighbor_finder=GPUNeighborFinder(
                    n_atoms=n_atoms,
                    dist_cutoff=r_cut,
                    excluded_pairs=excluded_pairs,
                    special_pairs=special_pairs,
                    device_vector_type=CuArray{Int32, 1},
                ),
                force_units=NoUnits,
                energy_units=NoUnits
            )

            eligible = trues(n_atoms, n_atoms)
            for (i, j) in excluded_pairs
                eligible[i, j] = eligible[j, i] = false
            end
            special = falses(n_atoms, n_atoms)
            for (i, j) in special_pairs
                special[i, j] = special[j, i] = true
                # For DistanceNeighborFinder, if it is special it must also be eligible 
                # to be included in the neighbor list with the special flag.
                eligible[i, j] = eligible[j, i] = true 
            end

            cpu_sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut)),),
                neighbor_finder=DistanceNeighborFinder(
                    eligible=eligible,
                    special=special,
                    dist_cutoff=r_cut,
                ),
                force_units=NoUnits,
                energy_units=NoUnits
            )

            fs_cpu = forces(cpu_sys, find_neighbors(cpu_sys))
            pe_cpu = potential_energy(cpu_sys, find_neighbors(cpu_sys))
            
            fs_gpu = forces(sys, nothing)
            pe_gpu = potential_energy(sys, nothing)
            
            fs_gpu_cpu = Array(fs_gpu)
            for i in 1:n_atoms
                @test isapprox(fs_gpu_cpu[i], fs_cpu[i], rtol=1e-8, atol=1e-10)
            end
            @test isapprox(pe_gpu, pe_cpu, rtol=1e-8, atol=1e-10)
        end

        @testset "Non-Neighborlist GPU Path" begin
            n_atoms = 20
            D = 3
            T = Float64
            boundary = CubicBoundary(T(10.0))
            coords = [SVector{D, T}(0.5 * i, 0.5 * i, 0.5 * i) for i in 1:n_atoms]
            atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0)) for i in 1:n_atoms]
            r_cut = T(5.0)

            # System on GPU with NoNeighborFinder
            sys = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=false, cutoff=DistanceCutoff(r_cut)),),
                neighbor_finder=NoNeighborFinder(),
                force_units=NoUnits,
                energy_units=NoUnits
            )
            
            # System on CPU
            cpu_sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=false, cutoff=DistanceCutoff(r_cut)),),
                neighbor_finder=NoNeighborFinder(),
                force_units=NoUnits,
                energy_units=NoUnits
            )
            
            fs_cpu = forces(cpu_sys)
            pe_cpu = potential_energy(cpu_sys)
            
            fs_gpu = forces(sys)
            pe_gpu = potential_energy(sys)
            
            fs_gpu_cpu = Array(fs_gpu)
            for i in 1:n_atoms
                @test isapprox(fs_gpu_cpu[i], fs_cpu[i], rtol=1e-8, atol=1e-10)
            end
            @test isapprox(pe_gpu, pe_cpu, rtol=1e-8, atol=1e-10)
        end

        @testset "GPU buffers refresh reordered data" begin
            n_atoms = 96
            D = 3
            T = Float64
            coords = [SVector{D, T}(0.12 * i, 0.08 * i, 0.05 * i) for i in 1:n_atoms]
            velocities = [SVector{D, T}(1e-3, -5e-4, 7.5e-4) for _ in 1:n_atoms]
            boundary = CubicBoundary(T(20.0), T(20.0), T(20.0))
            atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0)) for i in 1:n_atoms]
            r_cut = T(5.0)

            sys = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                velocities=CuArray(velocities),
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(r_cut)),),
                neighbor_finder=GPUNeighborFinder(
                    n_atoms=n_atoms,
                    dist_cutoff=r_cut,
                    n_steps_reorder=25,
                    device_vector_type=CuArray{Int32, 1},
                ),
                force_units=NoUnits,
                energy_units=NoUnits,
            )

            neighbors = find_neighbors(sys)
            buffers = init_buffers!(sys, 256)
            fs_reused = Molly.zero_forces(sys)

            Molly.forces!(fs_reused, sys, neighbors, buffers, Val(false), 0; n_threads=1)
            sys.coords .+= sys.velocities .* T(0.5)
            Molly.forces!(fs_reused, sys, neighbors, buffers, Val(false), 1; n_threads=1)

            fs_fresh = forces(sys, neighbors, 1; n_threads=1)
            fs_reused_host = Array(fs_reused)
            fs_fresh_host = Array(fs_fresh)

            for i in 1:n_atoms
                @test isapprox(fs_reused_host[i], fs_fresh_host[i], rtol=1e-8, atol=1e-10)
            end
        end
    else
        @warn "CUDA not functional, skipping GPU consistency tests"
    end
end
