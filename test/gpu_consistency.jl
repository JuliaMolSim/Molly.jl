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

        @testset "GPU remove_CM_motion!" begin
            n_atoms = 8
            T = Float32
            boundary = CubicBoundary(T(10.0)u"nm")
            coords = [SVector(T(i), T(i + 1), T(i + 2)) * u"nm" for i in 1:n_atoms]
            velocities = [SVector(T(0.1 * i), T(-0.2 * i), T(0.05 * (i - 3))) * u"nm/ps"
                          for i in 1:n_atoms]
            atoms = [Atom(
                index=i,
                mass=T(i + 1)u"g/mol",
                charge=T(0.0),
                σ=T(0.3)u"nm",
                ϵ=T(1.0)u"kJ * mol^-1",
            ) for i in 1:n_atoms]

            cpu_sys = System(
                atoms=atoms,
                coords=coords,
                velocities=velocities,
                boundary=boundary,
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )
            gpu_sys = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                velocities=CuArray(velocities),
                boundary=boundary,
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )

            remove_CM_motion!(cpu_sys)
            remove_CM_motion!(gpu_sys)
            gpu_velocities = Array(gpu_sys.velocities)

            for i in 1:n_atoms
                @test isapprox(gpu_velocities[i], cpu_sys.velocities[i], atol=T(1e-6)u"nm/ps")
            end
            cm_velocity = mapreduce((v, m) -> v * m, +, gpu_velocities, mass.(atoms)) / sum(mass.(atoms))
            @test isapprox(cm_velocity, zero(cm_velocity), atol=T(1e-6)u"nm/ps")
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
                neighbor_finder=NoNeighborFinder(),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )
            
            sys_gpu = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                velocities=CuArray(velocities),
                boundary=boundary,
                pairwise_inters=pairwise_inters_gpu,
                neighbor_finder=GPUNeighborFinder(
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
                @test isapprox(f_cpu[i], f_gpu_host[i], rtol=1e-8, atol=1e-10u"kJ * mol^-1 * nm^-1")
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
                neighbor_finder=GPUNeighborFinder(
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
                    UInt64(0),
                    0,
                )
            end

            overflow_force_buffers = with_tiny_tile_capacity(Molly.init_buffers!(sys_gpu_overflow, 1))
            overflow_energy_buffers = with_tiny_tile_capacity(Molly.init_buffers!(sys_gpu_overflow, 1, true))

            @test_throws ErrorException Molly.forces!(
                Molly.zero_forces(sys_gpu_overflow),
                sys_gpu_overflow,
                nothing,
                0,
                overflow_force_buffers,
                Val(false),
            )
            @test_throws ErrorException potential_energy(sys_gpu_overflow, nothing, 0,
                                                         overflow_energy_buffers)
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
            buffers = Molly.init_buffers!(sys, 256)
            fs_reused = Molly.zero_forces(sys)

            Molly.forces!(fs_reused, sys, neighbors, 0, buffers, Val(false); n_threads=1)
            sys.coords .+= sys.velocities .* T(0.5)
            Molly.forces!(fs_reused, sys, neighbors, 1, buffers, Val(false); n_threads=1)

            fs_fresh = forces(sys, neighbors, 1; n_threads=1)
            fs_reused_host = Array(fs_reused)
            fs_fresh_host = Array(fs_fresh)

            for i in 1:n_atoms
                @test isapprox(fs_reused_host[i], fs_fresh_host[i], rtol=1e-8, atol=1e-10)
            end
        end

        @testset "Sparse-pair updates refresh cached GPU tiles" begin
            n_atoms = 2
            D = 3
            T = Float64
            coords = [SVector{D, T}(0.0, 0.0, 0.0), SVector{D, T}(0.45, 0.0, 0.0)]
            boundary = CubicBoundary(T(5.0), T(5.0), T(5.0))
            atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0)) for i in 1:n_atoms]
            r_cut = T(2.0)

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
                energy_units=NoUnits,
            )

            buffers = Molly.init_buffers!(sys, 1, true)
            pe_before = potential_energy(sys, nothing, 0, buffers)

            Molly.append_excluded_pairs!(sys.neighbor_finder, ((1, 2),))

            pe_reused = potential_energy(sys, nothing, 0, buffers)
            pe_fresh = potential_energy(sys, nothing, 0, Molly.init_buffers!(sys, 1, true))

            @test pe_before != pe_reused
            @test isapprox(pe_reused, pe_fresh, rtol=1e-8, atol=1e-10)
        end

        @testset "AlchemicalPartition supports GPUNeighborFinder" begin
            n_atoms = 4
            atom_mass = 10.0u"g/mol"
            σ = 0.3u"nm"
            ϵ = 0.5u"kJ * mol^-1"
            temp = 298.0u"K"
            dt = 0.001u"ps"
            boundary = CubicBoundary(4.0u"nm")
            coords = [
                SVector(0.2, 0.2, 0.2),
                SVector(0.7, 0.2, 0.2),
                SVector(0.2, 0.8, 0.2),
                SVector(0.8, 0.8, 0.2),
            ] .* u"nm"
            atoms_ref = [Atom(index=i, mass=atom_mass, charge=0.0, σ=σ, ϵ=ϵ, λ=1.0) for i in 1:n_atoms]
            atoms_pert = copy(atoms_ref)
            atoms_pert[1] = Atom(index=1, mass=atom_mass, charge=0.0, σ=σ, ϵ=ϵ, λ=0.5)

            function gpu_state(atoms_state)
                sys = System(
                    atoms=CuArray(atoms_state),
                    coords=CuArray(coords),
                    boundary=boundary,
                    pairwise_inters=(LennardJones(use_neighbors=true, cutoff=DistanceCutoff(1.2u"nm")),),
                    neighbor_finder=GPUNeighborFinder(
                        n_atoms=n_atoms,
                        dist_cutoff=1.2u"nm",
                        device_vector_type=CuArray{Int32, 1},
                    ),
                    force_units=u"kJ * mol^-1 * nm^-1",
                    energy_units=u"kJ * mol^-1",
                )
                return ThermoState(sys, VelocityVerlet(dt=dt); temperature=temp)
            end

            thermo_states = [gpu_state(atoms_ref), gpu_state(atoms_pert)]

            for reuse_neighbors in (true, false)
                partition = Molly.AlchemicalPartition(thermo_states; reuse_neighbors=reuse_neighbors)
                energies = Molly.evaluate_energy_all!(
                    partition,
                    thermo_states[1].system.coords,
                    boundary,
                )
                direct = [potential_energy(ts.system) for ts in thermo_states]

                @test length(energies) == length(thermo_states)
                if reuse_neighbors
                    @test partition.λ_sys.neighbor_finder isa DistanceNeighborFinder
                else
                    @test partition.λ_sys.neighbor_finder isa GPUNeighborFinder
                end
                @test isfinite(ustrip(energies[1])) && isfinite(ustrip(energies[2]))
                @test isapprox(energies[1], direct[1], rtol=1e-8, atol=1e-10u"kJ * mol^-1")
                @test isapprox(energies[2], direct[2], rtol=1e-8, atol=1e-10u"kJ * mol^-1")
            end
        end

        @testset "Langevin random numbers FT=$FT" for FT in (Float32, Float64)
            n_atoms = 50_000
            boundary = CubicBoundary(FT(50.0)u"nm")
            atom_mass = FT(10.0)u"g/mol"
            temp = FT(298.0)u"K"
            AT = CuArray
            # Some atoms are virtual sites, which must have their velocities
            #   actively set to zero by random_velocities!.
            vs_inds = [2, 4, 7]
            virtual_sites = [
                OneParticleSite(2, 1, zero(FT)u"nm^-1"),
                TwoParticleAverageSite(4, 3, 5, FT(0.5), FT(0.5)),
                ThreeParticleAverageSite(7, 6, 8, 9, FT(0.2), FT(0.3), FT(0.5)),
            ]
            atoms = [Atom(mass=(i in vs_inds ? FT(0.0) : FT(10.0))u"g/mol")
                     for i in 1:n_atoms]
            coords = [SVector(FT(1.0)u"nm", FT(1.0)u"nm", FT(1.0)u"nm") for _ in 1:n_atoms]
            vel_unit = unit(random_velocity(atom_mass, temp)[1])
            velocities = [zero(SVector{3, FT}) * vel_unit for _ in 1:n_atoms]

            sys = System(
                atoms=copy(atoms),
                coords=copy(coords),
                boundary=boundary,
                velocities=copy(velocities),
                virtual_sites=copy(virtual_sites),
            )

            sys2 = System(
                atoms=AT(atoms),
                coords=AT(coords),
                boundary=boundary,
                velocities=AT(velocities),
                virtual_sites=AT(virtual_sites),
            )
            # Start the output buffers filled with non-zero velocities so we test
            #   that virtual site velocities are actively set to zero, not just
            #   left as they were.
            ones_vels = fill(SVector(FT(1), FT(1), FT(1)) * vel_unit, n_atoms)
            vels_gpu = AT(ones_vels)

            vels_cpu_inplace = copy(ones_vels)
            random_velocities!(vels_cpu_inplace, sys, temp; rng=Xoshiro(10))
            vels_cpu = random_velocities(sys, temp; rng=Xoshiro(10))
            random_velocities!(sys, temp; rng=Xoshiro(10))
            # These should match exactly
            @test vels_cpu_inplace == vels_cpu
            @test sys.velocities == vels_cpu

            random_velocities!(vels_gpu, sys2, temp; rng=Xoshiro(10))
            vels_gpu_cpu = Array(vels_gpu)

            # With the same rng source the velocities should match down to floating point error.
            vel_scale = sqrt(sys.k * temp / atom_mass)
            @test maximum(norm, vels_gpu_cpu .- vels_cpu) < 16 * eps(FT) * vel_scale

            # Virtual site velocities must be actively set to zero on both CPU and GPU.
            for i in vs_inds
                @test iszero(vels_cpu[i])
                @test iszero(vels_gpu_cpu[i])
            end
        end

        @testset "langevin_o_step! FT=$FT" for FT in (Float32, Float64)
            n_atoms = 50_000
            AT = CuArray

            rng = Xoshiro(15)
            init_vels = [randn(rng, SVector{3, FT}) for _ in 1:n_atoms]

            vel_scales = fill(FT(0.8), n_atoms)
            noise_scales = fill(FT(0.3), n_atoms)

            philox_key = 0x1234567890abcdef
            philox_ctr1 = 0xfedcba0987654321

            vels_cpu = copy(init_vels)
            vels_gpu = AT(init_vels)

            Molly.langevin_o_step!(vels_cpu, vel_scales, noise_scales,
                                   philox_ctr1, philox_key, FT)
            Molly.langevin_o_step!(vels_gpu, AT(vel_scales), AT(noise_scales),
                                   philox_ctr1, philox_key, FT)
            vels_gpu_cpu = Array(vels_gpu)

            @test vels_cpu != init_vels
            @test maximum(norm, vels_gpu_cpu .- vels_cpu) < 16 * eps(FT)
        end
    else
        @warn "CUDA not functional, skipping GPU consistency tests"
    end
end
