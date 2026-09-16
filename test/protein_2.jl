@testset "CHARMM OpenMM protein comparison" begin
    pme_mesh_dims = (46, 46, 51)
    ff = MolecularForceField(
        joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...;
        strictness=:nowarn,
    )
    show(devnull, ff)
    @test_throws ForceFieldXMLError MolecularForceField(
        joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...;
        strictness=:error,
    )
    ff_nounits = MolecularForceField(
        joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...;
        units=false,
        strictness=:nowarn,
    )
    start_temp = 485.281907022u"K" # High since it does not take into account constraints

    for constraint_algorithm in (SetupLINCS(), SetupSHAKE_RATTLE())
        sys = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            nonbonded_method=SetupPME(mesh_dims=pme_mesh_dims),
            center_coords=false,
            constraints=:hbonds,
            rigid_water=true,
            constraint_algorithm=constraint_algorithm,
            n_threads=1,
        )
        neighbors = find_neighbors(sys)
        @test length(sys.specific_inter_lists) == 7
        @test length(sys.specific_inter_lists[1]) == 1691
        @test length(sys.specific_inter_lists[2]) == 2137
        show(devnull, sys)
        for sil in sys.specific_inter_lists
            show(devnull, sil)
        end

        constrained_inds = Molly.constrained_atom_inds(sys)
        @test length(constrained_inds) == 15747
        @test count(i -> i <= 1170, constrained_inds) == 963
        constrained_pairs = Molly.constrained_atom_pairs(sys)
        @test length(constrained_pairs) == 15380
        @test count(p -> (p[1] <= 1170 && p[2] <= 1170), constrained_pairs) == 596

        bench_result = @benchmark potential_energy($sys, $neighbors; n_threads=1) samples=5 evals=1
        @test bench_result.allocs <= 16
        @test bench_result.memory <= 1000
        forces_t = Molly.zero_forces(sys)
        buffers = Molly.init_buffers!(sys, 1)
        bench_result = @benchmark Molly.forces!($forces_t, $sys, $neighbors, 0, $buffers, Val(false);
                                                n_threads=1) samples=5 evals=1
        @test bench_result.allocs <= 15
        @test bench_result.memory <= 1100

        # Use all threads
        sys = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            nonbonded_method=SetupPME(mesh_dims=pme_mesh_dims),
            center_coords=false,
            constraints=:hbonds,
            rigid_water=true,
            constraint_algorithm=constraint_algorithm,
        )
        scalar_vir = scalar_virial(sys)
        @test scalar_vir ≈ tr(virial(sys))
        @test scalar_vir ≈ scalar_virial(sys; n_threads=1)
        pressure_t = pressure(sys)
        scalar_P = scalar_pressure(sys)
        @test all(isfinite, pressure_t)
        @test scalar_P ≈ tr(pressure_t) / 3
        @test scalar_pressure(sys; n_threads=1) ≈ tr(pressure_t) / 3

        forces_molly = forces(sys, neighbors; n_threads=1)
        openmm_forces_fp = joinpath(openmm_dir, "charmm", "forces.txt")
        forces_openmm = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(forces_molly .- forces_openmm)) < 1e-3u"kJ * mol^-1 * nm^-1"

        E_molly = potential_energy(sys, neighbors)
        openmm_E_fp = joinpath(openmm_dir, "charmm", "energy.txt")
        E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
        @test abs(E_molly - E_openmm) < 0.2u"kJ * mol^-1"

        # Run a short simulation with all interactions
        n_steps = 100
        simulator = VelocityVerlet(dt=0.0005u"ps")
        start_vels_fp = joinpath(openmm_dir, "velocities_300K.txt")
        velocities_start = SVector{3}.(eachrow(readdlm(start_vels_fp)))u"nm * ps^-1"
        sys.velocities = copy(velocities_start)
        @test kinetic_energy(sys) ≈ 65524.08096011398u"kJ * mol^-1"
        @test total_energy(sys) ≈ -71600.777370857u"kJ * mol^-1"
        @test temperature(sys) ≈ start_temp

        simulate!(sys, simulator, n_steps; n_threads=Threads.nthreads())

        openmm_coords_fp = joinpath(openmm_dir, "charmm", "coordinates_$(n_steps)steps.txt")
        openmm_vels_fp   = joinpath(openmm_dir, "charmm", "velocities_$(n_steps)steps.txt" )
        coords_openmm = SVector{3}.(eachrow(readdlm(openmm_coords_fp)))u"nm"
        vels_openmm   = SVector{3}.(eachrow(readdlm(openmm_vels_fp)))u"nm * ps^-1"

        coords_diff = sys.coords .- wrap_coords.(coords_openmm, (sys.boundary,))
        vels_diff = sys.velocities .- vels_openmm
        @test maximum(norm.(coords_diff)) < 5e-4u"nm"
        @test maximum(norm.(vels_diff  )) < 0.5u"nm * ps^-1"

        # Test with no units
        sys_nounits = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff_nounits;
            velocities=copy(ustrip_vec.(velocities_start)),
            units=false,
            nonbonded_method=SetupPME(mesh_dims=pme_mesh_dims),
            center_coords=false,
            constraints=:hbonds,
            rigid_water=true,
            constraint_algorithm=constraint_algorithm,
        )
        show(devnull, sys_nounits)
        simulator_nounits = VelocityVerlet(dt=0.0005)
        @test kinetic_energy(sys_nounits)u"kJ * mol^-1" ≈ 65524.08096011398u"kJ * mol^-1"
        @test temperature(sys_nounits)u"K" ≈ start_temp
        @test scalar_virial(sys_nounits) ≈ tr(virial(sys_nounits))
        pressure_nounits = pressure(sys_nounits)
        @test all(isfinite, pressure_nounits)
        @test scalar_pressure(sys_nounits) ≈ tr(pressure_nounits) / 3

        neighbors_nounits = find_neighbors(sys_nounits)
        @test isapprox(potential_energy(sys_nounits, neighbors_nounits) * u"kJ * mol^-1",
                        E_openmm; atol=0.2u"kJ * mol^-1")

        simulate!(sys_nounits, simulator_nounits, n_steps; n_threads=Threads.nthreads())

        coords_diff = sys_nounits.coords * u"nm" .- wrap_coords.(coords_openmm, (sys.boundary,))
        vels_diff = sys_nounits.velocities * u"nm * ps^-1" .- vels_openmm
        @test maximum(norm.(coords_diff)) < 5e-4u"nm"
        @test maximum(norm.(vels_diff  )) < 0.5u"nm * ps^-1"

        params_dic = Molly.extract_parameters(sys_nounits, ff_nounits)
        sys_grad = inject_gradients(sys_nounits, params_dic)
        @test sys_grad.atoms == sys_nounits.atoms
        @test sys_grad.pairwise_inters == sys_nounits.pairwise_inters
        @test sys_grad.specific_inter_lists == sys_nounits.specific_inter_lists

        # Test the same simulation on the GPU
        for AT in array_list[2:end]
            sys = System(
                joinpath(data_dir, "6mrr_equil.pdb"),
                ff;
                array_type=AT,
                float_type=Float64,
                nonbonded_method=SetupPME(mesh_dims=pme_mesh_dims),
                center_coords=false,
                constraints=:hbonds,
                rigid_water=true,
                constraint_algorithm=constraint_algorithm,
            )
            show(devnull, sys)
            @test scalar_virial(sys) ≈ scalar_vir
            @test scalar_pressure(sys) ≈ scalar_P
            sys.velocities = to_device(copy(velocities_start), AT)
            @test kinetic_energy(sys) ≈ 65524.08096011398u"kJ * mol^-1"
            @test temperature(sys) ≈ start_temp

            neighbors = find_neighbors(sys)
            @test maximum(norm.(from_device(forces(sys, neighbors)) .- forces_openmm)) < 1e-3u"kJ * mol^-1 * nm^-1"
            @test isapprox(potential_energy(sys, neighbors), E_openmm; atol=0.2u"kJ * mol^-1")

            simulate!(sys, simulator, n_steps)

            coords_diff = from_device(sys.coords) .-
                                        wrap_coords.(coords_openmm, (sys.boundary,))
            vels_diff = from_device(sys.velocities) .- vels_openmm
            @test maximum(norm.(coords_diff)) < 5e-4u"nm"
            @test maximum(norm.(vels_diff  )) < 0.5u"nm * ps^-1"

            sys_nounits = System(
                joinpath(data_dir, "6mrr_equil.pdb"),
                ff_nounits;
                velocities=to_device(copy(ustrip_vec.(velocities_start)), AT),
                units=false,
                array_type=AT,
                float_type=Float64,
                nonbonded_method=SetupPME(mesh_dims=pme_mesh_dims),
                center_coords=false,
                constraints=:hbonds,
                rigid_water=true,
                constraint_algorithm=constraint_algorithm,
            )
            @test kinetic_energy(sys_nounits)u"kJ * mol^-1" ≈ 65524.08096011398u"kJ * mol^-1"
            @test temperature(sys_nounits)u"K" ≈ start_temp

            neighbors_nounits = find_neighbors(sys_nounits)
            forces_molly = from_device(forces(sys_nounits, neighbors)u"kJ * mol^-1 * nm^-1")
            @test maximum(norm.(forces_molly .- forces_openmm)) < 1e-3u"kJ * mol^-1 * nm^-1"
            @test isapprox(potential_energy(sys_nounits, neighbors_nounits) * u"kJ * mol^-1",
                        E_openmm; atol=0.2u"kJ * mol^-1")

            simulate!(sys_nounits, simulator_nounits, n_steps)

            coords_diff = from_device(sys_nounits.coords * u"nm") .-
                                        wrap_coords.(coords_openmm, (sys.boundary,))
            vels_diff = from_device(sys_nounits.velocities * u"nm * ps^-1") .- vels_openmm
            @test maximum(norm.(coords_diff)) < 5e-4u"nm"
            @test maximum(norm.(vels_diff  )) < 0.5u"nm * ps^-1"

            params_dic_gpu = Molly.extract_parameters(sys_nounits, ff_nounits)
            @test params_dic == params_dic_gpu
            sys_grad = inject_gradients(sys_nounits, params_dic_gpu)
            @test sys_grad.atoms == sys_nounits.atoms
            @test sys_grad.pairwise_inters == sys_nounits.pairwise_inters
            @test sys_grad.specific_inter_lists == sys_nounits.specific_inter_lists
        end
    end
end
