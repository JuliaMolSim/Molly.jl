@testset "Peptide" begin
    n_steps = 100
    temp = 298.0u"K"
    s = System(
        joinpath(data_dir, "5XER", "gmx_coords.gro"),
        joinpath(data_dir, "5XER", "gmx_top_ff.top");
        loggers=(
            temp=TemperatureLogger(10),
            coords=CoordinatesLogger(10),
            energy=TotalEnergyLogger(10),
            writer=StructureWriter(10, temp_fp_pdb),
            density=DensityLogger(10),
        ),
        data="test_data_peptide",
    )
    simulator = VelocityVerlet(dt=0.0002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))

    true_n_atoms = 5191
    @test length(s.atoms) == true_n_atoms
    @test length(s.coords) == true_n_atoms
    @test size(s.neighbor_finder.eligible) == (true_n_atoms, true_n_atoms)
    @test size(s.neighbor_finder.special) == (true_n_atoms, true_n_atoms)
    @test length(s.pairwise_inters) == 2
    @test length(s.specific_inter_lists) == 3
    @test s.boundary == CubicBoundary(3.7146u"nm")
    show(devnull, first(s.atoms))
    @test s.data == "test_data_peptide"

    @test length(s.topology.atom_molecule_inds) == length(s) == 5191
    @test s.topology.atom_molecule_inds[10] == 1
    @test length(s.topology.molecule_atom_counts) == 1678
    @test s.topology.molecule_atom_counts[1] == 164

    s.velocities = [random_velocity(mass(a), temp) .* 0.01 for a in s.atoms]
    @time simulate!(s, simulator, n_steps; n_threads=1)

    @test all(isapprox(1016.0870493u"kg * m^-3"), values(s.loggers.density))
    traj = read(temp_fp_pdb, BioStructures.PDBFormat)
    rm(temp_fp_pdb)
    @test BioStructures.countmodels(traj) == 11
    @test BioStructures.countatoms(first(traj)) == 5191
end

@testset "Peptide Float32 no units" begin
    n_steps = 1_000
    temp = 298.0f0
    press = Float32(ustrip(u"u * nm^-1 * ps^-2", 1.0f0u"bar"))
    s = System(
        Float32,
        joinpath(data_dir, "5XER", "gmx_coords.gro"),
        joinpath(data_dir, "5XER", "gmx_top_ff.top");
        loggers=(
            temp=TemperatureLogger(Float32, 10),
            coords=CoordinatesLogger(Float32, 10),
            energy=TotalEnergyLogger(Float32, 10),
        ),
        units=false,
    )
    thermostat = AndersenThermostat(temp, 10.0f0)
    barostat = MonteCarloBarostat(press, temp, s.boundary; n_steps=20)
    simulator = VelocityVerlet(dt=0.0002f0, coupling=(thermostat, barostat))

    s.velocities = [random_velocity(mass(a), temp) .* 0.01f0 for a in s.atoms]
    simulate!(deepcopy(s), simulator, 100; n_threads=1)
    @time simulate!(s, simulator, n_steps; n_threads=1)
    @test s.boundary != CubicBoundary(3.7146f0)
    @test 3.6f0 < s.boundary.side_lengths[1] < 3.8f0
end

@testset "OpenMM protein comparison" begin
    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...)
    show(devnull, ff)
    sys = System(joinpath(data_dir, "6mrr_equil.pdb"), ff; center_coords=false)
    neighbors = find_neighbors(sys)

    @test count(i -> is_any_atom(  sys.atoms[i], sys.atoms_data[i]), eachindex(sys)) == 15954
    @test count(i -> is_heavy_atom(sys.atoms[i], sys.atoms_data[i]), eachindex(sys)) == 5502
    @test length(sys.topology.atom_molecule_inds) == length(sys) == 15954
    @test sys.topology.atom_molecule_inds[10] == 1
    @test length(sys.topology.molecule_atom_counts) == 4929
    @test sys.topology.molecule_atom_counts[1] == 1170

    for inter in ("bond", "angle", "proptor", "improptor", "lj", "coul", "all")
        if inter == "all"
            pin = sys.pairwise_inters
        elseif inter == "lj"
            pin = sys.pairwise_inters[1:1]
        elseif inter == "coul"
            pin = sys.pairwise_inters[2:2]
        else
            pin = ()
        end

        if inter == "all"
            sils = sys.specific_inter_lists
        elseif inter == "bond"
            sils = sys.specific_inter_lists[1:1]
        elseif inter == "angle"
            sils = sys.specific_inter_lists[2:2]
        elseif inter == "proptor"
            sils = sys.specific_inter_lists[3:3]
        elseif inter == "improptor"
            sils = sys.specific_inter_lists[4:4]
        else
            sils = ()
        end

        sys_part = System(
            atoms=sys.atoms,
            coords=sys.coords,
            boundary=sys.boundary,
            pairwise_inters=pin,
            specific_inter_lists=sils,
            neighbor_finder=sys.neighbor_finder,
        )

        forces_molly = forces(sys_part, neighbors; n_threads=1)
        forces_openmm = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "forces_$(inter)_only.txt"))))u"kJ * mol^-1 * nm^-1"
        # All force terms on all atoms must match at some threshold
        @test !any(d -> any(abs.(d) .> 1e-6u"kJ * mol^-1 * nm^-1"), forces_molly .- forces_openmm)

        E_molly = potential_energy(sys_part, neighbors)
        E_openmm = readdlm(joinpath(openmm_dir, "energy_$(inter)_only.txt"))[1] * u"kJ * mol^-1"
        # Energy must match at some threshold
        @test E_molly - E_openmm < 1e-5u"kJ * mol^-1"
    end

    # Run a short simulation with all interactions
    n_steps = 100
    simulator = VelocityVerlet(dt=0.0005u"ps")
    velocities_start = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))u"nm * ps^-1"
    sys.velocities = copy(velocities_start)
    @test kinetic_energy(sys) ≈ 65521.87288132431u"kJ * mol^-1"
    @test temperature(sys) ≈ 329.3202932884933u"K"

    simulate!(sys, simulator, n_steps; n_threads=Threads.nthreads())

    coords_openmm = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "coordinates_$(n_steps)steps.txt"))))u"nm"
    vels_openmm   = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_$(n_steps)steps.txt" ))))u"nm * ps^-1"

    coords_diff = sys.coords .- wrap_coords.(coords_openmm, (sys.boundary,))
    vels_diff = sys.velocities .- vels_openmm
    # Coordinates and velocities at end must match at some threshold
    @test maximum(maximum(abs.(v)) for v in coords_diff) < 1e-9u"nm"
    @test maximum(maximum(abs.(v)) for v in vels_diff  ) < 1e-6u"nm * ps^-1"

    # Test with no units
    ff_nounits = MolecularForceField(
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...;
        units=false,
    )
    sys_nounits = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff_nounits;
        velocities=copy(ustrip_vec.(velocities_start)),
        units=false,
        center_coords=false,
    )
    simulator_nounits = VelocityVerlet(dt=0.0005)
    @test kinetic_energy(sys_nounits)u"kJ * mol^-1" ≈ 65521.87288132431u"kJ * mol^-1"
    @test temperature(sys_nounits)u"K" ≈ 329.3202932884933u"K"

    E_openmm = readdlm(joinpath(openmm_dir, "energy_all_only.txt"))[1] * u"kJ * mol^-1"
    neighbors_nounits = find_neighbors(sys_nounits)
    @test isapprox(potential_energy(sys_nounits, neighbors_nounits) * u"kJ * mol^-1",
                    E_openmm; atol=1e-5u"kJ * mol^-1")

    simulate!(sys_nounits, simulator_nounits, n_steps; n_threads=Threads.nthreads())

    coords_diff = sys_nounits.coords * u"nm" .- wrap_coords.(coords_openmm, (sys.boundary,))
    vels_diff = sys_nounits.velocities * u"nm * ps^-1" .- vels_openmm
    @test maximum(maximum(abs.(v)) for v in coords_diff) < 1e-9u"nm"
    @test maximum(maximum(abs.(v)) for v in vels_diff  ) < 1e-6u"nm * ps^-1"

    params_dic = extract_parameters(sys_nounits, ff_nounits)
    @test length(params_dic) == 638
    atoms_grad, pis_grad, sis_grad, gis_grad = inject_gradients(sys_nounits, params_dic)
    @test atoms_grad == sys_nounits.atoms
    @test pis_grad == sys_nounits.pairwise_inters

    # Test the same simulation on the GPU
    if run_gpu_tests
        sys = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            velocities=CuArray(copy(velocities_start)),
            gpu=true,
            center_coords=false,
        )
        @test kinetic_energy(sys) ≈ 65521.87288132431u"kJ * mol^-1"
        @test temperature(sys) ≈ 329.3202932884933u"K"

        neighbors = find_neighbors(sys)
        @test isapprox(potential_energy(sys, neighbors), E_openmm; atol=1e-5u"kJ * mol^-1")

        simulate!(sys, simulator, n_steps)

        coords_diff = Array(sys.coords) .- wrap_coords.(coords_openmm, (sys.boundary,))
        vels_diff = Array(sys.velocities) .- vels_openmm
        @test maximum(maximum(abs.(v)) for v in coords_diff) < 1e-9u"nm"
        @test maximum(maximum(abs.(v)) for v in vels_diff  ) < 1e-6u"nm * ps^-1"

        # Test Andersen thermostat on GPU
        simulator_and = Verlet(
            dt=0.0005u"ps",
            coupling=AndersenThermostat(321.0u"K", 10.0u"ps"),
        )
        simulate!(sys, simulator_and, n_steps)
        @test temperature(sys) > 400.0u"K"

        sys_nounits = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff_nounits;
            velocities=CuArray(copy(ustrip_vec.(velocities_start))),
            units=false,
            gpu=true,
            center_coords=false,
        )
        @test kinetic_energy(sys_nounits)u"kJ * mol^-1" ≈ 65521.87288132431u"kJ * mol^-1"
        @test temperature(sys_nounits)u"K" ≈ 329.3202932884933u"K"

        neighbors_nounits = find_neighbors(sys_nounits)
        @test isapprox(potential_energy(sys_nounits, neighbors_nounits) * u"kJ * mol^-1",
                        E_openmm; atol=1e-5u"kJ * mol^-1")

        simulate!(sys_nounits, simulator_nounits, n_steps)

        coords_diff = Array(sys_nounits.coords * u"nm") .- wrap_coords.(coords_openmm, (sys.boundary,))
        vels_diff = Array(sys_nounits.velocities * u"nm * ps^-1") .- vels_openmm
        @test maximum(maximum(abs.(v)) for v in coords_diff) < 1e-9u"nm"
        @test maximum(maximum(abs.(v)) for v in vels_diff  ) < 1e-6u"nm * ps^-1"

        simulator_and_nounits = Verlet(
            dt=0.0005,
            coupling=AndersenThermostat(321.0, 10.0),
        )
        simulate!(sys_nounits, simulator_and_nounits, n_steps)
        @test temperature(sys_nounits) > 400.0

        params_dic_gpu = extract_parameters(sys_nounits, ff_nounits)
        @test params_dic == params_dic_gpu
        atoms_grad, pis_grad, sis_grad, gis_grad = inject_gradients(sys_nounits, params_dic_gpu)
        @test atoms_grad == sys_nounits.atoms
        @test pis_grad == sys_nounits.pairwise_inters
    end
end

@testset "Implicit solvent" begin
    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "his.xml"])...)

    for gpu in gpu_list
        for solvent_model in ("obc2", "gbn2")
            sys = System(
                joinpath(data_dir, "6mrr_nowater.pdb"),
                ff;
                boundary=CubicBoundary(100.0u"nm"),
                gpu=gpu,
                dist_cutoff=5.0u"nm",
                dist_neighbors=5.0u"nm",
                implicit_solvent=solvent_model,
                kappa=1.0u"nm^-1",
            )
            neighbors = find_neighbors(sys)
            forces_molly = forces(sys)
            @test !any(d -> any(abs.(d) .> 1e-6u"kJ * mol^-1 * nm^-1"),
                        Array(forces_molly) .- Array(forces(sys, neighbors)))
            openmm_force_fp = joinpath(openmm_dir, "forces_$solvent_model.txt")
            forces_openmm = SVector{3}.(eachrow(readdlm(openmm_force_fp)))u"kJ * mol^-1 * nm^-1"
            @test !any(d -> any(abs.(d) .> 1e-3u"kJ * mol^-1 * nm^-1"),
                        Array(forces_molly) .- forces_openmm)

            E_molly = potential_energy(sys)
            @test E_molly ≈ potential_energy(sys, neighbors)
            openmm_E_fp = joinpath(openmm_dir, "energy_$solvent_model.txt")
            E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
            @test E_molly - E_openmm < 1e-2u"kJ * mol^-1"

            if solvent_model == "gbn2"
                sim = SteepestDescentMinimizer(tol=400.0u"kJ * mol^-1 * nm^-1")
                coords_start = copy(sys.coords)
                simulate!(sys, sim)
                @test potential_energy(sys) < E_molly
                @test rmsd(coords_start, sys.coords) < 0.1u"nm"
            end
        end
    end
end
