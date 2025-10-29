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
            dcd_writer=TrajectoryWriter(10, temp_fp_dcd; atom_inds=1001:2000),
            pdb_writer=TrajectoryWriter(10, temp_fp_pdb),
            density=DensityLogger(10),
        ),
        nonbonded_method=:cutoff,
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
    traj = Chemfiles.Trajectory(temp_fp_dcd)
    rm(temp_fp_dcd)
    @test Int(length(traj)) == 11
    frame = read(traj)
    @test length(frame) == 1000
    @test size(Chemfiles.positions(frame)) == (3, 1000)
    @test Chemfiles.lengths(Chemfiles.UnitCell(frame)) == [37.146, 37.146, 37.146]
    boundary = Molly.boundary_from_chemfiles(Chemfiles.UnitCell(frame))
    @test boundary.side_lengths ≈ SVector(3.7146, 3.7146, 3.7146)u"nm"

    @test readlines(temp_fp_pdb)[1] == "CRYST1   37.146   37.146   37.146  90.00  90.00  90.00 P 1           1"
    traj = read(temp_fp_pdb, BioStructures.PDBFormat)
    rm(temp_fp_pdb)
    @test BioStructures.countmodels(traj) == 11
    @test BioStructures.countatoms(first(traj)) == 5191

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
        nonbonded_method=:cutoff,
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
    sys = System(joinpath(data_dir, "6mrr_equil.pdb"), ff;
                 nonbonded_method=:cutoff, center_coords=false)
    sys_pme = System(joinpath(data_dir, "6mrr_equil.pdb"), ff;
                     nonbonded_method=:pme, center_coords=false)
    sys_pme_exact = System(joinpath(data_dir, "6mrr_equil.pdb"), ff;
                           nonbonded_method=:pme, approximate_pme=false, center_coords=false)
    zero(sys)
    zero(sys_pme)
    neighbors = find_neighbors(sys)

    @test count(i -> is_any_atom(  sys.atoms[i], sys.atoms_data[i]), eachindex(sys)) == 15954
    @test count(i -> is_heavy_atom(sys.atoms[i], sys.atoms_data[i]), eachindex(sys)) == 5502
    @test length(sys.topology.atom_molecule_inds) == length(sys) == 15954
    @test sys.topology.atom_molecule_inds[10] == 1
    @test length(sys.topology.molecule_atom_counts) == 4929
    @test sys.topology.molecule_atom_counts[1] == 1170

    bench_result = @benchmark potential_energy($sys, $neighbors; n_threads=1)
    @test bench_result.allocs <= 6
    @test bench_result.memory <= 192
    forces_t = Molly.zero_forces(sys)
    buffers = Molly.init_buffers!(sys, 1)
    bench_result = @benchmark Molly.forces!($forces_t, $sys, $neighbors, $buffers, Val(false);
                                            n_threads=1)
    @test bench_result.allocs <= 3
    @test bench_result.memory <= 144

    scalar_vir = scalar_virial(sys_pme)
    scalar_P = scalar_pressure(sys_pme)
    @test scalar_vir ≈ tr(virial(sys_pme))
    @test scalar_P ≈ tr(pressure(sys_pme)) / 3
    @test scalar_vir ≈ scalar_virial(sys_pme; n_threads=1)
    @test scalar_P ≈ scalar_pressure(sys_pme; n_threads=1)

    inters = (
        "bond_only", "angle_only", "proptor_only", "improptor_only", "lj_only", "coul_only",
        "all_cut", "all_pme", "all_pme_exact",
    )
    for inter in inters
        if inter == "all_cut"
            pin = sys.pairwise_inters
        elseif inter == "all_pme"
            pin = sys_pme.pairwise_inters
        elseif inter == "all_pme_exact"
            pin = sys_pme_exact.pairwise_inters
        elseif inter == "lj_only"
            pin = sys.pairwise_inters[1:1]
        elseif inter == "coul_only"
            pin = sys.pairwise_inters[2:2]
        else
            pin = ()
        end

        if startswith(inter, "all")
            sils = sys.specific_inter_lists
        elseif inter == "bond_only"
            sils = sys.specific_inter_lists[1:1]
        elseif inter == "angle_only"
            sils = sys.specific_inter_lists[2:2]
        elseif inter == "proptor_only"
            sils = sys.specific_inter_lists[3:3]
        elseif inter == "improptor_only"
            sils = sys.specific_inter_lists[4:4]
        else
            sils = ()
        end

        if inter == "all_pme"
            gis = sys_pme.general_inters
        elseif inter == "all_pme_exact"
            gis = sys_pme_exact.general_inters
        else
            gis = ()
        end

        sys_part = System(
            atoms=sys.atoms,
            coords=sys.coords,
            boundary=sys.boundary,
            pairwise_inters=pin,
            specific_inter_lists=sils,
            general_inters=gis,
            neighbor_finder=sys.neighbor_finder,
        )

        forces_molly = forces(sys_part, neighbors; n_threads=1)
        openmm_forces_fp = joinpath(openmm_dir, "forces_$inter.txt")
        forces_openmm = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        # All forces must match at some threshold
        ftol = (inter == "all_pme" ? 1e-3 : 1e-7)u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(forces_molly .- forces_openmm)) < ftol

        E_molly = potential_energy(sys_part, neighbors)
        openmm_E_fp = joinpath(openmm_dir, "energy_$inter.txt")
        E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
        # Energy must match at some threshold
        etol = (inter == "all_pme" ? 0.2 : 1e-5)u"kJ * mol^-1"
        @test abs(E_molly - E_openmm) < etol
    end

    # Run a short simulation with all interactions
    n_steps = 100
    simulator = VelocityVerlet(dt=0.0005u"ps")
    start_vels_fp = joinpath(openmm_dir, "velocities_300K.txt")
    velocities_start = SVector{3}.(eachrow(readdlm(start_vels_fp)))u"nm * ps^-1"
    sys_pme_exact.velocities = copy(velocities_start)
    @test kinetic_energy(sys_pme_exact) ≈ 65521.87288132431u"kJ * mol^-1"
    @test temperature(sys_pme_exact) ≈ 329.3202932884933u"K"

    simulate!(sys_pme_exact, simulator, n_steps; n_threads=Threads.nthreads())

    openmm_coords_fp = joinpath(openmm_dir, "coordinates_$(n_steps)steps.txt")
    openmm_vels_fp   = joinpath(openmm_dir, "velocities_$(n_steps)steps.txt" )
    coords_openmm = SVector{3}.(eachrow(readdlm(openmm_coords_fp)))u"nm"
    vels_openmm   = SVector{3}.(eachrow(readdlm(openmm_vels_fp)))u"nm * ps^-1"

    coords_diff = sys_pme_exact.coords .- wrap_coords.(coords_openmm, (sys_pme_exact.boundary,))
    vels_diff = sys_pme_exact.velocities .- vels_openmm
    # Coordinates and velocities at end must match at some threshold
    @test maximum(norm.(coords_diff)) < 1e-10u"nm"
    @test maximum(norm.(vels_diff  )) < 1e-7u"nm * ps^-1"

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
        nonbonded_method=:pme,
        approximate_pme=false,
        center_coords=false,
    )
    zero(sys_nounits)
    simulator_nounits = VelocityVerlet(dt=0.0005)
    @test kinetic_energy(sys_nounits)u"kJ * mol^-1" ≈ 65521.87288132431u"kJ * mol^-1"
    @test temperature(sys_nounits)u"K" ≈ 329.3202932884933u"K"
    @test scalar_virial(sys_nounits) ≈ tr(virial(sys_nounits))
    @test scalar_pressure(sys_nounits) ≈ tr(pressure(sys_nounits)) / 3

    E_openmm_pme = readdlm(joinpath(openmm_dir, "energy_all_pme_exact.txt"))[1] * u"kJ * mol^-1"
    neighbors_nounits = find_neighbors(sys_nounits)
    @test isapprox(potential_energy(sys_nounits, neighbors_nounits) * u"kJ * mol^-1",
                    E_openmm_pme; atol=1e-5u"kJ * mol^-1")

    simulate!(sys_nounits, simulator_nounits, n_steps; n_threads=Threads.nthreads())

    coords_diff = sys_nounits.coords * u"nm" .- wrap_coords.(coords_openmm, (sys.boundary,))
    vels_diff = sys_nounits.velocities * u"nm * ps^-1" .- vels_openmm
    @test maximum(norm.(coords_diff)) < 1e-10u"nm"
    @test maximum(norm.(vels_diff  )) < 1e-7u"nm * ps^-1"

    params_dic = Molly.extract_parameters(sys_nounits, ff_nounits)
    @test length(params_dic) == 637
    sys_nounits_nogi = System(sys_nounits; general_inters=())
    atoms_grad, pis_grad, sis_grad, gis_grad = Molly.inject_gradients(sys_nounits_nogi, params_dic)
    @test atoms_grad == sys_nounits.atoms
    @test pis_grad == sys_nounits.pairwise_inters

    # Test the same simulation on the GPU
    for AT in array_list[2:end]
        sys = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            velocities=to_device(copy(velocities_start), AT),
            array_type=AT,
            nonbonded_method=:cutoff,
            center_coords=false,
        )
        zero(sys)
        @test kinetic_energy(sys) ≈ 65521.87288132431u"kJ * mol^-1"
        @test temperature(sys) ≈ 329.3202932884933u"K"

        neighbors = find_neighbors(sys)
        openmm_forces_fp = joinpath(openmm_dir, "forces_all_cut.txt")
        forces_openmm = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(from_device(forces(sys, neighbors)) .- forces_openmm)) < 1e-7u"kJ * mol^-1 * nm^-1"
        E_openmm = readdlm(joinpath(openmm_dir, "energy_all_cut.txt"))[1] * u"kJ * mol^-1"
        @test isapprox(potential_energy(sys, neighbors), E_openmm; atol=1e-5u"kJ * mol^-1")

        sys_pme = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            velocities=to_device(copy(velocities_start), AT),
            array_type=AT,
            nonbonded_method=:pme,
            center_coords=false,
        )
        zero(sys_pme)

        neighbors = find_neighbors(sys_pme)
        openmm_forces_fp = joinpath(openmm_dir, "forces_all_pme.txt")
        forces_openmm_pme = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(from_device(forces(sys_pme, neighbors)) .- forces_openmm_pme)) < 1e-3u"kJ * mol^-1 * nm^-1"
        E_openmm_pme = readdlm(joinpath(openmm_dir, "energy_all_pme.txt"))[1] * u"kJ * mol^-1"
        @test isapprox(potential_energy(sys_pme, neighbors), E_openmm_pme; atol=0.2u"kJ * mol^-1")
        sys_pme.velocities .= (zero(SVector{3, Float64}) * u"nm * ps^-1",)
        @test scalar_virial(sys_pme) ≈ scalar_vir
        @test scalar_pressure(sys_pme) ≈ scalar_P

        sys_pme_exact = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            velocities=to_device(copy(velocities_start), AT),
            array_type=AT,
            nonbonded_method=:pme,
            approximate_pme=false,
            center_coords=false,
        )

        neighbors = find_neighbors(sys_pme_exact)
        openmm_forces_fp = joinpath(openmm_dir, "forces_all_pme_exact.txt")
        forces_openmm_pme = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(from_device(forces(sys_pme_exact, neighbors)) .- forces_openmm_pme)) < 1e-7u"kJ * mol^-1 * nm^-1"
        E_openmm_pme = readdlm(joinpath(openmm_dir, "energy_all_pme_exact.txt"))[1] * u"kJ * mol^-1"
        @test isapprox(potential_energy(sys_pme_exact, neighbors),
                       E_openmm_pme; atol=1e-5u"kJ * mol^-1")

        simulate!(sys_pme_exact, simulator, n_steps)

        coords_diff = from_device(sys_pme_exact.coords) .-
                                    wrap_coords.(coords_openmm, (sys_pme_exact.boundary,))
        vels_diff = from_device(sys_pme_exact.velocities) .- vels_openmm
        @test maximum(norm.(coords_diff)) < 1e-10u"nm"
        @test maximum(norm.(vels_diff  )) < 1e-7u"nm * ps^-1"

        # Test Andersen thermostat on GPU
        simulator_and = Verlet(
            dt=0.0005u"ps",
            coupling=AndersenThermostat(321.0u"K", 10.0u"ps"),
        )
        simulate!(sys_pme_exact, simulator_and, n_steps)
        @test temperature(sys_pme_exact) > 400.0u"K"

        sys_nounits = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff_nounits;
            velocities=to_device(copy(ustrip_vec.(velocities_start)), AT),
            units=false,
            array_type=AT,
            nonbonded_method=:pme,
            approximate_pme=false,
            center_coords=false,
        )
        zero(sys_nounits)
        @test kinetic_energy(sys_nounits)u"kJ * mol^-1" ≈ 65521.87288132431u"kJ * mol^-1"
        @test temperature(sys_nounits)u"K" ≈ 329.3202932884933u"K"

        neighbors_nounits = find_neighbors(sys_nounits)
        openmm_forces_fp = joinpath(openmm_dir, "forces_all_pme_exact.txt")
        forces_molly = from_device(forces(sys_nounits, neighbors)u"kJ * mol^-1 * nm^-1")
        @test maximum(norm.(forces_molly .- forces_openmm_pme)) < 1e-7u"kJ * mol^-1 * nm^-1"
        @test isapprox(potential_energy(sys_nounits, neighbors_nounits) * u"kJ * mol^-1",
                       E_openmm_pme; atol=1e-5u"kJ * mol^-1")

        simulate!(sys_nounits, simulator_nounits, n_steps)

        coords_diff = from_device(sys_nounits.coords * u"nm") .-
                                    wrap_coords.(coords_openmm, (sys.boundary,))
        vels_diff = from_device(sys_nounits.velocities * u"nm * ps^-1") .- vels_openmm
        @test maximum(norm.(coords_diff)) < 1e-10u"nm"
        @test maximum(norm.(vels_diff  )) < 1e-7u"nm * ps^-1"

        simulator_and_nounits = Verlet(
            dt=0.0005,
            coupling=AndersenThermostat(321.0, 10.0),
        )
        simulate!(sys_nounits, simulator_and_nounits, n_steps)
        @test temperature(sys_nounits) > 400.0

        params_dic_gpu = Molly.extract_parameters(sys_nounits, ff_nounits)
        @test params_dic == params_dic_gpu
        sys_nounits_nogi = System(sys_nounits; general_inters=())
        atoms_grad, pis_grad, sis_grad, gis_grad = Molly.inject_gradients(sys_nounits_nogi, params_dic_gpu)
        @test atoms_grad == sys_nounits.atoms
        @test pis_grad == sys_nounits.pairwise_inters
    end
end

@testset "Implicit solvent" begin
    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "his.xml"])...)

    for AT in array_list
        for solvent_model in (:obc2, :gbn2)
            sys = System(
                joinpath(data_dir, "6mrr_nowater.pdb"),
                ff;
                boundary=CubicBoundary(100.0u"nm"),
                array_type=AT,
                dist_cutoff=5.0u"nm",
                nonbonded_method=:none,
                implicit_solvent=solvent_model,
                kappa=1.0u"nm^-1",
            )
            neighbors = find_neighbors(sys)
            forces_molly = forces(sys)
            @test maximum(norm.(forces_molly .- forces_virial(sys)[1] )) < 1e-10u"kJ * mol^-1 * nm^-1"
            @test maximum(norm.(forces_molly .- forces(sys, neighbors))) < 1e-10u"kJ * mol^-1 * nm^-1"
            openmm_force_fp = joinpath(openmm_dir, "forces_$solvent_model.txt")
            forces_openmm = SVector{3}.(eachrow(readdlm(openmm_force_fp)))u"kJ * mol^-1 * nm^-1"
            @test maximum(norm.(from_device(forces_molly) .- forces_openmm)) < 1e-3u"kJ * mol^-1 * nm^-1"

            E_molly = potential_energy(sys)
            @test E_molly ≈ potential_energy(sys, neighbors)
            openmm_E_fp = joinpath(openmm_dir, "energy_$solvent_model.txt")
            E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
            @test abs(E_molly - E_openmm) < 1e-2u"kJ * mol^-1"

            if solvent_model == :gbn2
                sim = SteepestDescentMinimizer(tol=400.0u"kJ * mol^-1 * nm^-1")
                coords_start = copy(sys.coords)
                simulate!(sys, sim)
                @test potential_energy(sys) < E_molly
                @test rmsd(coords_start, sys.coords) < 0.1u"nm"
            end
        end
    end
end
