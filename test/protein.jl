@testset "Peptide" begin
    n_steps = 100
    temp = 298.0u"K"
    s = System(
        joinpath(data_dir, "5XER", "gmx_coords.gro"),
        joinpath(data_dir, "5XER", "gmx_top_ff.top");
        loggers=Dict(
            "temp"   => TemperatureLogger(10),
            "coords" => CoordinateLogger(10),
            "energy" => TotalEnergyLogger(10),
            "writer" => StructureWriter(10, temp_fp_pdb),
        ),
    )
    simulator = VelocityVerlet(dt=0.0002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))

    true_n_atoms = 5191
    @test length(s.atoms) == true_n_atoms
    @test length(s.coords) == true_n_atoms
    @test size(s.neighbor_finder.nb_matrix) == (true_n_atoms, true_n_atoms)
    @test size(s.neighbor_finder.matrix_14) == (true_n_atoms, true_n_atoms)
    @test length(s.general_inters) == 2
    @test length(s.specific_inter_lists) == 3
    @test s.box_size == SVector(3.7146, 3.7146, 3.7146)u"nm"
    show(devnull, first(s.atoms))

    s.velocities = [velocity(a.mass, temp) .* 0.01 for a in s.atoms]
    @time simulate!(s, simulator, n_steps; parallel=false)

    traj = read(temp_fp_pdb, BioStructures.PDB)
    rm(temp_fp_pdb)
    @test BioStructures.countmodels(traj) == 10
    @test BioStructures.countatoms(first(traj)) == 5191
end

@testset "Peptide Float32" begin
    n_steps = 100
    temp = 298.0f0u"K"
    s = System(
        Float32,
        joinpath(data_dir, "5XER", "gmx_coords.gro"),
        joinpath(data_dir, "5XER", "gmx_top_ff.top");
        loggers=Dict(
            "temp"   => TemperatureLogger(typeof(1.0f0u"K"), 10),
            "coords" => CoordinateLogger(typeof(1.0f0u"nm"), 10),
            "energy" => TotalEnergyLogger(typeof(1.0f0u"kJ * mol^-1"), 10),
        ),
    )
    simulator = VelocityVerlet(dt=0.0002f0u"ps", coupling=AndersenThermostat(temp, 10.0f0u"ps"))

    s.velocities = [velocity(a.mass, Float32(temp)) .* 0.01f0 for a in s.atoms]
    @time simulate!(s, simulator, n_steps; parallel=false)
end

@testset "OpenMM protein comparison" begin
    ff_dir = joinpath(data_dir, "force_fields")
    openmm_dir = joinpath(data_dir, "openmm_6mrr")

    ff = OpenMMForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...)
    sys = System(joinpath(data_dir, "6mrr_equil.pdb"), ff)
    neighbors = find_neighbors(sys, sys.neighbor_finder)

    for inter in ("bond", "angle", "proptor", "improptor", "lj", "coul", "all")
        if inter == "all"
            gin = sys.general_inters
        elseif inter == "lj"
            gin = sys.general_inters[1:1]
        elseif inter == "coul"
            gin = sys.general_inters[2:2]
        else
            gin = ()
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
            general_inters=gin,
            specific_inter_lists=sils,
            coords=sys.coords,
            box_size=sys.box_size,
            neighbor_finder=sys.neighbor_finder,
        )

        forces_molly = ustrip_vec.(accelerations(sys_part, neighbors; parallel=false) .* mass.(sys_part.atoms))
        forces_openmm = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "forces_$(inter)_only.txt"))))
        # All force terms on all atoms must match at some threshold
        @test !any(d -> any(abs.(d) .> 1e-6), forces_molly .- forces_openmm)

        E_molly = ustrip(potential_energy(sys_part, neighbors))
        E_openmm = readdlm(joinpath(openmm_dir, "energy_$(inter)_only.txt"))[1]
        # Energy must match at some threshold
        @test E_molly - E_openmm < 1e-5
    end

    # Run a short simulation with all interactions
    n_steps = 100
    simulator = VelocityVerlet(dt=0.0005u"ps")
    velocities_start = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))u"nm * ps^-1"
    sys.velocities = deepcopy(velocities_start)
    simulate!(sys, simulator, n_steps; parallel=true)

    coords_openmm = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "coordinates_$(n_steps)steps.txt"))))u"nm"
    vels_openmm   = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_$(n_steps)steps.txt" ))))u"nm * ps^-1"

    coords_diff = sys.coords .- wrap_coords_vec.(coords_openmm, (sys.box_size,))
    vels_diff = sys.velocities .- vels_openmm
    # Coordinates and velocities at end must match at some threshold
    @test maximum(maximum(abs.(v)) for v in coords_diff) < 1e-9u"nm"
    @test maximum(maximum(abs.(v)) for v in vels_diff  ) < 1e-6u"nm * ps^-1"

    # Test the same simulation on the GPU
    if run_gpu_tests
        sys = System(joinpath(data_dir, "6mrr_equil.pdb"), ff; gpu=true)
        sys.velocities = cu(deepcopy(velocities_start))    
        simulate!(sys, simulator, n_steps)

        coords_diff = Array(sys.coords) .- wrap_coords_vec.(coords_openmm, (sys.box_size,))
        vels_diff = Array(sys.velocities) .- vels_openmm
        @test maximum(maximum(abs.(v)) for v in coords_diff) < 1e-9u"nm"
        @test maximum(maximum(abs.(v)) for v in vels_diff  ) < 1e-6u"nm * ps^-1"
    end
end
