@testset "Lennard-Jones 2D" begin
    for AT in array_list
        n_atoms = 10
        n_steps = 20_000
        temp = 100.0u"K"
        boundary = RectangularBoundary(2.0u"nm")
        atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
                 for i in 1:n_atoms]
        simulator = VelocityVerlet(dt=0.001u"ps", coupling=(AndersenThermostat(temp, 10.0u"ps"),))
        gen_temp_wrapper(s, buffers, args...; kwargs...) = temperature(s; kin_tensor = buffers.kin_tensor)

        if Molly.uses_gpu_neighbor_finder(AT)
            neighbor_finder = GPUNeighborFinder(
                eligible=eligible=to_device(trues(n_atoms, n_atoms), AT),
                dist_cutoff=2.0u"nm",
            )
        else
            neighbor_finder = DistanceNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                n_steps=10,
                dist_cutoff=2.0u"nm",
            )
        end

        sys = System(
            atoms=to_device(atoms, AT),
            coords=to_device(place_atoms(n_atoms, boundary; min_dist=0.3u"nm"), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=neighbor_finder,
            loggers=(
                temp=TemperatureLogger(100),
                coords=CoordinatesLogger(100; dims=2),
                gen_temp=GeneralObservableLogger(gen_temp_wrapper, typeof(temp), 10),
                avg_temp=AverageObservableLogger(Molly.temperature_wrapper,
                                                    typeof(temp), 1; n_blocks=200),
            ),
        )

        random_velocities!(sys, temp)

        @test masses(sys) == to_device(fill(10.0u"g/mol", n_atoms), AT)
        @test AtomsBase.cell_vectors(sys) == (
            SVector(2.0, 0.0)u"nm",
            SVector(0.0, 2.0)u"nm",
        )

        show(devnull, sys)

        @time simulate!(sys, simulator, n_steps; n_threads=1)

        @test length(values(sys.loggers.coords)) == 201
        final_coords = last(values(sys.loggers.coords))
        @test all(all(c .> 0.0u"nm") for c in final_coords)
        @test all(all(c .< boundary) for c in final_coords)
        displacements(final_coords, boundary)
        distances(final_coords, boundary)
        rdf(final_coords, boundary)

        show(devnull, sys.loggers.gen_temp)
        show(devnull, sys.loggers.avg_temp)
        t, σ = values(sys.loggers.avg_temp)
        @test values(sys.loggers.avg_temp; std=false) == t
        @test isapprox(t, mean(values(sys.loggers.temp)); atol=3σ)
        run_visualize_tests && visualize(sys.loggers.coords, boundary, temp_fp_mp4)
    end
end

@testset "Lennard-Jones" begin
    n_atoms = 100
    atom_mass = 10.0u"g/mol"
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=(AndersenThermostat(temp, 10.0u"ps"),))

    TV = typeof(random_velocity(10.0u"g/mol", temp))
    TP = typeof(0.2u"kJ * mol^-1")

    V(sys, args...; kwargs...) = sys.velocities
    pot_obs(sys, neighbors, step_n; kwargs...) = potential_energy(sys, neighbors, step_n)
    kin_obs(sys, args...; kwargs...) = kinetic_energy(sys)

    for n_threads in n_threads_list
        s = System(
            atoms=[Atom(index=i, mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
                   for i in 1:n_atoms],
            coords=place_atoms(n_atoms, boundary; min_dist=0.3u"nm"),
            boundary=boundary,
            velocities=[random_velocity(atom_mass, temp) .* 0.01 for i in 1:n_atoms],
            atoms_data=[AtomData(atom_name="AR", res_number=i, res_name="AR",
                                 chain_id="B", element="Ar")
                        for i in 1:n_atoms],
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=DistanceNeighborFinder(
                eligible=trues(n_atoms, n_atoms),
                n_steps=10,
                dist_cutoff=2.0u"nm",
            ),
            loggers=(
                temp=TemperatureLogger(100),
                coords=CoordinatesLogger(100),
                vels=VelocitiesLogger(100),
                energy=TotalEnergyLogger(100),
                ke=KineticEnergyLogger(100),
                pe=PotentialEnergyLogger(100),
                force=ForcesLogger(100),
                dcd_writer=TrajectoryWriter(100, temp_fp_dcd),
                trr_writer=TrajectoryWriter(100, temp_fp_trr; write_velocities=true),
                pdb_writer=TrajectoryWriter(100, temp_fp_pdb),
                potkin_correlation=TimeCorrelationLogger(pot_obs, kin_obs, TP, TP, 1, 100),
                velocity_autocorrelation=AutoCorrelationLogger(V, TV, n_atoms, 100),
            ),
        )

        if n_threads == 1
            write_structure(temp_fp_pdb, s; atom_inds=[10, 12, 14, 16])
            @test readlines(temp_fp_pdb)[1] == "CRYST1     20.0     20.0     20.0  90.00  90.00  90.00 P 1           1"
            traj = read(temp_fp_pdb, BioStructures.PDBFormat)
            rm(temp_fp_pdb)
            @test BioStructures.countmodels(traj) == 1
            @test BioStructures.countatoms(first(traj)) == 4
            traj_atoms = BioStructures.collectatoms(traj)
            @test all(iszero, BioStructures.ishetero.(traj_atoms))
            @test BioStructures.serial.(traj_atoms) == [10, 12, 14, 16]
            @test BioStructures.chainids(traj) == ["B"]

            for write_boundary in (true, false)
                # Suppress sybyl type warning
                @suppress_err begin
                    write_structure(temp_fp_mol2, s; format="MOL2",
                                    write_boundary=write_boundary)
                    traj = Chemfiles.Trajectory(temp_fp_mol2)
                    rm(temp_fp_mol2)
                    @test Int(length(traj)) == 1
                    frame = read(traj)
                    @test length(frame) == 100
                    @test size(Chemfiles.positions(frame)) == (3, 100)
                    @test !iszero(sum(Array(Chemfiles.positions(frame))))
                    if write_boundary
                        @test Chemfiles.lengths(Chemfiles.UnitCell(frame)) == [20.0, 20.0, 20.0]
                    end
                end
            end

            write_structure(temp_fp_xyz, s)
            @test countlines(temp_fp_xyz) == 102
            traj = Chemfiles.Trajectory(temp_fp_xyz)
            rm(temp_fp_xyz)
            @test Int(length(traj)) == 1
            frame = read(traj)
            @test length(frame) == 100
            @test size(Chemfiles.positions(frame)) == (3, 100)
            @test !iszero(sum(Array(Chemfiles.positions(frame))))
            @test Chemfiles.lengths(Chemfiles.UnitCell(frame)) == [20.0, 20.0, 20.0]
        end

        # Test AtomsBase.jl interface
        @test length(s) == n_atoms
        @test eachindex(s) == Base.OneTo(n_atoms)
        @test length(s[2:4]) == 3
        @test length(s[[2, 4]]) == 2
        @test broadcast(a -> a.index, s) == collect(1:n_atoms)
        @test AtomsBase.position(s, :) == s.coords
        @test AtomsBase.position(s, 5) == s.coords[5]
        @test AtomsBase.velocity(s, :) == s.velocities
        @test AtomsBase.velocity(s, 5) == s.velocities[5]
        @test AtomsBase.mass(s, :) == fill(atom_mass, n_atoms)
        @test AtomsBase.mass(s, 5) == atom_mass
        @test AtomsBase.atomic_symbol(s) == fill(:Ar, n_atoms)
        @test AtomsBase.atomic_symbol(s, 5) == :Ar
        @test AtomsBase.cell_vectors(s) == (
            SVector(2.0, 0.0, 0.0)u"nm",
            SVector(0.0, 2.0, 0.0)u"nm",
            SVector(0.0, 0.0, 2.0)u"nm",
        )
        show(devnull, s[5])
        show(devnull, s[2:4])
        for a in s
            show(devnull, a)
        end

        nf_tree = TreeNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=10, dist_cutoff=2.0u"nm")
        neighbors = find_neighbors(s, s.neighbor_finder; n_threads=n_threads)
        neighbors_tree = find_neighbors(s, nf_tree; n_threads=n_threads)
        @test length(neighbors.list) == length(neighbors_tree.list)
        @test all(nn in neighbors_tree.list for nn in neighbors.list)

        @time simulate!(s, simulator, n_steps; n_threads=n_threads)

        show(devnull, s.loggers.temp)
        show(devnull, s.loggers.coords)
        show(devnull, s.loggers.vels)
        show(devnull, s.loggers.energy)
        show(devnull, s.loggers.ke)
        show(devnull, s.loggers.pe)
        show(devnull, s.loggers.force)
        show(devnull, s.loggers.dcd_writer)
        show(devnull, s.loggers.pdb_writer)
        show(devnull, s.loggers.potkin_correlation)
        show(devnull, s.loggers.velocity_autocorrelation)

        final_coords = last(values(s.loggers.coords))
        @test all(all(c .> 0.0u"nm") for c in final_coords)
        @test all(all(c .< boundary) for c in final_coords)
        displacements(final_coords, boundary)
        distances(final_coords, boundary)
        rdf(final_coords, boundary)
        @test unit(first(values(s.loggers.potkin_correlation))) == NoUnits
        @test unit(first(values(s.loggers.velocity_autocorrelation; normalize=false))) == u"nm^2 * ps^-2"

        traj = Chemfiles.Trajectory(temp_fp_dcd)
        rm(temp_fp_dcd)
        @test Int(length(traj)) == 201
        frame = read(traj)
        @test length(frame) == 100
        # Chemfiles does not write velocities to DCD files
        @test size(Chemfiles.positions(frame)) == (3, 100)
        @test !iszero(sum(Array(Chemfiles.positions(frame))))
        @test Chemfiles.lengths(Chemfiles.UnitCell(frame)) == [20.0, 20.0, 20.0]

        traj = Chemfiles.Trajectory(temp_fp_trr)
        rm(temp_fp_trr)
        @test Int(length(traj)) == 201
        frame = read(traj)
        @test length(frame) == 100
        @test size(Chemfiles.positions(frame)) == (3, 100)
        @test !iszero(sum(Array(Chemfiles.positions(frame))))
        @test size(Chemfiles.velocities(frame)) == (3, 100)
        @test !iszero(sum(Chemfiles.velocities(frame)[1, 1]))
        @test Chemfiles.lengths(Chemfiles.UnitCell(frame)) == [20.0, 20.0, 20.0]

        @test readlines(temp_fp_pdb)[1] == "CRYST1     20.0     20.0     20.0  90.00  90.00  90.00 P 1           1"
        traj = read(temp_fp_pdb, BioStructures.PDBFormat)
        rm(temp_fp_pdb)
        @test BioStructures.countmodels(traj) == 201
        @test BioStructures.countatoms(first(traj)) == 100

        run_visualize_tests && visualize(s.loggers.coords, boundary, temp_fp_mp4)

        coords_unc = [c .± (abs(randn()) / 100)u"nm"         for c in s.coords    ]
        vels_unc   = [v .± (abs(randn()) / 100)u"nm * ps^-1" for v in s.velocities]
        sys_unc = System(
            atoms=[Atom(index=i, mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
                   for i in 1:n_atoms],
            coords=coords_unc,
            boundary=boundary,
            velocities=vels_unc,
            atoms_data=[AtomData(atom_name="AR", res_number=i, res_name="AR",
                                 chain_id="B", element="Ar")
                        for i in 1:n_atoms],
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=DistanceNeighborFinder(
                eligible=trues(n_atoms, n_atoms),
                n_steps=10,
                dist_cutoff=2.0u"nm",
            ),
            loggers=(
                temp=TemperatureLogger(100),
                coords=CoordinatesLogger(100),
                vels=VelocitiesLogger(100),
                energy=TotalEnergyLogger(100),
                ke=KineticEnergyLogger(100),
                pe=PotentialEnergyLogger(100),
                force=ForcesLogger(100),
                dcd_writer=TrajectoryWriter(100, temp_fp_dcd),
                trr_writer=TrajectoryWriter(100, temp_fp_trr; write_velocities=true),
                pdb_writer=TrajectoryWriter(100, temp_fp_pdb),
                potkin_correlation=TimeCorrelationLogger(pot_obs, kin_obs, TP, TP, 1, 100),
                velocity_autocorrelation=AutoCorrelationLogger(V, TV, n_atoms, 100),
            ),
            strictness=:nowarn
        )
        for n_threads in n_threads_list
            @test typeof(potential_energy(sys_unc; n_threads=n_threads)) ==
                                typeof((1.0 ± 0.1)u"kJ * mol^-1")
            @test abs(potential_energy(sys_unc; n_threads=n_threads) -
                                potential_energy(s; n_threads=n_threads)) < 0.1u"kJ * mol^-1"
            @test typeof(kinetic_energy(sys_unc)) == typeof((1.0 ± 0.1)u"kJ * mol^-1")
            @test typeof(temperature(sys_unc)) == typeof((1.0 ± 0.1)u"K")
            @test abs(temperature(sys_unc) - temperature(s)) < 0.1u"K"
            @test eltype(eltype(forces(sys_unc; n_threads=n_threads))) ==
                                typeof((1.0 ± 0.1)u"kJ * mol^-1 * nm^-1")
        end
        simulator_unc = VelocityVerlet(dt=0.002u"ps")
        for n_threads in n_threads_list
            simulate!(sys_unc, simulator_unc, 1; n_threads=n_threads, run_loggers=false)
        end
    end
end

@testset "Lennard-Jones infinite boundaries" begin
    n_atoms = 100
    n_steps = 2_000
    temp = 298.0u"K"
    boundary = CubicBoundary(Inf * u"nm", Inf * u"nm", 2.0u"nm")
    coords = place_atoms(n_atoms, CubicBoundary(2.0u"nm"); min_dist=0.3u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=(AndersenThermostat(temp, 10.0u"ps"),))

    s = System(
        atoms=[Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        coords=coords,
        boundary=boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(coords=CoordinatesLogger(100),),
    )

    @test Molly.has_infinite_boundary(boundary)
    @test Molly.has_infinite_boundary(s)
    @test AtomsBase.atomic_symbol(s) == fill(:unknown, n_atoms)
    @test AtomsBase.atomic_symbol(s, 5) == :unknown

    random_velocities!(s, temp)

    @time simulate!(s, simulator, n_steps ÷ 2)
    @time simulate!(s, simulator, n_steps ÷ 2; run_loggers=:skipzero)

    @test length(values(s.loggers.coords)) == 21
    @test maximum(distances(s.coords, boundary)) > 5.0u"nm"

    run_visualize_tests && visualize(s.loggers.coords, boundary, temp_fp_mp4)
end

@testset "Lennard-Jones simulators" begin
    n_atoms = 100
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
             for i in 1:n_atoms]
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    simulators = [
        Verlet(dt=0.002u"ps", coupling=(AndersenThermostat(temp, 10.0u"ps"),)),
        StormerVerlet(dt=0.002u"ps"),
        Langevin(dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1"),
        OverdampedLangevin(dt=0.002u"ps", temperature=temp, friction=10.0u"ps^-1"),
    ]

    for AT in array_list
        if Molly.uses_gpu_neighbor_finder(AT)
            neighbor_finder = GPUNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                dist_cutoff=2.0u"nm",
            )
        else
            neighbor_finder = DistanceNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                n_steps=10,
                dist_cutoff=2.0u"nm",
            )
        end
        sys = System(
            atoms=to_device(atoms, AT),
            coords=to_device(coords, AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=neighbor_finder,
            loggers=(coords=CoordinatesLogger(100),),
        )
        random_velocities!(sys, temp)
        for simulator in simulators
            @time simulate!(sys, simulator, n_steps; n_threads=1)
        end
    end
end

@testset "Verlet integrators on CPU and GPU" begin
    n_atoms = 100
    n_steps = 1000
    temp = 298.0u"K"
    boundary = CubicBoundary(4.0u"nm")
    atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.1u"nm", ϵ=0.2u"kJ * mol^-1")
             for i in 1:n_atoms]
    coords = place_atoms(n_atoms, boundary; min_dist=0.2u"nm")
    velocities = [random_velocity(10.0u"g/mol", temp) .* 0.01 for i in 1:n_atoms]
    simulators = [
        VelocityVerlet(dt=0.002u"ps"),
        Verlet(dt=0.002u"ps"),
        StormerVerlet(dt=0.002u"ps"),
    ]

    sys = System(
        atoms=atoms,
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(
            coords=CoordinatesLogger(100),
            disp=DisplacementsLogger(100, coords),
        ),
    )

    @test_throws ArgumentError DisplacementsLogger(100, coords; n_steps_update=17)

    if run_cuda_tests
        sys_gpu = System(
            atoms=CuArray(atoms),
            coords=CuArray(coords),
            velocities=CuArray(velocities),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=GPUNeighborFinder(
                eligible=CuArray(trues(n_atoms, n_atoms)),
                dist_cutoff=2.0u"nm",
            ),
            loggers=(
                coords=CoordinatesLogger(100),
                disp=DisplacementsLogger(100, CuArray(coords)),
            ),
        )
    end

    for simulator in simulators
        @time simulate!(sys, simulator, n_steps; n_threads=1)
        @test all(isequal(0.0u"nm"), norm.(first(values(sys.loggers.disp))))
        @test mean(norm.(sys.loggers.disp.displacements[end])) > 0.005u"nm"
        if run_cuda_tests
            @time simulate!(sys_gpu, simulator, n_steps; n_threads=1)
            @test all(isequal(0.0u"nm"), norm.(first(values(sys_gpu.loggers.disp))))
            @test mean(norm.(sys.loggers.disp.displacements[end])) > 0.005u"nm"
            coord_diff = sys.coords .- from_device(sys_gpu.coords)
            coord_diff_size = sum(sum(map(x -> abs.(x), coord_diff))) / (3 * n_atoms)
            E_diff = abs(potential_energy(sys) - potential_energy(sys_gpu))
            @test coord_diff_size < 1e-4u"nm"
            @test E_diff < 5e-4u"kJ * mol^-1"
        end
    end
end

@testset "Diatomic molecules" begin
    n_atoms = 100
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms ÷ 2, boundary; min_dist=0.3u"nm")
    for i in eachindex(coords)
        push!(coords, coords[i] .+ [0.1, 0.0, 0.0]u"nm")
    end
    bonds = InteractionList2Atoms(
        collect(1:(n_atoms ÷ 2)),
        collect((1 + n_atoms ÷ 2):n_atoms),
        [HarmonicBond(k=300_000.0u"kJ * mol^-1 * nm^-2", r0=0.1u"nm") for i in 1:(n_atoms ÷ 2)],
        fill("", n_atoms ÷ 2),
    )
    eligible = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 2)
        eligible[i, i + (n_atoms ÷ 2)] = false
        eligible[i + (n_atoms ÷ 2), i] = false
    end
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=(BerendsenThermostat(temp, 1.0u"ps"),))

    s = System(
        atoms=[Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        coords=coords,
        boundary=boundary,
        velocities=[random_velocity(10.0u"g/mol", temp) .* 0.01 for i in 1:n_atoms],
        pairwise_inters=(LennardJones(use_neighbors=true),),
        specific_inter_lists=(bonds,),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(
            temp=TemperatureLogger(10),
            coords=CoordinatesLogger(10),
        ),
    )

    @time simulate!(s, simulator, n_steps; n_threads=1)

    if run_visualize_tests
        visualize(
            s.loggers.coords,
            boundary,
            temp_fp_mp4;
            connections=[(i, i + (n_atoms ÷ 2)) for i in 1:(n_atoms ÷ 2)],
            trails=2,
        )
    end
end

@testset "Pairwise interactions" begin
    n_atoms = 100
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    G = 10.0u"kJ * mol * nm * g^-2"
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=(AndersenThermostat(temp, 10.0u"ps"),))
    pairwise_inter_types = (
        LennardJones(use_neighbors=true), LennardJones(use_neighbors=false),
        LennardJones(cutoff=DistanceCutoff(1.0u"nm"), use_neighbors=true),
        LennardJones(cutoff=ShiftedPotentialCutoff(1.0u"nm"), use_neighbors=true),
        LennardJones(cutoff=ShiftedForceCutoff(1.0u"nm"), use_neighbors=true),
        LennardJones(cutoff=CubicSplineCutoff(0.6u"nm", 1.0u"nm"), use_neighbors=true),
        SoftSphere(use_neighbors=true), SoftSphere(use_neighbors=false),
        Mie(m=5, n=10, use_neighbors=true), Mie(m=5, n=10, use_neighbors=false),
        Coulomb(use_neighbors=true), Coulomb(use_neighbors=false),
        CoulombReactionField(dist_cutoff=1.0u"nm", use_neighbors=true),
        CoulombReactionField(dist_cutoff=1.0u"nm", use_neighbors=false),
        Gravity(G=G, use_neighbors=true), Gravity(G=G, use_neighbors=false),
    )

    for inter in pairwise_inter_types
        if use_neighbors(inter)
            neighbor_finder = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=10,
                                                        dist_cutoff=1.5u"nm")
        else
            neighbor_finder = NoNeighborFinder()
        end

        s = System(
            atoms=[Atom(mass=10.0u"g/mol", charge=(i % 2 == 0 ? -1.0 : 1.0), σ=0.2u"nm",
                        ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
            coords=place_atoms(n_atoms, boundary; min_dist=0.2u"nm"),
            boundary=boundary,
            velocities=[random_velocity(10.0u"g/mol", temp) .* 0.01 for i in 1:n_atoms],
            pairwise_inters=(inter,),
            neighbor_finder=neighbor_finder,
            loggers=(
                temp=TemperatureLogger(100),
                coords=CoordinatesLogger(100),
                energy=TotalEnergyLogger(100),
            ),
        )

        @time simulate!(s, simulator, n_steps)
    end
end

@testset "LJ on CPU and GPU" begin
    n_atoms = 100
    n_steps = 100
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps")
    pairwise_inter_types = (
        LennardJones(use_neighbors=true),
        LennardJones(use_neighbors=false),
        LennardJones(cutoff=DistanceCutoff(1.0u"nm"), use_neighbors=true),
        LennardJones(cutoff=ShiftedPotentialCutoff(1.0u"nm"), use_neighbors=true),
        LennardJones(cutoff=ShiftedForceCutoff(1.0u"nm"), use_neighbors=true),
        LennardJones(cutoff=CubicSplineCutoff(0.6u"nm", 1.0u"nm"), use_neighbors=true),
    )

    for inter in pairwise_inter_types
        if use_neighbors(inter)
            neighbor_finder = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=10,
                                                        dist_cutoff=1.2u"nm")
        else
            neighbor_finder = NoNeighborFinder()
        end

        if run_cuda_tests
            if use_neighbors(inter)
                neighbor_finder_gpu = GPUNeighborFinder(eligible=CuArray(trues(n_atoms, n_atoms)),
                                                        dist_cutoff=1.2u"nm")
            else
                neighbor_finder_gpu = NoNeighborFinder()
            end
        end

        atoms = [Atom(mass=10.0u"g/mol", charge=(i % 2 == 0 ? -1.0 : 1.0), σ=0.2u"nm", ϵ=0.2u"kJ * mol^-1")
                 for i in 1:n_atoms]
        coords = place_atoms(n_atoms, boundary; min_dist=0.2u"nm")
        velocities = [random_velocity(10.0u"g/mol", temp) .* 0.01 for i in 1:n_atoms]

        sys = System(
            atoms=copy(atoms),
            coords=copy(coords),
            boundary=boundary,
            velocities=copy(velocities),
            pairwise_inters=(inter,),
            neighbor_finder=neighbor_finder,
        )
        E0 = potential_energy(sys)
        @time simulate!(sys, simulator, n_steps)

        if run_cuda_tests
            sys_gpu = System(
                atoms=CuArray(atoms),
                coords=CuArray(coords),
                boundary=boundary,
                velocities=CuArray(velocities),
                pairwise_inters=(inter,),
                neighbor_finder=neighbor_finder_gpu,
            )
            E_diff_start = abs(E0 - potential_energy(sys_gpu))
            @test E_diff_start < 5e-4u"kJ * mol^-1"
            @time simulate!(sys_gpu, simulator, n_steps)
            coord_diff = sys.coords .- from_device(sys_gpu.coords)
            coord_diff_size = sum(sum(map(x -> abs.(x), coord_diff))) / (3 * n_atoms)
            E_diff = abs(potential_energy(sys) - potential_energy(sys_gpu))
            @test coord_diff_size < 5e-4u"nm"
            @test E_diff < 5e-3u"kJ * mol^-1"
        end
    end
end

@testset "Müller-Brown" begin
    atom_mass = 1.0u"g/mol"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")]
    boundary = RectangularBoundary(Inf*u"nm")
    coords = [SVector(-0.5, 0.25)u"nm"]
    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp; dims=2)]

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        general_inters=(MullerBrown(),),
        loggers=(coords=CoordinatesLogger(100; dims=2),),
    )

    simulator = VelocityVerlet(dt=0.002u"ps")
    simulate!(sys, simulator, 100_000)

    # Particle should end up at local minimum and stick due to no thermostat
    final_pos = values(sys.loggers.coords)[end][1]
    local_min = SVector(-0.05001082299878202, 0.46669410487256247)u"nm"
    @test isapprox(final_pos, local_min; atol=1e-7u"nm")
end

@testset "Units vs no units" begin
    n_atoms = 100
    n_steps = 2_000 # Does diverge for longer simulations or higher velocities
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    velocities = [random_velocity(10.0u"g/mol", temp) .* 0.01 for i in 1:n_atoms]
    simulator = VelocityVerlet(dt=0.002u"ps")
    simulator_nounits = VelocityVerlet(dt=0.002)

    vtype = eltype(velocities)
    V(sys::System, neighbors=nothing) = sys.velocities

    s = System(
        atoms=[Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(
            temp=TemperatureLogger(100),
            coords=CoordinatesLogger(100),
            energy=TotalEnergyLogger(100),
        ),
    )

    vtype_nounits = eltype(ustrip_vec.(velocities))

    s_nounits = System(
        atoms=[Atom(mass=10.0, charge=0.0, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
        coords=ustrip_vec.(coords),
        boundary=CubicBoundary(ustrip.(boundary)),
        velocities=ustrip_vec.(u"nm/ps",velocities),
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0,
        ),
        loggers=(
            temp=TemperatureLogger(Float64, 100),
            coords=CoordinatesLogger(Float64, 100),
            energy=TotalEnergyLogger(Float64, 100),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    neighbors = find_neighbors(s, s.neighbor_finder; n_threads=1)
    neighbors_nounits = find_neighbors(s_nounits, s_nounits.neighbor_finder; n_threads=1)
    a1 = accelerations(s, neighbors)
    a2 = accelerations(s_nounits, neighbors_nounits)u"kJ * nm^-1 * g^-1"
    @test all(all(a1[i] .≈ a2[i]) for i in eachindex(a1))

    simulate!(s, simulator, n_steps; n_threads=1)
    simulate!(s_nounits, simulator_nounits, n_steps; n_threads=1)

    coords_diff = last(values(s.loggers.coords)) .- last(values(s_nounits.loggers.coords)) * u"nm"
    @test median([maximum(abs.(c)) for c in coords_diff]) < 1e-8u"nm"

    final_energy = last(values(s.loggers.energy))
    final_energy_nounits = last(values(s_nounits.loggers.energy)) * u"kJ * mol^-1"
    @test isapprox(final_energy, final_energy_nounits; atol=5e-4u"kJ * mol^-1")
end

@testset "Position restraints" begin
    for AT in array_list
        n_atoms = 10
        n_atoms_res = n_atoms ÷ 2
        n_steps = 2_000
        boundary = CubicBoundary(2.0u"nm")
        starting_coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
        atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.2u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
        atoms_data = [AtomData(atom_type=(i <= n_atoms_res ? "A1" : "A2")) for i in 1:n_atoms]
        sim = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")

        sys = System(
            atoms=to_device(atoms, AT),
            coords=to_device(copy(starting_coords), AT),
            boundary=boundary,
            atoms_data=atoms_data,
            pairwise_inters=(LennardJones(),),
            loggers=(coords=CoordinatesLogger(100),),
        )

        atom_selector(at, at_data) = at_data.atom_type == "A1"

        sys_res = add_position_restraints(sys, 100_000.0u"kJ * mol^-1 * nm^-2";
                                          atom_selector=atom_selector)

        @time simulate!(sys_res, sim, n_steps)

        dists = norm.(vector.(starting_coords, from_device(sys_res.coords), (boundary,)))
        @test maximum(dists[1:n_atoms_res]) < 0.1u"nm"
        @test median(dists[(n_atoms_res + 1):end]) > 0.2u"nm"
    end
end

@testset "Langevin splitting" begin
    n_atoms = 400
    n_steps = 2000
    temp = 300.0u"K"
    boundary = CubicBoundary(10.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    velocities = [random_velocity(10.0u"g/mol", temp) .* 0.01 for i in 1:n_atoms]
    s1 = System(
        atoms=[Atom( mass=10.0u"g/mol", charge=0.0,σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(temp=TemperatureLogger(10),),
    )
    s2 = deepcopy(s1)
    rseed = 2022
    simulator1 = Langevin(dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1")
    simulator2 = LangevinSplitting(dt=0.002u"ps", temperature=temp,
                                   friction=10.0u"g * mol^-1 * ps^-1", splitting="BAOA")

    @time simulate!(s1, simulator1, n_steps; rng=MersenneTwister(rseed))
    @test 280.0u"K" <= mean(s1.loggers.temp.history[(end - 100):end]) <= 320.0u"K"

    @time simulate!(s2, simulator2, n_steps; rng=MersenneTwister(rseed))
    @test 280.0u"K" <= mean(s2.loggers.temp.history[(end - 100):end]) <= 320.0u"K"

    @test maximum(maximum(abs.(v)) for v in (s1.coords .- s2.coords)) < 1e-5u"nm"
end

@testset "Nosé-Hoover" begin
    n_atoms = 256
    atom_mass = 39.98u"g/mol"
    atoms = [Atom(mass=atom_mass, σ=0.34u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(4.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.36u"nm")
    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(),),
        loggers=(temp=TemperatureLogger(1),),
    )

    minimizer = SteepestDescentMinimizer()
    simulate!(sys, minimizer)

    simulator = NoseHoover(dt=0.002u"ps", temperature=temp)
    simulate!(sys, simulator, 50_000)

    @test (temp - 1.0u"K") < mean(values(sys.loggers.temp)) < (temp + 1.0u"K")
    @test std(values(sys.loggers.temp)) > 2.0u"K"
end

@testset "Temperature REMD" begin
    n_atoms = 100
    n_steps = 10_000
    atom_mass = 10.0u"g/mol"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")

    pairwise_inters = (LennardJones(use_neighbors=true),)

    eligible = trues(n_atoms, n_atoms)

    neighbor_finder = DistanceNeighborFinder(
        eligible=eligible,
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    
    # Define the unperturbed base system
    base_sys = System(
        atoms=atoms, 
        coords=coords, 
        boundary=boundary, 
        pairwise_inters=pairwise_inters, 
        neighbor_finder=neighbor_finder
    )

    n_replicas = 4
    temp_vals = [120.0u"K", 180.0u"K", 240.0u"K", 300.0u"K"]
    
    # Construct the array of thermodynamic states
    thermo_states = ThermoState[]
    for temp in temp_vals
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(base_sys, intg; temperature=temp))
    end

    replica_loggers = [(temp=TemperatureLogger(10), coords=CoordinatesLogger(10)) for i in 1:n_replicas]

    # Initialize ReplicaSystem using the generalized constructor
    repsys = ReplicaSystem(
        thermo_states,
        [copy(coords) for _ in 1:n_replicas];
        replica_loggers=replica_loggers,
    )

    @test !is_on_gpu(repsys)
    @test float_type(repsys) == Float64
    @test masses(repsys) == fill(atom_mass, n_atoms)
    @test length(repsys) == n_atoms
    @test eachindex(repsys) == Base.OneTo(n_atoms)
    @test AtomsBase.mass(repsys, :) == fill(atom_mass, n_atoms)
    @test AtomsBase.mass(repsys, 5) == atom_mass
    @test AtomsBase.cell_vectors(repsys) == (
        SVector(2.0, 0.0, 0.0)u"nm",
        SVector(0.0, 2.0, 0.0)u"nm",
        SVector(0.0, 0.0, 2.0)u"nm",
    )
    show(devnull, repsys)

    # Use the unified simulator
    simulator = ReplicaExchangeMD(dt=0.005u"ps", exchange_time=2.5u"ps")

    @time simulate!(repsys, simulator, n_steps; assign_velocities=true)
    @time simulate!(repsys, simulator, n_steps; assign_velocities=false)

    efficiency = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test efficiency > 0.2 # This is a fairly arbitrary threshold but it's a good test for very bad cases
    @test efficiency < 1.0 # Bad acceptance rate?
    @info "Exchange Efficiency: $efficiency"

    for id in 1:n_replicas
        mean_temp = mean(values(repsys.replica_loggers[id].temp))
        # Given physical coordinates swap thermal states, they should average out across the ladder bounds
        @test (0.9 * temp_vals[1]) < mean_temp < (1.1 * temp_vals[end])
    end
end

@testset "Hamiltonian REMD" begin
    n_atoms = 100
    n_steps = 10_000
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 100.0u"K"

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_replicas = 4
    λ_vals = [1.0, 0.9, 0.75, 0.6]
    
    thermo_states = ThermoState[]
    for i in 1:n_replicas
        # Embed the lambda values directly into the atoms for this thermodynamic state
        atoms_λ = [Atom(mass=atom_mass, charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", 
                        λ =λ_vals[i]) for _ in 1:n_atoms]
        
        sys = System(
            atoms=atoms_λ,
            coords=coords,
            boundary=boundary,
            # SoftCore no longer takes λ; it relies on the atom's λ properties
            pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
            neighbor_finder=neighbor_finder
        )
        # All states share the exact same temperature and integrator parameters
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(sys, intg; temperature=temp))
    end

    replica_loggers = [(temp=TemperatureLogger(10), ) for i in 1:n_replicas]

    # Initialize generalized ReplicaSystem
    repsys = ReplicaSystem(
        thermo_states,
        [copy(coords) for _ in 1:n_replicas];
        replica_loggers=replica_loggers,
    )

    # Use the unified simulator (implicitly handles Hamiltonian REMD based on the ThermoStates)
    simulator = ReplicaExchangeMD(dt=0.005u"ps", exchange_time=2.5u"ps")

    @time simulate!(repsys, simulator, n_steps; assign_velocities=true)
    @time simulate!(repsys, simulator, n_steps; assign_velocities=false)

    efficiency = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test efficiency > 0.2 # This is a fairly arbitrary threshold, but it's a good test for very bad cases
    @test efficiency < 1.0 # Bad acceptance rate?
    @info "Exchange Efficiency: $efficiency"

    for id in 1:n_replicas
        mean_temp = mean(values(repsys.replica_loggers[id].temp))
        # Since temperature is constant across the ladder, physical replicas should hover exactly around temp
        @test (0.9 * temp) < mean_temp < (1.1 * temp)
    end
end

@testset "Metropolis Monte Carlo" begin
    n_atoms = 100
    n_steps = 10_000
    atom_mass = 10.0u"g/mol"
    atoms = [Atom(mass=atom_mass, charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(4.0u"nm", 4.0u"nm", 4.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 198.0u"K"

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        pairwise_inters=(Coulomb(use_neighbors=true), ),
        neighbor_finder=neighbor_finder,
        loggers=(
            coords=CoordinatesLogger(10),
            mcl=MonteCarloLogger(),
            avgpe=AverageObservableLogger(Molly.potential_energy_wrapper, typeof(atoms[1].ϵ), 10),
        ),
    )

    simulator_uniform = MetropolisMonteCarlo(
        temperature=temp,
        trial_moves=random_uniform_translation!,
        trial_args=Dict(:shift_size => 0.1u"nm"),
    )

    simulator_gaussian = MetropolisMonteCarlo(
        temperature=temp,
        trial_moves=random_normal_translation!,
        trial_args=Dict(:shift_size => 0.1u"nm"),
    )

    @time simulate!(sys, simulator_uniform , n_steps)
    @time simulate!(sys, simulator_gaussian, n_steps)

    acceptance_rate = sys.loggers.mcl.n_accept / sys.loggers.mcl.n_trials
    @info "Acceptance Rate: $acceptance_rate"
    @test acceptance_rate > 0.05

    @test sys.loggers.avgpe.block_averages[end] < sys.loggers.avgpe.block_averages[1]

    distance_sum = 0.0u"nm"
    for i in eachindex(sys)
        ci = sys.coords[i]
        min_dist2 = Inf*u"nm^2"
        for j in eachindex(sys)
            if i == j
                continue
            end
            r2 = sum(abs2, vector(ci, sys.coords[j], sys.boundary))
            if r2 < min_dist2
                min_dist2 = r2
            end
        end
        distance_sum += sqrt(min_dist2)
    end
    mean_distance = distance_sum / length(sys)
    wigner_seitz_radius = cbrt(3 * volume(sys.boundary) / (4π * length(sys)))
    @test wigner_seitz_radius < mean_distance < 2 * wigner_seitz_radius
end

@testset "Crystals" begin
    r_cut = 0.85u"nm"
    a = 0.52468u"nm"
    atom_mass = 39.948u"g/mol"
    temp = 10.0u"K"

    fcc_crystal = SimpleCrystals.FCC(a, atom_mass, SVector(4, 4, 4))
    n_atoms = length(fcc_crystal)
    @test n_atoms == 256
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    sys = System(
        fcc_crystal;
        velocities=velocities,
        pairwise_inters=(LennardJones(cutoff=ShiftedForceCutoff(r_cut)),),
        loggers=(tot_eng=TotalEnergyLogger(100),),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )

    sys_cp = System(sys)
    @test sys_cp.atoms  == sys.atoms
    @test sys_cp.coords == sys.coords
    sys_mod = System(sys; coords=(sys.coords .* 0.5))
    @test sys_mod.atoms  == sys.atoms
    @test sys_mod.coords == sys.coords .* 0.5

    σ = 0.34u"nm"
    ϵ = (4.184 * 0.24037)u"kJ * mol^-1"
    updated_atoms = []

    for i in eachindex(sys)
        push!(updated_atoms, Atom(index=sys.atoms[i].index, atom_type=sys.atoms[i].atom_type,
                                  charge=sys.atoms[i].charge, mass=sys.atoms[i].mass,
                                  σ=σ, ϵ=ϵ))
    end

    sys = System(sys; atoms=[updated_atoms...])

    simulator = Langevin(
        dt=2.0u"fs",
        temperature=temp,
        friction=1.0u"ps^-1",
    )

    @time simulate!(sys, simulator, 25_000; run_loggers=false)
    @time simulate!(sys, simulator, 25_000)

    @test length(values(sys.loggers.tot_eng)) == 251
    @test -1800u"kJ * mol^-1" < mean(values(sys.loggers.tot_eng)) < -1600u"kJ * mol^-1"

    # Test unsupported crystals
    hex_crystal = SimpleCrystals.Hexagonal(a, :Ar, SVector(2, 2))
    @test_throws ArgumentError System(hex_crystal)

    # Make an invalid crystals (angle is too large)
    function MyInvalidCrystal(a, atomic_symbol::Symbol, N::SVector{3}; charge=0.0u"C")
        lattice = SimpleCrystals.BravaisLattice(
            SimpleCrystals.MonoclinicLattice(a, a, a, 120u"°"),
            SimpleCrystals.Primitive(),
        )
        z = zero(a)
        basis = [SimpleCrystals.Atom(atomic_symbol, [z, z, z], charge=charge)]
        return SimpleCrystals.Crystal(lattice, basis, N)
    end
    my_crystal = MyInvalidCrystal(a, :Ar, SVector(1, 1, 1))
    @test_throws ErrorException System(my_crystal)
end

@testset "Different implementations" begin
    n_atoms = 400
    atom_mass = 10.0u"g/mol"
    v1 = SVector(5.0u"nm", 0.0u"nm", 0.0u"nm")
    v2 = SVector(2.0u"nm", 6.0u"nm", 0.0u"nm")
    v3 = SVector(3.0u"nm", 4.0u"nm", 7.0u"nm")
    boundary_cubic = CubicBoundary(6.0u"nm")
    boundary_triclinic = TriclinicBoundary(v1, v2, v3)
    temp = 1.0u"K"
    starting_coords_cubic = place_diatomics(n_atoms ÷ 2, boundary_cubic, 0.2u"nm"; min_dist=0.2u"nm")
    starting_coords_f32_cubic = [Float32.(c) for c in starting_coords_cubic]
    starting_coords_triclinic = place_diatomics(n_atoms ÷ 2, boundary_triclinic, 0.2u"nm"; min_dist=0.2u"nm")
    starting_coords_f32_triclinic = [Float32.(c) for c in starting_coords_triclinic]
    starting_velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]
    starting_velocities_f32 = [Float32.(c) for c in starting_velocities]

    function test_sim(nft, parallel::Bool, f32::Bool, ::Type{AT}, triclinic::Bool) where AT
        T = (f32 ? Float32 : Float64)
        n_atoms = 400
        n_steps = 200
        atom_mass = T(10.0)u"g/mol"
        boundary = triclinic ? TriclinicBoundary(T.(v1), T.(v2), T.(v3)) : CubicBoundary(T(6.0)u"nm")
        starting_coords = triclinic ? starting_coords_triclinic : starting_coords_cubic
        starting_coords_f32 = triclinic ? starting_coords_f32_triclinic : starting_coords_f32_cubic
        simulator = VelocityVerlet(dt=T(0.02)u"ps")
        k = T(10_000.0)u"kJ * mol^-1 * nm^-2"
        r0 = T(0.2)u"nm"
        bonds = [HarmonicBond(k=k, r0=r0) for i in 1:(n_atoms ÷ 2)]
        specific_inter_lists = (InteractionList2Atoms(
            to_device(Int32.(collect(1:2:n_atoms)), AT),
            to_device(Int32.(collect(2:2:n_atoms)), AT),
            to_device(bonds, AT),
        ),)
        cutoff = DistanceCutoff(T(1.0)u"nm")

        if nft == GPUNeighborFinder
            neighbor_finder = GPUNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                dist_cutoff=T(1.0)u"nm",
            )
        elseif nft == DistanceNeighborFinder
            neighbor_finder = DistanceNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                n_steps=10,
                dist_cutoff=T(1.5)u"nm",
            )
        else
            neighbor_finder = NoNeighborFinder()
        end
        pairwise_inters = (LennardJones(use_neighbors=(nft != NoNeighborFinder), cutoff=cutoff),)
        show(devnull, neighbor_finder)

        coords = to_device(copy(f32 ? starting_coords_f32 : starting_coords), AT)
        velocities = to_device(copy(f32 ? starting_velocities_f32 : starting_velocities), AT)
        atoms = to_device([Atom(charge=zero(T), mass=atom_mass, σ=T(0.2)u"nm",
                                ϵ=T(0.2)u"kJ * mol^-1") for i in 1:n_atoms], AT)

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            specific_inter_lists=specific_inter_lists,
            neighbor_finder=neighbor_finder,
        )

        @test is_on_gpu(sys) == (AT <: AbstractGPUArray)
        @test float_type(sys) == T

        n_threads = (parallel ? Threads.nthreads() : 1)
        E_start = potential_energy(sys; n_threads=n_threads)

        simulate!(sys, simulator, n_steps; n_threads=n_threads)
        return sys.coords, E_start
    end

    runs = [
        ("CPU"       , [NoNeighborFinder      , false, false, Array]),
        ("CPU f32"   , [NoNeighborFinder      , false, true , Array]),
        ("CPU NL"    , [DistanceNeighborFinder, false, false, Array]),
        ("CPU f32 NL", [DistanceNeighborFinder, false, true , Array]),
    ]
    if run_parallel_tests
        push!(runs, ("CPU parallel"       , [NoNeighborFinder      , true , false, Array]))
        push!(runs, ("CPU parallel f32"   , [NoNeighborFinder      , true , true , Array]))
        push!(runs, ("CPU parallel NL"    , [DistanceNeighborFinder, true , false, Array]))
        push!(runs, ("CPU parallel f32 NL", [DistanceNeighborFinder, true , true , Array]))
    end
    for AT in array_list[2:end]
        push!(runs, ("$AT"       , [NoNeighborFinder      , false, false, AT]))
        push!(runs, ("$AT f32"   , [NoNeighborFinder      , false, true , AT]))
        push!(runs, ("$AT NL"    , [DistanceNeighborFinder, false, false, AT]))
        push!(runs, ("$AT f32 NL", [DistanceNeighborFinder, false, true , AT]))
    end
    if run_cuda_tests
        AT = CuArray
        push!(runs, ("$AT GPU NL"    , [GPUNeighborFinder, false, false, AT]))
        push!(runs, ("$AT f32 GPU NL", [GPUNeighborFinder, false, true , AT]))
    end
    if run_metal_tests
        AT = MtlArray
        push!(runs, ("$AT f32"   , [NoNeighborFinder      , false, true , AT]))
        push!(runs, ("$AT f32 NL", [DistanceNeighborFinder, false, true , AT]))
    end

    # Check all simulations give the same result to within some error
    for triclinic in (false, true)
        final_coords_ref, E_start_ref = test_sim(runs[1][2]..., triclinic)
        for (name, args) in runs
            final_coords, E_start = test_sim(args..., triclinic)
            final_coords_f64 = [Float64.(c) for c in from_device(final_coords)]
            coord_diff = final_coords_f64 .- final_coords_ref
            coord_diff_size = sum(sum(map(x -> abs.(x), coord_diff))) / (3 * n_atoms)
            E_diff = abs(Float64(E_start) - E_start_ref)
            name = (triclinic ? "$name triclinic" : "$name cubic")
            @info "$(rpad(name, 29)) - difference per coordinate $coord_diff_size - potential energy difference $E_diff"
            @test coord_diff_size < 1e-4u"nm"
            @test E_diff < 5e-4u"kJ * mol^-1"
        end
    end
end

@testset "Accelerated Weight Histogram (AWH)" begin
    n_atoms = 50
    n_steps = 2_000
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_windows = 4
    λ_vals = [1.0, 0.8, 0.6, 0.4]
    
    thermo_states = ThermoState[]
    for i in 1:n_windows
        # Embed the lambda values directly into the atoms
        atoms_λ = [Atom(mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", 
                        λ = λ_vals[i]) for _ in 1:n_atoms]
        
        # Define the system at this specific lambda state
        sys = System(
            atoms=atoms_λ,
            coords=coords,
            boundary=boundary,
            pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
            neighbor_finder=neighbor_finder
        )
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(sys, intg; temperature=temp))
    end

    # Initialize AWH state using the newly generalized array of ThermoStates
    # n_bias is set low (10) to guarantee the initial stage is rapidly saturated 
    # and weight updates trigger during a short 2000 step test
    awh_state = AWHState(thermo_states; first_state=1, n_bias=10)

    # Wrap in AWHSimulation
    awh_sim = AWHSimulation(
        awh_state;
        num_md_steps=10,
        update_freq=5,
        well_tempered_factor=10.0,
        coverage_threshold=1.0,
        log_freq=10
    )

    initial_f = copy(awh_sim.state.f)

    # Run the AWH simulation loop
    @time simulate!(awh_sim, n_steps)

    # Verification
    # 1. Active index must remain strictly within the bounds of the lambda ladder
    @test 1 <= awh_sim.state.active_idx <= n_windows
    
    # 2. Gibbs sampling should accumulate effective samples
    @test awh_sim.state.N_eff > 0
    
    # 3. The free energy array must update from its initial state
    @test awh_sim.state.f != initial_f
    
    # 4. AWH enforces a structural constraint where the first state acts as the reference (f = 0.0)
    @test awh_sim.state.f[1] == 0.0

    # 4b. simulate!(awh_sim, n_steps) must also execute remainder steps when
    #     n_steps is not divisible by num_md_steps.
    rem_boundary = CubicBoundary(10.0u"nm")
    rem_atoms = [Atom(index=1, mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)]
    rem_coords = [SVector(0.0, 0.0, 0.0)u"nm"]
    rem_vels = [SVector(1.0, 0.0, 0.0)u"nm/ps"]
    rem_sys = System(
        atoms=rem_atoms,
        coords=rem_coords,
        velocities=rem_vels,
        boundary=rem_boundary,
        pairwise_inters=(),
        neighbor_finder=NoNeighborFinder(),
    )
    rem_intg = VelocityVerlet(dt=0.1u"ps", remove_CM_motion=0)
    rem_states = [ThermoState(rem_sys, rem_intg; temperature=300.0u"K")]
    rem_awh_state = AWHState(rem_states; n_bias=5)
    rem_awh_sim = AWHSimulation(
        rem_awh_state;
        num_md_steps=4,
        update_freq=1,
        well_tempered_factor=Inf,
        log_freq=1000,
    )
    rem_x0 = rem_awh_sim.state.active_sys.coords[1][1]
    simulate!(rem_awh_sim, 6)
    rem_x1 = rem_awh_sim.state.active_sys.coords[1][1]
    @test isapprox(ustrip(u"nm", rem_x1 - rem_x0), 0.6; atol=1e-12, rtol=1e-12)
    @test simulate!(rem_awh_sim, 4) === rem_awh_sim
    @test simulate!(rem_awh_sim, 5) === rem_awh_sim
    @test_throws ArgumentError simulate!(rem_awh_sim, -1)

    # 5. Custom target distributions must be accepted and normalized
    custom_rho = [0.1, 0.2, 0.3, 0.4]
    awh_state_rho = AWHState(thermo_states; first_state=1, n_bias=10, ρ=custom_rho)
    @test isapprox(sum(awh_state_rho.rho), 1.0; atol=1e-12)
    @test awh_state_rho.rho ≈ custom_rho ./ sum(custom_rho)

    # 6. In the linear stage, Eq. (4) must use the reference N before
    #    adding the currently accumulated block (n_accum samples).
    awh_state_linear = AWHState(thermo_states; first_state=1, n_bias=20)
    awh_sim_linear = AWHSimulation(
        awh_state_linear;
        update_freq=3,
        well_tempered_factor=Inf,
        log_freq=1000
    )

    st = awh_sim_linear.state
    st.in_initial_stage = false
    st.N_eff = 11.0
    st.n_accum = 3
    st.w_seg .= [1.5, 0.5, 0.7, 0.3]
    st.rho .= fill(0.25, n_windows)
    st.log_rho .= log.(st.rho)
    st.f .= [0.0, 0.2, -0.1, 0.3]

    f_before = copy(st.f)
    N_ref = awh_sim_linear.initial_sampl_n + (st.N_eff - st.n_accum)
    delta_expected = log.((N_ref .* st.rho .+ st.w_seg) ./ (N_ref .* st.rho .+ st.n_accum .* st.rho))
    f_expected = f_before .- delta_expected
    f_expected .-= f_expected[1]

    Molly.update_awh_bias!(awh_sim_linear, 1)
    @test st.f ≈ f_expected

    # 7. Coverage in the initial stage must be evaluated over the active
    #    target support only (ρ > 0), otherwise zero-target windows can block exit.
    awh_state_cov = AWHState(thermo_states; first_state=1, n_bias=10, ρ=[0.5, 0.5, 0.0, 0.0])
    awh_sim_cov = AWHSimulation(
        awh_state_cov;
        update_freq=1,
        well_tempered_factor=Inf,
        coverage_threshold=1.0,
        log_freq=1000
    )

    st_cov = awh_sim_cov.state
    st_cov.n_accum = 1
    st_cov.w_seg .= st_cov.rho
    union!(st_cov.visited_windows, (1, 2))

    n_bias_before = st_cov.N_bias
    Molly.update_awh_bias!(awh_sim_cov, 1)
    @test st_cov.N_bias == 2n_bias_before

    # 8. Zero-mass atoms must not generate NaN velocities when swapping
    #    between states with different temperatures.
    zm_boundary = CubicBoundary(2.0u"nm")
    zm_atoms = [Atom(mass=0.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)]
    zm_coords = place_atoms(1, zm_boundary; min_dist=0.1u"nm")
    zm_nf = DistanceNeighborFinder(
        eligible=trues(1, 1),
        n_steps=1,
        dist_cutoff=1.0u"nm",
    )
    zm_sys_1 = System(
        atoms=zm_atoms,
        coords=zm_coords,
        boundary=zm_boundary,
        pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
        neighbor_finder=zm_nf
    )
    zm_sys_2 = System(
        atoms=zm_atoms,
        coords=zm_coords,
        boundary=zm_boundary,
        pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
        neighbor_finder=zm_nf
    )
    zm_intg_1 = Langevin(dt=0.005u"ps", temperature=300.0u"K", friction=0.1u"ps^-1")
    zm_intg_2 = Langevin(dt=0.005u"ps", temperature=600.0u"K", friction=0.1u"ps^-1")
    zm_states = [ThermoState(zm_sys_1, zm_intg_1), ThermoState(zm_sys_2, zm_intg_2)]
    awh_state_zm = AWHState(zm_states; first_state=1, n_bias=10)
    Molly.update_active_sys!(awh_state_zm, 2)
    @test all(isfinite, ustrip.(awh_state_zm.active_sys.velocities[1]))

    # 9. Temperature swaps with immutable velocity vectors (SVector) must
    #    rescale velocities without raising setindex! errors.
    tv_boundary = CubicBoundary(2.0u"nm")
    tv_atoms = [Atom(mass=2.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)]
    tv_coords = [SVector(0.0, 0.0, 0.0)u"nm"]
    tv_vels = [SVector(1.0, -0.5, 0.25)u"nm/ps"]
    tv_nf = DistanceNeighborFinder(eligible=trues(1, 1), n_steps=1, dist_cutoff=1.0u"nm")
    tv_sys_1 = System(
        atoms=tv_atoms,
        coords=tv_coords,
        velocities=tv_vels,
        boundary=tv_boundary,
        pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
        neighbor_finder=tv_nf,
    )
    tv_sys_2 = System(
        atoms=tv_atoms,
        coords=tv_coords,
        velocities=tv_vels,
        boundary=tv_boundary,
        pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
        neighbor_finder=tv_nf,
    )
    tv_intg_1 = Langevin(dt=0.005u"ps", temperature=300.0u"K", friction=0.1u"ps^-1")
    tv_intg_2 = Langevin(dt=0.005u"ps", temperature=600.0u"K", friction=0.1u"ps^-1")
    tv_states = [ThermoState(tv_sys_1, tv_intg_1), ThermoState(tv_sys_2, tv_intg_2)]
    awh_state_tv = AWHState(tv_states; first_state=1, n_bias=10)
    v_before = awh_state_tv.active_sys.velocities[1]
    β_scale = sqrt(awh_state_tv.λ_β[1] / awh_state_tv.λ_β[2])
    Molly.update_active_sys!(awh_state_tv, 2)
    v_after = awh_state_tv.active_sys.velocities[1]
    @test all(isapprox.(ustrip.(v_after), ustrip.(v_before .* β_scale); atol=1e-12, rtol=1e-12))

    # 10. Swapping to a new λ state must synchronize cached system fields
    #     that are used by integrators (masses/total_mass/df) and neighbor finder.
    ms_boundary = CubicBoundary(2.0u"nm")
    ms_coords = place_atoms(2, ms_boundary; min_dist=0.2u"nm")
    ms_atoms_1 = [
        Atom(index=1, mass=1.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=1.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
    ]
    ms_atoms_2 = [
        Atom(index=1, mass=2.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=2.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
    ]
    ms_nf_1 = DistanceNeighborFinder(eligible=trues(2, 2), n_steps=1, dist_cutoff=1.0u"nm")
    ms_nf_2 = DistanceNeighborFinder(eligible=trues(2, 2), n_steps=1, dist_cutoff=1.0u"nm")
    ms_intg = Langevin(dt=0.005u"ps", temperature=300.0u"K", friction=0.1u"ps^-1")
    ms_sys_1 = System(
        atoms=ms_atoms_1,
        coords=ms_coords,
        boundary=ms_boundary,
        pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
        neighbor_finder=ms_nf_1,
    )
    ms_sys_2 = System(
        atoms=ms_atoms_2,
        coords=ms_coords,
        boundary=ms_boundary,
        pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
        neighbor_finder=ms_nf_2,
    )
    ms_states = [ThermoState(ms_sys_1, ms_intg), ThermoState(ms_sys_2, ms_intg)]
    awh_state_ms = AWHState(ms_states; first_state=1, n_bias=10)
    Molly.update_active_sys!(awh_state_ms, 2)

    @test awh_state_ms.active_sys.neighbor_finder.dist_cutoff == 1.0u"nm"
    @test masses(awh_state_ms.active_sys) == mass.(awh_state_ms.active_sys.atoms)
    @test awh_state_ms.active_sys.total_mass == sum(mass.(awh_state_ms.active_sys.atoms))
    @test awh_state_ms.active_sys.df == ms_sys_2.df

    # 10b. update_active_sys! must support heterogeneous interaction and
    #      integrator types across states without type-conversion failures.
    sw_boundary = CubicBoundary(2.0u"nm")
    sw_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.45, 0.0, 0.0)u"nm"]
    sw_atoms = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
    ]
    sw_nf = DistanceNeighborFinder(eligible=trues(2, 2), n_steps=1, dist_cutoff=1.5u"nm")
    sw_sys_1 = System(
        atoms=sw_atoms,
        coords=sw_coords,
        boundary=sw_boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=sw_nf,
    )
    sw_sys_2 = System(
        atoms=sw_atoms,
        coords=sw_coords,
        boundary=sw_boundary,
        pairwise_inters=(SoftSphere(use_neighbors=true),),
        neighbor_finder=sw_nf,
    )
    sw_intg_1 = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    sw_intg_2 = VelocityVerlet(dt=0.001u"ps", remove_CM_motion=0)
    sw_states = [
        ThermoState(sw_sys_1, sw_intg_1; temperature=300.0u"K"),
        ThermoState(sw_sys_2, sw_intg_2; temperature=300.0u"K"),
    ]
    awh_state_sw = AWHState(sw_states; first_state=1, n_bias=10)
    Molly.update_active_sys!(awh_state_sw, 2)
    @test awh_state_sw.active_sys.pairwise_inters == awh_state_sw.state_pairwise_inters[2]
    @test awh_state_sw.active_intg === awh_state_sw.λ_integrators[2]
    @test isfinite(ustrip(Molly.process_sample(awh_state_sw)))
    Molly.update_active_sys!(awh_state_sw, 1)
    @test awh_state_sw.active_sys.pairwise_inters == awh_state_sw.state_pairwise_inters[1]
    @test awh_state_sw.active_intg === awh_state_sw.λ_integrators[1]

    # 11. Pairwise interactions that do not use neighbor lists must not be
    #     double-counted by the AlchemicalPartition split.
    nn_boundary = CubicBoundary(2.0u"nm")
    nn_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.4, 0.0, 0.0)u"nm", SVector(0.8, 0.0, 0.0)u"nm"]
    nn_atoms = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
        Atom(index=3, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
    ]
    nn_nf = DistanceNeighborFinder(eligible=trues(3, 3), n_steps=1, dist_cutoff=1.5u"nm")
    nn_sys = System(
        atoms=nn_atoms,
        coords=nn_coords,
        boundary=nn_boundary,
        pairwise_inters=(LennardJones(use_neighbors=false),),
        neighbor_finder=nn_nf,
        strictness=:nowarn,
    )
    nn_intg = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    nn_states = [ThermoState(nn_sys, nn_intg), ThermoState(nn_sys, nn_intg)]
    nn_part = AlchemicalPartition(nn_states)
    nn_ref = potential_energy(nn_sys)
    nn_eval = evaluate_energy!(nn_part, nn_sys.coords, nn_sys.boundary, 1; force_recompute=true)
    @test nn_eval ≈ nn_ref
    nn_energies = [zero(nn_eval) for _ in 1:length(nn_states)]
    nn_energies_out = evaluate_energy_all!(nn_part, nn_sys.coords, nn_sys.boundary, nn_energies)
    @test nn_energies_out === nn_energies
    @test all(E -> E ≈ nn_ref, nn_energies)
    nn_energies_alloc = evaluate_energy_all!(nn_part, nn_sys.coords, nn_sys.boundary)
    @test nn_energies_alloc ≈ nn_energies

    # 11b. Perturbed atom detection must not depend on Atom.index values,
    #      which are optional metadata and may be non-unique.
    idx_boundary = CubicBoundary(2.0u"nm")
    idx_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.4, 0.0, 0.0)u"nm", SVector(0.8, 0.0, 0.0)u"nm"]
    idx_nf = DistanceNeighborFinder(eligible=trues(3, 3), n_steps=1, dist_cutoff=1.5u"nm")
    idx_atoms_1 = [Atom(mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0) for _ in 1:3]
    idx_atoms_2 = [Atom(mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.6) for _ in 1:3]
    idx_sys_1 = System(
        atoms=idx_atoms_1,
        coords=idx_coords,
        boundary=idx_boundary,
        pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
        neighbor_finder=idx_nf,
    )
    idx_sys_2 = System(
        atoms=idx_atoms_2,
        coords=idx_coords,
        boundary=idx_boundary,
        pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
        neighbor_finder=idx_nf,
    )
    idx_intg = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    idx_states = [ThermoState(idx_sys_1, idx_intg), ThermoState(idx_sys_2, idx_intg)]
    idx_part = AlchemicalPartition(idx_states)
    idx_direct = potential_energy(idx_sys_2)
    idx_eval = evaluate_energy!(idx_part, idx_sys_2.coords, idx_sys_2.boundary, 2; force_recompute=true)
    @test idx_eval ≈ idx_direct

    # 11c. If pairwise interactions differ between states but atoms are unchanged,
    #      λ-specific pairwise terms must still be evaluated correctly.
    diff_boundary = CubicBoundary(2.0u"nm")
    diff_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.45, 0.0, 0.0)u"nm"]
    diff_nf = DistanceNeighborFinder(eligible=trues(2, 2), n_steps=1, dist_cutoff=1.5u"nm")
    diff_atoms = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
    ]
    diff_sys_1 = System(
        atoms=diff_atoms,
        coords=diff_coords,
        boundary=diff_boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=diff_nf,
    )
    diff_sys_2 = System(
        atoms=diff_atoms,
        coords=diff_coords,
        boundary=diff_boundary,
        pairwise_inters=(SoftSphere(use_neighbors=true),),
        neighbor_finder=diff_nf,
    )
    diff_intg = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    diff_states = [ThermoState(diff_sys_1, diff_intg), ThermoState(diff_sys_2, diff_intg)]
    diff_part = AlchemicalPartition(diff_states)
    diff_direct = potential_energy(diff_sys_2)
    diff_eval = evaluate_energy!(diff_part, diff_sys_2.coords, diff_sys_2.boundary, 2; force_recompute=true)
    @test diff_eval ≈ diff_direct

    # 11d. The master-energy cache must invalidate when coordinates are
    #      mutated in-place without changing the array object identity.
    cache_boundary = CubicBoundary(2.0u"nm")
    cache_atoms = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.2u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.2u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0),
    ]
    cache_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.9, 0.0, 0.0)u"nm"]
    cache_sys = System(
        atoms=cache_atoms,
        coords=cache_coords,
        boundary=cache_boundary,
        pairwise_inters=(LennardJones(use_neighbors=false),),
        neighbor_finder=NoNeighborFinder(),
    )
    cache_intg = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    cache_states = [ThermoState(cache_sys, cache_intg), ThermoState(cache_sys, cache_intg)]
    cache_part = AlchemicalPartition(cache_states)
    coords_mut = copy(cache_coords)
    E_cache_initial = evaluate_energy!(cache_part, coords_mut, cache_boundary, 1; force_recompute=true)
    coords_mut[2] = SVector(0.7, 0.0, 0.0)u"nm"
    E_cache_mut = evaluate_energy!(cache_part, coords_mut, cache_boundary, 1; force_recompute=false)
    E_cache_direct_mut = potential_energy(System(cache_sys; coords=coords_mut))
    @test E_cache_mut ≈ E_cache_direct_mut
    @test E_cache_mut != E_cache_initial

    # 11e. The master-energy cache must invalidate when only the boundary changes.
    boundary_new = CubicBoundary(1.0u"nm")
    E_cache_boundary = evaluate_energy!(cache_part, coords_mut, boundary_new, 1; force_recompute=false)
    E_cache_direct_boundary = potential_energy(System(cache_sys; coords=coords_mut, boundary=boundary_new))
    @test E_cache_boundary ≈ E_cache_direct_boundary

    # 11f. AlchemicalPartition enforces a shared neighbor-finder policy
    #      across all λ windows (and target), rejecting mismatched setups.
    nf_state_boundary = CubicBoundary(2.0u"nm")
    nf_state_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.9, 0.0, 0.0)u"nm"]
    nf_state_atoms_1 = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
    ]
    nf_state_atoms_2 = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.6),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
    ]
    nf_small = DistanceNeighborFinder(eligible=trues(2, 2), n_steps=1, dist_cutoff=0.8u"nm")
    nf_large = DistanceNeighborFinder(eligible=trues(2, 2), n_steps=1, dist_cutoff=1.2u"nm")
    nf_state_sys_1 = System(
        atoms=nf_state_atoms_1,
        coords=nf_state_coords,
        boundary=nf_state_boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=nf_small,
    )
    nf_state_sys_2 = System(
        atoms=nf_state_atoms_2,
        coords=nf_state_coords,
        boundary=nf_state_boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=nf_large,
    )
    nf_state_intg = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    nf_states = [ThermoState(nf_state_sys_1, nf_state_intg), ThermoState(nf_state_sys_2, nf_state_intg)]
    @test_throws ArgumentError AlchemicalPartition(nf_states)
    @test_throws ArgumentError AWHState(nf_states; n_bias=10)

    # 12. AWH must support NoNeighborFinder when pairwise interactions do not use neighbor lists.
    no_nf_boundary = CubicBoundary(2.0u"nm")
    no_nf_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.5, 0.0, 0.0)u"nm"]
    no_nf_atoms = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
    ]
    no_nf_sys = System(
        atoms=no_nf_atoms,
        coords=no_nf_coords,
        boundary=no_nf_boundary,
        pairwise_inters=(LennardJones(use_neighbors=false),),
        neighbor_finder=NoNeighborFinder(),
    )
    no_nf_intg = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    no_nf_states = [ThermoState(no_nf_sys, no_nf_intg), ThermoState(no_nf_sys, no_nf_intg)]
    awh_state_no_nf = AWHState(no_nf_states; n_bias=5)
    @test awh_state_no_nf.active_sys.neighbor_finder isa NoNeighborFinder
    pe_no_nf = Molly.process_sample(awh_state_no_nf)
    @test isfinite(ustrip(pe_no_nf))

    # 12b. Coverage accounting mode must support both reweighted and physical tracking.
    awh_state_no_nf_rw = AWHState(no_nf_states; n_bias=5)
    Molly.process_sample(awh_state_no_nf_rw; coverage_type=:reweighted)
    @test awh_state_no_nf_rw.visited_windows == Set([1, 2])

    awh_state_no_nf_phys = AWHState(no_nf_states; n_bias=5)
    Molly.process_sample(awh_state_no_nf_phys; coverage_type=:physical)
    @test awh_state_no_nf_phys.visited_windows == Set([awh_state_no_nf_phys.active_idx])

    # 13. Coverage criteria must require at least one visited window.
    awh_state_cov_min = AWHState(thermo_states; first_state=1, n_bias=10)
    awh_sim_cov_min = AWHSimulation(
        awh_state_cov_min;
        update_freq=1,
        well_tempered_factor=Inf,
        coverage_threshold=0.4,
        log_freq=1000,
    )
    st_cov_min = awh_sim_cov_min.state
    st_cov_min.n_accum = 1
    st_cov_min.w_seg .= st_cov_min.rho
    n_bias_before_min = st_cov_min.N_bias
    Molly.update_awh_bias!(awh_sim_cov_min, 1)
    @test st_cov_min.N_bias == n_bias_before_min
    @test st_cov_min.in_initial_stage

    # 14. Invalid AWHSimulation inputs should throw.
    @test_throws ArgumentError AWHSimulation(awh_state; log_freq=0)
    @test_throws ArgumentError AWHSimulation(awh_state; update_freq=0)
    @test_throws ArgumentError AWHSimulation(awh_state; num_md_steps=0)
    @test_throws ArgumentError AWHSimulation(awh_state; coverage_threshold=0.0)
    @test AWHSimulation(awh_state; coverage_type=:physical).coverage_type == :physical
    @test_throws ArgumentError AWHSimulation(awh_state; coverage_type=:invalid)

    # 15-17. PMF deconvolution validation and numerical safeguards.
    pmf_boundary = CubicBoundary(2.0u"nm")
    pmf_atoms = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0),
    ]
    pmf_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.5, 0.0, 0.0)u"nm"]
    pmf_nf = DistanceNeighborFinder(eligible=trues(2, 2), n_steps=1, dist_cutoff=1.5u"nm")
    pmf_cv = CalcDist([1], [2], CalcSingleDist(:raw), :pbc)
    pmf_bias_1 = BiasPotential(pmf_cv, SquareBias(100.0u"kJ * mol^-1 * nm^-2", 0.4u"nm"))
    pmf_bias_2 = BiasPotential(pmf_cv, SquareBias(100.0u"kJ * mol^-1 * nm^-2", 0.6u"nm"))
    pmf_sys_1 = System(
        atoms=pmf_atoms,
        coords=pmf_coords,
        boundary=pmf_boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        general_inters=(pmf_bias_1,),
        neighbor_finder=pmf_nf,
    )
    pmf_sys_2 = System(
        atoms=pmf_atoms,
        coords=pmf_coords,
        boundary=pmf_boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        general_inters=(pmf_bias_2,),
        neighbor_finder=pmf_nf,
    )
    pmf_target_sys = System(
        atoms=pmf_atoms,
        coords=pmf_coords,
        boundary=pmf_boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=pmf_nf,
    )
    pmf_intg = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    pmf_states = [ThermoState(pmf_sys_1, pmf_intg), ThermoState(pmf_sys_2, pmf_intg)]
    pmf_target_state = ThermoState(pmf_target_sys, pmf_intg)
    pmf_awh_state = AWHState(pmf_states; target_state=pmf_target_state, n_bias=10)

    @test_throws ArgumentError AWHSimulation(
        pmf_awh_state;
        pmf_grid=((0.0, 0.0), (1.0, 1.0), (10, 10)),
    )

    pmf_calc_unitful = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0u"nm",), (1.0u"nm",), (10,));
        cv_func=(coords) -> (norm(coords[2] - coords[1]),),
    )
    Molly.process_sample(pmf_awh_state)
    Molly.update_pmf!(
        pmf_calc_unitful,
        pmf_awh_state,
        pmf_awh_state.active_sys.coords;
        box_volume=ustrip(volume(pmf_awh_state.active_sys.boundary)),
        apply_forgetting=false,
    )
    @test pmf_calc_unitful.sample_count == 1
    @test all(isfinite, pmf_calc_unitful.numerator_hist)

    pmf_calc_guard = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (0.5,),
    )
    pmf_awh_state.scratch_z .= -Inf
    hist_before = copy(pmf_calc_guard.numerator_hist)
    count_before = pmf_calc_guard.sample_count
    Molly.update_pmf!(
        pmf_calc_guard,
        pmf_awh_state,
        pmf_awh_state.active_sys.coords;
        box_volume=ustrip(volume(pmf_awh_state.active_sys.boundary)),
        apply_forgetting=false,
    )
    @test pmf_calc_guard.sample_count == count_before
    @test pmf_calc_guard.numerator_hist == hist_before

    # 18. Additional AWHState/AWHSimulation input validation.
    @test_throws ArgumentError AWHState(ThermoState[]; n_bias=10, ρ=Float64[])
    @test_throws ArgumentError AWHState(thermo_states; first_state=0, n_bias=10)
    @test_throws ArgumentError AWHState(thermo_states; first_state=n_windows + 1, n_bias=10)
    @test_throws ArgumentError AWHState(thermo_states; n_bias=0)
    @test_throws ArgumentError AWHState(thermo_states; n_bias=10, ρ=[1.0, 0.0])
    @test_throws ArgumentError AWHState(thermo_states; n_bias=10, ρ=[0.25, NaN, 0.25, 0.5])
    @test_throws ArgumentError AWHState(thermo_states; n_bias=10, ρ=[0.5, -0.1, 0.4, 0.2])
    @test_throws ArgumentError AWHState(thermo_states; n_bias=10, ρ=zeros(4))
    @test_throws ArgumentError AWHSimulation(awh_state; significant_weight=-0.1)
    @test_throws ArgumentError AWHSimulation(awh_state; coverage_threshold=1.1)
    @test_throws ArgumentError AWHSimulation(awh_state; well_tempered_factor=0.0)

    # 19. PMF constructor and PMF-related argument validation.
    @test_throws ErrorException AWHSimulation(
        AWHState(pmf_states; n_bias=10);
        pmf_grid=((0.0u"nm",), (1.0u"nm",), (10,)),
    )
    @test_throws ArgumentError Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,));
        cv_func=(coords) -> (0.5,),
    )
    @test_throws ArgumentError Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((1.0,), (1.0,), (10,));
        cv_func=(coords) -> (0.5,),
    )
    @test_throws ArgumentError Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (0.5,),
        is_periodic=(true, false),
    )
    @test_throws ArgumentError Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0, 0.0), (1.0, 1.0), (10, 10));
        cv_func=(coords) -> (0.5,),
    )
    no_bias_states = [ThermoState(rem_sys, rem_intg; temperature=300.0u"K")]
    no_bias_target_state = ThermoState(rem_sys, rem_intg; temperature=300.0u"K")
    no_bias_awh_state = AWHState(no_bias_states; target_state=no_bias_target_state, n_bias=5)
    @test_throws ErrorException Molly.AWHPMFDeconvolution(
        no_bias_awh_state,
        ((0.0,), (1.0,), (10,)),
    )

    # 20. PMF auto-detection and periodic indexing paths should run.
    pmf_calc_auto = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0u"nm",), (1.0u"nm",), (10,)),
    )
    Molly.process_sample(pmf_awh_state)
    Molly.update_pmf!(
        pmf_calc_auto,
        pmf_awh_state,
        pmf_awh_state.active_sys.coords;
        box_volume=ustrip(volume(pmf_awh_state.active_sys.boundary)),
        apply_forgetting=false,
    )
    @test pmf_calc_auto.sample_count == 1

    pmf_calc_periodic = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (1.2,),
        is_periodic=(true,),
    )
    Molly.process_sample(pmf_awh_state)
    Molly.update_pmf!(
        pmf_calc_periodic,
        pmf_awh_state,
        pmf_awh_state.active_sys.coords;
        box_volume=ustrip(volume(pmf_awh_state.active_sys.boundary)),
        apply_forgetting=false,
    )
    rel_periodic = (1.2 - pmf_calc_periodic.min_vals[1]) / pmf_calc_periodic.bin_widths[1]
    idx_periodic = Int(floor(rel_periodic)) + 1
    idx_periodic = mod(idx_periodic - 1, pmf_calc_periodic.shape[1]) + 1
    @test pmf_calc_periodic.sample_count == 1
    @test pmf_calc_periodic.denominator_hist[idx_periodic] == 1.0
    @test count(>(0.0), pmf_calc_periodic.denominator_hist) == 1

    # 21. PMF update guards and forgetting behavior.
    pmf_calc_nan_cv = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (NaN,),
    )
    Molly.process_sample(pmf_awh_state)
    @test_throws ArgumentError Molly.update_pmf!(
        pmf_calc_nan_cv,
        pmf_awh_state,
        pmf_awh_state.active_sys.coords;
        box_volume=ustrip(volume(pmf_awh_state.active_sys.boundary)),
        apply_forgetting=false,
    )

    pmf_calc_nonfinite = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (0.5,),
    )
    Molly.process_sample(pmf_awh_state)
    pmf_calc_nonfinite.target_beta = Inf
    hist_before_nonfinite = copy(pmf_calc_nonfinite.numerator_hist)
    count_before_nonfinite = pmf_calc_nonfinite.sample_count
    Molly.update_pmf!(
        pmf_calc_nonfinite,
        pmf_awh_state,
        pmf_awh_state.active_sys.coords;
        box_volume=ustrip(volume(pmf_awh_state.active_sys.boundary)),
        apply_forgetting=false,
    )
    @test pmf_calc_nonfinite.sample_count == count_before_nonfinite
    @test pmf_calc_nonfinite.numerator_hist == hist_before_nonfinite

    pmf_calc_overflow = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (0.5,),
    )
    pmf_awh_state.scratch_z .= -1000.0
    pmf_calc_overflow.target_beta = 0.0
    hist_before_overflow = copy(pmf_calc_overflow.numerator_hist)
    count_before_overflow = pmf_calc_overflow.sample_count
    Molly.update_pmf!(
        pmf_calc_overflow,
        pmf_awh_state,
        pmf_awh_state.active_sys.coords;
        box_volume=ustrip(volume(pmf_awh_state.active_sys.boundary)),
        apply_forgetting=false,
    )
    @test pmf_calc_overflow.sample_count == count_before_overflow
    @test pmf_calc_overflow.numerator_hist == hist_before_overflow

    # 21b. NaN target energies should be guarded via typemax fallback in PMF updates.
    pmf_nan_target_atoms = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=NaN * u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0),
    ]
    pmf_nan_target_sys = System(
        atoms=pmf_nan_target_atoms,
        coords=pmf_coords,
        boundary=pmf_boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=pmf_nf,
    )
    pmf_nan_target_state = ThermoState(pmf_nan_target_sys, pmf_intg)
    pmf_awh_state_nan_target = AWHState(pmf_states; target_state=pmf_nan_target_state, n_bias=10)
    pmf_calc_nan_target = Molly.AWHPMFDeconvolution(
        pmf_awh_state_nan_target,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (0.5,),
    )
    Molly.process_sample(pmf_awh_state_nan_target)
    hist_before_nan_target = copy(pmf_calc_nan_target.numerator_hist)
    count_before_nan_target = pmf_calc_nan_target.sample_count
    Molly.update_pmf!(
        pmf_calc_nan_target,
        pmf_awh_state_nan_target,
        pmf_awh_state_nan_target.active_sys.coords;
        box_volume=ustrip(volume(pmf_awh_state_nan_target.active_sys.boundary)),
        apply_forgetting=false,
    )
    @test pmf_calc_nan_target.sample_count == count_before_nan_target
    @test pmf_calc_nan_target.numerator_hist == hist_before_nan_target

    pmf_calc_forgetting = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (0.5,),
    )
    pmf_calc_forgetting.numerator_hist[6] = 2.0
    pmf_calc_forgetting.denominator_hist[6] = 4.0
    Molly.process_sample(pmf_awh_state)
    Molly.update_pmf!(
        pmf_calc_forgetting,
        pmf_awh_state,
        pmf_awh_state.active_sys.coords;
        weight_factor=0.5,
        box_volume=ustrip(volume(pmf_awh_state.active_sys.boundary)),
        apply_forgetting=true,
    )
    @test pmf_calc_forgetting.sample_count == 1
    @test isapprox(pmf_calc_forgetting.denominator_hist[6], 3.0; atol=1e-12, rtol=0.0)

    # 22. PMF extraction utility must preserve Inf in unsampled bins and zero the minimum.
    pmf_calc_profile = Molly.AWHPMFDeconvolution(
        pmf_awh_state,
        ((0.0,), (1.0,), (10,));
        cv_func=(coords) -> (0.5,),
    )
    pmf_calc_profile.numerator_hist .= 0.0
    pmf_calc_profile.numerator_hist[2] = 4.0
    pmf_calc_profile.numerator_hist[5] = 1.0
    pmf_profile = calc_pmf(pmf_calc_profile)
    @test isinf(pmf_profile[1])
    @test isapprox(pmf_profile[2], 0.0; atol=1e-12, rtol=0.0)
    @test isapprox(pmf_profile[5], log(4.0); atol=1e-12, rtol=0.0)

    # 23. process_sample must fall back to uniform weights when all log-weights underflow.
    awh_state_underflow = AWHState(thermo_states; first_state=1, n_bias=10)
    awh_state_underflow.f .= -Inf
    Molly.process_sample(awh_state_underflow)
    uniform_weights = fill(1.0 / length(awh_state_underflow.w_last), length(awh_state_underflow.w_last))
    @test awh_state_underflow.w_last ≈ uniform_weights

    # 23b. NaN state energies should be converted to finite potentials in process_sample.
    nan_energy_boundary = CubicBoundary(2.0u"nm")
    nan_energy_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(0.5, 0.0, 0.0)u"nm"]
    nan_energy_atoms_ok = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0),
    ]
    nan_energy_atoms_bad = [
        Atom(index=1, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=NaN * u"kJ * mol^-1", λ=1.0),
        Atom(index=2, mass=12.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0),
    ]
    nan_energy_sys_ok = System(
        atoms=nan_energy_atoms_ok,
        coords=nan_energy_coords,
        boundary=nan_energy_boundary,
        pairwise_inters=(LennardJones(use_neighbors=false),),
        neighbor_finder=NoNeighborFinder(),
    )
    nan_energy_sys_bad = System(
        atoms=nan_energy_atoms_bad,
        coords=nan_energy_coords,
        boundary=nan_energy_boundary,
        pairwise_inters=(LennardJones(use_neighbors=false),),
        neighbor_finder=NoNeighborFinder(),
    )
    nan_energy_intg = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")
    nan_energy_states = [
        ThermoState(nan_energy_sys_ok, nan_energy_intg),
        ThermoState(nan_energy_sys_bad, nan_energy_intg),
    ]
    awh_state_nan_energy = AWHState(nan_energy_states; first_state=1, n_bias=5)
    Molly.process_sample(awh_state_nan_energy)
    @test !any(isnan, awh_state_nan_energy.scratch_potentials)

    # 24. simulate! should execute PMF updates, and extract_awh_data must deep-copy state.
    pmf_awh_state_loop = AWHState(pmf_states; target_state=pmf_target_state, n_bias=4)
    pmf_awh_sim_loop = AWHSimulation(
        pmf_awh_state_loop;
        num_md_steps=1,
        update_freq=1,
        well_tempered_factor=Inf,
        log_freq=1,
        pmf_grid=((0.0u"nm",), (1.0u"nm",), (10,)),
    )
    simulate!(pmf_awh_sim_loop, 5)
    @test pmf_awh_sim_loop.pmf_calc.sample_count > 0
    @test sum(pmf_awh_sim_loop.pmf_calc.denominator_hist) > 0
    pmf_loop = calc_pmf(pmf_awh_sim_loop.pmf_calc)
    @test size(pmf_loop) == (10,)
    finite_loop_vals = pmf_loop[isfinite.(pmf_loop)]
    @test !isempty(finite_loop_vals)
    @test isapprox(minimum(finite_loop_vals), 0.0; atol=1e-12, rtol=0.0)
    @test any(isinf, pmf_loop)

    awh_data = extract_awh_data(pmf_awh_sim_loop)
    awh_data_before = (
        f = copy(awh_data.f),
        rho = copy(awh_data.rho),
        log_rho = copy(awh_data.log_rho),
        stats = deepcopy(awh_data.stats),
    )
    pmf_awh_sim_loop.state.f .+= 1
    pmf_awh_sim_loop.state.rho .*= 0.5
    pmf_awh_sim_loop.state.log_rho .= log.(pmf_awh_sim_loop.state.rho)
    if !isempty(pmf_awh_sim_loop.state.stats.step_indices)
        pmf_awh_sim_loop.state.stats.step_indices[1] = -999
    end
    if !isempty(pmf_awh_sim_loop.state.stats.f_history)
        pmf_awh_sim_loop.state.stats.f_history[1][1] += 1
    end
    @test awh_data.f == awh_data_before.f
    @test awh_data.rho == awh_data_before.rho
    @test awh_data.log_rho == awh_data_before.log_rho
    @test awh_data.stats.step_indices == awh_data_before.stats.step_indices
    @test awh_data.stats.f_history == awh_data_before.stats.f_history

    # 25. Convergence threshold should stop simulate! early in linear stage.
    conv_awh_state = AWHState(rem_states; n_bias=5)
    conv_awh_sim = AWHSimulation(
        conv_awh_state;
        num_md_steps=1,
        update_freq=1,
        well_tempered_factor=Inf,
        log_freq=1,
    )
    conv_awh_sim.state.in_initial_stage = false
    simulate!(conv_awh_sim, 8; convergence_threshold=0.0)
    @test conv_awh_sim.state.N_eff == 1
    @test conv_awh_sim.state.n_accum == 0

    # 26. simulate! should synchronize :awh_logger.active_idx when present.
    awh_logger = Molly.AWHEnsembleLogger(Float64, Float64, Float64, 1)
    awh_logger.active_idx = 99
    logger_sys = System(rem_sys; loggers=(awh_logger=awh_logger,))
    logger_states = [ThermoState(logger_sys, rem_intg; temperature=300.0u"K")]
    logger_awh_state = AWHState(logger_states; n_bias=5)
    logger_awh_sim = AWHSimulation(
        logger_awh_state;
        num_md_steps=1,
        update_freq=1,
        well_tempered_factor=Inf,
        log_freq=1,
    )
    simulate!(logger_awh_sim, 1)
    @test logger_awh_sim.state.active_sys.loggers.awh_logger.active_idx == logger_awh_sim.state.active_idx
end
