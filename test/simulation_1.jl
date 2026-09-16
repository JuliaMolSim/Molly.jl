const temp_fp_dcd  = tempname(cleanup=true) * ".dcd"
const temp_fp_trr  = tempname(cleanup=true) * ".trr"
const temp_fp_pdb  = tempname(cleanup=true) * ".pdb"
const temp_fp_xyz  = tempname(cleanup=true) * ".xyz"
const temp_fp_mol2 = tempname(cleanup=true) * ".mol2"
const temp_fp_mp4  = tempname(cleanup=true) * ".mp4"

@testset "Lennard-Jones 2D" begin
    for AT in array_list
        n_atoms = 10
        n_steps = 20_000
        temp = 100.0u"K"
        boundary = RectangularBoundary(2.0u"nm")
        atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
                 for i in 1:n_atoms]
        simulator = VelocityVerlet(dt=0.001u"ps", coupling=(AndersenThermostat(temp, 10.0u"ps"),))
        gen_temp_wrapper(s, neighbors, step_n, buffers; kwargs...) =
            temperature(s; kin_tensor=buffers.kin_tensor)

        if Molly.uses_gpu_neighbor_finder(AT)
            neighbor_finder = GPUNeighborFinder(
                n_atoms=n_atoms,
                dist_cutoff=2.0u"nm",
                device_vector_type=AT{Int32, 1},
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

        simulate!(sys, simulator, n_steps; n_threads=1)

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
    n_steps = 5_000
    n_frames = (n_steps ÷ 100) + 1
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=(AndersenThermostat(temp, 10.0u"ps"),))

    TV = typeof(random_velocity(10.0u"g/mol", temp))
    TP = typeof(0.2u"kJ * mol^-1")

    V(sys, args...; kwargs...) = sys.velocities
    pot_obs(sys, neighbors, step_n, buffers; kwargs...) = potential_energy(sys, neighbors, step_n)
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

        simulate!(s, simulator, n_steps; n_threads=n_threads, show_progress=true)

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
        @test Int(length(traj)) == n_frames
        frame = read(traj)
        @test length(frame) == 100
        # Chemfiles does not write velocities to DCD files
        @test size(Chemfiles.positions(frame)) == (3, 100)
        @test !iszero(sum(Array(Chemfiles.positions(frame))))
        @test Chemfiles.lengths(Chemfiles.UnitCell(frame)) == [20.0, 20.0, 20.0]
        boundary_dcd = Molly.boundary_from_chemfiles(Chemfiles.UnitCell(frame))
        @test boundary_dcd.side_lengths ≈ SVector(2.0, 2.0, 2.0)u"nm"

        traj = Chemfiles.Trajectory(temp_fp_trr)
        rm(temp_fp_trr)
        @test Int(length(traj)) == n_frames
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
        @test BioStructures.countmodels(traj) == n_frames
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
            float_type_high=Measurement{Float64},
            strictness=:nowarn
        )

        @test typeof(potential_energy(sys_unc; n_threads=n_threads)) ==
                            typeof((1.0 ± 0.1)u"kJ * mol^-1")
        @test abs(potential_energy(sys_unc; n_threads=n_threads) -
                            potential_energy(s; n_threads=n_threads)) < 0.1u"kJ * mol^-1"
        @test typeof(kinetic_energy(sys_unc)) == typeof((1.0 ± 0.1)u"kJ * mol^-1")
        @test typeof(temperature(sys_unc)) == typeof((1.0 ± 0.1)u"K")
        @test abs(temperature(sys_unc) - temperature(s)) < 0.1u"K"
        @test eltype(eltype(forces(sys_unc; n_threads=n_threads))) ==
                            typeof((1.0 ± 0.1)u"kJ * mol^-1 * nm^-1")

        simulator_unc = VelocityVerlet(dt=0.002u"ps")
        simulate!(sys_unc, simulator_unc, 1; n_threads=n_threads, run_loggers=false)
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

    simulate!(s, simulator, n_steps ÷ 2)
    simulate!(s, simulator, n_steps ÷ 2; run_loggers=:skipstart)

    @test length(values(s.loggers.coords)) == 21
    @test maximum(distances(s.coords, boundary)) > 5.0u"nm"

    run_visualize_tests && visualize(s.loggers.coords, boundary, temp_fp_mp4)
end

@testset "Lennard-Jones simulators" begin
    n_atoms = 100
    n_steps = 2_000
    dt = 0.002u"ps"
    sim_time = n_steps * dt
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
             for i in 1:n_atoms]
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    coords_nan = copy(coords)
    coords_nan[10] = SVector(NaN, 1.0, 1.0)u"nm"
    simulators = [
        Verlet(dt=dt, coupling=(AndersenThermostat(temp, 10.0u"ps"),)),
        StormerVerlet(dt=dt),
        Langevin(dt=dt, temperature=temp, friction=1.0u"ps^-1"),
        OverdampedLangevin(dt=dt, temperature=temp, friction=10.0u"ps^-1"),
    ]

    for AT in array_list
        if Molly.uses_gpu_neighbor_finder(AT)
            neighbor_finder = GPUNeighborFinder(
                n_atoms=n_atoms,
                dist_cutoff=2.0u"nm",
                device_vector_type=AT{Int32, 1},
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
            simulate!(sys, simulator, sim_time; n_threads=1)
        end
        @test_throws ArgumentError simulate!(sys, simulators[1], 1; n_threads=1, strictness=:wrong)

        sys_nan = System(
            atoms=to_device(atoms, AT),
            coords=to_device(coords_nan, AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=neighbor_finder,
            strictness=:nowarn,
        )
        @test_throws NaNSimulationError System(
            atoms=to_device(atoms, AT),
            coords=to_device(coords_nan, AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=neighbor_finder,
            strictness=:error,
        )
        for simulator in simulators
            @test_throws NaNSimulationError simulate!(sys_nan, simulator, sim_time; n_threads=1,
                                                      check_nans=true)
        end
    end

    @test Molly.calc_n_steps(n_steps, dt) == n_steps
    @test Molly.calc_n_steps(sim_time, dt) == n_steps
    @test Molly.calc_n_steps(ustrip(sim_time), ustrip(dt)) == n_steps
    @test_throws ArgumentError Molly.calc_n_steps(ustrip(sim_time), dt)
end

# The fields required by the interactions used below are present on the type
struct AtomWithFields{T, M, S, E}
    index::Int32
    mass::M
    charge::T
    σ::S
    ϵ::E
    λ::T
end

# The mass and charge are read through `mass` and `charge`, so they do not have to
# be fields with those names
struct AtomWithAccessors{T, M, S, E}
    m::M
    q::T
    σ::S
    ϵ::E
    λ::T
end

Molly.mass(a::AtomWithAccessors) = a.m
Molly.charge(a::AtomWithAccessors) = a.q

@testset "Custom atom types" begin
    n_atoms = 100
    n_steps = 100
    dt = 0.001u"ps"
    temp = 100.0u"K"
    boundary = CubicBoundary(4.0u"nm")
    coords_start = place_diatomics(n_atoms ÷ 2, boundary, 0.2u"nm"; min_dist=0.2u"nm")
    vels_start = [random_velocity(10.0u"g/mol", temp; rng=Xoshiro(i)) for i in 1:n_atoms]
    bonds = [HarmonicBond(k=10_000.0u"kJ * mol^-1 * nm^-2", r0=0.2u"nm") for _ in 1:(n_atoms ÷ 2)]
    eligible = trues(n_atoms, n_atoms)
    for i in 1:2:n_atoms
        eligible[i, i + 1] = false
        eligible[i + 1, i] = false
    end
    charge_i(i) = (isodd(i) ? 0.2 : -0.2)
    make_ref_atom(i) = Atom(index=i, mass=10.0u"g/mol", charge=charge_i(i),
                            σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    custom_makers = (
        "AtomWithFields"    => i -> AtomWithFields(Int32(i), 10.0u"g/mol", charge_i(i),
                                                   0.3u"nm", 0.2u"kJ * mol^-1", 1.0),
        "AtomWithAccessors" => i -> AtomWithAccessors(10.0u"g/mol", charge_i(i),
                                                      0.3u"nm", 0.2u"kJ * mol^-1", 1.0),
    )

    # The narrowed list of atom fields read by the interactions on the CUDA path is only
    #   used when every entry is a field of the atom type, otherwise every field is used
    inters_test = (LennardJones(use_neighbors=true), Coulomb(use_neighbors=true))
    fields_test = Molly.combine_atom_fields(inters_test)
    @test Molly.resolve_atom_fields(fields_test, typeof(make_ref_atom(1))) ==
          (:σ, :ϵ, :λ, :charge)
    @test Molly.resolve_atom_fields(fields_test, typeof(custom_makers[1][2](1))) ==
          (:σ, :ϵ, :λ, :charge)
    @test Molly.resolve_atom_fields(fields_test, typeof(custom_makers[2][2](1))) ==
          (:m, :q, :σ, :ϵ, :λ)

    function build_sys(make_atom, AT, nf_type)
        if nf_type == GPUNeighborFinder
            neighbor_finder = GPUNeighborFinder(
                n_atoms=n_atoms,
                dist_cutoff=1.2u"nm",
                excluded_pairs=[(i, i + 1) for i in 1:2:n_atoms],
                device_vector_type=AT{Int32, 1},
            )
        else
            neighbor_finder = DistanceNeighborFinder(
                eligible=to_device(copy(eligible), AT),
                n_steps=10,
                dist_cutoff=1.2u"nm",
            )
        end
        cutoff = DistanceCutoff(1.0u"nm")
        return System(
            atoms=to_device([make_atom(i) for i in 1:n_atoms], AT),
            coords=to_device(copy(coords_start), AT),
            boundary=boundary,
            velocities=to_device(copy(vels_start), AT),
            pairwise_inters=(LennardJones(cutoff=cutoff, use_neighbors=true),
                             Coulomb(cutoff=cutoff, use_neighbors=true)),
            specific_inter_lists=(InteractionList2Atoms(
                to_device(Int32.(collect(1:2:n_atoms)), AT),
                to_device(Int32.(collect(2:2:n_atoms)), AT),
                to_device(bonds, AT),
            ),),
            neighbor_finder=neighbor_finder,
            loggers=(temp=TemperatureLogger(10), pe=PotentialEnergyLogger(10),
                     coords=CoordinatesLogger(10), svir=ScalarVirialLogger(10)),
        )
    end

    max_diff(a, b) = maximum(maximum(abs.(v)) for v in (from_device(a) .- from_device(b)))
    simulators = (VelocityVerlet(dt=dt),
                  Langevin(dt=dt, temperature=temp, friction=1.0u"ps^-1"))

    # Each backend is run with the neighbor finder it uses by default, since every
    #   combination of atom type and neighbor finder is a separate kernel compilation
    for AT in array_list
        let nf_type = (Molly.uses_gpu_neighbor_finder(AT) ? GPUNeighborFinder :
                                                            DistanceNeighborFinder)
            sys_ref = build_sys(make_ref_atom, AT, nf_type)
            fs_ref, pe_ref, vir_ref = forces(sys_ref), potential_energy(sys_ref), virial(sys_ref)
            # Reference trajectories to compare the custom atom types against
            ref_runs = map(simulators) do simulator
                sys_run = build_sys(make_ref_atom, AT, nf_type)
                simulate!(sys_run, simulator, n_steps; n_threads=1, rng=Xoshiro(20))
                (from_device(sys_run.coords), from_device(sys_run.velocities))
            end

            for (atom_i, (atom_name, make_atom)) in enumerate(custom_makers)
                sys = build_sys(make_atom, AT, nf_type)
                @test isbitstype(eltype(from_device(sys.atoms)))
                @test from_device(masses(sys))  == from_device(masses(sys_ref))
                @test from_device(charges(sys)) == from_device(charges(sys_ref))
                @test sys.total_mass == sys_ref.total_mass
                @test momentum(sys) ≈ momentum(sys_ref)
                @test temperature(sys) ≈ temperature(sys_ref)
                @test max_diff(forces(sys), fs_ref) < 1e-8u"kJ * mol^-1 * nm^-1"
                @test potential_energy(sys) ≈ pe_ref
                @test maximum(abs.(virial(sys) .- vir_ref)) < 1e-8u"kJ * mol^-1"

                # A deterministic and a stochastic simulator should match the reference.
                # Only the first atom type runs the stochastic simulator and the
                #   operations below, since each extra atom type is a separate compilation
                sim_inds = (atom_i == 1 ? eachindex(simulators) : 1:1)
                for si in sim_inds
                    simulator = simulators[si]
                    coords_ref, vels_ref = ref_runs[si]
                    sys_run = build_sys(make_atom, AT, nf_type)
                    simulate!(sys_run, simulator, n_steps; n_threads=1, rng=Xoshiro(20))
                    @test max_diff(sys_run.coords    , coords_ref) < 1e-6u"nm"
                    @test max_diff(sys_run.velocities, vels_ref  ) < 1e-6u"nm * ps^-1"
                    @test length(values(sys_run.loggers.coords)) == (n_steps ÷ 10) + 1
                    @test !any(isnan, values(sys_run.loggers.temp))
                end

                # Other common operations should not require the built-in Atom type
                sys_ops = build_sys(make_atom, AT, nf_type)
                random_velocities!(sys_ops, temp; rng=Xoshiro(40))
                remove_CM_motion!(sys_ops)
                @test maximum(abs.(momentum(sys_ops))) < 1e-8u"g * nm * mol^-1 * ps^-1"
                show(devnull, sys_ops)
                if atom_i == 1
                    simulate!(sys_ops,
                              SteepestDescentMinimizer(step_size=0.001u"nm", max_steps=20))
                    barostat = MonteCarloBarostat(1.0u"bar", temp, boundary; n_steps=5)
                    simulate!(sys_ops, VelocityVerlet(dt=dt, coupling=barostat), 20;
                              n_threads=1, rng=Xoshiro(30))
                    @test !any(isnan, ustrip.(Molly.box_sides(sys_ops.boundary)))
                end
            end
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
                n_atoms=n_atoms,
                dist_cutoff=2.0u"nm",
                device_vector_type=CuArray{Int32, 1},
            ),
            loggers=(
                coords=CoordinatesLogger(100),
                disp=DisplacementsLogger(100, CuArray(coords)),
            ),
        )
    end

    for simulator in simulators
        simulate!(sys, simulator, n_steps; n_threads=1)
        @test all(isequal(0.0u"nm"), norm.(first(values(sys.loggers.disp))))
        @test mean(norm.(sys.loggers.disp.displacements[end])) > 0.005u"nm"
        if run_cuda_tests
            simulate!(sys_gpu, simulator, n_steps; n_threads=1)
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

@testset "Pairwise interactions" begin
    n_atoms = 100
    n_steps = 1_000
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
        CoulombReactionField(dist_cutoff=1.0u"nm", use_neighbors=true, solvent_dielectric=Inf),
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

        simulate!(s, simulator, n_steps)
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
                neighbor_finder_gpu = GPUNeighborFinder(
                    n_atoms=n_atoms,
                    dist_cutoff=1.2u"nm",
                    device_vector_type=CuArray{Int32, 1},
                )
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
        simulate!(sys, simulator, n_steps)

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
            simulate!(sys_gpu, simulator, n_steps)
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
    a3 = accelerations(s)
    @test all(all(a1[i] .≈ a2[i]) for i in eachindex(a1))
    @test all(all(a1[i] .≈ a3[i]) for i in eachindex(a1))

    simulate!(s, simulator, n_steps; n_threads=1)
    simulate!(s_nounits, simulator_nounits, n_steps; n_threads=1)

    coords_diff = last(values(s.loggers.coords)) .- last(values(s_nounits.loggers.coords)) * u"nm"
    @test median([maximum(abs.(c)) for c in coords_diff]) < 1e-8u"nm"

    final_energy = last(values(s.loggers.energy))
    final_energy_nounits = last(values(s_nounits.loggers.energy)) * u"kJ * mol^-1"
    @test isapprox(final_energy, final_energy_nounits; atol=5e-4u"kJ * mol^-1")

    # Test init_step
    s2 = deepcopy(s)
    simulate!(s, simulator, 100; n_threads=1)
    simulate!(s2, simulator, 40; n_threads=1)
    simulate!(s2, simulator, 40; n_threads=1, init_step=40)
    simulate!(s2, simulator, 20; n_threads=1, init_step=80)
    @test maximum(norm.(s.coords .- s2.coords)) < 1e-8u"nm"
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

        simulate!(sys_res, sim, n_steps)

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

    simulate!(s1, simulator1, n_steps; rng=MersenneTwister(rseed))
    @test 280.0u"K" <= mean(s1.loggers.temp.history[(end - 100):end]) <= 320.0u"K"

    simulate!(s2, simulator2, n_steps; rng=MersenneTwister(rseed))
    @test 280.0u"K" <= mean(s2.loggers.temp.history[(end - 100):end]) <= 320.0u"K"

    @test maximum(maximum(abs.(v)) for v in (s1.coords .- s2.coords)) < 1e-5u"nm"
end

@testset "Reproducible randomness" begin
    n_atoms = 100
    n_steps = 200
    temp = 300.0u"K"
    boundary = CubicBoundary(4.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng=Xoshiro(2024))
    atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
             for _ in 1:n_atoms]
    make_sys() = System(
        atoms=atoms,
        coords=copy(coords),
        boundary=boundary,
        velocities=[random_velocity(10.0u"g/mol", temp; rng=Xoshiro(2024 + i))
                    for i in 1:n_atoms],
        pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm"), use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=1.2u"nm",
        ),
    )

    # Stochastic simulators and couplers should give identical trajectories for the
    #   same seed, and different trajectories for different seeds
    simulators = (
        ("Langevin", Langevin(dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1")),
        ("Andersen", VelocityVerlet(dt=0.002u"ps",
                        coupling=AndersenThermostat(temp, 1.0u"ps"))),
        ("MC barostat", VelocityVerlet(dt=0.002u"ps",
                        coupling=(AndersenThermostat(temp, 1.0u"ps"),
                                  MonteCarloBarostat(1.0u"bar", temp, boundary; n_steps=10)))),
    )
    for (sim_name, sim) in simulators
        sys_a, sys_b, sys_c = make_sys(), make_sys(), make_sys()
        simulate!(sys_a, deepcopy(sim), n_steps; n_threads=1, rng=Xoshiro(100))
        simulate!(sys_b, deepcopy(sim), n_steps; n_threads=1, rng=Xoshiro(100))
        simulate!(sys_c, deepcopy(sim), n_steps; n_threads=1, rng=Xoshiro(200))
        @test sys_a.coords == sys_b.coords
        @test sys_a.velocities == sys_b.velocities
        @test sys_a.boundary == sys_b.boundary
        @test sys_a.coords != sys_c.coords
    end

    # Velocity generation with the same seed is reproducible
    sys_v = make_sys()
    @test random_velocities(sys_v, temp; rng=Xoshiro(1)) ==
          random_velocities(sys_v, temp; rng=Xoshiro(1))
    @test random_velocities(sys_v, temp; rng=Xoshiro(1)) !=
          random_velocities(sys_v, temp; rng=Xoshiro(2))
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
