@testset "Lennard-Jones 2D" begin
    n_atoms = 10
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = RectangularBoundary(2.0u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))
    gen_temp_wrapper(s, args...; kwargs...) = temperature(s)

    s = System(
        atoms=[Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        coords=place_atoms(n_atoms, boundary; min_dist=0.3u"nm"),
        boundary=boundary,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(
            temp=TemperatureLogger(100),
            coords=CoordinateLogger(100; dims=2),
            gen_temp=GeneralObservableLogger(gen_temp_wrapper, typeof(temp), 10),
            avg_temp=AverageObservableLogger(Molly.temperature_wrapper,
                                                typeof(temp), 1; n_blocks=200),
        ),
    )
    random_velocities!(s, temp)

    @test masses(s) == fill(10.0u"g/mol", n_atoms)
    @test AtomsBase.bounding_box(s) == (
        SVector(2.0, 0.0)u"nm",
        SVector(0.0, 2.0)u"nm",
    )

    show(devnull, s)

    @time simulate!(s, simulator, n_steps; n_threads=1)

    @test length(values(s.loggers.coords)) == 201
    final_coords = last(values(s.loggers.coords))
    @test all(all(c .> 0.0u"nm") for c in final_coords)
    @test all(all(c .< boundary) for c in final_coords)
    displacements(final_coords, boundary)
    distances(final_coords, boundary)
    rdf(final_coords, boundary)

    show(devnull, s.loggers.gen_temp)
    show(devnull, s.loggers.avg_temp)
    t, σ = values(s.loggers.avg_temp)
    @test values(s.loggers.avg_temp; std=false) == t
    @test isapprox(t, mean(values(s.loggers.temp)); atol=3σ)
    run_visualize_tests && visualize(s.loggers.coords, boundary, temp_fp_viz)
end

@testset "Lennard-Jones" begin
    n_atoms = 100
    atom_mass = 10.0u"g/mol"
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))

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
            atoms_data=[AtomData(atom_name="AR", res_number=i, res_name="AR", element="Ar") for i in 1:n_atoms],
            pairwise_inters=(LennardJones(use_neighbors=true),),
            neighbor_finder=DistanceNeighborFinder(
                eligible=trues(n_atoms, n_atoms),
                n_steps=10,
                dist_cutoff=2.0u"nm",
            ),
            loggers=(
                temp=TemperatureLogger(100),
                coords=CoordinateLogger(100),
                vels=VelocityLogger(100),
                energy=TotalEnergyLogger(100),
                ke=KineticEnergyLogger(100),
                pe=PotentialEnergyLogger(100),
                force=ForceLogger(100),
                writer=StructureWriter(100, temp_fp_pdb),
                potkin_correlation=TimeCorrelationLogger(pot_obs, kin_obs, TP, TP, 1, 100),
                velocity_autocorrelation=AutoCorrelationLogger(V, TV, n_atoms, 100),
            ),
        )

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
        @test AtomsBase.bounding_box(s) == (
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
        show(devnull, s.loggers.writer)
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

        traj = read(temp_fp_pdb, BioStructures.PDBFormat)
        rm(temp_fp_pdb)
        @test BioStructures.countmodels(traj) == 201
        @test BioStructures.countatoms(first(traj)) == 100

        run_visualize_tests && visualize(s.loggers.coords, boundary, temp_fp_viz)
    end
end

@testset "Lennard-Jones infinite boundaries" begin
    n_atoms = 100
    n_steps = 2_000
    temp = 298.0u"K"
    boundary = CubicBoundary(Inf * u"nm", Inf * u"nm", 2.0u"nm")
    coords = place_atoms(n_atoms, CubicBoundary(2.0u"nm"); min_dist=0.3u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))

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
        loggers=(coords=CoordinateLogger(100),),
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

    run_visualize_tests && visualize(s.loggers.coords, boundary, temp_fp_viz)
end

@testset "Lennard-Jones simulators" begin
    n_atoms = 100
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    simulators = [
        Verlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps")),
        StormerVerlet(dt=0.002u"ps"),
        Langevin(dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1"),
        OverdampedLangevin(dt=0.002u"ps", temperature=temp, friction=10.0u"ps^-1"),
    ]

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
        loggers=(coords=CoordinateLogger(100),),
    )
    random_velocities!(s, temp)

    for simulator in simulators
        @time simulate!(s, simulator, n_steps; n_threads=1)
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
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=BerendsenThermostat(temp, 1.0u"ps"))

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
            coords=CoordinateLogger(10),
        ),
    )

    @time simulate!(s, simulator, n_steps; n_threads=1)

    if run_visualize_tests
        visualize(
            s.loggers.coords,
            boundary,
            temp_fp_viz;
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
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))
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

    @testset "$inter" for inter in pairwise_inter_types
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
                coords=CoordinateLogger(100),
                energy=TotalEnergyLogger(100),
            ),
        )

        @time simulate!(s, simulator, n_steps)
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
        loggers=(coords=CoordinateLogger(100; dims=2),),
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
            coords=CoordinateLogger(100),
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
            coords=CoordinateLogger(Float64, 100),
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
    for gpu in gpu_list
        n_atoms = 10
        n_atoms_res = n_atoms ÷ 2
        n_steps = 2_000
        boundary = CubicBoundary(2.0u"nm")
        starting_coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
        atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.2u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
        atoms_data = [AtomData(atom_type=(i <= n_atoms_res ? "A1" : "A2")) for i in 1:n_atoms]
        sim = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")

        sys = System(
            atoms=(gpu ? CuArray(atoms) : atoms),
            coords=(gpu ? CuArray(deepcopy(starting_coords)) : deepcopy(starting_coords)),
            boundary=boundary,
            atoms_data=atoms_data,
            pairwise_inters=(LennardJones(),),
            loggers=(coords=CoordinateLogger(100),),
        )

        atom_selector(at, at_data) = at_data.atom_type == "A1"

        sys_res = add_position_restraints(sys, 100_000.0u"kJ * mol^-1 * nm^-2";
                                          atom_selector=atom_selector)

        @time simulate!(sys_res, sim, n_steps)

        dists = norm.(vector.(starting_coords, Array(sys_res.coords), (boundary,)))
        @test maximum(dists[1:n_atoms_res]) < 0.1u"nm"
        @test median(dists[(n_atoms_res + 1):end]) > 0.2u"nm"
    end
end

@testset "Constraints diatomic" begin
    r_cut = 8.5u"Å"
    temp = 300.0u"K"
    atom_mass = 1.00794u"g/mol"
    n_atoms = 400
    hydrogen_data = readdlm(joinpath(data_dir, "initial_hydrogen_data.atom"); skipstart=9)
    coords_matrix = hydrogen_data[:, 2:4]
    vel_matrix = hydrogen_data[:, 5:7]

    simulators = (
        VelocityVerlet(dt=0.002u"ps"),
        Verlet(dt=0.002u"ps"),
        StormerVerlet(dt=0.002u"ps"),
        Langevin(dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1"),
    )

    bond_length = 0.74u"Å"
    constraints = [DistanceConstraint(j, j + 1, bond_length) for j in 1:2:n_atoms]
    atoms = [Atom(index=j, mass=atom_mass, σ=2.8279u"Å", ϵ=0.074u"kcal* mol^-1") for j in 1:n_atoms]
    cons = SHAKE_RATTLE(constraints, n_atoms, 1e-8u"Å", 1e-8u"Å^2 * ps^-1")
    boundary = CubicBoundary(200.0u"Å")
    lj = LennardJones(cutoff=ShiftedPotentialCutoff(r_cut), use_neighbors=true)
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        dist_cutoff=1.5*r_cut,
    )
    disable_constrained_interactions!(neighbor_finder, cons.clusters)

    for simulator in simulators
        coords = [SVector(coords_matrix[j, 1]u"Å", coords_matrix[j, 2]u"Å", coords_matrix[j, 3]u"Å") for j in 1:n_atoms]
        velocities = [1000 * SVector(vel_matrix[j, :]u"Å/ps"...) for j in 1:n_atoms]

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(lj,),
            neighbor_finder=neighbor_finder,
            constraints=(cons,),
            energy_units=u"kcal * mol^-1",
            force_units=u"kcal * mol^-1 * Å^-1",
        )

        simulate!(sys, simulator, 10_000)

        @test check_position_constraints(sys, cons)
        if simulator isa VelocityVerlet
            @test check_velocity_constraints(sys, cons)
        end
    end
end

@testset "Constraints triatomic" begin
    n_atoms = 30
    atom_mass = 10.0u"g/mol"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")

    coords = place_atoms(n_atoms ÷ 3, boundary, min_dist=0.3u"nm")

    for i in 1:(n_atoms ÷ 3)
        push!(coords, coords[i] .+ [0.13, 0.0, 0.0]u"nm")
    end

    for i in 1:(n_atoms ÷ 3)
        push!(coords, coords[i] .+ [0.26, 0.0, 0.0]u"nm")
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    eligible = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 3)
        eligible[i, i + (n_atoms ÷ 3)] = false
        eligible[i + (n_atoms ÷ 3), i] = false
        eligible[i + (n_atoms ÷ 3), i + 2 * (n_atoms ÷ 3)] = false
        eligible[i + 2 * (n_atoms ÷ 3), i + (n_atoms ÷ 3)] = false
    end

    neighbor_finder = DistanceNeighborFinder(eligible=eligible, n_steps=10, dist_cutoff=1.5u"nm")
    bond_length = 0.1u"nm"

    is = collect(1:(2 * (n_atoms ÷ 3)))
    js = collect(((n_atoms ÷ 3) + 1):n_atoms)
    constraints = [DistanceConstraint(is[idx], js[idx], bond_length) for idx in eachindex(is)]
    cons = SHAKE_RATTLE(constraints, n_atoms, 1e-8u"nm",  1e-8u"nm^2/ps")

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
        loggers=(coords=CoordinateLogger(10),),
    )

    old_coords = deepcopy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
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

    n_replicas = 4
    replica_loggers = [(temp=TemperatureLogger(10), coords=CoordinateLogger(10)) for i in 1:n_replicas]

    repsys = ReplicaSystem(
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        boundary=boundary,
        n_replicas=n_replicas,
        replica_velocities=nothing,
        pairwise_inters=pairwise_inters,
        neighbor_finder=neighbor_finder,
        replica_loggers=replica_loggers,
    )

    @test !is_on_gpu(repsys)
    @test float_type(repsys) == Float64
    @test masses(repsys) == fill(atom_mass, n_atoms)
    @test length(repsys) == n_atoms
    @test eachindex(repsys) == Base.OneTo(n_atoms)
    @test AtomsBase.mass(repsys, :) == fill(atom_mass, n_atoms)
    @test AtomsBase.mass(repsys, 5) == atom_mass
    @test AtomsBase.bounding_box(repsys) == (
        SVector(2.0, 0.0, 0.0)u"nm",
        SVector(0.0, 2.0, 0.0)u"nm",
        SVector(0.0, 0.0, 2.0)u"nm",
    )
    show(devnull, repsys)

    temp_vals = [120.0u"K", 180.0u"K", 240.0u"K", 300.0u"K"]
    simulator = TemperatureREMD(
        dt=0.005u"ps",
        temperatures=temp_vals,
        simulators=[
            Langevin(
                dt=0.005u"ps",
                temperature=temp,
                friction=0.1u"ps^-1",
            )
            for temp in temp_vals],
        exchange_time=2.5u"ps",
    )

    @time simulate!(repsys, simulator, n_steps; assign_velocities=true )
    @time simulate!(repsys, simulator, n_steps; assign_velocities=false)

    efficiency = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test efficiency > 0.2 # This is a fairly arbitrary threshold but it's a good test for very bad cases
    @test efficiency < 1.0 # Bad acceptance rate?
    @info "Exchange Efficiency: $efficiency"

    for id in eachindex(repsys.replicas)
        mean_temp = mean(values(repsys.replicas[id].loggers.temp))
        @test (0.9 * temp_vals[id]) < mean_temp < (1.1 * temp_vals[id])
    end
end

@testset "Hamiltonian REMD" begin
    n_atoms = 100
    n_steps = 10_000
    atom_mass = 10.0u"g/mol"
    atoms = [Atom(mass=atom_mass, charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")

    temp = 100.0u"K"
    velocities = [random_velocity(10.0u"g/mol", temp) for i in 1:n_atoms]

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_replicas = 4
    λ_vals = [0.0, 0.1, 0.25, 0.4]
    replica_pairwise_inters = [(LennardJonesSoftCore(α=1, λ=λ_vals[i], p=2, use_neighbors=true),)
                               for i in 1:n_replicas]

    replica_loggers = [(temp=TemperatureLogger(10), ) for i in 1:n_replicas]

    repsys = ReplicaSystem(
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        boundary=boundary,
        n_replicas=n_replicas,
        replica_velocities=nothing,
        replica_pairwise_inters=replica_pairwise_inters,
        neighbor_finder=neighbor_finder,
        replica_loggers=replica_loggers,
    )

    simulator = HamiltonianREMD(
        dt=0.005u"ps",
        temperature=temp,
        simulators=[
            Langevin(
                dt=0.005u"ps",
                temperature=temp,
                friction=0.1u"ps^-1",
            )
            for _ in 1:n_replicas],
        exchange_time=2.5u"ps",
    )

    @time simulate!(repsys, simulator, n_steps; assign_velocities=true )
    @time simulate!(repsys, simulator, n_steps; assign_velocities=false)

    efficiency = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test efficiency > 0.2 # This is a fairly arbitrary threshold, but it's a good tests for very bad cases
    @test efficiency < 1.0 # Bad acceptance rate?
    @info "Exchange Efficiency: $efficiency"
    for id in eachindex(repsys.replicas)
        mean_temp = mean(values(repsys.replicas[id].loggers.temp))
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
        pairwise_inters=(Coulomb(), ),
        neighbor_finder=neighbor_finder,
        loggers=(
            coords=CoordinateLogger(10),
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
    @test acceptance_rate > 0.2

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
    wigner_seitz_radius = cbrt(3 * box_volume(sys.boundary) / (4π * length(sys)))
    @test wigner_seitz_radius < mean_distance < 2 * wigner_seitz_radius
end

@testset "Monte Carlo barostat" begin
    # See http://www.sklogwiki.org/SklogWiki/index.php/Argon for parameters
    n_atoms = 25
    n_steps = 1_000_000
    atom_mass = 39.947u"g/mol"
    boundary = CubicBoundary(8.0u"nm")
    temp = 288.15u"K"
    press = 1.0u"bar"
    dt = 0.0005u"ps"
    friction = 1.0u"ps^-1"
    lang = Langevin(dt=dt, temperature=temp, friction=friction)
    atoms = fill(Atom(mass=atom_mass, σ=0.3345u"nm", ϵ=1.0451u"kJ * mol^-1"), n_atoms)
    coords = place_atoms(n_atoms, boundary; min_dist=1.0u"nm")
    n_log_steps = 500

    box_size_wrapper(sys, args...; kwargs...) = sys.boundary.side_lengths[1]
    BoundaryLogger(n_steps) = GeneralObservableLogger(box_size_wrapper, typeof(1.0u"nm"), n_steps)

    sys = System(
        atoms=atoms,
        coords=deepcopy(coords),
        boundary=boundary,
        pairwise_inters=(LennardJones(),),
        loggers=(
            temperature=TemperatureLogger(n_log_steps),
            total_energy=TotalEnergyLogger(n_log_steps),
            kinetic_energy=KineticEnergyLogger(n_log_steps),
            potential_energy=PotentialEnergyLogger(n_log_steps),
            virial=VirialLogger(n_log_steps),
            pressure=PressureLogger(n_log_steps),
            box_size=BoundaryLogger(n_log_steps),
        ),
    )

    simulate!(deepcopy(sys), lang, 1_000; n_threads=1)
    @time simulate!(sys, lang, n_steps; n_threads=1)

    @test 280.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
    @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy  )) < 120.0u"kJ * mol^-1"
    @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
    @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
    @test -5.0u"kJ * mol^-1" < mean(values(sys.loggers.virial)) < 5.0u"kJ * mol^-1"
    @test 1.7u"bar" < mean(values(sys.loggers.pressure)) < 2.2u"bar"
    @test 0.1u"bar" < std( values(sys.loggers.pressure)) < 0.5u"bar"
    @test all(values(sys.loggers.box_size) .== 8.0u"nm")
    @test sys.boundary == CubicBoundary(8.0u"nm")

    barostat = MonteCarloBarostat(press, temp, boundary)
    lang_baro = Langevin(dt=dt, temperature=temp, friction=friction, coupling=barostat)
    vvand_baro = VelocityVerlet(dt=dt, coupling=(AndersenThermostat(temp, 1.0u"ps"), barostat))

    for sim in (lang_baro, vvand_baro)
        for gpu in gpu_list
            if gpu && sim == vvand_baro
                continue
            end
            AT = gpu ? CuArray : Array

            sys = System(
                atoms=AT(atoms),
                coords=AT(deepcopy(coords)),
                boundary=boundary,
                pairwise_inters=(LennardJones(),),
                loggers=(
                    temperature=TemperatureLogger(n_log_steps),
                    total_energy=TotalEnergyLogger(n_log_steps),
                    kinetic_energy=KineticEnergyLogger(n_log_steps),
                    potential_energy=PotentialEnergyLogger(n_log_steps),
                    virial=VirialLogger(n_log_steps),
                    pressure=PressureLogger(n_log_steps),
                    box_size=BoundaryLogger(n_log_steps),
                ),
            )

            simulate!(deepcopy(sys), sim, 1_000; n_threads=1)
            @time simulate!(sys, sim, n_steps; n_threads=1)

            @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy  )) < 120.0u"kJ * mol^-1"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
            @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
            @test -5.0u"kJ * mol^-1" < mean(values(sys.loggers.virial)) < 5.0u"kJ * mol^-1"
            @test 0.8u"bar" < mean(values(sys.loggers.pressure)) < 1.2u"bar"
            @test 0.1u"bar" < std( values(sys.loggers.pressure)) < 0.5u"bar"
            @test 9.5u"nm" < mean(values(sys.loggers.box_size)) < 10.5u"nm"
            @test 0.2u"nm" < std( values(sys.loggers.box_size)) < 1.0u"nm"
            @test sys.boundary != CubicBoundary(8.0u"nm")
        end
    end
end

@testset "Monte Carlo anisotropic barostat" begin
    # See http://www.sklogwiki.org/SklogWiki/index.php/Argon for parameters
    n_atoms = 25
    n_steps = 1_000_000
    atom_mass = 39.947u"g/mol"
    boundary = CubicBoundary(8.0u"nm")
    temp = 288.15u"K"
    dt = 0.0005u"ps"
    friction = 1.0u"ps^-1"
    lang = Langevin(dt=dt, temperature=temp, friction=friction)
    atoms = fill(Atom(mass=atom_mass, σ=0.3345u"nm", ϵ=1.0451u"kJ * mol^-1"), n_atoms)
    coords = place_atoms(n_atoms, boundary; min_dist=1.0u"nm")
    n_log_steps = 500

    box_volume_wrapper(sys, args...; kwargs...) = box_volume(sys.boundary)
    VolumeLogger(n_steps) = GeneralObservableLogger(box_volume_wrapper, typeof(1.0u"nm^3"), n_steps)

    baro_f(pressure) = MonteCarloAnisotropicBarostat(pressure, temp, boundary)
    lang_f(barostat) = Langevin(dt=dt, temperature=temp, friction=friction, coupling=barostat)

    pressure_test_set = (
        SVector(1.0u"bar", 1.0u"bar", 1.0u"bar"), # XYZ-axes coupled with the same pressure value
        SVector(1.5u"bar", 0.5u"bar", 1.0u"bar"), # XYZ-axes coupled with different pressure values
        SVector(nothing  , 1.0u"bar", nothing  ), # Only Y-axis coupled
        SVector(nothing  , nothing  , nothing  ), # Uncoupled
    )

    for gpu in gpu_list
        AT = gpu ? CuArray : Array
        for (press_i, press) in enumerate(pressure_test_set)
            if gpu && press_i != 2
                continue
            end

            sys = System(
                atoms=AT(atoms),
                coords=AT(deepcopy(coords)),
                boundary=boundary,
                pairwise_inters=(LennardJones(),),
                loggers=(
                    temperature=TemperatureLogger(n_log_steps),
                    total_energy=TotalEnergyLogger(n_log_steps),
                    kinetic_energy=KineticEnergyLogger(n_log_steps),
                    potential_energy=PotentialEnergyLogger(n_log_steps),
                    virial=VirialLogger(n_log_steps),
                    pressure=PressureLogger(n_log_steps),
                    box_volume=VolumeLogger(n_log_steps),
                ),
            )

            sim = lang_f(baro_f(press))
            simulate!(deepcopy(sys), sim, 100; n_threads=1)
            @time simulate!(sys, sim, n_steps; n_threads=1)

            @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 330.0u"K"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy)) < 120.0u"kJ * mol^-1"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
            @test -5.0u"kJ * mol^-1" < mean(values(sys.loggers.virial)) < 5.0u"kJ * mol^-1"
            @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
            all(!isnothing, press) && @test 0.6u"bar" < mean(values(sys.loggers.pressure)) < 1.3u"bar"
            any(!isnothing, press) && @test 0.1u"bar" < std(values(sys.loggers.pressure)) < 2.5u"bar"
            any(!isnothing, press) && @test 800.0u"nm^3" < mean(values(sys.loggers.box_volume)) < 2000u"nm^3"
            any(!isnothing, press) && @test 80.0u"nm^3" < std(values(sys.loggers.box_volume)) < 500.0u"nm^3"
            axis_is_uncoupled = isnothing.(press)
            axis_is_unchanged = sys.boundary .== 8.0u"nm"
            @test all(axis_is_uncoupled .== axis_is_unchanged)
        end
    end
end

@testset "Monte Carlo membrane barostat" begin
    # See http://www.sklogwiki.org/SklogWiki/index.php/Argon for parameters
    n_atoms = 25
    n_steps = 1_000_000
    atom_mass = 39.947u"g/mol"
    boundary = CubicBoundary(8.0u"nm")
    temp = 288.15u"K"
    tens = 0.1u"bar * nm"
    press = 1.0u"bar"
    dt = 0.0005u"ps"
    friction = 1.0u"ps^-1"
    lang = Langevin(dt=dt, temperature=temp, friction=friction)
    atoms = fill(Atom(mass=atom_mass, σ=0.3345u"nm", ϵ=1.0451u"kJ * mol^-1"), n_atoms)
    coords = place_atoms(n_atoms, boundary; min_dist=1.0u"nm")
    n_log_steps = 500

    box_volume_wrapper(sys, args...; kwargs...) = box_volume(sys.boundary)
    VolumeLogger(n_steps) = GeneralObservableLogger(box_volume_wrapper, typeof(1.0u"nm^3"), n_steps)

    lang_f(barostat) = Langevin(dt=dt, temperature=temp, friction=friction, coupling=barostat)

    barostat_test_set = (
        MonteCarloMembraneBarostat(press, tens, temp, boundary),
        MonteCarloMembraneBarostat(press, tens, temp, boundary; xy_isotropy=true),
        MonteCarloMembraneBarostat(press, tens, temp, boundary; constant_volume=true),
        MonteCarloMembraneBarostat(press, tens, temp, boundary; z_axis_fixed=true),
    )

    for gpu in gpu_list
        AT = gpu ? CuArray : Array
        for (barostat_i, barostat) in enumerate(barostat_test_set)
            if gpu && barostat_i != 2
                continue
            end

            sys = System(
                atoms=AT(atoms),
                coords=AT(deepcopy(coords)),
                boundary=boundary,
                pairwise_inters=(LennardJones(),),
                loggers=(
                    temperature=TemperatureLogger(n_log_steps),
                    total_energy=TotalEnergyLogger(n_log_steps),
                    kinetic_energy=KineticEnergyLogger(n_log_steps),
                    potential_energy=PotentialEnergyLogger(n_log_steps),
                    virial=VirialLogger(n_log_steps),
                    pressure=PressureLogger(n_log_steps),
                    box_volume=VolumeLogger(n_log_steps),
                ),
            )

            sim = lang_f(barostat)
            simulate!(deepcopy(sys), sim, 100; n_threads=1)
            @time simulate!(sys, sim, n_steps; n_threads=1)

            @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy)) < 120.0u"kJ * mol^-1"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
            @test -5.0u"kJ * mol^-1" < mean(values(sys.loggers.virial)) < 5.0u"kJ * mol^-1"
            @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
            if barostat.xy_isotropy
                @test sys.boundary[1] == sys.boundary[2]
            end
            if !barostat.constant_volume && isnothing(barostat.pressure[3])
                @test sys.boundary[3] == 8.0u"nm"
                @test 0.8u"bar" < mean(values(sys.loggers.pressure)) < 1.2u"bar"
                @test 0.1u"bar" < std(values(sys.loggers.pressure))  < 0.5u"bar"
            end
        end
    end
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
        energy_units=u"kJ * mol^-1",
        force_units=u"kJ * mol^-1 * nm^-1",
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
    @test -1850u"kJ * mol^-1" < mean(values(sys.loggers.tot_eng)) < -1650u"kJ * mol^-1"

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
    boundary = CubicBoundary(6.0u"nm")
    temp = 1.0u"K"
    starting_coords = place_diatomics(n_atoms ÷ 2, boundary, 0.2u"nm"; min_dist=0.2u"nm")
    starting_velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]
    starting_coords_f32 = [Float32.(c) for c in starting_coords]
    starting_velocities_f32 = [Float32.(c) for c in starting_velocities]

    function test_sim(nl::Bool, parallel::Bool, f32::Bool, gpu::Bool)
        n_atoms = 400
        n_steps = 200
        atom_mass = f32 ? 10.0f0u"g/mol" : 10.0u"g/mol"
        boundary = f32 ? CubicBoundary(6.0f0u"nm") : CubicBoundary(6.0u"nm")
        simulator = VelocityVerlet(dt=f32 ? 0.02f0u"ps" : 0.02u"ps")
        k = f32 ? 10_000.0f0u"kJ * mol^-1 * nm^-2" : 10_000.0u"kJ * mol^-1 * nm^-2"
        r0 = f32 ? 0.2f0u"nm" : 0.2u"nm"
        bonds = [HarmonicBond(k=k, r0=r0) for i in 1:(n_atoms ÷ 2)]
        specific_inter_lists = (InteractionList2Atoms(
            gpu ? CuArray(Int32.(collect(1:2:n_atoms))) : Int32.(collect(1:2:n_atoms)),
            gpu ? CuArray(Int32.(collect(2:2:n_atoms))) : Int32.(collect(2:2:n_atoms)),
            gpu ? CuArray(bonds) : bonds,
        ),)

        neighbor_finder = NoNeighborFinder()
        cutoff = DistanceCutoff(f32 ? 1.0f0u"nm" : 1.0u"nm")
        pairwise_inters = (LennardJones(use_neighbors=false, cutoff=cutoff),)
        if nl
            neighbor_finder = DistanceNeighborFinder(
                eligible=gpu ? CuArray(trues(n_atoms, n_atoms)) : trues(n_atoms, n_atoms),
                n_steps=10,
                dist_cutoff=f32 ? 1.5f0u"nm" : 1.5u"nm",
            )
            pairwise_inters = (LennardJones(use_neighbors=true, cutoff=cutoff),)
        end
        show(devnull, neighbor_finder)

        if gpu
            coords = CuArray(deepcopy(f32 ? starting_coords_f32 : starting_coords))
            velocities = CuArray(deepcopy(f32 ? starting_velocities_f32 : starting_velocities))
            atoms = CuArray([Atom(mass=atom_mass, charge=f32 ? 0.0f0 : 0.0, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                                  ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms])
        else
            coords = deepcopy(f32 ? starting_coords_f32 : starting_coords)
            velocities = deepcopy(f32 ? starting_velocities_f32 : starting_velocities)
            atoms = [Atom(mass=atom_mass, charge=f32 ? 0.0f0 : 0.0, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                            ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms]
        end

        s = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            specific_inter_lists=specific_inter_lists,
            neighbor_finder=neighbor_finder,
        )

        @test is_on_gpu(s) == gpu
        @test float_type(s) == (f32 ? Float32 : Float64)

        n_threads = parallel ? Threads.nthreads() : 1
        E_start = potential_energy(s; n_threads=n_threads)

        simulate!(s, simulator, n_steps; n_threads=n_threads)
        return s.coords, E_start
    end

    runs = [
        ("CPU"       , [false, false, false, false]),
        ("CPU f32"   , [false, false, true , false]),
        ("CPU NL"    , [true , false, false, false]),
        ("CPU f32 NL", [true , false, true , false]),
    ]
    if run_parallel_tests
        push!(runs, ("CPU parallel"       , [false, true , false, false]))
        push!(runs, ("CPU parallel f32"   , [false, true , true , false]))
        push!(runs, ("CPU parallel NL"    , [true , true , false, false]))
        push!(runs, ("CPU parallel f32 NL", [true , true , true , false]))
    end
    if run_gpu_tests
        push!(runs, ("GPU"       , [false, false, false, true]))
        push!(runs, ("GPU f32"   , [false, false, true , true]))
        push!(runs, ("GPU NL"    , [true , false, false, true]))
        push!(runs, ("GPU f32 NL", [true , false, true , true]))
    end

    final_coords_ref, E_start_ref = test_sim(runs[1][2]...)
    # Check all simulations give the same result to within some error
    for (name, args) in runs
        final_coords, E_start = test_sim(args...)
        final_coords_f64 = [Float64.(c) for c in Array(final_coords)]
        coord_diff = sum(sum(map(x -> abs.(x), final_coords_f64 .- final_coords_ref))) / (3 * n_atoms)
        E_diff = abs(Float64(E_start) - E_start_ref)
        @info "$(rpad(name, 19)) - difference per coordinate $coord_diff - potential energy difference $E_diff"
        @test coord_diff < 1e-4u"nm"
        @test E_diff < 5e-4u"kJ * mol^-1"
    end
end
