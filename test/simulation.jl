@testset "Lennard-Jones 2D" begin
    n_atoms = 10
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = RectangularBoundary(2.0u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))
    gen_temp_wrapper(s, neighbors=nothing; n_threads::Integer=Threads.nthreads()) = temperature(s)

    s = System(
        atoms=[Atom(charge=0.0, mass=10.0u"u", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        pairwise_inters=(LennardJones(nl_only=true),),
        coords=place_atoms(n_atoms, boundary; min_dist=0.3u"nm"),
        boundary=boundary,
        neighbor_finder=DistanceNeighborFinder(
            nb_matrix=trues(n_atoms, n_atoms),
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

    @test masses(s) == fill(10.0u"u", n_atoms)
    @test typeof(boundary_conditions(s)) <: SVector
    @test bounding_box(s) == SVector(
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
    @test isapprox(t, mean(values(s.loggers.temp)); atol=3σ)
    run_visualize_tests && visualize(s.loggers.coords, boundary, temp_fp_viz)
end

@testset "Lennard-Jones" begin
    n_atoms = 100
    atom_mass = 10.0u"u"
    n_steps = 20_000
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))
    n_threads_list = run_parallel_tests ? (1, Threads.nthreads()) : (1,)

    TV = typeof(velocity(10.0u"u", temp))
    TP = typeof(0.2u"kJ * mol^-1")

    V(sys, args...; kwargs...) = sys.velocities
    pot_obs(sys, neighbors; kwargs...) = potential_energy(sys, neighbors)
    kin_obs(sys, args...; kwargs...) = kinetic_energy(sys)

    for n_threads in n_threads_list
        s = System(
            atoms=[Atom(charge=0.0, mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
            atoms_data=[AtomData(atom_name="AR", res_number=i, res_name="AR", element="Ar") for i in 1:n_atoms],
            pairwise_inters=(LennardJones(nl_only=true),),
            coords=place_atoms(n_atoms, boundary; min_dist=0.3u"nm"),
            velocities=[velocity(atom_mass, temp) .* 0.01 for i in 1:n_atoms],
            boundary=boundary,
            neighbor_finder=DistanceNeighborFinder(
                nb_matrix=trues(n_atoms, n_atoms),
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
        @test species_type(s) <: Atom
        @test typeof(s[5]) <: AtomView
        @test position(s) == s.coords
        @test position(s, 5) == s.coords[5]
        @test velocity(s) == s.velocities
        @test velocity(s, 5) == s.velocities[5]
        @test atomic_mass(s) == fill(atom_mass, 100)
        @test atomic_mass(s, 5) == atom_mass
        @test atomic_symbol(s) == fill(:Ar, 100)
        @test atomic_symbol(s, 5) == :Ar
        @test typeof(boundary_conditions(s)) <: SVector
        @test bounding_box(s) == SVector(
            SVector(2.0, 0.0, 0.0)u"nm",
            SVector(0.0, 2.0, 0.0)u"nm",
            SVector(0.0, 0.0, 2.0)u"nm",
        )

        nf_tree = TreeNeighborFinder(nb_matrix=trues(n_atoms, n_atoms), n_steps=10, dist_cutoff=2.0u"nm")
        neighbors = find_neighbors(s, s.neighbor_finder; n_threads=n_threads)
        neighbors_tree = find_neighbors(s, nf_tree; n_threads=n_threads)
        @test length(neighbors.list) == length(neighbors_tree.list) &&
                all(nn in neighbors_tree.list for nn in neighbors.list)

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
        @test unit(velocity_autocorr(s.loggers.vels)) == u"nm^2 * ps^-2"
        @test unit(first(values(s.loggers.potkin_correlation))) == NoUnits
        @test unit(first(values(s.loggers.velocity_autocorrelation; normalize=false))) == u"nm^2 * ps^-2"

        traj = read(temp_fp_pdb, BioStructures.PDB)
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
        atoms=[Atom(charge=0.0, mass=10.0u"u", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        pairwise_inters=(LennardJones(nl_only=true),),
        coords=coords,
        boundary=boundary,
        neighbor_finder=DistanceNeighborFinder(
            nb_matrix=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(coords=CoordinateLogger(100),),
    )
    random_velocities!(s, temp)

    @time simulate!(s, simulator, n_steps)

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
    ]

    s = System(
        atoms=[Atom(charge=0.0, mass=10.0u"u", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        pairwise_inters=(LennardJones(nl_only=true),),
        coords=coords,
        boundary=boundary,
        neighbor_finder=DistanceNeighborFinder(
            nb_matrix=trues(n_atoms, n_atoms),
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
        fill("", n_atoms ÷ 2),
        [HarmonicBond(k=300_000.0u"kJ * mol^-1 * nm^-2", r0=0.1u"nm") for i in 1:(n_atoms ÷ 2)],
    )
    nb_matrix = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 2)
        nb_matrix[i, i + (n_atoms ÷ 2)] = false
        nb_matrix[i + (n_atoms ÷ 2), i] = false
    end
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=BerendsenThermostat(temp, 1.0u"ps"))

    s = System(
        atoms=[Atom(charge=0.0, mass=10.0u"u", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        pairwise_inters=(LennardJones(nl_only=true),),
        specific_inter_lists=(bonds,),
        coords=coords,
        velocities=[velocity(10.0u"u", temp) .* 0.01 for i in 1:n_atoms],
        boundary=boundary,
        neighbor_finder=DistanceNeighborFinder(
            nb_matrix=trues(n_atoms, n_atoms),
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
    G = 10.0u"kJ * nm * u^-2 * mol^-1"
    simulator = VelocityVerlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 10.0u"ps"))
    pairwise_inter_types = (
        LennardJones(nl_only=true), LennardJones(nl_only=false),
        LennardJones(cutoff=DistanceCutoff(1.0u"nm"), nl_only=true),
        LennardJones(cutoff=ShiftedPotentialCutoff(1.0u"nm"), nl_only=true),
        LennardJones(cutoff=ShiftedForceCutoff(1.0u"nm"), nl_only=true),
        LennardJones(cutoff=CubicSplineCutoff(0.6u"nm", 1.0u"nm"), nl_only=true),
        SoftSphere(nl_only=true), SoftSphere(nl_only=false),
        Mie(m=5, n=10, nl_only=true), Mie(m=5, n=10, nl_only=false),
        Coulomb(nl_only=true), Coulomb(nl_only=false),
        CoulombReactionField(dist_cutoff=1.0u"nm", nl_only=true),
        CoulombReactionField(dist_cutoff=1.0u"nm", nl_only=false),
        Gravity(G=G, nl_only=true), Gravity(G=G, nl_only=false),
    )

    @testset "$inter" for inter in pairwise_inter_types
        if inter.nl_only
            neighbor_finder = DistanceNeighborFinder(nb_matrix=trues(n_atoms, n_atoms), n_steps=10,
                                                        dist_cutoff=1.5u"nm")
        else
            neighbor_finder = NoNeighborFinder()
        end

        s = System(
            atoms=[Atom(charge=i % 2 == 0 ? -1.0 : 1.0, mass=10.0u"u", σ=0.2u"nm",
                        ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
            pairwise_inters=(inter,),
            coords=place_atoms(n_atoms, boundary; min_dist=0.2u"nm"),
            velocities=[velocity(10.0u"u", temp) .* 0.01 for i in 1:n_atoms],
            boundary=boundary,
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

@testset "Units" begin
    n_atoms = 100
    n_steps = 2_000 # Does diverge for longer simulations or higher velocities
    temp = 298.0u"K"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    velocities = [velocity(10.0u"u", temp) .* 0.01 for i in 1:n_atoms]
    simulator = VelocityVerlet(dt=0.002u"ps")
    simulator_nounits = VelocityVerlet(dt=0.002)

    vtype = eltype(velocities)
    V(sys::System, neighbors=nothing) = sys.velocities

    s = System(
        atoms=[Atom(charge=0.0, mass=10.0u"u", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        pairwise_inters=(LennardJones(nl_only=true),),
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        neighbor_finder=DistanceNeighborFinder(
            nb_matrix=trues(n_atoms, n_atoms),
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
        atoms=[Atom(charge=0.0, mass=10.0, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
        pairwise_inters=(LennardJones(nl_only=true),),
        coords=ustrip_vec.(coords),
        velocities=ustrip_vec.(velocities),
        boundary=CubicBoundary(ustrip.(boundary)),
        neighbor_finder=DistanceNeighborFinder(
            nb_matrix=trues(n_atoms, n_atoms),
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
    a2 = accelerations(s_nounits, neighbors_nounits)u"kJ * mol^-1 * nm^-1 * u^-1"
    @test all(all(a1[i] .≈ a2[i]) for i in eachindex(a1))

    simulate!(s, simulator, n_steps; n_threads=1)
    simulate!(s_nounits, simulator_nounits, n_steps; n_threads=1)

    coords_diff = last(values(s.loggers.coords)) .- last(values(s_nounits.loggers.coords)) * u"nm"
    @test median([maximum(abs.(c)) for c in coords_diff]) < 1e-8u"nm"

    final_energy = last(values(s.loggers.energy))
    final_energy_nounits = last(values(s_nounits.loggers.energy)) * u"kJ * mol^-1"
    @test isapprox(final_energy, final_energy_nounits, atol=5e-4u"kJ * mol^-1")
end

@testset "Position restraints" begin
    gpu_list = run_gpu_tests ? (false, true) : (false,)
    for gpu in gpu_list
        n_atoms = 10
        n_atoms_res = n_atoms ÷ 2
        n_steps = 2_000
        boundary = CubicBoundary(2.0u"nm")
        starting_coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
        atoms = [Atom(charge=0.0, mass=10.0u"u", σ=0.2u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
        atoms_data = [AtomData(atom_type=(i <= n_atoms_res ? "A1" : "A2")) for i in 1:n_atoms]
        sim = Langevin(dt=0.001u"ps", temperature=300.0u"K", friction=1.0u"ps^-1")

        sys = System(
            atoms=gpu ? CuArray(atoms) : atoms,
            atoms_data=atoms_data,
            pairwise_inters=(LennardJones(),),
            coords=gpu ? CuArray(deepcopy(starting_coords)) : deepcopy(starting_coords),
            boundary=boundary,
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

@testset "SHAKE bond constraints" begin
    n_atoms = 100
    atom_mass = 10.0u"u"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")

    coords = place_atoms(n_atoms ÷ 2, boundary, min_dist=0.3u"nm")

    for i in 1:length(coords)
        push!(coords, coords[i] .+ [0.15, 0.0, 0.0]u"nm")
    end

    temp = 100.0u"K"

    velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]

    nb_matrix = trues(n_atoms, n_atoms)

    for i in (1:n_atoms÷2)
        nb_matrix[i, i + (n_atoms ÷ 2)] = false
        nb_matrix[i + (n_atoms ÷ 2), i] = false
    end

    neighbor_finder = DistanceNeighborFinder(nb_matrix=nb_matrix, n_steps=10, dist_cutoff=1.5u"nm")
    
    bond_lengths = [0.1u"nm" for i in 1:(n_atoms÷2)]

    sh = SHAKE(bond_lengths, collect(1:(n_atoms÷2)), collect((1+n_atoms÷2):n_atoms))
    
    constraint_list = (sh,)

    sys = System(
        atoms=atoms,
        pairwise_inters=(LennardJones(nl_only=true),),
        constraints=constraint_list,
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        neighbor_finder=neighbor_finder,
        loggers=Dict(
            "temp" => TemperatureLogger(10),
            "coords" => CoordinateLogger(10),
        ),
    )

    for i in 1:length(sys.coords)
        sys.coords[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"        
    end

    old_coords = sys.coords
    apply_constraints!(sys, sh, old_coords, 0.002u"ps")

    lengths = map(1:length(sh.is)) do r
        return norm(vector(sys.coords[sh.is[r]], sys.coords[sh.js[r]], sys.boundary))
    end

    @test maximum(abs.(lengths .- 0.1u"nm")) < 1e-10u"nm"

    simulator = VelocityVerlet(dt=0.002u"ps", coupling=AndersenThermostat(temp, 1.0u"ps"))

    @time simulate!(sys, simulator, 1_000)

    lengths = map(1:length(sh.is)) do r
        return norm(vector(sys.coords[sh.is[r]], sys.coords[sh.js[r]], sys.boundary))
    end

    @test maximum(abs.(lengths .- 0.1u"nm")) < 1e-10u"nm"
end

@testset "SHAKE triatomic" begin
    n_atoms = 30
    atom_mass = 10.0u"u"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")

    coords = place_atoms(n_atoms÷3, boundary, min_dist=0.3u"nm")

    for i in 1:(n_atoms÷3)
        push!(coords, coords[i] .+ [0.13, 0.0, 0.0]u"nm")
    end

    for i in 1:(n_atoms÷3)
        push!(coords, coords[i] .+ [0.26, 0.0, 0.0]u"nm")
    end

    temp = 100.0u"K"

    velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]

    nb_matrix = trues(n_atoms, n_atoms)

    for i in (1:n_atoms÷3)
        nb_matrix[i, i + (n_atoms÷3)] = false
        nb_matrix[i + (n_atoms÷3), i] = false

        nb_matrix[i + (n_atoms÷3), i + (n_atoms*2÷3)] = false
        nb_matrix[i + (n_atoms÷3) + 1, i + (n_atoms*2÷3)] = false
    end

    neighbor_finder = DistanceNeighborFinder(nb_matrix=nb_matrix, n_steps=10, dist_cutoff=1.5u"nm")

    bond_lengths = [0.1u"nm" for i in 1:2*(n_atoms÷3)]

    sh = SHAKE(
        bond_lengths,
        [collect(1:n_atoms÷3)..., collect((1+(n_atoms÷3)):2*(n_atoms÷3))...],
        [collect((1+(n_atoms÷3)):(2*n_atoms÷3))..., collect((1+(2*n_atoms÷3)):n_atoms)...],
    )

    constraints = (sh,)

    sys = System(
        atoms=atoms,
        pairwise_inters=(LennardJones(nl_only=true),),
        constraints=constraints,
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        neighbor_finder=neighbor_finder,
        loggers=Dict("coords" => CoordinateLogger(10)),
    )

    for i in 1:length(sys.coords)
        sys.coords[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
    end

    old_coords = sys.coords

    @time apply_constraints!(sys, sh, old_coords, 0.002u"ps")

    lengths = map(1:length(sh.is)) do r
        return norm(vector(sys.coords[sh.is[r]], sys.coords[sh.js[r]], sys.boundary))
    end

    @test maximum(abs.(lengths .- 0.1u"nm")) < 1e-10u"nm"
end

@testset "Langevin splitting" begin
    n_atoms = 400
    n_steps = 2000
    temp = 300.0u"K"
    boundary = CubicBoundary(10.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    velocities = [velocity(10.0u"u", temp) .* 0.01 for i in 1:n_atoms]
    s1 = System(
        atoms=[Atom(charge=0.0, mass=10.0u"u", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
        pairwise_inters=(LennardJones(nl_only=true),),
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        neighbor_finder=DistanceNeighborFinder(
            nb_matrix=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
        loggers=(temp=TemperatureLogger(10),),
    )
    s2 = deepcopy(s1)
    rseed = 2022
    simulator1 = Langevin(dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1")
    simulator2 = LangevinSplitting(dt=0.002u"ps", temperature=temp,
                                   friction=10.0u"u * ps^-1", splitting="BAOA")

    @time simulate!(s1, simulator1, n_steps; rng=MersenneTwister(rseed))
    @test 280.0u"K" <= mean(s1.loggers.temp.history[(end - 100):end]) <= 320.0u"K"

    @time simulate!(s2, simulator2, n_steps; rng=MersenneTwister(rseed))
    @test 280.0u"K" <= mean(s2.loggers.temp.history[(end - 100):end]) <= 320.0u"K"

    @test maximum(maximum(abs.(v)) for v in (s1.coords .- s2.coords)) < 1e-5u"nm"
end

@testset "Temperature REMD" begin
    n_atoms = 100
    n_steps = 10_000
    atom_mass = 10.0u"u"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")

    pairwise_inters = (LennardJones(nl_only=true),)

    nb_matrix = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 2)
        nb_matrix[i, i + (n_atoms ÷ 2)] = false
        nb_matrix[i + (n_atoms ÷ 2), i] = false
    end
    
    neighbor_finder = DistanceNeighborFinder(
        nb_matrix=nb_matrix,
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_replicas = 4
    replica_loggers = [(temp=TemperatureLogger(10), coords=CoordinateLogger(10)) for i in 1:n_replicas]

    repsys = ReplicaSystem(
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        replica_velocities=nothing,
        n_replicas=n_replicas,
        boundary=boundary,
        pairwise_inters=pairwise_inters,
        replica_loggers=replica_loggers,
        neighbor_finder=neighbor_finder,
    )

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
    @test efficiency > 0.2 # This is a fairly arbitrary threshold, but it's a good tests for very bad cases
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
    atom_mass = 10.0u"u"
    atoms = [Atom(mass=atom_mass, charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")

    temp = 100.0u"K"
    velocities = [velocity(10.0u"u", temp) for i in 1:n_atoms]

    nb_matrix = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 2)
        nb_matrix[i, i + (n_atoms ÷ 2)] = false
        nb_matrix[i + (n_atoms ÷ 2), i] = false
    end
    
    neighbor_finder = DistanceNeighborFinder(
        nb_matrix=nb_matrix,
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_replicas = 4
    λ_vals = [0.0, 0.1, 0.25, 0.4]
    replica_pairwise_inters = [(LennardJonesSoftCore(α=1, λ=λ_vals[i], p=2, nl_only=true),) for i in 1:n_replicas]

    replica_loggers = [(temp=TemperatureLogger(10), ) for i in 1:n_replicas]

    repsys = ReplicaSystem(
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        replica_velocities=nothing,
        n_replicas=n_replicas,
        boundary=boundary,
        replica_pairwise_inters=replica_pairwise_inters,
        replica_loggers=replica_loggers,
        neighbor_finder=neighbor_finder,
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

    # TODO: Possibly more tests?
end

@testset "Different implementations" begin
    n_atoms = 400
    atom_mass = 10.0u"u"
    boundary = CubicBoundary(6.0u"nm")
    temp = 1.0u"K"
    starting_coords = place_diatomics(n_atoms ÷ 2, boundary, 0.2u"nm"; min_dist=0.2u"nm")
    starting_velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]
    starting_coords_f32 = [Float32.(c) for c in starting_coords]
    starting_velocities_f32 = [Float32.(c) for c in starting_velocities]

    function test_sim(nl::Bool, parallel::Bool, f32::Bool, gpu::Bool)
        n_atoms = 400
        n_steps = 200
        atom_mass = f32 ? 10.0f0u"u" : 10.0u"u"
        boundary = f32 ? CubicBoundary(6.0f0u"nm") : CubicBoundary(6.0u"nm")
        simulator = VelocityVerlet(dt=f32 ? 0.02f0u"ps" : 0.02u"ps")
        k = f32 ? 10_000.0f0u"kJ * mol^-1 * nm^-2" : 10_000.0u"kJ * mol^-1 * nm^-2"
        r0 = f32 ? 0.2f0u"nm" : 0.2u"nm"
        bonds = [HarmonicBond(k=k, r0=r0) for i in 1:(n_atoms ÷ 2)]
        specific_inter_lists = (InteractionList2Atoms(
            gpu ? CuArray(Int32.(collect(1:2:n_atoms))) : Int32.(collect(1:2:n_atoms)),
            gpu ? CuArray(Int32.(collect(2:2:n_atoms))) : Int32.(collect(2:2:n_atoms)),
            fill("", length(bonds)),
            gpu ? CuArray(bonds) : bonds,
        ),)

        neighbor_finder = NoNeighborFinder()
        cutoff = DistanceCutoff(f32 ? 1.0f0u"nm" : 1.0u"nm")
        pairwise_inters = (LennardJones(nl_only=false, cutoff=cutoff),)
        if nl
            if gpu
                neighbor_finder = DistanceVecNeighborFinder(
                    nb_matrix=gpu ? CuArray(trues(n_atoms, n_atoms)) : trues(n_atoms, n_atoms),
                    n_steps=10,
                    dist_cutoff=f32 ? 1.5f0u"nm" : 1.5u"nm",
                )
            else
                neighbor_finder = DistanceNeighborFinder(
                    nb_matrix=trues(n_atoms, n_atoms),
                    n_steps=10,
                    dist_cutoff=f32 ? 1.5f0u"nm" : 1.5u"nm",
                )
            end
            pairwise_inters = (LennardJones(nl_only=true, cutoff=cutoff),)
        end
        show(devnull, neighbor_finder)

        if gpu
            coords = CuArray(deepcopy(f32 ? starting_coords_f32 : starting_coords))
            velocities = CuArray(deepcopy(f32 ? starting_velocities_f32 : starting_velocities))
            atoms = CuArray([Atom(charge=f32 ? 0.0f0 : 0.0, mass=atom_mass, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                                  ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms])
        else
            coords = deepcopy(f32 ? starting_coords_f32 : starting_coords)
            velocities = deepcopy(f32 ? starting_velocities_f32 : starting_velocities)
            atoms = [Atom(charge=f32 ? 0.0f0 : 0.0, mass=atom_mass, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                            ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms]
        end

        s = System(
            atoms=atoms,
            pairwise_inters=pairwise_inters,
            specific_inter_lists=specific_inter_lists,
            coords=coords,
            velocities=velocities,
            boundary=boundary,
            neighbor_finder=neighbor_finder,
        )

        @test float_type(s) == (f32 ? Float32 : Float64)

        n_threads = parallel ? Threads.nthreads() : 1
        neighbors = find_neighbors(s; n_threads=n_threads)
        E_start = potential_energy(s, neighbors; n_threads=n_threads)

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
