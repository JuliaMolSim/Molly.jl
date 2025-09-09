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
    cons = SHAKE_RATTLE(n_atoms, 1e-8u"Å", 1e-8u"Å^2 * ps^-1"; dist_constraints=constraints)

    @test length(cons.clusters12) == (n_atoms ÷ 2)

    boundary = CubicBoundary(200.0u"Å")
    lj = LennardJones(cutoff=ShiftedPotentialCutoff(r_cut), use_neighbors=true)
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        dist_cutoff=1.5*r_cut,
    )

    for simulator in simulators
        coords = [SVector(coords_matrix[j, 1]u"Å", coords_matrix[j, 2]u"Å", coords_matrix[j, 3]u"Å")
                  for j in 1:n_atoms]
        velocities = [1000 * SVector(vel_matrix[j, :]u"Å/ps"...) for j in 1:n_atoms]

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(lj,),
            neighbor_finder=neighbor_finder,
            constraints=(cons,),
            force_units=u"kcal * mol^-1 * Å^-1",
            energy_units=u"kcal * mol^-1",
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

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    bond_length = 0.13u"nm"

    is = collect(1:(2 * (n_atoms ÷ 3)))
    js = collect(((n_atoms ÷ 3) + 1):n_atoms)
    constraints = [DistanceConstraint(is[idx], js[idx], bond_length) for idx in eachindex(is)]
    cons = SHAKE_RATTLE(n_atoms, 1e-8u"nm",  1e-8u"nm^2/ps"; dist_constraints=constraints)

    @test length(cons.clusters23) == (n_atoms ÷ 3)

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
    )

    old_coords = copy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
end

@testset "Constraints 4-atom" begin
    n_atoms = 40
    atom_mass = 10.0u"g/mol"
    # Central atom
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(3.0u"nm")

    coords = place_atoms(n_atoms ÷ 4, boundary, min_dist=0.3u"nm")
    for i in 1:(n_atoms ÷ 4)
        push!(coords, coords[i] .+ [0.13, 0.0, 0.0]u"nm")
    end
    for i in 1:(n_atoms ÷ 4)
        push!(coords, coords[i] .- [0.13, 0.0, 0.0]u"nm")
    end
    for i in 1:(n_atoms ÷ 4)
        push!(coords, coords[i] .+ [0.0, 0.13, 0.0]u"nm")
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    bond_length = 0.13u"nm"

    is = repeat(1:(n_atoms ÷ 4), 3) # Central atom in each cluster
    js = collect(((n_atoms ÷ 4) + 1):n_atoms)
    constraints = [DistanceConstraint(is[idx], js[idx], bond_length) for idx in eachindex(is)]
    cons = SHAKE_RATTLE(n_atoms, 1e-8u"nm", 1e-8u"nm^2/ps"; dist_constraints=constraints)

    @test length(cons.clusters34) == (n_atoms ÷ 4)

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
    )

    old_coords = copy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
end

@testset "Constraints angle" begin
    n_atoms = 30
    atom_mass = 10.0u"g/mol"
    # Central atoms
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(3.0u"nm")

    coords = place_atoms(n_atoms ÷ 3, boundary, min_dist=0.3u"nm")
    for i in 1:(n_atoms ÷ 3)
        push!(coords, coords[i] .- [0.13, 0.0, 0.0]u"nm")
    end
    θ = 2π / 3
    for i in 1:(n_atoms ÷ 3)
        push!(coords, coords[i] .+ [0.13 * cos(π - θ), 0.13 * sin(π - θ), 0.0]u"nm")
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    bond_length = 0.13u"nm"

    js = collect(1:(n_atoms ÷ 3))
    is = collect(((n_atoms ÷ 3) + 1):(2 * (n_atoms ÷ 3)))
    ks = collect((2 * (n_atoms ÷ 3) + 1):n_atoms)
    angle_constraints = [AngleConstraint(is[idx], js[idx], ks[idx], θ, bond_length, bond_length)
                         for idx in eachindex(is)]
    cons = SHAKE_RATTLE(n_atoms, 1e-8u"nm", 1e-8u"nm^2/ps"; angle_constraints=angle_constraints)

    @test length(cons.angle_clusters) == (n_atoms ÷ 3)

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
    )

    old_coords = copy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
end

@testset "Constraints distance and angle" begin
    n_atoms = 60
    atom_mass = 10.0u"g/mol"
    # Central atoms
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(3.0u"nm")

    coords = place_atoms(20, boundary, min_dist=0.3u"nm")

    # Angle constraints
    for i in 1:(n_atoms ÷ 6) # Atoms 21 -30
        push!(coords, coords[i] .- [0.13, 0.0, 0.0]u"nm")
    end
    θ = 2π / 3
    for i in 1:(n_atoms ÷ 6) # Atoms 31 - 40
        push!(coords, coords[i] .+ [0.13 * cos(π - θ), 0.13 * sin(π - θ), 0.0]u"nm")
    end

    # Central atom constraints
    for i in ((n_atoms ÷ 6) + 1):(2 * (n_atoms ÷ 6)) # Atoms 41 - 50
        push!(coords, coords[i] .+ [0.13, 0.0, 0.0]u"nm")
    end
    for i in ((n_atoms ÷ 6) + 1):(2 * (n_atoms ÷ 6)) # Atoms 51 - 60
        push!(coords, coords[i] .+ [0.26, 0.0, 0.0]u"nm")
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    bond_length = 0.13u"nm"

    js = collect(1:(n_atoms ÷ 6)) # 1 - 10
    is = collect((2 * (n_atoms ÷ 6) + 1):(3 * (n_atoms ÷ 6))) # 21 - 30
    ks = collect((3 * (n_atoms ÷ 6) + 1):(4 * (n_atoms ÷ 6))) # 31 - 40
    angle_constraints = [AngleConstraint(is[idx], js[idx], ks[idx], θ, bond_length, bond_length)
                         for idx in eachindex(is)]

    is = repeat(collect(((n_atoms ÷ 6) + 1):(2 * (n_atoms ÷ 6))), 2) # 11 - 20
    js = collect((4 * (n_atoms ÷ 6) + 1):n_atoms) # 41 - 60
    distance_constraints = [DistanceConstraint(is[idx], js[idx], bond_length)
                            for idx in eachindex(is)]

    cons = SHAKE_RATTLE(n_atoms, 1e-8u"nm", 1e-8u"nm^2/ps"; dist_constraints=distance_constraints,
                        angle_constraints=angle_constraints)

    @test length(cons.angle_clusters) == length(angle_constraints)
    @test length(cons.clusters23) == 10

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder
    )

    old_coords = copy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
end

@testset "Constraints protein CPU/GPU" begin
    pdb_fp = joinpath(data_dir, "1ubq.pdb") # No solvent
    T = Float32
    ff = MolecularForceField(
        T,
        joinpath(data_dir, "force_fields", "ff99SBildn.xml"),
        joinpath(data_dir, "force_fields", "his.xml"),
    )
    boundary = CubicBoundary(T(10.0)u"nm")
    temp = T(100.0)u"K"
    minimizer = SteepestDescentMinimizer()
    simulator = VelocityVerlet(dt=T(0.001)u"ps")

    for AT in array_list
        for rigid_water in (false, true)
            sys = System(
                pdb_fp,
                ff;
                boundary=boundary,
                array_type=AT,
                constraints=:hbonds,
                rigid_water=rigid_water,
            )

            simulate!(sys, minimizer)
            random_velocities!(sys, temp)

            simulate!(sys, simulator, 20)
            @time simulate!(sys, simulator, 1000)

            @test check_position_constraints(sys, sys.constraints[1])
            @test check_velocity_constraints(sys, sys.constraints[1])

            coords_copy = copy(sys.coords)
            sys.coords     .+= randn(SVector{3, T}, length(sys))u"nm"         ./ 100
            sys.velocities .+= randn(SVector{3, T}, length(sys))u"nm * ps^-1" ./ 100
            @test !check_position_constraints(sys, sys.constraints[1])
            @test !check_velocity_constraints(sys, sys.constraints[1])

            apply_position_constraints!(sys, coords_copy)
            apply_velocity_constraints!(sys)
            @test check_position_constraints(sys, sys.constraints[1])
            @test check_velocity_constraints(sys, sys.constraints[1])
        end
    end
end
