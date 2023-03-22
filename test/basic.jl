@testset "Spatial" begin
    @test vector_1D(4.0, 6.0, 10.0) ==  2.0
    @test vector_1D(1.0, 9.0, 10.0) == -2.0
    @test vector_1D(6.0, 4.0, 10.0) == -2.0
    @test vector_1D(9.0, 1.0, 10.0) ==  2.0

    @test vector_1D(4.0u"nm", 6.0u"nm", 10.0u"nm") ==  2.0u"nm"
    @test vector_1D(1.0u"m" , 9.0u"m" , 10.0u"m" ) == -2.0u"m"
    @test_throws Unitful.DimensionError vector_1D(6.0u"nm", 4.0u"nm", 10.0)

    @test vector(
        SVector(4.0, 1.0, 6.0),
        SVector(6.0, 9.0, 4.0),
        CubicBoundary(SVector(10.0, 10.0, 10.0)),
    ) == SVector(2.0, -2.0, -2.0)
    @test vector(
        SVector(4.0, 1.0, 1.0),
        SVector(6.0, 4.0, 3.0),
        CubicBoundary(SVector(10.0, 5.0, 3.5)),
    ) == SVector(2.0, -2.0, -1.5)
    @test vector(
        SVector(4.0, 1.0),
        SVector(6.0, 9.0),
        RectangularBoundary(SVector(10.0, 10.0)),
    ) == SVector(2.0, -2.0)
    @test vector(
        SVector(4.0, 1.0, 6.0)u"nm",
        SVector(6.0, 9.0, 4.0)u"nm",
        CubicBoundary(SVector(10.0, 10.0, 10.0)u"nm"),
    ) == SVector(2.0, -2.0, -2.0)u"nm"

    @test wrap_coord_1D(8.0 , 10.0) == 8.0
    @test wrap_coord_1D(12.0, 10.0) == 2.0
    @test wrap_coord_1D(-2.0, 10.0) == 8.0

    @test wrap_coord_1D(8.0u"nm" , 10.0u"nm") == 8.0u"nm"
    @test wrap_coord_1D(12.0u"m" , 10.0u"m" ) == 2.0u"m"
    @test_throws ErrorException wrap_coord_1D(-2.0u"nm", 10.0)

    vels_units   = [maxwell_boltzmann(12.0u"u", 300.0u"K") for _ in 1:1_000]
    vels_nounits = [maxwell_boltzmann(12.0    , 300.0    ) for _ in 1:1_000]
    @test 0.35u"nm * ps^-1" < std(vels_units) < 0.55u"nm * ps^-1"
    @test 0.35 < std(vels_nounits) < 0.55

    b = CubicBoundary(4.0u"nm", 5.0u"nm", 6.0u"nm")
    @test box_volume(b) == 120.0u"nm^3"
    @test box_center(b) == SVector(2.0, 2.5, 3.0)u"nm"
    @test_throws DomainError CubicBoundary(-4.0u"nm", 5.0u"nm", 6.0u"nm")

    b = RectangularBoundary(4.0u"m", 5.0u"m")
    @test box_volume(b) == 20.0u"m^2"
    @test box_center(b) == SVector(2.0, 2.5)u"m"
    @test_throws DomainError RectangularBoundary(-4.0u"nm", 5.0u"nm")

    b = TriclinicBoundary(SVector(2.2, 2.0, 1.8)u"nm", deg2rad.(SVector(50.0, 40.0, 60.0)))
    @test isapprox(b.basis_vectors[1], SVector(2.2      , 0.0      , 0.0      )u"nm", atol=1e-6u"nm")
    @test isapprox(b.basis_vectors[2], SVector(1.0      , 1.7320508, 0.0      )u"nm", atol=1e-6u"nm")
    @test isapprox(b.basis_vectors[3], SVector(1.37888  , 0.5399122, 1.0233204)u"nm", atol=1e-6u"nm")

    @test isapprox(box_volume(b), 3.89937463181886u"nm^3")
    @test isapprox(box_center(b), SVector(2.28944, 1.1359815, 0.5116602)u"nm", atol=1e-6u"nm")

    @test_throws ArgumentError TriclinicBoundary(
        SVector(2.0, 1.0, 0.0)u"nm",
        SVector(1.0, 2.0, 0.0)u"nm",
        SVector(1.0, 1.0, 2.0)u"nm",
    )
    @test_throws ArgumentError TriclinicBoundary(
        SVector(2.2, 2.0, 1.8)u"nm",
        deg2rad.(SVector(190.0, 40.0, 60.0)),
    )

    n_atoms = 1_000
    coords = place_atoms(n_atoms, b; min_dist=0.01u"nm")
    @test wrap_coords.(coords, (b,)) == coords

    # Test approximation for minimum image is correct up to half the minimum height/width
    b_exact = TriclinicBoundary(b.basis_vectors; approx_images=false)
    correct_limit = min(b.basis_vectors[1][1], b.basis_vectors[2][2], b.basis_vectors[3][3]) / 2
    @test all(1:(n_atoms - 1)) do i
        c1 = coords[i]
        c2 = coords[i + 1]
        dr_exact = vector(c1, c2, b_exact)
        if norm(dr_exact) <= correct_limit
            dr_approx = vector(c1, c2, b)
            return isapprox(dr_exact, dr_approx)
        else
            return true
        end
    end

    atoms = fill(Atom(mass=1.0u"u"), n_atoms)
    loggers = (coords=CoordinateLogger(10),)
    temp = 100.0u"K"
    dt = 0.002u"ps"
    sim = VelocityVerlet(dt=dt, remove_CM_motion=false)
    sys = System(atoms=atoms, coords=coords, boundary=b, loggers=loggers)
    random_velocities!(sys, temp)
    starting_velocities = copy(sys.velocities)
    simulate!(sys, sim, 1_000)
    @test all(isapprox.(sys.velocities, starting_velocities))
    @test wrap_coords.(sys.coords, (b,)) == sys.coords

    # Test that displacements match those expected from the starting velocity,
    #   i.e. that coordinate wrapping hasn't caused any atoms to jump
    logged_coords = values(sys.loggers.coords)
    max_disps = map(1:n_atoms) do ci
        return maximum(1:(length(logged_coords) - 1)) do fi
            return norm(vector(logged_coords[fi][ci], logged_coords[fi + 1][ci], b_exact))
        end
    end
    @test isapprox(
        maximum(max_disps),
        norm(sys.velocities[argmax(max_disps)]) * sys.loggers.coords.n_steps * dt,
        atol=1e-9u"nm",
    )
end

@testset "Neighbor lists" begin
    for neighbor_finder in (DistanceNeighborFinder, TreeNeighborFinder, CellListMapNeighborFinder)
        nf = neighbor_finder(nb_matrix=trues(3, 3), n_steps=10, dist_cutoff=2.0u"nm")
        s = System(
            atoms=[Atom(), Atom(), Atom()],
            coords=[
                SVector(1.0, 1.0, 1.0)u"nm",
                SVector(2.0, 2.0, 2.0)u"nm",
                SVector(5.0, 5.0, 5.0)u"nm",
            ],
            boundary=CubicBoundary(10.0u"nm", 10.0u"nm", 10.0u"nm"),
            neighbor_finder=nf,
        )
        neighbors = find_neighbors(s, s.neighbor_finder; n_threads=1)
        @test neighbors.list == [(2, 1, false)] || neighbors.list == [(1, 2, false)]
        if run_parallel_tests
            neighbors = find_neighbors(s, s.neighbor_finder; n_threads=Threads.nthreads())
            @test neighbors.list == [(2, 1, false)] || neighbors.list == [(1, 2, false)]
        end
        show(devnull, nf)
    end

    # Test passing the boundary and coordinates as keyword arguments to CellListMapNeighborFinder
    coords = [
        SVector(1.0, 1.0, 1.0)u"nm",
        SVector(2.0, 2.0, 2.0)u"nm",
        SVector(5.0, 5.0, 5.0)u"nm",
    ]
    boundary = CubicBoundary(10.0u"nm", 10.0u"nm", 10.0u"nm")
    neighbor_finder=CellListMapNeighborFinder(
        nb_matrix=trues(3, 3), n_steps=10, x0=coords,
        unit_cell=boundary, dist_cutoff=2.0u"nm",
    )
    s = System(
        atoms=[Atom(), Atom(), Atom()],
        coords=coords,
        boundary=boundary,
        neighbor_finder=neighbor_finder,
    )
    neighbors = find_neighbors(s, s.neighbor_finder; n_threads=1)
    @test neighbors.list == [(2, 1, false)] || neighbors.list == [(1, 2, false)]
    if run_parallel_tests
        neighbors = find_neighbors(s, s.neighbor_finder; n_threads=Threads.nthreads())
        @test neighbors.list == [(2, 1, false)] || neighbors.list == [(1, 2, false)]
    end

    # Test CellListMapNeighborFinder with TriclinicBoundary
    boundary = TriclinicBoundary(
        SVector(2.0, 0.0, 0.0)u"nm",
        SVector(0.7, 1.8, 0.0)u"nm",
        SVector(0.5, 0.3, 1.6)u"nm",
    )
    n_atoms = 1_000
    coords = place_atoms(n_atoms, boundary; min_dist=0.01u"nm")
    atoms = fill(Atom(), n_atoms)
    dist_cutoff = 0.6u"nm"
    nf = CellListMapNeighborFinder(nb_matrix=trues(n_atoms, n_atoms), dist_cutoff=dist_cutoff)
    sys = System(atoms=atoms, coords=coords, boundary=boundary, neighbor_finder=nf)
    neighbors = find_neighbors(sys)
    neighbors_sorted = map(
        x -> (x[1], x[2]),
        sort(neighbors.list, lt=(x, y) -> x[1] < y[1] || (x[1] == y[1] && x[2] < y[2])),
    )
    neighbors_dist = Tuple{Int, Int}[]
    for i in 1:n_atoms
        for j in (i + 1):n_atoms
            if norm(vector(coords[i], coords[j], boundary)) <= dist_cutoff
                push!(neighbors_dist, (i, j))
            end
        end
    end
    @test neighbors_dist == neighbors_sorted
end

@testset "Analysis" begin
    pdb_path = joinpath(data_dir, "1ssu.pdb")
    struc = read(pdb_path, BioStructures.PDB)
    cm_1 = BioStructures.coordarray(struc[1], BioStructures.calphaselector)
    cm_2 = BioStructures.coordarray(struc[2], BioStructures.calphaselector)
    coords_1 = SVector{3, Float64}.(eachcol(cm_1)) / 10 * u"nm"
    coords_2 = SVector{3, Float64}.(eachcol(cm_2)) / 10 * u"nm"
    @test rmsd(coords_1, coords_2) ≈ 2.54859467758795u"Å"
    if run_gpu_tests
        @test rmsd(CuArray(coords_1), CuArray(coords_2)) ≈ 2.54859467758795u"Å"
    end

    bb_atoms = BioStructures.collectatoms(struc[1], BioStructures.backboneselector)
    coords = SVector{3, Float64}.(eachcol(BioStructures.coordarray(bb_atoms))) / 10 * u"nm"
    bb_to_mass = Dict("C" => 12.011u"u", "N" => 14.007u"u", "O" => 15.999u"u")
    atoms = [Atom(mass=bb_to_mass[BioStructures.element(bb_atoms[i])]) for i in 1:length(bb_atoms)]
    @test isapprox(radius_gyration(coords, atoms), 11.51225678195222u"Å", atol=1e-6u"nm")
end

@testset "Replica System" begin
    n_atoms = 100
    boundary = CubicBoundary(2.0u"nm", 2.0u"nm", 2.0u"nm") 
    temp = 298.0u"K"
    atom_mass = 10.0u"u"
    
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    replica_velocities = nothing
    pairwise_inters = (LennardJones(nl_only=true),)
    n_replicas = 4

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

    repsys = ReplicaSystem(;
        n_replicas=n_replicas,
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        replica_velocities=replica_velocities,
        pairwise_inters=pairwise_inters,
        boundary=boundary,
    )

    sys = System(;
        atoms=atoms,
        coords=coords,
        velocities=nothing,
        pairwise_inters=pairwise_inters,
        boundary=boundary,
    )

    for i in 1:n_replicas
        @test all(
            [getfield(repsys.replicas[i], f) for f in fieldnames(System)] .== [getfield(sys, f) for f in fieldnames(System)]
        )
    end

    repsys2 = ReplicaSystem(;
        n_replicas=n_replicas,
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        replica_velocities=replica_velocities,
        pairwise_inters=pairwise_inters,
        boundary=boundary,
        replica_loggers=[(temp=TemperatureLogger(10), coords=CoordinateLogger(10)) for i in 1:n_replicas],
        neighbor_finder=neighbor_finder,
    )

    sys2 = System(;
        atoms=atoms,
        coords=coords,
        velocities=nothing,
        pairwise_inters=pairwise_inters,
        boundary=boundary,
        loggers=(temp=TemperatureLogger(10), coords=CoordinateLogger(10)),
        neighbor_finder=neighbor_finder,
    )

    for i in 1:n_replicas
        l1 = repsys2.replicas[i].loggers
        l2 = sys2.loggers
        @test typeof(l1) == typeof(l2)
        @test propertynames(l1) == propertynames(l2)

        nf1 = [getproperty(repsys2.replicas[i].neighbor_finder, p) for p in propertynames(repsys2.replicas[i].neighbor_finder)]
        nf2 = [getproperty(sys2.neighbor_finder, p) for p in propertynames(sys2.neighbor_finder)]
        @test all(nf1 .== nf2)
    end
end
