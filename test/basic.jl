using LAMMPS
using Molly

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
        RectangularBoundary(10.0),
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

    vels_units_1    = [maxwell_boltzmann(12.0u"u", 300.0u"K", uconvert(u"u * nm^2 * ps^-2 * K^-1", Unitful.k)) for _ in 1:1_000]
    vels_units_2    = [maxwell_boltzmann(12.0u"u", 300.0u"K") for _ in 1:1_000]
    vels_molunits_1 = [maxwell_boltzmann(12.0u"g/mol", 300.0u"K", Unitful.k * Unitful.Na) for _ in 1:1_000]
    vels_molunits_2 = [maxwell_boltzmann(12.0u"g/mol", 300.0u"K") for _ in 1:1_000]
    vels_nounits_1  = [maxwell_boltzmann(12.0, 300.0, ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k)) for _ in 1:1_000]
    vels_nounits_2  = [maxwell_boltzmann(12.0, 300.0) for _ in 1:1_000]
    @test 0.35u"nm * ps^-1" < std(vels_units_1)    < 0.55u"nm * ps^-1"
    @test 0.35u"nm * ps^-1" < std(vels_units_2)    < 0.55u"nm * ps^-1"
    @test 0.35u"nm * ps^-1" < std(vels_molunits_1) < 0.55u"nm * ps^-1"
    @test 0.35u"nm * ps^-1" < std(vels_molunits_2) < 0.55u"nm * ps^-1"
    @test 0.35              < std(vels_nounits_1)  < 0.55
    @test 0.35              < std(vels_nounits_2)  < 0.55

    b = CubicBoundary(4.0u"nm", 5.0u"nm", 6.0u"nm")
    @test float_type(b) == Float64
    @test Molly.length_type(b) == typeof(1.0u"nm")
    @test ustrip(b) == CubicBoundary(4.0, 5.0, 6.0)
    @test ustrip(u"Å", b) == CubicBoundary(40.0, 50.0, 60.0)
    @test !Molly.has_infinite_boundary(b)
    @test volume(b) == 120.0u"nm^3"
    @test volume(CubicBoundary(0.0u"m"; check_positive=false)) == 0.0u"m^3"
    @test box_center(b) == SVector(2.0, 2.5, 3.0)u"nm"
    sb = scale_boundary(b, 1.1)
    @test sb.side_lengths ≈ SVector(4.4, 5.5, 6.6)u"nm"
    @test Molly.cubic_bounding_box(b) == SVector(4.0, 5.0, 6.0)u"nm"
    @test Molly.axis_limits(CubicBoundary(4.0, 5.0, 6.0), CoordinatesLogger(1), 2) == (0.0, 5.0)
    @test_throws DomainError CubicBoundary(-4.0u"nm", 5.0u"nm", 6.0u"nm")
    @test_throws DomainError CubicBoundary( 4.0u"nm", 0.0u"nm", 6.0u"nm")

    b = RectangularBoundary(4.0u"m", 5.0u"m")
    @test float_type(b) == Float64
    @test Molly.length_type(b) == typeof(1.0u"m")
    @test ustrip(b) == RectangularBoundary(4.0, 5.0)
    @test ustrip(u"km", b) == RectangularBoundary(4e-3, 5e-3)
    @test !Molly.has_infinite_boundary(b)
    @test volume(b) == 20.0u"m^2"
    @test volume(RectangularBoundary(0.0u"m"; check_positive=false)) == 0.0u"m^2"
    @test box_center(b) == SVector(2.0, 2.5)u"m"
    sb = scale_boundary(b, 0.9)
    @test sb.side_lengths ≈ SVector(3.6, 4.5)u"m"
    @test Molly.cubic_bounding_box(b) == SVector(4.0, 5.0)u"m"
    @test Molly.axis_limits(RectangularBoundary(4.0, 5.0), CoordinatesLogger(1), 2) == (0.0, 5.0)
    @test_throws DomainError RectangularBoundary(-4.0u"nm", 5.0u"nm")
    @test_throws DomainError RectangularBoundary( 4.0u"nm", 0.0u"nm")

    b = TriclinicBoundary(SVector(2.2, 2.0, 1.8)u"nm", deg2rad.(SVector(50.0, 40.0, 60.0)))
    @test float_type(b) == Float64
    @test Molly.length_type(b) == typeof(1.0u"nm")
    @test isapprox(b.basis_vectors[1], SVector(2.2      , 0.0      , 0.0      )u"nm"; atol=1e-6u"nm")
    @test isapprox(b.basis_vectors[2], SVector(1.0      , 1.7320508, 0.0      )u"nm"; atol=1e-6u"nm")
    @test isapprox(b.basis_vectors[3], SVector(1.37888  , 0.5399122, 1.0233204)u"nm"; atol=1e-6u"nm")
    @test TriclinicBoundary(b.basis_vectors) == b
    @test TriclinicBoundary([b.basis_vectors[1], b.basis_vectors[2], b.basis_vectors[3]]) == b

    @test AtomsBase.cell_vectors(b) == (b.basis_vectors[1], b.basis_vectors[2], b.basis_vectors[3])
    @test volume(b) ≈ 3.89937463181886u"nm^3"
    @test isapprox(box_center(b), SVector(2.28944, 1.1359815, 0.5116602)u"nm"; atol=1e-6u"nm")
    sb = scale_boundary(b, 1.2)
    @test [sb.α, sb.β, sb.γ] ≈ [b.α, b.β, b.γ]
    @test volume(sb) ≈ volume(b) * 1.2^3
    @test isapprox(
        Molly.cubic_bounding_box(b),
        SVector(4.5788800, 2.2719630, 1.0233205)u"nm";
        atol=1e-6u"nm",
    )

    @test_throws ArgumentError TriclinicBoundary(
        SVector(2.0, 1.0, 0.0)u"nm",
        SVector(1.0, 2.0, 0.0)u"nm",
        SVector(1.0, 1.0, 2.0)u"nm",
    )
    @test_throws ArgumentError TriclinicBoundary(
        SVector(-2.2, 2.0, 1.8)u"nm",
        deg2rad.(SVector(20.0, 40.0, 60.0)),
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
            return dr_exact ≈ dr_approx
        else
            return true
        end
    end

    atoms = fill(Atom(mass=1.0u"g/mol"), n_atoms)
    loggers = (coords=CoordinatesLogger(10),)
    temp = 100.0u"K"
    dt = 0.002u"ps"
    sim = VelocityVerlet(dt=dt, remove_CM_motion=false)
    sys = System(atoms=atoms, coords=coords, boundary=b, loggers=loggers)
    random_velocities!(sys, temp)
    starting_velocities = copy(sys.velocities)
    simulate!(sys, sim, 1_000)
    @test all(sys.velocities .≈ starting_velocities)
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
        norm(sys.velocities[argmax(max_disps)]) * sys.loggers.coords.n_steps * dt;
        atol=1e-9u"nm",
    )

    coords = [SVector(1.95, 0.0, 0.0), SVector(0.05, 0.0, 0.0), SVector(0.15, 0.0, 0.0),
              SVector(1.0 , 1.0, 1.0)]
    boundary = CubicBoundary(2.0)
    topology = MolecularTopology([1, 1, 1, 2], [3, 1], [(1, 2), (2, 3)])
    mcs = molecule_centers(coords, boundary, topology)
    @test mcs == [SVector(0.05, 0.0, 0.0), SVector(1.0, 1.0, 1.0)]

    coords = [SVector(1.95, 0.0), SVector(0.05, 0.0), SVector(0.15, 0.0),
              SVector(1.0 , 1.0)]
    boundary = RectangularBoundary(2.0)
    mcs = molecule_centers(coords, boundary, topology)
    @test mcs == [SVector(0.05, 0.0), SVector(1.0, 1.0)]

    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...)
    for AT in array_list
        sys = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            array_type=AT,
            nonbonded_method="cutoff",
            neighbor_finder_type=(Molly.uses_gpu_neighbor_finder(AT) ? GPUNeighborFinder :
                                    DistanceNeighborFinder),
        )
        mcs = molecule_centers(sys.coords, sys.boundary, sys.topology)
        @test isapprox(from_device(mcs)[1], mean(sys.coords[1:1170]); atol=0.08u"nm")

        # Mark all pairs as ineligible for pairwise interactions and check that the
        #   potential energy from the specific interactions does not change on scaling
        no_nbs = falses(length(sys), length(sys))
        if Molly.uses_gpu_neighbor_finder(AT)
            sys.neighbor_finder = GPUNeighborFinder(
                eligible=to_device(no_nbs, AT),
                dist_cutoff=1.0u"nm",
            )
        else
            sys.neighbor_finder = DistanceNeighborFinder(
                eligible=to_device(no_nbs, AT),
                dist_cutoff=1.0u"nm",
            )
        end
        coords_start = copy(sys.coords)
        pe_start = potential_energy(sys, find_neighbors(sys))
        scale_factor = 1.02
        n_scales = 10

        for i in 1:n_scales
            scale_coords!(sys, scale_factor)
            @test potential_energy(sys, find_neighbors(sys)) ≈ pe_start
        end
        for i in 1:n_scales
            scale_coords!(sys, inv(scale_factor))
            @test potential_energy(sys, find_neighbors(sys)) ≈ pe_start
        end
        coords_diff = from_device(sys.coords) .- from_device(coords_start)
        @test maximum(maximum(abs.(v)) for v in coords_diff) < 5e-4u"nm"
    end
end

@testset "Neighbor lists" begin
    reorder_neighbors(nbs) = map(t -> (min(t[1], t[2]), max(t[1], t[2]), t[3]), nbs)

    for neighbor_finder in (DistanceNeighborFinder, TreeNeighborFinder, CellListMapNeighborFinder)
        nf = neighbor_finder(eligible=trues(3, 3), n_steps=10, dist_cutoff=2.0u"nm")
        s = System(
            atoms=[Atom(), Atom(), Atom()],
            coords=[
                SVector(1.0, 1.0, 1.0)u"nm",
                SVector(2.0, 2.0, 2.0)u"nm",
                SVector(5.0, 5.0, 5.0)u"nm",
            ],
            boundary=CubicBoundary(10.0u"nm"),
            neighbor_finder=nf,
        )
        neighbors = find_neighbors(s, s.neighbor_finder; n_threads=1)
        @test reorder_neighbors(neighbors.list) == [(Int32(1), Int32(2), false)]
        if run_parallel_tests
            neighbors = find_neighbors(s, s.neighbor_finder; n_threads=Threads.nthreads())
            @test reorder_neighbors(neighbors.list) == [(Int32(1), Int32(2), false)]
        end
        show(devnull, nf)
    end

    # Test passing the boundary and coordinates as keyword arguments to CellListMapNeighborFinder
    coords = [
        SVector(1.0, 1.0, 1.0)u"nm",
        SVector(2.0, 2.0, 2.0)u"nm",
        SVector(5.0, 5.0, 5.0)u"nm",
    ]
    boundary = CubicBoundary(10.0u"nm")
    neighbor_finder=CellListMapNeighborFinder(
        eligible=trues(3, 3), n_steps=10, x0=coords,
        unit_cell=boundary, dist_cutoff=2.0u"nm",
    )
    sys = System(
        atoms=[Atom(), Atom(), Atom()],
        coords=coords,
        boundary=boundary,
        neighbor_finder=neighbor_finder,
    )
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=1)
    @test reorder_neighbors(neighbors.list) == [(Int32(1), Int32(2), false)]
    if run_parallel_tests
        neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=Threads.nthreads())
        @test reorder_neighbors(neighbors.list) == [(Int32(1), Int32(2), false)]
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
    nf = CellListMapNeighborFinder(eligible=trues(n_atoms, n_atoms), dist_cutoff=dist_cutoff)
    sys = System(atoms=atoms, coords=coords, boundary=boundary, neighbor_finder=nf)
    neighbors = find_neighbors(sys)

    function sort_nbs(nbs_dev)
        nbs = from_device(nbs_dev)
        return sort(
            reorder_neighbors(nbs),
            lt=(t1, t2) -> t1[1] < t2[1] || (t1[1] == t2[1] && t1[2] < t2[2]),
        )
    end

    neighbors_dist = Tuple{Int32, Int32, Bool}[]
    for i in 1:n_atoms
        for j in (i + 1):n_atoms
            if norm(vector(coords[i], coords[j], boundary)) <= dist_cutoff
                push!(neighbors_dist, (Int32(i), Int32(j), false))
            end
        end
    end
    @test neighbors_dist == sort_nbs(neighbors.list)

    # Test all neighbor finders agree for a larger system
    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...)
    dist_cutoff = 1.2u"nm"
    sys = System(joinpath(data_dir, "6mrr_equil.pdb"), ff; dist_neighbors=dist_cutoff)
    neighbors_ref = find_neighbors(sys)
    n_neighbors_ref = 4602420
    @test length(neighbors_ref) == neighbors_ref.n == n_neighbors_ref

    identical_neighbors(nl1, nl2) = (nl1.n == nl2.n && sort_nbs(nl1.list) == sort_nbs(nl2.list))

    for neighbor_finder in (DistanceNeighborFinder, TreeNeighborFinder, CellListMapNeighborFinder)
        nf = neighbor_finder(
            eligible=sys.neighbor_finder.eligible,
            special=sys.neighbor_finder.special,
            dist_cutoff=dist_cutoff,
        )
        for n_threads in n_threads_list
            neighbors = find_neighbors(sys, nf; n_threads=n_threads)
            @test length(neighbors) == n_neighbors_ref
            @test neighbors[10] isa Tuple{Int32, Int32, Bool}
            @test identical_neighbors(neighbors, neighbors_ref)
        end
    end

    for AT in array_list[2:end]
        sys_gpu = System(joinpath(data_dir, "6mrr_equil.pdb"), ff; array_type=AT)
        for neighbor_finder in (DistanceNeighborFinder,)
            nf_gpu = neighbor_finder(
                eligible=sys_gpu.neighbor_finder.eligible,
                special=sys_gpu.neighbor_finder.special,
                dist_cutoff=dist_cutoff,
            )
            neighbors_gpu = find_neighbors(sys_gpu, nf_gpu)
            @test length(neighbors_gpu) == n_neighbors_ref
            GPUArrays.allowscalar() do
                @test neighbors_gpu[10] isa Tuple{Int32, Int32, Bool}
            end
            @test identical_neighbors(neighbors_gpu, neighbors_ref)
        end
    end
end

@testset "Analysis" begin
    pdb_path = joinpath(data_dir, "1ssu.pdb")
    struc = read(pdb_path, BioStructures.PDBFormat)
    cm_1 = BioStructures.coordarray(struc[1], BioStructures.calphaselector)
    cm_2 = BioStructures.coordarray(struc[2], BioStructures.calphaselector)
    coords_1 = SVector{3, Float64}.(eachcol(cm_1)) / 10 * u"nm"
    coords_2 = SVector{3, Float64}.(eachcol(cm_2)) / 10 * u"nm"
    @test rmsd(coords_1, coords_2) ≈ 2.54859467758795u"Å"
    for AT in array_list[2:end]
        @test rmsd(to_device(coords_1, AT), to_device(coords_2, AT)) ≈ 2.54859467758795u"Å"
    end

    bb_atoms = BioStructures.collectatoms(struc[1], BioStructures.backboneselector)
    coords = SVector{3, Float64}.(eachcol(BioStructures.coordarray(bb_atoms))) / 10 * u"nm"
    bb_to_mass = Dict("C" => 12.011u"g/mol", "N" => 14.007u"g/mol", "O" => 15.999u"g/mol")
    atoms = [Atom(mass=bb_to_mass[BioStructures.element(bb_atoms[i])]) for i in eachindex(bb_atoms)]
    @test isapprox(radius_gyration(coords, atoms), 11.51225678195222u"Å"; atol=1e-6u"nm")
    boundary = CubicBoundary(10.0u"nm")
    coords_wrap = wrap_coords.(coords, (boundary,))
    @test isapprox(hydrodynamic_radius(coords_wrap, boundary), 21.00006825680275u"Å"; atol=1e-6u"nm")
end

@testset "Replica System" begin
    n_atoms = 100
    boundary = CubicBoundary(2.0u"nm")
    temp = 298.0u"K"
    atom_mass = 10.0u"g/mol"

    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    replica_velocities = nothing
    pairwise_inters = (LennardJones(use_neighbors=true),)
    n_replicas = 4

    eligible = trues(n_atoms, n_atoms)
    for i in 1:(n_atoms ÷ 2)
        eligible[i, i + (n_atoms ÷ 2)] = false
        eligible[i + (n_atoms ÷ 2), i] = false
    end

    neighbor_finder = DistanceNeighborFinder(
        eligible=eligible,
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    repsys = ReplicaSystem(
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        boundary=boundary,
        n_replicas=n_replicas,
        replica_velocities=replica_velocities,
        pairwise_inters=pairwise_inters,
    )

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=nothing,
        pairwise_inters=pairwise_inters,
    )

    for i in 1:n_replicas
        repsys_fields = [getfield(repsys.replicas[i], f) for f in fieldnames(System)]
        sys_fields = [getfield(sys, f) for f in fieldnames(System)]
        @test all(repsys_fields .== sys_fields)
    end

    repsys2 = ReplicaSystem(
        atoms=atoms,
        replica_coords=[copy(coords) for _ in 1:n_replicas],
        boundary=boundary,
        n_replicas=n_replicas,
        replica_velocities=replica_velocities,
        pairwise_inters=pairwise_inters,
        neighbor_finder=neighbor_finder,
        replica_loggers=[(temp=TemperatureLogger(10), coords=CoordinatesLogger(10))
                         for i in 1:n_replicas],
        data="test_data_repsys",
    )

    sys2 = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=nothing,
        pairwise_inters=pairwise_inters,
        neighbor_finder=neighbor_finder,
        loggers=(temp=TemperatureLogger(10), coords=CoordinatesLogger(10)),
        data="test_data_sys",
    )

    @test repsys2.data == "test_data_repsys"
    @test sys2.data == "test_data_sys"

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

@testset "Invalid units" begin
    @test_throws MethodError random_velocity(10.0u"g/mol", 10u"K", Unitful.k)
    @test_throws MethodError random_velocity(10.0u"g", 10u"K", Unitful.k * Unitful.Na)

    # Incorrect units in boundary and coords
    b = CubicBoundary(10.0u"Å")
    atoms = [Atom(mass=1.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")]
    coords = place_atoms(1, b; min_dist=0.01u"nm")
    @test_throws ArgumentError System(atoms=atoms, coords=coords, boundary=b)

    # Incorrect just in boundary
    b_wrong = CubicBoundary(10.0u"Å")
    b_right = CubicBoundary(10.0u"nm")
    atoms = [Atom(mass=1.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")]
    coords = place_atoms(1, b_right; min_dist=0.01u"nm")
    @test_throws ArgumentError System(atoms=atoms, coords=coords, boundary=b_wrong)

    # Mixed units or other invalid units
    bad_velo = [random_velocity(1.0u"g/mol",10u"K",Unitful.k*Unitful.Na) .* 2u"g"]
    @test_throws ArgumentError System(atoms=atoms, coords=coords, boundary=b_right,
                                      velocities=bad_velo)

    bad_coord = place_atoms(1, b_right; min_dist=0.01u"nm") .* u"ps"
    @test_throws ArgumentError System(atoms=atoms, coords=bad_coord, boundary=b_right)

    good_velo = [random_velocity(1.0u"g/mol", 10u"K",Unitful.k*Unitful.Na)]
    @test_throws ArgumentError System(atoms=atoms, coords=coords, boundary=b_right,
        velocities=good_velo, energy_units=NoUnits, force_units=NoUnits)

    # Inconsistent molar and non-molar quantities
    atoms = [Atom(mass=1.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ")]
    @test_throws ArgumentError System(atoms=atoms, coords=coords, boundary=b_right,
        velocities=good_velo, energy_units=u"kJ")
end

@testset "AtomsBase conversion" begin
    ab_sys_1 = make_test_system().system
    # Update values to be something that works with Molly
    ab_sys_2 = AtomsBase.AbstractSystem(
        ab_sys_1;
        cell_vectors = [[1.54732, 0.0      , 0.0],
                        [0.0    , 1.4654985, 0.0],
                        [0.0    , 0.0      , Inf]]u"Å",
    )
    # Suppress origin warning
    @suppress_err begin
        molly_sys = System(ab_sys_2; energy_units=u"kJ", force_units=u"kJ/Å")
        test_approx_eq(ab_sys_2, molly_sys; common_only=true)
    end
end

@testset "AtomsCalculators" begin
    ab_sys = AtomsBase.AbstractSystem(
        make_test_system().system;
        cell_vectors = [[1.54732, 0.0      , 0.0      ],
                        [0.0    , 1.4654985, 0.0      ],
                        [0.0    , 0.0      , 1.7928950]]u"Å",
    )
    coul = Coulomb(coulomb_const=2.307e-21u"kJ*Å")
    calc = MollyCalculator(pairwise_inters=(coul,), force_units=u"kJ/Å", energy_units=u"kJ")

    # Suppress origin warning
    @suppress_err begin
        pe = AtomsCalculators.potential_energy(ab_sys, calc)
        @test unit(pe) == u"kJ"
        fs = AtomsCalculators.forces(ab_sys, calc)
        @test length(fs) == length(ab_sys)
        @test unit(fs[1][1]) == u"kJ/Å"
        zfs = AtomsCalculators.zero_forces(ab_sys, calc)
        @test zfs == fill(SVector(0.0, 0.0, 0.0)u"kJ/Å", length(ab_sys))

        # AtomsCalculators.AtomsCalculatorsTesting functions
        test_potential_energy(ab_sys, calc)
        test_forces(ab_sys, calc)
    end
end


@testset "LAMMPSCalculator" begin
    
    LAMMPS.MPI.Init()

    dt = 1.0u"fs"
    damping = 0.5u"ps^-1"

    ar_crystal = FCC(5.2468u"Å", :Ar, SVector(4,4,4))
    ar_crystal_real = FCC(5.2468u"Å", 39.95u"g/mol", SVector(4,4,4))
    diamond_crystal = Diamond(5.43u"Å", :Si, SVector(3, 3, 3))
    al_crystal = FCC(4.041u"Å", :Al, SVector(4,4,4))

    pot_basepath = abspath(dirname(LAMMPS.locate()), "..", "share", "lammps", "potentials")

    eam_pot = joinpath(pot_basepath, "Al_zhou.eam.alloy")
    sw_pot = joinpath(pot_basepath, "Si.sw")

    # these are LJ argon params, but will use with silicon just to see if energy the same.
    lj_cmds = ["pair_style lj/cut 8.5", "pair_coeff * * 0.0104 3.4", "pair_modify shift yes"]
    lj_cmds_real = ["pair_style lj/cut 8.5", "pair_coeff * * 0.24037 3.4", "pair_modify shift yes"]
    sw_cmds = ["pair_style sw", "pair_coeff * * \"$(sw_pot)\" Si"]
    eam_cmds = ["pair_style eam/alloy", "pair_coeff * * \"$(eam_pot)\" Al"]

    pots = (
        (lj_cmds, ar_crystal, "LJ", -19.85644u"eV"),
        (lj_cmds_real, ar_crystal_real, "LJ-real", -458.93197u"kcal/mol"),
        (sw_cmds, diamond_crystal, "SW", -936.70522u"eV"),
        (eam_cmds, al_crystal, "EAM", -915.27403u"eV")
    )

    for (pot_cmd, crys, pot_type, E_pot) in pots

        unit_sys = pot_type == "LJ-real" ? "real" : "metal"

        sys = System(
            crys,
            energy_units = unit(E_pot),
            force_units = unit(E_pot) / u"angstrom",
            loggers = (PotentialEnergyLogger(typeof(1.0 * unit(E_pot)), 10),)
        )

        if pot_type == "LJ_real"
            new_atoms_data = []
            for i in eachindex(sys)
                push!(new_atoms_data, Molly.AtomData(element = "Ar"))
            end

            sys = System(sys; atoms_data=[new_atoms_data...])
        end

        inter = LAMMPSCalculator(
            sys,
            unit_sys, 
            pot_cmd;
            logfile_path = joinpath(@__DIR__, "log.lammps"),
            calculate_potential = true
        )

        random_velocities!(sys, 100u"K")

        sys = System(sys; general_inters = (inter, ))

        sim = Langevin(dt = dt, temperature = 100u"K", friction = damping)

        simulate!(sys, sim, 1)
        PE_LAMMPS = values(sys.loggers[1])[1]

        @test PE_LAMMPS ≈ E_pot

        LAMMPS.close!(inter.lmp)
    end

    sys = System(
            ar_crystal,
            energy_units = u"eV",
            force_units = u"eV / angstrom",
            loggers = (PotentialEnergyLogger(typeof(1.0u"eV"), 10),)
    )

    @test_throws ArgumentError LAMMPSCalculator(
        sys,
        "metal", 
        eam_cmds;
        label_type_map = Dict(:Al => 1, :Si => 2, :Ar => 3),
        logfile_path = joinpath(@__DIR__, "log.lammps"),
        calculate_potential = true
    )

    @test_throws ArgumentError LAMMPSCalculator(
        sys,
        "real", 
        eam_cmds;
        label_type_map = Dict(:Al => 1),
        logfile_path = joinpath(@__DIR__, "log.lammps"),
        calculate_potential = true
    )

    @test_throws ArgumentError LAMMPSCalculator(
        sys,
        "fake-unit-system", 
        eam_cmds;
        label_type_map = Dict(:Al => 1),
        logfile_path = joinpath(@__DIR__, "log.lammps"),
        calculate_potential = true
    )

    LAMMPS.MPI.Finalize()


end