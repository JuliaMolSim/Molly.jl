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
    mcs = Molly.molecule_centers(coords, boundary, topology)
    @test isapprox(mcs, [SVector(0.05, 0.0, 0.0), SVector(1.0, 1.0, 1.0)]; atol=1e-6)

    coords = [SVector(1.95, 0.0), SVector(0.05, 0.0), SVector(0.15, 0.0),
              SVector(1.0 , 1.0)]
    boundary = RectangularBoundary(2.0)
    mcs = Molly.molecule_centers(coords, boundary, topology)
    @test isapprox(mcs, [SVector(0.05, 0.0), SVector(1.0, 1.0)]; atol=1e-6)

    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...)
    for AT in array_list
        sys = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            array_type=AT,
            nonbonded_method=:cutoff,
            neighbor_finder_type=(Molly.uses_gpu_neighbor_finder(AT) ? GPUNeighborFinder :
                                    DistanceNeighborFinder),
        )
        mcs = Molly.molecule_centers(sys.coords, sys.boundary, sys.topology)
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
        scale_factor = SMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.02
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

@testset "Trajectory" begin
    trj_path = joinpath(data_dir, "water_frames", "water_trj.dcd")
    ff = MolecularForceField(joinpath(ff_dir, "tip3p_standard.xml"); units=true)
    # The small distance cutoff is required so that the neighbor finder does not
    #   complain about the small unit cell
    sys = System(joinpath(data_dir, "water_3mol_cubic.pdb"), ff; dist_cutoff=0.5u"nm")
    traj_sys = EnsembleSystem(sys, trj_path)
    n_frames = Int(length(traj_sys.trajectory))

    for n in 1:n_frames
        current_frame = read_frame!(traj_sys, n)
        pdb_sys = System(joinpath(data_dir, "water_frames", "frame_$(n).pdb"), ff;
                                  dist_cutoff=0.5u"nm")
        p1 = current_frame.coords[1]
        p2 = pdb_sys.coords[1]
        @test isapprox(p1, p2; rtol=0.001) # isapprox due to rounding errors in PDB file
    end
end

@testset "Structure file formats" begin
    ff = MolecularForceField(
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "gaff.xml", "imatinib.xml",
                           "imatinib_frcmod.xml"])...;
        units=true,
    )
    ff_custom = MolecularForceField(
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "gaff.xml", "imatinib.xml",
                           "imatinib_frcmod.xml"])...;
        units=true,
        custom_residue_templates=joinpath(data_dir, "imatinib_topo.xml"),
    )
    boundary = CubicBoundary(Inf*u"nm")

    # Suppress MOL2 invalid sybyl type warning
    @suppress_err begin
        sys_mol2         = System(joinpath(data_dir, "imatinib.mol2"), ff; boundary=boundary)
        sys_pdb_connect  = System(joinpath(data_dir, "imatinib_conect.pdb"), ff; boundary=boundary)
        sys_pdb          = System(joinpath(data_dir, "imatinib.pdb"), ff_custom; boundary=boundary)

        @test sys_mol2.topology.bonded_atoms == sys_pdb_connect.topology.bonded_atoms
        @test sys_mol2.topology.bonded_atoms == sys_pdb.topology.bonded_atoms
        @test_throws ArgumentError System(joinpath(data_dir, "imatinib.pdb"), ff; boundary=boundary)
    end
end

@testset "System setup" begin
    FT = Float64
    AT = Array

    ff = MolecularForceField(
        FT,
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
        units=true,
    )

    struc_names = [
        "a-synuclein_1",
        "barn_bar",
        "bpti",
        "cd2_cd58",
        "cole7_im7",
        "drkN_SH3_1",
        "gb3",
        "hewl",
        "NTail_1",
        "PaaA2_1",
        "sgpb_omtky3",
        "ubiquitin",
    ]

    for struc_name in struc_names
        dat_file = joinpath(data_dir, "openmm_refs", "$struc_name.dat")
        pdb_file = joinpath(data_dir, "openmm_refs", "$struc_name.pdb")

        sys = System(
            pdb_file,
            ff;
            array_type=AT,
            nonbonded_method=:pme,
            approximate_pme=false,
            disulfide_bonds=true,
        )

        if struc_name == "sgpb_omtky3"
            # Catch if disulfide bonds are not added properly
            @test_throws ArgumentError System(
                pdb_file,
                ff;
                array_type = AT,
                nonbonded_method=:pme,
                approximate_pme=false,
                disulfide_bonds=false,
            )
        end

        fs_openmm = SVector{3}[]
        open(dat_file, "r") do f
            for line in readlines(f)
                cols = split(line, ",")
                f = SVector{3}([parse(FT, split(val, " ")[1])*u"kJ * mol^-1 * nm^-1"
                                for val in cols])
                push!(fs_openmm, f)
            end
        end

        diff = mean(norm.(forces(sys) .- fs_openmm))
        @test diff < FT(0.15)u"kJ * mol^-1 * nm^-1"
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
    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...)
    dist_cutoff = 1.2u"nm"
    sys = System(joinpath(data_dir, "6mrr_equil.pdb"), ff;
                 dist_cutoff=dist_cutoff, dist_buffer=0.0u"nm")
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
        neighbor_finder=neighbor_finder,
    )

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=nothing,
        pairwise_inters=pairwise_inters,
        neighbor_finder=neighbor_finder,
    )

    sys_fields = [getfield(sys, f) for f in fieldnames(System) if f != :neighbor_finder]
    for i in 1:n_replicas
        repsys_fields = [getfield(repsys.replicas[i], f)
                         for f in fieldnames(System) if f != :neighbor_finder]
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

    l2 = sys2.loggers
    nf2 = [getproperty(sys2.neighbor_finder, p) for p in propertynames(sys2.neighbor_finder)]
    for i in 1:n_replicas
        l1 = repsys2.replicas[i].loggers
        @test typeof(l1) == typeof(l2)
        @test propertynames(l1) == propertynames(l2)
        nf1 = [getproperty(repsys2.replicas[i].neighbor_finder, p)
               for p in propertynames(repsys2.replicas[i].neighbor_finder)]
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
    bad_velo = [random_velocity(1.0u"g/mol", 10u"K", Unitful.k * Unitful.Na) .* 2u"g"]
    @test_throws ArgumentError System(atoms=atoms, coords=coords, boundary=b_right,
                                      velocities=bad_velo)

    bad_coord = place_atoms(1, b_right; min_dist=0.01u"nm") .* u"ps"
    @test_throws ArgumentError System(atoms=atoms, coords=bad_coord, boundary=b_right)

    good_velo = [random_velocity(1.0u"g/mol", 10u"K", Unitful.k * Unitful.Na)]
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

@testset "Virtual sites" begin
    for AT in array_list
        for units in (false, true)
            if units
                LU, MU, EU, FU = u"nm", u"g/mol", u"kJ * mol^-1", u"kJ * mol^-1 * nm^-1"
                TU, AU = u"K", u"nm * ps^-2" 
            else
                LU, MU, EU, FU, TU, AU = NoUnits, NoUnits, NoUnits, NoUnits, NoUnits, NoUnits
            end
            vs_flags = to_device(BitVector([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]), AT)
            atom_masses = map(x -> (x ? 0.0 : 10.0), vs_flags) .* MU
            atoms = to_device([Atom(mass=m, σ=(0.1 * LU), ϵ=(0.2 * EU))
                               for m in from_device(atom_masses)], AT)
            coords = to_device([
                SVector(2.0, 2.0, 2.0),
                SVector(2.0, 2.0, 2.2),
                SVector(1.0, 1.0, 1.0),
                SVector(3.0, 3.0, 3.0),
                SVector(3.0, 3.0, 3.2),
                SVector(3.0, 3.2, 3.0),
                SVector(1.0, 1.0, 1.0),
                SVector(4.0, 4.0, 4.0),
                SVector(4.0, 4.0, 4.2),
                SVector(4.0, 4.2, 4.0),
                SVector(1.0, 1.0, 1.0),
                SVector(1.0, 1.0, 1.0),
            ] * LU, AT)
            boundary = CubicBoundary(7.0 * LU)
            virtual_sites = to_device([
                TwoParticleAverageSite(3, 1, 2, 0.6, 0.4),
                ThreeParticleAverageSite(7, 4, 5, 6, 0.2, 0.3, 0.5),
                OutOfPlaneSite(11, 8, 9, 10, 0.4, 0.4, 0.2 / LU),
                TwoParticleAverageSite(12, 10, 9, 0.7, 0.3),
            ], AT)
            pis = (LennardJones(),)
            sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                pairwise_inters=pis,
                virtual_sites=virtual_sites,
                force_units=FU,
                energy_units=EU,
            )

            @test Molly.calc_virtual_site_flags(virtual_sites, atom_masses, AT) == vs_flags
            place_virtual_sites!(sys)
            coords_true = to_device([
                SVector(2.0  , 2.0 , 2.0 ),
                SVector(2.0  , 2.0 , 2.2 ),
                SVector(2.0  , 2.0 , 2.08),
                SVector(3.0  , 3.0 , 3.0 ),
                SVector(3.0  , 3.0 , 3.2 ),
                SVector(3.0  , 3.2 , 3.0 ),
                SVector(3.0  , 3.1 , 3.06),
                SVector(4.0  , 4.0 , 4.0 ),
                SVector(4.0  , 4.0 , 4.2 ),
                SVector(4.0  , 4.2 , 4.0 ),
                SVector(3.992, 4.08, 4.08),
                SVector(4.0  , 4.14, 4.06),
            ] * LU, AT)
            @test maximum(norm, sys.coords .- coords_true) < (1e-10 * LU)

            @test potential_energy(sys) ≈ 176.755677721122 * EU
            fs = forces(sys)
            fs_true = to_device([
                SVector(2.925933983564653e-7, 3.1125193300777895e-7, -603.9218830270745),
                SVector(3.599618866597804e-7, 3.822959459233062e-7, 603.9218836444344),
                SVector(0.0, 0.0, 0.0),
                SVector(2.5808104725193487e-9, 3.1230191568812646, 1.559911900509554),
                SVector(2.4391188018790913e-7, 0.42728790372500036, -1.9871993119974722),
                SVector(2.855779272863662e-7, -3.5503065786187276, 0.42728796565238336),
                SVector(0.0, 0.0, 0.0),
                SVector(-3.771624506043736e-7, -6239.653288137466, 1866.7884395038172),
                SVector(-3.6238225220586173e-7, -3179.2670135477856, 1312.478572953156),
                SVector(-4.4508124119602144e-7, 9418.92030050971, -3179.2670136284933),
                SVector(0.0, 0.0, 0.0),
                SVector(0.0, 0.0, 0.0),
            ] * FU, AT)
            @test maximum(norm, fs .- fs_true) < (1e-10 * FU)
            @test norm(sum(fs)) < (1e-10 * FU)

            accels = Molly.calc_accels.(fs, atom_masses, vs_flags)
            accels_true = to_device([
                SVector(2.9259339835646528e-8, 3.1125193300777893e-8, -60.39218830270745),
                SVector(3.599618866597804e-8, 3.822959459233062e-8, 60.39218836444344),
                SVector(0.0, 0.0, 0.0),
                SVector(2.5808104725193486e-10, 0.3123019156881265, 0.1559911900509554),
                SVector(2.4391188018790913e-8, 0.04272879037250003, -0.19871993119974724),
                SVector(2.855779272863662e-8, -0.35503065786187277, 0.042728796565238335),
                SVector(0.0, 0.0, 0.0),
                SVector(-3.7716245060437356e-8, -623.9653288137466, 186.67884395038172),
                SVector(-3.623822522058617e-8, -317.9267013547786, 131.24785729531558),
                SVector(-4.450812411960214e-8, 941.892030050971, -317.9267013628493),
                SVector(0.0, 0.0, 0.0),
                SVector(0.0, 0.0, 0.0),
            ] * AU, AT)
            @test maximum(norm, accels .- accels_true) < (1e-10 * AU)

            random_velocities!(sys, 300.0 * TU)
            @test all(map((v, f) -> (f ? iszero(v) : !iszero(v)), sys.velocities, vs_flags))
        end
    end
end
