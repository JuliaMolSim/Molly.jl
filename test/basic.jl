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

    b = TriclinicBoundary(SVector(2.2, 2.0, 1.8)u"nm", deg2rad.(SVector(50.0, 40.0, 60.0)))
    @test isapprox(b.basis_vectors[1], SVector(2.2      , 0.0      , 0.0      )u"nm", atol=1e-6u"nm")
    @test isapprox(b.basis_vectors[2], SVector(1.0      , 1.7320508, 0.0      )u"nm", atol=1e-6u"nm")
    @test isapprox(b.basis_vectors[3], SVector(1.37888  , 0.5399122, 1.0233204)u"nm", atol=1e-6u"nm")

    @test isapprox(box_volume(b), 3.89937463181886u"nm^3")

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
    coords = place_atoms(n_atoms, b, 0.01u"nm")
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
end

@testset "Neighbor lists" begin
    for neighbor_finder in (DistanceNeighborFinder, TreeNeighborFinder, CellListMapNeighborFinder)
        nf = neighbor_finder(nb_matrix=trues(3, 3), n_steps=10, dist_cutoff=2.0u"nm")
        s = System(
            atoms=[Atom(), Atom(), Atom()],
            coords=[SVector(1.0, 1.0, 1.0)u"nm", SVector(2.0, 2.0, 2.0)u"nm",
                    SVector(5.0, 5.0, 5.0)u"nm"],
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
    coords = [SVector(1.0, 1.0, 1.0)u"nm", SVector(2.0, 2.0, 2.0)u"nm", SVector(5.0, 5.0, 5.0)u"nm"]
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
