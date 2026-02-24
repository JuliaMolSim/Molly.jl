@testset "Analysis" begin
    # Displacements and distances
    coords = [SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0)]
    boundary = CubicBoundary(10.0)
    disps = displacements(coords, boundary)
    @test disps[1, 2] == SVector(1.0, 1.0, 1.0)
    @test disps[2, 1] == SVector(-1.0, -1.0, -1.0)
    dists = distances(coords, boundary)
    @test dists[1, 2] ≈ sqrt(3.0)
    @test dists[2, 1] ≈ sqrt(3.0)

    boundary_triclinic = TriclinicBoundary(
        SVector(10.0, 0.0, 0.0),
        SVector(5.0, 10.0, 0.0),
        SVector(5.0, 5.0, 10.0),
    )
    disps_triclinic = displacements(coords, boundary_triclinic)
    @test disps_triclinic[1, 2] == SVector(1.0, 1.0, 1.0)
    dists_triclinic = distances(coords, boundary_triclinic)
    @test dists_triclinic[1, 2] ≈ sqrt(3.0)

    # RMSD
    coords1 = [SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0)]
    coords2 = [SVector(2.0, 1.0, 1.0), SVector(3.0, 2.0, 2.0)]
    @test rmsd(coords1, coords2) ≈ 0.0 atol=1e-12

    coords3 = [SVector(10.0, 10.0, 10.0), SVector(11.0, 11.0, 11.0)]
    coords4 = [SVector(11.0, 10.0, 10.0), SVector(12.0, 11.0, 11.0)]
    @test rmsd(coords3, coords4) ≈ 0.0 atol=1e-12

    coords5 = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
    rot = @SMatrix [0.0 -1.0  0.0;
                    1.0  0.0  0.0;
                    0.0  0.0  1.0]
    coords6 = [rot * c for c in coords5]
    @test rmsd(coords5, coords6) ≈ 0.0 atol=1e-12

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

    # Radius of gyration and hydrodynamic radius
    atoms = [Atom(mass=1.0), Atom(mass=1.0)]
    coords = [SVector(1.0, 1.0, 1.0), SVector(3.0, 3.0, 3.0)]
    @test radius_gyration(coords, atoms) ≈ sqrt(3.0)

    atoms2 = [Atom(mass=1.0), Atom(mass=2.0), Atom(mass=3.0)]
    coords2 = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]
    center_of_mass = (1.0 * coords2[1] + 2.0 * coords2[2] + 3.0 * coords2[3]) / 6.0
    I = 1.0 * sum(abs2, coords2[1] - center_of_mass) +
        2.0 * sum(abs2, coords2[2] - center_of_mass) +
        3.0 * sum(abs2, coords2[3] - center_of_mass)
    rg_sq = I / 6.0
    @test radius_gyration(coords2, atoms2) ≈ sqrt(rg_sq) atol=0.05

    coords = [SVector(0.0, 0.0, 0.0), SVector(3.0, 4.0, 0.0)]
    boundary = CubicBoundary(10.0)
    @test hydrodynamic_radius(coords, boundary) ≈ 20.0 atol=1e-12

    coords2 = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]
    d12 = 1.0
    d13 = 1.0
    d23 = sqrt(2.0)
    inv_rh_formula = (1/d12 + 1/d13 + 1/d23) / 9
    @test hydrodynamic_radius(coords2, boundary) ≈ inv(inv_rh_formula) atol=1e-12

    bb_atoms = BioStructures.collectatoms(struc[1], BioStructures.backboneselector)
    coords = SVector{3, Float64}.(eachcol(BioStructures.coordarray(bb_atoms))) / 10 * u"nm"
    bb_to_mass = Dict("C" => 12.011u"g/mol", "N" => 14.007u"g/mol", "O" => 15.999u"g/mol")
    atoms = [Atom(mass=bb_to_mass[BioStructures.element(bb_atoms[i])]) for i in eachindex(bb_atoms)]
    @test isapprox(radius_gyration(coords, atoms), 11.51225678195222u"Å"; atol=1e-6u"nm")
    boundary = CubicBoundary(10.0u"nm")
    coords_wrap = wrap_coords.(coords, (boundary,))
    @test isapprox(hydrodynamic_radius(coords_wrap, boundary), 21.00006825680275u"Å"; atol=1e-6u"nm")
end
