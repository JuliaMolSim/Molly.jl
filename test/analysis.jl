@testset "Analysis" begin
    @testset "Displacements and distances" begin
        coords = [SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0)]
        boundary = CubicBoundary(10.0)
        disps = displacements(coords, boundary)
        @test disps[1, 2] == SVector(1.0, 1.0, 1.0)
        @test disps[2, 1] == SVector(-1.0, -1.0, -1.0)
        dists = distances(coords, boundary)
        @test dists[1, 2] ≈ sqrt(3.0) atol=0.05
        @test dists[2, 1] ≈ sqrt(3.0) atol=0.05

        boundary_triclinic = TriclinicBoundary(SVector(10.0, 0.0, 0.0), SVector(5.0, 10.0, 0.0), SVector(5.0, 5.0, 10.0))
        disps_triclinic = displacements(coords, boundary_triclinic)
        @test disps_triclinic[1, 2] == SVector(1.0, 1.0, 1.0)
        dists_triclinic = distances(coords, boundary_triclinic)
        @test dists_triclinic[1, 2] ≈ sqrt(3.0) atol=0.05
    end

    @testset "RMSD" begin
        coords1 = [SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0)]
        coords2 = [SVector(2.0, 1.0, 1.0), SVector(3.0, 2.0, 2.0)]
        @test rmsd(coords1, coords2) ≈ 0.0 atol=0.05

        # Test with non-zero center
        coords3 = [SVector(10.0, 10.0, 10.0), SVector(11.0, 11.0, 11.0)]
        coords4 = [SVector(11.0, 10.0, 10.0), SVector(12.0, 11.0, 11.0)]
        @test rmsd(coords3, coords4) ≈ 0.0 atol=0.05

        # Test with rotation
        coords5 = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
        rot = @SMatrix [ 0.0 -1.0  0.0 ;
                         1.0  0.0  0.0 ;
                         0.0  0.0  1.0 ]
        coords6 = [rot * c for c in coords5]
        @test rmsd(coords5, coords6) ≈ 0.0 atol=0.05
    end

    @testset "Radius of gyration" begin
        atoms = [Atom(mass=1.0), Atom(mass=1.0)]
        coords = [SVector(1.0, 1.0, 1.0), SVector(3.0, 3.0, 3.0)]
        @test radius_gyration(coords, atoms) ≈ sqrt(3.0) atol=0.05

        atoms2 = [Atom(mass=1.0), Atom(mass=2.0), Atom(mass=3.0)]
        coords2 = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]
        center_of_mass = (1.0 * coords2[1] + 2.0 * coords2[2] + 3.0 * coords2[3]) / 6.0
        I = 1.0 * sum(abs2, coords2[1] - center_of_mass) + 2.0 * sum(abs2, coords2[2] - center_of_mass) + 3.0 * sum(abs2, coords2[3] - center_of_mass)
        rg_sq = I / 6.0
        @test radius_gyration(coords2, atoms2) ≈ sqrt(rg_sq) atol=0.05
    end

    @testset "Hydrodynamic radius" begin
        coords = [SVector(0.0, 0.0, 0.0), SVector(3.0, 4.0, 0.0)]
        boundary = CubicBoundary(10.0)
        @test hydrodynamic_radius(coords, boundary) ≈ 20.0 atol=0.05

        coords2 = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]
        d12 = 1.0
        d13 = 1.0
        d23 = sqrt(2.0)
        inv_rh_formula = (1/d12 + 1/d13 + 1/d23) / 9
        @test hydrodynamic_radius(coords2, boundary) ≈ 1 / inv_rh_formula atol=0.05
    end
end
