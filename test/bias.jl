struct BiasNaNGradient end

Molly.bias_gradient(::BiasNaNGradient, cv_sim) = NaN * u"kJ * mol^-1 * nm^-1"

@testset "Collective variables" begin
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    c3 = SVector(0.1, 1.0, 1.0)u"nm"
    c4 = SVector(1.8, 1.0, 1.0)u"nm"
    c5 = SVector(1.0, 1.2, 1.3)u"nm"
    c6 = SVector(0.8, 0.7, 0.9)u"nm"

    a1 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a2 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a3 = Atom(mass=20u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a4 = Atom(mass=5u"g/mol" , charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a5 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a6 = Atom(mass=15u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")

    coords = [c1, c2, c3, c4, c5, c6]
    atoms = [a1, a2, a3, a4, a5, a6]
    boundary = CubicBoundary(2.0u"nm")

    atom_inds_1 = [1, 2, 3]
    atom_inds_2 = [4, 5, 6] 
    ArrayTypes = CUDA.functional() ? [Array, CuArray] : [Array]

    for AT in ArrayTypes
        coords_1 = AT(coords[atom_inds_1])
        coords_2 = AT(coords[atom_inds_2])
        atoms_1 = AT(atoms[atom_inds_1])
        atoms_2 = AT(atoms[atom_inds_2])

        @test isapprox(
            Molly.center_of_mass(coords_1, atoms_1),
            SVector(0.625, 1.0, 1.0)u"nm";
            atol=1e-9u"nm",
        )
        @test isapprox(
            Molly.center_of_mass(coords_2, atoms_2),
            SVector(1.0333333333333334, 0.9166666666666666, 1.05)u"nm";
            atol=1e-9u"nm",
        )

        calc_dist = CalcCMDist()
        dist_cv = CalcDist(atom_inds_1, atom_inds_2, calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.4197386753154344u"nm";
            atol=1e-9u"nm",
        )
        Molly.cv_gradient(dist_cv, coords, atoms, boundary)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            Molly.dist_between_groups(calc_dist, coords_1, coords_2, boundary, atoms_1, atoms_2);
            atol=1e-9u"nm",
        )

        calc_dist = CalcMinDist()
        dist_cv = CalcDist(atom_inds_1, atom_inds_2, calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.3u"nm";
            atol=1e-9u"nm",
        )
        Molly.cv_gradient(dist_cv, coords, atoms, boundary)

        calc_dist = CalcMinDist(:raw)
        dist_cv = CalcDist(atom_inds_1, atom_inds_2, calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.36055512754639896u"nm";
            atol=1e-9u"nm",
        )

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            Molly.dist_between_groups(calc_dist, coords_1, coords_2, boundary);
            atol=1e-9u"nm",
        )

        calc_dist = CalcMaxDist()
        dist_cv = CalcDist(atom_inds_1, atom_inds_2, calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.9695359714832659u"nm";
            atol=1e-9u"nm",
        )
        Molly.cv_gradient(dist_cv, coords, atoms, boundary)

        calc_dist = CalcMaxDist(:raw)
        dist_cv = CalcDist(atom_inds_1, atom_inds_2, calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            1.7u"nm";
            atol=1e-9u"nm",
        )

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            Molly.dist_between_groups(calc_dist, coords_1, coords_2, boundary);
            atol=1e-9u"nm",
        )

        # Differently-sized groups: regression test for a transpose bug in the :raw branch that
        # only manifests when the two groups have different sizes (undetectable for equal sizes)
        atom_inds_small = [1, 2]
        coords_small = AT(coords[atom_inds_small])

        calc_dist = CalcMinDist()
        dist_cv = CalcDist(atom_inds_small, atom_inds_2, calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.36055512754639896u"nm";
            atol=1e-9u"nm",
        )
        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            Molly.dist_between_groups(calc_dist, coords_small, coords_2, boundary);
            atol=1e-9u"nm",
        )
        Molly.cv_gradient(dist_cv, coords, atoms, boundary)

        calc_dist = CalcMaxDist()
        dist_cv = CalcDist(atom_inds_small, atom_inds_2, calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.8u"nm";
            atol=1e-9u"nm",
        )
        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            Molly.dist_between_groups(calc_dist, coords_small, coords_2, boundary);
            atol=1e-9u"nm",
        )
        Molly.cv_gradient(dist_cv, coords, atoms, boundary)

        calc_dist = CalcSingleDist()
        dist_cv = CalcDist([3], [4], calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.3u"nm";
            atol=1e-9u"nm",
        )
        Molly.cv_gradient(dist_cv, coords, atoms, boundary)

        calc_dist = CalcSingleDist(:raw)
        dist_cv = CalcDist([3], [4], calc_dist, :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            1.7u"nm";
            atol=1e-9u"nm",
        )

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            Molly.dist_between_groups(calc_dist, [c3], [c4], boundary);
            atol=1e-9u"nm",
        )

        dist_cv = CalcDist([1], [2], CalcSingleDist(), :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.3u"nm";
            atol=1e-9u"nm",
        )

        dist_cv = CalcDist([3], [4], CalcSingleDist(), :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.3u"nm";
            atol=1e-9u"nm",
        )

        dist_cv = CalcDist([5], [6], CalcSingleDist(), :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.6708203932499369u"nm";
            atol=1e-9u"nm",
        )

        dist_cv = CalcDist([1], [2], CalcSingleDist(:raw), :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.3u"nm";
            atol=1e-9u"nm",
        )

        dist_cv = CalcDist([3], [4], CalcSingleDist(:raw), :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            1.7u"nm";
            atol=1e-9u"nm",
        )

        dist_cv = CalcDist([5], [6], CalcSingleDist(:raw), :wrap)

        @test isapprox(
            calculate_cv(dist_cv, coords, atoms, boundary),
            0.6708203932499369u"nm";
            atol=1e-9u"nm",
        )
    end

    pdb_path = joinpath(data_dir, "1ssu.pdb")
    struc = read(pdb_path, BioStructures.PDBFormat)
    cm_1 = BioStructures.coordarray(struc[1], BioStructures.calphaselector)
    cm_2 = BioStructures.coordarray(struc[2], BioStructures.calphaselector)
    coords_1 = SVector{3, Float64}.(eachcol(cm_1)) / 10 * u"nm"
    coords_2 = SVector{3, Float64}.(eachcol(cm_2)) / 10 * u"nm"

    # RMSD of all atoms
    rmsd_cv = CalcRMSD(coords_2)
    @test calculate_cv(rmsd_cv, coords_1) ≈ 2.54859467758795u"Å"
    @test Molly.cv_gradient(rmsd_cv, coords_1)[2] ≈ 2.54859467758795u"Å"

    # RMSD of a subset of atoms
    n_atoms_subset = 20
    subset_inds = collect(1:n_atoms_subset)
    coords_1_subset = coords_1[1:n_atoms_subset]
    coords_2_subset = coords_2[1:n_atoms_subset]
    rmsd_cv = CalcRMSD(coords_2, subset_inds, subset_inds)
    @test isapprox(
        calculate_cv(rmsd_cv, coords_1),
        rmsd(coords_1_subset, coords_2_subset);
        atol=1e-9u"nm",
    )
    @test Molly.cv_gradient(rmsd_cv, coords_1)[2] ≈ calculate_cv(rmsd_cv, coords_1)

    # A reused grad buffer must be zeroed when rmsd_val == 0, not left stale.
    ref_coords_1atom = [SVector(1.0, 2.0, 3.0)u"nm"] # 1 atom: no Kabsch SVD rotational noise
    rmsd_cv_1atom = CalcRMSD(ref_coords_1atom)
    grad_sentinel = [SVector(99.0, 99.0, 99.0)]
    d_buf_1atom = similar(ref_coords_1atom, eltype(eltype(ref_coords_1atom)), 1)
    Molly.cv_gradient!(grad_sentinel, d_buf_1atom, rmsd_cv_1atom, ref_coords_1atom)
    @test only(d_buf_1atom) == 0.0u"nm"
    @test all(iszero, grad_sentinel[1])

    bb_atoms = BioStructures.collectatoms(struc[1], BioStructures.backboneselector)
    coords = SVector{3, Float64}.(eachcol(BioStructures.coordarray(bb_atoms))) / 10 * u"nm"
    bb_to_mass = Dict("C" => 12.011u"g/mol", "N" => 14.007u"g/mol", "O" => 15.999u"g/mol")
    atoms = [Atom(mass=bb_to_mass[BioStructures.element(bb_atoms[i])]) for i in eachindex(bb_atoms)]

    boundary_rg = CubicBoundary(20.0u"nm")

    # Rg of all atoms
    rg_cv = CalcRg()
    @test isapprox(
        calculate_cv(rg_cv, coords, atoms, boundary_rg),
        11.51225678195222u"Å";
        atol=1e-6u"nm",
    )
    @test isapprox(Molly.cv_gradient(rg_cv, coords, atoms, boundary_rg)[2],
                   calculate_cv(rg_cv, coords, atoms, boundary_rg),
                   atol = 1e-5u"nm")

    # Disparate masses, unlike the backbone's similar C/N/O masses above, separate the two
    # definitions enough to catch a value/gradient COM mismatch.
    atoms_disparate = [Atom(mass=10.0u"g/mol"), Atom(mass=20.0u"g/mol"),
                       Atom(mass=15.0u"g/mol"), Atom(mass=30.0u"g/mol")]
    coords_disparate = [SVector(0.0, 0.0, 0.0)u"nm", SVector(1.0, 0.0, 0.0)u"nm",
                        SVector(0.0, 2.0, 0.0)u"nm", SVector(3.0, 1.0, 0.0)u"nm"]
    rg_cv_disparate = CalcRg()
    @test isapprox(
        calculate_cv(rg_cv_disparate, coords_disparate, atoms_disparate, boundary_rg),
        Molly.cv_gradient(rg_cv_disparate, coords_disparate, atoms_disparate, boundary_rg)[2];
        atol=1e-9u"nm",
    )

    # Rg of a subset of atoms
    n_atoms_subset = 20
    coords_subset = coords[1:n_atoms_subset]
    atoms_subset = atoms[1:n_atoms_subset]
    rg_cv = CalcRg([i for i=1:n_atoms_subset])
    @test isapprox(
        calculate_cv(rg_cv, coords, atoms, boundary_rg),
        radius_gyration(coords_subset,atoms_subset);
        atol=1e-6u"nm",
    )
    @test isapprox(Molly.cv_gradient(rg_cv, coords, atoms, boundary_rg)[2],
                   calculate_cv(rg_cv, coords, atoms, boundary_rg),
                   atol = 1e-5u"nm")

    # Test CalcTorsion value calculation
    # Define four atoms forming a 90-degree (pi/2) dihedral angle
    c_t1 = SVector(0.0, 0.0, 0.0)u"nm"
    c_t2 = SVector(0.1, 0.0, 0.0)u"nm"
    c_t3 = SVector(0.1, 0.1, 0.0)u"nm"
    c_t4 = SVector(0.1, 0.1, 0.1)u"nm"
    
    coords_tor = [c_t1, c_t2, c_t3, c_t4]
    # Atoms and boundary are already defined in the existing testset context
    tor_cv = CalcTorsion([1, 2, 3, 4])
    @test tor_cv.gradient_singularity_tol == 1e-6
    
    @test isapprox(
        calculate_cv(tor_cv, coords_tor, atoms, boundary),
        1.5707963267948966; # pi/2 radians
        atol=1e-9
    )

    for AT in ArrayTypes
        coords_tor_dev = AT(coords_tor)
        @test isapprox(
            calculate_cv(tor_cv, coords_tor_dev, atoms, boundary),
            1.5707963267948966; # pi/2 radians
            atol=1e-9
        )
        grad_dev, phi_dev = Molly.cv_gradient(tor_cv, coords_tor_dev, atoms, boundary)
        @test isapprox(phi_dev, 1.5707963267948966; atol=1e-9)
        @test all(v -> all(x -> isfinite(ustrip(x)), v), grad_dev)
    end

    coords_tor_near = SVector{3, Float32}[
        SVector(0.0f0, 0.0f0, 0.0f0),
        SVector(1.0f0, 0.0f0, 0.0f0),
        SVector(2.0f0, 1.0f-7, 0.0f0),
        SVector(3.0f0, 1.0f0, 0.0f0),
    ]
    grad_near, phi_near = Molly.cv_gradient(
        tor_cv,
        coords_tor_near,
        atoms,
        CubicBoundary(100.0f0),
    )
    @test isfinite(phi_near)
    @test all(v -> all(isfinite, v), grad_near)
    @test maximum(norm, grad_near) < 2.0f6

    coords_tor_near_units = [
        SVector(0.0, 0.0, 0.0)u"nm",
        SVector(1.0, 0.0, 0.0)u"nm",
        SVector(2.0, 1.0e-7, 0.0)u"nm",
        SVector(3.0, 1.0, 0.0)u"nm",
    ]
    grad_near_units, phi_near_units = Molly.cv_gradient(
        tor_cv,
        coords_tor_near_units,
        atoms,
        CubicBoundary(100.0u"nm"),
    )
    @test isfinite(phi_near_units)
    @test all(v -> all(x -> isfinite(ustrip(x)), v), grad_near_units)

    coords_tor_zero_bond = SVector{3, Float64}[
        SVector(0.0, 0.0, 0.0),
        SVector(0.0, 0.0, 0.0),
        SVector(1.0, 0.0, 0.0),
        SVector(2.0, 0.0, 0.0),
    ]
    @test_throws ArgumentError Molly.cv_gradient(
        tor_cv,
        coords_tor_zero_bond,
        atoms,
        CubicBoundary(100.0),
    )

    # CalcSingleDist needs exactly one atom per group, checked at construction on every backend
    @test_throws ArgumentError CalcDist([1, 2], [3], CalcSingleDist())
    @test_throws ArgumentError CalcDist([1], [2, 3], CalcSingleDist())
    @test_throws ArgumentError CalcDist(Int[], [3], CalcSingleDist())
    # Test CalcAngle value calculation
    # Define three atoms forming a 90-degree (pi/2) angle at the middle atom
    coords_ang = [
        SVector(0.1, 0.0, 0.0)u"nm",
        SVector(0.0, 0.0, 0.0)u"nm",
        SVector(0.0, 0.1, 0.0)u"nm",
    ]
    ang_cv = CalcAngle([1, 2, 3])
    @test isapprox(
        calculate_cv(ang_cv, coords_ang, atoms, boundary),
        1.5707963267948966; # pi/2 radians
        atol=1e-9
    )

    # The analytical gradient matches finite differences of the angle
    coords_ang_gen = [
        SVector(1.0, 1.0, 1.0)u"nm",
        SVector(1.1, 1.05, 0.95)u"nm",
        SVector(1.05, 1.2, 1.1)u"nm",
    ]
    grad_ang, θ_ang = Molly.cv_gradient(ang_cv, coords_ang_gen, atoms, boundary)
    isapprox(θ_ang, calculate_cv(ang_cv, coords_ang_gen, atoms, boundary); atol=1e-12)
    h = 1e-6u"nm"
    shift = SVector(h, zero(h), zero(h))
    c_plus, c_minus = copy(coords_ang_gen), copy(coords_ang_gen)
    c_plus[1] += shift
    c_minus[1] -= shift

    grad_fd = (calculate_cv(ang_cv, c_plus, atoms, boundary) -
                    calculate_cv(ang_cv, c_minus, atoms, boundary)) / (2*h)
    @test isapprox(grad_ang[1][1],grad_fd)

    # Collinear atoms, where the gradient is singular, give an angle of pi and zero gradients
    coords_ang_line = [
        SVector(0.0, 0.0, 0.0)u"nm",
        SVector(0.1, 0.0, 0.0)u"nm",
        SVector(0.2, 0.0, 0.0)u"nm",
    ]
    grad_line, θ_line = Molly.cv_gradient(ang_cv, coords_ang_line, atoms, boundary)
    @test isapprox(θ_line, π; atol=1e-6)
    @test all(v -> all(iszero, v), grad_line)
end

@testset "Bias potentials" begin
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    c3 = SVector(1.4, 1.0, 1.0)u"nm"
    c4 = SVector(1.1, 1.0, 1.0)u"nm"

    a1 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a2 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a3 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a4 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")

    boundary = CubicBoundary(2.0u"nm")

    dr12 = vector(c1, c2, boundary)
    dr13 = vector(c1, c3, boundary)
    dr14 = vector(c1, c4, boundary)

    atoms = [a1, a2, a3, a4]
    coords = [c1, c2, c3, c4]
    velocities = [random_velocity(10u"g/mol", 300u"K") for i in 1:length(atoms)]

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
    )

    lb = LinearBias(1500u"kJ * mol^-1 * nm^-1", 0.5u"nm")

    cv_sim = 1u"nm"
    @test isapprox(
        potential_energy(lb, cv_sim),
        750u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    cv_sim = 0.5u"nm"
    @test isapprox(
        potential_energy(lb, cv_sim),
        0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    cv_sim = 1u"nm"
    @test isapprox(
        Molly.bias_gradient(lb, cv_sim),
        1500u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 0.1u"nm"
    @test isapprox(
        Molly.bias_gradient(lb, cv_sim),
        -1500u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 0.5u"nm"
    @test Molly.bias_gradient(lb, cv_sim) == 0u"kJ * mol^-1 * nm^-1"

    sb = SquareBias(3000u"kJ * mol^-1 * nm^-2", 0.75u"nm")

    cv_sim = 1u"nm"
    @test isapprox(
        potential_energy(sb, cv_sim),
        93.75u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    cv_sim = 0.75u"nm"
    @test isapprox(
        potential_energy(sb, cv_sim),
        0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    cv_sim = 1u"nm"
    @test isapprox(
        Molly.bias_gradient(sb, cv_sim),
        750u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 0.1u"nm"
    @test isapprox(
        Molly.bias_gradient(sb, cv_sim),
        -1950u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 0.75u"nm"
    @test isapprox(
        Molly.bias_gradient(sb, cv_sim),
        0u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    fb = FlatBottomSquareBias(3000u"kJ * mol^-1 * nm^-2", 0.5u"nm", 0.75u"nm")
    @test_throws ArgumentError FlatBottomSquareBias(
        3000u"kJ * mol^-1 * nm^-2",
        -0.5u"nm",
        0.75u"nm",
    )
    @test_throws ArgumentError FlatBottomSquareBias(
        3000u"kJ * mol^-1 * nm^-2",
        NaN * u"nm",
        0.75u"nm",
    )

    cv_sim = 1.5u"nm"
    @test isapprox(
        potential_energy(fb, cv_sim),
        93.75u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    cv_sim = 1u"nm"
    @test isapprox(
        potential_energy(fb, cv_sim),
        0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    cv_sim = 1.5u"nm"
    @test isapprox(
        Molly.bias_gradient(fb, cv_sim),
        750u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 1u"nm"
    @test isapprox(
        Molly.bias_gradient(fb, cv_sim),
        0u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 0.75u"nm"
    @test Molly.bias_gradient(fb, cv_sim) == 0u"kJ * mol^-1 * nm^-1"

    calc_dist = CalcDist([1], [2], CalcSingleDist(), :wrap)

    lb = LinearBias(7500u"kJ * mol^-1 * nm^-1", 0.5u"nm")
    @test isapprox(
        AtomsCalculators.potential_energy(sys, BiasPotential(calc_dist, lb)),
        1500u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    sb = SquareBias(7500u"kJ * mol^-1 * nm^-2", 0.5u"nm")
    @test isapprox(
        AtomsCalculators.potential_energy(sys, BiasPotential(calc_dist, sb)),
        150u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    fb = FlatBottomSquareBias(7500u"kJ * mol^-1 * nm^-2", 0.15u"nm", 0.5u"nm")
    @test isapprox(
        AtomsCalculators.potential_energy(sys, BiasPotential(calc_dist, fb)),
        9.375u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    calc_dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    lb = LinearBias(7500u"kJ * mol^-1 * nm^-1", 0.5u"nm")

    fs = Molly.zero_forces(sys)
    AtomsCalculators.forces!(fs, sys, BiasPotential(calc_dist, lb))
    @test isapprox(
        fs[1],
        SVector(-7500, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    @test isapprox(
        fs[2],
        SVector(7500, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    @test isapprox(
        fs[3],
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    fs_bad = Molly.zero_forces(sys)
    @test_throws ErrorException AtomsCalculators.forces!(
        fs_bad,
        sys,
        BiasPotential(calc_dist, BiasNaNGradient()),
    )

    # check_bias_finite's max_abs_component_fn is only called (and only appears in the error
    # message) when the value being checked is actually non-finite.
    bias_check = BiasPotential(calc_dist, SquareBias(300.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))
    fs_svec_inf = [SVector(Inf, 0.0, 0.0)u"kJ * mol^-1 * nm^-1"]
    err = try
        Molly.check_bias_finite(fs_svec_inf, "bias force", bias_check; cv_sim=1.0u"nm",
                                max_abs_component_fn=() -> Molly.bias_max_abs_ustrip(fs_svec_inf))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("max_abs_component=Inf", err.msg)

    fs_svec_finite = [SVector(2.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1"]
    fn_called = Ref(false)
    Molly.check_bias_finite(fs_svec_finite, "bias force", bias_check; cv_sim=1.0u"nm",
                            max_abs_component_fn=() -> (fn_called[] = true;
                                                        Molly.bias_max_abs_ustrip(fs_svec_finite)))
    @test !fn_called[]

    # PeriodicFlatBottomBias tests (Target: 0, Flat bottom width: 0.1)
    pb = PeriodicFlatBottomBias(1000.0u"kJ * mol^-1", 0.1, 0.0)
    @test pb.r_fb == 0.1
    @test_throws ArgumentError PeriodicFlatBottomBias(1000.0u"kJ * mol^-1", -0.1, 0.0)
    @test_throws ArgumentError PeriodicFlatBottomBias(1000.0u"kJ * mol^-1", NaN, 0.0)

    # Inside flat region (no penalty)
    cv_sim_in = 0.05
    @test potential_energy(pb, cv_sim_in) == 0.0u"kJ * mol^-1"
    @test Molly.bias_gradient(pb, cv_sim_in) == 0.0u"kJ * mol^-1"

    # Outside region (harmonic penalty)
    cv_sim_out = 0.2
    # Energy: 0.5 * k * (dist - r_fb)^2 = 0.5 * 1000 * (0.2 - 0.1)^2 = 5.0
    @test isapprox(
        potential_energy(pb, cv_sim_out),
        5.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1"
    )
    # Gradient: k * (dist - r_fb) * sign(d_wrapped) = 1000 * 0.1 * 1 = 100.0
    @test isapprox(
        Molly.bias_gradient(pb, cv_sim_out),
        100.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1"
    )

    # Periodic wrapping test (Target 0, width 0.1, Input ~ -0.2)
    cv_sim_wrap = 2π - 0.2
    # Wrapped distance is 0.2, outside the flat bottom
    @test isapprox(
        potential_energy(pb, cv_sim_wrap),
        5.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1"
    )
    # Gradient should point towards the target (negative direction)
    @test isapprox(
        Molly.bias_gradient(pb, cv_sim_wrap),
        -100.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1"
    )

    @test isapprox(
        potential_energy(pb, π),
        potential_energy(pb, -π);
        atol=1e-9u"kJ * mol^-1",
    )
    @test Molly.bias_gradient(pb, π) == Molly.bias_gradient(pb, -π)
    @test Molly.bias_gradient(pb, π) < 0u"kJ * mol^-1"
end

@testset "Bias correction and GPU" begin
    coords_uw = [SVector(1.95u"nm", 0.0u"nm", 0.0u"nm"), SVector(0.05u"nm", 0.0u"nm", 0.0u"nm"), SVector(0.15u"nm", 0.0u"nm", 0.0u"nm"),
                 SVector(1.0u"nm", 1.0u"nm", 1.0u"nm")]
    boundary_uw = CubicBoundary(2.0u"nm")
    topology = MolecularTopology([1, 1, 1, 2], [3, 1], [(1, 2), (2, 3)])
    atoms_uw = [Atom(mass=10.0u"g/mol") for _ in 1:4]

    rg_cv_pbc  = CalcRg([1, 2, 3])
    rg_cv_wrap = CalcRg([1, 2, 3], :wrap)

    # :pbc (topology-unwrapped) and :wrap (raw, PBC-aware COM) agree for a molecule straddling
    # the boundary when every atom is within one periodic image of the others.
    sys_cpu = System(atoms=atoms_uw, coords=coords_uw, boundary=boundary_uw, topology=topology)
    rg_pbc_cpu = calculate_cv(rg_cv_pbc, Molly.bias_coords(sys_cpu, rg_cv_pbc), sys_cpu.atoms, boundary_uw)
    rg_wrap_cpu = calculate_cv(rg_cv_wrap, Molly.bias_coords(sys_cpu, rg_cv_wrap), sys_cpu.atoms, boundary_uw)
    @test isapprox(rg_pbc_cpu, rg_wrap_cpu; atol=1e-9u"nm")

    if CUDA.functional()
        sys_gpu = System(
            atoms=CuArray(atoms_uw),
            coords=CuArray(coords_uw),
            boundary=boundary_uw,
            topology=topology,
        )
        # :pbc now runs fully on GPU (GPU-native unwrap_molecules) and matches the CPU :pbc result
        rg_pbc_gpu = calculate_cv(rg_cv_pbc, Molly.bias_coords(sys_gpu, rg_cv_pbc), sys_gpu.atoms, boundary_uw)
        @test isapprox(rg_pbc_gpu, rg_pbc_cpu; atol=1e-9u"nm")
        # :wrap stays fully GPU-resident and matches the CPU :wrap (raw coordinates) result
        rg_wrap_gpu = calculate_cv(rg_cv_wrap, Molly.bias_coords(sys_gpu, rg_cv_wrap), sys_gpu.atoms, boundary_uw)
        @test isapprox(rg_wrap_gpu, rg_wrap_cpu; atol=1e-9u"nm")
    end
end

@testset "Biased simulation" begin
    function pair_dist_wrapper_12(sys, args...; kwargs...)
        coords_1 = Molly.from_device(sys.coords)[1]
        coords_2 = Molly.from_device(sys.coords)[2]
        distances([coords_1, coords_2], sys.boundary)[2]
    end

    function pair_dist_wrapper_13(sys, args...; kwargs...)
        coords_1 = Molly.from_device(sys.coords)[1]
        coords_2 = Molly.from_device(sys.coords)[3]
        distances([coords_1, coords_2], sys.boundary)[2]
    end

    # No units
    n_steps, burn_in = 50_000, 250

    for AT in array_list
        n_atoms = 10
        boundary = CubicBoundary(10.0)
        temp = 298.0
        atom_mass = 10.0
        rng = Xoshiro(1000)

        atoms = to_device([Atom(mass=atom_mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms], AT)
        coords = to_device(place_atoms(n_atoms, boundary; min_dist=0.3, rng=rng), AT)
        velocities = to_device([random_velocity(atom_mass, temp; rng=rng) for i in 1:n_atoms], AT)
        pairwise_inters = (LennardJones(),)

        define_cv = CalcDist([1], [2], CalcSingleDist(), :wrap)
        define_bias = SquareBias(400, 1.5)
        general_inters = (BiasPotential(define_cv, define_bias),)
        simulator = VelocityVerlet(
            dt=0.002,
            coupling=AndersenThermostat(temp, 1.0),
        )

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            general_inters=general_inters,
            force_units=NoUnits,
            energy_units=NoUnits,
            loggers=(
                pair_dist_12=GeneralObservableLogger(pair_dist_wrapper_12, Float64, 10),
                pair_dist_13=GeneralObservableLogger(pair_dist_wrapper_13, Float64, 10),
                coords=CoordinatesLogger(Float64, 10)
            ),
        )

        simulate!(sys, simulator, n_steps; rng=rng)

        pair_dists_12 = values(sys.loggers.pair_dist_12)
        pair_dists_13 = values(sys.loggers.pair_dist_13)

        dist_12_mean = mean(pair_dists_12[burn_in:end])
        dist_13_mean = mean(pair_dists_13[burn_in:end])
        dist_12_std = std(pair_dists_12[burn_in:100:end])
        dist_13_std = std(pair_dists_13[burn_in:100:end])

        @test isapprox(dist_12_mean, 1.5; atol=0.05)
        @test !isapprox(dist_13_mean, 1.5; atol=0.05)
        @test dist_13_mean > dist_12_mean
        @test dist_13_std > dist_12_std
    end

    # Units
    for AT in array_list
        n_atoms = 5
        boundary = CubicBoundary(10.0u"nm")
        temp = 298.0u"K"
        atom_mass = 10.0u"g/mol"
        rng = Xoshiro(1000)

        atoms = to_device([Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms], AT)
        coords = to_device(place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng=rng), AT)
        velocities = to_device([random_velocity(atom_mass, temp; rng=rng) for i in 1:n_atoms], AT)
        pairwise_inters = (LennardJones(),)

        define_cv = CalcDist([1], [2], CalcSingleDist(), :wrap)
        define_bias = SquareBias(400u"kJ * mol^-1 * nm^-2", 1.5u"nm")
        general_inters = (BiasPotential(define_cv, define_bias),)
        simulator = VelocityVerlet(
            dt=0.002u"ps",
            coupling=AndersenThermostat(temp, 1.0u"ps"),
        )

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            general_inters=general_inters,
            loggers=(
                coords=CoordinatesLogger(10),
                pair_dist_12=GeneralObservableLogger(pair_dist_wrapper_12, Any, 10),
                pair_dist_13=GeneralObservableLogger(pair_dist_wrapper_13, Any, 10),
            ),
        )

        simulate!(sys, simulator, n_steps; rng=rng)

        pair_dists_12 = values(sys.loggers.pair_dist_12)
        pair_dists_13 =values(sys.loggers.pair_dist_13)

        dist_12_mean = mean(pair_dists_12[burn_in:end])
        dist_13_mean = mean(pair_dists_13[burn_in:end])
        dist_12_std = std(pair_dists_12[burn_in:100:end])
        dist_13_std = std(pair_dists_13[burn_in:100:end])

        @test isapprox(dist_12_mean, 1.5u"nm"; atol=0.05u"nm")
        @test !isapprox(dist_13_mean, 1.5u"nm"; atol=0.05u"nm")
        @test dist_13_mean > dist_12_mean
        @test dist_13_std > dist_12_std
    end
end

@testset "BiasPotential persistent buffers" begin
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(20.0u"nm")

    @testset "Cross-BiasPotential isolation" for AT in array_list
        n = 8
        coords = AT([SVector(Float64(i) * 0.3, 0.0, 0.0)u"nm" for i in 1:n])
        atoms = AT([Atom(mass=atom_mass) for _ in 1:n])
        cv1 = CalcDist([1], [2], CalcSingleDist(), :wrap)
        cv2 = CalcDist([3, 4], [6, 7], CalcMinDist(), :wrap)
        bias1 = BiasPotential(cv1, SquareBias(300.0u"kJ * mol^-1 * nm^-2", 0.8u"nm"))
        bias2 = BiasPotential(cv2, SquareBias(250.0u"kJ * mol^-1 * nm^-2", 1.2u"nm"))
        sys = System(atoms=atoms, coords=coords, boundary=boundary, general_inters=(bias1, bias2))
        buffers = Molly.init_buffers!(sys, 1)

        fs1 = Molly.zero_forces(sys)
        Molly.AtomsCalculators.forces!(fs1, sys, bias1; buffers=buffers)
        fs2 = Molly.zero_forces(sys)
        Molly.AtomsCalculators.forces!(fs2, sys, bias2; buffers=buffers)
        @test buffers.bias_scratch[bias1.id].grad !== buffers.bias_scratch[bias2.id].grad
        combined_expected = Molly.from_device(fs1) .+ Molly.from_device(fs2)

        fs_sum = Molly.zero_forces(sys)
        for gi in sys.general_inters
            Molly.AtomsCalculators.forces!(fs_sum, sys, gi; buffers=buffers)
        end
        @test all(isapprox.(Molly.from_device(fs_sum), combined_expected; atol=1e-9u"kJ * mol^-1 * nm^-1"))
    end

    @testset "Scratch isolation between structurally-identical biases" begin
        cv = CalcDist([1], [2], CalcSingleDist(), :wrap)
        bias1 = BiasPotential(cv, SquareBias(400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))
        bias2 = BiasPotential(cv, SquareBias(400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))
        @test !(bias1 === bias2) # the id field makes them distinct, as intended
        @test bias1.id != bias2.id
        # ...but they compare equal, so thermo.jl's intersect can move a shared restraint
        @test bias1 == bias2 && isequal(bias1, bias2) && hash(bias1) == hash(bias2)
        @test length(intersect([bias1], [bias2])) == 1
    end

    # A System rejects biases that would corrupt the scratch buffers or run the unchecked kernels
    # out of range: two biases sharing an id (as a field-wise copy or a bias loaded from disk
    # can give), and atom indices outside the system.
    @testset "Invalid biases are rejected when building a System" begin
        atoms = [Atom(mass=10.0u"g/mol") for _ in 1:4]
        coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(1.5, 0.0, 0.0)u"nm",
                  SVector(5.0, 0.0, 0.0)u"nm", SVector(7.0, 0.0, 0.0)u"nm"]
        bias_type = SquareBias(400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm")
        cv_a = CalcDist([1], [2], CalcSingleDist(), :wrap)
        cv_b = CalcDist([3], [4], CalcSingleDist(), :wrap)
        bias_a = BiasPotential(cv_a, bias_type)
        bias_b = BiasPotential(cv_b, bias_type)
        bias_same_id = BiasPotential{typeof(cv_b), typeof(bias_type)}(cv_b, bias_type, true, bias_a.id)
        build(inters) = System(atoms=atoms, coords=coords, boundary=CubicBoundary(100.0u"nm"),
                               general_inters=inters)
        @test build((bias_a, bias_b)) isa System
        @test_throws ArgumentError build((bias_a, bias_same_id))

        for cv in (CalcDist([1], [5], CalcSingleDist(), :wrap), CalcDist([1, 2], [3, 9], CalcMinDist(), :wrap),
                   CalcDist([1, 2], [3, 6], CalcMaxDist(), :wrap), CalcDist([0, 2], [3, 4], CalcCMDist(), :wrap),
                   CalcRg([1, 2, 7], :wrap), CalcRMSD(coords[1:3], [1, 2, 6], [], :wrap))
            @test_throws ArgumentError build((BiasPotential(cv, bias_type),))
        end
        @test_throws ArgumentError build((BiasPotential(CalcTorsion([1, 2, 3, 5], :wrap),
                                                        SquareBias(400.0u"kJ * mol^-1", 1.0)),))
        @test build((BiasPotential(CalcRg([], :wrap), bias_type),)) isa System # empty means all atoms
    end

    # A reused persistent `grad` buffer must not retain a stale force contribution from a
    # PREVIOUS step's CalcMinDist winning pair once the winner moves to a different pair.
    @testset "Stale-winner regression (CalcMinDist)" for AT in array_list
        atoms = AT([Atom(mass=atom_mass) for _ in 1:4])
        # Step N: atom 1 closest to atom 3; step N+1: atom 2 closest to atom 4, with atoms
        # 1 and 3 now far apart.
        coords_N = [SVector(0.0, 0.0, 0.0)u"nm", SVector(10.0, 0.0, 0.0)u"nm",
                    SVector(0.5, 0.0, 0.0)u"nm", SVector(15.0, 0.0, 0.0)u"nm"]
        coords_N1 = [SVector(0.0, 0.0, 0.0)u"nm", SVector(10.0, 0.0, 0.0)u"nm",
                     SVector(15.0, 0.0, 0.0)u"nm", SVector(10.5, 0.0, 0.0)u"nm"]
        cv = CalcDist([1, 2], [3, 4], CalcMinDist(), :wrap)
        bias = BiasPotential(cv, SquareBias(400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))

        sys_N = System(atoms=atoms, coords=AT(coords_N), boundary=boundary, general_inters=(bias,))
        buffers = Molly.init_buffers!(sys_N, 1)
        fs_N = Molly.zero_forces(sys_N)
        Molly.AtomsCalculators.forces!(fs_N, sys_N, bias; buffers=buffers)
        fs_N_cpu = Molly.from_device(fs_N)
        @test norm(ustrip.(fs_N_cpu[1])) > 0
        @test norm(ustrip.(fs_N_cpu[3])) > 0
        @test norm(ustrip.(fs_N_cpu[2])) == 0
        @test norm(ustrip.(fs_N_cpu[4])) == 0

        sys_N1 = System(atoms=atoms, coords=AT(coords_N1), boundary=boundary, general_inters=(bias,))
        fs_N1 = Molly.zero_forces(sys_N1)
        # Reuses the SAME buffers (hence the SAME buffers.bias_scratch[bias.id].grad) as step N.
        Molly.AtomsCalculators.forces!(fs_N1, sys_N1, bias; buffers=buffers)
        fs_N1_cpu = Molly.from_device(fs_N1)
        @test norm(ustrip.(fs_N1_cpu[1])) == 0 # not a stale leftover from step N
        @test norm(ustrip.(fs_N1_cpu[3])) == 0
        @test norm(ustrip.(fs_N1_cpu[2])) > 0
        @test norm(ustrip.(fs_N1_cpu[4])) > 0
    end

    @testset "Scratch isolation across MTS levels" for AT in array_list
        atoms = AT([Atom(mass=atom_mass) for _ in 1:4])
        boundary_mts = CubicBoundary(50.0u"nm")
        coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(20.0, 0.0, 0.0)u"nm",
                 SVector(0.5, 0.0, 0.0)u"nm", SVector(21.5, 0.0, 0.0)u"nm"]
        bias_a = BiasPotential(CalcDist([1, 2], [3], CalcMinDist(), :wrap),
                               SquareBias(400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))
        bias_b = BiasPotential(CalcDist([1, 2], [4], CalcMinDist(), :wrap),
                               SquareBias(400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))

        sys_a = System(atoms=atoms, coords=AT(coords), boundary=boundary_mts, general_inters=(bias_a,))
        fs_a_ref = Molly.from_device(forces(sys_a))
        sys_b = System(atoms=atoms, coords=AT(coords), boundary=boundary_mts, general_inters=(bias_b,))
        fs_b_ref = Molly.from_device(forces(sys_b))

        sys_mts = System(atoms=atoms, coords=AT(coords), boundary=boundary_mts, general_inters=(bias_a, bias_b))
        sim = MTSIntegrator(dt=1.0u"fs", gi_fractions=(1, 2), remove_CM_motion=false)
        fraction_inters = Molly.mts_interaction_groups(sys_mts, sim)
        buffers = Molly.init_buffers!(sys_mts, 1)

        fs_a_mts = Molly.zero_forces(sys_mts)
        Molly.forces!(fs_a_mts, sys_mts, nothing, 1, buffers, Val(false); n_threads=1,
                      general_inters=fraction_inters[1].general_inters)
        fs_b_mts = Molly.zero_forces(sys_mts)
        Molly.forces!(fs_b_mts, sys_mts, nothing, 1, buffers, Val(false); n_threads=1,
                      general_inters=fraction_inters[2].general_inters)

        @test all(isapprox.(Molly.from_device(fs_a_mts), fs_a_ref; atol=1e-9u"kJ * mol^-1 * nm^-1"))
        @test all(isapprox.(Molly.from_device(fs_b_mts), fs_b_ref; atol=1e-9u"kJ * mol^-1 * nm^-1"))
    end

    # forces! called twice at the SAME step_n with coordinates moved in between (as MTS substeps
    # or a post-coupling recompute do) must not reuse the first call's cached :pbc unwrap.
    if CUDA.functional()
        @testset "Unwrap cache does not persist across forces! calls at the same step_n" begin
            topology = MolecularTopology([1, 1, 1], [3], [(1, 2), (2, 3)])
            atoms = CuArray([Atom(mass=atom_mass) for _ in 1:3])
            coords = CuArray([SVector(1.95, 0.0, 0.0)u"nm", SVector(0.05, 0.0, 0.0)u"nm",
                              SVector(0.15, 0.0, 0.0)u"nm"])
            boundary_uw = CubicBoundary(2.0u"nm")
            cv = CalcRg([1, 2, 3])
            bias = BiasPotential(cv, SquareBias(400.0u"kJ * mol^-1 * nm^-2", 0.5u"nm"))

            sys = System(atoms=atoms, coords=coords, boundary=boundary_uw, topology=topology,
                         general_inters=(bias,))
            buffers = Molly.init_buffers!(sys, 1)

            fs1 = Molly.zero_forces(sys)
            Molly.forces!(fs1, sys, nothing, 1, buffers, Val(false))

            sys.coords .= CuArray([SVector(0.95, 0.0, 0.0)u"nm", SVector(0.05, 0.0, 0.0)u"nm",
                                   SVector(0.15, 0.0, 0.0)u"nm"])
            fs2 = Molly.zero_forces(sys)
            Molly.forces!(fs2, sys, nothing, 1, buffers, Val(false)) # same step_n as fs1

            sys_ref = System(atoms=atoms, coords=copy(sys.coords), boundary=boundary_uw,
                             topology=topology, general_inters=(bias,))
            buffers_ref = Molly.init_buffers!(sys_ref, 1)
            fs_ref = Molly.zero_forces(sys_ref)
            Molly.forces!(fs_ref, sys_ref, nothing, 1, buffers_ref, Val(false))

            @test all(isapprox.(Molly.from_device(fs2), Molly.from_device(fs_ref);
                                atol=1e-9u"kJ * mol^-1 * nm^-1"))
        end
    end

    # CalcCMDist gradient with overlapping atom_inds_1/atom_inds_2: the shared atom's force must
    # be the SUM of both groups' contributions, not whichever group writes/races last.
    @testset "CalcCMDist gradient with overlapping atom groups" for AT in array_list
        masses = [10.0, 20.0, 15.0, 5.0]u"g/mol"
        coords_host = [SVector(0.0, 0.0, 0.0)u"nm", SVector(1.0, 0.0, 0.0)u"nm",
                       SVector(2.0, 0.0, 0.0)u"nm", SVector(5.0, 0.0, 0.0)u"nm"]
        atoms = AT([Atom(mass=m) for m in masses])
        coords = AT(coords_host)
        cv = CalcDist([1, 2, 3], [3, 4], CalcCMDist(), :wrap) # atom 3 is shared
        k, target = 400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"
        bias = BiasPotential(cv, SquareBias(k, target))
        sys = System(atoms=atoms, coords=coords, boundary=boundary, general_inters=(bias,))
        buffers = Molly.init_buffers!(sys, 1)

        fs = Molly.zero_forces(sys)
        Molly.AtomsCalculators.forces!(fs, sys, bias; buffers=buffers)
        fs_cpu = Molly.from_device(fs)

        # Independent reference computed directly from the CMDist/SquareBias formulas, not by
        # calling into any code under test.
        g1, g2 = [1, 2, 3], [3, 4]
        M1, M2 = sum(masses[g1]), sum(masses[g2])
        com1 = sum(coords_host[g1] .* masses[g1]) / M1
        com2 = sum(coords_host[g2] .* masses[g2]) / M2
        d = norm(com2 - com1)
        dir = (com2 - com1) / d
        # Atom 3 is in both groups, so its CV gradient is the SUM of both groups' contributions.
        d_cv_d_atom3 = -dir * (masses[3] / M1) + dir * (masses[3] / M2)
        force_atom3 = -k * (d - target) * d_cv_d_atom3

        @test all(isapprox.(ustrip.(fs_cpu[3]), ustrip.(force_atom3); atol=1e-9))
    end

    # The same group of atoms in two periodic images -- crossing the box, and not -- must give the
    # same forces and virial.
    @testset "CalcRg :wrap virial for a group crossing the boundary" for AT in array_list
        coords_cross = [SVector(1.9, 0.0, 0.0)u"nm", SVector(1.95, 0.0, 0.0)u"nm",
                        SVector(0.05, 0.0, 0.0)u"nm", SVector(0.1, 0.0, 0.0)u"nm"]
        coords_whole = [SVector(-0.1, 0.0, 0.0)u"nm", SVector(-0.05, 0.0, 0.0)u"nm",
                        SVector(0.05, 0.0, 0.0)u"nm", SVector(0.1, 0.0, 0.0)u"nm"]
        bias = BiasPotential(CalcRg([1, 2, 3, 4], :wrap), SquareBias(300.0u"kJ * mol^-1 * nm^-2", 0.5u"nm"))
        function forces_virial(coords)
            sys = System(atoms=AT([Atom(mass=10.0u"g/mol") for _ in 1:4]), coords=AT(coords),
                         boundary=CubicBoundary(2.0u"nm"), general_inters=(bias,))
            buffers = Molly.init_buffers!(sys, 1)
            fs = Molly.zero_forces(sys)
            Molly.forces!(fs, sys, nothing, 1, buffers, Val(true); n_threads=1)
            return ustrip.(Molly.from_device(fs)), ustrip.(Molly.from_device(buffers.virial))
        end
        fs_cross, virial_cross = forces_virial(coords_cross)
        fs_whole, virial_whole = forces_virial(coords_whole)
        @test all(isapprox.(fs_cross, fs_whole; atol=1e-9))
        @test all(isapprox.(virial_cross, virial_whole; atol=1e-9))
        @test !iszero(virial_whole)
    end

    # Shared base system for the testsets below, remade per-testset via System(sys; coords=..,
    # atoms=..) instead of rebuilding from scratch.
    if CUDA.functional()
        atoms_v = [Atom(mass=10.0u"g/mol") for _ in 1:4]
        coords_v = [SVector(0.0, 0.0, 0.0)u"nm", SVector(1.0, 0.0, 0.0)u"nm",
                   SVector(5.0, 0.0, 0.0)u"nm", SVector(6.0, 0.0, 0.0)u"nm"]
        sys_cpu_base = System(atoms=atoms_v, coords=coords_v, boundary=boundary)
        sys_gpu_base = System(atoms=CuArray(atoms_v), coords=CuArray(coords_v), boundary=boundary)

        # Asserts potential_energy (and forces! if check_forces) through `bias` match on
        # sys_cpu/sys_gpu; returns fs_cpu (or nothing if !check_forces) for any extra caller checks.
        function pe_fs_gpu_vs_cpu(sys_cpu, sys_gpu, bias; check_forces=true)
            pe_cpu = Molly.AtomsCalculators.potential_energy(sys_cpu, bias)
            pe_gpu = Molly.AtomsCalculators.potential_energy(sys_gpu, bias)
            @test ustrip(pe_gpu) ≈ ustrip(pe_cpu)
            check_forces || return nothing
            fs_cpu = Molly.zero_forces(sys_cpu)
            Molly.AtomsCalculators.forces!(fs_cpu, sys_cpu, bias)
            fs_gpu = Molly.zero_forces(sys_gpu)
            Molly.AtomsCalculators.forces!(fs_gpu, sys_gpu, bias)
            @test all(isapprox.(ustrip.(Molly.from_device(fs_gpu)), ustrip.(fs_cpu); atol=1e-10))
            return fs_cpu
        end

        # Asserts potential_energy and forces! with virial through `bias` match on sys_cpu/sys_gpu.
        function fs_virial_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
            pe_cpu = Molly.AtomsCalculators.potential_energy(sys_cpu, bias)
            pe_gpu = Molly.AtomsCalculators.potential_energy(sys_gpu, bias)
            @test ustrip(pe_gpu) ≈ ustrip(pe_cpu)

            buffers_cpu = Molly.init_buffers!(sys_cpu, 1)
            fs_cpu = Molly.zero_forces(sys_cpu)
            Molly.forces!(fs_cpu, sys_cpu, nothing, 1, buffers_cpu, Val(true); n_threads=1)
            buffers_gpu = Molly.init_buffers!(sys_gpu, 1)
            fs_gpu = Molly.zero_forces(sys_gpu)
            Molly.forces!(fs_gpu, sys_gpu, nothing, 1, buffers_gpu, Val(true))
            @test all(isapprox.(Molly.from_device(fs_gpu), fs_cpu; atol=1e-9u"kJ * mol^-1 * nm^-1"))
            @test all(isapprox.(ustrip.(Molly.from_device(buffers_gpu.virial)), ustrip.(buffers_cpu.virial);
                                atol=1e-9))
            return nothing
        end

        @testset "CalcRMSD in BiasPotential on GPU matches CPU" begin
            ref_coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(1.0, 0.0, 0.0)u"nm",
                          SVector(0.0, 1.0, 0.0)u"nm", SVector(0.0, 0.0, 1.0)u"nm"]
            coords = [SVector(0.1, 0.0, 0.0)u"nm", SVector(1.1, 0.0, 0.0)u"nm",
                      SVector(0.0, 1.1, 0.0)u"nm", SVector(0.0, 0.0, 1.1)u"nm"]
            cv = CalcRMSD(ref_coords)
            bias = BiasPotential(cv, SquareBias(400.0u"kJ * mol^-1 * nm^-2", 0.0u"nm"))

            sys_cpu = System(sys_cpu_base; coords=coords)
            sys_gpu = System(sys_gpu_base; coords=CuArray(coords))
            pe_fs_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
        end

        # cv_gradient returns one SVector per atom with no atom-index list (bias.jl's non-built-in
        # path applies it as a dense per-atom array), so this needs its own atom count, not base.
        @testset "Custom CV type in BiasPotential on GPU matches CPU" begin
            struct XCoordDistCV
                correction::Symbol
            end
            Molly.calculate_cv(cv::XCoordDistCV, coords, atoms, boundary, velocities; kwargs...) =
                coords[2][1] - coords[1][1]
            Molly.cv_gradient(cv::XCoordDistCV, coords, atoms, boundary, velocities; kwargs...) =
                ([SVector(-1.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)] .* oneunit(coords[1][1] / coords[1][1]),
                 coords[2][1] - coords[1][1])

            cv = XCoordDistCV(:wrap)
            bias = BiasPotential(cv, SquareBias(400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))

            # coords_v[1:2]'s distance (1.0 nm) would exactly match dist0, giving a zero gradient.
            coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(1.5, 0.0, 0.0)u"nm"]
            sys_cpu = System(atoms=atoms_v[1:2], coords=coords, boundary=boundary)
            sys_gpu = System(atoms=CuArray(atoms_v[1:2]), coords=CuArray(coords), boundary=boundary)
            fs_cpu = pe_fs_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
            @test ustrip(fs_cpu[1][1]) != 0
        end

        # A custom atom type with no :mass field, only a mass(atom) overload, must work through
        # the GPU scratch path (bias_dist_scratch_types reads mass()'s return type).
        @testset "Custom atom type (no :mass field) in CMDist bias on GPU matches CPU" begin
            struct SimpleMassAtom
                m::Float64
            end
            Molly.mass(a::SimpleMassAtom) = a.m * u"g/mol"

            atoms = [SimpleMassAtom(10.0), SimpleMassAtom(20.0), SimpleMassAtom(15.0), SimpleMassAtom(5.0)]
            cv = CalcDist([1, 2], [3, 4], CalcCMDist(), :wrap)
            bias = BiasPotential(cv, SquareBias(400.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))

            sys_cpu = System(sys_cpu_base; atoms=atoms)
            sys_gpu = System(sys_gpu_base; atoms=CuArray(atoms))
            pe_fs_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
        end

        # CalcMinDist/CalcMaxDist forces AND virial on GPU must match CPU, since
        # calculate_virial_dist! recomputes the extremal pair instead of reading a cache.
        @testset "CalcMinDist/CalcMaxDist virial on GPU matches CPU" for dist_type in (CalcMinDist(), CalcMaxDist())
            cv = CalcDist([1, 2], [3, 4], dist_type, :wrap)
            bias = BiasPotential(cv, SquareBias(300.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))
            sys_cpu = System(sys_cpu_base; general_inters=(bias,))
            sys_gpu = System(sys_gpu_base; general_inters=(bias,))
            fs_virial_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
        end

        # CalcRg forces AND virial on GPU must match CPU, through BiasPotential's persistent-buffer
        # path (not just calculate_cv/cv_gradient called directly).
        @testset "CalcRg virial on GPU matches CPU" begin
            cv = CalcRg([1, 2, 3, 4], :wrap)
            bias = BiasPotential(cv, SquareBias(300.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))
            sys_cpu = System(sys_cpu_base; general_inters=(bias,))
            sys_gpu = System(sys_gpu_base; general_inters=(bias,))
            fs_virial_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
        end

        @testset "CalcCMDist virial on GPU matches CPU" begin
            cv = CalcDist([1, 2], [3, 4], CalcCMDist(), :wrap)
            bias = BiasPotential(cv, SquareBias(300.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))
            sys_cpu = System(sys_cpu_base; general_inters=(bias,))
            sys_gpu = System(sys_gpu_base; general_inters=(bias,))
            fs_virial_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
        end

        @testset "CalcSingleDist virial on GPU matches CPU" begin
            cv = CalcDist([1], [4], CalcSingleDist(), :wrap)
            bias = BiasPotential(cv, SquareBias(300.0u"kJ * mol^-1 * nm^-2", 1.0u"nm"))
            sys_cpu = System(sys_cpu_base; general_inters=(bias,))
            sys_gpu = System(sys_gpu_base; general_inters=(bias,))
            fs_virial_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
        end

        # sys_cpu_base's atoms are collinear (undefined torsion); use a non-planar fixture instead.
        @testset "CalcTorsion virial on GPU matches CPU" begin
            coords = [SVector(0.0, 0.0, 0.0)u"nm", SVector(1.0, 0.0, 0.0)u"nm",
                     SVector(1.0, 1.0, 0.0)u"nm", SVector(1.0, 1.0, 1.0)u"nm"]
            cv = CalcTorsion([1, 2, 3, 4], :wrap)
            bias = BiasPotential(cv, SquareBias(300.0u"kJ * mol^-1", 0.5))
            sys_cpu = System(sys_cpu_base; coords=coords, general_inters=(bias,))
            sys_gpu = System(sys_gpu_base; coords=CuArray(coords), general_inters=(bias,))
            fs_virial_gpu_vs_cpu(sys_cpu, sys_gpu, bias)
        end

        # The fused GPU kernel for CalcMinDist/CalcMaxDist uses O(group) memory, not O(group^2) --
        # assert via CUDA.@allocated rather than reproducing the OOM the old approach hit.
        @testset "CalcMinDist GPU memory is O(group), not O(group^2)" begin
            na = 500
            coords = CuArray([SVector(Float32(i % 100) * 0.01f0, 0f0, 0f0)u"nm" for i in 1:(2 * na)])
            atoms = CuArray([Atom(mass=10.0f0u"g/mol") for _ in 1:(2 * na)])
            boundary_f32 = CubicBoundary(100.0f0u"nm")
            cv = CalcDist(collect(1:na), collect((na + 1):(2 * na)), CalcMinDist(), :wrap)
            buff = similar(coords, eltype(eltype(coords)), 1)
            Molly.calculate_cv!(cv, coords, atoms, boundary_f32, buff) # warm up / compile
            bytes = CUDA.@allocated Molly.calculate_cv!(cv, coords, atoms, boundary_f32, buff)
            @test bytes < 500_000 # a few KB expected; O(group^2) at na=500 would be ~3MB

            coords_1, coords_2 = coords[1:na], coords[(na + 1):end]
            Molly.dist_between_groups(CalcMinDist(), coords_1, coords_2, boundary_f32) # warm up
            bytes = CUDA.@allocated Molly.dist_between_groups(CalcMinDist(), coords_1, coords_2, boundary_f32)
            @test bytes < 500_000
        end
@testset "Bias virial" begin
    # The virial of a bias is minus the derivative of its energy with respect to a
    #   homogeneous strain, W = -(dU/dε)ᵀ, here for a molecule split over the boundary
    boundary = CubicBoundary(2.0u"nm")
    coords_whole = [SVector(-0.15, -0.10, 0.05), SVector(0.05, -0.12, -0.08),
                    SVector(0.12, 0.08, 0.10), SVector(-0.05, 0.15, -0.12),
                    SVector(0.20, -0.05, 0.18), SVector(-0.18, 0.12, 0.15)]u"nm"
    atoms = [Atom(mass=m * u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
             for m in (12.0, 1.0, 16.0, 14.0, 12.0, 32.0)]
    topology = MolecularTopology([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], 6)
    ref_coords = [1.1 * c + SVector(0.02, -0.01, 0.03)u"nm" for c in coords_whole]
    k = 500.0u"kJ * mol^-1 * nm^-2"
    cvs_biases = (
        (CalcDist([1], [5], CalcSingleDist()), SquareBias(k, 0.2u"nm")),
        (CalcDist([1, 2], [4, 5, 6], CalcMinDist()), SquareBias(k, 0.4u"nm")),
        (CalcDist([1, 2, 3], [5, 6], CalcMaxDist(:raw)), SquareBias(k, 0.2u"nm")),
        (CalcDist([1, 2, 3], [4, 5, 6], CalcCMDist()), SquareBias(k, 0.3u"nm")),
        (CalcRg(), SquareBias(k, 0.1u"nm")),
        (CalcRMSD(ref_coords), SquareBias(k, 0.0u"nm")),
        (CalcTorsion([1, 2, 3, 4]), SquareBias(50.0u"kJ * mol^-1", 0.5)),
    )
    for (cv, bias) in cvs_biases
        sys = System(atoms=atoms, coords=wrap_coords.(coords_whole, (boundary,)),
                     boundary=boundary, topology=topology,
                     general_inters=(BiasPotential(cv, bias),))
        h = 1e-6
        dU_dε = map(Iterators.product(1:3, 1:3)) do (a, b)
            Us = map((h, -h)) do δ
                μ = SMatrix{3, 3}(I + δ * (1:3 .== a) * (1:3 .== b)')
                cv_val = calculate_cv(cv, [μ * c for c in coords_whole], atoms, boundary)
                return Molly.potential_energy(bias, cv_val)
            end
            return (Us[1] - Us[2]) / 2h
        end
        @test virial(sys) ≈ -transpose(dU_dε) rtol=1e-6
    end
end
