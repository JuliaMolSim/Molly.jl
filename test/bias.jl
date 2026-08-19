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
    coords_1 = coords[atom_inds_1]
    coords_2 = coords[atom_inds_2]
    atoms_1 = atoms[atom_inds_1]
    atoms_2 = atoms[atom_inds_2]

    @test isapprox(
        Molly.center_of_mass(coords_1,atoms_1),
        SVector(0.625, 1.0, 1.0)u"nm";
        atol=1e-9u"nm",
    )

    @test isapprox(
        Molly.center_of_mass(coords_2,atoms_2),
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

    bb_atoms = BioStructures.collectatoms(struc[1], BioStructures.backboneselector)
    coords = SVector{3, Float64}.(eachcol(BioStructures.coordarray(bb_atoms))) / 10 * u"nm"
    bb_to_mass = Dict("C" => 12.011u"g/mol", "N" => 14.007u"g/mol", "O" => 15.999u"g/mol")
    atoms = [Atom(mass=bb_to_mass[BioStructures.element(bb_atoms[i])]) for i in eachindex(bb_atoms)]

    # Rg of all atoms
    rg_cv = CalcRg()
    @test isapprox(
        calculate_cv(rg_cv, coords, atoms),
        11.51225678195222u"Å";
        atol=1e-6u"nm",
    )
    @test isapprox(Molly.cv_gradient(rg_cv, coords, atoms, CubicBoundary(20.0u"nm"))[2],
                   calculate_cv(rg_cv, coords, atoms),
                   atol = 1e-5u"nm")

    # Rg of a subset of atoms
    n_atoms_subset = 20
    coords_subset = coords[1:n_atoms_subset]
    atoms_subset = atoms[1:n_atoms_subset]
    rg_cv = CalcRg([i for i=1:n_atoms_subset])
    @test isapprox(
        calculate_cv(rg_cv, coords, atoms),
        radius_gyration(coords_subset,atoms_subset);
        atol=1e-6u"nm",
    )
    @test isapprox(Molly.cv_gradient(rg_cv, coords, atoms, CubicBoundary(20.0u"nm"))[2],
                   calculate_cv(rg_cv, coords, atoms),
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
    for AT in array_list
        n_atoms = 10
        boundary = CubicBoundary(10.0)
        temp = 298.0
        atom_mass = 10.0

        atoms = to_device([Atom(mass=atom_mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms], AT)
        coords = to_device(place_atoms(n_atoms, boundary; min_dist=0.3), AT)
        velocities = to_device([random_velocity(atom_mass, temp) for i in 1:n_atoms], AT)
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

        simulate!(sys, simulator, 200_000)

        pair_dists_12 = values(sys.loggers.pair_dist_12)
        pair_dists_13 = values(sys.loggers.pair_dist_13)

        dist_12_mean = mean(pair_dists_12[1000:end])
        dist_13_mean = mean(pair_dists_13[1000:end])
        dist_12_std = std(pair_dists_12[1000:100:end])
        dist_13_std = std(pair_dists_13[1000:100:end])

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

        atoms = to_device([Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms], AT)
        coords = to_device(place_atoms(n_atoms, boundary; min_dist=0.3u"nm"), AT)
        velocities = to_device([random_velocity(atom_mass, temp) for i in 1:n_atoms], AT)
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

        simulate!(sys, simulator, 200_000)

        pair_dists_12 = values(sys.loggers.pair_dist_12)
        pair_dists_13 =values(sys.loggers.pair_dist_13)

        dist_12_mean = mean(pair_dists_12[1000:end])
        dist_13_mean = mean(pair_dists_13[1000:end])
        dist_12_std = std(pair_dists_12[1000:100:end])
        dist_13_std = std(pair_dists_13[1000:100:end])

        @test isapprox(dist_12_mean, 1.5u"nm"; atol=0.05u"nm")
        @test !isapprox(dist_13_mean, 1.5u"nm"; atol=0.05u"nm")
        @test dist_13_mean > dist_12_mean
        @test dist_13_std > dist_12_std
    end
end

@testset "MetaDynamicsBias memory" begin
    # ListHills: a single CV
    lh = ListHills(2.0, 0.5, [1.0, 1.5, -0.5])
    @test length(lh.centers) == 3
    @test isapprox(potential_energy(lh, 1.2), 3.52295; atol=1e-3)
    @test isapprox(Molly.bias_gradient(lh, 1.2), 0.485656; atol=1e-3)

    add_hill!(lh, 2.0)
    @test length(lh.centers) == 4
    @test lh.centers[end] == 2.0

    @test_throws ArgumentError ListHills(2.0, -0.5)
    @test_throws ArgumentError ListHills(2.0, 0.0)

    # ListHills: multiple CVs at once, sigma/centers become tuples but k stays a single
    # scalar height for the joint Gaussian
    lh2 = ListHills(3.0, (0.5, 1.0), [(1.0, 2.0), (0.0, 0.0)])
    @test isapprox(potential_energy(lh2, (1.2, 2.5)), 2.451341; atol=1e-3)
    grad2 = Molly.bias_gradient(lh2, (1.2, 2.5))
    @test isapprox(grad2[1], -1.99067; atol=1e-3)
    @test isapprox(grad2[2], -1.24047; atol=1e-3)

    # GridHills: single CV only, deposited hills are accumulated onto the grid
    gh = GridHills(1.0, 0.2, 0.0, 2.0, 5)
    @test length(gh.values) == 5
    @test isapprox(gh.bin_width, 0.5; atol=1e-9)

    add_hill!(gh, 1.0)
    @test isapprox(potential_energy(gh, 1.0), 1.0; atol=1e-9) # Exactly on a grid point
    @test isapprox(potential_energy(gh, 0.75), 0.52196847; atol=1e-3) # Interpolated
    @test isapprox(Molly.bias_gradient(gh, 0.75), 1.91212614; atol=1e-3)

    @test_throws ArgumentError GridHills(1.0, 0.2, 0.0, 2.0, 1)
    @test_throws ArgumentError GridHills(1.0, -0.2, 0.0, 2.0, 5)
    @test_throws ArgumentError GridHills(1.0, 0.2, 2.0, 0.0, 5)
end

@testset "MetaDynamicsBias via BiasPotential" begin
    # A single CV evaluated externally, matching the interface of the other bias
    # potentials in this file (LinearBias, SquareBias, ...)
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    a1 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a2 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    boundary = CubicBoundary(2.0u"nm")

    sys = System(
        atoms=[a1, a2],
        coords=[c1, c2],
        boundary=boundary,
        velocities=[random_velocity(10u"g/mol", 300u"K") for i in 1:2],
    )

    calc_dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    md = MetaDynamicsBias(2.0u"kJ * mol^-1", 0.5u"nm")
    @test isempty(md.cvs)
    @test isapprox(
        AtomsCalculators.potential_energy(sys, BiasPotential(calc_dist, md)),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    add_hill!(md, 1.0u"nm")
    add_hill!(md, 1.5u"nm")
    add_hill!(md, -0.5u"nm")
    @test length(md.memory.centers) == 3

    # Actual distance between the two atoms is 0.3 nm
    @test isapprox(potential_energy(md, 0.3u"nm"), 1.41897u"kJ * mol^-1"; atol=1e-3u"kJ * mol^-1")
    @test isapprox(
        Molly.bias_gradient(md, 0.3u"nm"),
        0.86120u"kJ * mol^-1 * nm^-1";
        atol=1e-3u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        AtomsCalculators.potential_energy(sys, BiasPotential(calc_dist, md)),
        1.41897u"kJ * mol^-1";
        atol=1e-3u"kJ * mol^-1",
    )

    fs = Molly.zero_forces(sys)
    AtomsCalculators.forces!(fs, sys, BiasPotential(calc_dist, md))
    @test isapprox(
        fs[1], SVector(0.86120, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-3u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs[2], SVector(-0.86120, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-3u"kJ * mol^-1 * nm^-1",
    )
end

@testset "MetaDynamicsBias multi-CV calculator" begin
    # cvs stored directly on MetaDynamicsBias, making it a self-contained AtomsCalculators
    # calculator usable directly as a general_inters entry, evaluating several CVs at once
    atoms = [Atom(mass=10.0, σ=0.3, ϵ=0.2) for _ in 1:3]
    coords = [SVector(1.0, 1.0, 1.0), SVector(1.3, 1.0, 1.0), SVector(1.4, 1.0, 1.0)]
    boundary = CubicBoundary(5.0)
    velocities = [SVector(0.0, 0.0, 0.0) for _ in 1:3]

    cv1 = CalcDist([1], [2], CalcSingleDist(), :wrap) # Distance 1-2 is 0.3
    cv2 = CalcDist([1], [3], CalcSingleDist(), :wrap) # Distance 1-3 is 0.4

    @test_throws ArgumentError MetaDynamicsBias((cv1, cv2), GridHills(1.0, 0.1, 0.0, 1.0, 5))

    bias = MetaDynamicsBias((cv1, cv2), 5.0, (0.1, 0.1), Tuple{Float64, Float64}[])
    @test length(bias.cvs) == 2

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        general_inters=(bias,),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    @test_throws ArgumentError AtomsCalculators.potential_energy(sys, MetaDynamicsBias(2.0, 0.5))
    @test isapprox(AtomsCalculators.potential_energy(sys, bias), 0.0; atol=1e-9)

    add_hill!(bias, sys)
    @test length(bias.memory.centers) == 1
    @test isapprox(bias.memory.centers[1][1], 0.3; atol=1e-9)
    @test isapprox(bias.memory.centers[1][2], 0.4; atol=1e-9)

    # Evaluated exactly at the deposited hill's own centre, the potential equals its
    # height, and the gradient (the Gaussian's own peak) is zero in every CV dimension, so
    # the resulting force on every atom should be ~0
    @test isapprox(AtomsCalculators.potential_energy(sys, bias), 5.0; atol=1e-9)

    fs = Molly.zero_forces(sys)
    AtomsCalculators.forces!(fs, sys, bias)
    for f in fs
        @test isapprox(f, SVector(0.0, 0.0, 0.0); atol=1e-9)
    end
end

@testset "MetaDynamicsBias simulation" begin
    # Two atoms connected by a HarmonicBond, biased with MetaDynamicsBias used directly as
    # a general interaction. After a thermostatted run depositing a modest number of hills,
    # the accumulated bias should start to recover (the negative of) the underlying bond
    # potential: the standard metadynamics free energy reconstruction result, here checked
    # by verifying that adding the bias back to the true bond potential flattens it out
    # over the sampled region compared to the bare bond potential alone.
    mass = 10.0
    r0 = 1.0
    k_bond = 500.0
    temp = 298.0
    boundary = CubicBoundary(10.0)

    atoms = [Atom(mass=mass, σ=0.3, ϵ=0.2), Atom(mass=mass, σ=0.3, ϵ=0.2)]
    coords = [SVector(4.5, 5.0, 5.0), SVector(4.5 + r0, 5.0, 5.0)]
    velocities = [random_velocity(mass, temp) for i in 1:2]

    bond = HarmonicBond(k=k_bond, r0=r0)
    specific_inter_lists = (InteractionList2Atoms([1], [2], [bond]),)

    calc_dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    bias = MetaDynamicsBias((calc_dist,), 0.5, 0.02)

    simulator = VelocityVerlet(dt=0.002, coupling=AndersenThermostat(temp, 0.1))

    deposit_every = 200
    deposit_wrapper(sys, args...; kwargs...) = (add_hill!(bias, sys); true)

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        specific_inter_lists=specific_inter_lists,
        general_inters=(bias,),
        force_units=NoUnits,
        energy_units=NoUnits,
        loggers=(
            hill_deposit=GeneralObservableLogger(deposit_wrapper, Bool, deposit_every),
        ),
    )

    simulate!(sys, simulator, 20_000)

    centers = bias.memory.centers
    @test length(centers) >= 50
    @test all(isfinite, centers)

    # Every deposited hill contributes at least its own height to the bias evaluated at its
    # own centre, since every other hill's contribution is non-negative
    @test all(c -> potential_energy(bias.memory, c) >= bias.memory.k - 1e-9, centers)

    # Far outside anything ever visited, the accumulated bias has decayed to ~0
    c_min, c_max = extrema(centers)
    far = c_max + 50 * bias.memory.sigma
    @test potential_energy(bias.memory, far) < 1e-12 * bias.memory.k

    # The bond confines the atoms near r0, so the thermally-sampled centres should cluster
    # there more densely than at either edge of the range actually visited
    c_mean = mean(centers)
    @test potential_energy(bias.memory, c_mean) > potential_energy(bias.memory, c_min)
    @test potential_energy(bias.memory, c_mean) > potential_energy(bias.memory, c_max)

    # Recovering the bond potential: adding the accumulated bias back to the true harmonic
    # bond potential should flatten it out over the sampled region, compared to the bare
    # bond potential alone
    bond_pe(r) = potential_energy(bond, SVector(0.0, 0.0, 0.0), SVector(r, 0.0, 0.0), boundary)
    bare = [bond_pe(c) for c in centers]
    combined = [bond_pe(c) + potential_energy(bias.memory, c) for c in centers]
    @test std(combined) < 0.9 * std(bare)
end


