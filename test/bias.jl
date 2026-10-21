using Molly: NoTempering, AbstractMetadynamicsMemory, add_hill!, free_energy,
             metadynamics_log_weights

struct BiasNaNGradient end

Molly.bias_gradient(::BiasNaNGradient, cv_sim) = NaN * u"kJ * mol^-1 * nm^-1"

# A CV that always evaluates to NaN
struct CVNaN
    correction::Symbol
    has_virial::Bool
end

Molly.calculate_cv(::CVNaN, coords, args...; kwargs...) = NaN * u"nm"

# A custom memory implementing only the documented contract: a constant bias
struct ConstMemory{C} <: AbstractMetadynamicsMemory
    cvs::C
    height::Float64
end

Molly.potential_energy(m::ConstMemory, cv; kwargs...) = m.height
Molly.bias_gradient(::ConstMemory, cv; kwargs...) = 0.0
Molly.add_hill!(::ConstMemory, cv, height) = nothing
Molly.cv_gradient(::CVNaN, coords, args...; kwargs...) = (zero(coords) ./ oneunit(eltype(eltype(coords))), NaN * u"nm")

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

@testset "Metadynamics memory" begin
    # Non-periodic distance CVs, one or two
    dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    one_cv, two_cvs = (dist,), (dist, dist)
    @test_throws ArgumentError ListHills((), 2.0, 0.5)
    @test_throws ArgumentError GridHills((), 1.0, 0.2, 0.0, 2.0, 5)

    # ListHills: a single CV
    lh = ListHills(one_cv, 2.0, 0.5, [1.0, 1.5, -0.5])
    @test isapprox(potential_energy(lh, 1.2), 3.52295; atol=1e-3)
    @test isapprox(Molly.bias_gradient(lh, 1.2), 0.485656; atol=1e-3)

    add_hill!(lh, 2.0)
    @test length(lh.centers) == 4
    @test lh.centers[end] == 2.0

    @test_throws ArgumentError ListHills(one_cv, 2.0, -0.5)
    @test_throws ArgumentError ListHills(one_cv, 2.0, 0.0)

    # ListHills: multiple CVs at once, sigma/centers become tuples but k stays a single
    # scalar height for the joint Gaussian
    lh2 = ListHills(two_cvs, 3.0, (0.5, 1.0), [(1.0, 2.0), (0.0, 0.0)])
    @test isapprox(potential_energy(lh2, (1.2, 2.5)), 2.451341; atol=1e-3)
    grad2 = Molly.bias_gradient(lh2, (1.2, 2.5))
    @test isapprox(grad2[1], -1.99067; atol=1e-3)
    @test isapprox(grad2[2], -1.24047; atol=1e-3)

    # GridHills: a single CV, deposited hills are accumulated onto the grid. Scalar
    # arguments are internally wrapped into length-1 tuples (n_dims == 1), and bias_gradient
    # collapses its result back down to a bare scalar for this case.
    gh = GridHills(one_cv, 1.0, 0.2, 0.0, 2.0, 5)
    @test length(gh.values) == 5
    @test isapprox(only(gh.bin_width), 0.5; atol=1e-9)
    @test gh.cutoff == 6
    @test GridHills(one_cv, 1.0, 0.2, 0.0, 2.0, 5; cutoff=3).cutoff == 3
    # A bare CV is wrapped in a tuple; a tuple with wrong arguments isn't wrapped again
    @test GridHills(only(one_cv), 1.0, 0.2, 0.0, 2.0, 5; cutoff=3).cvs === one_cv
    @test ListHills(only(one_cv), 1.0, 0.2).cvs === one_cv
    @test_throws MethodError ListHills(one_cv, 1.0, 0.2, Float64[], Float64[], 1)
    @test_throws MethodError GridHills(one_cv, 1.0, 0.2, 0.0)
    # A vector of CVs is taken as a tuple
    @test ListHills([only(one_cv), only(one_cv)], 1.0, (0.1, 0.1)).cvs == (only(one_cv), only(one_cv))
    @test GridHills([only(one_cv)], 1.0, 0.2, 0.0, 2.0, 5).cvs === one_cv

    # The scalar form builds the same grid as per-dimension tuples
    gh_t = GridHills(one_cv, 1.0, (0.2,), (0.0,), (2.0,), (5,))
    @test gh_t.bin_width == gh.bin_width && gh_t.period == gh.period
    @test size(gh_t.values) == size(gh.values)

    add_hill!(gh, 1.0)
    @test isapprox(potential_energy(gh, 1.0), 1.0; atol=1e-9) # Exactly on a grid point
    @test isapprox(potential_energy(gh, 0.75), 0.52196847; atol=1e-3) # Interpolated
    @test isapprox(Molly.bias_gradient(gh, 0.75), 1.91212614; atol=1e-3)

    @test_throws ArgumentError GridHills(one_cv, 1.0,  0.2, 0.0, 2.0, 1)
    @test_throws ArgumentError GridHills(one_cv, 1.0, -0.2, 0.0, 2.0, 5)
    @test_throws ArgumentError GridHills(one_cv, 1.0,  0.2, 2.0, 0.0, 5)

    # GridHills: multiple CVs at once, sigma/grid_min/grid_max/n_points become tuples (one
    # entry per dimension) and hills are deposited/evaluated via N-linear interpolation
    gh2d = GridHills(two_cvs, 1.0, (0.1, 0.1), (0.0, 0.0), (2.0, 2.0), 9) # bin_width (0.25, 0.25)
    @test size(gh2d.values) == (9, 9)
    @test all(isapprox.(gh2d.bin_width, (0.25, 0.25); atol=1e-9))

    add_hill!(gh2d, (1.0, 1.0)) # Lands exactly on grid point (index 5, 5)
    # cutoff=6 default reaches +/-0.6 in each dimension from (1.0, 1.0), i.e. indices 2:8;
    # the grid corners are well outside that box and so are untouched (still exactly 0)
    @test gh2d.values[1, 1] == 0.0
    @test gh2d.values[1, 5] == 0.0
    @test gh2d.values[5, 1] == 0.0
    @test gh2d.values[9, 9] == 0.0

    # Evaluated exactly at the hill center, both potential_energy and bias_gradient reduce
    # to reading/differencing straight off the grid (no interpolation blending needed)
    @test isapprox(potential_energy(gh2d, (1.0, 1.0)), 1.0; atol=1e-9)
    grad2d = Molly.bias_gradient(gh2d, (1.0, 1.0))
    @test isapprox(grad2d[1], -3.82423; atol=1e-3)
    # Grid, sigma and hill placement are all symmetric between the two CV dimensions here,
    # so the gradient components must match by symmetry
    @test isapprox(grad2d[1], grad2d[2]; atol=1e-9)

    @test_throws ArgumentError GridHills(two_cvs, 1.0, (0.1, 0.1), (0.0, 0.0, 0.0), (2.0, 2.0), 9)
    @test_throws ArgumentError GridHills(two_cvs, 1.0, (0.1, 0.1), (0.0, 0.0), (2.0, 2.0), (5, 5, 5))

    # n_points can also be given per-dimension instead of a single shared Integer
    gh2d_uneven = GridHills(two_cvs, 1.0, (0.1, 0.2), (0.0, 0.0), (1.0, 2.0), (3, 5))
    @test size(gh2d_uneven.values) == (3, 5)
    @test all(isapprox.(gh2d_uneven.bin_width, (0.5, 0.5); atol=1e-9))

    # Integer grid bounds are converted to floats
    gh_int = GridHills(one_cv, 1.0, 0.1, 0, 5, 51)
    @test only(gh_int.bin_width) isa AbstractFloat
    @test isapprox(only(gh_int.bin_width), 0.1; atol=1e-9)

    # By default leaving the grid follows the strictness passed by the simulation
    @test gh.out_of_grid == :global_strictness
    @test_throws ArgumentError potential_energy(gh, 2.5; strictness=:error)
    @test_throws ArgumentError Molly.bias_gradient(gh, -0.5; strictness=:error)
    @test_logs potential_energy(gh, 2.5; strictness=:nowarn)
    @test_throws ArgumentError GridHills(one_cv, 1.0, 0.2, 0.0, 2.0, 5; out_of_grid=:maybe)
    # A strictness set on the grid overrides the simulation's
    gh_err = GridHills(one_cv, 1.0, 0.2, 0.0, 2.0, 5; out_of_grid=:error)
    @test_throws ArgumentError potential_energy(gh_err, 2.5; strictness=:nowarn)
    @test_throws ArgumentError add_hill!(gh_err, 2.5)

    # With :warn the bias warns once and is held constant outside the grid, so it exerts
    #   no force there
    gh_soft = GridHills(one_cv, 1.0, 0.2, 0.0, 2.0, 5; out_of_grid=:warn)
    add_hill!(gh_soft, 2.0)
    @test_logs (:warn,) begin
        potential_energy(gh_soft, 2.5; strictness=:error)
        potential_energy(gh_soft, 2.6)
    end
    # Its own report_issue id, so another warning with maxlog=1 doesn't suppress it
    @test_logs (:warn, "other") (:warn, r"outside the grid") begin
        Molly.report_issue("other", :warn; maxlog=1)
        potential_energy(gh_soft, 2.5)
    end
    # Held constant outside the grid; :nowarn on the grid silences it
    gh_quiet = GridHills(one_cv, 1.0, 0.2, 0.0, 2.0, 5; out_of_grid=:nowarn)
    add_hill!(gh_quiet, 2.0)
    @test_logs potential_energy(gh_quiet, 2.5; strictness=:warn)
    @test isapprox(potential_energy(gh_quiet, 2.5), potential_energy(gh_quiet, 2.0); atol=1e-9)
    @test Molly.bias_gradient(gh_quiet, 2.5) == 0.0
    @test !iszero(Molly.bias_gradient(gh_quiet, 1.9))

    gh2d_soft = GridHills(two_cvs, 1.0, (0.1, 0.1), (0.0, 0.0), (2.0, 2.0), 9; out_of_grid=:warn)
    add_hill!(gh2d_soft, (1.0, 1.9))
    grad_out = @test_logs (:warn,) Molly.bias_gradient(gh2d_soft, (1.1, 2.5))
    grad_edge = Molly.bias_gradient(gh2d_soft, (1.1, 2.0))
    @test grad_out[2] == 0.0
    @test isapprox(grad_out[1], grad_edge[1]; atol=1e-9)

    # A bias deposits no hill and records nothing while the CV is outside the grid
    md_soft = Metadynamics(GridHills(one_cv, 1.0, 0.2, 0.0, 2.0, 5; out_of_grid=:warn);
                           logger=MetadynamicsLogger(temperature=300.0))
    @test_logs (:warn,) add_hill!(md_soft, 2.5)
    @test all(iszero, md_soft.memory.values)
    @test md_soft.n_deposits == 0
    @test isempty(md_soft.logger.steps)
    add_hill!(md_soft, 1.0)
    @test md_soft.n_deposits == 1
    @test length(md_soft.logger.steps) == 1
end

@testset "Metadynamics default centers" begin
    # Without explicit centers, deposits keep the units and precision of sigma
    # One distance CV per sigma entry
    dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    bias_with(k, sigma; kwargs...) = Metadynamics(
        ListHills(sigma isa Tuple ? (dist, dist) : (dist,), k, sigma); kwargs...)
    md_units = bias_with(2.0u"kJ * mol^-1", 0.5u"nm")
    add_hill!(md_units, 1.0u"nm")
    @test md_units.memory.centers == [1.0u"nm"]

    md_tuple = bias_with(2.0, (0.5, 1.0))
    add_hill!(md_tuple, (1.0, 2.0))
    @test md_tuple.memory.centers == [(1.0, 2.0)]

    md_f64 = bias_with(2.0, 0.5)
    add_hill!(md_f64, 0.1)
    @test eltype(md_f64.memory.centers) == Float64
    @test md_f64.memory.centers[1] === 0.1

    md_f32 = bias_with(2.0f0, 0.5f0)
    @test eltype(md_f32.memory.centers) == Float32

    # Integer sigma still stores fractional CV values
    md_int = bias_with(1.0, 1)
    add_hill!(md_int, 0.5)
    @test md_int.memory.centers == [0.5]

    md_int_tuple = bias_with(1.0, (0.5, 1))
    add_hill!(md_int_tuple, (0.1, 0.2))
    @test md_int_tuple.memory.centers == [(0.1, 0.2)]

    # Integer k still stores fractional, e.g. tempered, heights
    @test eltype(ListHills((dist,), 1, 0.5).heights) == Float64
    md_int_k = bias_with(1, 0.5; tempering=WellTempered(10.0, 2.5))
    add_hill!(md_int_k, 0.1)
    add_hill!(md_int_k, 0.1)
    @test eltype(md_int_k.memory.heights) == Float64
    @test md_int_k.memory.heights[2] < 1.0
    gh_int_k = GridHills((dist,), 1, 0.2, 0.0, 2.0, 5)
    add_hill!(gh_int_k, 0.75)
    @test eltype(gh_int_k.values) == Float64
end

@testset "Metadynamics with units" begin
    # A single distance CV, with units
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
    md = Metadynamics(ListHills((calc_dist,), 2.0u"kJ * mol^-1", 0.5u"nm",
                                typeof(1.0u"nm")[]))
    @test isapprox(
        AtomsCalculators.potential_energy(sys, md),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    add_hill!(md, 1.0u"nm")
    add_hill!(md, 1.5u"nm")
    add_hill!(md, -0.5u"nm")
    @test length(md.memory.centers) == 3

    # Actual distance between the two atoms is 0.3 nm
    @test isapprox(potential_energy(md.memory, 0.3u"nm"), 1.41897u"kJ * mol^-1";
                   atol=1e-3u"kJ * mol^-1")
    @test isapprox(
        Molly.bias_gradient(md.memory, 0.3u"nm"),
        0.86120u"kJ * mol^-1 * nm^-1";
        atol=1e-3u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        AtomsCalculators.potential_energy(sys, md),
        1.41897u"kJ * mol^-1";
        atol=1e-3u"kJ * mol^-1",
    )

    fs = Molly.zero_forces(sys)
    AtomsCalculators.forces!(fs, sys, md)
    @test isapprox(
        fs[1], SVector(0.86120, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-3u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs[2], SVector(-0.86120, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-3u"kJ * mol^-1 * nm^-1",
    )

    # A height that doesn't convert leaves centers and heights the same length
    mem_units = ListHills((calc_dist,), 1.0u"kJ * mol^-1", 0.1)
    @test_throws Unitful.DimensionError add_hill!(mem_units, 0.3, 1.0)
    @test isempty(mem_units.centers) && isempty(mem_units.heights)

    # A non-finite CV value throws before any hill is deposited
    md_nan = Metadynamics(ListHills((calc_dist,), 2.0u"kJ * mol^-1", 0.5u"nm"))
    @test_throws ArgumentError add_hill!(md_nan, NaN * u"nm")
    @test md_nan.call_count == 0
    @test md_nan.n_deposits == 0
    @test isempty(md_nan.memory.centers)

    md_nan_cvs = Metadynamics(ListHills((CVNaN(:wrap, false),), 2.0u"kJ * mol^-1", 0.5u"nm"))
    @test_throws ErrorException AtomsCalculators.forces!(Molly.zero_forces(sys), sys, md_nan_cvs)
    @test_throws ErrorException AtomsCalculators.forces!(Molly.zero_forces(sys), sys, md_nan_cvs;
                                                         step_n=0)
    @test_throws ErrorException AtomsCalculators.potential_energy(sys, md_nan_cvs)
    @test_throws ErrorException add_hill!(md_nan_cvs, sys)
    @test md_nan_cvs.call_count == 0
    @test md_nan_cvs.n_deposits == 0
    @test isempty(md_nan_cvs.memory.centers)
end

@testset "Metadynamics multi-CV calculator" begin
    # CVs stored in the memory make the bias a self-contained AtomsCalculators calculator
    # usable directly as a general_inters entry, evaluating several CVs at once
    atoms = [Atom(mass=10.0, σ=0.3, ϵ=0.2) for _ in 1:3]
    coords = [SVector(1.0, 1.0, 1.0), SVector(1.3, 1.0, 1.0), SVector(1.4, 1.0, 1.0)]
    boundary = CubicBoundary(5.0)
    velocities = [SVector(0.0, 0.0, 0.0) for _ in 1:3]

    cv1 = CalcDist([1], [2], CalcSingleDist(), :wrap) # Distance 1-2 is 0.3
    cv2 = CalcDist([1], [3], CalcSingleDist(), :wrap) # Distance 1-3 is 0.4
    cvs = (cv1, cv2)

    # The memory needs one sigma entry per CV
    @test_throws ArgumentError GridHills(cvs, 1.0, 0.1, 0.0, 1.0, 5)
    @test_throws ArgumentError ListHills((cv1,), 5.0, (0.1, 0.1))
    @test_throws ArgumentError ListHills(cvs, 2.0, 0.5)
    @test_throws ArgumentError ListHills((cv1,), 5.0, 0.1, Tuple{Float64, Float64}[])
    @test ListHills((cv1,), 1.0, (0.1,)).sigma == (0.1,)
    @test_throws ArgumentError ListHills((cv1,), 1.0, 0.1, Tuple{Float64}[])
    @test_throws ArgumentError ListHills(cvs, 1.0, (0.1, 0.1), Tuple[])
    @test Metadynamics(ListHills((cv1,), 5.0, 0.1); n_steps=7).n_steps == 7
    @test Metadynamics(ListHills((cv1,), 5.0, 0.1)).memory.cvs === (cv1,)
    # The empty check of the constructor guards custom memories
    @test_throws ArgumentError Metadynamics(ConstMemory((), 2.0))

    # Each CV is evaluated on the coordinates its own correction asks for, so :wrap and
    #   :pbc CVs can be mixed
    md_mixed = Metadynamics(ListHills((cv1, CalcDist([1], [3], CalcSingleDist(), :pbc)),
                                      5.0, (0.1, 0.1)))
    sys_mixed = System(atoms=atoms, coords=coords, boundary=boundary, velocities=velocities,
                       force_units=NoUnits, energy_units=NoUnits)
    AtomsCalculators.forces!(Molly.zero_forces(sys_mixed), sys_mixed, md_mixed)
    @test all(isapprox.(only(md_mixed.memory.centers), (0.3, 0.4); atol=1e-9))

    # A custom memory with only the documented contract works through the System
    md_const = Metadynamics(ConstMemory((cv1,), 2.0))
    sys_const = System(atoms=atoms, coords=coords, boundary=boundary, velocities=velocities,
                       general_inters=(md_const,), force_units=NoUnits, energy_units=NoUnits)
    @test potential_energy(sys_const) ≈ 2.0
    @test all(iszero, forces(sys_const))

    # Biases built from one memory are walkers sharing their hills
    walkers_mem = ListHills(cvs, 5.0, (0.1, 0.1))
    walker_1, walker_2 = Metadynamics(walkers_mem), Metadynamics(walkers_mem)
    add_hill!(walker_1, (0.3, 0.4))
    @test walker_2.memory === walkers_mem
    @test length(walker_2.memory.centers) == 1

    # A GridHills built with one grid dimension per CV works directly as a multi-CV memory
    bias_grid = Metadynamics(GridHills(cvs, 5.0, (0.1, 0.1), (0.0, 0.0), (1.0, 1.0), 11))
    sys_grid = System(
        atoms=atoms, coords=coords, boundary=boundary, velocities=velocities,
        general_inters=(bias_grid,), force_units=NoUnits, energy_units=NoUnits,
    )
    fs_grid = Molly.zero_forces(sys_grid)
    AtomsCalculators.forces!(fs_grid, sys_grid, bias_grid) # Deposits a hill at (0.3, 0.4)
    @test isapprox(AtomsCalculators.potential_energy(sys_grid, bias_grid), 5.0; atol=1e-6)

    bias = Metadynamics(ListHills(cvs, 5.0, (0.1, 0.1), Tuple{Float64, Float64}[]))

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        general_inters=(bias,),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    @test isapprox(AtomsCalculators.potential_energy(sys, bias), 0.0; atol=1e-9)

    # forces! is self-updating: with the default n_steps=1, every call both
    # computes forces and deposits a hill, with no external logger needed. This first call
    # starts from an empty memory, so the resulting forces are trivially zero.
    fs = Molly.zero_forces(sys)
    AtomsCalculators.forces!(fs, sys, bias)
    @test length(bias.memory.centers) == 1
    @test isapprox(bias.memory.centers[1][1], 0.3; atol=1e-9)
    @test isapprox(bias.memory.centers[1][2], 0.4; atol=1e-9)
    for f in fs
        @test isapprox(f, SVector(0.0, 0.0, 0.0); atol=1e-9)
    end
    @test isapprox(AtomsCalculators.potential_energy(sys, bias), 5.0; atol=1e-9)

    # Coordinates haven't moved, so this second call evaluates the gradient exactly at the
    # hill deposited above -- a Gaussian's own peak has zero gradient in every CV
    # dimension, so the force on every atom should again be ~0, non-trivially this time
    # since the memory is no longer empty. It also deposits a second, identical hill.
    fs2 = Molly.zero_forces(sys)
    AtomsCalculators.forces!(fs2, sys, bias)
    @test length(bias.memory.centers) == 2
    for f in fs2
        @test isapprox(f, SVector(0.0, 0.0, 0.0); atol=1e-9)
    end
    @test isapprox(AtomsCalculators.potential_energy(sys, bias), 10.0; atol=1e-9)

    # Away from the hill, the force is the sum of the forces of both CVs: atom 1 is pulled
    # by both distances, atoms 2 and 3 only by their own
    md_off = Metadynamics(ListHills(cvs, 5.0, (0.1, 0.1), [(0.25, 0.45)]))
    fs_off = Molly.zero_forces(sys)
    AtomsCalculators.forces!(fs_off, sys, md_off)
    d_bias = Molly.bias_gradient(md_off.memory, (0.3, 0.4))
    @test isapprox(fs_off[2], SVector(-d_bias[1], 0.0, 0.0); atol=1e-9)
    @test isapprox(fs_off[3], SVector(-d_bias[2], 0.0, 0.0); atol=1e-9)
    @test isapprox(fs_off[1], -(fs_off[2] + fs_off[3]); atol=1e-9)

    # n_steps paces the automatic deposits: only every 3rd force evaluation
    # should actually add a hill
    md3 = Metadynamics(ListHills(cvs, 5.0, (0.1, 0.1), Tuple{Float64, Float64}[]);
                       n_steps=3)
    sys3 = System(
        atoms=atoms, coords=coords, boundary=boundary, velocities=velocities,
        general_inters=(md3,), force_units=NoUnits, energy_units=NoUnits,
    )
    fs3 = Molly.zero_forces(sys3)
    for i in 1:6
        AtomsCalculators.forces!(fs3, sys3, md3)
    end
    @test length(md3.memory.centers) == 2
    @test md3.n_deposits == 2

    # One hill per window of 3 steps (at steps 1, 3 and 6), none for recomputed forces
    md_step = Metadynamics(ListHills(cvs, 5.0, (0.1, 0.1)); n_steps=3)
    for step_n in 1:6, _ in 1:2
        AtomsCalculators.forces!(fs3, sys3, md_step; step_n=step_n)
    end
    @test md_step.n_deposits == 3
    @test md_step.last_deposit_step == 6

    # Chunks deposit as one continuous run, with or without continuing step numbers
    function run_chunks!(md, chunks)
        for chunk in chunks, step_n in chunk
            AtomsCalculators.forces!(fs3, sys3, md; step_n=step_n)
        end
        return md.n_deposits
    end
    new_bias(interval) = Metadynamics(ListHills(cvs, 5.0, (0.1, 0.1)); n_steps=interval)
    # Continued: hills at 0, 50, ..., 200 like one continuous run
    @test run_chunks!(new_bias(50), (0:100, 100:200)) == 5
    # Restarted: the second chunk's step 0 is the first one's last step, hills at 150 and 200
    restarted = new_bias(50)
    @test run_chunks!(restarted, (0:100, 0:100)) == 5
    @test restarted.last_deposit_step == 100
    @test run_chunks!(new_bias(100), fill(0:250, 4)) == 11
    # Runs of a single step deposit every step when continued, but repeat the same step
    #   number otherwise
    @test run_chunks!(new_bias(1), [i:i for i in 1:10]) == 10
    @test run_chunks!(new_bias(1), fill(1:1, 10)) == 1

    # Manual deposits use a separate call counter and leave the step schedule unchanged
    mixed = new_bias(3)
    run_chunks!(mixed, (0:2,))
    AtomsCalculators.forces!(fs3, sys3, mixed)
    AtomsCalculators.forces!(fs3, sys3, mixed)
    add_hill!(mixed, sys3) # 3rd manual call, deposits
    @test mixed.n_deposits == 2
    run_chunks!(mixed, (3:6,))
    @test mixed.last_deposit_step == 6
    @test mixed.n_deposits == 4
end

@testset "Metadynamics free energy" begin
    dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    γ = 10.0

    # The prefactor comes from the tempering: -1 untempered, -γ/(γ-1) well-tempered
    lh = ListHills((dist,), 1.0, 0.1)
    add_hill!(lh, 0.5, 0.8)
    @test free_energy(Metadynamics(lh), 0.55) ≈ -potential_energy(lh, 0.55)
    md_wt = Metadynamics(lh; tempering=WellTempered(γ, 2.5))
    @test free_energy(md_wt, 0.55) ≈ -potential_energy(lh, 0.55) * γ / (γ - 1)

    # Broadcasts over CV values, for any memory
    gh = GridHills((dist,), 1.0, 0.1, 0.0, 1.0, 101)
    add_hill!(gh, 0.5, 0.8)
    rs = [0.4, 0.5, 0.6]
    @test free_energy.(Metadynamics(gh), rs) ≈ [-potential_energy(gh, r) for r in rs]

    # Several CVs take a tuple
    lh2 = ListHills((dist, dist), 1.0, (0.1, 0.2))
    add_hill!(lh2, (0.5, 0.5), 0.8)
    @test free_energy(Metadynamics(lh2), (0.5, 0.6)) ≈ -potential_energy(lh2, (0.5, 0.6))

    # Keeps the energy units
    lh_u = ListHills((dist,), 1.0u"kJ * mol^-1", 0.1u"nm")
    add_hill!(lh_u, 0.5u"nm", 1.0u"kJ * mol^-1")
    @test unit(free_energy(Metadynamics(lh_u), 0.5u"nm")) == u"kJ * mol^-1"
end

@testset "Metadynamics AtomsCalculators interface" begin
    # forces, energy_forces and energy_forces! match forces! and potential_energy. A large
    #   n_steps keeps the manual call counter from depositing between the calls.
    dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    mem = GridHills((dist,), 1.0, 0.1, 0.0, 1.0, 101)
    add_hill!(mem, 0.45, 1.0)
    md = Metadynamics(mem; n_steps=1000)
    sys = System(
        atoms=[Atom(mass=10.0, σ=0.3, ϵ=0.2) for _ in 1:2],
        coords=[SVector(1.0, 1.0, 1.0), SVector(1.5, 1.0, 1.0)],
        boundary=CubicBoundary(3.0), general_inters=(md,),
        force_units=NoUnits, energy_units=NoUnits,
    )
    fs = AtomsCalculators.forces!(Molly.zero_forces(sys), sys, md)
    @test any(!iszero, fs)
    @test AtomsCalculators.forces(sys, md) ≈ fs
    ef = AtomsCalculators.energy_forces(sys, md)
    @test ef.forces ≈ fs
    @test ef.energy ≈ AtomsCalculators.potential_energy(sys, md)
    ef_inplace = AtomsCalculators.energy_forces!(Molly.zero_forces(sys), sys, md)
    @test ef_inplace.forces ≈ fs && ef_inplace.energy ≈ ef.energy
    @test md.n_deposits == 0

    # Without a step number they count towards the manual deposits, as forces! does
    md_1 = Metadynamics(ListHills((dist,), 1.0, 0.1))
    AtomsCalculators.forces(sys, md_1)
    AtomsCalculators.energy_forces(sys, md_1)
    @test md_1.n_deposits == 2
end

@testset "Metadynamics logger" begin
    dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    γ, kT = 10.0, 2.5
    T = kT / Molly.default_k(NoUnits) # The temperature giving this kT
    pts = range(0.0, 1.0; length=101)

    # Each deposit records the bias before its hill and the offset c(t) over cv_points
    md = Metadynamics(ListHills((dist,), 1.0, 0.1); tempering=WellTempered(γ, T),
                      logger=MetadynamicsLogger(cv_points=pts))
    test_values = [0.5, 0.52, 0.3]
    for s in test_values
        add_hill!(md, s)
    end
    lg = md.logger
    @test lg.steps == [-1, -1, -1]
    @test values(lg) == test_values
    mem_2 = ListHills((dist,), 1.0, 0.1, md.memory.centers[1:2], md.memory.heights[1:2])
    v_2 = [potential_energy(mem_2, s) for s in pts]
    a = γ / (γ - 1)
    @test lg.bias_energies[3] ≈ potential_energy(mem_2, 0.3)
    @test lg.offsets[3] ≈ kT * log(sum(exp.(a .* v_2 ./ kT)) / sum(exp.((a - 1) .* v_2 ./ kT)))
    @test lg.points.values ≈ v_2
    @test sum(metadynamics_weights(lg)) ≈ 1
    @test metadynamics_log_weights(lg) ≈ (lg.bias_energies .- lg.offsets) ./ kT

    # The buffer also catches hills of other walkers sharing the memory
    walker = Metadynamics(md.memory; tempering=WellTempered(γ, T))
    add_hill!(walker, 0.7)
    add_hill!(md, 0.6)
    @test lg.points.values ≈ [potential_energy(md.memory, s) for s in pts] .-
                        [Molly.hill_value(md.memory, s, 0.6, md.memory.heights[end]) for s in pts]

    # GridHills integrates over its grid, with units
    gh = GridHills((dist,), 1.0u"kJ * mol^-1", 0.1u"nm", 0.0u"nm", 1.0u"nm", 101)
    md_g = Metadynamics(gh; tempering=WellTempered(γ, T * u"K"),
                        logger=MetadynamicsLogger())
    add_hill!(md_g, 0.5u"nm")
    add_hill!(md_g, 0.55u"nm")
    @test unit(md_g.logger.offsets[2]) == u"kJ * mol^-1"
    @test sum(metadynamics_weights(md_g.logger)) ≈ 1

    # Records go to the file in batches of write_interval, flush writes the rest
    path = tempname()
    md_f = Metadynamics(ListHills((dist,), 1.0, 0.1); tempering=WellTempered(γ, T),
                        logger=MetadynamicsLogger(cv_points=pts, filepath=path, write_interval=2))
    for s in (0.5, 0.52, 0.3)
        add_hill!(md_f, s)
    end
    @test length(readlines(path)) == 4
    flush(md_f.logger)
    lines = readlines(path)
    @test lines[1] == "# kT 2.5"
    @test lines[2] == "# step cv_1 bias offset height log_weight"
    rows = [parse.(Float64, split(l)) for l in lines[3:end]]
    @test length(rows) == 3
    @test [r[2] for r in rows] == [0.5, 0.52, 0.3]
    @test [r[end] for r in rows] ≈ metadynamics_log_weights(md_f.logger)
    @test read_metadynamics_logger(path).heights == md_f.memory.heights # Unitless, scalar CVs
    md_o = Metadynamics(ListHills((dist,), 1.0, 0.1); tempering=WellTempered(γ, T),
                        logger=MetadynamicsLogger(cv_points=pts, filepath=path, overwrite=true))
    add_hill!(md_o, 0.5)
    flush(md_o.logger)
    @test length(readlines(path)) == 3
    @test flush(lg) === lg

    # Read back: the records, the weights, and units with several CVs
    gh2 = GridHills((dist, dist), 1.0u"kJ * mol^-1", (0.1u"nm", 0.1u"nm"), (0.0u"nm", 0.0u"nm"),
                    (1.0u"nm", 1.0u"nm"), 51)
    md_r = Metadynamics(gh2; tempering=WellTempered(γ, T * u"K"),
                        logger=MetadynamicsLogger(filepath=path, overwrite=true))
    for s in ((0.5, 0.4), (0.52, 0.4), (0.3, 0.6))
        add_hill!(md_r, s .* u"nm")
    end
    flush(md_r.logger)
    lg_r = read_metadynamics_logger(path)
    @test values(lg_r) == values(md_r.logger)
    @test lg_r.steps == md_r.logger.steps && lg_r.kT == md_r.logger.kT
    @test lg_r.bias_energies == md_r.logger.bias_energies && lg_r.offsets == md_r.logger.offsets
    @test lg_r.heights == md_r.logger.heights
    @test metadynamics_weights(lg_r) ≈ metadynamics_weights(md_r.logger)

    # A failed deposit records nothing, so the records keep one length
    md_e = Metadynamics(GridHills((dist,), 1.0, 0.1, 0.0, 1.0, 11; out_of_grid=:error);
                        logger=MetadynamicsLogger(temperature=T))
    add_hill!(md_e, 0.5)
    @test_throws ArgumentError add_hill!(md_e, 2.0)
    @test length.((md_e.logger.steps, md_e.logger.offsets, md_e.logger.heights)) == (1, 1, 1)

    # Errors at construction: temperature units of the logger or the tempering, cv_points for
    #   GridHills, a bad path, the spec in use
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0, 0.1);
                    logger=MetadynamicsLogger(temperature=T * u"K", cv_points=pts))
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0, 0.1);
                                            tempering=WellTempered(γ, T * u"K"))
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0, 0.1);
                    tempering=WellTempered(γ, T), logger=MetadynamicsLogger(temperature=T * u"K",
                                                                           cv_points=pts))
    @test_throws ArgumentError Metadynamics(GridHills((dist,), 1.0, 0.1, 0.0, 1.0, 11);
                                            logger=MetadynamicsLogger(temperature=T, cv_points=pts))
    @test_throws SystemError Metadynamics(ListHills((dist,), 1.0, 0.1);
                    logger=MetadynamicsLogger(temperature=T, cv_points=pts, filepath="/nonexistent_dir/x.txt"))
    spec = MetadynamicsLogger(temperature=T, cv_points=pts)
    Metadynamics(ListHills((dist,), 1.0, 0.1); logger=spec)
    @test_throws ArgumentError metadynamics_weights(spec)

    # An existing empty file gets a header, without a warning
    rm(path)
    touch(path)
    md_1 = @test_logs Metadynamics(ListHills((dist,), 1.0, 0.1);
                        logger=MetadynamicsLogger(temperature=T, cv_points=pts, filepath=path))
    add_hill!(md_1, 0.5)
    flush(md_1.logger)
    @test count(startswith("# step"), readlines(path)) == 1

    # Another writer on the file throws at the next flush: a shared path, or one replaced
    md_2 = @test_logs (:warn,) Metadynamics(ListHills((dist,), 1.0, 0.1);
                        logger=MetadynamicsLogger(temperature=T, cv_points=pts, filepath=path))
    add_hill!(md_1, 0.6)
    flush(md_1.logger)
    add_hill!(md_2, 0.6)
    @test_throws ArgumentError flush(md_2.logger)
    Metadynamics(ListHills((dist,), 1.0, 0.1);
                 logger=MetadynamicsLogger(temperature=T, cv_points=pts, filepath=path, overwrite=true))
    add_hill!(md_1, 0.7)
    @test_throws ArgumentError flush(md_1.logger)
    rm(path)

    # By default a run is appended with its own header; the reader keeps runs apart
    md_a = Metadynamics(ListHills((dist,), 1.0, 0.1);
                        logger=MetadynamicsLogger(temperature=T, cv_points=pts, filepath=path))
    add_hill!(md_a, 0.5)
    flush(md_a.logger)
    md_b = @test_logs (:warn,) Metadynamics(ListHills((dist, dist), 1.0, (0.1, 0.1));
                        logger=MetadynamicsLogger(temperature=2T, filepath=path,
                                                  cv_points=Iterators.product(pts, pts)))
    add_hill!(md_b, (0.5, 0.6))
    add_hill!(md_b, (0.6, 0.6))
    flush(md_b.logger)
    @test_throws ArgumentError read_metadynamics_logger(path)
    @test values(read_metadynamics_logger(path; run=1)) == [0.5]
    lg_b = read_metadynamics_logger(path; run=2)
    @test values(lg_b) == [(0.5, 0.6), (0.6, 0.6)] && lg_b.kT == md_b.logger.kT
    @test_throws ArgumentError read_metadynamics_logger(path; run=3)
    # Rows before the first header, e.g. of a lost header, throw instead of being dropped
    write(path, "1 0.5 0.0 0.0 1.0 0.0\n" * read(path, String))
    @test_throws ArgumentError read_metadynamics_logger(path; run=1)
    rm(path)

    # bias.logger is not a spec: each bias needs its own MetadynamicsLogger
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0, 0.1); logger=md_a.logger)
    @test_throws ArgumentError Metadynamics(GridHills((dist,), 1.0, 0.1, 0.0, 1.0, 11);
                                            logger=md_a.logger)
    # Nor can one spec serve two walkers
    spec_w = MetadynamicsLogger(temperature=T, cv_points=pts)
    Metadynamics(ListHills((dist,), 1.0, 0.1); logger=spec_w)
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0, 0.1); logger=spec_w)

    # An affine temperature (°C) gets the same clear error
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0u"kJ * mol^-1", 0.1u"nm");
                                            tempering=WellTempered(γ, 25u"°C"))

    # 2-CV cv_points from Iterators.product
    lh_2d = ListHills((dist, dist), 1.0, (0.1, 0.1))
    md_2d = Metadynamics(lh_2d; tempering=WellTempered(γ, T),
                         logger=MetadynamicsLogger(cv_points=Iterators.product(pts, pts)))
    add_hill!(md_2d, (0.5, 0.5))
    add_hill!(md_2d, (0.6, 0.5))
    pts_2d = vec(collect(Iterators.product(pts, pts)))
    @test md_2d.logger.points.values ≈ [potential_energy(lh_2d, s) for s in pts_2d] .-
                                  [Molly.hill_value(lh_2d, s, (0.6, 0.5), lh_2d.heights[end])
                                   for s in pts_2d]

    # A 1-tuple or SVector CV into GridHills records like a scalar
    md_s = Metadynamics(GridHills((dist,), 1.0, 0.1, 0.0, 1.0, 11); logger=MetadynamicsLogger(temperature=T))
    add_hill!(md_s, (0.5,))
    add_hill!(md_s, SVector(0.6))
    @test values(md_s.logger) == [0.5, 0.6]

    # cv_points are checked at construction: empty, wrong shape, missing units
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0, 0.1);
                                            logger=MetadynamicsLogger(temperature=T, cv_points=Float64[]))
    @test_throws MethodError Metadynamics(ListHills((dist, dist), 1.0, (0.1, 0.1));
                                          logger=MetadynamicsLogger(temperature=T, cv_points=pts))
    @test_throws Unitful.DimensionError Metadynamics(ListHills((dist,), 1.0u"kJ * mol^-1", 0.1u"nm");
                    logger=MetadynamicsLogger(temperature=T * u"K", cv_points=pts))

    # Unit labels use plain exponents, which uparse reads back
    @test Molly.unit_label(1.0u"kJ * mol^-1") == " [kJ mol^-1]"

    # A large bias factor gives a → 1, the NoTempering offset
    md_big = Metadynamics(ListHills((dist,), 1.0, 0.1); tempering=WellTempered(1e6, T),
                          logger=MetadynamicsLogger(cv_points=pts))
    foreach(s -> add_hill!(md_big, s), (0.5, 0.52, 0.3))
    v_b = [potential_energy(ListHills((dist,), 1.0, 0.1, md_big.memory.centers[1:2],
                                      md_big.memory.heights[1:2]), s) for s in pts]
    @test md_big.logger.offsets[3] ≈ kT * log(mean(exp.(v_b ./ kT))) rtol=1e-4

    # NoTempering: c = kT ln⟨exp(V/kT)⟩ over cv_points
    md_n = Metadynamics(ListHills((dist,), 1.0, 0.1); logger=MetadynamicsLogger(temperature=T, cv_points=pts))
    foreach(s -> add_hill!(md_n, s), (0.5, 0.52, 0.3))
    v_n = [potential_energy(ListHills((dist,), 1.0, 0.1, md_n.memory.centers[1:2]), s) for s in pts]
    @test md_n.logger.offsets[3] ≈ kT * log(mean(exp.(v_n ./ kT)))

    # Tempering and logger take kT from the Boltzmann constant of the System
    md_k = Metadynamics(ListHills((dist,), 1.0, 0.1); tempering=WellTempered(γ, T),
                        logger=MetadynamicsLogger(cv_points=pts))
    sys_k = System(
        atoms=[Atom(mass=10.0, σ=0.3, ϵ=0.2) for _ in 1:2],
        coords=[SVector(1.0, 1.0, 1.0), SVector(1.5, 1.0, 1.0)],
        boundary=CubicBoundary(3.0), general_inters=(md_k,),
        force_units=NoUnits, energy_units=NoUnits, k=2.0,
    )
    forces(sys_k, find_neighbors(sys_k), 0)
    forces(sys_k, find_neighbors(sys_k), 1)
    @test md_k.logger.kT == 2.0 * T
    v_1 = potential_energy(ListHills((dist,), 1.0, 0.1, md_k.memory.centers[1:1]), 0.5)
    @test md_k.memory.heights[2] ≈ exp(-v_1 / (2.0 * T * (γ - 1)))
    # kT is fixed by the first deposit: a manual deposit with the default k throws
    @test_throws ArgumentError add_hill!(md_k, 0.5)
    @test md_k.logger.kT == 2.0 * T && md_k.n_deposits == 2 && length(md_k.logger.steps) == 2
    @test md_k.call_count == 0 # Checked before the pacing
    sys_d = System(atoms=sys_k.atoms, coords=sys_k.coords, boundary=sys_k.boundary,
                   general_inters=(md_k,), force_units=NoUnits, energy_units=NoUnits)
    pacing = (md_k.step_count, md_k.last_step_n, md_k.last_deposit_step)
    @test_throws ArgumentError forces(sys_d, find_neighbors(sys_d), 2)
    @test (md_k.step_count, md_k.last_step_n, md_k.last_deposit_step) == pacing

    # NoTempering needs kT and ListHills needs cv_points; no logger by default
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0, 0.1);
                                            logger=MetadynamicsLogger(cv_points=pts))
    @test_throws ArgumentError Metadynamics(ListHills((dist,), 1.0, 0.1);
                                            logger=MetadynamicsLogger(temperature=T))
    @test isnothing(Metadynamics(ListHills((dist,), 1.0, 0.1)).logger)
end

@testset "Metadynamics Monte Carlo" begin
    # Monte Carlo deposits on the configuration after each accept/reject step
    cv = CalcDist([1], [2], CalcSingleDist(), :wrap)
    bias = Metadynamics(ListHills(cv, 0.1, 0.05); n_steps=5)
    sys = System(
        atoms=[Atom(mass=10.0, σ=0.3, ϵ=0.2) for _ in 1:2],
        coords=[SVector(1.0, 1.0, 1.0), SVector(1.3, 1.0, 1.0)],
        boundary=CubicBoundary(3.0),
        specific_inter_lists=(InteractionList2Atoms([1], [2], [HarmonicBond(k=500.0, r0=0.3)]),),
        general_inters=(bias,), force_units=NoUnits, energy_units=NoUnits,
    )
    simulator = MetropolisMonteCarlo(temperature=300.0,
                                     trial_moves=Molly.random_uniform_translation!,
                                     trial_args=Dict(:shift_size => 0.02))
    simulate!(sys, simulator, 50; rng=MersenneTwister(1))
    # Windows open at steps 1, 5, 10, ..., 50
    @test bias.n_deposits == 11
    @test bias.last_deposit_step == 50
    # The last hill is on the final configuration of the chain
    @test bias.memory.centers[end] ≈ calculate_cv(cv, sys.coords, sys.atoms, sys.boundary)
end

@testset "Metadynamics step pacing in simulation" begin
    # A ForcesLogger recomputes forces at logged steps, which must not add hills
    atoms = [Atom(mass=10.0, σ=0.3, ϵ=0.2) for _ in 1:2]
    coords = [SVector(1.0, 1.0, 1.0), SVector(1.3, 1.0, 1.0)]
    boundary = CubicBoundary(3.0)
    cv = CalcDist([1], [2], CalcSingleDist(), :wrap)
    bias = Metadynamics(ListHills((cv,), 0.1, 0.05); n_steps=5)
    n_steps = 20
    sys = System(
        atoms=atoms, coords=coords, boundary=boundary,
        velocities=[SVector(0.0, 0.0, 0.0) for _ in 1:2],
        general_inters=(bias,), force_units=NoUnits, energy_units=NoUnits,
        loggers=(forces=ForcesLogger(Float64, 1),),
    )
    simulator = VelocityVerlet(dt=0.002)
    simulate!(sys, simulator, n_steps)
    # Hills at steps 0, 5, 10, 15 and 20
    @test bias.n_deposits == n_steps ÷ 5 + 1
    @test length(bias.memory.centers) == bias.n_deposits

    # A second simulate! call continues the schedule without init_step, hills at counted
    #   steps 25, 30, 35 and 40
    simulate!(sys, simulator, n_steps)
    @test bias.n_deposits == n_steps ÷ 5 + 1 + n_steps ÷ 5
    @test bias.last_deposit_step == n_steps
    # And with init_step, hills at counted steps 45, 50, 55 and 60
    simulate!(sys, simulator, n_steps; init_step=n_steps)
    @test bias.n_deposits == n_steps ÷ 5 + 1 + 2 * (n_steps ÷ 5)
    @test bias.last_deposit_step == 2 * n_steps

    # A run shorter than n_steps deposits at step 0
    bias_long = Metadynamics(ListHills((cv,), 0.1, 0.05); n_steps=1000)
    sys_long = System(
        atoms=atoms, coords=coords, boundary=boundary,
        velocities=[SVector(0.0, 0.0, 0.0) for _ in 1:2],
        general_inters=(bias_long,), force_units=NoUnits, energy_units=NoUnits,
    )
    simulate!(sys_long, simulator, 500)
    @test bias_long.n_deposits == 1
    @test bias_long.last_deposit_step == 0

    # Langevin deposits the start configuration at step 1, with or without a logger at step 0
    function langevin_deposits(chunks, interval; loggers=(), continue_steps=true)
        bias_l = Metadynamics(ListHills((cv,), 0.1, 0.05); n_steps=interval)
        sys_l = System(
            atoms=atoms, coords=coords, boundary=boundary,
            velocities=[SVector(0.0, 0.0, 0.0) for _ in 1:2],
            general_inters=(bias_l,), force_units=NoUnits, energy_units=NoUnits,
            loggers=loggers,
        )
        sim_l = Langevin(dt=0.002, temperature=300.0, friction=1.0)
        step_n = 0
        for n in chunks
            simulate!(sys_l, sim_l, n; init_step=(continue_steps ? step_n : 0))
            step_n += n
        end
        return bias_l.n_deposits
    end
    @test langevin_deposits((n_steps, n_steps), 5) == langevin_deposits((2 * n_steps,), 5)
    @test langevin_deposits((n_steps, n_steps), 5; continue_steps=false) ==
          langevin_deposits((2 * n_steps,), 5)
    @test langevin_deposits((n_steps, n_steps), 5) == 2 * n_steps ÷ 5 + 1
    @test langevin_deposits(fill(1, 10), 1) == 10
    @test langevin_deposits((n_steps,), 5) == n_steps ÷ 5 + 1
    @test langevin_deposits((n_steps,), 5; loggers=(forces=ForcesLogger(Float64, 1),)) ==
          n_steps ÷ 5 + 1

    # Minimizers report a history-dependent interaction, which they would change
    minimizer_sys(inters) = System(
        atoms=atoms, coords=coords, boundary=boundary,
        specific_inter_lists=(InteractionList2Atoms([1], [2], [HarmonicBond(k=500.0, r0=0.3)]),),
        general_inters=inters, force_units=NoUnits, energy_units=NoUnits,
    )
    minimizer = SteepestDescentMinimizer(tol=1000.0)
    @test_logs (:warn, r"minimization") match_mode=:any simulate!(
        minimizer_sys((Metadynamics(ListHills(cv, 0.1, 0.05)),)), minimizer; strictness=:warn)
    @test_throws ErrorException simulate!(
        minimizer_sys((Metadynamics(ListHills(cv, 0.1, 0.05)),)), minimizer; strictness=:error)
    @test_logs simulate!(minimizer_sys(()), minimizer; strictness=:warn)
end

@testset "Metadynamics periodic CVs" begin
    # Hills on a periodic CV, such as a torsion in (-π, π], wrap around ±π. The memory takes
    #   the periods of its CVs (cv_period) when it is built.
    tor = CalcTorsion([1, 2, 3, 4])
    dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    @test Molly.cv_period(tor) == 2π
    @test iszero(Molly.cv_period(dist))
    @test iszero(Molly.cv_period(CalcAngle([1, 2, 3])))

    hill = π - 0.05
    lh = ListHills((tor,), 1.0, 0.1)
    gh = GridHills((tor,), 1.0, 0.1, -π, π, 360)
    fd_gradient(mem, x; h=1e-7) = (potential_energy(mem, x + h) - potential_energy(mem, x - h)) / 2h
    for (mem, rtol) in ((lh, 1e-12), (gh, 1e-2))
        add_hill!(mem, hill)
        # The same point seen from both sides of the wrap
        for δ in (0.02, 0.07, 0.15)
            @test isapprox(potential_energy(mem, hill + δ), potential_energy(mem, hill + δ - 2π);
                           atol=1e-12)
        end
        # A hill just below π raises the bias just above -π as much as at the same
        #   distance on its other side
        @test potential_energy(mem, -π + 0.05) > 0.5
        @test isapprox(potential_energy(mem, -π + 0.05), potential_energy(mem, hill - 0.1); rtol=rtol)
        # Energy is continuous and the gradient matches finite differences across ±π
        @test isapprox(potential_energy(mem, π - 1e-9), potential_energy(mem, -π + 1e-9); atol=1e-6)
        for x in (π - 0.031, -π + 0.031, -π + 0.007)
            @test isapprox(Molly.bias_gradient(mem, x), fd_gradient(mem, x); rtol=1e-4, atol=1e-6)
        end
    end
    @test isapprox(Molly.bias_gradient(lh, π), fd_gradient(lh, π); rtol=1e-6)

    # The periods are fixed when the memory is built, one per CV
    @test ListHills((tor, dist), 1.0, (0.1, 0.1)).period == (2π, 0.0)
    @test ListHills((dist,), 1.0, 0.1).period == (0.0,)
    @test GridHills((tor, dist), 1.0, (0.1, 0.1), (-π, 0.0), (π, 1.0), 10).period == (2π, 0.0)
    # The grid of a periodic CV must span one period
    @test_throws ArgumentError GridHills((tor,), 1.0, 0.1, 0.0, 1.0, 10)
    # Walkers sharing a memory share its CVs and periods
    walkers_mem = ListHills((tor,), 1.0, 0.1)
    @test Metadynamics(walkers_mem).memory.period == (2π,)
    @test Metadynamics(walkers_mem).memory.cvs === (tor,)

    # One periodic and one open dimension
    gh2 = GridHills((tor, dist), 1.0, (0.1, 0.1), (-π, 0.0), (π, 1.0), (360, 101))
    add_hill!(gh2, (hill, 0.5))
    @test isapprox(potential_energy(gh2, (hill, 0.5)), 1.0; atol=1e-2)
    @test isapprox(potential_energy(gh2, (-π + 0.05, 0.5)), potential_energy(gh2, (hill - 0.1, 0.5));
                   rtol=1e-2)
    # Periodic CV values outside the grid are wrapped, open ones still throw
    @test isapprox(potential_energy(gh2, (hill + 2π, 0.5)), potential_energy(gh2, (hill, 0.5));
                   atol=1e-12)
    @test_throws ArgumentError potential_energy(gh2, (hill, 1.5); strictness=:error)

    # A hill wider than the period adds to every grid point exactly once
    gh_wide = GridHills((tor,), 1.0, 4π, -π, π, 10)
    add_hill!(gh_wide, 0.0)
    @test all(v -> exp(-0.5 * 0.25^2) - 1e-12 <= v <= 1.0, gh_wide.values)
end

@testset "Metadynamics Float32" begin
    # Float32 parameters and CV values give Float32 energies, gradients and deposits
    # The torsion makes the memories periodic, with the 2π period in Float32
    tor = CalcTorsion([1, 2, 3, 4])
    dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    for cv in (dist, tor)
        lh = ListHills((cv,), 1f0, 0.1f0)
        add_hill!(lh, 0.3f0)
        add_hill!(lh, 3.1f0)
        @test (@inferred potential_energy(lh, 0.35f0)) isa Float32
        @test (@inferred Molly.bias_gradient(lh, 0.35f0)) isa Float32
        @test (@inferred potential_energy(lh, -3.1f0)) isa Float32
    end
    @test ListHills((tor,), 1f0, 0.1f0).period isa Tuple{Float32}
    @test ListHills((tor, dist), 1f0, (0.1f0, 0.2f0)).period === (2f0π, 0f0)

    lh2 = ListHills((dist, dist), 1f0, (0.1f0, 0.2f0))
    add_hill!(lh2, (0.3f0, 0.4f0))
    @test (@inferred potential_energy(lh2, (0.35f0, 0.45f0))) isa Float32
    @test (@inferred Molly.bias_gradient(lh2, (0.35f0, 0.45f0))) isa Tuple{Float32, Float32}

    # Integer heights and centers become float, so tempered fractional deposits fit
    lh_int = ListHills((dist,), 1, 0.5, [1], [1])
    @test eltype(lh_int.heights) == Float64
    @test eltype(lh_int.centers) == Float64
    md_int_h = Metadynamics(lh_int; tempering=WellTempered(10.0, 2.5))
    add_hill!(md_int_h, 0.1)
    @test length(md_int_h.memory.heights) == 2

    # Hills are deposited into the memory passed in
    @test md_int_h.memory === lh_int
    @test length(lh_int.heights) == 2

    for cv in (dist, tor)
        gh = GridHills((cv,), 1f0, 0.1f0, -1f0π, 1f0π, 100)
        add_hill!(gh, 0.3f0)
        @test eltype(gh.values) == Float32
        @test (@inferred potential_energy(gh, 0.35f0)) isa Float32
        @test (@inferred Molly.bias_gradient(gh, 0.35f0)) isa Float32
    end
end

@testset "Metadynamics simulation" begin
    # Two atoms connected by a HarmonicBond, with well-tempered Metadynamics on their
    # distance r. The converged bias V estimates the free energy along r as
    # F = -V * γ / (γ - 1), which for a distance in 3D is F(r) = U(r) - 2kT ln(r) up to a
    # constant, the second term coming from the r^2 Jacobian of the angular coordinates.
    rng = Xoshiro(1)
    mass = 10.0
    r0 = 1.0
    k_bond = 500.0
    temp = 298.0
    boundary = CubicBoundary(10.0)
    kT = ustrip(u"u * nm^2 * ps^-2 * K^-1", Unitful.k) * temp
    γ = 10.0

    atoms = [Atom(mass=mass, σ=0.3, ϵ=0.2), Atom(mass=mass, σ=0.3, ϵ=0.2)]
    coords = [SVector(4.5, 5.0, 5.0), SVector(4.5 + r0, 5.0, 5.0)]
    velocities = [random_velocity(mass, temp; rng=rng) for i in 1:2]

    bond = HarmonicBond(k=k_bond, r0=r0)
    specific_inter_lists = (InteractionList2Atoms([1], [2], [bond]),)

    calc_dist = CalcDist([1], [2], CalcSingleDist(), :wrap)
    # n_steps paces deposits directly off forces! (called every step regardless
    # of simulator), so no external logger is needed to drive hill deposition
    bias = Metadynamics(GridHills((calc_dist,), 0.2, 0.02, 0.3, 1.7, 701);
                        n_steps=100, tempering=WellTempered(γ, temp),
                        logger=MetadynamicsLogger())

    # remove_CM_motion=false: with 2 atoms, removing the centre of mass velocity after each
    #   Andersen collision cools the bond to about 2/3 of temp
    simulator = VelocityVerlet(dt=0.002, coupling=AndersenThermostat(temp, 0.1),
                               remove_CM_motion=false)

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        specific_inter_lists=specific_inter_lists,
        general_inters=(bias,),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    n_steps = 500_000
    simulate!(sys, simulator, n_steps; rng=rng)

    @test bias.n_deposits == n_steps ÷ 100 + 1
    @test all(isfinite, bias.memory.values)
    # Far outside anything ever visited, at the grid edges, no bias has accumulated
    @test potential_energy(bias.memory, 0.3) < 1e-12
    @test potential_energy(bias.memory, 1.7) < 1e-12

    # Over r0 ± 3 thermal widths, the estimate matches F(r) within a fraction of kT after
    # removing the unknown constant. Without the Jacobian term, the RMS deviation is above
    # 0.2kT.
    σ_thermal = sqrt(kT / k_bond)
    rs = range(r0 - 3σ_thermal, r0 + 3σ_thermal; length=61)
    bond_pe(r) = potential_energy(bond, SVector(0.0, 0.0, 0.0), SVector(r, 0.0, 0.0), boundary)
    F_est = free_energy.(bias, rs)
    F_ref = @. bond_pe(rs) - 2kT * log(rs)
    dev = (F_est .- mean(F_est)) .- (F_ref .- mean(F_ref))
    @test sqrt(mean(abs2, dev)) < 0.17kT
    @test maximum(abs, dev) < 0.5kT

    # Reweighted deposit frames recover the unbiased distribution P(r) ∝ r² exp(-U(r)/kT)
    @test length(values(bias.logger)) == bias.n_deposits
    @test bias.logger.steps[1:2] == [0, 100]
    w = metadynamics_weights(bias.logger)
    r_s = values(bias.logger)
    r_fine = range(r0 - 6σ_thermal, r0 + 6σ_thermal; length=2001)
    p = @. r_fine^2 * exp(-bond_pe(r_fine) / kT)
    p ./= sum(p)
    mean_ref, mean_rw = sum(p .* r_fine), sum(w .* r_s)
    std_ref = sqrt(sum(p .* (r_fine .- mean_ref).^2))
    std_rw = sqrt(sum(w .* (r_s .- mean_rw).^2))
    @test abs(mean_rw - mean_ref) < 0.1σ_thermal
    @test abs(std_rw / std_ref - 1) < 0.1
    # c(t) balances early against late frames; with c = 0 the ratio is about 0.01
    @test 0.5 < sum(w[1:end÷2]) / sum(w[end÷2+1:end]) < 2
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
