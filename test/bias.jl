# test cv calculations, bias potentials and simulation with biased cv 

@testset "cv" begin

    # test calculate_cv(CalcDist, args...)
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    c3 = SVector(0.1, 1.0, 1.0)u"nm"
    c4 = SVector(1.8, 1.0, 1.0)u"nm"
    c5 = SVector(1.0, 1.2, 1.3)u"nm"
    c6 = SVector(0.8, 0.7, 0.9)u"nm"

    a1 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a2 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a3 = Atom(mass=20u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a4 = Atom(mass=5u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a5 = Atom(mass=10u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a6 = Atom(mass=15u"g/mol", charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")

    coords = [c1, c2, c3, c4, c5, c6]
    atoms = [a1, a2, a3, a4, a5, a6]
    boundary = CubicBoundary(2.0u"nm")

    atom_inds_1 = [1,2,3]
    atom_inds_2 = [4,5,6]
    coords_1 = coords[atom_inds_1]
    coords_2 = coords[atom_inds_2]
    atoms_1 = atoms[atom_inds_1]
    atoms_2 = atoms[atom_inds_2]

    # centre of mass
    @test isapprox(
        Molly.centre_of_mass(coords_1,atoms_1),
        SVector(0.625, 1.0, 1.0)u"nm";    
        atol=1e-9u"nm",
    )

    @test isapprox(
        Molly.centre_of_mass(coords_2,atoms_2),
        SVector(1.0333333333333334, 0.9166666666666666, 1.05)u"nm";    
        atol=1e-9u"nm", 
    )

    # com-com distance between two atom groups
    cd = CalcComDist(:closest)
    dist_cv = CalcDist(atom_inds_1,atom_inds_2,:wrap,cd)

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        0.4197386753154344u"nm";    
        atol=1e-9u"nm", 
    )

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        Molly.dist_between_groups(cd, coords_1, coords_2, boundary, atoms_1, atoms_2);    
        atol=1e-9u"nm", 
    )

    # minimum distance between two atom groups
    cd = CalcMinDist(:closest)
    dist_cv = CalcDist(atom_inds_1,atom_inds_2,:wrap,cd)

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        0.3u"nm";    
        atol=1e-9u"nm", 
    )

    cd = CalcMinDist(:raw)
    dist_cv = CalcDist(atom_inds_1,atom_inds_2,:wrap,cd)

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        0.36055512754639896u"nm";  
        atol=1e-9u"nm", 
    )

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        Molly.dist_between_groups(cd, coords_1, coords_2, boundary); 
        atol=1e-9u"nm", 
    )

    # maximum distance between two atom groups
    cd = CalcMaxDist(:closest)
    dist_cv = CalcDist(atom_inds_1,atom_inds_2,:wrap,cd)

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        0.9695359714832659u"nm";
        atol=1e-9u"nm", 
    )

    cd = CalcMaxDist(:raw)
    dist_cv = CalcDist(atom_inds_1,atom_inds_2,:wrap,cd)

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        1.7u"nm";
        atol=1e-9u"nm", 
    )

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        Molly.dist_between_groups(cd, coords_1, coords_2, boundary); 
        atol=1e-9u"nm", 
    )

    # distance between two atoms 
    cd = CalcSingleDist(:closest)   
    dist_cv = CalcDist([3],[4],:wrap,cd)

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        0.3u"nm";
        atol=1e-9u"nm", 
    )

    cd = CalcSingleDist(:raw)   
    dist_cv = CalcDist([3],[4],:wrap,cd)

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        1.7u"nm";
        atol=1e-9u"nm", 
    )

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        Molly.dist_between_groups(cd, [c3], [c4], boundary); 
        atol=1e-9u"nm", 
    )

    dist_cv = CalcDist([1],[2],:wrap,CalcSingleDist(:closest))

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary), 
        0.3u"nm";
        atol=1e-9u"nm"
    )

    dist_cv = CalcDist([3],[4],:wrap,CalcSingleDist(:closest))

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary), 
        0.3u"nm";
        atol=1e-9u"nm"
    )

    dist_cv = CalcDist([5],[6],:wrap,CalcSingleDist(:closest))

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        0.6708203932499369u"nm";
        atol=1e-9u"nm"
    )

    dist_cv = CalcDist([1],[2],:wrap,CalcSingleDist(:raw))

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary), 
        0.3u"nm";
        atol=1e-9u"nm"
    )

    dist_cv = CalcDist([3],[4],:wrap,CalcSingleDist(:raw))

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary), 
        1.7u"nm";
        atol=1e-9u"nm"
    )

    dist_cv = CalcDist([5],[6],:wrap,CalcSingleDist(:raw))

    @test isapprox(
        calculate_cv(dist_cv, coords, atoms, boundary),
        0.6708203932499369u"nm";
        atol=1e-9u"nm"
    )

    # test calculate_cv(CalcRMSD, args...)
    pdb_path = joinpath(data_dir, "1ssu.pdb")
    struc = read(pdb_path, BioStructures.PDBFormat)
    cm_1 = BioStructures.coordarray(struc[1], BioStructures.calphaselector)
    cm_2 = BioStructures.coordarray(struc[2], BioStructures.calphaselector)
    coords_1 = SVector{3, Float64}.(eachcol(cm_1)) / 10 * u"nm"
    coords_2 = SVector{3, Float64}.(eachcol(cm_2)) / 10 * u"nm"

    # RMSD of all atoms
    rmsd_cv = CalcRMSD(Int[],Int[],coords_2,:pdb)
    @test calculate_cv(rmsd_cv, coords_1) ≈ 2.54859467758795u"Å"

    # RMSD of subset of atoms
    n_atoms_subset = 20
    subset_inds = [i for i=1:n_atoms_subset]
    coords_1_subset = coords_1[1:n_atoms_subset]
    coords_2_subset = coords_2[1:n_atoms_subset]
    rmsd_cv = CalcRMSD(subset_inds, subset_inds, coords_2, :pdb)
    @test isapprox(
        calculate_cv(rmsd_cv, coords_1), 
        rmsd(coords_1_subset,coords_2_subset); 
        atol=1e-9u"nm"
    )

    # test calculate_cv(CalcRg, args...)
    bb_atoms = BioStructures.collectatoms(struc[1], BioStructures.backboneselector)
    coords = SVector{3, Float64}.(eachcol(BioStructures.coordarray(bb_atoms))) / 10 * u"nm"
    bb_to_mass = Dict("C" => 12.011u"g/mol", "N" => 14.007u"g/mol", "O" => 15.999u"g/mol")
    atoms = [Atom(mass=bb_to_mass[BioStructures.element(bb_atoms[i])]) for i in eachindex(bb_atoms)]

    # Rg of all atoms
    rg_cv = CalcRg(Int[],:pdb)
    @test isapprox(
        calculate_cv(rg_cv, coords, atoms), 
        11.51225678195222u"Å"; 
        atol=1e-6u"nm"
    )

    # Rg of subset of atoms
    n_atoms_subset = 20
    coords_subset = coords[1:n_atoms_subset]
    atoms_subset = atoms[1:n_atoms_subset]
    rg_cv = CalcRg([i for i=1:n_atoms_subset], :pdb)
    @test isapprox(
        calculate_cv(rg_cv, coords, atoms), 
        radius_gyration(coords_subset,atoms_subset); 
        atol=1e-6u"nm"
    )
end

@testset "bias_potentials" begin

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
        velocities=velocities
    )

    # test LinearBias
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
        bias_gradient(lb, cv_sim), 
        1500u"kJ * mol^-1 * nm^-1"; 
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 0.1u"nm"
    @test isapprox(
        bias_gradient(lb, cv_sim), 
        -1500u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    # test SquareBias
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
        bias_gradient(sb, cv_sim), 
        750u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 0.1u"nm"
    @test isapprox(
        bias_gradient(sb, cv_sim), 
        -1950u"kJ * mol^-1 * nm^-1"; 
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 0.75u"nm"
    @test isapprox(
        bias_gradient(sb, cv_sim), 
        0u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    # test FlatBottomBias
    fb = FlatBottomBias(3000u"kJ * mol^-1 * nm^-2", 0.5u"nm", 0.75u"nm")

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
        bias_gradient(fb, cv_sim), 
        750u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    cv_sim = 1u"nm"
    @test isapprox(
        bias_gradient(fb, cv_sim), 
        0u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    # test AtomsCalculators.potential_energy for dist between two atoms
    cd = CalcDist([1],[2],:wrap,CalcSingleDist(:closest))

    lb = LinearBias(7500u"kJ * mol^-1 * nm^-1", 0.5u"nm") 
    @test isapprox(
        AtomsCalculators.potential_energy(sys, BiasPotential(cd, lb)), 
        1500u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    sb = SquareBias(7500u"kJ * mol^-1 * nm^-2", 0.5u"nm") 
    @test isapprox(
        AtomsCalculators.potential_energy(sys, BiasPotential(cd, sb)), 
        150u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    fb = FlatBottomBias(7500u"kJ * mol^-1 * nm^-2", 0.15u"nm", 0.5u"nm") 
    @test isapprox(
        AtomsCalculators.potential_energy(sys, BiasPotential(cd, fb)), 
        9.375u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    # test AtomsCalculators.forces! for dist between two atoms
    cd = CalcDist([1],[2],:wrap,CalcSingleDist(:closest))
    lb = LinearBias(7500u"kJ * mol^-1 * nm^-1", 0.5u"nm") 

    fs = Molly.zero_forces(sys)
    AtomsCalculators.forces!(fs, sys, BiasPotential(cd, lb))
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
end

@testset "biased_simulation_unitless" begin

    # define wrapper functions
    function pair_dist_wrapper_12(sys, args...; kwargs...)
        coords_1 = Molly.from_device(sys.coords)[1]
        coords_2 = Molly.from_device(sys.coords)[2] 
        distances([coords_1,coords_2], sys.boundary)[2]
    end

    function pair_dist_wrapper_13(sys, args...; kwargs...)
        coords_1 = Molly.from_device(sys.coords)[1]
        coords_2 = Molly.from_device(sys.coords)[3]
        distances([coords_1,coords_2], sys.boundary)[2]
    end

    for AT in array_list

        # system setup
        n_atoms = 10
        boundary = CubicBoundary(10.0) 
        temp = 298.0
        atom_mass = 10.0

        atoms = to_device([Atom(mass=atom_mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms], AT)
        coords = to_device(place_atoms(n_atoms, boundary; min_dist=0.3), AT)
        velocities = to_device([random_velocity(atom_mass, temp) for i in 1:n_atoms], AT)

        pairwise_inters = (LennardJones(),)

        define_cv = CalcDist([1],[2],:wrap,CalcSingleDist(:closest)) # bias distance between atoms 1 and 2
        define_bias = SquareBias(400,1.5) # apply square bias potential with k=400 and target distance 1.5 nm
        general_inters = (BiasPotential(define_cv, define_bias),) # add to general interactions

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

        # simulate
        simulate!(sys, simulator, 200_000)

        # analyse
        pair_dists_12 = values(sys.loggers.pair_dist_12)
        pair_dists_13 =values(sys.loggers.pair_dist_13)

        dist_12_mean = mean(pair_dists_12[1000:end])
        dist_13_mean = mean(pair_dists_13[1000:end]) 
        dist_12_std = std(pair_dists_12[1000:100:end]) 
        dist_13_std = std(pair_dists_13[1000:100:end])

        # test
        @test isapprox(dist_12_mean, 1.5; atol=0.05,)
        @test !isapprox(dist_13_mean, 1.5; atol=0.05,)

        @test dist_13_mean > dist_12_mean
        @test dist_13_std > dist_12_std
    end
end

@testset "biased_simulation_unitful" begin

    # define wrapper functions
    function pair_dist_wrapper_12(sys, args...; kwargs...)
        coords_1 = Molly.from_device(sys.coords)[1]
        coords_2 = Molly.from_device(sys.coords)[2] 
        distances([coords_1,coords_2], sys.boundary)[2]
    end

    function pair_dist_wrapper_13(sys, args...; kwargs...)
        coords_1 = Molly.from_device(sys.coords)[1]
        coords_2 = Molly.from_device(sys.coords)[3]
        distances([coords_1,coords_2], sys.boundary)[2]
    end

    for AT in array_list

        # system setup
        n_atoms = 5
        boundary = CubicBoundary(10.0u"nm") 
        temp = 298.0u"K"
        atom_mass = 10.0u"g/mol"

        atoms = to_device([Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms], AT)
        coords = to_device(place_atoms(n_atoms, boundary; min_dist=0.3u"nm"), AT)
        velocities = to_device([random_velocity(atom_mass, temp) for i in 1:n_atoms], AT)

        pairwise_inters = (LennardJones(),)

        define_cv = CalcDist([1],[2],:wrap,CalcSingleDist(:closest)) # bias distance between atoms 1 and 2
        define_bias = SquareBias(400u"kJ * mol^-1 * nm^-2",1.5u"nm") # apply square bias potential with k=400 and target distance 1.5 nm
        general_inters = (BiasPotential(define_cv, define_bias),) # add to general interactions

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

        # simulate
        simulate!(sys, simulator, 200_000)

        # analyse
        pair_dists_12 = values(sys.loggers.pair_dist_12)
        pair_dists_13 =values(sys.loggers.pair_dist_13)

        dist_12_mean = mean(pair_dists_12[1000:end])
        dist_13_mean = mean(pair_dists_13[1000:end]) 
        dist_12_std = std(pair_dists_12[1000:100:end]) 
        dist_13_std = std(pair_dists_13[1000:100:end])

        # test
        @test isapprox(dist_12_mean, 1.5u"nm"; atol=0.05u"nm",)
        @test !isapprox(dist_13_mean, 1.5u"nm"; atol=0.05u"nm",)

        @test dist_13_mean > dist_12_mean
        @test dist_13_std > dist_12_std
    end
end