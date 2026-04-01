@testset "ustrip coverage" begin
    atoms = [
        Atom(mass=12.0u"g/mol", charge=0.1, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1"),
        Atom(mass=14.0u"g/mol", charge=-0.1, σ=0.32u"nm", ϵ=0.25u"kJ * mol^-1"),
        Atom(mass=16.0u"g/mol", charge=0.0, σ=0.34u"nm", ϵ=0.3u"kJ * mol^-1"),
    ]
    atoms_data = [AtomData(element="O", atom_name="O", res_name="HOH") for _ in atoms]
    bonds = InteractionList2Atoms(Int32[1], Int32[2], [0])

    tri = TriclinicBoundary(
        SVector(2.0, 0.0, 0.0)u"nm",
        SVector(0.4, 1.8, 0.0)u"nm",
        SVector(0.2, 0.3, 1.7)u"nm",
    )
    tri_nounits = ustrip(tri)
    @test tri_nounits.basis_vectors[1][1] == 2.0

    buck = Buckingham(cutoff=DistanceCutoff(1.0u"nm"), weight_special=0.5)
    buck_nounits = ustrip(buck)
    @test buck_nounits.cutoff.dist_cutoff == 1.0
    @test buck_nounits.weight_special == 0.5

    mie = Mie(m=4, n=6, cutoff=DistanceCutoff(1.1u"nm"), weight_special=0.25)
    mie_nounits = ustrip(mie)
    @test mie_nounits.cutoff.dist_cutoff == 1.1
    @test mie_nounits.weight_special == 0.25

    soft = SoftSphere(cutoff=DistanceCutoff(0.9u"nm"))
    @test ustrip(soft).cutoff.dist_cutoff == 0.9

    grav = Gravity(cutoff=DistanceCutoff(1.2u"nm"))
    @test ustrip(grav).cutoff.dist_cutoff == 1.2

    dpd = DPDInteraction(
        a=25.0u"nm",
        γ=4.5u"ps",
        σ=3.0u"nm",
        r_c=1.0u"nm",
        dt=0.01u"ps",
    )
    dpd_nounits = ustrip(dpd)
    @test dpd_nounits.r_c == 1.0
    @test dpd_nounits.dt == 0.01

    cosine = CosineAngle(k=10.0u"kJ * mol^-1", θ0=0.5)
    @test ustrip(cosine).k == 10.0

    fene = FENEBond(k=30.0u"kJ * mol^-1 * nm^-2", r0=1.5u"nm", σ=0.4u"nm", ϵ=0.2u"kJ * mol^-1")
    @test ustrip(fene).r0 == 1.5

    urey = UreyBradley(kangle=20.0u"kJ * mol^-1", θ0=1.2, kbond=15.0u"kJ * mol^-1 * nm^-2", r0=0.3u"nm")
    @test ustrip(urey).r0 == 0.3

    morse = MorseBond(D=2.0u"kJ * mol^-1", a=3.0u"nm^-1", r0=0.2u"nm")
    @test ustrip(morse).a == 3.0

    rb = RBTorsion(
        1.0u"kJ * mol^-1",
        2.0u"kJ * mol^-1",
        3.0u"kJ * mol^-1",
        4.0u"kJ * mol^-1",
    )
    @test ustrip(rb).f4 == 4.0

    ljdc = LJDispersionCorrection(atoms[1:2], 1.0u"nm")
    @test ustrip(ljdc).factor ≈ ustrip(ljdc.factor)

    mb = MullerBrown()
    mb_nounits = ustrip(mb)
    @test mb_nounits.energy_units == NoUnits
    @test mb_nounits.force_units == NoUnits

    obc = ImplicitSolventOBC(atoms, atoms_data, bonds; kappa=0.2u"nm^-1")
    obc_nounits = ustrip(obc)
    @test obc_nounits.kappa == 0.2
    @test eltype(from_device(obc_nounits.offset_radii)) <: AbstractFloat

    gbn2 = ImplicitSolventGBN2(atoms, atoms_data, bonds; kappa=0.3u"nm^-1")
    gbn2_nounits = ustrip(gbn2)
    @test gbn2_nounits.kappa == 0.3
    @test eltype(from_device(gbn2_nounits.offset_radii)) <: AbstractFloat

    lb = LinearBias(100.0u"kJ * mol^-1 * nm^-1", 0.2u"nm")
    sb = SquareBias(200.0u"kJ * mol^-1 * nm^-2", 0.3u"nm")
    fb = FlatBottomSquareBias(200.0u"kJ * mol^-1 * nm^-2", 0.1u"nm", 0.3u"nm")
    pb = PeriodicFlatBottomBias(1000.0u"kJ * mol^-1", 0.1, 0.0)
    @test ustrip(lb).cv_target == 0.2
    @test ustrip(sb).k == 200.0
    @test ustrip(fb).r_fb == 0.1
    @test ustrip(pb).r_bf == 0.1

    ref_coords = [
        SVector(0.0, 0.0, 0.0)u"nm",
        SVector(0.2, 0.0, 0.0)u"nm",
        SVector(0.1, 0.15, 0.0)u"nm",
    ]
    cv_rmsd = CalcRMSD(ref_coords, [1, 2, 3], [1, 2, 3], :wrap)
    cv_rmsd_nounits = ustrip(cv_rmsd)
    @test eltype(cv_rmsd_nounits.ref_coords) <: SVector{3, Float64}

    bias = BiasPotential(cv_rmsd, sb)
    bias_nounits = ustrip(bias)
    @test bias_nounits.bias_type.k == 200.0
    @test eltype(bias_nounits.cv_type.ref_coords) <: SVector{3, Float64}

    dist_constraints = [DistanceConstraint(1, 2, 0.3u"nm")]
    sr = SHAKE_RATTLE(3, 1e-8u"nm", 1e-8u"nm^2/ps"; dist_constraints=dist_constraints)
    sr_nounits = ustrip(sr)
    @test sr_nounits.dist_tolerance == 1e-8
    @test first(from_device(sr_nounits.clusters12)).dist12 == 0.3
end

@testset "ustrip system with LJ dispersion correction" begin
    for AT in array_list
        atoms = to_device([Atom(mass=12.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:4], AT)
        boundary = CubicBoundary(2.5u"nm")
        coords = to_device([
            SVector(0.1, 0.1, 0.1)u"nm",
            SVector(0.6, 0.1, 0.1)u"nm",
            SVector(0.1, 0.6, 0.1)u"nm",
            SVector(0.6, 0.6, 0.1)u"nm",
        ], AT)
        velocities = to_device(fill(SVector(0.0, 0.0, 0.0)u"nm/ps", 4), AT)
        neighbor_finder = DistanceNeighborFinder(
            eligible=to_device(trues(4, 4), AT),
            n_steps=10,
            dist_cutoff=1.2u"nm",
        )

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm"), use_neighbors=true),),
            general_inters=(LJDispersionCorrection(atoms, 1.0u"nm"),),
            neighbor_finder=neighbor_finder,
        )

        sys_nounits = ustrip(sys)
        neighbors = find_neighbors(sys_nounits, sys_nounits.neighbor_finder; n_threads=1)
        @test sys_nounits.force_units == NoUnits
        @test sys_nounits.energy_units == NoUnits
        @test sys_nounits.general_inters[1] isa LJDispersionCorrection
        @test isfinite(potential_energy(sys_nounits, neighbors))
    end
end

@testset "ustrip composite system coverage" begin
    atoms = [Atom(mass=12.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:3]
    coords = [
        SVector(0.15, 0.15, 0.15)u"nm",
        SVector(0.45, 0.20, 0.18)u"nm",
        SVector(0.28, 0.52, 0.22)u"nm",
    ]
    boundary = TriclinicBoundary(
        SVector(1.5, 0.0, 0.0)u"nm",
        SVector(0.2, 1.4, 0.0)u"nm",
        SVector(0.1, 0.3, 1.3)u"nm",
    )
    velocities = fill(SVector(0.0, 0.0, 0.0)u"nm/ps", 3)
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(3, 3),
        n_steps=10,
        dist_cutoff=1.2u"nm",
    )
    angles = InteractionList3Atoms(
        Int32[1],
        Int32[2],
        Int32[3],
        [CosineAngle(k=10.0u"kJ * mol^-1", θ0=0.5)],
    )
    bias = BiasPotential(
        CalcDist([1], [2], CalcSingleDist(), :wrap),
        SquareBias(100.0u"kJ * mol^-1 * nm^-2", 0.4u"nm"),
    )
    cons = SHAKE_RATTLE(3, 1e-8u"nm", 1e-8u"nm^2/ps"; dist_constraints=[DistanceConstraint(1, 2, 0.3u"nm")])

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(Mie(m=4, n=6, cutoff=DistanceCutoff(1.0u"nm"), use_neighbors=true),),
        specific_inter_lists=(angles,),
        general_inters=(bias,),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
    )

    sys_nounits = ustrip(sys)
    neighbors = find_neighbors(sys_nounits, sys_nounits.neighbor_finder; n_threads=1)
    @test sys_nounits.boundary isa TriclinicBoundary
    @test sys_nounits.force_units == NoUnits
    @test sys_nounits.energy_units == NoUnits
    @test sys_nounits.constraints[1] isa SHAKE_RATTLE
    @test sys_nounits.specific_inter_lists[1].inters[1] isa CosineAngle
    @test sys_nounits.general_inters[1] isa BiasPotential
    @test isfinite(potential_energy(sys_nounits, neighbors))
end
