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

    dexp = DoubleExponential(cutoff=DistanceCutoff(1.1u"nm"), weight_special=0.25)
    dexp_nounits = ustrip(dexp)
    @test dexp_nounits.cutoff.dist_cutoff == 1.1
    @test dexp_nounits.weight_special == 0.25

    dexp_softcore = DoubleExponentialSoftCore(
        cutoff=DistanceCutoff(1.1u"nm"),
        weight_special=0.25,
    )
    dexp_softcore_nounits = ustrip(dexp_softcore)
    @test dexp_softcore_nounits.cutoff.dist_cutoff == 1.1
    @test dexp_softcore_nounits.weight_special == 0.25
    @test dexp_softcore_nounits.scheduler == dexp_softcore.scheduler

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

    dist_constraint = DistanceConstraint(1, 2, 0.3u"nm")
    @test ustrip(dist_constraint).dist == 0.3

    angle_constraint = AngleConstraint(1, 2, 3, 1.8, 0.2u"nm", 0.25u"nm")
    angle_constraint_nounits = ustrip(angle_constraint)
    @test angle_constraint_nounits.dist_ij == 0.2
    @test angle_constraint_nounits.dist_jk == 0.25
    @test angle_constraint_nounits.dist_ik ≈ ustrip(angle_constraint.dist_ik)

    lincs = LINCS(
        masses=[12.0u"g/mol", 14.0u"g/mol"],
        dist_constraints=[dist_constraint],
        dist_tolerance=1e-8u"nm",
        vel_tolerance=1e-8u"nm^2/ps",
    )
    lincs_nounits = ustrip(lincs)
    @test lincs_nounits.dist_tolerance == 1e-8
    @test lincs_nounits.vel_tolerance == 1e-8
    @test first(lincs_nounits.dist_constraints).dist == 0.3
    @test first(from_device(lincs_nounits.clusters)).dist12 == 0.3
    @test first(lincs_nounits.lincs_data.lengths) == 0.3

    ljdc = LJDispersionCorrection(atoms[1:2], 1.0u"nm")
    ljdc_nounits = ustrip(ljdc)
    @test ljdc_nounits.factor_6 ≈ ustrip(ljdc.factor_6)
    @test ljdc_nounits.factor_12 ≈ ustrip(ljdc.factor_12)

    coul_scaled = CoulombScaled(
        cutoff=DistanceCutoff(1.0u"nm"),
        weight_special=0.5,
        coulomb_const=2.0u"kJ * mol^-1 * nm",
    )
    coul_scaled_nounits = ustrip(coul_scaled)
    @test coul_scaled_nounits.cutoff.dist_cutoff == 1.0
    @test coul_scaled_nounits.weight_special == 0.5
    @test coul_scaled_nounits.coulomb_const == 2.0
    @test coul_scaled_nounits.scheduler == coul_scaled.scheduler

    crf_scaled = CoulombReactionFieldScaled(
        dist_cutoff=1.2u"nm",
        weight_special=0.25,
        coulomb_const=3.0u"kJ * mol^-1 * nm",
    )
    crf_scaled_nounits = ustrip(crf_scaled)
    @test crf_scaled_nounits.dist_cutoff == 1.2
    @test crf_scaled_nounits.weight_special == 0.25
    @test crf_scaled_nounits.coulomb_const == 3.0
    @test crf_scaled_nounits.scheduler == crf_scaled.scheduler

    ewald = Ewald(1.0u"nm")
    ewald_nounits = ustrip(ewald)
    @test ewald_nounits.dist_cutoff == 1.0
    @test ewald_nounits.error_tol == ewald.error_tol
    @test ewald_nounits.excluded_pairs == ewald.excluded_pairs

    coul_ewald_scaled = CoulombEwaldScaled(
        dist_cutoff=1.4u"nm",
        error_tol=0.001,
        weight_special=0.75,
        coulomb_const=4.0u"kJ * mol^-1 * nm",
        approximate_erfc=false,
    )
    coul_ewald_scaled_nounits = ustrip(coul_ewald_scaled)
    @test coul_ewald_scaled_nounits.dist_cutoff == 1.4
    @test coul_ewald_scaled_nounits.error_tol == 0.001
    @test coul_ewald_scaled_nounits.weight_special == 0.75
    @test coul_ewald_scaled_nounits.coulomb_const == 4.0
    @test coul_ewald_scaled_nounits.α ≈ ustrip(coul_ewald_scaled.α)
    @test coul_ewald_scaled_nounits.scheduler == coul_ewald_scaled.scheduler
    @test coul_ewald_scaled_nounits.approximate_erfc == false

    cubic = CubicBoundary(2.0u"nm")
    pme = PME(1.0u"nm", atoms, cubic)
    pme_nounits = ustrip(pme)
    @test pme_nounits.dist_cutoff == 1.0
    @test pme_nounits.α == ustrip(pme.α)
    @test pme_nounits.mesh_dims == pme.mesh_dims
    @test size(pme_nounits.charge_grid) == size(pme.charge_grid)

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

@testset "ustrip system with PME electrostatics" begin
    atoms = [
        Atom(mass=12.0u"g/mol", charge=0.3, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1"),
        Atom(mass=14.0u"g/mol", charge=-0.2, σ=0.32u"nm", ϵ=0.25u"kJ * mol^-1"),
        Atom(mass=16.0u"g/mol", charge=-0.1, σ=0.34u"nm", ϵ=0.3u"kJ * mol^-1"),
    ]
    boundary = CubicBoundary(2.5u"nm")
    coords = [
        SVector(0.2, 0.2, 0.2)u"nm",
        SVector(0.8, 0.3, 0.4)u"nm",
        SVector(1.1, 0.9, 0.7)u"nm",
    ]
    velocities = fill(SVector(0.0, 0.0, 0.0)u"nm/ps", 3)
    rc = 1.0u"nm"

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(CoulombEwald(dist_cutoff=rc),),
        general_inters=(PME(rc, atoms, boundary),),
    )

    sys_nounits = ustrip(sys)
    @test sys_nounits.force_units == NoUnits
    @test sys_nounits.energy_units == NoUnits
    @test sys_nounits.general_inters[1] isa PME
    @test sys_nounits.general_inters[1].dist_cutoff == 1.0
    @test isfinite(potential_energy(sys_nounits))
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

@testset "ustrip logger coverage" begin
    temp_logger = TemperatureLogger(typeof(1.0u"K"), 5)
    push!(temp_logger.history, 300.0u"K")
    temp_logger_nounits = ustrip(temp_logger)
    @test temp_logger_nounits.history == [300.0]

    temp_logger_empty_nounits = ustrip(TemperatureLogger(typeof(1.0u"K"), 5))
    @test isempty(temp_logger_empty_nounits.history)
    @test eltype(temp_logger_empty_nounits.history) == Float64

    tcl = TimeCorrelationLogger(
        (args...; kwargs...) -> 1.0u"nm",
        (args...; kwargs...) -> 2.0u"nm",
        typeof(1.0u"nm"),
        typeof(2.0u"nm"),
        1,
        3,
    )
    push!(tcl.history_A, 1.0u"nm")
    push!(tcl.history_B, 2.0u"nm")
    tcl.sum_offset_products[1] = 2.0u"nm^2"
    tcl.n_timesteps = 1
    tcl.sum_A += 1.0u"nm"
    tcl.sum_B += 2.0u"nm"
    tcl.sum_sq_A += 1.0u"nm^2"
    tcl.sum_sq_B += 4.0u"nm^2"
    tcl_nounits = ustrip(tcl)
    @test collect(tcl_nounits.history_A) == [1.0]
    @test collect(tcl_nounits.history_B) == [2.0]
    @test tcl_nounits.sum_offset_products[1] == 2.0

    aol = AverageObservableLogger(
        (args...; kwargs...) -> 1.0u"kJ * mol^-1",
        typeof(1.0u"kJ * mol^-1"),
        5,
    )
    push!(aol.block_averages, 1.0u"kJ * mol^-1")
    push!(aol.current_block, 2.0u"kJ * mol^-1")
    aol_nounits = ustrip(aol)
    @test aol_nounits.block_averages == [1.0]
    @test aol_nounits.current_block == [2.0]

    awh_empty_nounits = ustrip(AWHEnsembleLogger(
        typeof(1.0u"nm"),
        typeof(1.0u"nm^3"),
        typeof(1.0u"kJ * mol^-1"),
        10,
    ))
    @test isempty(awh_empty_nounits.coords_history)
    @test eltype(awh_empty_nounits.coords_history) == Vector{SVector{3, Float64}}
    @test eltype(awh_empty_nounits.volume_history) == Float64
    @test eltype(awh_empty_nounits.potential_energy_history) == Float64
end

@testset "ustrip system preserves loggers" begin
    atoms = [Atom(mass=12.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:3]
    coords = [
        SVector(0.15, 0.15, 0.15)u"nm",
        SVector(0.45, 0.20, 0.18)u"nm",
        SVector(0.28, 0.52, 0.22)u"nm",
    ]
    boundary = CubicBoundary(1.5u"nm")
    velocities = fill(SVector(0.0, 0.0, 0.0)u"nm/ps", 3)

    temp_logger = TemperatureLogger(typeof(1.0u"K"), 5)
    push!(temp_logger.history, 300.0u"K")

    disp_logger = DisplacementsLogger(10, coords)

    mc_logger = MonteCarloLogger()
    mc_logger.n_trials = 1
    mc_logger.n_accept = 1
    push!(mc_logger.energy_rates, 0.5)
    push!(mc_logger.state_changed, true)

    awh_logger = AWHEnsembleLogger(
        typeof(1.0u"nm"),
        typeof(1.0u"nm^3"),
        typeof(1.0u"kJ * mol^-1"),
        10,
    )
    push!(awh_logger.active_idx_history, 1)
    push!(awh_logger.coords_history, [
        SVector(0.15, 0.15, 0.15)u"nm",
        SVector(0.45, 0.20, 0.18)u"nm",
        SVector(0.28, 0.52, 0.22)u"nm",
    ])
    push!(awh_logger.volume_history, 1.5u"nm^3")
    push!(awh_logger.potential_energy_history, 2.0u"kJ * mol^-1")

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
        loggers=(temp=temp_logger, disp=disp_logger, mc=mc_logger, awh=awh_logger),
    )

    sys_nounits = ustrip(sys)
    @test keys(sys_nounits.loggers) == (:temp, :disp, :mc, :awh)
    @test sys_nounits.loggers.temp.history == [300.0]
    @test eltype(sys_nounits.loggers.disp.coords_ref) <: SVector{3, Float64}
    @test sys_nounits.loggers.mc.energy_rates == [0.5]
    @test eltype(sys_nounits.loggers.awh.coords_history[1]) <: SVector{3, Float64}
    @test sys_nounits.loggers.awh.volume_history == [1.5]
    @test sys_nounits.loggers.awh.potential_energy_history == [2.0]
end
