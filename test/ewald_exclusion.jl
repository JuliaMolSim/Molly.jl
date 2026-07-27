@testset "Ewald exclusion unit stripping and energy" begin
    atoms = [
        Atom(charge=-0.834),
        Atom(charge=0.417),
        Atom(charge=0.417),
    ]
    coords = [
        SVector(0.2, 0.2, 0.2)u"nm",
        SVector(0.29572, 0.2, 0.2)u"nm",
        SVector(0.176, 0.2927, 0.2)u"nm",
    ]
    boundary = CubicBoundary(2.5u"nm")
    velocities = fill(SVector(0.0, 0.0, 0.0)u"nm/ps", 3)
    data = Molly.EwaldExclusionData(1.0u"nm")
    exclusions = InteractionList2Atoms(
        Int32[1, 1, 2],
        Int32[2, 3, 3],
        fill(EwaldExclusion(), 3),
        fill("", 3),
        data,
    )

    @test ustrip(EwaldExclusion()) isa EwaldExclusion
    for input_data in (data, Molly.EwaldExclusionData(1.0))
        stripped = ustrip(input_data)
        @test stripped.dist_cutoff == 1.0
        @test stripped.dist_cutoff isa Float64
        @test stripped.error_tol == input_data.error_tol
        @test stripped.ϵr == input_data.ϵr
        @test stripped.α == ustrip(input_data.α)
        @test stripped.α isa Float64
        @test stripped.f_div_ϵr == ustrip(input_data.f_div_ϵr)
        @test stripped.f_div_ϵr isa Float64
        @test stripped.scheduler === input_data.scheduler
    end

    stripped_list = ustrip(exclusions)
    @test stripped_list.data isa Molly.EwaldExclusionData
    @test stripped_list.data.dist_cutoff == 1.0
    exclusions_cpu = from_device(exclusions)
    @test exclusions_cpu.data isa Molly.EwaldExclusionData
    @test exclusions_cpu.data.dist_cutoff == data.dist_cutoff
    exclusions_array = to_device(exclusions, Array)
    @test exclusions_array.data isa Molly.EwaldExclusionData
    @test exclusions_array.data.scheduler === data.scheduler
    for AT in array_list[2:end]
        exclusions_device = to_device(exclusions, AT)
        @test exclusions_device.data isa Molly.EwaldExclusionData
        @test exclusions_device.data.dist_cutoff == data.dist_cutoff
        @test from_device(exclusions_device).data isa Molly.EwaldExclusionData
    end

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        specific_inter_lists=(exclusions,),
    )
    sys_cpu = from_device(sys)
    @test sys_cpu.specific_inter_lists[1].data isa Molly.EwaldExclusionData
    @test sys_cpu.specific_inter_lists[1].data.dist_cutoff == data.dist_cutoff

    copied = System(sys_cpu; coords=copy(sys_cpu.coords))
    sys_nounits = ustrip(copied)
    @test sys_nounits.specific_inter_lists[1].data isa Molly.EwaldExclusionData
    @test sys_nounits.specific_inter_lists[1].data.dist_cutoff == 1.0
    @test isfinite(potential_energy(sys_nounits))

    bonds = InteractionList2Atoms(
        Int32[1],
        Int32[2],
        [HarmonicBond(k=100.0u"kJ/mol/nm^2", r0=0.1u"nm")],
    )
    ordinary_sys = ustrip(System(from_device(System(sys; specific_inter_lists=(bonds,)))))
    @test ordinary_sys.specific_inter_lists[1].data === nothing
    @test isfinite(potential_energy(ordinary_sys))
end

@testset "partitioned PME workspaces own mutable storage" begin
    boundary = CubicBoundary(2.5u"nm")
    coords = [
        SVector(0.2, 0.2, 0.2)u"nm",
        SVector(0.8, 0.5, 0.4)u"nm",
        SVector(1.4, 1.2, 0.9)u"nm",
    ]
    atoms = [
        Atom(mass=1.0u"g/mol", charge=0.8),
        Atom(mass=1.0u"g/mol", charge=-0.5),
        Atom(mass=1.0u"g/mol", charge=-0.3),
    ]
    cutoff = 1.0u"nm"
    system = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        pairwise_inters=(CoulombEwald(dist_cutoff=cutoff, use_neighbors=true),),
        general_inters=(PME(cutoff, atoms, boundary),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(3, 3),
            special=falses(3, 3),
            dist_cutoff=cutoff,
        ),
    )
    states = [
        ThermoState(
            deepcopy(system),
            VelocityVerlet(dt=0.001u"ps");
            temperature=temperature * u"K",
        )
        for temperature in (290.0, 300.0, 310.0)
    ]
    state_pmes = [
        only(filter(inter -> inter isa PME, state.system.general_inters))
        for state in states
    ]
    @test all(i -> all(j -> state_pmes[i].fft_plan !== state_pmes[j].fft_plan,
                       1:(i - 1)), eachindex(state_pmes))
    @test all(isfinite(potential_energy(state.system)) for state in states)

    workspaces = [Molly.PartitionedReducedPotentialWorkspace(states) for _ in 1:2]
    pmes = [
        only(filter(inter -> inter isa PME, workspace.partition.master_sys.general_inters))
        for workspace in workspaces
    ]
    @test pmes[1] !== pmes[2]
    @test pmes[1].charge_grid !== pmes[2].charge_grid
    @test pmes[1].grid_fractions !== pmes[2].grid_fractions
    @test pmes[1].bsplines_θ !== pmes[2].bsplines_θ
    @test pmes[1].fft_plan !== pmes[2].fft_plan
    @test pmes[1].bfft_plan !== pmes[2].bfft_plan

    translated = [
        coords .+ Ref(SVector(0.007i, 0.011i, 0.013i)u"nm")
        for i in 1:12
    ]
    coords_k = [translated[1:4], translated[5:8], translated[9:12]]
    boundaries_k = [fill(boundary, 4) for _ in states]
    direct = Molly.assemble_mbar_inputs_full(coords_k, boundaries_k, states)
    for _ in 1:3
        partitioned = Molly.assemble_mbar_inputs(coords_k, boundaries_k, states)
        @test all(isfinite, partitioned.u)
        @test partitioned.u ≈ direct.u rtol=1e-10 atol=1e-10
    end
end

@testset "Ewald component validation and scaled parameter injection" begin
    scheduler = Molly.EleScaledLambdaScheduler()
    boundary = CubicBoundary(2.5)
    atoms = [
        Atom(mass=1.0, charge=1.0, σ=0.3, ϵ=0.2,
             λ=0.75, alch_role=Molly.InsertRole),
        Atom(mass=1.0, charge=-1.0, σ=0.3, ϵ=0.2),
    ]
    pair = CoulombEwaldScaled(
        dist_cutoff=1.0,
        scheduler=scheduler,
        use_neighbors=true,
        weight_special=1.0,
    )
    exclusions = InteractionList2Atoms(
        Int32[1],
        Int32[2],
        [EwaldExclusion()],
        [""],
        Molly.EwaldExclusionData(1.0; scheduler=scheduler),
    )
    sys = System(
        atoms=atoms,
        coords=[SVector(0.2, 0.2, 0.2), SVector(0.8, 0.6, 0.4)],
        boundary=boundary,
        pairwise_inters=(pair,),
        specific_inter_lists=(exclusions,),
        general_inters=(PME(1.0, atoms, boundary; scheduler=scheduler),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(2, 2),
            special=falses(2, 2),
            dist_cutoff=1.0,
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    @test Molly.validate_ewald_components(sys; require_complete=true)
    params = Molly.extract_parameters(sys)
    params["inter_CE_weight_14"] = 0.5
    _, pairwise, _, _ = Molly.inject_gradients(sys, params)
    @test only(pairwise) isa CoulombEwaldScaled
    @test only(pairwise).scheduler == scheduler
    @test only(pairwise).weight_special == 0.5

    wrong_pme = PME(
        1.0,
        atoms,
        boundary;
        scheduler=Molly.DefaultLambdaScheduler(),
    )
    @test_throws ArgumentError Molly.validate_ewald_components(
        System(sys; general_inters=(wrong_pme,));
        require_complete=true,
    )

    ordinary = CoulombEwald(dist_cutoff=1.0, use_neighbors=true)
    @test_throws ArgumentError Molly.validate_ewald_components(
        System(sys; pairwise_inters=(ordinary,));
        require_complete=true,
    )
end
