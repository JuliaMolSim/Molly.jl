@testset "system device transfer reachability" begin
    atoms = [
        Atom(mass=12.0u"g/mol", charge=0.3, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1"),
        Atom(mass=14.0u"g/mol", charge=-0.2, σ=0.32u"nm", ϵ=0.25u"kJ * mol^-1"),
        Atom(mass=16.0u"g/mol", charge=-0.1, σ=0.34u"nm", ϵ=0.3u"kJ * mol^-1"),
    ]
    atoms_data = [AtomData(element="O", atom_name="O", res_name="HOH") for _ in atoms]
    coords = [
        SVector(0.2, 0.2, 0.2)u"nm",
        SVector(0.8, 0.3, 0.4)u"nm",
        SVector(1.1, 0.9, 0.7)u"nm",
    ]
    velocities = fill(SVector(0.0, 0.0, 0.0)u"nm/ps", 3)
    boundary = CubicBoundary(2.5u"nm")
    rc = 1.0u"nm"

    bonds = InteractionList2Atoms(
        Int32[1, 2],
        Int32[2, 3],
        [
            HarmonicBond(k=100.0u"kJ * mol^-1 * nm^-2", r0=0.3u"nm"),
            HarmonicBond(k=120.0u"kJ * mol^-1 * nm^-2", r0=0.31u"nm"),
        ],
        ["bond12", "bond23"],
    )
    dist_constraints = [DistanceConstraint(1, 2, 0.3u"nm")]
    shake = SHAKE_RATTLE(3, 1e-8u"nm", 1e-8u"nm^2/ps"; dist_constraints=dist_constraints)
    cv_rmsd = CalcRMSD(coords, [1, 2, 3], [1, 2, 3], :wrap)
    bias = BiasPotential(cv_rmsd, SquareBias(200.0u"kJ * mol^-1 * nm^-2", 0.3u"nm"))

    sys = System(
        atoms = atoms,
        atoms_data = atoms_data,
        coords = coords,
        boundary = boundary,
        velocities = velocities,
        pairwise_inters = (
            coul = CoulombEwald(dist_cutoff=rc, use_neighbors=true),
            lj = LennardJones(cutoff=DistanceCutoff(rc), use_neighbors=true),
            dexp = DoubleExponential(cutoff=DistanceCutoff(rc), use_neighbors=true),
            dexp_softcore = DoubleExponentialSoftCore(
                cutoff=DistanceCutoff(rc),
                use_neighbors=true,
            ),
        ),
        specific_inter_lists = (bonds = bonds,),
        general_inters = (
            pme = PME(rc, atoms, boundary),
            obc = ImplicitSolventOBC(atoms, atoms_data, bonds; kappa=0.2u"nm^-1"),
            gbn2 = ImplicitSolventGBN2(atoms, atoms_data, bonds; kappa=0.3u"nm^-1"),
            ljdc = LJDispersionCorrection(atoms, rc),
            bias = bias,
        ),
        constraints = (shake = shake,),
        neighbor_finder = DistanceNeighborFinder(
            eligible = trues(3, 3),
            dist_cutoff = rc,
            special = falses(3, 3),
            n_steps = 10,
        ),
        loggers = (
            disp = DisplacementsLogger(10, coords),
            mc = MonteCarloLogger(),
        ),
        launch_config = Molly.CUDALaunchConfig(force_block_y=2),
    )

    @test Molly.from_device(sys.general_inters.pme) isa PME
    @test Molly.from_device(sys.general_inters.obc) isa ImplicitSolventOBC
    @test Molly.from_device(sys.general_inters.gbn2) isa ImplicitSolventGBN2
    @test Molly.from_device(sys.constraints[1]) isa SHAKE_RATTLE
    @test Molly.from_device(sys.loggers.disp) isa DisplacementsLogger
    @test Molly.from_device(sys.neighbor_finder) isa DistanceNeighborFinder

    for AT in array_list
        sys_dev = Molly.to_device(sys, AT)
        sys_host = Molly.from_device(sys_dev)

        @test array_type(sys_dev) == AT
        @test sys_dev.pairwise_inters isa NamedTuple
        @test sys_dev.general_inters isa NamedTuple
        @test sys_dev.loggers isa NamedTuple
        @test sys_host.launch_config == sys.launch_config

        if AT <: GPUArrays.AbstractGPUArray
            @test sys_dev.general_inters.pme.charge_grid isa GPUArrays.AbstractGPUArray
            @test sys_dev.general_inters.obc.offset_radii isa GPUArrays.AbstractGPUArray
            @test sys_dev.general_inters.gbn2.offset_radii isa GPUArrays.AbstractGPUArray
            @test sys_dev.loggers.disp.coords_ref isa GPUArrays.AbstractGPUArray
            @test sys_dev.constraints[1].clusters12.k1 isa GPUArrays.AbstractGPUArray
        end

        @test sys_host.coords isa Array
        @test sys_host.general_inters.pme.charge_grid isa Array
        @test sys_host.general_inters.obc.offset_radii isa Array
        @test sys_host.general_inters.gbn2.offset_radii isa Array
        @test sys_host.loggers.disp.coords_ref isa Array
        @test sys_host.constraints[1].clusters12.k1 isa Vector{Int32}
        @test sys_host.neighbor_finder isa DistanceNeighborFinder
        @test sys_host.pairwise_inters.coul isa CoulombEwald
        @test sys_host.pairwise_inters.dexp isa DoubleExponential
        @test sys_host.pairwise_inters.dexp_softcore isa DoubleExponentialSoftCore
        @test sys_host.general_inters.ljdc isa LJDispersionCorrection
        @test sys_host.general_inters.bias isa BiasPotential
        @test sys_host.loggers.mc isa MonteCarloLogger
    end
end

function assert_roundtrip_realistic_system(sys; host_energy_atol, device_energy_atol, force_atol=nothing)
    energy_ref = potential_energy(sys; n_threads=1)
    forces_ref = isnothing(force_atol) ? nothing : from_device(forces(sys; n_threads=1))

    for AT in array_list
        sys_dev = Molly.to_device(sys, AT)
        sys_host = Molly.from_device(sys_dev)

        @test array_type(sys_dev) == AT
        @test sys_host.coords isa Array
        @test potential_energy(sys_host; n_threads=1) ≈ energy_ref atol=host_energy_atol

        if !isnothing(forces_ref)
            forces_host = from_device(forces(sys_host; n_threads=1))
            @test maximum(norm.(forces_host .- forces_ref)) < force_atol
        end

        if AT <: GPUArrays.AbstractGPUArray
            @test potential_energy(sys_dev; n_threads=1) ≈ energy_ref atol=device_energy_atol

            if !isnothing(forces_ref)
                forces_dev = from_device(forces(sys_dev; n_threads=1))
                @test maximum(norm.(forces_dev .- forces_ref)) < force_atol
            end
        end
    end
end

@testset "realistic system device transfer roundtrips" begin
    @testset "PME water constructor path" begin
        ff = MolecularForceField(Float32, joinpath(ff_dir, "tip3p_standard.xml"))
        sys = System(
            joinpath(data_dir, "water_3mol_cubic.pdb"),
            ff;
            array_type=Array,
            dist_cutoff=0.9f0u"nm",
            dist_buffer=0.0f0u"nm",
            nonbonded_method=:pme,
            dispersion_correction=false,
            center_coords=false,
            strictness=:nowarn,
        )
        sys = System(sys; loggers=(disp=DisplacementsLogger(10, sys.coords),))

        @test sys.general_inters[1] isa PME
        @test !isempty(sys.specific_inter_lists)
        @test sys.loggers.disp isa DisplacementsLogger

        assert_roundtrip_realistic_system(
            sys;
            host_energy_atol=1e-6u"kJ/mol",
            device_energy_atol=3e-4u"kJ/mol",
            force_atol=5e-4u"kJ * mol^-1 * nm^-1",
        )
    end

    @testset "Implicit solvent protein constructor path" begin
        ff = MolecularForceField(Float32, joinpath(ff_dir, "ff99SBildn.xml"); units=false)
        sys = System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            ff;
            units=false,
            array_type=Array,
            boundary=CubicBoundary(Float32(100.0)),
            dist_cutoff=Float32(5.0),
            nonbonded_method=:cutoff,
            implicit_solvent=:gbn2,
            kappa=Float32(0.7),
            dispersion_correction=false,
            strictness=:nowarn,
        )

        @test !isempty(sys.specific_inter_lists)
        @test sys.general_inters[1] isa ImplicitSolventGBN2
        @test sys.neighbor_finder isa Union{DistanceNeighborFinder, CellListMapNeighborFinder}

        assert_roundtrip_realistic_system(
            sys;
            host_energy_atol=1e-6,
            device_energy_atol=1e-2,
        )
    end
end
