simulation_step_wrapper(sys, neighbors, step_n, buffers; kwargs...) = step_n

mutable struct StepTrackingCoupler
    n_steps::Int
    history::Vector{Int}
end

StepTrackingCoupler(n_steps::Integer) = StepTrackingCoupler(Int(n_steps), Int[])

function Molly.apply_coupling!(sys, buffers, coupler::StepTrackingCoupler, sim, neighbors,
                               step_n; kwargs...)
    step_n % coupler.n_steps == 0 && push!(coupler.history, step_n)
    return false
end

@testset "Simulation continuation timing" begin
    atoms = [Atom(mass=10.0u"g/mol"), Atom(mass=12.0u"g/mol")]
    coords = [SVector(0.5, 0.5, 0.5)u"nm", SVector(1.0, 1.0, 1.0)u"nm"]
    velocities = [SVector(0.1, 0.0, 0.0)u"nm/ps", SVector(-0.1, 0.0, 0.0)u"nm/ps"]
    logger() = GeneralObservableLogger(simulation_step_wrapper, Int, 2)

    sys_continuous = System(
        atoms=atoms,
        coords=coords,
        velocities=velocities,
        boundary=CubicBoundary(2.0u"nm"),
        loggers=(step=logger(),),
    )
    sys_chunked = deepcopy(sys_continuous)
    coupler_continuous = StepTrackingCoupler(4)
    coupler_chunked = StepTrackingCoupler(4)
    sim_continuous = VelocityVerlet(
        dt=0.001u"ps",
        coupling=(coupler_continuous,),
        remove_CM_motion=0,
    )
    sim_chunked = VelocityVerlet(
        dt=0.001u"ps",
        coupling=(coupler_chunked,),
        remove_CM_motion=0,
    )

    simulate!(sys_continuous, sim_continuous, 10; n_threads=1)
    simulate!(sys_chunked, sim_chunked, 3; n_threads=1)
    simulate!(sys_chunked, sim_chunked, 3; n_threads=1, init_step=3,
              run_loggers=:skipstart)
    simulate!(sys_chunked, sim_chunked, 4; n_threads=1, init_step=6,
              run_loggers=:skipstart)

    @test sys_chunked.coords == sys_continuous.coords
    @test sys_chunked.velocities == sys_continuous.velocities
    @test coupler_chunked.history == coupler_continuous.history == [4, 8]
    @test values(sys_chunked.loggers.step) ==
          values(sys_continuous.loggers.step) == collect(0:2:10)
    @test_throws ArgumentError simulate!(sys_chunked, sim_chunked, 1; init_step=-1)
end

@testset "Temperature REMD" begin
    Random.seed!(1234)
    rng = Xoshiro(10)
    n_atoms = 100
    n_steps = 20_000
    atom_mass = 10.0u"g/mol"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng=rng)

    pairwise_inters = (LennardJones(use_neighbors=true),)

    eligible = trues(n_atoms, n_atoms)

    neighbor_finder = DistanceNeighborFinder(
        eligible=eligible,
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    
    # Define the unperturbed base system
    base_sys = System(
        atoms=atoms, 
        coords=coords, 
        boundary=boundary, 
        pairwise_inters=pairwise_inters, 
        neighbor_finder=neighbor_finder
    )

    n_replicas = 4
    temp_vals = [120.0u"K", 180.0u"K", 240.0u"K", 300.0u"K"]
    
    # Construct the array of thermodynamic states
    thermo_states = ThermoState[]
    for temp in temp_vals
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(base_sys, intg; temperature=temp))
    end

    replica_loggers = [
        (
            temp=TemperatureLogger(10),
            coords=CoordinatesLogger(10),
            step=GeneralObservableLogger(simulation_step_wrapper, Int, 10),
        )
        for i in 1:n_replicas
    ]

    # Initialize ReplicaSystem using the generalized constructor
    repsys = ReplicaSystem(
        thermo_states,
        [copy(coords) for _ in 1:n_replicas];
        replica_loggers=replica_loggers,
    )

    @test !is_on_gpu(repsys)
    @test float_type(repsys) == Float64
    @test masses(repsys) == fill(atom_mass, n_atoms)
    @test length(repsys) == n_atoms
    @test eachindex(repsys) == Base.OneTo(n_atoms)
    @test AtomsBase.mass(repsys, :) == fill(atom_mass, n_atoms)
    @test AtomsBase.mass(repsys, 5) == atom_mass
    @test AtomsBase.cell_vectors(repsys) == (
        SVector(2.0, 0.0, 0.0)u"nm",
        SVector(0.0, 2.0, 0.0)u"nm",
        SVector(0.0, 0.0, 2.0)u"nm",
    )
    show(devnull, repsys)

    # Use the unified simulator
    simulator = ReplicaExchangeMD(dt=0.005u"ps", exchange_time=2.5u"ps")

    @test_throws ArgumentError simulate!(repsys, simulator, n_steps; rng=rng)
    simulate!(repsys, simulator, n_steps; assign_velocities=true, n_threads=1)
    simulate!(repsys, simulator, n_steps; assign_velocities=false, n_threads=1)

    @test repsys.current_step == 2n_steps
    @test all(
        values(repsys.replica_loggers[id].step) == collect(0:10:(2n_steps))
        for id in 1:n_replicas
    )
    @test issorted(repsys.exchange_logger.steps)
    @test all(step -> 0 < step <= repsys.current_step, repsys.exchange_logger.steps)

    efficiency = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test efficiency > 0.16 # This is a fairly arbitrary threshold but it's a good test for very bad cases
    @test efficiency < 1.0 # Bad acceptance rate?

    for id in 1:n_replicas
        mean_temp = mean(values(repsys.replica_loggers[id].temp))
        # Given physical coordinates swap thermal states, they should average out across the ladder bounds
        @test (0.9 * temp_vals[1]) < mean_temp < (1.15 * temp_vals[end])
    end
end

@testset "Hamiltonian REMD" begin
    Random.seed!(1234)
    rng = Xoshiro(10)
    n_atoms = 100
    n_steps = 20_000
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng=rng)
    temp = 100.0u"K"

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_replicas = 4
    λ_vals = [1.0, 0.9, 0.75, 0.6]
    
    thermo_states = ThermoState[]
    for i in 1:n_replicas
        # Embed the lambda values directly into the atoms for this thermodynamic state
        atoms_λ = [Atom(mass=atom_mass, charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", 
                        λ =λ_vals[i]) for _ in 1:n_atoms]
        
        sys = System(
            atoms=atoms_λ,
            coords=coords,
            boundary=boundary,
            # SoftCore no longer takes λ; it relies on the atom's λ properties
            pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
            neighbor_finder=neighbor_finder
        )
        # All states share the exact same temperature and integrator parameters
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(sys, intg; temperature=temp))
    end

    replica_loggers = [(temp=TemperatureLogger(10), ) for i in 1:n_replicas]

    # Initialize generalized ReplicaSystem
    repsys = ReplicaSystem(
        thermo_states,
        [copy(coords) for _ in 1:n_replicas];
        replica_loggers=replica_loggers,
    )

    # Use the unified simulator (implicitly handles Hamiltonian REMD based on the ThermoStates)
    simulator = ReplicaExchangeMD(dt=0.005u"ps", exchange_time=2.5u"ps")

    @test_throws ArgumentError simulate!(repsys, simulator, n_steps; rng=rng)
    simulate!(repsys, simulator, n_steps; assign_velocities=true, n_threads=1)
    simulate!(repsys, simulator, n_steps; assign_velocities=false, n_threads=1)

    efficiency = repsys.exchange_logger.n_exchanges / repsys.exchange_logger.n_attempts
    @test efficiency > 0.08 # This is a fairly arbitrary threshold, but it's a good test for very bad cases
    @test efficiency < 1.0 # Bad acceptance rate?

    for id in 1:n_replicas
        mean_temp = mean(values(repsys.replica_loggers[id].temp))
        # Since temperature is constant across the ladder, physical replicas should hover exactly around temp
        @test (0.9 * temp) < mean_temp < (1.1 * temp)
    end
end

@testset "Metropolis Monte Carlo" begin
    n_atoms = 100
    n_steps = 10_000
    atom_mass = 10.0u"g/mol"
    atoms = [Atom(mass=atom_mass, charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(4.0u"nm", 4.0u"nm", 4.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 198.0u"K"

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        pairwise_inters=(Coulomb(use_neighbors=true), ),
        neighbor_finder=neighbor_finder,
        loggers=(
            coords=CoordinatesLogger(10),
            mcl=MonteCarloLogger(),
            avgpe=AverageObservableLogger(Molly.potential_energy_wrapper, typeof(atoms[1].ϵ), 10),
        ),
    )

    simulator_uniform = MetropolisMonteCarlo(
        temperature=temp,
        trial_moves=random_uniform_translation!,
        trial_args=Dict(:shift_size => 0.1u"nm"),
    )

    simulator_gaussian = MetropolisMonteCarlo(
        temperature=temp,
        trial_moves=random_normal_translation!,
        trial_args=Dict(:shift_size => 0.1u"nm"),
    )

    simulate!(sys, simulator_uniform , n_steps)
    simulate!(sys, simulator_gaussian, n_steps)

    acceptance_rate = sys.loggers.mcl.n_accept / sys.loggers.mcl.n_trials
    @test acceptance_rate > 0.05
    @test sys.loggers.avgpe.block_averages[end] < sys.loggers.avgpe.block_averages[1]

    distance_sum = 0.0u"nm"
    for i in eachindex(sys)
        ci = sys.coords[i]
        min_dist2 = Inf*u"nm^2"
        for j in eachindex(sys)
            if i == j
                continue
            end
            r2 = sum(abs2, vector(ci, sys.coords[j], sys.boundary))
            if r2 < min_dist2
                min_dist2 = r2
            end
        end
        distance_sum += sqrt(min_dist2)
    end
    mean_distance = distance_sum / length(sys)
    wigner_seitz_radius = cbrt(3 * volume(sys.boundary) / (4π * length(sys)))
    @test wigner_seitz_radius < mean_distance < 2 * wigner_seitz_radius
end

@testset "Crystals" begin
    r_cut = 0.85u"nm"
    a = 0.52468u"nm"
    atom_mass = 39.948u"g/mol"
    temp = 10.0u"K"

    fcc_crystal = SimpleCrystals.FCC(a, atom_mass, SVector(4, 4, 4))
    n_atoms = length(fcc_crystal)
    @test n_atoms == 256
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    sys = System(
        fcc_crystal;
        velocities=velocities,
        pairwise_inters=(LennardJones(cutoff=ShiftedForceCutoff(r_cut)),),
        loggers=(tot_eng=TotalEnergyLogger(100),),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )

    sys_cp = System(sys)
    @test sys_cp.atoms  == sys.atoms
    @test sys_cp.coords == sys.coords
    sys_mod = System(sys; coords=(sys.coords .* 0.5))
    @test sys_mod.atoms  == sys.atoms
    @test sys_mod.coords == sys.coords .* 0.5

    σ = 0.34u"nm"
    ϵ = (4.184 * 0.24037)u"kJ * mol^-1"
    updated_atoms = []

    for i in eachindex(sys)
        push!(updated_atoms, Atom(index=sys.atoms[i].index, atom_type=sys.atoms[i].atom_type,
                                  charge=sys.atoms[i].charge, mass=sys.atoms[i].mass,
                                  σ=σ, ϵ=ϵ))
    end

    sys = System(sys; atoms=[updated_atoms...])

    simulator = Langevin(
        dt=2.0u"fs",
        temperature=temp,
        friction=1.0u"ps^-1",
    )

    simulate!(sys, simulator, 25_000; run_loggers=false)
    simulate!(sys, simulator, 25_000)

    @test length(values(sys.loggers.tot_eng)) == 251
    @test -1800u"kJ * mol^-1" < mean(values(sys.loggers.tot_eng)) < -1600u"kJ * mol^-1"

    # Test unsupported crystals
    hex_crystal = SimpleCrystals.Hexagonal(a, :Ar, SVector(2, 2))
    @test_throws ArgumentError System(hex_crystal)

    # Make an invalid crystals (angle is too large)
    function MyInvalidCrystal(a, atomic_symbol::Symbol, N::SVector{3}; charge=0.0u"C")
        lattice = SimpleCrystals.BravaisLattice(
            SimpleCrystals.MonoclinicLattice(a, a, a, 120u"°"),
            SimpleCrystals.Primitive(),
        )
        z = zero(a)
        basis = [SimpleCrystals.Atom(atomic_symbol, [z, z, z], charge=charge)]
        return SimpleCrystals.Crystal(lattice, basis, N)
    end
    my_crystal = MyInvalidCrystal(a, :Ar, SVector(1, 1, 1))
    @test_throws ArgumentError System(my_crystal)
end

@testset "Different implementations" begin
    n_atoms = 400
    atom_mass = 10.0u"g/mol"
    v1 = SVector(5.0u"nm", 0.0u"nm", 0.0u"nm")
    v2 = SVector(2.0u"nm", 6.0u"nm", 0.0u"nm")
    v3 = SVector(3.0u"nm", 4.0u"nm", 7.0u"nm")
    boundary_cubic = CubicBoundary(6.0u"nm")
    boundary_triclinic = TriclinicBoundary(v1, v2, v3)
    temp = 1.0u"K"
    starting_coords_cubic = place_diatomics(n_atoms ÷ 2, boundary_cubic, 0.2u"nm"; min_dist=0.2u"nm")
    starting_coords_f32_cubic = [Float32.(c) for c in starting_coords_cubic]
    starting_coords_triclinic = place_diatomics(n_atoms ÷ 2, boundary_triclinic, 0.2u"nm"; min_dist=0.2u"nm")
    starting_coords_f32_triclinic = [Float32.(c) for c in starting_coords_triclinic]
    starting_velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]
    starting_velocities_f32 = [Float32.(c) for c in starting_velocities]

    function test_sim(nft, parallel::Bool, f32::Bool, ::Type{AT}, triclinic::Bool) where AT
        T = (f32 ? Float32 : Float64)
        n_atoms = 400
        n_steps = 200
        atom_mass = T(10.0)u"g/mol"
        boundary = triclinic ? TriclinicBoundary(T.(v1), T.(v2), T.(v3)) : CubicBoundary(T(6.0)u"nm")
        starting_coords = triclinic ? starting_coords_triclinic : starting_coords_cubic
        starting_coords_f32 = triclinic ? starting_coords_f32_triclinic : starting_coords_f32_cubic
        simulator = VelocityVerlet(dt=T(0.02)u"ps")
        k = T(10_000.0)u"kJ * mol^-1 * nm^-2"
        r0 = T(0.2)u"nm"
        bonds = [HarmonicBond(k=k, r0=r0) for i in 1:(n_atoms ÷ 2)]
        specific_inter_lists = (InteractionList2Atoms(
            to_device(Int32.(collect(1:2:n_atoms)), AT),
            to_device(Int32.(collect(2:2:n_atoms)), AT),
            to_device(bonds, AT),
        ),)
        cutoff = DistanceCutoff(T(1.0)u"nm")

        if nft == GPUNeighborFinder
            neighbor_finder = GPUNeighborFinder(
                n_atoms=n_atoms,
                dist_cutoff=T(1.0)u"nm",
                device_vector_type=AT{Int32, 1},
            )
        elseif nft == DistanceNeighborFinder
            neighbor_finder = DistanceNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                n_steps=10,
                dist_cutoff=T(1.5)u"nm",
            )
        else
            neighbor_finder = NoNeighborFinder()
        end
        pairwise_inters = (LennardJones(use_neighbors=(nft != NoNeighborFinder), cutoff=cutoff),)
        show(devnull, neighbor_finder)

        coords = to_device(copy(f32 ? starting_coords_f32 : starting_coords), AT)
        velocities = to_device(copy(f32 ? starting_velocities_f32 : starting_velocities), AT)
        atoms = to_device([Atom(charge=zero(T), mass=atom_mass, σ=T(0.2)u"nm",
                                ϵ=T(0.2)u"kJ * mol^-1", λ=one(T)) for i in 1:n_atoms], AT)

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            specific_inter_lists=specific_inter_lists,
            neighbor_finder=neighbor_finder,
        )

        @test is_on_gpu(sys) == (AT <: AbstractGPUArray)
        @test float_type(sys) == T

        n_threads = (parallel ? Threads.nthreads() : 1)
        E_start = potential_energy(sys; n_threads=n_threads)

        simulate!(sys, simulator, n_steps; n_threads=n_threads)
        return sys.coords, E_start
    end

    runs = [
        ("CPU"                , [NoNeighborFinder      , false, false, Array]),
        ("CPU f32"            , [NoNeighborFinder      , false, true , Array]),
        ("CPU NL"             , [DistanceNeighborFinder, false, false, Array]),
        ("CPU f32 NL"         , [DistanceNeighborFinder, false, true , Array]),
        ("CPU parallel"       , [NoNeighborFinder      , true , false, Array]),
        ("CPU parallel f32"   , [NoNeighborFinder      , true , true , Array]),
        ("CPU parallel NL"    , [DistanceNeighborFinder, true , false, Array]),
        ("CPU parallel f32 NL", [DistanceNeighborFinder, true , true , Array]),
    ]
    for AT in array_list[2:end]
        push!(runs, ("$AT"       , [NoNeighborFinder      , false, false, AT]))
        push!(runs, ("$AT f32"   , [NoNeighborFinder      , false, true , AT]))
        push!(runs, ("$AT NL"    , [DistanceNeighborFinder, false, false, AT]))
        push!(runs, ("$AT f32 NL", [DistanceNeighborFinder, false, true , AT]))
    end
    if run_cuda_tests
        AT = CuArray
        push!(runs, ("$AT GPU NL"    , [GPUNeighborFinder, false, false, AT]))
        push!(runs, ("$AT f32 GPU NL", [GPUNeighborFinder, false, true , AT]))
    end
    if run_metal_tests
        AT = MtlArray
        push!(runs, ("$AT f32"   , [NoNeighborFinder      , false, true , AT]))
        push!(runs, ("$AT f32 NL", [DistanceNeighborFinder, false, true , AT]))
    end

    # Check all simulations give the same result to within some error
    for triclinic in (false, true)
        final_coords_ref, E_start_ref = test_sim(runs[1][2]..., triclinic)
        for (name, args) in runs
            final_coords, E_start = test_sim(args..., triclinic)
            final_coords_f64 = [Float64.(c) for c in from_device(final_coords)]
            coord_diff = final_coords_f64 .- final_coords_ref
            coord_diff_size = sum(sum(map(x -> abs.(x), coord_diff))) / (3 * n_atoms)
            E_diff = abs(Float64(E_start) - E_start_ref)
            name = (triclinic ? "$name triclinic" : "$name cubic")
            @test coord_diff_size < 1e-4u"nm"
            @test E_diff < 5e-4u"kJ * mol^-1"
        end
    end
end

@testset "DPD simulation" begin
    n_atoms = 100
    n_steps = 10_000
    dt = 0.01
    r_c = 1.0
    box_size = 5.0
    density = n_atoms / box_size^3
    γ = 4.5
    kBT = 1.0
    σ = sqrt(2 * γ * kBT)
    a = 25.0

    boundary = CubicBoundary(box_size)
    rng = Xoshiro(12345)
    coords = [SVector{3}(rand(rng, 3) .* box_size) for _ in 1:n_atoms]
    velocities = [SVector{3}(randn(rng, 3)) for _ in 1:n_atoms]
    atoms = [Atom(index=i, mass=1.0, charge=0.0, σ=0.0, ϵ=0.0) for i in 1:n_atoms]

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(DPDInteraction(a=a, γ=γ, σ=σ, r_c=r_c, dt=dt, use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=1.5 * r_c,
        ),
        loggers=(
            temp=TemperatureLogger(Float64, 100),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
        k=1.0, # DPD uses reduced units where kB = 1; default_k(NoUnits) ≈ 0.008314
    )

    simulator = DPDVelocityVerlet(dt=dt, λ=0.65)
    simulate!(sys, simulator, n_steps; n_threads=1)

    temps = values(sys.loggers.temp)
    mean_temp = mean(temps[length(temps) ÷ 2 + 1:end])
    @test 0.5 < mean_temp < 1.5

    total_momentum = sum(sys.velocities .* mass.(atoms))
    @test all(abs.(total_momentum) .< 1.0)
end

@testset "MTSIntegrator" begin
    ff = MolecularForceField(joinpath(ff_dir, "tip4pfb.xml"))
    constraint_options = (
        (:none  , SetupSHAKE_RATTLE()),
        (:hbonds, SetupSHAKE_RATTLE()),
        (:hbonds, SetupLINCS()       ),
    )

    for AT in array_list
        for (constraints, constraint_algorithm) in constraint_options
            sys = System(
                joinpath(data_dir, "tip4pew.pdb"),
                ff;
                array_type=AT,
                float_type=Float64,
                constraints=constraints,
                constraint_algorithm=constraint_algorithm,
                nonbonded_method=SetupCoulombReactionField(),
                center_coords=false,
            )

            if constraints == :hbonds
                # Do not constrain angles, or there would be no specific interactions left
                cons_label = "cons"
                si_fractions = (4,)
            else
                cons_label = "nocons"
                si_fractions = (8, 4)
            end
            sim = MTSIntegrator(
                dt=1.0u"fs",
                pi_fractions=(1, 1),
                si_fractions=si_fractions,
                gi_fractions=(1,),
                remove_CM_motion=false,
            )

            forces_molly = from_device(forces(sys))
            openmm_forces_fp = joinpath(data_dir, "openmm_tip4pfb", "forces_$cons_label.txt")
            forces_openmm_vs = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
            forces_openmm = [(iszero(i % 4) ? zero(forces_openmm_vs[i]) : forces_openmm_vs[i])
                             for i in eachindex(sys)]
            @test maximum(norm.(forces_molly .- forces_openmm)) < 1e-6u"kJ * mol^-1 * nm^-1"

            E_molly = potential_energy(sys)
            openmm_E_fp = joinpath(data_dir, "openmm_tip4pfb", "energy_$cons_label.txt")
            E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
            @test abs(E_molly - E_openmm) < 1e-5u"kJ * mol^-1"

            n_steps = 10
            simulate!(sys, sim, n_steps)

            openmm_coords_fp = joinpath(data_dir, "openmm_tip4pfb",
                                        "coordinates_$(n_steps)steps_$cons_label.txt")
            openmm_vels_fp   = joinpath(data_dir, "openmm_tip4pfb",
                                        "velocities_$(n_steps)steps_$cons_label.txt" )
            coords_openmm = SVector{3}.(eachrow(readdlm(openmm_coords_fp)))u"nm"
            vels_openmm   = SVector{3}.(eachrow(readdlm(openmm_vels_fp)))u"nm * ps^-1"

            coords_diff = from_device(sys.coords) .- wrap_coords.(coords_openmm, (sys.boundary,))
            vels_diff = from_device(sys.velocities) .- vels_openmm
            @test maximum(norm.(coords_diff)) < 1e-3u"nm"
            @test maximum(norm.(vels_diff  )) < 0.1u"nm * ps^-1"

            temp = 300.0u"K"
            coupling = (CRescaleBarostat(1.0u"bar", 1.0u"fs"; max_scale_frac=0.01, n_steps=1),)
            sim_lang = MTSLangevinIntegrator(
                dt=1.0u"fs",
                temperature=temp,
                friction=10.0u"ps^-1",
                pi_fractions=(1, 1),
                si_fractions=si_fractions,
                gi_fractions=(1,),
                coupling=coupling,
                remove_CM_motion=false,
            )

            sys = System(
                sys;
                loggers=(
                    TemperatureLogger(5),
                    BoxLogger(5),
                ),
            )
            simulate!(sys, sim_lang, 600)

            @test 290u"K" < mean(values(sys.loggers[1])[81:end]) < 310u"K"
            @test 2.95u"nm" < mean(values(sys.loggers[2])[81:end])[1, 1] < 3.05u"nm"
        end

        # inner_step_neighbors recalculates the neighbors every inner step, giving correct
        #   dynamics with no neighbor list buffer even when the neighbor finder would not
        #   otherwise update the neighbors
        sys = System(
            joinpath(data_dir, "tip4pew.pdb"),
            ff;
            array_type=AT,
            float_type=Float64,
            nonbonded_method=SetupCoulombReactionField(),
            center_coords=false,
            dist_buffer=0.0u"nm",
        )
        nf = sys.neighbor_finder
        coords_start, velocities_start = copy(sys.coords), copy(sys.velocities)

        function run_mts(inner_step_neighbors, n_steps_neighbors)
            sys.coords .= coords_start
            sys.velocities .= velocities_start
            if nf isa GPUNeighborFinder
                nf.n_steps_reorder = n_steps_neighbors
            else
                nf.n_steps = n_steps_neighbors
            end
            sim = MTSIntegrator(
                dt=1.0u"fs",
                pi_fractions=(1, 1),
                si_fractions=(8, 4),
                gi_fractions=(1,),
                remove_CM_motion=false,
                inner_step_neighbors=inner_step_neighbors,
            )
            simulate!(sys, sim, 100)
            return copy(from_device(sys.coords))
        end

        # The neighbor finder cadence should make no difference when it is switched on
        coords_isn    = run_mts(true , 10^9)
        coords_isn_10 = run_mts(true , 10  )
        # With it switched off the neighbors are never updated after the first step
        coords_no_isn = run_mts(false, 10^9)
        @test maximum(norm.(coords_isn .- coords_isn_10)) < 1e-10u"nm"
        @test maximum(norm.(coords_isn .- coords_no_isn)) > 1e-3u"nm"
    end
end

@testset "Accelerated Weight Histogram (AWH)" begin
    n_atoms = 50
    n_steps = 2_000
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_windows = 4
    λ_vals = [1.0, 0.8, 0.6, 0.4]
    
    thermo_states = ThermoState[]
    for i in 1:n_windows
        # Embed the lambda values directly into the atoms
        atoms_λ = [Atom(mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", 
                        λ = λ_vals[i]) for _ in 1:n_atoms]
        
        # Define the system at this specific lambda state
        sys = System(
            atoms=atoms_λ,
            coords=coords,
            boundary=boundary,
            pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
            neighbor_finder=neighbor_finder,
        )
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(sys, intg; temperature=temp))
    end

    # Initialize AWH state using the newly generalized array of ThermoStates
    # n_bias is set low (10) to guarantee the initial stage is rapidly saturated 
    # and weight updates trigger during a short 2000 step test
    awh_state = AWHState(
        thermo_states;
        first_state=1,
        n_bias=10,
    )
    @test_throws MethodError AWHState(
        thermo_states;
        first_state=1,
        n_bias=10,
        loggers=(step=GeneralObservableLogger(simulation_step_wrapper, Int, 1),),
    )
    awh_state_show = sprint(show, awh_state)
    @test occursin("AWHState with 4 windows", awh_state_show)
    @test occursin("active window 1", awh_state_show)
    @test !occursin("scratch_potentials", awh_state_show)
    @test sprint(show, MIME"text/plain"(), awh_state) == awh_state_show

    awh_first_state = AWHState(thermo_states; first_state=3, n_bias=10)
    @test awh_first_state.active_idx == 3
    @test awh_first_state.active_intg === thermo_states[3].integrator
    @test awh_first_state.active_sys.pairwise_inters == thermo_states[3].system.pairwise_inters

    space = awh_state.state_space
    subset = [1, 3]
    full_energies = Molly.evaluate_energy_all!(space.partition, awh_state.active_sys.coords,
                                               awh_state.active_sys.boundary)
    subset_energies = Molly.evaluate_energy_subset(
        space.partition,
        awh_state.active_sys.coords,
        awh_state.active_sys.boundary,
        subset,
    )
    @test subset_energies[1] ≈ full_energies[1]
    @test subset_energies[2] ≈ full_energies[3]

    full_reduced = zeros(typeof(awh_state.N_bias), n_windows)
    subset_reduced = zeros(typeof(awh_state.N_bias), length(subset))
    Molly.reduced_potentials!(
        full_reduced,
        full_energies,
        space,
        awh_state.active_sys.boundary,
        Base.OneTo(n_windows),
    )
    Molly.reduced_potentials!(
        subset_reduced,
        subset_energies,
        space,
        awh_state.active_sys.boundary,
        subset,
    )
    @test subset_reduced ≈ full_reduced[subset]

    pressure_state = ThermoState(thermo_states[1].system, thermo_states[1].integrator;
                                 temperature=temp, pressure=1.0u"bar")
    pressure_space = Molly.ExtendedStateSpace([pressure_state])
    probe_energy = 1.25u"kJ * mol^-1"
    expected_reduced = pressure_space.betas[1] *
                       (ustrip(probe_energy) +
                        ustrip(pressure_space.pressures[1] * volume(boundary)))
    @test Molly.reduced_potential(pressure_space, probe_energy, boundary, 1) ≈ expected_reduced

    log_state_bias = [0.2, -0.4, 0.1]
    reduced = [1.0, 2.0, 0.5]
    weights = zeros(3)
    scratch = zeros(3)
    Molly.conditional_state_weights!(weights, log_state_bias, reduced, scratch)
    z = log_state_bias .- reduced
    log_den = maximum(z) + log(sum(exp.(z .- maximum(z))))
    @test weights ≈ exp.(z .- log_den)
    @test sum(weights) ≈ 1.0

    # Wrap in AWHSimulation
    awh_sim = AWHSimulation(
        awh_state;
        num_md_steps=10,
        update_freq=5,
        well_tempered_factor=10.0,
        coverage_threshold=1.0,
        log_freq=10,
        loggers=(step=GeneralObservableLogger(simulation_step_wrapper, Int, 1),),
    )
    awh_sim_show = sprint(show, awh_sim)
    @test occursin("AWHSimulation with 4 windows", awh_sim_show)
    @test occursin("PMF deconvolution disabled", awh_sim_show)
    @test !occursin("well_tempered_fac", awh_sim_show)
    @test sprint(show, MIME"text/plain"(), awh_sim) == awh_sim_show

    initial_f = copy(awh_sim.state.f)

    # Run the AWH simulation loop
    simulate!(awh_sim, n_steps)

    # Verification
    # 1. Active index must remain strictly within the bounds of the lambda ladder
    @test 1 <= awh_sim.state.active_idx <= n_windows
    
    # 2. Gibbs sampling should accumulate effective samples
    @test awh_sim.state.N_eff > 0
    
    # 3. The free energy array must update from its initial state
    @test awh_sim.state.f != initial_f
    
    # 4. AWH enforces a structural constraint where the first state acts as the reference (f = 0.0)
    @test awh_sim.state.f[1] == 0.0
    @test awh_sim.current_step == n_steps
    @test values(awh_sim.active_state.active_sys.loggers.step) == collect(0:n_steps)

    simulate!(awh_sim, 20)
    @test awh_sim.current_step == n_steps + 20
    @test values(awh_sim.active_state.active_sys.loggers.step) == collect(0:(n_steps + 20))
    @test_throws ArgumentError AWHSimulation(awh_state; initial_step=-1)
end

@testset "Agent-based modelling" begin
    Random.seed!(1234)

    @enum Status susceptible infected recovered

    # Custom atom type
    mutable struct Person
        i::Int
        status::Status
        mass::Float64
        σ::Float64
        ϵ::Float64
        λ::Float64
    end

    # Custom pairwise interaction
    struct SIRInteraction <: PairwiseInteraction
        dist_infection::Float64
        prob_infection::Float64
        prob_recovery::Float64
    end

    # Custom logger
    struct SIRLogger
        n_steps::Int
        fracs_sir::Vector{Vector{Float64}}
    end

    Base.values(logger::SIRLogger) = logger.fracs_sir

    # Custom force function
    function Molly.force(inter::SIRInteraction,
                            vec_ij,
                            atom_i,
                            atom_j,
                            args...)
        if (atom_i.status == infected && atom_j.status == susceptible) ||
                    (atom_i.status == susceptible && atom_j.status == infected)
            # Infect close people randomly
            r2 = sum(abs2, vec_ij)
            if r2 < inter.dist_infection^2 && rand() < inter.prob_infection
                atom_i.status = infected
                atom_j.status = infected
            end
        end
        # Workaround to obtain a self-interaction
        if atom_i.i == (atom_j.i - 1)
            # Recover randomly
            if atom_i.status == infected && rand() < inter.prob_recovery
                atom_i.status = recovered
            end
        end
        return zero(vec_ij)
    end

    # Test log_property! definition rather than just using GeneralObservableLogger
    function Molly.log_property!(logger::SIRLogger, sys, neighbors, step_n, buffers; kwargs...)
        if step_n % logger.n_steps == 0
            counts_sir = [
                count(p -> p.status == susceptible, sys.atoms),
                count(p -> p.status == infected   , sys.atoms),
                count(p -> p.status == recovered  , sys.atoms)
            ]
            push!(logger.fracs_sir, counts_sir ./ length(sys))
        end
    end

    rng = Xoshiro(15)
    temp = 1.0
    boundary = RectangularBoundary(10.0)
    n_steps = 1_000
    n_people = 500
    n_starting = 2
    atoms = [Person(i, i <= n_starting ? infected : susceptible, 1.0, 0.1, 0.02, 1.0) for i in 1:n_people]
    coords = place_atoms(n_people, boundary; min_dist=0.1, rng=rng)
    velocities = [random_velocity(1.0, temp; dims=2, rng=rng) for i in 1:n_people]

    lj = LennardJones(cutoff=DistanceCutoff(1.6), use_neighbors=true)
    sir = SIRInteraction(0.5, 0.06, 0.01)
    @test !use_neighbors(sir)
    pairwise_inters = (LennardJones=lj, SIR=sir)
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_people, n_people),
        n_steps=10,
        dist_cutoff=2.0,
    )
    simulator = VelocityVerlet(
        dt=0.02,
        coupling=(AndersenThermostat(temp, 5.0),),
    )

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=pairwise_inters,
        neighbor_finder=neighbor_finder,
        loggers=(
            coords=CoordinatesLogger(Float64, 10; dims=2),
            SIR=SIRLogger(10, []),
        ),
        force_units=NoUnits,
        energy_units=NoUnits,
        strictness=:nowarn,
    )

    simulate!(sys, simulator, n_steps; n_threads=1, rng=rng)

    s, i, r = values(sys.loggers.SIR)[end]
    @test s < 0.9
    @test i < 0.9
    @test r > 0.1
end
