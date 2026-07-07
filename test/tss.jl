tss_step_wrapper(sys, neighbors, step_n, buffers; kwargs...) = step_n
tss_step_logger() = (step=GeneralObservableLogger(tss_step_wrapper, Int, 1),)

tss_logsumexp_for_test(xs) = maximum(xs) + log(sum(exp.(xs .- maximum(xs))))

function make_tss_thermo_states(; n_atoms=6, n_states=3)
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    thermo_states = ThermoState[]
    for lambda in range(1.0, 0.6; length=n_states)
        atoms = [Atom(mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1",
                      λ=lambda) for _ in 1:n_atoms]
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
            neighbor_finder=neighbor_finder,
        )
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(sys, intg; temperature=temp))
    end
    return thermo_states
end

function make_tss_npt_thermo_states(; n_atoms=4, n_states=3)
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"
    press = 1.0u"bar"
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    thermo_states = ThermoState[]
    for lambda in range(1.0, 0.7; length=n_states)
        atoms = [Atom(mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1",
                      λ=lambda) for _ in 1:n_atoms]
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
            neighbor_finder=neighbor_finder,
        )
        barostat = CRescaleBarostat(press, 1.0u"ps"; n_steps=10)
        intg = Langevin(dt=0.001u"ps", temperature=temp, friction=0.1u"ps^-1",
                        coupling=(barostat,))
        push!(thermo_states, ThermoState(sys, intg; temperature=temp, pressure=press))
    end
    return thermo_states
end

function make_tss_local_estimator_for_test(thermo_states;
                                           first_state::Int = 1,
                                           reuse_neighbors::Bool = true,
                                           kwargs...)
    state_space = Molly.ExtendedStateSpace(thermo_states; reuse_neighbors = reuse_neighbors)
    active_state = Molly.ActiveThermoState(state_space, first_state)
    return Molly.make_tss_local_estimator(
        state_space,
        active_state;
        kwargs...,
        require_active_state = true,
    )
end

function visit_control_free_energies_for_test(state; reference_state::Integer = 1)
    Molly.update_window_probabilities!(state)
    Molly.solve_windowed_visit_control!(state)
    visit_control_f = copy(state.coupling.visit_control_f)
    visit_control_f .-= visit_control_f[reference_state]
    return visit_control_f
end

function make_tss_pmf_thermo_states(; n_atoms=4, n_states=4)
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"

    return [
        begin
            atoms = [Atom(mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1",
                          λ=1.0) for _ in 1:n_atoms]
            restraint = HarmonicPositionRestraint(
                k=10.0u"kJ * mol^-1 * nm^-2",
                x0=SVector(center, 0.0, 0.0)u"nm",
            )
            sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                specific_inter_lists=(InteractionList1Atoms([1], [restraint]),),
            )
            intg = Langevin(dt=0.001u"ps", temperature=temp, friction=0.1u"ps^-1")
            ThermoState(sys, intg; temperature=temp)
        end
        for center in range(0.2, 0.8; length=n_states)
    ]
end

@testset "Times Square Sampling (TSS)" begin
    thermo_states = make_tss_thermo_states()

    @testset "generalized ensemble logger handling" begin
        logged_sys = System(first(thermo_states).system; loggers=tss_step_logger())
        logged_state = @test_logs (:warn, r"Generalized ensemble methods ignore") ThermoState(
            logged_sys,
            first(thermo_states).integrator;
            temperature=298.0u"K",
        )
        state_space = Molly.ExtendedStateSpace([logged_state])
        @test isempty(values(state_space.partition.master_sys.loggers))
        @test all(sys -> isempty(values(sys.loggers)), state_space.partition.λ_systems)
        active_state = Molly.ActiveThermoState(state_space)
        @test isempty(values(active_state.active_sys.loggers))

        explicit_active = Molly.ActiveThermoState(state_space; loggers=tss_step_logger())
        @test hasproperty(explicit_active.active_sys.loggers, :step)
    end

    @testset "TSSState does not accept loggers" begin
        @test_throws MethodError Molly.TSSState(
            thermo_states;
            loggers=tss_step_logger(),
        )
    end

    @testset "local estimator construction and update" begin
        state = make_tss_local_estimator_for_test(thermo_states;
            first_state=2,
            gamma=[2.0, 1.0, 1.0],
            initial_f=[10.0, 11.0, 12.0],
            ETA=2.0,
            dens_reg=1e-4,
        )

        @test state.active_state.active_idx == 2
        @test state.gamma ≈ [0.5, 0.25, 0.25]
        @test state.f == [0.0, 1.0, 2.0]
        @test state.density ≈ state.gamma
        @test_throws ArgumentError Molly.TSSState(thermo_states; first_state=0)
        @test_throws ArgumentError Molly.TSSState(thermo_states; gamma=[1.0, 0.0, 1.0])

        state.density .= [0.2, 0.3, 0.5]
        state.log_dens .= log.(state.density)
        state.reduced_pot .= [1.0, 2.0, 0.5]
        @. state.log_state_bias = state.f + state.log_dens
        Molly.conditional_state_weights!(
            state.weights,
            state.log_state_bias,
            state.reduced_pot,
            state.scratch,
        )
        @test sum(state.weights) ≈ 1.0
        @test all(>=(0), state.weights)

        state.weights .= [0.2, 0.5, 0.3]
        state.reduced_pot .= state.f .+ state.log_dens .- log.(state.weights)
        max_delta_f = Molly.update_tss_estimates!(state; visited_state=2)
        @test state.iteration == 1
        @test isfinite(max_delta_f)
        @test all(isfinite, state.f)
        @test all(isfinite, state.tilts)
        @test sum(state.density) ≈ 1.0
        @test_throws ArgumentError Molly.update_tss_estimates!(state; visited_state=0)
    end

    @testset "history forgetting" begin
        keep_recent = make_tss_local_estimator_for_test(thermo_states;
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.5, phi=1.2),
        )

        for step in 1:10
            keep_recent.weights .= fill(1 / 3, 3)
            keep_recent.reduced_pot .= [0.1 * step, -0.05 * step, 0.2]
            Molly.update_tss_estimates!(keep_recent; visited_state=mod1(step, 3))
        end

        @test keep_recent.iteration == 10
        @test 0 < Molly.tss_recent_count(keep_recent) < keep_recent.iteration
        @test all(isfinite, keep_recent.f)
        @test all(>(0), keep_recent.density)
        @test sum(keep_recent.density) ≈ 1.0
    end

    @testset "single-window simulation" begin
        tss_state = Molly.TSSState(thermo_states;
            first_state=1,
            gamma=[1.0, 1.0, 1.0],
            initial_f=[1.0, 2.0, 4.0],
            ETA=2.0,
            dens_reg=1e-4,
        )
        @test occursin("TSSState with 3 states", sprint(show, tss_state))

        sim = Molly.TSSSimulation(
            tss_state;
            n_md_steps=1,
            n_cycles=3,
            log_freq=1,
            loggers=tss_step_logger(),
        )
        @test occursin("TSSSimulation with 1 replica", sprint(show, sim))
        Molly.simulate!(sim; rng=MersenneTwister(1))

        estimator = Molly.active_tss_estimator(tss_state)
        @test tss_state.iteration == 3
        @test estimator.iteration == 3
        @test estimator.stats.iterations == [1, 2, 3]
        @test all(in(1:3), estimator.stats.active_state)
        @test all(in(1:3), estimator.stats.sampled_next_state)
        @test all(isfinite, estimator.stats.max_abs_delta_f)
        @test all(f -> length(f) == 3 && all(isfinite, f), estimator.stats.f_history)
        @test estimator.f[1] == 0.0
        @test sim.current_step == 3
        @test values(sim.replicas[1].active_state.active_sys.loggers.step) == collect(0:3)

        Molly.simulate!(sim; rng=MersenneTwister(2))
        @test sim.current_step == 6
        @test values(sim.replicas[1].active_state.active_sys.loggers.step) == collect(0:6)

        @test_throws ArgumentError Molly.TSSSimulation(tss_state; n_md_steps=0, n_cycles=1)
        @test_throws ArgumentError Molly.TSSSimulation(
            tss_state;
            n_md_steps=1,
            n_cycles=1,
            initial_step=-1,
        )
        @test_throws ArgumentError Molly.TSSSimulation(
            tss_state;
            n_md_steps=1,
            n_cycles=1,
            loggers=tss_step_logger(),
            replica_loggers=[tss_step_logger()],
        )

        resumed_state = Molly.TSSState(make_tss_thermo_states();
            first_state=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        resumed_sim = Molly.TSSSimulation(
            resumed_state;
            n_md_steps=1,
            n_cycles=1,
            initial_step=5,
            loggers=tss_step_logger(),
        )
        Molly.simulate!(resumed_sim; rng=MersenneTwister(3))
        @test resumed_sim.current_step == 6
        @test values(resumed_sim.replicas[1].active_state.active_sys.loggers.step) == [5, 6]
    end

    @testset "windowed graph, visit control, and CovDet" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        true_f = [0.0, 1.0, 3.0, 6.0]
        state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=2,
            first_window=2,
            initial_f=true_f,
            ETA=1.0,
            dens_reg=1e-4,
            visit_control_tolerance=1e-10,
        )

        @test [window.state_indices for window in graph4.windows] ==
              [[1], [1, 2], [2, 3], [3, 4], [4]]
        @test state.state_to_windows == [[1, 2], [2, 3], [3, 4], [4, 5]]
        @test Molly.other_window_for_state(state, 2) == 3
        @test state.coupling.converged
        @test state.coupling.max_abs_residual <= state.coupling.tolerance
        @test visit_control_free_energies_for_test(state) ≈ true_f atol=1e-8
        @test Molly.tss_free_energies(state) ≈ true_f
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=graph4, first_state=2, first_window=1)

        covdet_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            adaptive_gamma=:covdet,
            global_visit_control=false,
        )
        @test all(est -> !isnothing(est.adaptive_gamma), covdet_state.estimators)
        @test all(est -> sum(est.gamma) ≈ 1.0 && all(>(0), est.gamma),
                  covdet_state.estimators)

        estimator = covdet_state.estimators[3]
        u_by_state = [0.0, 1.0, 4.0, 9.0]
        for (eval_i, state_i) in enumerate(estimator.evaluation_state_indices)
            estimator.evaluation_reduced_pot[eval_i] = u_by_state[state_i]
        end
        covdet_values = Molly.tss_covdet_moment_values(estimator)
        @test size(covdet_values, 1) == length(estimator.state_indices)
        @test all(isfinite, covdet_values)
    end

    @testset "windowed simulation and replicas" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        sim = Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=4,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
            replica_loggers=[tss_step_logger(), tss_step_logger()],
        )
        Molly.simulate!(sim; rng=MersenneTwister(14), n_threads=1, replica_parallel=:serial)

        @test length(sim.replicas) == 2
        @test length(sim.replica_workspaces) == 2
        @test state.iteration == 4
        @test sum(est.iteration for est in state.estimators) == 8
        @test sum(state.window_update_counts) == 8
        @test length(state.stats.replica_indices) == 4
        @test all(==([1, 2]), state.stats.replica_indices)
        @test all(replica -> replica.active_state.active_idx in
                             state.windows[replica.active_window].state_indices,
                  sim.replicas)
        @test sim.current_step == 8
        @test all(
            values(replica.active_state.active_sys.loggers.step) == collect(0:8)
            for replica in sim.replicas
        )
        @test all(isfinite, Molly.tss_free_energies(state; visited_only=true))
        @test_throws ArgumentError Molly.simulate!(sim; replica_parallel=:invalid)

        rng_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        rng_sim = Molly.TSSSimulation(rng_state;
            n_md_steps=1,
            n_cycles=0,
            n_replicas=2,
            first_states=[1, 3],
        )
        replica_rngs = [MersenneTwister(101), MersenneTwister(202)]
        expected_rngs = deepcopy(replica_rngs)
        Molly.simulate!(
            rng_sim;
            replica_rngs=replica_rngs,
            n_threads=1,
            replica_parallel=:serial,
        )
        @test rand(rng_sim.replica_workspaces[1].rng, UInt) == rand(expected_rngs[1], UInt)
        @test rand(rng_sim.replica_workspaces[2].rng, UInt) == rand(expected_rngs[2], UInt)
        @test_throws ArgumentError Molly.simulate!(
            rng_sim;
            replica_rngs=[MersenneTwister(1)],
            n_threads=1,
            replica_parallel=:serial,
        )

        @test_throws ArgumentError Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=1,
            n_replicas=2,
            first_states=[1, 3],
            replica_loggers=[tss_step_logger(), (step=tss_step_logger().step, temp=TemperatureLogger(1))],
        )

        @test_throws ArgumentError Molly.TSSSimulation(
            state;
            n_md_steps=1,
            n_cycles=1,
            n_replicas=2,
            first_states=[1, 3],
            loggers=tss_step_logger(),
        )
    end

    @testset "frozen replay archive" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        frozen_f = [copy(est.f) for est in state.estimators]
        frozen_density = [copy(est.density) for est in state.estimators]
        frozen_log_dens = [copy(est.log_dens) for est in state.estimators]
        frozen_tilts = [copy(est.tilts) for est in state.estimators]
        archive_logger = Molly.TSSReplayLogger(1; include_weights=true)
        sim = Molly.TSSSimulation(state;
            n_md_steps=2,
            n_cycles=2,
            frozen=true,
            replay_logger=archive_logger,
            loggers=(
                coords=CoordinatesLogger(1),
                box=BoxLogger(1),
            ),
        )

        @test Molly.tss_replay_archive(sim) === archive_logger
        Molly.simulate!(sim; rng=MersenneTwister(15), n_threads=1, replica_parallel=:serial)

        frames = values(Molly.tss_replay_archive(sim))
        coords_frames = values(sim.replicas[1].active_state.active_sys.loggers.coords)
        box_frames = values(sim.replicas[1].active_state.active_sys.loggers.box)
        @test length(frames) == length(coords_frames) == length(box_frames)
        @test [frame.step for frame in frames] == collect(0:4)
        @test [frame.coordinates for frame in frames] == coords_frames
        @test [frame.box_matrix for frame in frames] == box_frames

        for (estimator, f, density, log_dens, tilts) in
                zip(state.estimators, frozen_f, frozen_density, frozen_log_dens, frozen_tilts)
            @test estimator.f == f
            @test estimator.density == density
            @test estimator.log_dens == log_dens
            @test estimator.tilts == tilts
        end

        for frame in frames
            estimator = state.estimators[frame.active_window]
            @test frame.active_state in frame.state_indices
            @test frame.state_indices == estimator.state_indices
            recomputed = tss_logsumexp_for_test(
                estimator.f .+ estimator.log_dens .- frame.reduced_potentials,
            )
            @test frame.log_den ≈ recomputed
            @test !isnothing(frame.weights)
            @test sum(frame.weights) ≈ 1.0
        end

        @test_throws ArgumentError Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=1,
            frozen=false,
            replay_logger=Molly.TSSReplayLogger(1),
        )

        npt_state = Molly.TSSState(make_tss_npt_thermo_states(n_states=4);
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        npt_sim = Molly.TSSSimulation(npt_state;
            n_md_steps=1,
            n_cycles=1,
            frozen=true,
            replay_logger=Molly.TSSReplayLogger(1),
        )
        Molly.simulate!(npt_sim; rng=MersenneTwister(16), n_threads=1, replica_parallel=:serial)
        for frame in values(Molly.tss_replay_archive(npt_sim))
            estimator = npt_state.estimators[frame.active_window]
            recomputed = similar(frame.reduced_potentials)
            Molly.reduced_potentials!(
                recomputed,
                estimator.state_space,
                frame.coordinates,
                frame.boundary,
                frame.state_indices,
            )
            @test frame.box_matrix == Molly.boxmatrix(frame.boundary)
            @test recomputed ≈ frame.reduced_potentials
        end
    end

    @testset "windowed jackknife uncertainty" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        true_f = [0.0, 1.0, 3.0, 6.0]
        window_offsets = [10.0, -2.0, 5.0, 8.0, -4.0]
        state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=2.0),
        )
        state.iteration = 4

        for (window_i, estimator) in enumerate(state.estimators)
            local_f = true_f[state.windows[window_i].state_indices] .+ window_offsets[window_i]
            estimator.f .= local_f
            estimator.tilts .= 1.0
            estimator.density .= estimator.gamma
            estimator.log_dens .= log.(estimator.density)

            history = estimator.history
            empty!(history.epochs)
            Molly.ensure_tss_epoch_bounds!(history, state.iteration)
            for epoch_index in 2:4
                epoch = Molly.TSSEpoch(epoch_index, Float64, length(estimator.f))
                epoch.count = 1
                epoch.f .= local_f
                epoch.tilts .= 1.0
                push!(history.epochs, epoch)
            end
        end

        jackknife = Molly.tss_free_energy_uncertainties(state)
        @test jackknife isa Molly.TSSJackknifeResult
        @test jackknife.free_energies ≈ true_f
        @test jackknife.epoch_indices == [2, 3, 4]
        @test jackknife.epoch_weights ≈ [0.25, 0.25, 0.5]
        @test size(jackknife.replicates) == (length(true_f), 3)
        @test all(replicate -> replicate ≈ true_f, eachcol(jackknife.replicates))

        state.estimators[2].history.epochs[1].f[2] += 0.5
        noisy_jackknife = Molly.tss_free_energy_uncertainties(state)
        @test all(isfinite, noisy_jackknife.standard_errors)
        @test any(>(0), noisy_jackknife.standard_errors[2:end])
    end

    @testset "PMF accumulators and reporting" begin
        acc = Molly.OnlinePMFAccumulator(((0.0,), (2.0,), (2,)); T=Float64)
        Molly.accumulate!(acc, 0.25, log(2.0))
        Molly.accumulate!(acc, 1.25, log(1.0))
        Molly.accumulate!(acc, 3.0, log(1.0))
        @test (acc.total_samples, acc.accepted_samples, acc.out_of_grid_samples) == (3, 2, 1)
        @test acc.counts == [1, 1]
        @test Molly.pmf(acc).p ≈ [2 / 3, 1 / 3]
        @test Molly.total_effective_samples(acc) ≈ 9 / 5
        @test_throws ArgumentError Molly.OnlinePMFAccumulator(((0.0,), (1.0,), (0,)); T=Float64)
        @test_throws DimensionMismatch Molly.accumulate!(acc, (0.5, 0.5), 0.0)

        grid = Molly.PMFGrid((0.0, 3.0, 3); T=Float64)
        sampled = Molly.SampledPMFDeconvolutionAccumulator(grid)
        for _ in 1:20
            Molly.accumulate_pmf_deconvolution!(sampled, (0.5,), [0.0, 0.0, 0.0])
        end
        Molly.accumulate_pmf_deconvolution!(sampled, (1.5,), [0.0, log(1000.0), 0.0])
        for _ in 1:19
            Molly.accumulate_pmf_deconvolution!(sampled, (1.5,), [0.0, 0.0, 0.0])
        end
        Molly.accumulate_pmf_deconvolution!(sampled, (2.5,), [0.0, 0.0, log(2.0)])

        quality = Molly.pmf_bin_quality(sampled; min_count=20, min_ess=5.0,
                                        max_weight_fraction=0.5)
        @test quality.counts == [20, 20, 1]
        @test quality.reliable == [true, false, false]
        @test quality.ess[1] ≈ 20.0
        @test quality.ess[2] < 5.0
        @test quality.maxfrac[2] > 0.5

        raw = Molly.pmf_result_from_sampled_deconvolution(sampled)
        reported = Molly.pmf_result_from_sampled_deconvolution(
            sampled;
            quality=quality,
            gauge_reliable_only=true,
            mask_unreliable=true,
        )
        @test raw.F[2] ≈ 0.0
        @test reported.F[1] ≈ 0.0
        @test isnan(reported.F[2]) && isnan(reported.F[3])
        @test reported.p ≈ raw.p
        @test_throws ArgumentError Molly.pmf_result_from_sampled_deconvolution(
            sampled;
            quality=Molly.pmf_bin_quality(sampled; min_count=1000),
            gauge_reliable_only=true,
        )
    end

    @testset "PMF deconvolution arithmetic and sampling" begin
        pmf_states = make_tss_pmf_thermo_states(n_states=3)
        tss_state = Molly.TSSState(pmf_states)
        coupling = (xi, state_i) -> 0.17 * xi[1]^2 - 0.31 * xi[1] * state_i +
                                    0.23 * state_i^2
        deconv = Molly.PMFDeconvolution(
            tss_state;
            grid=(0.0, 5.0, 5),
            cv=active_state -> (0.5,),
            coupling=coupling,
        )
        estimator = first(tss_state.estimators)
        estimator.f .= [0.8, -0.35, 0.2]
        estimator.density .= [0.2, 0.5, 0.3]
        estimator.log_dens .= log.(estimator.density)

        sample = Molly.collect_tss_pmf_deconvolution_sample(
            deconv,
            estimator,
            tss_state.active_state;
            window_offset=-0.7,
        )
        @test length(sample.value) == 1
        @test length(sample.log_bin_weights) == 5
        @test all(isfinite, sample.log_bin_weights)

        manual_weights = similar(sample.log_bin_weights)
        Molly.pmf_log_bin_weights!(
            manual_weights,
            deconv.backend.log_coupling_matrix,
            estimator.f .+ estimator.log_dens .+ 0.7;
            state_indices=estimator.state_indices,
        )
        @test sample.log_bin_weights ≈ manual_weights

        target_probability = [0.12, 0.27, 0.08, 0.33, 0.20]
        biased_probability = target_probability .* exp.(-sample.log_bin_weights)
        biased_probability ./= sum(biased_probability)
        sampled = Molly.SampledPMFDeconvolutionAccumulator(deconv.backend.grid)
        for (bin_i, center) in enumerate(0.5:1.0:4.5)
            Molly.accumulate_pmf_deconvolution!(
                sampled,
                (center,),
                sample.log_bin_weights;
                log_reweight=log(biased_probability[bin_i]),
            )
        end
        expected_F = -log.(target_probability)
        expected_F .-= minimum(expected_F)
        @test Molly.sampled_pmf_probability(sampled) ≈ target_probability atol=1e-12 rtol=1e-12
        @test Molly.pmf_result_from_sampled_deconvolution(sampled).F ≈ expected_F atol=1e-12 rtol=1e-12

        left = Molly.SampledPMFDeconvolutionAccumulator(deconv.backend.grid)
        right = Molly.SampledPMFDeconvolutionAccumulator(deconv.backend.grid)
        merged = Molly.SampledPMFDeconvolutionAccumulator(deconv.backend.grid)
        Molly.accumulate_pmf_deconvolution!(left, (0.5,), sample.log_bin_weights)
        Molly.accumulate_pmf_deconvolution!(right, (1.5,), sample.log_bin_weights)
        Molly.merge_pmf_deconvolution_accumulator!(merged, left)
        Molly.merge_pmf_deconvolution_accumulator!(merged, right)
        @test merged.counts == [1, 1, 0, 0, 0]
        @test merged.total_samples == 2

        @test_throws ArgumentError Molly.PMFDeconvolution(
            tss_state;
            grid=(0.0, 3.0, 3),
            cv=active_state -> (0.5,),
            coupling=coupling,
            free_energies=zeros(3),
        )
    end

    @testset "PMF deconvolution with TSS history and simulation" begin
        pmf_states = make_tss_pmf_thermo_states(n_states=3)
        tss_state = Molly.TSSState(
            pmf_states;
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.5, phi=2.0),
        )
        deconv = Molly.PMFDeconvolution(
            tss_state;
            grid=(0.0, 3.0, 3),
            cv=active_state -> (0.5,),
            coupling=(xi, state_i) -> 0.0,
        )
        old_sample = Molly.PMFDeconvolutionSample((0.5,), [log(1000.0), 0.0, 0.0], 0.0)
        new_sample = Molly.PMFDeconvolutionSample((1.5,), [0.0, 0.0, 0.0], 0.0)
        make_obs(sample) = Molly.WindowedTSSObservation(
            1, 1, 1, 1, 0.0, zeros(3), fill(1 / 3, 3), nothing, Any[sample],
        )

        Molly.accumulate_tss_pmf_deconvolution!(
            deconv,
            tss_state,
            [make_obs(old_sample)];
            history_time=1,
        )
        Molly.accumulate_tss_pmf_deconvolution!(
            deconv,
            tss_state,
            [make_obs(new_sample)];
            history_time=8,
        )
        Molly.drop_old_tss_pmf_deconvolution_epochs!(deconv, tss_state, 8)
        @test length(deconv.backend.epoch_accumulators) == 1
        @test Molly.pmf(deconv).p ≈ [0.0, 1.0, 0.0]

        torsion_states = ThermoState[]
        boundary = CubicBoundary(2.0u"nm")
        coords = place_atoms(4, boundary; min_dist=0.3u"nm")
        cv = CalcTorsion([1, 2, 3, 4], :pbc, true)
        for target in range(-0.5, 0.5; length=3)
            atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm",
                          ϵ=0.2u"kJ * mol^-1", λ=1.0) for _ in 1:4]
            bias = BiasPotential(cv, PeriodicFlatBottomBias(10.0u"kJ * mol^-1", 0.1, target))
            sys = System(atoms=atoms, coords=coords, boundary=boundary, general_inters=(bias,))
            intg = Langevin(dt=0.001u"ps", temperature=298.0u"K", friction=0.1u"ps^-1")
            push!(torsion_states, ThermoState(sys, intg; temperature=298.0u"K"))
        end
        auto_state = Molly.TSSState(torsion_states)
        auto_deconv = Molly.PMFDeconvolution(auto_state; grid=(-π, π, 4))
        auto_sample = Molly.collect_tss_pmf_deconvolution_sample(
            auto_deconv,
            first(auto_state.estimators),
            auto_state.active_state,
        )
        @test length(auto_sample.value) == 1
        @test eltype(auto_sample.value) == Float64

        sim_state = Molly.TSSState(pmf_states)
        sim_deconv = Molly.PMFDeconvolution(
            sim_state;
            grid=(0.0, 3.0, 3),
            cv=active_state -> (0.5,),
            coupling=(xi, state_i) -> 0.0,
        )
        sim = Molly.TSSSimulation(
            sim_state;
            n_md_steps=1,
            n_cycles=1,
            self_adjustment_steps=3,
            pmf=sim_deconv,
        )
        Molly.simulate!(sim; rng=MersenneTwister(2), n_threads=1,
                        replica_parallel=:serial, show_progress=false)
        @test sim_deconv.backend.accumulator.accepted_samples == 1
    end

    @testset "partitioned workspace and MBAR assembly" begin
        pmf_states = make_tss_pmf_thermo_states()
        cv = CalcDist([1], [2], CalcSingleDist(), :wrap)
        bias = BiasPotential(cv, SquareBias(10.0u"kJ * mol^-1 * nm^-2", 0.4u"nm"))
        biased = System(pmf_states[1].system; general_inters=(bias,))
        unbiased = System(pmf_states[1].system; general_inters=())
        intg = pmf_states[1].integrator
        hetero_states = [
            ThermoState(biased, intg; temperature=298.0u"K"),
            ThermoState(unbiased, intg; temperature=298.0u"K"),
        ]
        coords = hetero_states[1].system.coords
        boundary = hetero_states[1].system.boundary
        partitioned = Molly.PartitionedReducedPotentialWorkspace(hetero_states)
        energies = Molly.evaluate_energy_all!(partitioned.partition, coords, boundary)
        @test energies[1] ≈ potential_energy(hetero_states[1].system)
        @test energies[2] ≈ potential_energy(hetero_states[2].system)

        coords_k = [[copy(pmf_states[1].system.coords)], [copy(pmf_states[2].system.coords)]]
        boundaries_k = [[pmf_states[1].system.boundary], [pmf_states[2].system.boundary]]
        partitioned_inputs = Molly.assemble_mbar_inputs(
            coords_k,
            boundaries_k,
            pmf_states[1:2];
            target_state=pmf_states[4],
            shift=true,
        )
        full_inputs = Molly.assemble_mbar_inputs_full(
            coords_k,
            boundaries_k,
            pmf_states[1:2];
            target_state=pmf_states[4],
            shift=true,
        )
        @test partitioned_inputs.u ≈ full_inputs.u
        @test partitioned_inputs.u_target ≈ full_inputs.u_target
        @test partitioned_inputs.shifts ≈ full_inputs.shifts
        @test partitioned_inputs.win_of == full_inputs.win_of
    end
end
