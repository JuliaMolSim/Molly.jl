tss_step_wrapper(sys, buffers, neighbors, step_n; kwargs...) = step_n
tss_step_logger() = (step=GeneralObservableLogger(tss_step_wrapper, Int, 1),)

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
end
