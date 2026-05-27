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
    lambdas = range(1.0, 0.6; length=n_states)
    for lambda in lambdas
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

function tss_retained_epoch_count_for_test(state)
    history = getfield(state, :history)
    return isnothing(history) ? 0 : length(history.epochs)
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

    @testset "constructor validation" begin
        state = make_tss_local_estimator_for_test(thermo_states;
            first_state=2,
            gamma=[2.0, 1.0, 1.0],
            initial_f=[10.0, 11.0, 12.0],
            ETA=0.0,
            dens_reg=1e-4,
        )

        @test state.active_state.active_idx == 2
        @test state.gamma ≈ [0.5, 0.25, 0.25]
        @test state.log_gamma ≈ log.(state.gamma)
        @test state.f == [0.0, 1.0, 2.0]
        @test state.density ≈ state.gamma
        @test sum(state.density) ≈ 1.0
        @test isempty(state.stats.iterations)

        @test_throws ArgumentError Molly.TSSState(thermo_states; first_state=0)
        @test_throws ArgumentError Molly.TSSState(thermo_states; gamma=[1.0, 1.0])
        @test_throws ArgumentError Molly.TSSState(thermo_states; gamma=[1.0, 0.0, 1.0])
        @test_throws ArgumentError Molly.TSSState(thermo_states; gamma=[1.0, Inf, 1.0])
        @test_throws ArgumentError Molly.TSSState(thermo_states; initial_f=[0.0, 1.0])
        @test_throws ArgumentError Molly.TSSState(thermo_states; initial_f=[0.0, NaN, 1.0])
        @test_throws ArgumentError Molly.TSSState(thermo_states; ETA=-1.0)
        @test_throws ArgumentError Molly.TSSState(thermo_states; dens_reg=0.0)

        default_history = Molly.TSSHistoryForgetting()
        @test default_history.alpha == 0.19
        @test default_history.target_n_epochs == 16
        @test default_history.phi ≈ 0.19^(-1 / 16)

        history = Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2)
        history_state = make_tss_local_estimator_for_test(thermo_states; history_forgetting=history)
        @test history_state.history !== nothing
        @test history_state.history.config.alpha == 0.0
        @test history_state.history.config.phi == 1.2
        @test_throws ArgumentError Molly.TSSHistoryForgetting(alpha=-0.1)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(alpha=1.0)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(n_epochs=0)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(phi=1.0)

        @test_throws ArgumentError Molly.TSSState(thermo_states; adaptive_gamma=:covdet)
    end

    @testset "subset constructor validation and index mapping" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        state = make_tss_local_estimator_for_test(thermo_states4;
            first_state=2,
            state_indices=[2, 3],
        )

        @test state.state_indices == [2, 3]
        @test state.local_index_by_state == [0, 1, 2, 0]
        @test state.active_state.active_idx == 2
        @test Molly.tss_local_index(state, 2) == 1
        @test Molly.tss_local_index(state, 3) == 2
        @test Molly.tss_global_index(state, 1) == 2
        @test Molly.tss_global_index(state, 2) == 3
        @test_throws ArgumentError Molly.tss_local_index(state, 1)
        @test_throws ArgumentError Molly.tss_local_index(state, 4)
        @test_throws ArgumentError Molly.tss_global_index(state, 0)
        @test_throws ArgumentError Molly.tss_global_index(state, 3)

        @test state.gamma ≈ [0.5, 0.5]
        @test sum(state.gamma) ≈ 1.0
        @test state.density ≈ state.gamma
        @test length(state.f) == 2
        @test length(state.weights) == 2
        @test length(state.reduced_pot) == 2

        custom_state = make_tss_local_estimator_for_test(thermo_states4;
            first_state=3,
            state_indices=2:3,
            gamma=[2.0, 1.0],
            initial_f=[10.0, 12.0],
            ETA=0.0,
            dens_reg=1e-4,
        )
        @test custom_state.state_indices == [2, 3]
        @test custom_state.gamma ≈ [2 / 3, 1 / 3]
        @test custom_state.f == [0.0, 2.0]
        @test custom_state.density ≈ custom_state.gamma

        @test_throws ArgumentError make_tss_local_estimator_for_test(thermo_states4; first_state=1,
            state_indices=[2, 3])
        @test_throws ArgumentError make_tss_local_estimator_for_test(thermo_states4; first_state=2,
            state_indices=[2, 2])
        @test_throws ArgumentError make_tss_local_estimator_for_test(thermo_states4; first_state=2,
            state_indices=[2, 5])
        @test_throws ArgumentError make_tss_local_estimator_for_test(thermo_states4; first_state=2,
            state_indices=[2, 3], gamma=[1.0, 1.0, 1.0, 1.0])
    end

    @testset "sampling distribution update" begin
        state = make_tss_local_estimator_for_test(thermo_states;
            gamma=[2.0, 1.0, 1.0],
            ETA=2.0,
            dens_reg=0.1,
        )

        state.tilts .= [1.0, 2.0, 4.0]
        Molly.update_tss_sampling_distribution!(state)

        raw = state.gamma .* state.tilts .^ (-state.ETA)
        raw ./= sum(raw)
        expected = (1 - state.dens_reg) .* raw .+ state.dens_reg .* state.gamma
        expected ./= sum(expected)

        @test state.density ≈ expected
        @test state.log_dens ≈ log.(expected)
        @test sum(state.density) ≈ 1.0
        @test all(>(0), state.density)
        @test state.density[1] > state.gamma[1]
        @test state.density[3] < state.gamma[3]

        state.tilts .= [0.0, 1.0, 2.0]
        Molly.update_tss_sampling_distribution!(state)
        @test all(isfinite, state.density)
        @test all(>(0), state.density)
        @test sum(state.density) ≈ 1.0
        @test state.density[1] > state.density[2]

        state_eta_zero = make_tss_local_estimator_for_test(thermo_states;
            gamma=[2.0, 1.0, 1.0],
            ETA=0.0,
            dens_reg=0.1,
        )
        state_eta_zero.tilts .= [1.0, 100.0, 0.5]
        Molly.update_tss_sampling_distribution!(state_eta_zero)
        @test state_eta_zero.density ≈ state_eta_zero.gamma
    end

    @testset "conditional weights and one-step estimator update" begin
        state = make_tss_local_estimator_for_test(thermo_states;
            gamma=[1.0, 1.0, 1.0],
            initial_f=[0.0, 0.1, -0.2],
            ETA=2.0,
            dens_reg=1e-4,
        )

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

        z = state.f .+ log.(state.density) .- state.reduced_pot
        expected_weights = exp.(z .- (maximum(z) + log(sum(exp.(z .- maximum(z))))))
        @test state.weights ≈ expected_weights
        @test sum(state.weights) ≈ 1.0

        state.f .= [0.0, 0.1, -0.2]
        state.density .= fill(1 / 3, 3)
        state.log_dens .= log.(state.density)
        state.weights .= [0.2, 0.5, 0.3]
        state.reduced_pot .= state.f .+ state.log_dens .- log.(state.weights)
        old_f = copy(state.f)
        old_tilts = copy(state.tilts)

        log_terms = old_f .+ log.(state.density) .- state.reduced_pot
        log_den = maximum(log_terms) + log(sum(exp.(log_terms .- maximum(log_terms))))
        ratio = exp.(old_f .- state.reduced_pot .- log_den)
        gain = 1.0
        delta_f = -log1p.(gain .* (ratio .- 1.0))
        expected_f = old_f .+ delta_f
        expected_f .-= expected_f[1]
        expected_tilts = copy(old_tilts)
        for k in eachindex(expected_tilts)
            target = (k == 2 ? 1.0 : 0.0) / state.gamma[k]
            expected_tilts[k] += gain * (target - expected_tilts[k])
        end

        max_delta_f = Molly.update_tss_estimates!(state; visited_state=2)

        @test state.iteration == 1
        @test state.f ≈ expected_f
        @test state.tilts ≈ expected_tilts
        @test max_delta_f ≈ maximum(abs, delta_f .- delta_f[1])
        @test sum(state.density) ≈ 1.0
        @test all(>(0), state.density)
        @test_throws ArgumentError Molly.update_tss_estimates!(state; visited_state=0)
        @test_throws ArgumentError Molly.update_tss_estimates!(state; visited_state=4)
    end

    @testset "zero-weight estimator update remains finite" begin
        state = make_tss_local_estimator_for_test(thermo_states;
            gamma=[1.0, 1.0, 1.0],
            ETA=0.0,
            dens_reg=1e-4,
        )

        state.f .= 0.0
        state.density .= fill(1 / 3, 3)
        state.log_dens .= log.(state.density)
        state.reduced_pot .= [0.0, 1_000.0, 2_000.0]
        @. state.log_state_bias = state.f + state.log_dens
        Molly.conditional_state_weights!(
            state.weights,
            state.log_state_bias,
            state.reduced_pot,
            state.scratch,
        )

        @test any(iszero, state.weights)

        max_delta_f = Molly.update_tss_estimates!(state; visited_state=1)

        @test isfinite(max_delta_f)
        @test all(isfinite, state.f)
        @test state.f ≈ [0.0, 1_000.0, 2_000.0]
        @test state.iteration == 1
        @test sum(state.density) ≈ 1.0

        state.reduced_pot[2] = NaN
        @test_throws ArgumentError Molly.update_tss_estimates!(state; visited_state=1)
    end

    @testset "history forgetting local estimator" begin
        no_history = make_tss_local_estimator_for_test(thermo_states;
            gamma=[1.0, 1.0, 1.0],
            ETA=1.0,
            dens_reg=1e-4,
        )
        keep_all = make_tss_local_estimator_for_test(thermo_states;
            gamma=[1.0, 1.0, 1.0],
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )

        samples = (
            ([0.0, 0.3, 1.1], 1),
            ([0.2, -0.1, 0.8], 2),
            ([0.4, 0.5, -0.2], 3),
            ([0.1, 0.7, 0.3], 2),
        )
        for (reduced_pot, visited_state) in samples
            for state in (no_history, keep_all)
                state.weights .= fill(1 / 3, 3)
                state.reduced_pot .= reduced_pot
                Molly.update_tss_estimates!(state; visited_state=visited_state)
            end
        end

        @test keep_all.iteration == no_history.iteration
        @test Molly.tss_recent_count(keep_all) == keep_all.iteration
        @test tss_retained_epoch_count_for_test(keep_all) >= 1
        @test keep_all.f ≈ no_history.f
        @test keep_all.tilts ≈ no_history.tilts
        @test keep_all.density ≈ no_history.density

        forget_old = make_tss_local_estimator_for_test(thermo_states;
            gamma=[1.0, 1.0, 1.0],
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.5, phi=1.2),
        )
        for step in 1:10
            forget_old.weights .= fill(1 / 3, 3)
            forget_old.reduced_pot .= [0.1 * step, -0.05 * step, 0.2]
            Molly.update_tss_estimates!(forget_old; visited_state=mod1(step, 3))
        end

        @test Molly.tss_recent_count(forget_old) < forget_old.iteration
        @test Molly.tss_recent_count(forget_old) > 0
        @test tss_retained_epoch_count_for_test(forget_old) < forget_old.iteration
        @test all(isfinite, forget_old.f)
        @test all(isfinite, forget_old.tilts)
        @test all(isfinite, forget_old.density)
        @test all(>(0), forget_old.density)
        @test sum(forget_old.density) ≈ 1.0
        @test forget_old.f[1] == 0.0

        cutoff_config = Molly.TSSHistoryForgetting(alpha=0.19, phi=1.2)
        cutoff_history = Molly.TSSEpochHistory(cutoff_config, Float64, 2)
        cutoff_time = 95
        Molly._ensure_tss_epoch_bounds!(cutoff_history, cutoff_time)
        floor_cutoff = Molly._tss_epoch_index!(
            cutoff_history,
            floor(Int, cutoff_config.alpha * cutoff_time),
        )
        first_retained = Molly._tss_first_retained_epoch_index!(
            cutoff_history,
            cutoff_time,
        )
        @test cutoff_history.taus[first_retained] >= cutoff_config.alpha * cutoff_time
        @test cutoff_history.taus[first_retained - 1] < cutoff_config.alpha * cutoff_time
        @test first_retained > floor_cutoff

        for epoch_index in (floor_cutoff, first_retained)
            epoch = Molly.TSSEpoch(epoch_index, Float64, 2)
            epoch.count = 1
            push!(cutoff_history.epochs, epoch)
        end
        Molly._drop_old_tss_epochs!(cutoff_history, cutoff_time)
        @test [epoch.index for epoch in cutoff_history.epochs] == [first_retained]
    end

    @testset "subset sample processing and estimator update" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        state = make_tss_local_estimator_for_test(thermo_states4;
            first_state=2,
            state_indices=[2, 3],
            gamma=[1.0, 1.0],
            initial_f=[0.0, 0.1],
            ETA=1.0,
            dens_reg=1e-4,
        )

        weights = Molly.process_tss_sample!(state)
        @test weights === state.weights
        @test length(weights) == 2
        @test sum(weights) ≈ 1.0
        @test all(>=(0), weights)
        @test all(isfinite, state.reduced_pot)

        rng = MersenneTwister(1)
        samples = [Molly.tss_sample_global_state(rng, state) for _ in 1:50]
        @test all(s -> s in (2, 3), samples)

        state.f .= [0.0, 0.1]
        state.density .= fill(0.5, 2)
        state.log_dens .= log.(state.density)
        state.weights .= [0.4, 0.6]
        state.reduced_pot .= state.f .+ state.log_dens .- log.(state.weights)

        max_delta_f = Molly.update_tss_estimates!(state; visited_state=3)

        @test isfinite(max_delta_f)
        @test state.iteration == 1
        @test all(isfinite, state.f)
        @test state.tilts ≈ [0.0, 2.0]
        @test sum(state.density) ≈ 1.0
        @test all(>(0), state.density)
        @test_throws ArgumentError Molly.update_tss_estimates!(state; visited_state=1)
    end

    @testset "sample processing and simulation logging" begin
        tss_state = Molly.TSSState(thermo_states;
            first_state=1,
            gamma=[1.0, 1.0, 1.0],
            initial_f=[1.0, 2.0, 4.0],
            ETA=2.0,
            dens_reg=1e-4,
        )
        tss_state_show = sprint(show, tss_state)
        @test occursin("TSSState with 3 states", tss_state_show)
        @test occursin("active state 1", tss_state_show)
        @test !occursin("window_update_counts", tss_state_show)
        @test sprint(show, MIME"text/plain"(), tss_state) == tss_state_show
        state = Molly.active_tss_estimator(tss_state)
        estimator_show = sprint(show, state)
        @test occursin("_TSSLocalEstimator with 3 states", estimator_show)
        @test !occursin("reduced_pot", estimator_show)

        weights = Molly.process_tss_sample!(state)
        @test weights === state.weights
        @test sum(state.weights) ≈ 1.0
        @test all(>=(0), state.weights)
        @test all(isfinite, state.reduced_pot)

        sim = Molly.TSSSimulation(tss_state; n_md_steps=1, n_cycles=3, log_freq=1)
        sim_show = sprint(show, sim)
        @test occursin("TSSSimulation with 1 replica", sim_show)
        @test occursin("PMF deconvolution disabled", sim_show)
        @test !occursin("replica_workspaces", sim_show)
        @test sprint(show, MIME"text/plain"(), sim) == sim_show
        Molly.simulate!(sim; rng=MersenneTwister(1))

        @test tss_state.iteration == 3
        @test state.iteration == 3
        @test state.stats.iterations == [1, 2, 3]
        @test length(state.stats.active_state) == 3
        @test length(state.stats.sampled_next_state) == 3
        @test length(state.stats.max_abs_delta_f) == 3
        @test all(in(1:3), state.stats.active_state)
        @test all(in(1:3), state.stats.sampled_next_state)
        @test all(isfinite, state.stats.max_abs_delta_f)
        @test all(f -> length(f) == 3 && all(isfinite, f), state.stats.f_history)
        @test all(d -> length(d) == 3 && sum(d) ≈ 1.0, state.stats.dens_history)
        @test all(o -> length(o) == 3 && all(isfinite, o), state.stats.tilt_history)
        @test 1 <= state.active_state.active_idx <= 3
        @test state.f[1] == 0.0

        first_f_snapshot = copy(state.stats.f_history[1])
        state.f[2] += 1.0
        @test state.stats.f_history[1] == first_f_snapshot

        @test_throws ArgumentError Molly.TSSSimulation(tss_state; n_md_steps=0, n_cycles=1)
        @test_throws ArgumentError Molly.TSSSimulation(tss_state; n_md_steps=1, n_cycles=-1)
        @test_throws ArgumentError Molly.TSSSimulation(tss_state; n_md_steps=1, n_cycles=1, log_freq=0)
    end

    @testset "self-adjustment simulation" begin
        tss_state = Molly.TSSState(thermo_states;
            first_state=1,
            gamma=[1.0, 1.0, 1.0],
            ETA=2.0,
            dens_reg=1e-4,
        )
        state = Molly.active_tss_estimator(tss_state)
        sim = Molly.TSSSimulation(tss_state;
            n_md_steps=1,
            n_cycles=3,
            self_adjustment_steps=4,
            log_freq=1,
        )
        Molly.simulate!(sim; rng=MersenneTwister(2))

        @test state.iteration == 3
        @test length(state.stats.active_state) == 3
        @test all(in(1:3), state.stats.sampled_next_state)

        zero_cycle_state = Molly.TSSState(thermo_states)
        zero_cycle_sim = Molly.TSSSimulation(zero_cycle_state; n_md_steps=1, n_cycles=0)
        Molly.simulate!(zero_cycle_sim; rng=MersenneTwister(4))
        @test zero_cycle_state.iteration == 0
        @test isempty(zero_cycle_state.stats.iterations)

        @test_throws ArgumentError Molly.TSSSimulation(tss_state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=0)
        @test_throws ArgumentError Molly.TSSSimulation(tss_state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=-1)
        @test_throws ArgumentError Molly.TSSSimulation(tss_state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=1.5)
    end

    @testset "windowed TSS construction and validation" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        windows = [window.state_indices for window in graph4.windows]
        state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=2,
            first_window=2,
            gamma=[1.0, 2.0, 3.0, 4.0],
            initial_f=[0.0, 1.0, 3.0, 6.0],
            ETA=1.0,
            dens_reg=1e-4,
        )

        @test length(state.windows) == 5
        @test [window.state_indices for window in state.windows] == [Int[w...] for w in windows]
        @test state.state_to_windows == [[1, 2], [2, 3], [3, 4], [4, 5]]
        @test state.active_window == 2
        @test state.active_state.active_idx == 2
        @test all(est -> est.state_space === state.state_space, state.estimators)
        @test all(est -> est.active_state === state.active_state, state.estimators)
        @test [est.state_indices for est in state.estimators] == [Int[w...] for w in windows]
        @test state.estimators[2].gamma ≈ [1 / 3, 2 / 3]
        @test state.estimators[3].gamma ≈ [2 / 5, 3 / 5]
        @test state.estimators[2].f == [0.0, 1.0]
        @test state.estimators[3].f == [0.0, 2.0]
        @test state.coupling !== nothing
        @test state.coupling.converged
        @test state.coupling.max_abs_residual <= state.coupling.tolerance
        @test visit_control_free_energies_for_test(state) ≈ [0.0, 1.0, 3.0, 6.0]
        @test Molly.tss_free_energies(state) ≈ [0.0, 1.0, 3.0, 6.0]

        linear_graph = Molly.tss_grid_graph((8,); window_size=(4,), periodic=false)
        linear_windows = linear_graph.windows
        @test [window.state_indices for window in linear_windows] ==
              [[1, 2], [1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8]]

        @test all(length.(graph4.state_to_windows) .== 2)
        @test all(window -> all(state_index -> state_index in window.evaluation_state_indices,
                                window.state_indices), graph4.windows)
        @test graph4.rung_volumes == [0.5, 1.0, 1.0, 0.5]
        @test graph4.rung_neighbors[1] == [(1, 2, 1)]
        @test graph4.rung_neighbors[4] == [(3, 4, 1)]
        single_window_state = Molly.TSSState(thermo_states4; global_visit_control=false)
        @test length(single_window_state.windows) == 1
        @test only(single_window_state.windows).state_indices == collect(1:4)
        @test single_window_state.state_to_windows == [[1], [1], [1], [1]]
        @test Molly.other_window_for_state(single_window_state, 1) == 1
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=graph4, first_state=2, first_window=1)
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=graph4, gamma=[1.0, 2.0])
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=Molly.tss_grid_graph((8,); window_size=(4,), periodic=false))
        @test_throws ArgumentError Molly.tss_grid_graph((7,); window_size=(4,), periodic=false)
        @test_throws ArgumentError Molly.tss_grid_graph((8,); window_size=(3,), periodic=true)
        @test_throws ArgumentError Molly.tss_grid_graph((9,); window_size=(4,), periodic=true)
        @test_throws ArgumentError Molly.tss_grid_graph((8,); window_size=(10,), periodic=true)
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=graph4, visit_control_tolerance=0.0)
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=graph4, visit_control_max_iterations=0)
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=graph4, visit_control_damping=0.0)
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=graph4, pi_regularization=0.0)
        @test_throws ArgumentError Molly.TSSState(thermo_states4;
            graph=graph4, adaptive_gamma=:unknown)

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
        @test all(est -> est.gamma ≈
                         graph4.rung_volumes[est.state_indices] ./
                         sum(graph4.rung_volumes[est.state_indices]),
                  covdet_state.estimators)

        derivative_estimator = covdet_state.estimators[
            findfirst(est -> all(state_i -> state_i in est.state_indices, (2, 3)),
                      covdet_state.estimators)
        ]
        u_by_state = [0.0, 1.0, 4.0, 9.0]
        for (eval_i, state_i) in enumerate(derivative_estimator.evaluation_state_indices)
            derivative_estimator.evaluation_reduced_pot[eval_i] = u_by_state[state_i]
        end
        covdet_values = Molly._tss_covdet_moment_values(derivative_estimator)
        for (local_i, state_i) in enumerate(derivative_estimator.state_indices)
            reverse, forward, denominator = only(graph4.rung_neighbors[state_i])
            expected_derivative = denominator == 0 ? 0.0 :
                                  (u_by_state[forward] - u_by_state[reverse]) / denominator
            @test covdet_values[local_i, 1] ≈ expected_derivative
            @test covdet_values[local_i, 2] ≈ expected_derivative^2
        end

        manual_covdet_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            adaptive_gamma=:covdet,
            global_visit_control=false,
        )
        for estimator in manual_covdet_state.estimators
            estimator.adaptive_moments = zeros(length(estimator.state_indices), 2)
            for (local_i, state_i) in enumerate(estimator.state_indices)
                estimator.adaptive_moments[local_i, 2] = state_i^2
            end
        end
        Molly._update_windowed_tss_adaptive_gamma!(manual_covdet_state)
        for estimator in manual_covdet_state.estimators
            raw = Float64.(estimator.state_indices)
            volumes = graph4.rung_volumes[estimator.state_indices]
            expected = ((1 - 0.01) .* raw .+ 0.01 .* 4.0) .* volumes
            expected ./= sum(expected)
            @test estimator.gamma ≈ expected
        end

        local_only_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            global_visit_control=false,
        )
        @test local_only_state.coupling === nothing
        @test isnothing(local_only_state.coupling)
        @test_throws ArgumentError Molly.tss_free_energies(local_only_state)
    end

    @testset "TSS graph construction" begin
        periodic_graph = Molly.tss_grid_graph((8,); window_size=(4,), periodic=true)
        periodic_windows = periodic_graph.windows
        @test [window.state_indices for window in periodic_windows] ==
              [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 1, 2]]

        coverage = zeros(Int, 8)
        for window in periodic_windows
            for state_index in window.state_indices
                coverage[state_index] += 1
            end
        end
        @test coverage == fill(2, 8)

        thermo_states8 = make_tss_thermo_states(n_states=8)
        state = Molly.TSSState(thermo_states8;
            graph=periodic_graph,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )

        @test length(state.windows) == 4
        @test state.state_to_windows == [[1, 4], [1, 4], [1, 2], [1, 2],
                                         [2, 3], [2, 3], [3, 4], [3, 4]]
        @test Molly.other_window_for_state(state, 1) == 4
        @test state.coupling !== nothing
        @test state.coupling.converged

        graph2d = Molly.tss_grid_graph((4, 4); window_size=(2, 2), periodic=(true, true))
        @test graph2d.n_states == 16
        @test all(length.(graph2d.state_to_windows) .== 2)
        @test all(length(neighbors) == 2 for neighbors in graph2d.rung_neighbors)
        @test graph2d.rung_neighbors[1] == [(4, 2, 2), (13, 5, 2)]
        @test all(volume == 1.0 for volume in graph2d.rung_volumes)

        nonperiodic_2d = Molly.tss_grid_graph((4, 4); window_size=(2, 2))
        @test nonperiodic_2d.n_states == 16
        @test length(nonperiodic_2d.windows) == 13
        @test all(length.(nonperiodic_2d.state_to_windows) .== 2)
        @test nonperiodic_2d.rung_volumes[1] == 0.25
        @test nonperiodic_2d.rung_volumes[6] == 1.0

        builder = Molly.TSSGraphBuilder()
        Molly.add_tss_edge!(builder, ("A", "B"), (4,); window_size=(2,))
        Molly.add_tss_edge!(builder, ("B", "C"), (4,); window_size=(2,))
        stitched_graph = Molly.build_tss_graph(builder)
        @test stitched_graph.n_states == 8
        @test all(length.(stitched_graph.state_to_windows) .== 2)
        @test all(length(window.state_indices) <= 2 for window in stitched_graph.windows)
        @test_throws ArgumentError Molly.add_tss_edge!(
            Molly.TSSGraphBuilder(), ("A", "A"), (4,); window_size=(2,))
    end

    @testset "windowed active-window switching and local processing" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=2,
            first_window=2,
            ETA=1.0,
            dens_reg=1e-4,
        )

        @test Molly.windows_for_state(state, 2) == [2, 3]
        @test Molly.other_window_for_state(state, 2) == 3
        Molly.switch_active_window!(state; current_state=2)
        @test state.active_window == 3
        @test Molly.window_contains_state(state.windows[state.active_window], state.active_state.active_idx)

        estimator = Molly.active_tss_estimator(state)
        weights = Molly.process_tss_sample!(estimator)
        @test weights === estimator.weights
        @test length(weights) == length(state.windows[state.active_window].state_indices)
        @test sum(weights) ≈ 1.0
        @test all(>=(0), weights)

        rng = MersenneTwister(11)
        samples = [Molly.tss_sample_global_state(rng, estimator) for _ in 1:50]
        @test all(s -> s in state.windows[state.active_window].state_indices, samples)
        @test_throws ArgumentError Molly.update_tss_estimates!(estimator; visited_state=1)

        state.active_window = 1
        @test_throws ArgumentError Molly.switch_active_window!(state; current_state=2)
    end

    @testset "windowed visit-control coupling" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        true_f = [0.0, 1.0, 3.0, 6.0]
        state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            initial_f=true_f,
            ETA=1.0,
            dens_reg=1e-4,
            visit_control_tolerance=1e-10,
        )

        coupling = state.coupling
        @test coupling !== nothing
        @test coupling.converged
        @test coupling.max_abs_residual <= coupling.tolerance
        @test isapprox(
            visit_control_free_energies_for_test(state),
            true_f;
            atol=1e-8,
        )
        @test Molly.tss_free_energies(state) ≈ true_f
        @test sum(coupling.window_probs) ≈ 1.0
        @test all(>(0), coupling.window_probs)
        @test all(d -> length(d) > 0 && all(>(0), d) && sum(d) ≈ 1.0,
                  coupling.candidate_densities)
        @test isapprox(coupling.global_rung_weights, coupling.rhs_marginal; atol=1e-8)
        @test all(isfinite, coupling.window_offsets)
        @test all(isfinite, coupling.reported_offsets)
        @test coupling.reported_f ≈ true_f

        coupling.visit_control_f .= [0.0, 2.0, 0.0, 0.0]
        Molly.compute_windowed_sampling_densities!(state)
        density_12 = coupling.candidate_densities[2]
        @test density_12[2] > state.estimators[2].gamma[2]

        old_f = [copy(est.f) for est in state.estimators]
        old_tilts = [copy(est.tilts) for est in state.estimators]
        Molly.apply_windowed_sampling_densities!(state)
        @test all(i -> state.estimators[i].density ≈ coupling.candidate_densities[i],
                  eachindex(state.estimators))
        @test all(i -> state.estimators[i].f == old_f[i], eachindex(state.estimators))
        @test all(i -> state.estimators[i].tilts == old_tilts[i], eachindex(state.estimators))

        state.window_update_counts .= [0, 2, 0, 1, 0]
        Molly.update_window_probabilities!(state)
        @test sum(coupling.window_probs) ≈ 1.0
        @test coupling.window_probs[1] == 0.0
        @test coupling.window_probs[3] == 0.0
        @test coupling.window_probs[5] == 0.0
        @test coupling.window_probs[2] > coupling.window_probs[1]
        @test coupling.window_probs[4] > coupling.window_probs[1]

        state.window_update_counts .= 1
        foreach(est -> fill!(est.tilts, 1.0), state.estimators)
        Molly.update_window_probabilities!(state)
        flat_probs = copy(coupling.window_probs)
        @test sum(flat_probs) ≈ 1.0
        @test all(>(0), flat_probs)

        state.estimators[2].tilts .= [0.25, 4.0]
        Molly.update_window_probabilities!(state)
        @test sum(coupling.window_probs) ≈ 1.0
        @test coupling.window_probs != flat_probs
    end

    @testset "windowed self-adjustment cycle order" begin
        thermo_states8 = make_tss_thermo_states(n_states=8)
        graph8 = Molly.tss_grid_graph((8,); window_size=(4,), periodic=false)

        state = Molly.TSSState(thermo_states8;
            graph=graph8,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        sim = Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=5,
            self_adjustment_steps=4,
            log_freq=1,
        )
        Molly.simulate!(sim; rng=MersenneTwister(31))

        @test state.iteration == 5
        @test sum(state.window_update_counts) == 5
        @test length(state.stats.update_window) == 5
        @test sum(est.iteration for est in state.estimators) == sim.n_cycles
        for i in eachindex(state.stats.iterations)
            update_window = state.stats.update_window[i]
            @test state.stats.visited_state[i] in state.windows[update_window].state_indices
            @test state.stats.sampled_next_state[i] in state.windows[update_window].state_indices
        end
        @test state.active_state.active_idx in state.windows[state.active_window].state_indices

        first_cycle_state = Molly.TSSState(thermo_states8;
            graph=graph8,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        expected_window = Molly.other_window_for_state(first_cycle_state, 1)
        first_cycle_sim = Molly.TSSSimulation(first_cycle_state;
            n_md_steps=1,
            n_cycles=1,
            self_adjustment_steps=1,
            log_freq=1,
        )
        Molly.simulate!(first_cycle_sim; rng=MersenneTwister(32))
        @test only(first_cycle_state.stats.update_window) == expected_window
        @test first_cycle_state.window_update_counts[expected_window] == 1

    end

    @testset "windowed TSS simulation and reported free energies" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=2,
            first_window=2,
            ETA=1.0,
            dens_reg=1e-4,
        )
        sim = Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=4,
            self_adjustment_steps=3,
            log_freq=1,
        )
        @test length(sim.replicas) == 1
        @test sim.replicas[1].active_state === state.active_state
        @test sim.replicas[1].active_window == state.active_window
        Molly.simulate!(sim; rng=MersenneTwister(12))

        @test state.iteration == 4
        @test sum(est.iteration for est in state.estimators) == 4
        @test sum(state.window_update_counts) == 4
        @test state.stats.iterations == [1, 2, 3, 4]
        @test length(state.stats.update_window) == 4
        @test all(w -> 1 <= w <= length(state.windows), state.stats.update_window)
        for i in eachindex(state.stats.iterations)
            update_window = state.stats.update_window[i]
            @test state.stats.visited_state[i] in state.windows[update_window].state_indices
            @test state.stats.sampled_next_state[i] in state.windows[update_window].state_indices
        end
        @test state.active_state.active_idx in state.windows[state.active_window].state_indices
        @test all(state.estimators[i].density ≈ state.coupling.candidate_densities[i]
                  for i in eachindex(state.estimators))

        history_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=2,
            first_window=2,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.5, phi=1.2),
        )
        history_sim = Molly.TSSSimulation(history_state;
            n_md_steps=1,
            n_cycles=6,
            self_adjustment_steps=2,
            log_freq=1,
        )
        Molly.simulate!(history_sim; rng=MersenneTwister(13))

        recent_counts = [Molly.tss_recent_count(est) for est in history_state.estimators]
        @test history_state.iteration == 6
        @test sum(est.iteration for est in history_state.estimators) == 6
        @test sum(recent_counts) <= 6
        @test any(>(0), recent_counts)
        @test Molly._windowed_tss_visited_mask(history_state) == (recent_counts .> 0)
        @test all(est -> est.history !== nothing, history_state.estimators)
        @test all(isfinite, Molly.tss_free_energies(history_state; visited_only=true))

        @test_throws ArgumentError Molly.TSSSimulation(state;
            n_md_steps=1, n_cycles=1, n_replicas=2)

        multi_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        multi_sim = Molly.TSSSimulation(multi_state;
            n_md_steps=1,
            n_cycles=4,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
        )
        Molly.simulate!(multi_sim;
            rng=MersenneTwister(14),
            n_threads=1,
            replica_parallel=:serial,
        )

        @test length(multi_sim.replicas) == 2
        @test length(multi_sim.replica_workspaces) == 2
        @test multi_state.iteration == 4
        @test sum(est.iteration for est in multi_state.estimators) == 8
        @test sum(multi_state.window_update_counts) == 8
        @test sum(Molly.tss_recent_count(est) for est in multi_state.estimators) == 8
        @test length(multi_state.stats.replica_indices) == 4
        @test all(==([1, 2]), multi_state.stats.replica_indices)
        @test all(windows -> length(windows) == 2, multi_state.stats.replica_update_windows)
        for log_i in eachindex(multi_state.stats.replica_indices)
            for obs_i in eachindex(multi_state.stats.replica_indices[log_i])
                update_window = multi_state.stats.replica_update_windows[log_i][obs_i]
                visited_state = multi_state.stats.replica_visited_states[log_i][obs_i]
                next_state = multi_state.stats.replica_sampled_next_states[log_i][obs_i]
                @test visited_state in multi_state.windows[update_window].state_indices
                @test next_state in multi_state.windows[update_window].state_indices
            end
        end
        @test all(replica -> replica.active_state.active_idx in
                             multi_state.windows[replica.active_window].state_indices,
                  multi_sim.replicas)
        @test all(isfinite, Molly.tss_free_energies(multi_state; visited_only=true))

        covdet_windowed_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            adaptive_gamma=:covdet,
        )
        covdet_windowed_sim = Molly.TSSSimulation(covdet_windowed_state;
            n_md_steps=1,
            n_cycles=4,
            self_adjustment_steps=1,
            log_freq=1,
        )
        Molly.simulate!(covdet_windowed_sim; rng=MersenneTwister(141))
        @test all(est -> all(>(0), est.gamma) && sum(est.gamma) ≈ 1.0,
                  covdet_windowed_state.estimators)
        @test any(est -> !isnothing(est.adaptive_moments),
                  covdet_windowed_state.estimators)
        @test all(est -> isnothing(est.adaptive_moments) ||
                         size(est.adaptive_moments, 1) == length(est.state_indices),
                  covdet_windowed_state.estimators)
        @test all(isfinite, visit_control_free_energies_for_test(covdet_windowed_state))

        threaded_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        threaded_sim = Molly.TSSSimulation(threaded_state;
            n_md_steps=1,
            n_cycles=4,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
        )
        Molly.simulate!(threaded_sim;
            rng=MersenneTwister(15),
            n_threads=2,
            replica_parallel=:threads,
        )
        @test threaded_state.iteration == 4
        @test sum(est.iteration for est in threaded_state.estimators) == 8
        @test sum(threaded_state.window_update_counts) == 8
        @test sum(Molly.tss_recent_count(est) for est in threaded_state.estimators) == 8
        @test all(replica -> replica.active_state.active_idx in
                             threaded_state.windows[replica.active_window].state_indices,
                  threaded_sim.replicas)
        @test all(isfinite, Molly.tss_free_energies(threaded_state; visited_only=true))

        serial_a = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        serial_b = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        serial_sim_a = Molly.TSSSimulation(serial_a;
            n_md_steps=1,
            n_cycles=3,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
        )
        serial_sim_b = Molly.TSSSimulation(serial_b;
            n_md_steps=1,
            n_cycles=3,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
        )
        Molly.simulate!(serial_sim_a;
            rng=MersenneTwister(16),
            n_threads=1,
            replica_parallel=:serial,
        )
        Molly.simulate!(serial_sim_b;
            rng=MersenneTwister(16),
            n_threads=1,
            replica_parallel=:serial,
        )
        @test serial_a.window_update_counts == serial_b.window_update_counts
        @test [replica.active_state.active_idx for replica in serial_sim_a.replicas] ==
              [replica.active_state.active_idx for replica in serial_sim_b.replicas]
        @test all(serial_a.estimators[i].f ≈ serial_b.estimators[i].f
                  for i in eachindex(serial_a.estimators))
        @test all(serial_a.estimators[i].density ≈ serial_b.estimators[i].density
                  for i in eachindex(serial_a.estimators))

        supplied_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        supplied_active_states = [
            Molly.ActiveThermoState(supplied_state.state_space, 1),
            Molly.ActiveThermoState(supplied_state.state_space, 3),
        ]
        supplied_sim = Molly.TSSSimulation(supplied_state;
            n_md_steps=1,
            n_cycles=2,
            self_adjustment_steps=1,
            log_freq=1,
            replica_active_states=supplied_active_states,
        )
        @test length(supplied_sim.replicas) == 2
        @test supplied_sim.replicas[1].active_state === supplied_active_states[1]
        @test supplied_sim.replicas[2].active_state === supplied_active_states[2]
        Molly.simulate!(supplied_sim;
            rng=MersenneTwister(17),
            n_threads=1,
            replica_parallel=:serial,
        )
        @test supplied_state.iteration == 2
        @test sum(est.iteration for est in supplied_state.estimators) == 4
        @test sum(supplied_state.window_update_counts) == 4
        @test all(replica -> replica.active_state.active_idx in
                             supplied_state.windows[replica.active_window].state_indices,
                  supplied_sim.replicas)

        @test_throws ArgumentError Molly.simulate!(multi_sim; replica_parallel=:invalid)
        @test_throws ArgumentError Molly.simulate!(multi_sim; n_threads=0)
        @test_throws ArgumentError Molly.TSSSimulation(multi_state;
            n_md_steps=1, n_cycles=1, n_replicas=2, first_states=[1])
        @test_throws ArgumentError Molly.TSSSimulation(multi_state;
            n_md_steps=1, n_cycles=1, n_replicas=2, first_states=[1, 3],
            first_windows=[1, 1])

        reported_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
        )
        true_f = [0.0, 1.0, 3.0, 6.0]
        window_offsets = [10.0, -2.0, 5.0, 8.0, -4.0]
        for (window_i, estimator) in enumerate(reported_state.estimators)
            estimator.f .= true_f[reported_state.windows[window_i].state_indices] .+
                           window_offsets[window_i]
        end

        reported_f = Molly.tss_free_energies(reported_state)
        visit_control_f = visit_control_free_energies_for_test(reported_state)
        @test reported_f ≈ true_f .- true_f[1]
        @test all(isfinite, visit_control_f)
        @test all(isfinite, reported_state.coupling.reported_gamma)
        @test sum(reported_state.coupling.reported_gamma) ≈ 1.0

        foreach(est -> fill!(est.tilts, 1.0), reported_state.estimators)
        reported_eta_zero = Molly.tss_free_energies(reported_state)
        foreach(est -> (est.ETA = 4.0), reported_state.estimators)
        reported_eta_four = Molly.tss_free_energies(reported_state)
        @test reported_eta_four ≈ reported_eta_zero

        @test_throws ArgumentError Molly.tss_free_energy_uncertainties(state)

        short_history_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=2.0),
        )
        short_history_state.iteration = 1
        @test_throws ArgumentError Molly.tss_free_energy_uncertainties(short_history_state)

        jackknife_state = Molly.TSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=2.0),
        )
        jackknife_state.iteration = 4
        for (window_i, estimator) in enumerate(jackknife_state.estimators)
            local_f = true_f[jackknife_state.windows[window_i].state_indices] .+
                      window_offsets[window_i]
            estimator.f .= local_f
            estimator.tilts .= 1.0
            estimator.density .= estimator.gamma
            estimator.log_dens .= log.(estimator.density)

            history = estimator.history
            empty!(history.epochs)
            Molly._ensure_tss_epoch_bounds!(history, jackknife_state.iteration)
            stale_epoch = Molly.TSSEpoch(1, Float64, length(estimator.f))
            stale_epoch.count = 50
            stale_epoch.f .= reverse(local_f) .+ 10.0
            stale_epoch.tilts .= 1.0
            push!(history.epochs, stale_epoch)
            for epoch_index in 2:4
                epoch = Molly.TSSEpoch(epoch_index, Float64, length(estimator.f))
                epoch.count = 1
                epoch.f .= local_f
                epoch.tilts .= 1.0
                push!(history.epochs, epoch)
            end
        end

        jackknife = Molly.tss_free_energy_uncertainties(jackknife_state)
        @test jackknife isa Molly.TSSJackknifeResult
        @test jackknife.reference_state == 1
        @test jackknife.free_energies ≈ true_f
        @test all(<=(1e-12), abs.(jackknife.standard_errors))
        @test all(<=(1e-24), abs.(jackknife.mse))
        @test jackknife.epoch_indices == [2, 3, 4]
        @test jackknife.epoch_weights ≈ [0.25, 0.25, 0.5]
        @test size(jackknife.replicates) == (length(true_f), 3)
        @test all(replicate -> replicate ≈ true_f, eachcol(jackknife.replicates))

        jackknife_state.estimators[2].history.epochs[1].f[2] += 0.5
        noisy_jackknife = Molly.tss_free_energy_uncertainties(jackknife_state)
        @test all(isfinite, noisy_jackknife.standard_errors)
        @test noisy_jackknife.standard_errors[1] == 0.0
        @test any(>(0), noisy_jackknife.standard_errors[2:end])

        empty_retained_state = deepcopy(jackknife_state)
        for epoch in empty_retained_state.estimators[1].history.epochs
            epoch.index == 1 || (epoch.count = 0)
        end
        empty_retained_err = try
            Molly.tss_free_energy_uncertainties(empty_retained_state)
            nothing
        catch err
            err
        end
        @test empty_retained_err isa ArgumentError
        @test occursin("have no samples in the shared retained epochs",
                       sprint(showerror, empty_retained_err))

        sparse_delete_state = deepcopy(jackknife_state)
        for estimator in sparse_delete_state.estimators
            for epoch in estimator.history.epochs
                epoch.index == 1 && continue
                epoch.count = epoch.index == 2 ? 1 : 0
            end
        end
        sparse_delete_err = try
            Molly.tss_free_energy_uncertainties(sparse_delete_state)
            nothing
        catch err
            err
        end
        @test sparse_delete_err isa ArgumentError
        @test occursin("cannot delete every retained epoch",
                       sprint(showerror, sparse_delete_err))
    end
end
