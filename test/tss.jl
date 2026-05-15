using Molly
using Random
using Test
using Unitful

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

@testset "Times Square Sampling (TSS)" begin
    thermo_states = make_tss_thermo_states()

    @testset "constructor validation" begin
        state = Molly.TSSState(thermo_states;
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

        history = Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2)
        history_state = Molly.TSSState(thermo_states; history_forgetting=history)
        @test history_state.history !== nothing
        @test history_state.history.config.alpha == 0.0
        @test history_state.history.config.phi == 1.2
        @test_throws ArgumentError Molly.TSSHistoryForgetting(alpha=-0.1)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(alpha=1.0)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(n_epochs=0)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(phi=1.0)
    end

    @testset "subset constructor validation and index mapping" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        state = Molly.TSSState(thermo_states4;
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

        custom_state = Molly.TSSState(thermo_states4;
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

        @test_throws ArgumentError Molly.TSSState(thermo_states4; first_state=1,
            state_indices=[2, 3])
        @test_throws ArgumentError Molly.TSSState(thermo_states4; first_state=2,
            state_indices=[2, 2])
        @test_throws ArgumentError Molly.TSSState(thermo_states4; first_state=2,
            state_indices=[0, 2])
        @test_throws ArgumentError Molly.TSSState(thermo_states4; first_state=2,
            state_indices=[-1, 2])
        @test_throws ArgumentError Molly.TSSState(thermo_states4; first_state=2,
            state_indices=[2, 5])
        @test_throws ArgumentError Molly.TSSState(thermo_states4; first_state=2,
            state_indices=Int[])
        @test_throws ArgumentError Molly.TSSState(thermo_states4; first_state=2,
            state_indices=[2, 3], gamma=[1.0, 1.0, 1.0, 1.0])
        @test_throws ArgumentError Molly.TSSState(thermo_states4; first_state=2,
            state_indices=[2, 3], initial_f=[0.0, 1.0, 2.0, 3.0])
    end

    @testset "sampling distribution update" begin
        state = Molly.TSSState(thermo_states;
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

        state_eta_zero = Molly.TSSState(thermo_states;
            gamma=[2.0, 1.0, 1.0],
            ETA=0.0,
            dens_reg=0.1,
        )
        state_eta_zero.tilts .= [1.0, 100.0, 0.5]
        Molly.update_tss_sampling_distribution!(state_eta_zero)
        @test state_eta_zero.density ≈ state_eta_zero.gamma
    end

    @testset "conditional weights and one-step estimator update" begin
        state = Molly.TSSState(thermo_states;
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
        state = Molly.TSSState(thermo_states;
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
        no_history = Molly.TSSState(thermo_states;
            gamma=[1.0, 1.0, 1.0],
            ETA=1.0,
            dens_reg=1e-4,
        )
        keep_all = Molly.TSSState(thermo_states;
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
        @test Molly.tss_retained_epoch_count(keep_all) >= 1
        @test keep_all.f ≈ no_history.f
        @test keep_all.tilts ≈ no_history.tilts
        @test keep_all.density ≈ no_history.density

        forget_old = Molly.TSSState(thermo_states;
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
        @test Molly.tss_retained_epoch_count(forget_old) < forget_old.iteration
        @test all(isfinite, forget_old.f)
        @test all(isfinite, forget_old.tilts)
        @test all(isfinite, forget_old.density)
        @test all(>(0), forget_old.density)
        @test sum(forget_old.density) ≈ 1.0
        @test forget_old.f[1] == 0.0
    end

    @testset "subset sample processing and estimator update" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        state = Molly.TSSState(thermo_states4;
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
        state = Molly.TSSState(thermo_states;
            first_state=1,
            gamma=[1.0, 1.0, 1.0],
            initial_f=[1.0, 2.0, 4.0],
            ETA=2.0,
            dens_reg=1e-4,
        )

        weights = Molly.process_tss_sample!(state)
        @test weights === state.weights
        @test sum(state.weights) ≈ 1.0
        @test all(>=(0), state.weights)
        @test all(isfinite, state.reduced_pot)

        sim = Molly.TSSSimulation(state; n_md_steps=1, n_cycles=3, log_freq=1)
        Molly.simulate!(sim; rng=MersenneTwister(1))

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

        @test_throws ArgumentError Molly.TSSSimulation(state; n_md_steps=0, n_cycles=1)
        @test_throws ArgumentError Molly.TSSSimulation(state; n_md_steps=1, n_cycles=-1)
        @test_throws ArgumentError Molly.TSSSimulation(state; n_md_steps=1, n_cycles=1, log_freq=0)
    end

    @testset "self-adjustment simulation" begin
        state = Molly.TSSState(thermo_states;
            first_state=1,
            gamma=[1.0, 1.0, 1.0],
            ETA=2.0,
            dens_reg=1e-4,
        )
        sim = Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=3,
            self_adjustment_steps=4,
            log_freq=1,
        )
        Molly.simulate!(sim; rng=MersenneTwister(2))

        @test state.iteration == 3
        @test state.stats.iterations == [1, 2, 3]
        @test length(state.stats.active_state) == 3
        @test length(state.stats.sampled_next_state) == 3
        @test all(in(1:3), state.stats.active_state)
        @test all(in(1:3), state.stats.sampled_next_state)
        @test all(isfinite, state.stats.max_abs_delta_f)
        @test all(f -> length(f) == 3 && all(isfinite, f), state.stats.f_history)
        @test all(d -> length(d) == 3 && sum(d) ≈ 1.0, state.stats.dens_history)

        subset_state = Molly.TSSState(make_tss_thermo_states(n_states=4);
            first_state=2,
            state_indices=[2, 3],
            ETA=1.0,
            dens_reg=1e-4,
        )
        subset_sim = Molly.TSSSimulation(subset_state;
            n_md_steps=1,
            n_cycles=2,
            self_adjustment_steps=3,
            log_freq=1,
        )
        Molly.simulate!(subset_sim; rng=MersenneTwister(3))

        @test subset_state.iteration == 2
        @test subset_state.stats.iterations == [1, 2]
        @test all(in([2, 3]), subset_state.stats.active_state)
        @test all(in([2, 3]), subset_state.stats.sampled_next_state)
        @test all(f -> length(f) == 2 && all(isfinite, f), subset_state.stats.f_history)
        @test all(d -> length(d) == 2 && sum(d) ≈ 1.0, subset_state.stats.dens_history)
        @test subset_state.active_state.active_idx in (2, 3)

        zero_cycle_state = Molly.TSSState(thermo_states)
        zero_cycle_sim = Molly.TSSSimulation(zero_cycle_state; n_md_steps=1, n_cycles=0)
        Molly.simulate!(zero_cycle_sim; rng=MersenneTwister(4))
        @test zero_cycle_state.iteration == 0
        @test isempty(zero_cycle_state.stats.iterations)

        @test_throws ArgumentError Molly.TSSSimulation(state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=0)
        @test_throws ArgumentError Molly.TSSSimulation(state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=-1)
        @test_throws ArgumentError Molly.TSSSimulation(state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=1.5)
    end

    @testset "windowed TSS construction and validation" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        windows = [[1], [1, 2], [2, 3], [3, 4], [4]]
        state = Molly.WindowedTSSState(thermo_states4;
            windows=windows,
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
        @test Molly.windowed_tss_visit_control_free_energies(state) ≈ [0.0, 1.0, 3.0, 6.0]
        @test Molly.windowed_tss_free_energies(state) ≈ [0.0, 1.0, 3.0, 6.0]

        linear_windows = Molly.linear_tss_windows(8; window_size=4)
        @test [window.state_indices for window in linear_windows] ==
              [[1, 2], [1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8]]

        @test_throws ArgumentError Molly.TSSWindow(1, [1, 3])
        custom_window = Molly.TSSWindow(1, [1, 3]; check_contiguous=false)
        @test custom_window.state_indices == [1, 3]
        @test_throws ArgumentError Molly.TSSWindow(1, [1, 1])
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=[[1], [1, 2], [2, 2], [3, 4], [4]])
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=[[1], [1, 2], [2, 3], [3, 4]])
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=[[1], [1, 2], [2], [2, 3], [3, 4], [4]])
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=[[1], [1, 2], [2], [3], [3, 4], [4]])
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=windows, first_state=2, first_window=1)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=windows, gamma=[1.0, 2.0])
        @test_throws ArgumentError Molly.linear_tss_windows(7; window_size=4)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=windows, visit_control_tolerance=0.0)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=windows, visit_control_max_iterations=0)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=windows, visit_control_damping=0.0)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=windows, pi_regularization=0.0)

        local_only_state = Molly.WindowedTSSState(thermo_states4;
            windows=windows,
            global_visit_control=false,
        )
        @test local_only_state.coupling === nothing
        @test_throws ArgumentError Molly.windowed_tss_visit_control_free_energies(local_only_state)
        @test_throws ArgumentError Molly.windowed_tss_free_energies(local_only_state)
    end

    @testset "periodic TSS windows" begin
        periodic_windows = Molly.periodic_tss_windows(8; window_size=4)
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
        state = Molly.WindowedTSSState(thermo_states8;
            windows=periodic_windows,
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

        edge_windows = Molly.periodic_tss_windows(8; window_size=2)
        @test first(edge_windows).state_indices == [1, 2]
        @test last(edge_windows).state_indices == [8, 1]
        edge_coverage = zeros(Int, 8)
        for window in edge_windows, state_index in window.state_indices
            edge_coverage[state_index] += 1
        end
        @test edge_coverage == fill(2, 8)

        @test_throws ArgumentError Molly.periodic_tss_windows(8; window_size=3)
        @test_throws ArgumentError Molly.periodic_tss_windows(9; window_size=4)
        @test_throws ArgumentError Molly.periodic_tss_windows(8; window_size=10)
    end

    @testset "windowed active-window switching and local processing" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        state = Molly.WindowedTSSState(thermo_states4;
            windows=[[1], [1, 2], [2, 3], [3, 4], [4]],
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
        windows = [[1], [1, 2], [2, 3], [3, 4], [4]]
        true_f = [0.0, 1.0, 3.0, 6.0]
        state = Molly.WindowedTSSState(thermo_states4;
            windows=windows,
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
            Molly.windowed_tss_visit_control_free_energies(state),
            true_f;
            atol=1e-8,
        )
        @test Molly.windowed_tss_free_energies(state) ≈ true_f
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
        windows = Molly.linear_tss_windows(8; window_size=4)

        state = Molly.WindowedTSSState(thermo_states8;
            windows=windows,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        sim = Molly.WindowedTSSSimulation(state;
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

        first_cycle_state = Molly.WindowedTSSState(thermo_states8;
            windows=windows,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        expected_window = Molly.other_window_for_state(first_cycle_state, 1)
        first_cycle_sim = Molly.WindowedTSSSimulation(first_cycle_state;
            n_md_steps=1,
            n_cycles=1,
            self_adjustment_steps=1,
            log_freq=1,
        )
        Molly.simulate!(first_cycle_sim; rng=MersenneTwister(32))
        @test only(first_cycle_state.stats.update_window) == expected_window
        @test first_cycle_state.window_update_counts[expected_window] == 1

        for self_adjustment_steps in (1, 5)
            count_state = Molly.WindowedTSSState(thermo_states8;
                windows=windows,
                first_state=1,
                first_window=1,
                ETA=1.0,
                dens_reg=1e-4,
            )
            count_sim = Molly.WindowedTSSSimulation(count_state;
                n_md_steps=1,
                n_cycles=3,
                self_adjustment_steps=self_adjustment_steps,
                log_freq=1,
            )
            Molly.simulate!(count_sim; rng=MersenneTwister(40 + self_adjustment_steps))

            @test count_state.iteration == 3
            @test sum(est.iteration for est in count_state.estimators) == 3
            @test sum(count_state.window_update_counts) == 3
        end
    end

    @testset "windowed TSS simulation and reported free energies" begin
        thermo_states4 = make_tss_thermo_states(n_states=4)
        windows = [[1], [1, 2], [2, 3], [3, 4], [4]]
        state = Molly.WindowedTSSState(thermo_states4;
            windows=windows,
            first_state=2,
            first_window=2,
            ETA=1.0,
            dens_reg=1e-4,
        )
        sim = Molly.WindowedTSSSimulation(state;
            n_md_steps=1,
            n_cycles=4,
            self_adjustment_steps=3,
            log_freq=1,
        )
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
        @test all(est -> all(isfinite, est.f), state.estimators)
        @test all(est -> all(isfinite, est.density) && sum(est.density) ≈ 1.0, state.estimators)
        @test all(est -> all(isfinite, est.tilts), state.estimators)
        @test all(f -> length(f) == 4 && all(isfinite, f), state.stats.reported_f_history)
        @test length(state.stats.visit_control_converged) == 4
        @test length(state.stats.visit_control_iterations) == 4
        @test length(state.stats.visit_control_max_abs_residual) == 4
        @test all(isfinite, state.stats.visit_control_max_abs_residual)
        @test all(p -> length(p) == length(state.windows) && sum(p) ≈ 1.0,
                  state.stats.window_prob_history)
        @test all(f -> length(f) == 4 && all(isfinite, f),
                  state.stats.visit_control_f_history)
        @test all(state.estimators[i].density ≈ state.coupling.candidate_densities[i]
                  for i in eachindex(state.estimators))

        history_state = Molly.WindowedTSSState(thermo_states4;
            windows=windows,
            first_state=2,
            first_window=2,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.5, phi=1.2),
        )
        history_sim = Molly.WindowedTSSSimulation(history_state;
            n_md_steps=1,
            n_cycles=6,
            self_adjustment_steps=2,
            log_freq=1,
        )
        Molly.simulate!(history_sim; rng=MersenneTwister(13))

        recent_counts = [Molly.tss_recent_count(est) for est in history_state.estimators]
        retained_epochs = [Molly.tss_retained_epoch_count(est) for est in history_state.estimators]
        @test history_state.iteration == 6
        @test sum(est.iteration for est in history_state.estimators) == 6
        @test sum(recent_counts) <= 6
        @test any(>(0), recent_counts)
        @test all(>=(0), retained_epochs)
        @test Molly._windowed_tss_visited_mask(history_state) == (recent_counts .> 0)
        @test all(est -> est.history !== nothing, history_state.estimators)
        @test all(est -> all(isfinite, est.f), history_state.estimators)
        @test all(est -> all(isfinite, est.density) && all(>(0), est.density) &&
                         sum(est.density) ≈ 1.0, history_state.estimators)
        @test all(isfinite, Molly.windowed_tss_free_energies(history_state; visited_only=true))
        @test all(isfinite, Molly.windowed_tss_visit_control_free_energies(history_state))

        @test_throws ArgumentError Molly.WindowedTSSSimulation(state;
            n_md_steps=0, n_cycles=1)
        @test_throws ArgumentError Molly.WindowedTSSSimulation(state;
            n_md_steps=1, n_cycles=-1)
        @test_throws ArgumentError Molly.WindowedTSSSimulation(state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=0)
        @test_throws ArgumentError Molly.WindowedTSSSimulation(state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=1.5)

        reported_state = Molly.WindowedTSSState(thermo_states4;
            windows=windows,
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

        reported_f = Molly.windowed_tss_free_energies(reported_state)
        visit_control_f = Molly.windowed_tss_visit_control_free_energies(reported_state)
        @test reported_f ≈ true_f .- true_f[1]
        @test all(isfinite, visit_control_f)
        @test all(isfinite, reported_state.coupling.reported_gamma)
        @test sum(reported_state.coupling.reported_gamma) ≈ 1.0

        foreach(est -> fill!(est.tilts, 1.0), reported_state.estimators)
        reported_eta_zero = Molly.windowed_tss_free_energies(reported_state)
        foreach(est -> (est.ETA = 4.0), reported_state.estimators)
        reported_eta_four = Molly.windowed_tss_free_energies(reported_state)
        @test reported_eta_four ≈ reported_eta_zero
    end
end
