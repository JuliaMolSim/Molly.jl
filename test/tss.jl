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

function prepare_tss_manual_sample!(state; reduced_pot=nothing)
    n_local = length(state.f)
    state.weights .= inv(n_local)
    state.density .= inv(n_local)
    state.log_dens .= log.(state.density)
    if isnothing(reduced_pot)
        state.reduced_pot .= 0.0
    else
        state.reduced_pot .= reduced_pot
    end
    return state
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

        default_history = Molly.TSSHistoryForgetting()
        @test default_history.alpha == 0.19
        @test default_history.target_n_epochs == 16
        @test default_history.phi ≈ 0.19^(-1 / 16)

        history = Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2)
        history_state = Molly.TSSState(thermo_states; history_forgetting=history)
        @test history_state.history !== nothing
        @test history_state.history.config.alpha == 0.0
        @test history_state.history.config.phi == 1.2
        @test_throws ArgumentError Molly.TSSHistoryForgetting(alpha=-0.1)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(alpha=1.0)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(n_epochs=0)
        @test_throws ArgumentError Molly.TSSHistoryForgetting(phi=1.0)

        adaptive = Molly.TSSAdaptiveGamma(
            context -> 1.0,
            (state_indices, means) -> ones(length(state_indices));
            epsilon_gamma=0.05,
        )
        @test adaptive.device_policy == :auto
        @test adaptive.epsilon_gamma == 0.05
        @test_throws ArgumentError Molly.TSSAdaptiveGamma(
            context -> 1.0,
            (state_indices, means) -> ones(length(state_indices));
            epsilon_gamma=-0.1,
        )
        @test_throws ArgumentError Molly.TSSAdaptiveGamma(
            context -> 1.0,
            (state_indices, means) -> ones(length(state_indices));
            device_policy=:gpu,
        )
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

    @testset "adaptive gamma observables" begin
        gamma_from_indices = (state_indices, means) -> Float64.(state_indices)

        raw_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> [context.active_global_state, context.step],
                gamma_from_indices;
                epsilon_gamma=0.1,
            ),
            ETA=0.0,
            dens_reg=1e-4,
        )
        prepare_tss_manual_sample!(raw_state)
        Molly.update_tss_estimates!(raw_state; visited_state=1)

        expected_raw_gamma = Float64.(1:3)
        expected_raw_gamma ./= sum(expected_raw_gamma)
        expected_gamma = 0.9 .* expected_raw_gamma .+ 0.1 .* fill(1 / 3, 3)
        expected_gamma ./= sum(expected_gamma)

        @test raw_state.observable_means ≈ repeat([1.0 1.0], 3, 1)
        @test raw_state.gamma ≈ expected_gamma
        @test raw_state.log_gamma ≈ log.(raw_state.gamma)
        @test raw_state.density ≈ raw_state.gamma

        scalar_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> 2.0u"nm",
                (state_indices, means) -> means[:, 1] .+ 1.0,
            ),
            ETA=0.0,
            dens_reg=1e-4,
        )
        prepare_tss_manual_sample!(scalar_state)
        Molly.update_tss_estimates!(scalar_state; visited_state=1)
        @test scalar_state.observable_means ≈ fill(2.0, 3, 1)
        @test scalar_state.gamma ≈ fill(1 / 3, 3)

        matrix_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> hcat(Float64.(context.state_indices),
                                fill(Float64(context.step), length(context.state_indices))),
                (state_indices, means) -> means[:, 1] .+ 1.0,
            ),
            ETA=0.0,
            dens_reg=1e-4,
        )
        prepare_tss_manual_sample!(matrix_state)
        Molly.update_tss_estimates!(matrix_state; visited_state=1)
        @test matrix_state.observable_means ≈ [1.0 1.0; 2.0 1.0; 3.0 1.0]
        expected_matrix_gamma = [2.0, 3.0, 4.0]
        expected_matrix_gamma ./= sum(expected_matrix_gamma)
        expected_matrix_gamma = 0.99 .* expected_matrix_gamma .+ 0.01 .* fill(1 / 3, 3)
        expected_matrix_gamma ./= sum(expected_matrix_gamma)
        @test matrix_state.gamma ≈ expected_matrix_gamma

        cv_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                Molly.TSSCVObservable(Molly.CalcDist([1], [2], Molly.CalcSingleDist(:raw))),
                (state_indices, means) -> ones(length(state_indices));
                device_policy=:auto,
            ),
            ETA=0.0,
            dens_reg=1e-4,
        )
        prepare_tss_manual_sample!(cv_state)
        Molly.update_tss_estimates!(cv_state; visited_state=1)
        @test size(cv_state.observable_means) == (3, 1)
        @test all(isfinite, cv_state.observable_means)
        @test cv_state.gamma ≈ fill(1 / 3, 3)

        logger_style_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                Molly.TSSSystemObservable(
                    (sys, buffers, neighbors, step_n; n_threads=1) ->
                        length(sys.coords) + step_n / 100,
                ),
                (state_indices, means) -> means[:, 1],
            ),
            ETA=0.0,
            dens_reg=1e-4,
        )
        prepare_tss_manual_sample!(logger_style_state)
        Molly.update_tss_estimates!(logger_style_state; visited_state=1)
        @test logger_style_state.observable_means ≈ fill(6.01, 3, 1)
        @test logger_style_state.gamma ≈ fill(1 / 3, 3)

        history_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> hcat(Float64.(context.state_indices),
                                fill(Float64(context.step), length(context.state_indices))),
                (state_indices, means) -> means[:, 1] .+ 1.0,
            ),
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.5, phi=1.2),
            ETA=0.0,
            dens_reg=1e-4,
        )
        for step in 1:8
            prepare_tss_manual_sample!(history_state; reduced_pot=[0.1, 0.2, 0.3] .* step)
            Molly.update_tss_estimates!(history_state; visited_state=mod1(step, 3))
        end
        @test size(history_state.observable_means) == (3, 2)
        @test all(isfinite, history_state.observable_means)
        @test all(epoch -> isnothing(epoch.observable_means) ||
                           size(epoch.observable_means) == (3, 2),
                  history_state.history.epochs)
        @test all(>(0), history_state.gamma)
        @test sum(history_state.gamma) ≈ 1.0

        bad_shape_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> ones(2, 2),
                (state_indices, means) -> ones(length(state_indices)),
            ),
        )
        prepare_tss_manual_sample!(bad_shape_state)
        @test_throws ArgumentError Molly.update_tss_estimates!(bad_shape_state; visited_state=1)

        nonfinite_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> NaN,
                (state_indices, means) -> ones(length(state_indices)),
            ),
        )
        prepare_tss_manual_sample!(nonfinite_state)
        @test_throws ArgumentError Molly.update_tss_estimates!(nonfinite_state; visited_state=1)

        bad_gamma_state = Molly.TSSState(thermo_states;
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> 1.0,
                (state_indices, means) -> [1.0, 0.0, 1.0],
            ),
        )
        prepare_tss_manual_sample!(bad_gamma_state)
        @test_throws ArgumentError Molly.update_tss_estimates!(bad_gamma_state; visited_state=1)
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
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        windows = [window.state_indices for window in graph4.windows]
        state = Molly.WindowedTSSState(thermo_states4;
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
        @test Molly.windowed_tss_visit_control_free_energies(state) ≈ [0.0, 1.0, 3.0, 6.0]
        @test Molly.windowed_tss_free_energies(state) ≈ [0.0, 1.0, 3.0, 6.0]

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
        @test !(:TSSWindow in names(Molly))
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            windows=windows)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            graph=graph4, first_state=2, first_window=1)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            graph=graph4, gamma=[1.0, 2.0])
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            graph=Molly.tss_grid_graph((8,); window_size=(4,), periodic=false))
        @test_throws ArgumentError Molly.tss_grid_graph((7,); window_size=(4,), periodic=false)
        @test_throws ArgumentError Molly.tss_grid_graph((8,); window_size=(3,), periodic=true)
        @test_throws ArgumentError Molly.tss_grid_graph((9,); window_size=(4,), periodic=true)
        @test_throws ArgumentError Molly.tss_grid_graph((8,); window_size=(10,), periodic=true)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            graph=graph4, visit_control_tolerance=0.0)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            graph=graph4, visit_control_max_iterations=0)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            graph=graph4, visit_control_damping=0.0)
        @test_throws ArgumentError Molly.WindowedTSSState(thermo_states4;
            graph=graph4, pi_regularization=0.0)

        local_only_state = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
            global_visit_control=false,
        )
        @test local_only_state.coupling === nothing
        @test_throws ArgumentError Molly.windowed_tss_visit_control_free_energies(local_only_state)
        @test_throws ArgumentError Molly.windowed_tss_free_energies(local_only_state)
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
        state = Molly.WindowedTSSState(thermo_states8;
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

        edge_graph = Molly.tss_grid_graph((8,); window_size=(2,), periodic=true)
        edge_windows = edge_graph.windows
        @test first(edge_windows).state_indices == [1, 2]
        @test last(edge_windows).state_indices == [8, 1]
        edge_coverage = zeros(Int, 8)
        for window in edge_windows, state_index in window.state_indices
            edge_coverage[state_index] += 1
        end
        @test edge_coverage == fill(2, 8)

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

        graph3d = Molly.tss_grid_graph((4, 4, 4);
            window_size=(2, 2, 2),
            periodic=(true, true, true),
        )
        @test graph3d.n_states == 64
        @test all(length.(graph3d.state_to_windows) .== 2)
        @test all(length(neighbors) == 3 for neighbors in graph3d.rung_neighbors)
        @test all(volume == 1.0 for volume in graph3d.rung_volumes)

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
        state = Molly.WindowedTSSState(thermo_states4;
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
        state = Molly.WindowedTSSState(thermo_states4;
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
        graph8 = Molly.tss_grid_graph((8,); window_size=(4,), periodic=false)

        state = Molly.WindowedTSSState(thermo_states8;
            graph=graph8,
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
            graph=graph8,
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
                graph=graph8,
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
        graph4 = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        state = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
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
            graph=graph4,
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
            n_md_steps=1, n_cycles=1, n_replicas=2)

        multi_state = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        multi_sim = Molly.WindowedTSSSimulation(multi_state;
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
        @test all(est -> all(isfinite, est.f), multi_state.estimators)
        @test all(est -> all(isfinite, est.density) && all(>(0), est.density) &&
                         sum(est.density) ≈ 1.0, multi_state.estimators)
        @test all(isfinite, Molly.windowed_tss_free_energies(multi_state; visited_only=true))
        @test all(isfinite, Molly.windowed_tss_visit_control_free_energies(multi_state))

        adaptive_windowed_state = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> hcat(Float64.(context.state_indices)),
                (state_indices, means) -> means[:, 1] .+ 1.0,
            ),
        )
        adaptive_windowed_sim = Molly.WindowedTSSSimulation(adaptive_windowed_state;
            n_md_steps=1,
            n_cycles=4,
            self_adjustment_steps=1,
            log_freq=1,
        )
        Molly.simulate!(adaptive_windowed_sim; rng=MersenneTwister(141))
        @test all(est -> all(>(0), est.gamma) && sum(est.gamma) ≈ 1.0,
                  adaptive_windowed_state.estimators)
        @test any(est -> !isnothing(est.observable_means),
                  adaptive_windowed_state.estimators)
        @test all(est -> isnothing(est.observable_means) ||
                         size(est.observable_means, 1) == length(est.state_indices),
                  adaptive_windowed_state.estimators)
        @test all(isfinite, Molly.windowed_tss_visit_control_free_energies(adaptive_windowed_state))

        adaptive_multi_state = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
            adaptive_gamma=Molly.TSSAdaptiveGamma(
                context -> hcat(Float64.(context.state_indices),
                                fill(Float64(context.active_global_state),
                                     length(context.state_indices))),
                (state_indices, means) -> means[:, 1] .+ 1.0,
            ),
        )
        adaptive_multi_sim = Molly.WindowedTSSSimulation(adaptive_multi_state;
            n_md_steps=1,
            n_cycles=3,
            self_adjustment_steps=1,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
        )
        Molly.simulate!(adaptive_multi_sim;
            rng=MersenneTwister(142),
            n_threads=1,
            replica_parallel=:serial,
        )
        @test sum(Molly.tss_recent_count(est) for est in adaptive_multi_state.estimators) == 6
        @test all(est -> all(>(0), est.gamma) && sum(est.gamma) ≈ 1.0,
                  adaptive_multi_state.estimators)
        @test all(est -> Molly.tss_recent_count(est) == 0 || !isnothing(est.observable_means),
                  adaptive_multi_state.estimators)
        @test all(isfinite, Molly.windowed_tss_free_energies(adaptive_multi_state; visited_only=true))

        threaded_state = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        threaded_sim = Molly.WindowedTSSSimulation(threaded_state;
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
        @test all(est -> all(isfinite, est.f), threaded_state.estimators)
        @test all(est -> all(isfinite, est.density) && all(>(0), est.density) &&
                         sum(est.density) ≈ 1.0, threaded_state.estimators)
        @test all(isfinite, Molly.windowed_tss_free_energies(threaded_state; visited_only=true))
        @test all(isfinite, Molly.windowed_tss_visit_control_free_energies(threaded_state))

        serial_a = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        serial_b = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        serial_sim_a = Molly.WindowedTSSSimulation(serial_a;
            n_md_steps=1,
            n_cycles=3,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
        )
        serial_sim_b = Molly.WindowedTSSSimulation(serial_b;
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

        supplied_state = Molly.WindowedTSSState(thermo_states4;
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
        supplied_sim = Molly.WindowedTSSSimulation(supplied_state;
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
        @test_throws ArgumentError Molly.WindowedTSSSimulation(multi_state;
            n_md_steps=1, n_cycles=1, n_replicas=2, first_states=[1])
        @test_throws ArgumentError Molly.WindowedTSSSimulation(multi_state;
            n_md_steps=1, n_cycles=1, n_replicas=2, first_states=[1, 3],
            first_windows=[1, 1])

        @test_throws ArgumentError Molly.WindowedTSSSimulation(state;
            n_md_steps=0, n_cycles=1)
        @test_throws ArgumentError Molly.WindowedTSSSimulation(state;
            n_md_steps=1, n_cycles=-1)
        @test_throws ArgumentError Molly.WindowedTSSSimulation(state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=0)
        @test_throws ArgumentError Molly.WindowedTSSSimulation(state;
            n_md_steps=1, n_cycles=1, self_adjustment_steps=1.5)

        reported_state = Molly.WindowedTSSState(thermo_states4;
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

        @test_throws ArgumentError Molly.windowed_tss_free_energy_uncertainties(state)

        short_history_state = Molly.WindowedTSSState(thermo_states4;
            graph=graph4,
            first_state=1,
            first_window=1,
            ETA=0.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=2.0),
        )
        short_history_state.iteration = 1
        @test_throws ArgumentError Molly.windowed_tss_free_energy_uncertainties(short_history_state)

        jackknife_state = Molly.WindowedTSSState(thermo_states4;
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

        jackknife = Molly.windowed_tss_free_energy_uncertainties(jackknife_state)
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
        noisy_jackknife = Molly.windowed_tss_free_energy_uncertainties(jackknife_state)
        @test all(isfinite, noisy_jackknife.standard_errors)
        @test noisy_jackknife.standard_errors[1] == 0.0
        @test any(>(0), noisy_jackknife.standard_errors[2:end])

        empty_retained_state = deepcopy(jackknife_state)
        for epoch in empty_retained_state.estimators[1].history.epochs
            epoch.index == 1 || (epoch.count = 0)
        end
        empty_retained_err = try
            Molly.windowed_tss_free_energy_uncertainties(empty_retained_state)
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
            Molly.windowed_tss_free_energy_uncertainties(sparse_delete_state)
            nothing
        catch err
            err
        end
        @test sparse_delete_err isa ArgumentError
        @test occursin("cannot delete every retained epoch",
                       sprint(showerror, sparse_delete_err))
    end
end
