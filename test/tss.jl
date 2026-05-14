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
    end

    @testset "sampling distribution update" begin
        state = Molly.TSSState(thermo_states;
            gamma=[2.0, 1.0, 1.0],
            ETA=2.0,
            dens_reg=0.1,
        )

        state.tilts .= [1.0, 2.0, 4.0]
        Molly.update_tss_sampling_distribution!(state)

        raw = state.gamma .* state.tilts .^ state.ETA
        raw ./= sum(raw)
        expected = (1 - state.dens_reg) .* raw .+ state.dens_reg .* state.gamma
        expected ./= sum(expected)

        @test state.density ≈ expected
        @test state.log_dens ≈ log.(expected)
        @test sum(state.density) ≈ 1.0
        @test all(>(0), state.density)

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
end
