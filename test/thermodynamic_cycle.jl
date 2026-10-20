# A leg source with known free energies and independent standard errors per state
struct ToyFreeEnergies{F, S}
    F::F
    σ::S
end

Molly.free_energy_difference(src::ToyFreeEnergies, i, j) = src.F[j] - src.F[i]
Molly.free_energy_difference_uncertainty(src::ToyFreeEnergies, i, j) = hypot(src.σ[i], src.σ[j])

function make_cycle_thermo_states(; temps=fill(298.0u"K", 4), names=nothing, n_atoms=4)
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    neighbor_finder = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=10,
                                             dist_cutoff=1.5u"nm")
    return map(enumerate(temps)) do (k, temp)
        lambda = 1.0 - 0.1 * (k - 1)
        atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1",
                      λ=lambda, alch_role=Molly.InsertRole) for _ in 1:n_atoms]
        sys = System(atoms=atoms, coords=coords, boundary=boundary,
                     pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true,
                                      scheduler=Molly.LinearLambdaScheduler(dual=true)),),
                     neighbor_finder=neighbor_finder)
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        name = isnothing(names) ? nothing : names[k]
        ThermoState(sys, intg; temperature=temp, name=name)
    end
end

# A TSSState with fixed free energies `true_f` and three history epochs, as in the
#   jackknife test of tss.jl; `noise` perturbs one epoch so that the errors are nonzero
function make_fixed_tss_state(thermo_states, true_f; noise=0.5)
    graph = Molly.tss_grid_graph((length(true_f),); window_size=(2,), periodic=false)
    state = TSSState(thermo_states; graph=graph, first_state=1, first_window=1, ETA=0.0,
                     dens_reg=1e-4,
                     history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=2.0))
    state.iteration = 4
    for (window_i, estimator) in enumerate(state.estimators)
        local_f = true_f[state.windows[window_i].state_indices]
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
    state.estimators[2].history.epochs[1].f[2] += noise
    return state
end

# Umbrella windows on a harmonic potential sampled exactly (reduced units), as in the
#   MBAR test of analysis.jl
function harmonic_umbrella_samples(rng; n=2_000)
    κ0, κ, μs = 1.0, 5.0, range(-2.0, 2.0; length=9)
    x = [κ * μ / (κ0 + κ) + randn(rng) / sqrt(κ0 + κ) for μ in μs for _ in 1:n]
    win_of, N_k = repeat(eachindex(μs); inner=n), fill(n, length(μs))
    u = [0.5 * κ0 * xi^2 + 0.5 * κ * (xi - μ)^2 for xi in x, μ in μs]
    exact_f = @.(0.5 * κ0 * κ / (κ0 + κ) * (μs^2 - μs[1]^2))
    return (; x, u, win_of, N_k, exact_f)
end

@testset "Thermodynamic cycles" begin
    @testset "cycle arithmetic" begin
        src_a = ToyFreeEnergies([0.0, 2.0, 5.0]u"kJ/mol", [0.0, 0.3, 0.4]u"kJ/mol")
        src_b = ToyFreeEnergies([1.0, -3.0]u"kJ/mol", [0.1, 0.2]u"kJ/mol")
        cycle = ThermodynamicCycle(
            Leg(src_a, 1 => 3; name=:a),
            Leg(src_b, 2 => 1; coefficient=-2, name=:b);
            correction=1.5u"kJ/mol",
        )
        @test length(cycle.legs) == 2
        @test cycle.legs[2].coefficient === -2.0
        @test cycle.legs[1].name == :a
        @test free_energy(cycle) ≈ (5.0 - 2 * 4.0 + 1.5)u"kJ/mol"
        @test free_energy_uncertainty(cycle) ≈ sqrt(0.4^2 + 4 * (0.1^2 + 0.2^2))u"kJ/mol"

        # Compatible units are converted to the units of the first leg
        src_kcal = ToyFreeEnergies([0.0, 1.0]u"kcal/mol", [0.0, 0.5]u"kcal/mol")
        mixed = ThermodynamicCycle(Leg(src_a, 1 => 2), Leg(src_kcal, 1 => 2);
                                   correction=1.0u"kcal/mol")
        @test unit(free_energy(mixed)) == u"kJ/mol"
        @test free_energy(mixed) ≈ (2.0 + 2 * 4.184)u"kJ/mol"
        @test free_energy_uncertainty(mixed) ≈ hypot(0.3, 0.5 * 4.184)u"kJ/mol"

        # Without a correction, unitless sources work too
        unitless = ThermodynamicCycle(Leg(ToyFreeEnergies([0.0, 3.0], [0.0, 0.4]), 2 => 1))
        @test free_energy(unitless) ≈ -3.0
        @test free_energy_uncertainty(unitless) ≈ 0.4

        @test_throws ArgumentError ThermodynamicCycle()
        @test_throws ArgumentError Leg(src_a, 1.0 => 2)
        @test_throws ArgumentError Leg(src_a, 1 => :b)
        src_nm = ToyFreeEnergies([0.0, 1.0]u"nm", [0.0, 0.1]u"nm")
        @test_throws ArgumentError free_energy(ThermodynamicCycle(Leg(src_a, 1 => 2),
                                                                  Leg(src_nm, 1 => 2)))
        @test_throws ArgumentError free_energy_uncertainty(
                                        ThermodynamicCycle(Leg(src_a, 1 => 2),
                                                           Leg(src_nm, 1 => 2)))
        @test_throws ArgumentError free_energy(ThermodynamicCycle(Leg(src_a, 1 => 2);
                                                                  correction=1.0u"K"))
        @test_throws ArgumentError free_energy(ThermodynamicCycle(
                        Leg(ToyFreeEnergies([0.0, 3.0], [0.0, 0.4]), 1 => 2);
                        correction=1.0u"kJ/mol"))
    end

    @testset "standard-state correction" begin
        T, P = 298.15u"K", 1.0u"bar"
        correction = standard_state_correction(T, P)
        @test unit(correction) == u"kJ/mol"
        # RT ln(RTc/P), about 7.96 kJ/mol at 298.15 K and 1 bar
        @test correction ≈ Unitful.R * T * log(ustrip(upreferred(Unitful.R * T * u"mol/L" / P)))
        @test isapprox(correction, 7.9585u"kJ/mol"; atol=1e-4u"kJ/mol")
        @test standard_state_correction(T, P; concentration=2u"mol/L") ≈
              correction + Unitful.R * T * log(2)
    end

    @testset "TSSState legs" begin
        names = ["decoupled", "s2", "s3", "coupled"]
        f_vac, f_solv = [0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 2.5, 4.0]
        vac = make_fixed_tss_state(make_cycle_thermo_states(names=names), f_vac)
        solv = make_fixed_tss_state(make_cycle_thermo_states(names=names), f_solv; noise=0.8)
        kBT = uconvert(u"kJ/mol", Unitful.R * 298.0u"K")
        @test free_energy_difference(vac, 1, 4) ≈ 6.0 * kBT
        @test free_energy_difference(vac, "coupled", "s2") ≈ -5.0 * kBT

        cycle = ThermodynamicCycle(
            Leg(vac, "coupled" => "decoupled"; name=:vacuum),
            Leg(solv, 4 => 1; coefficient=-1, name=:solvated);
            correction=standard_state_correction(298.0u"K", 1u"bar"),
        )
        jk_vac = Molly.tss_free_energy_uncertainties(vac)
        jk_solv = Molly.tss_free_energy_uncertainties(solv)
        f_vac_rep, f_solv_rep = Molly.tss_free_energies(vac), Molly.tss_free_energies(solv)
        β_vac, β_solv = vac.state_space.betas[1], solv.state_space.betas[1]
        manual = (f_vac_rep[1] - f_vac_rep[4]) / β_vac - (f_solv_rep[1] - f_solv_rep[4]) / β_solv
        @test free_energy(cycle) ≈ manual * u"kJ/mol" +
                                   standard_state_correction(298.0u"K", 1u"bar")
        @test free_energy(cycle) ≈ (-6.0 + 4.0) * kBT +
                                   standard_state_correction(298.0u"K", 1u"bar")
        σ_vac, σ_solv = jk_vac.standard_errors[4] / β_vac, jk_solv.standard_errors[4] / β_solv
        @test σ_vac > 0 && σ_solv > 0
        @test free_energy_uncertainty(cycle) ≈ hypot(σ_vac, σ_solv) * u"kJ/mol"
        # The jackknife reference state is the leg's `from` state
        @test free_energy_difference_uncertainty(vac, 2, 4) ≈
              Molly.tss_free_energy_uncertainties(vac; reference_state=2).standard_errors[4] /
              β_vac * u"kJ/mol"

        @test_throws ArgumentError free_energy_difference(vac, 1, 5)
        @test_throws ArgumentError free_energy_difference(vac, "absent", "coupled")
        # Default names are equal for states at the same temperature and pressure
        unnamed = make_fixed_tss_state(make_cycle_thermo_states(), f_vac)
        @test_throws ArgumentError free_energy_difference(unnamed, "state_T298.0 K", 1)

        # A leg between states at different temperatures needs a single kBT
        temps = [298.0, 298.0, 310.0, 310.0]u"K"
        mixed = make_fixed_tss_state(make_cycle_thermo_states(temps=temps), f_vac)
        @test free_energy_difference(mixed, 3, 4) ≈
              3.0 * uconvert(u"kJ/mol", Unitful.R * 310.0u"K")
        @test_throws ArgumentError free_energy_difference(mixed, 1, 4)
        @test_throws ArgumentError free_energy_difference_uncertainty(mixed, 1, 4)
        # Legs at different temperatures combine, each with its own kBT
        two_temps = ThermodynamicCycle(Leg(mixed, 1 => 2), Leg(mixed, 3 => 4))
        @test free_energy(two_temps) ≈ 1.0 * kBT +
                                       3.0 * uconvert(u"kJ/mol", Unitful.R * 310.0u"K")
    end

    @testset "AWHState legs" begin
        names = ["a", "b", "c", "d"]
        awh = AWHState(make_cycle_thermo_states(names=names); n_bias=10)
        awh.f .= [0.0, 1.0, 3.0, 6.0]
        kBT = uconvert(u"kJ/mol", Unitful.R * 298.0u"K")
        @test free_energy_difference(awh, "d", "a") ≈ -6.0 * kBT
        cycle = ThermodynamicCycle(Leg(awh, 1 => 4), Leg(awh, 2 => 3; coefficient=-1))
        @test free_energy(cycle) ≈ 4.0 * kBT
        @test_throws ArgumentError free_energy_difference_uncertainty(awh, 1, 4)
        @test_throws ArgumentError free_energy_uncertainty(cycle)
    end

    @testset "MBARResult legs" begin
        rng = Xoshiro(12)
        kBT = fill(2.5u"kJ/mol", 9)
        samples = harmonic_umbrella_samples(rng)
        result = Molly.MBARResult(samples.u, samples.win_of, samples.N_k, kBT;
                                  names=["w$k" for k in 1:9])
        f, _ = Molly.iterate_mbar(samples.u, samples.win_of, samples.N_k)
        @test result.f == f
        @test free_energy_difference(result, 2, 9) ≈ (f[9] - f[2]) * 2.5u"kJ/mol"
        @test free_energy_difference(result, "w9", "w2") ≈ (f[2] - f[9]) * 2.5u"kJ/mol"
        @test free_energy_difference_uncertainty(result, 3, 3) == 0.0u"kJ/mol"
        @test free_energy_difference_uncertainty(result, 1, 9) ≈
              free_energy_difference_uncertainty(result, 9, 1)
        @test abs(free_energy_difference(result, 1, 9) - samples.exact_f[9] * 2.5u"kJ/mol") <
              4 * free_energy_difference_uncertainty(result, 1, 9)
        # Shifted reduced potentials give the same result
        shifted = Molly.MBARResult(samples.u .- minimum(samples.u; dims=2), samples.win_of,
                                   samples.N_k, kBT)
        @test shifted.f ≈ result.f
        @test shifted.covariance ≈ result.covariance rtol=1e-6

        # The asymptotic standard error matches the spread over independent data sets
        ΔF = Float64[]
        σ = Float64[]
        for _ in 1:30
            s = harmonic_umbrella_samples(rng)
            r = Molly.MBARResult(s.u, s.win_of, s.N_k, kBT)
            push!(ΔF, ustrip(free_energy_difference(r, 1, 9)))
            push!(σ, ustrip(free_energy_difference_uncertainty(r, 1, 9)))
        end
        @test std(ΔF) ≈ mean(σ) rtol=0.3
        @test mean(ΔF) ≈ samples.exact_f[9] * 2.5 atol=3 * mean(σ) / sqrt(30)

        @test_throws ArgumentError free_energy_difference(
                    Molly.MBARResult(samples.u, samples.win_of, samples.N_k, kBT), "w1", 2)
        mixed_T = Molly.MBARResult(samples.u, samples.win_of, samples.N_k,
                                   [fill(2.5u"kJ/mol", 8); 2.6u"kJ/mol"])
        @test_throws ArgumentError free_energy_difference(mixed_T, 1, 9)
        @test_throws DimensionMismatch Molly.MBARResult(samples.u, samples.win_of,
                                                        samples.N_k, kBT[1:8])

        # From ThermoStates, through assemble_mbar_inputs
        states = make_cycle_thermo_states(names=["a", "b", "c", "d"])
        coords_k = [[copy(states[k].system.coords)] for k in 1:4]
        boundaries_k = [[states[k].system.boundary] for k in 1:4]
        input = Molly.assemble_mbar_inputs(coords_k, boundaries_k, states)
        from_states = Molly.MBARResult(input, states)
        @test from_states.names == ["a", "b", "c", "d"]
        @test from_states.kBT ≈ fill(uconvert(u"kJ/mol", Unitful.R * 298.0u"K"), 4)
        @test from_states.f ≈ first(Molly.iterate_mbar(input.u, input.win_of, input.N))
    end

    @testset "PMF legs" begin
        samples = harmonic_umbrella_samples(Xoshiro(13))
        f, logN = Molly.iterate_mbar(samples.u, samples.win_of, samples.N_k)
        R_k = [samples.x[samples.win_of .== k] for k in 1:9]
        u_target = 0.5 .* samples.x .^ 2
        kBT = 2.5u"kJ/mol"
        pmf = Molly.pmf_with_uncertainty(samples.u, u_target, f, samples.N_k, logN, R_k;
                                         edges=-1.5:0.25:1.5, kBT=kBT)
        @test free_energy_difference(pmf, 2, 10) ≈ (pmf.F[10] - pmf.F[2]) * kBT
        @test free_energy_difference_uncertainty(pmf, 2, 10) ≈
              hypot(pmf.sigma_F[2], pmf.sigma_F[10]) * kBT
        # Harmonic PMF, F(x) = x²/2 in kBT
        @test abs(free_energy_difference(pmf, 2, 10) -
                  0.5 * (pmf.centers[10]^2 - pmf.centers[2]^2) * kBT) <
              4 * free_energy_difference_uncertainty(pmf, 2, 10)

        reduced = Molly.pmf_with_uncertainty(samples.u, u_target, f, samples.N_k, logN, R_k;
                                             edges=-1.5:0.25:1.5)
        @test_throws ArgumentError free_energy_difference(reduced, 2, 10)
        @test_throws ArgumentError free_energy_difference(pmf, "a", 10)
        @test_throws ArgumentError free_energy_difference(pmf, 1, 13)
    end
end
