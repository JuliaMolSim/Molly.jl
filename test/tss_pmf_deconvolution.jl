function make_tss_pmf_thermo_states(; n_atoms=4, n_states=4)
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"

    thermo_states = ThermoState[]
    for restraint_center in range(0.2, 0.8; length=n_states)
        atoms = [Atom(mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1",
                      λ=1.0) for _ in 1:n_atoms]
        restraint = HarmonicPositionRestraint(
            k=10.0u"kJ * mol^-1 * nm^-2",
            x0=SVector(restraint_center, 0.0, 0.0)u"nm",
        )
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            specific_inter_lists=(InteractionList1Atoms([1], [restraint]),),
        )
        intg = Langevin(dt=0.001u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(sys, intg; temperature=temp))
    end
    return thermo_states
end

function make_heterogeneous_general_thermo_states()
    n_atoms = 4
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"
    atoms = [Atom(mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1",
                  λ=1.0) for _ in 1:n_atoms]
    cv = CalcDist([1], [2], CalcSingleDist(), :wrap)
    bias = BiasPotential(cv, SquareBias(10.0u"kJ * mol^-1 * nm^-2", 0.4u"nm"))
    biased_sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        general_inters=(bias,),
    )
    unbiased_sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        general_inters=(),
    )
    intg = Langevin(dt=0.001u"ps", temperature=temp, friction=0.1u"ps^-1")
    return [
        ThermoState(biased_sys, intg; temperature=temp),
        ThermoState(unbiased_sys, intg; temperature=temp),
    ]
end

@testset "TSS PMF deconvolution" begin
    @testset "online PMF accumulator" begin
        acc = Molly.OnlinePMFAccumulator(((0.0,), (2.0,), (2,)); T=Float64)
        Molly.accumulate!(acc, 0.25, log(2.0))
        Molly.accumulate!(acc, 1.25, log(1.0))
        Molly.accumulate!(acc, 3.0, log(1.0))

        @test acc.total_samples == 3
        @test acc.accepted_samples == 2
        @test acc.out_of_grid_samples == 1
        @test acc.counts == [1, 1]

        result = Molly.pmf(acc)
        @test result.F[1] ≈ 0.0
        @test result.F[2] ≈ log(2.0)
        @test result.p ≈ [2 / 3, 1 / 3]
        @test Molly.total_effective_samples(acc) ≈ 9 / 5
        @test_throws ArgumentError Molly.OnlinePMFAccumulator(((0.0,), (1.0,), (0,)); T=Float64)
        @test_throws DimensionMismatch Molly.accumulate!(acc, (0.5, 0.5), 0.0)
    end

    @testset "sampled TSS PMF deconvolution" begin
        thermo_states = make_tss_pmf_thermo_states(n_states=3)
        tss_state = Molly.TSSState(thermo_states)
        coupling = (xi, state_i) -> 0.25 * abs(xi[1] - state_i)
        deconv = Molly.PMFDeconvolution(
            tss_state;
            grid=(0.0, 3.0, 3),
            cv=active_state -> (0.5,),
            coupling=coupling,
        )

        estimator = first(tss_state.estimators)
        sample = Molly.collect_tss_pmf_deconvolution_sample(
            deconv,
            estimator,
            tss_state.active_state;
            window_offset=0.0,
        )
        @test length(sample.value) == 1
        @test length(sample.log_bin_weights) == 3
        @test all(isfinite, sample.log_bin_weights)

        Molly.accumulate_pmf_deconvolution!(deconv.backend.accumulator, sample)
        acc = deconv.backend.accumulator
        @test acc.accepted_samples == 1
        @test count(isfinite, acc.log_numerator_sums) == 1
        @test all(isfinite, Molly.pmf(deconv).F[isfinite.(Molly.pmf(deconv).F)])
        @test_throws ArgumentError Molly.PMFDeconvolution(
            tss_state;
            grid=(0.0, 3.0, 3),
            cv=active_state -> (0.5,),
            coupling=coupling,
            free_energies=zeros(3),
        )
    end

    @testset "TSS deconvolution arithmetic recovers exact PMF" begin
        thermo_states = make_tss_pmf_thermo_states(n_states=3)
        tss_state = Molly.TSSState(thermo_states)
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
        window_offset = -0.7

        tss_log_bin_weights = zeros(Float64, 5)
        Molly.tss_pmf_log_bin_weights!(
            tss_log_bin_weights,
            deconv.backend,
            estimator;
            window_offset=window_offset,
        )

        manual_log_state_weights = estimator.f .+ estimator.log_dens .- window_offset
        manual_log_bin_weights = similar(tss_log_bin_weights)
        Molly.pmf_log_bin_weights!(
            manual_log_bin_weights,
            deconv.backend.log_coupling_matrix,
            manual_log_state_weights;
            state_indices=estimator.state_indices,
        )
        @test tss_log_bin_weights ≈ manual_log_bin_weights

        target_probability = [0.12, 0.27, 0.08, 0.33, 0.20]
        biased_probability = target_probability .* exp.(-tss_log_bin_weights)
        biased_probability ./= sum(biased_probability)

        acc = Molly.SampledPMFDeconvolutionAccumulator(deconv.backend.grid)
        for (bin_i, center) in enumerate(0.5:1.0:4.5)
            Molly.accumulate_pmf_deconvolution!(
                acc,
                (center,),
                tss_log_bin_weights;
                log_reweight=log(biased_probability[bin_i]),
            )
        end

        recovered_probability = Molly.sampled_pmf_probability(acc)
        expected_F = -log.(target_probability)
        expected_F .-= minimum(expected_F)
        recovered_pmf = Molly.pmf_result_from_sampled_deconvolution(acc)

        @test recovered_probability ≈ target_probability atol=1e-12 rtol=1e-12
        @test recovered_pmf.F ≈ expected_F atol=1e-12 rtol=1e-12
    end

    @testset "sampled PMF accumulator merging" begin
        grid = Molly.PMFGrid((0.0, 2.0, 2); T=Float64)
        left = Molly.SampledPMFDeconvolutionAccumulator(grid)
        right = Molly.SampledPMFDeconvolutionAccumulator(grid)
        merged = Molly.SampledPMFDeconvolutionAccumulator(grid)
        direct = Molly.SampledPMFDeconvolutionAccumulator(grid)

        Molly.accumulate_pmf_deconvolution!(left, (0.5,), [log(2.0), 0.0])
        Molly.accumulate_pmf_deconvolution!(right, (1.5,), [0.0, log(3.0)])
        Molly.accumulate_pmf_deconvolution!(direct, (0.5,), [log(2.0), 0.0])
        Molly.accumulate_pmf_deconvolution!(direct, (1.5,), [0.0, log(3.0)])
        Molly.merge_pmf_deconvolution_accumulator!(merged, left)
        Molly.merge_pmf_deconvolution_accumulator!(merged, right)

        @test merged.counts == direct.counts
        @test merged.total_samples == direct.total_samples
        @test merged.accepted_samples == direct.accepted_samples
        @test Molly.sampled_pmf_probability(merged) ≈ Molly.sampled_pmf_probability(direct)
    end

    @testset "sampled PMF bin quality controls gauge and reporting" begin
        grid = Molly.PMFGrid((0.0, 3.0, 3); T=Float64)
        acc = Molly.SampledPMFDeconvolutionAccumulator(grid)

        for _ in 1:20
            Molly.accumulate_pmf_deconvolution!(acc, (0.5,), [0.0, 0.0, 0.0])
        end
        Molly.accumulate_pmf_deconvolution!(acc, (1.5,), [0.0, log(1000.0), 0.0])
        for _ in 1:19
            Molly.accumulate_pmf_deconvolution!(acc, (1.5,), [0.0, 0.0, 0.0])
        end
        Molly.accumulate_pmf_deconvolution!(acc, (2.5,), [0.0, 0.0, log(2.0)])

        quality = Molly.pmf_bin_quality(
            acc;
            min_count=20,
            min_ess=5.0,
            max_weight_fraction=0.5,
        )
        @test quality.counts == [20, 20, 1]
        @test quality.reliable == [true, false, false]
        @test quality.ess[1] ≈ 20.0
        @test quality.ess[2] < 5.0
        @test quality.maxfrac[2] > 0.5

        raw = Molly.pmf_result_from_sampled_deconvolution(acc)
        reported = Molly.pmf_result_from_sampled_deconvolution(
            acc;
            quality=quality,
            gauge_reliable_only=true,
            mask_unreliable=true,
        )
        @test raw.F[2] ≈ 0.0
        @test reported.F[1] ≈ 0.0
        @test isnan(reported.F[2])
        @test isnan(reported.F[3])
        @test reported.p ≈ raw.p
        reported_from_mask = Molly.pmf_result_from_sampled_deconvolution(
            acc;
            quality=quality.reliable,
            gauge_reliable_only=true,
            mask_unreliable=true,
        )
        @test isequal(reported_from_mask.F, reported.F)
        @test reported_from_mask.p ≈ raw.p

        no_reliable = Molly.pmf_bin_quality(acc; min_count=1000)
        @test_throws ArgumentError Molly.pmf_result_from_sampled_deconvolution(
            acc;
            quality=no_reliable,
            gauge_reliable_only=true,
        )
    end

    @testset "TSS PMF history retention" begin
        thermo_states = make_tss_pmf_thermo_states(n_states=2)
        tss_state = Molly.TSSState(
            thermo_states;
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.5, phi=2.0),
        )
        deconv = Molly.PMFDeconvolution(
            tss_state;
            grid=(0.0, 2.0, 2),
            cv=active_state -> (0.5,),
            coupling=(xi, state_i) -> 0.0,
        )
        old_sample = Molly.PMFDeconvolutionSample((0.5,), [log(1000.0), 0.0], 0.0)
        new_sample = Molly.PMFDeconvolutionSample((1.5,), [0.0, 0.0], 0.0)
        make_obs(sample) = Molly.WindowedTSSObservation(
            1,
            1,
            1,
            1,
            0.0,
            zeros(2),
            fill(0.5, 2),
            nothing,
            Any[sample],
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
        @test length(deconv.backend.epoch_accumulators) == 2

        Molly.drop_old_tss_pmf_deconvolution_epochs!(deconv, tss_state, 8)
        @test length(deconv.backend.epoch_accumulators) == 1
        @test Molly.pmf_deconvolution_accumulator(deconv).accepted_samples == 1
        result = Molly.pmf(deconv)
        @test result.p ≈ [0.0, 1.0]
    end

    @testset "automatic and windowed TSS deconvolution" begin
        torsion_states = ThermoState[]
        boundary = CubicBoundary(2.0u"nm")
        coords = place_atoms(4, boundary; min_dist=0.3u"nm")
        cv = CalcTorsion([1, 2, 3, 4], :pbc, true)
        for target in range(-0.5, 0.5; length=3)
            atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm",
                          ϵ=0.2u"kJ * mol^-1", λ=1.0) for _ in 1:4]
            bias = BiasPotential(
                cv,
                PeriodicFlatBottomBias(10.0u"kJ * mol^-1", 0.1, target),
            )
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

        graph = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        windowed_state = Molly.TSSState(
            make_tss_pmf_thermo_states(n_states=4);
            graph=graph,
            first_state=1,
            first_window=1,
            initial_f=[0.0, 1.0, 2.0, 3.0],
            ETA=1.0,
            dens_reg=1e-4,
        )
        windowed_deconv = Molly.PMFDeconvolution(
            windowed_state;
            grid=(0.0, 4.0, 4),
            cv=active_state -> (0.5,),
            coupling=(xi, state_i) -> 0.25 * abs(xi[1] - state_i),
        )
        sample = Molly.collect_tss_pmf_deconvolution_sample(
            windowed_deconv,
            windowed_state,
            first(windowed_state.estimators),
            windowed_state.active_state;
            window_offset=windowed_state.coupling.window_offsets[1],
        )
        acc = Molly.SampledPMFDeconvolutionAccumulator(windowed_deconv.backend.grid)
        Molly.accumulate_pmf_deconvolution!(acc, (0.5,), sample.log_bin_weights)
        Molly.accumulate_pmf_deconvolution!(acc, (1.5,), sample.log_bin_weights)
        pmf = Molly.pmf_result_from_sampled_deconvolution(acc)
        @test sum(pmf.p) ≈ 1.0
        @test count(>(0), pmf.p) == 2

        sim_state = Molly.TSSState(make_tss_pmf_thermo_states(n_states=3))
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
        Molly.simulate!(
            sim;
            rng=MersenneTwister(2),
            n_threads=1,
            replica_parallel=:serial,
            show_progress=false,
        )
        @test sim_deconv.backend.accumulator.total_samples == 1
        @test sim_deconv.backend.accumulator.accepted_samples == 1
    end

    @testset "partitioned workspace and MBAR assembly" begin
        hetero_states = make_heterogeneous_general_thermo_states()
        coords = hetero_states[1].system.coords
        boundary = hetero_states[1].system.boundary
        partitioned = Molly.PartitionedReducedPotentialWorkspace(hetero_states)
        energies = Molly.evaluate_energy_all!(partitioned.partition, coords, boundary)
        @test energies[1] ≈ potential_energy(hetero_states[1].system)
        @test energies[2] ≈ potential_energy(hetero_states[2].system)

        thermo_states = make_tss_pmf_thermo_states()
        coords_k = [[copy(thermo_states[1].system.coords)], [copy(thermo_states[2].system.coords)]]
        boundaries_k = [[thermo_states[1].system.boundary], [thermo_states[2].system.boundary]]
        sampled_states = thermo_states[1:2]
        target_state = thermo_states[4]
        partitioned_inputs = Molly.assemble_mbar_inputs(
            coords_k,
            boundaries_k,
            sampled_states;
            target_state=target_state,
            shift=true,
        )
        full_inputs = Molly.assemble_mbar_inputs_full(
            coords_k,
            boundaries_k,
            sampled_states;
            target_state=target_state,
            shift=true,
        )
        @test partitioned_inputs.u ≈ full_inputs.u
        @test partitioned_inputs.u_target ≈ full_inputs.u_target
        @test partitioned_inputs.shifts ≈ full_inputs.shifts
        @test partitioned_inputs.win_of == full_inputs.win_of
    end
end
