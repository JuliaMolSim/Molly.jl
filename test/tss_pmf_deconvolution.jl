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
