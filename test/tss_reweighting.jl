using Molly
using Random
using Test
using Unitful

function make_tss_reweighting_thermo_states(; n_atoms=4, n_states=4)
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"

    thermo_states = ThermoState[]
    for (state_i, restraint_center) in enumerate(range(0.2, 0.8; length=n_states))
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

tss_reweighting_x_observable(context) =
    ustrip(context.active_state.active_sys.coords[1][1])

@testset "TSS on-the-fly reweighting" begin
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
        @test Molly.effective_samples(acc) ≈ [1.0, 1.0]
        @test Molly.total_effective_samples(acc) ≈ 9 / 5
        @test Molly.max_weight_fraction(acc) ≈ [1.0, 1.0]

        acc_same_bin = Molly.OnlinePMFAccumulator((0.0, 1.0, 1); T=Float64)
        Molly.accumulate!(acc_same_bin, 0.25, log(2.0))
        Molly.accumulate!(acc_same_bin, 0.75, log(1.0))
        @test Molly.max_weight_fraction(acc_same_bin) ≈ [2 / 3]

        acc2 = Molly.OnlinePMFAccumulator(((0.0, 0.0), (1.0, 1.0), (2, 2)); T=Float64)
        Molly.accumulate!(acc2, (0.25, 0.75), 0.0)
        result2 = Molly.pmf(acc2)
        @test size(result2.F) == (2, 2)
        @test result2.F[1, 2] == 0.0
        @test isinf(result2.F[1, 1])
        @test isinf(result2.F[2, 1])
        @test isinf(result2.F[2, 2])

        @test_throws ArgumentError Molly.OnlinePMFAccumulator(((0.0,), (1.0,), (0,)); T=Float64)
        @test_throws ArgumentError Molly.accumulate!(acc, 0.5, Inf)
        @test_throws DimensionMismatch Molly.accumulate!(acc, (0.5, 0.5), 0.0)
    end

    @testset "TSS reweighting target validation" begin
        thermo_states = make_tss_reweighting_thermo_states()
        target = Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
            device_policy=:cpu,
            name=:x,
        )
        @test target.target_state === thermo_states[1]
        @test target.device_policy == :cpu
        @test target.name == :x
        @test target.sample_stride == 10

        @test_throws ArgumentError Molly.TSSReweightingTarget(
            thermo_states[1];
            grid=(0.0, 2.0, 4),
        )
        @test_throws ArgumentError Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            cv=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
        )
        @test_throws ArgumentError Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
            device_policy=:gpu,
        )
        @test_throws ArgumentError Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
            sample_stride=0,
        )
        @test_throws ArgumentError Molly.TSSReplayLogger(
            observable=tss_reweighting_x_observable,
            n_steps=0,
        )
    end

    @testset "partitioned reduced-potential workspace matches full target" begin
        thermo_states = make_tss_reweighting_thermo_states()
        coords = thermo_states[1].system.coords
        boundary = thermo_states[1].system.boundary
        target_state = thermo_states[3]

        partitioned = Molly.PartitionedReducedPotentialWorkspace([thermo_states[1], target_state])
        full = Molly.ReducedPotentialWorkspace(target_state)

        u_partitioned = Molly.reduced_potential(partitioned, coords, boundary, 2)
        u_full = Molly.reduced_potential(full, coords, boundary)
        @test u_partitioned ≈ u_full

        out = zeros(Float64, 2)
        Molly.reduced_potentials!(out, partitioned, coords, boundary, 1:2)
        @test out[2] ≈ u_full
    end

    @testset "partitioned workspace supports heterogeneous general interactions" begin
        thermo_states = make_heterogeneous_general_thermo_states()
        coords = thermo_states[1].system.coords
        boundary = thermo_states[1].system.boundary

        partitioned = Molly.PartitionedReducedPotentialWorkspace(thermo_states)
        full_target = Molly.ReducedPotentialWorkspace(thermo_states[2])

        energies = Molly.evaluate_energy_all!(partitioned.partition, coords, boundary)
        direct = [potential_energy(ts.system) for ts in thermo_states]
        @test energies[1] ≈ direct[1]
        @test energies[2] ≈ direct[2]

        out = zeros(Float64, 2)
        Molly.reduced_potentials!(out, partitioned, coords, boundary, 1:2)
        @test out[2] ≈ Molly.reduced_potential(full_target, coords, boundary)

        target_workspace = Molly.TSSTargetReducedPotentialWorkspace(
            partitioned,
            nothing,
            2,
            :partitioned,
        )
        @test Molly._tss_target_reduced_potential(target_workspace, coords, boundary) ≈
              Molly.reduced_potential(full_target, coords, boundary)
    end

    @testset "partitioned MBAR assembly matches full potentials" begin
        thermo_states = make_tss_reweighting_thermo_states()
        coords_k = [
            [copy(thermo_states[1].system.coords)],
            [copy(thermo_states[2].system.coords)],
        ]
        boundaries_k = [
            [thermo_states[1].system.boundary],
            [thermo_states[2].system.boundary],
        ]
        sampled_states = thermo_states[1:2]
        target_state = thermo_states[4]

        partitioned = Molly.assemble_mbar_inputs(
            coords_k,
            boundaries_k,
            sampled_states;
            target_state=target_state,
        )
        full = Molly._assemble_mbar_inputs_full(
            coords_k,
            boundaries_k,
            sampled_states;
            target_state=target_state,
        )
        @test partitioned.u ≈ full.u
        @test partitioned.u_target ≈ full.u_target
        @test partitioned.N == full.N
        @test partitioned.win_of == full.win_of

        partitioned_shifted = Molly.assemble_mbar_inputs(
            coords_k,
            boundaries_k,
            sampled_states;
            target_state=target_state,
            shift=true,
        )
        full_shifted = Molly._assemble_mbar_inputs_full(
            coords_k,
            boundaries_k,
            sampled_states;
            target_state=target_state,
            shift=true,
        )
        @test partitioned_shifted.u ≈ full_shifted.u
        @test partitioned_shifted.u_target ≈ full_shifted.u_target
        @test partitioned_shifted.shifts ≈ full_shifted.shifts

        hetero_states = make_heterogeneous_general_thermo_states()
        hetero_coords_k = [[copy(hetero_states[1].system.coords)]]
        hetero_boundaries_k = [[hetero_states[1].system.boundary]]

        hetero_partitioned = Molly.assemble_mbar_inputs(
            hetero_coords_k,
            hetero_boundaries_k,
            hetero_states[1:1];
            target_state=hetero_states[2],
        )
        hetero_full = Molly._assemble_mbar_inputs_full(
            hetero_coords_k,
            hetero_boundaries_k,
            hetero_states[1:1];
            target_state=hetero_states[2],
        )
        @test hetero_partitioned.u ≈ hetero_full.u
        @test hetero_partitioned.u_target ≈ hetero_full.u_target
    end

    @testset "windowed TSS reweights every x/k sample" begin
        thermo_states = make_tss_reweighting_thermo_states()
        graph = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        state = Molly.TSSState(thermo_states;
            graph=graph,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        reweighting = Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
            device_policy=:cpu,
            sample_stride=1,
        )
        sim = Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=3,
            self_adjustment_steps=2,
            log_freq=1,
            reweighting=reweighting,
        )
        Molly.simulate!(sim; rng=MersenneTwister(1))

        acc = sim.reweighting.accumulator
        @test state.iteration == 3
        @test sum(est.iteration for est in state.estimators) == 3
        @test eltype(acc.log_weight_sums) == Float64
        @test acc.total_samples == 6
        @test acc.accepted_samples + acc.out_of_grid_samples == acc.total_samples
        @test acc.accepted_samples > 0

        result = Molly.tss_reweighted_pmf(sim)
        @test any(isfinite, result.F)
        @test sum(result.p) ≈ 1.0

        no_reweighting_sim = Molly.TSSSimulation(
            Molly.TSSState(thermo_states; graph=graph);
            n_md_steps=1,
            n_cycles=0,
        )
        @test_throws ArgumentError Molly.tss_reweighted_pmf(no_reweighting_sim)

        strided_state = Molly.TSSState(thermo_states;
            graph=graph,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        strided_reweighting = Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
            device_policy=:cpu,
            sample_stride=2,
        )
        strided_sim = Molly.TSSSimulation(strided_state;
            n_md_steps=1,
            n_cycles=3,
            self_adjustment_steps=2,
            log_freq=1,
            reweighting=strided_reweighting,
        )
        Molly.simulate!(strided_sim; rng=MersenneTwister(4))
        @test strided_sim.reweighting.accumulator.total_samples == 3
    end

    @testset "multireplica TSS reweighting sample count" begin
        thermo_states = make_tss_reweighting_thermo_states()
        graph = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        state = Molly.TSSState(thermo_states;
            graph=graph,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        reweighting = Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
            device_policy=:cpu,
            sample_stride=1,
        )
        sim = Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=2,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
            reweighting=reweighting,
        )
        Molly.simulate!(sim;
            rng=MersenneTwister(2),
            n_threads=1,
            replica_parallel=:serial,
        )

        @test state.iteration == 2
        @test sum(est.iteration for est in state.estimators) == 4
        @test sim.reweighting.accumulator.total_samples == 8
        @test sim.reweighting.accumulator.accepted_samples > 0
        @test any(isfinite, Molly.tss_reweighted_pmf(sim).F)

        threaded_state = Molly.TSSState(thermo_states;
            graph=graph,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
            history_forgetting=Molly.TSSHistoryForgetting(alpha=0.0, phi=1.2),
        )
        threaded_sim = Molly.TSSSimulation(threaded_state;
            n_md_steps=1,
            n_cycles=1,
            self_adjustment_steps=1,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
            reweighting=reweighting,
        )
        Molly.simulate!(threaded_sim;
            rng=MersenneTwister(3),
            n_threads=2,
            replica_parallel=:threads,
        )

        @test threaded_state.iteration == 1
        @test sum(est.iteration for est in threaded_state.estimators) == 2
        @test threaded_sim.reweighting.accumulator.total_samples == 2
        @test any(isfinite, Molly.tss_reweighted_pmf(threaded_sim).F)
    end

    @testset "frozen TSS replay supports offline reweighting" begin
        thermo_states = make_tss_reweighting_thermo_states()
        graph = Molly.tss_grid_graph((4,); window_size=(2,), periodic=false)
        state = Molly.TSSState(thermo_states;
            graph=graph,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        target = Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
            device_policy=:cpu,
            sample_stride=1,
        )
        replay = Molly.TSSReplayLogger(
            observable=tss_reweighting_x_observable,
            device_policy=:cpu,
            n_steps=1,
        )

        initial_f = [copy(est.f) for est in state.estimators]
        initial_density = [copy(est.density) for est in state.estimators]
        initial_gamma = [copy(est.gamma) for est in state.estimators]
        initial_coupling_f = copy(state.coupling.visit_control_f)

        sim = Molly.TSSSimulation(state;
            n_md_steps=1,
            n_cycles=2,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
            replay_logger=replay,
            frozen=true,
        )
        Molly.simulate!(sim;
            rng=MersenneTwister(5),
            n_threads=1,
            replica_parallel=:serial,
        )

        @test state.iteration == 2
        @test sum(est.iteration for est in state.estimators) == 0
        @test sum(state.window_update_counts) == 4
        @test length(values(replay)) == 8
        @test all(i -> state.estimators[i].f == initial_f[i], eachindex(state.estimators))
        @test all(i -> state.estimators[i].density == initial_density[i], eachindex(state.estimators))
        @test all(i -> state.estimators[i].gamma == initial_gamma[i], eachindex(state.estimators))
        @test state.coupling.visit_control_f == initial_coupling_f

        offline_pmf = Molly.tss_offline_reweighted_pmf(replay, target)
        @test any(isfinite, offline_pmf.F)
        @test sum(offline_pmf.p) ≈ 1.0
        @test all(isfinite, offline_pmf.p)

        mbar_pmf = Molly.tss_mbar_pmf(values(replay), thermo_states, target)
        @test any(isfinite, mbar_pmf.F)
        @test sum(mbar_pmf.p) ≈ 1.0

        compact_state = Molly.TSSState(thermo_states;
            graph=graph,
            first_state=1,
            first_window=1,
            ETA=1.0,
            dens_reg=1e-4,
        )
        compact_replay = Molly.TSSReplayLogger(
            observable=tss_reweighting_x_observable,
            device_policy=:cpu,
            n_steps=1,
            store_coords=false,
        )
        compact_sim = Molly.TSSSimulation(compact_state;
            n_md_steps=1,
            n_cycles=2,
            self_adjustment_steps=2,
            log_freq=1,
            n_replicas=2,
            first_states=[1, 3],
            reweighting=target,
            replay_logger=compact_replay,
            frozen=true,
        )
        Molly.simulate!(compact_sim;
            rng=MersenneTwister(7),
            n_threads=1,
            replica_parallel=:serial,
        )
        @test all(record -> isnothing(record.coords), values(compact_replay))
        @test all(record -> !isnothing(record.log_weight), values(compact_replay))
        compact_pmf = Molly.tss_offline_reweighted_pmf(compact_replay, target)
        @test any(isfinite, compact_pmf.F)
        @test sum(compact_pmf.p) ≈ 1.0

        sparse_online_target = Molly.TSSReweightingTarget(
            thermo_states[1];
            observable=tss_reweighting_x_observable,
            grid=(0.0, 2.0, 4),
            device_policy=:cpu,
            sample_stride=100,
        )
        sparse_compact_replay = Molly.TSSReplayLogger(
            observable=tss_reweighting_x_observable,
            device_policy=:cpu,
            n_steps=1,
            store_coords=false,
        )
        sparse_compact_sim = Molly.TSSSimulation(
            Molly.TSSState(thermo_states; graph=graph);
            n_md_steps=1,
            n_cycles=1,
            self_adjustment_steps=2,
            log_freq=1,
            reweighting=sparse_online_target,
            replay_logger=sparse_compact_replay,
            frozen=true,
        )
        Molly.simulate!(sparse_compact_sim; rng=MersenneTwister(9))
        @test sparse_compact_sim.reweighting.accumulator.total_samples == 0
        @test all(record -> isnothing(record.coords), values(sparse_compact_replay))
        @test all(record -> !isnothing(record.log_weight), values(sparse_compact_replay))
        sparse_compact_pmf = Molly.tss_offline_reweighted_pmf(
            sparse_compact_replay,
            sparse_online_target,
        )
        @test any(isfinite, sparse_compact_pmf.F)
        @test sum(sparse_compact_pmf.p) ≈ 1.0

        dual_replay = Molly.TSSReplayLogger(
            observable=tss_reweighting_x_observable,
            device_policy=:cpu,
            n_steps=1,
            store_coords=true,
        )
        dual_sim = Molly.TSSSimulation(
            Molly.TSSState(thermo_states; graph=graph);
            n_md_steps=1,
            n_cycles=1,
            self_adjustment_steps=2,
            log_freq=1,
            reweighting=target,
            replay_logger=dual_replay,
            frozen=true,
        )
        Molly.simulate!(dual_sim; rng=MersenneTwister(8))
        compact_dual_pmf = Molly.tss_offline_reweighted_pmf(dual_replay, target)
        coordinate_records = [
            Molly.TSSReplayRecord(
                record.sample_index,
                record.replica_index,
                record.iteration,
                record.substep,
                record.active_state,
                record.update_window,
                record.coords,
                record.boundary,
                record.observable_values,
                record.log_den,
                record.window_offset,
                record.aligned_log_den,
                nothing,
                nothing,
            )
            for record in values(dual_replay)
        ]
        coordinate_pmf = Molly.tss_offline_reweighted_pmf(coordinate_records, target)
        @test compact_dual_pmf.p ≈ coordinate_pmf.p
        finite_mask = isfinite.(compact_dual_pmf.F) .& isfinite.(coordinate_pmf.F)
        @test compact_dual_pmf.F[finite_mask] ≈ coordinate_pmf.F[finite_mask]

        coords_free_replay = Molly.TSSReplayLogger(
            observable=tss_reweighting_x_observable,
            device_policy=:cpu,
            n_steps=1,
            store_coords=false,
        )
        coords_free_sim = Molly.TSSSimulation(
            Molly.TSSState(thermo_states; graph=graph);
            n_md_steps=1,
            n_cycles=1,
            replay_logger=coords_free_replay,
            frozen=true,
        )
        Molly.simulate!(coords_free_sim; rng=MersenneTwister(6))
        @test_throws ArgumentError Molly.tss_offline_reweighted_pmf(coords_free_replay, target)
    end
end
