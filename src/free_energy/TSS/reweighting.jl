export
    TSSReweightingTarget,
    TSSReplayLogger,
    tss_reweighted_pmf,
    tss_offline_reweighted_pmf,
    tss_mbar_pmf

struct TSSReweightingTarget{TS, O, G}
    target_state::TS
    observable::O
    grid::G
    device_policy::Symbol
    name::Symbol
    sample_stride::Int
end

"""
    TSSReweightingTarget(target_state::ThermoState; observable, grid,
                         device_policy=:auto, name=:pmf, sample_stride=10)
    TSSReweightingTarget(target_state::ThermoState; cv, grid,
                         cv_transform=identity, device_policy=:auto, name=:pmf,
                         sample_stride=10)

Configure on-the-fly TSS reweighting to a target thermodynamic state.

`target_state` describes the unbiased or otherwise desired state to reweight
sampled configurations to. Pass either a callable `observable` that accepts a
TSS context, or a Molly collective variable as `cv`. `grid` defines the PMF bins
using the same formats accepted by the online PMF accumulator. `sample_stride`
controls how often the online diagnostic accumulator stores an x/k sample;
frozen replay logging is configured separately with `TSSReplayLogger`.
"""
function TSSReweightingTarget(target_state::ThermoState;
                              observable = nothing,
                              cv = nothing,
                              cv_transform = identity,
                              grid = nothing,
                              device_policy = :auto,
                              name = :pmf,
                              sample_stride::Integer = 10)
    isnothing(grid) &&
        throw(ArgumentError("TSSReweightingTarget requires a PMF grid."))
    if isnothing(observable) == isnothing(cv)
        throw(ArgumentError("provide exactly one of observable or cv to TSSReweightingTarget."))
    end
    sample_stride > 0 ||
        throw(ArgumentError("sample_stride must be positive."))

    final_observable = isnothing(cv) ? observable : _TSSCVObservable(cv; transform = cv_transform)
    policy = _validate_tss_observable_device_policy(device_policy)
    return TSSReweightingTarget{typeof(target_state), typeof(final_observable), typeof(grid)}(
        target_state,
        final_observable,
        grid,
        policy,
        Symbol(name),
        Int(sample_stride),
    )
end

mutable struct TSSReweightingRuntime{T, A}
    target::T
    accumulator::A
end

struct TSSReweightingSample{T}
    value::Vector{T}
    log_weight::T
    target_reduced_potential::T
end

struct TSSReplayRecord{C, B, V, T, U, L}
    sample_index::Int
    replica_index::Int
    iteration::Int
    substep::Int
    active_state::Int
    update_window::Int
    coords::C
    boundary::B
    observable_values::V
    log_den::T
    window_offset::T
    aligned_log_den::T
    target_reduced_potential::U
    log_weight::L
end

mutable struct TSSReplayLogger{O}
    observable::O
    device_policy::Symbol
    n_steps::Int
    store_coords::Bool
    history::Vector{Any}
end

"""
    TSSReplayLogger(; observable=nothing, cv=nothing, cv_transform=identity,
                    device_policy=:auto, n_steps=1, store_coords=true)

Collect frozen-TSS replay records for offline reweighting. The logger follows
the Molly logger convention through `values(logger)` and `log_property!`, but is
attached to `TSSSimulation` rather than to `System.loggers` because each
record also needs TSS metadata such as the active state, update window, and
known frozen sampling log density.

Pass the same observable/grid semantics used by `TSSReweightingTarget` when the
records will be used by `tss_offline_reweighted_pmf` or `tss_mbar_pmf`. When a
frozen simulation is also configured with `reweighting=TSSReweightingTarget(...)`,
the replay records store compact log weights and `store_coords=false` is enough
for direct offline PMF reweighting. Coordinate storage is still required for
MBAR or for recomputing target potentials after the run.
"""
function TSSReplayLogger(; observable = nothing,
                         cv = nothing,
                         cv_transform = identity,
                         device_policy = :auto,
                         n_steps::Integer = 1,
                         store_coords::Bool = true)
    if !isnothing(observable) && !isnothing(cv)
        throw(ArgumentError("provide at most one of observable or cv to TSSReplayLogger."))
    end
    n_steps > 0 ||
        throw(ArgumentError("n_steps must be positive."))
    final_observable = isnothing(cv) ? observable : _TSSCVObservable(cv; transform = cv_transform)
    policy = _validate_tss_observable_device_policy(device_policy)
    return TSSReplayLogger(final_observable, policy, Int(n_steps), store_coords, Any[])
end

Base.values(logger::TSSReplayLogger) = logger.history

function log_property!(logger::TSSReplayLogger, sys, buffers, neighbors, step_n;
                       record = nothing, kwargs...)
    isnothing(record) &&
        throw(ArgumentError("TSSReplayLogger must be called with a prebuilt replay record."))
    if step_n % logger.n_steps == 0
        push!(logger.history, record)
    end
    return logger
end

function _prepare_tss_reweighting(::Nothing, state::TSSState)
    return nothing
end

function _prepare_tss_reweighting(target::TSSReweightingTarget,
                                  state::TSSState{FT}) where FT
    return TSSReweightingRuntime(
        target,
        OnlinePMFAccumulator(target.grid; T = Float64),
    )
end

struct TSSTargetReducedPotentialWorkspace{P, F}
    partitioned::P
    full::F
    target_index::Int
    mode::Symbol
end

function _tss_reweighting_workspace(::Nothing, state::TSSState)
    return nothing
end

function _tss_reweighting_workspace(runtime::TSSReweightingRuntime,
                                    state::TSSState)
    target_state = runtime.target.target_state
    reference_state = first(state.state_space.thermo_states)
    partitioned = try
        PartitionedReducedPotentialWorkspace([reference_state, target_state])
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
    if isnothing(partitioned)
        return TSSTargetReducedPotentialWorkspace(
            nothing,
            ReducedPotentialWorkspace(target_state),
            1,
            :full,
        )
    end
    return TSSTargetReducedPotentialWorkspace(partitioned, nothing, 2, :partitioned)
end

function _tss_target_reduced_potential(workspace::TSSTargetReducedPotentialWorkspace,
                                       coords,
                                       boundary;
                                       n_threads::Integer = 1)
    if workspace.mode == :partitioned
        return reduced_potential(
            workspace.partitioned,
            coords,
            boundary,
            workspace.target_index,
        )
    elseif workspace.mode == :full
        return reduced_potential(
            workspace.full,
            coords,
            boundary;
            n_threads = n_threads,
        )
    end
    throw(ArgumentError("unknown TSS target reduced-potential workspace mode " *
                        "$(workspace.mode)."))
end

function _tss_reweighting_value(::Type{FT}, value) where FT
    if value isa Number
        return FT[_tss_numeric_value(FT, value)]
    end

    value_cpu = value isa Tuple ? collect(value) : from_device(value)
    value_cpu isa AbstractVector ||
        throw(ArgumentError("TSS reweighting observable must return a scalar, tuple, " *
                            "or vector, got $(typeof(value))."))

    values = FT[_tss_numeric_value(FT, v) for v in value_cpu]
    !isempty(values) ||
        throw(ArgumentError("TSS reweighting observable must return at least one value."))
    all(isfinite, values) ||
        throw(ArgumentError("TSS reweighting observable contains non-finite values."))
    return values
end

function _tss_observable_value(::Type{FT}, observable, context) where FT
    isnothing(observable) && return nothing
    return _tss_reweighting_value(FT, _evaluate_tss_observable(observable, context))
end

function _tss_reweighting_observable_context(runtime::TSSReweightingRuntime,
                                             estimator::_TSSLocalEstimator,
                                             active_state::ActiveThermoState;
                                             log_den,
                                             history_time::Int,
                                             energies,
                                             reduced_potentials,
                                             n_threads::Int)
    return _TSSObservableContext(
        active_state,
        estimator.state_space,
        estimator.state_indices,
        active_state.active_idx,
        tss_local_index(estimator, active_state.active_idx),
        energies,
        reduced_potentials,
        log_den,
        history_time,
        nothing,
        nothing,
        n_threads,
        runtime.target.device_policy,
    )
end

function _tss_replay_observable_context(logger::TSSReplayLogger,
                                        estimator::_TSSLocalEstimator,
                                        active_state::ActiveThermoState;
                                        log_den,
                                        history_time::Int,
                                        energies,
                                        reduced_potentials,
                                        n_threads::Int)
    return _TSSObservableContext(
        active_state,
        estimator.state_space,
        estimator.state_indices,
        active_state.active_idx,
        tss_local_index(estimator, active_state.active_idx),
        energies,
        reduced_potentials,
        log_den,
        history_time,
        nothing,
        nothing,
        n_threads,
        logger.device_policy,
    )
end

function _collect_tss_reweighting_sample(runtime::TSSReweightingRuntime{<:Any, <:OnlinePMFAccumulator{N, WT}},
                                         target_workspace::TSSTargetReducedPotentialWorkspace,
                                         estimator::_TSSLocalEstimator{FT},
                                         active_state::ActiveThermoState;
                                         log_den,
                                         history_time::Int,
                                         energies,
                                             reduced_potentials,
                                             window_offset,
                                             sample_index::Int,
                                             n_threads::Int,
                                             force::Bool = false) where {N, WT, FT}
    force || sample_index % runtime.target.sample_stride == 0 || return nothing

    context = _tss_reweighting_observable_context(
        runtime,
        estimator,
        active_state;
        log_den = log_den,
        history_time = history_time,
        energies = energies,
        reduced_potentials = reduced_potentials,
        n_threads = n_threads,
    )
    value = _tss_reweighting_value(
        WT,
        _evaluate_tss_observable(runtime.target.observable, context),
    )
    length(value) == N ||
        throw(DimensionMismatch("TSS reweighting observable has $(length(value)) dimensions, " *
                                "but the PMF grid has $(N)."))

    u_target = _tss_target_reduced_potential(
        target_workspace,
        active_state.active_sys.coords,
        active_state.active_sys.boundary;
        n_threads = n_threads,
    )
    aligned_log_den = WT(log_den) - WT(window_offset)
    target_reduced_potential = WT(u_target)
    log_weight = -target_reduced_potential - aligned_log_den
    isfinite(log_weight) ||
        throw(ArgumentError("TSS reweighting produced non-finite log weight " *
                            "$(log_weight) for target $(runtime.target.name)."))

    return TSSReweightingSample(value, log_weight, target_reduced_potential)
end

function _collect_tss_reweighting_sample(::Nothing,
                                         target_workspace,
                                         estimator,
                                         active_state;
                                         kwargs...)
    return nothing
end

function _collect_tss_replay_record(::Nothing,
                                    estimator,
                                    active_state;
                                    kwargs...)
    return nothing
end

function _collect_tss_replay_record(logger::TSSReplayLogger,
                                    estimator::_TSSLocalEstimator{FT},
                                    active_state::ActiveThermoState;
                                    log_den,
                                    history_time::Int,
                                    energies,
                                    reduced_potentials,
                                    window_offset,
                                    sample_index::Int,
                                    replica_index::Int,
                                    update_window::Int,
                                    substep::Int,
                                    reweighting_sample,
                                    n_threads::Int) where FT
    sample_index % logger.n_steps == 0 || return nothing

    context = _tss_replay_observable_context(
        logger,
        estimator,
        active_state;
        log_den = log_den,
        history_time = history_time,
        energies = energies,
        reduced_potentials = reduced_potentials,
        n_threads = n_threads,
    )
    value = _tss_observable_value(FT, logger.observable, context)
    coords = logger.store_coords ? copy(from_device(active_state.active_sys.coords)) : nothing
    boundary = logger.store_coords ? deepcopy(active_state.active_sys.boundary) : nothing
    aligned_log_den = FT(log_den) - FT(window_offset)
    target_reduced_potential = isnothing(reweighting_sample) ?
                               nothing :
                               reweighting_sample.target_reduced_potential
    log_weight = isnothing(reweighting_sample) ? nothing : reweighting_sample.log_weight
    return TSSReplayRecord(
        sample_index,
        replica_index,
        history_time,
        substep,
        active_state.active_idx,
        update_window,
        coords,
        boundary,
        value,
        FT(log_den),
        FT(window_offset),
        aligned_log_den,
        target_reduced_potential,
        log_weight,
    )
end

function _accumulate_tss_reweighting!(::Nothing, observations)
    return nothing
end

function _accumulate_tss_reweighting!(runtime::TSSReweightingRuntime, observations)
    for observation in observations
        for sample in observation.reweighting_samples
            accumulate!(runtime.accumulator, sample.value, sample.log_weight)
        end
    end
    return runtime
end

function _append_tss_replay_records!(::Nothing, observations)
    return nothing
end

function _append_tss_replay_records!(logger::TSSReplayLogger, observations)
    for observation in observations
        for record in observation.replay_records
            log_property!(logger, nothing, nothing, nothing, record.sample_index; record = record)
        end
    end
    return logger
end

tss_reweighted_pmf(runtime::TSSReweightingRuntime; kwargs...) =
    pmf(runtime.accumulator; kwargs...)

function _tss_replay_records(records_or_logger)
    if records_or_logger isa TSSReplayLogger
        return values(records_or_logger)
    end
    return collect(records_or_logger)
end

function _check_tss_replay_records(records)
    isempty(records) &&
        throw(ArgumentError("no TSS replay records were provided."))
    return records
end

function _check_tss_replay_record_for_reweighting(record)
    isnothing(record.observable_values) &&
        throw(ArgumentError("TSS replay record does not contain observable values; " *
                            "configure TSSReplayLogger with observable or cv."))
    if isnothing(record.log_weight)
        _check_tss_replay_record_for_coordinates(record)
    end
    return record
end

function _check_tss_replay_record_for_coordinates(record)
    isnothing(record.coords) &&
        throw(ArgumentError("TSS replay record does not contain coordinates; " *
                            "construct TSSReplayLogger with store_coords=true, or " *
                            "collect replay during a simulation configured with " *
                            "reweighting=TSSReweightingTarget(...)."))
    isnothing(record.boundary) &&
        throw(ArgumentError("TSS replay record does not contain a boundary; " *
                            "construct TSSReplayLogger with store_coords=true."))
    return record
end

function _tss_offline_reweighting_accumulator(records_or_logger,
                                             target::TSSReweightingTarget;
                                             n_threads::Integer = 1)
    records = _check_tss_replay_records(_tss_replay_records(records_or_logger))
    needs_workspace = any(record -> isnothing(record.log_weight), records)
    workspace = needs_workspace ? ReducedPotentialWorkspace(target.target_state) : nothing
    accumulator = OnlinePMFAccumulator(target.grid; T = Float64)

    for record in records
        _check_tss_replay_record_for_reweighting(record)
        log_weight = if isnothing(record.log_weight)
            u_target = reduced_potential(
                workspace,
                record.coords,
                record.boundary;
                n_threads = n_threads,
            )
            -Float64(u_target) - Float64(record.aligned_log_den)
        else
            Float64(record.log_weight)
        end
        accumulate!(accumulator, record.observable_values, log_weight)
    end
    return accumulator
end

"""
    tss_offline_reweighted_pmf(records_or_logger, target; zero=:min, kBT=nothing)

Compute an offline target-state PMF from frozen-TSS replay records using the
known frozen extended-ensemble log density recorded with each sample. If records
contain precomputed log weights from a frozen simulation configured with
`reweighting=TSSReweightingTarget(...)`, no coordinates are needed. Otherwise the
target potential is recomputed from stored coordinates. This is the production
reweighting path; the online accumulator is intended as a cheap diagnostic.
"""
function tss_offline_reweighted_pmf(records_or_logger,
                                    target::TSSReweightingTarget;
                                    n_threads::Integer = 1,
                                    kwargs...)
    accumulator = _tss_offline_reweighting_accumulator(
        records_or_logger,
        target;
        n_threads = n_threads,
    )
    return pmf(accumulator; kwargs...)
end

function _tss_group_replay_records_by_state(records, thermo_states)
    coords_k = [Any[] for _ in eachindex(thermo_states)]
    boundaries_k = [Any[] for _ in eachindex(thermo_states)]
    values_k = [Vector{Float64}[] for _ in eachindex(thermo_states)]

    for record in records
        _check_tss_replay_record_for_reweighting(record)
        _check_tss_replay_record_for_coordinates(record)
        1 <= record.active_state <= length(thermo_states) ||
            throw(ArgumentError("TSS replay active state $(record.active_state) is outside " *
                                "the provided thermo_states range 1:$(length(thermo_states))."))
        push!(coords_k[record.active_state], record.coords)
        push!(boundaries_k[record.active_state], record.boundary)
        push!(values_k[record.active_state], Float64.(record.observable_values))
    end

    sampled_states = findall(k -> !isempty(coords_k[k]), eachindex(coords_k))
    isempty(sampled_states) &&
        throw(ArgumentError("no replay records matched the provided thermo_states."))

    coords_sampled = [coords_k[k] for k in sampled_states]
    boundaries_sampled = [boundaries_k[k] for k in sampled_states]
    values_flat = Vector{Vector{Float64}}()
    for k in sampled_states
        append!(values_flat, values_k[k])
    end
    return coords_sampled, boundaries_sampled, values_flat, sampled_states
end

"""
    tss_mbar_pmf(records_or_logger, thermo_states, target; shift=true, zero=:min)

Build an MBAR target-state PMF from frozen-TSS replay records grouped by their
active thermodynamic state. States with no replay samples are dropped before
solving MBAR, so this estimator should be treated as a support diagnostic when
coverage is sparse.
"""
function tss_mbar_pmf(records_or_logger,
                      thermo_states::Vector{ThermoState},
                      target::TSSReweightingTarget;
                      shift::Bool = true,
                      n_threads::Integer = 1,
                      kwargs...)
    records = _check_tss_replay_records(_tss_replay_records(records_or_logger))
    coords_k, boundaries_k, values_flat, sampled_states =
        _tss_group_replay_records_by_state(records, thermo_states)
    sampled_thermo_states = thermo_states[sampled_states]

    mbar_input = assemble_mbar_inputs(
        coords_k,
        boundaries_k,
        sampled_thermo_states;
        target_state = target.target_state,
        shift = shift,
    )
    f, logN = iterate_mbar(mbar_input.u, mbar_input.win_of, mbar_input.N)
    return mbar_pmf(
        mbar_input.u,
        mbar_input.u_target,
        f,
        mbar_input.N,
        logN,
        values_flat,
        target.grid;
        shifts = mbar_input.shifts,
        kwargs...,
    )
end
