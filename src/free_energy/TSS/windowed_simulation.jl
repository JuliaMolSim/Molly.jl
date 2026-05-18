function log_windowed_tss_stats!(
    stats::WindowedTSSStats{FT},
    state::WindowedTSSState{FT},
    update_window::Int,
    visited_state::Int,
    next_state::Int,
    max_delta_f::FT;
    replica_indices = [1],
    replica_update_windows = [update_window],
    replica_visited_states = [visited_state],
    replica_sampled_next_states = [next_state],
    replica_max_abs_delta_f = [max_delta_f]) where {FT}

    push!(stats.iterations, state.iteration)
    push!(stats.update_window, update_window)
    push!(stats.visited_state, visited_state)
    push!(stats.sampled_next_state, next_state)
    push!(stats.max_abs_delta_f, max_delta_f)
    push!(stats.active_window_history, state.active_window)
    push!(stats.replica_indices, Int.(collect(replica_indices)))
    push!(stats.replica_update_windows, Int.(collect(replica_update_windows)))
    push!(stats.replica_visited_states, Int.(collect(replica_visited_states)))
    push!(stats.replica_sampled_next_states, Int.(collect(replica_sampled_next_states)))
    push!(stats.replica_max_abs_delta_f, FT.(collect(replica_max_abs_delta_f)))
    push!(stats.reported_f_history, windowed_tss_free_energies(state))
    if isnothing(state.coupling)
        push!(stats.visit_control_converged, false)
        push!(stats.visit_control_iterations, 0)
        push!(stats.visit_control_max_abs_residual, FT(NaN))
        push!(stats.window_prob_history, FT[])
        push!(stats.visit_control_f_history, FT[])
    else
        coupling = state.coupling
        push!(stats.visit_control_converged, coupling.converged)
        push!(stats.visit_control_iterations, coupling.iterations)
        push!(stats.visit_control_max_abs_residual, coupling.max_abs_residual)
        push!(stats.window_prob_history, copy(coupling.window_probs))
        push!(stats.visit_control_f_history, copy(coupling.visit_control_f))
    end

    return stats
end

mutable struct WindowedTSSReplica{AS}
    active_state::AS
    active_window::Int
end

struct WindowedTSSObservation{T, V}
    replica_index::Int
    update_window::Int
    visited_state::Int
    sampled_next_state::Int
    log_den::T
    reduced_pot::V
    weights::V
    observable_values::Union{Nothing, Matrix{T}}
end

mutable struct WindowedTSSReplicaWorkspace{P, ST, T, R}
    partition::P
    energies::Vector{ST}
    reduced_pot::Vector{T}
    weights::Vector{T}
    scratch::Vector{T}
    log_state_bias::Vector{T}
    rng::R
end

mutable struct WindowedTSSSimulation{S, R, W}
    state::S
    replicas::Vector{R}
    replica_workspaces::Vector{W}
    n_md_steps::Int
    n_cycles::Int
    self_adjustment_steps::Int
    log_freq::Int
end

function WindowedTSSSimulation(state::WindowedTSSState;
                               n_md_steps::Int,
                               n_cycles::Int,
                               self_adjustment_steps = 1,
                               log_freq::Int = 1000,
                               n_replicas = nothing,
                               first_states = nothing,
                               first_windows = nothing,
                               replica_active_states = nothing)

    n_md_steps > 0 || throw(ArgumentError("n_md_steps must be positive."))
    n_cycles >= 0 || throw(ArgumentError("n_cycles must be non-negative."))
    self_adjustment_steps isa Integer ||
        throw(ArgumentError("self_adjustment_steps must be an integer."))
    self_adjustment_steps > 0 || throw(ArgumentError("self_adjustment_steps must be positive."))
    log_freq > 0 || throw(ArgumentError("log_freq must be positive."))

    replicas = _make_windowed_tss_replicas(
        state;
        n_replicas = n_replicas,
        first_states = first_states,
        first_windows = first_windows,
        replica_active_states = replica_active_states,
    )
    _validate_windowed_tss_replica_history_support(state, replicas)
    replica_workspaces = [
        WindowedTSSReplicaWorkspace(state)
        for _ in eachindex(replicas)
    ]

    return WindowedTSSSimulation(
        state,
        replicas,
        replica_workspaces,
        n_md_steps,
        n_cycles,
        Int(self_adjustment_steps),
        log_freq,
    )
end

function WindowedTSSReplicaWorkspace(state::WindowedTSSState)
    max_local_states = maximum(length(estimator.state_indices) for estimator in state.estimators)
    reference_estimator = first(state.estimators)
    FT = eltype(reference_estimator.f)
    ST = eltype(reference_estimator.energies)

    return WindowedTSSReplicaWorkspace(
        deepcopy(state.state_space.partition),
        Vector{ST}(undef, max_local_states),
        zeros(FT, max_local_states),
        zeros(FT, max_local_states),
        zeros(FT, max_local_states),
        zeros(FT, max_local_states),
        MersenneTwister(0),
    )
end

function _replica_count(n_replicas, replica_active_states)
    if isnothing(n_replicas)
        return isnothing(replica_active_states) ? 1 : length(replica_active_states)
    end
    n_replicas isa Integer ||
        throw(ArgumentError("n_replicas must be an integer."))
    n_replicas > 0 ||
        throw(ArgumentError("n_replicas must be positive."))
    return Int(n_replicas)
end

function _normalize_replica_values(values, n_replicas::Int, default, name::AbstractString)
    if isnothing(values)
        return fill(Int(default), n_replicas)
    end
    normalized = Int.(collect(values))
    length(normalized) == n_replicas ||
        throw(ArgumentError("$(name) must have length $(n_replicas)."))
    return normalized
end

function _default_tss_replica_windows(state::WindowedTSSState,
                                      first_states::AbstractVector{Int})
    return [first(windows_for_state(state, first_state)) for first_state in first_states]
end

function _validate_tss_replica_window(state::WindowedTSSState,
                                      first_state::Int,
                                      first_window::Int,
                                      replica_i::Int)
    1 <= first_state <= n_states(state.state_space) ||
        throw(ArgumentError("first_states[$(replica_i)] ($(first_state)) out of range."))
    1 <= first_window <= length(state.windows) ||
        throw(ArgumentError("first_windows[$(replica_i)] ($(first_window)) out of range."))
    window_contains_state(state.windows[first_window], first_state) ||
        throw(ArgumentError("first_windows[$(replica_i)] must contain first_states[$(replica_i)]."))
    return nothing
end

function _make_windowed_tss_replicas(state::WindowedTSSState;
                                     n_replicas,
                                     first_states,
                                     first_windows,
                                     replica_active_states)
    n_rep = _replica_count(n_replicas, replica_active_states)
    n_rep > 0 ||
        throw(ArgumentError("n_replicas must be positive."))

    if isnothing(replica_active_states)
        first_state_values = _normalize_replica_values(
            first_states,
            n_rep,
            state.active_state.active_idx,
            "first_states",
        )
        if n_rep == 1 && first_state_values[1] != state.active_state.active_idx
            throw(ArgumentError("single-replica WindowedTSSSimulation must use the " *
                                "WindowedTSSState active state; construct the state " *
                                "with the desired first_state instead."))
        end
        active_states = n_rep == 1 ?
                        [state.active_state] :
                        [ActiveThermoState(state.state_space, first_state)
                         for first_state in first_state_values]
    else
        active_states = collect(replica_active_states)
        length(active_states) == n_rep ||
            throw(ArgumentError("replica_active_states must have length $(n_rep)."))
        first_state_values = [active_state.active_idx for active_state in active_states]
        if !isnothing(first_states)
            supplied = _normalize_replica_values(first_states, n_rep, first(first_state_values), "first_states")
            supplied == first_state_values ||
                throw(ArgumentError("first_states must match replica_active_states active indices."))
        end
    end

    first_window_values = if isnothing(first_windows)
        n_rep == 1 && active_states[1] === state.active_state ?
        [state.active_window] :
        _default_tss_replica_windows(state, first_state_values)
    else
        _normalize_replica_values(first_windows, n_rep, state.active_window, "first_windows")
    end

    if n_rep == 1 && first_window_values[1] != state.active_window &&
            active_states[1] === state.active_state
        throw(ArgumentError("single-replica WindowedTSSSimulation must use the " *
                            "WindowedTSSState active window; construct the state " *
                            "with the desired first_window instead."))
    end

    for replica_i in 1:n_rep
        _validate_tss_replica_window(
            state,
            first_state_values[replica_i],
            first_window_values[replica_i],
            replica_i,
        )
    end

    return [
        WindowedTSSReplica(active_states[replica_i], first_window_values[replica_i])
        for replica_i in 1:n_rep
    ]
end

function _validate_windowed_tss_replica_history_support(state::WindowedTSSState,
                                                        replicas::AbstractVector)
    length(replicas) <= 1 && return nothing
    all(estimator -> !isnothing(estimator.history), state.estimators) ||
        throw(ArgumentError("multireplica WindowedTSSSimulation requires history_forgetting " *
                            "to be enabled in WindowedTSSState."))
    return nothing
end

function _windowed_tss_cycle_error(state::WindowedTSSState,
                                   message::AbstractString,
                                   cycle_window::Int,
                                   substep::Int)
    return ArgumentError(
        "TSS windowed cycle invariant failed at iteration $(state.iteration), " *
        "substep $(substep), fixed cycle window $(cycle_window): $(message)"
    )
end

function _check_windowed_tss_cycle_state!(state::WindowedTSSState,
                                          cycle_window::Int,
                                          state_idx::Int,
                                          substep::Int,
                                          label::AbstractString)
    if !window_contains_state(state.windows[cycle_window], state_idx)
        throw(_windowed_tss_cycle_error(
            state,
            "$(label) state $(state_idx) is not contained in fixed cycle window $(cycle_window)",
            cycle_window,
            substep,
        ))
    end
    return state
end

function _check_windowed_tss_cycle_window!(state::WindowedTSSState,
                                           cycle_window::Int,
                                           substep::Int)
    if state.active_window != cycle_window
        throw(_windowed_tss_cycle_error(
            state,
            "active window $(state.active_window) changed during the cycle",
            cycle_window,
            substep,
        ))
    end
    return state
end

function _drop_old_windowed_tss_histories!(state::WindowedTSSState, history_time::Int)
    for estimator in state.estimators
        isnothing(estimator.history) && continue

        _drop_old_tss_epochs!(estimator.history, history_time)
        if tss_recent_count(estimator) > 0
            _aggregate_tss_history!(estimator)
            _update_tss_adaptive_gamma!(estimator)
            update_tss_sampling_distribution!(estimator)
        end
    end
    return state
end

function _check_windowed_tss_replica_invariant(state::WindowedTSSState,
                                               replica::WindowedTSSReplica)
    1 <= replica.active_window <= length(state.windows) ||
        throw(ArgumentError("TSS replica active window $(replica.active_window) out of range."))
    window_contains_state(state.windows[replica.active_window], replica.active_state.active_idx) ||
        throw(ArgumentError("TSS replica active window $(replica.active_window) does not contain " *
                            "active state $(replica.active_state.active_idx)."))
    return replica
end

function _check_windowed_tss_replica_cycle_state!(state::WindowedTSSState,
                                                  replica::WindowedTSSReplica,
                                                  cycle_window::Int,
                                                  state_idx::Int,
                                                  substep::Int,
                                                  label::AbstractString)
    if !window_contains_state(state.windows[cycle_window], state_idx)
        throw(_windowed_tss_cycle_error(
            state,
            "$(label) state $(state_idx) is not contained in fixed cycle window $(cycle_window)",
            cycle_window,
            substep,
        ))
    end
    if replica.active_window != cycle_window
        throw(_windowed_tss_cycle_error(
            state,
            "replica active window $(replica.active_window) changed during the cycle",
            cycle_window,
            substep,
        ))
    end
    return replica
end

function _workspace_view(workspace::WindowedTSSReplicaWorkspace, field::Symbol, n::Int)
    return @view getfield(workspace, field)[1:n]
end

function _process_tss_sample!(workspace::WindowedTSSReplicaWorkspace,
                              estimator::TSSState{FT},
                              active_state::ActiveThermoState) where {FT}
    coords = active_state.active_sys.coords
    boundary = active_state.active_sys.boundary
    state_indices = estimator.state_indices
    n_local = length(state_indices)

    energies = _workspace_view(workspace, :energies, n_local)
    reduced_pot = _workspace_view(workspace, :reduced_pot, n_local)
    weights = _workspace_view(workspace, :weights, n_local)
    scratch = _workspace_view(workspace, :scratch, n_local)
    log_state_bias = _workspace_view(workspace, :log_state_bias, n_local)

    evaluate_energy_subset!(
        energies,
        workspace.partition,
        coords,
        boundary,
        state_indices,
    )
    reduced_potentials!(
        reduced_pot,
        energies,
        estimator.state_space,
        boundary,
        state_indices,
    )
    check_tss_finite!(reduced_pot, "reduced potentials", estimator)

    @. log_state_bias = estimator.f + estimator.log_dens
    check_tss_finite!(log_state_bias, "log state bias", estimator)
    conditional_state_weights!(
        weights,
        log_state_bias,
        reduced_pot,
        scratch,
    )
    check_tss_probabilities!(weights, "conditional weights", estimator)

    @. scratch = log_state_bias - reduced_pot
    log_den = logsumexp(scratch)
    isfinite(log_den) || throw(ArgumentError("TSS log normalization is non-finite " *
                                             "($(log_den)) at iteration " *
                                             "$(estimator.iteration) with active state " *
                                             "$(active_state.active_idx)."))

    return (
        reduced_pot = reduced_pot,
        weights = weights,
        log_den = log_den,
    )
end

function _tss_sample_global_state(rng::AbstractRNG,
                                  estimator::TSSState,
                                  weights::AbstractVector)
    idx = sample_state(rng, weights)
    return tss_global_index(estimator, idx)
end

function _collect_windowed_tss_observation!(state::WindowedTSSState{FT},
                                            replica::WindowedTSSReplica,
                                            workspace::WindowedTSSReplicaWorkspace,
                                            replica_index::Int,
                                            n_md_steps::Int,
                                            self_adjustment_steps::Int,
                                            n_threads::Int) where {FT}
    _check_windowed_tss_invariant(state)
    _check_windowed_tss_replica_invariant(state, replica)

    entry_state = replica.active_state.active_idx
    replica.active_window = other_window_for_state(state, replica.active_window, entry_state)
    cycle_window = replica.active_window
    estimator = state.estimators[cycle_window]
    _check_windowed_tss_replica_cycle_state!(
        state,
        replica,
        cycle_window,
        entry_state,
        0,
        "entry",
    )

    final_visited_state = entry_state
    final_next_state = entry_state
    final_log_den = zero(FT)

    for substep in 1:self_adjustment_steps
        visited_state = replica.active_state.active_idx
        _check_windowed_tss_replica_cycle_state!(
            state,
            replica,
            cycle_window,
            visited_state,
            substep,
            "visited",
        )

        simulate!(
            replica.active_state.active_sys,
            replica.active_state.active_integrator,
            n_md_steps;
            n_threads = n_threads,
            rng = workspace.rng,
        )

        sample = _process_tss_sample!(workspace, estimator, replica.active_state)
        final_log_den = sample.log_den
        next_state = _tss_sample_global_state(workspace.rng, estimator, sample.weights)
        _check_windowed_tss_replica_cycle_state!(
            state,
            replica,
            cycle_window,
            next_state,
            substep,
            "sampled next",
        )

        final_visited_state = visited_state
        final_next_state = next_state

        if substep < self_adjustment_steps
            set_active_state!(replica.active_state, state.state_space, next_state)
            _check_windowed_tss_replica_invariant(state, replica)
        end
    end

    observation = WindowedTSSObservation(
        replica_index,
        cycle_window,
        final_visited_state,
        final_next_state,
        final_log_den,
        copy(_workspace_view(workspace, :reduced_pot, length(estimator.state_indices))),
        copy(_workspace_view(workspace, :weights, length(estimator.state_indices))),
        _evaluate_tss_adaptive_observable(
            estimator,
            replica.active_state;
            log_den = final_log_den,
            history_time = state.iteration + 1,
            energies = _workspace_view(workspace, :energies, length(estimator.state_indices)),
            reduced_potentials = _workspace_view(workspace, :reduced_pot, length(estimator.state_indices)),
            n_threads = n_threads,
        ),
    )
    set_active_state!(replica.active_state, state.state_space, final_next_state)
    _check_windowed_tss_replica_invariant(state, replica)
    return observation
end

function _apply_tss_observation_standard!(estimator::TSSState,
                                          observation::WindowedTSSObservation,
                                          history_time::Int)
    estimator.reduced_pot .= observation.reduced_pot
    estimator.weights .= observation.weights
    return update_tss_estimates!(
        estimator;
        visited_state = observation.visited_state,
        history_time = history_time,
        observable_values = observation.observable_values,
    )
end

function _apply_tss_observation_to_history!(estimator::TSSState,
                                            observation::WindowedTSSObservation,
                                            history_time::Int)
    isnothing(estimator.history) &&
        throw(ArgumentError("multireplica TSS observation updates require history forgetting."))

    visited_local = tss_local_index(estimator, observation.visited_state)
    estimator.reduced_pot .= observation.reduced_pot
    _update_tss_history!(
        estimator,
        visited_local,
        observation.log_den,
        history_time;
        observable_values = observation.observable_values,
        aggregate = false,
    )
    estimator.iteration += 1
    return estimator
end

function _apply_windowed_tss_observations!(state::WindowedTSSState{FT},
                                           observations::AbstractVector) where {FT}
    history_time = state.iteration + 1
    old_f = [copy(estimator.f) for estimator in state.estimators]

    if length(observations) == 1
        observation = only(observations)
        estimator = state.estimators[observation.update_window]
        max_df = _apply_tss_observation_standard!(
            estimator,
            observation,
            history_time,
        )
        state.window_update_counts[observation.update_window] += 1
        state.iteration += 1
        _drop_old_windowed_tss_histories!(state, state.iteration)
        update_windowed_tss_coupling!(state)
        return max_df
    end

    for observation in observations
        estimator = state.estimators[observation.update_window]
        _apply_tss_observation_to_history!(estimator, observation, history_time)
        state.window_update_counts[observation.update_window] += 1
    end

    state.iteration += 1
    _drop_old_windowed_tss_histories!(state, state.iteration)
    update_windowed_tss_coupling!(state)

    max_df = zero(FT)
    for (window_i, estimator) in enumerate(state.estimators)
        max_df = max(max_df, maximum(abs, estimator.f .- old_f[window_i]))
    end
    return max_df
end

function _seed_windowed_tss_replica_rngs!(sim::WindowedTSSSimulation, rng::AbstractRNG)
    for workspace in sim.replica_workspaces
        Random.seed!(workspace.rng, rand(rng, UInt))
    end
    return sim
end

function _validate_tss_replica_parallel(replica_parallel)
    mode = replica_parallel isa Symbol ? replica_parallel : Symbol(replica_parallel)
    mode in (:auto, :serial, :threads) ||
        throw(ArgumentError("replica_parallel must be one of :auto, :serial, or :threads."))
    return mode
end

function _windowed_tss_uses_gpu_replicas(sim::WindowedTSSSimulation)
    return any(replica -> is_on_gpu(replica.active_state.active_sys), sim.replicas)
end

function _resolve_tss_replica_parallel(sim::WindowedTSSSimulation,
                                       replica_parallel,
                                       n_threads::Int)
    mode = _validate_tss_replica_parallel(replica_parallel)
    uses_gpu = _windowed_tss_uses_gpu_replicas(sim)

    if mode == :threads && uses_gpu
        throw(ArgumentError("replica_parallel=:threads is not supported for GPU-backed " *
                            "WindowedTSSSimulation replicas; use :serial or :auto."))
    end

    if mode == :auto
        return (!uses_gpu && length(sim.replicas) > 1 && n_threads > 1 &&
                Threads.nthreads() > 1) ? :threads : :serial
    end
    return mode
end

function _collect_windowed_tss_observations_serial!(sim::WindowedTSSSimulation,
                                                    thread_div)
    state = sim.state
    FT = eltype(first(state.estimators).f)
    observations = Vector{WindowedTSSObservation{FT, Vector{FT}}}(undef, length(sim.replicas))
    for replica_i in eachindex(sim.replicas)
        observations[replica_i] = _collect_windowed_tss_observation!(
            state,
            sim.replicas[replica_i],
            sim.replica_workspaces[replica_i],
            replica_i,
            sim.n_md_steps,
            sim.self_adjustment_steps,
            max(1, thread_div[replica_i]),
        )
    end
    return observations
end

function _collect_windowed_tss_observations_threaded!(sim::WindowedTSSSimulation,
                                                      thread_div)
    state = sim.state
    FT = eltype(first(state.estimators).f)
    observations = Vector{WindowedTSSObservation{FT, Vector{FT}}}(undef, length(sim.replicas))
    @sync for replica_i in eachindex(sim.replicas)
        Threads.@spawn begin
            observations[replica_i] = _collect_windowed_tss_observation!(
                state,
                sim.replicas[replica_i],
                sim.replica_workspaces[replica_i],
                replica_i,
                sim.n_md_steps,
                sim.self_adjustment_steps,
                max(1, thread_div[replica_i]),
            )
        end
    end
    return observations
end

function _sync_windowed_tss_state_to_replica!(state::WindowedTSSState,
                                              replica::WindowedTSSReplica)
    state.active_window = replica.active_window
    if state.active_state !== replica.active_state
        set_active_state!(
            state.active_state,
            state.state_space,
            replica.active_state.active_idx,
        )
        state.active_state.active_sys.coords .= replica.active_state.active_sys.coords
        state.active_state.active_sys.velocities .= replica.active_state.active_sys.velocities
        state.active_state.active_sys.boundary = replica.active_state.active_sys.boundary
    end
    return state
end

function _run_windowed_tss_cycle!(state::WindowedTSSState,
                                  n_md_steps::Int,
                                  self_adjustment_steps::Int,
                                  n_threads::Int,
                                  rng)
    _check_windowed_tss_invariant(state)

    entry_state = state.active_state.active_idx
    switch_active_window!(state; current_state = entry_state)
    cycle_window = state.active_window
    estimator = active_tss_estimator(state)
    _check_windowed_tss_cycle_state!(
        state,
        cycle_window,
        entry_state,
        0,
        "entry",
    )

    final_visited_state = entry_state
    final_next_state = entry_state

    for substep in 1:self_adjustment_steps
        visited_state = state.active_state.active_idx
        _check_windowed_tss_cycle_state!(
            state,
            cycle_window,
            visited_state,
            substep,
            "visited",
        )

        simulate!(
            state.active_state.active_sys,
            state.active_state.active_integrator,
            n_md_steps;
            n_threads = n_threads,
            rng = rng,
        )

        _check_windowed_tss_cycle_window!(state, cycle_window, substep)

        process_tss_sample!(estimator)
        next_state = tss_sample_global_state(rng, estimator)
        _check_windowed_tss_cycle_state!(
            state,
            cycle_window,
            next_state,
            substep,
            "sampled next",
        )

        final_visited_state = visited_state
        final_next_state = next_state

        if substep < self_adjustment_steps
            set_active_state!(state.active_state, state.state_space, next_state)
            _check_windowed_tss_invariant(state)
        end
    end

    history_time = state.iteration + 1
    max_df = update_tss_estimates!(
        estimator;
        visited_state = final_visited_state,
        history_time = history_time,
        n_threads = n_threads,
    )
    state.window_update_counts[cycle_window] += 1
    state.iteration += 1
    _drop_old_windowed_tss_histories!(state, state.iteration)
    update_windowed_tss_coupling!(state)

    set_active_state!(state.active_state, state.state_space, final_next_state)
    _check_windowed_tss_invariant(state)

    return (
        update_window = cycle_window,
        visited_state = final_visited_state,
        next_state = final_next_state,
        max_delta_f = max_df,
    )
end

function simulate!(sim::WindowedTSSSimulation;
                   rng = Random.default_rng(),
                   n_threads::Integer = Threads.nthreads(),
                   replica_parallel = :auto)
    state::WindowedTSSState = sim.state
    n_threads > 0 || throw(ArgumentError("n_threads must be positive."))
    length(sim.replica_workspaces) == length(sim.replicas) ||
        throw(ArgumentError("WindowedTSSSimulation replica workspaces do not match replicas."))

    parallel_mode = _resolve_tss_replica_parallel(sim, replica_parallel, Int(n_threads))
    thread_div = equal_parts(Int(n_threads), length(sim.replicas))
    legacy_single_path = length(sim.replicas) == 1 &&
                         sim.replicas[1].active_state === state.active_state &&
                         sim.replicas[1].active_window == state.active_window
    legacy_single_path || _seed_windowed_tss_replica_rngs!(sim, rng)

    for cycle in 1:sim.n_cycles
        if legacy_single_path
            result = _run_windowed_tss_cycle!(
                state,
                sim.n_md_steps,
                sim.self_adjustment_steps,
                Int(n_threads),
                rng,
            )
            sim.replicas[1].active_window = state.active_window
            estimator = state.estimators[result.update_window]

            if should_log_tss(estimator.iteration, sim.log_freq)
                log_tss_stats!(
                    estimator.stats,
                    estimator,
                    result.visited_state,
                    result.next_state,
                    result.max_delta_f,
                )
            end
            if should_log_tss(state.iteration, sim.log_freq)
                log_windowed_tss_stats!(
                    state.stats,
                    state,
                    result.update_window,
                    result.visited_state,
                    result.next_state,
                    result.max_delta_f,
                )
            end
            continue
        end

        observations = if parallel_mode == :threads
            _collect_windowed_tss_observations_threaded!(sim, thread_div)
        else
            _collect_windowed_tss_observations_serial!(sim, thread_div)
        end
        max_delta_f = _apply_windowed_tss_observations!(state, observations)
        _sync_windowed_tss_state_to_replica!(state, first(sim.replicas))

        if should_log_tss(state.iteration, sim.log_freq)
            for observation in observations
                estimator = state.estimators[observation.update_window]
                log_tss_stats!(
                    estimator.stats,
                    estimator,
                    observation.visited_state,
                    observation.sampled_next_state,
                    max_delta_f,
                )
            end
            first_observation = first(observations)
            log_windowed_tss_stats!(
                state.stats,
                state,
                first_observation.update_window,
                first_observation.visited_state,
                first_observation.sampled_next_state,
                max_delta_f;
                replica_indices = [observation.replica_index for observation in observations],
                replica_update_windows = [observation.update_window for observation in observations],
                replica_visited_states = [observation.visited_state for observation in observations],
                replica_sampled_next_states = [observation.sampled_next_state for observation in observations],
                replica_max_abs_delta_f = fill(max_delta_f, length(observations)),
            )
        end
    end

    return state
end
