function log_windowed_tss_stats!(
    stats::WindowedTSSStats{FT},
    state::WindowedTSSState{FT},
    update_window::Int,
    visited_state::Int,
    next_state::Int,
    max_delta_f::FT) where {FT}

    push!(stats.iterations, state.iteration)
    push!(stats.update_window, update_window)
    push!(stats.visited_state, visited_state)
    push!(stats.sampled_next_state, next_state)
    push!(stats.max_abs_delta_f, max_delta_f)
    push!(stats.active_window_history, state.active_window)
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

mutable struct WindowedTSSSimulation{S}
    state::S
    n_md_steps::Int
    n_cycles::Int
    self_adjustment_steps::Int
    log_freq::Int
end

function WindowedTSSSimulation(state::WindowedTSSState;
                               n_md_steps::Int,
                               n_cycles::Int,
                               self_adjustment_steps = 1,
                               log_freq::Int = 1000)

    n_md_steps > 0 || throw(ArgumentError("n_md_steps must be positive."))
    n_cycles >= 0 || throw(ArgumentError("n_cycles must be non-negative."))
    self_adjustment_steps isa Integer ||
        throw(ArgumentError("self_adjustment_steps must be an integer."))
    self_adjustment_steps > 0 || throw(ArgumentError("self_adjustment_steps must be positive."))
    log_freq > 0 || throw(ArgumentError("log_freq must be positive."))

    return WindowedTSSSimulation(
        state,
        n_md_steps,
        n_cycles,
        Int(self_adjustment_steps),
        log_freq,
    )
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
            update_tss_sampling_distribution!(estimator)
        end
    end
    return state
end

function _run_windowed_tss_cycle!(state::WindowedTSSState,
                                  n_md_steps::Int,
                                  self_adjustment_steps::Int,
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
            n_md_steps,
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

function simulate!(sim::WindowedTSSSimulation; rng = Random.default_rng())
    state::WindowedTSSState = sim.state

    for cycle in 1:sim.n_cycles
        result = _run_windowed_tss_cycle!(
            state,
            sim.n_md_steps,
            sim.self_adjustment_steps,
            rng,
        )
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
    end

    return state
end
