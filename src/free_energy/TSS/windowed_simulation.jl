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

function simulate!(sim::WindowedTSSSimulation; rng = Random.default_rng())
    state::WindowedTSSState = sim.state

    for cycle in 1:sim.n_cycles
        _check_windowed_tss_invariant(state)

        final_window = state.active_window
        final_visited_state = state.active_state.active_idx
        final_next_state = state.active_state.active_idx

        for substep in 1:sim.self_adjustment_steps
            visited_state = state.active_state.active_idx

            simulate!(
                state.active_state.active_sys,
                state.active_state.active_integrator,
                sim.n_md_steps,
            )

            switch_active_window!(state; current_state = visited_state)
            estimator = active_tss_estimator(state)

            process_tss_sample!(estimator)
            next_state = tss_sample_global_state(rng, estimator)

            final_window = state.active_window
            final_visited_state = visited_state
            final_next_state = next_state

            if substep < sim.self_adjustment_steps
                set_active_state!(state.active_state, state.state_space, next_state)
                _check_windowed_tss_invariant(state)
            end
        end

        estimator = state.estimators[final_window]
        max_df = update_tss_estimates!(estimator; visited_state = final_visited_state)
        state.window_update_counts[final_window] += 1
        state.iteration += 1
        update_windowed_tss_coupling!(state)

        if should_log_tss(estimator.iteration, sim.log_freq)
            log_tss_stats!(estimator.stats, estimator, final_visited_state, final_next_state, max_df)
        end
        if should_log_tss(state.iteration, sim.log_freq)
            log_windowed_tss_stats!(
                state.stats,
                state,
                final_window,
                final_visited_state,
                final_next_state,
                max_df,
            )
        end

        set_active_state!(state.active_state, state.state_space, final_next_state)
        _check_windowed_tss_invariant(state)
    end

    return state
end
