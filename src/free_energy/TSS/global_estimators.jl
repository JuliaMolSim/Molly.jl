function _windowed_tss_coupling(state::WindowedTSSState)
    isnothing(state.coupling) &&
        throw(ArgumentError("global TSS visit control is not enabled for this WindowedTSSState."))
    return state.coupling
end

function _validate_windowed_tss_coupling_params(::Type{FT};
                                                tolerance::Real,
                                                max_iterations::Integer,
                                                damping::Real,
                                                pi_regularization::Real) where {FT}
    isfinite(tolerance) && tolerance > 0 ||
        throw(ArgumentError("visit_control_tolerance must be finite and positive."))
    max_iterations > 0 ||
        throw(ArgumentError("visit_control_max_iterations must be positive."))
    isfinite(damping) && 0 < damping <= 1 ||
        throw(ArgumentError("visit_control_damping must be in the (0, 1] interval."))
    isfinite(pi_regularization) && 0 < pi_regularization < 1 ||
        throw(ArgumentError("pi_regularization must be in the (0, 1) interval."))

    return (
        tolerance = FT(tolerance),
        max_iterations = Int(max_iterations),
        damping = FT(damping),
        pi_regularization = FT(pi_regularization),
    )
end

function _gauge_windowed_visit_control!(coupling::WindowedTSSCoupling{FT}) where {FT}
    reference = coupling.visit_control_f[1]
    isfinite(reference) ||
        throw(ArgumentError("TSS visit-control gauge reference is non-finite ($(reference))."))
    @. coupling.visit_control_f -= reference
    all(isfinite, coupling.visit_control_f) ||
        throw(ArgumentError("TSS visit-control free energies contain non-finite values."))
    return coupling.visit_control_f
end

function _normalize_windowed_tss_probabilities!(weights::AbstractVector{FT},
                                                name::AbstractString,
                                                state::WindowedTSSState{FT}) where {FT}
    check_tss_probabilities!(weights, name, state)
    s = sum(weights)
    weights ./= s
    check_tss_positive_probabilities!(weights, name, state)
    return weights
end

function _normalize_windowed_tss_nonnegative_probabilities!(weights::AbstractVector{FT},
                                                            name::AbstractString,
                                                            state::WindowedTSSState{FT}) where {FT}
    check_tss_probabilities!(weights, name, state)
    weights ./= sum(weights)
    check_tss_probabilities!(weights, name, state)
    return weights
end

function _windowed_tss_visited_mask(state::WindowedTSSState)
    use_recent_counts = any(estimator -> !isnothing(estimator.history), state.estimators)
    visited = use_recent_counts ?
              [tss_recent_count(estimator) > 0 for estimator in state.estimators] :
              state.window_update_counts .> 0
    if !any(visited)
        visited .= true
    end
    return visited
end

function _windowed_current_local_free_energies(state::WindowedTSSState)
    return [estimator.f for estimator in state.estimators]
end

function _validate_windowed_local_free_energies(state::WindowedTSSState,
                                                local_f_by_window)
    length(local_f_by_window) == length(state.estimators) ||
        throw(ArgumentError("local_f_by_window must match the number of TSS windows."))

    for (window_i, estimator) in enumerate(state.estimators)
        local_f = local_f_by_window[window_i]
        length(local_f) == length(estimator.state_indices) ||
            throw(ArgumentError("local free-energy vector for TSS window $(window_i) " *
                                "has length $(length(local_f)); expected " *
                                "$(length(estimator.state_indices))."))
        all(isfinite, local_f) ||
            throw(ArgumentError("local free-energy vector for TSS window $(window_i) " *
                                "contains non-finite values."))
    end

    return local_f_by_window
end

function _windowed_local_average_free_energies(state::WindowedTSSState{FT},
                                               local_f_by_window) where {FT}
    _validate_windowed_local_free_energies(state, local_f_by_window)
    K = n_states(state.state_space)
    values = zeros(FT, K)
    counts = zeros(Int, K)

    for (window_i, estimator) in enumerate(state.estimators)
        for global_state in state.windows[window_i].state_indices
            local_state = tss_local_index(estimator, global_state)
            values[global_state] += local_f_by_window[window_i][local_state]
            counts[global_state] += 1
        end
    end

    for state_i in 1:K
        counts[state_i] > 0 ||
            throw(ArgumentError("state $(state_i) has no local TSS free-energy estimates."))
        values[state_i] /= FT(counts[state_i])
    end
    values .-= values[1]
    return values
end

function _windowed_local_average_free_energies(state::WindowedTSSState)
    return _windowed_local_average_free_energies(
        state,
        _windowed_current_local_free_energies(state),
    )
end

function initialize_windowed_tss_coupling(state::WindowedTSSState{FT};
                                          tolerance::Real = 1e-8,
                                          max_iterations::Integer = 1_000,
                                          damping::Real = 1.0,
                                          pi_regularization::Real = 1e-3) where {FT}
    params = _validate_windowed_tss_coupling_params(
        FT;
        tolerance = tolerance,
        max_iterations = max_iterations,
        damping = damping,
        pi_regularization = pi_regularization,
    )

    K = n_states(state.state_space)
    n_windows = length(state.windows)
    visit_control_f = _windowed_local_average_free_energies(state)

    coupling = WindowedTSSCoupling{FT}(
        visit_control_f,
        fill(inv(FT(n_windows)), n_windows),
        zeros(FT, n_windows, n_windows),
        zeros(FT, K),
        zeros(FT, n_windows),
        zeros(FT, K),
        zeros(FT, K),
        zeros(FT, K),
        [copy(estimator.density) for estimator in state.estimators],
        fill(inv(FT(n_windows)), n_windows),
        zeros(FT, K),
        zeros(FT, n_windows),
        copy(visit_control_f),
        0,
        false,
        FT(Inf),
        params.tolerance,
        params.max_iterations,
        params.damping,
        params.pi_regularization,
    )
    _gauge_windowed_visit_control!(coupling)
    return coupling
end

function _windowed_local_weight(estimator::TSSState{FT},
                                local_state::Integer,
                                use_tilts::Bool) where {FT}
    weight = estimator.gamma[local_state]
    if use_tilts
        weight *= max(estimator.tilts[local_state], tss_tilt_floor(FT))
    end
    return weight
end

function compute_window_transition_matrix!(state::WindowedTSSState{FT};
                                           use_tilts::Bool = true,
                                           visited_mask = _windowed_tss_visited_mask(state),
                                           transition = _windowed_tss_coupling(state).window_transition) where {FT}
    Q = transition
    fill!(Q, zero(FT))

    length(visited_mask) == length(state.windows) ||
        throw(ArgumentError("visited_mask must match the number of TSS windows."))
    any(visited_mask) ||
        throw(ArgumentError("at least one TSS window must be active in the window-probability solve."))

    for (window_j, estimator) in enumerate(state.estimators)
        visited_mask[window_j] || continue

        denom = zero(FT)
        for local_state in eachindex(estimator.state_indices)
            denom += _windowed_local_weight(estimator, local_state, use_tilts)
        end
        isfinite(denom) && denom > zero(FT) ||
            throw(ArgumentError("TSS window $(window_j) has invalid transition denominator $(denom)."))

        for local_state in eachindex(estimator.state_indices)
            global_state = estimator.state_indices[local_state]
            contribution = FT(0.5) * _windowed_local_weight(estimator, local_state, use_tilts) / denom
            for window_i in state.state_to_windows[global_state]
                if visited_mask[window_i]
                    Q[window_i, window_j] += contribution
                else
                    Q[window_j, window_j] += contribution
                end
            end
        end

        col_sum = sum(view(Q, :, window_j))
        if !isfinite(col_sum) || col_sum <= zero(FT)
            Q[window_j, window_j] = one(FT)
        else
            @views Q[:, window_j] ./= col_sum
        end
    end

    return Q
end

function solve_window_probability_eigenvector!(state::WindowedTSSState{FT};
                                               use_tilts::Bool = true,
                                               visited_mask = _windowed_tss_visited_mask(state),
                                               output = _windowed_tss_coupling(state).window_probs,
                                               transition = _windowed_tss_coupling(state).window_transition) where {FT}
    Q = compute_window_transition_matrix!(
        state;
        use_tilts = use_tilts,
        visited_mask = visited_mask,
        transition = transition,
    )
    visited = findall(identity, visited_mask)
    fill!(output, zero(FT))

    if length(visited) == 1
        output[first(visited)] = one(FT)
        return output
    end

    Q_sub = Matrix{FT}(Q[visited, visited])
    n = length(visited)
    A = Q_sub - Matrix{FT}(I, n, n)
    b = zeros(FT, n)
    A[n, :] .= one(FT)
    b[n] = one(FT)

    probs = pinv(A) * b
    for i in eachindex(probs)
        if probs[i] < zero(FT) && probs[i] > -sqrt(eps(FT))
            probs[i] = zero(FT)
        end
    end

    if any(<(zero(FT)), probs) || !all(isfinite, probs) || sum(probs) <= zero(FT)
        probs .= inv(FT(n))
    else
        probs ./= sum(probs)
    end

    for (local_i, window_i) in enumerate(visited)
        output[window_i] = probs[local_i]
    end
    _normalize_windowed_tss_nonnegative_probabilities!(output, "window probabilities", state)
    return output
end

function update_window_probabilities!(state::WindowedTSSState{FT}) where {FT}
    return solve_window_probability_eigenvector!(
        state;
        use_tilts = true,
        visited_mask = _windowed_tss_visited_mask(state),
        output = _windowed_tss_coupling(state).window_probs,
    )
end

function compute_global_rung_weights!(state::WindowedTSSState{FT}) where {FT}
    coupling = _windowed_tss_coupling(state)
    fill!(coupling.global_rung_weights, zero(FT))

    for (window_j, estimator) in enumerate(state.estimators)
        p_j = coupling.window_probs[window_j]
        p_j > zero(FT) || continue

        denom = zero(FT)
        for local_state in eachindex(estimator.state_indices)
            denom += _windowed_local_weight(estimator, local_state, true)
        end
        isfinite(denom) && denom > zero(FT) ||
            throw(ArgumentError("TSS window $(window_j) has invalid global-rung denominator $(denom)."))

        for local_state in eachindex(estimator.state_indices)
            global_state = estimator.state_indices[local_state]
            coupling.global_rung_weights[global_state] +=
                p_j * _windowed_local_weight(estimator, local_state, true) / denom
        end
    end

    total = sum(coupling.global_rung_weights)
    isfinite(total) && total > zero(FT) ||
        throw(ArgumentError("TSS global rung weights have invalid total $(total)."))
    coupling.global_rung_weights ./= total
    coupling.lhs_marginal .= coupling.global_rung_weights
    return coupling.global_rung_weights
end

function _windowed_log_offset_denominator(state::WindowedTSSState{FT},
                                          global_state::Int,
                                          eta_plus_one::FT) where {FT}
    coupling = _windowed_tss_coupling(state)
    log_den = -FT(Inf)

    for window_i in state.state_to_windows[global_state]
        p_i = coupling.window_probs[window_i]
        p_i > zero(FT) || continue
        estimator = state.estimators[window_i]
        local_state = tss_local_index(estimator, global_state)
        term = log(p_i) + estimator.log_gamma[local_state] +
               (estimator.f[local_state] - coupling.window_offsets[window_i]) / eta_plus_one
        log_den = logaddexp_tss(log_den, term)
    end

    isfinite(log_den) ||
        throw(ArgumentError("TSS visit-control offset denominator is non-finite for state $(global_state)."))
    return log_den
end

function _gauge_window_offsets!(offsets::AbstractVector{FT},
                                probs::AbstractVector{FT}) where {FT}
    weight = sum(probs)
    weight > zero(FT) || return offsets
    offset_mean = sum(probs .* offsets) / weight
    offsets .-= offset_mean
    return offsets
end

function solve_windowed_visit_control!(state::WindowedTSSState{FT}) where {FT}
    coupling = _windowed_tss_coupling(state)
    eta = first(state.estimators).ETA
    compute_global_rung_weights!(state)

    if iszero(eta)
        coupling.window_offsets .= zero(FT)
        coupling.visit_control_f .= _windowed_local_average_free_energies(state)
        compute_visit_control_residual!(state)
        coupling.iterations = 0
        coupling.converged = true
        return coupling
    end

    eta_plus_one = eta + one(FT)
    proposed = similar(coupling.window_offsets)
    coupling.converged = false
    coupling.iterations = 0

    for iteration in 1:coupling.max_iterations
        fill!(proposed, zero(FT))

        for (window_j, estimator) in enumerate(state.estimators)
            coupling.window_probs[window_j] > zero(FT) || continue
            log_sum = -FT(Inf)

            for local_state in eachindex(estimator.state_indices)
                global_state = estimator.state_indices[local_state]
                q_k = coupling.global_rung_weights[global_state]
                q_k > zero(FT) || continue
                log_den = _windowed_log_offset_denominator(state, global_state, eta_plus_one)
                term = log(q_k) + estimator.log_gamma[local_state] +
                       estimator.f[local_state] / eta_plus_one - log_den
                log_sum = logaddexp_tss(log_sum, term)
            end

            isfinite(log_sum) ||
                throw(ArgumentError("TSS window $(window_j) has non-finite visit-control offset update."))
            proposed[window_j] = eta_plus_one * log_sum
        end

        _gauge_window_offsets!(proposed, coupling.window_probs)
        coupling.iterations = iteration
        delta = maximum(abs, proposed .- coupling.window_offsets)

        @. coupling.window_offsets += coupling.damping * (proposed - coupling.window_offsets)
        _gauge_window_offsets!(coupling.window_offsets, coupling.window_probs)

        if delta <= coupling.tolerance
            coupling.converged = true
            break
        end
    end

    update_windowed_visit_control_free_energies!(state)
    compute_visit_control_residual!(state)
    return coupling
end

function update_windowed_visit_control_free_energies!(state::WindowedTSSState{FT}) where {FT}
    coupling = _windowed_tss_coupling(state)
    eta_plus_one = first(state.estimators).ETA + one(FT)
    fallback = _windowed_local_average_free_energies(state)

    for global_state in eachindex(coupling.visit_control_f)
        q_k = coupling.global_rung_weights[global_state]
        if q_k <= zero(FT)
            coupling.visit_control_f[global_state] = fallback[global_state]
            continue
        end

        log_sum = -FT(Inf)
        for window_j in state.state_to_windows[global_state]
            p_j = coupling.window_probs[window_j]
            p_j > zero(FT) || continue
            estimator = state.estimators[window_j]
            local_state = tss_local_index(estimator, global_state)
            term = log(p_j) + estimator.log_gamma[local_state] +
                   (estimator.f[local_state] - coupling.window_offsets[window_j]) / eta_plus_one
            log_sum = logaddexp_tss(log_sum, term)
        end

        isfinite(log_sum) ||
            throw(ArgumentError("TSS visit-control free energy is undefined for state $(global_state)."))
        coupling.visit_control_f[global_state] = eta_plus_one * (log_sum - log(q_k))
    end

    _gauge_windowed_visit_control!(coupling)
    return coupling.visit_control_f
end

function compute_windowed_sampling_densities!(state::WindowedTSSState{FT}) where {FT}
    coupling = _windowed_tss_coupling(state)
    check_tss_finite!(coupling.visit_control_f, "visit-control free energies", state)

    for (window_i, estimator) in enumerate(state.estimators)
        candidate = coupling.candidate_densities[window_i]
        length(candidate) == length(estimator.state_indices) ||
            throw(ArgumentError("candidate density for TSS window $(window_i) has invalid length."))

        check_tss_finite!(estimator.f, "window $(window_i) free energies", estimator)
        check_tss_positive_probabilities!(estimator.gamma, "window $(window_i) gamma", estimator)

        coupling_strength = estimator.ETA / (estimator.ETA + one(FT))
        for local_state in eachindex(estimator.state_indices)
            global_state = estimator.state_indices[local_state]
            estimator.scratch[local_state] =
                estimator.log_gamma[local_state] +
                coupling_strength * (coupling.visit_control_f[global_state] -
                                     estimator.f[local_state])
        end

        log_norm = logsumexp(estimator.scratch)
        isfinite(log_norm) ||
            throw(ArgumentError("TSS window $(window_i) candidate density normalization is non-finite."))

        for local_state in eachindex(candidate)
            candidate[local_state] = exp(estimator.scratch[local_state] - log_norm)
        end
        @. candidate = (one(FT) - coupling.pi_regularization) * candidate +
                       coupling.pi_regularization * estimator.gamma
        _normalize_windowed_tss_probabilities!(
            candidate,
            "candidate density for window $(window_i)",
            state,
        )
    end

    return coupling.candidate_densities
end

function compute_visit_control_rhs!(state::WindowedTSSState{FT}) where {FT}
    coupling = _windowed_tss_coupling(state)
    fill!(coupling.rhs_marginal, zero(FT))

    for (window_i, estimator) in enumerate(state.estimators)
        coupling.window_probs[window_i] > zero(FT) || continue

        for local_state in eachindex(estimator.state_indices)
            global_state = estimator.state_indices[local_state]
            estimator.scratch[local_state] =
                estimator.log_gamma[local_state] +
                (estimator.f[local_state] - coupling.visit_control_f[global_state]) /
                (estimator.ETA + one(FT))
        end

        log_den = logsumexp(estimator.scratch)
        isfinite(log_den) ||
            throw(ArgumentError("TSS window $(window_i) has non-finite visit-control rhs denominator."))

        for local_state in eachindex(estimator.state_indices)
            global_state = estimator.state_indices[local_state]
            coupling.rhs_marginal[global_state] +=
                coupling.window_probs[window_i] *
                exp(estimator.scratch[local_state] - log_den)
        end
    end

    if sum(coupling.rhs_marginal) > zero(FT)
        coupling.rhs_marginal ./= sum(coupling.rhs_marginal)
    end
    return coupling.rhs_marginal
end

function compute_visit_control_residual!(state::WindowedTSSState{FT}) where {FT}
    coupling = _windowed_tss_coupling(state)
    compute_visit_control_rhs!(state)
    coupling.max_abs_residual = zero(FT)

    for state_i in eachindex(coupling.residual)
        lhs = coupling.global_rung_weights[state_i]
        rhs = coupling.rhs_marginal[state_i]
        if lhs > zero(FT)
            isfinite(rhs) && rhs > zero(FT) ||
                throw(ArgumentError("TSS visit-control rhs is invalid at state $(state_i): $(rhs)."))
            coupling.residual[state_i] = log(rhs) - log(lhs)
            coupling.max_abs_residual = max(coupling.max_abs_residual, abs(coupling.residual[state_i]))
        else
            coupling.residual[state_i] = zero(FT)
        end
    end
    check_tss_finite!(coupling.residual, "visit-control residual", state)
    return coupling.residual
end

function _reported_active_mask(state::WindowedTSSState, visited_only::Bool)
    visited_only || return trues(length(state.windows))
    return _windowed_tss_visited_mask(state)
end

function _compute_reported_tss_free_energy_components(state::WindowedTSSState{FT},
                                                      local_f_by_window;
                                                      visited_only::Bool = false) where {FT}
    _validate_windowed_local_free_energies(state, local_f_by_window)
    active_mask = _reported_active_mask(state, visited_only)
    n_windows = length(state.windows)
    reported_window_probs = zeros(FT, n_windows)
    solve_window_probability_eigenvector!(
        state;
        use_tilts = false,
        visited_mask = active_mask,
        output = reported_window_probs,
        transition = zeros(FT, n_windows, n_windows),
    )

    K = n_states(state.state_space)
    reported_gamma = zeros(FT, K)
    for (window_j, estimator) in enumerate(state.estimators)
        p_j = reported_window_probs[window_j]
        p_j > zero(FT) || continue
        for local_state in eachindex(estimator.state_indices)
            global_state = estimator.state_indices[local_state]
            reported_gamma[global_state] += p_j * estimator.gamma[local_state]
        end
    end

    reported_total = sum(reported_gamma)
    isfinite(reported_total) && reported_total > zero(FT) ||
        throw(ArgumentError("TSS reported rung density has invalid total $(reported_total)."))
    reported_gamma ./= reported_total

    active_windows = findall(>(zero(FT)), reported_window_probs)
    isempty(active_windows) &&
        throw(ArgumentError("no TSS windows are available for reported free-energy estimation."))

    n_active = length(active_windows)
    global_weighted_f = zeros(FT, K)
    for global_state in 1:K
        gamma_tss = reported_gamma[global_state]
        gamma_tss > zero(FT) || continue
        for window_j in state.state_to_windows[global_state]
            p_j = reported_window_probs[window_j]
            p_j > zero(FT) || continue
            estimator = state.estimators[window_j]
            local_state = tss_local_index(estimator, global_state)
            global_weighted_f[global_state] +=
                p_j * estimator.gamma[local_state] *
                local_f_by_window[window_j][local_state] / gamma_tss
        end
    end

    transition = zeros(FT, n_active, n_active)
    rhs = zeros(FT, n_active)
    for (local_i, window_i) in enumerate(active_windows)
        estimator_i = state.estimators[window_i]

        for global_state in state.windows[window_i].state_indices
            gamma_tss = reported_gamma[global_state]
            gamma_tss > zero(FT) || continue
            local_i_state = tss_local_index(estimator_i, global_state)
            gamma_i = estimator_i.gamma[local_i_state]
            rhs[local_i] += gamma_i *
                            (local_f_by_window[window_i][local_i_state] -
                             global_weighted_f[global_state])

            for (local_j, window_j) in enumerate(active_windows)
                window_j in state.state_to_windows[global_state] || continue
                estimator_j = state.estimators[window_j]
                local_j_state = tss_local_index(estimator_j, global_state)
                transition[local_i, local_j] +=
                    gamma_i * reported_window_probs[window_j] *
                    estimator_j.gamma[local_j_state] / gamma_tss
            end
        end
    end

    A = Matrix{FT}(I, n_active, n_active) - transition
    b = rhs
    A[n_active, :] .= reported_window_probs[active_windows]
    b[n_active] = zero(FT)

    offsets = pinv(A) * b
    reported_offsets = zeros(FT, n_windows)
    for (local_i, window_i) in enumerate(active_windows)
        reported_offsets[window_i] = offsets[local_i]
    end
    _gauge_window_offsets!(reported_offsets, reported_window_probs)

    fallback = _windowed_local_average_free_energies(state, local_f_by_window)
    reported_f = zeros(FT, K)
    for global_state in 1:K
        gamma_tss = reported_gamma[global_state]
        if gamma_tss <= zero(FT)
            reported_f[global_state] = fallback[global_state]
            continue
        end

        value = zero(FT)
        for window_j in state.state_to_windows[global_state]
            p_j = reported_window_probs[window_j]
            p_j > zero(FT) || continue
            estimator = state.estimators[window_j]
            local_state = tss_local_index(estimator, global_state)
            value += p_j * estimator.gamma[local_state] *
                     (local_f_by_window[window_j][local_state] - reported_offsets[window_j])
        end
        reported_f[global_state] = value / gamma_tss
    end

    reported_f .-= reported_f[1]
    check_tss_finite!(reported_f, "reported free energies", state)
    check_tss_finite!(reported_offsets, "reported window offsets", state)
    check_tss_probabilities!(reported_gamma, "reported rung density", state)
    check_tss_probabilities!(reported_window_probs, "reported window probabilities", state)

    return (
        reported_f = reported_f,
        reported_gamma = reported_gamma,
        reported_offsets = reported_offsets,
        reported_window_probs = reported_window_probs,
    )
end

function compute_reported_tss_free_energies!(state::WindowedTSSState{FT};
                                             visited_only::Bool = false) where {FT}
    coupling = _windowed_tss_coupling(state)
    components = _compute_reported_tss_free_energy_components(
        state,
        _windowed_current_local_free_energies(state);
        visited_only = visited_only,
    )
    coupling.reported_window_probs .= components.reported_window_probs
    coupling.reported_gamma .= components.reported_gamma
    coupling.reported_offsets .= components.reported_offsets
    coupling.reported_f .= components.reported_f
    return coupling.reported_f
end

function apply_windowed_sampling_densities!(state::WindowedTSSState{FT}) where {FT}
    coupling = _windowed_tss_coupling(state)

    for (window_i, estimator) in enumerate(state.estimators)
        candidate = coupling.candidate_densities[window_i]
        _normalize_windowed_tss_probabilities!(
            candidate,
            "candidate density for window $(window_i)",
            state,
        )
        estimator.density .= candidate
        estimator.log_dens .= log.(estimator.density)
        check_tss_positive_probabilities!(
            estimator.density,
            "sampling density for window $(window_i)",
            estimator,
        )
    end

    return state
end

function update_windowed_tss_coupling!(state::WindowedTSSState)
    isnothing(state.coupling) && return nothing
    update_window_probabilities!(state)
    solve_windowed_visit_control!(state)
    compute_windowed_sampling_densities!(state)
    apply_windowed_sampling_densities!(state)
    compute_reported_tss_free_energies!(state)
    return state.coupling
end

function windowed_tss_free_energies(state::WindowedTSSState;
                                    reference_state::Integer = 1,
                                    visited_only::Bool = false)
    coupling = _windowed_tss_coupling(state)
    K = n_states(state.state_space)
    reference_state = Int(reference_state)
    1 <= reference_state <= K ||
        throw(ArgumentError("reference_state $(reference_state) out of bounds."))

    compute_reported_tss_free_energies!(state; visited_only = visited_only)
    reported = copy(coupling.reported_f)
    reported .-= reported[reference_state]
    return reported
end

struct TSSJackknifeResult{T}
    free_energies::Vector{T}
    standard_errors::Vector{T}
    mse::Vector{T}
    reference_state::Int
    epoch_indices::Vector{Int}
    epoch_weights::Vector{T}
    replicates::Matrix{T}
end

function _windowed_tss_jackknife_histories(state::WindowedTSSState)
    histories = map(state.estimators) do estimator
        isnothing(estimator.history) &&
            throw(ArgumentError("TSS jackknife requires history forgetting to be enabled."))
        estimator.history
    end

    config = first(histories).config
    for (window_i, history) in enumerate(histories)
        history.config.alpha == config.alpha &&
            history.config.phi == config.phi &&
            history.config.target_n_epochs == config.target_n_epochs ||
            throw(ArgumentError("TSS jackknife requires matching history-forgetting " *
                                "configuration in every window; window $(window_i) differs."))
    end

    return histories
end

function _windowed_tss_jackknife_epoch_indices!(histories, t::Int)
    epoch_indices = _tss_retained_epoch_indices!(first(histories), t)
    for history in histories
        current = _tss_retained_epoch_indices!(history, t)
        current == epoch_indices ||
            throw(ArgumentError("TSS jackknife retained epoch boundaries differ across windows."))
    end
    length(epoch_indices) >= 2 ||
        throw(ArgumentError("TSS jackknife requires at least two retained epochs; " *
                            "got $(length(epoch_indices))."))
    return epoch_indices
end

function _summarize_tss_indices(indices; max_items::Int = 8)
    values = collect(indices)
    isempty(values) && return "[]"
    if length(values) <= max_items
        return repr(values)
    end
    head = join(values[1:max_items], ", ")
    return "[$(head), ...]"
end

function _check_windowed_tss_jackknife_samples!(histories, epoch_indices)
    empty_windows = Int[]
    for (window_i, history) in enumerate(histories)
        if _tss_history_sample_count(history; epoch_indices = epoch_indices) == 0
            push!(empty_windows, window_i)
        end
    end
    isempty(empty_windows) ||
        throw(ArgumentError("TSS jackknife cannot be computed because windows " *
                            "$(_summarize_tss_indices(empty_windows)) have no samples " *
                            "in the shared retained epochs. Run longer, use more " *
                            "replicas, or reduce history forgetting."))

    invalid_deletions = Tuple{Int, Vector{Int}}[]
    for epoch_index in epoch_indices
        empty_after_delete = Int[]
        for (window_i, history) in enumerate(histories)
            if _tss_history_sample_count(history;
                    omit_epoch_index = epoch_index,
                    epoch_indices = epoch_indices,
                ) == 0
                push!(empty_after_delete, window_i)
            end
        end
        isempty(empty_after_delete) || push!(invalid_deletions,
                                             (epoch_index, empty_after_delete))
    end

    if !isempty(invalid_deletions)
        epoch_index, windows = first(invalid_deletions)
        throw(ArgumentError("TSS jackknife cannot delete every retained epoch: " *
                            "deleting epoch $(epoch_index) leaves windows " *
                            "$(_summarize_tss_indices(windows)) with no retained " *
                            "samples. Run longer, use more replicas, or reduce " *
                            "history forgetting."))
    end

    return nothing
end

function _windowed_tss_jackknife_local_free_energies(state::WindowedTSSState;
                                                     omit_epoch_index = nothing,
                                                     epoch_indices = nothing)
    return [
        _aggregate_tss_history_free_energies(estimator;
            omit_epoch_index = omit_epoch_index,
            epoch_indices = epoch_indices,
        )
        for estimator in state.estimators
    ]
end

function windowed_tss_free_energy_uncertainties(state::WindowedTSSState{FT};
                                                reference_state::Integer = 1) where {FT}
    _windowed_tss_coupling(state)

    K = n_states(state.state_space)
    reference_state = Int(reference_state)
    1 <= reference_state <= K ||
        throw(ArgumentError("reference_state $(reference_state) out of bounds."))
    state.iteration > 0 ||
        throw(ArgumentError("TSS jackknife requires at least one windowed update."))

    histories = _windowed_tss_jackknife_histories(state)
    epoch_indices = _windowed_tss_jackknife_epoch_indices!(histories, state.iteration)
    _check_windowed_tss_jackknife_samples!(histories, epoch_indices)
    epoch_weights = _tss_epoch_weights!(first(histories), epoch_indices, state.iteration)
    all(>(zero(FT)), epoch_weights) ||
        throw(ArgumentError("TSS jackknife epoch weights must be strictly positive."))

    full_local_f = _windowed_tss_jackknife_local_free_energies(
        state;
        epoch_indices = epoch_indices,
    )
    full_components = _compute_reported_tss_free_energy_components(state, full_local_f)
    full_f = copy(full_components.reported_f)
    full_f .-= full_f[reference_state]

    n_replicates = length(epoch_indices)
    replicates = zeros(FT, K, n_replicates)
    for (replicate_i, epoch_index) in enumerate(epoch_indices)
        local_f = _windowed_tss_jackknife_local_free_energies(
            state;
            omit_epoch_index = epoch_index,
            epoch_indices = epoch_indices,
        )
        components = _compute_reported_tss_free_energy_components(state, local_f)
        replicate = components.reported_f
        replicate .-= replicate[reference_state]
        replicates[:, replicate_i] .= replicate
    end

    mse = zeros(FT, K)
    denominator = FT(n_replicates - 1)
    for state_i in 1:K
        acc = zero(FT)
        for replicate_i in 1:n_replicates
            weight = epoch_weights[replicate_i]
            delta = replicates[state_i, replicate_i] - full_f[state_i]
            acc += ((one(FT) - weight)^2 / weight) * delta^2
        end
        mse[state_i] = acc / denominator
    end
    mse[reference_state] = zero(FT)
    check_tss_finite!(mse, "jackknife mean-square errors", state)

    standard_errors = sqrt.(max.(mse, zero(FT)))
    standard_errors[reference_state] = zero(FT)
    return TSSJackknifeResult(
        full_f,
        standard_errors,
        mse,
        reference_state,
        epoch_indices,
        epoch_weights,
        replicates,
    )
end

function windowed_tss_visit_control_free_energies(state::WindowedTSSState;
                                                  reference_state::Integer = 1)
    coupling = _windowed_tss_coupling(state)
    K = n_states(state.state_space)
    reference_state = Int(reference_state)
    1 <= reference_state <= K ||
        throw(ArgumentError("reference_state $(reference_state) out of bounds."))

    update_window_probabilities!(state)
    solve_windowed_visit_control!(state)
    visit_control_f = copy(coupling.visit_control_f)
    visit_control_f .-= visit_control_f[reference_state]
    return visit_control_f
end
