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

function _windowed_local_average_free_energies(state::WindowedTSSState{FT}) where {FT}
    K = n_states(state.state_space)
    values = zeros(FT, K)
    counts = zeros(Int, K)

    for (window_i, estimator) in enumerate(state.estimators)
        for global_state in state.windows[window_i].state_indices
            local_state = tss_local_index(estimator, global_state)
            values[global_state] += estimator.f[local_state]
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

function compute_reported_tss_free_energies!(state::WindowedTSSState{FT};
                                             visited_only::Bool = false) where {FT}
    coupling = _windowed_tss_coupling(state)
    active_mask = _reported_active_mask(state, visited_only)
    solve_window_probability_eigenvector!(
        state;
        use_tilts = false,
        visited_mask = active_mask,
        output = coupling.reported_window_probs,
        transition = zeros(FT, length(state.windows), length(state.windows)),
    )

    fill!(coupling.reported_gamma, zero(FT))
    K = n_states(state.state_space)
    for (window_j, estimator) in enumerate(state.estimators)
        p_j = coupling.reported_window_probs[window_j]
        p_j > zero(FT) || continue
        for local_state in eachindex(estimator.state_indices)
            global_state = estimator.state_indices[local_state]
            coupling.reported_gamma[global_state] += p_j * estimator.gamma[local_state]
        end
    end

    reported_total = sum(coupling.reported_gamma)
    isfinite(reported_total) && reported_total > zero(FT) ||
        throw(ArgumentError("TSS reported rung density has invalid total $(reported_total)."))
    coupling.reported_gamma ./= reported_total

    active_windows = findall(>(zero(FT)), coupling.reported_window_probs)
    isempty(active_windows) &&
        throw(ArgumentError("no TSS windows are available for reported free-energy estimation."))

    n_active = length(active_windows)
    global_weighted_f = zeros(FT, K)
    for global_state in 1:K
        gamma_tss = coupling.reported_gamma[global_state]
        gamma_tss > zero(FT) || continue
        for window_j in state.state_to_windows[global_state]
            p_j = coupling.reported_window_probs[window_j]
            p_j > zero(FT) || continue
            estimator = state.estimators[window_j]
            local_state = tss_local_index(estimator, global_state)
            global_weighted_f[global_state] +=
                p_j * estimator.gamma[local_state] * estimator.f[local_state] / gamma_tss
        end
    end

    transition = zeros(FT, n_active, n_active)
    rhs = zeros(FT, n_active)
    for (local_i, window_i) in enumerate(active_windows)
        estimator_i = state.estimators[window_i]

        for global_state in state.windows[window_i].state_indices
            gamma_tss = coupling.reported_gamma[global_state]
            gamma_tss > zero(FT) || continue
            local_i_state = tss_local_index(estimator_i, global_state)
            gamma_i = estimator_i.gamma[local_i_state]
            rhs[local_i] += gamma_i *
                            (estimator_i.f[local_i_state] - global_weighted_f[global_state])

            for (local_j, window_j) in enumerate(active_windows)
                window_j in state.state_to_windows[global_state] || continue
                estimator_j = state.estimators[window_j]
                local_j_state = tss_local_index(estimator_j, global_state)
                transition[local_i, local_j] +=
                    gamma_i * coupling.reported_window_probs[window_j] *
                    estimator_j.gamma[local_j_state] / gamma_tss
            end
        end
    end

    A = Matrix{FT}(I, n_active, n_active) - transition
    b = rhs
    A[n_active, :] .= coupling.reported_window_probs[active_windows]
    b[n_active] = zero(FT)

    offsets = pinv(A) * b
    fill!(coupling.reported_offsets, zero(FT))
    for (local_i, window_i) in enumerate(active_windows)
        coupling.reported_offsets[window_i] = offsets[local_i]
    end
    _gauge_window_offsets!(coupling.reported_offsets, coupling.reported_window_probs)

    fallback = _windowed_local_average_free_energies(state)
    for global_state in 1:K
        gamma_tss = coupling.reported_gamma[global_state]
        if gamma_tss <= zero(FT)
            coupling.reported_f[global_state] = fallback[global_state]
            continue
        end

        value = zero(FT)
        for window_j in state.state_to_windows[global_state]
            p_j = coupling.reported_window_probs[window_j]
            p_j > zero(FT) || continue
            estimator = state.estimators[window_j]
            local_state = tss_local_index(estimator, global_state)
            value += p_j * estimator.gamma[local_state] *
                     (estimator.f[local_state] - coupling.reported_offsets[window_j])
        end
        coupling.reported_f[global_state] = value / gamma_tss
    end

    coupling.reported_f .-= coupling.reported_f[1]
    check_tss_finite!(coupling.reported_f, "reported free energies", state)
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
