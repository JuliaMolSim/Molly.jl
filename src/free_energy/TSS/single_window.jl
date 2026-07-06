mutable struct TSSLocalEstimator{T, ES, AS, ST, AG}

    state_space::ES  # The different hamiltonians
    active_state::AS # The hamiltonian that is currently active

    state_indices::Vector{Int} # Local index to global state index
    local_index_by_state::Vector{Int} # Global index to local index, 0 mean not in local estimator
    evaluation_state_indices::Vector{Int} # States whose reduced potentials are evaluated for this estimator
    evaluation_local_index_by_state::Vector{Int} # Global index to evaluation index, 0 means not evaluated

    f::Vector{T} # Free energy estimate, dimensionless
    gamma::Vector{T} # Akin to target distribution. Must be positive and normalised!
    log_gamma::Vector{T} # Avoids recomputation of this magnitude that appears a lot

    tilts::Vector{T} # Empirical visit control of the rungs
    density::Vector{T} # Current TSS sampling density over rungs
    log_dens::Vector{T} # Log of prev. quantity

    weights::Vector{T} # Conditional state probabilities
    reduced_pot::Vector{T} # u_k(x) values for last configuration
    energies::Vector{ST} # Raw potential energy, before reduced
    evaluation_reduced_pot::Vector{T} # u_k(x) values for evaluation_state_indices
    evaluation_energies::Vector{ST} # Raw energies for evaluation_state_indices
    scratch::Vector{T} # Used for log-sum-exp
    log_state_bias::Vector{T} # f .+ log_dens in log form

    iteration::Int # Number of updates already applied
    ETA::T # Visit-control strength. ETA == 0 disables visit control, gives dens == gamma
    dens_reg::T # Small interp. towards gamma, keeps all states reachable
    stats::TSSStats{T} # Diagnostics
    history::Union{Nothing, TSSEpochHistory{T}} # Optional epoch history forgetting state
    adaptive_gamma::AG # Optional adaptive reference-density estimator
    adaptive_moments::Union{Nothing, Matrix{T}} # Retained moments for adaptive gamma

end

function Base.show(io::IO, state::TSSLocalEstimator)
    print(io, "_TSSLocalEstimator with ", tss_count(length(state.state_indices), "state"),
          ", active state ", state.active_state.active_idx,
          ", iteration ", state.iteration)
end

Base.show(io::IO, ::MIME"text/plain", state::TSSLocalEstimator) = show(io, state)

function make_tss_local_estimator(state_space::ExtendedStateSpace,
                                  active_state::ActiveThermoState;
                                  state_indices = nothing,
                                  evaluation_state_indices = nothing,
                                  gamma = nothing,
                                  initial_f = nothing,
                                  ETA::Real = 2.0,
                                  dens_reg::Real = 1e-6,
                                  history_forgetting = nothing,
                                  adaptive_gamma = nothing,
                                  require_active_state::Bool = false)

    FT = typeof(ustrip(active_state.active_sys.total_mass))
    EU = active_state.active_sys.energy_units
    K  = n_states(state_space)

    if !(K >= 1)
        throw(ArgumentError("Number of states must be larger or equal than 1."))
    end

    if isnothing(state_indices)
        state_indices = collect(Base.OneTo(K))
    else
        state_indices = Int.(collect(state_indices))
        if length(state_indices) > K
            throw(ArgumentError("state_indices cannot be longer than $(K)."))
        end
        if length(state_indices) == 0
            throw(ArgumentError("state_indices must be non-empty."))
        end
        if any(i -> !(1 <= i <= K), state_indices)
            throw(ArgumentError("state_indices entries must be in 1:$(K)."))
        end
        if !allunique(state_indices)
            throw(ArgumentError("state_indices entries must be unique."))
        end
    end

    if require_active_state && !(active_state.active_idx ∈ state_indices)
        throw(ArgumentError("active state must be contained in state_indices"))
    end

    local_index_by_state = zeros(Int, K)
    for (local_k, global_k) in enumerate(state_indices)
        local_index_by_state[global_k] = local_k
    end

    if isnothing(evaluation_state_indices)
        evaluation_state_indices = copy(state_indices)
    else
        evaluation_state_indices = unique(Int.(collect(vcat(state_indices, evaluation_state_indices))))
        if any(i -> !(1 <= i <= K), evaluation_state_indices)
            throw(ArgumentError("evaluation_state_indices entries must be in 1:$(K)."))
        end
    end

    evaluation_local_index_by_state = zeros(Int, K)
    for (eval_k, global_k) in enumerate(evaluation_state_indices)
        evaluation_local_index_by_state[global_k] = eval_k
    end

    if ETA < 0
        throw(ArgumentError("ETA must be larger or equal than 0."))
    end

    if !(0 < dens_reg < 1)
        throw(ArgumentError("dens_reg must be contained in the (0, 1) interval."))
    end

    local_K = length(state_indices)

    if isnothing(gamma)
        gamma = fill(inv(FT(local_K)), local_K)
    else
        if !(length(gamma) == local_K)
            throw(ArgumentError("gamma must be of length $(local_K)."))
        end

        if !all(isfinite, gamma)
            throw(ArgumentError("All gamma values must be finite."))
        end
        if !all(>(zero(FT)), gamma)
            throw(ArgumentError("All gamma values must be stictly positive."))
        end
        if sum(gamma) <= 0
            throw(ArgumentError("sum(gamma) must be strictly positive."))
        end
        gamma = FT.(gamma)
        s = sum(gamma)
        gamma ./= s
    end

    if isnothing(initial_f)
        initial_f = zeros(FT, local_K)
    else
        if !(length(initial_f) == local_K)
            throw(ArgumentError("initial_f must be of length $(local_K)."))
        end
        if !(length(initial_f) == length(state_indices))
            throw(ArgumentError("initial_f and state_indices must have the same length."))
        end
        if !all(isfinite, initial_f)
            throw(ArgumentError("All values of initial_f mus be finite."))
        end
        initial_f = FT.(initial_f)
        initial_f .-= initial_f[1]
    end

    tilts = ones(FT, local_K)

    density     = copy(gamma)
    log_density = log.(density)

    weights = zeros(FT, local_K)
    reduced_potentials = zeros(FT, local_K)
    energies = zeros(FT, local_K) .* EU
    evaluation_reduced_potentials = zeros(FT, length(evaluation_state_indices))
    evaluation_energies = zeros(FT, length(evaluation_state_indices)) .* EU
    scratch = zeros(FT, local_K)
    log_state_bias = zeros(FT, local_K)

    stats = TSSStats(FT)
    history = isnothing(history_forgetting) ?
              nothing :
              TSSEpochHistory(history_forgetting, FT, local_K)

    adaptive_gamma = validate_tss_local_adaptive_gamma(adaptive_gamma)
    adaptive_moments = nothing

    return TSSLocalEstimator{
        FT,
        typeof(state_space),
        typeof(active_state),
        eltype(energies),
        typeof(adaptive_gamma),
    }(
        state_space,
        active_state,
        state_indices,
        local_index_by_state,
        evaluation_state_indices,
        evaluation_local_index_by_state,
        initial_f,
        gamma,
        log.(gamma),
        tilts,
        density,
        log_density,
        weights,
        reduced_potentials,
        energies,
        evaluation_reduced_potentials,
        evaluation_energies,
        scratch,
        log_state_bias,
        0,
        FT(ETA),
        FT(dens_reg),
        stats,
        history,
        adaptive_gamma,
        adaptive_moments,
    )

end

function tss_local_index(state::TSSLocalEstimator, global_state::Int)
    if !(1 <= global_state <= n_states(state.state_space))
        throw(ArgumentError("global_state $(global_state) out of bounds."))
    end
    local_idx = state.local_index_by_state[global_state]
    if local_idx == 0
        throw(ArgumentError("$(global_state) does not map to any local state."))
    end
    return local_idx
end

function tss_global_index(state::TSSLocalEstimator, local_state::Int)
    if !(1 <= local_state <= length(state.state_indices))
        throw(ArgumentError("$(local_state) out of bounds."))
    end
    return state.state_indices[local_state]
end

function tss_sample_global_state(rng::AbstractRNG, state::TSSLocalEstimator)
    idx = sample_state(rng, state.weights)
    return tss_global_index(state, idx)
end

function process_tss_sample!(state::TSSLocalEstimator{FT}, active_state::ActiveThermoState) where {FT}
    coords = active_state.active_sys.coords
    boundary = active_state.active_sys.boundary

    evaluate_energy_subset!(
        state.evaluation_energies,
        state.state_space.partition,
        coords,
        boundary,
        state.evaluation_state_indices,
    )

    reduced_potentials!(
        state.evaluation_reduced_pot,
        state.evaluation_energies,
        state.state_space,
        boundary,
        state.evaluation_state_indices,
    )
    check_tss_finite!(state.evaluation_reduced_pot, "evaluation reduced potentials", state)

    for (local_k, global_k) in enumerate(state.state_indices)
        eval_k = state.evaluation_local_index_by_state[global_k]
        state.energies[local_k] = state.evaluation_energies[eval_k]
        state.reduced_pot[local_k] = state.evaluation_reduced_pot[eval_k]
    end

    check_tss_finite!(state.reduced_pot, "reduced potentials", state)

    @. state.log_state_bias = state.f + state.log_dens
    check_tss_finite!(state.log_state_bias, "log state bias", state)
    conditional_state_weights!(
        state.weights,
        state.log_state_bias,
        state.reduced_pot,
        state.scratch
    )
    check_tss_probabilities!(state.weights, "conditional weights", state)

    return state.weights

end

function process_tss_sample!(state::TSSLocalEstimator{FT}) where {FT}
    return process_tss_sample!(state, state.active_state)
end

function tss_log_den!(state::TSSLocalEstimator)
    @. state.log_state_bias = state.f + state.log_dens
    check_tss_finite!(state.log_state_bias, "log state bias", state)
    @. state.scratch = state.log_state_bias - state.reduced_pot
    log_den = logsumexp(state.scratch)
    isfinite(log_den) || throw(ArgumentError("TSS log normalization is non-finite " *
                                             "($(log_den)) at iteration " *
                                             "$(state.iteration) with active state " *
                                             "$(state.active_state.active_idx)."))
    return log_den
end

function update_tss_sampling_distribution!(state::TSSLocalEstimator{FT}) where {FT}

    check_tss_finite!(state.tilts, "visit tilts", state)
    if any(<(zero(FT)), state.tilts)
        throw(ArgumentError("TSS visit tilts contain negative values at iteration " *
                            "$(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end

    tilt_floor = tss_tilt_floor(FT)
    if iszero(state.ETA)
        state.scratch .= state.log_gamma
    else
        for k in eachindex(state.scratch)
            state.scratch[k] = state.log_gamma[k] - state.ETA * log(max(state.tilts[k], tilt_floor))
        end
    end

    log_norm = logsumexp(state.scratch)
    if !isfinite(log_norm)
        throw(ArgumentError("TSS raw sampling density has non-finite log normalization " *
                            "$(log_norm) at iteration $(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end

    for k in eachindex(state.density)
        state.density[k] = exp(state.scratch[k] - log_norm)
    end
    @. state.density = (one(FT) - state.dens_reg) * state.density + state.dens_reg * state.gamma
    s = sum(state.density)
    if !isfinite(s) || s <= zero(FT)
        throw(ArgumentError("TSS sampling density has invalid total $(s) " *
                            "at iteration $(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end
    state.density ./= s
    state.log_dens = log.(state.density)
    check_tss_positive_probabilities!(state.density, "sampling density", state)

    return

end

function update_tss_estimates!(state::TSSLocalEstimator{FT};
                               visited_state::Int,
                               history_time = nothing,
                               adaptive_values = nothing,
                               update_adaptive_gamma::Bool = true) where {FT}

    visited_local = tss_local_index(state, visited_state)

    if !(1 <= visited_state <= n_states(state.state_space))
        throw(ArgumentError("Visited state $(visited_state) out of state space."))
    end

    check_tss_probabilities!(state.weights, "conditional weights", state)
    check_tss_positive_probabilities!(state.density, "sampling density", state)
    check_tss_finite!(state.f, "free energy estimates", state)
    check_tss_finite!(state.reduced_pot, "reduced potentials", state)

    log_den = tss_log_den!(state)

    t_next = state.iteration + 1
    history_time_int = if isnothing(history_time)
        t_next
    else
        if !(history_time isa Integer)
            throw(ArgumentError("history_time must be an integer."))
        end
        Int(history_time)
    end
    if history_time_int <= 0
        throw(ArgumentError("history_time must be positive."))
    end
    old_f = copy(state.f)
    if isnothing(adaptive_values)
        adaptive_values = tss_covdet_moment_values(state)
    end
    use_standard_update = isnothing(state.history) || iszero(state.history.config.alpha)

    if use_standard_update
        gain = inv(FT(t_next))

        delta_f = similar(state.f)
        @inbounds for k in eachindex(delta_f)
            log_ratio = state.f[k] - state.reduced_pot[k] - log_den
            delta_f[k] = -tss_log_update_arg(log_ratio, gain)
        end
        check_tss_finite!(delta_f, "free energy update", state)

        @. state.f += delta_f
        @. state.f -= state.f[1]
        check_tss_finite!(state.f, "free energy estimates", state)
        update_tss_running_adaptive_moments!(
            state,
            old_f,
            log_den,
            gain,
            adaptive_values,
        )

        for k in eachindex(state.tilts)
            target = (k == visited_local ? one(FT) : zero(FT)) / state.gamma[k]
            state.tilts[k] += gain * (target - state.tilts[k])
        end
        check_tss_finite!(state.tilts, "visit tilts", state)

        if !isnothing(state.history)
            update_tss_history!(
                state,
                visited_local,
                log_den,
                history_time_int;
                adaptive_values = adaptive_values,
                aggregate = false,
            )
        end
    else
        update_tss_history!(
            state,
            visited_local,
            log_den,
            history_time_int;
            adaptive_values = adaptive_values,
        )
    end

    state.iteration += 1

    update_adaptive_gamma && update_tss_adaptive_gamma!(state)
    update_tss_sampling_distribution!(state)

    return maximum(abs, state.f .- old_f)

end

function log_tss_stats!(
    stats::TSSStats{FT},
    state::TSSLocalEstimator{FT},
    visited_state::Int,
    next_state::Int,
    max_delta_f::FT) where {FT}

    push!(stats.iterations, state.iteration)
    push!(stats.active_state, visited_state)
    push!(stats.sampled_next_state, next_state)
    push!(stats.max_abs_delta_f, max_delta_f)
    push!(stats.f_history, copy(state.f))
    push!(stats.dens_history, copy(state.density))
    push!(stats.tilt_history, copy(state.tilts))

    return stats

end
