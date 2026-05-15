mutable struct TSSState{T, ES, AS, ST}

    state_space::ES  # The different hamiltonians
    active_state::AS # The hamiltonian that is currently active

    state_indices::Vector{Int} # Local index to global state index
    local_index_by_state::Vector{Int} # Global index to local index, 0 mean not in local estimator

    f::Vector{T} # Free energy estimate, dimensionless
    gamma::Vector{T} # Akin to target distribution. Must be positive and normalised!
    log_gamma::Vector{T} # Avoids recomputation of this magnitude that appears a lot

    tilts::Vector{T} # Empirical visit control of the rungs
    density::Vector{T} # Current TSS sampling density over rungs
    log_dens::Vector{T} # Log of prev. quantity
    
    weights::Vector{T} # Conditional state probabilities
    reduced_pot::Vector{T} # u_k(x) values for last configuration
    energies::Vector{ST} # Raw potential energy, before reduced
    scratch::Vector{T} # Used for log-sum-exp 
    log_state_bias::Vector{T} # f .+ log_dens in log form

    iteration::Int # Number of updates already applied
    ETA::T # Visit-control strength. ETA == 0 disables visit control, gives dens == gamma
    dens_reg::T # Small interp. towards gamma, keeps all states reachable
    stats::TSSStats{T} # Diagnostics
    history::Union{Nothing, TSSEpochHistory{T}} # Optional epoch history forgetting state

end

function TSSState(thermo_states::AbstractVector{<:ThermoState};
                  first_state::Int      = 1,
                  state_indices         = nothing,
                  gamma                 = nothing,
                  initial_f             = nothing,
                  ETA::Real             = 2.0,
                  dens_reg::Real        = 1e-6,
                  reuse_neighbors::Bool = true,
                  history_forgetting    = nothing)

    state_space  = ExtendedStateSpace(thermo_states; reuse_neighbors = reuse_neighbors)
    K  = n_states(state_space)

    if !(1 <= first_state <= K)
        throw(ArgumentError("First state must be larger or equal than 1 and smaller or equal than $(K)."))
    end

    active_state = ActiveThermoState(state_space, first_state)

    return make_tss_local_estimator(
        state_space,
        active_state;
        state_indices = state_indices,
        gamma = gamma,
        initial_f = initial_f,
        ETA = ETA,
        dens_reg = dens_reg,
        history_forgetting = history_forgetting,
        require_active_state = true,
    )

end

function make_tss_local_estimator(state_space::ExtendedStateSpace,
                                  active_state::ActiveThermoState;
                                  state_indices = nothing,
                                  gamma = nothing,
                                  initial_f = nothing,
                                  ETA::Real = 2.0,
                                  dens_reg::Real = 1e-6,
                                  history_forgetting = nothing,
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
    scratch = zeros(FT, local_K)
    log_state_bias = zeros(FT, local_K)

    stats = TSSStats(FT)
    history = isnothing(history_forgetting) ?
              nothing :
              TSSEpochHistory(history_forgetting, FT, local_K)

    return TSSState(
        state_space, 
        active_state,
        state_indices,
        local_index_by_state,
        initial_f,
        gamma,
        log.(gamma),
        tilts,
        density,
        log_density,
        weights,
        reduced_potentials,
        energies,
        scratch,
        log_state_bias,
        0,
        FT(ETA),
        FT(dens_reg),
        stats,
        history
    )

end

function tss_local_index(state::TSSState, global_state::Int)
    if !(1 <= global_state <= n_states(state.state_space))
        throw(ArgumentError("global_state $(global_state) out of bounds."))
    end
    local_idx = state.local_index_by_state[global_state]
    if local_idx == 0
        throw(ArgumentError("$(global_state) does not map to any local state."))
    end
    return local_idx
end

function tss_global_index(state::TSSState, local_state::Int)
    if !(1 <= local_state <= length(state.state_indices))
        throw(ArgumentError("$(local_state) out of bounds."))
    end
    return state.state_indices[local_state]
end

function tss_active_local_index(state::TSSState)
    idx = state.active_state.active_idx
    return tss_local_index(state, idx)
end

function tss_sample_global_state(rng::AbstractRNG, state::TSSState)
    idx = sample_state(rng, state.weights)
    return tss_global_index(state, idx)
end

function process_tss_sample!(state::TSSState{FT}) where {FT}
    coords = state.active_state.active_sys.coords
    boundary = state.active_state.active_sys.boundary

    iter = state.state_indices

    evaluate_energy_subset!(
        state.energies,
        state.state_space.partition,
        coords,
        boundary,
        iter,
    )

    reduced_potentials!(
        state.reduced_pot,
        state.energies,
        state.state_space,
        boundary,
        iter,
    )

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

function update_tss_sampling_distribution!(state::TSSState{FT}) where {FT}

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
    isfinite(log_norm) ||
        throw(ArgumentError("TSS raw sampling density has non-finite log normalization " *
                            "$(log_norm) at iteration $(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))

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

function update_tss_estimates!(state::TSSState{FT};
                               visited_state::Int,
                               history_time = nothing) where {FT}

    visited_local = tss_local_index(state, visited_state)

    if !(1 <= visited_state <= n_states(state.state_space))
        throw(ArgumentError("Visited state $(visited_state) out of state space."))
    end

    check_tss_probabilities!(state.weights, "conditional weights", state)
    check_tss_positive_probabilities!(state.density, "sampling density", state)
    check_tss_finite!(state.f, "free energy estimates", state)
    check_tss_finite!(state.reduced_pot, "reduced potentials", state)

    @. state.log_state_bias = state.f + state.log_dens
    check_tss_finite!(state.log_state_bias, "log state bias", state)
    @. state.scratch = state.log_state_bias - state.reduced_pot
    log_den = logsumexp(state.scratch)
    isfinite(log_den) || throw(ArgumentError("TSS log normalization is non-finite " *
                                             "($(log_den)) at iteration " *
                                             "$(state.iteration) with active state " *
                                             "$(state.active_state.active_idx)."))

    t_next = state.iteration + 1
    history_time_int = if isnothing(history_time)
        t_next
    else
        history_time isa Integer ||
            throw(ArgumentError("history_time must be an integer."))
        Int(history_time)
    end
    history_time_int > 0 ||
        throw(ArgumentError("history_time must be positive."))
    old_f = copy(state.f)
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

        for k in eachindex(state.tilts)
            target = (k == visited_local ? one(FT) : zero(FT)) / state.gamma[k]
            state.tilts[k] += gain * (target - state.tilts[k])
        end
        check_tss_finite!(state.tilts, "visit tilts", state)

        if !isnothing(state.history)
            _update_tss_history!(
                state,
                visited_local,
                log_den,
                history_time_int;
                aggregate = false,
            )
        end
    else
        _update_tss_history!(state, visited_local, log_den, history_time_int)
    end

    state.iteration += 1

    update_tss_sampling_distribution!(state)

    return maximum(abs, state.f .- old_f)

end

function log_tss_stats!(
    stats::TSSStats{FT},
    state::TSSState{FT},
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

mutable struct TSSSimulation{S}
    state::S
    n_md_steps::Int
    n_cycles::Int
    self_adjustment_steps::Int
    log_freq::Int
end

function TSSSimulation(state::TSSState;
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

    return TSSSimulation(
        state,
        n_md_steps,
        n_cycles,
        Int(self_adjustment_steps),
        log_freq
    )

end

function simulate!(sim::TSSSimulation; rng = Random.default_rng())

    state::TSSState = sim.state

    for cycle in 1:sim.n_cycles

        final_visited_state = state.active_state.active_idx
        final_next_state    = state.active_state.active_idx

        for substep in 1:sim.self_adjustment_steps

            visited_state = state.active_state.active_idx

            simulate!(
                state.active_state.active_sys,
                state.active_state.active_integrator,
                sim.n_md_steps
            )

            process_tss_sample!(state)
            next_state = tss_sample_global_state(rng, state)

            final_visited_state = visited_state
            final_next_state    = next_state

            if substep < sim.self_adjustment_steps
                set_active_state!(state.active_state, state.state_space, next_state)
            end

        end

        max_df = update_tss_estimates!(state; visited_state = final_visited_state)

        if should_log_tss(state.iteration, sim.log_freq)
            log_tss_stats!(state.stats, state, final_visited_state, final_next_state, max_df)
        end

        set_active_state!(state.active_state, state.state_space, final_next_state)

    end

    return state

end
