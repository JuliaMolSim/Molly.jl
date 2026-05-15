struct TSSHistoryForgetting{T}
    alpha::T
    phi::T
    target_n_epochs::Int
end

function TSSHistoryForgetting(; alpha::Real = 0.19,
                              n_epochs::Integer = 32,
                              phi = nothing)
    isfinite(alpha) && 0 <= alpha < 1 ||
        throw(ArgumentError("alpha must be finite and in the [0, 1) interval."))
    n_epochs > 0 ||
        throw(ArgumentError("n_epochs must be positive."))

    phi_value = if isnothing(phi)
        iszero(alpha) ? 1.2 : alpha^(-inv(n_epochs))
    else
        phi
    end
    isfinite(phi_value) && phi_value > 1 ||
        throw(ArgumentError("phi must be finite and greater than 1."))

    FT = promote_type(typeof(float(alpha)), typeof(float(phi_value)))
    return TSSHistoryForgetting{FT}(FT(alpha), FT(phi_value), Int(n_epochs))
end

function _convert_tss_history_config(::Type{FT},
                                     config::TSSHistoryForgetting) where {FT}
    return TSSHistoryForgetting{FT}(
        FT(config.alpha),
        FT(config.phi),
        config.target_n_epochs,
    )
end

mutable struct TSSEpoch{T}
    index::Int
    count::Int
    f::Vector{T}
    tilts::Vector{T}
end

mutable struct TSSEpochHistory{T}
    config::TSSHistoryForgetting{T}
    taus::Vector{Int}
    epochs::Vector{TSSEpoch{T}}
end

function TSSEpoch(index::Integer, ::Type{FT}, n_states::Integer) where {FT}
    return TSSEpoch{FT}(
        Int(index),
        0,
        zeros(FT, n_states),
        zeros(FT, n_states),
    )
end

function TSSEpochHistory(config::TSSHistoryForgetting,
                         ::Type{FT},
                         n_states::Integer) where {FT}
    n_states > 0 || throw(ArgumentError("history n_states must be positive."))
    return TSSEpochHistory{FT}(
        _convert_tss_history_config(FT, config),
        [0, 1],
        TSSEpoch{FT}[],
    )
end

function _ensure_tss_epoch_bounds!(history::TSSEpochHistory, t::Integer)
    t <= last(history.taus) && return history.taus

    while t > last(history.taus)
        previous = last(history.taus)
        next_tau = ceil(Int, history.config.phi * previous)
        push!(history.taus, max(previous + 1, next_tau))
    end

    return history.taus
end

function _tss_epoch_index!(history::TSSEpochHistory, t)
    t <= 0 && return 1
    t_int = ceil(Int, t)
    _ensure_tss_epoch_bounds!(history, t_int)
    return searchsortedfirst(history.taus, t_int)
end

function _tss_epoch_for_update!(history::TSSEpochHistory{FT},
                                t::Integer,
                                n_states::Integer) where {FT}
    epoch_index = _tss_epoch_index!(history, t)
    existing = findfirst(epoch -> epoch.index == epoch_index, history.epochs)
    if isnothing(existing)
        push!(history.epochs, TSSEpoch(epoch_index, FT, n_states))
        return last(history.epochs)
    end
    return history.epochs[existing]
end

function _drop_old_tss_epochs!(history::TSSEpochHistory, t::Integer)
    first_recent = _tss_epoch_index!(
        history,
        floor(Int, history.config.alpha * t),
    )
    filter!(epoch -> epoch.index >= first_recent, history.epochs)
    return history
end

function _tss_retained_epoch_indices!(history::TSSEpochHistory, t::Integer)
    t > 0 || throw(ArgumentError("TSS jackknife requires a positive history time."))
    _ensure_tss_epoch_bounds!(history, t)
    first_recent = max(2, _tss_epoch_index!(history, floor(Int, history.config.alpha * t)))
    current = _tss_epoch_index!(history, t)
    current >= first_recent ||
        throw(ArgumentError("TSS jackknife could not identify retained epochs at time $(t)."))
    return collect(first_recent:current)
end

function _tss_epoch_weights!(history::TSSEpochHistory{FT},
                             epoch_indices::AbstractVector{Int},
                             t::Integer) where {FT}
    isempty(epoch_indices) && return FT[]
    _ensure_tss_epoch_bounds!(history, t)

    first_epoch = first(epoch_indices)
    first_epoch >= 2 ||
        throw(ArgumentError("TSS epoch indices must be at least 2."))
    denominator = FT(t - history.taus[first_epoch - 1])
    denominator > zero(FT) ||
        throw(ArgumentError("TSS jackknife retained-history duration must be positive."))

    weights = FT[]
    for epoch_index in epoch_indices
        epoch_index <= length(history.taus) ||
            throw(ArgumentError("TSS epoch index $(epoch_index) has no stored boundary."))
        lower = history.taus[epoch_index - 1]
        upper = min(history.taus[epoch_index], t)
        duration = upper - lower
        duration > 0 ||
            throw(ArgumentError("TSS epoch $(epoch_index) has non-positive duration $(duration)."))
        push!(weights, FT(duration) / denominator)
    end

    return weights
end

function tss_recent_count(state)
    history = getfield(state, :history)
    if isnothing(history)
        return state.iteration
    end
    isempty(history.epochs) && return 0
    return sum(epoch.count for epoch in history.epochs)
end

function tss_retained_epoch_count(state)
    history = getfield(state, :history)
    isnothing(history) && return 0
    return length(history.epochs)
end

function _aggregate_tss_history!(state)
    history = state.history
    total_count = tss_recent_count(state)
    total_count > 0 || return state

    FT = eltype(state.f)
    for k in eachindex(state.f)
        log_z = -FT(Inf)
        tilt_sum = zero(FT)
        for epoch in history.epochs
            epoch.count > 0 || continue
            log_z = logaddexp_tss(log_z, log(FT(epoch.count)) - epoch.f[k])
            tilt_sum += FT(epoch.count) * epoch.tilts[k]
        end
        state.f[k] = -(log_z - log(FT(total_count)))
        state.tilts[k] = tilt_sum / FT(total_count)
    end

    @. state.f -= state.f[1]
    check_tss_finite!(state.f, "history-aggregated free energies", state)
    check_tss_finite!(state.tilts, "history-aggregated visit tilts", state)
    return state
end

function _aggregate_tss_history_free_energies(state;
                                              omit_epoch_index = nothing,
                                              epoch_indices = nothing)
    history = state.history
    isnothing(history) &&
        throw(ArgumentError("TSS jackknife requires history forgetting to be enabled."))

    retained = isnothing(epoch_indices) ? nothing : Set(epoch_indices)
    total_count = 0
    for epoch in history.epochs
        epoch.count > 0 || continue
        !isnothing(retained) && !(epoch.index in retained) && continue
        !isnothing(omit_epoch_index) && epoch.index == omit_epoch_index && continue
        total_count += epoch.count
    end
    total_count > 0 ||
        throw(ArgumentError("TSS jackknife deletion of epoch $(omit_epoch_index) leaves " *
                            "window with no retained samples."))

    FT = eltype(state.f)
    f = similar(state.f)
    for k in eachindex(f)
        log_z = -FT(Inf)
        for epoch in history.epochs
            epoch.count > 0 || continue
            !isnothing(retained) && !(epoch.index in retained) && continue
            !isnothing(omit_epoch_index) && epoch.index == omit_epoch_index && continue
            log_z = logaddexp_tss(log_z, log(FT(epoch.count)) - epoch.f[k])
        end
        f[k] = -(log_z - log(FT(total_count)))
    end
    f .-= f[1]
    check_tss_finite!(f, "history-aggregated jackknife free energies", state)
    return f
end

function _update_tss_history!(state,
                              visited_local::Int,
                              log_den,
                              history_time::Int;
                              aggregate::Bool = true)
    FT = eltype(state.f)
    history = state.history
    history_time > 0 ||
        throw(ArgumentError("history_time must be positive."))
    epoch = _tss_epoch_for_update!(history, history_time, length(state.f))
    epoch.count += 1
    gain = inv(FT(epoch.count))

    for k in eachindex(epoch.f)
        log_ratio = epoch.f[k] - state.reduced_pot[k] - log_den
        epoch.f[k] -= tss_log_update_arg(log_ratio, gain)
    end
    check_tss_finite!(epoch.f, "epoch free energies", state)

    for k in eachindex(epoch.tilts)
        target = (k == visited_local ? one(FT) : zero(FT)) / state.gamma[k]
        epoch.tilts[k] += gain * (target - epoch.tilts[k])
    end
    check_tss_finite!(epoch.tilts, "epoch visit tilts", state)

    _drop_old_tss_epochs!(history, history_time)
    aggregate && _aggregate_tss_history!(state)
    return state
end
