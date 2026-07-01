const TSS_COVDET_GAMMA_EPSILON = 0.01

struct TSSCovDetAdaptiveGamma{T}
    epsilon_gamma::T
    rung_neighbors::Vector{Vector{NTuple{3, Int}}}
    rung_volumes::Vector{T}
    dimension::Int
end

function validate_tss_local_adaptive_gamma(adaptive_gamma)
    isnothing(adaptive_gamma) && return nothing
    adaptive_gamma isa TSSCovDetAdaptiveGamma && return adaptive_gamma
    if adaptive_gamma isa Symbol
        if adaptive_gamma == :covdet
            throw(ArgumentError("adaptive_gamma=:covdet is only supported by TSSState " *
                                "because it requires a TSSGraph."))
        end
        throw(ArgumentError("unknown TSS adaptive_gamma mode $(adaptive_gamma); " *
                            "the only supported mode is :covdet."))
    end
    throw(ArgumentError("TSS adaptive_gamma accepts only nothing or :covdet."))
end

function ensure_tss_adaptive_moments!(state, n_moments::Int)
    if isnothing(state.adaptive_moments)
        state.adaptive_moments = zeros(eltype(state.f), length(state.f), n_moments)
    elseif size(state.adaptive_moments) != (length(state.f), n_moments)
        throw(ArgumentError("TSS adaptive-gamma moment dimension changed from " *
                            "$(size(state.adaptive_moments, 2)) to $(n_moments)."))
    end
    return state.adaptive_moments
end

function ensure_tss_epoch_adaptive_moments!(epoch::TSSEpoch{FT},
                                             n_states::Int,
                                             n_moments::Int) where {FT}
    if isnothing(epoch.adaptive_moments)
        epoch.adaptive_moments = zeros(FT, n_states, n_moments)
    elseif size(epoch.adaptive_moments) != (n_states, n_moments)
        throw(ArgumentError("TSS epoch adaptive-gamma moment dimension changed."))
    end
    return epoch.adaptive_moments
end

function update_tss_adaptive_moments!(moments::AbstractMatrix{FT},
                                       old_f::AbstractVector{FT},
                                       reduced_pot::AbstractVector{FT},
                                       log_den::FT,
                                       gain::FT,
                                       adaptive_values::AbstractMatrix{FT}) where {FT}
    if size(moments) != size(adaptive_values)
        throw(DimensionMismatch("TSS adaptive moments and values must have matching sizes."))
    end
    if !(length(old_f) == length(reduced_pot) == size(moments, 1))
        throw(DimensionMismatch("TSS adaptive moments must match local state count."))
    end

    log_gain = log(gain)
    log_keep = gain == one(FT) ? -FT(Inf) : log1p(-gain)
    for k in axes(moments, 1)
        log_old_z = -old_f[k]
        log_sample_z = -reduced_pot[k] - log_den
        log_new_z = logaddexp_tss(log_keep + log_old_z, log_gain + log_sample_z)
        old_weight = exp(log_keep + log_old_z - log_new_z)
        sample_weight = exp(log_gain + log_sample_z - log_new_z)
        for m in axes(moments, 2)
            moments[k, m] = old_weight * moments[k, m] +
                            sample_weight * adaptive_values[k, m]
        end
    end

    if !all(isfinite, moments)
        throw(ArgumentError("TSS adaptive-gamma moments became non-finite."))
    end
    return moments
end

function update_tss_running_adaptive_moments!(state,
                                               old_f::AbstractVector,
                                               log_den,
                                               gain,
                                               adaptive_values)
    isnothing(adaptive_values) && return state
    FT = eltype(state.f)
    values = FT.(adaptive_values)
    moments =ensure_tss_adaptive_moments!(state, size(values, 2))
   update_tss_adaptive_moments!(
        moments,
        FT.(old_f),
        state.reduced_pot,
        FT(log_den),
        FT(gain),
        values,
    )
    return state
end

function aggregate_tss_history_adaptive_moments!(state)
    isnothing(state.adaptive_gamma) && return state
    history = state.history
    isnothing(history) && return state

    FT = eltype(state.f)

    n_moments = 0
    for epoch in history.epochs
        epoch.count > 0 || continue
        isnothing(epoch.adaptive_moments) && continue
        n_moments = size(epoch.adaptive_moments, 2)
        break
    end
    n_moments == 0 && return state

    moments = ensure_tss_adaptive_moments!(state, n_moments)
    for k in axes(moments, 1)
        log_norm = -FT(Inf)
        log_weights = FT[]
        epochs = TSSEpoch{FT}[]
        for epoch in history.epochs
            epoch.count > 0 || continue
            isnothing(epoch.adaptive_moments) && continue
            log_weight = log(FT(epoch.count)) - epoch.f[k]
            push!(log_weights, log_weight)
            push!(epochs, epoch)
            log_norm = logaddexp_tss(log_norm, log_weight)
        end
        isempty(epochs) && continue

        for m in axes(moments, 2)
            value = zero(FT)
            for (epoch_i, epoch) in enumerate(epochs)
                value += exp(log_weights[epoch_i] - log_norm) *
                         epoch.adaptive_moments[k, m]
            end
            moments[k, m] = value
        end
    end

    if !all(isfinite, moments)
        throw(ArgumentError("TSS history-aggregated adaptive-gamma moments are non-finite."))
    end
    return state
end

tss_covdet_moment_count(dim::Int) = dim + dim * dim
tss_covdet_outer_col(dim::Int, i::Int, j::Int) = dim + (j - 1) * dim + i

function tss_covdet_moment_values(state,
                                   evaluation_reduced_pot = state.evaluation_reduced_pot)
    adaptive_gamma = state.adaptive_gamma
    adaptive_gamma isa TSSCovDetAdaptiveGamma || return nothing

    FT = eltype(state.f)
    dim = adaptive_gamma.dimension
    n_moments = tss_covdet_moment_count(dim)
    values = zeros(FT, length(state.state_indices), n_moments)
    derivatives = zeros(FT, dim)

    for (local_i, global_state) in enumerate(state.state_indices)
        neighbors = adaptive_gamma.rung_neighbors[global_state]
        if length(neighbors) != dim
            throw(ArgumentError("TSS CovDet rung $(global_state) has $(length(neighbors)) " *
                                "derivative dimensions, expected $(dim)."))
        end

        for d in 1:dim
            reverse, forward, denominator = neighbors[d]
            if denominator == 0
                derivatives[d] = zero(FT)
                continue
            end
            reverse_eval = state.evaluation_local_index_by_state[reverse]
            forward_eval = state.evaluation_local_index_by_state[forward]
            if reverse_eval == 0 || forward_eval == 0
                throw(ArgumentError("TSS CovDet derivative for rung $(global_state) " *
                                    "requires states $(reverse) and $(forward), but they " *
                                    "were not included in the window evaluation set."))
            end
            derivatives[d] = (FT(evaluation_reduced_pot[forward_eval]) -
                              FT(evaluation_reduced_pot[reverse_eval])) / FT(denominator)
            values[local_i, d] = derivatives[d]
        end

        for j in 1:dim, i in 1:dim
            values[local_i,tss_covdet_outer_col(dim, i, j)] =
                derivatives[i] * derivatives[j]
        end
    end

    if !all(isfinite, values)
        throw(ArgumentError("TSS CovDet adaptive-gamma moments contain non-finite values."))
    end
    return values
end

function tss_covdet_raw_values(state)
    adaptive_gamma = state.adaptive_gamma
    adaptive_gamma isa TSSCovDetAdaptiveGamma || return nothing
    isnothing(state.adaptive_moments) && return nothing

    FT = eltype(state.f)
    dim = adaptive_gamma.dimension
    if size(state.adaptive_moments, 2) != tss_covdet_moment_count(dim)
        throw(ArgumentError("TSS CovDet adaptive moments have invalid dimension."))
    end

    raw = zeros(FT, length(state.state_indices))
    covariance = Matrix{Float64}(undef, dim, dim)
    for local_i in eachindex(raw)
        for j in 1:dim, i in 1:dim
            mean_outer = Float64(state.adaptive_moments[
                local_i,
               tss_covdet_outer_col(dim, i, j),
            ])
            mean_i = Float64(state.adaptive_moments[local_i, i])
            mean_j = Float64(state.adaptive_moments[local_i, j])
            covariance[i, j] = mean_outer - mean_i * mean_j
        end
        for j in 1:dim, i in 1:(j - 1)
            value = 0.5 * (covariance[i, j] + covariance[j, i])
            covariance[i, j] = value
            covariance[j, i] = value
        end
        detcov = dim == 1 ? covariance[1, 1] : det(Symmetric(covariance))
        raw[local_i] = FT(sqrt(max(detcov, 0.0)))
    end
    if !all(isfinite, raw)
        throw(ArgumentError("TSS CovDet adaptive-gamma estimates are non-finite."))
    end
    return raw
end

function volume_weighted_tss_gamma!(state)
    adaptive_gamma = state.adaptive_gamma
    adaptive_gamma isa TSSCovDetAdaptiveGamma || return state
    FT = eltype(state.f)
    weights = FT.(adaptive_gamma.rung_volumes)
    total = sum(weights)
    if !(isfinite(total) && total > zero(FT))
        throw(ArgumentError("TSS CovDet rung volumes have invalid total $(total)."))
    end
    state.gamma .= weights ./ total
    state.log_gamma .= log.(state.gamma)
    check_tss_positive_probabilities!(state.gamma, "CovDet adaptive gamma", state)
    return state
end

function apply_tss_covdet_gamma!(state, raw_values, max_detcov)
    adaptive_gamma = state.adaptive_gamma
    adaptive_gamma isa TSSCovDetAdaptiveGamma || return state

    FT = eltype(state.f)
    raw = isnothing(raw_values) ? zeros(FT, length(state.gamma)) : FT.(raw_values)
    if length(raw) != length(state.gamma)
        throw(ArgumentError("TSS CovDet adaptive gamma has invalid length $(length(raw)); " *
                            "expected $(length(state.gamma))."))
    end
    if !all(isfinite, raw)
        throw(ArgumentError("TSS CovDet adaptive gamma contains non-finite raw values."))
    end

    if !isfinite(max_detcov) || max_detcov <= zero(FT)
        return volume_weighted_tss_gamma!(state)
    end

    epsilon = FT(adaptive_gamma.epsilon_gamma)
    for k in eachindex(state.gamma)
        state.gamma[k] = ((one(FT) - epsilon) * max(raw[k], zero(FT)) +
                          epsilon * FT(max_detcov)) *
                         FT(adaptive_gamma.rung_volumes[k])
    end
    total = sum(state.gamma)
    if !(isfinite(total) && total > zero(FT))
        throw(ArgumentError("TSS CovDet adaptive gamma has invalid total $(total)."))
    end
    state.gamma ./= total
    state.log_gamma .= log.(state.gamma)
    check_tss_positive_probabilities!(state.gamma, "CovDet adaptive gamma", state)
    return state
end

function update_tss_adaptive_gamma!(state)
    isnothing(state.adaptive_gamma) && return state
    raw =tss_covdet_raw_values(state)
    max_detcov = isnothing(raw) ? zero(eltype(state.f)) : maximum(raw)
    return apply_tss_covdet_gamma!(state, raw, max_detcov)
end
