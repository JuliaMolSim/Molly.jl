"""
    TSSAdaptiveGamma(observable, gamma_from_means; epsilon_gamma=0.01,
                     device_policy=:auto)

Configure adaptive reference-density estimates for Times Square Sampling.

`observable` is evaluated at each TSS estimator update and may be a
[`TSSCVObservable`](@ref), [`TSSSystemObservable`](@ref), or any callable that
accepts a [`TSSObservableContext`](@ref). Its return value is converted to a
numeric matrix with one row per local TSS state. Scalars and vectors are treated
as configuration observables and broadcast to every local state; matrices must
have size `n_local_states x n_observable_components`.

`gamma_from_means(state_indices, means)` receives the retained-history means and
must return positive unnormalised gamma values for the local states.
`epsilon_gamma` mixes the normalized result with a local uniform density to keep
all states reachable.

`device_policy` controls how observable inputs are passed:

- `:auto`: built-in CV observables are evaluated on CPU copies via
  [`from_device`](@ref); other observables receive the native system.
- `:cpu`: evaluate with CPU copies of system arrays.
- `:native`: pass the live system and arrays as-is.
"""
struct TSSAdaptiveGamma{O, G, T}
    observable::O
    gamma_from_means::G
    epsilon_gamma::T
    device_policy::Symbol
end

function TSSAdaptiveGamma(observable, gamma_from_means;
                          epsilon_gamma::Real = 0.01,
                          device_policy = :auto)
    isfinite(epsilon_gamma) && 0 <= epsilon_gamma <= 1 ||
        throw(ArgumentError("epsilon_gamma must be finite and in the [0, 1] interval."))
    policy = _validate_tss_observable_device_policy(device_policy)
    FT = typeof(float(epsilon_gamma))
    return TSSAdaptiveGamma{typeof(observable), typeof(gamma_from_means), FT}(
        observable,
        gamma_from_means,
        FT(epsilon_gamma),
        policy,
    )
end

"""
    TSSCVObservable(cv; transform=identity)

Use a Molly collective variable as a TSS observable. The wrapper evaluates
`calculate_cv(cv, coords, atoms, boundary, velocities)` on the active system and
then applies `transform` to the result.

For periodic CVs such as torsions, use `transform` or a custom observable when
the adaptive gamma rule needs a continuous representation, e.g. return
`SVector(sin(phi), cos(phi))` instead of the raw wrapped angle.
"""
struct TSSCVObservable{CV, F}
    cv::CV
    transform::F
end

TSSCVObservable(cv; transform = identity) =
    TSSCVObservable{typeof(cv), typeof(transform)}(cv, transform)

"""
    TSSSystemObservable(observable)

Adapt a logger-style Molly observable for TSS adaptive gamma. The wrapped
function is called as

```julia
observable(sys, buffers, neighbors, step_n; n_threads)
```

matching the convention used by [`GeneralObservableLogger`](@ref) and related
loggers. TSS does not mutate logger objects; this wrapper only reuses the
observable function signature.
"""
struct TSSSystemObservable{F}
    observable::F
end

"""
    TSSObservableContext

Context passed to raw TSS adaptive-gamma observables. User observables can read
the active system, state indices, current reduced potentials, and update
metadata from this object and return a scalar, vector, or `n_local_states x M`
matrix of finite numeric values.
"""
struct TSSObservableContext{AS, ES, B, N, E, R, T}
    active_state::AS
    state_space::ES
    state_indices::Vector{Int}
    active_global_state::Int
    active_local_state::Int
    energies::E
    reduced_potentials::R
    log_den::T
    step::Int
    buffers::B
    neighbors::N
    n_threads::Int
    device_policy::Symbol
end

function _validate_tss_observable_device_policy(device_policy)
    policy = device_policy isa Symbol ? device_policy : Symbol(device_policy)
    policy in (:auto, :cpu, :native) ||
        throw(ArgumentError("device_policy must be one of :auto, :cpu, or :native."))
    return policy
end

function _tss_cpu_active_state(active_state::ActiveThermoState)
    sys = active_state.active_sys
    cpu_sys = System(
        sys;
        atoms = from_device(sys.atoms),
        coords = from_device(sys.coords),
        velocities = from_device(sys.velocities),
    )
    return ActiveThermoState(
        active_state.active_idx,
        cpu_sys,
        active_state.active_integrator,
    )
end

function _tss_cpu_observable_context(context::TSSObservableContext)
    return TSSObservableContext(
        _tss_cpu_active_state(context.active_state),
        context.state_space,
        context.state_indices,
        context.active_global_state,
        context.active_local_state,
        from_device(context.energies),
        from_device(context.reduced_potentials),
        context.log_den,
        context.step,
        context.buffers,
        context.neighbors,
        context.n_threads,
        context.device_policy,
    )
end

function _tss_context_for_observable(::TSSCVObservable, context::TSSObservableContext)
    if context.device_policy in (:auto, :cpu)
        return _tss_cpu_observable_context(context)
    end
    return context
end

function _tss_context_for_observable(::TSSSystemObservable, context::TSSObservableContext)
    context.device_policy == :cpu && return _tss_cpu_observable_context(context)
    return context
end

function _tss_context_for_observable(observable, context::TSSObservableContext)
    context.device_policy == :cpu && return _tss_cpu_observable_context(context)
    return context
end

function evaluate_tss_observable(observable::TSSCVObservable, context::TSSObservableContext)
    context = _tss_context_for_observable(observable, context)
    sys = context.active_state.active_sys
    value = calculate_cv(
        observable.cv,
        sys.coords,
        sys.atoms,
        sys.boundary,
        sys.velocities,
    )
    return observable.transform(value)
end

function evaluate_tss_observable(observable::TSSSystemObservable,
                                 context::TSSObservableContext)
    context = _tss_context_for_observable(observable, context)
    return observable.observable(
        context.active_state.active_sys,
        context.buffers,
        context.neighbors,
        context.step;
        n_threads = context.n_threads,
    )
end

function evaluate_tss_observable(observable, context::TSSObservableContext)
    context = _tss_context_for_observable(observable, context)
    return observable(context)
end

_tss_numeric_value(::Type{FT}, value) where {FT} = FT(ustrip(value))

function _tss_observable_matrix(values,
                                ::Type{FT},
                                n_local_states::Int,
                                name::AbstractString) where {FT}
    n_local_states > 0 ||
        throw(ArgumentError("TSS $(name) requires at least one local state."))

    if values isa Number
        out = fill(_tss_numeric_value(FT, values), n_local_states, 1)
    else
        values_cpu = if values isa Tuple
            collect(values)
        else
            from_device(values)
        end

        if values_cpu isa AbstractVector
            row = [_tss_numeric_value(FT, value) for value in values_cpu]
            out = Matrix{FT}(undef, n_local_states, length(row))
            for k in 1:n_local_states
                out[k, :] .= row
            end
        elseif values_cpu isa AbstractMatrix
            size(values_cpu, 1) == n_local_states ||
                throw(ArgumentError("TSS $(name) matrix must have $(n_local_states) " *
                                    "rows, got $(size(values_cpu, 1))."))
            out = Matrix{FT}(undef, size(values_cpu, 1), size(values_cpu, 2))
            for idx in eachindex(values_cpu)
                out[idx] = _tss_numeric_value(FT, values_cpu[idx])
            end
        else
            throw(ArgumentError("TSS $(name) must return a scalar, vector, tuple, " *
                                "or matrix, got $(typeof(values))."))
        end
    end

    size(out, 2) > 0 ||
        throw(ArgumentError("TSS $(name) must contain at least one observable component."))
    all(isfinite, out) ||
        throw(ArgumentError("TSS $(name) contains non-finite values."))
    return out
end

function _tss_observable_context(state,
                                 active_state::ActiveThermoState;
                                 log_den,
                                 history_time::Int,
                                 energies = state.energies,
                                 reduced_potentials = state.reduced_pot,
                                 buffers = nothing,
                                 neighbors = nothing,
                                 n_threads::Int = Threads.nthreads())
    return TSSObservableContext(
        active_state,
        state.state_space,
        state.state_indices,
        active_state.active_idx,
        tss_local_index(state, active_state.active_idx),
        energies,
        reduced_potentials,
        log_den,
        history_time,
        buffers,
        neighbors,
        n_threads,
        state.adaptive_gamma.device_policy,
    )
end

function _evaluate_tss_adaptive_observable(state,
                                           active_state::ActiveThermoState;
                                           log_den,
                                           history_time::Int,
                                           energies = state.energies,
                                           reduced_potentials = state.reduced_pot,
                                           buffers = nothing,
                                           neighbors = nothing,
                                           n_threads::Int = Threads.nthreads())
    isnothing(state.adaptive_gamma) && return nothing
    context = _tss_observable_context(
        state,
        active_state;
        log_den = log_den,
        history_time = history_time,
        energies = energies,
        reduced_potentials = reduced_potentials,
        buffers = buffers,
        neighbors = neighbors,
        n_threads = n_threads,
    )
    values = evaluate_tss_observable(state.adaptive_gamma.observable, context)
    return _tss_observable_matrix(
        values,
        eltype(state.f),
        length(state.f),
        "adaptive-gamma observable",
    )
end

function _ensure_tss_observable_means!(state, n_observables::Int)
    if isnothing(state.observable_means)
        state.observable_means = zeros(eltype(state.f), length(state.f), n_observables)
    elseif size(state.observable_means) != (length(state.f), n_observables)
        throw(ArgumentError("TSS adaptive-gamma observable dimension changed from " *
                            "$(size(state.observable_means, 2)) to $(n_observables)."))
    end
    return state.observable_means
end

function _ensure_tss_epoch_observable_means!(epoch::TSSEpoch{FT},
                                             n_states::Int,
                                             n_observables::Int) where {FT}
    if isnothing(epoch.observable_means)
        epoch.observable_means = zeros(FT, n_states, n_observables)
    elseif size(epoch.observable_means) != (n_states, n_observables)
        throw(ArgumentError("TSS epoch adaptive-gamma observable dimension changed."))
    end
    return epoch.observable_means
end

function _update_tss_observable_means!(means::AbstractMatrix{FT},
                                       old_f::AbstractVector{FT},
                                       reduced_pot::AbstractVector{FT},
                                       log_den::FT,
                                       gain::FT,
                                       observable_values::AbstractMatrix{FT}) where {FT}
    size(means) == size(observable_values) ||
        throw(DimensionMismatch("TSS observable means and values must have matching sizes."))
    length(old_f) == length(reduced_pot) == size(means, 1) ||
        throw(DimensionMismatch("TSS observable means must match local state count."))

    log_gain = log(gain)
    log_keep = gain == one(FT) ? -FT(Inf) : log1p(-gain)
    for k in axes(means, 1)
        log_old_z = -old_f[k]
        log_sample_z = -reduced_pot[k] - log_den
        log_new_z = logaddexp_tss(log_keep + log_old_z, log_gain + log_sample_z)
        old_weight = exp(log_keep + log_old_z - log_new_z)
        sample_weight = exp(log_gain + log_sample_z - log_new_z)
        for m in axes(means, 2)
            means[k, m] = old_weight * means[k, m] +
                          sample_weight * observable_values[k, m]
        end
    end

    all(isfinite, means) ||
        throw(ArgumentError("TSS adaptive-gamma observable means became non-finite."))
    return means
end

function _update_tss_running_observable_means!(state,
                                               old_f::AbstractVector,
                                               log_den,
                                               gain,
                                               observable_values)
    isnothing(observable_values) && return state
    FT = eltype(state.f)
    values = FT.(observable_values)
    means = _ensure_tss_observable_means!(state, size(values, 2))
    _update_tss_observable_means!(
        means,
        FT.(old_f),
        state.reduced_pot,
        FT(log_den),
        FT(gain),
        values,
    )
    return state
end

function _aggregate_tss_history_observable_means!(state)
    isnothing(state.adaptive_gamma) && return state
    history = state.history
    isnothing(history) && return state

    FT = eltype(state.f)

    n_observables = 0
    for epoch in history.epochs
        epoch.count > 0 || continue
        isnothing(epoch.observable_means) && continue
        n_observables = size(epoch.observable_means, 2)
        break
    end
    n_observables == 0 && return state

    means = _ensure_tss_observable_means!(state, n_observables)
    for k in axes(means, 1)
        log_norm = -FT(Inf)
        log_weights = FT[]
        epochs = TSSEpoch{FT}[]
        for epoch in history.epochs
            epoch.count > 0 || continue
            isnothing(epoch.observable_means) && continue
            log_weight = log(FT(epoch.count)) - epoch.f[k]
            push!(log_weights, log_weight)
            push!(epochs, epoch)
            log_norm = logaddexp_tss(log_norm, log_weight)
        end
        isempty(epochs) && continue

        for m in axes(means, 2)
            value = zero(FT)
            for (epoch_i, epoch) in enumerate(epochs)
                value += exp(log_weights[epoch_i] - log_norm) *
                         epoch.observable_means[k, m]
            end
            means[k, m] = value
        end
    end

    all(isfinite, means) ||
        throw(ArgumentError("TSS history-aggregated adaptive-gamma means are non-finite."))
    return state
end

function _update_tss_adaptive_gamma!(state)
    isnothing(state.adaptive_gamma) && return state
    isnothing(state.observable_means) && return state

    FT = eltype(state.f)
    raw_values = state.adaptive_gamma.gamma_from_means(
        state.state_indices,
        state.observable_means,
    )
    raw_source = raw_values isa Tuple ? collect(raw_values) : from_device(raw_values)
    raw = FT.(collect(vec(raw_source)))
    length(raw) == length(state.gamma) ||
        throw(ArgumentError("TSS adaptive gamma rule must return $(length(state.gamma)) " *
                            "values, got $(length(raw))."))
    all(isfinite, raw) ||
        throw(ArgumentError("TSS adaptive gamma rule returned non-finite values."))
    all(>(zero(FT)), raw) ||
        throw(ArgumentError("TSS adaptive gamma rule must return strictly positive values."))

    total = sum(raw)
    total > zero(FT) ||
        throw(ArgumentError("TSS adaptive gamma rule returned an invalid total $(total)."))
    raw ./= total

    epsilon = FT(state.adaptive_gamma.epsilon_gamma)
    uniform = inv(FT(length(raw)))
    @. state.gamma = (one(FT) - epsilon) * raw + epsilon * uniform
    state.gamma ./= sum(state.gamma)
    state.log_gamma .= log.(state.gamma)
    check_tss_positive_probabilities!(state.gamma, "adaptive gamma", state)
    return state
end
