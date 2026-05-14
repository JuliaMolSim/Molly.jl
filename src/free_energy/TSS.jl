export
    TSSState,
    TSSSimulation,
    WindowedTSSState,
    WindowedTSSSimulation

mutable struct TSSStats{T}
    iterations::Vector{Int}
    active_state::Vector{Int}
    sampled_next_state::Vector{Int}
    max_abs_delta_f::Vector{T}
    f_history::Vector{Vector{T}}
    dens_history::Vector{Vector{T}}
    tilt_history::Vector{Vector{T}}
end

function TSSStats(::Type{FT}) where {FT}
    return TSSStats{FT}(
        Int[],
        Int[],
        Int[],
        FT[],
        Vector{FT}[],
        Vector{FT}[],
        Vector{FT}[],
    )

end

function should_log_tss(iteration::Int, log_freq::Int)
    return iteration == 1 || (iteration % log_freq == 0)
end

function tss_vector_diagnostic(name, values, state)
    bad = findall(x -> !isfinite(x), values)
    n_show = min(length(bad), 8)
    shown_bad = bad[1:n_show]
    finite_values = filter(isfinite, collect(values))
    finite_msg = isempty(finite_values) ?
                 "no finite values" :
                 "finite range $(minimum(finite_values)) to $(maximum(finite_values))"

    return "TSS $(name) contains non-finite values at iteration $(state.iteration) " *
           "with active state $(state.active_state.active_idx); " *
           "bad indices $(shown_bad)$(length(bad) > n_show ? " ..." : ""); " *
           finite_msg
end

function check_tss_finite!(values, name::AbstractString, state)
    all(isfinite, values) || throw(ArgumentError(tss_vector_diagnostic(name, values, state)))
    return values
end

function check_tss_probabilities!(weights::AbstractVector{FT},
                                  name::AbstractString,
                                  state) where {FT}
    check_tss_finite!(weights, name, state)
    if any(<(zero(FT)), weights)
        throw(ArgumentError("TSS $(name) contains negative values at iteration " *
                            "$(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end

    s = sum(weights)
    if !isfinite(s) || s <= zero(FT)
        throw(ArgumentError("TSS $(name) has invalid total weight $(s) at iteration " *
                            "$(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end
    return weights
end

function check_tss_positive_probabilities!(weights::AbstractVector{FT},
                                           name::AbstractString,
                                           state) where {FT}
    check_tss_probabilities!(weights, name, state)
    if any(<=(zero(FT)), weights)
        throw(ArgumentError("TSS $(name) contains non-positive values at iteration " *
                            "$(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end
    return weights
end

@inline function logaddexp_tss(a::T, b::T) where T
    if a == -T(Inf)
        return b
    elseif b == -T(Inf)
        return a
    end

    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

@inline function tss_log_update_arg(log_ratio::T, gain::T) where T
    gain == one(T) && return log_ratio
    return logaddexp_tss(log1p(-gain), log(gain) + log_ratio)
end


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

end

function TSSState(thermo_states::AbstractVector{<:ThermoState};
                  first_state::Int      = 1,
                  state_indices         = nothing,
                  gamma                 = nothing,
                  initial_f             = nothing,
                  ETA::Real             = 2.0,
                  dens_reg::Real        = 1e-6,
                  reuse_neighbors::Bool = true)

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

    density_raw   = FT.(gamma .* tilts.^ETA)
    density_raw ./= sum(density_raw)

    density     = FT.((1 - dens_reg) .* density_raw .+ dens_reg .* gamma)
    density   ./= sum(density)
    log_density = log.(density)

    weights = zeros(FT, local_K)
    reduced_potentials = zeros(FT, local_K)
    energies = zeros(FT, local_K) .* EU
    scratch = zeros(FT, local_K)
    log_state_bias = zeros(FT, local_K)

    stats = TSSStats(FT)

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
        stats
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

    density_raw   = FT.(state.gamma .* state.tilts.^state.ETA)
    raw_sum = sum(density_raw)
    if !isfinite(raw_sum) || raw_sum <= zero(FT)
        throw(ArgumentError("TSS raw sampling density has invalid total $(raw_sum) " *
                            "at iteration $(state.iteration) with active state " *
                            "$(state.active_state.active_idx)."))
    end
    density_raw ./= raw_sum

    state.density   = FT.((1 - state.dens_reg) .* density_raw .+ state.dens_reg .* state.gamma)
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

function update_tss_estimates!(state::TSSState{FT}; visited_state::Int) where {FT}

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
    gain   = inv(FT(t_next))

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

    state.iteration += 1

    update_tss_sampling_distribution!(state)

    return maximum(abs, delta_f .- delta_f[1])

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

struct TSSWindow
    index::Int
    state_indices::Vector{Int}
    function TSSWindow(index::Integer, state_indices)
        if !(index > 0)
            throw(ArgumentError("index must be strictly positive."))
        end

        state_indices = Int.(collect(state_indices))
        if isempty(state_indices)
            throw(ArgumentError("state_indices must be non-empty"))
        end
        if any(<=(0), state_indices)
            throw(ArgumentError("state_indices entries must be positive."))
        end
        if !allunique(state_indices)
            throw(ArgumentError("state_indices entries must be unique."))
        end

        sort!(state_indices)
        if length(state_indices) > 1 && any(!=(1), diff(state_indices))
            throw(ArgumentError("state_indices must be contiguous for linear TSS windows."))
        end

        return new(Int(index), state_indices)
    end
end

mutable struct WindowedTSSStats{T}
    iterations::Vector{Int}
    update_window::Vector{Int}
    visited_state::Vector{Int}
    sampled_next_state::Vector{Int}
    max_abs_delta_f::Vector{T}
    active_window_history::Vector{Int}
    reported_f_history::Vector{Vector{T}}
end

function WindowedTSSStats(::Type{FT}) where {FT}
    return WindowedTSSStats{FT}(
        Int[],
        Int[],
        Int[],
        Int[],
        FT[],
        Int[],
        Vector{FT}[],
    )
end

mutable struct WindowedTSSState{T, ES, AS, W, E}
    state_space::ES # Shared ExtendedStateSpace among windows
    active_state::AS # Shared ActiveThermoState among windows
    windows::Vector{W} # Window definitions
    estimators::Vector{E} # One local TSSState per window
    state_to_windows::Vector{Vector{Int}} # Global state index to containing windows
    active_window::Int # Current window index
    iteration::Int # Total number of windowed updates
    stats::WindowedTSSStats{T} # Diagnostics
end

function _coerce_tss_windows(windows, K::Int)
    isempty(windows) && throw(ArgumentError("at least one TSS window is required."))

    normalized = TSSWindow[]
    for (i, window) in enumerate(windows)
        normalized_window = if window isa TSSWindow
            window.index == i ||
                throw(ArgumentError("TSSWindow index $(window.index) must match position $(i)."))
            TSSWindow(i, window.state_indices)
        else
            TSSWindow(i, window)
        end

        if any(k -> k > K, normalized_window.state_indices)
            throw(ArgumentError("window $(i) contains state index larger than $(K)."))
        end
        push!(normalized, normalized_window)
    end

    return normalized
end

function _build_tss_state_to_windows(windows::AbstractVector{TSSWindow}, K::Int)
    state_to_windows = [Int[] for _ in 1:K]
    for window in windows
        for state_index in window.state_indices
            push!(state_to_windows[state_index], window.index)
        end
    end
    return state_to_windows
end

function _tss_window_overlap_adjacency(windows::AbstractVector{TSSWindow})
    adjacency = [Int[] for _ in eachindex(windows)]

    for i in eachindex(windows)
        for j in (i + 1):length(windows)
            if !isempty(intersect(windows[i].state_indices, windows[j].state_indices))
                push!(adjacency[i], j)
                push!(adjacency[j], i)
            end
        end
    end

    return adjacency
end

function _check_tss_window_graph_connected(windows::AbstractVector{TSSWindow})
    adjacency = _tss_window_overlap_adjacency(windows)
    visited = falses(length(windows))
    queue = [1]
    visited[1] = true

    while !isempty(queue)
        window = popfirst!(queue)
        for neighbor in adjacency[window]
            if !visited[neighbor]
                visited[neighbor] = true
                push!(queue, neighbor)
            end
        end
    end

    all(visited) || throw(ArgumentError("TSS window overlap graph must be connected."))
    return adjacency
end

function _validate_tss_window_coverage!(windows::AbstractVector{TSSWindow},
                                        state_to_windows::Vector{Vector{Int}},
                                        K::Int)
    for state_index in 1:K
        n_cover = length(state_to_windows[state_index])
        n_cover == 2 ||
            throw(ArgumentError("state $(state_index) must be covered by exactly two windows; got $(n_cover)."))
    end

    _check_tss_window_graph_connected(windows)
    return windows
end

function _windowed_global_vector(values, name::AbstractString, K::Int)
    isnothing(values) && return nothing
    values = collect(values)
    length(values) == K ||
        throw(ArgumentError("$(name) must have length $(K) for WindowedTSSState."))
    return values
end

function WindowedTSSState(thermo_states::AbstractVector{<:ThermoState};
                          windows,
                          first_state::Int = 1,
                          first_window = nothing,
                          gamma = nothing,
                          initial_f = nothing,
                          ETA::Real = 2.0,
                          dens_reg::Real = 1e-6,
                          reuse_neighbors::Bool = true)

    state_space = ExtendedStateSpace(thermo_states; reuse_neighbors = reuse_neighbors)
    K = n_states(state_space)
    1 <= first_state <= K ||
        throw(ArgumentError("first_state ($(first_state)) out of range 1:$(K)."))

    active_state = ActiveThermoState(state_space, first_state)
    FT = typeof(ustrip(active_state.active_sys.total_mass))

    normalized_windows = _coerce_tss_windows(windows, K)
    state_to_windows = _build_tss_state_to_windows(normalized_windows, K)
    _validate_tss_window_coverage!(normalized_windows, state_to_windows, K)

    first_windows = state_to_windows[first_state]
    active_window = if isnothing(first_window)
        first(first_windows)
    else
        first_window isa Integer ||
            throw(ArgumentError("first_window must be an integer window index."))
        first_window = Int(first_window)
        first_window in first_windows ||
            throw(ArgumentError("first_window must contain first_state."))
        first_window
    end

    global_gamma = _windowed_global_vector(gamma, "gamma", K)
    global_initial_f = _windowed_global_vector(initial_f, "initial_f", K)

    estimators = [
        make_tss_local_estimator(
            state_space,
            active_state;
            state_indices = window.state_indices,
            gamma = isnothing(global_gamma) ? nothing : global_gamma[window.state_indices],
            initial_f = isnothing(global_initial_f) ? nothing : global_initial_f[window.state_indices],
            ETA = ETA,
            dens_reg = dens_reg,
            require_active_state = false,
        )
        for window in normalized_windows
    ]

    stats = WindowedTSSStats(FT)

    return WindowedTSSState{
        FT,
        typeof(state_space),
        typeof(active_state),
        TSSWindow,
        eltype(estimators),
    }(
        state_space,
        active_state,
        normalized_windows,
        estimators,
        state_to_windows,
        active_window,
        0,
        stats,
    )
end

function linear_tss_windows(n_states::Integer; window_size::Integer)
    n_states > 0 || throw(ArgumentError("n_states must be positive."))
    window_size >= 2 || throw(ArgumentError("window_size must be at least 2."))
    iseven(window_size) || throw(ArgumentError("window_size must be even."))
    n_states >= window_size || throw(ArgumentError("n_states must be at least window_size."))
    n_states % window_size == 0 ||
        throw(ArgumentError("n_states must be divisible by window_size."))

    n_states = Int(n_states)
    window_size = Int(window_size)
    half_width = window_size ÷ 2
    raw_windows = Vector{Vector{Int}}()

    push!(raw_windows, collect(1:half_width))
    for start in 1:window_size:(n_states - window_size + 1)
        push!(raw_windows, collect(start:(start + window_size - 1)))
    end
    for start in (1 + half_width):window_size:(n_states - window_size + 1)
        push!(raw_windows, collect(start:(start + window_size - 1)))
    end
    push!(raw_windows, collect((n_states - half_width + 1):n_states))

    sort!(raw_windows; by = window -> (first(window), last(window), length(window)))
    unique_windows = Vector{Vector{Int}}()
    for window in raw_windows
        if !any(existing -> existing == window, unique_windows)
            push!(unique_windows, window)
        end
    end

    return [TSSWindow(i, window) for (i, window) in enumerate(unique_windows)]
end

active_tss_estimator(state::WindowedTSSState) = state.estimators[state.active_window]

window_contains_state(window::TSSWindow, global_state::Integer) =
    Int(global_state) in window.state_indices

function windows_for_state(state::WindowedTSSState, global_state::Integer)
    global_state = Int(global_state)
    1 <= global_state <= n_states(state.state_space) ||
        throw(ArgumentError("global_state $(global_state) out of bounds."))
    return state.state_to_windows[global_state]
end

function other_window_for_state(state::WindowedTSSState, global_state::Integer)
    win = windows_for_state(state, global_state)
    length(win) == 2 ||
        throw(ArgumentError("state $(global_state) is not covered by exactly two windows."))

    if state.active_window == win[1]
        return win[2]
    elseif state.active_window == win[2]
        return win[1]
    else
        throw(ArgumentError("active window $(state.active_window) does not contain state $(global_state)."))
    end
end

function switch_active_window!(state::WindowedTSSState; current_state::Integer = state.active_state.active_idx)
    state.active_window = other_window_for_state(state, current_state)
    return state
end

function _check_windowed_tss_invariant(state::WindowedTSSState)
    active_window = state.windows[state.active_window]
    window_contains_state(active_window, state.active_state.active_idx) ||
        throw(ArgumentError("active window $(state.active_window) does not contain active state " *
                            "$(state.active_state.active_idx)."))
    return state
end

function align_window_free_energies(state::WindowedTSSState{FT};
                                    reference_state::Integer = 1) where {FT}
    K = n_states(state.state_space)
    reference_state = Int(reference_state)
    1 <= reference_state <= K ||
        throw(ArgumentError("reference_state $(reference_state) out of bounds."))

    adjacency = _tss_window_overlap_adjacency(state.windows)
    reference_window = first(windows_for_state(state, reference_state))

    offsets = zeros(FT, length(state.windows))
    aligned = falses(length(state.windows))
    aligned[reference_window] = true
    queue = [reference_window]

    while !isempty(queue)
        window_i = popfirst!(queue)
        estimator_i = state.estimators[window_i]

        for window_j in adjacency[window_i]
            aligned[window_j] && continue

            estimator_j = state.estimators[window_j]
            shared_states = intersect(state.windows[window_i].state_indices,
                                      state.windows[window_j].state_indices)
            isempty(shared_states) && continue

            offset_sum = zero(FT)
            for global_state in shared_states
                local_i = tss_local_index(estimator_i, global_state)
                local_j = tss_local_index(estimator_j, global_state)
                offset_sum += estimator_i.f[local_i] + offsets[window_i] - estimator_j.f[local_j]
            end

            offsets[window_j] = offset_sum / FT(length(shared_states))
            aligned[window_j] = true
            push!(queue, window_j)
        end
    end

    all(aligned) || throw(ArgumentError("could not align all TSS windows."))

    reported = zeros(FT, K)
    counts = zeros(Int, K)
    residuals = FT[]

    for (window_index, estimator) in enumerate(state.estimators)
        for global_state in state.windows[window_index].state_indices
            local_state = tss_local_index(estimator, global_state)
            reported[global_state] += estimator.f[local_state] + offsets[window_index]
            counts[global_state] += 1
        end
    end

    all(>(0), counts) || throw(ArgumentError("not all states have aligned free-energy contributions."))
    for state_index in 1:K
        reported[state_index] /= FT(counts[state_index])
    end

    for i in eachindex(state.windows)
        for j in (i + 1):length(state.windows)
            shared_states = intersect(state.windows[i].state_indices, state.windows[j].state_indices)
            for global_state in shared_states
                local_i = tss_local_index(state.estimators[i], global_state)
                local_j = tss_local_index(state.estimators[j], global_state)
                push!(residuals,
                      state.estimators[i].f[local_i] + offsets[i] -
                      state.estimators[j].f[local_j] - offsets[j])
            end
        end
    end

    reported .-= reported[reference_state]
    return reported, offsets, residuals
end

function windowed_tss_free_energies(state::WindowedTSSState; reference_state::Integer = 1)
    reported, _, _ = align_window_free_energies(state; reference_state = reference_state)
    return reported
end

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
        state.iteration += 1

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
