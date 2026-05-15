struct TSSWindow
    index::Int
    state_indices::Vector{Int}
    function TSSWindow(index::Integer, state_indices; check_contiguous::Bool = true)
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

        if check_contiguous
            sort!(state_indices)
            if length(state_indices) > 1 && any(!=(1), diff(state_indices))
                throw(ArgumentError("state_indices must be contiguous for linear TSS windows; " *
                                    "use check_contiguous=false for advanced non-linear windows."))
            end
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
    visit_control_converged::Vector{Bool}
    visit_control_iterations::Vector{Int}
    visit_control_max_abs_residual::Vector{T}
    window_prob_history::Vector{Vector{T}}
    visit_control_f_history::Vector{Vector{T}}
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
        Bool[],
        Int[],
        FT[],
        Vector{FT}[],
        Vector{FT}[],
    )
end

mutable struct WindowedTSSCoupling{T}
    visit_control_f::Vector{T}
    window_probs::Vector{T}
    window_transition::Matrix{T}
    global_rung_weights::Vector{T}
    window_offsets::Vector{T}
    lhs_marginal::Vector{T}
    rhs_marginal::Vector{T}
    residual::Vector{T}
    candidate_densities::Vector{Vector{T}}
    reported_window_probs::Vector{T}
    reported_gamma::Vector{T}
    reported_offsets::Vector{T}
    reported_f::Vector{T}
    iterations::Int
    converged::Bool
    max_abs_residual::T
    tolerance::T
    max_iterations::Int
    damping::T
    pi_regularization::T
end

mutable struct WindowedTSSState{T, ES, AS, W, E}
    state_space::ES # Shared ExtendedStateSpace among windows
    active_state::AS # Shared ActiveThermoState among windows
    windows::Vector{W} # Window definitions
    estimators::Vector{E} # One local TSSState per window
    state_to_windows::Vector{Vector{Int}} # Global state index to containing windows
    active_window::Int # Current window index
    window_update_counts::Vector{Int} # Number of estimator updates per window
    iteration::Int # Total number of windowed updates
    stats::WindowedTSSStats{T} # Diagnostics
    coupling::Union{Nothing, WindowedTSSCoupling{T}} # Global visit-control state
end

function _coerce_tss_windows(windows, K::Int)
    isempty(windows) && throw(ArgumentError("at least one TSS window is required."))

    normalized = TSSWindow[]
    for (i, window) in enumerate(windows)
        normalized_window = if window isa TSSWindow
            window.index == i ||
                throw(ArgumentError("TSSWindow index $(window.index) must match position $(i)."))
            TSSWindow(i, window.state_indices; check_contiguous = false)
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
                          reuse_neighbors::Bool = true,
                          history_forgetting = nothing,
                          global_visit_control::Bool = true,
                          visit_control_tolerance::Real = 1e-8,
                          visit_control_max_iterations::Integer = 1_000,
                          visit_control_damping::Real = 1.0,
                          pi_regularization::Real = 1e-3)

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
            history_forgetting = history_forgetting,
            require_active_state = false,
        )
        for window in normalized_windows
    ]

    stats = WindowedTSSStats(FT)

    state = WindowedTSSState{
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
        zeros(Int, length(normalized_windows)),
        0,
        stats,
        nothing,
    )

    if global_visit_control
        state.coupling = initialize_windowed_tss_coupling(
            state;
            tolerance = visit_control_tolerance,
            max_iterations = visit_control_max_iterations,
            damping = visit_control_damping,
            pi_regularization = pi_regularization,
        )
        update_windowed_tss_coupling!(state)
    end

    return state
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

function _periodic_tss_window_indices(n_states::Int, start::Int, window_size::Int)
    return [mod1(start + offset, n_states) for offset in 0:(window_size - 1)]
end

function periodic_tss_windows(n_states::Integer; window_size::Integer)
    n_states > 0 || throw(ArgumentError("n_states must be positive."))
    window_size >= 2 || throw(ArgumentError("window_size must be at least 2."))
    iseven(window_size) || throw(ArgumentError("window_size must be even."))
    n_states >= window_size || throw(ArgumentError("n_states must be at least window_size."))

    n_states = Int(n_states)
    window_size = Int(window_size)
    stride = window_size ÷ 2
    n_states % stride == 0 ||
        throw(ArgumentError("n_states must be divisible by half the window_size for periodic TSS windows."))

    starts = collect(1:stride:n_states)
    return [
        TSSWindow(
            window_i,
            _periodic_tss_window_indices(n_states, start, window_size);
            check_contiguous = false,
        )
        for (window_i, start) in enumerate(starts)
    ]
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
    length(state.window_update_counts) == length(state.windows) ||
        throw(ArgumentError("window update counts do not match the number of TSS windows."))
    if !isnothing(state.coupling)
        coupling = state.coupling
        K = n_states(state.state_space)
        length(coupling.visit_control_f) == K ||
            throw(ArgumentError("visit-control free energies do not match the number of states."))
        length(coupling.window_probs) == length(state.windows) ||
            throw(ArgumentError("visit-control window probabilities do not match the number of windows."))
        length(coupling.candidate_densities) == length(state.windows) ||
            throw(ArgumentError("visit-control candidate densities do not match the number of windows."))
    end
    return state
end
