struct TSSWindow
    index::Int
    state_indices::Vector{Int}
    evaluation_state_indices::Vector{Int}
    function TSSWindow(index::Integer, state_indices;
                       evaluation_state_indices = state_indices,
                       check_contiguous::Bool = true)
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

        evaluation_state_indices = unique(Int.(collect(vcat(state_indices, evaluation_state_indices))))
        if isempty(evaluation_state_indices)
            throw(ArgumentError("evaluation_state_indices must be non-empty"))
        end
        if any(<=(0), evaluation_state_indices)
            throw(ArgumentError("evaluation_state_indices entries must be positive."))
        end
        if !allunique(evaluation_state_indices)
            throw(ArgumentError("evaluation_state_indices entries must be unique."))
        end
        if !all(state_index -> state_index in evaluation_state_indices, state_indices)
            throw(ArgumentError("evaluation_state_indices must contain all state_indices."))
        end

        return new(Int(index), state_indices, evaluation_state_indices)
    end
end

function Base.show(io::IO, window::TSSWindow)
    print(io, "TSSWindow ", window.index, " with ",
          tss_count(length(window.state_indices), "state"), ", ",
          tss_count(length(window.evaluation_state_indices), "evaluated state"))
end

Base.show(io::IO, ::MIME"text/plain", window::TSSWindow) = show(io, window)

# TSSGraph
#
# A window graph for TSS expanded-ensemble states.
#
# The graph stores the global state count, overlapping local windows, state to
# window membership, neighbor relationships between rungs, and the geometric
# metadata used by adaptive TSS estimators.
struct TSSGraph
    n_states::Int
    windows::Vector{TSSWindow}
    state_to_windows::Vector{Vector{Int}}
    rung_neighbors::Vector{Vector{NTuple{3, Int}}}
    rung_volumes::Vector{Float64}
end

function Base.show(io::IO, graph::TSSGraph)
    print(io, "TSSGraph with ", tss_count(graph.n_states, "state"), ", ",
          tss_count(length(graph.windows), "window"))
end

Base.show(io::IO, ::MIME"text/plain", graph::TSSGraph) = show(io, graph)

struct TSSGraphEdge
    nodes::Any
    shape::Vector{Int}
    window_size::Vector{Int}
    periodic::Vector{Bool}
    primary_window_tiling_only::Bool
end

# TSSGraphBuilder()
#
# Create an empty builder for a multi-edge TSS graph.
#
# Add one or more edges with add_tss_edge!, then call build_tss_graph to
# construct the immutable TSSGraph.
mutable struct TSSGraphBuilder
    edges::Vector{TSSGraphEdge}
end

TSSGraphBuilder() = TSSGraphBuilder(TSSGraphEdge[])

struct TSSPartialMembership
    dimension::Int
    side::Int
end

struct TSSDimWindow
    start::Int
    size::Int
    partials::Vector{TSSPartialMembership}
end

struct TSSWindowSpec
    sort_key::Vector{Int}
    state_indices::Vector{Int}
    partial_signature::Union{Nothing, Tuple{Vararg{String}}}
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
    replica_indices::Vector{Vector{Int}}
    replica_update_windows::Vector{Vector{Int}}
    replica_visited_states::Vector{Vector{Int}}
    replica_sampled_next_states::Vector{Vector{Int}}
    replica_max_abs_delta_f::Vector{Vector{T}}
end

function Base.show(io::IO, stats::WindowedTSSStats)
    print(io, "WindowedTSSStats with ",
          tss_count(length(stats.iterations), "logged entry", "logged entries"),
          " and ", tss_count(length(stats.replica_indices), "replica log entry",
                              "replica log entries"))
end

Base.show(io::IO, ::MIME"text/plain", stats::WindowedTSSStats) = show(io, stats)

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
        Vector{Int}[],
        Vector{Int}[],
        Vector{Int}[],
        Vector{Int}[],
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

function Base.show(io::IO, coupling::WindowedTSSCoupling)
    print(io, "WindowedTSSCoupling with ",
          tss_count(length(coupling.visit_control_f), "state"), ", ",
          tss_count(length(coupling.window_probs), "window"), ", converged=",
          coupling.converged, ", iterations=", coupling.iterations)
end

Base.show(io::IO, ::MIME"text/plain", coupling::WindowedTSSCoupling) = show(io, coupling)

"""
    TSSState
    TSSState(thermo_states; graph=nothing, first_state=1, first_window=nothing,
             gamma=nothing, initial_f=nothing, ETA=2.0, dens_reg=1e-6,
             reuse_neighbors=true, history_forgetting=nothing,
             adaptive_gamma=nothing, global_visit_control=true,
             visit_control_tolerance=1e-8,
             visit_control_max_iterations=1000,
             visit_control_damping=1.0, pi_regularization=1e-3)

Mutable state for a TSS simulation over a set of thermodynamic states.

`thermo_states` defines the global expanded ensemble. `graph` may be omitted for
a single-window TSS run, or provided as a `TSSGraph` for windowed TSS.
The state owns the active thermodynamic state, local window estimators,
visit-control coupling state, and diagnostic histories.
Loggers are attached by [`TSSSimulation`](@ref), not by `TSSState`.
"""
mutable struct TSSState{T, ES, AS, G, W, E}
    state_space::ES # Shared ExtendedStateSpace among windows
    active_state::AS # Shared ActiveThermoState among windows
    graph::G # Graph ladder defining rung/window topology
    windows::Vector{W} # Window definitions
    estimators::Vector{E} # One local TSS estimator per window
    state_to_windows::Vector{Vector{Int}} # Global state index to containing windows
    active_window::Int # Current window index
    window_update_counts::Vector{Int} # Number of estimator updates per window
    iteration::Int # Total number of windowed updates
    stats::WindowedTSSStats{T} # Diagnostics
    coupling::Union{Nothing, WindowedTSSCoupling{T}} # Global visit-control state
end

function Base.show(io::IO, state::TSSState)
    visit_control = isnothing(state.coupling) ? "disabled" : "enabled"
    print(io, "TSSState with ", tss_count(n_states(state.state_space), "state"), ", ",
          tss_count(length(state.windows), "window"), ", active state ",
          state.active_state.active_idx, ", active window ", state.active_window,
          ", iteration ", state.iteration, ", visit control ", visit_control)
end

Base.show(io::IO, ::MIME"text/plain", state::TSSState) = show(io, state)

function build_tss_state_to_windows(windows::AbstractVector{TSSWindow}, K::Int)
    state_to_windows = [Int[] for _ in 1:K]
    for window in windows
        for state_index in window.state_indices
            push!(state_to_windows[state_index], window.index)
        end
    end
    return state_to_windows
end

function tss_window_overlap_adjacency(windows::AbstractVector{TSSWindow})
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

function check_tss_window_graph_connected(windows::AbstractVector{TSSWindow})
    adjacency = tss_window_overlap_adjacency(windows)
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

function validate_tss_state_window_coverage!(windows::AbstractVector{TSSWindow},
                                              state_to_windows::Vector{Vector{Int}},
                                              K::Int;
                                              required_coverage::Int = length(windows) == 1 ? 1 : 2)
    for state_index in 1:K
        n_cover = length(state_to_windows[state_index])
        if n_cover != required_coverage
            throw(ArgumentError("state $(state_index) must be covered by exactly " *
                                "$(required_coverage) window(s); got $(n_cover)."))
        end
    end

    check_tss_window_graph_connected(windows)
    return windows
end

function single_window_tss_graph(K::Int)
    K >= 1 || throw(ArgumentError("Number of states must be larger or equal than 1."))
    window = TSSWindow(1, Base.OneTo(K))
    state_to_windows = [[1] for _ in 1:K]
    rung_neighbors = [NTuple{3, Int}[] for _ in 1:K]
    rung_volumes = ones(Float64, K)
    return TSSGraph(
        K,
        [window],
        state_to_windows,
        rung_neighbors,
        rung_volumes,
    )
end

function tss_tuple(value, n_dims::Int, name::AbstractString, ::Type{T}) where {T}
    values = if value isa Tuple || value isa AbstractVector
        collect(value)
    else
        fill(value, n_dims)
    end
    if length(values) != n_dims
        throw(ArgumentError("$(name) must have length $(n_dims)."))
    end
    return T[values...]
end

tss_shape_tuple(value, name::AbstractString = "shape") =
    value isa Integer ? [Int(value)] : Int.(collect(value))

function tss_node_name(nodes, corner::Vector{Int})
    value = if nodes isa AbstractArray && ndims(nodes) == length(corner)
        nodes[corner...]
    else
        current = nodes
        for c in corner
            current = current[c]
        end
        current
    end
    return String(value)
end

function validate_tss_edge_nodes(nodes, n_dims::Int)
    seen = Dict{String, Int}()
    for corner in CartesianIndices(ntuple(_ -> 2, n_dims))
        name = tss_node_name(nodes, collect(Tuple(corner)))
        name == "_" && continue
        seen[name] = get(seen, name, 0) + 1
        if seen[name] != 1
            throw(ArgumentError("TSS edge node name $(name) is repeated within one edge."))
        end
    end
    return nodes
end

# add_tss_edge!(builder::TSSGraphBuilder, nodes, shape; window_size,
#               periodic=false, primary_window_tiling_only=false)
#
# Add one edge to a TSSGraphBuilder.
#
# `shape` gives the number of rungs along each edge dimension. `window_size`
# controls the local TSS window size, and `periodic` marks periodic dimensions.
# `nodes` names shared edge corners so separate edges can be joined in the final
# graph.
function add_tss_edge!(builder::TSSGraphBuilder,
                       nodes,
                       shape;
                       window_size,
                       periodic = false,
                       primary_window_tiling_only::Bool = false)
    shape = tss_shape_tuple(shape)
    n_dims = length(shape)
    n_dims > 0 || throw(ArgumentError("TSS edge shape must have at least one dimension."))
    all(>(0), shape) || throw(ArgumentError("TSS edge shape entries must be positive."))

    window_size = tss_tuple(window_size, n_dims, "window_size", Int)
    if !all(>(0), window_size)
        throw(ArgumentError("TSS window_size entries must be positive."))
    end
    periodic = tss_tuple(periodic, n_dims, "periodic", Bool)
    validate_tss_edge_nodes(nodes, n_dims)

    push!(
        builder.edges,
        TSSGraphEdge(nodes, shape, window_size, periodic, primary_window_tiling_only),
    )
    return builder
end

function anonymous_tss_nodes(n_dims::Int)
    return fill("_", ntuple(_ -> 2, n_dims))
end

"""
    tss_grid_graph(shape; window_size, periodic=false)

Construct a regular TSS grid graph.

This is a convenience wrapper around `TSSGraphBuilder`, `add_tss_edge!`, and
`build_tss_graph` for a single anonymous edge with regular windows.
"""
function tss_grid_graph(shape; window_size, periodic = false)
    shape = tss_shape_tuple(shape)
    builder = TSSGraphBuilder()
    add_tss_edge!(
        builder,
        anonymous_tss_nodes(length(shape)),
        shape;
        window_size = window_size,
        periodic = periodic,
    )
    return build_tss_graph(builder)
end

function tss_edge_offsets(edges::Vector{TSSGraphEdge})
    offsets = Int[]
    next_index = 1
    for edge in edges
        push!(offsets, next_index)
        next_index += prod(edge.shape)
    end
    return offsets
end

function tss_edge_rung_index(edge::TSSGraphEdge, offset::Int, coord::Vector{Int})
    return offset + LinearIndices(Tuple(edge.shape))[CartesianIndex(Tuple(coord))] - 1
end

function tss_edge_coordinates(edge::TSSGraphEdge)
    return [collect(Tuple(coord)) for coord in CartesianIndices(Tuple(edge.shape))]
end

function tss_rung_volume(edge::TSSGraphEdge, coord::Vector{Int})
    n_edge_dims = count(eachindex(coord)) do dim
        !edge.periodic[dim] && (coord[dim] == 1 || coord[dim] == edge.shape[dim])
    end
    return 0.5 ^ n_edge_dims
end

function tss_neighbor_coord(edge::TSSGraphEdge,
                             coord::Vector{Int},
                             dim::Int,
                             step::Int)
    n = edge.shape[dim]
    n == 1 && return coord

    neighbor = copy(coord)
    trial = coord[dim] + step
    if edge.periodic[dim]
        neighbor[dim] = mod1(trial, n)
    elseif 1 <= trial <= n
        neighbor[dim] = trial
    else
        neighbor[dim] = coord[dim]
    end
    return neighbor
end

function tss_rung_neighbors(edge::TSSGraphEdge, offset::Int, coord::Vector{Int})
    neighbors = NTuple{3, Int}[]
    self = tss_edge_rung_index(edge, offset, coord)
    for dim in eachindex(coord)
        reverse = tss_edge_rung_index(
            edge,
            offset,
            tss_neighbor_coord(edge, coord, dim, -1),
        )
        forward = tss_edge_rung_index(
            edge,
            offset,
            tss_neighbor_coord(edge, coord, dim, 1),
        )
        real_neighbor_count = (reverse != self) + (forward != self)
        push!(neighbors, (reverse, forward, real_neighbor_count))
    end
    return neighbors
end

function tss_dim_windows(n_states::Int,
                          window_size::Int,
                          periodic::Bool,
                          dim::Int,
                          generate_overlapping::Bool)
    if n_states < window_size
        throw(ArgumentError("TSS window_size[$(dim)] must not exceed shape[$(dim)]."))
    end
    if n_states % window_size != 0
        throw(ArgumentError("TSS shape[$(dim)] must be divisible by window_size[$(dim)]."))
    end

    regular = [TSSDimWindow(start, window_size, TSSPartialMembership[])
               for start in 1:window_size:n_states]
    !generate_overlapping && return regular, TSSDimWindow[]

    if !iseven(window_size)
        throw(ArgumentError("TSS window_size[$(dim)] must be even for overlapping windows."))
    end
    half_width = window_size ÷ 2
    overlap = TSSDimWindow[]

    if periodic
        for start in (1 + half_width):window_size:n_states
            push!(overlap, TSSDimWindow(start, window_size, TSSPartialMembership[]))
        end
    else
        for start in (1 + half_width):window_size:(n_states - window_size + 1)
            push!(overlap, TSSDimWindow(start, window_size, TSSPartialMembership[]))
        end
        push!(
            overlap,
            TSSDimWindow(1, half_width, [TSSPartialMembership(dim, 0)]),
        )
        push!(
            overlap,
            TSSDimWindow(
                n_states - half_width + 1,
                half_width,
                [TSSPartialMembership(dim, 1)],
            ),
        )
    end
    return regular, overlap
end

function tss_dim_state_values(dim_window::TSSDimWindow,
                               n_states::Int,
                               periodic::Bool)
    return [periodic ? mod1(dim_window.start + offset, n_states) :
                       dim_window.start + offset
            for offset in 0:(dim_window.size - 1)]
end

function tss_partial_signature(edge::TSSGraphEdge,
                                partials::Vector{TSSPartialMembership})
    isempty(partials) && return nothing
    fixed = Dict(partial.dimension => partial.side + 1 for partial in partials)
    names = String[]
    for corner in CartesianIndices(ntuple(_ -> 2, length(edge.shape)))
        corner_vec = collect(Tuple(corner))
        if all(get(fixed, dim, corner_vec[dim]) == corner_vec[dim]
               for dim in eachindex(corner_vec))
            name = tss_node_name(edge.nodes, corner_vec)
            name == "_" || push!(names, name)
        end
    end
    isempty(names) && return nothing
    return Tuple(sort!(unique(names)))
end

function tss_window_spec(edge::TSSGraphEdge,
                          offset::Int,
                          windows_by_dim)
    n_dims = length(edge.shape)
    values_by_dim = [
        tss_dim_state_values(windows_by_dim[dim], edge.shape[dim], edge.periodic[dim])
        for dim in 1:n_dims
    ]
    state_indices = Int[]
    for coord_tuple in Iterators.product(values_by_dim...)
        push!(state_indices, tss_edge_rung_index(edge, offset, collect(coord_tuple)))
    end

    partials = reduce(
        vcat,
        (window.partials for window in windows_by_dim);
        init = TSSPartialMembership[],
    )
    return TSSWindowSpec(
        [window.start for window in windows_by_dim],
        state_indices,
        tss_partial_signature(edge, partials),
    )
end

function tss_edge_window_specs(edge::TSSGraphEdge, offset::Int)
    regular_by_dim = Vector{TSSDimWindow}[]
    overlap_by_dim = Vector{TSSDimWindow}[]
    for dim in eachindex(edge.shape)
        regular, overlap = tss_dim_windows(
            edge.shape[dim],
            edge.window_size[dim],
            edge.periodic[dim],
            dim,
            !edge.primary_window_tiling_only,
        )
        push!(regular_by_dim, regular)
        push!(overlap_by_dim, overlap)
    end

    specs = TSSWindowSpec[]
    for window_tuple in Iterators.product(regular_by_dim...)
        push!(specs, tss_window_spec(edge, offset, window_tuple))
    end
    if !edge.primary_window_tiling_only
        for window_tuple in Iterators.product(overlap_by_dim...)
            push!(specs, tss_window_spec(edge, offset, window_tuple))
        end
    end
    return specs
end

function merge_tss_window_specs(specs::Vector{TSSWindowSpec})
    full_specs = TSSWindowSpec[]
    unmerged_partials = TSSWindowSpec[]
    partial_groups = Dict{Tuple{Vararg{String}}, Vector{TSSWindowSpec}}()

    for spec in specs
        if isnothing(spec.partial_signature)
            push!(full_specs, spec)
        else
            group = get!(partial_groups, spec.partial_signature, TSSWindowSpec[])
            push!(group, spec)
        end
    end

    merged_specs = copy(full_specs)
    for group in values(partial_groups)
        if length(group) == 1
            push!(unmerged_partials, only(group))
            continue
        end

        state_indices = Int[]
        for spec in group
            append!(state_indices, spec.state_indices)
        end
        state_indices = unique(state_indices)
        sort_key = copy(first(group).sort_key)
        for spec in Iterators.drop(group, 1)
            if spec.sort_key < sort_key
                sort_key = copy(spec.sort_key)
            end
        end
        push!(merged_specs, TSSWindowSpec(sort_key, state_indices, nothing))
    end
    append!(merged_specs, unmerged_partials)
    sort!(merged_specs; by = spec -> (spec.sort_key, length(spec.state_indices), spec.state_indices))
    return merged_specs
end

function evaluation_states_for_window(state_indices::Vector{Int},
                                       rung_neighbors::Vector{Vector{NTuple{3, Int}}})
    evaluation_states = copy(state_indices)
    for state_index in state_indices
        for (reverse, forward, ) in rung_neighbors[state_index]
            push!(evaluation_states, reverse)
            push!(evaluation_states, forward)
        end
    end
    return unique(evaluation_states)
end

# build_tss_graph(builder::TSSGraphBuilder)
#
# Build a TSSGraph from all edges stored in `builder`.
#
# The builder must contain at least one edge. The resulting graph validates
# window coverage, constructs rung neighbor lists, and merges compatible window
# specifications across named edge nodes.
function build_tss_graph(builder::TSSGraphBuilder)
    if isempty(builder.edges)
        throw(ArgumentError("TSSGraphBuilder must contain at least one edge."))
    end

    offsets = tss_edge_offsets(builder.edges)
    n_total = sum(prod(edge.shape) for edge in builder.edges)
    rung_neighbors = [NTuple{3, Int}[] for _ in 1:n_total]
    rung_volumes = zeros(Float64, n_total)

    specs = TSSWindowSpec[]
    for (edge_i, edge) in enumerate(builder.edges)
        offset = offsets[edge_i]
        for coord in tss_edge_coordinates(edge)
            state_index = tss_edge_rung_index(edge, offset, coord)
            rung_neighbors[state_index] = tss_rung_neighbors(edge, offset, coord)
            rung_volumes[state_index] = tss_rung_volume(edge, coord)
        end
        append!(specs, tss_edge_window_specs(edge, offset))
    end

    merged_specs = merge_tss_window_specs(specs)
    windows = [
        TSSWindow(
            window_i,
            spec.state_indices;
            evaluation_state_indices = evaluation_states_for_window(
                spec.state_indices,
                rung_neighbors,
            ),
            check_contiguous = false,
        )
        for (window_i, spec) in enumerate(merged_specs)
    ]
    state_to_windows = build_tss_state_to_windows(windows, n_total)
    validate_tss_state_window_coverage!(
        windows,
        state_to_windows,
        n_total;
        required_coverage = 2,
    )

    return TSSGraph(
        n_total,
        windows,
        state_to_windows,
        rung_neighbors,
        rung_volumes,
    )
end

function tss_swap_window(graph::TSSGraph, active_window::Integer, state_index::Integer)
    state_index = Int(state_index)
    if !(1 <= state_index <= graph.n_states)
        throw(ArgumentError("state $(state_index) is out of TSS graph bounds."))
    end
    active_window = Int(active_window)
    windows = graph.state_to_windows[state_index]
    if length(windows) != 2
        throw(ArgumentError("state $(state_index) is not covered by exactly two windows."))
    end
    if active_window == windows[1]
        return windows[2]
    elseif active_window == windows[2]
        return windows[1]
    end
    throw(ArgumentError("active window $(active_window) does not contain state $(state_index)."))
end

function windowed_global_vector(values, name::AbstractString, K::Int)
    isnothing(values) && return nothing
    values = collect(values)
    if length(values) != K
        throw(ArgumentError("$(name) must have length $(K) for TSSState."))
    end
    return values
end

function windowed_tss_adaptive_gamma_mode(adaptive_gamma)
    isnothing(adaptive_gamma) && return nothing
    if adaptive_gamma isa Symbol
        adaptive_gamma == :covdet && return :covdet
        throw(ArgumentError("unknown TSS adaptive_gamma mode $(adaptive_gamma); " *
                            "the only supported mode is :covdet."))
    end
    throw(ArgumentError("TSSState adaptive_gamma accepts only nothing or :covdet."))
end

function windowed_tss_adaptive_gamma_for_window(mode,
                                                 graph::TSSGraph,
                                                 window::TSSWindow,
                                                 ::Type{FT}) where {FT}
    isnothing(mode) && return nothing
    if mode != :covdet
        throw(ArgumentError("unknown TSS adaptive_gamma mode $(mode)."))
    end

    first_state = first(window.state_indices)
    dimension = length(graph.rung_neighbors[first_state])
    for state_index in window.state_indices
        if length(graph.rung_neighbors[state_index]) != dimension
            throw(ArgumentError("TSS CovDet adaptive gamma requires all rungs in a " *
                                "window to have the same lambda dimension."))
        end
    end
    volumes = FT.(graph.rung_volumes[window.state_indices])
    if !(all(isfinite, volumes) && all(>=(zero(FT)), volumes) && sum(volumes) > zero(FT))
        throw(ArgumentError("TSS CovDet adaptive gamma requires finite positive rung volumes."))
    end
    return TSSCovDetAdaptiveGamma(
        FT(TSS_COVDET_GAMMA_EPSILON),
        graph.rung_neighbors,
        volumes,
        dimension,
    )
end

function TSSState(thermo_states::AbstractVector{<:ThermoState};
                          graph = nothing,
                          first_state::Int = 1,
                          first_window = nothing,
                          gamma = nothing,
                          initial_f = nothing,
                          ETA::Real = 2.0,
                          dens_reg::Real = 1e-6,
                          reuse_neighbors::Bool = true,
                          history_forgetting = nothing,
                          adaptive_gamma = nothing,
                          global_visit_control::Bool = true,
                          visit_control_tolerance::Real = 1e-8,
                          visit_control_max_iterations::Integer = 1_000,
                          visit_control_damping::Real = 1.0,
                          pi_regularization::Real = 1e-3)

    state_space = ExtendedStateSpace(thermo_states; reuse_neighbors = reuse_neighbors)
    K = n_states(state_space)
    explicit_graph = !isnothing(graph)
    if isnothing(graph)
        graph = single_window_tss_graph(K)
    elseif !(graph isa TSSGraph)
        throw(ArgumentError("graph must be a TSSGraph; construct one with " *
                            "tss_grid_graph or build_tss_graph."))
    end
    if graph.n_states != K
        throw(ArgumentError("TSS graph contains $(graph.n_states) states but " *
                            "TSSState was given $(K) thermodynamic states."))
    end
    if !(1 <= first_state <= K)
        throw(ArgumentError("first_state ($(first_state)) out of range 1:$(K)."))
    end

    active_state = ActiveThermoState(state_space, first_state)
    FT = typeof(ustrip(active_state.active_sys.total_mass))

    normalized_windows = graph.windows
    state_to_windows = graph.state_to_windows
    validate_tss_state_window_coverage!(normalized_windows, state_to_windows, K)

    first_windows = state_to_windows[first_state]
    active_window = if isnothing(first_window)
        first(first_windows)
    else
        if !(first_window isa Integer)
            throw(ArgumentError("first_window must be an integer window index."))
        end
        first_window = Int(first_window)
        if !(first_window in first_windows)
            throw(ArgumentError("first_window must contain first_state."))
        end
        first_window
    end

    global_gamma = windowed_global_vector(gamma, "gamma", K)
    global_initial_f = windowed_global_vector(initial_f, "initial_f", K)
    adaptive_mode = windowed_tss_adaptive_gamma_mode(adaptive_gamma)
    if adaptive_mode == :covdet && !explicit_graph
        throw(ArgumentError("adaptive_gamma=:covdet requires an explicit TSS graph " *
                            "with lambda-neighbor topology."))
    end

    estimators = [
        make_tss_local_estimator(
            state_space,
            active_state;
            state_indices = window.state_indices,
            evaluation_state_indices = window.evaluation_state_indices,
            gamma = isnothing(global_gamma) ? nothing : global_gamma[window.state_indices],
            initial_f = isnothing(global_initial_f) ? nothing : global_initial_f[window.state_indices],
            ETA = ETA,
            dens_reg = dens_reg,
            history_forgetting = history_forgetting,
            adaptive_gamma = windowed_tss_adaptive_gamma_for_window(
                adaptive_mode,
                graph,
                window,
                FT,
            ),
            require_active_state = false,
        )
        for window in normalized_windows
    ]

    stats = WindowedTSSStats(FT)

    state = TSSState{
        FT,
        typeof(state_space),
        typeof(active_state),
        typeof(graph),
        TSSWindow,
        eltype(estimators),
    }(
        state_space,
        active_state,
        graph,
        normalized_windows,
        estimators,
        state_to_windows,
        active_window,
        zeros(Int, length(normalized_windows)),
        0,
        stats,
        nothing,
    )

    update_windowed_tss_adaptive_gamma!(state)

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

active_tss_estimator(state::TSSState) = state.estimators[state.active_window]

window_contains_state(window::TSSWindow, global_state::Integer) =
    Int(global_state) in window.state_indices

function windows_for_state(state::TSSState, global_state::Integer)
    global_state = Int(global_state)
    if !(1 <= global_state <= n_states(state.state_space))
        throw(ArgumentError("global_state $(global_state) out of bounds."))
    end
    return state.state_to_windows[global_state]
end

function other_window_for_state(state::TSSState,
                                active_window::Integer,
                                global_state::Integer)
    windows = windows_for_state(state, global_state)
    if length(windows) == 1
        only_window = only(windows)
        if Int(active_window) != only_window
            throw(ArgumentError("active window $(active_window) does not contain state $(global_state)."))
        end
        return only_window
    end
    return tss_swap_window(state.graph, active_window, global_state)
end

function other_window_for_state(state::TSSState, global_state::Integer)
    return other_window_for_state(state, state.active_window, global_state)
end

function switch_active_window!(state::TSSState; current_state::Integer = state.active_state.active_idx)
    state.active_window = other_window_for_state(state, current_state)
    return state
end

function check_windowed_tss_invariant(state::TSSState)
    active_window = state.windows[state.active_window]
    if !window_contains_state(active_window, state.active_state.active_idx)
        throw(ArgumentError("active window $(state.active_window) does not contain active state " *
                            "$(state.active_state.active_idx)."))
    end
    if length(state.window_update_counts) != length(state.windows)
        throw(ArgumentError("window update counts do not match the number of TSS windows."))
    end
    if state.graph.n_states != n_states(state.state_space)
        throw(ArgumentError("TSS graph state count does not match the state space."))
    end
    if !isnothing(state.coupling)
        coupling = state.coupling
        K = n_states(state.state_space)
        if length(coupling.visit_control_f) != K
            throw(ArgumentError("visit-control free energies do not match the number of states."))
        end
        if length(coupling.window_probs) != length(state.windows)
            throw(ArgumentError("visit-control window probabilities do not match the number of windows."))
        end
        if length(coupling.candidate_densities) != length(state.windows)
            throw(ArgumentError("visit-control candidate densities do not match the number of windows."))
        end
    end
    return state
end
