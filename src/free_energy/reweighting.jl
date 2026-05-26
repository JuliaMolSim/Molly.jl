@inline function _online_pmf_logaddexp(a::T, b::T) where T
    if a == -T(Inf)
        return b
    elseif b == -T(Inf)
        return a
    end
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

function _online_pmf_tuple(values::Tuple, ::Type{T}) where T
    return tuple((_safe_ustrip(T, value) for value in values)...)
end

function _online_pmf_tuple(value::Number, ::Type{T}) where T
    return (_safe_ustrip(T, value),)
end

function _online_pmf_tuple(value, ::Type{T}) where T
    values = from_device(value)
    values isa AbstractVector ||
        throw(ArgumentError("PMF coordinate must be a scalar, tuple, or vector."))
    return tuple((_safe_ustrip(T, v) for v in values)...)
end

function _online_pmf_edges_from_bounds(mins, maxs, bins, ::Type{T}) where T
    length(mins) == length(maxs) == length(bins) ||
        throw(ArgumentError("PMF grid minima, maxima, and bin counts must have matching lengths."))
    return ntuple(length(bins)) do d
        min_d = _safe_ustrip(T, mins[d])
        max_d = _safe_ustrip(T, maxs[d])
        n_d = Int(bins[d])
        isfinite(min_d) && isfinite(max_d) && max_d > min_d ||
            throw(ArgumentError("PMF grid dimension $(d) must have finite max > min."))
        n_d > 0 ||
            throw(ArgumentError("PMF grid dimension $(d) must have a positive bin count."))
        step = (max_d - min_d) / T(n_d)
        [min_d + T(i) * step for i in 0:n_d]
    end
end

function _online_pmf_edges(grid, ::Type{T}) where T
    if grid isa Tuple && length(grid) == 3 && grid[3] isa Integer
        return _online_pmf_edges_from_bounds((grid[1],), (grid[2],), (grid[3],), T)
    elseif grid isa Tuple && length(grid) == 3 && !(grid[1] isa AbstractVector) &&
           !(grid[1] isa Tuple)
        return _online_pmf_edges_from_bounds((grid[1],), (grid[2],), (grid[3],), T)
    elseif grid isa Tuple && length(grid) == 3 &&
           (grid[3] isa Tuple || (grid[3] isa AbstractVector && all(x -> x isa Integer, grid[3]))) &&
           all(x -> x isa Tuple || x isa AbstractVector, grid)
        return _online_pmf_edges_from_bounds(grid[1], grid[2], grid[3], T)
    elseif grid isa Tuple && all(edge -> edge isa AbstractVector, grid)
        return ntuple(length(grid)) do d
            edges_d = T.(_safe_ustrip.(Ref(T), grid[d]))
            length(edges_d) >= 2 ||
                throw(ArgumentError("PMF grid dimension $(d) must have at least two edges."))
            all(isfinite, edges_d) ||
                throw(ArgumentError("PMF grid dimension $(d) contains non-finite edges."))
            all(>(zero(T)), diff(edges_d)) ||
                throw(ArgumentError("PMF grid dimension $(d) edges must be strictly increasing."))
            collect(edges_d)
        end
    else
        throw(ArgumentError("PMF grid must be (min, max, bins), " *
                            "((mins...), (maxs...), (bins...)), or a tuple of edge vectors."))
    end
end

function _online_pmf_centers(edges::NTuple{N, <:AbstractVector{T}}) where {N, T}
    return ntuple(N) do d
        [(edges[d][i] + edges[d][i + 1]) / T(2) for i in 1:(length(edges[d]) - 1)]
    end
end

function _online_pmf_widths(edges::NTuple{N, <:AbstractVector{T}}) where {N, T}
    return ntuple(N) do d
        [edges[d][i + 1] - edges[d][i] for i in 1:(length(edges[d]) - 1)]
    end
end

mutable struct OnlinePMFAccumulator{N, T, E, C, W}
    edges::E
    centers::C
    widths::W
    log_weight_sums::Array{T, N}
    log_weight_sq_sums::Array{T, N}
    max_log_weights::Array{T, N}
    counts::Array{Int, N}
    total_samples::Int
    accepted_samples::Int
    out_of_grid_samples::Int
end

"""
    OnlinePMFAccumulator(grid; T=Float64)

Online weighted histogram accumulator for target-state PMFs.

`grid` can be `(min, max, bins)` for 1D,
`((mins...), (maxs...), (bins...))` for N dimensions, or a tuple of explicit
edge vectors. Samples are added with `accumulate!(acc, value, log_weight)`, and
`pmf(acc)` returns a `PMF` result.
"""
function OnlinePMFAccumulator(grid; T::Type = Float64)
    edges = _online_pmf_edges(grid, T)
    centers = _online_pmf_centers(edges)
    widths = _online_pmf_widths(edges)
    shape = ntuple(d -> length(edges[d]) - 1, length(edges))
    return OnlinePMFAccumulator(
        edges,
        centers,
        widths,
        fill(-T(Inf), shape),
        fill(-T(Inf), shape),
        fill(-T(Inf), shape),
        zeros(Int, shape),
        0,
        0,
        0,
    )
end

function _online_pmf_bin_index(edges::NTuple{N, <:AbstractVector{T}},
                               values::NTuple{N, T}) where {N, T}
    return ntuple(N) do d
        edges_d = edges[d]
        idx = searchsortedlast(edges_d, values[d])
        if idx == length(edges_d) && values[d] == edges_d[end]
            idx -= 1
        end
        if idx < 1 || idx >= length(edges_d)
            return 0
        end
        idx
    end
end

function accumulate!(acc::OnlinePMFAccumulator{N, T}, value, log_weight) where {N, T}
    lw = T(log_weight)
    isfinite(lw) ||
        throw(ArgumentError("online PMF log weight is non-finite ($(log_weight))."))

    values = _online_pmf_tuple(value, T)
    length(values) == N ||
        throw(DimensionMismatch("PMF coordinate has $(length(values)) dimensions, expected $(N)."))
    all(isfinite, values) ||
        throw(ArgumentError("PMF coordinate contains non-finite values."))

    acc.total_samples += 1
    idx = _online_pmf_bin_index(acc.edges, values)
    if any(==(0), idx)
        acc.out_of_grid_samples += 1
        return acc
    end

    cart_idx = CartesianIndex(idx)
    acc.log_weight_sums[cart_idx] =
        _online_pmf_logaddexp(acc.log_weight_sums[cart_idx], lw)
    acc.log_weight_sq_sums[cart_idx] =
        _online_pmf_logaddexp(acc.log_weight_sq_sums[cart_idx], T(2) * lw)
    acc.max_log_weights[cart_idx] = max(acc.max_log_weights[cart_idx], lw)
    acc.counts[cart_idx] += 1
    acc.accepted_samples += 1
    return acc
end

function _online_pmf_total_log_weight(log_weight_sums::AbstractArray{T}) where T
    total = -T(Inf)
    for lw in log_weight_sums
        total = _online_pmf_logaddexp(total, lw)
    end
    return total
end

function _online_pmf_bin_volumes(widths::NTuple{N, <:AbstractVector{T}},
                                 shape::NTuple{N, Int}) where {N, T}
    volumes = Array{T, N}(undef, shape)
    for idx in CartesianIndices(volumes)
        v = one(T)
        for d in 1:N
            v *= widths[d][idx[d]]
        end
        volumes[idx] = v
    end
    return volumes
end

"""
    effective_samples(acc::OnlinePMFAccumulator)

Return per-bin effective sample sizes from the accumulated log weights.
"""
function effective_samples(acc::OnlinePMFAccumulator{N, T}) where {N, T}
    ess = zeros(T, size(acc.log_weight_sums))
    for idx in CartesianIndices(ess)
        lw = acc.log_weight_sums[idx]
        lw2 = acc.log_weight_sq_sums[idx]
        if isfinite(lw) && isfinite(lw2)
            ess[idx] = exp(T(2) * lw - lw2)
        end
    end
    return ess
end

"""
    total_effective_samples(acc::OnlinePMFAccumulator)

Return the effective sample size of the complete accumulated PMF histogram.
"""
function total_effective_samples(acc::OnlinePMFAccumulator{N, T}) where {N, T}
    lw = _online_pmf_total_log_weight(acc.log_weight_sums)
    lw2 = _online_pmf_total_log_weight(acc.log_weight_sq_sums)
    if isfinite(lw) && isfinite(lw2)
        return exp(T(2) * lw - lw2)
    end
    return zero(T)
end

"""
    max_weight_fraction(acc::OnlinePMFAccumulator)

Return the largest normalized single-sample weight per occupied PMF bin.
Values near one indicate that a bin is dominated by one configuration, which is
a useful support diagnostic for offline reweighting.
"""
function max_weight_fraction(acc::OnlinePMFAccumulator{N, T}) where {N, T}
    fraction = zeros(T, size(acc.log_weight_sums))
    for idx in CartesianIndices(fraction)
        lw = acc.log_weight_sums[idx]
        max_lw = acc.max_log_weights[idx]
        if isfinite(lw) && isfinite(max_lw)
            fraction[idx] = exp(max_lw - lw)
        end
    end
    return fraction
end

"""
    pmf(acc::OnlinePMFAccumulator; zero=:min, kBT=nothing)

Convert an online PMF accumulator to a `PMF` result. PMF values are reported in
reduced units, shifted according to `zero`; when `kBT` is provided, `F_energy`
is also populated.
"""
function pmf(acc::OnlinePMFAccumulator{N, T}; zero::Symbol = :min, kBT = nothing) where {N, T}
    zero in (:min, :last, :none) ||
        throw(ArgumentError("zero must be one of :min, :last, or :none."))

    log_total = _online_pmf_total_log_weight(acc.log_weight_sums)
    isfinite(log_total) ||
        throw(ArgumentError("cannot compute PMF before at least one in-grid weighted sample."))

    volumes = _online_pmf_bin_volumes(acc.widths, size(acc.log_weight_sums))
    probability = zeros(T, size(acc.log_weight_sums))
    F = fill(T(Inf), size(acc.log_weight_sums))
    for idx in CartesianIndices(acc.log_weight_sums)
        lw = acc.log_weight_sums[idx]
        if isfinite(lw)
            log_probability = lw - log_total
            log_density = log_probability - log(volumes[idx])
            F[idx] = -log_density

            # This may underflow; that is acceptable for p,
            # but should not control whether F is finite.
            probability[idx] = exp(log_probability)
        end
    end

    finite_mask = isfinite.(F)
    if any(finite_mask)
        offset = if zero == :min
            minimum(F[finite_mask])
        elseif zero == :last
            last_finite = findlast(finite_mask)
            isnothing(last_finite) ? Base.zero(T) : F[last_finite]
        else
            Base.zero(T)
        end
        F[finite_mask] .-= offset
    end

    centers = N == 1 ? acc.centers[1] : acc.centers
    widths = N == 1 ? acc.widths[1] : acc.widths
    edges = N == 1 ? acc.edges[1] : acc.edges
    F_energy = isnothing(kBT) ? nothing : F .* kBT

    return PMF(centers, widths, edges, F, F_energy, nothing, nothing, probability, nothing)
end

function _target_coords_for_system(coords, sys::System)
    target_array_type = array_type(sys)
    array_type(coords) == target_array_type && return coords
    return to_device(from_device(coords), target_array_type)
end

mutable struct ReducedPotentialWorkspace{TS, B}
    target_state::TS
    buffers::B
end

function ReducedPotentialWorkspace(target_state::ThermoState)
    target_copy = deepcopy(target_state)
    buffers = is_on_gpu(target_copy.system) ? init_buffers!(target_copy.system, 1, true) : nothing
    return ReducedPotentialWorkspace(target_copy, buffers)
end

function reduced_potential(workspace::ReducedPotentialWorkspace,
                           coords,
                           boundary;
                           n_threads::Integer = 1)
    target = workspace.target_state
    sys = target.system
    sys.coords .= _target_coords_for_system(coords, sys)
    sys.boundary = boundary
    energy = potential_energy(
        sys,
        find_neighbors(sys),
        workspace.buffers;
        n_threads = n_threads,
    )
    return reduced_potential(target, energy, boundary)
end
