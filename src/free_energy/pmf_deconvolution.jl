export
    PMFDeconvolution,
    pmf

struct PMFGrid{N, T, E, C, W, V}
    edges::E
    centers::C
    widths::W
    shape::NTuple{N, Int}
    volumes::V
end

function PMFGrid(grid; T::Type = Float64)
    edges = online_pmf_edges(grid, T)
    centers = online_pmf_centers(edges)
    widths = online_pmf_widths(edges)
    shape = ntuple(d -> length(edges[d]) - 1, length(edges))
    volumes = online_pmf_bin_volumes(widths, shape)
    N = length(shape)
    return PMFGrid{N, T, typeof(edges), typeof(centers), typeof(widths), typeof(volumes)}(
        edges,
        centers,
        widths,
        shape,
        volumes,
    )
end

pmf_centers(grid::PMFGrid{1}) = grid.centers[1]
pmf_centers(grid::PMFGrid) = grid.centers
pmf_widths(grid::PMFGrid{1}) = grid.widths[1]
pmf_widths(grid::PMFGrid) = grid.widths
pmf_edges(grid::PMFGrid{1}) = grid.edges[1]
pmf_edges(grid::PMFGrid) = grid.edges

function pmf_bin_center(grid::PMFGrid{N}, idx::CartesianIndex{N}) where N
    return ntuple(N) do d
        grid.centers[d][idx[d]]
    end
end

function pmf_reference_index(F::AbstractArray, zero::Symbol)
    if !(zero in (:min, :last, :none))
        throw(ArgumentError("zero must be one of :min, :last, or :none."))
    end
    zero == :none && return nothing

    finite_mask = isfinite.(F)
    if !any(finite_mask)
        throw(ArgumentError("cannot gauge a PMF without finite bins."))
    end
    if zero == :min
        finite_indices = findall(finite_mask)
        return finite_indices[argmin(F[finite_indices])]
    else
        return findlast(finite_mask)
    end
end

function pmf_probability_from_raw_free_energy(grid::PMFGrid{N, T},
                                               F::AbstractArray{T, N}) where {N, T}
    probability = zeros(T, size(F))
    for idx in CartesianIndices(F)
        if isfinite(F[idx])
            probability[idx] = exp(-F[idx]) * grid.volumes[idx]
        end
    end
    total = sum(probability)
    if total <= zero(T)
        throw(ArgumentError("PMF probabilities cannot be normalized; all bins have zero weight."))
    end
    probability ./= total
    return probability
end

function pmf_raw_free_energy_from_probability(grid::PMFGrid{N, T},
                                               probability::AbstractArray) where {N, T}
    if size(probability) != grid.shape
        throw(DimensionMismatch("PMF probability shape $(size(probability)) does not match " *
                                "grid shape $(grid.shape)."))
    end
    F = fill(T(Inf), grid.shape)
    for idx in CartesianIndices(F)
        p = T(probability[idx])
        if p < zero(T)
            throw(ArgumentError("PMF probabilities must be non-negative."))
        end
        if p > zero(T)
            F[idx] = -log(p / grid.volumes[idx])
        end
    end
    return F
end

function pmf_result_from_raw_free_energy(grid::PMFGrid{N, T},
                                          F_raw::AbstractArray{T, N};
                                          probability = nothing,
                                          zero::Symbol = :min,
                                          kBT = nothing,
                                          sigma_F = nothing,
                                          reference_index = nothing) where {N, T}
    F = copy(F_raw)
    ref_idx = isnothing(reference_index) ? pmf_reference_index(F, zero) : reference_index
    if !isnothing(ref_idx)
        if !isfinite(F[ref_idx])
            throw(ArgumentError("PMF reference bin $(ref_idx) is not finite."))
        end
        offset = F[ref_idx]
        finite_mask = isfinite.(F)
        F[finite_mask] .-= offset
    end

    p = isnothing(probability) ? pmf_probability_from_raw_free_energy(grid, F_raw) :
        Array{T, N}(probability)
    F_energy = isnothing(kBT) ? nothing : F .* kBT
    sigma_F_energy = (isnothing(kBT) || isnothing(sigma_F)) ? nothing : sigma_F .* kBT

    return PMF(
        pmf_centers(grid),
        pmf_widths(grid),
        pmf_edges(grid),
        F,
        F_energy,
        sigma_F,
        sigma_F_energy,
        p,
        nothing,
    )
end

function pmf_result_from_probability(grid::PMFGrid{N, T},
                                      probability::AbstractArray;
                                      kwargs...) where {N, T}
    prob = Array{T, N}(probability)
    total = sum(prob)
    if total <= zero(T)
        throw(ArgumentError("PMF probabilities cannot be normalized; all bins have zero weight."))
    end
    prob ./= total
    F_raw = pmf_raw_free_energy_from_probability(grid, prob)
    return pmf_result_from_raw_free_energy(grid, F_raw; probability = prob, kwargs...)
end

function pmf_build_log_coupling_matrix(state_space,
                                        grid::PMFGrid{N, T};
                                        coupling = nothing) where {N, T}
    n_thermo_states = n_states(state_space)
    matrix = Matrix{T}(undef, prod(grid.shape), n_thermo_states)
    cart_indices = CartesianIndices(grid.shape)
    linear_indices = LinearIndices(grid.shape)

    if !isnothing(coupling)
        for idx in cart_indices
            xi = pmf_bin_center(grid, idx)
            bin_i = linear_indices[idx]
            for state_i in 1:n_thermo_states
                dimless_bias = T(coupling(xi, state_i))
                if !isfinite(dimless_bias)
                    throw(ArgumentError("PMF coupling returned non-finite value " *
                                        "$(dimless_bias) for bin $(idx), state $(state_i)."))
                end
                matrix[bin_i, state_i] = -dimless_bias
            end
        end
        return matrix
    end

    for state_i in 1:n_thermo_states
        ham = state_space.partition.λ_hamiltonians[state_i]
        bias_indices = findall(inter -> inter isa BiasPotential, ham.general_inters)
        if length(bias_indices) != N
            throw(ArgumentError("automatic PMF deconvolution found $(length(bias_indices)) " *
                                "BiasPotential interactions in state $(state_i), but the " *
                                "PMF grid is $(N)D. Provide an explicit coupling function."))
        end
        beta_i = state_space.betas[state_i]
        for idx in cart_indices
            xi = pmf_bin_center(grid, idx)
            physical_bias_energy = zero(T) * state_space.systems[state_i].energy_units
            for (dim_i, bias_idx) in enumerate(bias_indices)
                bias_inter = ham.general_inters[bias_idx]
                physical_bias_energy += potential_energy(bias_inter.bias_type, xi[dim_i])
            end
            matrix[linear_indices[idx], state_i] =
                -beta_i * safe_ustrip(T, physical_bias_energy)
        end
    end
    return matrix
end

"""
    PMFDeconvolution(state; grid, coupling=nothing, cv=nothing, ...)

Prepare a sampled PMF deconvolution object for an enhanced-sampling state.

The same user-facing object is used by AWH, TSS, and future methods. The
backend is selected from the type of `state`. `grid` follows the same format as
[`OnlinePMFAccumulator`](@ref). `coupling(xi, state_i)` should return the
dimensionless bias energy at PMF coordinate `xi` in thermodynamic state
`state_i`; when omitted, Molly attempts to infer matching [`BiasPotential`](@ref)
interactions from the state Hamiltonians.

The estimator follows the sampled deconvolution form used by AWH: each sampled
PMF coordinate is accumulated in a self-normalized weighted histogram using the
inverse time-dependent effective bias at the observed PMF bin. Direct
deconvolution of final state free energies is not supported. Call [`pmf`](@ref)
on the returned object to extract the current PMF estimate.
"""
struct PMFDeconvolution{B}
    backend::B
end

struct PMFDeconvolutionSample{N, T, W}
    value::NTuple{N, T}
    log_bin_weights::W
    log_reweight::T
end

mutable struct SampledPMFDeconvolutionAccumulator{N, T, G}
    grid::G
    log_numerator_sums::Array{T, N}
    log_numerator_sq_sums::Array{T, N}
    max_log_numerator_weights::Array{T, N}
    counts::Array{Int, N}
    total_samples::Int
    accepted_samples::Int
    out_of_grid_samples::Int
end

function SampledPMFDeconvolutionAccumulator(grid::PMFGrid{N}) where N
    T = Float64
    shape = grid.shape
    grid64 = grid isa PMFGrid{N, T} ? grid : PMFGrid(grid.edges; T = T)
    return SampledPMFDeconvolutionAccumulator(
        grid64,
        fill(-T(Inf), shape),
        fill(-T(Inf), shape),
        fill(-T(Inf), shape),
        zeros(Int, shape),
        0,
        0,
        0,
    )
end

function pmf_deconvolution_check_log_weight(lw::T, label::AbstractString) where T
    if isnan(lw) || lw == T(Inf)
        throw(ArgumentError("sampled PMF deconvolution $(label) is non-finite ($(lw))."))
    end
    return lw
end

function accumulate_pmf_deconvolution!(acc::SampledPMFDeconvolutionAccumulator{N, T},
                                        value,
                                        log_bin_weights;
                                        log_reweight = zero(T)) where {N, T}
    if length(log_bin_weights) != prod(acc.grid.shape)
        throw(DimensionMismatch("PMF deconvolution sample has $(length(log_bin_weights)) " *
                                "bin weights, expected $(prod(acc.grid.shape))."))
    end
    lr = pmf_deconvolution_check_log_weight(T(log_reweight), "reweighting factor")

    values = online_pmf_tuple(value, T)
    if length(values) != N
        throw(DimensionMismatch("PMF coordinate has $(length(values)) dimensions, expected $(N)."))
    end
    if !all(isfinite, values)
        throw(ArgumentError("PMF coordinate contains non-finite values."))
    end

    acc.total_samples += 1
    bin_index = online_pmf_bin_index(acc.grid.edges, values)
    if any(==(0), bin_index)
        acc.out_of_grid_samples += 1
        return acc
    end

    linear_indices = LinearIndices(acc.grid.shape)
    current_idx = CartesianIndex(bin_index)
    current_linear = linear_indices[current_idx]

    ln = pmf_deconvolution_check_log_weight(T(log_bin_weights[current_linear]) + lr,
                                            "observed-bin weight")
    if !isfinite(ln)
        throw(ArgumentError("sampled PMF deconvolution produced zero support for the " *
                            "observed bin $(current_idx)."))
    end
    acc.log_numerator_sums[current_idx] =
        online_pmf_logaddexp(acc.log_numerator_sums[current_idx], ln)
    acc.log_numerator_sq_sums[current_idx] =
        online_pmf_logaddexp(acc.log_numerator_sq_sums[current_idx], T(2) * ln)
    acc.max_log_numerator_weights[current_idx] =
        max(acc.max_log_numerator_weights[current_idx], ln)
    acc.counts[current_idx] += 1
    acc.accepted_samples += 1
    return acc
end

function accumulate_pmf_deconvolution!(acc::SampledPMFDeconvolutionAccumulator,
                                        sample::PMFDeconvolutionSample)
    return accumulate_pmf_deconvolution!(
        acc,
        sample.value,
        sample.log_bin_weights;
        log_reweight = sample.log_reweight,
    )
end

function sampled_pmf_probability(acc::SampledPMFDeconvolutionAccumulator{N, T}) where {N, T}
    log_total = online_pmf_total_log_weight(acc.log_numerator_sums)
    if !isfinite(log_total)
        throw(ArgumentError("cannot compute PMF before at least one in-grid weighted sample."))
    end

    probability = zeros(T, acc.grid.shape)
    for idx in CartesianIndices(acc.log_numerator_sums)
        lw = acc.log_numerator_sums[idx]
        if isfinite(lw)
            probability[idx] = exp(lw - log_total)
        end
    end
    return probability
end

function pmf_result_from_sampled_deconvolution(acc::SampledPMFDeconvolutionAccumulator;
                                               zero::Symbol = :min,
                                               kBT = nothing)
    probability = sampled_pmf_probability(acc)
    return pmf_result_from_probability(acc.grid, probability; zero = zero, kBT = kBT)
end

function pmf_log_bin_weights!(dest::AbstractVector{T},
                               log_coupling_matrix::AbstractMatrix,
                               log_state_weights::AbstractVector;
                               state_indices = eachindex(log_state_weights),
                               log_weight_factor = zero(T)) where T
    if length(dest) != size(log_coupling_matrix, 1)
        throw(DimensionMismatch("PMF deconvolution scratch has $(length(dest)) bins, " *
                                "expected $(size(log_coupling_matrix, 1))."))
    end
    if length(log_state_weights) != length(state_indices)
        throw(DimensionMismatch("PMF deconvolution state weights and state indices mismatch."))
    end
    lwf = pmf_deconvolution_check_log_weight(T(log_weight_factor), "sample scale")
    for bin_i in eachindex(dest)
        log_den = -T(Inf)
        for (local_i, state_i) in enumerate(state_indices)
            log_coupling = T(log_coupling_matrix[bin_i, state_i])
            pmf_deconvolution_check_log_weight(log_coupling, "coupling")
            isfinite(log_coupling) || continue
            log_state_weight =
                pmf_deconvolution_check_log_weight(T(log_state_weights[local_i]),
                                                    "state weight")
            isfinite(log_state_weight) || continue
            log_den = online_pmf_logaddexp(
                log_den,
                log_state_weight + log_coupling,
            )
        end
        dest[bin_i] = isfinite(log_den) ? lwf - log_den : -T(Inf)
    end
    return dest
end

function pmf_positive_log(value, label::AbstractString, ::Type{T}) where T
    v = T(value)
    if !(isfinite(v) && v > zero(T))
        throw(ArgumentError("$(label) must be finite and positive, got $(value)."))
    end
     return log(v)
end

pmf(deconv::PMFDeconvolution; kwargs...) = pmf(deconv.backend; kwargs...)
