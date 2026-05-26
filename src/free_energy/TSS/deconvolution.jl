mutable struct _TSSPMFDeconvolutionBackend{N, T, S, F_CV, G, C, A}
    state::S
    cv_function::F_CV
    grid::G
    log_coupling_matrix::C
    accumulator::A
end

function _tss_auto_pmf_cv(state::TSSState, grid::_PMFGrid{N}) where N
    first_ham = state.state_space.partition.λ_hamiltonians[1]
    bias_indices = findall(inter -> inter isa BiasPotential, first_ham.general_inters)
    length(bias_indices) == N ||
        throw(ArgumentError("automatic PMF deconvolution found $(length(bias_indices)) " *
                            "BiasPotential interactions in the first TSS state, but the " *
                            "PMF grid is $(N)D. Provide explicit cv and coupling functions."))
    cv_types = [first_ham.general_inters[i].cv_type for i in bias_indices]
    return active_state -> begin
        sys = active_state.active_sys
        coords = from_device(sys.coords)
        atoms = from_device(sys.atoms)
        velocities = from_device(sys.velocities)
        ntuple(N) do dim_i
            calculate_cv(
                cv_types[dim_i],
                coords,
                atoms,
                sys.boundary,
                velocities,
            )
        end
    end
end

function PMFDeconvolution(state::TSSState{T};
                          grid,
                          coupling = nothing,
                          cv = nothing,
                          target_temp = nothing,
                          target_pressure = nothing,
                          free_energies = nothing,
                          uncertainty = nothing,
                          solver = nothing,
                          kwargs...) where T
    isempty(kwargs) ||
        throw(ArgumentError("unsupported PMFDeconvolution keyword(s): " *
                            "$(join(keys(kwargs), ", "))."))
    isnothing(target_temp) && isnothing(target_pressure) ||
        throw(ArgumentError("target_temp and target_pressure are not supported by " *
                            "TSS sampled PMF deconvolution."))
    isnothing(free_energies) && isnothing(uncertainty) && isnothing(solver) ||
        throw(ArgumentError("direct TSS PMF deconvolution from final free energies has " *
                            "been removed. Create PMFDeconvolution(tss_state; grid=...) " *
                            "before the run and pass it to TSSSimulation(...; pmf=...)."))
    if !isnothing(cv) && isnothing(coupling)
        throw(ArgumentError("provide coupling when using a custom cv function for PMF deconvolution."))
    end

    pmf_grid = _PMFGrid(grid; T = Float64)
    cv_function = isnothing(cv) ? _tss_auto_pmf_cv(state, pmf_grid) : cv
    log_coupling_matrix = _pmf_build_log_coupling_matrix(
        state.state_space,
        pmf_grid;
        coupling = coupling,
    )
    accumulator = _SampledPMFDeconvolutionAccumulator(pmf_grid)
    backend = _TSSPMFDeconvolutionBackend{
        length(pmf_grid.shape),
        Float64,
        typeof(state),
        typeof(cv_function),
        typeof(pmf_grid),
        typeof(log_coupling_matrix),
        typeof(accumulator),
    }(
        state,
        cv_function,
        pmf_grid,
        log_coupling_matrix,
        accumulator,
    )
    return PMFDeconvolution(backend)
end

function _collect_tss_pmf_deconvolution_sample(::Nothing,
                                               estimator,
                                               active_state;
                                               window_offset = 0)
    return nothing
end

function _collect_tss_pmf_deconvolution_sample(::Nothing,
                                               state,
                                               estimator,
                                               active_state;
                                               window_offset = 0)
    return nothing
end

function _tss_pmf_log_bin_weights!(dest::AbstractVector{T},
                                   backend::_TSSPMFDeconvolutionBackend,
                                   estimator::_TSSLocalEstimator;
                                   window_offset = 0) where T
    local_log_state_weights = Vector{T}(undef, length(estimator.state_indices))
    offset = T(window_offset)
    for local_i in eachindex(local_log_state_weights)
        local_log_state_weights[local_i] =
            T(estimator.f[local_i]) + T(estimator.log_dens[local_i]) - offset
    end
    return _pmf_log_bin_weights!(
        dest,
        backend.log_coupling_matrix,
        local_log_state_weights;
        state_indices = estimator.state_indices,
    )
end

function _collect_tss_pmf_deconvolution_sample(
    deconv::PMFDeconvolution{<:_TSSPMFDeconvolutionBackend{N, T}},
    estimator::_TSSLocalEstimator,
    active_state;
    window_offset = 0) where {N, T}

    backend = deconv.backend
    value = _online_pmf_tuple(backend.cv_function(active_state), T)
    length(value) == N ||
        throw(DimensionMismatch("PMF CV returned $(length(value)) dimensions, expected $(N)."))

    log_bin_weights = Vector{T}(undef, prod(backend.grid.shape))
    _tss_pmf_log_bin_weights!(
        log_bin_weights,
        backend,
        estimator;
        window_offset = window_offset,
    )
    return _PMFDeconvolutionSample(value, log_bin_weights, zero(T))
end

function _collect_tss_pmf_deconvolution_sample(
    deconv::PMFDeconvolution{<:_TSSPMFDeconvolutionBackend{N, T}},
    state::TSSState,
    estimator::_TSSLocalEstimator,
    active_state;
    window_offset = 0) where {N, T}

    backend = deconv.backend
    value = _online_pmf_tuple(backend.cv_function(active_state), T)
    length(value) == N ||
        throw(DimensionMismatch("PMF CV returned $(length(value)) dimensions, expected $(N)."))

    log_bin_weights = Vector{T}(undef, prod(backend.grid.shape))
    _tss_pmf_log_bin_weights!(
        log_bin_weights,
        backend,
        estimator;
        window_offset = window_offset,
    )
    return _PMFDeconvolutionSample(value, log_bin_weights, zero(T))
end

function _accumulate_tss_pmf_deconvolution!(::Nothing, observations)
    return nothing
end

function _accumulate_tss_pmf_deconvolution!(
    deconv::PMFDeconvolution{<:_TSSPMFDeconvolutionBackend},
    observations)

    for observation in observations
        for sample in observation.pmf_deconvolution_samples
            _accumulate_pmf_deconvolution!(deconv.backend.accumulator, sample)
        end
    end
    return deconv
end

function pmf(backend::_TSSPMFDeconvolutionBackend;
             zero::Symbol = :min,
             kBT = nothing)
    return _pmf_result_from_sampled_deconvolution(
        backend.accumulator;
        zero = zero,
        kBT = kBT,
    )
end
