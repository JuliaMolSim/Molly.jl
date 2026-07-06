mutable struct TSSPMFEpochAccumulator{A}
    index::Int
    accumulator::A
end

mutable struct TSSPMFDeconvolutionBackend{N, T, S, F_CV, G, C, A}
    state::S
    cv_function::F_CV
    grid::G
    log_coupling_matrix::C
    accumulator::A
    epoch_accumulators::Vector{TSSPMFEpochAccumulator{A}}
end

function tss_auto_pmf_cv(state::TSSState, grid::PMFGrid{N}) where N
    first_ham = state.state_space.partition.λ_hamiltonians[1]
    bias_indices = findall(inter -> inter isa BiasPotential, first_ham.general_inters)
    if length(bias_indices) != N
        throw(ArgumentError("automatic PMF deconvolution found $(length(bias_indices)) " *
                            "BiasPotential interactions in the first TSS state, but the " *
                            "PMF grid is $(N)D. Provide explicit cv and coupling functions."))
    end
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
    if !isempty(kwargs)
        throw(ArgumentError("unsupported PMFDeconvolution keyword(s): " *
                            "$(join(keys(kwargs), ", "))."))
    end
    if !(isnothing(target_temp) && isnothing(target_pressure))
        throw(ArgumentError("target_temp and target_pressure are not supported by " *
                            "TSS sampled PMF deconvolution."))
    end
    if !(isnothing(free_energies) && isnothing(uncertainty) && isnothing(solver))
        throw(ArgumentError("direct TSS PMF deconvolution from final free energies has " *
                            "been removed. Create PMFDeconvolution(tss_state; grid=...) " *
                            "before the run and pass it to TSSSimulation(...; pmf=...)."))
    end
    if !isnothing(cv) && isnothing(coupling)
        throw(ArgumentError("provide coupling when using a custom cv function for PMF deconvolution."))
    end

    pmf_grid =PMFGrid(grid; T = Float64)
    cv_function = isnothing(cv) ? tss_auto_pmf_cv(state, pmf_grid) : cv
    log_coupling_matrix =pmf_build_log_coupling_matrix(
        state.state_space,
        pmf_grid;
        coupling = coupling,
    )
    accumulator =SampledPMFDeconvolutionAccumulator(pmf_grid)
    backend =TSSPMFDeconvolutionBackend{
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
        TSSPMFEpochAccumulator{typeof(accumulator)}[],
    )
    return PMFDeconvolution(backend)
end

function collect_tss_pmf_deconvolution_sample(::Nothing,
                                               state,
                                               estimator,
                                               active_state;
                                               window_offset = 0)
    return nothing
end

function tss_pmf_log_bin_weights!(dest::AbstractVector{T},
                                   backend::TSSPMFDeconvolutionBackend,
                                   estimator::TSSLocalEstimator;
                                   window_offset = 0,
                                   log_weight_factor = zero(T)) where T
    local_log_state_weights = Vector{T}(undef, length(estimator.state_indices))
    offset = T(window_offset)
    for local_i in eachindex(local_log_state_weights)
        local_log_state_weights[local_i] =
            T(estimator.f[local_i]) + T(estimator.log_dens[local_i]) - offset
    end
    return pmf_log_bin_weights!(
        dest,
        backend.log_coupling_matrix,
        local_log_state_weights;
        state_indices = estimator.state_indices,
        log_weight_factor = T(log_weight_factor),
    )
end

function collect_tss_pmf_deconvolution_sample(
    deconv::PMFDeconvolution{<:TSSPMFDeconvolutionBackend{N, T}},
    estimator::TSSLocalEstimator,
    active_state;
    window_offset = 0) where {N, T}

    backend = deconv.backend
    value =online_pmf_tuple(backend.cv_function(active_state), T)
    if length(value) != N
        throw(DimensionMismatch("PMF CV returned $(length(value)) dimensions, expected $(N)."))
    end

    log_bin_weights = Vector{T}(undef, prod(backend.grid.shape))
    tss_pmf_log_bin_weights!(
        log_bin_weights,
        backend,
        estimator;
        window_offset = window_offset,
    )
    return PMFDeconvolutionSample(value, log_bin_weights, zero(T))
end

function collect_tss_pmf_deconvolution_sample(
    deconv::PMFDeconvolution{<:TSSPMFDeconvolutionBackend},
    state::TSSState,
    estimator::TSSLocalEstimator,
    active_state;
    window_offset = 0)
    return collect_tss_pmf_deconvolution_sample(
        deconv,
        estimator,
        active_state;
        window_offset = window_offset,
    )
end

function tss_pmf_uses_epoch_history(estimator::TSSLocalEstimator)
    return !isnothing(estimator.history) && !iszero(estimator.history.config.alpha)
end

function tss_pmf_epoch_accumulator!(backend::TSSPMFDeconvolutionBackend, epoch_index::Int)
    existing = findfirst(epoch -> epoch.index == epoch_index, backend.epoch_accumulators)
    if isnothing(existing)
        push!(
            backend.epoch_accumulators,
            TSSPMFEpochAccumulator(
                epoch_index,
                SampledPMFDeconvolutionAccumulator(backend.grid),
            ),
        )
        return last(backend.epoch_accumulators).accumulator
    end
    return backend.epoch_accumulators[existing].accumulator
end

function accumulate_tss_pmf_deconvolution!(
    ::Nothing,
    state::TSSState,
    observations;
    history_time::Integer,
)
    return nothing
end

function accumulate_tss_pmf_deconvolution!(
    deconv::PMFDeconvolution{<:TSSPMFDeconvolutionBackend},
    state::TSSState,
    observations;
    history_time::Integer,
)
    history_time > 0 || throw(ArgumentError("history_time must be positive."))
    backend = deconv.backend
    for observation in observations
        estimator = state.estimators[observation.update_window]
        accumulator = if tss_pmf_uses_epoch_history(estimator)
            epoch_index = tss_epoch_index!(estimator.history, Int(history_time))
            tss_pmf_epoch_accumulator!(backend, epoch_index)
        else
            backend.accumulator
        end
        for sample in observation.pmf_deconvolution_samples
            accumulate_pmf_deconvolution!(accumulator, sample)
        end
    end
    return deconv
end

function drop_old_tss_pmf_deconvolution_epochs!(::Nothing, state::TSSState, history_time::Integer)
    return nothing
end

function drop_old_tss_pmf_deconvolution_epochs!(
    deconv::PMFDeconvolution{<:TSSPMFDeconvolutionBackend},
    state::TSSState,
    history_time::Integer,
)
    backend = deconv.backend
    isempty(backend.epoch_accumulators) && return deconv
    retained = Set{Int}()
    for estimator in state.estimators
        tss_pmf_uses_epoch_history(estimator) || continue
        first_recent = tss_first_retained_epoch_index!(estimator.history, Int(history_time))
        current = tss_epoch_index!(estimator.history, Int(history_time))
        for epoch_index in first_recent:current
            push!(retained, epoch_index)
        end
    end
    isempty(retained) && return deconv
    filter!(epoch -> epoch.index in retained, backend.epoch_accumulators)
    return deconv
end

function tss_pmf_retained_accumulator(backend::TSSPMFDeconvolutionBackend)
    isempty(backend.epoch_accumulators) && return backend.accumulator
    acc = SampledPMFDeconvolutionAccumulator(backend.grid)
    for epoch in backend.epoch_accumulators
        merge_pmf_deconvolution_accumulator!(acc, epoch.accumulator)
    end
    return acc
end

pmf_deconvolution_accumulator(backend::TSSPMFDeconvolutionBackend) =
    tss_pmf_retained_accumulator(backend)

function pmf(backend::TSSPMFDeconvolutionBackend;
             zero::Symbol = :min,
             kBT = nothing,
             kwargs...)
    return pmf_result_from_sampled_deconvolution(
        tss_pmf_retained_accumulator(backend);
        zero = zero,
        kBT = kBT,
        kwargs...,
    )
end
