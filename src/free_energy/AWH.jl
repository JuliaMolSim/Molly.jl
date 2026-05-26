export 
    AWHState,
    AWHSimulation
    
# Convenience struct to store relevant things
# when running an AWH simulation.
mutable struct AWHStats{T}
    step_indices::Vector{Int}
    active_λ::Vector{Int}
    f_history::Vector{Vector{T}}
    n_effective_history::Vector{T}
    stage_history::Vector{Symbol}
    max_delta_f_history::Vector{T}
end


@doc raw"""
    AWHState(thermo_states::AbstractArray{ThermoState};
             <keyword arguments>)

State of an Accelerated Weight Histogram (AWH) simulation.

Maintains the physical state of the system across multiple λ windows, as well as the
accumulated statistical weights, free energy estimates, and target distribution parameters.

# Arguments
- `thermo_states::AbstractArray{ThermoState}`: Iterable containing the [`ThermoState`](@ref) structs 
    defining each λ window of the AWH simulation.
- `first_state::Int=1`: The index of the initial active λ state.
- `n_bias::Int=100`: Fictitious effective sampling size used during the initial stage. Smaller values
    yield more aggressive free energy updates.
- `ρ::Union{Nothing, AbstractArray{T}}=nothing`: Target distribution array along λ. If `nothing`,
    a uniform distribution is used.
- `reuse_neighbors::Bool=true`: Whether to reuse the active system's neighbor list when calculating
    energies for adjacent λ windows. Generally improves performance.
"""
mutable struct AWHState{T, ES, AS}
    state_space::ES
    active_state::AS

    # Probability & Free Energy
    f::Vector{T}
    rho::Vector{T}
    log_rho::Vector{T}
    
    # Weight Accumulators
    w_seg::Vector{T}
    w_last::Vector{T}

    scratch_potentials::Vector{T}
    scratch_z::Vector{T}
    
    N_eff::T
    N_bias::T          
    n_accum::Int

    in_initial_stage::Bool
    visited_windows::Set{Int}

    stats::AWHStats{T}
end

function Base.getproperty(state::AWHState, name::Symbol)
    if name === :partition
        return getfield(getfield(state, :state_space), :partition)
    elseif name === :active_idx
        return getfield(getfield(state, :active_state), :active_idx)
    elseif name === :active_sys
        return getfield(getfield(state, :active_state), :active_sys)
    elseif name === :active_intg
        return getfield(getfield(state, :active_state), :active_integrator)
    elseif name === :λ_integrators
        return getfield(getfield(state, :state_space), :integrators)
    elseif name === :λ_β
        return getfield(getfield(state, :state_space), :betas)
    elseif name === :λ_p
        return getfield(getfield(state, :state_space), :pressures)
    elseif name === :state_pairwise_inters
        return getfield(getfield(state, :state_space), :state_pairwise_inters)
    elseif name === :state_specific_inter_lists
        return getfield(getfield(state, :state_space), :state_specific_inter_lists)
    elseif name === :state_general_inters
        return getfield(getfield(state, :state_space), :state_general_inters)
    elseif name === :λ_hamiltonians
        return getfield(getfield(getfield(state, :state_space), :partition), :λ_hamiltonians)
    end
    return getfield(state, name)
end

function Base.setproperty!(state::AWHState, name::Symbol, value)
    if name === :active_idx
        getfield(state, :active_state).active_idx = value
    elseif name === :active_sys
        getfield(state, :active_state).active_sys = value
    elseif name === :active_intg
        getfield(state, :active_state).active_integrator = value
    else
        setfield!(state, name, value)
    end
    return value
end

function Base.propertynames(state::AWHState, private::Bool=false)
    names = collect(fieldnames(typeof(state)))
    append!(names, (:partition, :active_idx, :active_sys, :active_intg, :λ_integrators,
                    :λ_β, :λ_p, :state_pairwise_inters, :state_specific_inter_lists,
                    :state_general_inters, :λ_hamiltonians))
    return Tuple(names)
end

function AWHState(thermo_states::AbstractArray{<:ThermoState};
                  first_state::Int = 1,
                  n_bias::Int = 100,
                  ρ::Union{Nothing, AbstractArray} = nothing,
                  reuse_neighbors::Bool = true)

    state_space = ExtendedStateSpace(thermo_states; reuse_neighbors=reuse_neighbors)
    active_state = ActiveThermoState(state_space, first_state)
    n_λ = n_states(state_space)
    ref_sys = state_space.systems[first_state]
    FT = typeof(ustrip(ref_sys.total_mass))

    # Handle Target Distribution (ρ)
    if isnothing(ρ)
        rho_val = fill(FT(1/n_λ), n_λ)
    else
        rho_val = eltype(ρ) != FT ? FT.(ρ) : ρ
    end
    log_ρ = log.(rho_val)

    stats = AWHStats(Int[], Int[], Vector{FT}[], FT[], Symbol[], FT[])

    return AWHState{FT, typeof(state_space), typeof(active_state)}(
        state_space,
        active_state,
        zeros(FT, n_λ),
        rho_val,
        log_ρ,
        zeros(FT, n_λ),
        zeros(FT, n_λ),
        zeros(FT, n_λ),
        zeros(FT, n_λ),
        zero(FT),
        FT(n_bias),
        0,
        true,
        Set{Int}(),
        stats
    )
end

mutable struct _AWHPMFDeconvolutionBackend{N, T, F_CV, G, C, A}
    grid::G
    cv_function::F_CV
    log_coupling_matrix::C
    accumulator::A
    target_beta::Union{T, Nothing}
    target_pressure::Union{T, Nothing}
    cv_history::Vector{NTuple{N, T}}
    active_idx_history::Vector{Int}
    scratch_g::Vector{T}
    scratch_log_bin_weights::Vector{T}
end

function _awh_auto_pmf_cv(awh_state::AWHState, grid::_PMFGrid{N}) where N
    first_ham = awh_state.partition.λ_hamiltonians[1]
    bias_indices = findall(inter -> inter isa BiasPotential, first_ham.general_inters)
    length(bias_indices) == N ||
        throw(ArgumentError("automatic PMF deconvolution found $(length(bias_indices)) " *
                            "BiasPotential interactions in the first AWH state, but the " *
                            "PMF grid is $(N)D. Provide explicit cv and coupling functions."))
    cv_types = [first_ham.general_inters[i].cv_type for i in bias_indices]
    return coords -> begin
        sys = awh_state.active_sys
        coords_cpu = from_device(coords)
        atoms_cpu = from_device(sys.atoms)
        velocities_cpu = from_device(sys.velocities)
        ntuple(N) do dim_i
            calculate_cv(
                cv_types[dim_i],
                coords_cpu,
                atoms_cpu,
                sys.boundary,
                velocities_cpu,
            )
        end
    end
end

function _awh_target_beta_pressure(sys, ::Type{T}, target_temp, target_pressure) where T
    e_unit = sys.energy_units

    target_beta = nothing
    if !isnothing(target_temp)
        kBT_q = uconvert(e_unit, Unitful.R * target_temp)
        target_beta = T(1 / ustrip(kBT_q))
    end

    target_press = nothing
    if !isnothing(target_pressure)
        l_unit = unit(sys.boundary.side_lengths[1])
        p_unit = e_unit / l_unit^3
        e_val = one(T) * e_unit
        p_val_scaled = target_pressure * (e_val / energy_remove_mol(e_val))
        target_press = T(ustrip(uconvert(p_unit, p_val_scaled)))
    end

    return target_beta, target_press
end

function PMFDeconvolution(awh_state::AWHState{T};
                          grid,
                          coupling = nothing,
                          cv = nothing,
                          target_temp = nothing,
                          target_pressure = nothing,
                          kwargs...) where T
    isempty(kwargs) ||
        throw(ArgumentError("unsupported PMFDeconvolution keyword(s): " *
                            "$(join(keys(kwargs), ", "))."))
    pmf_grid = _PMFGrid(grid; T = Float64)
    if !isnothing(cv) && isnothing(coupling)
        throw(ArgumentError("provide coupling when using a custom cv function for PMF deconvolution."))
    end
    cv_function = isnothing(cv) ? _awh_auto_pmf_cv(awh_state, pmf_grid) : cv
    log_coupling_matrix = _pmf_build_log_coupling_matrix(
        awh_state.state_space,
        pmf_grid;
        coupling = coupling,
    )
    target_beta, target_press = _awh_target_beta_pressure(
        awh_state.active_sys,
        Float64,
        target_temp,
        target_pressure,
    )
    backend = _AWHPMFDeconvolutionBackend(
        pmf_grid,
        cv_function,
        log_coupling_matrix,
        _SampledPMFDeconvolutionAccumulator(pmf_grid),
        target_beta,
        target_press,
        Vector{NTuple{length(pmf_grid.shape), Float64}}(),
        Int[],
        zeros(Float64, n_states(awh_state.state_space)),
        zeros(Float64, prod(pmf_grid.shape)),
    )
    return PMFDeconvolution(backend)
end

function update_pmf!(deconv::PMFDeconvolution{<:_AWHPMFDeconvolutionBackend{N, T}},
                     awh_state,
                     curr_coords;
                     weight_factor = one(T),
                     potential_energy = zero(T),
                     box_volume = zero(T),
                     current_beta = one(T),
                     current_pressure = zero(T)) where {N, T}
    backend = deconv.backend
    val = backend.cv_function(from_device(curr_coords))
    current_cv = _online_pmf_tuple(val, T)
    length(current_cv) == N ||
        throw(DimensionMismatch("PMF CV returned $(length(current_cv)) dimensions, " *
                                "expected $(N)."))
    push!(backend.cv_history, current_cv)
    push!(backend.active_idx_history, awh_state.active_idx)

    @. backend.scratch_g = Float64(awh_state.f) + Float64(awh_state.log_rho)
    _pmf_log_bin_weights!(
        backend.scratch_log_bin_weights,
        backend.log_coupling_matrix,
        backend.scratch_g;
        log_weight_factor = _pmf_positive_log(weight_factor, "PMF deconvolution weight_factor", Float64),
    )

    reweight_log = 0.0
    if !isnothing(backend.target_beta)
        reweight_log -= (backend.target_beta - Float64(current_beta)) * Float64(potential_energy)
    end
    if !isnothing(backend.target_pressure)
        target_beta = isnothing(backend.target_beta) ? Float64(current_beta) : backend.target_beta
        target_work = target_beta * backend.target_pressure
        current_work = Float64(current_beta) * Float64(current_pressure)
        reweight_log -= (target_work - current_work) * Float64(box_volume)
    end

    _accumulate_pmf_deconvolution!(
        backend.accumulator,
        current_cv,
        backend.scratch_log_bin_weights;
        log_reweight = reweight_log,
    )
    return deconv
end

function pmf(backend::_AWHPMFDeconvolutionBackend{N, T};
             zero::Symbol = :min,
             kBT = nothing) where {N, T}
    return _pmf_result_from_sampled_deconvolution(
        backend.accumulator;
        zero = zero,
        kBT = kBT,
    )
end

@doc raw"""
    AWHSimulation(awh_state::AWHState{T};
                  <keyword arguments>)

Prepares an Accelerated Weight Histogram (AWH) simulation.

This struct stores the parameters controlling the AWH updates. It can optionally update a
[`PMFDeconvolution`](@ref) object on the fly to obtain an unbiased PMF, as described in
[Lindahl et al. (2014)](https://doi.org/10.1063/1.4890371). Deconvolution is recommended
for simulations where the λ coordinate represents a biased reaction coordinate (e.g. an
umbrella potential). It is typically not required for standard alchemical transformations.

# Arguments
- `awh_state::AWHState{T}`: The [`AWHState`](@ref) containing the λ windows and accumulators.
- `num_md_steps::Int=10`: Number of MD integration steps to perform between AWH coordinate 
    sampling and reweighting.
- `update_freq::Int=1`: Number of samples to collect before applying an update to the AWH bias.
- `well_tempered_factor::Real=10.0`: If finite, the AWH target distribution (ρ) is dynamically 
    scaled to favor low-energy λ windows. Smaller values accentuate this behavior. Use `Inf` to disable.
- `coverage_threshold::Real=1.0`: Proportion of λ windows that must be visited to double the
    fictitious sample size and advance the initial stage.
- `significant_weight::Real=0.1`: The fractional weight threshold (relative to an ideally uniform 
    distribution) required for a λ window to be considered "visited". This filters out sampling noise.
- `log_freq::Int=100`: Number of AWH iterations between logging statistics.
- `pmf=nothing`: Optional [`PMFDeconvolution`](@ref) object created from `awh_state`.
"""
struct AWHSimulation{T}
    n_windows::Int
    initial_sampl_n::T
    n_md_steps::Int 
    update_freq::Int        
    well_tempered_fac::T    
    coverage_threshold::T   
    significant_weight::T   
    log_freq::Int           
    state::AWHState{T}
    pmf::Union{PMFDeconvolution, Nothing}
end

function AWHSimulation(
    awh_state::AWHState{T};
    num_md_steps::Int = 10,
    update_freq::Int = 1,
    well_tempered_factor::Real = 10.0,
    coverage_threshold::Real = 1.0,
    significant_weight::Real = 0.1,
    log_freq::Int = 100,
    pmf = nothing
) where T

    n_win = n_states(awh_state.state_space)

    if !isnothing(pmf) && !(pmf isa PMFDeconvolution{<:_AWHPMFDeconvolutionBackend})
        throw(ArgumentError("AWHSimulation pmf must be created with PMFDeconvolution(awh_state; ...)."))
    end

    return AWHSimulation(n_win,
                         copy(awh_state.N_bias),
                         num_md_steps,
                         update_freq,
                         T(well_tempered_factor),
                         T(coverage_threshold),
                         T(significant_weight),
                         log_freq, awh_state,
                         pmf)
end

# Swaps Hamiltionians
function update_active_sys!(awh_state::AWHState, active_idx::Int)
    set_active_state!(awh_state.active_state, awh_state.state_space, active_idx)
    return awh_state
end

# Reweights coordinates along λ windows and accumulates histogram
function process_sample(awh::AWHState{FT}; weight_relevance::Real = 0.1) where FT
    n_win = n_states(awh.state_space)
    coords = awh.active_sys.coords
    bound  = awh.active_sys.boundary

    state_indices = Base.OneTo(n_win)
    energies = evaluate_energy_subset(awh.partition, coords, bound, state_indices)
    reduced_potentials!(awh.scratch_potentials, energies, awh.state_space, bound, state_indices)
    active_pe = energies[awh.active_idx]

    @. awh.scratch_z = awh.log_rho + awh.f
    conditional_state_weights!(awh.w_last, awh.scratch_z, awh.scratch_potentials, awh.scratch_z)

    # Accumulate
    awh.w_seg .+= awh.w_last
    awh.n_accum += 1
    awh.N_eff   += 1

    # Check visited windows using w_last
    for (i, val) in enumerate(awh.w_last)
        if val > weight_relevance/n_win
            push!(awh.visited_windows, i)
        end
    end
    
    return active_pe
end

# Decides which is the new Hamiltonian given some weights
function gibbs_sample_window(state::AWHState)
    return sample_state(state.w_last)
end

# Logs important stuff
function log_awh_statistics!(state::AWHState, step_idx::Int, delta_f, current_N)
    stats = state.stats
    max_change = maximum(abs.(delta_f))
    push!(stats.step_indices, step_idx)
    push!(stats.active_λ, state.active_idx)
    push!(stats.f_history, copy(state.f))
    push!(stats.n_effective_history, current_N)
    push!(stats.stage_history, state.in_initial_stage ? :initial : :linear)
    push!(stats.max_delta_f_history, max_change)
end

# Calculates the update free energy for each λ window
# Updates target distribution if well tempered is required
# Decides if to remain in initial sampling stage or move to linear stage
function update_awh_bias!(awh_sim::AWHSimulation, iteration_n::Int)
    if awh_sim.state.n_accum < awh_sim.update_freq
        return nothing 
    end

    current_N = awh_sim.state.in_initial_stage ? awh_sim.state.N_bias : (awh_sim.initial_sampl_n + awh_sim.state.N_eff)

    numerator   = current_N .* awh_sim.state.rho .+ awh_sim.state.w_seg
    denominator = current_N .* awh_sim.state.rho .+ (awh_sim.state.n_accum .* awh_sim.state.rho)
    
    # Safe log ratio to prevent 0/0 -> NaN if rho and w_seg underflow
    delta_f = zeros(eltype(numerator), length(numerator))
    for i in eachindex(delta_f)
        if denominator[i] > 0
            delta_f[i] = log(numerator[i] / denominator[i])
        else
            delta_f[i] = zero(eltype(numerator))
        end
    end
    
    awh_sim.state.f .-= delta_f
    awh_sim.state.f .-= awh_sim.state.f[1] 

    if iteration_n % awh_sim.log_freq == 0
        log_awh_statistics!(awh_sim.state, iteration_n, delta_f, current_N)
    end

    if isfinite(awh_sim.well_tempered_fac)
        f_min = minimum(awh_sim.state.f)
        @. awh_sim.state.rho = exp( - (awh_sim.state.f - f_min) / awh_sim.well_tempered_fac )
        
        sum_rho = sum(awh_sim.state.rho)
        if sum_rho > 0
            awh_sim.state.rho ./= sum_rho
        end
        
        # Clamp to prevent log(0)
        @. awh_sim.state.rho = max(awh_sim.state.rho, floatmin(eltype(awh_sim.state.rho)))
        @. awh_sim.state.log_rho = log(awh_sim.state.rho)
    end

    if awh_sim.state.in_initial_stage
        cov_count = length(awh_sim.state.visited_windows)
        if cov_count >= floor(Int, awh_sim.coverage_threshold * awh_sim.n_windows)
            awh_sim.state.N_bias *= 2
            empty!(awh_sim.state.visited_windows)
            if awh_sim.state.N_bias >= (awh_sim.initial_sampl_n + awh_sim.state.N_eff)
                awh_sim.state.in_initial_stage = false
            end
        end
    end

    fill!(awh_sim.state.w_seg, 0)
    awh_sim.state.n_accum = 0
    
    return delta_f
end

function simulate!(awh_sim::AWHSimulation{T},
                   n_steps::Int;
                   show_progress = default_show_progress()) where T

    n_iterations = Int(floor(n_steps / awh_sim.n_md_steps))

    progress = setup_progress(n_iterations, show_progress)
    for iteration_n in 1:n_iterations
        simulate!(awh_sim.state.active_sys, awh_sim.state.active_intg, awh_sim.n_md_steps)


        active_pe_units = process_sample(awh_sim.state)

        if !isnothing(awh_sim.pmf)
            # Calculate a(t) factor for PMF Deconvolution [Lindahl et al. 2014, Eq. 9 text]
            # Initial Stage: N is constant => Delta N = 0 => a = N / (N + n_lambda)
            # Linear Stage: N grows => Delta N = n_lambda => a = 1.0
            
            w_fac = one(T)
            if awh_sim.state.in_initial_stage
                current_N = awh_sim.state.N_bias
                n_lambda  = T(awh_sim.update_freq)
                w_fac = current_N / (current_N + n_lambda)
            end

            # Extract System Info
            sys = awh_sim.state.active_sys
            e_unit = sys.energy_units
            
            pot_eng = ustrip(e_unit, active_pe_units)
            
            vol_val = T(ustrip(volume(sys.boundary)))

            cur_beta = awh_sim.state.λ_β[awh_sim.state.active_idx]            
            cur_press = awh_sim.state.λ_p[awh_sim.state.active_idx]

            update_pmf!(
                awh_sim.pmf,
                awh_sim.state, 
                awh_sim.state.active_sys.coords; 
                weight_factor=w_fac,
                potential_energy=pot_eng,
                box_volume=vol_val,
                current_beta=cur_beta,
                current_pressure=cur_press
            )
        end

        active_idx_new = gibbs_sample_window(awh_sim.state)
        update_active_sys!(awh_sim.state, active_idx_new)
        update_awh_bias!(awh_sim, iteration_n)
        next_nograd!(progress)
    end
end
