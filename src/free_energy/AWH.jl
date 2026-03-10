export 
    AWHState,
    AWHSimulation,
    calc_pmf
    
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
mutable struct AWHState{T, P, S, I, B, PR, SPI, SSI, SGI}
    partition::P
    
    active_idx::Int
    active_sys::S
    active_intg::I
    
    λ_integrators::Vector{I}
    λ_β::B
    λ_p::PR
    
    target_β::Union{T, Nothing}
    target_p::Union{T, Nothing}
    
    # Store full interactions for accurate MD integration forces
    state_pairwise_inters::SPI
    state_specific_inter_lists::SSI
    state_general_inters::SGI

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

function AWHState(thermo_states::AbstractArray{<:ThermoState};
                  target_state::Union{ThermoState, Nothing} = nothing,
                  first_state::Int = 1,
                  n_bias::Int = 100,
                  ρ::Union{Nothing, AbstractArray{T}} = nothing,
                  reuse_neighbors::Bool = true) where T

    n_λ = length(thermo_states)
    n_λ > 0 || throw(ArgumentError("`thermo_states` cannot be empty."))
    1 <= first_state <= n_λ || throw(ArgumentError("`first_state`=$first_state is out of bounds for $n_λ states."))
    n_bias > 0 || throw(ArgumentError("`n_bias` must be positive, got $n_bias."))
    ref_sys = thermo_states[first_state].system
    FT = typeof(ustrip(ref_sys.total_mass))

    partition = AlchemicalPartition(thermo_states; 
                                    target_state=target_state, 
                                    reuse_neighbors=reuse_neighbors)

    # Extract integrators and parameters
    λ_integrators = [ts.integrator for ts in thermo_states]
    λ_β = [ts.beta for ts in thermo_states]
    λ_p = [isnothing(ts.p) ? zero(FT) : ts.p for ts in thermo_states]

    # Extract Target Thermodynamics for PMF
    local target_β_val = nothing
    local target_p_val = nothing
    
    if !isnothing(target_state)
        # ThermoState stores beta and pressure in Molly-internal units already.
        target_β_val = FT(target_state.beta)
        target_p_val = isnothing(target_state.p) ? zero(FT) : FT(target_state.p)
    end

    # Extract complete interaction lists for standard MD forces
    state_pairwise_inters = [ts.system.pairwise_inters for ts in thermo_states]
    state_specific_inter_lists = [ts.system.specific_inter_lists for ts in thermo_states]
    state_general_inters = [ts.system.general_inters for ts in thermo_states]

    active_sys = System(deepcopy(ref_sys);
        atoms = partition.λ_atoms[first_state],
        pairwise_inters = state_pairwise_inters[first_state],
        specific_inter_lists = state_specific_inter_lists[first_state],
        general_inters = state_general_inters[first_state]
    )
  
    active_intg = λ_integrators[first_state]

    # Handle Target Distribution (ρ)
    if isnothing(ρ)
        rho_val = fill(FT(1/n_λ), n_λ)
    else
        rho_val = FT.(ρ)
        length(rho_val) == n_λ || throw(ArgumentError("`ρ` length $(length(rho_val)) does not match number of states $n_λ."))
        all(isfinite, rho_val) || throw(ArgumentError("`ρ` must contain only finite values."))
        any(x -> x < zero(FT), rho_val) && throw(ArgumentError("`ρ` must be nonnegative."))
        rho_sum = sum(rho_val)
        rho_sum > zero(FT) || throw(ArgumentError("`ρ` must have positive total weight."))
        rho_val ./= rho_sum
    end
    log_ρ = log.(rho_val)

    stats = AWHStats(Int[], Int[], Vector{FT}[], FT[], Symbol[], FT[])

    return AWHState{FT, typeof(partition), typeof(active_sys), typeof(active_intg), 
                    typeof(λ_β), typeof(λ_p), typeof(state_pairwise_inters),
                    typeof(state_specific_inter_lists), typeof(state_general_inters)}(
        partition,
        first_state,
        active_sys,
        active_intg,
        λ_integrators,
        λ_β,
        λ_p,
        target_β_val,
        target_p_val,
        state_pairwise_inters,
        state_specific_inter_lists,
        state_general_inters,
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

# Implements global Multistate Bennett Acceptance Ratio (MBAR) reweighting
# to obtain the unbiased PMF across distinct thermodynamic states.
# See Lundborg et al. 2021 https://doi.org/10.1063/5.0044352
#
# Note on algorithmic transition:
# The previous implementation utilized the spatial deconvolution method 
# introduced in Lindahl et al. 2014 (https://doi.org/10.1063/1.4890371). 
# That method assumes the total energy separates strictly into a static 
# core potential and a varying geometric bias: U_λ(x) = U_core(x) + V(ξ(x), λ).
# By assuming U_core(x) is identical across all λ states, it precomputes a 
# static coupling matrix to unbias the trajectory.
# 
# However, during alchemical transformations or temperature-replica AWH runs, 
# the core physical interactions fundamentally change across windows (ΔU_core or Δβ). 
# The spatial deconvolution method cannot account for these Hamiltonian differences,
# resulting in invalid free energy estimates for non-geometric variables.
# 
# To ensure thermodynamic consistency across all AWH application types, this 
# implementation now abandons the spatial coupling matrix. Instead using 
# exact MBAR reweighting by evaluating the full physical Hamiltonian for every 
# sampled coordinate frame across all states to compute the exact mixture weight.
mutable struct AWHPMFDeconvolution{N, T, F_CV}
    min_vals::NTuple{N, T}
    bin_widths::NTuple{N, T}
    shape::NTuple{N, Int}
    is_periodic::NTuple{N, Bool} 
    cv_function::F_CV         

    numerator_hist::Array{T, N}
    denominator_hist::Array{T, N}
    sample_count::Int

    target_beta::T
    target_pressure::T
end

function AWHPMFDeconvolution(
    awh_state::AWHState{T},
    pmf_grid::Tuple;
    cv_func = nothing,
    is_periodic = nothing
) where T
    
    min_vals = T.(pmf_grid[1])
    max_vals = T.(pmf_grid[2])
    n_bins   = Int.(pmf_grid[3])
    N = length(n_bins)
    bin_widths = (max_vals .- min_vals) ./ n_bins
    periodic_flags = isnothing(is_periodic) ? ntuple(_ -> false, N) : Tuple(Bool.(is_periodic))
    
    local final_cv_func
    
    if !isnothing(cv_func)
        final_cv_func = cv_func
    else
        first_ham = awh_state.partition.λ_hamiltonians[1]
        bias_indices = findall(x -> x isa BiasPotential, first_ham.general_inters)
        
        if isempty(bias_indices)
            error("No BiasPotential found in AWHState. Cannot auto-detect PMF settings.")
        end

        cv_types = [first_ham.general_inters[i].cv_type for i in bias_indices]
        
        final_cv_func = (coords) -> begin
            sys = awh_state.active_sys
            return ntuple(N) do d
                Molly.calculate_cv(cv_types[d], coords, sys.atoms, sys.boundary, sys.velocities)
            end
        end
    end

    return AWHPMFDeconvolution(
        min_vals, bin_widths, n_bins, periodic_flags, final_cv_func,
        zeros(T, n_bins), zeros(T, n_bins), 0, awh_state.target_β, awh_state.target_p
    )
end

function update_pmf!(
    pmf::AWHPMFDeconvolution{N, T, F_CV},
    awh_state, 
    curr_coords;
    weight_factor::T = one(T), 
    box_volume::T = zero(T),
    apply_forgetting::Bool = true
) where {N, T, F_CV}
    
    val = pmf.cv_function(from_device(curr_coords))
    current_cv = val isa Tuple ? val : (val,)

    # Determine Index with conditional periodic wrapping
    current_indices = ntuple(N) do d
        rel = (current_cv[d] - pmf.min_vals[d]) / pmf.bin_widths[d]
        idx = Int(floor(rel)) + 1
        
        if pmf.is_periodic[d]
            mod(idx - 1, pmf.shape[d]) + 1
        else
            clamp(idx, 1, pmf.shape[d])
        end
    end
    current_linear_idx = LinearIndices(pmf.shape)[CartesianIndex(current_indices)]

    # 1. Mixture Denominator W_mix = \sum_k exp(f_k + ln ρ_k - u_k(x))
    # Read directly from the preceding process_sample step
    log_W_mix = Molly.logsumexp(awh_state.scratch_z)
    
    # 2. Evaluate Exact Target Energy
    target_pe_units = evaluate_energy!(
        awh_state.partition, 
        curr_coords, 
        awh_state.active_sys.boundary, 
        awh_state.partition.target_hamiltonian, 
        awh_state.partition.target_atoms
    )
    
    unbiased_pe = ustrip(target_pe_units)
    if isnan(unbiased_pe)
        unbiased_pe = typemax(T)
    end
    
    # 3. Calculate MBAR Reweighting Factor targeting the exact physical state
    u_target = pmf.target_beta * (unbiased_pe + pmf.target_pressure * box_volume)
    unbias_log = -u_target - log_W_mix
    w_frame = exp(unbias_log)
    
    # 4. Exponential Forgetting & Accumulation
    if apply_forgetting && weight_factor < one(T)
        pmf.numerator_hist .*= weight_factor
        pmf.denominator_hist .*= weight_factor
    end
    
    pmf.numerator_hist[current_linear_idx] += w_frame
    pmf.denominator_hist[current_linear_idx] += one(T)
    pmf.sample_count += 1
end

@doc raw"""
    calc_pmf(pmf_calc::AWHPMFDeconvolution)

Extracts the unbiased Potential of Mean Force (PMF) from the accumulated numerator 
histograms. Unsampled bins are assigned a value of `Inf`, and the 
global minimum of the valid PMF is shifted to zero.
"""
function calc_pmf(pmf_calc::AWHPMFDeconvolution{N, T, F_CV}) where {N, T, F_CV}
    num = pmf_calc.numerator_hist
    
    # Identify bins with non-zero reweighted samples to avoid domain errors in log.
    # The denominator histogram is no longer used for the free energy calculation.
    valid = num .> zero(T)
    
    # Initialize PMF array with infinity for unsampled regions
    pmf = fill(T(Inf), size(num))
    
    # Calculate absolute unbiased PMF for valid bins based directly on the sum of MBAR weights
    pmf[valid] .= -log.(num[valid])
    
    # Shift the global minimum to zero
    if any(valid)
        min_val = minimum(pmf[valid])
        pmf[valid] .-= min_val
    end
    
    return pmf
end

@doc raw"""
    AWHSimulation(awh_state::AWHState{T};
                  <keyword arguments>)

Prepares an Accelerated Weight Histogram (AWH) simulation.

This struct stores the parameters controlling the AWH updates. It can optionally perform
on-the-fly PMF deconvolution to obtain an unbiased estimate of the true PMF, as described
in [Lindahl et al. (2014)](https://doi.org/10.1063/1.4890371). Deconvolution is recommended
for simulations where the λ coordinate represents a biased reaction coordinate (e.g., an 
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
- `pmf_grid=nothing`: Tuple of tuples defining the PMF grid `((min_1, ...), (max_1, ...), (bins_1, ...))`.
    Required if running with PMF deconvolution.
- `pmf_cv=nothing`: Function taking system coordinates and returning a tuple of scalar Collective 
    Variables (CVs). If omitted when `pmf_grid` is provided, Molly attempts to auto-detect CVs from 
    the active `BiasPotential`s.
- `pmf_coupling=nothing`: Function returning the dimensionless bias energy given the CV tuple and 
    a `lambda_idx`. If omitted when `pmf_grid` is provided, Molly computes this automatically.
- `target_temp=nothing`: Target temperature for the unbiased PMF. Recommended for proper reweighting 
    if λ states have varying temperatures.
- `target_pressure=nothing`: Target pressure for the unbiased PMF. Recommended for proper reweighting 
    if λ states have varying pressures.
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
    
    pmf_calc::Union{AWHPMFDeconvolution, Nothing}
end

function AWHSimulation(
    awh_state::AWHState{T};
    num_md_steps::Int = 10,
    update_freq::Int = 1,
    well_tempered_factor::Real = 10.0,
    coverage_threshold::Real = 1.0,
    significant_weight::Real = 0.1,
    log_freq::Int = 100,
    pmf_grid = nothing,
    pmf_cv = nothing,
    pmf_is_periodic = nothing
) where T

    n_win = length(awh_state.partition.λ_hamiltonians)

    pmf_calc = nothing
    if !isnothing(pmf_grid)
        if isnothing(awh_state.target_β)
            error("PMF deconvolution requires a target_state to be passed during AWHState initialization.")
        end
        
        pmf_calc = AWHPMFDeconvolution(
            awh_state, 
            pmf_grid;
            cv_func=pmf_cv, 
            is_periodic=pmf_is_periodic
        )
    end

    return AWHSimulation(n_win,
                         copy(awh_state.N_bias),
                         num_md_steps,
                         update_freq,
                         T(well_tempered_factor),
                         T(coverage_threshold),
                         T(significant_weight),
                         log_freq, awh_state,
                         pmf_calc)
end

# Swaps Hamiltonians and enforces kinetic energy continuity across temperature jumps and mass changes
function update_active_sys!(awh_state::AWHState, active_idx::Int)
    old_idx = awh_state.active_idx
    
    # Perform instantaneous velocity rescaling if the temperature target or atomic masses change
    if old_idx != active_idx
        β_old = awh_state.λ_β[old_idx]
        β_new = awh_state.λ_β[active_idx]
        
        old_atoms = awh_state.partition.λ_atoms[old_idx]
        new_atoms = awh_state.partition.λ_atoms[active_idx]
        
        for i in eachindex(awh_state.active_sys.velocities)
            m_old = mass(old_atoms[i])
            m_new = mass(new_atoms[i])
            
            if β_old != β_new || m_old != m_new
                # Zero-mass particles have no kinetic contribution; avoid 0/0
                # when swapping between temperatures or mass-scaled states.
                if iszero(m_old) || iszero(m_new)
                    continue
                end
                # ustrip ensures the resulting scalar is a raw float, preventing Unitful type instability
                velocity_scaling_factor = sqrt(ustrip((β_old * m_old) / (β_new * m_new)))
                awh_state.active_sys.velocities[i] .*= velocity_scaling_factor
            end
        end
    end

    awh_state.active_idx = active_idx
    awh_state.active_sys.atoms = awh_state.partition.λ_atoms[active_idx]
    awh_state.active_sys.pairwise_inters = awh_state.state_pairwise_inters[active_idx]
    awh_state.active_sys.specific_inter_lists = awh_state.state_specific_inter_lists[active_idx]
    awh_state.active_sys.general_inters = awh_state.state_general_inters[active_idx]
    awh_state.active_intg = awh_state.λ_integrators[active_idx]
end

# Reweights coordinates along λ windows and accumulates histogram
function process_sample(awh::AWHState{FT}; weight_relevance::Real = 0.1) where FT
    n_states = length(awh.λ_β)
    coords = awh.active_sys.coords
    bound  = awh.active_sys.boundary

    # Exploit the AlchemicalPartition abstraction
    energies = evaluate_energy_all!(awh.partition, coords, bound)
    
    # Extract volume in consistent units
    vol_val = FT(ustrip(volume(bound)))
    
    potentials = awh.scratch_potentials 
    active_pe = nothing

    for n in 1:n_states
        pe = energies[n]
        if n == awh.active_idx
            active_pe = pe
        end
        
        pe_val = ustrip(pe)
        
        # Trap r=0 Lennard-Jones overlaps (Inf - Inf = NaN)
        if isnan(pe_val)
            pe_val = typemax(FT)
        end
        
        # Include Pressure-Volume work for NPT ensemble validity
        pv_work = awh.λ_p[n] * vol_val
        
        # β is already converted to a raw float in correct units
        potentials[n] = awh.λ_β[n] * (pe_val + pv_work)
    end

    # Calculate Z in-place
    @. awh.scratch_z = awh.log_rho + awh.f - potentials
    
    log_den = Molly.logsumexp(awh.scratch_z)
    
    # Prevent NaN propagation if all potentials evaluate to typemax(FT)
    if isinf(log_den) && log_den < zero(FT)
        fill!(awh.w_last, one(FT) / n_states)
    else
        # Calculate W directly into w_last
        @. awh.w_last = exp(awh.scratch_z - log_den)
    end

    # Accumulate
    awh.w_seg .+= awh.w_last
    awh.n_accum += 1
    awh.N_eff   += 1

    # Check visited windows using w_last
    for (i, val) in enumerate(awh.w_last)
        if val > weight_relevance * awh.rho[i]
            push!(awh.visited_windows, i)
        end
    end
    
    return active_pe
end

# Decides which is the new Hamiltonian given some weights
function gibbs_sample_window(state::AWHState)
    return sample(1:length(state.w_last), Weights(state.w_last))
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

    current_N = if awh_sim.state.in_initial_stage
        awh_sim.state.N_bias
    else
        # In the linear stage, Eq. (4) uses the reference histogram size
        # before the current update block is folded in.
        awh_sim.initial_sampl_n + (awh_sim.state.N_eff - awh_sim.state.n_accum)
    end

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
        n_target_windows = count(x -> x > zero(eltype(awh_sim.state.rho)), awh_sim.state.rho)
        required_cov = floor(Int, awh_sim.coverage_threshold * n_target_windows)
        if cov_count >= required_cov
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

function simulate!(awh_sim::AWHSimulation{T}, n_steps::Int) where T

    n_iterations = Int(floor(n_steps / awh_sim.n_md_steps))
    active_idx = awh_sim.state.active_idx

    for iteration_n in 1:n_iterations
        simulate!(awh_sim.state.active_sys, awh_sim.state.active_intg, awh_sim.n_md_steps)


        process_sample(awh_sim.state; weight_relevance=awh_sim.significant_weight)

        if !isnothing(awh_sim.pmf_calc)
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
            vol_val = T(ustrip(volume(sys.boundary)))
            apply_forgetting = awh_sim.state.n_accum == 1

            update_pmf!(
                awh_sim.pmf_calc, 
                awh_sim.state, 
                awh_sim.state.active_sys.coords; 
                weight_factor=w_fac,
                box_volume=vol_val,
                apply_forgetting=apply_forgetting
            )
        end

        active_idx_new = gibbs_sample_window(awh_sim.state)
        if active_idx_new != active_idx
            active_idx = active_idx_new
        end

        update_active_sys!(awh_sim.state, active_idx)
        update_awh_bias!(awh_sim, iteration_n)
    end
end
