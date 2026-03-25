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
                  first_state::Int = 1,
                  n_bias::Int = 100,
                  ρ::Union{Nothing, AbstractArray} = nothing,
                  reuse_neighbors::Bool = true)

    n_λ = length(thermo_states)
    ref_sys = thermo_states[first_state].system
    FT = typeof(ustrip(ref_sys.total_mass))

    # Delegate the separation of interactions to the core abstraction
    partition = AlchemicalPartition(thermo_states; reuse_neighbors=reuse_neighbors)

    # Extract integrators and parameters
    λ_integrators = [ts.integrator for ts in thermo_states]
    λ_β = [ts.beta for ts in thermo_states]
    λ_p = [isnothing(ts.p) ? zero(FT) : ts.p for ts in thermo_states]

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
        rho_val = eltype(ρ) != FT ? FT.(ρ) : ρ
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

# Implements the deconvolution method to obtain the unbiased PMF
# See Lindahl et al. 2014 https://doi.org/10.1063/1.4890371

mutable struct AWHPMFDeconvolution{N, T, F_CV}
    min_vals::NTuple{N, T}
    bin_widths::NTuple{N, T}
    shape::NTuple{N, Int}

    cv_function::F_CV         

    # Optimization: Pre-computed exp(-Q(xi, lambda)) = exp(-beta_k * V_bias(xi))
    # Rows: Linearized Bin Index, Cols: Window Index
    coupling_matrix::Matrix{T} 

    numerator_hist::Array{T, N}
    denominator_hist::Array{T, N}
    
    sample_count::Int

    target_beta::Union{T, Nothing}
    target_pressure::Union{T, Nothing}
    
    cv_history::Vector{NTuple{N, T}}
    active_idx_history::Vector{Int}

    scratch_g::Vector{T}        # For g(λ)
    scratch_weights::Vector{T}  # For exp(g)
    scratch_denom::Vector{T}    # For coupling_matrix * weights
end

function AWHPMFDeconvolution(
    awh_state::AWHState{T},
    pmf_grid::Tuple;
    cv_func = nothing,
    coupling_func = nothing,
    target_temp = nothing,
    target_pressure = nothing
) where T
    
    # Parse Grid Dimensions
    # pmf_grid format: ((min_1, min_2...), (max_1, max_2...), (bins_1, bins_2...))
    min_vals = T.(pmf_grid[1])
    max_vals = T.(pmf_grid[2])
    n_bins   = Int.(pmf_grid[3])
    
    N = length(n_bins)
    bin_widths = (max_vals .- min_vals) ./ n_bins
    
    # --- CV Function Setup ---
    local final_cv_func
    
    if !isnothing(cv_func)
        # User provided explicit CV function
        final_cv_func = cv_func
    else
        # Default: Auto-detect from BiasPotentials in the first window
        first_ham = awh_state.partition.λ_hamiltonians[1]
        bias_indices = findall(x -> x isa BiasPotential, first_ham.general_inters)
        
        if isempty(bias_indices)
            error("No BiasPotential found in AWHState. Cannot auto-detect PMF settings.")
        end
        if length(bias_indices) != N
            error("Found $(length(bias_indices)) BiasPotentials but grid is $(N)D.")
        end

        cv_types = [first_ham.general_inters[i].cv_type for i in bias_indices]
        
        # Capture awh_state for system info
        final_cv_func = (coords) -> begin
            sys = awh_state.active_sys
            return ntuple(N) do d
                Molly.calculate_cv(
                    cv_types[d], 
                    coords, 
                    sys.atoms, 
                    sys.boundary, 
                    sys.velocities
                )
            end
        end
    end

    # --- Coupling Matrix Setup ---
    total_bins = prod(n_bins)
    n_windows  = length(awh_state.partition.λ_hamiltonians)
    coupling_mat = Matrix{T}(undef, total_bins, n_windows)
    
    cart_indices = CartesianIndices(n_bins)
    linear_indices = LinearIndices(n_bins)

    # Pre-compute exp(-Q(xi, lambda))
    # If user provides coupling_func, it must return dimensionless energy (beta * E)
    if !isnothing(cv_func) && isnothing(coupling_func)
        error("If a custom `pmf_cv` is provided, `pmf_coupling` must also be provided.")
    end

    for (k, ham_k) in enumerate(awh_state.partition.λ_hamiltonians)
        beta_k = awh_state.λ_β[k]
        
        # If auto-detecting, find bias indices for this window once
        local bias_indices_k
        if isnothing(coupling_func)
             bias_indices_k = findall(x -> x isa BiasPotential, ham_k.general_inters)
        end

        for idx in cart_indices
            linear_i = linear_indices[idx]
            
            # Determine CV value at the center of this bin
            cv_val_tuple = ntuple(N) do d
                min_vals[d] + (idx[d] - 0.5) * bin_widths[d]
            end
            
            dim_less_bias = zero(T)

            if !isnothing(coupling_func)
                # User provided coupling: returns dimensionless energy directly
                dim_less_bias = coupling_func(cv_val_tuple, k)
            else
                # Auto-detect: Calculate physical energy and multiply by beta
                physical_bias_energy = zero(T) * awh_state.active_sys.energy_units
                
                for (d, bias_idx) in enumerate(bias_indices_k)
                    bias_inter = ham_k.general_inters[bias_idx]
                    physical_bias_energy += potential_energy(bias_inter.bias_type, cv_val_tuple[d])
                end
                
                if beta_k isa Quantity
                    dim_less_bias = ustrip(beta_k * physical_bias_energy)
                else
                    dim_less_bias = beta_k * ustrip(physical_bias_energy)
                end
            end
            coupling_mat[linear_i, k] = exp(-dim_less_bias)
        end
    end

    # [NEW] Setup Reweighting Constants
    sys = awh_state.active_sys
    e_unit = sys.energy_units
    
    tgt_beta = nothing
    if !isnothing(target_temp)
        # Convert kBT to system energy unit (kJ/mol) using R (Gas Constant)
        kBT_q = uconvert(e_unit, Unitful.R * target_temp)
        # Invert and Strip to get Beta in [1/energy_unit]
        tgt_beta = T(1.0 / ustrip(kBT_q))
    end
    
    tgt_press = nothing
    if !isnothing(target_pressure)
        l_unit = unit(sys.boundary.side_lengths[1])
        v_unit = l_unit^3
        p_unit = e_unit / v_unit
        
        # Deal with molar units
        e_val = 1.0 * e_unit
        molar_scaling = e_val / energy_remove_mol(e_val)
        
        # Scale macroscopic pressure to match the system's internal pressure dimensionality
        p_val_scaled = target_pressure * molar_scaling
        
        tgt_press = T(ustrip(uconvert(p_unit, p_val_scaled)))
    end

    num_hist = zeros(T, n_bins)
    den_hist = zeros(T, n_bins)

    n_wins = length(awh_state.λ_hamiltonians)
    n_bins_total = prod(n_bins)

    return AWHPMFDeconvolution(
        min_vals, 
        bin_widths, 
        n_bins,
        final_cv_func,
        coupling_mat,
        num_hist,
        den_hist,
        0,
        tgt_beta,
        tgt_press,
        Vector{NTuple{N, T}}(),
        Vector{Int}(),
        zeros(T, n_wins),
        zeros(T, n_wins),
        zeros(T, n_bins_total)
    )
end

function update_pmf!(
    pmf::AWHPMFDeconvolution{N, T, F_CV},
    awh_state, 
    curr_coords;
    weight_factor::T = one(T), 
    potential_energy::T = zero(T),
    box_volume::T = zero(T),
    current_beta::T = one(T),
    current_pressure::T = zero(T)
) where {N, T, F_CV}
    
    # Calculate current CV
    val = pmf.cv_function(from_device(curr_coords))
    current_cv = val isa Tuple ? val : (val,)

    # Convert to T to ensure type stability within the struct
    push!(pmf.cv_history, T.(current_cv))
    push!(pmf.active_idx_history, awh_state.active_idx)

    # Determine Index
    current_indices = ntuple(N) do d
        rel = (current_cv[d] - pmf.min_vals[d]) / pmf.bin_widths[d]
        idx = Int(floor(rel)) + 1
        clamp(idx, 1, pmf.shape[d])
    end
    current_cartesian = CartesianIndex(current_indices)
    current_linear_idx = LinearIndices(pmf.shape)[current_cartesian]

    # Compute Dynamic Weights (In-Place)
    # g(λ) = f(λ) + ln ρ(λ)
    @. pmf.scratch_g = awh_state.f + awh_state.log_rho
    @. pmf.scratch_weights = exp(pmf.scratch_g)
    
    # Global Update (In-Place Matrix Multiplication)
    # denom_vector = pmf.coupling_matrix * weights
    mul!(pmf.scratch_denom, pmf.coupling_matrix, pmf.scratch_weights)

    # Calculate Reweighting Factor W
    # W = exp( - [ (beta_tgt - beta_curr)U + (beta_tgt*P_tgt - beta_curr*P_curr)V ] )
    reweight_log = zero(T)
    
    if !isnothing(pmf.target_beta)
        reweight_log -= (pmf.target_beta - current_beta) * potential_energy
    end
    
    if !isnothing(pmf.target_pressure)
        target_work  = pmf.target_beta * pmf.target_pressure
        current_work = current_beta * current_pressure
        reweight_log -= (target_work - current_work) * box_volume
    end
    
    w_reweight = exp(reweight_log)
    
    # Iterate to update arrays using scratch_denom
    for i in eachindex(pmf.denominator_hist)
        term = weight_factor / pmf.scratch_denom[i]
        
        pmf.denominator_hist[i] += term
        
        if i == current_linear_idx
            pmf.numerator_hist[i] += term * w_reweight
        end
    end
    
    pmf.sample_count += 1
end

@doc raw"""
    calc_pmf(pmf_calc::AWHPMFDeconvolution)

Extracts the unbiased Potential of Mean Force (PMF) from the accumulated numerator 
and denominator histograms. Unsampled bins are assigned a value of `Inf`, and the 
global minimum of the valid PMF is shifted to zero.
"""
function calc_pmf(pmf_calc::AWHPMFDeconvolution{N, T, F_CV}) where {N, T, F_CV}
    num = pmf_calc.numerator_hist
    den = pmf_calc.denominator_hist
    
    # Identify bins with non-zero samples to avoid domain errors in log
    valid = (num .> zero(T)) .& (den .> zero(T))
    
    # Initialize PMF array with infinity for unsampled regions
    pmf = fill(T(Inf), size(num))
    
    # Calculate unbiased PMF for valid bins
    pmf[valid] .= -log.(num[valid] ./ den[valid])
    
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
    # Optional PMF args
    pmf_grid = nothing,
    pmf_cv = nothing,
    pmf_coupling = nothing,
    target_temp = nothing,
    target_pressure = nothing
) where T

    n_win = length(awh_state.partition.λ_hamiltonians)

    pmf_calc = nothing
    if !isnothing(pmf_grid)
        pmf_calc = AWHPMFDeconvolution(
            awh_state, 
            pmf_grid; 
            cv_func=pmf_cv, 
            coupling_func=pmf_coupling,
            target_temp=target_temp,
            target_pressure=target_pressure
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

# Swaps Hamiltionians
function update_active_sys!(awh_state::AWHState, active_idx::Int)
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
        
        # β is already converted to a raw float in correct units
        potentials[n] = awh.λ_β[n] * pe_val
    end

    # Calculate Z in-place
    @. awh.scratch_z = awh.log_rho + awh.f - potentials
    
    log_den = Molly.logsumexp(awh.scratch_z)
    
    # Calculate W directly into w_last
    @. awh.w_last = exp(awh.scratch_z - log_den)

    # Accumulate
    awh.w_seg .+= awh.w_last
    awh.n_accum += 1
    awh.N_eff   += 1

    # Check visited windows using w_last
    for (i, val) in enumerate(awh.w_last)
        if val > weight_relevance/n_states
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

function simulate!(awh_sim::AWHSimulation{T}, n_steps::Int) where T

    n_iterations = Int(floor(n_steps / awh_sim.n_md_steps))
    active_idx = 1

    for iteration_n in 1:n_iterations
        simulate!(awh_sim.state.active_sys, awh_sim.state.active_intg, awh_sim.n_md_steps)


        active_pe_units = process_sample(awh_sim.state)

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
            e_unit = sys.energy_units
            
            pot_eng = ustrip(e_unit, active_pe_units)
            
            vol_val = T(ustrip(volume(sys.boundary)))

            cur_beta = awh_sim.state.λ_β[awh_sim.state.active_idx]            
            cur_press = awh_sim.state.λ_p[awh_sim.state.active_idx]

            update_pmf!(
                awh_sim.pmf_calc, 
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
        if active_idx_new != active_idx
            active_idx = active_idx_new
        end

        update_active_sys!(awh_sim.state, active_idx)
        update_awh_bias!(awh_sim, iteration_n)
    end
end