export 
    AWHState,
    AWHSimulation,
    calc_pmf,
    extract_awh_data
    
# Convenience struct to store relevant things
# when running an AWH simulation.
mutable struct AWHStats{T}
    step_indices::Vector{Int}
    active_λ::Vector{Int}
    f_history::Vector{Vector{T}}
    n_effective_history::Vector{T}
    ess_history::Vector{Vector{T}}
    stage_history::Vector{Symbol}
    max_delta_f_history::Vector{T}
end

@inline host_view_for_cv!(::Nothing, arr) = arr
@inline function host_view_for_cv!(buffer, arr)
    copyto!(buffer, arr)
    return buffer
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
mutable struct AWHState{T, P, S, I, B, PR, SPI, SSI, SGI, SNF, SCN, SVS, SVF, SLG, SM, STM, SE}
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
    state_neighbor_finders::SNF
    state_constraints::SCN
    state_virtual_sites::SVS
    state_virtual_site_flags::SVF
    state_loggers::SLG
    state_masses::SM
    state_total_masses::STM
    state_dfs::Vector{Int}

    # Probability & Free Energy
    f::Vector{T}
    rho::Vector{T}
    log_rho::Vector{T}
   
    # Weight Accumulators
    w_seg::Vector{T}
    w2_seg::Vector{T}
    w_last::Vector{T}

    scratch_energies::SE
    scratch_potentials::Vector{T}
    scratch_z::Vector{T}
    
    N_eff::T
    N_bias::T          
    n_accum::Int
    inefficiency::T

    in_initial_stage::Bool
    visited_windows::Set{Int}

    stats::AWHStats{T}

    # Buffer for autocorrelation scaling
    cv_buffer::Vector{Vector{T}}
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
    # Keep as Vector{Any} so mixed integrator types can be switched safely.
    λ_integrators = Any[ts.integrator for ts in thermo_states]
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
    state_neighbor_finders = [deepcopy(ts.system.neighbor_finder) for ts in thermo_states]
    state_constraints = [ts.system.constraints for ts in thermo_states]
    state_virtual_sites = [ts.system.virtual_sites for ts in thermo_states]
    state_virtual_site_flags = [ts.system.virtual_site_flags for ts in thermo_states]
    state_loggers = [ts.system.loggers for ts in thermo_states]
    state_masses = [ts.system.masses for ts in thermo_states]
    state_total_masses = [ts.system.total_mass for ts in thermo_states]
    state_dfs = [ts.system.df for ts in thermo_states]

    active_sys = System(deepcopy(ref_sys);
        atoms = partition.λ_atoms[first_state],
        pairwise_inters = state_pairwise_inters[first_state],
        specific_inter_lists = state_specific_inter_lists[first_state],
        general_inters = state_general_inters[first_state],
        constraints = state_constraints[first_state],
        virtual_sites = state_virtual_sites[first_state],
        neighbor_finder = state_neighbor_finders[first_state],
        loggers = state_loggers[first_state]
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
    scratch_energies = [zero(partition.cached_master_pe) for _ in 1:n_λ]

    stats = AWHStats(Int[], Int[], Vector{FT}[], FT[], Vector{FT}[], Symbol[], FT[])

    return AWHState{FT, typeof(partition), System, Any, 
                    typeof(λ_β), typeof(λ_p), typeof(state_pairwise_inters),
                    typeof(state_specific_inter_lists), typeof(state_general_inters),
                    typeof(state_neighbor_finders), typeof(state_constraints),
                    typeof(state_virtual_sites), typeof(state_virtual_site_flags),
                    typeof(state_loggers), typeof(state_masses), typeof(state_total_masses),
                    typeof(scratch_energies)}(
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
        state_neighbor_finders,
        state_constraints,
        state_virtual_sites,
        state_virtual_site_flags,
        state_loggers,
        state_masses,
        state_total_masses,
        state_dfs,
        zeros(FT, n_λ),
        rho_val,
        log_ρ,
        zeros(FT, n_λ),
        zeros(FT, n_λ),
        zeros(FT, n_λ),
        scratch_energies,
        zeros(FT, n_λ),
        zeros(FT, n_λ),
        zero(FT),
        FT(n_bias),
        0,
        one(FT),
        true,
        Set{Int}(),
        stats,
        Vector{FT}[]
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
mutable struct AWHPMFDeconvolution{N, T, F_CV, MV, BW, SH, TP}
    min_vals::MV
    bin_widths::BW
    shape::SH
    is_periodic::NTuple{N, Bool} 
    cv_function::F_CV         

    numerator_hist::Array{T, N}
    denominator_hist::Array{T, N}
    denominator_w2_hist::Array{T, N}
    sample_count::Int

    target_beta::T
    target_pressure::TP
end

function AWHPMFDeconvolution(
    awh_state::AWHState{T},
    pmf_grid::Tuple;
    cv_func = nothing,
    is_periodic = nothing
) where T
    length(pmf_grid) == 3 || throw(ArgumentError(
        "`pmf_grid` must be a 3-tuple: (mins, maxs, bins)."
    ))

    min_vals = Tuple(pmf_grid[1])
    max_vals = Tuple(pmf_grid[2])
    n_bins   = Tuple(Int.(pmf_grid[3]))
    N = length(n_bins)
    length(min_vals) == N || throw(ArgumentError(
        "PMF mins length $(length(min_vals)) must match bins length $N."
    ))
    length(max_vals) == N || throw(ArgumentError(
        "PMF maxs length $(length(max_vals)) must match bins length $N."
    ))
    all(>(0), n_bins) || throw(ArgumentError("All PMF bin counts must be positive."))

    bin_widths = ntuple(N) do d
        width = (max_vals[d] - min_vals[d]) / n_bins[d]
        if !(width > zero(width))
            throw(ArgumentError(
                "PMF axis $d must satisfy max > min; got min=$(min_vals[d]), max=$(max_vals[d])."
            ))
        end
        width
    end

    periodic_flags = if isnothing(is_periodic)
        ntuple(_ -> false, N)
    else
        flags = Tuple(Bool.(is_periodic))
        length(flags) == N || throw(ArgumentError(
            "`pmf_is_periodic` length $(length(flags)) must match PMF dimensionality $N."
        ))
        flags
    end
    
    local final_cv_func
    coords_host_buffer = awh_state.active_sys.coords isa Array ? nothing : copy(from_device(awh_state.active_sys.coords))
    
    if !isnothing(cv_func)
        final_cv_func = (coords) -> begin
            coords_cv = host_view_for_cv!(coords_host_buffer, coords)
            return cv_func(coords_cv)
        end
    else
        first_ham = awh_state.partition.λ_hamiltonians[1]
        bias_indices = findall(x -> x isa BiasPotential, first_ham.general_inters)
        
        if isempty(bias_indices)
            error("No BiasPotential found in AWHState. Cannot auto-detect PMF settings.")
        end

        cv_types = [first_ham.general_inters[i].cv_type for i in bias_indices]
        length(cv_types) == N || throw(ArgumentError(
            "Auto-detected $(length(cv_types)) bias CVs but PMF dimensionality is $N. " *
            "Provide a matching `pmf_grid`, or pass `pmf_cv` explicitly."
        ))
        state_atoms_host = [from_device(atoms) for atoms in awh_state.partition.λ_atoms]
        velocities_host_buffer = awh_state.active_sys.velocities isa Array ? nothing : copy(from_device(awh_state.active_sys.velocities))
        
        final_cv_func = (coords) -> begin
            sys = awh_state.active_sys
            coords_cv = host_view_for_cv!(coords_host_buffer, coords)
            velocities_cv = host_view_for_cv!(velocities_host_buffer, sys.velocities)
            atoms_cv = state_atoms_host[awh_state.active_idx]
            return ntuple(N) do d
                Molly.calculate_cv(
                    cv_types[d],
                    coords_cv,
                    atoms_cv,
                    sys.boundary,
                    velocities_cv,
                )
            end
        end
    end

    sample_val = final_cv_func(awh_state.active_sys.coords)
    sample_tuple = sample_val isa Tuple ? sample_val : (sample_val,)
    length(sample_tuple) == N || throw(ArgumentError(
        "`pmf_cv` returned $(length(sample_tuple)) values but PMF dimensionality is $N."
    ))

    return AWHPMFDeconvolution(
        min_vals, bin_widths, n_bins, periodic_flags, final_cv_func,
        zeros(T, n_bins), zeros(T, n_bins), zeros(T, n_bins), 0, awh_state.target_β, awh_state.target_p
    )
end

function update_pmf!(
    pmf::AWHPMFDeconvolution{N, T},
    awh_state, 
    curr_coords;
    weight_factor::T = one(T), 
    box_volume::T = zero(T),
    apply_forgetting::Bool = true
) where {N, T}
    
    val = pmf.cv_function(curr_coords)
    current_cv = val isa Tuple ? val : (val,)
    length(current_cv) == N || throw(ArgumentError(
        "PMF CV function returned $(length(current_cv)) values, expected $N."
    ))

    # Determine Index with conditional periodic wrapping
    current_indices = ntuple(N) do d
        rel = (current_cv[d] - pmf.min_vals[d]) / pmf.bin_widths[d]
        rel_val = ustrip(rel)
        isfinite(rel_val) || throw(ArgumentError("Non-finite PMF CV value encountered on axis $d."))
        idx = Int(floor(rel_val)) + 1
        
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
    if !isfinite(log_W_mix)
        return nothing
    end
    
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
    if !isfinite(unbias_log)
        return nothing
    end
    w_frame = exp(unbias_log)
    if !isfinite(w_frame)
        return nothing
    end
    
    # Scale by statistical inefficiency to account for temporal correlations
    ineff = awh_state.inefficiency
    w_frame_eff = w_frame / ineff
    count_eff = 1 / ineff
    w2_eff = 1 / (ineff^2)
    
    # 4. Exponential Forgetting & Accumulation
    if apply_forgetting && weight_factor < one(T)
        pmf.numerator_hist .*= weight_factor
        pmf.denominator_hist .*= weight_factor
        pmf.denominator_w2_hist .*= weight_factor^2
    end
    
    pmf.numerator_hist[current_linear_idx] += w_frame_eff
    pmf.denominator_hist[current_linear_idx] += count_eff
    pmf.denominator_w2_hist[current_linear_idx] += w2_eff
    pmf.sample_count += 1
end

@doc raw"""
    calc_pmf(pmf_calc::AWHPMFDeconvolution)

Extracts the unbiased Potential of Mean Force (PMF) from the accumulated numerator 
histograms. Unsampled bins are assigned a value of `Inf`, and the 
global minimum of the valid PMF is shifted to zero.
"""
function calc_pmf(pmf_calc::AWHPMFDeconvolution{N, T}) where {N, T}
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
- `coverage_type::Symbol=:reweighted`: Coverage accounting mode. Use `:reweighted` for coordinate-based
    AWH and `:physical` for strictly physical (active-window-only) tracking.
- `log_freq::Int=100`: Number of AWH iterations between logging statistics.
- `pmf_grid=nothing`: Tuple of tuples defining the PMF grid `((min_1, ...), (max_1, ...), (bins_1, ...))`.
    Required if running with PMF deconvolution.
- `pmf_cv=nothing`: Function taking system coordinates and returning a tuple of scalar Collective 
    Variables (CVs). If omitted when `pmf_grid` is provided, Molly attempts to auto-detect CVs from 
    the active `BiasPotential`s.
- `target_state` in [`AWHState`](@ref): Required when using PMF deconvolution (`pmf_grid`) and
    defines the exact unbiased thermodynamic state used for MBAR frame reweighting.
"""
struct AWHSimulation{T}
    n_windows::Int
    initial_sampl_n::T
    n_md_steps::Int 
    update_freq::Int        
    well_tempered_fac::T    
    coverage_threshold::T   
    significant_weight::T   
    coverage_type::Symbol
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
    coverage_type::Symbol = :reweighted,
    log_freq::Int = 100,
    pmf_grid = nothing,
    pmf_cv = nothing,
    pmf_is_periodic = nothing
) where T

    num_md_steps > 0 || throw(ArgumentError("`num_md_steps` must be positive, got $num_md_steps."))
    update_freq > 0 || throw(ArgumentError("`update_freq` must be positive, got $update_freq."))
    log_freq > 0 || throw(ArgumentError("`log_freq` must be positive, got $log_freq."))
    0 < coverage_threshold <= 1 || throw(ArgumentError(
        "`coverage_threshold` must be in (0, 1], got $coverage_threshold."
    ))
    significant_weight >= 0 || throw(ArgumentError(
        "`significant_weight` must be non-negative, got $significant_weight."
    ))
    (isinf(well_tempered_factor) || well_tempered_factor > 0) || throw(ArgumentError(
        "`well_tempered_factor` must be positive or `Inf`, got $well_tempered_factor."
    ))
    coverage_type in (:reweighted, :physical) || throw(ArgumentError(
        "`coverage_type` must be either `:reweighted` or `:physical`, got $coverage_type."
    ))

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
                         coverage_type,
                         log_freq, awh_state,
                         pmf_calc)
end

# Fast-path check: if all target state fields are type-compatible with the
# current `active_sys`, update in place; otherwise rebuild `active_sys`.
function can_update_active_sys_inplace(awh_state::AWHState, active_idx::Int)
    sys = awh_state.active_sys
    return awh_state.partition.λ_atoms[active_idx] isa typeof(sys.atoms) &&
           awh_state.state_pairwise_inters[active_idx] isa typeof(sys.pairwise_inters) &&
           awh_state.state_specific_inter_lists[active_idx] isa typeof(sys.specific_inter_lists) &&
           awh_state.state_general_inters[active_idx] isa typeof(sys.general_inters) &&
           awh_state.state_constraints[active_idx] isa typeof(sys.constraints) &&
           awh_state.state_virtual_sites[active_idx] isa typeof(sys.virtual_sites) &&
           awh_state.state_virtual_site_flags[active_idx] isa typeof(sys.virtual_site_flags) &&
           awh_state.state_neighbor_finders[active_idx] isa typeof(sys.neighbor_finder) &&
           awh_state.state_loggers[active_idx] isa typeof(sys.loggers) &&
           awh_state.state_masses[active_idx] isa typeof(sys.masses) &&
           awh_state.state_total_masses[active_idx] isa typeof(sys.total_mass)
end

# Swaps Hamiltonians and enforces kinetic energy continuity across temperature jumps and mass changes
function update_active_sys!(awh_state::AWHState, active_idx::Int)
    old_idx = awh_state.active_idx
    
    # Perform instantaneous velocity rescaling if the temperature target or atomic masses change
    if old_idx != active_idx
        β_old = awh_state.λ_β[old_idx]
        β_new = awh_state.λ_β[active_idx]
        
        old_masses = from_device(awh_state.state_masses[old_idx])
        new_masses = from_device(awh_state.state_masses[active_idx])
        temps_or_masses_change = (β_old != β_new)

        if !temps_or_masses_change
            for i in eachindex(old_masses, new_masses)
                if old_masses[i] != new_masses[i]
                    temps_or_masses_change = true
                    break
                end
            end
        end

        if temps_or_masses_change
            vels_cpu = copy(from_device(awh_state.active_sys.velocities))

            for i in eachindex(vels_cpu, old_masses, new_masses)
                m_old = old_masses[i]
                m_new = new_masses[i]

                if β_old != β_new || m_old != m_new
                    # Zero-mass particles have no kinetic contribution; avoid 0/0
                    # when swapping between temperatures or mass-scaled states.
                    if iszero(m_old) || iszero(m_new)
                        continue
                    end
                    # ustrip ensures the resulting scalar is a raw float, preventing Unitful type instability
                    velocity_scaling_factor = sqrt(ustrip((β_old * m_old) / (β_new * m_new)))
                    vels_cpu[i] = vels_cpu[i] * velocity_scaling_factor
                end
            end

            awh_state.active_sys.velocities .= to_device(vels_cpu, array_type(awh_state.active_sys))
        end
    end

    awh_state.active_idx = active_idx
    if can_update_active_sys_inplace(awh_state, active_idx)
        awh_state.active_sys.atoms = awh_state.partition.λ_atoms[active_idx]
        awh_state.active_sys.pairwise_inters = awh_state.state_pairwise_inters[active_idx]
        awh_state.active_sys.specific_inter_lists = awh_state.state_specific_inter_lists[active_idx]
        awh_state.active_sys.general_inters = awh_state.state_general_inters[active_idx]
        awh_state.active_sys.constraints = awh_state.state_constraints[active_idx]
        awh_state.active_sys.virtual_sites = awh_state.state_virtual_sites[active_idx]
        awh_state.active_sys.virtual_site_flags = awh_state.state_virtual_site_flags[active_idx]
        awh_state.active_sys.neighbor_finder = awh_state.state_neighbor_finders[active_idx]
        awh_state.active_sys.loggers = awh_state.state_loggers[active_idx]
        awh_state.active_sys.masses = awh_state.state_masses[active_idx]
        awh_state.active_sys.total_mass = awh_state.state_total_masses[active_idx]
        awh_state.active_sys.df = awh_state.state_dfs[active_idx]
    else
        sys = awh_state.active_sys
        awh_state.active_sys = System(
            sys;
            atoms = awh_state.partition.λ_atoms[active_idx],
            coords = sys.coords,
            boundary = sys.boundary,
            velocities = sys.velocities,
            pairwise_inters = awh_state.state_pairwise_inters[active_idx],
            specific_inter_lists = awh_state.state_specific_inter_lists[active_idx],
            general_inters = awh_state.state_general_inters[active_idx],
            constraints = awh_state.state_constraints[active_idx],
            virtual_sites = awh_state.state_virtual_sites[active_idx],
            neighbor_finder = awh_state.state_neighbor_finders[active_idx],
            loggers = awh_state.state_loggers[active_idx],
            strictness = :nowarn,
        )
        # Keep per-state cached values exactly as provided by ThermoState.
        awh_state.active_sys.virtual_site_flags = awh_state.state_virtual_site_flags[active_idx]
        awh_state.active_sys.masses = awh_state.state_masses[active_idx]
        awh_state.active_sys.total_mass = awh_state.state_total_masses[active_idx]
        awh_state.active_sys.df = awh_state.state_dfs[active_idx]
    end
    awh_state.active_intg = awh_state.λ_integrators[active_idx]
end

# Reweights coordinates along λ windows and accumulates histogram
function process_sample(
    awh::AWHState{FT};
    weight_relevance::Real = 0.1,
    coverage_type::Symbol = :reweighted,
    pmf_calc::Union{AWHPMFDeconvolution, Nothing} = nothing,
) where FT
    n_states = length(awh.λ_β)
    coords = awh.active_sys.coords
    bound  = awh.active_sys.boundary

    # Exploit the AlchemicalPartition abstraction
    energies = evaluate_energy_all!(awh.partition, coords, bound, awh.scratch_energies)
    
    # Extract volume in consistent units
    vol_val = FT(ustrip(volume(bound)))
    
    potentials = awh.scratch_potentials 
    active_pe = energies[awh.active_idx]

    for n in 1:n_states
        pe = energies[n]
        
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
    awh.w2_seg .+= awh.w_last .^ 2
    awh.n_accum += 1

    # Buffer CV for autocorrelation calculation
    val = if !isnothing(pmf_calc)
        pmf_calc.cv_function(coords)
    else
        # Fallback to active index if no CV is defined (alchemical mode)
        awh.active_idx
    end
    cv_val = val isa Tuple ? [ustrip(v) for v in val] : [ustrip(val)]
    push!(awh.cv_buffer, cv_val)

    if coverage_type == :reweighted
        # Coordinate-style AWH: one frame may validly contribute to neighboring bins.
        for (i, val) in enumerate(awh.w_last)
            if val > weight_relevance * awh.rho[i]
                push!(awh.visited_windows, i)
            end
        end
    elseif coverage_type == :physical
        # Alchemical-style AWH: only the physically propagated window counts as visited.
        push!(awh.visited_windows, awh.active_idx)
    else
        throw(ArgumentError(
            "`coverage_type` must be either `:reweighted` or `:physical`, got $coverage_type."
        ))
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
    
    # Calculate Kish ESS per window for the current segment
    ess = zeros(eltype(state.w_seg), length(state.w_seg))
    for i in eachindex(ess)
        if state.w2_seg[i] > 0
            ess[i] = (state.w_seg[i]^2) / state.w2_seg[i]
        end
    end

    push!(stats.step_indices, step_idx)
    push!(stats.active_λ, state.active_idx)
    push!(stats.f_history, copy(state.f))
    push!(stats.n_effective_history, current_N)
    push!(stats.ess_history, ess)
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

    # Use the current best estimate of inefficiency to determine effective weights
    ineff = awh_sim.state.inefficiency
    eff_n_accum = awh_sim.state.n_accum / ineff
    eff_w_seg = awh_sim.state.w_seg ./ ineff

    current_N = if awh_sim.state.in_initial_stage
        awh_sim.state.N_bias
    else
        # In the linear stage, use the accumulated effective sample size
        # before the current update block is folded in.
        awh_sim.initial_sampl_n + (awh_sim.state.N_eff - eff_n_accum)
    end

    numerator   = current_N .* awh_sim.state.rho .+ eff_w_seg
    denominator = current_N .* awh_sim.state.rho .+ (eff_n_accum .* awh_sim.state.rho)
    
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
        required_cov = max(1, ceil(Int, awh_sim.coverage_threshold * n_target_windows))
        if cov_count >= required_cov
            awh_sim.state.N_bias *= 2
            empty!(awh_sim.state.visited_windows)
            if awh_sim.state.N_bias >= (awh_sim.initial_sampl_n + awh_sim.state.N_eff)
                awh_sim.state.in_initial_stage = false
            end
        end
    end

    fill!(awh_sim.state.w_seg, 0)
    fill!(awh_sim.state.w2_seg, 0)
    awh_sim.state.n_accum = 0
    
    return delta_f
end

function simulate!(awh_sim::AWHSimulation{T}, n_steps::Int; convergence_threshold=nothing, max_lag::Int=10) where T
    n_steps >= 0 || throw(ArgumentError("`n_steps` must be non-negative, got $n_steps."))
    max_lag > 0 || throw(ArgumentError("`max_lag` must be positive, got $max_lag."))

    n_iterations = fld(n_steps, awh_sim.n_md_steps)
    remaining_steps = n_steps - n_iterations * awh_sim.n_md_steps
    active_idx = awh_sim.state.active_idx
    converged = false

    df_hist = T[]

    for iteration_n in 1:n_iterations
        # --- NEW: Sync active index to the logger ---
        if hasproperty(awh_sim.state.active_sys.loggers, :awh_logger)
            awh_sim.state.active_sys.loggers.awh_logger.active_idx = active_idx
        end

        simulate!(awh_sim.state.active_sys, awh_sim.state.active_intg, awh_sim.n_md_steps)


        process_sample(
            awh_sim.state;
            weight_relevance=awh_sim.significant_weight,
            coverage_type=awh_sim.coverage_type,
            pmf_calc=awh_sim.pmf_calc,
        )

        # Increment N_eff based on the current best estimate of inefficiency
        # This ensures that N_eff grows even when cv_buffer is not yet full.
        awh_sim.state.N_eff += (1 / awh_sim.state.inefficiency)

        # Periodically update the statistical inefficiency estimate using CV history
        if length(awh_sim.state.cv_buffer) >= 50
            # Use the max inefficiency across all CV dimensions as a conservative scaling factor
            cv_data = awh_sim.state.cv_buffer
            n_cv = length(cv_data[1])
            
            new_ineff = 1.0
            for d in 1:n_cv
                cv_series = [v[d] for v in cv_data]
                d_ineff = statistical_inefficiency(cv_series).inefficiency
                new_ineff = max(new_ineff, d_ineff)
            end
            
            # Correct N_eff for the last 50 steps using the improved inefficiency estimate
            awh_sim.state.N_eff -= (50 / awh_sim.state.inefficiency)
            awh_sim.state.N_eff += (50 / new_ineff)
            
            awh_sim.state.inefficiency = new_ineff
            empty!(awh_sim.state.cv_buffer)
        end

        if !isnothing(awh_sim.pmf_calc)
            # PMF update logic
            w_fac = one(T)
            if awh_sim.state.in_initial_stage
                current_N = awh_sim.state.N_bias
                # Use effective segment length for forgetting scaling
                eff_n_lambda  = T(awh_sim.update_freq) / awh_sim.state.inefficiency
                w_fac = current_N / (current_N + eff_n_lambda)
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
        
        delta_f = update_awh_bias!(awh_sim, iteration_n)

        # --- NEW: Convergence Evaluation ---
        if !isnothing(convergence_threshold) && !isnothing(delta_f)
            # Strict enforcement: only evaluate convergence during the linear stage
            if !awh_sim.state.in_initial_stage
                max_change = maximum(abs.(delta_f))
                push!(df_hist, max_change)
                if length(df_hist) > max_lag
                    popfirst!(df_hist)
                end
                nonzero_changes = filter(x -> x != zero(T), df_hist)
                mean_change = isempty(nonzero_changes) ? zero(T) : mean(nonzero_changes)
                if mean_change <= convergence_threshold
                    @info "AWH converged at iteration $iteration_n (max ΔF = $mean_change <= $convergence_threshold)"
                    converged = true
                    break
                end
            end
        end
    end

    # Preserve the exact total number of requested MD steps unless converged early.
    if !converged && remaining_steps > 0
        simulate!(awh_sim.state.active_sys, awh_sim.state.active_intg, remaining_steps)
    end

    return awh_sim
end

@doc raw"""
    extract_awh_data(awh_sim::AWHSimulation)

Extracts the converged free energy profile (f), target distribution (rho), 
log target distribution (log_rho), and statistics from an AWH simulation.
Returns a NamedTuple containing deepcopied arrays to ensure the reference 
data is preserved independently of the ongoing simulation state.
"""
function extract_awh_data(awh_sim::AWHSimulation)
    return (
        f = copy(awh_sim.state.f),
        rho = copy(awh_sim.state.rho),
        log_rho = copy(awh_sim.state.log_rho),
        ess_history = copy(awh_sim.state.stats.ess_history),
        stats = deepcopy(awh_sim.state.stats)
    )
end
