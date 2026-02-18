export 
    AWHState,
    AWHSimulation,
    simulate!
    
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

# Convenience function that builds neighbor finders.
# This is needed to be able to properly separate 
# alchemical atoms from normal atoms
function build_neighbor_finder(ref_nfinder, eligible; 
                               reuse_neighbors::Bool = true)

    if ref_nfinder isa DistanceNeighborFinder
        nf = DistanceNeighborFinder(
            eligible = eligible, 
            dist_cutoff = ref_nfinder.dist_cutoff,
            special   = ref_nfinder.special,
            n_steps = 1,
            neighbors = ref_nfinder.neighbors
        )
    elseif ref_nfinder isa CellListMapNeighborFinder
        nf = CellListMapNeighborFinder(
            eligible = eligible,
            dist_cutoff = ref_nfinder.dist_cutoff,
            special = ref_nfinder.special,
            n_steps = 1,
            x0 = ref_nfinder.x0,
            unit_cell = ref_nfinder.unit_cell,
            dims = ref_nfinder.dims
        )
    elseif ref_nfinder isa GPUNeighborFinder
        if !reuse_neighbors
            nf = GPUNeighborFinder(
                eligible = eligible,
                dist_cutoff = ref_nfinder.dist_cutoff,
                special = ref_nfinder.special,
                n_steps_reorder = 1,
                initialized = ref_nfinder.initialized
            )
        else
            nf = DistanceNeighborFinder(
                eligible = eligible, 
                dist_cutoff = ref_nfinder.dist_cutoff,
                special   = ref_nfinder.special,
                n_steps = 1
            )
        end
    elseif ref_nfinder isa TreeNeighborFinder
        nf = TreeNeighborFinder(
            eligible = eligible,
            dist_cutoff = ref_nfinder.dist_cutoff,
            special = ref_nfinder.special,
            n_steps = 1
        )
    elseif ref_nfinder isa NoNeighborFinder
        nf = NoNeighborFinder()
    else
        @warn "Unknown NeighborFinder type $(typeof(ref_nfinder)). Using default DistanceNeighborFinder for Master System."
        nf = DistanceNeighborFinder(
            eligible = eligible,
            dist_cutoff = 1.0u"nm",
            n_steps = 1
        )
    end

    return nf
end

# Convenience struct that stores the interactions
# lists that change along the RC when performing
# AWH. Allows efficient swap of Hamiltonians
struct LambdaHamiltonian{PI, SI, GI}
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
end

"""
    AWHState(thermo_states::AbstractArray{ThermoState};
             first_state::Int                    = 1,
             n_bias::Int                         = 100,
             ρ::Union{Nothing, AbstractArray{T}} = nothing,
             reuse_neighbors::Bool               = true)

Represents the state of an AWH simulation.

# Arguments
- `thermo_states::AbstractArray{ThermoState}`: Iterable that carries the [`ThermoState`](@ref) structs that are used
    as the different λ states used when running AWH.
- `first_state::Int = 1`: The index of the [`ThermoState`](@ref) that will be used as the starting point of the AWH
    simulation
- `n_bias::Int = 100`: Fictitious effective sampling size to be used during the initial stage of the AWH simulation.
    Smaller values imply a more aggressive sampling during this stage.
- `ρ::Union{Nothing, AbstractArray{T}} = Nothing`: Target distribution along λ. If `nothing` is passed, defaults to 
    uniform distribution.
- `reuse_neighbors::Bool = true`: Wether to reuse the same neighbor finder for the reweighting step for a given sampled
    conformation. Usually improves performance.
"""
mutable struct AWHState{T}
    # Master System (Common interactions, e.g., Solvent-Solvent)
    master_sys::System
    # λ System (Unique interactions per λ window)
    λ_sys::System

    active_idx::Int

    # Active System, contains all interactions to run actual MD simulation
    active_sys::System

    # Active integrator
    active_intg

    # λ Integrators, each window carries its own integrator to account for different temperatures / pressures
    λ_integrators::Vector

    # λ Inverse temperatures
    λ_β::Vector

    # λ Target Pressures (Precomputed in internal units)
    λ_p::Vector

    # Per λ interactions
    λ_hamiltonians::Vector{LambdaHamiltonian}
    # Per λ atoms
    λ_atoms::Vector

    # Probability & Free Energy
    f::Vector{T}            # Free energy of each window
    rho::Vector{T}          # Target distribution
    log_rho::Vector{T}      # Log of target distribution (precomputed)
    
    # Weight Accumulators
    w_seg::Vector{T}        # For current update segment
    w_last::Vector{T}       # For Gibbs sampling
    
    # Dynamics Variables
    N_eff::T                # Real effective samples
    N_bias::T               # Artificial bias samples for the initial stage
    n_accum::Int            # Samples in current segment

    # State Flags
    in_initial_stage::Bool    # Keeps track if we are in initial or linear Δf scaling stage
    visited_windows::Set{Int} # Keeps track of λ windows representative of sampled frames 

    # Stats
    stats::AWHStats
end

function AWHState(thermo_states::AbstractArray{ThermoState};
                  first_state::Int = 1,
                  n_bias::Int = 100,
                  ρ::Union{Nothing, AbstractArray{T}} = nothing,
                  reuse_neighbors::Bool = true) where T

    n_λ = length(thermo_states)
    ref_sys = thermo_states[first_state].system

    FT = typeof(ustrip(ref_sys.total_mass))

    # 1. Identify Global Solute Indices
    solute_indices = Set{Int}()
    
    λ_atoms = []
    λ_integrators = []
    λ_β = []
    λ_p = [] # New pressure vector

    # Pre-fetch units for pressure conversion
    e_unit = ref_sys.energy_units
    l_unit = unit(ref_sys.boundary.side_lengths[1])
    p_unit = e_unit / (l_unit^3)

    for tstate in thermo_states
        atoms = tstate.system.atoms
        intg  = tstate.integrator
        push!(λ_atoms, atoms)
        push!(λ_integrators, intg)
        push!(λ_β, tstate.beta)
        
        # Precompute Pressure
        p_val = zero(FT)
        if hasfield(typeof(intg), :coupling)
            couplers = intg.coupling isa Tuple ? intg.coupling : (intg.coupling,)
            for c in couplers
                if hasfield(typeof(c), :pressure)
                    # Convert P_bar to P_molar (internal E/V)
                    # P_molar = P_bar * Na
                    p_molar = FT(1/3 * tr(c.pressure)) * Unitful.Na
                    p_val = FT(ustrip(uconvert(p_unit, p_molar)))
                end
            end
        end
        push!(λ_p, p_val)

        atoms_cpu = from_device(atoms)
        for atom in atoms_cpu
            if atom.λ < 1.0
                push!(solute_indices, atom.index)
            end
        end
    end

    active_intg = λ_integrators[first_state]
    
    # 2. Partition Neighbor Lists
    ref_nfinder = ref_sys.neighbor_finder
    base_eligible_cpu = copy(from_device(ref_nfinder.eligible))
    n_atoms = size(base_eligible_cpu, 1)
    
    master_eligible_cpu = copy(base_eligible_cpu)
    for idx in solute_indices
        master_eligible_cpu[idx, :] .= false
        master_eligible_cpu[:, idx] .= false
    end
    
    specific_eligible_cpu = copy(base_eligible_cpu)
    solvent_indices = [i for i in 1:n_atoms if !(i in solute_indices)]
    
    if !isempty(solvent_indices)
        specific_eligible_cpu[solvent_indices, solvent_indices] .= false
    end

    AT = array_type(ref_sys)
    master_eligible = to_device(master_eligible_cpu, AT)
    λ_eligible      = to_device(specific_eligible_cpu, AT)
    
    # 3. Extract and Partition Interaction Lists
    list_1a = [Vector{InteractionList1Atoms}() for _ in 1:n_λ]
    list_2a = [Vector{InteractionList2Atoms}() for _ in 1:n_λ]
    list_3a = [Vector{InteractionList3Atoms}() for _ in 1:n_λ]
    list_4a = [Vector{InteractionList4Atoms}() for _ in 1:n_λ]

    @inbounds for (i, tstate) in enumerate(thermo_states)
        sils = tstate.system.specific_inter_lists
        for inter in sils
            if inter isa InteractionList1Atoms push!(list_1a[i], inter)
            elseif inter isa InteractionList2Atoms push!(list_2a[i], inter)
            elseif inter isa InteractionList3Atoms push!(list_3a[i], inter)
            elseif inter isa InteractionList4Atoms push!(list_4a[i], inter)
            end
        end
    end

    all_gils = [collect(thermo_states[n].system.general_inters) for n in 1:n_λ]
    all_pils = [collect(thermo_states[n].system.pairwise_inters) for n in 1:n_λ]

    master_sils_1a = intersect(list_1a...)
    master_sils_2a = intersect(list_2a...)
    master_sils_3a = intersect(list_3a...)
    master_sils_4a = intersect(list_4a...)
    master_gils    = intersect(all_gils...)
    master_pils    = intersect(all_pils...)

    λ_specific = Vector{Tuple}(undef, n_λ)
    λ_general  = Vector{Tuple}(undef, n_λ)
    λ_pairwise = Vector{Tuple}(undef, n_λ)

    for i in 1:n_λ
        u_1 = setdiff(list_1a[i], master_sils_1a)
        u_2 = setdiff(list_2a[i], master_sils_2a)
        u_3 = setdiff(list_3a[i], master_sils_3a)
        u_4 = setdiff(list_4a[i], master_sils_4a)
        λ_specific[i] = (u_1..., u_2..., u_3..., u_4...)
        
        u_g = setdiff(all_gils[i], master_gils)
        λ_general[i] = (u_g...,)
        λ_pairwise[i] = (all_pils[i]...,)
    end

    # 4. Construct Master System
    master_nf = build_neighbor_finder(ref_nfinder, master_eligible; reuse_neighbors = reuse_neighbors)
    λ_nf      = build_neighbor_finder(ref_nfinder, λ_eligible; reuse_neighbors = reuse_neighbors)

    active_sys = System(deepcopy(ref_sys))

    master_sys = System(deepcopy(ref_sys); 
        pairwise_inters      = (master_pils...,),
        general_inters       = (master_gils...,),
        specific_inter_lists = (master_sils_1a..., 
                                master_sils_2a..., 
                                master_sils_3a..., 
                                master_sils_4a...),
        neighbor_finder      = master_nf
    )

    λ_sys = System(deepcopy(ref_sys); 
        pairwise_inters      = (λ_pairwise[1]...,),
        general_inters       = (λ_general[1]...,),
        specific_inter_lists = (λ_specific[1]...,),
        neighbor_finder      = λ_nf
    )

    # FT = typeof(ustrip(master_sys.total_mass))
    hamiltonians = LambdaHamiltonian[]
    for (λ_p, λ_s, λ_g) in zip(λ_pairwise, λ_specific, λ_general)
        ham = LambdaHamiltonian(λ_p, λ_s, λ_g)
        push!(hamiltonians, ham)
    end

    # 5. Handle Target Distribution (ρ)
    if isnothing(ρ)
        rho_val = fill(FT(1/n_λ), n_λ)
    else
        if eltype(ρ) != FT
             rho_val = FT.(ρ) 
        else
             rho_val = ρ
        end
    end
    log_ρ = log.(rho_val)

    stats = AWHStats(
        Int[],              # step_indices
        Int[],             # Active λ
        Vector{FT}[],        # f_history
        FT[],                # n_effective_history
        Symbol[],           # stage_history
        FT[]                 # max_delta_f_history
    )

    return AWHState(
        master_sys,
        λ_sys,
        first_state,
        active_sys,
        active_intg,
        λ_integrators,
        λ_β,
        λ_p,
        hamiltonians,
        λ_atoms,        
        zeros(FT, n_λ),    # f
        rho_val,           # rho
        log_ρ,             # log_rho
        zeros(FT, n_λ),    # w_seg
        zeros(FT, n_λ),    # w_last
        zero(FT),          # N_eff
        FT(n_bias),        # N_bias
        0,                 # n_accum
        true,              # in_initial_stage
        Set{Int}(),        # visited_windows
        stats,
    )
end

# Implements the deconvolution method to obtain the unbiased PMF
# See Lindahl et al. 2014
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
        first_ham = awh_state.λ_hamiltonians[1]
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
    n_windows  = length(awh_state.λ_hamiltonians)
    coupling_mat = Matrix{T}(undef, total_bins, n_windows)
    
    cart_indices = CartesianIndices(n_bins)
    linear_indices = LinearIndices(n_bins)

    # Pre-compute exp(-Q(xi, lambda))
    # If user provides coupling_func, it must return dimensionless energy (beta * E)
    if !isnothing(cv_func) && isnothing(coupling_func)
        error("If a custom `pmf_cv` is provided, `pmf_coupling` must also be provided.")
    end

    for (k, ham_k) in enumerate(awh_state.λ_hamiltonians)
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
                    physical_bias_energy += Molly.potential_energy(bias_inter.bias_type, cv_val_tuple[d])
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
        # FIXED: DimensionError fix
        # Molly Energies are per mole. P*V is just energy.
        # We need (P*V) to have units of J/mol.
        # P (bar) * Na (mol^-1) gives the correct dimensionality to match e_unit/v_unit.
        
        l_unit = unit(sys.boundary.side_lengths[1])
        v_unit = l_unit^3
        
        # Internal pressure unit: Energy / Volume (per mole)
        p_unit = e_unit / v_unit # e.g. kJ mol^-1 nm^-3
        
        # We scale the macroscopic pressure by Avogadro's constant to get "Molar Pressure"
        p_val_molar = target_pressure * Unitful.Na
        
        tgt_press = T(ustrip(uconvert(p_unit, p_val_molar)))
    end

    num_hist = zeros(T, n_bins)
    den_hist = zeros(T, n_bins)

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
        tgt_press
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

    # Determine Index
    current_indices = ntuple(N) do d
        rel = (current_cv[d] - pmf.min_vals[d]) / pmf.bin_widths[d]
        idx = Int(floor(rel)) + 1
        clamp(idx, 1, pmf.shape[d])
    end
    current_cartesian = CartesianIndex(current_indices)
    current_linear_idx = LinearIndices(pmf.shape)[current_cartesian]

    # Compute Dynamic Weights
    # g(λ) = f(λ) + ln ρ(λ)
    g = awh_state.f .+ awh_state.log_rho
    weights = exp.(g) 
    
    # Global Update
    # denom_vector[i] corresponds to e^{-gamma(xi)}
    denom_vector = pmf.coupling_matrix * weights

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
    
    # Iterate to update arrays
    for i in eachindex(pmf.denominator_hist)
        # Apply the a(t) factor here. 
        # term = a(t) * e^{gamma(xi)}
        term = weight_factor / denom_vector[i]
        
        # Update Denominator
        pmf.denominator_hist[i] += term
        
        # Update Numerator
        if i == current_linear_idx
            pmf.numerator_hist[i] += term * w_reweight
        end
    end
    
    pmf.sample_count += 1
end

"""
    AWHSimulation(awh_state::AWHState{T};
                  num_md_steps::Int          = 10,
                  update_freq::Int           = 1,
                  well_tempered_factor::Real = 10.0,
                  coverage_threshold::Real   = 1.0,
                  significant_weight::Real   = 0.1,
                  log_freq::Int              = 100,
                  pmf_cv                     = nothing,
                  pmf_coupling               = nothing,
                  pmf_grid                   = nothing)

Prepares and runs an AWH simulation. This can be run with or without the deconvolution
method described in [Lindahl et al.](https://doi.org/10.1063/1.4890371), in order to
obtain an unbiased estimator og the true PMF. 
For simulations where the λ coordinate is a biased reaction coordinate, e.g., using an 
umbrella potential, the deconvolution approach is preferred. Alchemical transformations 
do not require this method.

# Arguments
- `awh_state::AWHState{T}`: The [`AWHState`](@ref) defined to carry the necessary λ windows to run 
    the simulation
- `num_md_steps::Int = 10`: Number of MD steps to run between coordinate sampling.
- `update_freq::Int = 1`: Number of samples to collect before updating the AWH bias. Note: the original
    developers of AWH mention that there is not a clear advantage of setting this number > 1.
- `well_tempered_factor::Real = 10.0`: If this number is set to anything other than `Inf` 
    the AWH target distribution (ρ) is dynamically updated to favour low energy λ windows. Smaller
    values accentuate this behaviour.
- `coverage_threshold::Real = 1.0`: Proportion of λ windows that must be visited to advance the
    initial stage of the algorithm, moving away from the fictitious effective sampling size.
- `significant_weight::Real = 0.1`: When checking if a λ window has been visited, this value 
    represents the fraction of an ideally uniform histogram of weights at a given bin, in order
    to consider said bin has been actually visited. This effectively filts sampling noise.
- `log_freq::Int = 100`: Number of AWH iterations between logging statistics.
- `pmf_cv = nothing`: If something other than `nothing` is passed, the deconvolution method 
    to obtain the unbiased PMF will be used. This must be a function that takes the coordinates 
    of the system and returns a tuple of scalar Collective Variables `ξ::Real`.
- `pmf_coupling = nothing`: Reqired if the previous argument was provided. Must be a function 
    that returns the dimensionless bias energy given the tuple of CVs and a `lambda_idx::Int`.
- `pmf_grid = nothing`: If the two previous arguments are passed this must be a tuple of tuples,
    indicating the `((min,), (max,), (number_of_bins,))` for each CV.
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

    n_win = length(awh_state.λ_hamiltonians)

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
    master_specific = awh_state.master_sys.specific_inter_lists
    master_general  = awh_state.master_sys.general_inters
    awh_state.active_sys.atoms = awh_state.λ_atoms[active_idx]
    awh_state.active_sys.specific_inter_lists = (master_specific..., awh_state.λ_hamiltonians[active_idx].specific_inter_lists...)
    awh_state.active_sys.general_inters       = (master_general..., awh_state.λ_hamiltonians[active_idx].general_inters...)
    awh_state.active_intg = awh_state.λ_integrators[active_idx]
end

# Reweights coordinates along λ windows and accumulates histogram
function process_sample(awh::AWHState{FT};
                        weight_relevance::Real = 0.1) where FT
    n_states = length(awh.λ_hamiltonians)
    coords = awh.active_sys.coords
    bound  = awh.active_sys.boundary

    awh.master_sys.coords   = coords
    awh.master_sys.boundary = bound
    master_pe = potential_energy(awh.master_sys)

    awh.λ_sys.coords   = coords
    awh.λ_sys.boundary = bound
    nbrs = find_neighbors(awh.λ_sys)

    potentials = Vector{FT}(undef, n_states)
    active_pe = nothing # Store active PE here

    for (n, (atoms, haml, β)) in enumerate(zip(awh.λ_atoms, awh.λ_hamiltonians, awh.λ_β))
        awh.λ_sys.atoms = atoms
        awh.λ_sys.pairwise_inters      = haml.pairwise_inters
        awh.λ_sys.specific_inter_lists = haml.specific_inter_lists
        awh.λ_sys.general_inters       = haml.general_inters
        
        # Total PE for this lambda state
        pe = master_pe + potential_energy(awh.λ_sys, nbrs, 0)
        
        # Capture the PE if this is the active window
        if n == awh.active_idx
            active_pe = pe
        end
        
        if β isa Quantity
            potentials[n] = ustrip(β * pe)
        else
            potentials[n] = β * ustrip(pe)
        end
    end

    z = awh.log_rho .+ awh.f .- potentials
    log_den = Molly.logsumexp(z)
    w = exp.(z .- log_den)

    awh.w_seg .+= w
    awh.w_last .= w
    awh.n_accum += 1
    awh.N_eff   += 1

    for (i, val) in enumerate(w)
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
    delta_f     = log.(numerator ./ denominator)
    
    awh_sim.state.f .-= delta_f
    awh_sim.state.f .-= awh_sim.state.f[1] 

    if iteration_n % awh_sim.log_freq == 0
        log_awh_statistics!(awh_sim.state, iteration_n, delta_f, current_N)
    end

    if isfinite(awh_sim.well_tempered_fac)
        f_min = minimum(awh_sim.state.f)
        @. awh_sim.state.rho = exp( - (awh_sim.state.f - f_min) / awh_sim.well_tempered_fac )
        awh_sim.state.rho ./= sum(awh_sim.state.rho)
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

"""
    simulate!(awh_sim::AWHSimulation, n_steps::Int)

Runs an AWH simulation. Number of AWH iterations is automatically determinded
from the total number of MD steps requested and the number of MD steps to be
taken between AWH loop.

# Arguments
- `awh_sim::AWHSimulation`: The [`AWHSimulation`](@ref) struct defining the system to be run.
- `n_steps::Int`: Total number of MD steps to be run.
"""
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

            # Beta processing
            raw_beta = awh_sim.state.λ_β[awh_sim.state.active_idx]
            cur_beta = zero(T)
            if raw_beta isa Quantity
                cur_beta = T(ustrip(uconvert(inv(e_unit), raw_beta)))
            else
                cur_beta = T(raw_beta)
            end
            
            # Read precomputed pressure
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
            println("New active Hamiltonian $(active_idx) -> $(active_idx_new). Iteration $(iteration_n) out of $(n_iterations)")
            active_idx = active_idx_new
        end

        update_active_sys!(awh_sim.state, active_idx)
        update_awh_bias!(awh_sim, iteration_n)
    end
end