export 
    AWHState,
    AWHSimulation,
    AWHPMFDeconvolution,
    simulate!

# --- Statistics & Logging (Storage) ---
mutable struct AWHStats{T}
    step_indices::Vector{Int}
    active_λ::Vector{Int}
    f_history::Vector{Vector{T}}
    n_effective_history::Vector{T} # Logs the ACTIVE N (N_bias or N_ref)
    stage_history::Vector{Symbol}  # :initial or :linear
    max_delta_f_history::Vector{T}
end

# --- PMF Deconvolution Struct (Generalized) ---
"""
    AWHPMFDeconvolution{N, T, F_CV, F_Coup}

Stores the histograms required to compute the Potential of Mean Force (PMF) 
using the AWH deconvolution method (Lindahl et al. 2014, Eq. 9).

# Fields
- `min_vals`: Tuple of minimum boundaries for the PMF grid (per dimension).
- `bin_widths`: Tuple of bin widths.
- `shape`: Tuple of grid dimensions (number of bins).
- `cv_function`: Function `coords -> ξ` (Tuple). Calculates the reaction coordinate.
- `coupling_function`: Function `(ξ, λ_index) -> Q`. Returns the *dimensionless* bias energy.
- `numerator_hist`: Accumulator for Eq. 9 numerator.
- `denominator_hist`: Accumulator for Eq. 9 denominator.
- `sample_count`: Total samples accumulated.
"""
mutable struct AWHPMFDeconvolution{N, T, F_CV, F_Coup}
    min_vals::NTuple{N, T}
    bin_widths::NTuple{N, T}
    shape::NTuple{N, Int}

    cv_function::F_CV         
    coupling_function::F_Coup 

    numerator_hist::Array{T, N}
    denominator_hist::Array{T, N}
    
    sample_count::Int
end

function AWHPMFDeconvolution(
    min_vals::NTuple{N, T},
    max_vals::NTuple{N, T},
    n_bins::NTuple{N, Int},
    cv_function::F_CV,
    coupling_function::F_Coup
) where {N, T, F_CV, F_Coup}
    
    bin_widths = (max_vals .- min_vals) ./ n_bins
    
    return AWHPMFDeconvolution{N, T, F_CV, F_Coup}(
        min_vals,
        bin_widths,
        n_bins,
        cv_function,
        coupling_function,
        zeros(T, n_bins...),
        zeros(T, n_bins...),
        0
    )
end

function update_pmf!(
    pmf::AWHPMFDeconvolution{N, T, F_CV, F_Coup},
    awh_state, # Untyped to avoid circular dependency issues in definitions
    curr_coords
) where {N, T, F_CV, F_Coup}
    
    # 1. Calculate current CV ξ(t)
    # The cv_function should return a Tuple (even for 1D) or we wrap it
    val = pmf.cv_function(curr_coords)
    current_cv = val isa Tuple ? val : (val,)

    # 2. Determine which bin the current sample falls into
    # We use index math: idx = floor((val - min)/width) + 1
    # This assumes the grid covers the domain of interest.
    current_indices = ntuple(N) do d
        # Clamp to grid edges to avoid bounds errors, or let it fail if stricter
        rel = (current_cv[d] - pmf.min_vals[d]) / pmf.bin_widths[d]
        idx = Int(floor(rel)) + 1
        clamp(idx, 1, pmf.shape[d])
    end
    current_cartesian = CartesianIndex(current_indices)

    # 3. Pre-calc effective bias factors g(λ) = f(λ) + ln ρ(λ)
    # P_sim ~ exp(-E + g). We need to unbias by exp(-g).
    # Lindahl Eq 9 uses e^{-gamma} = sum_lambda exp(g - Q).
    g = awh_state.f .+ awh_state.log_rho
    n_windows = length(g)

    # 4. Iterate over the entire PMF Grid (The Deconvolution)
    # We calculate gamma(xi) for every bin center.
    
    # Helper to get bin center coordinate
    get_bin_center(d, i) = pmf.min_vals[d] + (i - 0.5) * pmf.bin_widths[d]

    for I in CartesianIndices(pmf.shape)
        # Construct ξ for this bin
        xi = ntuple(d -> get_bin_center(d, I[d]), N)
        
        # Calculate sum_lambda exp(g(λ) - Q(ξ, λ))
        sum_exp = zero(T)
        for k in 1:n_windows
            # Generalized coupling: User provides Q(ξ, k)
            Q = pmf.coupling_function(xi, k)
            sum_exp += exp(g[k] - Q)
        end
        
        # Unbiasing factor e^{gamma(xi)} = 1 / sum_exp
        term = 1.0 / sum_exp
        
        # Update Denominator (Everywhere)
        pmf.denominator_hist[I] += term
        
        # Update Numerator (Only at sampled position)
        if I == current_cartesian
            pmf.numerator_hist[I] += term
        end
    end
    
    pmf.sample_count += 1
end


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
            x0 = ref_nfinder.x0, # Assuming x0 available or generic ref
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

struct LambdaHamiltonian{PI, SI, GI}
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
end

"""
    AWHState(thermo_states; n_bias=100, ρ=nothing)

Represents the state of an AWH (Adaptive Biasing Force) simulation. 
This struct manages the "Master" system (common solvent-solvent interactions) and the 
differential "Specific" interactions (solute-solute and solute-solvent) for each lambda window.
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
    
    # 1. Identify Global Solute Indices
    solute_indices = Set{Int}()
    λ_atoms = []
    λ_integrators = []
    λ_β = []
    for tstate in thermo_states
        atoms = tstate.system.atoms
        intg  = tstate.integrator
        push!(λ_atoms, atoms)
        push!(λ_integrators, intg)
        push!(λ_β, tstate.beta)
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

    FT = typeof(ustrip(master_sys.total_mass))
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

struct AWHSimulation{T}
    n_windows::Int
    initial_sampl_n::T
    n_md_steps::Int         # Number of MD steps to run between samplings
    update_freq::Int        # Number of samples between f-updates
    well_tempered_fac::T    # Well-tempered factor (Inf = Standard AWH/Uniform)
    coverage_threshold::T   # Fraction of windows visited to trigger doubling
    significant_weight::T   # Threshold to count a window as "visited"
    log_freq::Int           # Frequency of storing data to history
    state::AWHState{T}
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
    pmf_cv = nothing,
    pmf_coupling = nothing,
    pmf_grid = nothing
) where T

    n_win = length(awh_state.λ_hamiltonians)

    pmf_calc = nothing
    if !isnothing(pmf_cv) && !isnothing(pmf_coupling) && !isnothing(pmf_grid)
        min_vals, max_vals, n_bins = pmf_grid
        pmf_calc = AWHPMFDeconvolution(min_vals, max_vals, n_bins, pmf_cv, pmf_coupling)
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

function update_active_sys!(awh_state::AWHState, active_idx::Int)
    awh_state.active_idx = active_idx
    master_specific = awh_state.master_sys.specific_inter_lists
    master_general  = awh_state.master_sys.general_inters
    awh_state.active_sys.atoms = awh_state.λ_atoms[active_idx]
    awh_state.active_sys.specific_inter_lists = (master_specific..., awh_state.λ_hamiltonians[active_idx].specific_inter_lists...)
    awh_state.active_sys.general_inters       = (master_general..., awh_state.λ_hamiltonians[active_idx].general_inters...)
    awh_state.active_intg = awh_state.λ_integrators[active_idx]
end

function process_sample(awh::AWHState{FT}; weight_relevance::Real = 0.1) where FT
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

    for (n, (atoms, haml, β)) in enumerate(zip(awh.λ_atoms, awh.λ_hamiltonians, awh.λ_β))
        awh.λ_sys.atoms          = atoms
        awh.λ_sys.pairwise_inters      = haml.pairwise_inters
        awh.λ_sys.specific_inter_lists = haml.specific_inter_lists
        awh.λ_sys.general_inters       = haml.general_inters
        pe = master_pe + potential_energy(awh.λ_sys, nbrs, 0)
        potentials[n] = β * pe
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
end

function gibbs_sample_window(state::AWHState)
    return sample(1:length(state.w_last), Weights(state.w_last))
end

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

function update_awh_bias!(awh_sim::AWHSimulation, iteration_n::Int)
    if awh_sim.state.n_accum < awh_sim.update_freq
        return nothing 
    end

    current_N = awh_sim.state.in_initial_stage ?
        awh_sim.state.N_bias : (awh_sim.initial_sampl_n + awh_sim.state.N_eff)

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

function simulate!(awh_sim::AWHSimulation, n_steps::Int)

    n_iterations = Int(floor(n_steps / awh_sim.n_md_steps))
    active_idx = 1

    for iteration_n in 1:n_iterations
        simulate!(awh_sim.state.active_sys, awh_sim.state.active_intg, awh_sim.n_md_steps)

        process_sample(awh_sim.state)

        # --- PMF Deconvolution Update ---
        if !isnothing(awh_sim.pmf_calc)
            update_pmf!(awh_sim.pmf_calc, awh_sim.state, awh_sim.state.active_sys.coords)
        end
        # --------------------------------

        active_idx_new = gibbs_sample_window(awh_sim.state)
        if active_idx_new != active_idx
            println("New active Hamiltonian $(active_idx) -> $(active_idx_new). Iteration $(iteration_n)")
            active_idx = active_idx_new
        end

        update_active_sys!(awh_sim.state, active_idx)
        update_awh_bias!(awh_sim, iteration_n)
    end
end