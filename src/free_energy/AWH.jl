export AWHState

function build_neighbor_finder(ref_nfinder, eligible; 
                               reuse_neighbors::Bool = true)

    if ref_nfinder isa DistanceNeighborFinder
        nf = DistanceNeighborFinder(
            eligible = eligible, 
            dist_cutoff = ref_nfinder.dist_cutoff,
            special   = ref_nfinder.special,
            n_steps = ref_nfinder.n_steps,
            neighbors = ref_nfinder.neighbors
        )
    elseif ref_nfinder isa CellListMapNeighborFinder
        nf = CellListMapNeighborFinder(
            eligible = eligible,
            dist_cutoff = ref_nfinder.dist_cutoff,
            special = ref_nfinder.special,
            n_steps = ref_nfinder.n_steps,
            x0 = ref_sys.coords,
            unit_cell = ref_sys.boundary,
            dims = ref_nfinder.dims
        )
    elseif ref_nfinder isa GPUNeighborFinder
        if !reuse_neighbors
            nf = GPUNeighborFinder(
                eligible = eligible,
                dist_cutoff = ref_nfinder.dist_cutoff,
                special = ref_nfinder.special,
                n_steps_reorder = ref_nfinder.n_steps_reorder,
                initialized = ref_nfinder.initialized
            )
        else
            nf = DistanceNeighborFinder(
                eligible = eligible, 
                dist_cutoff = ref_nfinder.dist_cutoff,
                special   = ref_nfinder.special,
                n_steps = ref_nfinder.n_steps_reorder
            )
        end
    elseif ref_nfinder isa TreeNeighborFinder
        nf = TreeNeighborFinder(
            eligible = eligible,
            dist_cutoff = ref_nfinder.dist_cutoff,
            special = ref_nfinder.special,
            n_steps = ref_nfinder.n_steps
        )
    elseif ref_nfinder isa NoNeighborFinder
        nf = NoNeighborFinder()
    else
        @warn "Unknown NeighborFinder type $(typeof(ref_nfinder)). Using default DistanceNeighborFinder for Master System."
        nf = DistanceNeighborFinder(
            eligible = master_eligible,
            dist_cutoff = 1.0u"nm",
            n_steps = 10
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
end

function AWHState(thermo_states::AbstractArray{ThermoState};
                  n_bias::Int = 100,
                  ρ::Union{Nothing, AbstractArray{T}} = nothing,
                  reuse_neighbors::Bool = true) where T

    n_λ = length(thermo_states)
    ref_sys = thermo_states[1].system
    
    # ---------------------------------------------------------
    # 1. Identify Global Solute Indices (Union of all λ < 1)
    # ---------------------------------------------------------
    # We MUST exclude the UNION of all atoms that ever change from the Master system.
    # If we didn't, Master would contain an interaction that is valid for Window A
    # but invalid for Window B.
    solute_indices = Set{Int}()
    λ_atoms = []
    for tstate in thermo_states
        atoms     = tstate.system.atoms
        push!(λ_atoms, atoms)
        atoms_cpu = from_device(atoms)
        for atom in atoms_cpu
            # Any atom that is not fully coupled (λ < 1) in THIS window is active
            if atom.λ < 1.0
                push!(solute_indices, atom.index)
            end
        end
    end
    
    # ---------------------------------------------------------
    # 2. Partition Neighbor Lists (Eligible Matrices)
    # ---------------------------------------------------------
    # Retrieve base matrix on CPU for manipulation
    ref_nfinder = ref_sys.neighbor_finder
    base_eligible_cpu = copy(from_device(ref_nfinder.eligible))
    n_atoms = size(base_eligible_cpu, 1)
    
    # A. Generate Master Eligible Matrix (Solvent-Solvent ONLY)
    # Logic: Mask out any pair involving a 'solute' atom.
    master_eligible_cpu = copy(base_eligible_cpu)
    for idx in solute_indices
        master_eligible_cpu[idx, :] .= false
        master_eligible_cpu[:, idx] .= false
    end
    
    # B. Generate Specific Eligible Matrix Pattern (Inverse of Master)
    # Logic: Mask out the pure Solvent-Solvent block.
    # We keep Solute-Solute and Solute-Solvent interactions enabled.
    # This must cover ALL atoms in `solute_indices` to ensure we compute the 
    # interactions excluded from Master.
    specific_eligible_cpu = copy(base_eligible_cpu)
    solvent_indices = [i for i in 1:n_atoms if !(i in solute_indices)]
    
    if !isempty(solvent_indices)
        specific_eligible_cpu[solvent_indices, solvent_indices] .= false
    end

    # C. Move matrices to the correct device
    # We use ref_sys.coords to detect if we should use CuArray or Array
    AT = array_type(ref_sys)
    master_eligible = to_device(master_eligible_cpu, AT)
    λ_eligible      = to_device(specific_eligible_cpu, AT)
    

    # ---------------------------------------------------------
    # 3. Extract and Partition Interaction Lists
    # ---------------------------------------------------------
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

    # Compute "Master" (Common) Sets
    master_sils_1a = intersect(list_1a...)
    master_sils_2a = intersect(list_2a...)
    master_sils_3a = intersect(list_3a...)
    master_sils_4a = intersect(list_4a...)
    master_gils    = intersect(all_gils...)
    master_pils    = intersect(all_pils...)

    # Compute Per-State Residuals (The "Unique" parts)
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

        # For Pairwise interactions, we do NOT use setdiff.
        # Since we use the eligible matrix to split the "Pair Space" (Solvent-Solvent vs The Rest),
        # we must apply the full set of interactions to the Residual term.
        λ_pairwise[i] = (all_pils[i]...,)
    end

    # ---------------------------------------------------------
    # 4. Construct Master System
    # ---------------------------------------------------------
    
    # Reconstruct the NeighborFinder using the new master_eligible matrix
    master_nf = build_neighbor_finder(ref_nfinder, master_eligible; reuse_neighbors = reuse_neighbors)
    λ_nf      = build_neighbor_finder(ref_nfinder, λ_eligible; reuse_neighbors = reuse_neighbors)

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

    # ---------------------------------------------------------
    # 5. Handle Target Distribution (ρ)
    # ---------------------------------------------------------
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

    return AWHState(
        master_sys,
        λ_sys,
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
    )
end
