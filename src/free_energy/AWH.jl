export
    AWHState

"""
    AWHState()

Represents the current state of an AWH run.
TODO: IMPROVE DOCSTRING
"""
mutable struct AWHState{T}

    # Molly system, the simulation driver
    system::System

    # Per λ interactions
    λ_pairwise
    λ_specific
    λ_general

    # Probability & Free Energy
    f::Vector{T}            # Free energy of each window
    rho::Vector{T}          # Target distriibution
    log_rho::Vector{T}      # Log of target distribution, to save computation
    
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
) where T

    n_λ = length(thermo_states)
    
    # 1. Extract Lists
    # We keep them separate to compute intersections easily
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

    # 2. Compute "Master" (Common) Sets
    master_sils_1a = intersect(list_1a...)
    master_sils_2a = intersect(list_2a...)
    master_sils_3a = intersect(list_3a...)
    master_sils_4a = intersect(list_4a...)
    master_gils    = intersect(all_gils...)
    master_pils    = intersect(all_pils...)

    # 3. Compute Per-State Residuals (The "Unique" parts)
    # We want λ_specific[i] to be a Tuple of all unique interactions for state i.
    λ_specific = Vector{Tuple}(undef, n_λ)
    λ_general  = Vector{Tuple}(undef, n_λ)
    λ_pairwise = Vector{Tuple}(undef, n_λ)

    for i in 1:n_λ
        # setdiff returns the elements in A not in B.
        # We combine 1a, 2a, 3a, 4a into one tuple for this state.
        u_1 = setdiff(list_1a[i], master_sils_1a)
        u_2 = setdiff(list_2a[i], master_sils_2a)
        u_3 = setdiff(list_3a[i], master_sils_3a)
        u_4 = setdiff(list_4a[i], master_sils_4a)
        
        λ_specific[i] = (u_1..., u_2..., u_3..., u_4...)
        
        # Do the same for general and pairwise
        u_g = setdiff(all_gils[i], master_gils)
        λ_general[i] = (u_g...,)

        u_p = setdiff(all_pils[i], master_pils)
        λ_pairwise[i] = (u_p...,)
    end

    # 4. Construct Master System
    # We recreate the system using the COMMON interactions.
    # Note: We must splat (...) the vectors into Tuples as System expects Tuples.
    master_sys = System(thermo_states[1].system; # Use convenience constructor to copy props
                        pairwise_inters      = (master_pils...,),
                        general_inters       = (master_gils...,),
                        specific_inter_lists = (master_sils_1a..., 
                                                master_sils_2a..., 
                                                master_sils_3a..., 
                                                master_sils_4a...)
                        )

    FT = typeof(ustrip(master_sys.total_mass)) # Get float type from system mass

    # 5. Handle Target Distribution (ρ)
    if isnothing(ρ)
        rho_val = fill(FT(1/n_λ), n_λ)
    else
        if eltype(ρ) != FT
             # Convert if necessary, or throw error as you did
             rho_val = FT.(ρ) 
        else
             rho_val = ρ
        end
    end
    log_ρ = log.(rho_val)

    return AWHState(
        master_sys,
        λ_pairwise, 
        λ_specific, 
        λ_general,
        zeros(FT, n_λ),   # f
        rho_val,          # rho
        log_ρ,            # log_rho
        zeros(FT, n_λ),   # w_seg
        zeros(FT, n_λ),   # w_last
        zero(FT),         # N_eff
        FT(n_bias),       # N_bias
        0,                # n_accum
        true,             # in_initial_stage
        Set{Int}(),       # visited_windows
    )
end