export
    LambdaHamiltonian,
    AlchemicalPartition,
    evaluate_energy!,
    evaluate_energy_all!

# Convenience struct that stores the interaction lists that change 
# across thermodynamic states (e.g., along a reaction coordinate or replica states).
struct LambdaHamiltonian{PI, SI, GI}
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
end

@doc raw"""
    AlchemicalPartition(thermo_states::AbstractArray{ThermoState}; <keyword arguments>)

Isolates shared topological and interactive components (the `master_sys`) from components 
that are unique to specific thermodynamic states (the `λ_sys` and `λ_hamiltonians`).
This guarantees that unperturbed components (e.g., bulk solvent) are evaluated 
exactly once when checking cross-energies or evaluating multiple states.
"""
mutable struct AlchemicalPartition{S, L, A, H, C, T, TH, TA}
    master_sys::S
    λ_sys::L
    λ_atoms::A
    λ_hamiltonians::H
    
    # Pre-compiled arbitrary target state for PMF deconvolution
    target_hamiltonian::TH
    target_atoms::TA

    cached_coords::C
    cached_master_pe::T
end

function AlchemicalPartition(thermo_states::AbstractArray{<:ThermoState}; 
                             target_state::Union{ThermoState, Nothing}=nothing,
                             reuse_neighbors::Bool=true)
    
    n_λ = length(thermo_states)
    ref_sys = thermo_states[1].system
    FT = typeof(ustrip(ref_sys.total_mass))

    # Append target state for comprehensive intersection
    all_states = isnothing(target_state) ? collect(thermo_states) : [thermo_states..., target_state]
    n_all = length(all_states)
    
    # 1. Identify Global Solute Indices (Perturbed Atoms)
    solute_indices = Set{Int}()
    λ_atoms = []

    ref_atoms_cpu = from_device(ref_sys.atoms)

    for tstate in all_states
        atoms = tstate.system.atoms
        push!(λ_atoms, atoms)

        atoms_cpu = from_device(atoms)
        for atom in atoms_cpu
            if (atom.λ < 1.0) || (atom.λ != ref_atoms_cpu[atom.index].λ)
                push!(solute_indices, atom.index)
            end
        end
    end

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
    # Size correctly for ALL states including the target
    list_1a = [Vector{InteractionList1Atoms}() for _ in 1:n_all]
    list_2a = [Vector{InteractionList2Atoms}() for _ in 1:n_all]
    list_3a = [Vector{InteractionList3Atoms}() for _ in 1:n_all]
    list_4a = [Vector{InteractionList4Atoms}() for _ in 1:n_all]

    # Iterate over all_states to capture target interactions for valid intersection
    @inbounds for (i, tstate) in enumerate(all_states)
        sils = tstate.system.specific_inter_lists
        for inter in sils
            if inter isa InteractionList1Atoms 
                push!(list_1a[i], inter)
            elseif inter isa InteractionList2Atoms 
                push!(list_2a[i], inter)
            elseif inter isa InteractionList3Atoms 
                push!(list_3a[i], inter)
            elseif inter isa InteractionList4Atoms 
                push!(list_4a[i], inter)
            end
        end
    end

    all_gils = [collect(tstate.system.general_inters) for tstate in all_states]
    all_pils = [collect(tstate.system.pairwise_inters) for tstate in all_states]

    # Calculate interactions identical across ALL simulated windows AND the target state
    master_sils_1a = intersect(list_1a...)
    master_sils_2a = intersect(list_2a...)
    master_sils_3a = intersect(list_3a...)
    master_sils_4a = intersect(list_4a...)
    master_gils    = intersect(all_gils...)
    master_pils    = intersect(all_pils...)

    # Extract λ-specific Hamiltonians for the simulated windows
    hamiltonians = LambdaHamiltonian[]
    for i in 1:n_λ
        λ_p = (all_pils[i]...,)
        λ_s = (setdiff(list_1a[i], master_sils_1a)..., 
               setdiff(list_2a[i], master_sils_2a)...,
               setdiff(list_3a[i], master_sils_3a)...,
               setdiff(list_4a[i], master_sils_4a)...)
        λ_g = (setdiff(all_gils[i], master_gils)...,)
        push!(hamiltonians, LambdaHamiltonian(λ_p, λ_s, λ_g))
    end

    # Extract Target Hamiltonian (Safe because lists are length n_all)
    if !isnothing(target_state)
        tgt_idx = n_all
        tgt_p = (all_pils[tgt_idx]...,)
        tgt_s = (setdiff(list_1a[tgt_idx], master_sils_1a)..., 
                 setdiff(list_2a[tgt_idx], master_sils_2a)...,
                 setdiff(list_3a[tgt_idx], master_sils_3a)...,
                 setdiff(list_4a[tgt_idx], master_sils_4a)...)
        tgt_g = (setdiff(all_gils[tgt_idx], master_gils)...,)
        target_hamiltonian = LambdaHamiltonian(tgt_p, tgt_s, tgt_g)
        target_atoms_array = λ_atoms[tgt_idx]
    else
        target_hamiltonian = nothing
        target_atoms_array = nothing
    end

    # 4. Construct Partitioned Systems
    master_nf = build_neighbor_finder(ref_nfinder, master_eligible; reuse_neighbors=reuse_neighbors)
    λ_nf      = build_neighbor_finder(ref_nfinder, λ_eligible; reuse_neighbors=reuse_neighbors)

    master_sys = System(deepcopy(ref_sys); 
        pairwise_inters      = (master_pils...,),
        general_inters       = (master_gils...,),
        specific_inter_lists = (master_sils_1a..., 
                                master_sils_2a..., 
                                master_sils_3a..., 
                                master_sils_4a...),
        neighbor_finder      = master_nf
    )

    # Use the first simulated state's difference Hamiltonian to initialize λ_sys
    λ_sys = System(deepcopy(ref_sys); 
        pairwise_inters      = hamiltonians[1].pairwise_inters,
        general_inters       = hamiltonians[1].general_inters,
        specific_inter_lists = hamiltonians[1].specific_inter_lists,
        neighbor_finder      = λ_nf
    )

    initial_pe = zero(FT) * master_sys.energy_units
    
    return AlchemicalPartition(
        master_sys, λ_sys, λ_atoms[1:n_λ], hamiltonians, 
        target_hamiltonian, target_atoms_array,
        copy(ref_sys.coords), initial_pe
    )
end

function build_neighbor_finder(ref_nfinder, eligible; reuse_neighbors::Bool = true)
    if ref_nfinder isa DistanceNeighborFinder
        return DistanceNeighborFinder(
            eligible = eligible, 
            dist_cutoff = ref_nfinder.dist_cutoff,
            special   = ref_nfinder.special,
            n_steps = 1
        )
    elseif ref_nfinder isa CellListMapNeighborFinder
        return CellListMapNeighborFinder(
            eligible = eligible,
            dist_cutoff = ref_nfinder.dist_cutoff,
            special = ref_nfinder.special,
            n_steps = 1
        )
    elseif ref_nfinder isa GPUNeighborFinder
        if !reuse_neighbors
            return GPUNeighborFinder(
                eligible = eligible,
                dist_cutoff = ref_nfinder.dist_cutoff,
                special = ref_nfinder.special,
                n_steps_reorder = 1,
                initialized = ref_nfinder.initialized
            )
        else
            return DistanceNeighborFinder(
                eligible = eligible, 
                dist_cutoff = ref_nfinder.dist_cutoff,
                special   = ref_nfinder.special,
                n_steps = 1
            )
        end
    elseif ref_nfinder isa TreeNeighborFinder
        return TreeNeighborFinder(
            eligible = eligible,
            dist_cutoff = ref_nfinder.dist_cutoff,
            special = ref_nfinder.special,
            n_steps = 1
        )
    elseif ref_nfinder isa NoNeighborFinder
        return NoNeighborFinder()
    else
        throw(ArgumentError("Unknown NeighborFinder type $(typeof(ref_nfinder))"))
    end
end

"""
    evaluate_energy!(partition::AlchemicalPartition, coords, boundary, state_index::Int; force_recompute::Bool=false)

Calculates the total potential energy for a specific thermodynamic state. Caches the `master_sys` 
energy. If `coords` is identical to `cached_coords`, the `master_sys` energy is not recomputed 
unless `force_recompute` is true.
"""
function evaluate_energy!(partition::AlchemicalPartition, coords, boundary, state_index::Int; 
                          force_recompute::Bool=false)
    # Check if the master system needs re-evaluation
    if force_recompute || partition.cached_coords !== coords
        partition.master_sys.coords = coords
        partition.master_sys.boundary = boundary
        partition.cached_master_pe = potential_energy(partition.master_sys)
        # Update cache tracking. Only update reference to avoid allocation where possible.
        partition.cached_coords = coords
    end
    
    partition.λ_sys.coords = coords
    partition.λ_sys.boundary = boundary
    nbrs = find_neighbors(partition.λ_sys)
    
    # Load specific interactions for the requested state
    partition.λ_sys.atoms = partition.λ_atoms[state_index]
    partition.λ_sys.pairwise_inters = partition.λ_hamiltonians[state_index].pairwise_inters
    partition.λ_sys.specific_inter_lists = partition.λ_hamiltonians[state_index].specific_inter_lists
    partition.λ_sys.general_inters = partition.λ_hamiltonians[state_index].general_inters
    
    pe_specific = potential_energy(partition.λ_sys, nbrs, 0)
    
    return partition.cached_master_pe + pe_specific
end

function evaluate_energy!(partition::AlchemicalPartition, coords, boundary, 
                          target_ham::LambdaHamiltonian, target_atoms; 
                          force_recompute::Bool=false)
    if force_recompute || partition.cached_coords !== coords
        partition.master_sys.coords = coords
        partition.master_sys.boundary = boundary
        partition.cached_master_pe = potential_energy(partition.master_sys)
        partition.cached_coords = coords
    end
    
    partition.λ_sys.coords = coords
    partition.λ_sys.boundary = boundary
    nbrs = find_neighbors(partition.λ_sys)
    
    partition.λ_sys.atoms = target_atoms
    partition.λ_sys.pairwise_inters = target_ham.pairwise_inters
    partition.λ_sys.specific_inter_lists = target_ham.specific_inter_lists
    partition.λ_sys.general_inters = target_ham.general_inters
    
    pe_specific = potential_energy(partition.λ_sys, nbrs, 0)
    
    return partition.cached_master_pe + pe_specific
end

"""
    evaluate_energy_all!(partition::AlchemicalPartition, coords, boundary)

Efficiently evaluates the potential energy of the current coordinates mapped across all 
thermodynamic states simultaneously, evaluating the unperturbed `master_sys` only once.
"""
function evaluate_energy_all!(partition::AlchemicalPartition, coords, boundary)
    partition.master_sys.coords = coords
    partition.master_sys.boundary = boundary
    partition.cached_master_pe = potential_energy(partition.master_sys)
    partition.cached_coords = coords
    
    partition.λ_sys.coords = coords
    partition.λ_sys.boundary = boundary
    nbrs = find_neighbors(partition.λ_sys)
    
    energies = Vector{typeof(partition.cached_master_pe)}(undef, length(partition.λ_hamiltonians))
    
    for state_index in eachindex(partition.λ_hamiltonians)
        partition.λ_sys.atoms = partition.λ_atoms[state_index]
        partition.λ_sys.pairwise_inters = partition.λ_hamiltonians[state_index].pairwise_inters
        partition.λ_sys.specific_inter_lists = partition.λ_hamiltonians[state_index].specific_inter_lists
        partition.λ_sys.general_inters = partition.λ_hamiltonians[state_index].general_inters
        
        pe_specific = potential_energy(partition.λ_sys, nbrs, 0)
        energies[state_index] = partition.cached_master_pe + pe_specific
    end
    
    return energies
end

function logsumexp(x::AbstractVector{T}) where T
    isempty(x) && return -T(Inf)
    x_max = maximum(x)
    # If all weights are -Inf (e.g. huge energy overlap issues), return -Inf
    !isfinite(x_max) && return x_max 
    
    s = zero(T)
    for val in x
        s += exp(val - x_max)
    end
    return x_max + log(s)
end
