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
mutable struct AlchemicalPartition{S, L, A, H, C, T}
    master_sys::S
    λ_sys::L
    λ_atoms::A
    λ_hamiltonians::H
    
    # State cache for master energy to prevent redundant calculations 
    # across multiple single-state queries
    cached_coords::C
    cached_master_pe::T
end

function AlchemicalPartition(thermo_states::AbstractArray{<:ThermoState}; 
                             reuse_neighbors::Bool=true)
    n_λ = length(thermo_states)
    ref_sys = thermo_states[1].system
    FT = typeof(ustrip(ref_sys.total_mass))

    # 1. Identify Global Solute Indices (Perturbed Atoms)
    solute_indices = Set{Int}()
    λ_atoms = []

    # Bring the reference atoms off the device outside of the loop
    ref_atoms_cpu = from_device(ref_sys.atoms)

    for tstate in thermo_states
        atoms = tstate.system.atoms
        push!(λ_atoms, atoms)

        atoms_cpu = from_device(atoms)
        for atom in atoms_cpu
            # Flag if the atom possesses alchemical scaling properties,
            # or if its properties explicitly diverge from the reference.
            if (atom.λ < 1.0) || 
               (atom.λ != ref_atoms_cpu[atom.index].λ)
                push!(solute_indices, atom.index)
            end
        end
    end

    # 2. Partition Neighbor Lists
    ref_nfinder = ref_sys.neighbor_finder
    base_eligible_cpu, base_special_cpu = neighbor_finder_masks(ref_nfinder, length(ref_sys))
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
    special_mask    = to_device(base_special_cpu, AT)
    
    # 3. Extract and Partition Interaction Lists
    list_1a = [Vector{InteractionList1Atoms}() for _ in 1:n_λ]
    list_2a = [Vector{InteractionList2Atoms}() for _ in 1:n_λ]
    list_3a = [Vector{InteractionList3Atoms}() for _ in 1:n_λ]
    list_4a = [Vector{InteractionList4Atoms}() for _ in 1:n_λ]

    @inbounds for (i, tstate) in enumerate(thermo_states)
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

    all_gils = [collect(tstate.system.general_inters) for tstate in thermo_states]
    all_pils = [collect(tstate.system.pairwise_inters) for tstate in thermo_states]

    # Calculate interactions that are identical across ALL states
    master_sils_1a = intersect(list_1a...)
    master_sils_2a = intersect(list_2a...)
    master_sils_3a = intersect(list_3a...)
    master_sils_4a = intersect(list_4a...)
    master_gils    = intersect(all_gils...)
    master_pils    = intersect(all_pils...)

    λ_specific = Vector{Tuple}(undef, n_λ)
    λ_general  = Vector{Tuple}(undef, n_λ)
    λ_pairwise = Vector{Tuple}(undef, n_λ)

    # Distill state-specific interactions by subtracting the master subset
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

    # 4. Construct Partitioned Systems
    master_nf = build_neighbor_finder(ref_nfinder, master_eligible, special_mask;
                                      reuse_neighbors=reuse_neighbors)
    λ_nf      = build_neighbor_finder(ref_nfinder, λ_eligible, special_mask;
                                      reuse_neighbors=reuse_neighbors)

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

    hamiltonians = LambdaHamiltonian[]
    for (λ_p, λ_s, λ_g) in zip(λ_pairwise, λ_specific, λ_general)
        ham = LambdaHamiltonian(λ_p, λ_s, λ_g)
        push!(hamiltonians, ham)
    end

    # Initialize cache values with safe defaults
    initial_pe = zero(FT) * master_sys.energy_units
    
    return AlchemicalPartition(
        master_sys,
        λ_sys,
        λ_atoms,
        hamiltonians,
        copy(ref_sys.coords),
        initial_pe
    )
end

function build_neighbor_finder(ref_nfinder, eligible, special; reuse_neighbors::Bool = true)
    if ref_nfinder isa DistanceNeighborFinder
        return DistanceNeighborFinder(
            eligible = eligible, 
            dist_cutoff = ref_nfinder.dist_cutoff,
            special   = special,
            n_steps = 1
        )
    elseif ref_nfinder isa CellListMapNeighborFinder
        return CellListMapNeighborFinder(
            eligible = eligible,
            dist_cutoff = ref_nfinder.dist_cutoff,
            special = special,
            n_steps = 1
        )
    elseif ref_nfinder isa GPUNeighborFinder
        if !reuse_neighbors
            return GPUNeighborFinder(
                eligible = eligible,
                dist_cutoff = ref_nfinder.dist_cutoff,
                special = special,
                n_steps_reorder = 1,
                initialized = ref_nfinder.initialized
            )
        else
            return DistanceNeighborFinder(
                eligible = eligible, 
                dist_cutoff = ref_nfinder.dist_cutoff,
                special   = special,
                n_steps = 1
            )
        end
    elseif ref_nfinder isa TreeNeighborFinder
        return TreeNeighborFinder(
            eligible = eligible,
            dist_cutoff = ref_nfinder.dist_cutoff,
            special = special,
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
    if force_recompute || partition.cached_coords != coords
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
    
    pe_specific = potential_energy(partition.λ_sys, nbrs)
    
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
        
        pe_specific = potential_energy(partition.λ_sys, nbrs)
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
