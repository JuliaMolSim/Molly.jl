# Convenience struct that stores the interaction lists that change
# across thermodynamic states (e.g., along a reaction coordinate or replica states).
struct LambdaHamiltonian{PI, SI, GI}
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
end

# AlchemicalPartition(thermo_states::AbstractArray{ThermoState}; <keyword arguments>)
#
# Isolates shared topological and interactive components (the `master_sys`) from
# components that are unique to specific thermodynamic states (the `λ_systems`
# and `λ_hamiltonians`). This guarantees that unperturbed components (e.g.,
# bulk solvent) are evaluated exactly once when checking cross-energies or
# evaluating multiple states.
mutable struct AlchemicalPartition{S, L, LS, A, H, T}
    master_sys::S
    # First lambda system, kept as a convenience alias for older callers/tests.
    λ_sys::L
    λ_systems::LS
    λ_atoms::A
    λ_hamiltonians::H

    # State cache for master energy to prevent redundant calculations
    # across multiple single-state queries
    cached_coords::Any
    cached_master_pe::T
    cached_master_state::Int
    shared_master_neighbor_finder::Bool
    shared_λ_neighbor_finder::Bool
end

function alchemical_atom_changed(atom, ref_atom)
    return atom.λ < 1.0 ||
           atom.λ != ref_atom.λ ||
           atom.alch_role != ref_atom.alch_role ||
           charge(atom) != charge(ref_atom)
end

function effective_charges(inter::AbstractEwald, atoms)
    T = typeof(inter.error_tol)
    atoms_cpu = from_device(atoms)
    return [effective_charge(inter.scheduler, atom, Val(T)) for atom in atoms_cpu]
end

function effective_charges_vary(inter::AbstractEwald, λ_atoms)
    ref_charges = effective_charges(inter, first(λ_atoms))
    return any(atoms -> effective_charges(inter, atoms) != ref_charges, λ_atoms[2:end])
end

keep_master_general_inter(inter, λ_atoms) = !(inter isa AbstractEwald && effective_charges_vary(inter, λ_atoms))

function ewald_exclusion_charges_vary(inter_list::InteractionList2Atoms, λ_atoms)
    inter_list.data isa EwaldExclusionData || return false

    inds = unique!(vcat(Int.(from_device(inter_list.is)), Int.(from_device(inter_list.js))))
    isempty(inds) && return false

    T = typeof(inter_list.data.error_tol)
    scheduler = inter_list.data.scheduler
    ref_atoms = from_device(first(λ_atoms))
    ref_charges = [effective_charge(scheduler, ref_atoms[i], Val(T)) for i in inds]

    for atoms in λ_atoms[2:end]
        atoms_cpu = from_device(atoms)
        charges = [effective_charge(scheduler, atoms_cpu[i], Val(T)) for i in inds]
        charges != ref_charges && return true
    end
    return false
end

keep_master_specific_inter(inter_list, λ_atoms) = true
keep_master_specific_inter(inter_list::InteractionList2Atoms, λ_atoms) =
    !ewald_exclusion_charges_vary(inter_list, λ_atoms)

function AlchemicalPartition(thermo_states::AbstractArray{<:ThermoState};
                             reuse_neighbors::Bool=true)
    
    n_λ = length(thermo_states)
    n_λ > 0 || throw(ArgumentError("`thermo_states` cannot be empty."))
    ref_sys = thermo_states[1].system
    FT = typeof(ustrip(ref_sys.total_mass))

    # Append target state for comprehensive intersection
    all_states = isnothing(target_state) ? collect(thermo_states) : [thermo_states..., target_state]
    n_all = length(all_states)
    
    # 1. Identify Global Solute Indices (Perturbed Atoms)
    solute_indices = Set{Int}()
    λ_atoms = Any[]

    ref_atoms_cpu = from_device(ref_sys.atoms)
    n_ref_atoms = length(ref_atoms_cpu)

    for tstate in all_states
        atoms = tstate.system.atoms
        push!(λ_atoms, atoms)

        atoms_cpu = from_device(atoms)
        for (atom_i, atom) in pairs(atoms_cpu)
            # Flag if the atom possesses alchemical scaling properties,
            # or if its properties explicitly diverge from the reference.
            if alchemical_atom_changed(atom, ref_atoms_cpu[atom_i])
                push!(solute_indices, atom_i)
            end
        end
    end
    solute_indices = sort!(collect(solute_indices))
    split_pairwise_by_atoms = !isempty(solute_indices)
    solvent_indices = if split_pairwise_by_atoms
        [i for i in 1:n_ref_atoms if !(i in solute_indices)]
    else
        Int[]
    end

    # 2. Neighbor Finder Policy
    # Enforce a shared finder configuration across all λ windows (and target state),
    # then build exactly one partitioned finder for master and one for λ interactions.
    ref_nfinder = ref_sys.neighbor_finder
    @inbounds for (i, tstate) in enumerate(all_states)
        state_nf = tstate.system.neighbor_finder
        if typeof(state_nf) != typeof(ref_nfinder)
            throw(ArgumentError(
                "All thermodynamic states must use the same neighbor finder type. " *
                "Reference uses $(typeof(ref_nfinder)), state $i uses $(typeof(state_nf))."
            ))
        end
        if !neighbor_finders_equivalent(ref_nfinder, state_nf)
            throw(ArgumentError(
                "All thermodynamic states must use equivalent neighbor finder settings. " *
                "State $i differs from the reference in cutoff/eligible/special or update parameters."
            ))
        end
    end

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
    λ_eligible = to_device(λ_eligible_cpu, AT)
    special_mask    = to_device(base_special_cpu, AT)

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
    # Pairwise interactions that do not use a neighbor list cannot be safely split
    # by an eligibility mask. Keep those fully state-specific to avoid double counting.
    all_pils_nl = [filter(use_neighbors, pils) for pils in all_pils]
    all_pils_nonl = [filter(!use_neighbors, pils) for pils in all_pils]

    # Calculate interactions identical across ALL simulated windows AND the target state
    master_sils_1a = intersect(list_1a...)
    master_sils_2a = intersect(list_2a...)
    master_sils_3a = intersect(list_3a...)
    master_sils_4a = intersect(list_4a...)
    master_gils    = intersect(all_gils...)
    master_pils    = intersect(all_pils_nl...)

    filter!(inter -> keep_master_specific_inter(inter, λ_atoms), master_sils_2a)
    filter!(inter -> keep_master_general_inter(inter, λ_atoms), master_gils)

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

    # Extract Target Hamiltonian (Safe because lists are length n_all)
    if !isnothing(target_state)
        tgt_idx = n_all
        tgt_p_nl = split_pairwise_by_atoms ? all_pils_nl[tgt_idx] : setdiff(all_pils_nl[tgt_idx], master_pils)
        tgt_p = (tgt_p_nl..., all_pils_nonl[tgt_idx]...)
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
    master_nf = build_neighbor_finder(ref_nfinder, master_eligible, special_mask;
                                      reuse_neighbors=reuse_neighbors, boundary=ref_sys.boundary)

    master_sys = System(deepcopy(ref_sys);
        pairwise_inters      = (master_pils...,),
        general_inters       = (master_gils...,),
        specific_inter_lists = (master_sils_1a...,
                                master_sils_2a...,
                                master_sils_3a...,
                                master_sils_4a...),
        neighbor_finder      = master_nf,
        loggers              = (),
    )

    hamiltonians = LambdaHamiltonian[]
    λ_systems = Any[]
    for (λ_p, λ_s, λ_g) in zip(λ_pairwise, λ_specific, λ_general)
        ham = LambdaHamiltonian(λ_p, λ_s, λ_g)
        push!(hamiltonians, ham)

        λ_nf = build_neighbor_finder(ref_nfinder, λ_eligible, special_mask;
                                     reuse_neighbors=reuse_neighbors,
                                     boundary=ref_sys.boundary)
        λ_sys = System(deepcopy(ref_sys);
            atoms                = λ_atoms[length(λ_systems) + 1],
            pairwise_inters      = (λ_p...,),
            general_inters       = (λ_g...,),
            specific_inter_lists = (λ_s...,),
            neighbor_finder      = λ_nf,
            loggers              = (),
        )
        push!(λ_systems, λ_sys)
    end

    # Initialize cache values with safe defaults
    initial_pe = zero(FT) * master_sys.energy_units

    return AlchemicalPartition(
        master_sys,
        first(λ_systems),
        λ_systems,
        λ_atoms,
        hamiltonians,
        nothing,
        initial_pe
    )
end

function build_neighbor_finder(ref_nfinder, eligible, special; reuse_neighbors::Bool = true, boundary = nothing)
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
            n_steps = 1,
            boundary = boundary,
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

function update_master_energy!(partition::AlchemicalPartition, coords, boundary;
                               force_recompute::Bool=false,
                               n_threads::Integer=Threads.nthreads())
    if force_recompute || isnothing(partition.cached_coords) || partition.cached_coords != coords
        partition.master_sys.coords = coords
        partition.master_sys.boundary = boundary
        partition.cached_master_pe = potential_energy(partition.master_sys;
                                                      n_threads=n_threads)
        # Forced multi-state calls already know they need a fresh master energy. Only the
        # single-state cached path keeps a coordinate snapshot for repeated same-frame queries.
        partition.cached_coords = force_recompute ? nothing : copy(coords)
    end
    return partition.cached_master_pe
end

function evaluate_λ_energy!(λ_sys, coords, boundary;
                            n_threads::Integer=Threads.nthreads())
    λ_sys.coords = coords
    λ_sys.boundary = boundary
    nbrs = find_neighbors(λ_sys; n_threads=n_threads)
    return potential_energy(λ_sys, nbrs, 0; n_threads=n_threads)
end

# evaluate_energy!(partition::AlchemicalPartition, coords, boundary, state_index::Int;
#                  force_recompute::Bool=false)
#
# Calculates the total potential energy for a specific thermodynamic state.
# Caches the `master_sys` energy. If `coords` is identical to `cached_coords`,
# the `master_sys` energy is not recomputed unless `force_recompute` is true.
function evaluate_energy!(partition::AlchemicalPartition, coords, boundary, state_index::Int;
                          force_recompute::Bool=false,
                          n_threads::Integer=Threads.nthreads())
    1 <= state_index <= length(partition.λ_systems) ||
        throw(ArgumentError("state_index ($state_index) out of range " *
                            "1:$(length(partition.λ_systems))"))

    update_master_energy!(partition, coords, boundary;
                          force_recompute=force_recompute, n_threads=n_threads)
    pe_specific = evaluate_λ_energy!(partition.λ_systems[state_index], coords, boundary;
                                     n_threads=n_threads)
    return partition.cached_master_pe + pe_specific
end

# evaluate_energy_all!(partition::AlchemicalPartition, coords, boundary)
#
# Efficiently evaluates the potential energy of the current coordinates mapped
# across all thermodynamic states simultaneously, evaluating the unperturbed
# `master_sys` only once.
function evaluate_energy_all!(partition::AlchemicalPartition, coords, boundary;
                              n_threads::Integer=Threads.nthreads())
    update_master_energy!(partition, coords, boundary;
                          force_recompute=true, n_threads=n_threads)
    energies = Vector{typeof(partition.cached_master_pe)}(undef, length(partition.λ_systems))

    for state_index in eachindex(partition.λ_systems)
        pe_specific = evaluate_λ_energy!(partition.λ_systems[state_index], coords, boundary;
                                         n_threads=n_threads)
        energies[state_index] = partition.cached_master_pe + pe_specific
    end

    return energies
end

function evaluate_energy_all!(partition::AlchemicalPartition, coords, boundary)
    energies = Vector{typeof(partition.cached_master_pe)}(undef, length(partition.λ_hamiltonians))
    return evaluate_energy_all!(partition, coords, boundary, energies)
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
