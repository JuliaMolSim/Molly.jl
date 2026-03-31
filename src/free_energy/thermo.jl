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
mutable struct AlchemicalPartition{S, L, A, H, MNF, LNF, TMNF, TLNF, C, T, TH, TA}
    master_sys::S
    λ_sys::L
    λ_atoms::A
    λ_hamiltonians::H
    master_neighbor_finders::MNF
    λ_neighbor_finders::LNF

    target_master_neighbor_finder::TMNF
    target_λ_neighbor_finder::TLNF

    # Pre-compiled arbitrary target state for PMF deconvolution
    target_hamiltonian::TH
    target_atoms::TA

    cached_coords::C
    cached_master_pe::T
    cached_master_state::Int
    shared_master_neighbor_finder::Bool
    shared_λ_neighbor_finder::Bool
end

function AlchemicalPartition(thermo_states::AbstractArray{<:ThermoState}; 
                             target_state::Union{ThermoState, Nothing}=nothing,
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
        length(atoms_cpu) == n_ref_atoms || throw(ArgumentError(
            "All thermodynamic states must contain the same number of atoms. " *
            "Reference has $n_ref_atoms atoms, found $(length(atoms_cpu))."
        ))
        for i in eachindex(atoms_cpu, ref_atoms_cpu)
            if atoms_cpu[i] != ref_atoms_cpu[i]
                push!(solute_indices, i)
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

    size(base_eligible_cpu, 2) == n_atoms || throw(ArgumentError(
        "NeighborFinder eligible matrix must be square, found size $(size(base_eligible_cpu))."
    ))
    n_atoms == n_ref_atoms || throw(ArgumentError(
        "NeighborFinder eligible matrix has size $n_atoms, " *
        "but systems contain $n_ref_atoms atoms."
    ))

    master_eligible_cpu = copy(base_eligible_cpu)
    for idx in solute_indices
        master_eligible_cpu[idx, :] .= false
        master_eligible_cpu[:, idx] .= false
    end

    λ_eligible_cpu = copy(base_eligible_cpu)
    if split_pairwise_by_atoms && !isempty(solvent_indices)
        λ_eligible_cpu[solvent_indices, solvent_indices] .= false
    end

    AT = array_type(ref_sys)
    master_eligible = to_device(master_eligible_cpu, AT)
    λ_eligible = to_device(λ_eligible_cpu, AT)
    special_mask    = to_device(base_special_cpu, AT)
    master_nf = build_neighbor_finder(ref_nfinder, master_eligible, special_mask;
                                      reuse_neighbors=reuse_neighbors)
    λ_nf = build_neighbor_finder(ref_nfinder, λ_eligible, special_mask;
                                 reuse_neighbors=reuse_neighbors)

    state_master_neighbor_finders = Any[master_nf for _ in 1:n_λ]
    state_λ_neighbor_finders = Any[λ_nf for _ in 1:n_λ]
    
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

    # Extract λ-specific Hamiltonians for the simulated windows
    hamiltonians = LambdaHamiltonian[]
    for i in 1:n_λ
        # Keep neighbor-list interactions in both master and λ layers only
        # when atom-based splitting is active; then the eligibility masks are
        # disjoint and there is no double-counting.
        # If no perturbed atoms are detected, evaluate only the state-specific
        # remainder in λ over the full eligibility mask.
        λ_p_nl = split_pairwise_by_atoms ? all_pils_nl[i] : setdiff(all_pils_nl[i], master_pils)
        λ_p = (λ_p_nl..., all_pils_nonl[i]...)
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
    target_master_neighbor_finder = isnothing(target_state) ? nothing : master_nf
    target_λ_neighbor_finder = isnothing(target_state) ? nothing : λ_nf
    shared_master_neighbor_finder = all_neighbor_finders_equivalent(state_master_neighbor_finders)
    shared_λ_neighbor_finder = all_neighbor_finders_equivalent(state_λ_neighbor_finders)
    
    return AlchemicalPartition(
        master_sys, λ_sys, λ_atoms[1:n_λ], hamiltonians,
        state_master_neighbor_finders, state_λ_neighbor_finders,
        target_master_neighbor_finder, target_λ_neighbor_finder,
        target_hamiltonian, target_atoms_array,
        copy(ref_sys.coords), initial_pe, 0,
        shared_master_neighbor_finder, shared_λ_neighbor_finder
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

@inline function masks_equivalent(a, b)
    a === b && return true
    if a isa AbstractArray && b isa AbstractArray
        size(a) == size(b) || return false
        return isequal(from_device(a), from_device(b))
    end
    return a == b
end

neighbor_finders_equivalent(::NoNeighborFinder, ::NoNeighborFinder) = true

function neighbor_finders_equivalent(a::DistanceNeighborFinder, b::DistanceNeighborFinder)
    return a.dist_cutoff == b.dist_cutoff &&
           a.n_steps == b.n_steps &&
           masks_equivalent(a.eligible, b.eligible) &&
           masks_equivalent(a.special, b.special)
end

function neighbor_finders_equivalent(a::TreeNeighborFinder, b::TreeNeighborFinder)
    return a.dist_cutoff == b.dist_cutoff &&
           a.n_steps == b.n_steps &&
           masks_equivalent(a.eligible, b.eligible) &&
           masks_equivalent(a.special, b.special)
end

function neighbor_finders_equivalent(a::CellListMapNeighborFinder, b::CellListMapNeighborFinder)
    return a.dist_cutoff == b.dist_cutoff &&
           a.n_steps == b.n_steps &&
           masks_equivalent(a.eligible, b.eligible) &&
           masks_equivalent(a.special, b.special)
end

function neighbor_finders_equivalent(a::GPUNeighborFinder, b::GPUNeighborFinder)
    return a.dist_cutoff == b.dist_cutoff &&
           a.n_steps_reorder == b.n_steps_reorder &&
           masks_equivalent(a.eligible, b.eligible) &&
           masks_equivalent(a.special, b.special)
end

neighbor_finders_equivalent(a, b) = false

function all_neighbor_finders_equivalent(neighbor_finders)
    n_states = length(neighbor_finders)
    n_states <= 1 && return true
    ref_nf = neighbor_finders[1]
    @inbounds for i in 2:n_states
        if !neighbor_finders_equivalent(ref_nf, neighbor_finders[i])
            return false
        end
    end
    return true
end

"""
    evaluate_energy!(partition::AlchemicalPartition, coords, boundary, state_index::Int; force_recompute::Bool=false)

Calculates the total potential energy for a specific thermodynamic state. Caches the `master_sys` 
energy. If `coords` and `boundary` are unchanged from the previous cached values, the
`master_sys` energy is not recomputed unless `force_recompute` is true.
"""
function refresh_master_energy!(partition::AlchemicalPartition, coords, boundary, state_tag::Int,
                                master_neighbor_finder; force_recompute::Bool=false)
    needs_master_recompute = force_recompute ||
        partition.cached_master_state != state_tag ||
        partition.master_sys.boundary != boundary ||
        partition.cached_coords != coords

    partition.master_sys.coords = coords
    partition.master_sys.boundary = boundary
    partition.master_sys.neighbor_finder = master_neighbor_finder

    if needs_master_recompute
        partition.cached_master_pe = potential_energy(partition.master_sys)
        # Keep a value snapshot so in-place coordinate changes invalidate the cache.
        partition.cached_coords .= coords
        partition.cached_master_state = state_tag
    end

    return partition.cached_master_pe
end

function evaluate_energy!(partition::AlchemicalPartition, coords, boundary, state_index::Int; 
                          force_recompute::Bool=false)
    n_states = length(partition.λ_hamiltonians)
    1 <= state_index <= n_states || throw(BoundsError(partition.λ_hamiltonians, state_index))
    master_state_index = partition.shared_master_neighbor_finder ? 1 : state_index
    master_neighbor_finder = partition.master_neighbor_finders[master_state_index]
    master_pe = refresh_master_energy!(
        partition,
        coords,
        boundary,
        master_state_index,
        master_neighbor_finder;
        force_recompute=force_recompute
    )

    pe_specific = evaluate_hamiltonian_energy!(
        partition,
        coords,
        boundary,
        partition.λ_hamiltonians[state_index],
        partition.λ_atoms[state_index],
        partition.λ_neighbor_finders[state_index],
    )
    
    return master_pe + pe_specific
end

function can_reuse_lambda_sys(partition::AlchemicalPartition, ham::LambdaHamiltonian, atoms, λ_neighbor_finder)
    λ_sys = partition.λ_sys
    return ham.pairwise_inters isa typeof(λ_sys.pairwise_inters) &&
           ham.specific_inter_lists isa typeof(λ_sys.specific_inter_lists) &&
           ham.general_inters isa typeof(λ_sys.general_inters) &&
           atoms isa typeof(λ_sys.atoms) &&
           λ_neighbor_finder isa typeof(λ_sys.neighbor_finder)
end

struct NoPrecomputedNeighbors end
const NO_PRECOMPUTED_NEIGHBORS = NoPrecomputedNeighbors()

function evaluate_hamiltonian_energy!(partition::AlchemicalPartition, coords, boundary,
                                      ham::LambdaHamiltonian, atoms, λ_neighbor_finder,
                                      precomputed_neighbors=NO_PRECOMPUTED_NEIGHBORS)
    if can_reuse_lambda_sys(partition, ham, atoms, λ_neighbor_finder)
        partition.λ_sys.coords = coords
        partition.λ_sys.boundary = boundary
        partition.λ_sys.atoms = atoms
        partition.λ_sys.pairwise_inters = ham.pairwise_inters
        partition.λ_sys.specific_inter_lists = ham.specific_inter_lists
        partition.λ_sys.general_inters = ham.general_inters
        partition.λ_sys.neighbor_finder = λ_neighbor_finder
        nbrs = if precomputed_neighbors === NO_PRECOMPUTED_NEIGHBORS
            find_neighbors(partition.λ_sys)
        else
            precomputed_neighbors
        end
        return potential_energy(partition.λ_sys, nbrs, 0)
    else
        tmp_sys = System(
            partition.λ_sys;
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=ham.pairwise_inters,
            specific_inter_lists=ham.specific_inter_lists,
            general_inters=ham.general_inters,
            neighbor_finder=λ_neighbor_finder,
            strictness=:nowarn,
        )
        nbrs = find_neighbors(tmp_sys)
        return potential_energy(tmp_sys, nbrs, 0)
    end
end

function evaluate_energy!(partition::AlchemicalPartition, coords, boundary, 
                          target_ham::LambdaHamiltonian, target_atoms; 
                          force_recompute::Bool=false,
                          target_master_neighbor_finder=partition.target_master_neighbor_finder,
                          target_λ_neighbor_finder=partition.target_λ_neighbor_finder)
    master_neighbor_finder = isnothing(target_master_neighbor_finder) ?
        partition.master_neighbor_finders[1] : target_master_neighbor_finder
    λ_neighbor_finder = isnothing(target_λ_neighbor_finder) ?
        partition.λ_neighbor_finders[1] : target_λ_neighbor_finder
    target_state_tag = length(partition.λ_hamiltonians) + 1
    master_pe = refresh_master_energy!(
        partition,
        coords,
        boundary,
        target_state_tag,
        master_neighbor_finder;
        force_recompute=force_recompute
    )

    pe_specific = evaluate_hamiltonian_energy!(
        partition, coords, boundary, target_ham, target_atoms, λ_neighbor_finder
    )
    
    return master_pe + pe_specific
end

"""
    evaluate_energy_all!(partition::AlchemicalPartition, coords, boundary)

Efficiently evaluates the potential energy of the current coordinates mapped across all 
thermodynamic states simultaneously, evaluating the unperturbed `master_sys` only once.
"""
function evaluate_energy_all!(partition::AlchemicalPartition, coords, boundary, energies)
    n_states = length(partition.λ_hamiltonians)
    length(energies) == n_states || throw(ArgumentError(
        "`energies` length $(length(energies)) does not match number of states $n_states."
    ))

    master_pe = zero(partition.cached_master_pe)
    if partition.shared_master_neighbor_finder
        master_pe = refresh_master_energy!(
            partition,
            coords,
            boundary,
            1,
            partition.master_neighbor_finders[1]
        )
    end

    reuse_lambda_nbrs = partition.shared_λ_neighbor_finder
    if reuse_lambda_nbrs
        @inbounds for state_index in eachindex(partition.λ_hamiltonians)
            if !can_reuse_lambda_sys(
                partition,
                partition.λ_hamiltonians[state_index],
                partition.λ_atoms[state_index],
                partition.λ_neighbor_finders[state_index],
            )
                reuse_lambda_nbrs = false
                break
            end
        end
    end

    precomputed_neighbors = NO_PRECOMPUTED_NEIGHBORS
    if reuse_lambda_nbrs
        partition.λ_sys.coords = coords
        partition.λ_sys.boundary = boundary
        partition.λ_sys.neighbor_finder = partition.λ_neighbor_finders[1]
        precomputed_neighbors = find_neighbors(partition.λ_sys)
    end

    @inbounds for state_index in eachindex(partition.λ_hamiltonians)
        local_master_pe = if partition.shared_master_neighbor_finder
            master_pe
        else
            refresh_master_energy!(
                partition,
                coords,
                boundary,
                state_index,
                partition.master_neighbor_finders[state_index];
                force_recompute=true,
            )
        end
        pe_specific = evaluate_hamiltonian_energy!(
            partition,
            coords,
            boundary,
            partition.λ_hamiltonians[state_index],
            partition.λ_atoms[state_index],
            partition.λ_neighbor_finders[state_index],
            precomputed_neighbors,
        )
        energies[state_index] = local_master_pe + pe_specific
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
