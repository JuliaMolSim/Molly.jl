export ActiveThermoState

# ExtendedStateSpace(thermo_states; reuse_neighbors=true)
#
# An expanded ensemble over a collection of thermodynamic states.
#
# `thermo_states` supplies the systems, integrators, temperatures, and optional
# pressures for each state. The resulting state space stores shared alchemical
# partition data for evaluating energies across states while preserving each
# state's interactions and integrator.
struct ExtendedStateSpace{P, TS, S, I, B, PR, HP, SPI, SSI, SGI}
    partition::P
    thermo_states::TS
    systems::S
    integrators::I
    betas::B
    pressures::PR
    has_pressure::HP
    state_pairwise_inters::SPI
    state_specific_inter_lists::SSI
    state_general_inters::SGI
end

function ExtendedStateSpace(thermo_states::AbstractArray{<:ThermoState};
                            reuse_neighbors::Bool=true,
                            build_λ_systems::Bool=true)
    isempty(thermo_states) && throw(ArgumentError("at least one ThermoState is required"))

    thermo_states_vec = collect(thermo_states)
    ref_sys = first(thermo_states_vec).system
    TH = float_type_high(ref_sys)
    partition = AlchemicalPartition(thermo_states_vec; reuse_neighbors=reuse_neighbors,
                                    build_λ_systems=build_λ_systems)

    systems = [ts.system for ts in thermo_states_vec]
    integrators = [ts.integrator for ts in thermo_states_vec]
    betas = [ts.beta for ts in thermo_states_vec]
    has_pressure = [!isnothing(ts.p) for ts in thermo_states_vec]
    pressures = [isnothing(ts.p) ? zero(TH) : ts.p for ts in thermo_states_vec]

    state_pairwise_inters = [ts.system.pairwise_inters for ts in thermo_states_vec]
    state_specific_inter_lists = [ts.system.specific_inter_lists for ts in thermo_states_vec]
    state_general_inters = [ts.system.general_inters for ts in thermo_states_vec]

    return ExtendedStateSpace(
        partition,
        thermo_states_vec,
        systems,
        integrators,
        betas,
        pressures,
        has_pressure,
        state_pairwise_inters,
        state_specific_inter_lists,
        state_general_inters,
    )
end

# The partition evaluates the energies of the states with frozen copies of the
#   interactions, so a changing bias would be ignored
function check_thermo_states_history_inters(thermo_states, sim_name)
    for ts in thermo_states
        check_no_history_inters(values(ts.system.general_inters), sim_name)
    end
    return nothing
end

function ExtendedStateSpace(space::ExtendedStateSpace;
                            partition=space.partition,
                            thermo_states=space.thermo_states,
                            systems=space.systems,
                            integrators=space.integrators,
                            state_pairwise_inters=space.state_pairwise_inters,
                            state_specific_inter_lists=space.state_specific_inter_lists,
                            state_general_inters=space.state_general_inters)
    return ExtendedStateSpace(
        partition,
        thermo_states,
        systems,
        integrators,
        space.betas,
        space.pressures,
        space.has_pressure,
        state_pairwise_inters,
        state_specific_inter_lists,
        state_general_inters,
    )
end

n_states(space::ExtendedStateSpace) = length(space.betas)

# Only the interactions of the carrier are read, but the base system and the finder are still
# constrained twice over: `neighbor_finder=nothing` is rejected whenever a pairwise interaction uses
# the neighbour list, and the base has to be on the same device as the interactions handed in, since
# `System` checks that before it applies `array_type`. So the base is the source-type system and its
# own finder comes along, which costs one mask conversion per call and is thrown away with it
function convert_state_inters(ref_sys, pairwise, specific, general, ::Type{AT}) where AT
    carrier = System(ref_sys;
        pairwise_inters      = pairwise,
        specific_inter_lists = specific,
        general_inters       = general,
        loggers              = (),
        array_type           = AT,
        strictness           = :nowarn,
    )
    return (carrier.pairwise_inters, carrier.specific_inter_lists, carrier.general_inters)
end

# The exclusion masks in the form the neighbour finders of the target device store them, so that
# every state shares one object instead of bit-packing its own: `to_bitmatrix` returns a `BitMatrix`
# unchanged, but builds a new one from the dense `Matrix{Bool}` that a device array converts to
function localize_mask(mask, ::Type{AT}) where AT
    if AT <: AbstractGPUArray
        # A device array can not be built from the `BitMatrix` that a host copy holds
        return to_device(mask isa BitArray ? Array(mask) : mask, AT)
    else
        return cpu_neighbor_matrix(mask)
    end
end

# An expanded ensemble holding only `states`, on the given array type. Per-state vectors keep their
# global indices and hold `nothing` elsewhere, so indexing them needs no remapping. The exclusion
# matrices are converted twice however many states are resident, and shared, as are the interactions
# of the states that share them. `build_λ_systems=false` gives a copy that can be shipped to a
# worker, which rebuilds the λ systems itself, but that can not evaluate an energy here
function localize_state_space(space::ExtendedStateSpace, states, ::Type{AT};
                              build_λ_systems::Bool=true) where AT
    resident = sort!(unique!(Int.(collect(states))))
    ref_sys = space.systems[first(resident)]
    reference = System(ref_sys; array_type=AT, loggers=())
    master = System(space.partition.master_sys; array_type=AT, loggers=())
    λ_eligible = localize_mask(space.partition.λ_eligible, AT)
    special_mask = localize_mask(space.partition.special_mask, AT)

    K = n_states(space)
    λ_systems, λ_atoms, hamiltonians = (Vector{Any}(nothing, K) for _ in 1:3)
    systems, integrators = (Vector{Any}(nothing, K) for _ in 1:2)
    pairwise, specific, general = (Vector{Any}(nothing, K) for _ in 1:3)
    # States built from one system hold the same interaction objects, and converting such a set once
    # keeps them shared: the key is the identity of the source, so no state is given interactions it
    # did not already share, and the mask that every conversion throws away is converted once too
    converted = IdDict{Any, Any}()
    convert_once(p, s, g) = get!(converted, (p, s, g)) do
        convert_state_inters(ref_sys, p, s, g, AT)
    end

    for k in resident
        source = space.partition.λ_hamiltonians[k]
        hamiltonians[k] = LambdaHamiltonian(convert_once(source.pairwise_inters,
            source.specific_inter_lists, source.general_inters)...)
        λ_atoms[k] = to_device(space.partition.λ_atoms[k], AT)
        if build_λ_systems
            # Every part is already on the target array type, so this converts nothing and the
            # masks are shared rather than copied per state
            λ_systems[k] = System(master;
                atoms                = λ_atoms[k],
                pairwise_inters      = hamiltonians[k].pairwise_inters,
                specific_inter_lists = hamiltonians[k].specific_inter_lists,
                general_inters       = hamiltonians[k].general_inters,
                neighbor_finder      = build_neighbor_finder(master.neighbor_finder, λ_eligible,
                                            special_mask;
                                            reuse_neighbors=space.partition.reuse_neighbors,
                                            boundary=master.boundary),
                loggers              = (),
            )
        end
        pairwise[k], specific[k], general[k] = convert_once(space.state_pairwise_inters[k],
            space.state_specific_inter_lists[k], space.state_general_inters[k])
        # One object at every resident index: `ActiveThermoState` deepcopies it, so nothing aliases
        systems[k] = reference
        integrators[k] = space.integrators[k]
    end

    return ExtendedStateSpace(space;
        partition = AlchemicalPartition(space.partition;
            master_sys     = master,
            λ_systems      = λ_systems,
            λ_atoms        = λ_atoms,
            λ_hamiltonians = hamiltonians,
            λ_eligible     = λ_eligible,
            special_mask   = special_mask,
        ),
        thermo_states = nothing,
        systems = systems,
        integrators = integrators,
        state_pairwise_inters = pairwise,
        state_specific_inter_lists = specific,
        state_general_inters = general,
    )
end

mutable struct PartitionedReducedPotentialWorkspace{P, TS, E}
    partition::P
    thermo_states::TS
    energies::E
end

function PartitionedReducedPotentialWorkspace(thermo_states::AbstractArray{<:ThermoState};
                                              reuse_neighbors::Bool=true)
    thermo_states_vec = collect(thermo_states)
    if isempty(thermo_states_vec)
        throw(ArgumentError("at least one ThermoState is required"))
    end
    partition = AlchemicalPartition(thermo_states_vec; reuse_neighbors=reuse_neighbors)
    energies = Vector{typeof(partition.cached_master_pe)}(undef, length(thermo_states_vec))
    return PartitionedReducedPotentialWorkspace(partition, thermo_states_vec, energies)
end

function partitioned_workspace_energy_view(workspace::PartitionedReducedPotentialWorkspace,
                                            n::Integer)
    length(workspace.energies) < n && resize!(workspace.energies, n)
    return @view workspace.energies[1:n]
end

"""
    ActiveThermoState(space, first_state=1; loggers=())

Mutable active state for simulations over the thermodynamic states of a state space,
such as the `state_space` field of a [`TSSState`](@ref) or an [`AWHState`](@ref).

The active state owns a simulation system and integrator corresponding to the
current thermodynamic state index, starting at `first_state`, and is moved between
states without reallocating the active wrapper.
`loggers` are attached to the active system.
For example, explicit active states can be given as `replica_active_states` to
[`TSSSimulation`](@ref).
"""
mutable struct ActiveThermoState{S, I}
    active_idx::Int
    active_sys::S
    active_integrator::I
end

function ActiveThermoState(space::ExtendedStateSpace, first_state::Integer=1; loggers=())
    if !(1 <= first_state <= n_states(space))
        throw(ArgumentError("first_state ($first_state) out of range 1:$(n_states(space))"))
    end

    ref_sys = space.systems[first_state]
    active_sys = System(deepcopy(ref_sys);
        atoms = space.partition.λ_atoms[first_state],
        pairwise_inters = space.state_pairwise_inters[first_state],
        specific_inter_lists = space.state_specific_inter_lists[first_state],
        general_inters = space.state_general_inters[first_state],
        loggers = loggers,
    )
    return ActiveThermoState{typeof(active_sys), eltype(typeof(space.integrators))}(
        first_state,
        active_sys,
        space.integrators[first_state],
    )
end

function sync_active_state_dynamics!(dest::ActiveThermoState, src::ActiveThermoState)
    dest.active_sys.coords .= src.active_sys.coords
    dest.active_sys.velocities .= src.active_sys.velocities
    dest.active_sys.boundary = src.active_sys.boundary
    return dest
end

# set_active_state!(active::ActiveThermoState, space::ExtendedStateSpace, state_index)
#
# Retarget an ActiveThermoState to another thermodynamic state.
#
# The active state index, atoms, interactions, and integrator are updated from
# `space`. Coordinates, velocities, and boundary already stored in the active
# system are preserved.
function set_active_state!(active::ActiveThermoState,
                           space::ExtendedStateSpace,
                           state_index::Integer)
    if !(1 <= state_index <= n_states(space))
        throw(ArgumentError("state_index ($state_index) out of range 1:$(n_states(space))"))
    end

    active.active_idx = Int(state_index)
    active.active_sys.atoms = space.partition.λ_atoms[state_index]
    active.active_sys.pairwise_inters = space.state_pairwise_inters[state_index]
    active.active_sys.specific_inter_lists = space.state_specific_inter_lists[state_index]
    active.active_sys.general_inters = space.state_general_inters[state_index]
    active.active_integrator = space.integrators[state_index]
    return active
end

# evaluate_energy_subset!(energies, partition::AlchemicalPartition,
#                         coords, boundary, state_indices)
#
# Fill `energies` with total potential energies for selected alchemical states.
#
# The unperturbed master energy is evaluated once for `coords` and `boundary`;
# the state-specific energy contribution is then evaluated for each entry in
# `state_indices`. The output vector length must match `state_indices`.
function evaluate_energy_subset!(energies::AbstractVector,
                                 partition::AlchemicalPartition,
                                 coords,
                                 boundary,
                                 state_indices;
                                 n_threads::Integer=Threads.nthreads())
    if length(energies) != length(state_indices)
        throw(DimensionMismatch("energies length ($(length(energies))) must match " *
                                "state_indices length ($(length(state_indices)))"))
    end

    update_master_energy!(partition, coords, boundary;
                          force_recompute=true, n_threads=n_threads)

    @inbounds for (out_i, state_index) in pairs(state_indices)
        if !(1 <= state_index <= length(partition.λ_systems))
            throw(ArgumentError("state_index ($state_index) out of range " *
                                "1:$(length(partition.λ_systems))"))
        end
        pe_specific = evaluate_λ_energy!(
            partition.λ_systems[state_index],
            coords,
            boundary,
            n_threads = n_threads,
        )
        energies[out_i] = partition.cached_master_pe + pe_specific
    end

    return energies
end

# evaluate_energy_subset(partition::AlchemicalPartition, coords, boundary, state_indices)
#
# Return total potential energies for selected alchemical states.
#
# This allocating wrapper calls evaluate_energy_subset! with a freshly created
# output vector.
function evaluate_energy_subset(partition::AlchemicalPartition,
                                coords,
                                boundary,
                                state_indices;
                                n_threads::Integer=Threads.nthreads())
    energies = Vector{typeof(partition.cached_master_pe)}(undef, length(state_indices))
    return evaluate_energy_subset!(energies, partition, coords, boundary, state_indices;
                                   n_threads=n_threads)
end

@inline function safe_ustrip(::Type{T}, x) where T
    val = T(ustrip(x))
    return isnan(val) ? typemax(T) : val
end

# reduced_potential(space::ExtendedStateSpace, energy, boundary, state_index)
# reduced_potential(state::ThermoState, energy, boundary)
# reduced_potential(workspace, coords, boundary, state_index)
#
# Convert an energy, and optional pressure-volume term, to a dimensionless
# reduced potential.
#
# For `ExtendedStateSpace` and `ThermoState` inputs the supplied `energy` is used
# directly. For a partitioned workspace, the energy for `coords`, `boundary`, and
# `state_index` is evaluated before conversion.
function reduced_potential(space::ExtendedStateSpace,
                           energy,
                           boundary,
                           state_index::Integer)
    T = typeof(space.betas[state_index])
    red = space.betas[state_index] * safe_ustrip(T, energy)
    if space.has_pressure[state_index]
        red += space.betas[state_index] *
               safe_ustrip(T, space.pressures[state_index] * volume(boundary))
    end
    return red
end

function reduced_potential(state::ThermoState, energy, boundary)
    T = typeof(state.beta)
    red = state.beta * safe_ustrip(T, energy)
    if !isnothing(state.p)
        red += state.beta * safe_ustrip(T, state.p * volume(boundary))
    end
    return red
end

function reduced_potential(workspace::PartitionedReducedPotentialWorkspace,
                           coords,
                           boundary,
                           state_index::Integer;
                           n_threads::Integer=Threads.nthreads())
    if !(1 <= state_index <= length(workspace.thermo_states))
        throw(ArgumentError("state_index ($state_index) out of range " *
                            "1:$(length(workspace.thermo_states))"))
    end
    energy = evaluate_energy!(
        workspace.partition,
        coords,
        boundary,
        Int(state_index);
        force_recompute = true,
        n_threads = n_threads,
    )
    return reduced_potential(workspace.thermo_states[state_index], energy, boundary)
end

# reduced_potentials!(out, energies, space::ExtendedStateSpace, boundary, state_indices)
# reduced_potentials!(out, workspace, coords, boundary, state_indices)
# reduced_potentials!(out, space::ExtendedStateSpace, coords, boundary, state_indices)
#
# Fill `out` with reduced potentials for selected thermodynamic states.
#
# Depending on the method, potential energies are supplied explicitly in
# `energies`, evaluated through a reusable workspace, or evaluated from
# `space.partition`. The output length must match `state_indices`.
function reduced_potentials!(out::AbstractVector,
                             energies::AbstractVector,
                             space::ExtendedStateSpace,
                             boundary,
                             state_indices)
    if length(out) != length(state_indices)
        throw(DimensionMismatch("out length ($(length(out))) must match " *
                                "state_indices length ($(length(state_indices)))"))
    end
    if length(energies) != length(state_indices)
        throw(DimensionMismatch("energies length ($(length(energies))) must match " *
                                "state_indices length ($(length(state_indices)))"))
    end

    @inbounds for (out_i, state_index) in pairs(state_indices)
        out[out_i] = reduced_potential(space, energies[out_i], boundary, state_index)
    end
    return out
end

function reduced_potentials!(out::AbstractVector,
                             workspace::PartitionedReducedPotentialWorkspace,
                             coords,
                             boundary,
                             state_indices=Base.OneTo(length(workspace.thermo_states));
                             n_threads::Integer=Threads.nthreads())
    if length(out) != length(state_indices)
        throw(DimensionMismatch("out length ($(length(out))) must match " *
                                "state_indices length ($(length(state_indices)))"))
    end
    energies = partitioned_workspace_energy_view(workspace, length(state_indices))
    evaluate_energy_subset!(
        energies,
        workspace.partition,
        coords,
        boundary,
        state_indices,
        n_threads = n_threads,
    )
    @inbounds for (out_i, state_index) in pairs(state_indices)
        out[out_i] = reduced_potential(
            workspace.thermo_states[state_index],
            energies[out_i],
            boundary,
        )
    end
    return out
end

function reduced_potentials!(out::AbstractVector,
                             space::ExtendedStateSpace,
                             coords,
                             boundary,
                             state_indices=Base.OneTo(n_states(space)))
    energies = evaluate_energy_subset(space.partition, coords, boundary, state_indices)
    reduced_potentials!(out, energies, space, boundary, state_indices)
    return out
end

# conditional_state_weights!(weights, log_state_bias, reduced_potentials, scratch)
#
# Normalize conditional expanded-ensemble state weights in log space.
#
# The normalized weights are proportional to
# `exp(log_state_bias .- reduced_potentials)`. `scratch` is used as temporary
# storage and all input/output vectors must have matching lengths.
function conditional_state_weights!(weights::AbstractVector,
                                    log_state_bias::AbstractVector,
                                    reduced_potentials::AbstractVector,
                                    scratch::AbstractVector)
    if !(length(weights) == length(log_state_bias) == length(reduced_potentials) == length(scratch))
        throw(DimensionMismatch("weights, log_state_bias, reduced_potentials, and scratch " *
                                "must have matching lengths"))
    end

    @. scratch = log_state_bias - reduced_potentials
    log_den = logsumexp(scratch)
    @. weights = exp(scratch - log_den)
    s = sum(weights)
    @. weights /= s
    return weights
end

# sample_state(weights)
# sample_state(rng, weights)
#
# Sample a state index from normalized state weights.
#
# The returned index is in `1:length(weights)`. Pass an explicit random number
# generator for reproducible sampling.
sample_state(weights::AbstractVector) = sample(1:length(weights), Weights(weights))
sample_state(rng::AbstractRNG, weights::AbstractVector) = sample(rng, 1:length(weights), Weights(weights))
