export
    ExtendedStateSpace,
    ActiveThermoState,
    n_states,
    set_active_state!,
    evaluate_energy_subset,
    evaluate_energy_subset!,
    reduced_potential,
    reduced_potentials!,
    conditional_state_weights!,
    sample_state

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
                            reuse_neighbors::Bool=true)
    isempty(thermo_states) && throw(ArgumentError("at least one ThermoState is required"))

    thermo_states_vec = collect(thermo_states)
    ref_sys = first(thermo_states_vec).system
    FT = typeof(ustrip(ref_sys.total_mass))
    partition = AlchemicalPartition(thermo_states_vec; reuse_neighbors=reuse_neighbors)

    systems = [ts.system for ts in thermo_states_vec]
    integrators = [ts.integrator for ts in thermo_states_vec]
    betas = [ts.beta for ts in thermo_states_vec]
    has_pressure = [!isnothing(ts.p) for ts in thermo_states_vec]
    pressures = [isnothing(ts.p) ? zero(FT) : ts.p for ts in thermo_states_vec]

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

n_states(space::ExtendedStateSpace) = length(space.betas)

mutable struct PartitionedReducedPotentialWorkspace{P, TS, E}
    partition::P
    thermo_states::TS
    energies::E
end

function PartitionedReducedPotentialWorkspace(thermo_states::AbstractArray{<:ThermoState};
                                              reuse_neighbors::Bool=true)
    thermo_states_vec = collect(thermo_states)
    isempty(thermo_states_vec) &&
        throw(ArgumentError("at least one ThermoState is required"))
    partition = AlchemicalPartition(thermo_states_vec; reuse_neighbors=reuse_neighbors)
    energies = Vector{typeof(partition.cached_master_pe)}(undef, length(thermo_states_vec))
    return PartitionedReducedPotentialWorkspace(partition, thermo_states_vec, energies)
end

function _partitioned_workspace_energy_view(workspace::PartitionedReducedPotentialWorkspace,
                                            n::Integer)
    length(workspace.energies) < n && resize!(workspace.energies, n)
    return @view workspace.energies[1:n]
end

mutable struct ActiveThermoState{S, I}
    active_idx::Int
    active_sys::S
    active_integrator::I
end

function ActiveThermoState(space::ExtendedStateSpace, first_state::Integer=1)
    1 <= first_state <= n_states(space) ||
        throw(ArgumentError("first_state ($first_state) out of range 1:$(n_states(space))"))

    ref_sys = space.systems[first_state]
    active_sys = System(deepcopy(ref_sys);
        atoms = space.partition.λ_atoms[first_state],
        pairwise_inters = space.state_pairwise_inters[first_state],
        specific_inter_lists = space.state_specific_inter_lists[first_state],
        general_inters = space.state_general_inters[first_state],
    )
    return ActiveThermoState{typeof(active_sys), eltype(typeof(space.integrators))}(
        first_state,
        active_sys,
        space.integrators[first_state],
    )
end

function set_active_state!(active::ActiveThermoState,
                           space::ExtendedStateSpace,
                           state_index::Integer)
    1 <= state_index <= n_states(space) ||
        throw(ArgumentError("state_index ($state_index) out of range 1:$(n_states(space))"))

    active.active_idx = Int(state_index)
    active.active_sys.atoms = space.partition.λ_atoms[state_index]
    active.active_sys.pairwise_inters = space.state_pairwise_inters[state_index]
    active.active_sys.specific_inter_lists = space.state_specific_inter_lists[state_index]
    active.active_sys.general_inters = space.state_general_inters[state_index]
    active.active_integrator = space.integrators[state_index]
    return active
end

function evaluate_energy_subset!(energies::AbstractVector,
                                 partition::AlchemicalPartition,
                                 coords,
                                 boundary,
                                 state_indices)
    length(energies) == length(state_indices) ||
        throw(DimensionMismatch("energies length ($(length(energies))) must match " *
                                "state_indices length ($(length(state_indices)))"))

    _update_master_energy!(partition, coords, boundary; force_recompute=true)

    @inbounds for (out_i, state_index) in pairs(state_indices)
        1 <= state_index <= length(partition.λ_systems) ||
            throw(ArgumentError("state_index ($state_index) out of range " *
                                "1:$(length(partition.λ_systems))"))
        pe_specific = _evaluate_λ_energy!(
            partition.λ_systems[state_index],
            coords,
            boundary,
        )
        energies[out_i] = partition.cached_master_pe + pe_specific
    end

    return energies
end

function evaluate_energy_subset(partition::AlchemicalPartition,
                                coords,
                                boundary,
                                state_indices)
    energies = Vector{typeof(partition.cached_master_pe)}(undef, length(state_indices))
    return evaluate_energy_subset!(energies, partition, coords, boundary, state_indices)
end

@inline function _safe_ustrip(::Type{T}, x) where T
    val = T(ustrip(x))
    return isnan(val) ? typemax(T) : val
end

function reduced_potential(space::ExtendedStateSpace,
                           energy,
                           boundary,
                           state_index::Integer)
    T = typeof(space.betas[state_index])
    red = space.betas[state_index] * _safe_ustrip(T, energy)
    if space.has_pressure[state_index]
        red += space.betas[state_index] *
               _safe_ustrip(T, space.pressures[state_index] * volume(boundary))
    end
    return red
end

function reduced_potential(state::ThermoState, energy, boundary)
    T = typeof(state.beta)
    red = state.beta * _safe_ustrip(T, energy)
    if !isnothing(state.p)
        red += state.beta * _safe_ustrip(T, state.p * volume(boundary))
    end
    return red
end

function reduced_potential(workspace::PartitionedReducedPotentialWorkspace,
                           coords,
                           boundary,
                           state_index::Integer)
    1 <= state_index <= length(workspace.thermo_states) ||
        throw(ArgumentError("state_index ($state_index) out of range " *
                            "1:$(length(workspace.thermo_states))"))
    energy = evaluate_energy!(
        workspace.partition,
        coords,
        boundary,
        Int(state_index);
        force_recompute = true,
    )
    return reduced_potential(workspace.thermo_states[state_index], energy, boundary)
end

function reduced_potentials!(out::AbstractVector,
                             energies::AbstractVector,
                             space::ExtendedStateSpace,
                             boundary,
                             state_indices)
    length(out) == length(state_indices) ||
        throw(DimensionMismatch("out length ($(length(out))) must match " *
                                "state_indices length ($(length(state_indices)))"))
    length(energies) == length(state_indices) ||
        throw(DimensionMismatch("energies length ($(length(energies))) must match " *
                                "state_indices length ($(length(state_indices)))"))

    @inbounds for (out_i, state_index) in pairs(state_indices)
        out[out_i] = reduced_potential(space, energies[out_i], boundary, state_index)
    end
    return out
end

function reduced_potentials!(out::AbstractVector,
                             workspace::PartitionedReducedPotentialWorkspace,
                             coords,
                             boundary,
                             state_indices=Base.OneTo(length(workspace.thermo_states)))
    length(out) == length(state_indices) ||
        throw(DimensionMismatch("out length ($(length(out))) must match " *
                                "state_indices length ($(length(state_indices)))"))
    energies = _partitioned_workspace_energy_view(workspace, length(state_indices))
    evaluate_energy_subset!(
        energies,
        workspace.partition,
        coords,
        boundary,
        state_indices,
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

function conditional_state_weights!(weights::AbstractVector,
                                    log_state_bias::AbstractVector,
                                    reduced_potentials::AbstractVector,
                                    scratch::AbstractVector)
    length(weights) == length(log_state_bias) == length(reduced_potentials) == length(scratch) ||
        throw(DimensionMismatch("weights, log_state_bias, reduced_potentials, and scratch " *
                                "must have matching lengths"))

    @. scratch = log_state_bias - reduced_potentials
    log_den = logsumexp(scratch)
    @. weights = exp(scratch - log_den)
    s = sum(weights)
    @. weights /= s
    return weights
end

sample_state(weights::AbstractVector) = sample(1:length(weights), Weights(weights))
sample_state(rng::AbstractRNG, weights::AbstractVector) = sample(rng, 1:length(weights), Weights(weights))
