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

struct ExtendedStateSpace{P, S, I, B, PR, HP, SPI, SSI, SGI}
    partition::P
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

    ref_sys = first(thermo_states).system
    FT = typeof(ustrip(ref_sys.total_mass))
    partition = AlchemicalPartition(thermo_states; reuse_neighbors=reuse_neighbors)

    systems = [ts.system for ts in thermo_states]
    integrators = [ts.integrator for ts in thermo_states]
    betas = [ts.beta for ts in thermo_states]
    has_pressure = [!isnothing(ts.p) for ts in thermo_states]
    pressures = [isnothing(ts.p) ? zero(FT) : ts.p for ts in thermo_states]

    state_pairwise_inters = [ts.system.pairwise_inters for ts in thermo_states]
    state_specific_inter_lists = [ts.system.specific_inter_lists for ts in thermo_states]
    state_general_inters = [ts.system.general_inters for ts in thermo_states]

    return ExtendedStateSpace(
        partition,
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

    partition.master_sys.coords = coords
    partition.master_sys.boundary = boundary
    partition.cached_master_pe = potential_energy(partition.master_sys)
    partition.cached_coords = coords

    partition.λ_sys.coords = coords
    partition.λ_sys.boundary = boundary
    nbrs = find_neighbors(partition.λ_sys)

    @inbounds for (out_i, state_index) in pairs(state_indices)
        partition.λ_sys.atoms = partition.λ_atoms[state_index]
        partition.λ_sys.pairwise_inters = partition.λ_hamiltonians[state_index].pairwise_inters
        partition.λ_sys.specific_inter_lists = partition.λ_hamiltonians[state_index].specific_inter_lists
        partition.λ_sys.general_inters = partition.λ_hamiltonians[state_index].general_inters

        pe_specific = potential_energy(partition.λ_sys, nbrs, 0)
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
