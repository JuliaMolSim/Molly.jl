# -----------------------------------------------------------------------------
# Unit stripping helpers for late-defined Molly types
# -----------------------------------------------------------------------------

@inline _strip_units(x) = ustrip(x)
@inline _strip_units(::Nothing) = nothing
@inline _strip_units(x::Tuple) = map(_strip_units, x)
@inline _strip_units(x::NamedTuple) = map(_strip_units, x)

@inline function _strip_units_array(x)
    AT = array_type(x)
    return to_device(_maybe_strip_units.(from_device(x)), AT)
end

@inline _maybe_strip_units(::Nothing) = nothing
@inline _maybe_strip_units(x::Tuple) = map(_maybe_strip_units, x)
@inline _maybe_strip_units(x::NamedTuple) = map(_maybe_strip_units, x)
@inline _maybe_strip_units(x::CircularBuffer) = ustrip(x)
@inline _maybe_strip_units(x::StaticArray) = ustrip(x)
@inline _maybe_strip_units(x::AbstractArray) = _strip_units_array(x)
@inline _maybe_strip_units(x) = (applicable(ustrip, x) ? _strip_units(x) : x)

function Unitful.ustrip(cb::CircularBuffer)
    stripped_values = _maybe_strip_units.(collect(cb))
    stripped = CircularBuffer{eltype(stripped_values)}(capacity(cb))
    append!(stripped, stripped_values)
    return stripped
end

Unitful.ustrip(a::Atom) = Atom(
    index = a.index,
    atom_type = a.atom_type,
    mass = ustrip(a.mass),
    charge = ustrip(a.charge),
    σ = ustrip(a.σ),
    ϵ = ustrip(a.ϵ),
    λ = a.λ,
    alch_role = a.alch_role,
)

function Unitful.ustrip(il::InteractionList1Atoms)
    AT = array_type(il.inters)
    unitless_inters = to_device(ustrip.(from_device(il.inters)), AT)
    return InteractionList1Atoms(il.is, unitless_inters, il.types)
end

function Unitful.ustrip(il::InteractionList2Atoms)
    AT = array_type(il.inters)
    unitless_inters = to_device(ustrip.(from_device(il.inters)), AT)
    return InteractionList2Atoms(il.is, il.js, unitless_inters, il.types)
end

function Unitful.ustrip(il::InteractionList3Atoms)
    AT = array_type(il.inters)
    unitless_inters = to_device(ustrip.(from_device(il.inters)), AT)
    return InteractionList3Atoms(il.is, il.js, il.ks, unitless_inters, il.types)
end

function Unitful.ustrip(il::InteractionList4Atoms)
    AT = array_type(il.inters)
    unitless_inters = to_device(ustrip.(from_device(il.inters)), AT)
    return InteractionList4Atoms(il.is, il.js, il.ks, il.ls, unitless_inters, il.types)
end

function Unitful.ustrip(logger::GeneralObservableLogger)
    history = _maybe_strip_units(logger.history)
    return GeneralObservableLogger{eltype(history), typeof(logger.observable)}(
        logger.observable,
        logger.n_steps,
        history,
    )
end

function Unitful.ustrip(logger::DisplacementsLogger)
    displacements = _maybe_strip_units(logger.displacements)
    coords_ref = _maybe_strip_units(logger.coords_ref)
    last_displacements = _maybe_strip_units(logger.last_displacements)
    return DisplacementsLogger{typeof(displacements), typeof(coords_ref)}(
        displacements,
        coords_ref,
        last_displacements,
        logger.n_steps,
        logger.n_steps_update,
    )
end

function Unitful.ustrip(logger::TrajectoryWriter)
    atom_inds = _maybe_strip_units(logger.atom_inds)
    topology = _maybe_strip_units(logger.topology)
    return TrajectoryWriter{typeof(atom_inds), typeof(topology)}(
        logger.n_steps,
        logger.filepath,
        logger.format,
        logger.correction,
        atom_inds,
        copy(logger.excluded_res),
        logger.write_velocities,
        logger.write_boundary,
        topology,
        logger.topology_written,
        logger.structure_n,
    )
end

function Unitful.ustrip(logger::TimeCorrelationLogger)
    history_A = ustrip(logger.history_A)
    history_B = ustrip(logger.history_B)
    sum_offset_products = _maybe_strip_units(logger.sum_offset_products)
    sum_A = _maybe_strip_units(logger.sum_A)
    sum_B = _maybe_strip_units(logger.sum_B)
    sum_sq_A = _maybe_strip_units(logger.sum_sq_A)
    sum_sq_B = _maybe_strip_units(logger.sum_sq_B)
    return TimeCorrelationLogger{
        typeof(sum_A),
        typeof(sum_sq_A),
        typeof(sum_B),
        typeof(sum_sq_B),
        eltype(sum_offset_products),
        typeof(logger.observableA),
        typeof(logger.observableB),
    }(
        logger.observableA,
        logger.observableB,
        logger.n_correlation,
        history_A,
        history_B,
        sum_offset_products,
        logger.n_timesteps,
        sum_A,
        sum_B,
        sum_sq_A,
        sum_sq_B,
    )
end

function Unitful.ustrip(logger::AverageObservableLogger)
    block_averages = _maybe_strip_units(logger.block_averages)
    current_block = _maybe_strip_units(logger.current_block)
    return AverageObservableLogger{eltype(block_averages), typeof(logger.observable)}(
        logger.observable,
        logger.n_steps,
        logger.n_blocks,
        logger.current_block_size,
        block_averages,
        current_block,
    )
end

function Unitful.ustrip(logger::ReplicaExchangeLogger)
    deltas = _maybe_strip_units(logger.deltas)
    return ReplicaExchangeLogger{eltype(deltas)}(
        logger.n_replicas,
        logger.n_attempts,
        logger.n_exchanges,
        copy(logger.indices),
        copy(logger.steps),
        deltas,
        logger.end_step,
    )
end

function Unitful.ustrip(logger::MonteCarloLogger)
    energy_rates = _maybe_strip_units(logger.energy_rates)
    return MonteCarloLogger{eltype(energy_rates)}(
        logger.n_trials,
        logger.n_accept,
        energy_rates,
        copy(logger.state_changed),
    )
end

function Unitful.ustrip(logger::AWHEnsembleLogger)
    coords_history = _maybe_strip_units(logger.coords_history)
    volume_history = _maybe_strip_units(logger.volume_history)
    potential_energy_history = _maybe_strip_units(logger.potential_energy_history)
    return AWHEnsembleLogger{
        eltype(eltype(eltype(coords_history))),
        eltype(volume_history),
        eltype(potential_energy_history),
    }(
        logger.n_steps,
        logger.global_step,
        logger.should_log,
        logger.active_idx,
        copy(logger.active_idx_history),
        coords_history,
        volume_history,
        potential_energy_history,
    )
end

function Unitful.ustrip(sys::System)
    AT = array_type(sys.coords)

    # Safely strip units on the CPU for all arrays to prevent GPU kernel crashes.
    unitless_atoms  = to_device(ustrip.(from_device(sys.atoms)), AT)
    unitless_coords = to_device(ustrip.(from_device(sys.coords)), AT)
    unitless_vels   = to_device(ustrip.(from_device(sys.velocities)), AT)

    return System(
        atoms = unitless_atoms,
        atoms_data = sys.atoms_data,
        topology = sys.topology,
        pairwise_inters = _strip_units(sys.pairwise_inters),
        specific_inter_lists = _strip_units(sys.specific_inter_lists),
        general_inters = _strip_units(sys.general_inters),
        constraints = _strip_units(sys.constraints),
        coords = unitless_coords,
        velocities = unitless_vels,
        boundary = ustrip(sys.boundary),
        virtual_sites = _strip_units_array(sys.virtual_sites),
        neighbor_finder = _strip_units(sys.neighbor_finder),
        loggers = _maybe_strip_units(sys.loggers),
        force_units = NoUnits,
        energy_units = NoUnits,
        k = _strip_units(sys.k),
        data = _maybe_strip_units(sys.data),
        launch_config = sys.launch_config,
        nonbonded_energy_type = sys.nonbonded_energy_type,
    )
end
