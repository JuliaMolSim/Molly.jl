# -----------------------------------------------------------------------------
# Device transfer helpers for late-defined Molly types
# -----------------------------------------------------------------------------

from_device(x::Array) = x
from_device(x) = Array(x)

to_device(x::Array, ::Type{<:Array}) = x
to_device(x::Array, ::Type{AT}) where {AT <: AbstractGPUArray} =
    (isbitstype(eltype(x)) ? AT(x) : x)
to_device(x, ::Type{AT}) where AT = AT(x)

@inline _transfer_storage(x, ::Type{AT}) where {AT} = to_device(from_device(x), AT)

@inline function _rebuild_structarray(x::StructArrays.StructArray)
    return StructArray{eltype(x)}(map(from_device, StructArrays.components(x)))
end

@inline _rebuild_structarray(x) = from_device(x)

@inline function _rebuild_structarray(x::StructArrays.StructArray, ::Type{AT}) where {AT}
    return StructArray{eltype(x)}(map(c -> to_device(from_device(c), AT), StructArrays.components(x)))
end

@inline _rebuild_structarray(x, ::Type{AT}) where {AT} = x

from_device(il::InteractionList1Atoms) = InteractionList1Atoms(from_device(il.is), from_device(il.inters), il.types)
from_device(il::InteractionList2Atoms) = InteractionList2Atoms(from_device(il.is), from_device(il.js), from_device(il.inters), il.types)
from_device(il::InteractionList3Atoms) = InteractionList3Atoms(from_device(il.is), from_device(il.js), from_device(il.ks), from_device(il.inters), il.types)
from_device(il::InteractionList4Atoms) = InteractionList4Atoms(from_device(il.is), from_device(il.js), from_device(il.ks), from_device(il.ls), from_device(il.inters), il.types)

to_device(il::InteractionList1Atoms, ::Type{AT}) where {AT} = InteractionList1Atoms(to_device(il.is, AT), to_device(il.inters, AT), il.types)
to_device(il::InteractionList2Atoms, ::Type{AT}) where {AT} = InteractionList2Atoms(to_device(il.is, AT), to_device(il.js, AT), to_device(il.inters, AT), il.types)
to_device(il::InteractionList3Atoms, ::Type{AT}) where {AT} = InteractionList3Atoms(to_device(il.is, AT), to_device(il.js, AT), to_device(il.ks, AT), to_device(il.inters, AT), il.types)
to_device(il::InteractionList4Atoms, ::Type{AT}) where {AT} = InteractionList4Atoms(to_device(il.is, AT), to_device(il.js, AT), to_device(il.ks, AT), to_device(il.ls, AT), to_device(il.inters, AT), il.types)

_from_device_inter_lists(x::Tuple) = map(from_device, x)
_from_device_inter_lists(x::NamedTuple) = map(from_device, x)
_from_device_inter_lists(x) = x

_to_device_inter_lists(x::Tuple, ::Type{AT}) where {AT} = map(i -> to_device(i, AT), x)
_to_device_inter_lists(x::NamedTuple, ::Type{AT}) where {AT} = map(i -> to_device(i, AT), x)
_to_device_inter_lists(x, ::Type{AT}) where {AT} = x

@inline function _maybe_from_device(x)
    fallback = which(from_device, Tuple{Any})
    if x isa AbstractArray || which(from_device, Tuple{typeof(x)}) !== fallback
        return from_device(x)
    end
    return x
end

@inline function _maybe_to_device(x, ::Type{AT}) where {AT}
    fallback = which(to_device, Tuple{Any, Type{AT}})
    if x isa AbstractArray || which(to_device, Tuple{typeof(x), Type{AT}}) !== fallback
        return to_device(x, AT)
    end
    return x
end

_from_device_system_fields(x::Tuple) = map(_maybe_from_device, x)
_from_device_system_fields(x::NamedTuple) = map(_maybe_from_device, x)
_from_device_system_fields(x) = _maybe_from_device(x)

_to_device_system_fields(x::Tuple, ::Type{AT}) where {AT} = map(i -> _maybe_to_device(i, AT), x)
_to_device_system_fields(x::NamedTuple, ::Type{AT}) where {AT} = map(i -> _maybe_to_device(i, AT), x)
_to_device_system_fields(x, ::Type{AT}) where {AT} = _maybe_to_device(x, AT)

function from_device(sys::System)
    return System(
        atoms = from_device(sys.atoms),
        coords = from_device(sys.coords),
        boundary = sys.boundary,
        velocities = from_device(sys.velocities),
        atoms_data = sys.atoms_data,
        topology = sys.topology,
        pairwise_inters = _from_device_system_fields(sys.pairwise_inters),
        specific_inter_lists = _from_device_inter_lists(sys.specific_inter_lists),
        general_inters = _from_device_system_fields(sys.general_inters),
        constraints = _from_device_system_fields(sys.constraints),
        virtual_sites = from_device(sys.virtual_sites),
        neighbor_finder = _maybe_from_device(sys.neighbor_finder),
        loggers = _from_device_system_fields(sys.loggers),
        force_units = sys.force_units,
        energy_units = sys.energy_units,
        k = sys.k,
        data = sys.data,
        launch_config = sys.launch_config,
        nonbonded_energy_type = sys.nonbonded_energy_type,
    )
end

function to_device(sys::System, ::Type{AT}) where {AT}
    return System(
        atoms = to_device(sys.atoms, AT),
        coords = to_device(sys.coords, AT),
        boundary = sys.boundary,
        velocities = to_device(sys.velocities, AT),
        atoms_data = sys.atoms_data,
        topology = sys.topology,
        pairwise_inters = _to_device_system_fields(sys.pairwise_inters, AT),
        specific_inter_lists = _to_device_inter_lists(sys.specific_inter_lists, AT),
        general_inters = _to_device_system_fields(sys.general_inters, AT),
        constraints = _from_device_system_fields(sys.constraints),
        virtual_sites = to_device(sys.virtual_sites, AT),
        neighbor_finder = _maybe_to_device(sys.neighbor_finder, AT),
        loggers = _to_device_system_fields(sys.loggers, AT),
        force_units = sys.force_units,
        energy_units = sys.energy_units,
        k = sys.k,
        data = sys.data,
        launch_config = sys.launch_config,
        nonbonded_energy_type = sys.nonbonded_energy_type,
    )
end

from_device(::NoNeighborFinder) = NoNeighborFinder()
to_device(::NoNeighborFinder, ::Type{AT}) where {AT} = NoNeighborFinder()

from_device(nf::GPUNeighborFinder) = DistanceNeighborFinder(
    eligible = neighbor_finder_masks(nf, nf.n_atoms)[1],
    dist_cutoff = nf.dist_cutoff,
    special = neighbor_finder_masks(nf, nf.n_atoms)[2],
    n_steps = nf.n_steps_reorder,
)

from_device(nf::DistanceNeighborFinder) = DistanceNeighborFinder(
    eligible = neighbor_finder_masks(nf)[1],
    dist_cutoff = nf.dist_cutoff,
    special = neighbor_finder_masks(nf)[2],
    n_steps = nf.n_steps,
)

from_device(nf::TreeNeighborFinder) = nf
from_device(nf::CellListMapNeighborFinder) = nf

to_device(nf::GPUNeighborFinder, ::Type{AT}) where {AT} = to_device(from_device(nf), AT)

function to_device(nf::DistanceNeighborFinder, ::Type{AT}) where {AT}
    eligible_cpu, special_cpu = neighbor_finder_masks(nf)
    if uses_gpu_neighbor_finder(AT)
        return GPUNeighborFinder(
            eligible = to_device(eligible_cpu, AT),
            dist_cutoff = nf.dist_cutoff,
            special = to_device(special_cpu, AT),
            n_steps_reorder = nf.n_steps,
        )
    end
    return DistanceNeighborFinder(
        eligible = to_device(eligible_cpu, AT),
        dist_cutoff = nf.dist_cutoff,
        special = to_device(special_cpu, AT),
        n_steps = nf.n_steps,
    )
end

to_device(nf::TreeNeighborFinder, ::Type{AT}) where {AT} = nf
to_device(nf::CellListMapNeighborFinder, ::Type{AT}) where {AT} = nf

from_device(inter::Ewald) = inter
to_device(inter::Ewald, ::Type{AT}) where {AT} = inter

function _pme_buffers(::Type{AT}, ::Type{T}, mesh_dims, charge_grid, excluded_pairs,
                      n_atoms) where {AT, T}
    if AT <: AbstractGPUArray
        charge_grid_buffer = to_device(zeros(T, size(charge_grid)), AT)
        recip_conv_buffer  = to_device(zeros(T, mesh_dims...), AT)
        excluded_buffer_Fs = to_device(zeros(T, 3, n_atoms), AT)
        excluded_buffer_Es = to_device(zeros(T, length(excluded_pairs)), AT)
        virial_buffer      = to_device(zeros(T, 3, 3), AT)
    elseif Threads.nthreads() > 1
        charge_grid_buffer = [zero(charge_grid) for _ in 1:Threads.nthreads()]
        recip_conv_buffer = zeros(T, Threads.nthreads())
        excluded_buffer_Fs, excluded_buffer_Es = nothing, nothing
        virial_buffer = [zeros(T, 3, 3) for _ in 1:Threads.nthreads()]
    else
        charge_grid_buffer, recip_conv_buffer = nothing, nothing
        excluded_buffer_Fs, excluded_buffer_Es = nothing, nothing
        virial_buffer = [zeros(T, 3, 3)]
    end
    return charge_grid_buffer, excluded_buffer_Fs, excluded_buffer_Es,
           recip_conv_buffer, virial_buffer
end

function _transfer_pme(inter::PME, ::Type{AT}, ::Type{T}) where {AT, T}
    n_atoms = size(inter.grid_indices, 2)
    excluded_pairs = _transfer_storage(inter.excluded_pairs, AT)
    grid_indices = to_device(zeros(Int, 3, n_atoms), AT)
    grid_fractions = to_device(zeros(T, 3, n_atoms), AT)
    bsplines_θ = to_device(zeros(T, inter.order * n_atoms, 3), AT)
    bsplines_dθ = zero(bsplines_θ)
    bsplines_moduli = _pme_bspline_moduli(T, inter.order, inter.mesh_dims)
    bsplines_moduli_x = to_device(bsplines_moduli[1], AT)
    bsplines_moduli_y = to_device(bsplines_moduli[2], AT)
    bsplines_moduli_z = to_device(bsplines_moduli[3], AT)
    charge_grid = to_device(zeros(Complex{T}, inter.mesh_dims[3], inter.mesh_dims[2], inter.mesh_dims[1]), AT)
    charge_grid_buffer, excluded_buffer_Fs, excluded_buffer_Es, recip_conv_buffer,
    virial_buffer = _pme_buffers(AT, T, inter.mesh_dims, charge_grid, excluded_pairs, n_atoms)
    fft_plan = plan_fft!(charge_grid)
    bfft_plan = plan_bfft!(charge_grid)

    return PME(
        _float_precision_convert(inter.dist_cutoff, T),
        _float_precision_convert(inter.error_tol, T),
        inter.order,
        _float_precision_convert(inter.ϵr, T),
        excluded_pairs,
        _float_precision_convert(inter.α, T),
        inter.mesh_dims,
        grid_indices,
        grid_fractions,
        bsplines_θ,
        bsplines_dθ,
        bsplines_moduli_x,
        bsplines_moduli_y,
        bsplines_moduli_z,
        charge_grid,
        charge_grid_buffer,
        excluded_buffer_Fs,
        excluded_buffer_Es,
        recip_conv_buffer,
        virial_buffer,
        _float_precision_convert(inter.pc_sum, T),
        _float_precision_convert(inter.pc_abs2_sum, T),
        fft_plan,
        bfft_plan,
        inter.scheduler,
        inter.grad_safe,
    )
end

_transfer_pme(inter::PME{T}, ::Type{AT}) where {T, AT} = _transfer_pme(inter, AT, T)

@inline _float_precision_convert(inter::PME, ::Type{T}) where {T <: AbstractFloat} =
    _transfer_pme(inter, array_type(inter.charge_grid), T)

from_device(inter::PME) = _transfer_pme(inter, Array)
to_device(inter::PME, ::Type{AT}) where {AT} = _transfer_pme(inter, AT)

function _transfer_obc(inter::ImplicitSolventOBC, ::Type{AT}) where {AT}
    offset_radii = _transfer_storage(inter.offset_radii, AT)
    scaled_offset_radii = _transfer_storage(inter.scaled_offset_radii, AT)
    is = _transfer_storage(inter.is, AT)
    js = _transfer_storage(inter.js, AT)
    oris = @view offset_radii[is]
    orjs = @view offset_radii[js]
    srjs = @view scaled_offset_radii[js]
    return ImplicitSolventOBC(
        offset_radii,
        scaled_offset_radii,
        inter.solvent_dielectric,
        inter.solute_dielectric,
        inter.kappa,
        inter.offset,
        inter.dist_cutoff,
        inter.use_ACE,
        inter.α,
        inter.β,
        inter.γ,
        inter.probe_radius,
        inter.sa_factor,
        inter.factor_solute,
        inter.factor_solvent,
        is,
        js,
        oris,
        orjs,
        srjs,
    )
end

from_device(inter::ImplicitSolventOBC) = _transfer_obc(inter, Array)
to_device(inter::ImplicitSolventOBC, ::Type{AT}) where {AT} = _transfer_obc(inter, AT)

function _transfer_gbn2(inter::ImplicitSolventGBN2, ::Type{AT}) where {AT}
    offset_radii = _transfer_storage(inter.offset_radii, AT)
    scaled_offset_radii = _transfer_storage(inter.scaled_offset_radii, AT)
    αs = _transfer_storage(inter.αs, AT)
    βs = _transfer_storage(inter.βs, AT)
    γs = _transfer_storage(inter.γs, AT)
    is = _transfer_storage(inter.is, AT)
    js = _transfer_storage(inter.js, AT)
    d0s = _transfer_storage(inter.d0s, AT)
    m0s = _transfer_storage(inter.m0s, AT)
    oris = @view offset_radii[is]
    orjs = @view offset_radii[js]
    srjs = @view scaled_offset_radii[js]
    return ImplicitSolventGBN2(
        offset_radii,
        scaled_offset_radii,
        inter.solvent_dielectric,
        inter.solute_dielectric,
        inter.kappa,
        inter.offset,
        inter.dist_cutoff,
        inter.use_ACE,
        αs,
        βs,
        γs,
        inter.probe_radius,
        inter.sa_factor,
        inter.factor_solute,
        inter.factor_solvent,
        is,
        js,
        d0s,
        m0s,
        inter.neck_scale,
        inter.neck_cut,
        oris,
        orjs,
        srjs,
    )
end

from_device(inter::ImplicitSolventGBN2) = _transfer_gbn2(inter, Array)
to_device(inter::ImplicitSolventGBN2, ::Type{AT}) where {AT} = _transfer_gbn2(inter, AT)

function from_device(sr::SHAKE_RATTLE)
    return SHAKE_RATTLE(
        sr,
        _rebuild_structarray(sr.clusters12),
        _rebuild_structarray(sr.clusters23),
        _rebuild_structarray(sr.clusters34),
        _rebuild_structarray(sr.angle_clusters),
    )
end

function to_device(sr::SHAKE_RATTLE, ::Type{AT}) where {AT}
    sr_cpu = from_device(sr)
    if AT <: AbstractGPUArray
        return SHAKE_RATTLE(
            sr_cpu,
            _rebuild_structarray(sr_cpu.clusters12, AT),
            _rebuild_structarray(sr_cpu.clusters23, AT),
            _rebuild_structarray(sr_cpu.clusters34, AT),
            _rebuild_structarray(sr_cpu.angle_clusters, AT),
        )
    end
    return sr_cpu
end

from_device(inter::LJDispersionCorrection) = inter
to_device(inter::LJDispersionCorrection, ::Type{AT}) where {AT} = inter

from_device(inter::MullerBrown) = inter
to_device(inter::MullerBrown, ::Type{AT}) where {AT} = inter

from_device(cv::CalcRMSD) = cv
to_device(cv::CalcRMSD, ::Type{AT}) where {AT} = cv

from_device(bp::BiasPotential) = bp
to_device(bp::BiasPotential, ::Type{AT}) where {AT} = bp

from_device(logger::GeneralObservableLogger) = logger
to_device(logger::GeneralObservableLogger, ::Type{AT}) where {AT} = logger

function from_device(logger::DisplacementsLogger)
    displacements = map(from_device, logger.displacements)
    return DisplacementsLogger(
        displacements,
        from_device(logger.coords_ref),
        from_device(logger.last_displacements),
        logger.n_steps,
        logger.n_steps_update,
    )
end

function to_device(logger::DisplacementsLogger, ::Type{AT}) where {AT}
    displacements = map(from_device, logger.displacements)
    return DisplacementsLogger(
        displacements,
        to_device(from_device(logger.coords_ref), AT),
        to_device(from_device(logger.last_displacements), AT),
        logger.n_steps,
        logger.n_steps_update,
    )
end

from_device(logger::TrajectoryWriter) = logger
to_device(logger::TrajectoryWriter, ::Type{AT}) where {AT} = logger

from_device(logger::TimeCorrelationLogger) = logger
to_device(logger::TimeCorrelationLogger, ::Type{AT}) where {AT} = logger

from_device(logger::AverageObservableLogger) = logger
to_device(logger::AverageObservableLogger, ::Type{AT}) where {AT} = logger

from_device(logger::ReplicaExchangeLogger) = logger
to_device(logger::ReplicaExchangeLogger, ::Type{AT}) where {AT} = logger

from_device(logger::MonteCarloLogger) = logger
to_device(logger::MonteCarloLogger, ::Type{AT}) where {AT} = logger

from_device(logger::AWHEnsembleLogger) = logger
to_device(logger::AWHEnsembleLogger, ::Type{AT}) where {AT} = logger
