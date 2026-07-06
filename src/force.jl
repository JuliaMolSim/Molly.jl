# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See OpenMM documentation and Gromacs manual for other aspects of forces

export
    accelerations,
    force,
    pairwise_force,
    SpecificForce1Atoms,
    SpecificForce2Atoms,
    SpecificForce3Atoms,
    SpecificForce4Atoms,
    SpecificForce5Atoms,
    forces,
    forces_virial

# Apply F = ma but give massless particles zero acceleration
calc_accels(f, m) = (iszero(m) ? zero(f / oneunit(m)) : (f / m))

"""
    accelerations(system, neighbors=find_neighbors(system), step_n=0;
                  n_threads=Threads.nthreads(), pairwise_inters=system.pairwise_inters,
                  specific_inter_lists=system.specific_inter_lists,
                  general_inters=system.general_inters)

Calculate the accelerations of all atoms in a system using the pairwise,
specific and general interactions and Newton's second law of motion.
"""
function accelerations(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return accelerations(sys, find_neighbors(sys; n_threads=n_threads);
                         n_threads=n_threads, kwargs...)
end

function accelerations(sys, neighbors, step_n::Integer=0; kwargs...)
    fs = forces(sys, neighbors, step_n; kwargs...)
    return calc_accels.(fs, masses(sys))
end

"""
    force(inter, vec_ij, atom_i, atom_j, force_units, special, coord_i, coord_j,
          boundary, velocity_i, velocity_j, step_n)
    force(inter, coord_i, boundary, atom_i, force_units, velocity_i, step_n)
    force(inter, coord_i, coord_j, boundary, atom_i, atom_j, force_units, velocity_i,
          velocity_j, step_n)
    force(inter, coord_i, coord_j, coord_k, boundary, atom_i, atom_j, atom_k,
          force_units, velocity_i, velocity_j, velocity_k, step_n)
    force(inter, coord_i, coord_j, coord_k, coord_l, boundary, atom_i, atom_j, atom_k,
          atom_l, force_units, velocity_i, velocity_j, velocity_k, velocity_l, step_n)

Calculate the force between atoms due to a given interaction type.

For pairwise interactions returns a single force vector and for specific interactions
returns a type such as [`SpecificForce2Atoms`](@ref).
Custom pairwise and specific interaction types should implement this function.
"""
function force end

# Allow GPU-specific force functions to be defined if required
force_gpu(inter, dr, ai, aj, fu, sp, ci, cj, bnd, vi, vj, sn) = force(inter, dr, ai, aj, fu, sp, ci, cj, bnd, vi, vj, sn)
force_gpu(inter, ci, bnd, ai, fu, vi, sn, data) = force(inter, ci, bnd, ai, fu, vi, sn, data)
force_gpu(inter, ci, cj, bnd, ai, aj, fu, vi, vj, sn, data) = force(inter, ci, cj, bnd, ai, aj, fu, vi, vj, sn, data)
force_gpu(inter, ci, cj, ck, bnd, ai, aj, ak, fu, vi, vj, vk, sn, data) = force(inter, ci, cj, ck, bnd, ai, aj, ak, fu, vi, vj, vk, sn, data)
force_gpu(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, fu, vi, vj, vk, vl, sn, data) = force(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, fu, vi, vj, vk, vl, sn, data)
force_gpu(inter, ci, cj, ck, cl, cm, bnd, ai, aj, ak, al, am, fu, vi, vj, vk, vl, vm, sn, data) = force(inter, ci, cj, ck, cl, cm, bnd, ai, aj, ak, al, am, fu, vi, vj, vk, vl, vm, sn, data)

@inline zero_pairwise_force(dr, force_units) = ustrip.(zero(dr)) * force_units

@inline function zero_pairwise_force(dr::SVector{N, <:Unitful.Quantity{T}}, force_units) where {N, T}
    return zero(SVector{N, T}) * force_units
end

@inline function zero_pairwise_force(dr::SVector{N, T}, ::typeof(NoUnits)) where {N, T <: AbstractFloat}
    return zero(SVector{N, T})
end

@inline function radial_force_vector(f, r, dr, force_units)
    return iszero_value(r) ? zero_pairwise_force(dr, force_units) : (f / r) * dr
end

@inline function sum_pairwise_forces(inters::Tuple{T}, dr, atom_i, atom_j, force_units,
                                     special, coord_i, coord_j, boundary, vel_i, vel_j,
                                     step_n) where {T}
    return force(inters[1], dr, atom_i, atom_j, force_units, special, coord_i, coord_j,
                 boundary, vel_i, vel_j, step_n)
end

@inline function sum_pairwise_forces(inters::Tuple, dr, atom_i, atom_j, force_units,
                                     special, coord_i, coord_j, boundary, vel_i, vel_j, step_n)
    return force(first(inters), dr, atom_i, atom_j, force_units, special, coord_i, coord_j,
                 boundary, vel_i, vel_j, step_n) +
           sum_pairwise_forces(Base.tail(inters), dr, atom_i, atom_j, force_units, special,
                               coord_i, coord_j, boundary, vel_i, vel_j, step_n)
end

"""
    pairwise_force(inter, r, params)

Calculate the force magnitude between two atoms separated by distance `r` due to a
pairwise interaction.

This function is used in [`force`](@ref) to apply cutoff strategies by calculating
the force at different values of `r`.
Consequently, the parameters `params` should not include terms that depend on distance.
"""
function pairwise_force end

"""
    SpecificForce1Atoms(f1)

Force on one atom arising from an interaction such as a position restraint.
"""
struct SpecificForce1Atoms{D, T}
    f1::SVector{D, T}
end

"""
    SpecificForce2Atoms(f1, f2)

Forces on two atoms arising from an interaction such as a bond potential.
"""
struct SpecificForce2Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
end

"""
    SpecificForce3Atoms(f1, f2, f3)

Forces on three atoms arising from an interaction such as a bond angle potential.
"""
struct SpecificForce3Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
end

"""
    SpecificForce4Atoms(f1, f2, f3, f4)

Forces on four atoms arising from an interaction such as a torsion potential.
"""
struct SpecificForce4Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
    f4::SVector{D, T}
end

"""
    SpecificForce5Atoms(f1, f2, f3, f4, f5)

Forces on five atoms arising from an interaction such as a CMAP torsion potential.
"""
struct SpecificForce5Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
    f4::SVector{D, T}
    f5::SVector{D, T}
end

function SpecificForce1Atoms(f1::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce1Atoms{D, T}(f1)
end

function SpecificForce2Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce2Atoms{D, T}(f1, f2)
end

function SpecificForce3Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                             f3::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce3Atoms{D, T}(f1, f2, f3)
end

function SpecificForce4Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                             f3::StaticArray{Tuple{D}, T}, f4::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce4Atoms{D, T}(f1, f2, f3, f4)
end

function SpecificForce5Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                             f3::StaticArray{Tuple{D}, T}, f4::StaticArray{Tuple{D}, T},
                             f5::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce5Atoms{D, T}(f1, f2, f3, f4, f5)
end

Base.:+(x::SpecificForce1Atoms, y::SpecificForce1Atoms) = SpecificForce1Atoms(x.f1 + y.f1)
Base.:+(x::SpecificForce2Atoms, y::SpecificForce2Atoms) = SpecificForce2Atoms(x.f1 + y.f1, x.f2 + y.f2)
Base.:+(x::SpecificForce3Atoms, y::SpecificForce3Atoms) = SpecificForce3Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3)
Base.:+(x::SpecificForce4Atoms, y::SpecificForce4Atoms) = SpecificForce4Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3, x.f4 + y.f4)
Base.:+(x::SpecificForce5Atoms, y::SpecificForce5Atoms) = SpecificForce5Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3, x.f4 + y.f4, x.f5 + y.f5)

const INVALID_BUFFER_STEP = -1

# Tracks which step cached virial and pressure buffers are valid for.
mutable struct BufferValidity
    interaction_virial_step::Int
    constraint_virial_step::Int
    total_virial_step::Int
    pressure_step::Int
    pre_coupling_virial_step::Int
    pre_coupling_pressure_step::Int
end

BufferValidity() = BufferValidity(
    INVALID_BUFFER_STEP,
    INVALID_BUFFER_STEP,
    INVALID_BUFFER_STEP,
    INVALID_BUFFER_STEP,
    INVALID_BUFFER_STEP,
    INVALID_BUFFER_STEP,
)

function invalidate_interaction_virial!(v::BufferValidity)
    v.interaction_virial_step = INVALID_BUFFER_STEP
    return v
end

function invalidate_constraint_virial!(v::BufferValidity)
    v.constraint_virial_step = INVALID_BUFFER_STEP
    return v
end

function invalidate_total_virial!(v::BufferValidity)
    v.total_virial_step = INVALID_BUFFER_STEP
    return v
end

function invalidate_pressure!(v::BufferValidity)
    v.pressure_step = INVALID_BUFFER_STEP
    return v
end

function invalidate_pre_coupling_virial!(v::BufferValidity)
    v.pre_coupling_virial_step = INVALID_BUFFER_STEP
    return v
end

function invalidate_pre_coupling_pressure!(v::BufferValidity)
    v.pre_coupling_pressure_step = INVALID_BUFFER_STEP
    return v
end

function mark_interaction_virial!(v::BufferValidity, step_n::Integer)
    v.interaction_virial_step = Int(step_n)
    invalidate_total_virial!(v)
    return v
end

function mark_constraint_virial!(v::BufferValidity, step_n::Integer)
    v.constraint_virial_step = Int(step_n)
    return v
end

function mark_total_virial!(v::BufferValidity, step_n::Integer)
    v.total_virial_step = Int(step_n)
    return v
end

function mark_pressure!(v::BufferValidity, step_n::Integer)
    v.pressure_step = Int(step_n)
    return v
end

function mark_pre_coupling_virial!(v::BufferValidity, step_n::Integer)
    v.pre_coupling_virial_step = Int(step_n)
    return v
end

function mark_pre_coupling_pressure!(v::BufferValidity, step_n::Integer)
    v.pre_coupling_pressure_step = Int(step_n)
    return v
end

function has_interaction_virial(v::BufferValidity, step_n::Integer)
    return v.interaction_virial_step == step_n
end

function has_constraint_virial(v::BufferValidity, step_n::Integer)
    return v.constraint_virial_step == step_n
end

function has_total_virial(v::BufferValidity, step_n::Integer)
    return v.total_virial_step == step_n
end

function has_pressure(v::BufferValidity, step_n::Integer)
    return v.pressure_step == step_n
end

function has_pre_coupling_virial(v::BufferValidity, step_n::Integer)
    return v.pre_coupling_virial_step == step_n
end

function has_pre_coupling_pressure(v::BufferValidity, step_n::Integer)
    return v.pre_coupling_pressure_step == step_n && has_pre_coupling_virial(v, step_n)
end

function has_interaction_virial(buffers, step_n::Integer)
    return has_interaction_virial(buffers.validity, step_n)
end

function has_constraint_virial(buffers, step_n::Integer)
    return has_constraint_virial(buffers.validity, step_n)
end

function has_total_virial(buffers, step_n::Integer)
    return has_total_virial(buffers.validity, step_n)
end

function has_pressure(buffers, step_n::Integer)
    return has_pressure(buffers.validity, step_n)
end

function has_pre_coupling_virial(buffers, step_n::Integer)
    return has_pre_coupling_virial(buffers.validity, step_n)
end

function has_pre_coupling_pressure(buffers, step_n::Integer)
    return has_pre_coupling_pressure(buffers.validity, step_n)
end

function clear_constraint_virial!(buffers, step_n::Integer)
    fill!(buffers.constraint_virial, zero(eltype(buffers.constraint_virial)))
    fill!(buffers.constraint_virial_nounits, zero(eltype(buffers.constraint_virial_nounits)))
    mark_constraint_virial!(buffers.validity, step_n)
    invalidate_total_virial!(buffers.validity)
    invalidate_pressure!(buffers.validity)
    return buffers
end

clear_constraint_virial!(buffers, sys, step_n::Integer) = clear_constraint_virial!(buffers, step_n)

function constraint_virial_nounits(buffers, contribution)
    e_unit = unit(eltype(buffers.constraint_virial))
    if e_unit == NoUnits
        return ustrip.(contribution)
    else
        return ustrip.(e_unit, contribution)
    end
end

function accumulate_constraint_virial!(buffers, contribution)
    buffers.constraint_virial_nounits .+= constraint_virial_nounits(buffers, contribution)
    invalidate_total_virial!(buffers.validity)
    invalidate_pressure!(buffers.validity)
    return buffers
end

function accumulate_constraint_virial!(buffers, contribution, context)
    context.needs_virial || return buffers
    return accumulate_constraint_virial!(buffers, contribution .* context.virial_scale)
end

pre_coupling_ref() = Ref{Any}(nothing)
constraint_scratch_ref() = Ref{Any}(nothing)

struct BuffersCPU{F, A, V, VN, VC, KT, PT, FM}
    fs_nounits::F
    fs_chunks::A
    virial::V
    vir_nounits::VN
    vir_chunks::VC
    constraint_virial::V
    constraint_virial_nounits::VN
    constraint_virial_chunks::VC
    kin_tensor::KT
    pres_tensor::PT
    pre_coupling_virial::Base.RefValue{Any}
    pre_coupling_kin_tensor::Base.RefValue{Any}
    pre_coupling_volume::Base.RefValue{Any}
    constraint_coords_buffer::Base.RefValue{Any}
    constraint_velocities_buffer::Base.RefValue{Any}
    constraint_preview_coords_buffer::Base.RefValue{Any}
    fs_mat::FM
    validity::BufferValidity
end

function BuffersCPU(fs_nounits, fs_chunks, virial, vir_nounits, vir_chunks,
                    kin_tensor, pres_tensor, fs_mat)
    constraint_virial = zero(virial)
    constraint_virial_nounits = zero(vir_nounits)
    constraint_virial_chunks = similar(vir_chunks)
    for i in eachindex(vir_chunks)
        constraint_virial_chunks[i] = zero(vir_chunks[i])
    end
    return BuffersCPU(fs_nounits, fs_chunks, virial, vir_nounits, vir_chunks,
                      constraint_virial, constraint_virial_nounits,
                      constraint_virial_chunks, kin_tensor, pres_tensor,
                      pre_coupling_ref(), pre_coupling_ref(), pre_coupling_ref(),
                      constraint_scratch_ref(), constraint_scratch_ref(),
                      constraint_scratch_ref(), fs_mat, BufferValidity())
end

function BuffersCPU(fs_nounits, fs_chunks, virial, vir_nounits, vir_chunks,
                    constraint_virial, constraint_virial_nounits,
                    constraint_virial_chunks, kin_tensor, pres_tensor, fs_mat)
    return BuffersCPU(fs_nounits, fs_chunks, virial, vir_nounits, vir_chunks,
                      constraint_virial, constraint_virial_nounits,
                      constraint_virial_chunks, kin_tensor, pres_tensor,
                      pre_coupling_ref(), pre_coupling_ref(), pre_coupling_ref(),
                      constraint_scratch_ref(), constraint_scratch_ref(),
                      constraint_scratch_ref(), fs_mat, BufferValidity())
end

function init_buffers!(sys::System{D}, n_threads) where D
    # Allows propagation of uncertainties to tensors
    CT = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
    fs_nounits    = ustrip_vec.(zero(sys.coords))
    vir           = zeros(CT, D, D) .* sys.energy_units
    vir_nounits   = ustrip_vec.(zero(vir))
    constr_vir    = zero(vir)
    constr_vir_nu = zero(vir_nounits)
    kin           = zero(vir)
    pres          = zero(vir_nounits) .* (sys.energy_units == NoUnits ? NoUnits : u"bar")
    # Enzyme errors with nothing when n_threads is 1
    n_copies = (n_threads == 1 ? 0 : n_threads)
    fs_chunks         = [zero(fs_nounits) for _ in 1:n_copies]
    vir_chunks        = [zero(vir_nounits) for _ in 1:n_copies]
    constr_vir_chunks = [zero(vir_nounits) for _ in 1:n_copies]
    # fs_mat is only used for virtual sites to do atomic addition
    fs_mat = (length(sys.virtual_sites) > 0 ? zeros(CT, D, length(sys)) : nothing)
    return BuffersCPU(
        fs_nounits, fs_chunks,
        vir, vir_nounits, vir_chunks,
        constr_vir, constr_vir_nu, constr_vir_chunks,
        kin,
        pres,
        pre_coupling_ref(), pre_coupling_ref(), pre_coupling_ref(),
        constraint_scratch_ref(), constraint_scratch_ref(), constraint_scratch_ref(),
        fs_mat,
        BufferValidity(),
    )
end

upper_tile_count(n_blocks::Integer) = (Int64(n_blocks) * (Int64(n_blocks) + 1)) ÷ 2

#=
    BuffersGPU

Mutable struct holding GPU-resident buffers and state for pairwise force and 
energy calculations.

# Fields
- `fs_mat`: Force matrix of size `(D, N)`.
- `pe_vec_nounits`: Vector of size 1 for potential energy summation.
- `virial`: 3x3 virial tensor (with units).
- `virial_nounits`: 3x3 virial tensor on GPU (without units).
- `constraint_virial`: 3x3 constraint virial tensor (with units).
- `constraint_virial_nounits`: 3x3 constraint virial tensor on GPU (without units).
- `kin_tensor`: 3x3 kinetic energy tensor (with units).
- `pres_tensor`: 3x3 pressure tensor (with units).
- `pre_coupling_virial`, `pre_coupling_kin_tensor`, `pre_coupling_volume`:
  Optional state used by pressure/virial loggers after coordinate-scaling coupling.
- `constraint_coords_buffer`, `constraint_velocities_buffer`,
  `constraint_preview_coords_buffer`: Reusable scratch arrays for constraint virial
  snapshots and initial-step previews.
- `validity`: Step metadata describing which tensor buffers are current.
- `box_mins`, `box_maxs`: Bounding boxes for each 32-atom block.
- `morton_seq`, `morton_seq_buffer_1`, `morton_seq_buffer_2`, `morton_seq_inv`:
  Morton-order indices and temporary buffers for reordering atoms on the GPU.
- `compressed_masks`: 32x32 bitmasks (eligibility and special flags) for each
  upper-triangular tile in Morton order.
- `tile_is_clean`: Boolean flag for each tile indicating whether it contains no
  exclusions or special pairs and can skip mask lookups.
- `interacting_tiles_i`, `interacting_tiles_j`, `interacting_tiles_type`:
  parallel 1D vectors describing the compact list of tiles currently inside the
  interaction cutoff.
- `num_interacting_tiles`: device-side atomic counter for the number of valid
  entries in the interacting-tile vectors.
- `interacting_tiles_overflow`: device-side overflow flag set when the compact
  tile list exceeds the allocated capacity.
- `coords_reordered`, `velocities_reordered`, `atoms_reordered`: Cached reordered arrays.
- `fs_mat_reordered`: Force matrix in reordered space.
- `step_n_preprocessed`: Last simulation step where preprocessing was done.
- `sparse_pair_generation`: sparse-pair generation reflected in the cached masks
  and tile metadata.
- `num_pairs`: host-side cached copy of the current interacting-tile count,
  used to size kernel launches.
=#
mutable struct BuffersGPU{F, P, V, VN, KT, PT, C, M, R, IT, ITT, NIT, OIT, CR, VR, AR, fs_re, TIC}
    fs_mat::F
    pe_vec_nounits::P
    virial::V
    virial_nounits::VN
    constraint_virial::V
    constraint_virial_nounits::VN
    kin_tensor::KT
    pres_tensor::PT
    pre_coupling_virial::Base.RefValue{Any}
    pre_coupling_kin_tensor::Base.RefValue{Any}
    pre_coupling_volume::Base.RefValue{Any}
    constraint_coords_buffer::Base.RefValue{Any}
    constraint_velocities_buffer::Base.RefValue{Any}
    constraint_preview_coords_buffer::Base.RefValue{Any}
    validity::BufferValidity
    box_mins::C
    box_maxs::C
    morton_seq::M
    morton_seq_buffer_1::M
    morton_seq_buffer_2::M
    morton_seq_inv::M
    compressed_masks::R
    tile_is_clean::TIC
    interacting_tiles_i::IT
    interacting_tiles_j::IT
    interacting_tiles_type::ITT
    num_interacting_tiles::NIT
    interacting_tiles_overflow::OIT
    coords_reordered::CR
    velocities_reordered::VR
    atoms_reordered::AR
    fs_mat_reordered::fs_re
    step_n_preprocessed::Int
    sparse_pair_generation::UInt64
    num_pairs::Int
end

function BuffersGPU(fs_mat, pe_vec_nounits, virial, virial_nounits, kin_tensor, pres_tensor,
                    box_mins, box_maxs, morton_seq, morton_seq_buffer_1,
                    morton_seq_buffer_2, morton_seq_inv, compressed_masks, tile_is_clean,
                    interacting_tiles_i, interacting_tiles_j, interacting_tiles_type,
                    num_interacting_tiles, interacting_tiles_overflow, coords_reordered,
                    velocities_reordered, atoms_reordered, fs_mat_reordered,
                    step_n_preprocessed, sparse_pair_generation, num_pairs)
    constraint_virial = zero(virial)
    constraint_virial_nounits = similar(virial_nounits)
    fill!(constraint_virial_nounits, zero(eltype(virial_nounits)))
    return BuffersGPU(fs_mat, pe_vec_nounits, virial, virial_nounits,
                      constraint_virial, constraint_virial_nounits, kin_tensor, pres_tensor,
                      pre_coupling_ref(), pre_coupling_ref(), pre_coupling_ref(),
                      constraint_scratch_ref(), constraint_scratch_ref(),
                      constraint_scratch_ref(), BufferValidity(), box_mins, box_maxs, morton_seq,
                      morton_seq_buffer_1, morton_seq_buffer_2, morton_seq_inv,
                      compressed_masks, tile_is_clean, interacting_tiles_i,
                      interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
                      interacting_tiles_overflow, coords_reordered, velocities_reordered,
                      atoms_reordered, fs_mat_reordered, step_n_preprocessed,
                      sparse_pair_generation, num_pairs)
end

function clear_constraint_virial!(buffers::BuffersCPU, step_n::Integer)
    fill!(buffers.constraint_virial, zero(eltype(buffers.constraint_virial)))
    fill!(buffers.constraint_virial_nounits, zero(eltype(buffers.constraint_virial_nounits)))
    for chunk in buffers.constraint_virial_chunks
        fill!(chunk, zero(eltype(chunk)))
    end
    mark_constraint_virial!(buffers.validity, step_n)
    invalidate_total_virial!(buffers.validity)
    invalidate_pressure!(buffers.validity)
    return buffers
end

function merge_constraint_virial!(buffers::BuffersCPU, sys, step_n::Integer)
    has_interaction_virial(buffers, step_n) ||
        error("cannot merge constraint virial before interaction virial is valid for step $step_n")
    has_constraint_virial(buffers, step_n) ||
        error("cannot merge constraint virial before constraint virial is valid for step $step_n")

    buffers.constraint_virial .= buffers.constraint_virial_nounits .* sys.energy_units
    buffers.virial .+= buffers.constraint_virial
    mark_total_virial!(buffers.validity, step_n)
    invalidate_pressure!(buffers.validity)
    return buffers
end

function merge_constraint_virial!(buffers::BuffersGPU, sys, step_n::Integer)
    has_interaction_virial(buffers, step_n) ||
        error("cannot merge constraint virial before interaction virial is valid for step $step_n")
    has_constraint_virial(buffers, step_n) ||
        error("cannot merge constraint virial before constraint virial is valid for step $step_n")

    buffers.constraint_virial .= from_device(buffers.constraint_virial_nounits) .* sys.energy_units
    buffers.virial .+= buffers.constraint_virial
    mark_total_virial!(buffers.validity, step_n)
    invalidate_pressure!(buffers.validity)
    return buffers
end

function save_pre_coupling_virial!(buffers, step_n::Integer)
    buffers.pre_coupling_virial[] = copy(buffers.virial)
    mark_pre_coupling_virial!(buffers.validity, step_n)
    invalidate_pre_coupling_pressure!(buffers.validity)
    return buffers
end

function save_pre_coupling_pressure!(buffers, sys, step_n::Integer, kin_tensor)
    if isnothing(kin_tensor)
        kinetic_energy_tensor!(buffers.kin_tensor, sys)
        buffers.pre_coupling_kin_tensor[] = copy(buffers.kin_tensor)
    else
        buffers.pre_coupling_kin_tensor[] = copy(kin_tensor)
    end
    buffers.pre_coupling_volume[] = volume(sys.boundary)
    mark_pre_coupling_pressure!(buffers.validity, step_n)
    return buffers
end

#=
    init_buffers!(sys::System{D, <:AbstractGPUArray, T}, n_threads, for_pe=false)

Initialize and return a [`BuffersGPU`](@ref) struct for a GPU-based system.

Allocates the necessary arrays on the GPU using the system's backend. If `for_pe`
is `true`, the neighbor finder initialization state is preserved; otherwise, it
is reset if it is a [`GPUNeighborFinder`](@ref).
=#
function init_buffers!(sys::System{D, <:AbstractGPUArray, T}, n_threads,
                   for_pe::Bool=false) where {D, T}
    N = length(sys)
    C = eltype(eltype(sys.coords))
    CT = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
    ET = nonbonded_energy_type(sys)
    n_blocks = cld(N, 32)
    n_upper_tiles = upper_tile_count(n_blocks)
    backend = get_backend(sys.coords)

    fs_mat       = KernelAbstractions.zeros(backend, T, D, N)
    fs_mat_reordered = KernelAbstractions.zeros(backend, T, D, N)
    pe_vec_noun  = KernelAbstractions.zeros(backend, ET, 1)
    virial       = zeros(CT, D, D) .* sys.energy_units
    virial_nu    = KernelAbstractions.zeros(backend, T, D, D)
    constr_vir   = zero(virial)
    constr_vir_nu = KernelAbstractions.zeros(backend, T, D, D)
    kin          = zero(virial)
    pres         = ustrip_vec.(zero(virial)) * (sys.energy_units == NoUnits ? NoUnits : u"bar")
    box_mins = KernelAbstractions.zeros(backend, C, n_blocks, D)
    box_maxs = KernelAbstractions.zeros(backend, C, n_blocks, D)
    morton_seq = KernelAbstractions.zeros(backend, Int32, N)
    morton_seq_buffer_1 = KernelAbstractions.zeros(backend, Int32, N)
    morton_seq_buffer_2 = KernelAbstractions.zeros(backend, Int32, N)
    morton_seq_inv = KernelAbstractions.zeros(backend, Int32, N)
    compressed_masks = KernelAbstractions.zeros(backend, UInt32, 32, 2, n_upper_tiles)
    tile_is_clean = KernelAbstractions.zeros(backend, Bool, n_upper_tiles)

    max_interacting_blocks = min(n_upper_tiles, 1024 * n_blocks) # TODO: Implement dynamic resizing for this buffer
    interacting_tiles_i = KernelAbstractions.zeros(backend, Int32, max_interacting_blocks)
    interacting_tiles_j = KernelAbstractions.zeros(backend, Int32, max_interacting_blocks)
    interacting_tiles_type = KernelAbstractions.zeros(backend, UInt8, max_interacting_blocks)
    num_interacting_tiles = KernelAbstractions.zeros(backend, Int32, 1)
    interacting_tiles_overflow = KernelAbstractions.zeros(backend, Int32, 1)

    coords_reordered = zero(sys.coords)
    velocities_reordered = zero(sys.velocities)
    atoms_reordered = zero(sys.atoms)

    if !for_pe && sys.neighbor_finder isa GPUNeighborFinder
        sys.neighbor_finder.initialized = false
    end

    return BuffersGPU(fs_mat, pe_vec_noun, virial, virial_nu, constr_vir, constr_vir_nu,
                      kin, pres, pre_coupling_ref(), pre_coupling_ref(),
                      pre_coupling_ref(), constraint_scratch_ref(), constraint_scratch_ref(),
                      constraint_scratch_ref(), BufferValidity(), box_mins, box_maxs, morton_seq,
                      morton_seq_buffer_1, morton_seq_buffer_2, morton_seq_inv,
                      compressed_masks, tile_is_clean, interacting_tiles_i,
                      interacting_tiles_j, interacting_tiles_type, num_interacting_tiles,
                      interacting_tiles_overflow, coords_reordered, velocities_reordered,
                      atoms_reordered, fs_mat_reordered, -1, UInt64(0), 0)
end
zero_forces(sys) = ustrip_vec.(zero(sys.coords)) .* sys.force_units

"""
    forces(system, neighbors=find_neighbors(system), step_n=0;
           n_threads=Threads.nthreads(), pairwise_inters=system.pairwise_inters,
           specific_inter_lists=system.specific_inter_lists,
           general_inters=system.general_inters)

Calculate the forces on all atoms in a system using the pairwise, specific and
general interactions.
"""
function forces(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return forces(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads, kwargs...)
end

function forces(sys, neighbors, step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...)
    buffers = init_buffers!(sys, n_threads)
    fs = zero_forces(sys)
    forces!(fs, sys, neighbors, step_n, buffers, Val(false); n_threads=n_threads, kwargs...)
    return fs
end

"""
    forces_virial(system, neighbors=find_neighbors(system), step_n=0;
                  n_threads=Threads.nthreads(), pairwise_inters=system.pairwise_inters,
                  specific_inter_lists=system.specific_inter_lists,
                  general_inters=system.general_inters)

Calculate the forces on all atoms in a system and the virial using the pairwise,
specific and general interactions.
For constrained systems, constraint virial contributions are approximated using
the same deterministic small-step constraint preview as [`virial`](@ref).

Returns a tuple of the forces and the virial.
This is faster than calling [`forces`](@ref) and [`virial`](@ref) separately.
"""
function forces_virial(sys; n_threads::Integer=Threads.nthreads(), kwargs...)
    return forces_virial(sys, find_neighbors(sys; n_threads=n_threads);
                         n_threads=n_threads, kwargs...)
end

function forces_virial(sys, neighbors, step_n::Integer=0; n_threads::Integer=Threads.nthreads(),
                       kwargs...)
    buffers = init_buffers!(sys, n_threads)
    if length(sys.constraints) > 0
        fs, _ = compute_initial_total_virial!(buffers, sys, neighbors, step_n;
                                              n_threads=n_threads, kwargs...)
        return fs, buffers.virial
    end
    fs = zero_forces(sys)
    forces!(fs, sys, neighbors, step_n, buffers, Val(true); n_threads=n_threads, kwargs...)
    return fs, buffers.virial
end

function forces!(fs,
                 sys::System{<:Any, <:Any, T},
                 neighbors,
                 step_n::Integer,
                 buffers::BuffersCPU,
                 ::Val{needs_vir};
                 n_threads::Integer=Threads.nthreads(),
                 pairwise_inters=sys.pairwise_inters,
                 specific_inter_lists=sys.specific_inter_lists,
                 general_inters=sys.general_inters) where {T, needs_vir}
    if needs_vir
        fill!(buffers.virial, zero(T) * sys.energy_units)
    else
        invalidate_interaction_virial!(buffers.validity)
        invalidate_total_virial!(buffers.validity)
    end
    invalidate_pressure!(buffers.validity)
    fill!(buffers.kin_tensor,  zero(T) * sys.energy_units)
    fill!(buffers.pres_tensor, zero(T) * (sys.energy_units == NoUnits ? NoUnits : u"bar"))

    FT = eltype(buffers.fs_nounits)
    if n_threads == 1
        fill!(buffers.fs_nounits, zero(FT))
        if needs_vir
            fill!(buffers.vir_nounits, zero(eltype(buffers.vir_nounits)))
        end
    else
        Threads.@threads for chunk_i in 1:n_threads
            fill!(buffers.fs_chunks[chunk_i], zero(FT))
        end
        if needs_vir
            Threads.@threads for chunk_i in 1:n_threads
                fill!(buffers.vir_chunks[chunk_i], zero(eltype(buffers.vir_nounits)))
            end
        end
    end

    if length(pairwise_inters) > 0
        pairwise_inters_nonl = filter(!use_neighbors, values(pairwise_inters))
        pairwise_inters_nl   = filter( use_neighbors, values(pairwise_inters))
        pairwise_forces_loop!(buffers.fs_nounits, buffers.fs_chunks, buffers.vir_nounits,
                buffers.vir_chunks, sys.atoms, sys.coords, sys.velocities, sys.boundary,
                neighbors, sys.force_units, length(sys), pairwise_inters_nonl,
                pairwise_inters_nl, step_n, Val(n_threads), Val(needs_vir))
    end

    if length(specific_inter_lists) > 0
        sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(specific_inter_lists))
        sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(specific_inter_lists))
        sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(specific_inter_lists))
        sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(specific_inter_lists))
        sils_5_atoms = filter(il -> il isa InteractionList5Atoms, values(specific_inter_lists))
        specific_forces_loop!(buffers.fs_nounits, buffers.fs_chunks, buffers.vir_nounits,
                              buffers.vir_chunks, sys.atoms, sys.coords, sys.velocities,
                              sys.boundary, sys.force_units, sils_1_atoms, sils_2_atoms,
                              sils_3_atoms, sils_4_atoms, sils_5_atoms, step_n, Val(n_threads),
                              Val(needs_vir))
    end

    if n_threads > 1
        reduce_force_chunks!(buffers.fs_nounits, buffers.fs_chunks, buffers.vir_nounits,
                             buffers.vir_chunks, Val(n_threads), Val(needs_vir))
    end

    fs .= buffers.fs_nounits .* sys.force_units
    if needs_vir
        buffers.virial .= buffers.vir_nounits .* sys.energy_units
    end

    for inter in values(general_inters)
        AtomsCalculators.forces!(fs, sys, inter; neighbors=neighbors, step_n=step_n,
                                 n_threads=n_threads, buffers=buffers, needs_vir=needs_vir)
    end
    distribute_forces!(fs, sys, buffers)

    if needs_vir
        mark_interaction_virial!(buffers.validity, step_n)
        if length(sys.constraints) == 0
            mark_total_virial!(buffers.validity, step_n)
        end
    end

    return fs, buffers
end

function reduce_force_chunks!(fs_nounits, fs_chunks, vir_nounits, vir_chunks,
                              ::Val{n_threads}, ::Val{needs_vir}) where {n_threads, needs_vir}
    FT = eltype(fs_nounits)
    @inbounds Threads.@threads for i in eachindex(fs_nounits)
        f = zero(FT)
        for chunk_i in 1:n_threads
            f += fs_chunks[chunk_i][i]
        end
        fs_nounits[i] = f
    end

    if needs_vir
        @inbounds vir_nounits .= vir_chunks[1]
        @inbounds for chunk_i in 2:n_threads
            vir_nounits .+= vir_chunks[chunk_i]
        end
    end
    return fs_nounits
end

function pairwise_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords,
                               velocities, boundary, neighbors, force_units, n_atoms,
                               pairwise_inters_nonl, pairwise_inters_nl, step_n, ::Val{1},
                               ::Val{needs_vir}) where needs_vir
    @inbounds if length(pairwise_inters_nonl) > 0
        for i in 1:n_atoms
            coord_i = coords[i]
            atom_i = atoms[i]
            vel_i = velocities[i]
            for j in (i + 1):n_atoms
                coord_j = coords[j]
                atom_j = atoms[j]
                vel_j = velocities[j]
                dr = vector(coord_i, coord_j, boundary)
                f = sum_pairwise_forces(pairwise_inters_nonl, dr, atom_i, atom_j, force_units,
                                        false, coord_i, coord_j, boundary, vel_i, vel_j, step_n)
                f_ustrip = checked_ustrip(f, force_units)
                fs_nounits[i] -= f_ustrip
                fs_nounits[j] += f_ustrip

                if needs_vir
                    # Kronecker product of vector along which force acts and force itself
                    v = dr * transpose(f)
                    vir_nounits .+= ustrip.(v)
                end
            end
        end
    end

    @inbounds if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        for ni in eachindex(neighbors)
            i, j, special = neighbors[ni]
            coord_i = coords[i]
            coord_j = coords[j]
            atom_i = atoms[i]
            atom_j = atoms[j]
            vel_i = velocities[i]
            vel_j = velocities[j]
            dr = vector(coord_i, coord_j, boundary)
            f = sum_pairwise_forces(pairwise_inters_nl, dr, atom_i, atom_j, force_units,
                                    special, coord_i, coord_j, boundary, vel_i, vel_j, step_n)
            f_ustrip = checked_ustrip(f, force_units)
            fs_nounits[i] -= f_ustrip
            fs_nounits[j] += f_ustrip

            if needs_vir
                v = dr * transpose(f)
                vir_nounits .+= ustrip.(v)
            end
        end
    end

    return fs_nounits
end

function pairwise_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords,
                               velocities, boundary, neighbors, force_units, n_atoms,
                               pairwise_inters_nonl, pairwise_inters_nl, step_n, ::Val{n_threads},
                               ::Val{needs_vir}) where {n_threads, needs_vir}
    if isnothing(fs_chunks) || (needs_vir && isnothing(vir_chunks))
        throw(ArgumentError("fs_chunks / vir_chunks is not set but n_threads is > 1"))
    end
    if (length(fs_chunks) != n_threads) || (needs_vir && length(vir_chunks) != n_threads)
        throw(ArgumentError("length of fs_chunks ($(length(fs_chunks))) or vir_chunks " *
                            "($(length(vir_chunks))) does not match n_threads ($n_threads)"))
    end

    @inbounds if length(pairwise_inters_nonl) > 0
        Threads.@threads for chunk_i in 1:n_threads
            fs_chunk = fs_chunks[chunk_i]
            vir_chunk = (needs_vir ? vir_chunks[chunk_i] : nothing)
            for i in chunk_i:n_threads:n_atoms
                coord_i = coords[i]
                atom_i = atoms[i]
                vel_i = velocities[i]
                for j in (i + 1):n_atoms
                    coord_j = coords[j]
                    atom_j = atoms[j]
                    vel_j = velocities[j]
                    dr = vector(coord_i, coord_j, boundary)
                    f = sum_pairwise_forces(pairwise_inters_nonl, dr, atom_i, atom_j, force_units,
                                            false, coord_i, coord_j, boundary, vel_i, vel_j,
                                            step_n)
                    f_ustrip = checked_ustrip(f, force_units)
                    fs_chunk[i] -= f_ustrip
                    fs_chunk[j] += f_ustrip

                    if needs_vir
                        v = dr * transpose(f)
                        vir_chunk .+= ustrip.(v)
                    end
                end
            end
        end
    end

    @inbounds if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        n_neighbors = length(neighbors)
        block_size = 512
        next_block_start = Threads.Atomic{Int}(1)
        @sync for chunk_i in 1:n_threads
            Threads.@spawn begin
                fs_chunk = fs_chunks[chunk_i]
                vir_chunk = (needs_vir ? vir_chunks[chunk_i] : nothing)
                while true
                    block_start = Threads.atomic_add!(next_block_start, block_size)
                    block_start > n_neighbors && break
                    block_stop = min(block_start + block_size - 1, n_neighbors)
                    for ni in block_start:block_stop
                        i, j, special = neighbors[ni]
                        coord_i = coords[i]
                        coord_j = coords[j]
                        atom_i = atoms[i]
                        atom_j = atoms[j]
                        vel_i = velocities[i]
                        vel_j = velocities[j]
                        dr = vector(coord_i, coord_j, boundary)
                        f = sum_pairwise_forces(pairwise_inters_nl, dr, atom_i, atom_j, force_units,
                                                special, coord_i, coord_j, boundary, vel_i, vel_j,
                                                step_n)
                        f_ustrip = checked_ustrip(f, force_units)
                        fs_chunk[i] -= f_ustrip
                        fs_chunk[j] += f_ustrip

                        if needs_vir
                            v = dr * transpose(f)
                            vir_chunk .+= ustrip.(v)
                        end
                    end
                end
            end
        end
    end

    return fs_nounits
end

@inline function specific_force!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                                 force_units, step_n, inter_list::InteractionList1Atoms, inter_i,
                                 ::Val{needs_vir}) where needs_vir
    i = inter_list.is[inter_i]
    inter = inter_list.inters[inter_i]
    sf = force(inter, coords[i], boundary, atoms[i], force_units, velocities[i], step_n,
               inter_list.data)
    fs_nounits[i] += checked_ustrip(sf.f1, force_units)

    if needs_vir
        r_i = coords[i]
        λ = λ_mixing(MinimumMixing(), atoms[i], atoms[i])
        v = λ * r_i * transpose(sf.f1)
        vir_nounits .+= ustrip.(v)
    end
    return fs_nounits
end

@inline function specific_force!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                                 force_units, step_n, inter_list::InteractionList2Atoms, inter_i,
                                 ::Val{needs_vir}) where needs_vir
    i = inter_list.is[inter_i]
    j = inter_list.js[inter_i]
    inter = inter_list.inters[inter_i]
    sf = force(inter, coords[i], coords[j], boundary, atoms[i], atoms[j], force_units,
               velocities[i], velocities[j], step_n, inter_list.data)
    fs_nounits[i] += checked_ustrip(sf.f1, force_units)
    fs_nounits[j] += checked_ustrip(sf.f2, force_units)

    if needs_vir
        r_ji = vector(coords[j], coords[i], boundary) # Second atom is the reference
        λ = λ_mixing(MinimumMixing(), atoms[i], atoms[j])
        v = λ * r_ji * transpose(sf.f1)
        vir_nounits .+= ustrip.(v)
    end
    return fs_nounits
end

@inline function specific_force!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                                 force_units, step_n, inter_list::InteractionList3Atoms, inter_i,
                                 ::Val{needs_vir}) where needs_vir
    i = inter_list.is[inter_i]
    j = inter_list.js[inter_i]
    k = inter_list.ks[inter_i]
    inter = inter_list.inters[inter_i]
    sf = force(inter, coords[i], coords[j], coords[k], boundary, atoms[i], atoms[j], atoms[k],
               force_units, velocities[i], velocities[j], velocities[k], step_n, inter_list.data)
    fs_nounits[i] += checked_ustrip(sf.f1, force_units)
    fs_nounits[j] += checked_ustrip(sf.f2, force_units)
    fs_nounits[k] += checked_ustrip(sf.f3, force_units)

    if needs_vir
        r_ji = vector(coords[j], coords[i], boundary) # r_i - r_j (second atom is the reference, MIC)
        r_jk = vector(coords[j], coords[k], boundary) # r_k - r_j (second atom is the reference)
        λ_ji = λ_mixing(MinimumMixing(), atoms[j], atoms[i])
        λ_jk = λ_mixing(MinimumMixing(), atoms[j], atoms[k])
        λ = minimum((λ_ji, λ_jk))
        vir_nounits .+= λ * ustrip.(r_ji * transpose(sf.f1) + r_jk * transpose(sf.f3))
    end
    return fs_nounits
end

@inline function specific_force!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                                 force_units, step_n, inter_list::InteractionList4Atoms, inter_i,
                                 ::Val{needs_vir}) where needs_vir
    i = inter_list.is[inter_i]
    j = inter_list.js[inter_i]
    k = inter_list.ks[inter_i]
    l = inter_list.ls[inter_i]
    inter = inter_list.inters[inter_i]
    sf = force(inter, coords[i], coords[j], coords[k], coords[l], boundary, atoms[i], atoms[j],
               atoms[k], atoms[l], force_units, velocities[i], velocities[j], velocities[k],
               velocities[l], step_n, inter_list.data)
    fs_nounits[i] += checked_ustrip(sf.f1, force_units)
    fs_nounits[j] += checked_ustrip(sf.f2, force_units)
    fs_nounits[k] += checked_ustrip(sf.f3, force_units)
    fs_nounits[l] += checked_ustrip(sf.f4, force_units)

    if needs_vir
        r_ji = vector(coords[j], coords[i], boundary) # r_i - r_j
        r_jk = vector(coords[j], coords[k], boundary) # r_k - r_j
        r_jl = vector(coords[j], coords[l], boundary) # r_l - r_j (direct MIC, not sum)
        λ_ji = λ_mixing(MinimumMixing(), atoms[j], atoms[i])
        λ_jk = λ_mixing(MinimumMixing(), atoms[j], atoms[k])
        λ_jl = λ_mixing(MinimumMixing(), atoms[j], atoms[l])
        λ = minimum((λ_ji, λ_jk, λ_jl))
        vir_nounits .+= λ * ustrip.(r_ji * transpose(sf.f1) +
                                r_jk * transpose(sf.f3) +
                                r_jl * transpose(sf.f4) )
    end
    return fs_nounits
end

@inline function specific_force!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                                 force_units, step_n, inter_list::InteractionList5Atoms, inter_i,
                                 ::Val{needs_vir}) where needs_vir
    i = inter_list.is[inter_i]
    j = inter_list.js[inter_i]
    k = inter_list.ks[inter_i]
    l = inter_list.ls[inter_i]
    m = inter_list.ms[inter_i]
    inter = inter_list.inters[inter_i]
    sf = force(inter, coords[i], coords[j], coords[k], coords[l], coords[m], boundary, atoms[i],
               atoms[j], atoms[k], atoms[l], atoms[m], force_units, velocities[i], velocities[j],
               velocities[k], velocities[l], velocities[m], step_n, inter_list.data)
    fs_nounits[i] += checked_ustrip(sf.f1, force_units)
    fs_nounits[j] += checked_ustrip(sf.f2, force_units)
    fs_nounits[k] += checked_ustrip(sf.f3, force_units)
    fs_nounits[l] += checked_ustrip(sf.f4, force_units)
    fs_nounits[m] += checked_ustrip(sf.f5, force_units)

    if needs_vir
        r_ji = vector(coords[j], coords[i], boundary) # r_i - r_j
        r_jk = vector(coords[j], coords[k], boundary) # r_k - r_j
        r_jl = vector(coords[j], coords[l], boundary) # r_l - r_j (direct MIC, not sum)
        r_jm = vector(coords[j], coords[m], boundary) # r_m - r_j
        λ_ji = λ_mixing(MinimumMixing(), atoms[j], atoms[i])
        λ_jk = λ_mixing(MinimumMixing(), atoms[j], atoms[k])
        λ_jl = λ_mixing(MinimumMixing(), atoms[j], atoms[l])
        λ_jm = λ_mixing(MinimumMixing(), atoms[j], atoms[m])
        λ = minimum((λ_ji, λ_jk, λ_jl, λ_jm))
        vir_nounits .+= λ * ustrip.(r_ji * transpose(sf.f1) +
                                r_jk * transpose(sf.f3) +
                                r_jl * transpose(sf.f4) +
                                r_jm * transpose(sf.f5))
    end
    return fs_nounits
end

function specific_forces_inter_list!(fs_nounits, vir_nounits, atoms, coords, velocities,
                                     boundary, force_units, step_n, inter_list, inter_range,
                                     ::Val{needs_vir}) where needs_vir
    for inter_i in inter_range
        specific_force!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                        force_units, step_n, inter_list, inter_i, Val(needs_vir))
    end
    return fs_nounits
end

function specific_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords,
                               velocities, boundary, force_units, sils_1_atoms, sils_2_atoms,
                               sils_3_atoms, sils_4_atoms, sils_5_atoms, step_n, ::Val{1},
                               ::Val{needs_vir}) where needs_vir
    for inter_list in sils_1_atoms
        specific_forces_inter_list!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                    force_units, step_n, inter_list, eachindex(inter_list.inters), Val(needs_vir))
    end

    for inter_list in sils_2_atoms
        specific_forces_inter_list!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                    force_units, step_n, inter_list, eachindex(inter_list.inters), Val(needs_vir))
    end

    for inter_list in sils_3_atoms
        specific_forces_inter_list!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                    force_units, step_n, inter_list, eachindex(inter_list.inters), Val(needs_vir))
    end

    for inter_list in sils_4_atoms
        specific_forces_inter_list!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                    force_units, step_n, inter_list, eachindex(inter_list.inters), Val(needs_vir))
    end

    for inter_list in sils_5_atoms
        specific_forces_inter_list!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary,
                    force_units, step_n, inter_list, eachindex(inter_list.inters), Val(needs_vir))
    end

    return fs_nounits
end

function chunk_range(n_items, chunk_i, n_chunks)
    chunk_size = cld(n_items, n_chunks)
    first_i = (chunk_i - 1) * chunk_size + 1
    last_i = min(first_i + chunk_size - 1, n_items)
    return first_i:last_i
end

function specific_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords,
                               velocities, boundary, force_units, sils_1_atoms, sils_2_atoms,
                               sils_3_atoms, sils_4_atoms, sils_5_atoms, step_n, ::Val{n_threads},
                               ::Val{needs_vir}) where {n_threads, needs_vir}
    if isnothing(fs_chunks) || (needs_vir && isnothing(vir_chunks))
        throw(ArgumentError("fs_chunks / vir_chunks is not set but n_threads is > 1"))
    end
    if (length(fs_chunks) != n_threads) || (needs_vir && length(vir_chunks) != n_threads)
        throw(ArgumentError("length of fs_chunks ($(length(fs_chunks))) or vir_chunks " *
                            "($(length(vir_chunks))) does not match n_threads ($n_threads)"))
    end

    @sync for chunk_i in 1:n_threads
        Threads.@spawn begin
            fs_chunk = fs_chunks[chunk_i]
            vir_chunk = (needs_vir ? vir_chunks[chunk_i] : nothing)
            for inter_list in sils_1_atoms
                cr = chunk_range(length(inter_list.inters), chunk_i, n_threads)
                specific_forces_inter_list!(fs_chunk, vir_chunk, atoms, coords, velocities,
                            boundary, force_units, step_n, inter_list, cr, Val(needs_vir))
            end
            for inter_list in sils_2_atoms
                cr = chunk_range(length(inter_list.inters), chunk_i, n_threads)
                specific_forces_inter_list!(fs_chunk, vir_chunk, atoms, coords, velocities,
                            boundary, force_units, step_n, inter_list, cr, Val(needs_vir))
            end
            for inter_list in sils_3_atoms
                cr = chunk_range(length(inter_list.inters), chunk_i, n_threads)
                specific_forces_inter_list!(fs_chunk, vir_chunk, atoms, coords, velocities,
                            boundary, force_units, step_n, inter_list, cr, Val(needs_vir))
            end
            for inter_list in sils_4_atoms
                cr = chunk_range(length(inter_list.inters), chunk_i, n_threads)
                specific_forces_inter_list!(fs_chunk, vir_chunk, atoms, coords, velocities,
                            boundary, force_units, step_n, inter_list, cr, Val(needs_vir))
            end
            for inter_list in sils_5_atoms
                cr = chunk_range(length(inter_list.inters), chunk_i, n_threads)
                specific_forces_inter_list!(fs_chunk, vir_chunk, atoms, coords, velocities,
                            boundary, force_units, step_n, inter_list, cr, Val(needs_vir))
            end
        end
    end

    return fs_nounits
end

function forces!(fs,
                 sys::System{D, <:AbstractGPUArray, T},
                 neighbors,
                 step_n::Integer,
                 buffers::BuffersGPU,
                 ::Val{needs_vir};
                 n_threads::Integer=Threads.nthreads(),
                 pairwise_inters=sys.pairwise_inters,
                 specific_inter_lists=sys.specific_inter_lists,
                 general_inters=sys.general_inters) where {D, T, needs_vir}
    if needs_vir
        fill!(buffers.virial, zero(T) * sys.energy_units)
        fill!(buffers.virial_nounits, zero(T))
    else
        invalidate_interaction_virial!(buffers.validity)
        invalidate_total_virial!(buffers.validity)
    end
    invalidate_pressure!(buffers.validity)
    fill!(buffers.kin_tensor, zero(T) * sys.energy_units)
    fill!(buffers.pres_tensor, zero(T) * (sys.energy_units == NoUnits ? NoUnits : u"bar"))
    fill!(buffers.fs_mat, zero(T))
    fill!(buffers.fs_mat_reordered, zero(T))

    pairwise_inters_nonl = filter(!use_neighbors, values(pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        n = length(sys)
        nbs = NoNeighborList(n)
        pairwise_forces_loop_gpu!(buffers, sys, pairwise_inters_nonl, nbs, Val(needs_vir), step_n)
    end

    pairwise_inters_nl = filter(use_neighbors, values(pairwise_inters))
    if length(pairwise_inters_nl) > 0
        pairwise_forces_loop_gpu!(buffers, sys, pairwise_inters_nl, neighbors, Val(needs_vir), step_n)
    end

    for inter_list in values(specific_inter_lists)
        specific_forces_gpu!(buffers.fs_mat, buffers.virial_nounits,
                            inter_list, sys.coords, sys.velocities, sys.atoms,
                            sys.boundary, Val(needs_vir), step_n, sys.force_units, Val(T))
    end

    apply_force_units_gpu!(fs, buffers.fs_mat, sys.force_units, Val(D), Val(T))

    if needs_vir
        buffers.virial .+= from_device(buffers.virial_nounits) .* sys.energy_units
    end

    for inter in values(general_inters)
        AtomsCalculators.forces!(fs, sys, inter; neighbors=neighbors, step_n=step_n,
                                 n_threads=n_threads, buffers=buffers, needs_vir=needs_vir)
    end
    distribute_forces!(fs, sys, buffers)

    if needs_vir
        mark_interaction_virial!(buffers.validity, step_n)
        if length(sys.constraints) == 0
            mark_total_virial!(buffers.validity, step_n)
        end
    end

    return fs, buffers
end
