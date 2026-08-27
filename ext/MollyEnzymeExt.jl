# Code for taking gradients with Enzyme
# This file is only loaded when Enzyme is imported

module MollyEnzymeExt

using Molly
using Enzyme
using Enzyme.EnzymeCore.EnzymeRules: EnzymeRules, RevConfig, Annotation
using FFTW
using GPUArrays
using KernelAbstractions

const GPUArraysCore = GPUArrays.GPUArraysCore

EnzymeRules.inactive(::typeof(is_on_gpu), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.default_strictness), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_strictness), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_specific_inter_lists), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_neighbor_finder), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_float_types), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_n_dims), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_float_type_consistency), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_cutoff_box_size), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.report_issue), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_force_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_energy_units), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.has_infinite_boundary), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.n_infinite_dims), args...) = nothing
EnzymeRules.inactive(::typeof(random_coord), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocity), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocities), args...) = nothing
EnzymeRules.inactive(::typeof(random_velocities!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.setup_virtual_sites), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_gbsa_n_threads), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.atoms_bonded_to_N), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.lookup_table), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.radius_classes), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.gb_log_scaling), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.calculate_n_dof), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.default_show_progress), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.default_check_nans), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.check_array_nans), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.isnan_svec_array), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.setup_progress), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.setup_progress_minimizer), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.next_nograd!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.update_nograd!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.needs_virial_schedule), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.parse_splitting), args...) = nothing
EnzymeRules.inactive(::typeof(use_neighbors), args...) = nothing
EnzymeRules.inactive(::typeof(find_neighbors), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.uses_gpu_neighbor_finder), args...) = nothing
EnzymeRules.inactive_type(::Type{<:NoNeighborFinder}) = true
EnzymeRules.inactive_type(::Type{<:GPUNeighborFinder}) = true
EnzymeRules.inactive_type(::Type{<:DistanceNeighborFinder}) = true
EnzymeRules.inactive_type(::Type{<:TreeNeighborFinder}) = true
EnzymeRules.inactive_type(::Type{<:CellListMapNeighborFinder}) = true
EnzymeRules.inactive(::typeof(visualize), args...) = nothing
EnzymeRules.inactive(::typeof(place_atoms), args...) = nothing
EnzymeRules.inactive(::typeof(place_diatomics), args...) = nothing
EnzymeRules.inactive(::typeof(read_frame!), args...) = nothing
EnzymeRules.inactive(::typeof(Molly.kabsch_rotation_nograd), args...) = nothing

# With the whole-function rules below, the GPU buffers are pure scratch: the rules
# recompute every cotangent from the system and the incoming seed and never read a buffer
# shadow. Allocating them is therefore not part of the derivative, which also keeps the
# ~20 device arrays `init_buffers!` allocates out of Enzyme's shadow bookkeeping. The CPU
# method is untouched, where the buffers do carry gradients.
EnzymeRules.inactive(::typeof(Molly.init_buffers!),
                     ::Molly.System{<:Any, <:AbstractGPUArray}, args...) = nothing

# Differentiable PME

# See https://github.com/EnzymeAD/Enzyme.jl/issues/2298
EnzymeRules.inactive(::typeof(plan_rfft ), args...) = nothing
EnzymeRules.inactive(::typeof(plan_brfft), args...) = nothing

# See rfft and brfft rrules in AbstractFFTs.jl

# The modes of the half spectrum that are not their own conjugate each stand for two modes
# of the full mesh. The real transforms fold that factor of two in, so the adjoints have to
# take it back out.
function hermitian_scale(charge_grid::AbstractArray{T, 3}, recip_grid) where T
    n = size(charge_grid, 1)
    return reshape([(isone(i) || 2*(i-1) == n ? one(T) : T(2))
                    for i in 1:size(recip_grid, 1)], :, 1, 1)
end

function EnzymeRules.augmented_primal(config, ::Const{typeof(Molly.grad_safe_fft!)}, t,
                                      charge_grid, recip_grid, fft_plan)
    Molly.grad_safe_fft!(charge_grid.val, recip_grid.val, fft_plan.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config, ::Const{typeof(Molly.grad_safe_fft!)}, dret, tape,
                             charge_grid, recip_grid, fft_plan)
    scale = hermitian_scale(charge_grid.val, recip_grid.val)
    # The real grid is read but not written, so its shadow is added to
    charge_grid.dval .+= brfft(recip_grid.dval ./ scale, size(charge_grid.val, 1))
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(config, ::Const{typeof(Molly.grad_safe_bfft!)}, t,
                                      charge_grid, recip_grid, bfft_plan)
    Molly.grad_safe_bfft!(charge_grid.val, recip_grid.val, bfft_plan.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config, ::Const{typeof(Molly.grad_safe_bfft!)}, dret, tape,
                             charge_grid, recip_grid, bfft_plan)
    scale = hermitian_scale(charge_grid.val, recip_grid.val)
    # The real grid is overwritten, so its shadow is consumed here
    recip_grid.dval .= scale .* rfft(charge_grid.dval)
    charge_grid.dval .= zero(eltype(charge_grid.dval))
    return (nothing, nothing, nothing)
end

# Calculate the gradient of a CV with respect to the input coordinates
# Works for systems with and without units
function Molly.cv_gradient(cv_type, coords, atoms=nothing, boundary=nothing, velocities=nothing)
    d_coords = zero(coords)
    unit_arr = Any[u"nm"]

    _, cv_val_ustrip = autodiff(
        set_runtime_activity(ReverseWithPrimal), # set_runtime_activity necessary for units
        Molly.calculate_cv_ustrip!,
        Active,
        Const(unit_arr),
        Const(cv_type),
        Duplicated(coords, d_coords),
        Const(atoms),
        Const(boundary),
        Const(velocities),
    )

    # Correct the units after the ustrip
    u = only(unit_arr)
    d_coords = d_coords .* u ./ unit(d_coords[1][1])^2

    return d_coords, cv_val_ustrip * u
end

# Optimisations for dict_get gradients

# Tape encoding: the value came from `default`, the slot to accumulate into in the shadow
# dictionary, or the value came from the dictionary but there is nowhere to accumulate it
const dict_get_from_default = 0
const dict_get_no_shadow = -1

@inline dict_get_default_cotangent(::Const, dret, tape) = nothing
@inline function dict_get_default_cotangent(default::Active, dret, tape)
    (dret isa Type) && return zero(default.val)
    return tape == dict_get_from_default ? dret.val : zero(default.val)
end
@inline function dict_get_default_cotangent(default::Annotation, dret, tape)
    error("Enzyme passed $(typeof(default)) for the default value of dict_get, which the " *
          "rule cannot accumulate into")
end

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Const{typeof(Molly.dict_get)},
                                      ::Type{RT}, dic::Annotation{<:Dict},
                                      key::Annotation, default::Annotation) where RT
    EnzymeRules.width(config) == 1 ||
        error("the Molly dict_get reverse rule does not support batch mode")
    idx = Base.ht_keyindex(dic.val, key.val)
    T = typeof(default.val)
    val = (idx > 0 ? T(@inbounds dic.val.vals[idx]) : default.val)
    tape = if idx <= 0
        dict_get_from_default
    elseif dic isa Const || dic.dval === dic.val
        # `set_runtime_activity` can hand a rule a shadow that is the primal itself, and
        # accumulating into that would corrupt the parameter dictionary
        dict_get_no_shadow
    else
        Base.ht_keyindex(dic.dval, key.val) # -1 when the shadow lacks the key
    end
    primal = (EnzymeRules.needs_primal(config) ? val : nothing)
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       Int}(primal, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{typeof(Molly.dict_get)},
                             dret, tape::Int, dic::Annotation{<:Dict},
                             key::Annotation, default::Annotation)
    if tape > 0 && !(dret isa Type)
        vals = dic.dval.vals
        @inbounds vals[tape] += convert(eltype(vals), dret.val)
    end
    return (nothing, nothing, dict_get_default_cotangent(default, dret, tape))
end

# Reverse-mode rules for the GPU force and energy kernel launchers
#
# The adjoint kernels do not differentiate the primal kernel. They call
# `Enzyme.autodiff_deferred` on the *per-interaction* function - a small scalar function
# of a few coordinates, atoms and one interaction - contracted against the incoming
# cotangent, which turns the vector-Jacobian product into a scalar reverse-mode call.
# Enzyme then scatters the cotangents into the shadow arrays itself. That is safe across
# threads because Enzyme emits atomic adds for every adjoint memory update when the
# target is a GPU (`atomicAdd = parallel = true` in Enzyme's compiler.jl), so the fact
# that many pairs touch the same atom needs no handling here.
#
# The interaction *tuple* of a pairwise interaction is passed through a one-element
# device array rather than by value, so its cotangent is accumulated in device memory by
# the same mechanism instead of needing a cross-thread reduction.

EnzymeRules.inactive(::typeof(Base.task_local_storage), args...) = nothing
EnzymeRules.inactive_noinl(::typeof(Base.task_local_storage), args...) = nothing
EnzymeRules.inactive(::typeof(GPUArraysCore.assertscalar), args...) = nothing
EnzymeRules.inactive_noinl(::typeof(GPUArraysCore.assertscalar), args...) = nothing
if isdefined(GPUArraysCore, :_assertscalar)
    EnzymeRules.inactive(::typeof(GPUArraysCore._assertscalar), args...) = nothing
    EnzymeRules.inactive_noinl(::typeof(GPUArraysCore._assertscalar), args...) = nothing
end

# CUDA.jl defines an Enzyme rule for `GPUArrays._mapreduce` only for an `Active` return,
# so a reduction whose result Enzyme has already decided is inactive - `sum` over the
# `Bool` virtual site flags in the `System` constructor, for instance - finds the rule by
# signature and then fails to dispatch. A constant result carries no derivative, so the
# reverse pass is empty.
function EnzymeRules.augmented_primal(config::RevConfig,
                                      ofn::Const{typeof(GPUArrays._mapreduce)},
                                      ::Type{RT}, f::Const, op::Const,
                                      A::Annotation{<:AbstractGPUArray};
                                      dims, init) where {RT <: Const}
    primal = EnzymeRules.needs_primal(config) ?
             ofn.val(f.val, op.val, A.val; dims=dims, init=init) : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       Nothing}(primal, nothing, nothing)
end

function EnzymeRules.reverse(config::RevConfig, ofn::Const{typeof(GPUArrays._mapreduce)},
                             ::Type{RT}, tape, f::Const, op::Const,
                             A::Annotation{<:AbstractGPUArray};
                             dims, init) where {RT <: Const}
    return (nothing, nothing, nothing)
end

# Shadow of a field of an annotated argument, or `nothing` when there is no gradient to
# accumulate. Under runtime activity Enzyme may hand back a shadow that aliases the
# primal for an inactive value; writing to that would corrupt the primal, so it is
# treated as constant.
@inline grad_shadow(::Const, _) = nothing
@inline function grad_shadow(x::Annotation, f::Symbol)
    val, dval = getfield(x.val, f), getfield(x.dval, f)
    return val === dval ? nothing : dval
end

@inline whole_shadow(::Const) = nothing
@inline whole_shadow(x::Annotation) = x.val === x.dval ? nothing : x.dval

# `Duplicated` when there is a shadow to write into, `Const` otherwise. Splitting on the
# `Nothing` type means each combination gets its own kernel specialisation, so the
# unused branches cost nothing on the device.
@inline dup(x, dx) = Duplicated(x, dx, false)
@inline dup(x, ::Nothing) = Const(x)

# The neighbour list view used by the primal launchers
@inline function grad_nbs(neighbors)
    if typeof(neighbors) == Molly.NoNeighborList
        return neighbors
    else
        return @view neighbors.list[1:neighbors.n]
    end
end

# Cotangent of an argument Enzyme passes by value. `Active` arguments take their
# cotangent through the rule's return value, mixed-activity ones through a shadow
# reference. Anything else has no gradient to accumulate.
@inline add_cotangents(a::Tuple, b::Tuple) = map(add_cotangents, a, b)
@inline add_cotangents(a, b) = a + b

@inline value_cotangent!(::Active, dval) = dval
@inline value_cotangent!(::Const, dval) = nothing
@inline function value_cotangent!(arg::MixedDuplicated, dval)
    arg.dval[] = add_cotangents(arg.dval[], dval)
    return nothing
end
@inline function value_cotangent!(arg::Annotation, dval)
    error("Enzyme passed $(typeof(arg)) for an interaction tuple, which the GPU reverse " *
          "rules cannot accumulate into")
end

# Scratch the reverse pass needs on the device: an array holding the pairwise interaction
# tuple, its shadow, a one-element array for the scalar seed, and a host array to read the
# interaction cotangent back into. All four are tiny and are reused across calls through
# `buffers.grad_scratch`, so a gradient inside `simulate!` allocates nothing per step. They
# are only live inside one reverse pass, so pooling them is safe - unlike the tape below,
# which must be per call.
#
# The interaction tuple is replicated over `n_grad_slots` slots and each workgroup uses
# its own, because every thread accumulating into a single shadow serialises and costs
# more than the rest of the kernel put together.
const n_grad_slots = 512

struct GradScratch{I, S, H}
    inters_arr::I
    dinters_arr::I
    seed::S
    dinters_host::H
end

function GradScratch(inters, seed_eltype, AT)
    zi = Enzyme.make_zero(inters)
    inters_arr = Molly.to_device(fill(inters, n_grad_slots), AT)
    dinters_arr = Molly.to_device(fill(zi, n_grad_slots), AT)
    seed = Molly.to_device([zero(seed_eltype)], AT)
    return GradScratch(inters_arr, dinters_arr, seed, fill(zi, n_grad_slots))
end

# `nothing` buffers (or a scratch built for a different interaction tuple) means allocate
@inline scratch_matches(::Nothing, inters, seed_eltype) = false
@inline function scratch_matches(sc::GradScratch, inters, seed_eltype)
    return eltype(sc.inters_arr) === typeof(inters) && eltype(sc.seed) === seed_eltype
end

function grad_scratch!(buffers, inters, seed_eltype, AT)
    sc = buffers.grad_scratch[]
    if !scratch_matches(sc, inters, seed_eltype)
        sc = GradScratch(inters, seed_eltype, AT)
        buffers.grad_scratch[] = sc
    end
    # `fill!` runs on the device, so no host allocation and no synchronisation
    fill!(sc.inters_arr, inters)
    fill!(sc.dinters_arr, Enzyme.make_zero(inters))
    return sc::GradScratch
end

# The kernel-level rules below allocate their own one-element arrays; they are not on the
# hot path (the whole-function rules take over) so they are left simple. A one-element
# array means every workgroup lands on slot 1, which is the old behaviour.
function inters_device_pair(inters, AT)
    return Molly.to_device([inters], AT), Molly.to_device([Enzyme.make_zero(inters)], AT)
end

# One slot per workgroup, so the threads that share a slot are the ones already sharing a
# cache line. `group` comes from `@index(Group, Linear)` inside the kernel.
@inline grad_slot(group, inters_arr) = mod1(Int(group), length(inters_arr))

# Pairwise potential energy

@inline function pairwise_pe_contrib(coords, atoms, inters_arr, slot, velocities, boundary,
                                     i, j, special, step_n, ::Val{E}, seed) where E
    @inbounds begin
        inters = inters_arr[slot]
        coord_i, coord_j = coords[i], coords[j]
        vel_i = Molly.kernel_maybe_velocity(velocities, i)
        vel_j = Molly.kernel_maybe_velocity(velocities, j)
        dr = vector(coord_i, coord_j, boundary)
        pe = Molly.sum_pairwise_potentials_gpu(inters, dr, atoms[i], atoms[j], Val(E), special,
                                               coord_i, coord_j, boundary, vel_i, vel_j, step_n)
        return seed * ustrip(pe[1])
    end
end

@kernel inbounds=true function pairwise_pe_rev_kernel!(coords, dcoords, atoms, datoms,
                        inters_arr, dinters_arr, velocities, dvelocities, boundary,
                        neighbors, step_n, eu, dpe_vec)
    inter_i = @index(Global, Linear)

    if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        seed = dpe_vec[1]
        slot = grad_slot(@index(Group, Linear), inters_arr)
        Enzyme.autodiff_deferred(
            Reverse, Const(pairwise_pe_contrib), Active{eltype(dpe_vec)},
            dup(coords, dcoords), dup(atoms, datoms), dup(inters_arr, dinters_arr),
            Const(slot), dup(velocities, dvelocities), Const(boundary), Const(i), Const(j),
            Const(special), Const(step_n), Const(eu), Const(seed),
        )
    end
end

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Const{typeof(Molly.pairwise_pe_loop_gpu!)},
                                      ::Type{RT}, pe_vec_nounits::Annotation,
                                      buffers::Annotation, sys::Annotation,
                                      pairwise_inters::Annotation, neighbors::Annotation,
                                      step_n::Annotation) where RT
    func.val(pe_vec_nounits.val, buffers.val, sys.val, pairwise_inters.val, neighbors.val,
             step_n.val)
    primal = EnzymeRules.needs_primal(config) ? pe_vec_nounits.val : nothing
    shadow = (EnzymeRules.needs_shadow(config) && !(pe_vec_nounits isa Const)) ?
             pe_vec_nounits.dval : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       Nothing}(primal, shadow, nothing)
end

function EnzymeRules.reverse(config::RevConfig,
                             func::Const{typeof(Molly.pairwise_pe_loop_gpu!)},
                             ::Type{RT}, tape, pe_vec_nounits::Annotation,
                             buffers::Annotation, sys::Annotation,
                             pairwise_inters::Annotation, neighbors::Annotation,
                             step_n::Annotation) where RT
    dinters = nothing
    dpe_vec = whole_shadow(pe_vec_nounits)
    if !isnothing(dpe_vec) && length(neighbors.val) > 0
        s = sys.val
        AT = array_type(s)
        nbs = grad_nbs(neighbors.val)
        vels = (Molly.any_uses_velocity(pairwise_inters.val) ? s.velocities : nothing)
        dvels = isnothing(vels) ? nothing : grad_shadow(sys, :velocities)
        inters_arr, dinters_arr = inters_device_pair(pairwise_inters.val, AT)

        backend = get_backend(s.coords)
        kernel! = pairwise_pe_rev_kernel!(backend, Molly.gpu_threads_pairwise(length(nbs)))
        kernel!(s.coords, grad_shadow(sys, :coords), s.atoms, grad_shadow(sys, :atoms),
                inters_arr, dinters_arr, vels, dvels, s.boundary, nbs, step_n.val,
                Val(s.energy_units), dpe_vec; ndrange=length(nbs))
        dinters = Molly.from_device(dinters_arr)[1]
    end
    if !(pairwise_inters isa Const)
        dinters = isnothing(dinters) ? Enzyme.make_zero(pairwise_inters.val) : dinters
        return (nothing, nothing, nothing, value_cotangent!(pairwise_inters, dinters),
                nothing, nothing)
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

# Pairwise forces

@inline function pairwise_force_contrib(coords, atoms, inters_arr, slot, velocities,
                                        boundary, i, j, special, step_n, ::Val{F}, ::Val{D},
                                        dfs_mat) where {F, D}
    @inbounds begin
        inters = inters_arr[slot]
        coord_i, coord_j = coords[i], coords[j]
        vel_i = Molly.kernel_maybe_velocity(velocities, i)
        vel_j = Molly.kernel_maybe_velocity(velocities, j)
        dr = vector(coord_i, coord_j, boundary)
        f = Molly.sum_pairwise_forces_gpu(inters, dr, atoms[i], atoms[j], Val(F), special,
                                          coord_i, coord_j, boundary, vel_i, vel_j, step_n)
        s = zero(eltype(dfs_mat))
        for dim in 1:D
            fval = ustrip(f[dim])
            # The primal does fs_mat[dim, i] += -fval and fs_mat[dim, j] += fval
            s += fval * (dfs_mat[dim, j] - dfs_mat[dim, i])
        end
        return s
    end
end

@kernel inbounds=true function pairwise_force_rev_kernel!(coords, dcoords, atoms, datoms,
                        inters_arr, dinters_arr, velocities, dvelocities, boundary,
                        neighbors, step_n, fu, dv, dfs_mat)
    inter_i = @index(Global, Linear)

    if inter_i <= length(neighbors)
        i, j, special = neighbors[inter_i]
        slot = grad_slot(@index(Group, Linear), inters_arr)
        Enzyme.autodiff_deferred(
            Reverse, Const(pairwise_force_contrib), Active{eltype(dfs_mat)},
            dup(coords, dcoords), dup(atoms, datoms), dup(inters_arr, dinters_arr),
            Const(slot), dup(velocities, dvelocities), Const(boundary), Const(i), Const(j),
            Const(special), Const(step_n), Const(fu), Const(dv), Const(dfs_mat),
        )
    end
end

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Const{typeof(Molly.pairwise_forces_loop_gpu!)},
                                      ::Type{RT}, buffers::Annotation, sys::Annotation,
                                      pairwise_inters::Annotation, neighbors::Annotation,
                                      needs_vir::Const, step_n::Annotation) where RT
    func.val(buffers.val, sys.val, pairwise_inters.val, neighbors.val, needs_vir.val,
             step_n.val)
    primal = EnzymeRules.needs_primal(config) ? buffers.val : nothing
    shadow = (EnzymeRules.needs_shadow(config) && !(buffers isa Const)) ? buffers.dval : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       Nothing}(primal, shadow, nothing)
end

function EnzymeRules.reverse(config::RevConfig,
                             func::Const{typeof(Molly.pairwise_forces_loop_gpu!)},
                             ::Type{RT}, tape, buffers::Annotation, sys::Annotation,
                             pairwise_inters::Annotation, neighbors::Annotation,
                             needs_vir::Const{Val{NV}}, step_n::Annotation) where {RT, NV}
    NV && error("gradients through the virial are not implemented on the GPU, use " *
                "forces rather than forces_virial")
    dinters = nothing
    dfs_mat = grad_shadow(buffers, :fs_mat)
    if !isnothing(dfs_mat) && length(neighbors.val) > 0
        s = sys.val
        D = length(eltype(s.coords))
        AT = array_type(s)
        nbs = grad_nbs(neighbors.val)
        vels = (Molly.any_uses_velocity(pairwise_inters.val) ? s.velocities : nothing)
        dvels = isnothing(vels) ? nothing : grad_shadow(sys, :velocities)
        inters_arr, dinters_arr = inters_device_pair(pairwise_inters.val, AT)

        backend = get_backend(s.coords)
        kernel! = pairwise_force_rev_kernel!(backend, Molly.gpu_threads_pairwise(length(nbs)))
        kernel!(s.coords, grad_shadow(sys, :coords), s.atoms, grad_shadow(sys, :atoms),
                inters_arr, dinters_arr, vels, dvels, s.boundary, nbs, step_n.val,
                Val(s.force_units), Val(D), dfs_mat; ndrange=length(nbs))
        dinters = Molly.from_device(dinters_arr)[1]
    end
    if !(pairwise_inters isa Const)
        dinters = isnothing(dinters) ? Enzyme.make_zero(pairwise_inters.val) : dinters
        return (nothing, nothing, value_cotangent!(pairwise_inters, dinters), nothing,
                nothing, nothing)
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

# Specific interactions
# The five arities differ only in how many atom indices they carry, so the per-interaction
# functions, the adjoint kernels and the rules are generated from one template each.

for (N, idxs) in ((1, (:is,)), (2, (:is, :js)), (3, (:is, :js, :ks)),
                  (4, (:is, :js, :ks, :ls)), (5, (:is, :js, :ks, :ls, :ms)))
    pe_contrib = Symbol("specific_pe_contrib_", N)
    force_contrib = Symbol("specific_force_contrib_", N)
    pe_kernel = Symbol("specific_pe_", N, "_rev_kernel!")
    force_kernel = Symbol("specific_force_", N, "_rev_kernel!")
    inds = [Symbol("i", n) for n in 1:N]
    fnames = [Symbol("f", n) for n in 1:N]
    coord_args = [:(coords[$(inds[n])]) for n in 1:N]
    atom_args = [:(atoms[$(inds[n])]) for n in 1:N]
    vel_args = [:(velocities[$(inds[n])]) for n in 1:N]
    idx_reads = [:($(inds[n]) = $(idxs[n])[inter_i]) for n in 1:N]

    @eval begin
        @inline function $pe_contrib(coords, atoms, inters, velocities, data, boundary,
                                     inter_i, $(inds...), step_n, ::Val{E}, seed) where E
            @inbounds begin
                pe = Molly.potential_energy_gpu(inters[inter_i], $(coord_args...), boundary,
                                                $(atom_args...), E, $(vel_args...), step_n, data)
                return seed * ustrip(pe)
            end
        end

        @kernel inbounds=true function $pe_kernel(coords, dcoords, atoms, datoms, inters,
                                dinters, velocities, dvelocities, data, ddata, boundary,
                                step_n, $(idxs...), eu, dpe_vec)
            inter_i = @index(Global, Linear)

            if inter_i <= length(is)
                $(idx_reads...)
                seed = dpe_vec[1]
                Enzyme.autodiff_deferred(
                    Reverse, Const($pe_contrib), Active{eltype(dpe_vec)},
                    dup(coords, dcoords), dup(atoms, datoms), dup(inters, dinters),
                    dup(velocities, dvelocities), dup(data, ddata), Const(boundary),
                    Const(inter_i), $([:(Const($(inds[n]))) for n in 1:N]...),
                    Const(step_n), Const(eu), Const(seed),
                )
            end
        end

        @inline function $force_contrib(coords, atoms, inters, velocities, data, boundary,
                                        inter_i, $(inds...), step_n, ::Val{F}, ::Val{D},
                                        dfs_mat) where {F, D}
            @inbounds begin
                fs = Molly.force_gpu(inters[inter_i], $(coord_args...), boundary,
                                     $(atom_args...), F, $(vel_args...), step_n, data)
                s = zero(eltype(dfs_mat))
                for dim in 1:D
                    $([:(s += ustrip(fs.$(fnames[n])[dim]) * dfs_mat[dim, $(inds[n])])
                       for n in 1:N]...)
                end
                return s
            end
        end

        @kernel inbounds=true function $force_kernel(coords, dcoords, atoms, datoms, inters,
                                dinters, velocities, dvelocities, data, ddata, boundary,
                                step_n, $(idxs...), fu, dv, dfs_mat)
            inter_i = @index(Global, Linear)

            if inter_i <= length(is)
                $(idx_reads...)
                Enzyme.autodiff_deferred(
                    Reverse, Const($force_contrib), Active{eltype(dfs_mat)},
                    dup(coords, dcoords), dup(atoms, datoms), dup(inters, dinters),
                    dup(velocities, dvelocities), dup(data, ddata), Const(boundary),
                    Const(inter_i), $([:(Const($(inds[n]))) for n in 1:N]...),
                    Const(step_n), Const(fu), Const(dv), Const(dfs_mat),
                )
            end
        end
    end
end

# The interaction list arity is a type, so the kernel for it is picked at compile time
for (N, IL) in ((1, :InteractionList1Atoms), (2, :InteractionList2Atoms),
                (3, :InteractionList3Atoms), (4, :InteractionList4Atoms),
                (5, :InteractionList5Atoms))
    @eval begin
        @inline specific_pe_rev_kernel(::$IL) = $(Symbol("specific_pe_", N, "_rev_kernel!"))
        @inline specific_force_rev_kernel(::$IL) = $(Symbol("specific_force_", N, "_rev_kernel!"))
    end
end

@inline inter_list_indices(il::InteractionList1Atoms) = (il.is,)
@inline inter_list_indices(il::InteractionList2Atoms) = (il.is, il.js)
@inline inter_list_indices(il::InteractionList3Atoms) = (il.is, il.js, il.ks)
@inline inter_list_indices(il::InteractionList4Atoms) = (il.is, il.js, il.ks, il.ls)
@inline inter_list_indices(il::InteractionList5Atoms) = (il.is, il.js, il.ks, il.ls, il.ms)

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Const{typeof(Molly.specific_pe_gpu!)},
                                      ::Type{RT}, pe_vec_nounits::Annotation,
                                      inter_list::Annotation, coords::Annotation{<:AbstractGPUArray},
                                      velocities::Annotation, atoms::Annotation,
                                      boundary::Annotation, step_n::Annotation,
                                      energy_units::Annotation, TH::Const) where RT
    func.val(pe_vec_nounits.val, inter_list.val, coords.val, velocities.val, atoms.val,
             boundary.val, step_n.val, energy_units.val, TH.val)
    primal = EnzymeRules.needs_primal(config) ? pe_vec_nounits.val : nothing
    shadow = (EnzymeRules.needs_shadow(config) && !(pe_vec_nounits isa Const)) ?
             pe_vec_nounits.dval : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       Nothing}(primal, shadow, nothing)
end

function EnzymeRules.reverse(config::RevConfig,
                             func::Const{typeof(Molly.specific_pe_gpu!)},
                             ::Type{RT}, tape, pe_vec_nounits::Annotation,
                             inter_list::Annotation, coords::Annotation{<:AbstractGPUArray},
                             velocities::Annotation, atoms::Annotation,
                             boundary::Annotation, step_n::Annotation,
                             energy_units::Annotation, TH::Const) where RT
    dpe_vec = whole_shadow(pe_vec_nounits)
    il = inter_list.val
    if !isnothing(dpe_vec) && length(il) > 0
        backend = get_backend(coords.val)
        kernel! = specific_pe_rev_kernel(il)(backend, Molly.gpu_threads_specific(length(il)))
        kernel!(coords.val, whole_shadow(coords), atoms.val, whole_shadow(atoms),
                il.inters, grad_shadow(inter_list, :inters), velocities.val,
                whole_shadow(velocities), il.data, grad_shadow(inter_list, :data),
                boundary.val, step_n.val, inter_list_indices(il)...,
                Val(energy_units.val), dpe_vec; ndrange=length(il))
    end
    return ntuple(Returns(nothing), Val(9))
end

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Const{typeof(Molly.specific_forces_gpu!)},
                                      ::Type{RT}, fs_mat::Annotation, vir::Annotation,
                                      inter_list::Annotation, coords::Annotation{<:AbstractGPUArray},
                                      velocities::Annotation, atoms::Annotation,
                                      boundary::Annotation, needs_vir::Const,
                                      step_n::Annotation, force_units::Annotation,
                                      TH::Const) where RT
    func.val(fs_mat.val, vir.val, inter_list.val, coords.val, velocities.val, atoms.val,
             boundary.val, needs_vir.val, step_n.val, force_units.val, TH.val)
    primal = EnzymeRules.needs_primal(config) ? fs_mat.val : nothing
    shadow = (EnzymeRules.needs_shadow(config) && !(fs_mat isa Const)) ? fs_mat.dval : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       Nothing}(primal, shadow, nothing)
end

function EnzymeRules.reverse(config::RevConfig,
                             func::Const{typeof(Molly.specific_forces_gpu!)},
                             ::Type{RT}, tape, fs_mat::Annotation, vir::Annotation,
                             inter_list::Annotation, coords::Annotation{<:AbstractGPUArray},
                             velocities::Annotation, atoms::Annotation,
                             boundary::Annotation, needs_vir::Const{Val{NV}},
                             step_n::Annotation, force_units::Annotation,
                             TH::Const) where {RT, NV}
    NV && error("gradients through the virial are not implemented on the GPU, use " *
                "forces rather than forces_virial")
    dfs_mat = whole_shadow(fs_mat)
    il = inter_list.val
    if !isnothing(dfs_mat) && length(il) > 0
        D = length(eltype(coords.val))
        backend = get_backend(coords.val)
        kernel! = specific_force_rev_kernel(il)(backend, Molly.gpu_threads_specific(length(il)))
        kernel!(coords.val, whole_shadow(coords), atoms.val, whole_shadow(atoms),
                il.inters, grad_shadow(inter_list, :inters), velocities.val,
                whole_shadow(velocities), il.data, grad_shadow(inter_list, :data),
                boundary.val, step_n.val, inter_list_indices(il)...,
                Val(force_units.val), Val(D), dfs_mat; ndrange=length(il))
    end
    return ntuple(Returns(nothing), Val(11))
end

# Applying force units
# A straight copy of the force matrix into the force vector, so the adjoint is a copy the
# other way. Each thread owns one atom, so no atomics are needed. `fs` is overwritten by
# the primal, so its cotangent is consumed here rather than accumulated.

@kernel inbounds=true function apply_force_units_rev_kernel!(dfs, dfs_mat, ::Val{D}) where D
    atom_i = @index(Global, Linear)

    if atom_i <= length(dfs)
        df = dfs[atom_i]
        for dim in 1:D
            dfs_mat[dim, atom_i] = ustrip(df[dim])
        end
        dfs[atom_i] = zero(eltype(dfs))
    end
end

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Const{typeof(Molly.apply_force_units_gpu!)},
                                      ::Type{RT}, fs::Annotation{<:AbstractGPUArray}, fs_mat::Annotation,
                                      force_units::Annotation, D::Const, T::Const) where RT
    func.val(fs.val, fs_mat.val, force_units.val, D.val, T.val)
    primal = EnzymeRules.needs_primal(config) ? fs.val : nothing
    shadow = (EnzymeRules.needs_shadow(config) && !(fs isa Const)) ? fs.dval : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       Nothing}(primal, shadow, nothing)
end

function EnzymeRules.reverse(config::RevConfig,
                             func::Const{typeof(Molly.apply_force_units_gpu!)},
                             ::Type{RT}, tape, fs::Annotation{<:AbstractGPUArray}, fs_mat::Annotation,
                             force_units::Annotation, D::Const{Val{DD}},
                             T::Const) where {RT, DD}
    dfs, dfs_mat = whole_shadow(fs), whole_shadow(fs_mat)
    if !isnothing(dfs) && !isnothing(dfs_mat)
        backend = get_backend(dfs)
        kernel! = apply_force_units_rev_kernel!(backend, Molly.gpu_threads_copy(length(dfs)))
        kernel!(dfs, dfs_mat, Val(DD); ndrange=length(dfs))
    end
    return (nothing, nothing, nothing, nothing, nothing)
end

# Whole-function rules for gpu_potential_energy and gpu_forces!

# The reverse pass evaluates the interactions again, so it needs the coordinates (and
# atoms, and velocities where an interaction uses them) as they were during the forward
# pass. For a single energy or force evaluation they are still there, but inside a
# `simulate!` loop the integrator overwrites the coordinates between the two passes, so
# they are copied into the tape. The shadows are always the live arrays - only the primal
# values are cached. Cost is one coordinate-sized array per call, which is what reverse
# mode over a trajectory needs anyway.
struct GPUAdjointTape{C, A, V}
    coords::C
    atoms::A
    velocities::V
end

@inline function adjoint_tape(sys, active::Bool, uses_velocity::Bool)
    active || return GPUAdjointTape(nothing, nothing, nothing)
    return GPUAdjointTape(copy(sys.coords), copy(sys.atoms),
                          uses_velocity ? copy(sys.velocities) : nothing)
end

# Primal values to differentiate at: the cached copies when there are any
@inline tape_or_live(::Nothing, live) = live
@inline tape_or_live(cached, live) = cached
@inline function primal_arrays(sys, tape::GPUAdjointTape)
    return (tape_or_live(tape.coords, sys.coords), tape_or_live(tape.atoms, sys.atoms),
            tape_or_live(tape.velocities, sys.velocities))
end

# Shadow of the `inters` array of each specific interaction list, as a tuple
@inline specific_inter_shadows(::Const, sils) = map(Returns(nothing), sils)
@inline function specific_inter_shadows(x::Annotation, sils)
    return map((il, dil) -> il.inters === dil.inters ? nothing : dil.inters, sils, x.dval)
end

# Pairwise contribution. The interaction cotangent is left on the device in
# `sc.dinters_arr`; `read_dinters!` brings it back, and only when it is wanted.
function pairwise_adjoint!(rev_kernel, sys, prim, neighbors, step_n, pairwise_inters, sc,
                           dinters_arr, seed_arg, dcoords, datoms, dvelocities, extra...)
    pcoords, patoms, pvels = prim
    pis = values(pairwise_inters)
    (length(pis) == 0) && return false
    use_nl = map(use_neighbors, pis)
    if all(use_nl)
        nbs_in = neighbors
    elseif !any(use_nl)
        nbs_in = Molly.NoNeighborList(length(sys))
    else
        error("a mix of pairwise interactions with and without neighbour lists is not " *
              "supported by the GPU reverse rules yet")
    end
    length(nbs_in) == 0 && return false

    nbs = grad_nbs(nbs_in)
    backend = get_backend(pcoords)
    vels = (Molly.any_uses_velocity(pis) ? pvels : nothing)
    dvels = isnothing(vels) ? nothing : dvelocities
    kernel! = rev_kernel(backend, Molly.gpu_threads_pairwise(length(nbs)))
    kernel!(pcoords, dcoords, patoms, datoms, sc.inters_arr, dinters_arr, vels, dvels,
            sys.boundary, nbs, step_n, extra..., seed_arg; ndrange=length(nbs))
    return true
end

# The device to host copy synchronises, so it is the only place the reverse pass waits.
# `buffers.grad_scratch` is a `Ref{Any}`, so `sc` is only known at run time; the return is
# annotated because Enzyme checks the *inferred* type of what a reverse rule returns.
@inline function read_dinters!(sc::GradScratch, ran::Bool, inters::I) where I
    ran || return Enzyme.make_zero(inters)::I
    copyto!(sc.dinters_host, sc.dinters_arr)
    return reduce(add_cotangents, sc.dinters_host)::I
end

function specific_adjoint!(rev_kernel_for, sys, prim, step_n, specific_inter_lists,
                           dsil_inters, seed_arg, dcoords, datoms, dvelocities, extra...)
    pcoords, patoms, pvels = prim
    backend = get_backend(pcoords)
    for (i, inter_list) in enumerate(values(specific_inter_lists))
        length(inter_list) == 0 && continue
        kernel! = rev_kernel_for(inter_list)(backend,
                                        Molly.gpu_threads_specific(length(inter_list)))
        kernel!(pcoords, dcoords, patoms, datoms, inter_list.inters, dsil_inters[i],
                pvels, dvelocities, inter_list.data, nothing, sys.boundary,
                step_n, inter_list_indices(inter_list)..., extra..., seed_arg;
                ndrange=length(inter_list))
    end
    return nothing
end

@inline function check_no_general_inters(general_inters)
    length(general_inters) > 0 && error(
        "reverse-mode gradients through general interactions (such as implicit solvent) " *
        "are not implemented on the GPU yet; the pairwise and specific interactions are")
    return nothing
end

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Const{typeof(Molly.gpu_potential_energy)},
                                      ::Type{RT}, sys::Annotation, neighbors::Annotation,
                                      step_n::Annotation, buffers::Annotation,
                                      pairwise_inters::Annotation,
                                      specific_inter_lists::Annotation,
                                      general_inters::Annotation,
                                      n_threads::Annotation) where RT
    check_no_general_inters(general_inters.val)
    pe = func.val(sys.val, neighbors.val, step_n.val, buffers.val, pairwise_inters.val,
                  specific_inter_lists.val, general_inters.val, n_threads.val)
    tape = adjoint_tape(sys.val, !(sys isa Const) || !(pairwise_inters isa Const) ||
                                 !(specific_inter_lists isa Const),
                        Molly.any_uses_velocity(values(pairwise_inters.val)))
    primal = EnzymeRules.needs_primal(config) ? pe : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       typeof(tape)}(primal, nothing, tape)
end

function EnzymeRules.reverse(config::RevConfig,
                             func::Const{typeof(Molly.gpu_potential_energy)},
                             dret, tape, sys::Annotation, neighbors::Annotation,
                             step_n::Annotation, buffers::Annotation,
                             pairwise_inters::Annotation,
                             specific_inter_lists::Annotation,
                             general_inters::Annotation, n_threads::Annotation)
    dpe = ustrip(dret isa Type ? zero(Float64) : dret.val)
    dinters = nothing
    if !iszero(dpe)
        s = sys.val
        prim = primal_arrays(s, tape)
        sc = grad_scratch!(buffers.val, values(pairwise_inters.val), typeof(dpe),
                           array_type(s))
        fill!(sc.seed, dpe)
        dcoords = grad_shadow(sys, :coords)
        datoms = grad_shadow(sys, :atoms)
        dvels = grad_shadow(sys, :velocities)
        # The interaction cotangent is the expensive part of the kernel even with the
        # slots, so skip it entirely when there is nothing to accumulate into
        dia = (pairwise_inters isa Const ? nothing : sc.dinters_arr)
        ran = pairwise_adjoint!(pairwise_pe_rev_kernel!, s, prim, neighbors.val,
                                step_n.val, pairwise_inters.val, sc, dia, sc.seed, dcoords,
                                datoms, dvels, Val(s.energy_units))
        dsil = specific_inter_shadows(specific_inter_lists, values(specific_inter_lists.val))
        specific_adjoint!(specific_pe_rev_kernel, s, prim, step_n.val,
                          specific_inter_lists.val, dsil, sc.seed, dcoords, datoms, dvels,
                          Val(s.energy_units))
        if !(pairwise_inters isa Const)
            dinters = read_dinters!(sc, ran, values(pairwise_inters.val))
        end
    end
    dpis = if pairwise_inters isa Const
        nothing
    else
        value_cotangent!(pairwise_inters,
                         isnothing(dinters) ? Enzyme.make_zero(pairwise_inters.val) : dinters)
    end
    return (nothing, nothing, nothing, nothing, dpis, nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Const{typeof(Molly.gpu_forces!)},
                                      ::Type{RT}, fs::Annotation, sys::Annotation,
                                      neighbors::Annotation, step_n::Annotation,
                                      buffers::Annotation, needs_vir::Const,
                                      pairwise_inters::Annotation,
                                      specific_inter_lists::Annotation,
                                      general_inters::Annotation,
                                      n_threads::Annotation) where RT
    check_no_general_inters(general_inters.val)
    func.val(fs.val, sys.val, neighbors.val, step_n.val, buffers.val, needs_vir.val,
             pairwise_inters.val, specific_inter_lists.val, general_inters.val,
             n_threads.val)
    tape = adjoint_tape(sys.val, !(fs isa Const),
                        Molly.any_uses_velocity(values(pairwise_inters.val)))
    primal = EnzymeRules.needs_primal(config) ? (fs.val, buffers.val) : nothing
    shadow = (EnzymeRules.needs_shadow(config) && !(fs isa Const)) ?
             (fs.dval, buffers.val) : nothing
    return EnzymeRules.AugmentedReturn{EnzymeRules.primal_type(config, RT),
                                       EnzymeRules.shadow_type(config, RT),
                                       typeof(tape)}(primal, shadow, tape)
end

function EnzymeRules.reverse(config::RevConfig,
                             func::Const{typeof(Molly.gpu_forces!)},
                             ::Type{RT}, tape, fs::Annotation, sys::Annotation,
                             neighbors::Annotation, step_n::Annotation,
                             buffers::Annotation, needs_vir::Const{Val{NV}},
                             pairwise_inters::Annotation,
                             specific_inter_lists::Annotation,
                             general_inters::Annotation,
                             n_threads::Annotation) where {RT, NV}
    NV && error("gradients through the virial are not implemented on the GPU, use " *
                "forces rather than forces_virial")
    dfs = whole_shadow(fs)
    dinters = nothing
    if !isnothing(dfs)
        s = sys.val
        D = length(eltype(s.coords))
        backend = get_backend(s.coords)
        # d(fs) is the seed; move it into the fs_mat layout the kernels contract against.
        # The kernel writes rather than accumulates, so fs_mat needs no zeroing first.
        dfs_mat = buffers.val.fs_mat
        kernel_u! = apply_force_units_rev_kernel!(backend, Molly.gpu_threads_copy(length(dfs)))
        kernel_u!(dfs, dfs_mat, Val(D); ndrange=length(dfs))

        prim = primal_arrays(s, tape)
        sc = grad_scratch!(buffers.val, values(pairwise_inters.val), eltype(dfs_mat),
                           array_type(s))
        dcoords = grad_shadow(sys, :coords)
        datoms = grad_shadow(sys, :atoms)
        dvels = grad_shadow(sys, :velocities)
        dia = (pairwise_inters isa Const ? nothing : sc.dinters_arr) # As above
        ran = pairwise_adjoint!(pairwise_force_rev_kernel!, s, prim, neighbors.val,
                                step_n.val, pairwise_inters.val, sc, dia, dfs_mat, dcoords,
                                datoms, dvels, Val(s.force_units), Val(D))
        dsil = specific_inter_shadows(specific_inter_lists, values(specific_inter_lists.val))
        specific_adjoint!(specific_force_rev_kernel, s, prim, step_n.val,
                          specific_inter_lists.val, dsil, dfs_mat, dcoords, datoms, dvels,
                          Val(s.force_units), Val(D))
        if !(pairwise_inters isa Const)
            dinters = read_dinters!(sc, ran, values(pairwise_inters.val))
        end
    end
    dpis = if pairwise_inters isa Const
        nothing
    else
        value_cotangent!(pairwise_inters,
                         isnothing(dinters) ? Enzyme.make_zero(pairwise_inters.val) : dinters)
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing, dpis, nothing, nothing,
            nothing)
end

end
