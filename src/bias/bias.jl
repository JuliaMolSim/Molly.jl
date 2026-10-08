# Bias potentials

export
    LinearBias,
    bias_gradient,
    SquareBias,
    FlatBottomSquareBias,
    PeriodicFlatBottomBias,
    BiasPotential


@doc raw"""
    LinearBias(k, cv_target)

A linear bias on a collective variable (CV) towards a target value.

The potential energy is defined as
```math
V(\boldsymbol{s}) = k |\boldsymbol{s} - \boldsymbol{s}_t|
```
where $s$ and $s_t$ are the system and target CV values respectively.

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `cv_target`: The target value of the collective variable.
"""
struct LinearBias{K, C}
    k::K
    cv_target::C
end

function potential_energy(lb::LinearBias, cv_sim; kwargs...)
    return lb.k * abs(cv_sim - lb.cv_target)
end

"""
    bias_gradient(bias::BiasType, cv_sim::Real)

Calculate the gradient of a bias potential with respect to the value of a
collective variable.

# Arguments
- `b::BiasType`: A struct that defines the type of bias to be used.
- `cv_sim::Real`: The value of a measured collective variable given
    the coordinates of a simulation.
"""
function bias_gradient(lb::LinearBias, cv_sim)
    d = cv_sim - lb.cv_target
    iszero(d) && return zero(lb.k)
    return lb.k * d / abs(d)
end

@doc raw"""
    SquareBias(k, cv_target)

A harmonic bias on a collective variable (CV) towards a target value.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (\boldsymbol{s} - \boldsymbol{s}_t)^2
```
where $s$ and $s_t$ are the system and target CV values respectively.

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `cv_target`: The target value of the collective variable.
"""
struct SquareBias{K, C}
    k::K
    cv_target::C
end

function potential_energy(sb::SquareBias, cv_sim; kwargs...)
    return (sb.k / 2) * (cv_sim - sb.cv_target)^2
end

function bias_gradient(sb::SquareBias, cv_sim)
    return sb.k * (cv_sim - sb.cv_target)
end


 function validate_flat_bottom_width(r_fb, label::AbstractString)
    if !isfinite(ustrip(r_fb)) || r_fb < zero(r_fb)
        throw(ArgumentError("$label flat-bottom width must be finite and non-negative, got $r_fb"))
    end
    return r_fb
end

@doc raw"""
    FlatBottomSquareBias(k, r_fb, cv_target)

A flat-bottomed square (harmonic) bias on a collective variable (CV) towards a target value.

The bias is zero when the value of the collective variable does not deviate
from `cv_target` by more than `r_fb`, and is square (harmonic) outside this range.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (|\boldsymbol{s} - \boldsymbol{s}_t| - r_{fb})^2 H
```
where $s$ and $s_t$ are the system and target CV values respectively, and
```math
H = \left\{ \begin{array}{cl}
0 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| < r_{fb} \\
1 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| \geq r_{fb} \\
\end{array} \right.
```

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `r_fb`: Width of flat-bottom potential well. Inside this region the
    bias potential is always 0.
- `cv_target`: The target value of the collective variable.
"""
struct FlatBottomSquareBias{K, R, C}
    k::K
    r_fb::R
    cv_target::C

    function FlatBottomSquareBias(k::K, r_fb::R, cv_target::C) where {K, R, C}
        validate_flat_bottom_width(r_fb, "FlatBottomSquareBias")
        return new{K, R, C}(k, r_fb, cv_target)
    end
end

function potential_energy(fb::FlatBottomSquareBias, cv_sim; kwargs...)
    d_abs = abs(cv_sim - fb.cv_target)
    H = (d_abs < fb.r_fb ? 0 : 1)
    return (fb.k / 2) * (d_abs - fb.r_fb)^2 * H
end

function bias_gradient(fb::FlatBottomSquareBias, cv_sim)
    d = cv_sim - fb.cv_target
    d_abs = abs(d)
    d_abs <= fb.r_fb && return zero(fb.k * fb.r_fb)
    return fb.k * (d_abs - fb.r_fb) * d / d_abs
end

@doc raw"""
    PeriodicFlatBottomBias(k, r_fb, cv_target)

A flat-bottomed square (harmonic) bias on a collective variable (CV) towards a target value.

The bias is zero when the value of the collective variable does not deviate
from `cv_target` by more than `r_fb`, and is square (harmonic) outside this range.

This variant handles periodicity in the CV wrapping around the (-π, π) range.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (|\boldsymbol{s} - \boldsymbol{s}_t| - r_{fb})^2 H
```
where $s$ and $s_t$ are the system and target CV values respectively, and
```math
H = \left\{ \begin{array}{cl}
0 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| < r_{fb} \\
1 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| \geq r_{fb} \\
\end{array} \right.
```

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `r_fb`: Width of flat-bottom potential well. Inside this region the
    bias potential is always 0.
- `cv_target`: The target value of the collective variable.
"""
struct PeriodicFlatBottomBias{K, R, T}
    k::K
    r_fb::R
    cv_target::T

    function PeriodicFlatBottomBias(k::K, r_fb::R, cv_target::T) where {K, R, T}
        validate_flat_bottom_width(r_fb, "PeriodicFlatBottomBias")
        return new{K, R, T}(k, r_fb, cv_target)
    end
end

 function periodic_flat_bottom_displacement(cv_sim, cv_target)
    d = cv_sim - cv_target
    FT = typeof(float(ustrip(d)))
    twopi = FT(2π) * oneunit(d)
    half_period = twopi / FT(2)
    return mod(d + half_period, twopi) - half_period
end

function potential_energy(pb::PeriodicFlatBottomBias, cv_sim; kwargs...)
    FT = typeof(float(ustrip(cv_sim - pb.cv_target)))
    d_wrapped = periodic_flat_bottom_displacement(cv_sim, pb.cv_target)
    
    dist = abs(d_wrapped)
    
    if dist <= pb.r_fb
        return zero(pb.k * pb.r_fb^2)
    else
        disp = dist - pb.r_fb
        return FT(0.5) * pb.k * disp^2
    end
end

function bias_gradient(pb::PeriodicFlatBottomBias, cv_sim)
    d_wrapped = periodic_flat_bottom_displacement(cv_sim, pb.cv_target)
    
    dist = abs(d_wrapped)
    
    if dist <= pb.r_fb
        return zero(pb.k * pb.r_fb)
    else
        disp = dist - pb.r_fb
        return pb.k * disp * sign(d_wrapped)
    end
end

const BIAS_POTENTIAL_ID_COUNTER = Threads.Atomic{UInt64}(0)
next_bias_potential_id() = Threads.atomic_add!(BIAS_POTENTIAL_ID_COUNTER, UInt64(1))

"""
    BiasPotential(cv_type, bias_type)

A potential to bias a simulation along a collective variable (CV), implemented
as an AtomsCalculators.jl calculator.

The `cv_type` could for example be [`CalcDist`](@ref) and the `bias_type`
could be [`LinearBias`](@ref).

Forces resulting from the bias potential are evaluated in two steps, specfically by
(1) calculating the gradient of the bias potential with respect to the value of the CV, and
(2) calculating the gradient of the CV with respect to the atomic coordinates.

Gradients can be calculated with either automatic differentiation or explicitly defined
gradient functions.
Enzyme should be imported in the first case.

Virial contributions must be explicitly defined.

When the `System` is GPU-resident, built-in CV types other than `CalcRMSD` run fully GPU-resident,
with no host transfer of coordinates, atoms or forces, including `cv_type.correction = :pbc` (the
default for the built-in CV types), which unwraps bonded molecules across the periodic boundary
using a GPU-native spanning-forest traversal. `CalcRMSD` performs a host-side Kabsch alignment
step every call. Custom (non-built-in) CV types always round-trip coordinates/atoms/gradient to
and from the host, since arbitrary user code isn't guaranteed GPU-safe.

Each `BiasPotential` gets a unique internal id when it is constructed, and no two biases
in a [`System`](@ref) may share an id (an `ArgumentError` is thrown when the `System` is built).
To copy a bias with a different CV, build a new one with `BiasPotential(cv_type, bias_type)`
instead of copying its fields.
"""
struct BiasPotential{C, B}
    cv_type::C
    bias_type::B
    uses_persistent_buffers::Bool   # set at construction, see uses_builtin_cv_gradient
    id::UInt64                      # set at construction, distinguishes structurally-equal biases
end

function BiasPotential(cv_type::C, bias_type::B) where {C, B}
    return BiasPotential{C, B}(cv_type, bias_type, uses_builtin_cv_gradient(cv_type),
                               next_bias_potential_id())
end

# id is the scratch key only; equality and hashing ignore it (thermo.jl's master-system intersect).
Base.:(==)(a::BiasPotential, b::BiasPotential) = a.cv_type == b.cv_type && a.bias_type == b.bias_type
Base.isequal(a::BiasPotential, b::BiasPotential) = isequal(a.cv_type, b.cv_type) && isequal(a.bias_type, b.bias_type)
Base.hash(b::BiasPotential, h::UInt) = hash(b.cv_type, hash(b.bias_type, hash(:BiasPotential, h)))

# Scratch buffers are keyed by id, so no two biases in a System may share one.
function check_bias_ids(general_inters)
    ids = [inter.id for inter in values(general_inters) if inter isa BiasPotential]
    length(ids) == length(Set(ids)) || throw(ArgumentError("BiasPotentials in general_inters " *
        "share an id and would share scratch buffers, build each one with " *
        "BiasPotential(cv_type, bias_type)"))
    return nothing
end

# The CV kernels run without bounds checks, so atom indices must be in range before they get there.
function check_bias_atom_inds(general_inters, n_atoms)
    for inter in values(general_inters)
        inter isa BiasPotential || continue
        for inds in atom_index_lists(inter.cv_type), i in inds
            1 <= i <= n_atoms || throw(ArgumentError("atom index $i of the " *
                "$(nameof(typeof(inter.cv_type))) in a BiasPotential is outside the $n_atoms " *
                "atoms of the system"))
        end
    end
    return nothing
end

# Per-BiasPotential lazy scratch (grad/d_buf/fs_svec on both backends; dist_scratch GPU-only
# fused-kernel state; d_bias_buf/bad_step, captured-path-only). One per bias in
# buffers.bias_scratch (force.jl), keyed on bias.id.
mutable struct BiasScratch
    grad::Any
    d_buf::Any
    fs_svec::Any
    dist_scratch::Any
    d_bias_buf::Any                 # lazy bias_gradient output, cuda_graph_capturing path only
    bad_step::Any                   # step of first deferred finite-check failure, or 0
end

BiasScratch() = BiasScratch(nothing, nothing, nothing, nothing, nothing, nothing)

# Locate bias's BiasScratch by its id, not its position in general_inters -- a position can
# differ between calls (e.g. MTS integrators pass a different per-level subset each time), so
# scratch must be looked up by the bias itself. Falls back to a fresh throwaway BiasScratch when
# buffers is nothing (a bare, bufferless call) -- no reuse across calls in that case, matching
# pre-persistent-buffer behaviour.
bias_scratch(buffers, bias::BiasPotential) =
    get!(BiasScratch, buffers.bias_scratch, bias.id)
bias_scratch(::Nothing, ::BiasPotential) = BiasScratch()

bias_all_finite(values::AbstractArray) = all(bias_all_finite, values)
bias_all_finite(value) = isfinite(ustrip(value))

 function bias_max_abs_ustrip(values::AbstractArray)
    isempty(values) && return 0.0
    return mapreduce(bias_max_abs_ustrip, max, values)
end

bias_max_abs_ustrip(value) = abs(ustrip(value))

 function check_bias_finite(value, label::AbstractString, bias::BiasPotential;
                            cv_sim=nothing, max_abs_component_fn=nothing)
    bias_all_finite(value) && return value
    msg = "BiasPotential with CV $(typeof(bias.cv_type)) and bias " *
          "$(typeof(bias.bias_type)) produced non-finite $(label)"
    if !isnothing(cv_sim)
        msg *= ", cv_sim=$(cv_sim)"
    end
    if !isnothing(max_abs_component_fn)
        msg *= ", max_abs_component=$(max_abs_component_fn())"
    end
    error(msg)
end

# Coordinates a BiasPotential should use, unwrapped for correction==:pbc. With buffers, routes
# through the shared once-per-forces!-call unwrap cache (ensure_unwrapped_coords!, force.jl)
# instead of recomputing per bias.
function bias_coords(sys, cv_type, buffers=nothing, step_n=nothing)
    cv_type.correction != :pbc && return sys.coords
    if !isnothing(buffers) && !isnothing(step_n) && hasproperty(buffers, :unwrapped_coords)
        return ensure_unwrapped_coords!(buffers, sys)
    end
    return unwrap_molecules(sys)
end

# Partition a heterogeneous tuple of interactions into its BiasPotential entries and everything
# else, preserving each group's relative order. Recurses rather than using filter, which isn't
# type-stable on a mixed-type Tuple.
@inline split_biases(::Tuple{}) = (), ()
@inline function split_biases(t::Tuple)
    rest_biases, rest_others = split_biases(Base.tail(t))
    x = first(t)
    return x isa BiasPotential ? ((x, rest_biases...), rest_others) : (rest_biases, (x, rest_others...))
end

# Lazily allocates scratch.grad/scratch.d_buf. Only reached when bias.uses_persistent_buffers
# is true (a CV type with a real cv_gradient!/calculate_cv! -- see uses_builtin_cv_gradient in
# cv.jl).
function ensure_bias_buffers!(scratch::BiasScratch, cv_type, coords)
    if scratch.grad === nothing
        scratch.grad, scratch.d_buf = zero_cv_gradient_buffers(cv_type, coords)
    end
    return nothing
end

# Uploads a host index Vector to the device once, avoiding the per-call re-upload that
# `@view coords[cv.atom_inds_1]` would otherwise do on every call (see MinMaxScratch's
# docstring, cv.jl).
function upload_idx(coords, inds::Vector{Int})
    idx_dev = similar(coords, Int, length(inds))
    copyto!(idx_dev, inds)
    return idx_dev
end

# For each atom in inds_a, its position in inds_b (0 if absent). Used by CMDistScratch so
# cmdist_grad_write_kernel! can detect a shared atom between the two groups without a race.
function partner_indices(inds_a::Vector{Int}, inds_b::Vector{Int})
    pos_in_b = Dict(atom => k for (k, atom) in enumerate(inds_b))
    return [get(pos_in_b, atom, 0) for atom in inds_a]
end

function bias_dist_scratch_types(coords, atoms)
    CT = eltype(coords)
    MT = Base.promote_op(mass, eltype(atoms))
    WT = typeof(zero(CT) * zero(MT))
    IT = typeof(sum_abs2(zero(CT)) * zero(MT))
    return CT, MT, WT, IT
end

# Lazily allocates CV-type-specific fused-kernel scratch. GPU-only device-sized buffers.
function ensure_bias_dist_scratch!(scratch::BiasScratch, cv, coords, atoms)
    if cv isa CalcDist{<:Union{CalcMinDist, CalcMaxDist}} && scratch.dist_scratch === nothing &&
       is_gpu_resident(coords)
        idx1_dev, idx2_dev = upload_idx(coords, cv.atom_inds_1), upload_idx(coords, cv.atom_inds_2)
        na, nb = length(cv.atom_inds_1), length(cv.atom_inds_2)
        # T caps kernel worker count at MINDIST_TILE_CAP regardless of na*nb -- see
        # MinMaxScratch's docstring (cv.jl).
        T = min(na * nb, MINDIST_TILE_CAP)
        scratch.dist_scratch = MinMaxScratch(
            idx1_dev,
            idx2_dev,
            similar(coords, eltype(eltype(coords)), T),
            similar(coords, Int, T),
            similar(coords, Int, T),
            similar(coords, T),
            similar(coords, 1),
        )
    elseif cv isa CalcDist{CalcCMDist} && scratch.dist_scratch === nothing && is_gpu_resident(coords)
        na, nb = length(cv.atom_inds_1), length(cv.atom_inds_2)
        T1, T2 = min(na, TILE_CAP_DEFAULT), min(nb, TILE_CAP_DEFAULT)
        CT, MT, WT, _ = bias_dist_scratch_types(coords, atoms)
        DT = typeof(zero(CT) / oneunit(eltype(CT))) # unit-stripped direction vector
        scratch.dist_scratch = CMDistScratch(
            upload_idx(coords, cv.atom_inds_1), upload_idx(coords, cv.atom_inds_2),
            upload_idx(coords, partner_indices(cv.atom_inds_1, cv.atom_inds_2)),
            upload_idx(coords, partner_indices(cv.atom_inds_2, cv.atom_inds_1)),
            similar(coords, MT, T1), similar(coords, WT, T1),
            similar(coords, MT, T2), similar(coords, WT, T2),
            similar(coords, DT, 1),
            similar(coords, MT, 1), similar(coords, MT, 1),
        )
    elseif cv isa CalcRg && scratch.dist_scratch === nothing && is_gpu_resident(coords)
        inds = iszero(length(cv.atom_inds)) ? collect(1:length(coords)) : cv.atom_inds
        n = length(inds)
        T = min(n, TILE_CAP_DEFAULT)
        CT, MT, WT, IT = bias_dist_scratch_types(coords, atoms)
        scratch.dist_scratch = RgScratch(
            upload_idx(coords, inds),
            similar(coords, MT, T), similar(coords, WT, T),
            similar(coords, IT, T),
            similar(coords, CT, 1), similar(coords, MT, 1),
        )
    elseif cv isa CalcRMSD && scratch.dist_scratch === nothing && is_gpu_resident(coords)
        inds = iszero(length(cv.atom_inds)) ? collect(1:length(coords)) : cv.atom_inds
        ref_inds = iszero(length(cv.ref_atom_inds)) ? collect(1:length(cv.ref_coords)) : cv.ref_atom_inds
        ref_coords_host = cv.ref_coords[ref_inds]
        AT = array_type(coords)
        ref_coords_used = ref_coords_host isa AT ? ref_coords_host : to_device(ref_coords_host, AT)
        # cv.ref_coords never changes after construction, so its Kabsch-centered form is computed
        # once here rather than on every cv_gradient!/calculate_cv! call (RmsdScratch docstring, cv.jl).
        n = length(inds)
        T = min(n, TILE_CAP_DEFAULT)
        CT = eltype(coords)
        IT = typeof(sum_abs2(zero(CT))) # unweighted nm^2, unlike bias_dist_scratch_types' mass-weighted IT
        scratch.dist_scratch = RmsdScratch(upload_idx(coords, inds), similar(coords, length(inds)),
                                           ref_coords_used, kabsch_centered(ref_coords_used),
                                           similar(coords, IT, T))
    end
    return nothing
end

function ensure_bias_gradient_buffer!(scratch::BiasScratch, bias_type)
    if isnothing(scratch.d_bias_buf)
        # bias_gradient's output units differ from d_buf's own, so derive the eltype from a
        # sample call rather than `similar(scratch.d_buf, 1)`.
        sample = bias_gradient(bias_type, oneunit(eltype(scratch.d_buf)))
        scratch.d_bias_buf = similar(scratch.d_buf, typeof(sample), 1)
    end
    return nothing
end

function ensure_bias_finite_buffer!(scratch::BiasScratch)
    if isnothing(scratch.bad_step)
        scratch.bad_step = similar(scratch.d_buf, Int, 1)
        scratch.bad_step .= 0
    end
    return nothing
end

# Concurrent writers only ever write the same step_n into bad_step[1], so no atomic is needed.
# Allocation-free by design: an earlier `mapreduce` version allocated per call, which inside a
# captured CUDA graph degraded @captured's cuGraphExecUpdate to a slow path scaling badly with
# n_bias (confirmed live).
@kernel inbounds=true function bias_finite_check_kernel!(bad_step, @Const(value), step_n::Int)
    idx = @index(Global, Linear)
    if idx <= length(value) && !bias_all_finite(value[idx]) && bad_step[1] == 0
        bad_step[1] = step_n
    end
end

# Device-resident, allocation-free stand-in for check_bias_finite on array-valued checks
# (d_buf/grad/fs_svec): records the first non-finite step into scratch.bad_step with no host
# sync, so it is safe to call every step, including inside a captured CUDA graph. The actual
# error, if any, is only raised later by check_bias_finite_periodic's cheap 1-element readback.
function check_bias_finite_deferred!(value::AbstractArray, scratch::BiasScratch, step_n::Integer)
    backend = get_backend(value)
    n = length(value)
    kernel! = bias_finite_check_kernel!(backend, min(n, 256))
    kernel!(scratch.bad_step, value, Int(step_n); ndrange=n)
    return nothing
end

# --- Batched captured-path tail across every attached BiasPotential at once ------------------
#
# cv_gradient! stays per-bias (different CVs need different reduction kernels -- see bias_cv_step!
# below). Everything after it (bias_gradient dispatch, finite check, force apply) is generic
# per-element work, so a tuple-recursion-unrolled kernel can dispatch each bias's own
# bias_type/array at compile time and batch all biases into 3 kernel launches per step instead of
# 3*n_bias -- see force.jl's cuda_graph_capturing branch for the call site.
#
# Recursion over the NTuples keeps indexing type-stable/
# GPU-codegen-safe when the tuple element types differ per bias (d_buf/d_bias_buf/bad_step have a
# different Unitful eltype per bias). The helpers below assume at least one bias, so the base case is the 1-tuple.
@inline function _bias_apply_recurse(grads::Tuple{Any}, fs_svecs::Tuple{Any}, d_bias_bufs::Tuple{Any}, i)
    v = d_bias_bufs[1][1] * grads[1][i]
    fs_svecs[1][i] = v
    return v
end
@inline function _bias_apply_recurse(grads::Tuple, fs_svecs::Tuple, d_bias_bufs::Tuple, i)
    v = d_bias_bufs[1][1] * grads[1][i]
    fs_svecs[1][i] = v
    return v + _bias_apply_recurse(Base.tail(grads), Base.tail(fs_svecs), Base.tail(d_bias_bufs), i)
end

@kernel inbounds=true function bias_batched_apply_kernel!(fs, grads::NTuple{N}, fs_svecs::NTuple{N},
                                                            d_bias_bufs::NTuple{N}) where N
    i = @index(Global, Linear)
    fs[i] -= _bias_apply_recurse(grads, fs_svecs, d_bias_bufs, i)
end

@inline function _bias_check_recurse(values::Tuple{Any}, bad_steps::Tuple{Any}, i, step_n)
    v, bs = values[1], bad_steps[1]
    if i <= length(v) && !bias_all_finite(v[i]) && bs[1] == 0
        bs[1] = step_n
    end
    return nothing
end
@inline function _bias_check_recurse(values::Tuple, bad_steps::Tuple, i, step_n)
    v, bs = values[1], bad_steps[1]
    if i <= length(v) && !bias_all_finite(v[i]) && bs[1] == 0
        bs[1] = step_n
    end
    return _bias_check_recurse(Base.tail(values), Base.tail(bad_steps), i, step_n)
end

# One launch checks every attached bias's d_buf/grad/fs_svec at once; each shorter than
# ndrange=max(n_atoms) simply reads out via its own bounds check (as bias_finite_check_kernel!
# does per-array). Each bias's bad_step is written independently, needing no atomic (see
# check_bias_finite_deferred!).
@kernel inbounds=true function bias_batched_finite_kernel!(values::NTuple{N}, bad_steps::NTuple{N},
                                                             step_n::Int) where N
    i = @index(Global, Linear)
    _bias_check_recurse(values, bad_steps, i, step_n)
end

# Per-bias half of the captured-path tail: computes scratch.grad/d_buf, the one CV-type-specific
# step that can't batch across biases. Pair with bias_batched_tail!, which does the batched
# remainder for every attached bias at once. do_check gates the finite-check kernels here and in
# bias_batched_tail! together, needed by the two-graph capture-once design
# (captured_forces_once!, MollyCUDAExt.jl): the no-check graph must carry no baked-in step_n,
# while the with-check graph is captured once with a sentinel step_n and only ever replayed.
function bias_cv_step!(bias::BiasPotential, scratch::BiasScratch, sys, coords, fs, step_n,
                       do_check::Bool=true)
    ensure_bias_buffers!(scratch, bias.cv_type, coords)
    ensure_bias_dist_scratch!(scratch, bias.cv_type, coords, sys.atoms)
    ensure_bias_gradient_buffer!(scratch, bias.bias_type)
    ensure_bias_finite_buffer!(scratch)
    cv_gradient!(scratch.grad, scratch.d_buf, bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities;
                scratch=scratch.dist_scratch)
    if do_check
        check_bias_finite_deferred!(scratch.d_buf, scratch, step_n)
        check_bias_finite_deferred!(scratch.grad, scratch, step_n)
    end
    isnothing(scratch.fs_svec) && (scratch.fs_svec = similar(fs))
    return nothing
end

@inline function _bias_gradient_recurse(d_bias_bufs::Tuple{Any}, d_bufs::Tuple{Any}, bias_types::Tuple{Any})
    d_bias_bufs[1][1] = bias_gradient(bias_types[1], d_bufs[1][1])
    return nothing
end
@inline function _bias_gradient_recurse(d_bias_bufs::Tuple, d_bufs::Tuple, bias_types::Tuple)
    d_bias_bufs[1][1] = bias_gradient(bias_types[1], d_bufs[1][1])
    return _bias_gradient_recurse(Base.tail(d_bias_bufs), Base.tail(d_bufs), Base.tail(bias_types))
end

# Batches bias_gradient across every bias in one launch; bias_types is a plain NTuple of
# bias_type values passed through as kernel arguments (like other bitstype-ish scalar args).
@kernel inbounds=true function bias_batched_gradient_kernel!(d_bias_bufs::NTuple{N}, @Const(d_bufs::NTuple{N}),
                                                               bias_types::NTuple{N}) where N
    idx = @index(Global, Linear)
    idx == 1 && _bias_gradient_recurse(d_bias_bufs, d_bufs, bias_types)
end

# Batched remainder of the captured-path tail, run once every step after every bias in biases
# has already run bias_cv_step!: computes each bias's bias_gradient (d_bias_buf), applies all
# their force contributions to fs, and finite-checks the array the tail itself produces
# (fs_svec). 3 kernel launches total instead of 3 per bias. scratches pairs 1:1 with biases.
function bias_batched_tail!(fs, biases::Tuple, scratches::Tuple, step_n, do_check::Bool=true)
    isempty(biases) && return nothing
    grads       = map(s -> s.grad,       scratches)
    fs_svecs    = map(s -> s.fs_svec,    scratches)
    d_bias_bufs = map(s -> s.d_bias_buf, scratches)
    bias_types  = map(b -> b.bias_type,  biases)
    backend = get_backend(fs)
    n_atoms = length(first(grads))
    grad_kernel! = bias_batched_gradient_kernel!(backend, 1)
    grad_kernel!(d_bias_bufs, map(s -> s.d_buf, scratches), bias_types; ndrange=1)
    apply_kernel! = bias_batched_apply_kernel!(backend, min(n_atoms, 256))
    apply_kernel!(fs, grads, fs_svecs, d_bias_bufs; ndrange=n_atoms)
    if do_check
        bad_steps = map(s -> s.bad_step, scratches)
        check_kernel! = bias_batched_finite_kernel!(backend, min(n_atoms, 256))
        check_kernel!(fs_svecs, bad_steps, Int(step_n); ndrange=n_atoms)
    end
    return nothing
end

# Out-of-graph readback of a deferred finite check (see check_bias_finite_deferred!); call every
# finite_check_every steps from simulate!'s step loop. A no-op until a deferred check has
# actually run.
function check_bias_finite_periodic(bias::BiasPotential, scratch::BiasScratch)
    isnothing(scratch.bad_step) && return nothing
    bad_step = only(from_device(scratch.bad_step))
    if bad_step != 0
        error("BiasPotential with CV $(typeof(bias.cv_type)) and bias $(typeof(bias.bias_type)) " *
              "first produced a non-finite value at step $bad_step.")
    end
    return nothing
end

# Same as check_bias_finite_periodic, but for every attached BiasPotential at once: one host
# sync total instead of one per bias, since a from_device round trip costs tens of microseconds
# of driver/sync overhead regardless of payload size (confirmed via profiling). Skips any bias
# whose bad_step hasn't been allocated yet.
function check_bias_finite_periodic_batched!(general_inters, buffers)
    live = Tuple((b, bias_scratch(buffers, b)) for b in general_inters
                 if b isa BiasPotential && !isnothing(bias_scratch(buffers, b).bad_step))
    isempty(live) && return nothing
    bad_steps_h = from_device(reduce(vcat, map(bs -> bs[2].bad_step, live)))
    for ((bias, _), bad_step) in zip(live, bad_steps_h)
        if bad_step != 0
            error("BiasPotential with CV $(typeof(bias.cv_type)) and bias $(typeof(bias.bias_type)) " *
                  "first produced a non-finite value at step $bad_step.")
        end
    end
    return nothing
end

function AtomsCalculators.potential_energy(
    sys, bias::BiasPotential;
    buffers = nothing,
    kwargs...
)
    coords = bias_coords(sys, bias.cv_type)
    scratch = bias_scratch(buffers, bias)

    if bias.uses_persistent_buffers
        ensure_bias_buffers!(scratch, bias.cv_type, coords)
        ensure_bias_dist_scratch!(scratch, bias.cv_type, coords, sys.atoms)
        calculate_cv_buffered!(bias.cv_type, coords, sys.atoms, sys.boundary, scratch.d_buf, sys.velocities;
                               scratch=scratch.dist_scratch, kwargs...)
        cv_sim = only(from_device(scratch.d_buf))
    else
        # Custom CV types aren't guaranteed GPU-safe, so run them on the CPU.
        cv_sim = calculate_cv(bias.cv_type, from_device(coords), from_device(sys.atoms),
                              sys.boundary, from_device(sys.velocities); kwargs...)
    end
    check_bias_finite(cv_sim, "collective variable", bias)

    pe = potential_energy(bias.bias_type, cv_sim; kwargs...)
    return check_bias_finite(pe, "potential energy", bias; cv_sim=cv_sim)
end

function AtomsCalculators.forces!(
    fs, sys, bias::BiasPotential;
    needs_vir::Bool = false,
    buffers = nothing, # Dummy to be able to have explicit kwarg. In reality a buffer will always be passed
    step_n = nothing,
    cuda_graph_capturing::Bool = false,
    defer_finite_check::Bool = false,
    kwargs...
)
    coords = bias_coords(sys, bias.cv_type, buffers, step_n)
    scratch = bias_scratch(buffers, bias)

    # cuda_graph_capturing=true: zero-host-sync path for use inside a captured CUDA graph.
    # check_bias_finite itself are done outside forces! for graph usage
    if cuda_graph_capturing
        # needs_vir is excluded from the captured region entirely, so no calculate_virial! call
        # here. see force.jl's cuda_graph_capturing branch for the n_bias>1 batching this enables.
        bias_cv_step!(bias, scratch, sys, coords, fs, step_n)
        bias_batched_tail!(fs, (bias,), (scratch,), step_n)
        return fs
    end

    # Gradient of CV with respect to coordinates
    if bias.uses_persistent_buffers
        ensure_bias_buffers!(scratch, bias.cv_type, coords)
        ensure_bias_dist_scratch!(scratch, bias.cv_type, coords, sys.atoms)
        # Warm up d_bias_buf too, so a later cuda_graph_capturing call isn't the one allocating it.
        is_gpu_resident(coords) && ensure_bias_gradient_buffer!(scratch, bias.bias_type)
        cv_gradient!(scratch.grad, scratch.d_buf, bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities;
                    scratch=scratch.dist_scratch)
        d_coords, cv_sim = scratch.grad, only(from_device(scratch.d_buf))
    else
        # Custom CV types aren't guaranteed GPU-safe, so run them on the CPU.
        d_coords, cv_sim = cv_gradient(
            bias.cv_type,
            from_device(coords),
            from_device(sys.atoms),
            sys.boundary,
            from_device(sys.velocities),
        )
    end
    check_bias_finite(cv_sim, "collective variable", bias)

    # Gradient of bias function with respect to CV
    d_bias = bias_gradient(bias.bias_type, cv_sim)
    check_bias_finite(d_bias, "bias gradient", bias; cv_sim=cv_sim)

    if bias.uses_persistent_buffers
        if scratch.fs_svec === nothing
            scratch.fs_svec = d_bias .* d_coords
        else
            scratch.fs_svec .= d_bias .* d_coords
        end
        fs_svec = scratch.fs_svec
    else
        fs_svec = d_bias .* d_coords
    end
    # d_bias is finite, so a non-finite d_coords always gives a non-finite fs_svec
    if defer_finite_check
        ensure_bias_finite_buffer!(scratch)
        check_bias_finite_deferred!(fs_svec, scratch, step_n)
    elseif !bias_all_finite(fs_svec)
        check_bias_finite(d_coords, "CV gradient", bias; cv_sim=cv_sim)
        check_bias_finite(
            fs_svec,
            "bias force",
            bias;
            cv_sim=cv_sim,
            max_abs_component_fn = () -> bias_max_abs_ustrip(fs_svec),
        )
    end

    if needs_vir && bias.cv_type.has_virial
        if bias.uses_persistent_buffers
            calculate_virial!(buffers.virial, bias.cv_type, coords, -fs_svec, sys.atoms, sys.boundary;
                             scratch=scratch.dist_scratch)
        else
            calculate_virial!(buffers.virial, bias.cv_type, from_device(coords), -fs_svec,
                              from_device(sys.atoms), sys.boundary)
        end
    end

    fs .-= bias.uses_persistent_buffers ? fs_svec : to_device(fs_svec, typeof(fs))
    return fs
end
