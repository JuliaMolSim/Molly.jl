# Bias potentials

export
    LinearBias,
    bias_gradient,
    uses_device_bias_gradient,
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
    d_abs <= fb.r_fb && return zero(fb.k * d)
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
        return zero(pb.k * d_wrapped)
    else
        disp = dist - pb.r_fb
        return pb.k * disp * sign(d_wrapped)
    end
end

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

When the `System` is GPU-resident, [`CalcDist`](@ref), [`CalcRg`](@ref), [`CalcTorsion`](@ref)
and [`CalcAngle`](@ref) run on the GPU, also with `cv_type.correction = :pbc`, which unwraps
bonded molecules on the GPU. On steps without a virial they need no transfer to the host.
[`CalcRMSD`](@ref) does the same apart from a host-side Kabsch alignment every step.
Other CV types, including custom ones, copy the coordinates and atoms to the host and the
gradient back, since arbitrary user code isn't guaranteed to be GPU-safe.
With the GPU CV types, `bias_gradient` runs on the GPU for bias types where
[`uses_device_bias_gradient`](@ref) is `true`: the built-in ones with plain-data parameters, and
custom ones that opt in. The others run it on the host.
Built-in bias types need plain-data parameters such as Float32 or Float64 on GPU systems;
building a GPU `System` throws otherwise.
For the GPU CV types, non-finite bias forces are detected on the GPU and reported every 20 steps,
at the end of [`simulate!`](@ref) and when one-off calls such as [`forces`](@ref) return, with
the step at which they first appeared.
"""
struct BiasPotential{C, B}
    cv_type::C
    bias_type::B
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

const BuiltinBias = Union{LinearBias, SquareBias, FlatBottomSquareBias, PeriodicFlatBottomBias}

# Built-in bias types with parameters that aren't plain data (e.g. BigFloat) can't be used with
# GPU arrays, so their forces would fail on a GPU system. Float64 parameters lose precision on
# backends without Float64.
function check_bias_gpu_types(general_inters, coords)
    is_on_gpu(coords) || return nothing
    for inter in values(general_inters)
        (inter isa BiasPotential && inter.bias_type isa BuiltinBias) || continue
        b = inter.bias_type
        if !isbits(b)
            throw(ArgumentError("$(typeof(b)) has parameters that can't be " *
                  "used in GPU arrays, so its forces can't be computed on a GPU system; " *
                  "use Float32 or Float64 parameters, or run on the CPU"))
        end
        if !KernelAbstractions.supports_float64(get_backend(coords)) &&
                any(x -> ustrip(x) isa Float64, ntuple(i -> getfield(b, i), fieldcount(typeof(b))))
            @warn "$(typeof(b)) has Float64 parameters, which this GPU backend doesn't support, " *
                  "so they are computed in lower precision; use Float32 parameters to avoid this"
        end
    end
    return nothing
end

# Per-BiasPotential lazy scratch (grad/d_buf/fs_svec on both backends; dist_scratch,
# d_bias_buf and bad_step GPU-only). One per bias in buffers.bias_scratch (force.jl), keyed by
# the bias.
mutable struct BiasScratch
    grad::Any
    d_buf::Any
    fs_svec::Any
    dist_scratch::Any
    d_bias_buf::Any # bias_gradient output on the device
    bad_step::Any   # first step with a non-finite value, or -1
end

BiasScratch() = BiasScratch(nothing, nothing, nothing, nothing, nothing, nothing)

# Keyed by the bias itself (===), since its position in general_inters can differ between calls
# (e.g. MTS levels). Identical biases compute identical values, so they can share a scratch.
# A bufferless call gets a throwaway BiasScratch.
bias_scratch(buffers, bias::BiasPotential) = get!(BiasScratch, buffers.bias_scratch, bias)
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
    return unwrap_molecules(sys.coords, sys.boundary, sys.topology)
end

"""
    uses_device_bias_gradient(bias_type)

Whether `bias_gradient` of `bias_type` runs inside the GPU kernel of a [`BiasPotential`](@ref)
with a GPU CV type, without a transfer to the host.

Defaults to `false`, so custom bias types run `bias_gradient` on the host.
The built-in bias types return `true` when their parameters are plain data; a GPU `System`
rejects them otherwise.
Define `Molly.uses_device_bias_gradient(::MyBias) = true` only if `bias_gradient` for `MyBias`
is GPU-compilable and reads no host memory, for example no global arrays.
It must also return the same type for every CV value, since the GPU buffer type comes from one
call.
A wrong opt-in can cause an illegal memory access, which leaves the GPU unusable for the rest
of the Julia session.
"""
uses_device_bias_gradient(bias_type) = false
uses_device_bias_gradient(b::BuiltinBias) = isbits(b)

# Partition a heterogeneous tuple of interactions into the BiasPotentials that take the sync-free
# path and everything else, preserving each group's relative order. Recurses rather than
# using filter, which isn't type-stable on a mixed-type Tuple.
@inline split_biases(::Tuple{}) = (), ()
@inline function split_biases(t::Tuple)
    rest_biases, rest_others = split_biases(Base.tail(t))
    x = first(t)
    if x isa BiasPotential && uses_builtin_cv_gradient(x.cv_type) &&
            uses_device_bias_gradient(x.bias_type)
        return (x, rest_biases...), rest_others
    else
        return rest_biases, (x, rest_others...)
    end
end

# Lazily allocates scratch.grad/scratch.d_buf. Only reached for CV types with
# uses_builtin_cv_gradient (cv.jl), which have a real cv_gradient!/calculate_cv!.
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
# Builds the GPU scratch of a CV once, see cv_scratch (cv.jl)
function ensure_bias_dist_scratch!(scratch::BiasScratch, cv, coords, atoms)
    if isnothing(scratch.dist_scratch) && is_gpu_resident(coords)
        scratch.dist_scratch = cv_scratch(cv, coords, atoms)
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
        scratch.bad_step .= -1
    end
    return nothing
end

# Sync-free GPU path for the biases with persistent buffers: bias_cv_step! runs each bias's CV
# kernels, then bias_batched_tail! evaluates bias_gradient, applies the forces and checks them
# for every bias at once in 3 launches, without reading anything back to the host. Non-finite
# values are recorded on the device and reported by check_bias_finite_flags!.
#
# The kernels recurse over tuples with one entry per bias, which keeps indexing type-stable when
# the element types differ per bias (d_buf/d_bias_buf have a different Unitful eltype per bias).
# The helpers assume at least one bias, so the base case is the 1-tuple.
@inline function _bias_gradient_recurse(d_bias_bufs::Tuple{Any}, d_bufs::Tuple{Any}, bias_types::Tuple{Any})
    d_bias_bufs[1][1] = bias_gradient(bias_types[1], d_bufs[1][1])
    return nothing
end
@inline function _bias_gradient_recurse(d_bias_bufs::Tuple, d_bufs::Tuple, bias_types::Tuple)
    d_bias_bufs[1][1] = bias_gradient(bias_types[1], d_bufs[1][1])
    return _bias_gradient_recurse(Base.tail(d_bias_bufs), Base.tail(d_bufs), Base.tail(bias_types))
end

# bias_types is a plain Tuple of bias_type values passed through as kernel arguments
@kernel inbounds=true function bias_batched_gradient_kernel!(d_bias_bufs::Tuple, @Const(d_bufs::Tuple),
                                                               bias_types::Tuple)
    idx = @index(Global, Linear)
    idx == 1 && _bias_gradient_recurse(d_bias_bufs, d_bufs, bias_types)
end

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

@kernel inbounds=true function bias_batched_apply_kernel!(fs, grads::Tuple, fs_svecs::Tuple,
                                                            d_bias_bufs::Tuple)
    i = @index(Global, Linear)
    fs[i] -= _bias_apply_recurse(grads, fs_svecs, d_bias_bufs, i)
end

# d_bias is finite whenever the CV value is, so a non-finite CV gradient always gives a
# non-finite force: checking the CV value and the forces covers everything. Each bias's bad_step
# is only ever set to the same step_n, so no atomic is needed.
@inline function _bias_check_one(fs_svec, d_buf, bad_step, i, step_n)
    bad = (i <= length(fs_svec) && !bias_all_finite(fs_svec[i])) ||
          (i == 1 && !bias_all_finite(d_buf[1]))
    if bad && bad_step[1] < 0
        bad_step[1] = step_n
    end
    return nothing
end
@inline function _bias_check_recurse(fs_svecs::Tuple{Any}, d_bufs::Tuple{Any}, bad_steps::Tuple{Any},
                                     i, step_n)
    return _bias_check_one(fs_svecs[1], d_bufs[1], bad_steps[1], i, step_n)
end
@inline function _bias_check_recurse(fs_svecs::Tuple, d_bufs::Tuple, bad_steps::Tuple, i, step_n)
    _bias_check_one(fs_svecs[1], d_bufs[1], bad_steps[1], i, step_n)
    return _bias_check_recurse(Base.tail(fs_svecs), Base.tail(d_bufs), Base.tail(bad_steps), i, step_n)
end

@kernel inbounds=true function bias_batched_finite_kernel!(fs_svecs::Tuple, @Const(d_bufs::Tuple),
                                                             bad_steps::Tuple, step_n::Int)
    i = @index(Global, Linear)
    _bias_check_recurse(fs_svecs, d_bufs, bad_steps, i, step_n)
end

# Per-bias half of the sync-free path: the CV kernels, which differ per CV type
function bias_cv_step!(bias::BiasPotential, scratch::BiasScratch, sys, coords, fs)
    ensure_bias_buffers!(scratch, bias.cv_type, coords)
    ensure_bias_dist_scratch!(scratch, bias.cv_type, coords, sys.atoms)
    ensure_bias_gradient_buffer!(scratch, bias.bias_type)
    ensure_bias_finite_buffer!(scratch)
    cv_gradient!(scratch.grad, scratch.d_buf, bias.cv_type, coords, sys.atoms, sys.boundary,
                 sys.velocities; scratch=scratch.dist_scratch)
    isnothing(scratch.fs_svec) && (scratch.fs_svec = similar(fs))
    return nothing
end

# Batched half of the sync-free path, after bias_cv_step! has run for every bias.
# scratches pairs 1:1 with biases.
function bias_batched_tail!(fs, biases::Tuple, scratches::Tuple, step_n)
    isempty(biases) && return nothing
    grads       = map(s -> s.grad,       scratches)
    d_bufs      = map(s -> s.d_buf,      scratches)
    fs_svecs    = map(s -> s.fs_svec,    scratches)
    d_bias_bufs = map(s -> s.d_bias_buf, scratches)
    bias_types  = map(b -> b.bias_type,  biases)
    backend = get_backend(fs)
    n_atoms = length(first(grads))
    grad_kernel! = bias_batched_gradient_kernel!(backend, 1)
    grad_kernel!(d_bias_bufs, d_bufs, bias_types; ndrange=1)
    apply_kernel! = bias_batched_apply_kernel!(backend, min(n_atoms, 256))
    apply_kernel!(fs, grads, fs_svecs, d_bias_bufs; ndrange=n_atoms)
    check_kernel! = bias_batched_finite_kernel!(backend, min(n_atoms, 256))
    check_kernel!(fs_svecs, d_bufs, map(s -> s.bad_step, scratches), Int(step_n); ndrange=n_atoms)
    return nothing
end

# Steps between readbacks of the non-finite flags in the GPU forces!
const BIAS_FINITE_CHECK_EVERY = 20

# Reads back the non-finite flags of every attached BiasPotential with one host sync: every
# BIAS_FINITE_CHECK_EVERY steps, when forces! gets check_flags=true, and at the end of
# simulate!. A no-op for biases that haven't used the sync-free path.
function check_bias_finite_flags!(general_inters, buffers)
    scratches = Tuple((b, get(buffers.bias_scratch, b, nothing)) for b in general_inters
                      if b isa BiasPotential)
    live = filter(((_, s),) -> !isnothing(s) && !isnothing(s.bad_step), scratches)
    isempty(live) && return nothing
    bad_steps_h = from_device(reduce(vcat, map(bs -> bs[2].bad_step, live)))
    for ((bias, _), bad_step) in zip(live, bad_steps_h)
        if bad_step >= 0
            error("BiasPotential with CV $(typeof(bias.cv_type)) and bias " *
                  "$(typeof(bias.bias_type)) first produced a non-finite value at step $bad_step")
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

    if uses_builtin_cv_gradient(bias.cv_type)
        ensure_bias_buffers!(scratch, bias.cv_type, coords)
        ensure_bias_dist_scratch!(scratch, bias.cv_type, coords, sys.atoms)
        calculate_cv!(bias.cv_type, coords, sys.atoms, sys.boundary, scratch.d_buf, sys.velocities;
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
    kwargs...
)
    coords = bias_coords(sys, bias.cv_type, buffers, step_n)
    scratch = bias_scratch(buffers, bias)

    # Gradient of CV with respect to coordinates
    if uses_builtin_cv_gradient(bias.cv_type)
        ensure_bias_buffers!(scratch, bias.cv_type, coords)
        ensure_bias_dist_scratch!(scratch, bias.cv_type, coords, sys.atoms)
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

    if uses_builtin_cv_gradient(bias.cv_type)
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
    if !bias_all_finite(fs_svec)
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
        if uses_builtin_cv_gradient(bias.cv_type)
            calculate_virial!(buffers.virial, bias.cv_type, coords, -fs_svec, sys.atoms, sys.boundary;
                             scratch=scratch.dist_scratch)
        else
            calculate_virial!(buffers.virial, bias.cv_type, from_device(coords), -fs_svec,
                              from_device(sys.atoms), sys.boundary)
        end
    end

    fs .-= uses_builtin_cv_gradient(bias.cv_type) ? fs_svec : to_device(fs_svec, typeof(fs))
    return fs
end
