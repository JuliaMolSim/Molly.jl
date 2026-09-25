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

CV computation runs fully on the GPU when the `System` is GPU-resident, with no host transfer of
coordinates, atoms or forces, including `cv_type.correction = :pbc` (the default for the built-in
CV types), which unwraps bonded molecules across the periodic boundary using a GPU-native
spanning-forest traversal.
"""
mutable struct BiasPotential{C, B}
    cv_type::C
    bias_type::B
    uses_persistent_buffers::Bool   # set at construction, see uses_builtin_cv_gradient!
    grad::Any                       # lazy CV gradient buffer
    d_buf::Any                      # lazy CV-value buffer
    fs_svec::Any                    # lazy bias-force buffer
    dist_scratch::Any               # CalcMinDist/CalcMaxDist etc: fused-kernel scratch
    extremal_cache::Any             # CalcMinDist/CalcMaxDist: cached extremal pair for virial reuse
end

function BiasPotential(cv_type::C, bias_type::B) where {C, B}
    return BiasPotential{C, B}(cv_type, bias_type, uses_builtin_cv_gradient!(cv_type),
                               nothing, nothing, nothing, nothing, nothing)
end

bias_all_finite(values::AbstractArray) = all(bias_all_finite, values)
bias_all_finite(value) = isfinite(ustrip(value))

 function bias_max_abs_ustrip(values::AbstractArray)
    isempty(values) && return 0.0
    return mapreduce(bias_max_abs_ustrip, max, values)
end

bias_max_abs_ustrip(value) = abs(ustrip(value))

 function check_bias_finite(value, label::AbstractString, bias::BiasPotential;
                            cv_sim=nothing, max_abs_component=nothing)
    bias_all_finite(value) && return value
    msg = "BiasPotential with CV $(typeof(bias.cv_type)) and bias " *
          "$(typeof(bias.bias_type)) produced non-finite $(label)"
    if !isnothing(cv_sim)
        msg *= ", cv_sim=$(cv_sim)"
    end
    if !isnothing(max_abs_component)
        msg *= ", max_abs_component=$(max_abs_component)"
    end
    error(msg)
end

"""
    bias_coords(sys, cv_type, buffers=nothing, step_n=nothing)

Return the coordinates a `BiasPotential` should use, unwrapping across periodic
boundaries when `cv_type.correction == :pbc`.

When `buffers`/`step_n` are supplied, routes through the shared, once-per-step
unwrap cache (`ensure_unwrapped_coords!`, `src/force.jl`) instead of recomputing
`unwrap_molecules` independently for every attached `BiasPotential`.
"""
function bias_coords(sys, cv_type, buffers=nothing, step_n=nothing)
    cv_type.correction != :pbc && return sys.coords
    if !isnothing(buffers) && !isnothing(step_n) && hasproperty(buffers, :unwrapped_coords)
        return ensure_unwrapped_coords!(buffers, sys, step_n)
    end
    return unwrap_molecules(sys)
end

bias_needs_unwrap(b::BiasPotential) = b.cv_type.correction == :pbc

# Lazily allocates bias.grad/bias.d_buf. Only reached when bias.uses_persistent_buffers is true
# (a CV type with a real cv_gradient!/calculate_cv! -- see uses_builtin_cv_gradient! in cv.jl).
function ensure_bias_buffers!(bias::BiasPotential, coords)
    if bias.grad === nothing
        bias.grad, bias.d_buf = zero_cv_gradient_buffers(bias.cv_type, coords)
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

function bias_dist_scratch_types(coords, atoms)
    CT = eltype(coords)
    MT = fieldtype(eltype(atoms), :mass)
    WT = typeof(zero(CT) * zero(MT))
    IT = typeof(sum_abs2(zero(CT)) * zero(MT))
    return CT, MT, WT, IT
end

"""
    ensure_bias_dist_scratch!(bias::BiasPotential, coords, atoms)

Lazily allocate CV-type-specific fused-kernel scratch (`bias.dist_scratch`, and for
CalcMinDist/CalcMaxDist also `bias.extremal_cache`) the first time it's needed.

Only allocated when `coords` is GPU-resident (the CPU path never uses it). Each CV
type's own `calculate_cv!`/`cv_gradient!` dispatch only takes its fused-kernel fast
path once its matching scratch struct has been populated here, falling back to the
generic broadcast path otherwise. See the matching `*Scratch` docstring in `cv.jl`
for why each one exists.
"""
function ensure_bias_dist_scratch!(bias::BiasPotential, coords, atoms)
    cv = bias.cv_type
    if cv isa CalcDist{<:Union{CalcMinDist, CalcMaxDist}} && bias.extremal_cache === nothing
        bias.extremal_cache = ExtremalPairCache(false, 0, 0, nothing, nothing)
        if is_gpu_resident(coords)
            idx1_dev, idx2_dev = upload_idx(coords, cv.atom_inds_1), upload_idx(coords, cv.atom_inds_2)
            na, nb = length(cv.atom_inds_1), length(cv.atom_inds_2)
            # T caps kernel worker count at MINDIST_TILE_CAP regardless of na*nb -- see
            # MinMaxScratch's docstring (cv.jl).
            T = min(na * nb, MINDIST_TILE_CAP)
            bias.dist_scratch = MinMaxScratch(
                idx1_dev,
                idx2_dev,
                similar(coords, eltype(eltype(coords)), T),
                similar(coords, Int, T),
                similar(coords, Int, T),
                similar(coords, T),
                similar(coords, Int, 1),
                similar(coords, Int, 1),
                similar(coords, 1),
            )
        end
    elseif cv isa CalcDist{CalcCMDist} && bias.dist_scratch === nothing && is_gpu_resident(coords)
        na, nb = length(cv.atom_inds_1), length(cv.atom_inds_2)
        T1, T2 = min(na, 1024), min(nb, 1024)
        CT, MT, WT, _ = bias_dist_scratch_types(coords, atoms)
        DT = typeof(zero(CT) / oneunit(eltype(CT))) # unit-stripped direction vector
        bias.dist_scratch = CMDistScratch(
            upload_idx(coords, cv.atom_inds_1), upload_idx(coords, cv.atom_inds_2),
            similar(coords, MT, T1), similar(coords, WT, T1),
            similar(coords, MT, T2), similar(coords, WT, T2),
            similar(coords, DT, 1),
            similar(coords, MT, 1), similar(coords, MT, 1),
        )
    elseif cv isa CalcRg && bias.dist_scratch === nothing && is_gpu_resident(coords)
        inds = iszero(length(cv.atom_inds)) ? collect(1:length(coords)) : cv.atom_inds
        n = length(inds)
        T = min(n, RG_TILE_CAP)
        CT, MT, WT, IT = bias_dist_scratch_types(coords, atoms)
        bias.dist_scratch = RgScratch(
            upload_idx(coords, inds),
            similar(coords, MT, T), similar(coords, WT, T),
            similar(coords, IT, T),
            similar(coords, CT, 1), similar(coords, MT, 1),
        )
    elseif cv isa CalcRMSD && bias.dist_scratch === nothing && is_gpu_resident(coords)
        inds = iszero(length(cv.atom_inds)) ? collect(1:length(coords)) : cv.atom_inds
        ref_inds = iszero(length(cv.ref_atom_inds)) ? collect(1:length(cv.ref_coords)) : cv.ref_atom_inds
        ref_coords_used = cv.ref_coords[ref_inds]
        # cv.ref_coords never changes after construction, so its Kabsch-centered form is computed
        # once here rather than on every cv_gradient!/calculate_cv! call (RmsdScratch docstring, cv.jl).
        n = length(inds)
        T = min(n, RMSD_TILE_CAP)
        _, _, _, IT = bias_dist_scratch_types(coords, atoms)
        bias.dist_scratch = RmsdScratch(upload_idx(coords, inds), similar(coords, length(inds)),
                                        ref_coords_used, kabsch_centered(ref_coords_used),
                                        similar(coords, IT, T))
    end
    return nothing
end

function AtomsCalculators.potential_energy(sys, bias::BiasPotential; kwargs...)
    coords = bias_coords(sys, bias.cv_type)

    if bias.uses_persistent_buffers
        ensure_bias_buffers!(bias, coords)
        ensure_bias_dist_scratch!(bias, coords, sys.atoms)
        calculate_cv_buffered!(bias.cv_type, coords, sys.atoms, sys.boundary, bias.d_buf, sys.velocities;
                               scratch=bias.dist_scratch, kwargs...)
        cv_sim = only(from_device(bias.d_buf))
    else
        cv_sim = calculate_cv(bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities; kwargs...)
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

    # Gradient of CV with respect to coordinates
    if bias.uses_persistent_buffers
        ensure_bias_buffers!(bias, coords)
        ensure_bias_dist_scratch!(bias, coords, sys.atoms)
        cv_gradient!(bias.grad, bias.d_buf, bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities;
                    extremal_cache=bias.extremal_cache, scratch=bias.dist_scratch)
        d_coords, cv_sim = bias.grad, only(from_device(bias.d_buf))
    else
        d_coords, cv_sim = cv_gradient(
            bias.cv_type,
            coords,
            sys.atoms,
            sys.boundary,
            sys.velocities,
        )
    end
    check_bias_finite(cv_sim, "collective variable", bias)
    check_bias_finite(d_coords, "CV gradient", bias; cv_sim=cv_sim)

    # Gradient of bias function with respect to CV
    d_bias = bias_gradient(bias.bias_type, cv_sim)
    check_bias_finite(d_bias, "bias gradient", bias; cv_sim=cv_sim)

    if bias.uses_persistent_buffers
        if bias.fs_svec === nothing
            bias.fs_svec = d_bias .* d_coords
        else
            bias.fs_svec .= d_bias .* d_coords
        end
        fs_svec = bias.fs_svec
    else
        fs_svec = d_bias .* d_coords
    end
    check_bias_finite(
        fs_svec,
        "bias force",
        bias;
        cv_sim=cv_sim,
        max_abs_component = bias_max_abs_ustrip(fs_svec),
    )

    if needs_vir && bias.cv_type.has_virial
        calculate_virial!(buffers.virial, bias.cv_type, coords, -fs_svec, sys.atoms, sys.boundary;
                          precomputed_extremum=bias.extremal_cache)
    end

    fs .-= fs_svec
    return fs
end
