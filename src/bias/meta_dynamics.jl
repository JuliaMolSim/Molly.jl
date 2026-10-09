# Metadynamics bias potential

export
    MetaDynamicsBias,
    AbstractMetaDynamicsMemory,
    ListHills,
    GridHills,
    AbstractTempering,
    NoTempering,
    WellTemperedTempering,
    tempering_height,
    add_hill!

function validate_positive_finite(value, label::AbstractString)
    if !isfinite(ustrip(value)) || value <= zero(value)
        throw(ArgumentError("$(label) must be finite and positive, got $(value)."))
    end
    return value
end

"""
    AbstractMetaDynamicsMemory

Abstract type for the different ways a [`MetaDynamicsBias`](@ref) can store deposited
Gaussian hills.

Subtypes must implement `potential_energy`, `bias_gradient` and `add_hill!`. Built-in
subtypes are [`ListHills`](@ref) and [`GridHills`](@ref).
"""
abstract type AbstractMetaDynamicsMemory end

# Empty hill-center vector with the element type of sigma, which is in CV units (a tuple
# for several CVs), so deposits keep the user's precision and units.
empty_centers(sigma) = typeof(sigma)[]

@doc raw"""
    ListHills(k, sigma, centers=typeof(sigma)[], heights=fill(k, length(centers)))

Metadynamics memory storing every deposited hill explicitly, summed at each evaluation:
```math
V(s) = \sum_{t' < t} h_{t'} \exp\left(-\frac{|s - s(t')|^2}{2\sigma^2}\right)
```
O(n_hills) per evaluation; see [`GridHills`](@ref) for an O(1) alternative.

`k`, `sigma` and the elements of `centers` are scalars for a single CV, or same-length
tuples for biasing several CVs at once (see [`MetaDynamicsBias`](@ref)).

# Arguments
- `k`: Default hill height, used unless [`add_hill!`](@ref) is given an explicit height.
- `sigma`: Gaussian width (standard deviation), in CV units.
- `centers`: CV values where hills have already been deposited. Defaults to an empty
    vector with the element type of `sigma`; its element type sets the precision and
    units of every later deposit.
- `heights`: Deposited height per entry in `centers`; defaults to `k`.
"""
struct ListHills{K, R, V, H} <: AbstractMetaDynamicsMemory
    k::K
    sigma::R
    centers::V
    heights::H

    function ListHills(k::K, sigma::R, centers::V=empty_centers(sigma),
                       heights::H=fill(k, length(centers))) where {K, R, V, H}
        validate_positive_finite.(sigma, "ListHills sigma")
        if length(heights) != length(centers)
            throw(ArgumentError(
                "ListHills heights must be the same length as centers, got " *
                "$(length(heights)) and $(length(centers))."))
        end
        return new{K, R, V, H}(k, sigma, centers, heights)
    end
end

function potential_energy(mem::ListHills, cv_sim; kwargs...)
    total = zero(mem.k)
    for (center, height) in zip(mem.centers, mem.heights)
        scaled_diff = (cv_sim .- center) ./ mem.sigma
        total += height * exp(-0.5 * sum(abs2, scaled_diff))
    end
    return total
end

function bias_gradient(mem::ListHills, cv_sim)
    total = zero(mem.k) ./ mem.sigma
    for (center, height) in zip(mem.centers, mem.heights)
        diff = cv_sim .- center
        scaled_diff = diff ./ mem.sigma
        weight = height * exp(-0.5 * sum(abs2, scaled_diff))
        total = total .+ weight .* (.-diff) ./ (mem.sigma .^ 2)
    end
    return total
end

function add_hill!(mem::ListHills, cv_value, height=mem.k)
    push!(mem.centers, cv_value)
    push!(mem.heights, height)
    return nothing
end

@doc raw"""
    GridHills(k, sigma, grid_min, grid_max, n_bins, cutoff=6; out_of_grid_error=true)

Metadynamics memory accumulating deposited hills onto a discretized grid spanning
`[grid_min, grid_max]`, evaluated by (N-linear) interpolation. O(1) per evaluation
regardless of the number of deposited hills; see [`ListHills`](@ref) for an exact but
O(n_hills) alternative.

For a single CV, `sigma`, `grid_min` and `grid_max` are scalars and `n_bins` an Integer,
giving a 1D grid. For several CVs, pass same-length tuples (one entry per CV, matching
`cvs`) and `n_bins` as an Integer or a matching tuple, giving a dense N-dimensional grid.
Cost scales as roughly O(prod(n_bins)) memory and O(cutoff^N) per deposit, so this is only
practical for a handful of CVs (2-3); use [`ListHills`](@ref) for more.

# Arguments
- `k`: Default hill height, used unless [`add_hill!`](@ref) is given an explicit height.
- `sigma`: Gaussian width (standard deviation), scalar or one per dimension.
- `grid_min`, `grid_max`: Extent of the grid, scalar or one per dimension.
- `n_bins`: Number of grid points, at least 2 in every dimension; a single Integer or one
    per dimension.
- `cutoff`: Standard deviations beyond which a deposited hill's contribution is ignored.
- `out_of_grid_error::Bool=true`: Whether a CV value outside `[grid_min, grid_max]` throws
    an error. If `false`, a warning is given once and the bias is held constant at its
    edge value outside the grid, so it exerts no force there.
"""
mutable struct GridHills{K, R, G, E, T} <: AbstractMetaDynamicsMemory
    k::K
    sigma::R
    grid_min::G
    grid_max::G
    bin_width::G
    values::E
    cutoff::T
    out_of_grid_error::Bool
end

# Wraps a scalar into a length-1 tuple, so a single N-dimensional implementation below
# covers both the single- and multi-CV cases.
as_dims_tuple(x::Tuple) = x
as_dims_tuple(x::AbstractVector) = Tuple(x)
as_dims_tuple(x) = (x,)

function GridHills(k, sigma, grid_min, grid_max, n_bins, cutoff=6;
                   out_of_grid_error::Bool=true)
    sigma_t = as_dims_tuple(sigma)
    # Float bounds of a common type per dimension, so bin_width can share their type
    grid_min_t = float.(as_dims_tuple(grid_min))
    grid_max_t = float.(as_dims_tuple(grid_max))
    if length(grid_min_t) == length(grid_max_t)
        bounds_t = map(promote, grid_min_t, grid_max_t)
        grid_min_t, grid_max_t = first.(bounds_t), last.(bounds_t)
    end
    n_dims = length(grid_min_t)
    n_bins_t = n_bins isa Integer ? ntuple(_ -> n_bins, n_dims) : as_dims_tuple(n_bins)

    if length(sigma_t) != n_dims || length(grid_max_t) != n_dims || length(n_bins_t) != n_dims
        throw(ArgumentError(
            "GridHills sigma, grid_min, grid_max and n_bins must all cover the same number " *
            "of dimensions, got $(length(sigma_t)), $(n_dims), $(length(grid_max_t)) and " *
            "$(length(n_bins_t))."))
    end
    validate_positive_finite.(sigma_t, "GridHills sigma")
    if any(nb -> nb < 2, n_bins_t)
        throw(ArgumentError("GridHills n_bins must be at least 2 in every dimension, got $(n_bins_t)."))
    end
    if any(gmax <= gmin for (gmin, gmax) in zip(grid_min_t, grid_max_t))
        throw(ArgumentError("GridHills grid_max must be greater than grid_min in every dimension."))
    end

    bin_width_t = (grid_max_t .- grid_min_t) ./ (n_bins_t .- 1)
    values = fill(zero(k), n_bins_t)
    return GridHills{typeof(k), typeof(sigma_t), typeof(grid_min_t), typeof(values), typeof(cutoff)}(
        k, sigma_t, grid_min_t, grid_max_t, bin_width_t, values, cutoff, out_of_grid_error,
    )
end

# Errors, or warns once when out_of_grid_error is false, if cv_sim lies outside the grid
function check_in_grid(mem::GridHills, cv_sim)
    for d in eachindex(mem.grid_min)
        if !(mem.grid_min[d] <= cv_sim[d] <= mem.grid_max[d])
            msg = "GridHills CV value $(cv_sim[d]) in dimension $d is outside the grid " *
                  "[$(mem.grid_min[d]), $(mem.grid_max[d])]"
            if mem.out_of_grid_error
                throw(ArgumentError(msg * "; widen the grid, add walls to confine the " *
                                    "CV, or construct the GridHills with " *
                                    "out_of_grid_error=false to hold the bias constant " *
                                    "outside the grid"))
            else
                @warn msg * ", the bias is held constant outside the grid" maxlog=1
            end
        end
    end
    return nothing
end

grid_axis_value(mem::GridHills, d::Integer, i::Integer) = mem.grid_min[d] + (i - 1) * mem.bin_width[d]

# Per-dimension bracket indices and fractional distance, for N-linear interpolation.
# cv_sim[d] on a scalar cv_sim (1D case) just returns cv_sim itself.
function grid_bracket_dims(mem::GridHills, cv_sim)
    n_dims = length(mem.grid_min)
    return ntuple(n_dims) do d
        n = size(mem.values, d)
        clamped = clamp(cv_sim[d], mem.grid_min[d], mem.grid_max[d])
        frac_bin = ustrip((clamped - mem.grid_min[d]) / mem.bin_width[d])
        i0 = clamp(floor(Int, frac_bin) + 1, 1, n - 1)
        (i0, i0 + 1, frac_bin - (i0 - 1))
    end
end

function potential_energy(mem::GridHills, cv_sim; kwargs...)
    check_in_grid(mem, cv_sim)
    dims = grid_bracket_dims(mem, cv_sim)
    n_dims = length(dims)
    total = zero(mem.k)
    for corner in CartesianIndices(ntuple(_ -> 2, n_dims))
        idx = ntuple(d -> corner[d] == 1 ? dims[d][1] : dims[d][2], n_dims)
        weight = prod(d -> corner[d] == 1 ? (1 - dims[d][3]) : dims[d][3], 1:n_dims)
        total += weight * mem.values[idx...]
    end
    return total
end

# Partial derivative of the N-linear interpolant along each dimension (chain rule through
# frac_d = (cv_sim[d] - grid_min[d]) / bin_width[d]). Collapses to a scalar for a 1D grid.
# Zero along dimensions where cv_sim is outside the grid, matching the clamped energy.
function bias_gradient(mem::GridHills, cv_sim)
    check_in_grid(mem, cv_sim)
    dims = grid_bracket_dims(mem, cv_sim)
    n_dims = length(dims)
    grad = ntuple(n_dims) do d
        total = zero(mem.k) / mem.bin_width[d]
        if !(mem.grid_min[d] <= cv_sim[d] <= mem.grid_max[d])
            return total
        end
        for corner in CartesianIndices(ntuple(_ -> 2, n_dims))
            idx = ntuple(e -> corner[e] == 1 ? dims[e][1] : dims[e][2], n_dims)
            other_weight = prod(1:n_dims) do e
                e == d ? one(dims[e][3]) : (corner[e] == 1 ? (1 - dims[e][3]) : dims[e][3])
            end
            sign = corner[d] == 1 ? -1 : 1
            total += sign * other_weight * mem.values[idx...] / mem.bin_width[d]
        end
        total
    end
    return n_dims == 1 ? only(grad) : grad
end

function add_hill!(mem::GridHills, cv_value, height=mem.k)
    check_in_grid(mem, cv_value)
    n_dims = length(mem.grid_min)
    ranges = ntuple(n_dims) do d
        n = size(mem.values, d)
        cutoff_d = mem.cutoff * mem.sigma[d]
        lo = clamp(cv_value[d] - cutoff_d, mem.grid_min[d], mem.grid_max[d])
        hi = clamp(cv_value[d] + cutoff_d, mem.grid_min[d], mem.grid_max[d])
        i_lo = clamp(floor(Int, ustrip((lo - mem.grid_min[d]) / mem.bin_width[d])) + 1, 1, n)
        i_hi = clamp(ceil(Int, ustrip((hi - mem.grid_min[d]) / mem.bin_width[d])) + 1, 1, n)
        i_lo:i_hi
    end
    for idx in CartesianIndices(ranges)
        scaled_sq = sum(1:n_dims) do d
            diff = grid_axis_value(mem, d, idx[d]) - cv_value[d]
            (diff / mem.sigma[d])^2
        end
        mem.values[idx] += height * exp(-0.5 * scaled_sq)
    end
    return nothing
end

"""
    AbstractTempering

Controls how the height of each newly deposited hill is scaled before being added to a
[`MetaDynamicsBias`](@ref)'s memory.

Subtypes must implement `tempering_height(tempering, bias::MetaDynamicsBias, cv_sim,
base_height)`, returning the height to actually deposit; `bias` gives access to
`bias.cvs`, `bias.memory`, `bias.n_deposits[]`/`bias.deposit_interval` and `cv_sim`.

Built-in subtypes are [`NoTempering`](@ref) (the default) and
[`WellTemperedTempering`](@ref).
"""
abstract type AbstractTempering end

"""
    NoTempering()

Default tempering: every hill keeps its full base height (standard, constant-height
Metadynamics).
"""
struct NoTempering <: AbstractTempering end

tempering_height(::NoTempering, bias, cv_sim, base_height) = base_height

@doc raw"""
    WellTemperedTempering(bias_factor, kT)

Standard well-tempered Metadynamics height decay (Barducci, Bussi & Parrinello, 2008):
```math
h(\boldsymbol{s}) = h_0 \exp\left(-\frac{V_{bias}(\boldsymbol{s})}{k_B (\gamma - 1) T}\right)
```
where $h_0$ is the base height, $V_{bias}$ the bias accumulated so far at $\boldsymbol{s}$,
$\gamma$ is `bias_factor` (> 1), and $k_B T$ is `kT`. Deposits become self-limiting, so the
bias converges to an estimate of the free energy (scaled by `bias_factor`) instead of
growing without bound.

# Arguments
- `bias_factor`: Temperature boost $\gamma > 1$; larger tempers more slowly (approaching
    untempered Metadynamics as $\gamma \to \infty$).
- `kT`: Thermal energy $k_B T$, in the same units as the bias height.
"""
struct WellTemperedTempering{Γ, T} <: AbstractTempering
    bias_factor::Γ
    kT::T

    function WellTemperedTempering(bias_factor::Γ, kT::T) where {Γ, T}
        if bias_factor <= one(bias_factor)
            throw(ArgumentError(
                "WellTemperedTempering bias_factor must be greater than 1, got " *
                "$(bias_factor)."))
        end
        validate_positive_finite(kT, "WellTemperedTempering kT")
        return new{Γ, T}(bias_factor, kT)
    end
end

function tempering_height(wt::WellTemperedTempering, bias, cv_sim, base_height)
    v_bias = potential_energy(bias.memory, cv_sim)
    return base_height * exp(-v_bias / (wt.kT * (wt.bias_factor - 1)))
end

@doc raw"""
    MetaDynamicsBias(k, sigma, centers=typeof(sigma)[]; deposit_interval=1, tempering=NoTempering())
    MetaDynamicsBias(cvs, k, sigma, centers=typeof(sigma)[]; deposit_interval=1, tempering=NoTempering())
    MetaDynamicsBias(cvs, memory; deposit_interval=1, tempering=NoTempering())

A history-dependent bias potential for Metadynamics: a sum of Gaussians deposited at
previously visited collective variable (CV) values.
```math
V(\boldsymbol{s}) = \sum_{t' < t} h_{t'} \exp\left(-\frac{|\boldsymbol{s} - \boldsymbol{s}(t')|^2}{2\sigma^2}\right)
```

The `memory` argument (an [`AbstractMetaDynamicsMemory`](@ref)) controls how hills are
stored: [`ListHills`](@ref) (exact, O(n_hills)) or [`GridHills`](@ref) (O(1), grid-based).

Two usage modes:
- **Single CV, evaluated externally**: `MetaDynamicsBias(k, sigma)`
    builds a `ListHills`-backed bias with no CVs of its own, for use as the `bias_type` of
    a [`BiasPotential`](@ref).
- **One or more CVs, evaluated internally**: `MetaDynamicsBias(cvs, memory)` stores a tuple
    `cvs` of CV descriptors (e.g. [`CalcDist`](@ref)) and is itself an AtomsCalculators.jl
    calculator usable directly as a `general_inters` entry. Forces are computed every
    simulation step regardless of simulator, and a hill is deposited every
    `deposit_interval` steps -- no external logger is needed. Pacing uses the simulation
    step number, so recomputing forces within a step (e.g. by a logger or coupler) adds no
    extra hills. Energy minimizers also pass step numbers, so minimize without the bias.
    The number of hills deposited so far is `bias.n_deposits[]`.

[`add_hill!`](@ref) is the lower-level deposit entry point, useful directly for the
externally-evaluated single-CV form (which has no `forces!` of its own to hook into).

`tempering` (an [`AbstractTempering`](@ref)) scales each deposited hill's height; the
default [`NoTempering`](@ref) always deposits the full base height.

# Arguments
- `cvs`: Tuple of CV descriptors; omit to evaluate the CV externally via [`BiasPotential`](@ref).
- `memory::AbstractMetaDynamicsMemory`: Storage and evaluation strategy for deposited hills.
- `deposit_interval::Integer=1`: Number of simulation steps between deposits into
    `memory`. For `forces!` calls without a step number and for `add_hill!` calls, the
    number of calls between deposits.
- `tempering::AbstractTempering=NoTempering()`: Scales the height of each deposited hill.
"""
struct MetaDynamicsBias{C <: Tuple, M <: AbstractMetaDynamicsMemory, TP <: AbstractTempering}
    cvs::C
    memory::M
    deposit_interval::Int
    call_count::Base.RefValue{Int}
    last_deposit_step::Base.RefValue{Int}
    n_deposits::Base.RefValue{Int}
    tempering::TP

    function MetaDynamicsBias(cvs::C, memory::M;
                              deposit_interval::Integer=1,
                              tempering::TP=NoTempering()) where {
                                  C <: Tuple, M <: AbstractMetaDynamicsMemory, TP <: AbstractTempering}
        # Empty cvs means the CV is evaluated externally via BiasPotential (see
        # check_meta_dynamics_cvs).
        if memory isa GridHills && !isempty(cvs) && length(cvs) != length(memory.grid_min)
            throw(ArgumentError(
                "GridHills memory has $(length(memory.grid_min)) dimension(s) but " *
                "$(length(cvs)) CVs were given; these must match. Build the GridHills with " *
                "one grid_min/grid_max/sigma entry per CV, or use ListHills instead."))
        end
        if deposit_interval < 1
            throw(ArgumentError("deposit_interval must be at least 1, got $(deposit_interval)."))
        end
        return new{C, M, TP}(cvs, memory, Int(deposit_interval), Ref(0), Ref(-1), Ref(0),
                             tempering)
    end
end

MetaDynamicsBias(k, sigma, centers=empty_centers(sigma); deposit_interval::Integer=1,
                 tempering::AbstractTempering=NoTempering()) =
    MetaDynamicsBias((), ListHills(k, sigma, centers);
                     deposit_interval=deposit_interval, tempering=tempering)
MetaDynamicsBias(cvs::Tuple, k, sigma, centers=empty_centers(sigma); deposit_interval::Integer=1,
                 tempering::AbstractTempering=NoTempering()) =
    MetaDynamicsBias(cvs, ListHills(k, sigma, centers);
                     deposit_interval=deposit_interval, tempering=tempering)

function potential_energy(md::MetaDynamicsBias, cv_sim; kwargs...)
    return potential_energy(md.memory, cv_sim; kwargs...)
end

function bias_gradient(md::MetaDynamicsBias, cv_sim)
    return bias_gradient(md.memory, cv_sim)
end

"""
    add_hill!(bias::MetaDynamicsBias, cv_value)
    add_hill!(bias::MetaDynamicsBias, sys)

Deposit a new Gaussian hill.

Called automatically from `forces!` when `bias` has a non-empty `cvs` -- there is usually
no need to call this directly. It remains useful for the externally-evaluated single-CV
form (via [`BiasPotential`](@ref)), which has no `forces!` to hook a deposit into.

`cv_value` matches the shape `bias` was constructed with (scalar or tuple); the `sys` form
evaluates `bias.cvs` itself. Every call counts towards `bias.deposit_interval`; only every
`deposit_interval`-th call actually updates the memory, at height
`tempering_height(bias.tempering, bias, cv_sim, bias.memory.k)`.
"""
function add_hill!(md::MetaDynamicsBias, cv_value)
    if should_deposit_hill!(md)
        deposit_hill!(md, cv_value)
    end
    return nothing
end

function add_hill!(md::MetaDynamicsBias, sys::System)
    check_meta_dynamics_cvs(md)
    if should_deposit_hill!(md)
        deposit_hill!(md, evaluate_meta_dynamics_cvs(md, sys))
    end
    return nothing
end

function deposit_hill!(md::MetaDynamicsBias, cv_sim)
    height = tempering_height(md.tempering, md, cv_sim, md.memory.k)
    add_hill!(md.memory, cv_sim, height)
    md.n_deposits[] += 1
    return nothing
end

# True when this call lands on the configured deposit_interval pace. Without a step number
#   every call counts; with one, only the first call of every deposit_interval-th step
#   deposits, so recomputing forces within a step (loggers, couplers, the initial virial)
#   adds no extra hills.
function should_deposit_hill!(md::MetaDynamicsBias, ::Nothing=nothing)
    md.call_count[] += 1
    return (md.call_count[] % md.deposit_interval) == 0
end

function should_deposit_hill!(md::MetaDynamicsBias, step_n::Integer)
    if step_n % md.deposit_interval != 0 || step_n == md.last_deposit_step[]
        return false
    end
    md.last_deposit_step[] = step_n
    return true
end

function check_meta_dynamics_cvs(md::MetaDynamicsBias)
    if isempty(md.cvs)
        throw(ArgumentError(
            "This MetaDynamicsBias has no stored collective variables, so it cannot be " *
            "used directly as a general interaction or with add_hill!(bias, sys). Either " *
            "construct it as MetaDynamicsBias(cvs, memory) with a tuple of CV descriptors, " *
            "or wrap it in a BiasPotential with an externally supplied CV: " *
            "BiasPotential(cv, bias)."))
    end
    return nothing
end

# Single CV -> bare scalar; multiple CVs -> tuple, matching what ListHills/GridHills expect.
reshape_meta_dynamics_cvs(cv_values::Tuple) = (length(cv_values) == 1) ? only(cv_values) : cv_values

function evaluate_meta_dynamics_cvs(md::MetaDynamicsBias, sys)
    coords_pbc = any(cv -> cv.correction == :pbc, md.cvs) ? unwrap_molecules(sys) : nothing
    cv_values = map(md.cvs) do cv
        coords = from_device(cv.correction == :pbc ? coords_pbc : sys.coords)
        calculate_cv(cv, coords, from_device(sys.atoms), sys.boundary, from_device(sys.velocities))
    end
    return reshape_meta_dynamics_cvs(cv_values)
end

function AtomsCalculators.potential_energy(sys, md::MetaDynamicsBias; kwargs...)
    check_meta_dynamics_cvs(md)
    cv_sim = evaluate_meta_dynamics_cvs(md, sys)
    return potential_energy(md.memory, cv_sim; kwargs...)
end

function AtomsCalculators.forces!(
    fs, sys, md::MetaDynamicsBias;
    needs_vir::Bool = false,
    buffers = nothing,
    step_n = nothing,
    kwargs...
)
    check_meta_dynamics_cvs(md)
    coords_pbc = any(cv -> cv.correction == :pbc, md.cvs) ? unwrap_molecules(sys) : nothing

    per_cv = map(md.cvs) do cv
        coords = from_device(cv.correction == :pbc ? coords_pbc : sys.coords)
        atoms = from_device(sys.atoms)
        d_coords, cv_val = cv_gradient(cv, coords, atoms, sys.boundary, from_device(sys.velocities))
        (cv=cv, coords=coords, atoms=atoms, d_coords=d_coords, cv_val=cv_val)
    end

    cv_sim = reshape_meta_dynamics_cvs(map(x -> x.cv_val, per_cv))
    d_bias = bias_gradient(md.memory, cv_sim)

    for (i, x) in enumerate(per_cv)
        fs_svec = d_bias[i] .* x.d_coords
        if needs_vir && x.cv.has_virial
            calculate_virial!(buffers.virial, x.cv, x.coords, -fs_svec, x.atoms, sys.boundary)
        end
        fs .-= to_device(fs_svec, typeof(fs))
    end

    # Self-paced deposit, by simulation step when called from a simulator
    if should_deposit_hill!(md, step_n)
        deposit_hill!(md, cv_sim)
    end

    return fs
end
