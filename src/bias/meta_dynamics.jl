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

Abstract type for the different ways a [`MetaDynamicsBias`](@ref) can store the history
of deposited Gaussian hills.

Subtypes must implement `potential_energy`, `bias_gradient` and `add_hill!`.
Built-in subtypes are [`ListHills`](@ref) and [`GridHills`](@ref).
"""
abstract type AbstractMetaDynamicsMemory end

@doc raw"""
    ListHills(k, sigma, centers=Float64[], heights=fill(k, length(centers)))

Metadynamics memory that stores every deposited hill explicitly as a list of CV centers,
each with its own deposited height, and a shared width `sigma`.

The bias potential and its gradient are evaluated by summing the contribution of every
deposited hill:
```math
V(s) = \sum_{t' < t} h_{t'} \exp\left(-\frac{|s - s(t')|^2}{2\sigma^2}\right)
```
Evaluation cost scales as O(n_hills) with the number of deposited hills, so this is best
suited to short or infrequently biased simulations. See [`GridHills`](@ref) for O(1)
evaluation independent of the number of hills.

`k`, `sigma` and the elements of `centers` can either be scalars, for biasing a single CV,
or same-length tuples, for biasing several CVs at once as used by [`MetaDynamicsBias`](@ref)
when given multiple CV descriptors. Tuples allow each CV dimension to carry its own units
(e.g. a distance in nm and a torsion angle in radians biased together).

# Arguments
- `k`: Default/base height of each Gaussian hill, used unless [`add_hill!`](@ref) is given
    an explicit height (e.g. via tempering). Must match system energy units.
- `sigma`: Width (standard deviation) of the Gaussians. Must match CV units.
- `centers`: Vector of CV values where hills have already been deposited.
- `heights`: The deposited height of each entry in `centers`; defaults to `k` for all of
    them if not given.
"""
struct ListHills{K, R, V, H} <: AbstractMetaDynamicsMemory
    k::K
    sigma::R
    centers::V
    heights::H

    function ListHills(k::K, sigma::R, centers::V=DefaultFloat[],
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
    GridHills(k, sigma, grid_min, grid_max, n_bins, cutoff=6)

Metadynamics memory that accumulates deposited hills directly onto a discretized grid
spanning `[grid_min, grid_max]`.

For a single CV, `sigma`, `grid_min` and `grid_max` are scalars and `n_bins` is an Integer,
giving a 1D grid stored as a `Vector` of `n_bins` points.

For several CVs biased together, pass `sigma`, `grid_min` and `grid_max` as same-length
tuples (one entry per CV, matching the `cvs` order given to [`MetaDynamicsBias`](@ref)) and
`n_bins` as either a single Integer (used for every dimension) or a matching tuple of
per-dimension bin counts. This gives a dense N-dimensional grid stored as an `Array`, with
hills deposited/evaluated via N-linear interpolation instead of the 1D linear interpolation
used for a single CV. Both memory and per-deposit cost scale as roughly O(prod(n_bins)) and
O(cutoff^N) respectively, so this is only practical for a handful of CVs (2-3) with modest
bin counts; use [`ListHills`](@ref) instead for more CVs or finer/larger grids.

Each deposit adds the new hill's Gaussian contribution onto every grid point within
`cutoff` standard deviations of the hill center (per dimension), an operation performed
once per deposit. Evaluating the potential or gradient at simulation time is O(1)
(independent of the number of deposited hills), using linear (or N-linear) interpolation
between the grid points bracketing the current CV value.

This trades some accuracy (interpolation, and hills deposited near or outside
`[grid_min, grid_max]` are truncated at the boundary) for evaluation speed independent of
the number of deposited hills, making it well suited to long biased simulations. See
[`ListHills`](@ref) for an exact but O(n_hills) alternative.

# Arguments
- `k`: Default/base height of each Gaussian hill, used unless [`add_hill!`](@ref) is given
    an explicit height (e.g. via tempering). Must match system energy units.
- `sigma`: Width (standard deviation) of the Gaussians, scalar or one per dimension. Must
    match CV units.
- `grid_min`, `grid_max`: The extent of the grid, in CV units; scalar or one per dimension.
- `n_bins`: Number of grid points; for a multi-dimensional grid, a single Integer (applied
    to every dimension) or one per dimension. Must be at least 2 in every dimension.
- `cutoff`: Number of standard deviations beyond which a deposited hill's contribution to
    the grid is ignored.
"""
mutable struct GridHills{K, R, G, E, T} <: AbstractMetaDynamicsMemory
    k::K
    sigma::R    # NTuple{n_dims}, one per CV dimension
    grid_min::G # NTuple{n_dims}
    grid_max::G # NTuple{n_dims}
    bin_width::G # NTuple{n_dims}
    values::E   # n_dims-dimensional Array
    cutoff::T
end

# Wraps a scalar into a length-1 tuple so a single n_dims-dimensional implementation below
# (n_dims >= 1) covers both the single-CV and multi-CV cases; Tuples/AbstractVectors (one
# entry per CV dimension) pass through as-is.
as_dims_tuple(x::Tuple) = x
as_dims_tuple(x::AbstractVector) = Tuple(x)
as_dims_tuple(x) = (x,)

function GridHills(k, sigma, grid_min, grid_max, n_bins, cutoff=6)
    sigma_t = as_dims_tuple(sigma)
    grid_min_t = as_dims_tuple(grid_min)
    grid_max_t = as_dims_tuple(grid_max)
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
        k, sigma_t, grid_min_t, grid_max_t, bin_width_t, values, cutoff,
    )
end

grid_axis_value(mem::GridHills, d::Integer, i::Integer) = mem.grid_min[d] + (i - 1) * mem.bin_width[d]

# Per-dimension bracketing indices (i0, i1) and fractional distance between them, for
# N-linear interpolation. cv_sim is a scalar when n_dims == 1 (Number indexing cv_sim[1]
# just returns cv_sim itself) or an n_dims-tuple otherwise.
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

# Partial derivative of the N-linear interpolant along each dimension, via the chain rule
# through frac_d = (cv_sim[d] - grid_min[d]) / bin_width[d]. Returns a scalar for a 1D grid,
# matching potential_energy/bias_gradient's convention elsewhere in this file of collapsing
# a length-1 tuple down to its single value (see reshape_meta_dynamics_cvs).
function bias_gradient(mem::GridHills, cv_sim)
    dims = grid_bracket_dims(mem, cv_sim)
    n_dims = length(dims)
    grad = ntuple(n_dims) do d
        total = zero(mem.k) / mem.bin_width[d]
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

Abstract type controlling how the height of each newly deposited Metadynamics hill is
scaled before being added to a [`MetaDynamicsBias`](@ref)'s memory.

Subtypes must implement
`tempering_height(tempering, bias::MetaDynamicsBias, cv_sim, base_height)`, returning the
actual height to deposit. The function receives the whole `bias`, so it has access to the
CVs (`bias.cvs`), the memory accumulated so far (`bias.memory`), and how many hills have
been deposited so far (`bias.call_count[]`, `bias.deposit_interval`), alongside whatever
parameters the tempering subtype itself stores, and the CV value the hill is about to be
deposited at (`cv_sim`).

Built-in subtypes are [`NoTempering`](@ref) (the default) and
[`WellTemperedTempering`](@ref).
"""
abstract type AbstractTempering end

"""
    NoTempering()

The default tempering: every deposited hill keeps its full base height, i.e. standard
(non-tempered) Metadynamics with a constant hill height.
"""
struct NoTempering <: AbstractTempering end

tempering_height(::NoTempering, bias, cv_sim, base_height) = base_height

@doc raw"""
    WellTemperedTempering(bias_factor, kT)

The standard well-tempered Metadynamics height decay (Barducci, Bussi & Parrinello, 2008).
The height of each new hill is scaled down according to how much bias has already been
deposited at the current CV value:
```math
h(\boldsymbol{s}) = h_0 \exp\left(-\frac{V_{bias}(\boldsymbol{s})}{k_B (\gamma - 1) T}\right)
```
where $h_0$ is the base hill height, $V_{bias}(\boldsymbol{s})$ is the bias accumulated so
far at $\boldsymbol{s}$ (evaluated via [`potential_energy`](@ref) on the current memory),
$\gamma$ is `bias_factor` (must be greater than 1), and $k_B T$ is `kT`.

This makes deposits self-limiting: hills shrink as a region fills up, so the bias
converges towards an estimate of the free energy surface (scaled by `bias_factor`) instead
of growing without bound, as constant-height Metadynamics does. See
[`MetaDynamicsBias`](@ref) for how to attach this to a bias.

# Arguments
- `bias_factor`: The temperature boost factor $\gamma > 1$. Larger values temper more
    slowly, approaching standard (untempered) Metadynamics as $\gamma \to \infty$; values
    close to 1 temper very aggressively (near-immediate saturation).
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
    MetaDynamicsBias(k, sigma, centers=Float64[]; deposit_interval=1, tempering=NoTempering())
    MetaDynamicsBias(cvs, k, sigma, centers=Float64[]; deposit_interval=1, tempering=NoTempering())
    MetaDynamicsBias(cvs, memory; deposit_interval=1, tempering=NoTempering())

A history-dependent bias potential for standard Metadynamics.

The bias potential is a sum of Gaussians deposited at previously visited collective
variable (CV) values:
```math
V(\boldsymbol{s}) = \sum_{t' < t} h_{t'} \exp\left(-\frac{|\boldsymbol{s} - \boldsymbol{s}(t')|^2}{2\sigma^2}\right)
```

How the history of deposited hills is stored and evaluated is controlled by the `memory`
argument, an [`AbstractMetaDynamicsMemory`](@ref):
- [`ListHills`](@ref): an explicit list of hill centers, summed from scratch at every
    evaluation. Exact, O(n_hills) per evaluation, and supports biasing multiple CVs at once.
- [`GridHills`](@ref): a discretized grid accumulating the sum of deposited hills.
    Approximate (interpolation), O(1) per evaluation. Supports multiple CVs via an
    N-dimensional grid, but is only practical for a handful of them (2-3).

`MetaDynamicsBias` can be used in two ways:
- **Single CV, evaluated externally**: the one-CV-worth constructor `MetaDynamicsBias(k,
    sigma, centers=Float64[])` builds a `ListHills`-backed bias with no CVs of its own. Use
    it as the `bias_type` argument to [`BiasPotential`](@ref), which evaluates the CV and
    calls `potential_energy`/`bias_gradient` on this bias, exactly like [`SquareBias`](@ref)
    or [`LinearBias`](@ref).
- **One or more CVs, evaluated internally**: `MetaDynamicsBias(cvs, memory)` (or
    `MetaDynamicsBias(cvs, k, sigma, centers=Float64[])`) stores a tuple `cvs` of CV
    descriptors (e.g. [`CalcDist`](@ref), [`CalcTorsion`](@ref)). This form is itself an
    AtomsCalculators.jl calculator: it evaluates every CV in `cvs` against the current
    system state before calling the bias potential, and can be used directly as a
    `general_inters` entry. This is the natural way to run metadynamics on more than one
    CV at a time. When biasing several CVs, `k`, `sigma` and any hill `centers` should be
    passed as tuples matching the length of `cvs` (see [`ListHills`](@ref)).

    Used this way, the bias is fully self-updating: forces are computed every simulation
    step regardless of simulator, and `deposit_interval` (below) paces how often those
    force evaluations also deposit a hill. No external logger is needed to drive
    metadynamics -- just add the bias to `general_inters` and simulate as normal.

[`add_hill!`](@ref) is the lower-level entry point used internally for depositing (also
useful to call directly, e.g. for the externally-evaluated single-CV form, which has no
`forces!` method of its own to hook into). By default every call deposits a hill;
`deposit_interval` throttles that down to every `deposit_interval`-th call, whether the
call comes from `forces!` or from `add_hill!` directly.

`tempering` controls how much of `memory.k` each individual deposit actually uses (see
[`AbstractTempering`](@ref)): the default [`NoTempering`](@ref) always deposits the full
`memory.k`, giving standard constant-height Metadynamics. [`WellTemperedTempering`](@ref)
is provided as a simple, standard alternative that decays each hill's height based on how
much bias has already been deposited nearby, so the bias self-limits rather than growing
without bound. Custom schemes can be added by subtyping `AbstractTempering` and
implementing `tempering_height`.

# Arguments
- `cvs`: A tuple of collective variable descriptors, evaluated in order to build the CV
    vector the bias acts on. Omit to evaluate the CV externally via [`BiasPotential`](@ref).
- `memory::AbstractMetaDynamicsMemory`: Storage and evaluation strategy for deposited hills.
- `deposit_interval::Integer=1`: Number of deposit-triggering calls (force evaluations when
    used as a `general_inters` entry, or calls to `add_hill!` otherwise) between actual
    deposits into `memory`. For example `deposit_interval=500` deposits every 500 calls.
- `tempering::AbstractTempering=NoTempering()`: Scales the height of each deposited hill;
    see above.
"""
struct MetaDynamicsBias{C <: Tuple, M <: AbstractMetaDynamicsMemory, TP <: AbstractTempering}
    cvs::C
    memory::M
    deposit_interval::Int
    call_count::Base.RefValue{Int}
    tempering::TP

    function MetaDynamicsBias(cvs::C, memory::M;
                              deposit_interval::Integer=1,
                              tempering::TP=NoTempering()) where {
                                  C <: Tuple, M <: AbstractMetaDynamicsMemory, TP <: AbstractTempering}
        # An empty cvs is a valid, intentional state: it means the CV is evaluated
        # externally via BiasPotential rather than by this struct's own calculator methods
        # (see check_meta_dynamics_cvs, called from those methods instead).
        if memory isa GridHills && !isempty(cvs) && length(cvs) != length(memory.grid_min)
            throw(ArgumentError(
                "GridHills memory has $(length(memory.grid_min)) dimension(s) but " *
                "$(length(cvs)) CVs were given; these must match. Build the GridHills with " *
                "one grid_min/grid_max/sigma entry per CV, or use ListHills instead."))
        end
        if deposit_interval < 1
            throw(ArgumentError("deposit_interval must be at least 1, got $(deposit_interval)."))
        end
        return new{C, M, TP}(cvs, memory, Int(deposit_interval), Ref(0), tempering)
    end
end

MetaDynamicsBias(k, sigma, centers=DefaultFloat[]; deposit_interval::Integer=1,
                 tempering::AbstractTempering=NoTempering()) =
    MetaDynamicsBias((), ListHills(k, sigma, centers);
                     deposit_interval=deposit_interval, tempering=tempering)
MetaDynamicsBias(cvs::Tuple, k, sigma, centers=DefaultFloat[]; deposit_interval::Integer=1,
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

When `bias` has a non-empty `cvs`, this is called automatically from `forces!` every
simulation step it is used in `general_inters` (see [`MetaDynamicsBias`](@ref)) -- there is
usually no need to call it directly. It remains useful for the externally-evaluated
single-CV form (constructed via `MetaDynamicsBias(k, sigma)`, wrapped in
[`BiasPotential`](@ref)), which has no `forces!` method of its own to hook a deposit into,
so must be called manually, typically from a logger.

The single-value form takes `cv_value` matching the shape `bias` was constructed with: a
scalar for a single CV, or a tuple of values (one per entry in `bias.cvs`) for multiple
CVs. The `sys` form instead evaluates `bias.cvs` against the current system state itself
and deposits the result with the correct shape; it requires `bias` to have been
constructed with a non-empty `cvs`.

Every call counts towards `bias.deposit_interval` (see [`MetaDynamicsBias`](@ref)); only
every `deposit_interval`-th call actually updates the memory. The height actually
deposited is `tempering_height(bias.tempering, bias, cv_sim, bias.memory.k)` (see
[`AbstractTempering`](@ref)).
"""
function add_hill!(md::MetaDynamicsBias, cv_value)
    if should_deposit_hill!(md)
        height = tempering_height(md.tempering, md, cv_value, md.memory.k)
        add_hill!(md.memory, cv_value, height)
    end
    return nothing
end

function add_hill!(md::MetaDynamicsBias, sys::System)
    check_meta_dynamics_cvs(md)
    if should_deposit_hill!(md)
        cv_sim = evaluate_meta_dynamics_cvs(md, sys)
        height = tempering_height(md.tempering, md, cv_sim, md.memory.k)
        add_hill!(md.memory, cv_sim, height)
    end
    return nothing
end

# Advances bias's internal call counter and reports whether this call lands on the
# configured deposit_interval pace.
function should_deposit_hill!(md::MetaDynamicsBias)
    md.call_count[] += 1
    return (md.call_count[] % md.deposit_interval) == 0
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

# A single CV evaluates to a bare scalar (matching the shape ListHills/GridHills expect
# by default), while multiple CVs evaluate to a tuple, one value per CV.
reshape_meta_dynamics_cvs(cv_values::Tuple) = (length(cv_values) == 1) ? only(cv_values) : cv_values

function evaluate_meta_dynamics_cvs(md::MetaDynamicsBias, sys)
    coords_pbc = any(cv -> cv.correction == :pbc, md.cvs) ? unwrap_molecules(sys) : nothing
    cv_values = map(md.cvs) do cv
        coords = from_device(cv.correction == :pbc ? coords_pbc : sys.coords) #TODO compute on gpu
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

    # Forces are computed every simulation step regardless of simulator, so depositing here
    # (paced by deposit_interval) makes MetaDynamicsBias fully self-updating as a
    # general_inters entry -- no external logger is needed to drive add_hill!. Note a small
    # caveat: couplings that trigger a same-step force recomputation (e.g. some barostats)
    # will count as an extra call towards deposit_interval, a minor pacing perturbation
    # rather than a correctness issue.
    if should_deposit_hill!(md)
        height = tempering_height(md.tempering, md, cv_sim, md.memory.k)
        add_hill!(md.memory, cv_sim, height)
    end

    return fs
end
