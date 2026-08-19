# Metadynamics bias potential

export
    MetaDynamicsBias,
    AbstractMetaDynamicsMemory,
    ListHills,
    GridHills,
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
    ListHills(k, sigma, centers=Float64[])

Metadynamics memory that stores every deposited hill explicitly as a list of CV centers,
all sharing a common height `k` and width `sigma`.

The bias potential and its gradient are evaluated by summing the contribution of every
deposited hill:
```math
V(s) = \sum_{t' < t} k \exp\left(-\frac{|s - s(t')|^2}{2\sigma^2}\right)
```
Evaluation cost scales as O(n_hills) with the number of deposited hills, so this is best
suited to short or infrequently biased simulations. See [`GridHills`](@ref) for O(1)
evaluation independent of the number of hills.

`k`, `sigma` and the elements of `centers` can either be scalars, for biasing a single CV,
or same-length tuples, for biasing several CVs at once as used by [`MetaDynamicsBias`](@ref)
when given multiple CV descriptors. Tuples allow each CV dimension to carry its own units
(e.g. a distance in nm and a torsion angle in radians biased together).

# Arguments
- `k`: Height (weight) of each Gaussian hill. Must match system energy units.
- `sigma`: Width (standard deviation) of the Gaussians. Must match CV units.
- `centers`: Vector of CV values where hills have already been deposited.
"""
struct ListHills{K, R, V} <: AbstractMetaDynamicsMemory
    k::K
    sigma::R
    centers::V

    function ListHills(k::K, sigma::R, centers::V=DefaultFloat[]) where {K, R, V}
        validate_positive_finite.(sigma, "ListHills sigma")
        return new{K, R, V}(k, sigma, centers)
    end
end

function potential_energy(mem::ListHills, cv_sim; kwargs...)
    total = zero(mem.k)
    for center in mem.centers
        scaled_diff = (cv_sim .- center) ./ mem.sigma
        total += mem.k * exp(-0.5 * sum(abs2, scaled_diff))
    end
    return total
end

function bias_gradient(mem::ListHills, cv_sim)
    total = zero(mem.k) ./ mem.sigma
    for center in mem.centers
        diff = cv_sim .- center
        scaled_diff = diff ./ mem.sigma
        weight = mem.k * exp(-0.5 * sum(abs2, scaled_diff))
        total = total .+ weight .* (.-diff) ./ (mem.sigma .^ 2)
    end
    return total
end

function add_hill!(mem::ListHills, cv_value)
    push!(mem.centers, cv_value)
    return nothing
end

@doc raw"""
    GridHills(k, sigma, grid_min, grid_max, n_bins, cutoff=6)

Metadynamics memory that accumulates deposited hills directly onto a discretized 1D grid
of `n_bins` points spanning `[grid_min, grid_max]`.

Only supports a single CV; use [`ListHills`](@ref) for multiple CVs.

Each deposit adds the new hill's Gaussian contribution onto every grid point within
`cutoff` standard deviations of the hill center, an O(n_bins) operation performed once per
deposit. Evaluating the potential or gradient at simulation time is then O(1), using linear
interpolation between the two grid points bracketing the current CV value.

This trades some accuracy (linear interpolation, and hills deposited near or outside
`[grid_min, grid_max]` are truncated at the boundary) for evaluation speed independent of
the number of deposited hills, making it well suited to long biased simulations. See
[`ListHills`](@ref) for an exact but O(n_hills) alternative.

# Arguments
- `k`: Height (weight) of each Gaussian hill. Must match system energy units.
- `sigma`: Width (standard deviation) of the Gaussians. Must match CV units.
- `grid_min`, `grid_max`: The extent of the grid, in CV units.
- `n_bins`: Number of grid points, must be at least 2.
- `cutoff`: Number of standard deviations beyond which a deposited hill's contribution to
    the grid is ignored.
"""
mutable struct GridHills{K, R, G, E, T} <: AbstractMetaDynamicsMemory
    k::K
    sigma::R
    grid_min::G
    grid_max::G
    bin_width::G
    values::E
    cutoff::T
end

function GridHills(k::K, sigma::R, grid_min::G, grid_max::G, n_bins::Integer,
                   cutoff=6) where {K, R, G}
    validate_positive_finite(sigma, "GridHills sigma")
    if n_bins < 2
        throw(ArgumentError("GridHills n_bins must be at least 2, got $(n_bins)."))
    end
    if !(grid_max > grid_min)
        throw(ArgumentError("GridHills grid_max must be greater than grid_min."))
    end

    bin_width = (grid_max - grid_min) / (n_bins - 1)
    values = fill(zero(k), n_bins)
    return GridHills{K, R, G, typeof(values), typeof(cutoff)}(
        k, sigma, grid_min, grid_max, bin_width, values, cutoff,
    )
end

grid_cv_value(mem::GridHills, i::Integer) = mem.grid_min + (i - 1) * mem.bin_width

# Returns the indices of the two grid points bracketing cv_sim (clamped to the grid
# extent) and the fractional distance between them, for linear interpolation.
function grid_bracket(mem::GridHills, cv_sim)
    n = length(mem.values)
    clamped = clamp(cv_sim, mem.grid_min, mem.grid_max)
    frac_bin = ustrip((clamped - mem.grid_min) / mem.bin_width)
    i0 = clamp(floor(Int, frac_bin) + 1, 1, n - 1)
    i1 = i0 + 1
    frac = frac_bin - (i0 - 1)
    return i0, i1, frac
end

function potential_energy(mem::GridHills, cv_sim; kwargs...)
    i0, i1, frac = grid_bracket(mem, cv_sim)
    return mem.values[i0] + frac * (mem.values[i1] - mem.values[i0])
end

function bias_gradient(mem::GridHills, cv_sim)
    i0, i1, _ = grid_bracket(mem, cv_sim)
    return (mem.values[i1] - mem.values[i0]) / mem.bin_width
end

function add_hill!(mem::GridHills, cv_value)
    n = length(mem.values)
    cutoff = mem.cutoff * mem.sigma
    lo = clamp(cv_value - cutoff, mem.grid_min, mem.grid_max)
    hi = clamp(cv_value + cutoff, mem.grid_min, mem.grid_max)
    i_lo = clamp(floor(Int, ustrip((lo - mem.grid_min) / mem.bin_width)) + 1, 1, n)
    i_hi = clamp(ceil(Int, ustrip((hi - mem.grid_min) / mem.bin_width)) + 1, 1, n)
    for i in i_lo:i_hi
        d = grid_cv_value(mem, i) - cv_value
        mem.values[i] += mem.k * exp(-0.5 * (d / mem.sigma)^2)
    end
    return nothing
end


@doc raw"""
    MetaDynamicsBias(k, sigma, centers=Float64[])
    MetaDynamicsBias(cvs, k, sigma, centers=Float64[])
    MetaDynamicsBias(cvs, memory)

A history-dependent bias potential for standard Metadynamics.

The bias potential is a sum of Gaussians deposited at previously visited collective
variable (CV) values:
```math
V(\boldsymbol{s}) = \sum_{t' < t} k \exp\left(-\frac{|\boldsymbol{s} - \boldsymbol{s}(t')|^2}{2\sigma^2}\right)
```

How the history of deposited hills is stored and evaluated is controlled by the `memory`
argument, an [`AbstractMetaDynamicsMemory`](@ref):
- [`ListHills`](@ref): an explicit list of hill centers, summed from scratch at every
    evaluation. Exact, O(n_hills) per evaluation, and supports biasing multiple CVs at once.
- [`GridHills`](@ref): a discretized grid accumulating the sum of deposited hills.
    Approximate (linear interpolation), O(1) per evaluation, single CV only.

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

# Arguments
- `cvs`: A tuple of collective variable descriptors, evaluated in order to build the CV
    vector the bias acts on. Omit to evaluate the CV externally via [`BiasPotential`](@ref).
- `memory::AbstractMetaDynamicsMemory`: Storage and evaluation strategy for deposited hills.
- `deposit_interval::Integer=1`: Number of deposit-triggering calls (force evaluations when
    used as a `general_inters` entry, or calls to `add_hill!` otherwise) between actual
    deposits into `memory`. For example `deposit_interval=500` deposits every 500 calls.
"""
struct MetaDynamicsBias{C <: Tuple, M <: AbstractMetaDynamicsMemory}
    cvs::C
    memory::M
    deposit_interval::Int
    call_count::Base.RefValue{Int}

    function MetaDynamicsBias(cvs::C, memory::M;
                              deposit_interval::Integer=1) where {C <: Tuple, M <: AbstractMetaDynamicsMemory}
        # An empty cvs is a valid, intentional state: it means the CV is evaluated
        # externally via BiasPotential rather than by this struct's own calculator methods
        # (see check_meta_dynamics_cvs, called from those methods instead).
        if memory isa GridHills && length(cvs) > 1
            throw(ArgumentError(
                "GridHills memory only supports a single collective variable, got " *
                "$(length(cvs)) CVs. Use ListHills for multiple CVs."))
        end
        if deposit_interval < 1
            throw(ArgumentError("deposit_interval must be at least 1, got $(deposit_interval)."))
        end
        return new{C, M}(cvs, memory, Int(deposit_interval), Ref(0))
    end
end

MetaDynamicsBias(k, sigma, centers=DefaultFloat[]; deposit_interval::Integer=1) =
    MetaDynamicsBias((), ListHills(k, sigma, centers); deposit_interval=deposit_interval)
MetaDynamicsBias(cvs::Tuple, k, sigma, centers=DefaultFloat[]; deposit_interval::Integer=1) =
    MetaDynamicsBias(cvs, ListHills(k, sigma, centers); deposit_interval=deposit_interval)

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
every `deposit_interval`-th call actually updates the memory.
"""
function add_hill!(md::MetaDynamicsBias, cv_value)
    should_deposit_hill!(md) && add_hill!(md.memory, cv_value)
    return nothing
end

function add_hill!(md::MetaDynamicsBias, sys::System)
    check_meta_dynamics_cvs(md)
    should_deposit_hill!(md) && add_hill!(md.memory, evaluate_meta_dynamics_cvs(md, sys))
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
    should_deposit_hill!(md) && add_hill!(md.memory, cv_sim)

    return fs
end
