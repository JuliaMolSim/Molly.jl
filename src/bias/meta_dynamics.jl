# Metadynamics bias potential

export
    MetaDynamics,
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

Abstract type for the different ways a [`MetaDynamics`](@ref) can store deposited
Gaussian hills.

Subtypes must have the fields `cvs`, the tuple of CVs, and `k`, the default hill height,
and implement
`potential_energy(memory, cv)`, `bias_gradient(memory, cv)` and
`add_hill!(memory, cv, height)`, where `cv` is a scalar for one CV and a tuple for several.
Subtypes that support periodic CVs take the periods from [`cv_period`](@ref) in their
constructor. Built-in subtypes are [`ListHills`](@ref) and [`GridHills`](@ref).
"""
abstract type AbstractMetaDynamicsMemory end

# Empty hill-center vector with the float element type of sigma, which is in CV units (a
# tuple for several CVs), so deposits keep the user's precision and units. Integer sigma is
# promoted to float, since CV values are generally fractional.
empty_centers(sigma) = typeof(float.(sigma))[]

# Elements converted to float, also inside tuples and units
function float_elements(v)
    T = Base.promote_op(x -> float.(x), eltype(v))
    return T == eltype(v) ? v : map(x -> float.(x), v)
end

@doc raw"""
    ListHills(cvs, k, sigma, centers=typeof(float.(sigma))[], heights=fill(k, length(centers)))

Metadynamics memory for the CVs in the tuple `cvs`, storing every deposited hill
explicitly, summed at each evaluation:
```math
V(s) = \sum_{t' < t} h_{t'} \exp\left(-\frac{|s - s(t')|^2}{2\sigma^2}\right)
```
O(n_hills) per evaluation; see [`GridHills`](@ref) for an O(1) alternative.

`sigma` and the elements of `centers` are scalars for a single CV, or tuples with one
entry per CV for several. Hills on periodic CVs such as torsions wrap around (see
[`cv_period`](@ref)).

# Arguments
- `cvs`: Tuple of CV descriptors, e.g. [`CalcDist`](@ref).
- `k`: Default hill height, used unless [`add_hill!`](@ref) is given an explicit height.
- `sigma`: Gaussian width (standard deviation), in CV units.
- `centers`: CV values where hills have already been deposited. Defaults to an empty
    vector with the float element type of `sigma`; its element type sets the precision and
    units of every later deposit.
- `heights`: Deposited height per entry in `centers`; defaults to `k`.
"""
struct ListHills{C, K, R, V, H, P} <: AbstractMetaDynamicsMemory
    cvs::C
    k::K
    sigma::R
    centers::V
    heights::H
    period::P # Shaped like sigma, zero for a non-periodic CV

    # k, centers and heights are converted to float, since tempered heights and CV values
    #   are generally fractional
    function ListHills(cvs::C, k, sigma::R, centers=empty_centers(sigma),
                       heights=fill(float(k), length(centers))) where {C <: Tuple, R}
        validate_positive_finite.(sigma, "ListHills sigma")
        check_sigma_matches_cvs(cvs, sigma, "ListHills")
        if length(heights) != length(centers)
            throw(ArgumentError(
                "ListHills heights must be the same length as centers, got " *
                "$(length(heights)) and $(length(centers))."))
        end
        T = eltype(centers)
        shape_ok = sigma isa Tuple ?
            (T <: Tuple && isconcretetype(T) && fieldcount(T) == length(sigma)) : !(T <: Tuple)
        if !shape_ok
            throw(ArgumentError("ListHills centers of type $T don't match sigma $(sigma); " *
                                "give one sigma entry per CV and tuple centers for several CVs"))
        end
        centers_f = float_elements(centers)
        heights_f = float_elements(heights)
        k_float = float(k)
        period = cv_periods(cvs, sigma)
        return new{C, typeof(k_float), R, typeof(centers_f), typeof(heights_f),
                   typeof(period)}(cvs, k_float, sigma, centers_f, heights_f, period)
    end
end

function check_has_cvs(cvs::Tuple, label)
    if isempty(cvs)
        throw(ArgumentError("$(label) needs at least one CV, e.g. ListHills((cv,), k, sigma)"))
    end
    return nothing
end

# A scalar sigma for one CV, a tuple with one entry per CV for several
function check_sigma_matches_cvs(cvs::Tuple, sigma, label)
    check_has_cvs(cvs, label)
    ok = length(cvs) == 1 ? !(sigma isa Tuple) :
                            (sigma isa Tuple && length(sigma) == length(cvs))
    if !ok
        throw(ArgumentError("$(label) got sigma $(sigma) for $(length(cvs)) CVs; give a " *
                            "scalar sigma for one CV and one entry per CV for several"))
    end
    return nothing
end

# Period of each CV in the float type of the matching sigma, zero for a non-periodic CV;
#   a scalar for one CV, a tuple for several
function cv_periods(cvs::Tuple, sigma)
    periods = map(cvs, as_dims_tuple(sigma)) do cv, s
        p = cv_period(cv)
        iszero(p) ? zero(float(s)) : convert(typeof(float(s)), p)
    end
    return sigma isa Tuple ? periods : only(periods)
end

function potential_energy(mem::ListHills, cv_sim; kwargs...)
    total = zero(mem.k)
    for (center, height) in zip(mem.centers, mem.heights)
        scaled_diff = periodic_displacement.(cv_sim .- center, mem.period) ./ mem.sigma
        total += height * exp(-sum(abs2, scaled_diff) / 2)
    end
    return total
end

function bias_gradient(mem::ListHills, cv_sim)
    total = zero(mem.k) ./ mem.sigma
    for (center, height) in zip(mem.centers, mem.heights)
        diff = periodic_displacement.(cv_sim .- center, mem.period)
        scaled_diff = diff ./ mem.sigma
        weight = height * exp(-sum(abs2, scaled_diff) / 2)
        total = total .+ weight .* (.-diff) ./ (mem.sigma .^ 2)
    end
    return total
end

function add_hill!(mem::ListHills, cv_value, height=mem.k)
    # Converted first, so a failing conversion leaves centers and heights the same length
    center, h = convert(eltype(mem.centers), cv_value), convert(eltype(mem.heights), height)
    push!(mem.centers, center)
    push!(mem.heights, h)
    return nothing
end

@doc raw"""
    GridHills(cvs, k, sigma, grid_min, grid_max, n_bins, cutoff=6; out_of_grid_error=true)

Metadynamics memory for the CVs in the tuple `cvs`, accumulating deposited hills onto a
discretized grid spanning `[grid_min, grid_max]`, evaluated by (N-linear) interpolation.
O(1) per evaluation regardless of the number of deposited hills; see [`ListHills`](@ref)
for an exact but O(n_hills) alternative.

For a single CV, `sigma`, `grid_min` and `grid_max` are scalars and `n_bins` an Integer,
giving a 1D grid. For several CVs, pass same-length tuples (one entry per CV in `cvs`) and
`n_bins` as an Integer or a matching tuple, giving a dense N-dimensional grid.
Cost scales as roughly O(prod(n_bins)) memory and O(cutoff^N) per deposit, so this is only
practical for a handful of CVs (2-3); use [`ListHills`](@ref) for more.

# Arguments
- `cvs`: Tuple of CV descriptors, e.g. [`CalcDist`](@ref).
- `k`: Default hill height, used unless [`add_hill!`](@ref) is given an explicit height.
- `sigma`: Gaussian width (standard deviation), scalar or one per dimension.
- `grid_min`, `grid_max`: Extent of the grid, scalar or one per dimension.
- `n_bins`: Number of grid points, at least 2 in every dimension; a single Integer or one
    per dimension.
- `cutoff`: Standard deviations beyond which a deposited hill's contribution is ignored.
- `out_of_grid_error::Bool=true`: Whether a CV value outside `[grid_min, grid_max]` throws
    an error. If `false`, a warning is given once and the bias is held constant at its
    edge value outside the grid, so it exerts no force there.

The grid of a periodic CV such as a torsion (see [`cv_period`](@ref)) must span exactly
one period, e.g. `-π` to `π`, with `grid_max` the same point as `grid_min`: its `n_bins`
points are then spaced by `period / n_bins`, hills and interpolation wrap around, and CV
values outside `[grid_min, grid_max]` are wrapped into the grid.
"""
struct GridHills{C, K, R, G, E, T} <: AbstractMetaDynamicsMemory
    cvs::C
    k::K
    sigma::R
    grid_min::G
    grid_max::G
    bin_width::G
    values::E
    cutoff::T
    out_of_grid_error::Bool
    period::G # One entry per dimension, zero for a non-periodic one
    warned_out_of_grid::Base.RefValue{Bool} # Whether this grid has warned about a CV outside it
end

is_periodic(mem::GridHills, d::Integer) = !iszero(mem.period[d])

# Wraps a scalar into a length-1 tuple, so a single N-dimensional implementation below
# covers both the single- and multi-CV cases.
as_dims_tuple(x::Tuple) = x
as_dims_tuple(x::AbstractVector) = Tuple(x)
as_dims_tuple(x) = (x,)

function GridHills(cvs::Tuple, k, sigma, grid_min, grid_max, n_bins, cutoff=6;
                   out_of_grid_error::Bool=true)
    k = float(k) # Tempered heights are generally fractional
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
    check_has_cvs(cvs, "GridHills")
    if length(cvs) != n_dims
        throw(ArgumentError("GridHills has $n_dims dimension(s) but $(length(cvs)) CVs " *
                            "were given; give one sigma, grid_min and grid_max entry per CV"))
    end

    # Periods in the float type of the grid, zero for a non-periodic dimension
    period_t = cv_periods(cvs, grid_min_t)
    for (d, p) in enumerate(period_t)
        if !iszero(p) && !isapprox(grid_max_t[d] - grid_min_t[d], p)
            throw(ArgumentError(
                "GridHills grid of the periodic CV $d must span exactly one period, e.g. " *
                "-π to π for a torsion, got [$(grid_min_t[d]), $(grid_max_t[d])] for " *
                "period $p."))
        end
    end
    # A periodic dimension doesn't store grid_max, which is the same point as grid_min
    bin_width_t = map(grid_min_t, grid_max_t, n_bins_t, period_t) do gmin, gmax, n, p
        (gmax - gmin) / (iszero(p) ? n - 1 : n)
    end
    values = fill(zero(k), n_bins_t)
    return GridHills{typeof(cvs), typeof(k), typeof(sigma_t), typeof(grid_min_t),
                     typeof(values), typeof(cutoff)}(
        cvs, k, sigma_t, grid_min_t, grid_max_t, bin_width_t, values, cutoff,
        out_of_grid_error, period_t, Ref(false),
    )
end

# Periodic dimensions wrap around, so every CV value is inside them
in_grid(mem::GridHills, d::Integer, x) =
    is_periodic(mem, d) || mem.grid_min[d] <= x <= mem.grid_max[d]

# Errors, or warns once per grid when out_of_grid_error is false, if cv_sim lies outside
#   the grid
function check_in_grid(mem::GridHills, cv_sim)
    for d in eachindex(mem.grid_min)
        in_grid(mem, d, cv_sim[d]) && continue
        !mem.out_of_grid_error && mem.warned_out_of_grid[] && continue
        msg = "GridHills CV value $(cv_sim[d]) in dimension $d is outside the grid " *
              "[$(mem.grid_min[d]), $(mem.grid_max[d])]"
        if mem.out_of_grid_error
            throw(ArgumentError(msg * "; widen the grid, add walls to confine the " *
                                "CV, or construct the GridHills with " *
                                "out_of_grid_error=false to hold the bias constant " *
                                "outside the grid"))
        end
        @warn msg * ", the bias is held constant outside the grid"
        mem.warned_out_of_grid[] = true
    end
    return nothing
end

grid_axis_value(mem::GridHills, d::Integer, i::Integer) = mem.grid_min[d] + (i - 1) * mem.bin_width[d]

# Per-dimension bracket indices and fractional distance, for N-linear interpolation.
# cv_sim[d] on a scalar cv_sim (1D case) just returns cv_sim itself. A periodic dimension
# wraps the CV into the grid, and its last point is bracketed with the first one.
function grid_bracket_dims(mem::GridHills, cv_sim)
    n_dims = length(mem.grid_min)
    return ntuple(n_dims) do d
        n = size(mem.values, d)
        if is_periodic(mem, d)
            offset = mod(cv_sim[d] - mem.grid_min[d], mem.period[d])
            frac_bin = ustrip(offset / mem.bin_width[d])
            i0 = clamp(floor(Int, frac_bin) + 1, 1, n)
            (i0, mod1(i0 + 1, n), frac_bin - (i0 - 1))
        else
            clamped = clamp(cv_sim[d], mem.grid_min[d], mem.grid_max[d])
            frac_bin = ustrip((clamped - mem.grid_min[d]) / mem.bin_width[d])
            i0 = clamp(floor(Int, frac_bin) + 1, 1, n - 1)
            (i0, i0 + 1, frac_bin - (i0 - 1))
        end
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
        if !in_grid(mem, d, cv_sim[d])
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

# Grid indices within the cutoff of a hill at cv_value[d]. A periodic dimension isn't
#   clamped, its indices are wrapped by the caller, and covers each grid point at most once.
function hill_index_range(mem::GridHills, d::Integer, cv_value)
    n = size(mem.values, d)
    cutoff_d = mem.cutoff * mem.sigma[d]
    lo, hi = cv_value[d] - cutoff_d, cv_value[d] + cutoff_d
    if !is_periodic(mem, d)
        lo = clamp(lo, mem.grid_min[d], mem.grid_max[d])
        hi = clamp(hi, mem.grid_min[d], mem.grid_max[d])
    end
    i_lo = floor(Int, ustrip((lo - mem.grid_min[d]) / mem.bin_width[d])) + 1
    i_hi = ceil(Int, ustrip((hi - mem.grid_min[d]) / mem.bin_width[d])) + 1
    if is_periodic(mem, d)
        return i_hi - i_lo + 1 >= n ? (1:n) : (i_lo:i_hi)
    else
        return clamp(i_lo, 1, n):clamp(i_hi, 1, n)
    end
end

function add_hill!(mem::GridHills, cv_value, height=mem.k)
    check_in_grid(mem, cv_value)
    n_dims = length(mem.grid_min)
    ranges = ntuple(d -> hill_index_range(mem, d, cv_value), n_dims)
    for idx in CartesianIndices(ranges)
        grid_idx = ntuple(d -> is_periodic(mem, d) ? mod1(idx[d], size(mem.values, d)) : idx[d],
                          n_dims)
        scaled_sq = sum(1:n_dims) do d
            diff = periodic_displacement(grid_axis_value(mem, d, grid_idx[d]) - cv_value[d],
                                         mem.period[d])
            (diff / mem.sigma[d])^2
        end
        mem.values[grid_idx...] += height * exp(-scaled_sq / 2)
    end
    return nothing
end


"""
    AbstractTempering

Controls how the height of each newly deposited hill is scaled before being added to a
[`MetaDynamics`](@ref)'s memory.

Subtypes must implement `tempering_height(tempering, bias::MetaDynamics, cv_sim,
base_height)`, returning the height to actually deposit; `bias` gives access to
`bias.memory`, `bias.n_deposits[]`/`bias.deposit_interval` and `cv_sim`.

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

"""
    tempering_height(tempering, bias, cv_sim, base_height)

Height of the hill `bias` deposits at the CV value `cv_sim`, given its base height
`base_height`, as scaled by `tempering` (an [`AbstractTempering`](@ref)).
"""
function tempering_height end

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
    MetaDynamics(memory; deposit_interval=1, tempering=NoTempering())

A history-dependent bias potential for Metadynamics: a sum of Gaussians deposited at
previously visited collective variable (CV) values.
```math
V(\boldsymbol{s}) = \sum_{t' < t} h_{t'} \exp\left(-\frac{|\boldsymbol{s} - \boldsymbol{s}(t')|^2}{2\sigma^2}\right)
```

The `memory` argument (an [`AbstractMetaDynamicsMemory`](@ref)) holds the CVs and controls
how hills are stored: [`ListHills`](@ref) (exact, O(n_hills)) or [`GridHills`](@ref) (O(1),
grid-based).

With the CVs in the memory, e.g. `MetaDynamics(ListHills((cv,), k, sigma))`, the bias
is an AtomsCalculators.jl calculator usable directly as a `general_inters` entry. Each CV
is evaluated on the coordinates its own `correction` asks for, and hills on periodic CVs
such as torsions wrap around (see [`cv_period`](@ref)). Hills are deposited into the memory
passed in, which is also `bias.memory`, so biases built from one memory are walkers sharing
their hills. Deposits into a memory aren't thread-safe, so walkers sharing a memory must
run one after the other, not in parallel.

Forces are computed every simulation step regardless of simulator, and a hill is deposited
on the first step of each window of `deposit_interval` steps -- no external logger is
needed. The first hill is on the start configuration, at step 0, or at step 1 for
integrators such as [`Langevin`](@ref) that first compute forces there. Recomputing forces
within a step (e.g. by a logger or coupler) adds no hill and a multiple time step
integrator deposits once per outer step. As for loggers, the step numbers restart with
every `simulate!` call, so pass `init_step` to continue the schedule of a run split into
chunks. A force call at step 0 between runs, e.g. `forces(sys)`, opens a new window, so
the next run's first force call deposits a hill too. Energy minimizers also count as
steps, so minimize without the bias. The number of hills deposited so far is
`bias.n_deposits[]` and the step of the last automatic deposit is
`bias.last_deposit_step[]`.

[`add_hill!`](@ref) deposits a hill manually.

`tempering` (an [`AbstractTempering`](@ref)) scales each deposited hill's height; the
default [`NoTempering`](@ref) always deposits the full base height.

# Arguments
- `memory::AbstractMetaDynamicsMemory`: The CVs, and the storage and evaluation strategy
    for deposited hills.
- `deposit_interval::Integer=1`: Number of simulation steps between deposits into
    `memory`, starting with the start configuration. [`add_hill!`](@ref) calls and `forces!` calls without a
    step number use a separate call counter, depositing on every `deposit_interval`-th
    such call without affecting the simulation schedule.
- `tempering::AbstractTempering=NoTempering()`: Scales the height of each deposited hill.
"""
struct MetaDynamics{M <: AbstractMetaDynamicsMemory, TP <: AbstractTempering}
    memory::M # Holds the CVs, memory.cvs
    deposit_interval::Int
    call_count::Base.RefValue{Int}
    last_step_n::Base.RefValue{Int}
    last_deposit_step::Base.RefValue{Int}
    n_deposits::Base.RefValue{Int}
    tempering::TP

    # The memory checks its CVs against its own dimensions; the empty check guards custom
    #   memories
    function MetaDynamics(memory::M; deposit_interval::Integer=1,
                          tempering::TP=NoTempering()) where {
                              M <: AbstractMetaDynamicsMemory, TP <: AbstractTempering}
        check_has_cvs(memory.cvs, "MetaDynamics")
        if deposit_interval < 1
            throw(ArgumentError("deposit_interval must be at least 1, got $(deposit_interval)."))
        end
        return new{M, TP}(memory, Int(deposit_interval), Ref(0), Ref(-1), Ref(-1), Ref(0),
                          tempering)
    end
end

"""
    add_hill!(bias::MetaDynamics, cv_value)
    add_hill!(bias::MetaDynamics, sys)

Deposit a new Gaussian hill.

Called automatically from `forces!` during a simulation -- there is usually no need to call
this directly.

`cv_value` matches the shape `bias` was constructed with (scalar or tuple); the `sys` form
evaluates the CVs of `bias.memory` itself. Calls are counted by a manual call counter, separate from
the step schedule of automatic deposits during a simulation; only every
`deposit_interval`-th call actually updates the memory, at height
`tempering_height(bias.tempering, bias, cv_value, bias.memory.k)`, where `cv_value` is the
deposited CV value (given, or evaluated from `sys`).
"""
function add_hill!(md::MetaDynamics, cv_value)
    if !bias_all_finite(cv_value)
        throw(ArgumentError("add_hill! got a non-finite CV value $(cv_value)"))
    end
    if should_deposit_hill!(md)
        deposit_hill!(md, cv_value)
    end
    return nothing
end

function add_hill!(md::MetaDynamics, sys::System)
    cv_sim = check_meta_dynamics_finite(evaluate_meta_dynamics_cvs(md, sys), "collective variable")
    if should_deposit_hill!(md)
        deposit_hill!(md, cv_sim)
    end
    return nothing
end

function deposit_hill!(md::MetaDynamics, cv_sim)
    height = tempering_height(md.tempering, md, cv_sim, md.memory.k)
    add_hill!(md.memory, cv_sim, height)
    md.n_deposits[] += 1
    return nothing
end

# Manual pacing, for add_hill! and forces! calls without a step number: every
#   deposit_interval-th call deposits. Independent of the step-paced schedule below.
function should_deposit_hill!(md::MetaDynamics, ::Nothing=nothing)
    md.call_count[] += 1
    return (md.call_count[] % md.deposit_interval) == 0
end

# Step pacing: deposits on the first force call in each window of deposit_interval steps,
#   so the start configuration is deposited at step 0, or at step 1 by integrators that
#   first compute forces there (e.g. Langevin), whether or not a logger computed forces at
#   step 0. Recomputing forces within a step (loggers, couplers, the initial virial, the
#   inner steps of a multiple time step integrator) stays in the window and adds no hill.
#   As for loggers, a run continues the schedule of the previous one only with init_step;
#   a force call at step 0 between runs opens a new window, so the next run deposits too.
function should_deposit_hill!(md::MetaDynamics, step_n::Integer)
    last_window = fld(md.last_step_n[], md.deposit_interval)
    md.last_step_n[] = step_n
    if fld(step_n, md.deposit_interval) == last_window
        return false
    end
    md.last_deposit_step[] = step_n
    return true
end

# Errors on non-finite values, which must never be deposited or applied as forces
function check_meta_dynamics_finite(value, label::AbstractString, cv_sim=nothing)
    bias_all_finite(value) && return value
    msg = "MetaDynamics produced non-finite $(label)"
    if !isnothing(cv_sim)
        msg *= ", cv_sim=$(cv_sim)"
    end
    error(msg)
end

# Single CV -> bare scalar; multiple CVs -> tuple, matching what ListHills/GridHills expect.
reshape_meta_dynamics_cvs(cv_values::Tuple) = (length(cv_values) == 1) ? only(cv_values) : cv_values

# Host copies of the system made once per call: the coordinates each CV is evaluated on
#   (unwrapped for correction=:pbc, as stored for :wrap), the atoms and the velocities
function meta_dynamics_host_state(md::MetaDynamics, sys)
    cvs = md.memory.cvs
    coords = any(cv -> cv.correction != :pbc, cvs) ? from_device(sys.coords) : nothing
    coords_pbc = any(cv -> cv.correction == :pbc, cvs) ? from_device(unwrap_molecules(sys)) :
                                                          nothing
    cv_coords = map(cv -> cv.correction == :pbc ? coords_pbc : coords, cvs)
    return cv_coords, from_device(sys.atoms), from_device(sys.velocities)
end

function evaluate_meta_dynamics_cvs(md::MetaDynamics, sys)
    cv_coords, atoms, velocities = meta_dynamics_host_state(md, sys)
    cv_values = map(md.memory.cvs, cv_coords) do cv, coords
        calculate_cv(cv, coords, atoms, sys.boundary, velocities)
    end
    return reshape_meta_dynamics_cvs(cv_values)
end

function AtomsCalculators.potential_energy(sys, md::MetaDynamics; kwargs...)
    cv_sim = check_meta_dynamics_finite(evaluate_meta_dynamics_cvs(md, sys),
                                        "collective variable")
    pe = potential_energy(md.memory, cv_sim)
    return check_meta_dynamics_finite(pe, "potential energy", cv_sim)
end

function AtomsCalculators.forces!(
    fs, sys, md::MetaDynamics;
    needs_vir::Bool = false,
    buffers = nothing,
    step_n = nothing,
    kwargs...
)
    cv_coords, atoms, velocities = meta_dynamics_host_state(md, sys)

    # Each CV on the coordinates its own correction asks for
    per_cv = map(md.memory.cvs, cv_coords) do cv, coords
        d_coords, cv_val = cv_gradient(cv, coords, atoms, sys.boundary, velocities)
        (cv=cv, coords=coords, atoms=atoms, d_coords=d_coords, cv_val=cv_val)
    end

    cv_sim = reshape_meta_dynamics_cvs(map(x -> x.cv_val, per_cv))
    check_meta_dynamics_finite(cv_sim, "collective variable")
    d_bias = check_meta_dynamics_finite(bias_gradient(md.memory, cv_sim), "bias gradient", cv_sim)

    fs_svecs = map(per_cv, ntuple(identity, length(per_cv))) do x, i
        check_meta_dynamics_finite(x.d_coords, "CV gradient", cv_sim)
        check_meta_dynamics_finite(d_bias[i] .* x.d_coords, "bias force", cv_sim)
    end
    for (x, fs_svec) in zip(per_cv, fs_svecs)
        if needs_vir && x.cv.has_virial
            calculate_virial!(buffers.virial, x.cv, x.coords, -fs_svec, x.atoms, sys.boundary)
        end
        fs .-= to_device(fs_svec, typeof(fs))
    end

    # Self-paced deposit, by simulation step when called from a simulator, after every
    #   finite check so non-finite values are never deposited
    if should_deposit_hill!(md, step_n)
        deposit_hill!(md, cv_sim)
    end

    return fs
end
