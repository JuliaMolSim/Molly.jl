# Metadynamics bias potential

export
    Metadynamics,
    AbstractMetadynamicsMemory,
    ListHills,
    GridHills,
    AbstractTempering,
    NoTempering,
    WellTempered,
    add_hill!,
    free_energy,
    MetadynamicsLogger,
    metadynamics_weights,
    metadynamics_log_weights,
    read_metadynamics_logger

@public tempering_height, free_energy_scale, default_temperature, bias_points_cache,
        bias_at_points

function validate_positive_finite(value, label::AbstractString)
    if !isfinite(ustrip(value)) || value <= zero(value)
        throw(ArgumentError("$(label) must be finite and positive, got $(value)."))
    end
    return value
end

"""
    AbstractMetadynamicsMemory

Abstract type for the different ways a [`Metadynamics`](@ref) can store deposited
Gaussian hills.

Subtypes must have the fields `cvs`, the tuple of CVs, and `height`, the default hill height,
and implement
`potential_energy(memory, cv_value)`, `bias_gradient(memory, cv_value)` and
`add_hill!(memory, cv_value, height)`, where `cv_value` is a scalar for one CV and a tuple
for several.
Subtypes that support periodic CVs take the periods from [`cv_period`](@ref) in their
constructor. Optionally, they implement [`bias_points_cache`](@ref) and
[`bias_at_points`](@ref) for a cheaper offset of a [`MetadynamicsLogger`](@ref).
Built-in subtypes are [`ListHills`](@ref) and [`GridHills`](@ref).
"""
abstract type AbstractMetadynamicsMemory end

# CV-shaped values: memories hold sigma and period as tuples, also for one CV; users
#   pass and get a scalar for one CV and a tuple for several
to_dims(x::Tuple) = x
to_dims(x::AbstractVector) = Tuple(x)
to_dims(x) = (x,)
from_dims(t::Tuple) = length(t) == 1 ? only(t) : t

# Empty hill-center vector with the float element type of sigma, which is in CV units (a
# tuple for several CVs), so deposits keep the user's precision and units. Integer sigma is
# promoted to float, since CV values are generally fractional.
empty_centers(sigma) = typeof(from_dims(float.(to_dims(sigma))))[]

# Elements converted to float, also inside tuples and units
function float_elements(v)
    T = Base.promote_op(x -> float.(x), eltype(v))
    return T == eltype(v) ? v : map(x -> float.(x), v)
end

@doc raw"""
    ListHills(cvs, height, sigma, centers=typeof(float.(sigma))[],
              heights=fill(height, length(centers)))

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
- `cvs`: Tuple of CV descriptors, e.g. [`CalcDist`](@ref); a single CV may be passed
    without the tuple.
- `height`: Default hill height, used unless [`add_hill!`](@ref) is given an explicit height.
- `sigma`: Gaussian width (standard deviation), in CV units.
- `centers`: CV values where hills have already been deposited. Defaults to an empty
    vector with the float element type of `sigma`; its element type sets the precision and
    units of every later deposit.
- `heights`: Deposited height per entry in `centers`; defaults to `height`.
"""
struct ListHills{C, K, R, V, H, P} <: AbstractMetadynamicsMemory
    cvs::C
    height::K
    sigma::R
    centers::V
    heights::H
    period::P # One entry per CV, zero for a non-periodic CV

    # height, centers and heights are converted to float, since tempered heights and CV
    #   values are generally fractional
    function ListHills(cvs::C, height, sigma, centers=empty_centers(sigma),
                       heights=fill(float(height), length(centers))) where {C <: Tuple}
        sigma_t = to_dims(sigma)
        validate_positive_finite.(sigma_t, "ListHills sigma")
        check_sigma_matches_cvs(cvs, sigma_t, "ListHills")
        if length(heights) != length(centers)
            throw(ArgumentError(
                "ListHills heights must be the same length as centers, got " *
                "$(length(heights)) and $(length(centers))."))
        end
        T = eltype(centers)
        shape_ok = length(cvs) == 1 ? !(T <: Tuple) :
            (T <: Tuple && isconcretetype(T) && fieldcount(T) == length(cvs))
        if !shape_ok
            throw(ArgumentError("ListHills centers of type $T don't match $(length(cvs)) " *
                                "CV(s); give scalar centers for one CV and tuples for several"))
        end
        centers_f = float_elements(centers)
        heights_f = float_elements(heights)
        height_f = float(height)
        period = cv_periods(cvs, sigma_t)
        return new{C, typeof(height_f), typeof(sigma_t), typeof(centers_f), typeof(heights_f),
                   typeof(period)}(cvs, height_f, sigma_t, centers_f, heights_f, period)
    end
end

# A bare CV is wrapped in a tuple; a tuple reaching here has the wrong arguments, so it
#   isn't wrapped again
function ListHills(cv, args...)
    cv isa Tuple && throw(MethodError(ListHills, (cv, args...)))
    return ListHills(cv_tuple(cv), args...)
end

# A bare CV as a 1-tuple, and a vector of CVs as a tuple
cv_tuple(cvs::AbstractVector) = to_dims(cvs)
cv_tuple(cv) = (cv,)

function check_has_cvs(cvs::Tuple, label)
    if isempty(cvs)
        throw(ArgumentError("$(label) needs at least one CV, e.g. ListHills((cv,), height, sigma)"))
    end
    return nothing
end

# One sigma entry per CV
function check_sigma_matches_cvs(cvs::Tuple, sigma::Tuple, label)
    check_has_cvs(cvs, label)
    if length(sigma) != length(cvs)
        throw(ArgumentError("$(label) got sigma $(sigma) for $(length(cvs)) CVs; give one " *
                            "sigma entry per CV"))
    end
    return nothing
end

# Period of each CV in the float type of the matching entry of the tuple sigma, zero for a
#   non-periodic CV
cv_periods(cvs::Tuple, sigma::Tuple) = map(cvs, sigma) do cv, s
    p = cv_period(cv)
    iszero(p) ? zero(float(s)) : convert(typeof(float(s)), p)
end

# Value of one hill at cv_sim
function hill_value(mem::ListHills, cv_sim, center, height)
    scaled_diff = periodic_displacement.(to_dims(cv_sim) .- to_dims(center), mem.period) ./
                  mem.sigma
    return height * exp(-sum(abs2, scaled_diff) / 2)
end

function potential_energy(mem::ListHills, cv_sim; kwargs...)
    total = zero(mem.height)
    for (center, height) in zip(mem.centers, mem.heights)
        total += hill_value(mem, cv_sim, center, height)
    end
    return total
end

function bias_gradient(mem::ListHills, cv_sim)
    total = zero(mem.height) ./ mem.sigma
    for (center, height) in zip(mem.centers, mem.heights)
        diff = periodic_displacement.(to_dims(cv_sim) .- to_dims(center), mem.period)
        scaled_diff = diff ./ mem.sigma
        weight = height * exp(-sum(abs2, scaled_diff) / 2)
        total = total .+ weight .* (.-diff) ./ (mem.sigma .^ 2)
    end
    return from_dims(total)
end

function add_hill!(mem::ListHills, cv_value, height=mem.height)
    # Converted first, so a failing conversion leaves centers and heights the same length
    center, h = convert(eltype(mem.centers), cv_value), convert(eltype(mem.heights), height)
    push!(mem.centers, center)
    push!(mem.heights, h)
    return nothing
end

@doc raw"""
    GridHills(cvs, height, sigma, grid_min, grid_max, n_points; cutoff=6,
              out_of_grid_error=true)

Metadynamics memory for the CVs in the tuple `cvs`, accumulating deposited hills onto a
discretized grid spanning `[grid_min, grid_max]`, evaluated by (N-linear) interpolation.
O(1) per evaluation regardless of the number of deposited hills; see [`ListHills`](@ref)
for an exact but O(n_hills) alternative.

For a single CV, `sigma`, `grid_min` and `grid_max` are scalars and `n_points` an Integer,
giving a 1D grid. For several CVs, pass same-length tuples (one entry per CV in `cvs`) and
`n_points` as an Integer or a matching tuple, giving a dense N-dimensional grid.
Cost scales as roughly O(prod(n_points)) memory and O(cutoff^N) per deposit, so this is only
practical for a handful of CVs (2-3); use [`ListHills`](@ref) for more.

# Arguments
- `cvs`: Tuple of CV descriptors, e.g. [`CalcDist`](@ref); a single CV may be passed
    without the tuple.
- `height`: Default hill height, used unless [`add_hill!`](@ref) is given an explicit height.
- `sigma`: Gaussian width (standard deviation), scalar or one per dimension.
- `grid_min`, `grid_max`: Extent of the grid, scalar or one per dimension.
- `n_points`: Number of grid points, at least 2 in every dimension; a single Integer or one
    per dimension.
- `cutoff=6`: Standard deviations beyond which a deposited hill's contribution is ignored.
- `out_of_grid_error::Bool=true`: Whether a CV value outside `[grid_min, grid_max]` throws
    an error. If `false`, a warning is given once and the bias is held constant at its
    edge value outside the grid, so it exerts no force there.

The grid of a periodic CV such as a torsion (see [`cv_period`](@ref)) must span exactly
one period, e.g. `-π` to `π`, with `grid_max` the same point as `grid_min`: its `n_points`
points are then spaced by `period / n_points`, hills and interpolation wrap around, and CV
values outside `[grid_min, grid_max]` are wrapped into the grid.
"""
mutable struct GridHills{C, K, R, G, E, T} <: AbstractMetadynamicsMemory
    const cvs::C
    const height::K
    const sigma::R
    const grid_min::G
    const grid_max::G
    const bin_width::G
    const values::E
    const cutoff::T
    const out_of_grid_error::Bool
    const period::G # One entry per dimension, zero for a non-periodic one
    warned_out_of_grid::Bool # Whether this grid has warned about a CV outside it

    # The only constructor of the struct: checks the per-dimension tuples and derives the
    #   grid from them, so no grid is built unchecked
    function GridHills(cvs::Tuple, height, sigma::Tuple, grid_min::Tuple, grid_max::Tuple,
                       n_points::Tuple; cutoff=6, out_of_grid_error::Bool=true)
        height = float(height) # Tempered heights are generally fractional
        # Float bounds of a common type per dimension, so bin_width can share their type
        grid_min_t, grid_max_t = float.(grid_min), float.(grid_max)
        if length(grid_min_t) == length(grid_max_t)
            bounds_t = map(promote, grid_min_t, grid_max_t)
            grid_min_t, grid_max_t = first.(bounds_t), last.(bounds_t)
        end
        n_dims = length(grid_min_t)
        if length(sigma) != n_dims || length(grid_max_t) != n_dims || length(n_points) != n_dims
            throw(ArgumentError(
                "GridHills sigma, grid_min, grid_max and n_points must all cover the same " *
                "number of dimensions, got $(length(sigma)), $(n_dims), " *
                "$(length(grid_max_t)) and $(length(n_points))."))
        end
        validate_positive_finite.(sigma, "GridHills sigma")
        if any(nb -> nb < 2, n_points)
            throw(ArgumentError("GridHills n_points must be at least 2 in every dimension, " *
                                "got $(n_points)."))
        end
        if any(gmax <= gmin for (gmin, gmax) in zip(grid_min_t, grid_max_t))
            throw(ArgumentError("GridHills grid_max must be greater than grid_min in every " *
                                "dimension."))
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
                    "GridHills grid of the periodic CV $d must span exactly one period, " *
                    "e.g. -π to π for a torsion, got [$(grid_min_t[d]), $(grid_max_t[d])] " *
                    "for period $p."))
            end
        end
        # A periodic dimension doesn't store grid_max, which is the same point as grid_min
        bin_width_t = map(grid_min_t, grid_max_t, n_points, period_t) do gmin, gmax, n, p
            (gmax - gmin) / (iszero(p) ? n - 1 : n)
        end
        values = fill(zero(height), n_points)
        return new{typeof(cvs), typeof(height), typeof(sigma), typeof(grid_min_t),
                   typeof(values), typeof(cutoff)}(
            cvs, height, sigma, grid_min_t, grid_max_t, bin_width_t, values, cutoff,
            out_of_grid_error, period_t, false,
        )
    end
end

# Scalars for one CV, and an Integer n_points for every dimension, as tuples
function GridHills(cvs::Tuple, height, sigma, grid_min, grid_max, n_points; kwargs...)
    grid_min_t = to_dims(grid_min)
    n_points_t = n_points isa Integer ? ntuple(_ -> n_points, length(grid_min_t)) :
                                        to_dims(n_points)
    return GridHills(cvs, height, to_dims(sigma), grid_min_t, to_dims(grid_max), n_points_t;
                     kwargs...)
end

# A bare CV is wrapped in a tuple, as for ListHills
function GridHills(cv, args...; kwargs...)
    cv isa Tuple && throw(MethodError(GridHills, (cv, args...)))
    return GridHills(cv_tuple(cv), args...; kwargs...)
end

is_periodic(mem::GridHills, d::Integer) = !iszero(mem.period[d])

# Periodic dimensions wrap around, so every CV value is inside them
in_grid(mem::GridHills, d::Integer, x) =
    is_periodic(mem, d) || mem.grid_min[d] <= x <= mem.grid_max[d]

# Errors, or warns once per grid when out_of_grid_error is false, if cv_sim lies outside
#   the grid
function check_in_grid(mem::GridHills, cv_sim)
    for d in eachindex(mem.grid_min)
        in_grid(mem, d, cv_sim[d]) && continue
        !mem.out_of_grid_error && mem.warned_out_of_grid && continue
        msg = "GridHills CV value $(cv_sim[d]) in dimension $d is outside the grid " *
              "[$(mem.grid_min[d]), $(mem.grid_max[d])]"
        if mem.out_of_grid_error
            throw(ArgumentError(msg * "; widen the grid, add walls to confine the " *
                                "CV, or construct the GridHills with " *
                                "out_of_grid_error=false to hold the bias constant " *
                                "outside the grid"))
        end
        @warn msg * ", the bias is held constant outside the grid"
        mem.warned_out_of_grid = true
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

# The 2^N corners of the cell bracketing a CV value
grid_corners(dims) = CartesianIndices(ntuple(_ -> 2, length(dims)))

# Grid value at one corner of the bracketing cell, times the N-linear weight of every
#   dimension except `skip` (0 for none)
function grid_corner_value(mem::GridHills, dims, corner, skip=0)
    n_dims = length(dims)
    idx = ntuple(d -> corner[d] == 1 ? dims[d][1] : dims[d][2], n_dims)
    weight = prod(1:n_dims) do d
        d == skip ? one(dims[d][3]) : (corner[d] == 1 ? 1 - dims[d][3] : dims[d][3])
    end
    return weight * mem.values[idx...]
end

function potential_energy(mem::GridHills, cv_sim; kwargs...)
    check_in_grid(mem, cv_sim)
    dims = grid_bracket_dims(mem, cv_sim)
    return sum(c -> grid_corner_value(mem, dims, c), grid_corners(dims); init=zero(mem.height))
end

# Partial derivative of the N-linear interpolant along each dimension (chain rule through
# frac_d = (cv_sim[d] - grid_min[d]) / bin_width[d]). Collapses to a scalar for a 1D grid.
# Zero along dimensions where cv_sim is outside the grid, matching the clamped energy.
function bias_gradient(mem::GridHills, cv_sim)
    check_in_grid(mem, cv_sim)
    dims = grid_bracket_dims(mem, cv_sim)
    grad = ntuple(length(dims)) do d
        total = zero(mem.height) / mem.bin_width[d]
        in_grid(mem, d, cv_sim[d]) || return total
        for corner in grid_corners(dims)
            sign = corner[d] == 1 ? -1 : 1
            total += sign * grid_corner_value(mem, dims, corner, d) / mem.bin_width[d]
        end
        total
    end
    return from_dims(grad)
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

function add_hill!(mem::GridHills, cv_value, height=mem.height)
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
[`Metadynamics`](@ref)'s memory.

Subtypes must implement [`Molly.tempering_height`](@ref), the height to actually deposit,
and [`Molly.free_energy_scale`](@ref), the prefactor of the free energy estimate.
Temperings with a temperature also implement [`Molly.default_temperature`](@ref).

Built-in subtypes are [`NoTempering`](@ref) (the default) and
[`WellTempered`](@ref).
"""
abstract type AbstractTempering end

"""
    NoTempering()

Default tempering: every hill keeps its full base height (standard, constant-height
Metadynamics).
"""
struct NoTempering <: AbstractTempering end

"""
    tempering_height(tempering, v_bias, base_height, k)

Height of a new hill, given its base height `base_height` and the bias `v_bias` accumulated
so far at its CV value, as scaled by `tempering` (an [`AbstractTempering`](@ref)). `k` is
the Boltzmann constant of the System.
"""
function tempering_height end

tempering_height(::NoTempering, v_bias, base_height, k) = base_height

@doc raw"""
    WellTempered(bias_factor, temperature)

Standard well-tempered Metadynamics height decay (Barducci, Bussi & Parrinello, 2008):
```math
h(\boldsymbol{s}) = h_0 \exp\left(-\frac{V_{bias}(\boldsymbol{s})}{k_B (\gamma - 1) T}\right)
```
where $h_0$ is the base height, $V_{bias}$ the bias accumulated so far at $\boldsymbol{s}$,
$\gamma$ is `bias_factor` (> 1), $T$ is `temperature` and $k_B$ the Boltzmann constant of
the System. Deposits become self-limiting, so the bias converges to an estimate of the free
energy (scaled by `bias_factor`) instead of growing without bound.

# Arguments
- `bias_factor`: Temperature boost $\gamma > 1$; larger tempers more slowly (approaching
    untempered Metadynamics as $\gamma \to \infty$).
- `temperature`: Temperature $T$ of the System.
"""
struct WellTempered{Γ, T} <: AbstractTempering
    bias_factor::Γ
    temperature::T

    function WellTempered(bias_factor::Γ, temperature::T) where {Γ, T}
        if bias_factor <= one(bias_factor)
            throw(ArgumentError(
                "WellTempered bias_factor must be greater than 1, got " *
                "$(bias_factor)."))
        end
        validate_positive_finite(temperature, "WellTempered temperature")
        return new{Γ, T}(bias_factor, temperature)
    end
end

# kT in the type and unit of energy, e.g. a Float32 height in kJ mol^-1
thermal_energy(k, temperature, energy) = convert(typeof(energy), k * temperature)

function tempering_height(wt::WellTempered, v_bias, base_height, k)
    kT = thermal_energy(k, wt.temperature, base_height)
    return base_height * exp(-v_bias / (kT * (wt.bias_factor - 1)))
end

"""
    free_energy_scale(tempering)

Prefactor relating the converged bias to the free energy,
`F(s) ≈ free_energy_scale(tempering) * V_bias(s) + const`.
Custom [`AbstractTempering`](@ref) types should implement it.
"""
free_energy_scale(::NoTempering) = -1
free_energy_scale(wt::WellTempered) = -wt.bias_factor / (wt.bias_factor - 1)

@doc raw"""
    Metadynamics(memory; n_steps=1, tempering=NoTempering(), logger=nothing)

A history-dependent bias potential for Metadynamics: a sum of Gaussians deposited at
previously visited collective variable (CV) values.
```math
V(\boldsymbol{s}) = \sum_{t' < t} h_{t'} \exp\left(-\frac{|\boldsymbol{s} - \boldsymbol{s}(t')|^2}{2\sigma^2}\right)
```

The `memory` argument (an [`AbstractMetadynamicsMemory`](@ref)) holds the CVs and controls
how hills are stored: [`ListHills`](@ref) (exact, O(n_hills)) or [`GridHills`](@ref) (O(1),
grid-based).

With the CVs in the memory, e.g. `Metadynamics(ListHills((cv,), height, sigma))`, the bias
is an AtomsCalculators.jl calculator usable directly as a `general_inters` entry. Each CV
is evaluated on the coordinates its own `correction` asks for, and hills on periodic CVs
such as torsions wrap around (see [`cv_period`](@ref)). Hills are deposited into the memory
passed in, which is also `bias.memory`, so biases built from one memory are walkers sharing
their hills. Deposits into a memory aren't thread-safe, so walkers sharing a memory must
run one after the other, not in parallel.

Forces are computed every simulation step regardless of simulator, and a hill is deposited
on the first step of each window of `n_steps` steps -- no external logger is
needed. The first hill is on the start configuration, at step 0, or at step 1 for
integrators such as [`Langevin`](@ref) that first compute forces there. Recomputing forces
within a step (e.g. by a logger or coupler) adds no hill and a multiple time step
integrator deposits once per outer step. Chunked `simulate!` calls deposit as one
continuous run, with or without `init_step`, as long as a bias uses one of the two
throughout: mixing them, or calling `forces(sys)` between runs and then continuing with
`init_step`, shifts the schedule. Energy minimizers also count as
steps, so minimize without the bias. The number of hills deposited so far is
`bias.n_deposits` and the step of the last automatic deposit is
`bias.last_deposit_step`.

[`add_hill!`](@ref) deposits a hill manually. Force evaluations outside a simulator, e.g.
`AtomsCalculators.forces(sys, bias)` or `energy_forces`, have no step number and count
towards the same manual call counter, so they deposit hills too (every call for
`n_steps=1`).

`tempering` (an [`AbstractTempering`](@ref)) scales each deposited hill's height; the
default [`NoTempering`](@ref) always deposits the full base height.

# Arguments
- `memory::AbstractMetadynamicsMemory`: The CVs, and the storage and evaluation strategy
    for deposited hills.
- `n_steps::Integer=1`: Number of simulation steps between deposits into
    `memory`, starting with the start configuration. [`add_hill!`](@ref) calls and force calls without a
    step number use a separate call counter, depositing on every `n_steps`-th
    such call without affecting the simulation schedule.
- `tempering::AbstractTempering=NoTempering()`: Scales the height of each deposited hill.
- `logger=nothing`: A [`MetadynamicsLogger`](@ref) to record each deposit for reweighting,
    available as `bias.logger`.
"""
mutable struct Metadynamics{M <: AbstractMetadynamicsMemory, TP <: AbstractTempering, L}
    const memory::M # Holds the CVs, memory.cvs
    const n_steps::Int
    call_count::Int
    last_step_n::Int
    step_count::Int # Steps counted across runs, for the pacing
    last_deposit_step::Int
    n_deposits::Int
    const tempering::TP
    const logger::L

    # The memory checks its CVs against its own dimensions; the empty check guards custom
    #   memories
    function Metadynamics(memory::M; n_steps::Integer=1,
                          tempering::TP=NoTempering(), logger=nothing) where {
                              M <: AbstractMetadynamicsMemory, TP <: AbstractTempering}
        check_has_cvs(memory.cvs, "Metadynamics")
        if n_steps < 1
            throw(ArgumentError("n_steps must be at least 1, got $(n_steps)."))
        end
        check_temperature_units(default_temperature(tempering), memory,
                                string(nameof(typeof(tempering))))
        typed_logger = init_logger(logger, memory, tempering)
        return new{M, TP, typeof(typed_logger)}(memory, Int(n_steps), 0, -1, -1, -1, 0,
                                                tempering, typed_logger)
    end
end

"""
    add_hill!(md::Metadynamics, cv_value; k=Molly.default_k(unit(md.memory.height)))
    add_hill!(md::Metadynamics, sys)

Deposit a new Gaussian hill.

Called automatically from `forces!` during a simulation -- there is usually no need to call
this directly.

`cv_value` matches the shape `md` was constructed with (scalar or tuple); the `sys` form
evaluates the CVs of `md.memory` itself. Calls are counted by a manual call counter, separate from
the step schedule of automatic deposits during a simulation; only every
`n_steps`-th call actually updates the memory, at height
`tempering_height(md.tempering, v_bias, md.memory.height, k)`, where `v_bias` is the bias at
the deposited CV value (given, or evaluated from `sys`). `k` is the Boltzmann constant, by
default that of a [`System`](@ref) with the energy units of the memory, and `sys.k` for
the `sys` form.
"""
function add_hill!(md::Metadynamics, cv_value; k=default_k(unit(md.memory.height)))
    if !bias_all_finite(cv_value)
        throw(ArgumentError("add_hill! got a non-finite CV value $(cv_value)"))
    end
    check_logger_kT(md.logger, md, k)
    if should_deposit_hill!(md)
        deposit_hill!(md, cv_value, -1, k)
    end
    return nothing
end

function add_hill!(md::Metadynamics, sys::System)
    cv_sim = check_bias_finite(evaluate_metadynamics_cvs(md, sys), "collective variable", md)
    check_logger_kT(md.logger, md, sys.k)
    if should_deposit_hill!(md)
        deposit_hill!(md, cv_sim, -1, sys.k)
    end
    return nothing
end

"""
    free_energy(bias::Metadynamics, cv_value)

Estimate of the free energy at `cv_value` (scalar, or a tuple for several CVs) from the
current bias, up to an additive constant. Broadcasts over CV values:
`free_energy.(bias, cv_values)`.
"""
free_energy(md::Metadynamics, cv_value) =
    free_energy_scale(md.tempering) * potential_energy(md.memory, cv_value)

Base.broadcastable(md::Metadynamics) = Ref(md)

"""
    MetadynamicsLogger(; temperature=nothing, cv_points=nothing, filepath=nothing,
                       write_interval=100, overwrite=false)

Records each deposit of a [`Metadynamics`](@ref) bias for reweighting: the step, the CV
value s, the bias V(s, t) it was sampled under and the offset c(t) (Tiwary & Parrinello,
2015). Pass it as `Metadynamics(memory; logger=MetadynamicsLogger())`; the records are in
`bias.logger`, with `values(bias.logger)` the CV values.

`temperature` defaults to that of [`WellTempered`](@ref) and must be given
otherwise. kT, `bias.logger.kT`, is fixed at the first deposit with the Boltzmann constant
of its System; a later deposit with a different kT throws.
c(t) integrates the bias over the grid of a [`GridHills`](@ref), and over `cv_points`,
a uniform grid of CV values, for other memories.
See [`metadynamics_weights`](@ref).

c(t) costs O(grid points or `cv_points`) per deposit, so 2D/3D grids want a larger
`n_steps`.

With a `filepath`, the records are also appended to that text file, with the columns step,
CV values, bias, offset, hill height and log weight, in batches of `write_interval` records.
Call `flush(bias.logger)` after the run to write the last batch. An existing file is
appended to as a new run with its own header, unless `overwrite=true`, which replaces it.
Use one file per logger, also for walkers: a flush that finds its file changed by another
writer throws.
[`read_metadynamics_logger`](@ref) reads the file back.
"""
mutable struct MetadynamicsLogger{T, K, P, S, E, F}
    const temperature::T
    kT::K # Fixed at the first deposit, from the Boltzmann constant of the System
    const points::P # cv_points as given in the spec, the bias_points_cache in bias.logger
    const steps::Vector{Int}
    const cvs::Vector{S}
    const bias_energies::Vector{E}
    const offsets::Vector{E}
    const heights::Vector{E}
    const filepath::F # Nothing to record in memory only
    const write_interval::Int
    const overwrite::Bool
    n_written::Int
    file_end::Int # File size after this logger's last write, to catch other writers
end

# Logger from only the fields that differ from an empty, memory-only one
function make_logger(; temperature=nothing, kT=nothing, points=nothing, steps=Int[],
                     cvs=Any[], bias_energies=Any[], offsets=Any[], heights=Any[],
                     filepath=nothing, write_interval=100, overwrite=false, n_written=0,
                     file_end=0)
    return MetadynamicsLogger(temperature, kT, points, steps, cvs, bias_energies, offsets,
                              heights, filepath, write_interval, overwrite, n_written, file_end)
end

function MetadynamicsLogger(; temperature=nothing, cv_points=nothing, filepath=nothing,
                            write_interval::Integer=100, overwrite::Bool=false)
    if write_interval < 1
        throw(ArgumentError("MetadynamicsLogger write_interval must be at least 1, got " *
                            "$(write_interval)."))
    end
    return make_logger(; temperature, points=cv_points, filepath,
                       write_interval=Int(write_interval), overwrite)
end

# The spec passed to Metadynamics has untyped records, bias.logger typed ones
is_logger_spec(logger::MetadynamicsLogger) = eltype(logger.bias_energies) === Any

# The spec passed to Metadynamics stays empty; its records are in bias.logger
function check_logger_ready(logger::MetadynamicsLogger)
    if is_logger_spec(logger)
        throw(ArgumentError("use bias.logger, not the MetadynamicsLogger passed to Metadynamics"))
    end
    return logger
end

Base.values(logger::MetadynamicsLogger) = check_logger_ready(logger).cvs

"""
    default_temperature(tempering)

Temperature of `tempering` (an [`AbstractTempering`](@ref)), or `nothing` if it has none.
Its units are checked when the bias is built, and it is the default temperature of a
[`MetadynamicsLogger`](@ref). Custom temperings with a temperature should add a method.
"""
default_temperature(wt::WellTempered) = wt.temperature
default_temperature(tempering) = nothing

# Fails at construction, not at the first deposit, on a temperature whose units don't fit
#   the energies of the memory. It uses the default Boltzmann constant of those units, so
#   a System with a custom k is only checked at the first deposit.
check_temperature_units(::Nothing, mem, label) = nothing
function check_temperature_units(temperature, mem, label)
    try
        thermal_energy(default_k(unit(mem.height)), temperature, mem.height)
    catch e
        e isa Union{Unitful.DimensionError, Unitful.AffineError} || rethrow()
        throw(ArgumentError("$(label) temperature $(temperature) doesn't fit the energies of " *
                            "the memory, whose hill height is $(mem.height); give a " *
                            "temperature in K for unitful energies and a plain number otherwise"))
    end
    return nothing
end

# CV value type of a memory, for the logger history
cv_value_type(mem::ListHills) = eltype(mem.centers)
cv_value_type(mem::GridHills) = typeof(from_dims(mem.grid_min))
cv_value_type(mem) = Any

"""
    bias_points_cache(memory, cv_points)

Cache for [`bias_at_points`](@ref): the points over which a [`MetadynamicsLogger`](@ref)
integrates the bias for its offset c(t), and what `memory` needs to update the bias there.
`cv_points` is a uniform grid of CV values, or `nothing` for a memory with its own grid
such as [`GridHills`](@ref). The default keeps the points; custom memories can add a
method for a cheaper update.
"""
function bias_points_cache(mem, cv_points)
    points = points_vector(mem, cv_points)
    potential_energy(mem, first(points)) # Shape and units, before the run
    return points
end

function bias_points_cache(mem::GridHills, cv_points)
    if !isnothing(cv_points)
        throw(ArgumentError("GridHills integrates over its own grid; don't pass cv_points"))
    end
    return nothing
end

# Bias of a ListHills at the points, updated with each new hill
mutable struct ListHillsPoints{P, V}
    const points::P
    const values::V
    n_hills::Int
end

function bias_points_cache(mem::ListHills, cv_points)
    points = points_vector(mem, cv_points)
    convert(eltype(mem.centers), first(points)) # Shape and units, before the run
    return ListHillsPoints(points, fill(zero(mem.height), length(points)), 0)
end

"""
    bias_at_points(memory, cache)

The bias of `memory` at the points of `cache`, from [`bias_points_cache`](@ref).
"""
bias_at_points(mem, points) = [potential_energy(mem, s) for s in points]
bias_at_points(mem::GridHills, ::Nothing) = vec(mem.values)
function bias_at_points(mem::ListHills, cache::ListHillsPoints)
    # Adds the hills deposited since the last call, also those of other walkers
    for i in (cache.n_hills + 1):length(mem.centers)
        cache.values .+= hill_value.((mem,), cache.points, (mem.centers[i],), (mem.heights[i],))
    end
    cache.n_hills = length(mem.centers)
    return cache.values
end

# cv_points as a vector, required unless the memory has its own grid
function points_vector(mem, cv_points)
    if isnothing(cv_points) || isempty(cv_points)
        throw(ArgumentError("MetadynamicsLogger needs cv_points, a uniform grid of CV " *
                            "values, for $(nameof(typeof(mem)))"))
    end
    return vec(collect(cv_points))
end

# Typed logger from the user's spec
init_logger(::Nothing, mem, tempering) = nothing
function init_logger(spec::MetadynamicsLogger, mem, tempering)
    if !is_logger_spec(spec) || spec.n_written < 0
        throw(ArgumentError("pass a new MetadynamicsLogger(...) to each bias, not bias.logger " *
                            "or a logger already passed to another bias"))
    end
    temperature = isnothing(spec.temperature) ? default_temperature(tempering) : spec.temperature
    if isnothing(temperature)
        throw(ArgumentError("MetadynamicsLogger needs a temperature for " *
                            "$(nameof(typeof(tempering)))"))
    end
    check_temperature_units(temperature, mem, "MetadynamicsLogger")
    points = bias_points_cache(mem, spec.points)
    # kT with the default Boltzmann constant until the first deposit. Fails here, not at the
    #   first deposit, on a missing free_energy_scale.
    kT = thermal_energy(default_k(unit(mem.height)), temperature, mem.height)
    metadynamics_offset(tempering, [zero(mem.height)], kT)
    filepath = isnothing(spec.filepath) ? nothing : abspath(spec.filepath) # Robust to a cd
    file_end = prepare_logger_file(filepath, spec.overwrite)
    E = typeof(mem.height)
    spec.n_written = -1 # Marks the spec as used, so two walkers can't share it and its file
    return make_logger(; temperature, kT, points, cvs=cv_value_type(mem)[],
                       bias_energies=E[], offsets=E[], heights=E[], filepath,
                       write_interval=spec.write_interval, overwrite=spec.overwrite, file_end)
end

# As for TrajectoryWriter, overwrite deletes an existing file, otherwise it is appended to.
#   Opening it here makes a bad path fail at construction. Returns the file size.
prepare_logger_file(::Nothing, overwrite) = 0
function prepare_logger_file(filepath::AbstractString, overwrite::Bool)
    if isfile(filepath)
        if overwrite
            rm(filepath)
        elseif filesize(filepath) > 0
            @warn "MetadynamicsLogger file $filepath exists, appending a new run to it; use " *
                  "overwrite=true to delete it instead"
        end
    end
    open(_ -> nothing, filepath, "a")
    return filesize(filepath)
end

# Plain exponents (kJ mol^-1), which uparse reads back, also where Unitful prints superscripts
unit_label(x) = unit(x) == NoUnits ? "" :
    " [" * sprint(print, unit(x); context=:fancy_exponent => false) * "]"

function write_logger_header(io, logger::MetadynamicsLogger)
    cv_cols = ["cv_$i$(unit_label(x))" for (i, x) in enumerate(to_dims(first(logger.cvs)))]
    e = unit_label(first(logger.bias_energies))
    println(io, "# kT ", ustrip(logger.kT), unit_label(logger.kT))
    println(io, "# step ", join(cv_cols, " "), " bias$e offset$e height$e log_weight")
end

"""
    flush(logger::MetadynamicsLogger)

Write the records not yet in the file of `logger`, which are otherwise written in batches of
`write_interval`. Call it after the run. Does nothing without a `filepath`.
"""
function Base.flush(logger::MetadynamicsLogger)
    check_logger_ready(logger)
    n = length(logger.steps)
    (isnothing(logger.filepath) || logger.n_written == n) && return logger
    if filesize(logger.filepath) != logger.file_end
        throw(ArgumentError("MetadynamicsLogger file $(logger.filepath) was changed by another " *
                            "writer; give each logger its own file"))
    end
    open(logger.filepath, "a") do io
        logger.n_written == 0 && write_logger_header(io, logger) # Each logger writes one run
        for i in (logger.n_written + 1):n
            v, c = logger.bias_energies[i], logger.offsets[i]
            log_w = ustrip(NoUnits, (v - c) / logger.kT)
            println(io, join((logger.steps[i], ustrip.(to_dims(logger.cvs[i]))...,
                              ustrip(v), ustrip(c), ustrip(logger.heights[i]), log_w), " "))
        end
    end
    logger.n_written = n
    logger.file_end = filesize(logger.filepath)
    return logger
end

# Tiwary–Parrinello offset c(t) = kT ln(∫exp(aV/kT) / ∫exp((a-1)V/kT)), a = -free_energy_scale
function metadynamics_offset(tempering, v, kT)
    a = -free_energy_scale(tempering)
    return kT * (logsumexp(ustrip.(NoUnits, a .* v ./ kT)) -
                 logsumexp(ustrip.(NoUnits, (a - 1) .* v ./ kT)))
end

# Before the pacing and any force, so a mismatch leaves the bias unchanged
check_logger_kT(::Nothing, md, k) = nothing
function check_logger_kT(logger::MetadynamicsLogger, md, k)
    kT = thermal_energy(k, logger.temperature, md.memory.height)
    if !isempty(logger.steps) && !isapprox(kT, logger.kT)
        throw(ArgumentError("MetadynamicsLogger got kT $(kT), but its earlier deposits used " *
                            "$(logger.kT); deposit with the Boltzmann constant of one System"))
    end
    return nothing
end

# Bias and offset c(t) at a deposit, with kT from the Boltzmann constant k of the System
bias_and_offset(::Nothing, md, v_bias, k) = nothing
function bias_and_offset(logger::MetadynamicsLogger, md, v_bias, k)
    # The first deposit fixes kT; construction only set a provisional one
    isempty(logger.steps) && (logger.kT = thermal_energy(k, logger.temperature, md.memory.height))
    return (v_bias,
            metadynamics_offset(md.tempering, bias_at_points(md.memory, logger.points), logger.kT))
end

record_deposit!(::Nothing, step_n, cv_sim, v_c, height) = nothing
function record_deposit!(logger::MetadynamicsLogger, step_n, cv_sim, v_c, height)
    push!(logger.steps, step_n)
    push!(logger.cvs, cv_sim)
    push!(logger.bias_energies, v_c[1])
    push!(logger.offsets, v_c[2])
    push!(logger.heights, height)
    if length(logger.steps) - logger.n_written >= logger.write_interval
        flush(logger)
    end
    return logger
end

"""
    metadynamics_log_weights(logger::MetadynamicsLogger)

Unnormalised log weights `(V(s, t) - c(t)) / kT` of the recorded frames, e.g. for an online
PMF accumulator.
"""
metadynamics_log_weights(logger::MetadynamicsLogger) =
    ustrip.(NoUnits, (check_logger_ready(logger).bias_energies .- logger.offsets) ./ logger.kT)

"""
    metadynamics_weights(logger::MetadynamicsLogger)

Normalised weights `exp((V(s, t) - c(t)) / kT)` of the recorded frames, to reweight any
observable of those frames to the unbiased ensemble.
"""
function metadynamics_weights(logger::MetadynamicsLogger)
    log_w = metadynamics_log_weights(logger)
    return exp.(log_w .- logsumexp(log_w))
end

# Column names and units of a header line, e.g. "cv_1 [nm]"; no brackets means unitless
header_columns(text) = [(m[1], isnothing(m[2]) ? NoUnits : uparse(replace(m[2], " " => "*")))
                        for m in eachmatch(r"([\w.+-]+)(?: \[([^\]]*)\])?", text)]

"""
    read_metadynamics_logger(filepath; run=nothing)

Restore the records of a [`MetadynamicsLogger`](@ref) file, e.g. for
[`metadynamics_weights`](@ref) after the run. Units from the header are reattached.
The returned logger records nothing further.

A file written with `overwrite=false` can hold several appended runs, each with its own
header. They come from separate biases and must not be mixed: pick one with `run=i`.
"""
function read_metadynamics_logger(filepath::AbstractString; run=nothing)
    # Each "# kT" header starts a run, followed by its "# step" header and rows
    lines = readlines(filepath)
    is_row(line) = !startswith(line, "#") && !isempty(strip(line))
    starts = findall(startswith("# kT "), lines)
    if isempty(starts)
        throw(ArgumentError("$(filepath) has no MetadynamicsLogger header"))
    elseif any(is_row, lines[1:(first(starts) - 1)])
        throw(ArgumentError("$(filepath) has data rows before its first header"))
    end
    n_runs = length(starts)
    if isnothing(run)
        if n_runs > 1
            throw(ArgumentError("$(filepath) holds $(n_runs) runs of separate biases; " *
                                "pick one with run=i"))
        end
        run = 1
    elseif !(1 <= run <= n_runs)
        throw(ArgumentError("$(filepath) holds $(n_runs) run(s), got run=$(run)"))
    end
    block = lines[starts[run]:(run < n_runs ? starts[run + 1] - 1 : end)]
    (value, kT_unit), = header_columns(block[1][6:end])
    kT = parse(Float64, value) * kT_unit
    i_cols = findfirst(startswith("# step "), block)
    if isnothing(i_cols)
        throw(ArgumentError("$(filepath) has no MetadynamicsLogger header"))
    end
    cols = header_columns(block[i_cols][3:end])
    rows = [parse.(Float64, split(line)) for line in block if is_row(line)]
    # Columns: step, CVs, bias, offset, height, log weight
    n_cvs = length(cols) - 5
    cv_units = ntuple(i -> cols[1 + i][2], n_cvs)
    e_unit = cols[n_cvs + 2][2]
    cvs = [n_cvs == 1 ? r[2] * cv_units[1] : ntuple(i -> r[1 + i] * cv_units[i], n_cvs)
           for r in rows]
    energies(j) = [r[n_cvs + j] * e_unit for r in rows]
    return make_logger(; kT, steps=[Int(r[1]) for r in rows], cvs,
                       bias_energies=energies(2), offsets=energies(3), heights=energies(4),
                       n_written=length(rows))
end

# k is the Boltzmann constant of the System; v_bias the bias at cv_sim, if already computed
function deposit_hill!(md::Metadynamics, cv_sim, step_n, k; v_bias=nothing)
    cv_sim = from_dims(to_dims(cv_sim)) # Scalar for one CV, tuple for several
    # Before the hill: the bias this frame was sampled under, evaluated once per deposit
    isnothing(v_bias) && (v_bias = potential_energy(md.memory, cv_sim))
    height = tempering_height(md.tempering, v_bias, md.memory.height, k)
    v_c = bias_and_offset(md.logger, md, v_bias, k)
    add_hill!(md.memory, cv_sim, height)
    md.n_deposits += 1
    record_deposit!(md.logger, step_n, cv_sim, v_c, height) # After, so a failed deposit records nothing
    return nothing
end

# Manual pacing, for add_hill! and forces! calls without a step number: every
#   n_steps-th call deposits. Independent of the step-paced schedule below.
function should_deposit_hill!(md::Metadynamics, ::Nothing=nothing)
    md.call_count += 1
    return (md.call_count % md.n_steps) == 0
end

# Step pacing: one hill per window of n_steps counted steps, on the first force
#   call in the window, so the start configuration is deposited at step 0, or at step 1 by
#   integrators that first compute forces there (e.g. Langevin). Recomputing forces within
#   a step (loggers, couplers, the initial virial, the inner steps of a multiple time step
#   integrator) adds no hill. A lower step number is a new run, whose step 0 is the previous
#   run's last configuration, so chunked runs deposit as one run with or without init_step.
#   Mixing runs with and without init_step shifts the schedule (documented, not caught).
function should_deposit_hill!(md::Metadynamics, step_n::Integer)
    count = if md.step_count < 0
        step_n
    elseif step_n >= md.last_step_n
        md.step_count + step_n - md.last_step_n
    else
        md.step_count + step_n
    end
    deposit = fld(count, md.n_steps) != fld(md.step_count, md.n_steps)
    md.step_count, md.last_step_n = count, step_n
    deposit && (md.last_deposit_step = step_n)
    return deposit
end

# For check_bias_finite, which also keeps non-finite values from being deposited
bias_description(md::Metadynamics) =
    "Metadynamics with CVs $(join(nameof.(typeof.(md.memory.cvs)), ", "))"


# Host copies of the system made once per call: the coordinates each CV is evaluated on
#   (unwrapped for correction=:pbc, as stored for :wrap), the atoms and the velocities
function metadynamics_host_state(md::Metadynamics, sys)
    cvs = md.memory.cvs
    coords = any(cv -> cv.correction != :pbc, cvs) ? from_device(sys.coords) : nothing
    coords_pbc = any(cv -> cv.correction == :pbc, cvs) ? from_device(unwrap_molecules(sys)) :
                                                          nothing
    cv_coords = map(cv -> cv.correction == :pbc ? coords_pbc : coords, cvs)
    return cv_coords, from_device(sys.atoms), from_device(sys.velocities)
end

function evaluate_metadynamics_cvs(md::Metadynamics, sys)
    cv_coords, atoms, velocities = metadynamics_host_state(md, sys)
    cv_values = map(md.memory.cvs, cv_coords) do cv, coords
        calculate_cv(cv, coords, atoms, sys.boundary, velocities)
    end
    return from_dims(cv_values)
end

function AtomsCalculators.potential_energy(sys, md::Metadynamics; kwargs...)
    cv_sim = check_bias_finite(evaluate_metadynamics_cvs(md, sys), "collective variable", md)
    pe = potential_energy(md.memory, cv_sim)
    return check_bias_finite(pe, "potential energy", md; cv_sim)
end

# Subtracts the bias forces from fs and returns the bias energy before any deposit (nothing
#   without with_energy), from one evaluation of the CVs
function metadynamics_forces!(fs, sys, md::Metadynamics; needs_vir::Bool=false,
                              buffers=nothing, step_n=nothing, with_energy::Bool=false)
    check_logger_kT(md.logger, md, sys.k)
    cvs = md.memory.cvs
    cv_coords, atoms, velocities = metadynamics_host_state(md, sys)

    # CV gradients and values, each CV on the coordinates its own correction asks for
    grads = map(cvs, cv_coords) do cv, coords
        cv_gradient(cv, coords, atoms, sys.boundary, velocities)
    end
    cv_sim = check_bias_finite(from_dims(map(last, grads)), "collective variable", md)
    d_bias = check_bias_finite(bias_gradient(md.memory, cv_sim), "bias gradient", md; cv_sim)

    # All finite checks come before any force is applied or hill deposited
    fs_svecs = map(grads, to_dims(d_bias)) do (d_coords, _), d_b
        check_bias_finite(d_coords, "CV gradient", md; cv_sim)
        check_bias_finite(d_b .* d_coords, "bias force", md; cv_sim)
    end
    for (cv, coords, fs_svec) in zip(cvs, cv_coords, fs_svecs)
        if needs_vir && cv.has_virial
            calculate_virial!(buffers.virial, cv, coords, -fs_svec, atoms, sys.boundary)
        end
        fs .-= to_device(fs_svec, typeof(fs))
    end
    pe = with_energy ?
         check_bias_finite(potential_energy(md.memory, cv_sim), "potential energy", md; cv_sim) :
         nothing

    # Self-paced deposit, by simulation step when called from a simulator, after every
    #   finite check so non-finite values are never deposited
    if should_deposit_hill!(md, step_n)
        # step_n is nothing outside a simulator, recorded as -1
        deposit_hill!(md, cv_sim, something(step_n, -1), sys.k; v_bias=pe)
    end
    return pe
end

function AtomsCalculators.forces!(fs, sys, md::Metadynamics; needs_vir::Bool=false,
                                  buffers=nothing, step_n=nothing, kwargs...)
    metadynamics_forces!(fs, sys, md; needs_vir, buffers, step_n)
    return fs
end

# As for Ewald, the allocating and energy forms use Molly's zero_forces
AtomsCalculators.forces(sys::System, md::Metadynamics; kwargs...) =
    AtomsCalculators.forces!(zero_forces(sys), sys, md; kwargs...)

function AtomsCalculators.energy_forces!(fs::AbstractVector, # Required for disambiguation
                                         sys::System, md::Metadynamics; needs_vir::Bool=false,
                                         buffers=nothing, step_n=nothing, kwargs...)
    pe = metadynamics_forces!(fs, sys, md; needs_vir, buffers, step_n, with_energy=true)
    return (energy=pe, forces=fs)
end

AtomsCalculators.energy_forces(sys::System, md::Metadynamics; kwargs...) =
    AtomsCalculators.energy_forces!(zero_forces(sys), sys, md; kwargs...)
