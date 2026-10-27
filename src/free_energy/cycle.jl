export
    ThermodynamicCycleLeg,
    ThermodynamicCycle

@public free_energy,
    free_energy_uncertainty,
    free_energy_difference,
    free_energy_difference_uncertainty,
    standard_state_correction

"""
    ThermodynamicCycleLeg(source, from => to; coefficient=1.0, name=Symbol(""))

One leg of a [`ThermodynamicCycle`](@ref): the free-energy difference
``F_\\text{to} - F_\\text{from}`` reported by `source`, multiplied by `coefficient`
(usually ±1).

`from` and `to` are state indices or [`ThermoState`](@ref) names. `source` is any
object that implements [`Molly.free_energy_difference`](@ref) and
[`Molly.free_energy_difference_uncertainty`](@ref), such as a [`TSSState`](@ref), an
[`AWHState`](@ref), a [`Molly.MBARResult`](@ref) or a PMF.
"""
struct ThermodynamicCycleLeg{S, I, J}
    source::S
    from::I
    to::J
    coefficient::Float64
    name::Symbol
end

function ThermodynamicCycleLeg(source, states::Pair; coefficient::Real=1.0, name::Symbol=Symbol(""))
    from, to = states
    for state in (from, to)
        if !(state isa Union{Integer, AbstractString}) || state isa Bool
            throw(ArgumentError("leg states must be indices or ThermoState names, " *
                                "got $(repr(state))"))
        end
    end
    return ThermodynamicCycleLeg(source, from, to, Float64(coefficient), name)
end

"""
    ThermodynamicCycle(legs::ThermodynamicCycleLeg...; correction=nothing)

A free energy composed from the free-energy differences of several legs.

[`Molly.free_energy`](@ref) returns ``\\sum_i c_i \\Delta F_i`` plus `correction`, and
[`Molly.free_energy_uncertainty`](@ref) its standard error.

`correction` is an analytic term in energy units, e.g. a standard-state correction
(see [`Molly.standard_state_correction`](@ref)), the release of restraints or a long-range
correction. `nothing` adds no term.

Each leg is converted to energy units with the temperature of its own states, so legs
at different temperatures may be combined. A source raises an error when the two
states of one leg have different temperatures, since their difference would need a
single ``k_B T``.
"""
struct ThermodynamicCycle{L <: Tuple, C}
    legs::L
    correction::C
end

function ThermodynamicCycle(legs::ThermodynamicCycleLeg...; correction=nothing)
    isempty(legs) && throw(ArgumentError("a ThermodynamicCycle needs at least one leg"))
    return ThermodynamicCycle(legs, correction)
end

leg_label(leg::ThermodynamicCycleLeg, i) = (leg.name == Symbol("") ? "leg $i" : "leg $i ($(leg.name))")

# Every leg value and the correction must have the same dimension
function check_cycle_units(cycle::ThermodynamicCycle, values, what)
    dim = Unitful.dimension(first(values))
    for (i, (leg, value)) in enumerate(zip(cycle.legs, values))
        if Unitful.dimension(value) != dim
            throw(ArgumentError("$(what) of $(leg_label(leg, i)) has units $(unit(value)), " *
                                "which are not compatible with $(unit(first(values))) " *
                                "of the first leg"))
        end
    end
    correction = cycle.correction
    if !isnothing(correction) && Unitful.dimension(correction) != dim
        throw(ArgumentError("the cycle correction has units $(unit(correction)), which " *
                            "are not compatible with $(unit(first(values))) of the legs"))
    end
    return unit(first(values))
end

"""
    free_energy(cycle::ThermodynamicCycle)

Free energy of a [`ThermodynamicCycle`](@ref), ``\\sum_i c_i \\Delta F_i`` plus the
correction, in the energy units of the first leg.

    free_energy(bias::Metadynamics, cv_value)

Estimate of the free energy at `cv_value` (scalar, or a tuple for several CVs) from the
current bias, up to an additive constant. Broadcasts over CV values:
`Molly.free_energy.(bias, cv_values)`.
"""
function free_energy(cycle::ThermodynamicCycle)
    ΔF = map(leg -> free_energy_difference(leg.source, leg.from, leg.to), cycle.legs)
    energy_units = check_cycle_units(cycle, ΔF, "the free-energy difference")
    total = sum(leg.coefficient * ΔF_i for (leg, ΔF_i) in zip(cycle.legs, ΔF))
    if !isnothing(cycle.correction)
        total += cycle.correction
    end
    return uconvert(energy_units, total)
end

"""
    free_energy_uncertainty(cycle::ThermodynamicCycle)

Standard error of [`Molly.free_energy`](@ref)`(cycle)`, ``\\sqrt{\\sum_i c_i^2 \\sigma_i^2}``,
in the energy units of the first leg.

The legs are assumed to be independent, e.g. from separate simulations. Legs that share
a source, such as two legs of one [`TSSState`](@ref), are correlated and this estimate
does not account for it. The correction is taken to be exact.
"""
function free_energy_uncertainty(cycle::ThermodynamicCycle)
    σ = map(leg -> free_energy_difference_uncertainty(leg.source, leg.from, leg.to),
            cycle.legs)
    energy_units = check_cycle_units(cycle, σ, "the uncertainty")
    return uconvert(energy_units, sqrt(sum((leg.coefficient * σ_i)^2
                                           for (leg, σ_i) in zip(cycle.legs, σ))))
end

# Index of a state given by index or by name, among `n` states with `names`
function cycle_state_index(names, n::Integer, i::Integer)
    (1 <= i <= n) || throw(ArgumentError("state index $i out of bounds for $n states"))
    return Int(i)
end

function cycle_state_index(names, n::Integer, name::AbstractString)
    isnothing(names) && throw(ArgumentError("states have no names; refer to them by index"))
    matches = findall(==(name), names)
    if length(matches) != 1
        throw(ArgumentError("state name \"$name\" matches $(length(matches)) states; " *
                            "it must match exactly one"))
    end
    return only(matches)
end

function check_same_temperature(kBT_i, kBT_j, i, j)
    if !isapprox(kBT_i, kBT_j)
        throw(ArgumentError("states $i and $j have different temperatures (kBT $kBT_i and " *
                            "$kBT_j); their free-energy difference in energy units would " *
                            "need a single kBT"))
    end
    return kBT_i
end

# Indices of states i and j of an ExtendedStateSpace and their shared thermal energy
function state_space_pair_kBT(space::ExtendedStateSpace, i, j)
    names = [ts.name for ts in space.thermo_states]
    i = cycle_state_index(names, n_states(space), i)
    j = cycle_state_index(names, n_states(space), j)
    kBT(k) = Float64(inv(space.betas[k])) * space.systems[k].energy_units
    return i, j, check_same_temperature(kBT(i), kBT(j), i, j)
end

"""
    free_energy_difference(source, i, j)

Free-energy difference ``F_j - F_i`` between states `i` and `j` of a leg source, in
energy units.

States are given by index or by [`ThermoState`](@ref) name. This is the interface a
[`ThermodynamicCycleLeg`](@ref) source implements, together with
[`Molly.free_energy_difference_uncertainty`](@ref). Molly implements it for:
- [`TSSState`](@ref), from [`Molly.tss_free_energies`](@ref), with the jackknife
  standard error of [`Molly.tss_free_energy_uncertainties`](@ref);
- [`AWHState`](@ref), from its current free energies `f`. AWH has no uncertainty
  estimate, so `Molly.free_energy_difference_uncertainty` raises an error;
- [`Molly.MBARResult`](@ref), the MBAR estimate for umbrella sampling, REMD or any
  other set of samples, with its asymptotic standard error;
- a `PMF` from [`Molly.pmf_with_uncertainty`](@ref) or [`Molly.pmf`](@ref), between two
  bins given by linear index. The PMF must have been computed with `kBT`. The
  uncertainty, when the PMF has one, treats the two bins as independent, an approximation
  that can be too large or too small since the bins share the normalisation.

States of a [`TSSState`](@ref), an [`AWHState`](@ref) or an [`Molly.MBARResult`](@ref)
built from [`ThermoState`](@ref)s are converted with ``k_B T`` of the states; both must
have the same temperature.
"""
function free_energy_difference(state::TSSState, i, j)
    i, j, kBT = state_space_pair_kBT(state.state_space, i, j)
    return tss_free_energies(state; reference_state=i)[j] * kBT
end

"""
    free_energy_difference_uncertainty(source, i, j)

Standard error of [`Molly.free_energy_difference`](@ref)`(source, i, j)`, in energy units.
"""
function free_energy_difference_uncertainty(state::TSSState, i, j)
    i, j, kBT = state_space_pair_kBT(state.state_space, i, j)
    return tss_free_energy_uncertainties(state; reference_state=i).standard_errors[j] * kBT
end

# The current estimate, `state.f`
function free_energy_difference(state::AWHState, i, j)
    i, j, kBT = state_space_pair_kBT(state.state_space, i, j)
    return (state.f[j] - state.f[i]) * kBT
end

function free_energy_difference_uncertainty(::AWHState, i, j)
    throw(ArgumentError("AWHState has no free-energy uncertainty estimate; estimate it " *
                        "from independent runs or from the samples with MBAR (MBARResult)"))
end

function mbar_pair_kBT(result::MBARResult, i, j)
    i = cycle_state_index(result.names, length(result.f), i)
    j = cycle_state_index(result.names, length(result.f), j)
    return i, j, check_same_temperature(result.kBT[i], result.kBT[j], i, j)
end

function free_energy_difference(result::MBARResult, i, j)
    i, j, kBT = mbar_pair_kBT(result, i, j)
    return (result.f[j] - result.f[i]) * kBT
end

function free_energy_difference_uncertainty(result::MBARResult, i, j)
    i, j, kBT = mbar_pair_kBT(result, i, j)
    Θ = result.covariance
    return sqrt(max(Θ[i, i] + Θ[j, j] - 2 * Θ[i, j], 0.0)) * kBT
end

# PMF bins i and j (linear indices), in the units of `F_energy`
function pmf_bin_indices(pmf::PMF, i, j, field)
    if isnothing(getfield(pmf, field))
        throw(ArgumentError("the PMF has no $(field); compute it with kBT given" *
                            (field == :sigma_F_energy ? " and with uncertainties" : "")))
    end
    n = length(pmf.F)
    i, j = cycle_state_index(nothing, n, i), cycle_state_index(nothing, n, j)
    for k in (i, j)
        isfinite(pmf.F[k]) || throw(ArgumentError("PMF bin $k has no finite free energy"))
    end
    return i, j
end

# Bins share the temperature of the PMF's target state
function free_energy_difference(pmf::PMF, i, j)
    i, j = pmf_bin_indices(pmf, i, j, :F_energy)
    return pmf.F_energy[j] - pmf.F_energy[i]
end

# Treats the two bins as independent, the covariance between bins is not available; the
#   shared normalisation makes this an approximation in either direction
function free_energy_difference_uncertainty(pmf::PMF, i, j)
    i, j = pmf_bin_indices(pmf, i, j, :sigma_F_energy)
    return sqrt(pmf.sigma_F_energy[i]^2 + pmf.sigma_F_energy[j]^2)
end

"""
    standard_state_correction(temperature, pressure; concentration=1u"mol/L")

Free energy of taking an ideal gas from `pressure` to `concentration`,
``k_B T \\ln(v_\\text{gas} / v_\\text{std})`` with ``v_\\text{gas} = k_B T / P`` and
``v_\\text{std} = 1 / (c N_A)``, in kJ/mol.

An alchemical solvation cycle gives the transfer free energy at equal concentrations in
the gas and the solution. Adding this correction gives the standard solvation free
energy from a gas at `pressure` to a solution at `concentration`.
"""
function standard_state_correction(temperature, pressure; concentration=1u"mol/L")
    v_gas = Unitful.k * temperature / pressure
    v_std = 1 / (concentration * Unitful.Na)
    return uconvert(u"kJ/mol", Unitful.R * temperature * log(uconvert(NoUnits, v_gas / v_std)))
end
