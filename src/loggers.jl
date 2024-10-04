# Loggers to record properties throughout a simulation

export
    apply_loggers!,
    GeneralObservableLogger,
    values,
    log_property!,
    TemperatureLogger,
    CoordinateLogger,
    VelocityLogger,
    TotalEnergyLogger,
    KineticEnergyLogger,
    PotentialEnergyLogger,
    ForceLogger,
    VirialLogger,
    VolumeLogger,
    DensityLogger,
    PressureLogger,
    StructureWriter,
    TimeCorrelationLogger,
    AutoCorrelationLogger,
    AverageObservableLogger,
    ReplicaExchangeLogger,
    MonteCarloLogger

"""
    apply_loggers!(system, neighbors=nothing, step_n=0, run_loggers=true;
                   n_threads=Threads.nthreads(), kwargs...)

Run the loggers associated with a system.

`run_loggers` can be `true`, `false` or `:skipzero`, in which case the loggers
are not run before the first step.
Additional keyword arguments can be passed to the loggers if required.
Ignored for gradient calculation during automatic differentiation.
"""
function apply_loggers!(sys::System, neighbors=nothing, step_n::Integer=0, run_loggers=true;
                        n_threads::Integer=Threads.nthreads(), kwargs...)
    if run_loggers == true || (run_loggers == :skipzero && step_n != 0)
        for logger in values(sys.loggers)
            log_property!(logger, sys, neighbors, step_n; n_threads=n_threads, kwargs...)
        end
    end
    return sys
end

"""
    GeneralObservableLogger(observable::Function, T, n_steps)

A logger which holds a record of regularly sampled observations of a system.

`observable` should return an object of type `T` and support the method
`observable(s::System, neighbors; n_threads::Integer)::T`.
"""
struct GeneralObservableLogger{T, F}
    observable::F
    n_steps::Int
    history::Vector{T}
end

function GeneralObservableLogger(observable::Function, T::DataType, n_steps::Integer)
    return GeneralObservableLogger{T, typeof(observable)}(observable, n_steps, T[])
end

"""
    values(logger)
    values(logger::TimeCorrelationLogger; normalize::Bool=true)
    values(logger::AverageObservableLogger; std::Bool=true)

Access the stored observations in a logger.
"""
Base.values(logger::GeneralObservableLogger) = logger.history

"""
    log_property!(logger, system, neighbors=nothing, step_n=0;
                  n_threads=Threads.nthreads(), kwargs...)

Log a property of a system throughout a simulation.

Custom loggers should implement this function.
Additional keyword arguments can be passed to the logger if required.
"""
function log_property!(logger::GeneralObservableLogger, s::System, neighbors=nothing,
                        step_n::Integer=0; kwargs...)
    if (step_n % logger.n_steps) == 0
        obs = logger.observable(s, neighbors, step_n; kwargs...)
        push!(logger.history, obs)
    end
end

function Base.show(io::IO, gol::GeneralObservableLogger)
    print(io, "GeneralObservableLogger with n_steps ", gol.n_steps, ", ",
            length(gol.history), " values recorded for observable ",
            gol.observable)
end

temperature_wrapper(sys, args...; kwargs...) = temperature(sys)

"""
    TemperatureLogger(n_steps)
    TemperatureLogger(T, n_steps)

Log the [`temperature`](@ref) throughout a simulation.
"""
function TemperatureLogger(T::DataType, n_steps::Integer)
    return GeneralObservableLogger(temperature_wrapper, T, n_steps)
end

TemperatureLogger(n_steps::Integer) = TemperatureLogger(typeof(one(DefaultFloat)u"K"), n_steps)

function Base.show(io::IO, tl::GeneralObservableLogger{T, typeof(temperature_wrapper)}) where T
    print(io, "TemperatureLogger{", eltype(values(tl)), "} with n_steps ",
            tl.n_steps, ", ", length(values(tl)), " temperatures recorded")
end

coordinates_wrapper(sys, args...; kwargs...) = copy(sys.coords)

"""
    CoordinateLogger(n_steps; dims=3)
    CoordinateLogger(T, n_steps; dims=3)

Log the coordinates throughout a simulation.
"""
function CoordinateLogger(T, n_steps::Integer; dims::Integer=3)
    return GeneralObservableLogger(
        coordinates_wrapper,
        Array{SArray{Tuple{dims}, T, 1, dims}, 1},
        n_steps,
    )
end

CoordinateLogger(n_steps::Integer; dims::Integer=3) = CoordinateLogger(typeof(one(DefaultFloat)u"nm"), n_steps; dims=dims)

function Base.show(io::IO, cl::GeneralObservableLogger{T, typeof(coordinates_wrapper)}) where T
    print(io, "CoordinateLogger{", eltype(eltype(values(cl))), "} with n_steps ",
            cl.n_steps, ", ", length(values(cl)), " frames recorded for ",
            length(values(cl)) > 0 ? length(first(values(cl))) : "?", " atoms")
end

velocities_wrapper(sys, args...; kwargs...) = copy(sys.velocities)

"""
    VelocityLogger(n_steps; dims=3)
    VelocityLogger(T, n_steps; dims=3)

Log the velocities throughout a simulation.
"""
function VelocityLogger(T, n_steps::Integer; dims::Integer=3)
    return GeneralObservableLogger(
        velocities_wrapper,
        Array{SArray{Tuple{dims}, T, 1, dims}, 1},
        n_steps,
    )
end

VelocityLogger(n_steps::Integer; dims::Integer=3) = VelocityLogger(typeof(one(DefaultFloat)u"nm * ps^-1"), n_steps; dims=dims)

function Base.show(io::IO, vl::GeneralObservableLogger{T, typeof(velocities_wrapper)}) where T
    print(io, "VelocityLogger{", eltype(eltype(values(vl))), "} with n_steps ",
            vl.n_steps, ", ", length(values(vl)), " frames recorded for ",
            length(values(vl)) > 0 ? length(first(values(vl))) : "?", " atoms")
end

kinetic_energy_wrapper(sys, args...; kwargs...) = kinetic_energy(sys)

"""
    KineticEnergyLogger(n_steps)
    KineticEnergyLogger(T, n_steps)

Log the [`kinetic_energy`](@ref) of a system throughout a simulation.
"""
function KineticEnergyLogger(T::Type, n_steps::Integer)
    return GeneralObservableLogger(kinetic_energy_wrapper, T, n_steps)
end

KineticEnergyLogger(n_steps::Integer) = KineticEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)

function Base.show(io::IO, el::GeneralObservableLogger{T, typeof(kinetic_energy_wrapper)}) where T
    print(io, "KineticEnergyLogger{", eltype(values(el)), "} with n_steps ",
            el.n_steps, ", ", length(values(el)), " energies recorded")
end

function potential_energy_wrapper(sys, neighbors, step_n::Integer; n_threads::Integer,
                                  current_potential_energy=nothing, kwargs...)
    if isnothing(current_potential_energy)
        return potential_energy(sys, neighbors, step_n; n_threads=n_threads)
    else
        return current_potential_energy
    end
end

"""
    PotentialEnergyLogger(n_steps)
    PotentialEnergyLogger(T, n_steps)

Log the [`potential_energy`](@ref) of a system throughout a simulation.
"""
function PotentialEnergyLogger(T::Type, n_steps::Integer)
    return GeneralObservableLogger(potential_energy_wrapper, T, n_steps)
end

PotentialEnergyLogger(n_steps::Integer) = PotentialEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)

function Base.show(io::IO, el::GeneralObservableLogger{T, typeof(potential_energy_wrapper)}) where T
    print(io, "PotentialEnergyLogger{", eltype(values(el)), "} with n_steps ",
            el.n_steps, ", ", length(values(el)), " energies recorded")
end

function total_energy_wrapper(sys, args...; kwargs...)
    return kinetic_energy(sys) + potential_energy_wrapper(sys, args...; kwargs...)
end

"""
    TotalEnergyLogger(n_steps)
    TotalEnergyLogger(T, n_steps)

Log the [`total_energy`](@ref) of a system throughout a simulation.
"""
TotalEnergyLogger(T::DataType, n_steps) = GeneralObservableLogger(total_energy_wrapper, T, n_steps)
TotalEnergyLogger(n_steps) = TotalEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)

function Base.show(io::IO, el::GeneralObservableLogger{T, typeof(total_energy_wrapper)}) where T
    print(io, "TotalEnergyLogger{", eltype(values(el)), "} with n_steps ",
            el.n_steps, ", ", length(values(el)), " energies recorded")
end

function forces_wrapper(sys, neighbors, step_n::Integer; n_threads::Integer,
                        current_forces=nothing, kwargs...)
    if isnothing(current_forces)
        return forces(sys, neighbors, step_n; n_threads=n_threads)
    else
        return copy(current_forces)
    end
end

"""
    ForceLogger(n_steps; dims=3)
    ForceLogger(T, n_steps; dims=3)

Log the [`forces`](@ref) throughout a simulation.
"""
function ForceLogger(T, n_steps::Integer; dims::Integer=3)
    return GeneralObservableLogger(
        forces_wrapper,
        Array{SArray{Tuple{dims}, T, 1, dims}, 1},
        n_steps,
    )
end

ForceLogger(n_steps::Integer; dims::Integer=3) = ForceLogger(typeof(one(DefaultFloat)u"kJ * mol^-1 * nm^-1"), n_steps; dims=dims)

function Base.show(io::IO, fl::GeneralObservableLogger{T, typeof(forces_wrapper)}) where T
    print(io, "ForceLogger{", eltype(eltype(values(fl))), "} with n_steps ",
            fl.n_steps, ", ", length(values(fl)), " frames recorded for ",
            length(values(fl)) > 0 ? length(first(values(fl))) : "?", " atoms")
end

function virial_wrapper(sys, neighbors, step_n; n_threads, kwargs...)
    return virial(sys, neighbors, step_n; n_threads=n_threads)
end

"""
    VirialLogger(n_steps)
    VirialLogger(T, n_steps)

Log the [`virial`](@ref) of a system throughout a simulation.

This should only be used on systems containing just pairwise interactions, or
where the specific interactions, general interactions and constraints do not
contribute to the virial.
"""
VirialLogger(T::Type, n_steps::Integer) = GeneralObservableLogger(virial_wrapper, T, n_steps)
VirialLogger(n_steps::Integer) = VirialLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)

function Base.show(io::IO, vl::GeneralObservableLogger{T, typeof(virial_wrapper)}) where T
    print(io, "VirialLogger{", eltype(values(vl)), "} with n_steps ",
            vl.n_steps, ", ", length(values(vl)), " virials recorded")
end

volume_wrapper(sys, args...; kwargs...) = volume(sys)

"""
    VolumeLogger(n_steps)
    VolumeLogger(T, n_steps)

Log the [`volume`](@ref) of a system throughout a simulation.

Not compatible with infinite boundaries.
"""
VolumeLogger(T::Type, n_steps::Integer) = GeneralObservableLogger(volume_wrapper, T, n_steps)
VolumeLogger(n_steps::Integer) = VolumeLogger(typeof(one(DefaultFloat)u"nm^3"), n_steps)

function Base.show(io::IO, vl::GeneralObservableLogger{T, typeof(volume_wrapper)}) where T
    print(io, "VolumeLogger{", eltype(values(vl)), "} with n_steps ",
            vl.n_steps, ", ", length(values(vl)), " volumes recorded")
end

density_wrapper(sys, args...; kwargs...) = density(sys)

"""
    DensityLogger(n_steps)
    DensityLogger(T, n_steps)

Log the [`density`](@ref) of a system throughout a simulation.

Not compatible with infinite boundaries.
"""
DensityLogger(T::Type, n_steps::Integer) = GeneralObservableLogger(density_wrapper, T, n_steps)
DensityLogger(n_steps::Integer) = DensityLogger(typeof(one(DefaultFloat)u"kg * m^-3"), n_steps)

function Base.show(io::IO, dl::GeneralObservableLogger{T, typeof(density_wrapper)}) where T
    print(io, "DensityLogger{", eltype(values(dl)), "} with n_steps ",
            dl.n_steps, ", ", length(values(dl)), " densities recorded")
end

function pressure_wrapper(sys, neighbors, step_n; n_threads, kwargs...)
    return pressure(sys, neighbors, step_n; n_threads=n_threads)
end

"""
    PressureLogger(n_steps)
    PressureLogger(T, n_steps)

Log the [`pressure`](@ref) of a system throughout a simulation.

This should only be used on systems containing just pairwise interactions, or
where the specific interactions, general interactions and constraints do not
contribute to the pressure.
"""
PressureLogger(T::Type, n_steps::Integer) = GeneralObservableLogger(pressure_wrapper, T, n_steps)
PressureLogger(n_steps::Integer) = PressureLogger(typeof(one(DefaultFloat)u"bar"), n_steps)

function Base.show(io::IO, pl::GeneralObservableLogger{T, typeof(pressure_wrapper)}) where T
    print(io, "PressureLogger{", eltype(values(pl)), "} with n_steps ",
            pl.n_steps, ", ", length(values(pl)), " pressures recorded")
end

"""
    StructureWriter(n_steps, filepath, excluded_res=String[])

Write 3D output structures to a file in the PDB format throughout a simulation.

The [`System`](@ref) should have `atoms_data` defined.
"""
mutable struct StructureWriter
    n_steps::Int
    filepath::String
    excluded_res::Set{String}
    structure_n::Int
end

function StructureWriter(n_steps::Integer, filepath::AbstractString, excluded_res=String[])
    return StructureWriter(n_steps, filepath, Set(excluded_res), 0)
end

function Base.show(io::IO, sw::StructureWriter)
    print(io, "StructureWriter with n_steps ", sw.n_steps, ", filepath \"",
            sw.filepath, "\", ", sw.structure_n, " frames written")
end

function log_property!(logger::StructureWriter, sys::System, neighbors=nothing,
                        step_n::Integer=0; kwargs...)
    if step_n % logger.n_steps == 0
        if length(sys) != length(sys.atoms_data)
            error("number of atoms is ", length(sys), " but number of atom data entries is ",
                  length(sys.atoms_data), ", StructureWriter requires them to be the same")
        end
        append_model!(logger, sys)
    end
end

function append_model!(logger::StructureWriter, sys)
    logger.structure_n += 1
    open(logger.filepath, "a") do output
        println(output, "MODEL     ", lpad(logger.structure_n, 4))
        for (i, coord) in enumerate(Array(sys.coords))
            atom_data = sys.atoms_data[i]
            if unit(first(coord)) == NoUnits
                # If not told, assume coordinates are in nm and convert to Å
                coord_convert = 10 .* coord
            else
                coord_convert = ustrip.(u"Å", coord)
            end
            if !(atom_data.res_name in logger.excluded_res)
                at_rec = atom_record(atom_data, i, coord_convert)
                println(output, BioStructures.pdbline(at_rec))
            end
        end
        println(output, "ENDMDL")
    end
end

atom_record(at_data, i, coord) = BioStructures.AtomRecord(
    false, i, at_data.atom_name, ' ', at_data.res_name, "A",
    at_data.res_number, ' ', coord, 1.0, 0.0,
    at_data.element == "?" ? "  " : at_data.element, "  "
)

@doc raw"""
    TimeCorrelationLogger(observableA::Function, observableB::Function,
                            TA::DataType, TB::DataType,
                            observable_length::Integer, n_correlation::Integer)

A time correlation logger.

Estimates statistical correlations of normalized form
```math
C(t)=\frac{\langle A_t\cdot B_0\rangle -\langle A\rangle\cdot \langle B\rangle}{\sqrt{\langle |A|^2\rangle\langle |B|^2\rangle}}
```
or unnormalized form
```math
C(t)=\langle A_t\cdot B_0\rangle -\langle A \rangle\cdot \langle B\rangle
```
These can be used to estimate statistical error, or to compute transport
coefficients from Green-Kubo type formulas.
*A* and *B* are observables, functions of the form
`observable(sys::System, neighbors; n_threads::Integer)`.
The return values of *A* and *B* can be of scalar or vector type (including
`Vector{SVector{...}}`, like positions or velocities) and must implement `dot`.

`n_correlation` should typically be chosen so that
`dt * n_correlation > t_corr`, where `dt` is the simulation timestep and
`t_corr` is the decorrelation time for the considered system and observables.
For the purpose of numerical stability, the logger internally records sums
instead of running averages.
The normalized and unnormalized form of the correlation function can be
retrieved through `values(logger::TimeCorrelationLogger; normalize::Bool)`.

# Arguments
- `observableA::Function`: the function corresponding to observable A.
- `observableB::Function`: the function corresponding to observable B.
- `TA::DataType`: the type returned by `observableA`, supporting `zero(TA)`.
- `TB::DataType`: the type returned by `observableB`, supporting `zero(TB)`.
- `observable_length::Integer`: the length of the observables if they are
    vectors, or `1` if they are scalar-valued.
- `n_correlation::Integer`: the length of the computed correlation vector.
"""
mutable struct TimeCorrelationLogger{T_A, T_A2, T_B, T_B2, T_AB, TF_A, TF_B}
    observableA::TF_A
    observableB::TF_B
    n_correlation::Int
    history_A::CircularBuffer{T_A}
    history_B::CircularBuffer{T_B}
    sum_offset_products::Vector{T_AB}
    n_timesteps::Int
    sum_A::T_A
    sum_B::T_B
    sum_sq_A::T_A2
    sum_sq_B::T_B2
end

function TimeCorrelationLogger(observableA::TF_A, observableB::TF_B,
                                TA::DataType, TB::DataType,
                                observable_length::Integer,
                                n_correlation::Integer) where {TF_A, TF_B}
    ini_sum_A = (observable_length > 1) ? zeros(TA, observable_length) : zero(TA)
    ini_sum_B = (observable_length > 1) ? zeros(TB, observable_length) : zero(TB)

    ini_sum_sq_A = dot(ini_sum_A, ini_sum_A)
    ini_sum_sq_B = dot(ini_sum_B, ini_sum_B)

    T_A = typeof(ini_sum_A)
    T_A2 = typeof(ini_sum_sq_A)

    T_B = typeof(ini_sum_B)
    T_B2 = typeof(ini_sum_sq_B)

    T_AB = typeof(dot(zero(TA), zero(TB)))

    return TimeCorrelationLogger{T_A, T_A2, T_B, T_B2, T_AB, TF_A, TF_B}(
        observableA, observableB, n_correlation,
        CircularBuffer{T_A}(n_correlation), CircularBuffer{T_B}(n_correlation),
        zeros(T_AB, n_correlation), 0, ini_sum_A, ini_sum_B,
        ini_sum_sq_A, ini_sum_sq_B,
    )
end

function Base.show(io::IO, tcl::TimeCorrelationLogger)
    print(io, "TimeCorrelationLogger with n_correlation ", tcl.n_correlation,
            " and ", tcl.n_timesteps, " samples collected for observables ",
            tcl.observableA, " and ", tcl.observableB)
end

"""
    AutoCorrelationLogger(observable::Function, TA::DataType,
                            observable_length::Integer, n_correlation::Integer)

An autocorrelation logger, equivalent to a [`TimeCorrelationLogger`](@ref) in the case
that `observableA == observableB`.
"""
function AutoCorrelationLogger(observable, TA, observable_length::Integer,
                                n_correlation::Integer)
    return TimeCorrelationLogger(observable, observable, TA, TA,
                                    observable_length, n_correlation)
end

function Base.show(io::IO, tcl::TimeCorrelationLogger{TA, TA2, TA, TA2, TAB, TFA, TFA}) where {TA, TA2, TAB, TFA}
    print(io, "AutoCorrelationLogger with n_correlation ", tcl.n_correlation,
            " and ", tcl.n_timesteps, " samples collected for observable ",
            tcl.observableA)
end

function log_property!(logger::TimeCorrelationLogger, s::System, neighbors=nothing,
                        step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...)
    A = logger.observableA(s, neighbors, step_n; n_threads=n_threads, kwargs...)
    if logger.observableA != logger.observableB
        B = logger.observableB(s, neighbors, step_n; n_threads=n_threads, kwargs...)
    else
        B = A
    end

    logger.n_timesteps += 1

    # Update history lists
    # Values of A and B older than `n_correlation` steps are overwritten
    # See DataStructures.jl `CircularBuffer`
    push!(logger.history_A, A)
    push!(logger.history_B, B)

    # Update running sums
    # Numerically stable method sums and computes averages at output time
    logger.sum_A += A
    logger.sum_B += B

    logger.sum_sq_A += dot(A, A)
    logger.sum_sq_B += dot(B, B)

    buff_length = length(logger.history_A)

    if n_threads > 1
        chunk_size = Int(ceil(buff_length / n_threads))
        ix_ranges = [i:min(i + chunk_size - 1, buff_length) for i in 1:chunk_size:buff_length]
        Threads.@threads for ixs in ix_ranges
            logger.sum_offset_products[ixs] .+= dot.(logger.history_A[ixs], (first(logger.history_B),))
        end
    else
        logger.sum_offset_products[1:buff_length] .+= dot.(logger.history_A, (first(logger.history_B),))
    end
end

function Base.values(logger::TimeCorrelationLogger; normalize::Bool=true)
    n_samps = logger.n_timesteps
    C = zero(logger.sum_offset_products)
    C_bar = dot(logger.sum_A / n_samps, logger.sum_B / n_samps)
    for i in 1:logger.n_correlation
        C[i] = logger.sum_offset_products[i] / (n_samps - i + 1)
    end
    C .-= C_bar
    if normalize
        denom = sqrt(logger.sum_sq_A * logger.sum_sq_B) / n_samps
        return C / denom
    else
        return C
    end
end

"""
    AverageObservableLogger(observable::Function, T::DataType, n_steps::Integer;
                            n_blocks::Integer=1024)

A logger that periodically records observations of a system and keeps a running
empirical average.

While [`GeneralObservableLogger`](@ref) holds a full record of observations,
[`AverageObservableLogger`](@ref) does not.
In addition, calling `values(logger::AverageObservableLogger; std::Bool=true)`
returns two values: the current running average, and an estimate of the standard
deviation for this average based on the block averaging method described in
[Flyvbjerg and Petersen 1989](https://doi.org/10.1063/1.457480).

# Arguments
- `observable::Function`: the observable whose mean is recorded, must support
    the method `observable(s::System, neighbors; n_threads::Integer)`.
- `T::DataType`: the type returned by `observable`.
- `n_steps::Integer`: number of simulation steps between observations.
- `n_blocks::Integer=1024`: the number of blocks used in the block averaging
    method, should be an even number.
"""
mutable struct AverageObservableLogger{T, F}
    observable::F
    n_steps::Int
    n_blocks::Int
    current_block_size::Int
    block_averages::Vector{T}
    current_block::Vector{T}
end

function AverageObservableLogger(observable::Function, T::DataType, n_steps::Integer;
                                    n_blocks::Integer=1024)
    return AverageObservableLogger{T, typeof(observable)}(observable, n_steps, n_blocks, 1, T[], T[])
end

function Base.values(aol::AverageObservableLogger; std::Bool=true)
    # Could add some logic to use the samples in the hanging block
    avg = mean(aol.block_averages)
    variance = var(aol.block_averages) / length(aol.block_averages)
    if std
        return (avg, sqrt(variance))
    else
        return avg
    end
end

function log_property!(aol::AverageObservableLogger{T}, s::System, neighbors=nothing,
                        step_n::Integer=0; kwargs...) where T
    if (step_n % aol.n_steps) == 0
        obs = aol.observable(s, neighbors, step_n; kwargs...)
        push!(aol.current_block, obs)

        if length(aol.current_block) == aol.current_block_size
            # Current block is full
            push!(aol.block_averages, mean(aol.current_block))
            aol.current_block = T[]
        end

        if length(aol.block_averages) == aol.n_blocks
            # Block averages buffer is full
            aol.block_averages = T[(avg1 + avg2) / 2 for (avg1, avg2) in zip(aol.block_averages[1:2:end], aol.block_averages[2:2:end])]
            aol.current_block_size = aol.current_block_size * 2
        end
    end
end

function Base.show(io::IO, aol::AverageObservableLogger)
    print(io, "AverageObservableLogger with n_steps ", aol.n_steps, ", ",
            aol.current_block_size * length(aol.block_averages),
            " samples collected for observable ", aol.observable)
end

"""
    ReplicaExchangeLogger(n_replicas)
    ReplicaExchangeLogger(T, n_replicas)

A logger that records exchanges in a replica exchange simulation.

The logged quantities include the number of exchange attempts (`n_attempts`),
number of successful exchanges (`n_exchanges`), exchanged replica indices (`indices`),
exchange steps (`steps`) and the value of Δ i.e. the argument of Metropolis rate for
the exchanges (`deltas`).
"""
mutable struct ReplicaExchangeLogger{T}
    n_replicas::Int
    n_attempts::Int
    n_exchanges::Int
    indices::Vector{Tuple{Int, Int}}
    steps::Vector{Int}
    deltas::Vector{T}
    end_step::Int
end

function ReplicaExchangeLogger(T::DataType, n_replicas::Integer)
    return ReplicaExchangeLogger{T}(n_replicas, 0, 0, Tuple{Int, Int}[], Int[], T[], 0)
end

ReplicaExchangeLogger(n_replicas::Integer) = ReplicaExchangeLogger(DefaultFloat, n_replicas)

function log_property!(rexl::ReplicaExchangeLogger,
                       sys::ReplicaSystem,
                       neighbors=nothing,
                       step_n::Integer=0;
                       indices,
                       delta,
                       n_threads::Integer=Threads.nthreads(),
                       kwargs...)
    push!(rexl.indices, indices)
    push!(rexl.steps, step_n + rexl.end_step)
    push!(rexl.deltas, delta)
    rexl.n_exchanges += 1
end

function finish_logs!(rexl::ReplicaExchangeLogger; n_steps::Integer=0, n_attempts::Integer=0)
    rexl.end_step += n_steps
    rexl.n_attempts += n_attempts
end

@doc raw"""
    MonteCarloLogger()
    MonteCarloLogger(T)

A logger that records acceptances in a Monte Carlo simulation.

The logged quantities include the number of new selections (`n_select`),
the number of successful acceptances (`n_accept`), an array named `energy_rates` which stores
the value of ``\frac{E}{k_B T}`` i.e. the argument of the Boltzmann factor for the states,
and a `BitVector` named `state_changed` that stores whether a new state was accepted for the
logged step.
"""
mutable struct MonteCarloLogger{T}
    n_trials::Int
    n_accept::Int
    energy_rates::Vector{T}
    state_changed::BitVector
end

MonteCarloLogger(T::DataType=DefaultFloat) = MonteCarloLogger{T}(0, 0, T[], BitVector())

function log_property!(mcl::MonteCarloLogger{T},
                        sys::System,
                        neighbors=nothing,
                        step_n::Integer=0;
                        success::Bool,
                        energy_rate::T,
                        kwargs...) where T
    mcl.n_trials += 1
    if success
        mcl.n_accept += 1
    end
    push!(mcl.state_changed, success)
    push!(mcl.energy_rates, energy_rate)
end
