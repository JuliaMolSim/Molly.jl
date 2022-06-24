# Loggers to record properties throughout a simulation

export
    run_loggers!,
    GeneralObservableLogger,
    log_property!,
    TemperatureLogger,
    CoordinateLogger,
    VelocityLogger,
    TotalEnergyLogger,
    KineticEnergyLogger,
    PotentialEnergyLogger,
    ForceLogger,
    StructureWriter,
    TimeCorrelationLogger,
    AutoCorrelationLogger,
    AverageObservableLogger

"""
    run_loggers!(system, neighbors=nothing, step_n=0; parallel=true)

Run the loggers associated with the system.
"""
function run_loggers!(s::System, neighbors=nothing, step_n::Integer=0; parallel::Bool=true)
    for logger in values(s.loggers)
        log_property!(logger, s, neighbors, step_n; parallel=parallel)
    end
end

"""
GeneralObservableLogger(observable::Function,T::DataType,n_steps::Int)

Returns a logger which hold a record of regularly sampled observation on the system. 
The observable should return an object of type T and support the following method.
    observable(s::System,neighbors;parallel::Bool)::T
"""
struct GeneralObservableLogger{T,F}
    n_steps::Int64
    observable::F
    history::Vector{T}
end

GeneralObservableLogger(observable::Function,T::DataType,n_steps::Int) = GeneralObservableLogger{T,typeof(observable)}(n_steps,observable,T[])
Base.values(logger::GeneralObservableLogger)=logger.history
"""
    log_property!(logger, system, neighbors=nothing, step_n=0; parallel=true)

Log a property of the system thoughout a simulation.
Custom loggers should implement this function.
"""
function log_property!(logger::GeneralObservableLogger,s::System,neighbors=nothing,step_n::Integer=0;parallel::Bool=true)
    if (step_n % logger.n_steps) == 0
        obs=logger.observable(s,neighbors;parallel=parallel)
        push!(logger.history,obs)
    end
end



temperature_wrapper(s,neighbors=nothing;parallel::Bool=true)=temperature(s)
"""
    TemperatureLogger(n_steps)
    TemperatureLogger(T, n_steps)

Log the temperature throughout a simulation.
"""
TemperatureLogger(T::DataType,n_steps::Integer)=GeneralObservableLogger(temperature_wrapper,T,n_steps)
TemperatureLogger(n_steps::Integer) = TemperatureLogger(typeof(one(DefaultFloat)u"K"), n_steps)

function Base.show(io::IO, tl::GeneralObservableLogger{T,typeof(temperature_wrapper)}) where {T}
    print(io, "TemperatureLogger{", eltype(values(tl)), "} with n_steps ",
                tl.n_steps, ", ", length(values(tl)),
                " temperatures recorded")
end

coordinates_wrapper(s,neighbors=nothing;parallel::Bool=true)=s.coords
"""
    CoordinateLogger(n_steps; dims=3)
    CoordinateLogger(T, n_steps; dims=3)

Log the coordinates throughout a simulation.
"""
CoordinateLogger(T, n_steps::Integer; dims::Integer=3)=GeneralObservableLogger(coordinates_wrapper,Array{SArray{Tuple{dims}, T, 1, dims}, 1},n_steps)
CoordinateLogger(n_steps::Integer; dims::Integer=3)=CoordinateLogger(typeof(one(DefaultFloat)u"nm"), n_steps; dims=dims)

function Base.show(io::IO, cl::GeneralObservableLogger{T,typeof(coordinates_wrapper)}) where {T}
    print(io, "CoordinateLogger{", eltype(eltype(cl.history)), "} with n_steps ",
            cl.n_steps, ", ", length(cl.history), " frames recorded for ",
            length(cl.history) > 0 ? length(first(cl.history)) : "?", " atoms")
end

velocities_wrapper(s::System,neighbors=nothing;parallel::Bool=true)=s.velocities
"""
    VelocityLogger(n_steps; dims=3)
    VelocityLogger(T, n_steps; dims=3)

Log the velocities throughout a simulation.
"""
VelocityLogger(T, n_steps::Integer; dims::Integer=3)=GeneralObservableLogger(velocities_wrapper,Array{SArray{Tuple{dims}, T, 1, dims}, 1},n_steps)
VelocityLogger(n_steps::Integer; dims::Integer=3)=VelocityLogger(typeof(one(DefaultFloat)u"nm * ps^-1"), n_steps; dims=dims)

function Base.show(io::IO, vl::GeneralObservableLogger{T,typeof(velocities_wrapper)}) where {T}
    print(io, "VelocityLogger{", eltype(eltype(vl.history)), "} with n_steps ",
            vl.n_steps, ", ", length(vl.history), " frames recorded for ",
            length(vl.history) > 0 ? length(first(vl.history)) : "?", " atoms")
end

total_energy_wrapper(s::System,neighbors=nothing;parallel::Bool=true)=total_energy(s,neighbors)
"""
    TotalEnergyLogger(n_steps)
    TotalEnergyLogger(T, n_steps)

Log the total energy of the system throughout a simulation.
"""
TotalEnergyLogger(T::DataType,n_steps)=GeneralObservableLogger(total_energy_wrapper,T,n_steps)
TotalEnergyLogger(n_steps)=TotalEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)

function Base.show(io::IO, el::GeneralObservableLogger{T,typeof(total_energy_wrapper)}) where {T}
    print(io, "TotalEnergyLogger{", eltype(el.history), "} with n_steps ",
                el.n_steps, ", ", length(el.history), " energies recorded")
end


kinetic_energy_wrapper(s::System,neighbors=nothing;parallel::Bool=true)=kinetic_energy(s)
"""
    KineticEnergyLogger(n_steps)
    KineticEnergyLogger(T, n_steps)

Log the kinetic energy of the system throughout a simulation.
"""
KineticEnergyLogger(T::Type, n_steps::Integer) = GeneralObservableLogger(kinetic_energy_wrapper,T,n_steps)
KineticEnergyLogger(n_steps::Integer)=KineticEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)

function Base.show(io::IO, el::GeneralObservableLogger{T,typeof(kinetic_energy_wrapper)}) where {T}
    print(io, "KineticEnergyLogger{", eltype(el.history), "} with n_steps ",
                el.n_steps, ", ", length(el.history), " energies recorded")
end

potential_energy_wrapper(s::System,neighbors=nothing;parallel::Bool=true)=potential_energy(s,neighbors)
"""
    PotentialEnergyLogger(n_steps)
    PotentialEnergyLogger(T, n_steps)

Log the potential energy of the system throughout a simulation.
"""
PotentialEnergyLogger(T::Type, n_steps::Integer) = GeneralObservableLogger(potential_energy_wrapper,T,n_steps)
PotentialEnergyLogger(n_steps::Integer) = PotentialEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)

function Base.show(io::IO, el::GeneralObservableLogger{T,typeof(potential_energy_wrapper)}) where {T}
    print(io, "PotentialEnergyLogger{", eltype(el.history), "} with n_steps ",
                el.n_steps, ", ", length(el.history), " energies recorded")
end


"""
    ForceLogger(n_steps; dims=3)
    ForceLogger(T, n_steps; dims=3)

Log the forces throughout a simulation.
"""
ForceLogger(T, n_steps::Integer; dims::Integer=3)=GeneralObservableLogger(forces,Array{SArray{Tuple{dims}, T, 1, dims}, 1},n_steps)
ForceLogger(n_steps::Integer; dims::Integer=3)=ForceLogger(typeof(one(DefaultFloat)u"kJ * mol^-1 * nm^-1"), n_steps; dims=dims)
function Base.show(io::IO, fl::GeneralObservableLogger{T,typeof(forces)}) where {T}
    print(io, "ForceLogger{", eltype(eltype(fl.history)), "} with n_steps ",
            fl.n_steps, ", ", length(fl.history), " frames recorded for ",
            length(fl.history) > 0 ? length(first(fl.history)) : "?", " atoms")
end

"""
    StructureWriter(n_steps, filepath, excluded_res=String[])

Write 3D output structures to the PDB file format throughout a simulation.
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

function log_property!(logger::StructureWriter, s::System, neighbors=nothing,
                        step_n::Integer=0; kwargs...)
    if step_n % logger.n_steps == 0
        if length(s) != length(s.atoms_data)
            error("Number of atoms is ", length(s), " but number of atom data entries is ",
                    length(s.atoms_data))
        end
        append_model!(logger, s)
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
`TimeCorrelationLogger(TA::DataType, TB::DataType, observableA::Function, observableB::Function, observable_length::Integer, n_correlation::Integer)`
A time correlation logger, which allow to estimate statistical correlations of normalized form
```math
C(t)=\frac{\langle A_t\cdot B_0\rangle -\langle A\rangle\cdot \langle B\rangle}{\sqrt{\langle |A|^2\rangle\langle |B|^2\rangle}}
```
or unnormalized form
```math
C(t)=\langle A_t\cdot B_0\rangle -\langle A \rangle\cdot \langle B\rangle
```
These can be used to estimate statistical error, or to compute transport coefficients from Green-Kubo type formulas.
A and B are observables, functions of the form `observable(sys::System,neighbors; parallel::Bool)`.    
The return values of A and B can be of scalar or vector type (including `Vector{SVector{...}}`s, like positions or velocities), and must implement `dot`
# Arguments
- `TA::DataType`: The type returned by `observableA`, suppporting `zero(TA)`.
- `TB::DataType`: The type returned by `observableB`, supporting `zero(TB)`.
- `observableA::Function`: The function corresponding to observable A.
- `observableB::Function`: The function corresponding to observable B.
- `observable_length::Integer`: The length of the observables if they are vectors, or `1` if they are scalar-valued.
- `n_correlation::Integer`: The length of the computed correlation vector.

 `n_correlation` should typically be chosen so that `dt*n_correlation>t_corr`,
  where `dt` is the simulation timestep and `t_corr` is the decorrelation time for the considered system and observables.

  For the purpose of numerical stability, the logger internally records sums instead of running averages. The normalized and unnormalized form of the correlation function can be retrieved 
  through `values(logger::TimeCorrelationLogger; normalize::Bool)``
"""
mutable struct TimeCorrelationLogger{T_A,T_A2,T_B,T_B2,T_AB,TF_A,TF_B}
    observableA::TF_A
    observableB::TF_B

    n_correlation::Integer

    history_A::CircularBuffer{T_A}
    history_B::CircularBuffer{T_B}

    sum_offset_products::Vector{T_AB} 

    n_timesteps::Int64

    sum_A::T_A
    sum_B::T_B

    sum_sq_A::T_A2
    sum_sq_B::T_B2
end

function TimeCorrelationLogger(TA::DataType, TB::DataType, observableA::TF_A, observableB::TF_B, observable_length::Integer, n_correlation::Integer) where {TF_A,TF_B}
    ini_sum_A = (observable_length > 1) ? zeros(TA, observable_length) : zero(TA)
    ini_sum_B = (observable_length > 1) ? zeros(TB, observable_length) : zero(TB)

    ini_sum_sq_A = dot(ini_sum_A, ini_sum_A)
    ini_sum_sq_B = dot(ini_sum_B, ini_sum_B)

    T_A = typeof(ini_sum_A)
    T_A2 = typeof(ini_sum_sq_A)

    T_B = typeof(ini_sum_B)
    T_B2 = typeof(ini_sum_sq_B)

    T_AB = typeof(dot(zero(TA),zero(TB)))

    return TimeCorrelationLogger{T_A,T_A2,T_B,T_B2,T_AB,TF_A,TF_B}(observableA, observableB, n_correlation, CircularBuffer{T_A}(n_correlation), CircularBuffer{T_B}(n_correlation), zeros(T_AB, n_correlation), 0, ini_sum_A, ini_sum_B, ini_sum_sq_A, ini_sum_sq_B)

end

"""
`function AutoCorrelationLogger(TA::DataType, observable::Function, observable_length::Integer, n_correlation::Integer)`
An autocorrelation logger, equivalent to a `TimeCorrelationLogger` in the case `observableA == observableB`.
"""
AutoCorrelationLogger(TA::DataType, observable::TF_A, observable_length::Integer, n_correlation::Integer) where {TF_A} = TimeCorrelationLogger(TA, TA, observable, observable, observable_length, n_correlation)

function Base.show(io::IO, tcl::TimeCorrelationLogger)
    print(io, "TimeCorrelationLogger with n_correlation ", tcl.n_correlation, ", and ", tcl.n_timesteps, " samples collected for observables ",
                tcl.observableA, " and ",tcl.observableB)
end

function Base.show(io::IO, tcl::TimeCorrelationLogger{TA,TA2,TA,TA2,TAB,TFA,TFA}) where {TA,TA2,TAB,TFA}
    print(io, "AutoCorrelationLogger with n_correlation ", tcl.n_correlation, ", and ", tcl.n_timesteps, " samples collected for observable ",
                tcl.observableA)
end

function log_property!(logger::TimeCorrelationLogger, s::System, neighbors=nothing, step_n::Integer=0; parallel::Bool=true)

    #compute observables
    A = logger.observableA(s, neighbors; parallel=parallel)
    B = (logger.observableA != logger.observableB) ? logger.observableB(s, neighbors; parallel=parallel) : A

    logger.n_timesteps += 1

    #update history lists -- values of A and B older than n_correlation steps are overwritten (see DataStructures.jl CircularBuffer)

    push!(logger.history_A, A)
    push!(logger.history_B, B)

    #update running sums (numerically stable method consists of summing and computing averages at output time)
    logger.sum_A += A
    logger.sum_B += B

    logger.sum_sq_A += dot(A, A)
    logger.sum_sq_B += dot(B, B)
    
    buff_length=length(logger.history_A)
    
    if parallel
        chunk_size = Int64(ceil(buff_length / nthreads()))
        ix_ranges=[i:min(i + chunk_size - 1, buff_length) for i in 1:chunk_size:buff_length]
        @threads for ixs in ix_ranges
            logger.sum_offset_products[ixs] .+= dot.(logger.history_A[ixs],(first(logger.history_B),))
        end
    else
        logger.sum_offset_products[1:buff_length] .+= dot.(logger.history_A,(first(logger.history_B),))
    end
end

function Base.values(logger::TimeCorrelationLogger; normalize::Bool=true)
    n_samps = logger.n_timesteps
        C = zero(logger.sum_offset_products)
        C_bar = dot(logger.sum_A / n_samps, logger.sum_B / n_samps)
        for i=1:logger.n_correlation
            C[i]=logger.sum_offset_products[i] / (n_samps - i + 1)
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
`AverageObservableLogger(observable::Function,T::DataType,n_steps::Integer;n_blocks::Integer=1024)`
A logger that periodically records observations of a system, and keeps a running empirical average.
While `GeneralObservableLogger` holds a full record of observations, `AverageObservableLogger` does not.
In addition, calling `values(logger::AverageObservableLogger; std::Bool=true)` returns two values: the current running average,
 and an estimate of the standard deviation for this average, based on the [block averaging method](https://aip.scitation.org/doi/10.1063/1.457480).

# Arguments
- `observable::Function`: The observable whose mean is recorded. Must support the method `observable(s::System, neighbors; parallel::Bool)`.
- `T::DataType`: The type returned by `observable`.
- `n_blocks::Integer`(optional): The number of blocks used in the block averaging method. It should be an even number.
"""
mutable struct AverageObservableLogger{T,F}
    observable::F
    n_steps::Int64
    n_blocks::Int64
    current_block_size::Int64
    block_averages::Vector{T}
    current_block::Vector{T}
end

function AverageObservableLogger(observable::Function, T::DataType, n_steps::Integer; n_blocks::Integer=1024)
    return AverageObservableLogger{T,typeof(observable)}(observable, n_steps, n_blocks, 1, T[],T[])
end

function log_property!(aol::AverageObservableLogger{T}, s::System, neighbors=nothing, step_n::Integer=0; parallel::Bool=true) where {T}
    if (step_n % aol.n_steps) == 0
        obs=aol.observable(s,neighbors;parallel=parallel)
        push!(aol.current_block,obs)

        if length(aol.current_block)==aol.current_block_size #current block is full
            push!(aol.block_averages,mean(aol.current_block))
            aol.current_block=T[]
        end

        if length(aol.block_averages)==aol.n_blocks #block averages buffer is full
            aol.block_averages = T[(avg1 + avg2) / 2 for (avg1,avg2)=zip(aol.block_averages[1:2:end],aol.block_averages[2:2:end])]
            aol.current_block_size = aol.current_block_size * 2
        end
        
    end
end

function Base.values(aol::AverageObservableLogger) #TODO add some logic to use the samples in the hanging block
    avg = mean(aol.block_averages)
    variance = var(aol.block_averages) / length(aol.block_averages)

    return [avg, sqrt(variance)]
end

function Base.show(io::IO, aol::AverageObservableLogger)
    print(io, "AverageObservableLogger with n_steps ", aol.n_steps, ", and ", (aol.current_block_size * length(aol.block_averages)) , " samples collected for observable ",
                aol.observable)
end