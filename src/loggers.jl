# Loggers to record properties throughout a simulation

export
    run_loggers!,
    TemperatureLogger,
    log_property!,
    CoordinateLogger,
    VelocityLogger,
    TotalEnergyLogger,
    KineticEnergyLogger,
    PotentialEnergyLogger,
    ForceLogger,
    StructureWriter,
    TimeCorrelationLogger,
    compute_statistical_correlation

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
    TemperatureLogger(n_steps)
    TemperatureLogger(T, n_steps)

Log the temperature throughout a simulation.
"""
struct TemperatureLogger{T}
    n_steps::Int
    temperatures::Vector{T}
end

TemperatureLogger(T::Type, n_steps::Integer) = TemperatureLogger(n_steps, T[])

TemperatureLogger(n_steps::Integer) = TemperatureLogger(typeof(one(DefaultFloat)u"K"), n_steps)

function Base.show(io::IO, tl::TemperatureLogger)
    print(io, "TemperatureLogger{", eltype(tl.temperatures), "} with n_steps ",
        tl.n_steps, ", ", length(tl.temperatures),
        " temperatures recorded")
end

"""
    log_property!(logger, system, neighbors=nothing, step_n=0; parallel=true)

Log a property of the system thoughout a simulation.
Custom loggers should implement this function.
"""
function log_property!(logger::TemperatureLogger, s::System, neighbors=nothing,
    step_n::Integer=0; kwargs...)
    if step_n % logger.n_steps == 0
        push!(logger.temperatures, temperature(s))
    end
end

"""
    CoordinateLogger(n_steps; dims=3)
    CoordinateLogger(T, n_steps; dims=3)

Log the coordinates throughout a simulation.
"""
struct CoordinateLogger{T}
    n_steps::Int
    coords::Vector{Vector{T}}
end

function CoordinateLogger(T, n_steps::Integer; dims::Integer=3)
    return CoordinateLogger(n_steps,
        Array{SArray{Tuple{dims},T,1,dims},1}[])
end

function CoordinateLogger(n_steps::Integer; dims::Integer=3)
    return CoordinateLogger(typeof(one(DefaultFloat)u"nm"), n_steps; dims=dims)
end

function Base.show(io::IO, cl::CoordinateLogger)
    print(io, "CoordinateLogger{", eltype(eltype(cl.coords)), "} with n_steps ",
        cl.n_steps, ", ", length(cl.coords), " frames recorded for ",
        length(cl.coords) > 0 ? length(first(cl.coords)) : "?", " atoms")
end

function log_property!(logger::CoordinateLogger, s::System, neighbors=nothing,
    step_n::Integer=0; kwargs...)
    if step_n % logger.n_steps == 0
        push!(logger.coords, deepcopy(s.coords))
    end
end

"""
    VelocityLogger(n_steps; dims=3)
    VelocityLogger(T, n_steps; dims=3)

Log the velocities throughout a simulation.
"""
struct VelocityLogger{T}
    n_steps::Int
    velocities::Vector{Vector{T}}
end

function VelocityLogger(T, n_steps::Integer; dims::Integer=3)
    return VelocityLogger(n_steps,
        Array{SArray{Tuple{dims},T,1,dims},1}[])
end

function VelocityLogger(n_steps::Integer; dims::Integer=3)
    return VelocityLogger(typeof(one(DefaultFloat)u"nm * ps^-1"), n_steps; dims=dims)
end

function Base.show(io::IO, vl::VelocityLogger)
    print(io, "VelocityLogger{", eltype(eltype(vl.velocities)), "} with n_steps ",
        vl.n_steps, ", ", length(vl.velocities), " frames recorded for ",
        length(vl.velocities) > 0 ? length(first(vl.velocities)) : "?", " atoms")
end

function log_property!(logger::VelocityLogger, s::System, neighbors=nothing,
    step_n::Integer=0; kwargs...)
    if step_n % logger.n_steps == 0
        push!(logger.velocities, deepcopy(s.velocities))
    end
end

"""
    TotalEnergyLogger(n_steps)
    TotalEnergyLogger(T, n_steps)

Log the total energy of the system throughout a simulation.
"""
struct TotalEnergyLogger{T}
    n_steps::Int
    energies::Vector{T}
end

TotalEnergyLogger(T::Type, n_steps::Integer) = TotalEnergyLogger(n_steps, T[])

function TotalEnergyLogger(n_steps::Integer)
    return TotalEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)
end

function Base.show(io::IO, el::TotalEnergyLogger)
    print(io, "TotalEnergyLogger{", eltype(el.energies), "} with n_steps ",
        el.n_steps, ", ", length(el.energies), " energies recorded")
end

function log_property!(logger::TotalEnergyLogger, s::System, neighbors=nothing,
    step_n::Integer=0; kwargs...)
    if step_n % logger.n_steps == 0
        push!(logger.energies, total_energy(s, neighbors))
    end
end

"""
    KineticEnergyLogger(n_steps)
    KineticEnergyLogger(T, n_steps)

Log the kinetic energy of the system throughout a simulation.
"""
struct KineticEnergyLogger{T}
    n_steps::Int
    energies::Vector{T}
end

KineticEnergyLogger(T::Type, n_steps::Integer) = KineticEnergyLogger(n_steps, T[])

function KineticEnergyLogger(n_steps::Integer)
    return KineticEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)
end

function Base.show(io::IO, el::KineticEnergyLogger)
    print(io, "KineticEnergyLogger{", eltype(el.energies), "} with n_steps ",
        el.n_steps, ", ", length(el.energies), " energies recorded")
end

function log_property!(logger::KineticEnergyLogger, s::System, neighbors=nothing,
    step_n::Integer=0; kwargs...)
    if step_n % logger.n_steps == 0
        push!(logger.energies, kinetic_energy(s))
    end
end

"""
    PotentialEnergyLogger(n_steps)
    PotentialEnergyLogger(T, n_steps)

Log the potential energy of the system throughout a simulation.
"""
struct PotentialEnergyLogger{T}
    n_steps::Int
    energies::Vector{T}
end

PotentialEnergyLogger(T::Type, n_steps::Integer) = PotentialEnergyLogger(n_steps, T[])

function PotentialEnergyLogger(n_steps::Integer)
    return PotentialEnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)
end

function Base.show(io::IO, el::PotentialEnergyLogger)
    print(io, "PotentialEnergyLogger{", eltype(el.energies), "} with n_steps ",
        el.n_steps, ", ", length(el.energies), " energies recorded")
end

function log_property!(logger::PotentialEnergyLogger, s::System, neighbors=nothing,
    step_n::Integer=0; kwargs...)
    if step_n % logger.n_steps == 0
        push!(logger.energies, potential_energy(s, neighbors))
    end
end

"""
    ForceLogger(n_steps; dims=3)
    ForceLogger(T, n_steps; dims=3)

Log the forces throughout a simulation.
"""
struct ForceLogger{T}
    n_steps::Int
    forces::Vector{Vector{T}}
end

function ForceLogger(T, n_steps::Integer; dims::Integer=3)
    return ForceLogger(n_steps,
        Array{SArray{Tuple{dims},T,1,dims},1}[])
end

function ForceLogger(n_steps::Integer; dims::Integer=3)
    return ForceLogger(typeof(one(DefaultFloat)u"kJ * mol^-1 * nm^-1"), n_steps; dims=dims)
end

function Base.show(io::IO, fl::ForceLogger)
    print(io, "ForceLogger{", eltype(eltype(fl.forces)), "} with n_steps ",
        fl.n_steps, ", ", length(fl.forces), " frames recorded for ",
        length(fl.forces) > 0 ? length(first(fl.forces)) : "?", " atoms")
end

function log_property!(logger::ForceLogger, s::System, neighbors=nothing,
    step_n::Integer=0; parallel::Bool=true)
    if step_n % logger.n_steps == 0
        push!(logger.forces, forces(s, neighbors; parallel=parallel))
    end
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

"""
A time correlation logger, which allow to estimate statistical correlations of the form
\$\$ C(t)=\\left(\\left\\langle A(t)\\cdot B(0)\\right\\rangle -\\left\\langle A\\right\\rangle\\cdot \\left\\langle B\\right\\rangle\\right)\\left(\\sqrt{\\left\\langle |A|^2\\right\\rangle\\left\\langle |B|^2\\right\\rangle}\\right)^{-1}\$\$
(normalized form), or
\$\$C(t)=\\left(\\left\\langle A(t)\\cdot B(0)\\right\\rangle -\\left\\langle \\right\\rangle\\cdot \\left\\langle B\\right\\rangle\\right)\$\$
(unnormalized form). These can be used to estimate statistical error, or to compute transport coefficients from Green-Kubo type formulas.
A and B are observables, functions of the form f(sys::System,neighbors=nothing).    
The return values of A and B can be of scalar or vector type (including vectors of `SVector`s, like positions or velocities), and must implement `dot`
# Arguments
- `TA::DataType`: The `DataType` returned by `A`, suppporting `zero(TA)`.
- `observableA::Function`: The function corresponding to observable A.
- `observableB::Function`: The function corresponding to observable B.
- `observable_length::Integer`: The length of the observables if they are vectors, or one if they are scalar-valued.
- `n_correlation::Integer`: The length of the computed correlation vector.

 `n_correlation` should typically be chosen so that `dt*n_correlation>t_corr`,
  where `dt` is the simulation timestep and `t_corr` is the decorrelation time for the considered system and observables.

  For the purpose of numerical stability, the logger internally records sums instead of running averages. The normalized and unnormalized form of the correlation function can be retrieved 
 from a`logger::TimeCorrelationLogger` through accessing the `logger.normalized_correlations` and `logger.unnormalized_correlations` properties.
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

function log_property!(logger::TimeCorrelationLogger, s::System, neighbors=nothing, step_n::Integer=0; parallel::Bool=true)

    #compute observables
    A = logger.observableA(s, neighbors)
    B = (logger.observableA != logger.observableB) ? logger.observableB(s, neighbors) : A

    logger.n_timesteps += 1

    #update history lists -- values of A and B older than n_correlation steps are overwritten (see DataStructures.jl CircularBuffer)

    push!(logger.history_A, A)
    push!(logger.history_B, B)

    #update running sums (numerically stable method consists of summing and computing averages at output time)
    logger.sum_A += A
    logger.sum_B += B

    logger.sum_sq_A += dot(A, A)
    logger.sum_sq_B += dot(B, B)

    B1 = first(logger.history_B)

    for i = 1:min(logger.n_correlation, logger.n_timesteps)
        logger.sum_offset_products[i] += dot(logger.history_A[i], B1)
    end

end

function Base.getproperty(logger::TimeCorrelationLogger,s::Symbol)
    if (s != :normalized_correlations) && (s != :unnormalized_correlations)
        return getfield(logger,s)
    else
        n_samps = getfield(logger, :n_timesteps)
        C = zero(getfield(logger, :sum_offset_products))
        C_bar = dot(getfield(logger, :sum_A) / n_samps, getfield(logger, :sum_B) / n_samps)
        for i=1:getfield(logger, :n_correlation)
            C[i]=getfield(logger, :sum_offset_products)[i] / (n_samps - i + 1)
        end
        C .-= C_bar
        if s == :unnormalized_correlations
            return C
        else
            denom = sqrt((getfield(logger, :sum_sq_A) / n_samps) * (getfield(logger, :sum_sq_B) / n_samps))
            return C / denom
        end
    end
end
