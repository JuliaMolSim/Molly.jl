# Loggers to record properties throughout a simulation

export
    TemperatureLogger,
    log_property!,
    CoordinateLogger,
    EnergyLogger,
    StructureWriter

"""
    TemperatureLogger(n_steps)
    TemperatureLogger(T, n_steps)

Log the temperature throughout a simulation.
"""
struct TemperatureLogger{T} <: Logger
    n_steps::Int
    temperatures::Vector{T}
end

TemperatureLogger(T::Type, n_steps::Integer) = TemperatureLogger(n_steps, T[])

TemperatureLogger(n_steps::Integer) = TemperatureLogger(DefaultFloat, n_steps)

function Base.show(io::IO, tl::TemperatureLogger)
    print(io, "TemperatureLogger{", eltype(tl.temperatures), "} with n_steps ",
                tl.n_steps, ", ", length(tl.temperatures),
                " temperatures recorded")
end

"""
    log_property!(logger, simulation, step_n)

Log a property thoughout a simulation.
Custom loggers should implement this function.
"""
function log_property!(logger::TemperatureLogger, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        push!(logger.temperatures, temperature(s))
    end
end

"""
    CoordinateLogger(n_steps; dims=3)

Log the coordinates throughout a simulation.
"""
struct CoordinateLogger{T} <: Logger
    n_steps::Int
    coords::Vector{Vector{T}}
end

function CoordinateLogger(T, n_steps::Integer; dims::Integer=3)
    return CoordinateLogger(n_steps,
                            Array{SArray{Tuple{dims}, T, 1, dims}, 1}[])
end

function CoordinateLogger(n_steps::Integer; dims::Integer=3)
    return CoordinateLogger(DefaultFloat, n_steps, dims=dims)
end

function Base.show(io::IO, cl::CoordinateLogger)
    print(io, "CoordinateLogger{", eltype(eltype(cl.coords)), "} with n_steps ",
                cl.n_steps, ", ", length(cl.coords), " frames recorded for ",
                length(first(cl.coords)), " atoms")
end

function log_property!(logger::CoordinateLogger, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        push!(logger.coords, deepcopy(s.coords))
    end
end

"""
    EnergyLogger(n_steps)

Log the energy of the system throughout a simulation.
"""
struct EnergyLogger{T} <: Logger
    n_steps::Int
    energy::Vector{T}
end

EnergyLogger(T::Type, n_steps::Integer) = EnergyLogger(n_steps, T[])

function EnergyLogger(n_steps::Integer)
    return EnergyLogger(DefaultFloat, n_steps)
end

function Base.show(io::IO, el::EnergyLogger)
    print(io, "EnergyLogger{", eltype(el.energy), "} with n_steps ",
                el.n_steps, ", ", length(el.energy), " energies recorded")
end

function log_property!(logger::EnergyLogger, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        push!(logger.energy, energy(s))
    end
end

"""
    StructureWriter(n_steps, filepath)

Write 3D output structures to the PDB file format throughout a simulation.
"""
mutable struct StructureWriter <: Logger
    n_steps::Int
    filepath::String
    structure_n::Int
end

StructureWriter(n_steps::Integer, filepath::AbstractString) = StructureWriter(
        n_steps, filepath, 1)

function Base.show(io::IO, sw::StructureWriter)
    print(io, "StructureWriter with n_steps ", sw.n_steps, ", filepath \"",
                sw.filepath, "\", ", sw.structure_n - 1, " frames written")
end

function log_property!(logger::StructureWriter, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        append_model(logger, s)
        logger.structure_n += 1
    end
end

function append_model(logger::StructureWriter, s::Simulation)
    open(logger.filepath, "a") do output
        println(output, "MODEL     ", lpad(logger.structure_n, 4))
        for (i, coord) in enumerate(s.coords)
            at = s.atoms[i]
            at_rec = AtomRecord(
                false, i, at.name, ' ', at.resname, "A", at.resnum,
                ' ', 10 .* coord, 1.0, 0.0, "  ", "  "
            )
            println(output, pdbline(at_rec))
        end
        println(output, "ENDMDL")
    end
end
