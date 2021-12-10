# Loggers to record properties throughout a simulation

export
    run_loggers!,
    TemperatureLogger,
    log_property!,
    CoordinateLogger,
    VelocityLogger,
    EnergyLogger,
    StructureWriter

"""
    run_loggers!(system, neighbors=nothing, step_n=0)

Run the loggers associated with the system.
"""
function run_loggers!(s::System, neighbors=nothing, step_n::Integer=0)
    for logger in values(s.loggers)
        log_property!(logger, s, neighbors, step_n)
    end
end

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

TemperatureLogger(n_steps::Integer) = TemperatureLogger(typeof(one(DefaultFloat)u"K"), n_steps)

function Base.show(io::IO, tl::TemperatureLogger)
    print(io, "TemperatureLogger{", eltype(tl.temperatures), "} with n_steps ",
                tl.n_steps, ", ", length(tl.temperatures),
                " temperatures recorded")
end

"""
    log_property!(logger, system, neighbors=nothing, step_n=0)

Log a property of the system thoughout a simulation.
Custom loggers should implement this function.
"""
function log_property!(logger::TemperatureLogger, s::System, neighbors=nothing, step_n::Integer=0)
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
    return CoordinateLogger(typeof(one(DefaultFloat)u"nm"), n_steps; dims=dims)
end

function Base.show(io::IO, cl::CoordinateLogger)
    print(io, "CoordinateLogger{", eltype(eltype(cl.coords)), "} with n_steps ",
            cl.n_steps, ", ", length(cl.coords), " frames recorded for ",
            length(cl.coords) > 0 ? length(first(cl.coords)) : "?", " atoms")
end

function log_property!(logger::CoordinateLogger, s::System, neighbors=nothing, step_n::Integer=0)
    if step_n % logger.n_steps == 0
        push!(logger.coords, deepcopy(s.coords))
    end
end

"""
    VelocityLogger(n_steps; dims=3)

Log the velocities throughout a simulation.
"""
struct VelocityLogger{T} <: Logger
    n_steps::Int
    velocities::Vector{Vector{T}}
end

function VelocityLogger(T, n_steps::Integer; dims::Integer=3)
    return VelocityLogger(n_steps,
                            Array{SArray{Tuple{dims}, T, 1, dims}, 1}[])
end

function VelocityLogger(n_steps::Integer; dims::Integer=3)
    return VelocityLogger(typeof(one(DefaultFloat)u"nm * ps^-1"), n_steps; dims=dims)
end

function Base.show(io::IO, vl::VelocityLogger)
    print(io, "VelocityLogger{", eltype(eltype(vl.velocities)), "} with n_steps ",
            vl.n_steps, ", ", length(vl.velocities), " frames recorded for ",
            length(vl.velocities) > 0 ? length(first(vl.velocities)) : "?", " atoms")
end

function log_property!(logger::VelocityLogger, s::System, neighbors=nothing, step_n::Integer=0)
    if step_n % logger.n_steps == 0
        push!(logger.velocities, deepcopy(s.velocities))
    end
end

"""
    EnergyLogger(n_steps)

Log the total energy of the system throughout a simulation.
"""
struct EnergyLogger{T} <: Logger
    n_steps::Int
    energies::Vector{T}
end

EnergyLogger(T::Type, n_steps::Integer) = EnergyLogger(n_steps, T[])

function EnergyLogger(n_steps::Integer)
    return EnergyLogger(typeof(one(DefaultFloat)u"kJ * mol^-1"), n_steps)
end

function Base.show(io::IO, el::EnergyLogger)
    print(io, "EnergyLogger{", eltype(el.energies), "} with n_steps ",
                el.n_steps, ", ", length(el.energies), " energies recorded")
end

function log_property!(logger::EnergyLogger, s::System, neighbors=nothing, step_n::Integer=0)
    if step_n % logger.n_steps == 0
        push!(logger.energies, energy(s, neighbors))
    end
end

"""
    StructureWriter(n_steps, filepath, excluded_res=String[])

Write 3D output structures to the PDB file format throughout a simulation.
"""
mutable struct StructureWriter <: Logger
    n_steps::Int
    filepath::String
    excluded_res::Set{String}
    structure_n::Int
end

function StructureWriter(n_steps::Integer, filepath::AbstractString, excluded_res=String[])
    return StructureWriter(n_steps, filepath, Set(excluded_res), 1)
end

function Base.show(io::IO, sw::StructureWriter)
    print(io, "StructureWriter with n_steps ", sw.n_steps, ", filepath \"",
                sw.filepath, "\", ", sw.structure_n - 1, " frames written")
end

function log_property!(logger::StructureWriter, s::System, neighbors=nothing, step_n::Integer=0)
    if step_n % logger.n_steps == 0
        if length(s) != length(s.atoms_data)
            error("Number of atoms is ", length(s), " but number of atom data entries is ",
                    length(s.atoms_data))
        end
        append_model(logger, s)
        logger.structure_n += 1
    end
end

function append_model(logger::StructureWriter, s::System)
    open(logger.filepath, "a") do output
        println(output, "MODEL     ", lpad(logger.structure_n, 4))
        for (i, coord) in enumerate(s.coords)
            atom_data = s.atoms_data[i]
            if unit(first(coord)) == NoUnits
                coord_convert = 10 .* coord # If not told, assume coordinates are in nm
            else
                coord_convert = ustrip.(u"Å", coord)
            end
            if !(atom_data.res_name in logger.excluded_res)
                at_rec = atomrecord(atom_data, i, coord_convert)
                println(output, pdbline(at_rec))
            end
        end
        println(output, "ENDMDL")
    end
end

atomrecord(at_data, i, coord) = AtomRecord(
    false, i, at_data.atom_name, ' ', at_data.res_name, "A",
    at_data.res_number, ' ', coord, 1.0, 0.0,
    at_data.element == "?" ? "  " : at_data.element, "  "
)
