# Loggers to record properties throughout a simulation

export
    TemperatureLogger,
    log_property!,
    CoordinateLogger,
    StructureWriter,
    append_model

"Log the temperature throughout a simulation."
struct TemperatureLogger <: Logger
    n_steps::Int
    temperatures::Vector{Float64}
end

TemperatureLogger(n_steps::Integer) = TemperatureLogger(n_steps, [])

"Log a property thoughout a simulation."
function log_property!(logger::TemperatureLogger, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        push!(logger.temperatures, temperature(s))
    end
end

"Log the coordinates throughout a simulation."
struct CoordinateLogger <: Logger
    n_steps::Int
    coords::Vector{Vector{Coordinates}}
end

CoordinateLogger(n_steps::Integer) = CoordinateLogger(n_steps, [])

function log_property!(logger::CoordinateLogger, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        push!(logger.coords, deepcopy(s.coords))
    end
end

"Write output structures to the PDB file format throughout a simulation."
mutable struct StructureWriter <: Logger
    n_steps::Int
    filepath::String
    structure_n::Int
end

StructureWriter(n_steps::Integer, filepath::AbstractString) = StructureWriter(
        n_steps, filepath, 1)

function log_property!(logger::StructureWriter, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        append_model(logger, s)
        logger.structure_n += 1
    end
end

function append_model(logger::StructureWriter, s::Simulation)
    open(logger.filepath, "a") do output
        println(output, "MODEL     ", lpad(logger.structure_n, 4))
        for (i, c) in enumerate(s.coords)
            at = s.atoms[i]
            at_rec = AtomRecord(
                false,
                i,
                at.name,
                ' ',
                at.resname,
                "A",
                at.resnum,
                ' ',
                [10 * c.x, 10 * c.y, 10 * c.z],
                1.0,
                0.0,
                "  ",
                "  "
            )
            println(output, pdbline(at_rec))
        end
        println(output, "ENDMDL")
    end
end
