# Loggers to record properties throughout a simulation

export
    Logger,
    TemperatureLogger,
    log_property!,
    StructureWriter,
    writepdb

"A way to record a property, e.g. the temperature, throughout a simulation."
abstract type Logger end

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

"Write output structures to the PDB file format throughout a simulation."
mutable struct StructureWriter <: Logger
    n_steps::Int
    out_dir::String
    structure_n::Int
end

StructureWriter(n_steps::Integer, out_dir::AbstractString) = StructureWriter(
        n_steps, out_dir, 1)

function log_property!(logger::StructureWriter, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        writepdb("$(logger.out_dir)/model_$(logger.structure_n).pdb", s)
        logger.structure_n += 1
    end
end

# Extension of method from BioStructures
function BioStructures.writepdb(filepath::AbstractString, s::Simulation)
    open(filepath, "w") do output
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
    end
end
