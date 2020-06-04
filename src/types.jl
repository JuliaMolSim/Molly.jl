# Types

export
    Interaction,
    GeneralInteraction,
    SpecificInteraction,
    Simulator,
    Thermostat,
    NeighbourFinder,
    Logger,
    Atom,
    Simulation

"An interaction between atoms that contributes to forces on the atoms."
abstract type Interaction end

"A general interaction that will apply to all atom pairs."
abstract type GeneralInteraction <: Interaction end

"A specific interaction between sets of specific atoms, e.g. a bond angle."
abstract type SpecificInteraction <: Interaction end

"A type of simulation to run, e.g. leap-frog integration or energy minimisation."
abstract type Simulator end

"A way to keep the temperature of a simulation constant."
abstract type Thermostat end

"A way to find near atoms to save on simulation time."
abstract type NeighbourFinder end

"A way to record a property, e.g. the temperature, throughout a simulation."
abstract type Logger end

"An atom and its associated information."
struct Atom{T}
    attype::String
    name::String
    resnum::Int
    resname::String
    charge::T
    mass::T
    σ::T
    ϵ::T
end

# We define constructors rather than using Base.@kwdef as it makes conversion
#   more convenient with the parametric type
function Atom(;
                attype="",
                name="",
                resnum=0,
                resname="",
                charge=0.0,
                mass=0.0,
                σ=0.0,
                ϵ=0.0)
    return Atom{typeof(charge)}(attype, name, resnum, resname, charge, mass, σ, ϵ)
end

"The data associated with a molecular simulation."
struct Simulation{T}
    simulator::Simulator
    atoms::Vector{<:Any}
    specific_inter_lists::Dict{String, Vector{<:SpecificInteraction}}
    general_inters::Dict{String, <:GeneralInteraction}
    coords::T # Typically a vector of static vectors
    velocities::T
    temperature::Float64
    box_size::Float64
    neighbour_finder::NeighbourFinder
    thermostat::Thermostat
    loggers::Dict{String, <:Logger}
    timestep::Float64
    n_steps::Int
    n_steps_made::Vector{Int} # This is a vector to keep the struct immutable
end

function Simulation(;
                    simulator,
                    atoms,
                    specific_inter_lists=Dict(),
                    general_inters=Dict(),
                    coords,
                    velocities,
                    temperature,
                    box_size,
                    neighbour_finder=NoNeighbourFinder(),
                    thermostat=NoThermostat(),
                    loggers=Dict(),
                    timestep,
                    n_steps,
                    n_steps_made=[0])
    return Simulation{typeof(coords)}(simulator, atoms, specific_inter_lists,
                general_inters, coords, velocities, temperature, box_size,
                neighbour_finder, thermostat, loggers, timestep, n_steps,
                n_steps_made)
end
