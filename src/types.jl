# Types

export
    Atom,
    Interaction,
    GeneralInteraction,
    SpecificInteraction,
    Logger,
    NeighbourFinder,
    Thermostat,
    Simulator,
    Simulation

"An atom and its associated information."
Base.@kwdef struct Atom
    attype::String = ""
    name::String = ""
    resnum::Int = 0
    resname::String = ""
    charge::Float64 = 0.0
    mass::Float64 = 0.0
    σ::Float64 = 0.0
    ϵ::Float64 = 0.0
end

"An interaction between atoms that contributes to forces on the atoms."
abstract type Interaction end

"A general interaction that will apply to all atom pairs."
abstract type GeneralInteraction <: Interaction end

"A specific interaction between sets of specific atoms, e.g. a bond angle."
abstract type SpecificInteraction <: Interaction end

"A way to record a property, e.g. the temperature, throughout a simulation."
abstract type Logger end

"A way to find near atoms to save on simulation time."
abstract type NeighbourFinder end

"A way to keep the temperature of a simulation constant."
abstract type Thermostat end

"A type of simulation to run, e.g. leap-frog integration or energy minimisation."
abstract type Simulator end

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

# This constructor makes conversion more convenient with the parametric type
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
