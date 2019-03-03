# Types

export
    Interaction,
    GeneralInteraction,
    SpecificInteraction,
    Logger,
    NeighbourFinder,
    Thermostat,
    Simulator,
    Simulation

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
mutable struct Simulation
    simulator::Simulator
    atoms::Vector{Atom}
    specific_inter_lists::Dict{String, Vector{T} where T <: SpecificInteraction}
    general_inters::Dict{String, GeneralInteraction}
    coords::Vector{Coordinates}
    velocities::Vector{Velocity}
    temperature::Float64
    box_size::Float64
    neighbour_list::Vector{Tuple{Int, Int}}
    neighbour_finder::NeighbourFinder
    thermostat::Thermostat
    loggers::Vector{Logger}
    timestep::Float64
    n_steps::Int
    n_steps_made::Int
end
