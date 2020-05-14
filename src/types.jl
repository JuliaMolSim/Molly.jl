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
struct Atom
    attype::String
    name::String
    resnum::Int
    resname::String
    charge::Float64
    mass::Float64
    σ::Float64
    ϵ::Float64
end

function Atom(;
            attype="",
            name="",
            resnum=0,
            resname="",
            charge=0.0,
            mass=0.0,
            σ=0.0,
            ϵ=0.0)
    return Atom(attype, name, resnum, resname, charge, mass, σ, ϵ)
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
mutable struct Simulation
    simulator::Simulator
    atoms::Vector{Atom}
    specific_inter_lists::Dict{String, Vector{S} where S <: SpecificInteraction}
    general_inters::Dict{String, GeneralInteraction}
    coords::Array{SArray{Tuple{3}, Float64, 1, 3}, 1}
    velocities::Array{SArray{Tuple{3}, Float64, 1, 3}, 1}
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

function Simulation(;
                    simulator,
                    atoms,
                    specific_inter_lists=Dict(),
                    general_inters=Dict(),
                    coords,
                    velocities,
                    temperature,
                    box_size,
                    neighbour_list=[],
                    neighbour_finder=NoNeighbourFinder(),
                    thermostat=NoThermostat(),
                    loggers=[],
                    timestep,
                    n_steps,
                    n_steps_made=0)
    return Simulation(simulator, atoms, specific_inter_lists,
                general_inters, coords, velocities, temperature,
                box_size, neighbour_list, neighbour_finder,
                thermostat, loggers, timestep, n_steps,
                n_steps_made)
end
