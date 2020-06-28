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

const DefaultFloat = Float64

"An interaction between atoms that contributes to forces on the atoms."
abstract type Interaction end

"""
A general interaction that will apply to all atom pairs.
Custom general interactions should sub-type this type.
"""
abstract type GeneralInteraction <: Interaction end

"""
A specific interaction between sets of specific atoms, e.g. a bond angle.
Custom specific interactions should sub-type this type.
"""
abstract type SpecificInteraction <: Interaction end

"""
A type of simulation to run, e.g. leap-frog integration or energy minimisation.
Custom simulators should sub-type this type.
"""
abstract type Simulator end

"""
A way to keep the temperature of a simulation constant.
Custom thermostats should sub-type this type.
"""
abstract type Thermostat end

"""
A way to find near atoms to save on simulation time.
Custom neighbour finders should sub-type this type.
"""
abstract type NeighbourFinder end

"""
A way to record a property, e.g. the temperature, throughout a simulation.
Custom loggers should sub-type this type.
"""
abstract type Logger end

"""
    Atom(; <keyword arguments>)

An atom and its associated information.
Properties unused in the simulation or in analysis can be left with their
default values.

# Arguments
- `attype::AbstractString=""`: the type of the atom.
- `name::AbstractString=""`: the name of the atom.
- `resnum::Integer=0`: the residue number if the atom is part of a polymer.
- `resname::AbstractString=""`: the residue name if the atom is part of a
    polymer.
- `charge::T=0.0`: the charge of the atom, used for electrostatic interactions.
- `mass::T=0.0`: the mass of the atom.
- `σ::T=0.0`: the Lennard-Jones finite distance at which the inter-particle
    potential is zero.
- `ϵ::T=0.0`: the Lennard-Jones depth of the potential well.
"""
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

"""
    Simulation(; <keyword arguments>)

The data needed to define and run a molecular simulation.
Properties unused in the simulation or in analysis can be left with their
default values.

# Arguments
- `simulator::Simulator`: the type of simulation to run.
- `atoms::Vector{A}`: the atoms, or atom equivalents, in the simulation. Can be
    of any type.
- `specific_inter_lists::SI=()`: the specific interactions in the simulation,
    i.e. interactions between specific atoms such as bonds or angles. Typically
    a `Tuple`.
- `general_inters::GI=()`: the general interactions in the simulation, i.e.
    interactions between all or most atoms such as electrostatics. Typically a
    `Tuple`.
- `coords::C`: the coordinates of the atoms in the simulation. Typically a
    `Vector` of `SVector`s of any dimension and type `T`, where `T` is an
    `AbstractFloat` type.
- `velocities::C`: the velocities of the atoms in the simulation, which should
    be the same type as the coordinates. The meaning of the velocities depends
    on the simulator used, e.g. for the `VelocityFreeVerlet` simulator they
    represent the previous step coordinates for the first step.
- `temperature::T=0.0`: the temperature of the simulation.
- `box_size::T`: the size of the cube in which the simulation takes place.
- `neighbour_finder::NeighbourFinder=NoNeighbourFinder()`: the neighbour finder
    used to find close atoms and save on computation.
- `thermostat::Thermostat=NoThermostat()`: the thermostat which applies during
    the simulation.
- `loggers::Dict{String, <:Logger}=Dict()`: the loggers that record properties
    of interest during the simulation.
- `timestep::T`: the timestep of the simulation.
- `n_steps::Integer`: the number of steps in the simulation.
- `n_steps_made::Vector{Int}=[]`: the number of steps already made during the
    simulation. This is a `Vector` to allow the `struct` to be immutable.
"""
struct Simulation{T, A, C, GI, SI}
    simulator::Simulator
    atoms::Vector{A}
    specific_inter_lists::SI
    general_inters::GI
    coords::C
    velocities::C
    temperature::T
    box_size::T
    neighbour_finder::NeighbourFinder
    thermostat::Thermostat
    loggers::Dict{String, <:Logger}
    timestep::T
    n_steps::Int
    n_steps_made::Vector{Int}
end

function Simulation(;
                    simulator,
                    atoms,
                    specific_inter_lists=(),
                    general_inters=(),
                    coords,
                    velocities,
                    temperature=0.0,
                    box_size,
                    neighbour_finder=NoNeighbourFinder(),
                    thermostat=NoThermostat(),
                    loggers=Dict{String, Logger}(),
                    timestep,
                    n_steps,
                    n_steps_made=[0])
    T = typeof(timestep)
    A = eltype(atoms)
    C = typeof(coords)
    GI = typeof(general_inters)
    SI = typeof(specific_inter_lists)
    return Simulation{T, A, C, GI, SI}(
                simulator, atoms, specific_inter_lists, general_inters, coords,
                velocities, temperature, box_size, neighbour_finder, thermostat,
                loggers, timestep, n_steps, n_steps_made)
end
