# Types

export
    Interaction,
    GeneralInteraction,
    SpecificInteraction,
    AbstractCutoff,
    Simulator,
    AbstractCoupler,
    NeighborFinder,
    Logger,
    Atom,
    mass,
    AtomData,
    NeighborList,
    NeighborListVec,
    Simulation

const DefaultFloat = Float64

"An interaction between atoms that contributes to forces on the atoms."
abstract type Interaction end

"""
A general interaction that will apply to all or most atom pairs.
Custom general interactions should sub-type this type.
"""
abstract type GeneralInteraction <: Interaction end

"""
A specific interaction between sets of specific atoms, e.g. a bond angle.
Custom specific interactions should sub-type this type.
"""
abstract type SpecificInteraction <: Interaction end

"""
A general type of cutoff encoding the approximation used for a potential.
Interactions can be parameterized by the cutoff behavior.
"""
abstract type AbstractCutoff end

"""
A type of simulation to run, e.g. leap-frog integration or energy minimisation.
Custom simulators should sub-type this type.
"""
abstract type Simulator end

"""
A way to keep properties of a simulation constant.
Custom temperature and pressure couplers should sub-type this type.
"""
abstract type AbstractCoupler end

"""
A way to find near atoms to save on simulation time.
Custom neighbor finders should sub-type this type.
"""
abstract type NeighborFinder end

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
The types used should be bits types if the GPU is going to be used.

# Arguments
- `index::Int`: the index of the atom in the system.
- `charge::C=0.0`: the charge of the atom, used for electrostatic interactions.
- `mass::M=0.0u"u"`: the mass of the atom.
- `σ::S=0.0u"nm"`: the Lennard-Jones finite distance at which the inter-particle
    potential is zero.
- `ϵ::E=0.0u"kJ * mol^-1"`: the Lennard-Jones depth of the potential well.
"""
struct Atom{C, M, S, E}
    index::Int
    charge::C
    mass::M
    σ::S
    ϵ::E
end

function Atom(;
                index=1,
                charge=0.0,
                mass=0.0u"u",
                σ=0.0u"nm",
                ϵ=0.0u"kJ * mol^-1")
    return Atom(index, charge, mass, σ, ϵ)
end

"""
    mass(atom)

The mass of an atom.
"""
mass(atom::Atom) = atom.mass

function Base.show(io::IO, a::Atom)
    print(io, "Atom with index ", a.index, ", charge=", a.charge,
            ", mass=", a.mass, ", σ=", a.σ, ", ϵ=", a.ϵ)
end

"""
    AtomData(atom_type, atom_name, res_number, res_name)

Data associated with an atom.
Storing this separately allows the atom types to be bits types and hence
work on the GPU.
"""
struct AtomData
    atom_type::String
    atom_name::String
    res_number::Int
    res_name::String
    element::String
end

function AtomData(;
                    atom_type="?",
                    atom_name="?",
                    res_number=1,
                    res_name="???",
                    element="?")
    return AtomData(atom_type, atom_name, res_number, res_name, element)
end

"""
    NeighborList()
    NeighborList(n, list)

Structure to contain pre-allocated neighbor lists.
"""
mutable struct NeighborList
    n::Int # Number of neighbors in list (n <= length(list))
    list::Vector{Tuple{Int, Int, Bool}}
end

NeighborList() = NeighborList(0, [])

function Base.empty!(nl::NeighborList)
    nl.n = 0
    return nl
end

function Base.push!(nl::NeighborList, element::Tuple{Int, Int, Bool})
    nl.n += 1
    if nl.n > length(nl.list)
        push!(nl.list, element)
    else
        nl.list[nl.n] = element
    end
    return nl
end

function Base.append!(nl::NeighborList, list::AbstractVector{Tuple{Int, Int, Bool}})
    for element in list
        push!(nl, element)
    end
    return nl
end

Base.append!(nl::NeighborList, nl_app::NeighborList) = append!(nl, @view(nl_app.list[1:nl_app.n]))

"""
    NeighborListVec(n, list)

Structure to contain neighbor lists for broadcasting.
"""
struct NeighborListVec{T}
    nbsi::Vector{Int} # Sorted ascending
    nbsj::Vector{Int}
    atom_bounds_i::Vector{Int}
    atom_bounds_j::Vector{Int}
    sortperm_j::Vector{Int}
    weights_14::T
end

"""
    Simulation(; <keyword arguments>)

The data needed to define and run a molecular simulation.
Properties unused in the simulation or in analysis can be left with their
default values.

# Arguments
- `simulator::Simulator`: the type of simulation to run.
- `atoms::A`: the atoms, or atom equivalents, in the simulation. Can be
    of any type.
- `atoms_data::AD`: data associated with the atoms, allowing the atoms to be
    bits types and hence work on the GPU.
- `specific_inter_lists::SI=()`: the specific interactions in the simulation,
    i.e. interactions between specific atoms such as bonds or angles. Typically
    a `Tuple`.
- `general_inters::GI=()`: the general interactions in the simulation, i.e.
    interactions between all or most atoms such as electrostatics. Typically a
    `Tuple`.
- `coords::C`: the coordinates of the atoms in the simulation. Typically a
    `Vector` of `SVector`s of any dimension and type `T`, where `T` is an
    `AbstractFloat` type.
- `velocities::V=zero(coords)`: the velocities of the atoms in the simulation,
    which should be the same type as the coordinates. The meaning of the
    velocities depends on the simulator used, e.g. for the `VelocityFreeVerlet`
    simulator they represent the previous step coordinates for the first step.
- `box_size::B`: the size of the box in which the simulation takes place.
- `neighbor_finder::NeighborFinder=NoNeighborFinder()`: the neighbor finder
    used to find close atoms and save on computation.
- `coupling::AbstractCoupler=NoCoupling()`: the coupling which applies during
    the simulation.
- `loggers::Dict{String, <:Logger}=Dict()`: the loggers that record properties
    of interest during the simulation.
- `timestep::S=0.0`: the timestep of the simulation.
- `n_steps::Integer=0`: the number of steps in the simulation.
- `force_unit::F=u"kJ * mol^-1 * nm^-1"`: the unit of force used in the
    simulation.
- `energy_unit::E=u"kJ * mol^-1"`: the unit of energy used in the simulation.
- `gpu_diff_safe::Bool`: whether to use the GPU implementation. Defaults to
    `isa(coords, CuArray)`.
"""
mutable struct Simulation{D, A, AD, C, V, GI, SI, B, S, F, E, NF, CO}
    simulator::Simulator
    atoms::A
    atoms_data::AD
    specific_inter_lists::SI
    general_inters::GI
    coords::C
    velocities::V
    box_size::B
    neighbor_finder::NF
    coupling::CO
    loggers::Dict{String, <:Logger}
    timestep::S
    n_steps::Int
    n_steps_made::Int
    force_unit::F
    energy_unit::E
end

Simulation{D}(args...) where {D, A, AD, C, V, GI, SI, B, S, F, E, NF, CO} = 
    Simulation{D, A, AD, C, V, GI, SI, B, S, F, E, NF, CO}(args...)

function Simulation(;
                    simulator=VelocityVerlet(),
                    atoms,
                    atoms_data=[],
                    specific_inter_lists=(),
                    general_inters=(),
                    coords,
                    velocities=zero(coords),
                    box_size,
                    neighbor_finder=NoNeighborFinder(),
                    coupling=NoCoupling(),
                    loggers=Dict{String, Logger}(),
                    timestep=0.0u"ps",
                    n_steps=0,
                    n_steps_made=0,
                    force_unit=u"kJ * mol^-1 * nm^-1",
                    energy_unit=u"kJ * mol^-1",
                    gpu_diff_safe=isa(coords, CuArray))
    A = typeof(atoms)
    AD = typeof(atoms_data)
    C = typeof(coords)
    V = typeof(velocities)
    GI = typeof(general_inters)
    SI = typeof(specific_inter_lists)
    B = typeof(box_size)
    S = typeof(timestep)
    F = typeof(force_unit)
    E = typeof(energy_unit)
    NF = typeof(neighbor_finder)
    CO = typeof(coupling)
    return Simulation{gpu_diff_safe, A, AD, C, V, GI, SI, B, S, F, E, NF, CO}(
                simulator, atoms, atoms_data, specific_inter_lists, general_inters,
                coords, velocities, box_size, neighbor_finder, coupling, loggers,
                timestep, n_steps, n_steps_made, force_unit, energy_unit)
end

function Base.show(io::IO, s::Simulation)
    print(io, "Simulation with ", length(s.coords), " atoms, ",
                typeof(s.simulator), " simulator, ", s.timestep, " timestep, ",
                s.n_steps, " steps, ", first(s.n_steps_made), " steps made")
end
