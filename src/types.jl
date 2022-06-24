# Types

export
    Interaction,
    PairwiseInteraction,
    SpecificInteraction,
    AbstractNeighborFinder,
    InteractionList2Atoms,
    InteractionList3Atoms,
    InteractionList4Atoms,
    Atom,
    charge,
    mass,
    AtomData,
    NeighborList,
    NeighborListVec,
    System,
    is_gpu_diff_safe,
    float_type

const DefaultFloat = Float64

"An interaction between atoms that contributes to forces on the atoms."
abstract type Interaction end

"""
A pairwise interaction that will apply to all or most atom pairs.
Custom pairwise interactions should sub-type this type.
"""
abstract type PairwiseInteraction <: Interaction end

"""
A specific interaction between sets of specific atoms, e.g. a bond angle.
Custom specific interactions should sub-type this type.
"""
abstract type SpecificInteraction <: Interaction end

"""
A way to find near atoms to save on simulation time.
Custom neighbor finders should sub-type this type.
"""
abstract type AbstractNeighborFinder end

"""
    InteractionList2Atoms(is, js, types, inters)
    InteractionList2Atoms(inter_type)

A list of specific interactions between two atoms.
"""
struct InteractionList2Atoms{T}
    is::Vector{Int}
    js::Vector{Int}
    types::Vector{String}
    inters::T
end

"""
    InteractionList3Atoms(is, js, ks, types, inters)
    InteractionList3Atoms(inter_type)

A list of specific interactions between three atoms.
"""
struct InteractionList3Atoms{T}
    is::Vector{Int}
    js::Vector{Int}
    ks::Vector{Int}
    types::Vector{String}
    inters::T
end

"""
    InteractionList4Atoms(is, js, ks, ls, types, inters)
    InteractionList4Atoms(inter_type)

A list of specific interactions between four atoms.
"""
struct InteractionList4Atoms{T}
    is::Vector{Int}
    js::Vector{Int}
    ks::Vector{Int}
    ls::Vector{Int}
    types::Vector{String}
    inters::T
end

InteractionList2Atoms(T) = InteractionList2Atoms{Vector{T}}([], [], [], T[])
InteractionList3Atoms(T) = InteractionList3Atoms{Vector{T}}([], [], [], [], T[])
InteractionList4Atoms(T) = InteractionList4Atoms{Vector{T}}([], [], [], [], [], T[])

interaction_type(::InteractionList2Atoms{T}) where {T} = eltype(T)
interaction_type(::InteractionList3Atoms{T}) where {T} = eltype(T)
interaction_type(::InteractionList4Atoms{T}) where {T} = eltype(T)

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
- `solute::Bool=false`: whether the atom is part of the solute.
"""
struct Atom{C, M, S, E}
    index::Int
    charge::C
    mass::M
    σ::S
    ϵ::E
    solute::Bool
end

function Atom(;
                index=1,
                charge=0.0,
                mass=0.0u"u",
                σ=0.0u"nm",
                ϵ=0.0u"kJ * mol^-1",
                solute=false)
    return Atom(index, charge, mass, σ, ϵ, solute)
end

"""
    charge(atom)

The partial charge of an atom.
"""
charge(atom::Atom) = atom.charge

"""
    mass(atom)

The mass of an atom.
"""
mass(atom::Atom) = atom.mass

function Base.show(io::IO, a::Atom)
    print(io, "Atom with index ", a.index, ", charge=", charge(a),
            ", mass=", mass(a), ", σ=", a.σ, ", ϵ=", a.ϵ)
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

struct NeighborsVec{T}
    nbsi::Vector{Int} # Sorted ascending
    nbsj::Vector{Int}
    atom_bounds_i::Vector{Int}
    atom_bounds_j::Vector{Int}
    sortperm_j::Vector{Int}
    weights_14::T
end

NeighborsVec() = NeighborsVec{Nothing}([], [], [], [], [], nothing)

"""
    NeighborListVec(close, all)

Structure to contain neighbor lists for broadcasting.
Each component may be present or absent depending on the interactions in
the system.
"""
struct NeighborListVec{C, A}
    close::NeighborsVec{C}
    all::NeighborsVec{A}
end

"""
    System(; <keyword arguments>)

A physical system to be simulated.
Properties unused in the simulation or in analysis can be left with their
default values.
`atoms`, `atoms_data`, `coords` and `velocities` should have the same length.
This is a sub-type of `AbstractSystem` from AtomsBase.jl and implements the
interface described there.

# Arguments
- `atoms::A`: the atoms, or atom equivalents, in the system. Can be
    of any type but should be a bits type if the GPU is used.
- `atoms_data::AD`: other data associated with the atoms, allowing the atoms to
    be bits types and hence work on the GPU.
- `pairwise_inters::PI=()`: the pairwise interactions in the system, i.e.
    interactions between all or most atom pairs such as electrostatics.
    Typically a `Tuple`.
- `specific_inter_lists::SI=()`: the specific interactions in the system,
    i.e. interactions between specific atoms such as bonds or angles. Typically
    a `Tuple`.
- `general_inters::GI=()`: the general interactions in the system,
    i.e. interactions involving all atoms such as implicit solvent. Typically
    a `Tuple`.
- `coords::C`: the coordinates of the atoms in the system. Typically a
    vector of `SVector`s of 2 or 3 dimensions.
- `velocities::V=zero(coords) * u"ps^-1"`: the velocities of the atoms in the
    system.
- `boundary::B`: the bounding box in which the simulation takes place.
- `neighbor_finder::NF=NoNeighborFinder()`: the neighbor finder used to find
    close atoms and save on computation.
- `loggers::L=()`: the loggers that record properties of interest during a
    simulation.
- `force_units::F=u"kJ * mol^-1 * nm^-1"`: the units of force of the system.
    Should be set to `NoUnits` if units are not being used.
- `energy_units::E=u"kJ * mol^-1"`: the units of energy of the system. Should
    be set to `NoUnits` if units are not being used.
- `k::K=Unitful.k`: the Boltzmann constant, which may be modified in some
    simulations.
- `gpu_diff_safe::Bool`: whether to use the code path suitable for the
    GPU and taking gradients. Defaults to `isa(coords, CuArray)`.
"""
mutable struct System{D, G, T, A, AD, PI, SI, GI, C, V, B, NF, L, F, E, K} <: AbstractSystem{D}
    atoms::A
    atoms_data::AD
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
    coords::C
    velocities::V
    boundary::B
    neighbor_finder::NF
    loggers::L
    force_units::F
    energy_units::E
    k::K
end

function System(;
                atoms,
                atoms_data=[],
                pairwise_inters=(),
                specific_inter_lists=(),
                general_inters=(),
                coords,
                velocities=zero(coords) * u"ps^-1",
                boundary,
                neighbor_finder=NoNeighborFinder(),
                loggers=(),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
                k=Unitful.k,
                gpu_diff_safe=isa(coords, CuArray))
    D = n_dimensions(boundary)
    G = gpu_diff_safe
    T = float_type(boundary)
    A = typeof(atoms)
    AD = typeof(atoms_data)
    PI = typeof(pairwise_inters)
    SI = typeof(specific_inter_lists)
    GI = typeof(general_inters)
    C = typeof(coords)
    V = typeof(velocities)
    B = typeof(boundary)
    NF = typeof(neighbor_finder)
    L = typeof(loggers)
    F = typeof(force_units)
    E = typeof(energy_units)

    if energy_units == NoUnits
        # Remove this ignore block once Unitful gradient support is added
        Zygote.ignore() do
            if unit(k) == NoUnits
                # Use user-supplied unitless Boltzmann constant
                k_converted = T(k)
            else
                # Otherwise assume energy units are (u* nm^2 * ps^-2)
                k_converted = T(ustrip(u"u * nm^2 * ps^-2 * K^-1", k))
            end
        end
    elseif dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        k_converted = T(uconvert(energy_units * u"mol * K^-1", k))
    else
        k_converted = T(uconvert(energy_units * u"K^-1", k))
    end
    
    K = typeof(k_converted)

    return System{D, G, T, A, AD, PI, SI, GI, C, V, B, NF, L, F, E, K}(
                    atoms, atoms_data, pairwise_inters, specific_inter_lists,
                    general_inters, coords, velocities, boundary, neighbor_finder,
                    loggers, force_units, energy_units,k_converted)
end

"""
    is_gpu_diff_safe(sys)

Whether a `System` uses the code path suitable for the GPU and
for taking gradients.
"""
is_gpu_diff_safe(::System{D, G}) where {D, G} = G

"""
    float_type(sys)
    float_type(boundary)

The float type a `System` or bounding box uses.
"""
float_type(::System{D, G, T}) where {D, G, T} = T

AtomsBase.species_type(s::System) = eltype(s.atoms)

Base.getindex(s::System, i::Integer) = AtomView(s, i)
Base.length(s::System) = length(s.atoms)

AtomsBase.position(s::System) = s.coords
AtomsBase.position(s::System, i::Integer) = s.coords[i]

AtomsBase.velocity(s::System) = s.velocities
AtomsBase.velocity(s::System, i::Integer) = s.velocities[i]

AtomsBase.atomic_mass(s::System, i::Integer) = mass(s.atoms[i])
AtomsBase.atomic_symbol(s::System, i::Integer) = Symbol(s.atoms_data[i].element)
AtomsBase.atomic_number(s::System, i::Integer) = missing

AtomsBase.boundary_conditions(::System{3}) = SVector(Periodic(), Periodic(), Periodic())
AtomsBase.boundary_conditions(::System{2}) = SVector(Periodic(), Periodic())

edges_to_box(bs::SVector{3}, z) = SVector{3}([
    SVector(bs[1], z    , z    ),
    SVector(z    , bs[2], z    ),
    SVector(z    , z    , bs[3]),
])
edges_to_box(bs::SVector{2}, z) = SVector{2}([
    SVector(bs[1], z    ),
    SVector(z    , bs[2]),
])

function AtomsBase.bounding_box(s::System)
    bs = s.boundary.side_lengths
    z = zero(bs[1])
    bb = edges_to_box(bs, z)
    return unit(z) == NoUnits ? (bb)u"nm" : bb # Assume nm without other information
end

function Base.show(io::IO, s::System)
    print(io, "System with ", length(s), " atoms, boundary ", s.boundary)
end
