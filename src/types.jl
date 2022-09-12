# Types

export
    PairwiseInteraction,
    SpecificInteraction,
    InteractionList1Atoms,
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
    ReplicaSystem,
    is_gpu_diff_safe,
    float_type,
    is_on_gpu,
    masses

const DefaultFloat = Float64

"""
A pairwise interaction that will apply to all or most atom pairs.
Custom pairwise interactions should sub-type this type.
"""
abstract type PairwiseInteraction end

"""
A specific interaction between sets of specific atoms, e.g. a bond angle.
Custom specific interactions should sub-type this type.
"""
abstract type SpecificInteraction end

"""
    InteractionList1Atoms(is, types, inters)
    InteractionList1Atoms(inter_type)

A list of specific interactions that involve one atom such as position restraints.
"""
struct InteractionList1Atoms{T}
    is::Vector{Int}
    types::Vector{String}
    inters::T
end

"""
    InteractionList2Atoms(is, js, types, inters)
    InteractionList2Atoms(inter_type)

A list of specific interactions that involve two atoms such as bond potentials.
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

A list of specific interactions that involve three atoms such as bond angle potentials.
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

A list of specific interactions that involve four atoms such as torsion potentials.
"""
struct InteractionList4Atoms{T}
    is::Vector{Int}
    js::Vector{Int}
    ks::Vector{Int}
    ls::Vector{Int}
    types::Vector{String}
    inters::T
end

InteractionList1Atoms(T) = InteractionList1Atoms{Vector{T}}([], [], T[])
InteractionList2Atoms(T) = InteractionList2Atoms{Vector{T}}([], [], [], T[])
InteractionList3Atoms(T) = InteractionList3Atoms{Vector{T}}([], [], [], [], T[])
InteractionList4Atoms(T) = InteractionList4Atoms{Vector{T}}([], [], [], [], [], T[])

interaction_type(::InteractionList1Atoms{T}) where {T} = eltype(T)
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
- `mass::M=1.0u"u"`: the mass of the atom.
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
                mass=1.0u"u",
                σ=0.0u"nm",
                ϵ=0.0u"kJ * mol^-1",
                solute=false)
    return Atom(index, charge, mass, σ, ϵ, solute)
end

"""
    charge(atom)

The partial charge of an [`Atom`](@ref).
"""
charge(atom) = atom.charge

"""
    mass(atom)

The mass of an [`Atom`](@ref).
Custom atom types should implement this function unless they have a `mass` field
defined, which the function accesses by default.
"""
mass(atom) = atom.mass

function Base.show(io::IO, a::Atom)
    print(io, "Atom with index ", a.index, ", charge=", charge(a),
            ", mass=", mass(a), ", σ=", a.σ, ", ϵ=", a.ϵ)
end

"""
    AtomData(atom_type, atom_name, res_number, res_name)

Data associated with an atom.
Storing this separately allows the [`Atom`](@ref) types to be bits types and hence
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

function Base.append!(nl::NeighborList, list::Base.AbstractVecOrTuple{Tuple{Int, Int, Bool}})
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

# Convert the Boltzmann constant k to suitable units and float type
function convert_k_units(T, k, energy_units)
    if energy_units == NoUnits
        if unit(k) == NoUnits
            # Use user-supplied unitless Boltzmann constant
            k_converted = T(k)
        else
            # Otherwise assume energy units are (u* nm^2 * ps^-2)
            k_converted = T(ustrip(u"u * nm^2 * ps^-2 * K^-1", k))
        end
    elseif dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        k_converted = T(uconvert(energy_units * u"mol * K^-1", k))
    else
        k_converted = T(uconvert(energy_units * u"K^-1", k))
    end
    return k_converted
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
- `constraints::CN=()`: the constraints for bonds and angles in the system. Typically
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
    GPU and taking gradients. Defaults to
    `isa(coords, AT) where AT <: Union{CuArray, ROCArray}`.
"""
mutable struct System{D, G, T, CU, A, AD, PI, SI, GI, CN, C, V, B, NF, L, F, E, K} <: AbstractSystem{D}
    atoms::A
    atoms_data::AD
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
    constraints::CN
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
                constraints=(),
                coords,
                velocities=nothing,
                boundary,
                neighbor_finder=NoNeighborFinder(),
                loggers=(),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
                k=Unitful.k,
                gpu_diff_safe=isa(coords, AT)) where AT <: Union{CuArray,
                                                                 ROCArray}
    D = n_dimensions(boundary)
    G = gpu_diff_safe
    T = float_type(boundary)
    CU = isa(coords, AT)
    A = typeof(atoms)
    AD = typeof(atoms_data)
    PI = typeof(pairwise_inters)
    SI = typeof(specific_inter_lists)
    GI = typeof(general_inters)
    CN = typeof(constraints)
    C = typeof(coords)
    B = typeof(boundary)
    NF = typeof(neighbor_finder)
    L = typeof(loggers)
    F = typeof(force_units)
    E = typeof(energy_units)

    if isnothing(velocities)
        if force_units == NoUnits
            vels = zero(coords)
        else
            vels = zero(coords) * u"ps^-1"
        end
    else
        vels = velocities
    end
    V = typeof(vels)

    if length(atoms) != length(coords)
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(coords)) coordinates"))
    end
    if length(atoms) != length(vels)
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(vels)) velocities"))
    end
    if length(atoms_data) > 0 && length(atoms) != length(atoms_data)
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(atoms_data)) atom data entries"))
    end

    if isa(atoms, AT) && !isa(coords, AT)
        throw(ArgumentError("The atoms are on the GPU but the coordinates are not"))
    end
    if isa(coords, AT) && !isa(atoms, AT)
        throw(ArgumentError("The coordinates are on the GPU but the atoms are not"))
    end
    if isa(atoms, AT) && !isa(vels, AT)
        throw(ArgumentError("The atoms are on the GPU but the velocities are not"))
    end
    if isa(vels, AT) && !isa(atoms, AT)
        throw(ArgumentError("The velocities are on the GPU but the atoms are not"))
    end

    k_converted = convert_k_units(T, k, energy_units)
    K = typeof(k_converted)

    return System{D, G, T, CU, A, AD, PI, SI, GI, CN, C, V, B, NF, L, F, E, K}(
                  atoms, atoms_data, pairwise_inters, specific_inter_lists,
                  general_inters, constraints, coords, vels, boundary, neighbor_finder,
                  loggers, force_units, energy_units, k_converted)
end

"""
    ReplicaSystem(; <keyword arguments>)

A wrapper for replicas in a replica exchange simulation.
Each individual replica is a [`System`](@ref).
Properties unused in the simulation or in analysis can be left with their default values.
`atoms`, `atoms_data` and the elements in `replica_coords` and `replica_velocities`
should have the same length.
The number of elements in `replica_coords`, `replica_velocities`, `replica_loggers` and 
the interaction arguments `replica_pairwise_inters`, `replica_specific_inter_lists`, 
`replica_general_inters` and `replica_constraints` should be equal to `n_replicas`.
This is a sub-type of `AbstractSystem` from AtomsBase.jl and implements the
interface described there.

When using `ReplicaSystem` with [`CellListMapNeighborFinder`](@ref), The number of threads used for both the simulation
of replicas and the neighbor finder should be set up to be same. This can be done by passing `nbatches=(min(n, 8), n)` to
`CellListMapNeighborFinder` during construction where `n` is the number of threads to be used per replica.

# Arguments
- `atoms::A`: the atoms, or atom equivalents, in the system. Can be
    of any type but should be a bits type if the GPU is used.
- `atoms_data::AD`: other data associated with the atoms, allowing the atoms to
    be bits types and hence work on the GPU.
- `pairwise_inters::PI=()`: the pairwise interactions in the system, i.e. interactions 
    between all or most atom pairs such as electrostatics (to be used if same for all replicas).
    Typically a `Tuple`. *Note: This is only used if no value is passed to the argument 
    `replica_pairwise_inters`*.
- `replica_pairwise_inters=[() for _ in 1:n_replicas]`: the pairwise interactions for 
    each replica.
- `specific_inter_lists::SI=()`: the specific interactions in the system, i.e. interactions 
    between specific atoms such as bonds or angles (to be used if same for all replicas). 
    Typically a `Tuple`. *Note: This is only used if no value is passed to the argument 
    `replica_specific_inter_lists`*.
- `replica_specific_inter_lists=[() for _ in 1:n_replicas]`: the specific interactions in 
    each replica.
- `general_inters::GI=()`: the general interactions in the system, i.e. interactions involving 
    all atoms such as implicit solvent (to be used if same for all replicas). Typically a `Tuple`. 
    *Note: This is only used if no value is passed to the argument `replica_general_inters`*.
- `replica_general_inters=[() for _ in 1:n_replicas]`: the general interactions for 
    each replica.
- `constraints::CN=()`: the constraints for bonds and angles in the system (to be used if same 
    for all replicas). Typically a `Tuple`.
- `replica_constraints=[() for _ in 1:n_replicas]`: the constraints for bonds and angles in each
    replica. *Note: This is only used if no value is passed to the argument `replica_constraints`*.
- `n_replicas::Integer`: the number of replicas of the system.
- `replica_coords`: the coordinates of the atoms in each replica.
- `replica_velocities=[zero(replica_coords[1]) * u"ps^-1" for _ in 1:n_replicas]`:
    the velocities of the atoms in each replica.
- `boundary::B`: the bounding box in which the simulation takes place.
- `neighbor_finder::NF=NoNeighborFinder()`: the neighbor finder used to find
    close atoms and save on computation.
- `exchange_logger::EL=ReplicaExchangeLogger(n_replicas)`: the logger used to record
    the exchange of replicas.
- `replica_loggers=[() for _ in 1:n_replicas]`: the loggers for each replica 
    that record properties of interest during a simulation.
- `force_units::F=u"kJ * mol^-1 * nm^-1"`: the units of force of the system.
    Should be set to `NoUnits` if units are not being used.
- `energy_units::E=u"kJ * mol^-1"`: the units of energy of the system. Should
    be set to `NoUnits` if units are not being used.
- `k::K=Unitful.k`: the Boltzmann constant, which may be modified in some
    simulations.
- `gpu_diff_safe::Bool`: whether to use the code path suitable for the
    GPU and taking gradients. Defaults to
    `isa(replica_coords[1], AT) where AT <: Union{CuArray, ROCArray}`.
"""
mutable struct ReplicaSystem{D, G, T, CU, A, AD, RS, B, EL, F, E, K} <: AbstractSystem{D}
    atoms::A
    atoms_data::AD
    n_replicas::Int
    replicas::RS
    boundary::B
    exchange_logger::EL
    force_units::F
    energy_units::E
    k::K
end

function ReplicaSystem(;
                        atoms,
                        atoms_data=[],
                        pairwise_inters=(),
                        replica_pairwise_inters=nothing,
                        specific_inter_lists=(),
                        replica_specific_inter_lists=nothing,
                        general_inters=(),
                        replica_general_inters=nothing,
                        constraints=(),
                        replica_constraints=nothing,
                        n_replicas,
                        replica_coords,
                        replica_velocities=nothing,
                        boundary,
                        neighbor_finder=NoNeighborFinder(),
                        exchange_logger=nothing,
                        replica_loggers=[() for _ in 1:n_replicas],
                        force_units=u"kJ * mol^-1 * nm^-1",
                        energy_units=u"kJ * mol^-1",
                        k=Unitful.k,
                        gpu_diff_safe=isa(replica_coords[1],
                                          AT)) where AT <: Union{CuArray,
                                                                 ROCArray}
    D = n_dimensions(boundary)
    G = gpu_diff_safe
    T = float_type(boundary)
    CU = isa(replica_coords[1], AT)
    A = typeof(atoms)
    AD = typeof(atoms_data)
    C = typeof(replica_coords[1])
    B = typeof(boundary)
    NF = typeof(neighbor_finder)
    F = typeof(force_units)
    E = typeof(energy_units)

    if isnothing(replica_velocities)
        if force_units == NoUnits
            replica_velocities = [zero(replica_coords[1]) for _ in 1:n_replicas]
        else
            replica_velocities = [zero(replica_coords[1]) * u"ps^-1" for _ in 1:n_replicas]
        end
    end
    V = typeof(replica_velocities[1])

    if isnothing(exchange_logger)
        exchange_logger = ReplicaExchangeLogger(T, n_replicas)
    end
    EL = typeof(exchange_logger)
    
    if isnothing(replica_pairwise_inters)
        replica_pairwise_inters = [pairwise_inters for _ in 1:n_replicas]
    elseif length(replica_pairwise_inters) != n_replicas
        throw(ArgumentError("Number of pairwise interactions ($(length(replica_pairwise_inters)))"
        * "does not match number of replicas ($(n_replicas))"))
    end

    if isnothing(replica_specific_inter_lists)
        replica_specific_inter_lists = [specific_inter_lists for _ in 1:n_replicas]
    elseif length(replica_specific_inter_lists) != n_replicas
        throw(ArgumentError("Number of specific interaction lists ($(length(replica_specific_inter_lists)))"
        * "does not match number of replicas ($(n_replicas))"))
    end

    if isnothing(replica_general_inters)
        replica_general_inters = [general_inters for _ in 1:n_replicas]
    elseif length(replica_general_inters) != n_replicas
        throw(ArgumentError("Number of general interactions ($(length(replica_general_inters)))"
        * "does not match number of replicas ($(n_replicas))"))
    end

    PI = eltype(replica_pairwise_inters)
    SI = eltype(replica_specific_inter_lists)
    GI = eltype(replica_general_inters)

    if isnothing(replica_constraints)
        replica_constraints = [constraints for _ in 1:n_replicas]
    elseif length(replica_constraints) != n_replicas
        throw(ArgumentError("Number of constraints ($(length(replica_general_inters)))"
        * "does not match number of replicas ($(n_replicas))"))
    end
    CN = eltype(replica_constraints)
    
    if !all(y -> typeof(y) == C, replica_coords)
        throw(ArgumentError("The coordinates for all the replicas are not of the same type"))
    end
    if !all(y -> typeof(y) == V, replica_velocities)
        throw(ArgumentError("The velocities for all the replicas are not of the same type"))
    end

    if length(replica_coords) != n_replicas
        throw(ArgumentError("There are $(length(replica_coords)) coordinates for replicas but $n_replicas replicas"))
    end
    if length(replica_velocities) != n_replicas
        throw(ArgumentError("There are $(length(replica_velocities)) velocities for replicas but $n_replicas replicas"))
    end
    if length(replica_loggers) != n_replicas
        throw(ArgumentError("There are $(length(replica_loggers)) loggers but $n_replicas replicas"))
    end

    if !all(y -> length(y) == length(replica_coords[1]), replica_coords)
        throw(ArgumentError("Some replicas have different number of coordinates"))
    end
    if !all(y -> length(y) == length(replica_velocities[1]), replica_velocities)
        throw(ArgumentError("Some replicas have different number of velocities"))
    end

    if length(atoms) != length(replica_coords[1])
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(replica_coords[1])) coordinates"))
    end
    if length(atoms) != length(replica_velocities[1])
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(replica_velocities[1])) velocities"))
    end
    if length(atoms_data) > 0 && length(atoms) != length(atoms_data)
        throw(ArgumentError("There are $(length(atoms)) atoms but $(length(atoms_data)) atom data entries"))
    end

    n_cuarray = sum(y -> isa(y, AT), replica_coords)
    if !(n_cuarray == n_replicas || n_cuarray == 0)
        throw(ArgumentError("The coordinates for $n_cuarray out of $n_replicas replicas are on GPU"))
    end
    if isa(atoms, AT) && n_cuarray != n_replicas
        throw(ArgumentError("The atoms are on the GPU but the coordinates are not"))
    end
    if n_cuarray == n_replicas && !isa(atoms, AT)
        throw(ArgumentError("The coordinates are on the GPU but the atoms are not"))
    end

    n_cuarray = sum(y -> isa(y, AT), replica_velocities)
    if !(n_cuarray == n_replicas || n_cuarray == 0)
        throw(ArgumentError("The velocities for $n_cuarray out of $n_replicas replicas are on GPU"))
    end
    if isa(atoms, AT) && n_cuarray != n_replicas
        throw(ArgumentError("The atoms are on the GPU but the velocities are not"))
    end
    if n_cuarray == n_replicas && !isa(atoms, AT)
        throw(ArgumentError("The velocities are on the GPU but the atoms are not"))
    end

    k_converted = convert_k_units(T, k, energy_units)
    K = typeof(k_converted)

    replicas = Tuple(System{D, G, T, CU, A, AD, PI, SI, GI, CN, C, V, B, NF, typeof(replica_loggers[i]), F, E, K}(
            atoms, atoms_data, replica_pairwise_inters[i], replica_specific_inter_lists[i],
            replica_general_inters[i], replica_constraints[i], replica_coords[i], 
            replica_velocities[i], boundary, deepcopy(neighbor_finder), replica_loggers[i],
            force_units, energy_units, k_converted) for i in 1:n_replicas)
    RS = typeof(replicas)

    return ReplicaSystem{D, G, T, CU, A, AD, RS, B, EL, F, E, K}(
            atoms, atoms_data, n_replicas, replicas, boundary, 
            exchange_logger, force_units, energy_units, k_converted)
end

"""
    is_gpu_diff_safe(sys)

Whether a [`System`](@ref) or [`ReplicaSystem`](@ref) uses the code path suitable
for the GPU and for taking gradients.
"""
is_gpu_diff_safe(::Union{System{D, G}, ReplicaSystem{D, G}}) where {D, G} = G

"""
    float_type(sys)
    float_type(boundary)

The float type a [`System`](@ref), [`ReplicaSystem`](@ref) or bounding box uses.
"""
float_type(::Union{System{D, G, T}, ReplicaSystem{D, G, T}}) where {D, G, T} = T

"""
    is_on_gpu(sys)

Whether a [`System`](@ref) or [`ReplicaSystem`](@ref) is on the GPU.
"""
is_on_gpu(::Union{System{D, G, T, CU}, ReplicaSystem{D, G, T, CU}}) where {D, G, T, CU} = CU

"""
    masses(sys)

The masses of the atoms in a [`System`](@ref) or [`ReplicaSystem`](@ref).
"""
masses(s::Union{System, ReplicaSystem}) = mass.(s.atoms)

# Move an array to the GPU depending on whether the system is on the GPU
move_array(arr, ::System{D, G, T, false}) where {D, G, T} = arr
move_array(arr::AT, ::System{D, G, T, true }) where {AT <: Union{CuArray,
                                                                 ROCArray},
                                                     D, G, T} = AT(arr)

AtomsBase.species_type(s::Union{System, ReplicaSystem}) = eltype(s.atoms)

Base.getindex(s::Union{System, ReplicaSystem}, i::Integer) = AtomView(s, i)
Base.length(s::Union{System, ReplicaSystem}) = length(s.atoms)

AtomsBase.position(s::System) = s.coords
AtomsBase.position(s::System, i::Integer) = s.coords[i]
AtomsBase.position(s::ReplicaSystem) = s.replicas[1].coords
AtomsBase.position(s::ReplicaSystem, i::Integer) = s.replicas[1].coords[i]

AtomsBase.velocity(s::System) = s.velocities
AtomsBase.velocity(s::System, i::Integer) = s.velocities[i]
AtomsBase.velocity(s::ReplicaSystem) = s.replicas[1].velocities
AtomsBase.velocity(s::ReplicaSystem, i::Integer) = s.replicas[1].velocities[i]

AtomsBase.atomic_mass(s::Union{System, ReplicaSystem}, i::Integer) = mass(s.atoms[i])
AtomsBase.atomic_symbol(s::Union{System, ReplicaSystem}, i::Integer) = Symbol(s.atoms_data[i].element)
AtomsBase.atomic_number(s::Union{System, ReplicaSystem}, i::Integer) = missing

AtomsBase.boundary_conditions(::Union{System{3}, ReplicaSystem{3}}) = SVector(Periodic(), Periodic(), Periodic())
AtomsBase.boundary_conditions(::Union{System{2}, ReplicaSystem{2}}) = SVector(Periodic(), Periodic())

AtomsBase.bounding_box(s::Union{System, ReplicaSystem}) = bounding_box(s.boundary)

function Base.show(io::IO, s::System)
    print(io, "System with ", length(s), " atoms, boundary ", s.boundary)
end

function Base.show(io::IO, s::ReplicaSystem)
    print(io, "ReplicaSystem containing ",  s.n_replicas, " replicas with ", length(s),
          " atoms, boundary ", s.boundary)
end
