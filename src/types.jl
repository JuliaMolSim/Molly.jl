# Types

export
    PairwiseInteraction,
    use_neighbors,
    SpecificInteraction,
    InteractionList1Atoms,
    InteractionList2Atoms,
    InteractionList3Atoms,
    InteractionList4Atoms,
    Atom,
    charge,
    mass,
    AtomData,
    MolecularTopology,
    NeighborList,
    System,
    ReplicaSystem,
    is_on_gpu,
    float_type,
    masses,
    charges,
    MollyCalculator

const DefaultFloat = Float64

"""
A pairwise interaction that will apply to all or most atom pairs.

Custom pairwise interactions should sub-type this abstract type.
"""
abstract type PairwiseInteraction end

"""
    use_neighbors(inter)

Whether a pairwise interaction uses the neighbor list, default `false`.

Custom pairwise interactions can define a method for this function.
For built-in interactions such as [`LennardJones`](@ref) this function accesses
the `use_neighbors` field of the struct.
"""
use_neighbors(::PairwiseInteraction) = false

"""
A specific interaction between sets of specific atoms, e.g. a bond angle.

Custom specific interactions should sub-type this abstract type.
"""
abstract type SpecificInteraction end

"""
    InteractionList1Atoms(is, inters)
    InteractionList1Atoms(is, inters, types)
    InteractionList1Atoms(inter_type)

A list of specific interactions that involve one atom such as position restraints.
"""
struct InteractionList1Atoms{I, T}
    is::I
    inters::T
    types::Vector{String}
end

"""
    InteractionList2Atoms(is, js, inters)
    InteractionList2Atoms(is, js, inters, types)
    InteractionList2Atoms(inter_type)

A list of specific interactions that involve two atoms such as bond potentials.
"""
struct InteractionList2Atoms{I, T}
    is::I
    js::I
    inters::T
    types::Vector{String}
end

"""
    InteractionList3Atoms(is, js, ks, inters)
    InteractionList3Atoms(is, js, ks, inters, types)
    InteractionList3Atoms(inter_type)

A list of specific interactions that involve three atoms such as bond angle potentials.
"""
struct InteractionList3Atoms{I, T}
    is::I
    js::I
    ks::I
    inters::T
    types::Vector{String}
end

"""
    InteractionList4Atoms(is, js, ks, ls, inters)
    InteractionList4Atoms(is, js, ks, ls, inters, types)
    InteractionList4Atoms(inter_type)

A list of specific interactions that involve four atoms such as torsion potentials.
"""
struct InteractionList4Atoms{I, T}
    is::I
    js::I
    ks::I
    ls::I
    inters::T
    types::Vector{String}
end

InteractionList1Atoms(is, inters) = InteractionList1Atoms(is, inters, fill("", length(is)))
InteractionList2Atoms(is, js, inters) = InteractionList2Atoms(is, js, inters, fill("", length(is)))
InteractionList3Atoms(is, js, ks, inters) = InteractionList3Atoms(is, js, ks, inters,
                                                                  fill("", length(is)))
InteractionList4Atoms(is, js, ks, ls, inters) = InteractionList4Atoms(is, js, ks, ls, inters,
                                                                      fill("", length(is)))

InteractionList1Atoms(T) = InteractionList1Atoms{Vector{Int32}, Vector{T}}([], T[], [])
InteractionList2Atoms(T) = InteractionList2Atoms{Vector{Int32}, Vector{T}}([], [], T[], [])
InteractionList3Atoms(T) = InteractionList3Atoms{Vector{Int32}, Vector{T}}([], [], [], T[], [])
InteractionList4Atoms(T) = InteractionList4Atoms{Vector{Int32}, Vector{T}}([], [], [], [], T[], [])

interaction_type(::InteractionList1Atoms{I, T}) where {I, T} = eltype(T)
interaction_type(::InteractionList2Atoms{I, T}) where {I, T} = eltype(T)
interaction_type(::InteractionList3Atoms{I, T}) where {I, T} = eltype(T)
interaction_type(::InteractionList4Atoms{I, T}) where {I, T} = eltype(T)

Base.length(inter_list::Union{InteractionList1Atoms, InteractionList2Atoms,
                              InteractionList3Atoms, InteractionList4Atoms}) = length(inter_list.is)

function Base.zero(inter_list::InteractionList1Atoms{I, T}) where {I, T}
    n_inters = length(inter_list)
    return InteractionList1Atoms{I, T}(
        fill(0, n_inters),
        zero.(inter_list.inters),
        fill("", n_inters),
    )
end

function Base.zero(inter_list::InteractionList2Atoms{I, T}) where {I, T}
    n_inters = length(inter_list)
    return InteractionList2Atoms{I, T}(
        fill(0, n_inters),
        fill(0, n_inters),
        zero.(inter_list.inters),
        fill("", n_inters),
    )
end

function Base.zero(inter_list::InteractionList3Atoms{I, T}) where {I, T}
    n_inters = length(inter_list)
    return InteractionList3Atoms{I, T}(
        fill(0, n_inters),
        fill(0, n_inters),
        fill(0, n_inters),
        zero.(inter_list.inters),
        fill("", n_inters),
    )
end

function Base.zero(inter_list::InteractionList4Atoms{I, T}) where {I, T}
    n_inters = length(inter_list)
    return InteractionList4Atoms{I, T}(
        fill(0, n_inters),
        fill(0, n_inters),
        fill(0, n_inters),
        fill(0, n_inters),
        zero.(inter_list.inters),
        fill("", n_inters),
    )
end

function Base.:+(il1::InteractionList1Atoms{I, T}, il2::InteractionList1Atoms{I, T}) where {I, T}
    return InteractionList1Atoms{I, T}(
        il1.is,
        il1.inters .+ il2.inters,
        il1.types,
    )
end

function Base.:+(il1::InteractionList2Atoms{I, T}, il2::InteractionList2Atoms{I, T}) where {I, T}
    return InteractionList2Atoms{I, T}(
        il1.is,
        il1.js,
        il1.inters .+ il2.inters,
        il1.types,
    )
end

function Base.:+(il1::InteractionList3Atoms{I, T}, il2::InteractionList3Atoms{I, T}) where {I, T}
    return InteractionList3Atoms{I, T}(
        il1.is,
        il1.js,
        il1.ks,
        il1.inters .+ il2.inters,
        il1.types,
    )
end

function Base.:+(il1::InteractionList4Atoms{I, T}, il2::InteractionList4Atoms{I, T}) where {I, T}
    return InteractionList4Atoms{I, T}(
        il1.is,
        il1.js,
        il1.ks,
        il1.ls,
        il1.inters .+ il2.inters,
        il1.types,
    )
end

"""
    Atom(; <keyword arguments>)

An atom and its associated information.

Properties unused in the simulation or in analysis can be left with their
default values.
The types used should be bits types if the GPU is going to be used.

# Arguments
- `index::Int`: the index of the atom in the system.
- `charge::C=0.0`: the charge of the atom, used for electrostatic interactions.
- `mass::M=1.0u"g/mol"`: the mass of the atom.
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
                mass=1.0u"g/mol",
                σ=0.0u"nm",
                ϵ=0.0u"kJ * mol^-1",
                solute=false)
    return Atom(index, charge, mass, σ, ϵ, solute)
end

function Base.zero(::Type{Atom{T, T, T, T}}) where T
    z = zero(T)
    return Atom(0, z, z, z, z, false)
end

function Base.:+(a1::Atom, a2::Atom)
    return Atom(0, a1.charge + a2.charge, a1.mass + a2.mass, a1.σ + a2.σ, a1.ϵ + a2.ϵ, false)
end

function Base.:-(a1::Atom, a2::Atom)
    return Atom(0, a1.charge - a2.charge, a1.mass - a2.mass, a1.σ - a2.σ, a1.ϵ - a2.ϵ, false)
end

Base.getindex(at::Atom, x::Symbol) = hasfield(Atom, x) ? getfield(atom, x) : KeyError("no field $x in Atom")

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
    MolecularTopology(bond_is, bond_js, n_atoms)
    MolecularTopology(atom_molecule_inds, molecule_atom_counts)

Topology information for a system.

Stores the index of the molecule each atom belongs to and the number of
atoms in each molecule.
"""
struct MolecularTopology
    atom_molecule_inds::Vector{Int32}
    molecule_atom_counts::Vector{Int32}
end

function bond_graph(bond_is, bond_js, n_atoms)
    g = SimpleGraph(n_atoms)
    for (i, j) in zip(bond_is, bond_js)
        add_edge!(g, i, j)
    end
    return g
end

function MolecularTopology(bond_is, bond_js, n_atoms)
    g = bond_graph(bond_is, bond_js, n_atoms)
    cc = connected_components(g)
    atom_molecule_inds = zeros(Int32, n_atoms)
    for (mi, atom_inds) in enumerate(cc)
        for ai in atom_inds
            atom_molecule_inds[ai] = mi
        end
    end
    molecule_atom_counts = length.(cc)
    return MolecularTopology(atom_molecule_inds, molecule_atom_counts)
end

"""
    NeighborList(n, list)
    NeighborList()

Structure to contain neighbor lists.
"""
mutable struct NeighborList{T}
    n::Int # Number of neighbors in list (n <= length(list))
    list::T
end

NeighborList() = NeighborList{Vector{Tuple{Int32, Int32, Bool}}}(0, [])

Base.length(nl::NeighborList) = nl.n
Base.getindex(nl::NeighborList, ind::Integer) = nl.list[ind]
Base.firstindex(::NeighborList) = 1
Base.lastindex(nl::NeighborList) = length(nl)
Base.eachindex(nl::NeighborList) = Base.OneTo(length(nl))

function Base.empty!(nl::NeighborList)
    nl.n = 0
    return nl
end

function Base.push!(nl::NeighborList, element)
    nl.n += 1
    if nl.n > length(nl.list)
        push!(nl.list, element)
    else
        nl.list[nl.n] = element
    end
    return nl
end

function Base.append!(nl::NeighborList, list)
    for element in list
        push!(nl, element)
    end
    return nl
end

Base.append!(nl::NeighborList, nl_app::NeighborList) = append!(nl, @view(nl_app.list[1:nl_app.n]))

# Placeholder struct to access N^2 interactions by indexing
struct NoNeighborList
    n_atoms::Int
end

n_atoms_to_n_pairs(n_atoms::Integer) = (n_atoms * (n_atoms - 1)) ÷ 2

Base.length(nl::NoNeighborList) = n_atoms_to_n_pairs(nl.n_atoms)

function pair_index(n_atoms::Integer, ind::Integer)
    kz = ind - 1
    iz = n_atoms - 2 - Int(floor(sqrt(-8 * kz + 4 * n_atoms * (n_atoms - 1) - 7) / 2 - 0.5))
    jz = kz + iz + 1 - (n_atoms * (n_atoms - 1)) ÷ 2 + ((n_atoms - iz) * ((n_atoms - iz) - 1)) ÷ 2
    i = iz + 1
    j = jz + 1
    return i, j
end

function Base.getindex(nl::NoNeighborList, ind::Integer)
    i, j = pair_index(nl.n_atoms, ind)
    return i, j, false
end

Base.firstindex(::NoNeighborList) = 1
Base.lastindex(nl::NoNeighborList) = length(nl)
Base.eachindex(nl::NoNeighborList) = Base.OneTo(length(nl))

CUDA.Const(nl::NoNeighborList) = nl

"""
    System(; <keyword arguments>)

A physical system to be simulated.

Properties unused in the simulation or in analysis can be left with their
default values.
The minimal required arguments are `atoms`, `coords` and `boundary`.
`atoms` and `coords` should have the same length, along with `velocities` and
`atoms_data` if these are provided.
This is a sub-type of `AbstractSystem` from AtomsBase.jl and implements the
interface described there.

# Arguments
- `atoms::A`: the atoms, or atom equivalents, in the system. Can be
    of any type but should be a bits type if the GPU is used.
- `coords::C`: the coordinates of the atoms in the system. Typically a
    vector of `SVector`s of 2 or 3 dimensions.
- `boundary::B`: the bounding box in which the simulation takes place.
- `velocities::V=zero(coords) * u"ps^-1"`: the velocities of the atoms in the
    system.
- `atoms_data::AD=[]`: other data associated with the atoms, allowing the atoms to
    be bits types and hence work on the GPU.
- `topology::TO=nothing`: topological information about the system such as which
    atoms are in the same molecule.
- `pairwise_inters::PI=()`: the pairwise interactions in the system, i.e.
    interactions between all or most atom pairs such as electrostatics.
    Typically a `Tuple`.
- `specific_inter_lists::SI=()`: the specific interactions in the system,
    i.e. interactions between specific atoms such as bonds or angles. Typically
    a `Tuple`.
- `general_inters::GI=()`: the general interactions in the system,
    i.e. interactions involving all atoms such as implicit solvent. Each should
    implement the AtomsCalculators.jl interface. Typically a `Tuple`.
- `constraints::CN=()` : The constraint algorithm(s) used to apply
    bond and angle constraints to the system. (e.g. SHAKE_RATTLE)
- `neighbor_finder::NF=NoNeighborFinder()`: the neighbor finder used to find
    close atoms and save on computation.
- `loggers::L=()`: the loggers that record properties of interest during a
    simulation.
- `force_units::F=u"kJ * mol^-1 * nm^-1"`: the units of force of the system.
    Should be set to `NoUnits` if units are not being used.
- `energy_units::E=u"kJ * mol^-1"`: the units of energy of the system. Should
    be set to `NoUnits` if units are not being used.
- `k::K=Unitful.k` or `Unitful.k * Unitful.Na`: the Boltzmann constant, which may be
    modified in some simulations. `k` is chosen based on the `energy_units` given.
- `data::DA=nothing`: arbitrary data associated with the system.
"""
mutable struct System{D, G, T, A, C, B, V, AD, TO, PI, SI, GI, CN, NF,
                      L, F, E, K, M, DA} <: AbstractSystem{D}
    atoms::A
    coords::C
    boundary::B
    velocities::V
    atoms_data::AD
    topology::TO
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
    constraints::CN
    neighbor_finder::NF
    loggers::L
    df::Int
    force_units::F
    energy_units::E
    k::K
    masses::M
    data::DA
end

function System(;
                atoms,
                coords,
                boundary,
                velocities=nothing,
                atoms_data=[],
                topology=nothing,
                pairwise_inters=(),
                specific_inter_lists=(),
                general_inters=(),
                constraints=(),
                neighbor_finder=NoNeighborFinder(),
                loggers=(),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
                k=default_k(energy_units),
                data=nothing)
    D = n_dimensions(boundary)
    G = isa(coords, CuArray)
    T = float_type(boundary)
    A = typeof(atoms)
    C = typeof(coords)
    B = typeof(boundary)
    AD = typeof(atoms_data)
    TO = typeof(topology)
    PI = typeof(pairwise_inters)
    SI = typeof(specific_inter_lists)
    GI = typeof(general_inters)
    CN = typeof(constraints)
    NF = typeof(neighbor_finder)
    L = typeof(loggers)
    F = typeof(force_units)
    E = typeof(energy_units)
    DA = typeof(data)


    if isnothing(velocities)
        if force_units == NoUnits
            vels = zero(coords)
        else
            # Assume time units are ps
            vels = zero(coords) * u"ps^-1"
        end
    else
        vels = velocities
    end
    V = typeof(vels)

    if length(atoms) != length(coords)
        throw(ArgumentError("there are $(length(atoms)) atoms but $(length(coords)) coordinates"))
    end
    if length(atoms) != length(vels)
        throw(ArgumentError("there are $(length(atoms)) atoms but $(length(vels)) velocities"))
    end
    if length(atoms_data) > 0 && length(atoms) != length(atoms_data)
        throw(ArgumentError("there are $(length(atoms)) atoms but $(length(atoms_data)) atom data entries"))
    end

    df = n_dof(D, length(atoms), boundary)
    if length(constraints) > 0
        for ca in constraints
            df -= n_dof_lost(D, ca.clusters)
        end
    end


    if isa(atoms, CuArray) && !isa(coords, CuArray)
        throw(ArgumentError("the atoms are on the GPU but the coordinates are not"))
    end
    if isa(coords, CuArray) && !isa(atoms, CuArray)
        throw(ArgumentError("the coordinates are on the GPU but the atoms are not"))
    end
    if isa(atoms, CuArray) && !isa(vels, CuArray)
        throw(ArgumentError("the atoms are on the GPU but the velocities are not"))
    end
    if isa(vels, CuArray) && !isa(atoms, CuArray)
        throw(ArgumentError("the velocities are on the GPU but the atoms are not"))
    end

    atom_masses = mass.(atoms)
    M = typeof(atom_masses)

    k_converted = convert_k_units(T, k, energy_units)
    K = typeof(k_converted)


    if !isbitstype(eltype(coords)) || !isbitstype(eltype(vels))
        @warn "eltype of coords or velocities is not isbits, it is recomended to use a vector of SVector's for performance"
    end

    check_units(atoms, coords, vels, energy_units, force_units, pairwise_inters,
                specific_inter_lists, general_inters, boundary)


    return System{D, G, T, A, C, B, V, AD, TO, PI, SI, GI, CN, NF, L, F, E, K, M, DA}(
                    atoms, coords, boundary, vels, atoms_data, topology, pairwise_inters,
                    specific_inter_lists, general_inters, constraints,
                    neighbor_finder, loggers, df, force_units, energy_units, k_converted, atom_masses, data)
end

"""
    System(sys; <keyword arguments>)

Convenience constructor for changing properties in a `System`.

A copy of the `System` is returned with the provided keyword arguments modified.
"""
function System(sys::System;
                atoms=sys.atoms,
                coords=sys.coords,
                boundary=sys.boundary,
                velocities=sys.velocities,
                atoms_data=sys.atoms_data,
                topology=sys.topology,
                pairwise_inters=sys.pairwise_inters,
                specific_inter_lists=sys.specific_inter_lists,
                general_inters=sys.general_inters,
                constraints=sys.constraints,
                neighbor_finder=sys.neighbor_finder,
                loggers=sys.loggers,
                force_units=sys.force_units,
                energy_units=sys.energy_units,
                k=sys.k,
                data=sys.data)
    return System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        atoms_data=atoms_data,
        topology=topology,
        pairwise_inters=pairwise_inters,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        constraints=constraints,
        neighbor_finder=neighbor_finder,
        loggers=loggers,
        force_units=force_units,
        energy_units=energy_units,
        k=k,
        data=data,
    )
end

"""
    System(crystal; <keyword arguments>)

Construct a `System` from a SimpleCrystals.jl `Crystal` struct.

Properties unused in the simulation or in analysis can be left with their
default values.
`atoms`, `atoms_data`, `coords` and `boundary` are automatically calcualted from
the `Crystal` struct.
Extra atom paramaters like `σ` have to be added manually after construction using
the convenience constructor `System(sys; <keyword arguments>)`.
"""
function System(crystal::Crystal{D};
                velocities=nothing,
                topology=nothing,
                pairwise_inters=(),
                specific_inter_lists=(),
                general_inters=(),
                constraints=(),
                neighbor_finder=NoNeighborFinder(),
                loggers=(),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
                k=default_k(energy_units),
                data=nothing) where D
    atoms = [Atom(index=i, charge=charge(a), mass=atomic_mass(a)) for (i, a) in enumerate(crystal.atoms)]
    atoms_data = [AtomData(element=String(atomic_symbol(a))) for a in crystal.atoms]
    coords = [SVector{D}(SimpleCrystals.position(crystal, i)) for i in 1:length(crystal)]

    # Build bounding box
    side_lengths = norm.(bounding_box(crystal))
    if any(typeof(crystal.lattice.crystal_family) .<: [CubicLattice, OrthorhombicLattice, TetragonalLattice])
        boundary = CubicBoundary(side_lengths...)
    elseif any(typeof(crystal.lattice.crystal_family) .<: [SquareLattice, RectangularLattice])
        boundary = RectangularBoundary(side_lengths...)
    elseif D == 2 # Honeycomb, Hex2D and Oblique
        throw(ArgumentError("$(crystal.lattice.crystal_family) is not supported as it would need " *
            "a 2D triclinic boundary, try defining the crystal with a rectangular or square unit cell"))
    else # 3D non-cubic systems
        if !all(crystal.lattice.crystal_family.lattice_angles .< 90u"°")
            throw(error("all crystal lattice angles must be less than 90°"))
        end
        boundary = TriclinicBoundary(side_lengths, crystal.lattice_angles)
    end

    return System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        atoms_data=atoms_data,
        topology=topology,
        pairwise_inters=pairwise_inters,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        constraints=constraints,
        neighbor_finder=neighbor_finder,
        loggers=loggers,
        force_units=force_units,
        energy_units=energy_units,
        k=k,
        data=data,
    )
end

"""
    ReplicaSystem(; <keyword arguments>)

A wrapper for replicas in a replica exchange simulation.

Each individual replica is a [`System`](@ref).
Properties unused in the simulation or in analysis can be left with their default values.
The minimal required arguments are `atoms`, `replica_coords`, `boundary` and `n_replicas`.
`atoms` and the elements in `replica_coords` should have the same length, along with
`atoms_data` and the elements in `replica_velocities` if these are provided.
The number of elements in `replica_coords`, `replica_velocities`, `replica_loggers` and 
the interaction arguments `replica_pairwise_inters`, `replica_specific_inter_lists`, 
`replica_general_inters` and `replica_constraints` should be equal to `n_replicas`.
This is a sub-type of `AbstractSystem` from AtomsBase.jl and implements the
interface described there.

When using `ReplicaSystem` with [`CellListMapNeighborFinder`](@ref), the number of threads used for
both the simulation of replicas and the neighbor finder should be set to be the same.
This can be done by passing `nbatches=(min(n, 8), n)` to [`CellListMapNeighborFinder`](@ref) during
construction where `n` is the number of threads to be used per replica.

# Arguments
- `atoms::A`: the atoms, or atom equivalents, in the system. Can be
    of any type but should be a bits type if the GPU is used.
- `replica_coords`: the coordinates of the atoms in each replica.
- `boundary::B`: the bounding box in which the simulation takes place.
- `n_replicas::Integer`: the number of replicas of the system.
- `replica_velocities=[zero(replica_coords[1]) * u"ps^-1" for _ in 1:n_replicas]`:
    the velocities of the atoms in each replica.
- `atoms_data::AD`: other data associated with the atoms, allowing the atoms to
    be bits types and hence work on the GPU.
- `topology::TO=nothing`: topological information about the system such as which
    atoms are in the same molecule (to be used if the same for all replicas).
    This is only used if no value is passed to the argument `replica_topology`.
- `replica_topology=[nothing for _ in 1:n_replicas]`: the topological information for 
    each replica.
- `pairwise_inters::PI=()`: the pairwise interactions in the system, i.e. interactions 
    between all or most atom pairs such as electrostatics (to be used if the same for all replicas).
    Typically a `Tuple`. This is only used if no value is passed to the argument 
    `replica_pairwise_inters`.
- `replica_pairwise_inters=[() for _ in 1:n_replicas]`: the pairwise interactions for 
    each replica.
- `specific_inter_lists::SI=()`: the specific interactions in the system, i.e. interactions 
    between specific atoms such as bonds or angles (to be used if the same for all replicas). 
    Typically a `Tuple`. This is only used if no value is passed to the argument 
    `replica_specific_inter_lists`.
- `replica_specific_inter_lists=[() for _ in 1:n_replicas]`: the specific interactions in 
    each replica.
- `general_inters::GI=()`: the general interactions in the system, i.e. interactions involving 
    all atoms such as implicit solvent (to be used if the same for all replicas). Each should
    implement the AtomsCalculators.jl interface. Typically a `Tuple`. This is only used if no
    value is passed to the argument `replica_general_inters`.
- `replica_general_inters=[() for _ in 1:n_replicas]`: the general interactions for 
    each replica.
- `constraints::CN=()` : The constraint algorithm(s) used to apply
    the same bond and angle constraints to all replicas. This is only used if
    not value is passed to the argument `replica_constraints`.
- `replica_constraints=()` : The constraint algorithm(s) used to apply
    bond and angle constraints to each replica.
- `neighbor_finder::NF=NoNeighborFinder()`: the neighbor finder used to find
    close atoms and save on computation. It is duplicated for each replica.
- `replica_loggers=[() for _ in 1:n_replicas]`: the loggers for each replica 
    that record properties of interest during a simulation.
- `exchange_logger::EL=ReplicaExchangeLogger(n_replicas)`: the logger used to record
    the exchange of replicas.
- `force_units::F=u"kJ * mol^-1 * nm^-1"`: the units of force of the system.
    Should be set to `NoUnits` if units are not being used.
- `energy_units::E=u"kJ * mol^-1"`: the units of energy of the system. Should
    be set to `NoUnits` if units are not being used.
- `k::K=Unitful.k` or `Unitful.k * Unitful.Na`: the Boltzmann constant, which may be
    modified in some simulations. `k` is chosen based on the `energy_units` given.
- `data::DA=nothing`: arbitrary data associated with the replica system.
"""
mutable struct ReplicaSystem{D, G, T, A, AD, EL, F, E, K, R, DA} <: AbstractSystem{D}
    atoms::A
    n_replicas::Int
    atoms_data::AD
    exchange_logger::EL
    force_units::F
    energy_units::E
    k::K
    replicas::R
    data::DA
end

function ReplicaSystem(;
                        atoms,
                        replica_coords,
                        boundary,
                        n_replicas,
                        replica_velocities=nothing,
                        atoms_data=[],
                        topology=nothing,
                        replica_topology=nothing,
                        pairwise_inters=(),
                        replica_pairwise_inters=nothing,
                        specific_inter_lists=(),
                        replica_specific_inter_lists=nothing,
                        general_inters=(),
                        replica_general_inters=nothing,
                        constraints=(),
                        replica_constraints=nothing,
                        neighbor_finder=NoNeighborFinder(),
                        replica_loggers=[() for _ in 1:n_replicas],
                        exchange_logger=nothing,
                        force_units=u"kJ * mol^-1 * nm^-1",
                        energy_units=u"kJ * mol^-1",
                        k=default_k(energy_units),
                        data=nothing)
    D = n_dimensions(boundary)
    G = isa(replica_coords[1], CuArray)
    T = float_type(boundary)
    A = typeof(atoms)
    AD = typeof(atoms_data)
    F = typeof(force_units)
    E = typeof(energy_units)
    DA = typeof(data)
    C = typeof(replica_coords[1])
    B = typeof(boundary)
    NF = typeof(neighbor_finder)


    if isnothing(replica_velocities)
        if force_units == NoUnits
            replica_velocities = [zero(replica_coords[1]) for _ in 1:n_replicas]
        else
            replica_velocities = [zero(replica_coords[1]) * u"ps^-1" for _ in 1:n_replicas]
        end
    end
    V = typeof(replica_velocities[1])

    if isnothing(replica_topology)
        replica_topology = [topology for _ in 1:n_replicas]
    elseif length(replica_topology) != n_replicas
        throw(ArgumentError("number of topologies ($(length(replica_topology)))"
                            * "does not match number of replicas ($n_replicas)"))
    end
    TO = eltype(replica_topology)

    if isnothing(replica_pairwise_inters)
        replica_pairwise_inters = [pairwise_inters for _ in 1:n_replicas]
    elseif length(replica_pairwise_inters) != n_replicas
        throw(ArgumentError("number of pairwise interactions ($(length(replica_pairwise_inters)))"
                            * "does not match number of replicas ($n_replicas)"))
    end
    PI = eltype(replica_pairwise_inters)

    if isnothing(replica_specific_inter_lists)
        replica_specific_inter_lists = [specific_inter_lists for _ in 1:n_replicas]
    elseif length(replica_specific_inter_lists) != n_replicas
        throw(ArgumentError("number of specific interaction lists ($(length(replica_specific_inter_lists)))"
                            * "does not match number of replicas ($n_replicas)"))
    end
    SI = eltype(replica_specific_inter_lists)

    if isnothing(replica_general_inters)
        replica_general_inters = [general_inters for _ in 1:n_replicas]
    elseif length(replica_general_inters) != n_replicas
        throw(ArgumentError("number of general interactions ($(length(replica_general_inters)))"
                            * "does not match number of replicas ($n_replicas)"))
    end
    GI = eltype(replica_general_inters)

    df = n_dof(D, length(atoms), boundary)
    if isnothing(replica_constraints)
        if length(constraints) > 0
            for ca in constraints
                df -= n_dof_lost(D, ca.clusters)
            end
            replica_constraints = [constraints for _ in 1:n_replicas]
            replica_dfs = [df for _ in 1:n_replicas]
        end

    elseif length(replica_constraints) != n_replicas
        throw(ArgumentError("number of constraints ($(length(replica_constraints)))"
                            * "does not match number of replicas ($n_replicas)"))
    else #if replicas passed & correct length
        replica_dfs = df.*ones(typeof(df), length(replica_constraints))
        for (i,cosntraints) in enumerate(replica_constraints)
            if length(constraints) > 0
                for ca in cosntraints
                    replica_dfs[i] -= n_dof_lost(D, ca.clusters)
                end
            end
        end
    end
    CN = eltype(replica_constraints)


    if isnothing(exchange_logger)
        exchange_logger = ReplicaExchangeLogger(T, n_replicas)
    end
    EL = typeof(exchange_logger)

    if !all(y -> typeof(y) == C, replica_coords)
        throw(ArgumentError("the coordinates for all the replicas are not of the same type"))
    end
    if !all(y -> typeof(y) == V, replica_velocities)
        throw(ArgumentError("the velocities for all the replicas are not of the same type"))
    end

    if length(replica_coords) != n_replicas
        throw(ArgumentError("there are $(length(replica_coords)) coordinates for replicas but $n_replicas replicas"))
    end
    if length(replica_velocities) != n_replicas
        throw(ArgumentError("there are $(length(replica_velocities)) velocities for replicas but $n_replicas replicas"))
    end
    if length(replica_loggers) != n_replicas
        throw(ArgumentError("there are $(length(replica_loggers)) loggers but $n_replicas replicas"))
    end

    if !all(y -> length(y) == length(replica_coords[1]), replica_coords)
        throw(ArgumentError("some replicas have different number of coordinates"))
    end
    if !all(y -> length(y) == length(replica_velocities[1]), replica_velocities)
        throw(ArgumentError("some replicas have different number of velocities"))
    end

    if length(atoms) != length(replica_coords[1])
        throw(ArgumentError("there are $(length(atoms)) atoms but $(length(replica_coords[1])) coordinates"))
    end
    if length(atoms) != length(replica_velocities[1])
        throw(ArgumentError("there are $(length(atoms)) atoms but $(length(replica_velocities[1])) velocities"))
    end
    if length(atoms_data) > 0 && length(atoms) != length(atoms_data)
        throw(ArgumentError("there are $(length(atoms)) atoms but $(length(atoms_data)) atom data entries"))
    end

    n_cuarray = sum(y -> isa(y, CuArray), replica_coords)
    if !(n_cuarray == n_replicas || n_cuarray == 0)
        throw(ArgumentError("the coordinates for $n_cuarray out of $n_replicas replicas are on GPU"))
    end
    if isa(atoms, CuArray) && n_cuarray != n_replicas
        throw(ArgumentError("the atoms are on the GPU but the coordinates are not"))
    end
    if n_cuarray == n_replicas && !isa(atoms, CuArray)
        throw(ArgumentError("the coordinates are on the GPU but the atoms are not"))
    end

    n_cuarray = sum(y -> isa(y, CuArray), replica_velocities)
    if !(n_cuarray == n_replicas || n_cuarray == 0)
        throw(ArgumentError("the velocities for $n_cuarray out of $n_replicas replicas are on GPU"))
    end
    if isa(atoms, CuArray) && n_cuarray != n_replicas
        throw(ArgumentError("the atoms are on the GPU but the velocities are not"))
    end
    if n_cuarray == n_replicas && !isa(atoms, CuArray)
        throw(ArgumentError("the velocities are on the GPU but the atoms are not"))
    end

    atom_masses = mass.(atoms)
    M = typeof(atom_masses)

    k_converted = convert_k_units(T, k, energy_units)
    K = typeof(k_converted)

    replicas = Tuple(System{D, G, T, A, C, B, V, AD, TO, PI, SI, GI, CN, NF,
                            typeof(replica_loggers[i]), F, E, K, M, Nothing}(
            atoms, replica_coords[i], boundary, replica_velocities[i], atoms_data,
            replica_topology[i], replica_pairwise_inters[i], replica_specific_inter_lists[i],
            replica_general_inters[i], replica_constraints[i], 
            deepcopy(neighbor_finder), replica_loggers[i], replica_dfs[i],
            force_units, energy_units, k_converted, atom_masses, nothing) for i in 1:n_replicas)
    R = typeof(replicas)

    return ReplicaSystem{D, G, T, A, AD, EL, F, E, K, R, DA}(
            atoms, n_replicas, atoms_data, exchange_logger, force_units,
            energy_units, k_converted, replicas, data)
end

"""
    is_on_gpu(sys)

Whether a [`System`](@ref) or [`ReplicaSystem`](@ref) is on the GPU.
"""
is_on_gpu(::Union{System{D, G}, ReplicaSystem{D, G}}) where {D, G} = G

"""
    float_type(sys)
    float_type(boundary)

The float type a [`System`](@ref), [`ReplicaSystem`](@ref) or bounding box uses.
"""
float_type(::Union{System{D, G, T}, ReplicaSystem{D, G, T}}) where {D, G, T} = T

"""
    masses(sys)

The masses of the atoms in a [`System`](@ref) or [`ReplicaSystem`](@ref).
"""
masses(s::System) = s.masses
masses(s::ReplicaSystem) = mass.(s.atoms)

"""
    charges(sys)

The partial charges of the atoms in a [`System`](@ref) or [`ReplicaSystem`](@ref).
"""
charges(s::Union{System, ReplicaSystem}) = charge.(s.atoms)
charge(s::Union{System, ReplicaSystem}, i::Integer) = charge(s.atoms[i])

# Move an array to the GPU depending on whether the system is on the GPU
move_array(arr, ::System{D, false}) where {D} = arr
move_array(arr, ::System{D, true }) where {D} = CuArray(arr)

Base.getindex(s::Union{System, ReplicaSystem}, i::Integer) = AtomView(s, i)
Base.length(s::Union{System, ReplicaSystem}) = length(s.atoms)
Base.eachindex(s::Union{System, ReplicaSystem}) = Base.OneTo(length(s))

AtomsBase.species_type(s::Union{System, ReplicaSystem}) = typeof(s[1])
AtomsBase.atomkeys(s::Union{System, ReplicaSystem}) = (:position, :velocity, :atomic_mass, :atomic_number, :charge)
AtomsBase.hasatomkey(s::Union{System, ReplicaSystem}, x::Symbol) = x in atomkeys(s)
AtomsBase.keys(sys::Union{System, ReplicaSystem}) = fieldnames(typeof(sys))
AtomsBase.haskey(sys::Union{System, ReplicaSystem}, x::Symbol) = hasfield(typeof(sys), x)
Base.getindex(sys::Union{System, ReplicaSystem}, x::Symbol) = 
    hasfield(typeof(sys), x) ? getfield(sys, x) : KeyError("no field `$x`, allowed keys are $(keys(sys))")
Base.pairs(sys::Union{System, ReplicaSystem}) = (k => sys[k] for k in keys(sys))
Base.get(sys::Union{System, ReplicaSystem}, x::Symbol, default) = 
    haskey(sys, x) ? getfield(sys, x) : default

AtomsBase.position(s::System) = s.coords
AtomsBase.position(s::System, i::Integer) = s.coords[i]
AtomsBase.position(s::ReplicaSystem) = s.replicas[1].coords
AtomsBase.position(s::ReplicaSystem, i::Integer) = s.replicas[1].coords[i]

AtomsBase.velocity(s::System) = s.velocities
AtomsBase.velocity(s::System, i::Integer) = s.velocities[i]
AtomsBase.velocity(s::ReplicaSystem) = s.replicas[1].velocities
AtomsBase.velocity(s::ReplicaSystem, i::Integer) = s.replicas[1].velocities[i]

AtomsBase.atomic_mass(s::Union{System, ReplicaSystem}) = masses(s)
AtomsBase.atomic_mass(s::Union{System, ReplicaSystem}, i::Integer) = mass(s.atoms[i])

function Base.getindex(sys::Union{System, ReplicaSystem}, i, x::Symbol)
    atomsbase_keys = (:position, :velocity, :atomic_mass, :atomic_number)
    if hasatomkey(sys, x)
        if x in atomsbase_keys
            return getproperty(AtomsBase, x)(sys, i)
        elseif x == :charge
            return getproperty(Molly, x)(sys, i) * u"e_au"
        end
    else
        throw(KeyError("key $x not present in the system"))
    end
end

function AtomsBase.atomic_symbol(s::Union{System, ReplicaSystem})
    if length(s.atoms_data) > 0
        return map(ad -> Symbol(ad.element), s.atoms_data)
    else
        return fill(:unknown, length(s))
    end
end

function AtomsBase.atomic_symbol(s::Union{System, ReplicaSystem}, i::Integer)
    if length(s.atoms_data) > 0
        return Symbol(s.atoms_data[i].element)
    else
        return :unknown
    end
end

function AtomsBase.atomic_number(s::Union{System, ReplicaSystem})
    if length(s.atoms_data) > 0
        return map(s.atoms_data) do ad
            if ad.element != "?"
                PeriodicTable.elements[Symbol(ad.element)].number
            else
                :unknown
            end
        end
    else
        return fill(:unknown, length(s))
    end
end

function AtomsBase.atomic_number(s::Union{System, ReplicaSystem}, i::Integer)
    if length(s.atoms_data) > 0 && s.atoms_data[i].element != "?"
        return PeriodicTable.elements[Symbol(s.atoms_data[i].element)].number
    else
        return :unknown
    end
end

function AtomsBase.boundary_conditions(::Union{System{3}, ReplicaSystem{3}})
    return SVector(Periodic(), Periodic(), Periodic())
end

function AtomsBase.boundary_conditions(::Union{System{2}, ReplicaSystem{2}})
    return SVector(Periodic(), Periodic())
end

AtomsBase.bounding_box(s::System) = bounding_box(s.boundary)
AtomsBase.bounding_box(s::ReplicaSystem) = bounding_box(s.replicas[1].boundary)

function Base.show(io::IO, s::System)
    print(io, "System with ", length(s), " atoms, boundary ", s.boundary)
end

function Base.show(io::IO, s::ReplicaSystem)
    print(io, "ReplicaSystem containing ",  s.n_replicas, " replicas with ", length(s),
          " atoms each")
end

# Take precedence over AtomsBase.jl show function
Base.show(io::IO, ::MIME"text/plain", s::Union{System, ReplicaSystem}) = show(io, s)

"""
    System(abstract_system)

Convert an AtomsBase `AbstractSystem` to a Molly `System`.

To add properties not present in the AtomsBase interface (e.g. pair potentials) use the
convenience constructor `System(sys::System)`.
"""
function System(sys::AbstractSystem{D}, energy_units, force_units) where D
    # Convert BC to Molly types
    bb = bounding_box(sys)
    bcs = AtomsBase.boundary_conditions(sys)

    if any(typeof.(bcs) .== DirichletZero)
        throw(ArgumentError("Molly does not support DirichletZero boundary conditions"))
    end

    # Check whether box is cubic
    angles = []
    for i in 1:D
        for j in (i + 1):D
            push!(angles, ustrip(dot(bb[i], bb[j])))
        end
    end
    is_cubic = all(angles .== 0.0)
    box_lengths = norm.(bb)

    if is_cubic && D == 2
        @warn "Molly RectangularBoundary assumes origin at (0, 0, 0)"
        molly_boundary = RectangularBoundary(box_lengths...)
    elseif is_cubic && D == 3
        @warn "Molly CubicSystem assumes origin at (0, 0, 0)"
        molly_boundary = CubicBoundary(box_lengths...)
    elseif D == 3
        @warn "Molly TriclinicBoundary assumes origin at (0, 0, 0)"
        if any(ustrip.(box_lengths) .== Inf)
            throw(ArgumentError("TriclinicBoundary does not support infinite boundaries"))
        end
        molly_boundary = TriclinicBoundary(bb...)
    else
        throw(ArgumentError("Molly does not support 2D triclinic domains"))
    end

    length_unit = unit(first(position(sys, 1)))
    atoms = Vector{Atom}(undef, (length(sys),))
    atoms_data = Vector{AtomData}(undef, (length(sys),))
    for (i, atom) in enumerate(sys)
        atoms[i] = Atom(
            index=i,
            charge=ustrip(get(atom, :charge, 0.0)), # Remove e unit
            mass=atomic_mass(atom),
            σ=(0.0 * length_unit),
            ϵ=(0.0 * energy_units),
        )
        atoms_data[i] = AtomData(element=String(atomic_symbol(atom)))
    end

    # AtomsBase does not specify type for coordinates or velocities
    # so it is best to use unified type. Thus conversion to SVector.
    coords = map(position(sys)) do r
        SVector(r...)
    end
    vels = map(velocity(sys)) do v
        SVector(v...)
    end

    mass_dim = dimension(atomic_mass(sys, 1))
    if mass_dim == u"𝐌" && dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        throw(ArgumentError("When constructing System from AbstractSystem, energy units " *
                            "are molar but mass units are not"))
    elseif mass_dim == u"𝐌 * 𝐍^-1" && dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐓^-2"
        throw(ArgumentError("When constructing System from AbstractSystem, mass units " *
                            "are molar but energy units are not"))
    end

    return System(
        atoms=atoms,
        coords=coords,
        boundary=molly_boundary,
        velocities=vels,
        atoms_data=atoms_data,
        energy_units=energy_units,
        force_units=force_units,
    )
end

"""
    MollyCalculator(; <keyword arguments>)

A calculator for use with the AtomsCalculators.jl interface.

`neighbors` can optionally be given as a keyword argument when calling the
calculation functions to save on computation when the neighbors are the same
for multiple calls.
In a similar way, `n_threads` can be given to determine the number of threads
to use when running the calculation function.
Note that this calculator is designed for using Molly in other contexts; if you
want to use another calculator in Molly it can be given as `general_inters` when
creating a [`System`](@ref).

Not currently compatible with virial calculation.
Not currently compatible with using atom properties such as `σ` and `ϵ`.

# Arguments
- `pairwise_inters::PI=()`: the pairwise interactions in the system, i.e.
    interactions between all or most atom pairs such as electrostatics.
    Typically a `Tuple`.
- `specific_inter_lists::SI=()`: the specific interactions in the system,
    i.e. interactions between specific atoms such as bonds or angles. Typically
    a `Tuple`.
- `general_inters::GI=()`: the general interactions in the system,
    i.e. interactions involving all atoms such as implicit solvent. Each should
    implement the AtomsCalculators.jl interface. Typically a `Tuple`.
- `neighbor_finder::NF=NoNeighborFinder()`: the neighbor finder used to find
    close atoms and save on computation.
- `force_units::F=u"kJ * mol^-1 * nm^-1"`: the units of force of the system.
    Should be set to `NoUnits` if units are not being used.
- `energy_units::E=u"kJ * mol^-1"`: the units of energy of the system. Should
    be set to `NoUnits` if units are not being used.
- `k::K=Unitful.k` or `Unitful.k * Unitful.Na`: the Boltzmann constant, which may be
    modified in some simulations. `k` is chosen based on the `energy_units` given.
"""
struct MollyCalculator{PI, SI, GI, NF, F, E, K}
    pairwise_inters::PI
    specific_inter_lists::SI
    general_inters::GI
    neighbor_finder::NF
    force_units::F
    energy_units::E
    k::K
end

function MollyCalculator(;
        pairwise_inters=(),
        specific_inter_lists=(),
        general_inters=(),
        neighbor_finder=NoNeighborFinder(),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
        k=default_k(energy_units),
    )
    return MollyCalculator(pairwise_inters, specific_inter_lists, general_inters, neighbor_finder,
                           force_units, energy_units, k)
end

# Doesn't work for Float32
function AtomsCalculators.promote_force_type(::Any, calc::MollyCalculator)
    return typeof(SVector(1.0, 1.0, 1.0) * calc.force_units)
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(
        abstract_sys,
        calc::MollyCalculator;
        neighbors=nothing,
        n_threads::Integer=Threads.nthreads(),
        kwargs...,
    )
    sys_nointers = System(abstract_sys, calc.energy_units, calc.force_units)
    sys = System(
        sys_nointers;
        pairwise_inters=calc.pairwise_inters,
        specific_inter_lists=calc.specific_inter_lists,
        general_inters=calc.general_inters,
        neighbor_finder=calc.neighbor_finder,
        k=calc.k,
    )
    nbs = isnothing(neighbors) ? find_neighbors(sys) : neighbors
    return forces(sys, nbs; n_threads=n_threads)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
        abstract_sys,
        calc::MollyCalculator;
        neighbors=nothing,
        n_threads::Integer=Threads.nthreads(),
        kwargs...,
    )
    sys_nointers = System(abstract_sys, calc.energy_units, calc.force_units)
    sys = System(
        sys_nointers;
        pairwise_inters=calc.pairwise_inters,
        specific_inter_lists=calc.specific_inter_lists,
        general_inters=calc.general_inters,
        neighbor_finder=calc.neighbor_finder,
        k=calc.k,
    )
    nbs = isnothing(neighbors) ? find_neighbors(sys) : neighbors
    return potential_energy(sys, nbs; n_threads=n_threads)
end
