# Types
import Base: ==, hash

export
    PairwiseInteraction,
    InteractionList1Atoms,
    InteractionList2Atoms,
    InteractionList3Atoms,
    InteractionList4Atoms,
    Atom,
    mass,
    charge,
    AtomData,
    MolecularTopology,
    NeighborList,
    System,
    ThermoState,
    ReplicaSystem,
    array_type,
    is_on_gpu,
    float_type,
    masses,
    charges,
    MollyCalculator,
    ASECalculator

# This is not the only place that the default float is set, for example
#   some function argument defaults are Float64
const DefaultFloat = Float64

# Base type for Molly interaction types
abstract type AbstractInteraction end

# Base type for n-body interactions, `N` is the body order
# Only NBodyInteraction{2} is currently supported
abstract type NBodyInteraction{N} <: AbstractInteraction end

# Base type for all specific interaction lists, `N` is the number of atoms in the interaction
abstract type SpecificInteractionList{N} <: AbstractInteraction end

"""
Base type for pairwise interactions.

An alias for NBodyInteraction{2}.
Custom pairwise interactions should subtype this.
"""
const PairwiseInteraction = NBodyInteraction{2}

"""
    InteractionList1Atoms(is, inters)
    InteractionList1Atoms(is, inters, types)
    InteractionList1Atoms(inter_type)

A list of specific interactions that involve one atom such as position restraints.
"""
struct InteractionList1Atoms{I, T} <: SpecificInteractionList{1}
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
struct InteractionList2Atoms{I, T} <: SpecificInteractionList{2}
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
struct InteractionList3Atoms{I, T} <: SpecificInteractionList{3}
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
struct InteractionList4Atoms{I, T} <: SpecificInteractionList{4}
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

function ==(a::InteractionList1Atoms, b::InteractionList1Atoms)
    return a.is == b.is && 
           a.inters == b.inters && 
           a.types == b.types
end

function ==(a::InteractionList2Atoms, b::InteractionList2Atoms)
    return a.is == b.is && 
           a.js == b.js && 
           a.inters == b.inters && 
           a.types == b.types
end

function ==(a::InteractionList3Atoms, b::InteractionList3Atoms)
    return a.is == b.is && 
           a.js == b.js && 
           a.ks == b.ks && 
           a.inters == b.inters && 
           a.types == b.types
end

function ==(a::InteractionList4Atoms, b::InteractionList4Atoms)
    return a.is == b.is && 
           a.js == b.js && 
           a.ks == b.ks && 
           a.ls == b.ls && 
           a.inters == b.inters && 
           a.types == b.types
end

function hash(a::InteractionList1Atoms, h::UInt)
    is     = from_device(a.is)
    inters = from_device(a.inters)
    types  = from_device(a.types)
    return hash(is, hash(inters, hash(types, h)))
end

function hash(a::InteractionList2Atoms, h::UInt)
    is     = from_device(a.is)
    js     = from_device(a.js)
    inters = from_device(a.inters)
    types  = from_device(a.types)
    return hash(is, hash(js, hash(inters, hash(types, h))))
end

function hash(a::InteractionList3Atoms, h::UInt)
    is     = from_device(a.is)
    js     = from_device(a.js)
    ks     = from_device(a.ks)
    inters = from_device(a.inters)
    types  = from_device(a.types)
    return hash(is, hash(js, hash(ks, hash(inters, hash(types, h)))))
end

function hash(a::InteractionList4Atoms, h::UInt)
    is     = from_device(a.is)
    js     = from_device(a.js)
    ks     = from_device(a.ks)
    ls     = from_device(a.ls)
    inters = from_device(a.inters)
    types  = from_device(a.types)
    return hash(is, hash(js, hash(ks, hash(ls, hash(inters, hash(types, h))))))
end

function inject_interaction_list(inter::InteractionList1Atoms, params_dic, AT)
    inters_grad = to_device(inject_interaction.(from_device(inter.inters),
                                inter.types, (params_dic,)), AT)
    InteractionList1Atoms(inter.is, inters_grad, inter.types)
end

function inject_interaction_list(inter::InteractionList2Atoms, params_dic, AT)
    inters_grad = to_device(inject_interaction.(from_device(inter.inters),
                                inter.types, (params_dic,)), AT)
    InteractionList2Atoms(inter.is, inter.js, inters_grad, inter.types)
end

function inject_interaction_list(inter::InteractionList3Atoms, params_dic, AT)
    inters_grad = to_device(inject_interaction.(from_device(inter.inters),
                                inter.types, (params_dic,)), AT)
    InteractionList3Atoms(inter.is, inter.js, inter.ks, inters_grad, inter.types)
end

function inject_interaction_list(inter::InteractionList4Atoms, params_dic, AT)
    inters_grad = to_device(inject_interaction.(from_device(inter.inters),
                                inter.types, (params_dic,)), AT)
    InteractionList4Atoms(inter.is, inter.js, inter.ks, inter.ls, inters_grad, inter.types)
end

"""
    Atom(; <keyword arguments>)

An atom and its associated information.

Properties unused in the simulation or in analysis can be left with their
default values.
The types used should be bits types if the GPU is going to be used.

# Arguments
- `index::Int=1`: the index of the atom in the system. This only needs to be set if
    it is used in the interactions. The order of atoms is determined by their order
    in the atom vector.
- `atom_type::T=1`: the type of the atom. This only needs to be set if
    it is used in the interactions.
- `mass::M=1.0u"g/mol"`: the mass of the atom.
- `charge::C=0.0`: the charge of the atom, used for electrostatic interactions.
- `σ::S=0.0u"nm"`: the Lennard-Jones finite distance at which the inter-particle
    potential is zero.
- `ϵ::E=0.0u"kJ * mol^-1"`: the Lennard-Jones depth of the potential well.
- `λ_coul::L=1.0`: scaling parameter of Coulombic interactions, used for alchemical 
    transformations
- `λ_vdw::L=1.0`: scaling parameter of Van der Waals interactions, used for alchemical 
    transformations
"""
@kwdef struct Atom{T, M, C, S, E, L}
    index::Int = 1
    atom_type::T = 1
    mass::M = 1.0u"g/mol"
    charge::C = 0.0
    σ::S = 0.0u"nm"
    ϵ::E = 0.0u"kJ * mol^-1"
    λ_coul::L = 1.0
    λ_vdw::L = 1.0
end

function Base.zero(::Atom{T, M, C, S, E, L}) where {T, M, C, S, E, L}
    return Atom(0, zero(T), zero(M), zero(C), zero(S), zero(E), zero(L), zero(L))
end

function Base.:+(a1::Atom, a2::Atom)
    return Atom(a1.index, a1.atom_type, a1.mass + a2.mass, a1.charge + a2.charge,
                a1.σ + a2.σ, a1.ϵ + a2.ϵ, a1.λ_coul + a2.λ_coul, a1.λ_vdw + a2.λ_vdw)
end

# get function errors with AD
dict_get(dic, key, default::T) where {T} = (haskey(dic, key) ? T(dic[key]) : default)

function inject_atom(at, at_data, params_dic)
    key_prefix = "atom_$(at_data.atom_type)_"
    Atom(
        at.index,
        at.atom_type,
        dict_get(params_dic, key_prefix * "mass"  , at.mass),
        at.charge, # Residue-specific
        dict_get(params_dic, key_prefix * "σ"     , at.σ   ),
        dict_get(params_dic, key_prefix * "ϵ"     , at.ϵ   ),
        at.λ_coul, # Preserve lambda from existing atom,
        at.λ_vdw
    )
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
Virtual sites should have zero mass, and non-virtual sites should have non-zero mass.
"""
mass(atom) = atom.mass

function Base.getindex(at::Atom, x::Symbol)
    if x == :charge
        return charge(at) * u"e_au" # AtomsBase has units on charge
    elseif hasfield(Atom, x)
        return getfield(at, x)
    else
        throw(KeyError("no field $x in Atom"))
    end
end

function Base.show(io::IO, a::Atom)
    print(io, "Atom with index=", a.index, ", atom_type=", a.atom_type, ", mass=", mass(a),
          ", charge=", charge(a), ", σ=", a.σ, ", ϵ=", a.ϵ, ", λ_coul=", a.λ_coul, " λ_vdw=", a.λ_vdw)
end

function lj_zero_shortcut(atom_i, atom_j)
    return iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
           iszero_value(atom_i.σ) || iszero_value(atom_j.σ)
end

no_shortcut(atom_i, atom_j) = false

lorentz_σ_mixing(atom_i, atom_j) = (atom_i.σ + atom_j.σ) / 2
lorentz_ϵ_mixing(atom_i, atom_j) = (atom_i.ϵ + atom_j.ϵ) / 2
lorentz_λ_coul_mixing(atom_i, atom_j) = (atom_i.λ_coul + atom_j.λ_coul) / 2
lorentz_λ_vdw_mixing(atom_i, atom_j) = (atom_i.λ_vdw + atom_j.λ_vdw) / 2

geometric_σ_mixing(atom_i, atom_j) = sqrt(atom_i.σ * atom_j.σ)
geometric_ϵ_mixing(atom_i, atom_j) = sqrt(atom_i.ϵ * atom_j.ϵ)
geometric_λ_coul_mixing(atom_i, atom_j) = sqrt(atom_i.λ_coul * atom_j.λ_coul)
geometric_λ_vdw_mixing(atom_i, atom_j) = sqrt(atom_i.λ_vdw * atom_j.λ_vdw)

function waldman_hagler_σ_mixing(atom_i, atom_j)
    T = typeof(ustrip(atom_i.σ))
    return ((atom_i.σ^6 + atom_j.σ^6) / 2) ^ T(1/6)
end

function waldman_hagler_ϵ_mixing(atom_i, atom_j)
    return 2 * sqrt(atom_i.ϵ * atom_j.ϵ) * ((atom_i.σ^3 * atom_j.σ^3) / (atom_i.σ^6 + atom_j.σ^6))
end

fender_halsey_ϵ_mixing(atom_i, atom_j) = (2 * atom_i.ϵ * atom_j.ϵ) / (atom_i.ϵ + atom_j.ϵ)

"""
    AtomData(; atom_type="?", atom_name="?", res_number=1, res_name="???",
             chain_id="A", element="?", hetero_atom=false)

Data associated with an atom.

Storing this separately allows the [`Atom`](@ref) types to be bits types and hence
work on the GPU.
"""
@kwdef struct AtomData
    atom_type::String = "?"
    atom_name::String = "?"
    res_number::Int = 1
    res_name::String = "???"
    chain_id::String = "A"
    element::String = "?"
    hetero_atom::Bool = false
end

"""
    MolecularTopology(bond_is, bond_js, n_atoms)
    MolecularTopology(atom_molecule_inds, molecule_atom_counts, bonded_atoms=[])

Topology information for a system.

Stores the index of the molecule each atom belongs to, the number of
atoms in each molecule and the list of bonded atom pairs.
"""
struct MolecularTopology
    atom_molecule_inds::Vector{Int32}
    molecule_atom_counts::Vector{Int32}
    bonded_atoms::Vector{Tuple{Int32, Int32}}
end

function bond_graph(bond_is, bond_js, n_atoms)
    g = SimpleGraph(n_atoms)
    for (i, j) in zip(bond_is, bond_js)
        add_edge!(g, i, j)
    end
    return g
end

MolecularTopology(amis, macs) = MolecularTopology(amis, macs, [])

function MolecularTopology(bond_is, bond_js, n_atoms::Integer)
    g = bond_graph(bond_is, bond_js, n_atoms)
    cc = connected_components(g)
    atom_molecule_inds = zeros(Int32, n_atoms)
    for (mi, atom_inds) in enumerate(cc)
        for ai in atom_inds
            atom_molecule_inds[ai] = mi
        end
    end
    molecule_atom_counts = length.(cc)
    bonded_atoms = collect(zip(bond_is, bond_js))
    return MolecularTopology(atom_molecule_inds, molecule_atom_counts, bonded_atoms)
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
    Should be a `Tuple` or `NamedTuple` of `PairwiseInteraction`s.
- `specific_inter_lists::SI=()`: the specific interactions in the system,
    i.e. interactions between specific atoms such as bonds or angles.
    Should be a `Tuple` or `NamedTuple`.
- `general_inters::GI=()`: the general interactions in the system,
    i.e. interactions involving all atoms such as implicit solvent. Each should
    implement the AtomsCalculators.jl interface. Should be a `Tuple` or `NamedTuple`.
- `constraints::CN=()`: the constraints for bonds and angles in the system.
    Should be a `Tuple` or `NamedTuple`.
- `virtual_sites::VS=[]`: the virtual sites present in the system; these are
    mass-less particles determined by the positions of other atoms.
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
- `strictness=:warn`: determines behavior when encountering possible problems,
    options are `:warn` to emit warnings, `:nowarn` to suppress warnings or
    `:error` to error.
"""
mutable struct System{D, AT, T, A, C, B, V, AD, TO, PI, SI, GI, CN, VS, VF, NF,
                      L, F, E, K, M, TM, DA} <: AtomsBase.AbstractSystem{D}
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
    virtual_sites::VS
    virtual_site_flags::VF
    neighbor_finder::NF
    loggers::L
    df::Int
    force_units::F
    energy_units::E
    k::K
    masses::M
    total_mass::TM
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
                virtual_sites=[],
                neighbor_finder=NoNeighborFinder(),
                loggers=(),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
                k=default_k(energy_units),
                data=nothing,
                strictness=:warn)
    check_strictness(strictness)
    D = AtomsBase.n_dimensions(boundary)
    AT = array_type(coords)
    T = float_type(boundary)
    A = typeof(atoms)
    C = typeof(coords)
    B = typeof(boundary)
    AD = typeof(atoms_data)
    TO = typeof(topology)
    PI = typeof(pairwise_inters)
    SI = typeof(specific_inter_lists)
    GI = typeof(general_inters)
    VS = typeof(virtual_sites)
    NF = typeof(neighbor_finder)
    L = typeof(loggers)
    F = typeof(force_units)
    E = typeof(energy_units)
    DA = typeof(data)
    n_atoms = length(atoms)

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

    if n_atoms != length(coords)
        throw(ArgumentError("there are $n_atoms atoms but $(length(coords)) coordinates"))
    end
    if n_atoms != length(vels)
        throw(ArgumentError("there are $n_atoms atoms but $(length(vels)) velocities"))
    end
    if length(atoms_data) > 0 && n_atoms != length(atoms_data)
        throw(ArgumentError("there are $n_atoms atoms but $(length(atoms_data)) atom data entries"))
    end

    if isa(atoms, AbstractGPUArray) && !isbitstype(eltype(atoms))
        throw(ArgumentError("the atoms are on the GPU but are not a bits type, found " *
                            "atom type $(eltype(atoms))"))
    end
    if isa(atoms, AbstractGPUArray) && !isa(coords, AbstractGPUArray)
        throw(ArgumentError("the atoms are on the GPU but the coordinates are not"))
    end
    if isa(coords, AbstractGPUArray) && !isa(atoms, AbstractGPUArray)
        throw(ArgumentError("the coordinates are on the GPU but the atoms are not"))
    end
    if isa(atoms, AbstractGPUArray) && !isa(vels, AbstractGPUArray)
        throw(ArgumentError("the atoms are on the GPU but the velocities are not"))
    end
    if isa(vels, AbstractGPUArray) && !isa(atoms, AbstractGPUArray)
        throw(ArgumentError("the velocities are on the GPU but the atoms are not"))
    end
    if length(virtual_sites) > 0
        if isa(atoms, AbstractGPUArray) && !isa(virtual_sites, AbstractGPUArray)
            throw(ArgumentError("the atoms are on the GPU but the virtual sites are not"))
        end
        if isa(virtual_sites, AbstractGPUArray) && !isa(atoms, AbstractGPUArray)
            throw(ArgumentError("the virtual sites are on the GPU but the atoms are not"))
        end
    end

    if !any(TT -> (pairwise_inters isa TT), (Tuple, NamedTuple))
        throw(ArgumentError("pairwise_inters should be a Tuple or a NamedTuple but has " *
                            "type $(typeof(pairwise_inters))"))
    end
    if !any(TT -> (specific_inter_lists isa TT), (Tuple, NamedTuple))
        throw(ArgumentError("specific_inter_lists should be a Tuple or a NamedTuple but has " *
                            "type $(typeof(specific_inter_lists))"))
    end
    if !any(TT -> (general_inters isa TT), (Tuple, NamedTuple))
        throw(ArgumentError("general_inters should be a Tuple or a NamedTuple but has " *
                            "type $(typeof(general_inters))"))
    end

    if !all(i -> i isa PairwiseInteraction, values(pairwise_inters))
        throw(ArgumentError("not all pairwise_inters are a subtype of PairwiseInteraction, " *
                            "found types $(typeof.(pairwise_inters))"))
    end
    if !all(i -> i isa SpecificInteractionList, values(specific_inter_lists))
        throw(ArgumentError("not all specific_inter_lists are a subtype of SpecificInteractionList, " *
                            "found types $(typeof.(specific_inter_lists))"))
    end

    if neighbor_finder isa NoNeighborFinder && any(use_neighbors, values(pairwise_inters))
        throw(ArgumentError("neighbor_finder is NoNeighborFinder but one of pairwise_inters " *
                            "uses the neighbor list"))
    end
    if !(neighbor_finder isa NoNeighborFinder) && !all(use_neighbors, values(pairwise_inters))
        err_str = "A neighbor finder is used but one of pairwise_inters does not use the " *
                  "neighbor finder, this may not be intended"
        report_issue(err_str, strictness)
    end

    atom_masses = mass.(atoms)
    M = typeof(atom_masses)
    total_mass = sum(atom_masses)
    TM = typeof(total_mass)

    k_converted = convert_k_units(T, k, energy_units, strictness)
    K = typeof(k_converted)

    if !isbitstype(eltype(coords)) || !isbitstype(eltype(vels))
        err_str = "eltype of coords or velocities is not isbits, it is recomended to use a " *
                  "vector of SVectors for performance"
        report_issue(err_str, strictness)
    end

    virtual_site_flags = setup_virtual_sites(virtual_sites, atom_masses, constraints,
                                             AT, D, strictness)
    VF = typeof(virtual_site_flags)
    n_virtual_sites = sum(virtual_site_flags)

    df = n_dof(D, n_atoms - n_virtual_sites, boundary)
    if length(constraints) > 0
        for ca in constraints
            for cluster_type in cluster_keys(ca)
                clusters = getproperty(ca, cluster_type)
                df -= n_dof_lost(D, clusters)
            end
        end
    end
    constraints = Tuple(setup_constraints!(ca, neighbor_finder, AT) for ca in constraints)
    CN = typeof(constraints)

    check_units(atoms, coords, vels, energy_units, force_units, pairwise_inters,
                specific_inter_lists, general_inters, boundary)

    return System{D, AT, T, A, C, B, V, AD, TO, PI, SI, GI, CN, VS, VF, NF, L, F, E, K, M, TM, DA}(
                    atoms, coords, boundary, vels, atoms_data, topology, pairwise_inters,
                    specific_inter_lists, general_inters, constraints, virtual_sites,
                    virtual_site_flags, neighbor_finder, loggers, df, force_units, energy_units,
                    k_converted, atom_masses, total_mass, data)
end

"""
    System(sys; <keyword arguments>)

Convenience constructor for changing properties in a `System`.

The `System` is returned with the provided keyword arguments modified.
Give `deepcopy(sys)` as the argument to make a new copy of the system.
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
                virtual_sites=sys.virtual_sites,
                neighbor_finder=sys.neighbor_finder,
                loggers=sys.loggers,
                force_units=sys.force_units,
                energy_units=sys.energy_units,
                k=sys.k,
                data=sys.data,
                strictness=:warn)
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
        virtual_sites=virtual_sites,
        neighbor_finder=neighbor_finder,
        loggers=loggers,
        force_units=force_units,
        energy_units=energy_units,
        k=k,
        data=data,
        strictness=strictness,
    )
end

"""
    System(crystal; <keyword arguments>)

Construct a `System` from a SimpleCrystals.jl `Crystal` struct.

Properties unused in the simulation or in analysis can be left with their
default values.
`atoms`, `atoms_data`, `coords` and `boundary` are automatically calculated from
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
    atoms = [Atom(index=i, charge=ustrip(uconvert(u"C", charge(a)) / Unitful.q), mass=AtomsBase.mass(a))
             for (i, a) in enumerate(crystal.atoms)]
    atoms_data = [AtomData(element=String(atomic_symbol(a))) for a in crystal.atoms]
    coords = [SVector{D}(AtomsBase.position(crystal, i)) for i in 1:length(crystal)]

    # Build bounding box
    side_lengths = norm.(AtomsBase.cell_vectors(crystal))
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

function Base.zero(sys::System{D, AT, T, A, C, B, V,
                   AD, TO, PI, SI, GI, CN, VS, VF, NF, L, F, E, K, M, TM, DA}) where {D, AT, T,
                            A, C, B, V, AD, TO, PI, SI, GI, CN, VS, VF, NF, L, F, E, K, M, TM, DA}
    return System{D, AT, T, A, C, B, V, AD, TO, PI, SI, GI, CN, VS, VF, NF, L, F, E, K, M, TM, DA}(
        zero.(sys.atoms),
        zero(sys.coords),
        zero(sys.boundary),
        zero(sys.velocities),
        sys.atoms_data,
        sys.topology,
        zero.(sys.pairwise_inters),
        zero.(sys.specific_inter_lists),
        zero.(sys.general_inters),
        sys.constraints,
        sys.virtual_sites,
        sys.virtual_site_flags,
        sys.neighbor_finder,
        sys.loggers,
        sys.df,
        sys.force_units,
        sys.energy_units,
        zero(sys.k),
        zero(sys.masses),
        zero(sys.total_mass),
        sys.data,
    )
end

# Add parameters from a dictionary to a system, allowing gradients to be tracked
function inject_gradients(sys::System{<:Any, AT}, params_dic) where AT
    atoms_grad = to_device(inject_atom.(from_device(sys.atoms), sys.atoms_data, (params_dic,)), AT)
    if length(sys.pairwise_inters) > 0
        pis_grad = inject_interaction.(sys.pairwise_inters, (params_dic,))
    else
        pis_grad = sys.pairwise_inters
    end
    if length(sys.specific_inter_lists) > 0
        sis_grad = inject_interaction_list.(sys.specific_inter_lists, (params_dic,), AT)
    else
        sis_grad = sys.specific_inter_lists
    end
    if length(sys.general_inters) > 0
        gis_grad = inject_interaction.(sys.general_inters, (params_dic,), (sys,))
    else
        gis_grad = sys.general_inters
    end
    return atoms_grad, pis_grad, sis_grad, gis_grad
end

# Form a dictionary of all parameters in a system, allowing gradients to be tracked
function extract_parameters(sys, ff)
    params_dic = Dict()

    for at_data in sys.atoms_data
        key_prefix = "atom_$(at_data.atom_type)_"
        if !haskey(params_dic, key_prefix * "mass")
            at = ff.atom_types[at_data.atom_type]
            params_dic[key_prefix * "mass"] = at.mass
            params_dic[key_prefix * "σ"   ] = at.σ
            params_dic[key_prefix * "ϵ"   ] = at.ϵ
        end
    end

    for inter in values(sys.pairwise_inters)
        extract_parameters!(params_dic, inter, ff)
    end

    for inter in values(sys.specific_inter_lists)
        extract_parameters!(params_dic, inter, ff)
    end

    return params_dic
end

@doc raw"""
    ThermoState(system::System, integrator; <keyword arguments>)

Thermodynamic state wrapper carrying the system, integrator, and derived thermodynamic properties 
(inverse temperature `beta` and pressure `p`). This serves as the definitive container for a 
single thermodynamic state across all generalized ensemble methods.

# Arguments
- `system::System`: The simulation system used to evaluate energies.
- `integrator`: The integrator used to simulate the system.
- `temperature=nothing`: Explicit target temperature. If `nothing`, it is inferred from the integrator's 
    thermostat or implicit temperature coupling.
- `pressure=nothing`: Explicit target pressure. If `nothing`, it is inferred from the integrator's barostat.
- `name::AbstractString=nothing`: A label for the state. If not provided, a default name based on 
    the temperature and pressure is generated.
"""
struct ThermoState{S, I, B, P}
    system::S
    integrator::I
    beta::B         # Inverse temperature in internal energy units
    p::P            # Isotropic pressure in internal pressure units
    name::String
end

function ThermoState(sys::System{D, AT, FT}, integrator; 
                     temperature=nothing, pressure=nothing,
                     name::Union{Nothing, AbstractString}=nothing) where {D, AT, FT}

    temp_source = temperature
    press_source = pressure

    # Infer thermodynamic targets from the integrator if not explicitly overridden
    if hasproperty(integrator, :coupling) && !(integrator.coupling isa NoCoupling)
        couplers = integrator.coupling isa Tuple ? integrator.coupling : (integrator.coupling,)
        for coupler in couplers
            if isnothing(temp_source) && coupler isa AbstractThermostat
                temp_source = coupler.temperature
            end
            if isnothing(press_source) && coupler isa AbstractBarostat
                if hasproperty(coupler, :pressure)
                    press_source = coupler.pressure
                end
            end
        end
    end

    # Check for integrators with implicit temperature control (e.g., Langevin, NoseHoover)
    if isnothing(temp_source) && hasproperty(integrator, :temperature)
        temp_source = integrator.temperature
    end

    if isnothing(temp_source)
        throw(ArgumentError("No temperature provided or inferred from the integrator. " * "You must provide an explicit temperature, use a thermostat, or " * "use an integrator with an implicit temperature."))
    end

    # Calculate beta (inverse temperature) in system-compatible units (e.g., mol/kJ)
    # Molly evaluates energy per mole by default, requiring the molar gas constant R
    beta_val = try
        kBT = uconvert(sys.energy_units, Unitful.R * temp_source)
        FT(ustrip(1/kBT))
    catch
        throw(ArgumentError("Temperature provided is not compatible with system energy units."))
    end

    # Calculate isotropic pressure in internal units (Energy / Volume) if applicable
    p_val = nothing
    if !isnothing(press_source)
        # Handle scalar pressure or tensor (matrix) pressure representations
        p_raw = press_source isa AbstractArray ? (1/3 * tr(press_source)) : press_source
        
        l_unit = unit(sys.boundary.side_lengths[1])
        v_unit = l_unit^3
        p_unit = sys.energy_units / v_unit

        # Convert macroscopic pressure (e.g., bar) to internal molar pressure
        p_molar = p_raw * Unitful.Na
        p_val = FT(ustrip(uconvert(p_unit, p_molar)))
    end

    final_name = isnothing(name) ? "state_T$(temp_source)" * (isnothing(p_val) ? "" : "_P$(press_source)") : String(name)
    
    return ThermoState{typeof(sys), typeof(integrator), typeof(beta_val), typeof(p_val)}(
        sys, integrator, beta_val, p_val, final_name
    )
end

@doc raw"""
    ReplicaSystem(thermo_states, replica_coords; <keyword arguments>)

A wrapper for replicas in a generalized ensemble or replica exchange simulation (REMD).

Instead of instantiating completely disjoint [`System`](@ref) objects, `ReplicaSystem` automatically compiles 
an [`AlchemicalPartition`](@ref) from the provided [`ThermoState`](@ref) vector. This isolates shared, unperturbed 
topological and interactive components (e.g., bulk solvent) from state-specific perturbations. 
During an exchange attempt, cross-energies are evaluated efficiently by querying only the 
necessary subset of perturbed interactions, completely avoiding redundant evaluations of the shared system.

Furthermore, upon a successful exchange, coordinates and velocities are no longer physically swapped in memory; 
the system simply updates internal pointers mapping the physical replica to its new thermodynamic state.

This is a sub-type of [`AbstractSystem`](@ref) from AtomsBase.jl and implements the interface described there, 
routing standard atomic property queries back to the unperturbed master system.

When using `ReplicaSystem` with [`CellListMapNeighborFinder`](@ref), the number of threads used for
both the simulation of replicas and the neighbor finder should be set to be the same.
This can be done by passing `nbatches=(min(n, 8), n)` to [`CellListMapNeighborFinder`](@ref) during
construction where `n` is the number of threads to be used per replica.

# Arguments
- `thermo_states::AbstractArray{<:ThermoState}`: An array of thermodynamic states defining the 
    replica ladder. Each state encapsulates its specific `System` interactions, integrator, 
    inverse temperature (`beta`), and pressure (`p`). The number of states dictates `n_replicas`.
- `replica_coords`: The coordinates of the atoms in each replica. The number of elements 
    must equal the length of `thermo_states`.
- `replica_velocities=nothing`: The velocities of the atoms in each replica. If not provided, 
    they default to zero velocities using the system's units.
- `replica_boundaries=nothing`: The bounding box for each replica. If not provided, it defaults 
    to duplicating the boundary of the reference system (the first `ThermoState`).
- `exchange_logger::EL=ReplicaExchangeLogger(n_replicas)`: The logger used to record
    the exchange of replicas.
- `data::DA=nothing`: Arbitrary data associated with the replica system.
- `reuse_neighbors::Bool=true`: Whether to reuse the active system's neighbor list when calculating
    energies for perturbed state differences. Generally improves performance.
"""
mutable struct ReplicaSystem{D, AT, T, P, C, V, B, EL, DA} <: AtomsBase.AbstractSystem{D}
    partition::P
    n_replicas::Int
    replica_coords::C
    replica_velocities::V
    replica_boundaries::B
    state_indices::Vector{Int}
    exchange_logger::EL
    data::DA
end

function ReplicaSystem(thermo_states::AbstractArray{<:ThermoState},
                       replica_coords;
                       replica_velocities=nothing,
                       replica_boundaries=nothing,
                       exchange_logger=nothing,
                       data=nothing,
                       reuse_neighbors::Bool=true)
    
    n_replicas = length(thermo_states)
    
    if length(replica_coords) != n_replicas
        throw(ArgumentError("Number of replica_coords ($(length(replica_coords))) " *
                            "does not match number of ThermoStates ($n_replicas)"))
    end

    ref_sys = thermo_states[1].system
    D = AtomsBase.n_dimensions(ref_sys.boundary)
    T = float_type(ref_sys.boundary)
    AT = array_type(replica_coords[1])

    if isnothing(replica_boundaries)
        replica_boundaries = [ref_sys.boundary for _ in 1:n_replicas]
    elseif length(replica_boundaries) != n_replicas
        throw(ArgumentError("Number of boundaries ($(length(replica_boundaries))) " *
                            "does not match number of replicas ($n_replicas)"))
    end

    if isnothing(replica_velocities)
        if ref_sys.force_units == NoUnits
            replica_velocities = [zero(replica_coords[1]) for _ in 1:n_replicas]
        else
            replica_velocities = [zero(replica_coords[1]) * u"ps^-1" for _ in 1:n_replicas]
        end
    elseif length(replica_velocities) != n_replicas
        throw(ArgumentError("Number of velocities ($(length(replica_velocities))) " *
                            "does not match number of replicas ($n_replicas)"))
    end

    if isnothing(exchange_logger)
        exchange_logger = ReplicaExchangeLogger(T, n_replicas)
    end

    # Initialize the AlchemicalPartition using the array of ThermoStates
    partition = AlchemicalPartition(thermo_states; reuse_neighbors=reuse_neighbors)
    
    # Track which thermodynamic state is currently assigned to each replica
    # Initially, replica i is in state i
    state_indices = collect(1:n_replicas)

    return ReplicaSystem{D, AT, T, typeof(partition), typeof(replica_coords), 
                         typeof(replica_velocities), typeof(replica_boundaries), 
                         typeof(exchange_logger), typeof(data)}(
        partition, n_replicas, replica_coords, replica_velocities, replica_boundaries, 
        state_indices, exchange_logger, data
    )
end


function AtomsBase.atomic_number(s::ReplicaSystem)
    if length(s.partition.master_sys.atoms_data) > 0
        return map(s.partition.master_sys.atoms_data) do ad
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


# Avoid unnecessary Array calls on CPU
from_device(x::Array) = x
from_device(x) = Array(x)

to_device(x::Array, ::Type{<:Array}) = x
to_device(x, ::Type{AT}) where AT = AT(x)

"""
    array_type(sys)
    array_type(arr)

The array type of a [`System`](@ref), [`ReplicaSystem`](@ref) or array, for example
`Array` for systems on CPU or `CuArray` for systems on a NVIDIA GPU.
"""
array_type(::AT) where AT = AT.name.wrapper
array_type(::Union{System{<:Any, AT}, ReplicaSystem{<:Any, AT}}) where {AT} = AT

"""
    is_on_gpu(sys)
    is_on_gpu(arr)

Whether a [`System`](@ref), [`ReplicaSystem`](@ref) or array type is on the GPU.
"""
function is_on_gpu(::Union{System{<:Any, AT}, ReplicaSystem{<:Any, AT}, AT}) where AT
    return AT <: AbstractGPUArray
end

"""
    float_type(sys)
    float_type(boundary)

The float type a [`System`](@ref), [`ReplicaSystem`](@ref) or bounding box uses.
"""
float_type(::Union{System{<:Any, <:Any, T}, ReplicaSystem{<:Any, <:Any, T}}) where {T} = T

"""
    masses(sys)

The masses of the atoms in a [`System`](@ref) or [`ReplicaSystem`](@ref).
"""
masses(s::System) = s.masses
masses(s::ReplicaSystem) = mass.(s.partition.master_sys.atoms)

"""
    charges(sys)

The partial charges of the atoms in a [`System`](@ref) or [`ReplicaSystem`](@ref).
"""
charges(s::System) = charge.(s.atoms)
charges(s::ReplicaSystem) = charge.(s.partition.master_sys.atoms)
charge(s::System, i::Integer) = charge(s.atoms[i])
charge(s::ReplicaSystem, i::Integer) = charge(s.partition.master_sys.atoms[i])
charge(s::System, ::Colon) = charge.(s.atoms)
charge(s::ReplicaSystem, ::Colon) = charge.(s.partition.master_sys.atoms)

# Separate methods to avoid method ambiguity with AtomsBase
Base.getindex(s::System, i::Integer) = s.atoms[i]
Base.getindex(s::ReplicaSystem, i::Integer) = s.partition.master_sys.atoms[i]
Base.getindex(s::System, is::AbstractVector{Bool}) = s.atoms[is]
Base.getindex(s::ReplicaSystem, is::AbstractVector{Bool}) = s.partition.master_sys.atoms[is]
Base.length(s::System) = length(s.atoms)
Base.length(s::ReplicaSystem) = length(s.partition.master_sys.atoms)
Base.eachindex(s::Union{System, ReplicaSystem}) = Base.OneTo(length(s))

AtomsBase.atomkeys(s::Union{System, ReplicaSystem}) = (:position, :velocity, :mass, :atomic_number, :charge)
AtomsBase.haskey(at::Atom, x::Symbol) = x in (:position, :velocity, :mass, :atomic_number, :charge)
AtomsBase.hasatomkey(s::Union{System, ReplicaSystem}, x::Symbol) = x in atomkeys(s)
AtomsBase.keys(sys::Union{System, ReplicaSystem}) = fieldnames(typeof(sys))
AtomsBase.haskey(sys::Union{System, ReplicaSystem}, x::Symbol) = hasfield(typeof(sys), x)
Base.getindex(sys::Union{System, ReplicaSystem}, x::Symbol) =
    hasfield(typeof(sys), x) ? getfield(sys, x) : KeyError("no field `$x`, allowed keys are $(keys(sys))")
Base.pairs(sys::Union{System, ReplicaSystem}) = (k => sys[k] for k in keys(sys))
Base.get(sys::Union{System, ReplicaSystem}, x::Symbol, default) =
    haskey(sys, x) ? getfield(sys, x) : default

AtomsBase.position(s::System, i::Union{Integer, AbstractVector}) = s.coords[i]
AtomsBase.position(s::System, ::Colon) = s.coords
AtomsBase.position(s::ReplicaSystem, i::Union{Integer, AbstractVector}) = s.replica_coords[1][i]
AtomsBase.position(s::ReplicaSystem, ::Colon) = s.replica_coords[1]

AtomsBase.velocity(s::System, i::Union{Integer, AbstractVector}) = s.velocities[i]
AtomsBase.velocity(s::System, ::Colon) = s.velocities
AtomsBase.velocity(s::ReplicaSystem, i::Union{Integer, AbstractVector}) = s.replica_velocities[1][i]
AtomsBase.velocity(s::ReplicaSystem, ::Colon) = s.replica_velocities[1]

AtomsBase.mass(s::Union{System, ReplicaSystem}, i::Union{Integer, AbstractVector}) = mass(s.atoms[i])
AtomsBase.mass(s::System, ::Colon) = s.masses
AtomsBase.mass(s::ReplicaSystem, ::Colon) = mass.(s.partition.master_sys.atoms)

function AtomsBase.species(s::System, i::Integer)
    return AtomsBase.ChemicalSpecies(Symbol(s.atoms_data[i].element))
end
function AtomsBase.species(s::ReplicaSystem, i::Integer)
    return AtomsBase.ChemicalSpecies(Symbol(s.partition.master_sys.atoms_data[i].element))
end

function AtomsBase.species(s::System, i::Union{AbstractVector, Colon})
    return AtomsBase.ChemicalSpecies.(Symbol.(getfield.(s.atoms_data[i], :element)))
end
function AtomsBase.species(s::ReplicaSystem, i::Union{AbstractVector, Colon})
    return AtomsBase.ChemicalSpecies.(Symbol.(getfield.(s.partition.master_sys.atoms_data[i], :element)))
end

function Base.getindex(sys::Union{System, ReplicaSystem}, i, x::Symbol)
    atomsbase_keys = (:position, :velocity, :mass, :atomic_number)
    if hasatomkey(sys, x)
        if x in atomsbase_keys
            return getproperty(AtomsBase, x)(sys, i)
        elseif x == :charge
            return getproperty(Molly, x)(sys, i) * u"e_au" # AtomsBase has units on charge
        end
    else
        throw(KeyError("key $x not present in the system"))
    end
end

function AtomsBase.atomic_symbol(s::System)
    if length(s.atoms_data) > 0
        return map(ad -> Symbol(ad.element), s.atoms_data)
    else
        return fill(:unknown, length(s))
    end
end
function AtomsBase.atomic_symbol(s::ReplicaSystem)
    if length(s.partition.master_sys.atoms_data) > 0
        return map(ad -> Symbol(ad.element), s.partition.master_sys.atoms_data)
    else
        return fill(:unknown, length(s))
    end
end

function AtomsBase.atomic_symbol(s::System, i::Integer)
    if length(s.atoms_data) > 0
        return Symbol(s.atoms_data[i].element)
    else
        return :unknown
    end
end
function AtomsBase.atomic_symbol(s::ReplicaSystem, i::Integer)
    if length(s.partition.master_sys.atoms_data) > 0
        return Symbol(s.partition.master_sys.atoms_data[i].element)
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

function AtomsBase.atomic_number(s::System , i::Integer)
    if length(s.atoms_data) > 0 && s.atoms_data[i].element != "?"
        return PeriodicTable.elements[Symbol(s.atoms_data[i].element)].number
    else
        return :unknown
    end
end
function AtomsBase.atomic_number(s::ReplicaSystem, i::Integer)
    if length(s.partition.master_sys.atoms_data) > 0 && s.partition.master_sys.atoms_data[i].element != "?"
        return PeriodicTable.elements[Symbol(s.partition.master_sys.atoms_data[i].element)].number
    else
        return :unknown
    end
end

AtomsBase.cell_vectors(s::System) = AtomsBase.cell_vectors(s.boundary)
AtomsBase.cell_vectors(s::ReplicaSystem) = AtomsBase.cell_vectors(s.replica_boundaries[1])

function AtomsBase.cell(sys::System{D}) where D
    return AtomsBase.PeriodicCell(
        AtomsBase.cell_vectors(sys),
        has_infinite_boundary(sys) ? ntuple(i -> !isinf(sys.boundary.side_lengths[i]), D) :
                                     ntuple(i -> true, D),
    )
end

AtomsBase.cell(sys::ReplicaSystem) = AtomsBase.cell(sys.partition.master_sys)

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
    System(abstract_system; <keyword arguments>)

Convert an AtomsBase `AbstractSystem` to a Molly `System`.

The keyword arguments `force_units` and `energy_units` should be set as appropriate.
Other keyword arguments are the same as for the main `System` constructor.
"""
function System(sys::AtomsBase.AbstractSystem{D};
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
    bb = AtomsBase.cell_vectors(sys)
    is_cubic = true
    for (i, bv) in enumerate(bb)
        for j in 1:(i - 1)
            if !iszero_value(bv[j])
                is_cubic = false
            end
        end
    end
    box_lengths = norm.(bb)

    if is_cubic && D == 2
        @warn "Molly RectangularBoundary assumes origin at (0, 0, 0)"
        molly_boundary = RectangularBoundary(box_lengths...)
    elseif is_cubic && D == 3
        @warn "Molly CubicBoundary assumes origin at (0, 0, 0)"
        molly_boundary = CubicBoundary(box_lengths...)
    elseif D == 3
        @warn "Molly TriclinicBoundary assumes origin at (0, 0, 0)"
        if any(isinf, box_lengths)
            throw(ArgumentError("TriclinicBoundary does not support infinite boundaries"))
        end
        molly_boundary = TriclinicBoundary(bb...)
    else
        throw(ArgumentError("Molly does not support 2D triclinic boundaries"))
    end

    length_unit = unit(first(AtomsBase.position(sys, 1)))
    atoms = Vector{Atom}(undef, (length(sys),))
    atoms_data = Vector{AtomData}(undef, (length(sys),))
    for (i, atom) in enumerate(sys)
        atoms[i] = Atom(
            index=i,
            charge=ustrip(get(atom, :charge, 0.0)), # Remove e unit
            mass=AtomsBase.mass(atom),
            σ=(0.0 * length_unit),
            ϵ=(0.0 * energy_units),
        )
        atoms_data[i] = AtomData(element=String(Symbol(AtomsBase.atomic_symbol(atom))))
    end

    # AtomsBase does not specify a type for coordinates or velocities so we convert to SVector
    if !(:position in AtomsBase.atomkeys(sys))
        throw(ArgumentError("failed to construct Molly System from AbstractSystem, " *
                            "missing position key"))
    end
    coords = map(AtomsBase.position(sys, :)) do r
        SVector(r...)
    end

    if :velocity in AtomsBase.atomkeys(sys)
        vels = map(AtomsBase.velocity(sys, :)) do v
            SVector(v...)
        end
    else
        vels = nothing
    end

    mass_dim = dimension(AtomsBase.mass(sys, 1))
    if mass_dim == u"𝐌" && dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        throw(ArgumentError("when constructing System from AbstractSystem, energy units " *
                            "are molar but mass units are not"))
    elseif mass_dim == u"𝐌 * 𝐍^-1" && dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐓^-2"
        throw(ArgumentError("when constructing System from AbstractSystem, mass units " *
                            "are molar but energy units are not"))
    end

    return System(
        atoms=atoms,
        coords=coords,
        boundary=molly_boundary,
        velocities=vels,
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
    Should be a `Tuple` or `NamedTuple` of `PairwiseInteraction`s.
- `specific_inter_lists::SI=()`: the specific interactions in the system,
    i.e. interactions between specific atoms such as bonds or angles.
    Should be a `Tuple` or `NamedTuple`.
- `general_inters::GI=()`: the general interactions in the system,
    i.e. interactions involving all atoms such as implicit solvent. Each should
    implement the AtomsCalculators.jl interface. Should be a `Tuple` or `NamedTuple`.
- `neighbor_finder::NF=NoNeighborFinder()`: the neighbor finder used to find
    close atoms and save on computation.
- `force_units::F=u"kJ * mol^-1 * nm^-1"`: the units of force of the system.
    Should be set to `NoUnits` if units are not being used.
- `energy_units::E=u"kJ * mol^-1"`: the units of energy of the system. Should
    be set to `NoUnits` if units are not being used.
- `k::K=Unitful.k` or `Unitful.k * Unitful.Na`: the Boltzmann constant, which may be
    modified in some simulations. `k` is chosen based on the `energy_units` given.
- `dims::Integer=3`: the number of dimensions in the system.
"""
struct MollyCalculator{D, PI, SI, GI, NF, F, E, K}
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
        dims::Integer=3,
    )
    return MollyCalculator{dims, typeof(pairwise_inters), typeof(specific_inter_lists),
                           typeof(general_inters), typeof(neighbor_finder), typeof(force_units),
                           typeof(energy_units), typeof(k)}(
            pairwise_inters, specific_inter_lists, general_inters, neighbor_finder,
            force_units, energy_units, k)
end

AtomsCalculators.energy_unit(calc::MollyCalculator) = calc.energy_units
AtomsCalculators.length_unit(calc::MollyCalculator) = calc.energy_units / calc.force_units
AtomsCalculators.force_unit(calc::MollyCalculator) = calc.force_units

function AtomsCalculators.promote_force_type(::AtomsBase.AbstractSystem{D},
                                             calc::MollyCalculator{D}) where D
    T = typeof(ustrip(calc.k))
    return typeof(ones(SVector{D, T}) * calc.force_units)
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(
        abstract_sys,
        calc::MollyCalculator;
        neighbors=nothing,
        step_n::Integer=0,
        n_threads::Integer=Threads.nthreads(),
        kwargs...,
    )
    sys_nointers = System(abstract_sys; force_units=calc.force_units,
                          energy_units=calc.energy_units)
    sys = System(
        sys_nointers;
        pairwise_inters=calc.pairwise_inters,
        specific_inter_lists=calc.specific_inter_lists,
        general_inters=calc.general_inters,
        neighbor_finder=calc.neighbor_finder,
        k=calc.k,
    )
    nbs = (isnothing(neighbors) ? find_neighbors(sys) : neighbors)
    return forces(sys, nbs, step_n; n_threads=n_threads)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
        abstract_sys,
        calc::MollyCalculator;
        neighbors=nothing,
        step_n::Integer=0,
        n_threads::Integer=Threads.nthreads(),
        kwargs...,
    )
    sys_nointers = System(abstract_sys; force_units=calc.force_units,
                          energy_units=calc.energy_units)
    sys = System(
        sys_nointers;
        pairwise_inters=calc.pairwise_inters,
        specific_inter_lists=calc.specific_inter_lists,
        general_inters=calc.general_inters,
        neighbor_finder=calc.neighbor_finder,
        k=calc.k,
    )
    nbs = (isnothing(neighbors) ? find_neighbors(sys) : neighbors)
    return potential_energy(sys, nbs, nothing, step_n; n_threads=n_threads)
end

"""
    ASECalculator(; <keyword arguments>)

A Python [ASE](https://wiki.fysik.dtu.dk/ase) calculator.

This calculator is only available when PythonCall is imported.
It is the user's responsibility to have the required Python packages installed.
This includes ASE and any packages providing the calculator.

Contrary to the rest of Molly, unitless quantities are assumed to have ASE units:
Å for length, eV for energy, u for mass, and Å sqrt(u/eV) for time.
Unitful quantities will be converted as appropriate.

Not currently compatible with [`TriclinicBoundary`](@ref).
Not currently compatible with virial calculation.

# Arguments
- `ase_calc`: the ASE calculator created with PythonCall.
- `atoms`: the atoms, or atom equivalents, in the system.
- `coords`: the coordinates of the atoms in the system. Typically a
    vector of `SVector`s of 2 or 3 dimensions.
- `boundary`: the bounding box in which the simulation takes place.
- `elements=nothing`: vector of atom elements as a string, either `elements` or
    `atoms_data` (which contains element data) must be provided.
- `atoms_data=nothing`: other data associated with the atoms.
- `velocities=nothing`: the velocities of the atoms in the system, only required
    if the velocities contribute to the potential energy or forces.
"""
struct ASECalculator{T}
    ase_atoms::T # T will be Py but that is not available here
    ase_calc::T
end

function update_ase_calc! end

# ForwardDiff.jl checks both value and derivative
# This could be extended to only check the value for Duals
iszero_value(x) = iszero(x)

# Only use threading if a condition is true
macro maybe_threads(flag, expr)
    quote
        if $(flag)
            Threads.@threads $expr
        else
            $expr
        end
    end |> esc
end

function check_strictness(strictness)
    if !(strictness in (:warn, :nowarn, :error))
        throw(ArgumentError("strictness argument must be :warn, :nowarn or :error, " *
                            "found $strictness"))
    end
end

function report_issue(err_str, strictness)
    if strictness == :warn
        @warn err_str
    elseif strictness == :error
        error(err_str)
    end
end
