# Read files to set up a system
# See OpenMM source code
# See http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    place_atoms,
    place_diatomics,
    AtomType,
    ResidueType,
    PeriodicTorsionType,
    MolecularForceField,
    is_any_atom,
    is_heavy_atom,
    add_position_restraints

"""
    place_atoms(n_atoms, boundary; min_dist=nothing, max_attempts=100)

Generate random coordinates.

Obtain `n_atoms` coordinates in bounding box `boundary` where no two
points are closer than `min_dist`, accounting for periodic boundary conditions.
The keyword argument `max_attempts` determines the number of failed tries after
which to stop placing atoms.
Can not be used if one or more dimensions has infinite boundaries.
"""
function place_atoms(n_atoms::Integer,
                     boundary;
                     min_dist=zero(length_type(boundary)),
                     max_attempts::Integer=100)
    if has_infinite_boundary(boundary)
        throw(ArgumentError("one or more dimension has infinite boundaries, boundary is $boundary"))
    end
    dims = AtomsBase.n_dimensions(boundary)
    max_atoms = box_volume(boundary) / (min_dist ^ dims)
    if n_atoms > max_atoms
        throw(ArgumentError("boundary $boundary too small for $n_atoms atoms with minimum distance $min_dist"))
    end
    min_dist_sq = min_dist ^ 2
    coords = SArray[]
    sizehint!(coords, n_atoms)
    failed_attempts = 0
    while length(coords) < n_atoms
        new_coord = random_coord(boundary)
        okay = true
        if min_dist > zero(min_dist)
            for coord in coords
                if sum(abs2, vector(coord, new_coord, boundary)) < min_dist_sq
                    okay = false
                    failed_attempts += 1
                    break
                end
            end
        end
        if okay
            push!(coords, new_coord)
            failed_attempts = 0
        elseif failed_attempts >= max_attempts
            error("failed to place atom $(length(coords) + 1) after $max_attempts (max_attempts) tries")
        end
    end
    return [coords...]
end

"""
    place_diatomics(n_molecules, boundary, bond_length; min_dist=nothing,
                    max_attempts=100, aligned=false)

Generate random diatomic molecule coordinates.

Obtain coordinates for `n_molecules` diatomics in bounding box `boundary`
where no two points are closer than `min_dist` and the bond length is `bond_length`,
accounting for periodic boundary conditions.
The keyword argument `max_attempts` determines the number of failed tries after
which to stop placing atoms.
The keyword argument `aligned` determines whether the bonds all point the same direction
(`true`) or random directions (`false`).
Can not be used if one or more dimensions has infinite boundaries.
"""
function place_diatomics(n_molecules::Integer,
                         boundary,
                         bond_length;
                         min_dist=zero(length_type(boundary)),
                         max_attempts::Integer=100,
                         aligned::Bool=false)
    if has_infinite_boundary(boundary)
        throw(ArgumentError("one or more dimension has infinite boundaries, boundary is $boundary"))
    end
    dims = AtomsBase.n_dimensions(boundary)
    max_molecules = box_volume(boundary) / ((min_dist + bond_length) ^ dims)
    if n_molecules > max_molecules
        throw(ArgumentError("boundary $boundary too small for $n_molecules diatomics with minimum distance $min_dist"))
    end
    min_dist_sq = min_dist ^ 2
    coords = SArray[]
    sizehint!(coords, 2 * n_molecules)
    failed_attempts = 0
    while length(coords) < (n_molecules * 2)
        new_coord_a = random_coord(boundary)
        if aligned
            shift = SVector{dims}([bond_length, [zero(bond_length) for d in 1:(dims - 1)]...])
        else
            shift = bond_length * normalize(randn(SVector{dims, typeof(ustrip(bond_length))}))
        end
        new_coord_b = copy(new_coord_a) + shift
        okay = true
        if min_dist > zero(min_dist)
            for coord in coords
                if sum(abs2, vector(coord, new_coord_a, boundary)) < min_dist_sq ||
                        sum(abs2, vector(coord, new_coord_b, boundary)) < min_dist_sq
                    okay = false
                    failed_attempts += 1
                    break
                end
            end
        end
        if okay
            push!(coords, new_coord_a)
            push!(coords, new_coord_b)
            failed_attempts = 0
        elseif failed_attempts >= max_attempts
            error("failed to place atom $(length(coords) + 1) after $max_attempts (max_attempts) tries")
        end
    end
    # Second atom in each molecule may be outside boundary
    return wrap_coords.([coords...], (boundary,))
end

"""
    AtomType(type, class, element, charge, mass, σ, ϵ)

An atom type.
"""
struct AtomType{C, M, S, E}
    type::String
    class::String # Currently this is not used
    element::String
    charge::Union{C, Missing}
    mass::M
    σ::S
    ϵ::E
end

"""
    ResidueType(name, types, charges, indices)

A residue type.
"""
struct ResidueType{C}
    name::String
    types::Dict{String, String}
    charges::Dict{String, Union{C, Missing}}
    indices::Dict{String, Int}
end

"""
    PeriodicTorsionType(periodicities, phases, ks, proper)

A periodic torsion type.
"""
struct PeriodicTorsionType{T, E}
    periodicities::Vector{Int}
    phases::Vector{T}
    ks::Vector{E}
    proper::Bool
end

"""
    MolecularForceField(ff_files...; units=true)
    MolecularForceField(T, ff_files...; units=true)
    MolecularForceField(atom_types, residue_types, bond_types, angle_types,
                        torsion_types, torsion_order, weight_14_coulomb,
                        weight_14_lj, attributes_from_residue)

A molecular force field.

Read one or more OpenMM force field XML files by passing them to the constructor.
"""
struct MolecularForceField{T, M, D, E, K}
    atom_types::Dict{String, AtomType{T, M, D, E}}
    residue_types::Dict{String, ResidueType{T}}
    bond_types::Dict{Tuple{String, String}, HarmonicBond{K, D}}
    angle_types::Dict{Tuple{String, String, String}, HarmonicAngle{E, T}}
    torsion_types::Dict{Tuple{String, String, String, String}, PeriodicTorsionType{T, E}}
    torsion_order::String
    weight_14_coulomb::T
    weight_14_lj::T
    attributes_from_residue::Vector{String}
end

function MolecularForceField(T::Type, ff_files::AbstractString...; units::Bool=true)
    atom_types = Dict{String, AtomType}()
    residue_types = Dict{String, ResidueType}()
    bond_types = Dict{Tuple{String, String}, HarmonicBond}()
    angle_types = Dict{Tuple{String, String, String}, HarmonicAngle}()
    torsion_types = Dict{Tuple{String, String, String, String}, PeriodicTorsionType}()
    torsion_order = ""
    weight_14_coulomb, weight_14_lj = one(T), one(T)
    weight_14_coulomb_set, weight_14_lj_set = false, false
    attributes_from_residue = String[]

    for ff_file in ff_files
        ff_xml = parsexml(read(ff_file))
        ff = root(ff_xml)
        for entry in eachelement(ff)
            entry_name = entry.name
            if entry_name == "AtomTypes"
                for atom_type in eachelement(entry)
                    at_type = atom_type["name"]
                    at_class = atom_type["class"]
                    element = haskey(atom_type, "element") ? atom_type["element"] : "?"
                    ch = missing # Updated later or defined in residue
                    atom_mass = units ? parse(T, atom_type["mass"])u"g/mol" : parse(T, atom_type["mass"])
                    σ = units ? T(-1u"nm") : T(-1) # Updated later
                    ϵ = units ? T(-1u"kJ * mol^-1") : T(-1) # Updated later
                    atom_types[at_type] = AtomType{T, typeof(atom_mass), typeof(σ), typeof(ϵ)}(
                                                at_type, at_class, element, ch, atom_mass, σ, ϵ)
                end
            elseif entry_name == "Residues"
                for residue in eachelement(entry)
                    name = residue["name"]
                    types = Dict{String, String}()
                    atom_charges = Dict{String, Union{T, Missing}}()
                    indices = Dict{String, Int}()
                    index = 1
                    for atom_or_bond in eachelement(residue)
                        # Ignore bonds because they are specified elsewhere
                        if atom_or_bond.name == "Atom"
                            atom_name = atom_or_bond["name"]
                            types[atom_name] = atom_or_bond["type"]
                            if haskey(atom_or_bond, "charge")
                                atom_charges[atom_name] = parse(T, atom_or_bond["charge"])
                            else
                                atom_charges[atom_name] = missing
                            end
                            indices[atom_name] = index
                            index += 1
                        elseif atom_or_bond.name == "VirtualSite"
                            @warn "Virtual sites not currently supported, this entry will be ignored"
                        end
                    end
                    residue_types[name] = ResidueType(name, types, atom_charges, indices)
                end
            elseif entry_name == "HarmonicBondForce"
                for bond in eachelement(entry)
                    if haskey(bond, "class1")
                        @warn "Atom classes not currently supported, this $entry_name entry will be ignored"
                        continue
                    end
                    atom_type_1 = bond["type1"]
                    atom_type_2 = bond["type2"]
                    k = units ? parse(T, bond["k"])u"kJ * mol^-1 * nm^-2" : parse(T, bond["k"])
                    r0 = units ? parse(T, bond["length"])u"nm" : parse(T, bond["length"])
                    bond_types[(atom_type_1, atom_type_2)] = HarmonicBond(k, r0)
                end
            elseif entry_name == "HarmonicAngleForce"
                for ang in eachelement(entry)
                    if haskey(ang, "class1")
                        @warn "Atom classes not currently supported, this $entry_name entry will be ignored"
                        continue
                    end
                    atom_type_1 = ang["type1"]
                    atom_type_2 = ang["type2"]
                    atom_type_3 = ang["type3"]
                    k = units ? parse(T, ang["k"])u"kJ * mol^-1" : parse(T, ang["k"])
                    θ0 = parse(T, ang["angle"])
                    angle_types[(atom_type_1, atom_type_2, atom_type_3)] = HarmonicAngle(k, θ0)
                end
            elseif entry_name == "PeriodicTorsionForce"
                torsion_order = haskey(entry, "ordering") ? entry["ordering"] : "default"
                for torsion in eachelement(entry)
                    if haskey(torsion, "class1")
                        @warn "Atom classes not currently supported, this $entry_name entry will be ignored"
                        continue
                    end
                    proper = torsion.name == "Proper"
                    atom_type_1 = torsion["type1"]
                    atom_type_2 = torsion["type2"]
                    atom_type_3 = torsion["type3"]
                    atom_type_4 = torsion["type4"]
                    periodicities = Int[]
                    phases = T[]
                    ks = units ? typeof(T(1u"kJ * mol^-1"))[] : T[]
                    phase_i = 1
                    phase_present = true
                    while phase_present
                        push!(periodicities, parse(Int, torsion["periodicity$phase_i"]))
                        push!(phases, parse(T, torsion["phase$phase_i"]))
                        push!(ks, units ? parse(T, torsion["k$phase_i"])u"kJ * mol^-1" : parse(T, torsion["k$phase_i"]))
                        phase_i += 1
                        phase_present = haskey(torsion, "periodicity$phase_i")
                    end
                    torsion_type = PeriodicTorsionType(periodicities, phases, ks, proper)
                    torsion_types[(atom_type_1, atom_type_2, atom_type_3, atom_type_4)] = torsion_type
                end
            elseif entry_name == "NonbondedForce"
                if haskey(entry, "coulomb14scale")
                    weight_14_coulomb_new = parse(T, entry["coulomb14scale"])
                    if weight_14_coulomb_set && weight_14_coulomb_new != weight_14_coulomb
                        error("found multiple NonbondedForce entries with different coulomb14scale values")
                    end
                    weight_14_coulomb = weight_14_coulomb_new
                    weight_14_coulomb_set = true
                end
                if haskey(entry, "lj14scale")
                    weight_14_lj_new = parse(T, entry["lj14scale"])
                    if weight_14_lj_set && weight_14_lj_new != weight_14_lj
                        error("found multiple NonbondedForce entries with different lj14scale values")
                    end
                    weight_14_lj = weight_14_lj_new
                    weight_14_lj_set = true
                end
                for atom_or_attr in eachelement(entry)
                    if atom_or_attr.name == "Atom"
                        if !haskey(atom_or_attr, "type")
                            @warn "Atom classes not currently supported, this $entry_name entry will be ignored"
                            continue
                        end
                        # Update previous atom types
                        atom_type = atom_or_attr["type"]
                        if !haskey(atom_types, atom_type)
                            # Skip types not defined above
                            continue
                        end
                        partial_type = atom_types[atom_type]
                        ch = haskey(atom_or_attr, "charge") ? parse(T, atom_or_attr["charge"]) : missing
                        σ = units ? parse(T, atom_or_attr["sigma"])u"nm" : parse(T, atom_or_attr["sigma"])
                        ϵ = units ? parse(T, atom_or_attr["epsilon"])u"kJ * mol^-1" : parse(T, atom_or_attr["epsilon"])
                        complete_type = AtomType{T, typeof(partial_type.mass), typeof(σ), typeof(ϵ)}(
                                    partial_type.type, partial_type.class, partial_type.element,
                                    ch, partial_type.mass, σ, ϵ)
                        atom_types[atom_type] = complete_type
                    elseif atom_or_attr.name == "UseAttributeFromResidue"
                        if !(atom_or_attr["name"] in attributes_from_residue)
                            push!(attributes_from_residue, atom_or_attr["name"])
                        end
                        if atom_or_attr["name"] != "charge"
                            @warn "UseAttributeFromResidue only currently supported for charge, " *
                                  "this entry will be ignored"
                        end
                    end
                end
            elseif entry_name == "Patches"
                @warn "Residue patches not currently supported, this entry will be ignored"
            elseif entry_name == "Include"
                @warn "File includes not currently supported, this entry will be ignored"
            elseif entry_name in ("RBTorsionForce", "CMAPTorsionForce", "GBSAOBCForce",
                                  "CustomBondForce", "CustomAngleForce", "CustomTorsionForce",
                                  "CustomNonbondedForce", "CustomGBForce", "CustomHbondForce",
                                  "CustomManyParticleForce", "LennardJonesForce")
                @warn "$entry_name entries not currently supported, this entry will be ignored"
            end
        end
    end

    if units
        M = typeof(T(1u"g/mol"))
        D = typeof(T(1u"nm"))
        E = typeof(T(1u"kJ * mol^-1"))
        K = typeof(T(1u"kJ * mol^-1 * nm^-2"))
    else
        M, D, E, K = T, T, T, T
    end
    return MolecularForceField{T, M, D, E, K}(atom_types, residue_types, bond_types, angle_types,
            torsion_types, torsion_order, weight_14_coulomb, weight_14_lj, attributes_from_residue)
end

function MolecularForceField(ff_files::AbstractString...; kwargs...)
    return MolecularForceField(DefaultFloat, ff_files...; kwargs...)
end

function Base.show(io::IO, ff::MolecularForceField)
    print(io, "MolecularForceField with ", length(ff.atom_types), " atom types, ",
            length(ff.residue_types), " residue types, ", length(ff.bond_types), " bond types, ",
            length(ff.angle_types), " angle types and ", length(ff.torsion_types), " torsion types")
end

get_res_id(res) = (Chemfiles.id(res), Chemfiles.property(res, "chainid"))

# Return the residue name with N or C added for terminal residues
function residue_name(res, res_id_to_standard::Dict, rename_terminal_res::Bool=true)
    res_id = get_res_id(res)
    res_name = Chemfiles.name(res)
    if rename_terminal_res && res_id_to_standard[res_id]
        prev_res_id = (res_id[1] - 1, res_id[2])
        next_res_id = (res_id[1] + 1, res_id[2])
        if !haskey(res_id_to_standard, prev_res_id) || !res_id_to_standard[prev_res_id]
            res_name = "N" * res_name
        elseif !haskey(res_id_to_standard, next_res_id) || !res_id_to_standard[next_res_id]
            res_name = "C" * res_name
        end
    end
    return res_name
end

atom_types_to_string(atom_types...) = join(map(at -> at == "" ? "-" : at, atom_types), "/")

atom_types_to_tuple(atom_types) = tuple(map(at -> at == "-" ? "" : at, split(atom_types, "/"))...)

const standard_res_names = [keys(BioStructures.threeletter_to_aa)..., "HID", "HIE", "HIP"]

"""
    System(coordinate_file, force_field; <keyword arguments>)

Read a coordinate file in a file format readable by Chemfiles and apply a
force field to it.

Atom names should exactly match residue templates - no searching of residue
templates is carried out.

    System(coordinate_file, topology_file; <keyword arguments>)
    System(T, coordinate_file, topology_file; <keyword arguments>)

Read a Gromacs coordinate file and a Gromacs topology file with all
includes collapsed into one file.

Gromacs file reading should be considered experimental.
The `implicit_solvent`, `kappa` and `rename_terminal_res` keyword arguments
are not available when reading Gromacs files.

# Arguments
- `boundary=nothing`: the bounding box used for simulation, read from the
    file by default.
- `velocities=nothing`: the velocities of the atoms in the system, set to
    zero by default.
- `loggers=()`: the loggers that record properties of interest during a
    simulation.
- `units::Bool=true`: whether to use Unitful quantities.
- `gpu::Bool=false`: whether to move the relevant parts of the system onto
    the GPU.
- `dist_cutoff=1.0u"nm"`: cutoff distance for long-range interactions.
- `dist_neighbors=1.2u"nm"`: cutoff distance for the neighbor list, should be
    greater than `dist_cutoff`.
- `center_coords::Bool=true`: whether to center the coordinates in the
    simulation box.
- `use_cell_list::Bool=true`: whether to use [`CellListMapNeighborFinder`](@ref)
    on CPU. If `false`, [`DistanceNeighborFinder`](@ref) is used.
- `data=nothing`: arbitrary data associated with the system.
- `implicit_solvent=nothing`: specify a string to add an implicit solvent
    model, options are "obc1", "obc2" and "gbn2".
- `kappa=0.0u"nm^-1"`: the kappa value for the implicit solvent model if one
    is used.
- `rename_terminal_res=true`: whether to rename the first and last residues
    to match the appropriate atom templates, for example the first (N-terminal)
    residue could be changed from "MET" to "NMET".
"""
function System(coord_file::AbstractString,
                force_field::MolecularForceField;
                boundary=nothing,
                velocities=nothing,
                loggers=(),
                units::Bool=true,
                gpu::Bool=false,
                dist_cutoff=units ? 1.0u"nm" : 1.0,
                dist_neighbors=units ? 1.2u"nm" : 1.2,
                center_coords::Bool=true,
                use_cell_list::Bool=true,
                data=nothing,
                implicit_solvent=nothing,
                kappa=0.0u"nm^-1",
                rename_terminal_res::Bool=true)
    T = typeof(force_field.weight_14_coulomb)

    # Chemfiles uses zero-based indexing, be careful
    trajectory = Chemfiles.Trajectory(coord_file)
    frame = Chemfiles.read(trajectory)
    top = Chemfiles.Topology(frame)
    n_atoms = size(top)

    atoms = Atom[]
    atoms_data = AtomData[]
    bonds = InteractionList2Atoms(HarmonicBond)
    angles = InteractionList3Atoms(HarmonicAngle)
    torsions = InteractionList4Atoms(PeriodicTorsion)
    impropers = InteractionList4Atoms(PeriodicTorsion)
    eligible = trues(n_atoms, n_atoms)
    special = falses(n_atoms, n_atoms)
    torsion_n_terms = 6

    top_bonds     = Vector{Int}[is for is in eachcol(Int.(Chemfiles.bonds(    top)))]
    top_angles    = Vector{Int}[is for is in eachcol(Int.(Chemfiles.angles(   top)))]
    top_torsions  = Vector{Int}[is for is in eachcol(Int.(Chemfiles.dihedrals(top)))]
    top_impropers = Vector{Int}[is for is in eachcol(Int.(Chemfiles.impropers(top)))]

    res_id_to_standard = Dict{Tuple{Int, String}, Bool}()
    for ri in 1:Chemfiles.count_residues(top)
        res = Chemfiles.Residue(top, ri - 1)
        res_id = get_res_id(res)
        res_name = Chemfiles.name(res)
        standard_res = res_name in standard_res_names
        res_id_to_standard[res_id] = standard_res

        if standard_res && residue_name(res, res_id_to_standard, rename_terminal_res) == "N" * res_name
            # Add missing N-terminal amide bonds, angles and torsions
            # See https://github.com/chemfiles/chemfiles/issues/429
            atom_inds_zero = Int.(Chemfiles.atoms(res))
            atom_names = Chemfiles.name.(Chemfiles.Atom.((top,), atom_inds_zero))
            nterm_atom_names = ("N", "H1", "H2", "H3", "CA", "CB", "HA", "HA2", "HA3", "C")
            ai_N, ai_H1, ai_H2, ai_H3, ai_CA, ai_CB, ai_HA, ai_HA2, ai_HA3, ai_C = [findfirst(isequal(an), atom_names) for an in nterm_atom_names]
            if !isnothing(ai_H1)
                push!(top_bonds, [atom_inds_zero[ai_N], atom_inds_zero[ai_H1]])
                push!(top_angles, [atom_inds_zero[ai_H1], atom_inds_zero[ai_N], atom_inds_zero[ai_CA]])
                push!(top_angles, [atom_inds_zero[ai_H1], atom_inds_zero[ai_N], atom_inds_zero[ai_H2]])
                push!(top_torsions, [atom_inds_zero[ai_H1], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_C]])
                if !isnothing(ai_CB)
                    push!(top_torsions, [atom_inds_zero[ai_H1], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_CB]])
                    push!(top_torsions, [atom_inds_zero[ai_H1], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_HA]])
                else
                    push!(top_torsions, [atom_inds_zero[ai_H1], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_HA2]])
                    push!(top_torsions, [atom_inds_zero[ai_H1], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_HA3]])
                end
            end
            if !isnothing(ai_H3)
                push!(top_bonds, [atom_inds_zero[ai_N], atom_inds_zero[ai_H3]])
                push!(top_angles, [atom_inds_zero[ai_H3], atom_inds_zero[ai_N], atom_inds_zero[ai_CA]])
                push!(top_angles, [atom_inds_zero[ai_H3], atom_inds_zero[ai_N], atom_inds_zero[ai_H2]])
                push!(top_torsions, [atom_inds_zero[ai_H3], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_C]])
                if !isnothing(ai_CB)
                    push!(top_torsions, [atom_inds_zero[ai_H3], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_CB]])
                    push!(top_torsions, [atom_inds_zero[ai_H3], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_HA]])
                else
                    push!(top_torsions, [atom_inds_zero[ai_H3], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_HA2]])
                    push!(top_torsions, [atom_inds_zero[ai_H3], atom_inds_zero[ai_N], atom_inds_zero[ai_CA], atom_inds_zero[ai_HA3]])
                end
            end
            if !isnothing(ai_H1) && !isnothing(ai_H3)
                push!(top_angles, [atom_inds_zero[ai_H1], atom_inds_zero[ai_N], atom_inds_zero[ai_H3]])
            end
        elseif res_name == "HOH"
            # Add missing water bonds and angles
            atom_inds_zero = Int.(Chemfiles.atoms(res))
            atom_names = Chemfiles.name.(Chemfiles.Atom.((top,), atom_inds_zero))
            ai_O, ai_H1, ai_H2 = [findfirst(isequal(an), atom_names) for an in ("O", "H1", "H2")]
            push!(top_bonds, [atom_inds_zero[ai_O], atom_inds_zero[ai_H1]])
            push!(top_bonds, [atom_inds_zero[ai_O], atom_inds_zero[ai_H2]])
            push!(top_angles, [atom_inds_zero[ai_H1], atom_inds_zero[ai_O], atom_inds_zero[ai_H2]])
        end
    end

    use_charge_from_residue = "charge" in force_field.attributes_from_residue
    for ai in 1:n_atoms
        atom_name = Chemfiles.name(Chemfiles.Atom(top, ai - 1))
        res = Chemfiles.residue_for_atom(top, ai - 1)
        res_id = get_res_id(res)
        res_name = residue_name(res, res_id_to_standard, rename_terminal_res)
        at_type = force_field.residue_types[res_name].types[atom_name]
        at = force_field.atom_types[at_type]
        if use_charge_from_residue
            ch = force_field.residue_types[res_name].charges[atom_name]
        else
            ch = at.charge
        end
        if ismissing(ch)
            error("atom of type ", at.type, " has not had charge set")
        end
        solute = res_id_to_standard[res_id] || res_name in ("ACE", "NME")
        if (units && at.σ < zero(T)u"nm") || (!units && at.σ < zero(T))
            error("atom of type ", at.type, " has not had σ or ϵ set")
        end
        push!(atoms, Atom(index=ai, charge=ch, mass=at.mass, σ=at.σ, ϵ=at.ϵ, solute=solute))
        push!(atoms_data, AtomData(atom_type=at_type, atom_name=atom_name, res_number=Chemfiles.id(res),
                                    res_name=Chemfiles.name(res), element=at.element))
        eligible[ai, ai] = false
    end

    for (a1z, a2z) in top_bonds
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = Chemfiles.name(Chemfiles.Atom(top, a2z))
        res_name_1 = residue_name(Chemfiles.residue_for_atom(top, a1z), res_id_to_standard, rename_terminal_res)
        res_name_2 = residue_name(Chemfiles.residue_for_atom(top, a2z), res_id_to_standard, rename_terminal_res)
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        push!(bonds.is, a1z + 1)
        push!(bonds.js, a2z + 1)
        if haskey(force_field.bond_types, (atom_type_1, atom_type_2))
            bond_type = force_field.bond_types[(atom_type_1, atom_type_2)]
            push!(bonds.types, atom_types_to_string(atom_type_1, atom_type_2))
        else
            bond_type = force_field.bond_types[(atom_type_2, atom_type_1)]
            push!(bonds.types, atom_types_to_string(atom_type_2, atom_type_1))
        end
        push!(bonds.inters, HarmonicBond(k=bond_type.k, r0=bond_type.r0))
        eligible[a1z + 1, a2z + 1] = false
        eligible[a2z + 1, a1z + 1] = false
    end

    for (a1z, a2z, a3z) in top_angles
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = Chemfiles.name(Chemfiles.Atom(top, a2z))
        atom_name_3 = Chemfiles.name(Chemfiles.Atom(top, a3z))
        res_name_1 = residue_name(Chemfiles.residue_for_atom(top, a1z), res_id_to_standard, rename_terminal_res)
        res_name_2 = residue_name(Chemfiles.residue_for_atom(top, a2z), res_id_to_standard, rename_terminal_res)
        res_name_3 = residue_name(Chemfiles.residue_for_atom(top, a3z), res_id_to_standard, rename_terminal_res)
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        atom_type_3 = force_field.residue_types[res_name_3].types[atom_name_3]
        push!(angles.is, a1z + 1)
        push!(angles.js, a2z + 1)
        push!(angles.ks, a3z + 1)
        if haskey(force_field.angle_types, (atom_type_1, atom_type_2, atom_type_3))
            angle_type = force_field.angle_types[(atom_type_1, atom_type_2, atom_type_3)]
            push!(angles.types, atom_types_to_string(atom_type_1, atom_type_2, atom_type_3))
        else
            angle_type = force_field.angle_types[(atom_type_3, atom_type_2, atom_type_1)]
            push!(angles.types, atom_types_to_string(atom_type_3, atom_type_2, atom_type_1))
        end
        push!(angles.inters, HarmonicAngle(k=angle_type.k, θ0=angle_type.θ0))
        eligible[a1z + 1, a3z + 1] = false
        eligible[a3z + 1, a1z + 1] = false
    end

    for (a1z, a2z, a3z, a4z) in top_torsions
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = Chemfiles.name(Chemfiles.Atom(top, a2z))
        atom_name_3 = Chemfiles.name(Chemfiles.Atom(top, a3z))
        atom_name_4 = Chemfiles.name(Chemfiles.Atom(top, a4z))
        res_name_1 = residue_name(Chemfiles.residue_for_atom(top, a1z), res_id_to_standard, rename_terminal_res)
        res_name_2 = residue_name(Chemfiles.residue_for_atom(top, a2z), res_id_to_standard, rename_terminal_res)
        res_name_3 = residue_name(Chemfiles.residue_for_atom(top, a3z), res_id_to_standard, rename_terminal_res)
        res_name_4 = residue_name(Chemfiles.residue_for_atom(top, a4z), res_id_to_standard, rename_terminal_res)
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        atom_type_3 = force_field.residue_types[res_name_3].types[atom_name_3]
        atom_type_4 = force_field.residue_types[res_name_4].types[atom_name_4]
        atom_types = (atom_type_1, atom_type_2, atom_type_3, atom_type_4)
        if haskey(force_field.torsion_types, atom_types) && force_field.torsion_types[atom_types].proper
            torsion_type = force_field.torsion_types[atom_types]
            best_key = atom_types
        elseif haskey(force_field.torsion_types, reverse(atom_types)) && force_field.torsion_types[reverse(atom_types)].proper
            torsion_type = force_field.torsion_types[reverse(atom_types)]
            best_key = reverse(atom_types)
        else
            # Search wildcard entries
            best_score = -1
            best_key = ("", "", "", "")
            for k in keys(force_field.torsion_types)
                if force_field.torsion_types[k].proper
                    for ke in (k, reverse(k))
                        valid = true
                        score = 0
                        for (i, v) in enumerate(ke)
                            if v == atom_types[i]
                                score += 1
                            elseif v != ""
                                valid = false
                                break
                            end
                        end
                        if valid && (score >= best_score)
                            best_score = score
                            best_key = k
                        end
                    end
                end
            end
            torsion_type = force_field.torsion_types[best_key]
        end
        n_terms = length(torsion_type.periodicities)
        for start_i in 1:torsion_n_terms:n_terms
            push!(torsions.is, a1z + 1)
            push!(torsions.js, a2z + 1)
            push!(torsions.ks, a3z + 1)
            push!(torsions.ls, a4z + 1)
            push!(torsions.types, atom_types_to_string(best_key...))
            end_i = min(start_i + torsion_n_terms - 1, n_terms)
            push!(torsions.inters, PeriodicTorsion(
                        periodicities=torsion_type.periodicities[start_i:end_i],
                        phases=torsion_type.phases[start_i:end_i],
                        ks=torsion_type.ks[start_i:end_i],
                        proper=true,
            ))
        end
        special[a1z + 1, a4z + 1] = true
        special[a4z + 1, a1z + 1] = true
    end

    # Note the order here - Chemfiles puts the central atom second
    for (a2z, a1z, a3z, a4z) in top_impropers
        inds_no1 = (a2z, a3z, a4z)
        atom_names = [Chemfiles.name(Chemfiles.Atom(top, a)) for a in inds_no1]
        res_names = [residue_name(Chemfiles.residue_for_atom(top, a), res_id_to_standard, rename_terminal_res) for a in inds_no1]
        atom_types = [force_field.residue_types[res_names[i]].types[atom_names[i]] for i in 1:3]
        # Amber sorts atoms alphabetically with hydrogen last
        if force_field.torsion_order == "amber"
            order = sortperm([t[1] == 'H' ? 'z' * t : t for t in atom_types])
        else
            order = [1, 2, 3]
        end
        a2z, a3z, a4z = [inds_no1[i] for i in order]
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = atom_names[order[1]]
        atom_name_3 = atom_names[order[2]]
        atom_name_4 = atom_names[order[3]]
        res_name_1 = residue_name(Chemfiles.residue_for_atom(top, a1z), res_id_to_standard, rename_terminal_res)
        res_name_2 = res_names[order[1]]
        res_name_3 = res_names[order[2]]
        res_name_4 = res_names[order[3]]
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        atom_type_3 = force_field.residue_types[res_name_3].types[atom_name_3]
        atom_type_4 = force_field.residue_types[res_name_4].types[atom_name_4]
        atom_types_no1 = (atom_type_2, atom_type_3, atom_type_4)
        best_score = -1
        best_key = ("", "", "", "")
        best_key_perm = ("", "", "", "")
        for k in keys(force_field.torsion_types)
            if !force_field.torsion_types[k].proper && (k[1] == atom_type_1 || k[1] == "")
                for ke2 in permutations(k[2:end])
                    valid = true
                    score = k[1] == atom_type_1 ? 1 : 0
                    for (i, v) in enumerate(ke2)
                        if v == atom_types_no1[i]
                            score += 1
                        elseif v != ""
                            valid = false
                            break
                        end
                    end
                    if valid && (score == 4 || best_score == -1)
                        best_score = score
                        best_key = k
                        best_key_perm = (k[1], ke2[1], ke2[2], ke2[3])
                    end
                end
            end
        end
        # Not all possible impropers are defined
        if best_score != -1
            torsion_type = force_field.torsion_types[best_key]
            a1, a2, a3, a4 = a1z + 1, a2z + 1, a3z + 1, a4z + 1
            # Follow Amber assignment rules from OpenMM
            if force_field.torsion_order == "amber"
                r2 = Chemfiles.id(Chemfiles.residue_for_atom(top, a2z))
                r3 = Chemfiles.id(Chemfiles.residue_for_atom(top, a3z))
                r4 = Chemfiles.id(Chemfiles.residue_for_atom(top, a4z))
                ta2 = force_field.residue_types[res_name_2].indices[atom_name_2]
                ta3 = force_field.residue_types[res_name_3].indices[atom_name_3]
                ta4 = force_field.residue_types[res_name_4].indices[atom_name_4]
                e2 = force_field.atom_types[atom_type_2].element
                e3 = force_field.atom_types[atom_type_3].element
                e4 = force_field.atom_types[atom_type_4].element
                t2, t3, t4 = atom_type_2, atom_type_3, atom_type_4
                if !("" in best_key_perm)
                    if t2 == t4 && (r2 > r4 || (r2 == r4 && ta2 > ta4))
                        a2, a4 = a4, a2
                        r2, r4 = r4, r2
                        ta2, ta4 = ta4, ta2
                    end
                    if t3 == t4 && (r3 > r4 || (r3 == r4 && ta3 > ta4))
                        a3, a4 = a4, a3
                        r3, r4 = r4, r3
                        ta3, ta4 = ta4, ta3
                    end
                    if t2 == t3 && (r2 > r3 || (r2 == r3 && ta2 > ta3))
                        a2, a3 = a3, a2
                    end
                else
                    if e2 == e4 && (r2 > r4 || (r2 == r4 && ta2 > ta4))
                        a2, a4 = a4, a2
                        r2, r4 = r4, r2
                        ta2, ta4 = ta4, ta2
                    end
                    if e3 == e4 && (r3 > r4 || (r3 == r4 && ta3 > ta4))
                        a3, a4 = a4, a3
                        r3, r4 = r4, r3
                        ta3, ta4 = ta4, ta3
                    end
                    if r2 > r3 || (r2 == r3 && ta2 > ta3)
                        a2, a3 = a3, a2
                    end
                end
            end
            push!(impropers.is, a2)
            push!(impropers.js, a3)
            push!(impropers.ks, a1)
            push!(impropers.ls, a4)
            push!(impropers.types, atom_types_to_string(best_key...))
            push!(impropers.inters, PeriodicTorsion(periodicities=torsion_type.periodicities,
                    phases=torsion_type.phases, ks=torsion_type.ks, proper=false))
        end
    end

    if units
        force_units = u"kJ * mol^-1 * nm^-1"
        energy_units = u"kJ * mol^-1"
    else
        force_units = NoUnits
        energy_units = NoUnits
    end

    lj = LennardJones(
        cutoff=DistanceCutoff(T(dist_cutoff)),
        use_neighbors=true,
        weight_special=force_field.weight_14_lj,
    )
    if isnothing(implicit_solvent)
        crf = CoulombReactionField(
            dist_cutoff=T(dist_cutoff),
            solvent_dielectric=T(crf_solvent_dielectric),
            use_neighbors=true,
            weight_special=force_field.weight_14_coulomb,
            coulomb_const=(units ? T(coulombconst) : T(ustrip(coulombconst))),
        )
    else
        crf = Coulomb(
            cutoff=DistanceCutoff(T(dist_cutoff)),
            use_neighbors=true,
            weight_special=force_field.weight_14_coulomb,
            coulomb_const=(units ? T(coulombconst) : T(ustrip(coulombconst))),
        )
    end
    pairwise_inters = (lj, crf)

    # All torsions should have the same number of terms for speed, GPU compatibility
    #   and for taking gradients
    # For now always pad to 6 terms
    torsion_inters_pad = [PeriodicTorsion(periodicities=t.periodicities, phases=t.phases, ks=t.ks,
                                            proper=t.proper, n_terms=torsion_n_terms) for t in torsions.inters]
    improper_inters_pad = [PeriodicTorsion(periodicities=t.periodicities, phases=t.phases, ks=t.ks,
                                            proper=t.proper, n_terms=torsion_n_terms) for t in impropers.inters]

    # Only add present interactions and ensure that array types are concrete
    specific_inter_array = []
    if length(bonds.is) > 0
        push!(specific_inter_array, InteractionList2Atoms(
            gpu ? CuArray(bonds.is) : bonds.is,
            gpu ? CuArray(bonds.js) : bonds.js,
            gpu ? CuArray([bonds.inters...]) : [bonds.inters...],
            bonds.types,
        ))
        topology = MolecularTopology(bonds.is, bonds.js, n_atoms)
    else
        topology = nothing
    end
    if length(angles.is) > 0
        push!(specific_inter_array, InteractionList3Atoms(
            gpu ? CuArray(angles.is) : angles.is,
            gpu ? CuArray(angles.js) : angles.js,
            gpu ? CuArray(angles.ks) : angles.ks,
            gpu ? CuArray([angles.inters...]) : [angles.inters...],
            angles.types,
        ))
    end
    if length(torsions.is) > 0
        push!(specific_inter_array, InteractionList4Atoms(
            gpu ? CuArray(torsions.is) : torsions.is,
            gpu ? CuArray(torsions.js) : torsions.js,
            gpu ? CuArray(torsions.ks) : torsions.ks,
            gpu ? CuArray(torsions.ls) : torsions.ls,
            gpu ? CuArray(torsion_inters_pad) : torsion_inters_pad,
            torsions.types,
        ))
    end
    if length(impropers.is) > 0
        push!(specific_inter_array, InteractionList4Atoms(
            gpu ? CuArray(impropers.is) : impropers.is,
            gpu ? CuArray(impropers.js) : impropers.js,
            gpu ? CuArray(impropers.ks) : impropers.ks,
            gpu ? CuArray(impropers.ls) : impropers.ls,
            gpu ? CuArray(improper_inters_pad) : improper_inters_pad,
            impropers.types,
        ))
    end
    specific_inter_lists = tuple(specific_inter_array...)

    if isnothing(boundary)
        # Read from file and convert from Å
        if units
            box_size = SVector{3}(T.(Chemfiles.lengths(Chemfiles.UnitCell(frame))u"nm" / 10.0))
        else
            box_size = SVector{3}(T.(Chemfiles.lengths(Chemfiles.UnitCell(frame)) / 10.0))
        end
        boundary_used = CubicBoundary(box_size)
    else
        boundary_used = boundary
    end

    # Convert from Å
    if units
        coords = [T.(SVector{3}(col)u"nm" / 10.0) for col in eachcol(Chemfiles.positions(frame))]
    else
        coords = [T.(SVector{3}(col) / 10.0) for col in eachcol(Chemfiles.positions(frame))]
    end
    if center_coords
        coords = coords .- (mean(coords),) .+ (box_center(boundary_used),)
    end
    coords = wrap_coords.(coords, (boundary_used,))

    atoms = [atoms...]
    if gpu || !use_cell_list
        neighbor_finder = DistanceNeighborFinder(
            eligible=(gpu ? CuArray(eligible) : eligible),
            special=(gpu ? CuArray(special) : special),
            n_steps=10,
            dist_cutoff=T(dist_neighbors),
        )
    else
        neighbor_finder = CellListMapNeighborFinder(
            eligible=eligible,
            special=special,
            n_steps=10,
            x0=coords,
            unit_cell=boundary_used,
            dist_cutoff=T(dist_neighbors),
        )
    end
    if gpu
        atoms = CuArray(atoms)
        coords = CuArray(coords)
    end

    if isnothing(velocities)
        if units
            vels = zero(ustrip_vec.(coords))u"nm * ps^-1"
        else
            vels = zero(coords)
        end
    else
        vels = velocities
    end

    if !isnothing(implicit_solvent)
        if implicit_solvent == "obc1"
            general_inters = (ImplicitSolventOBC(atoms, atoms_data, bonds;
                                kappa=kappa, use_OBC2=false),)
        elseif implicit_solvent == "obc2"
            general_inters = (ImplicitSolventOBC(atoms, atoms_data, bonds;
                                kappa=kappa, use_OBC2=true),)
        elseif implicit_solvent == "gbn2"
            general_inters = (ImplicitSolventGBN2(atoms, atoms_data, bonds; kappa=kappa),)
        else
            error("unknown implicit solvent model: \"$implicit_solvent\"")
        end
    else
        general_inters = ()
    end

    k = units ? Unitful.Na * Unitful.k : ustrip(u"kJ * K^-1 * mol^-1", Unitful.Na * Unitful.k)
    return System(
        atoms=atoms,
        coords=coords,
        boundary=boundary_used,
        velocities=vels,
        atoms_data=atoms_data,
        topology=topology,
        pairwise_inters=pairwise_inters,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        neighbor_finder=neighbor_finder,
        loggers=loggers,
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
        k=k,
        data=data,
    )
end

function System(T::Type,
                coord_file::AbstractString,
                top_file::AbstractString;
                boundary=nothing,
                velocities=nothing,
                loggers=(),
                units::Bool=true,
                gpu::Bool=false,
                dist_cutoff=units ? 1.0u"nm" : 1.0,
                dist_neighbors=units ? 1.2u"nm" : 1.2,
                center_coords::Bool=true,
                use_cell_list::Bool=true,
                data=nothing)
    # Read force field and topology file
    atomtypes = Dict{String, Atom}()
    bondtypes = Dict{String, HarmonicBond}()
    angletypes = Dict{String, HarmonicAngle}()
    torsiontypes = Dict{String, RBTorsion}()
    atomnames = Dict{String, String}()

    name = "?"
    atoms = Atom[]
    atoms_data = AtomData[]
    bonds = InteractionList2Atoms(HarmonicBond)
    pairs = Tuple{Int, Int}[]
    angles = InteractionList3Atoms(HarmonicAngle)
    possible_torsions = Tuple{Int, Int, Int, Int}[]
    torsions = InteractionList4Atoms(RBTorsion)

    if units
        force_units = u"kJ * mol^-1 * nm^-1"
        energy_units = u"kJ * mol^-1"
    else
        force_units = NoUnits
        energy_units = NoUnits
    end

    current_field = ""
    for l in eachline(top_file)
        sl = strip(l)
        if length(sl) == 0 || startswith(sl, ';')
            continue
        end
        if startswith(sl, '[') && endswith(sl, ']')
            current_field = strip(sl[2:end-1])
            continue
        end
        c = split(rstrip(first(split(sl, ";", limit=2))), r"\s+")
        if current_field == "bondtypes"
            if units
                bondtype = HarmonicBond(parse(T, c[5])u"kJ * mol^-1 * nm^-2", parse(T, c[4])u"nm")
            else
                bondtype = HarmonicBond(parse(T, c[5]), parse(T, c[4]))
            end
            bondtypes["$(c[1])/$(c[2])"] = bondtype
            bondtypes["$(c[2])/$(c[1])"] = bondtype
        elseif current_field == "angletypes"
            # Convert θ0 to radians
            if units
                angletype = HarmonicAngle(parse(T, c[6])u"kJ * mol^-1", deg2rad(parse(T, c[5])))
            else
                angletype = HarmonicAngle(parse(T, c[6]), deg2rad(parse(T, c[5])))
            end
            angletypes["$(c[1])/$(c[2])/$(c[3])"] = angletype
            angletypes["$(c[3])/$(c[2])/$(c[1])"] = angletype
        elseif current_field == "dihedraltypes" && c[1] != "#define"
            # Convert back to OPLS types
            f4 = parse(T, c[10]) / -4
            f3 = parse(T, c[9]) / -2
            f2 = 4 * f4 - parse(T, c[8])
            f1 = 3 * f3 - 2 * parse(T, c[7])
            if units
                torsiontype = RBTorsion((f1)u"kJ * mol^-1", (f2)u"kJ * mol^-1",
                                        (f3)u"kJ * mol^-1", (f4)u"kJ * mol^-1")
            else
                torsiontype = RBTorsion(f1, f2, f3, f4)
            end
            torsiontypes["$(c[1])/$(c[2])/$(c[3])/$(c[4])"] = torsiontype
        elseif current_field == "atomtypes" && length(c) >= 8
            atomname = uppercase(c[2])
            atomnames[c[1]] = atomname
            # Take the first version of each atom type only
            if !haskey(atomtypes, atomname)
                if units
                    atomtypes[atomname] = Atom(charge=parse(T, c[5]), mass=parse(T, c[4])u"g/mol",
                            σ=parse(T, c[7])u"nm", ϵ=parse(T, c[8])u"kJ * mol^-1")
                else
                    atomtypes[atomname] = Atom(charge=parse(T, c[5]), mass=parse(T, c[4]),
                            σ=parse(T, c[7]), ϵ=parse(T, c[8]))
                end
            end
        elseif current_field == "atoms"
            attype = atomnames[c[2]]
            ch = parse(T, c[7])
            if units
                atom_mass = parse(T, c[8])u"g/mol"
            else
                atom_mass = parse(T, c[8])
            end
            solute = c[4] in standard_res_names
            atom_index = length(atoms) + 1
            push!(atoms, Atom(index=atom_index, charge=ch, mass=atom_mass, σ=atomtypes[attype].σ,
                                ϵ=atomtypes[attype].ϵ, solute=solute))
            push!(atoms_data, AtomData(atom_type=attype, atom_name=c[5], res_number=parse(Int, c[3]),
                                        res_name=c[4]))
        elseif current_field == "bonds"
            i, j = parse.(Int, c[1:2])
            bn = "$(atoms_data[i].atom_type)/$(atoms_data[j].atom_type)"
            bondtype = bondtypes[bn]
            push!(bonds.is, i)
            push!(bonds.js, j)
            push!(bonds.types, bn)
            push!(bonds.inters, HarmonicBond(k=bondtype.k, r0=bondtype.r0))
        elseif current_field == "pairs"
            push!(pairs, (parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "angles"
            i, j, k = parse.(Int, c[1:3])
            an = "$(atoms_data[i].atom_type)/$(atoms_data[j].atom_type)/$(atoms_data[k].atom_type)"
            angletype = angletypes[an]
            push!(angles.is, i)
            push!(angles.js, j)
            push!(angles.ks, k)
            push!(angles.types, an)
            push!(angles.inters, HarmonicAngle(k=angletype.k, θ0=angletype.θ0))
        elseif current_field == "dihedrals"
            i, j, k, l = parse.(Int, c[1:4])
            push!(possible_torsions, (i, j, k, l))
        elseif current_field == "system"
            name = rstrip(first(split(sl, ";", limit=2)))
        end
    end

    # Add torsions based on wildcard torsion types
    for inds in possible_torsions
        at_types = [atoms_data[x].atom_type for x in inds]
        desired_key = join(at_types, "/")
        if haskey(torsiontypes, desired_key)
            d = torsiontypes[desired_key]
            push!(torsions.is, inds[1])
            push!(torsions.js, inds[2])
            push!(torsions.ks, inds[3])
            push!(torsions.ls, inds[4])
            push!(torsions.types, desired_key)
            push!(torsions.inters, RBTorsion(f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4))
        else
            best_score = 0
            best_key = ""
            for k in keys(torsiontypes)
                c = split(k, "/")
                for a in (c, reverse(c))
                    valid = true
                    score = 0
                    for (i, v) in enumerate(a)
                        if v == at_types[i]
                            score += 1
                        elseif v != "X"
                            valid = false
                            break
                        end
                    end
                    if valid && (score > best_score)
                        best_score = score
                        best_key = k
                    end
                end
            end
            # If a wildcard match is found, add a new specific torsion type
            if best_key != ""
                d = torsiontypes[best_key]
                push!(torsions.is, inds[1])
                push!(torsions.js, inds[2])
                push!(torsions.ks, inds[3])
                push!(torsions.ls, inds[4])
                push!(torsions.types, best_key)
                push!(torsions.inters, RBTorsion(f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4))
            end
        end
    end

    # Read coordinate file and add solvent atoms
    lines = readlines(coord_file)
    coords = SArray[]
    for (i, l) in enumerate(lines[3:end-1])
        coord = SVector(parse(T, l[21:28]), parse(T, l[29:36]), parse(T, l[37:44]))
        if units
            push!(coords, (coord)u"nm")
        else
            push!(coords, coord)
        end

        # Some atoms are not specified explicitly in the topology so are added here
        if i > length(atoms)
            atname = strip(l[11:15])
            attype = replace(atname, r"\d+" => "")
            temp_charge = atomtypes[attype].charge
            if attype == "CL" # Temp hack to fix charges
                temp_charge = T(-1.0)
            end
            atom_index = length(atoms) + 1
            push!(atoms, Atom(index=atom_index, charge=temp_charge, mass=atomtypes[attype].mass,
                                σ=atomtypes[attype].σ, ϵ=atomtypes[attype].ϵ, solute=false))
            push!(atoms_data, AtomData(atom_type=attype, atom_name=atname, res_number=parse(Int, l[1:5]),
                                        res_name=strip(l[6:10])))

            # Add O-H bonds and H-O-H angle in water
            if atname == "OW"
                bondtype = bondtypes["OW/HW"]
                push!(bonds.is, i)
                push!(bonds.js, i + 1)
                push!(bonds.types, "OW/HW")
                push!(bonds.inters, HarmonicBond(k=bondtype.k, r0=bondtype.r0))
                push!(bonds.is, i)
                push!(bonds.js, i + 2)
                push!(bonds.types, "OW/HW")
                push!(bonds.inters, HarmonicBond(k=bondtype.k, r0=bondtype.r0))
                angletype = angletypes["HW/OW/HW"]
                push!(angles.is, i + 1)
                push!(angles.js, i)
                push!(angles.ks, i + 2)
                push!(angles.types, "HW/OW/HW")
                push!(angles.inters, HarmonicAngle(k=angletype.k, θ0=angletype.θ0))
            end
        end
    end

    # Calculate matrix of pairs eligible for non-bonded interactions
    n_atoms = length(coords)
    eligible = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    for (i, j) in zip(bonds.is, bonds.js)
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, k) in zip(angles.is, angles.ks)
        # Assume bonding is already specified
        eligible[i, k] = false
        eligible[k, i] = false
    end

    # Calculate matrix of pairs eligible for halved non-bonded interactions
    # This applies to specified pairs in the topology file, usually 1-4 bonded
    special = falses(n_atoms, n_atoms)
    for (i, j) in pairs
        special[i, j] = true
        special[j, i] = true
    end

    lj = LennardJones(
        cutoff=DistanceCutoff(T(dist_cutoff)),
        use_neighbors=true,
        weight_special=T(0.5),
    )
    crf = CoulombReactionField(
        dist_cutoff=T(dist_cutoff),
        solvent_dielectric=T(crf_solvent_dielectric),
        use_neighbors=true,
        weight_special=T(0.5),
        coulomb_const=(units ? T(coulombconst) : T(ustrip(coulombconst))),
    )

    if isnothing(boundary)
        box_size_vals = SVector{3}(parse.(T, split(strip(lines[end]), r"\s+")))
        box_size = units ? (box_size_vals)u"nm" : box_size_vals
        boundary_used = CubicBoundary(box_size)
    else
        boundary_used = boundary
    end
    coords = [coords...]
    if center_coords
        coords = coords .- (mean(coords),) .+ (box_center(boundary_used),)
    end
    coords = wrap_coords.(coords, (boundary_used,))

    pairwise_inters = (lj, crf)

    # Only add present interactions and ensure that array types are concrete
    specific_inter_array = []
    if length(bonds.is) > 0
        push!(specific_inter_array, InteractionList2Atoms(
            gpu ? CuArray(bonds.is) : bonds.is,
            gpu ? CuArray(bonds.js) : bonds.js,
            gpu ? CuArray([bonds.inters...]) : [bonds.inters...],
            bonds.types,
        ))
        topology = MolecularTopology(bonds.is, bonds.js, n_atoms)
    else
        topology = nothing
    end
    if length(angles.is) > 0
        push!(specific_inter_array, InteractionList3Atoms(
            gpu ? CuArray(angles.is) : angles.is,
            gpu ? CuArray(angles.js) : angles.js,
            gpu ? CuArray(angles.ks) : angles.ks,
            gpu ? CuArray([angles.inters...]) : [angles.inters...],
            angles.types,
        ))
    end
    if length(torsions.is) > 0
        push!(specific_inter_array, InteractionList4Atoms(
            gpu ? CuArray(torsions.is) : torsions.is,
            gpu ? CuArray(torsions.js) : torsions.js,
            gpu ? CuArray(torsions.ks) : torsions.ks,
            gpu ? CuArray(torsions.ls) : torsions.ls,
            gpu ? CuArray([torsions.inters...]) : [torsions.inters...],
            torsions.types,
        ))
    end
    specific_inter_lists = tuple(specific_inter_array...)

    atoms = [Atom(index=a.index, charge=a.charge, mass=a.mass, σ=a.σ, ϵ=a.ϵ, solute=a.solute) for a in atoms]

    if gpu || !use_cell_list
        neighbor_finder = DistanceNeighborFinder(
            eligible=(gpu ? CuArray(eligible) : eligible),
            special=(gpu ? CuArray(special) : special),
            n_steps=10,
            dist_cutoff=T(dist_neighbors),
        )
    else
        neighbor_finder = CellListMapNeighborFinder(
            eligible=eligible,
            special=special,
            n_steps=10,
            x0=coords,
            unit_cell=boundary_used,
            dist_cutoff=T(dist_neighbors),
        )
    end
    if gpu
        atoms = CuArray(atoms)
        coords = CuArray(coords)
    end

    if isnothing(velocities)
        if units
            vels = zero(ustrip_vec.(coords))u"nm * ps^-1"
        else
            vels = zero(coords)
        end
    else
        vels = velocities
    end

    k = units ? Unitful.Na * Unitful.k : ustrip(u"kJ * K^-1 * mol^-1", Unitful.Na * Unitful.k)
    return System(
        atoms=atoms,
        coords=coords,
        boundary=boundary_used,
        velocities=vels,
        atoms_data=atoms_data,
        topology=topology,
        pairwise_inters=pairwise_inters,
        specific_inter_lists=specific_inter_lists,
        neighbor_finder=neighbor_finder,
        loggers=loggers,
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
        k=k,
        data=data,
    )
end

function System(coord_file::AbstractString, top_file::AbstractString; kwargs...)
    return System(DefaultFloat, coord_file, top_file; kwargs...)
end

"""
    is_any_atom(at, at_data)

Placeholder function that returns `true`, used to select any [`Atom`](@ref).
"""
is_any_atom(at, at_data) = true

"""
    is_heavy_atom(at, at_data)

Determines whether an [`Atom`](@ref) is a heavy atom, i.e. any element other than hydrogen.
"""
function is_heavy_atom(at, at_data)
    if isnothing(at_data) || at_data.element in ("?", "")
        return mass(at) > 1.01u"g/mol"
    else
        return !(at_data.element in ("H", "D"))
    end
end

"""
    add_position_restraints(sys, k; atom_selector=is_any_atom, restrain_coords=sys.coords)

Return a copy of a [`System`](@ref) with [`HarmonicPositionRestraint`](@ref)s added to restrain the
atoms.

The force constant `k` can be a single value or an array of equal length to the number of atoms
in the system.
The `atom_selector` function takes in each atom and atom data and determines whether to restrain
that atom.
For example, [`is_heavy_atom`](@ref) means non-hydrogen atoms are restrained.
"""
function add_position_restraints(sys,
                                 k;
                                 atom_selector::Function=is_any_atom,
                                 restrain_coords=sys.coords)
    k_array = isa(k, AbstractArray) ? k : fill(k, length(sys))
    if length(k_array) != length(sys)
        throw(ArgumentError("the system has $(length(sys)) atoms but there are $(length(k_array)) k values"))
    end
    is = Int32[]
    types = String[]
    inters = HarmonicPositionRestraint[]
    atoms_data = length(sys.atoms_data) > 0 ? sys.atoms_data : fill(nothing, length(sys))
    for (i, (at, at_data, k_res, x0)) in enumerate(zip(Array(sys.atoms), atoms_data, k_array,
                                                       Array(restrain_coords)))
        if atom_selector(at, at_data)
            push!(is, i)
            push!(types, "")
            push!(inters, HarmonicPositionRestraint(k_res, x0))
        end
    end
    restraints = InteractionList1Atoms(move_array(is, sys), move_array([inters...], sys), types)
    sis = (sys.specific_inter_lists..., restraints)
    return System(
        atoms=deepcopy(sys.atoms),
        coords=deepcopy(sys.coords),
        boundary=deepcopy(sys.boundary),
        velocities=deepcopy(sys.velocities),
        atoms_data=deepcopy(sys.atoms_data),
        topology=deepcopy(sys.topology),
        pairwise_inters=deepcopy(sys.pairwise_inters),
        specific_inter_lists=sis,
        general_inters=deepcopy(sys.general_inters),
        constraints=deepcopy(sys.constraints),
        neighbor_finder=deepcopy(sys.neighbor_finder),
        loggers=deepcopy(sys.loggers),
        force_units=sys.force_units,
        energy_units=sys.energy_units,
        k=sys.k,
        data=sys.data,
    )
end
