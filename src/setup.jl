# Read files to set up a simulation
# See http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    Atomtype,
    Bondtype,
    Angletype,
    Torsiontype,
    placeatoms,
    placediatomics,
    readinputs

"""
    Atomtype(charge, mass, σ, ϵ)

Gromacs atom type.
"""
struct Atomtype{C, M, S, E}
    charge::C
    mass::M
    σ::S
    ϵ::E
end

"""
    Bondtype(b0, kb)

Gromacs bond type.
"""
struct Bondtype{D, K}
    b0::D
    kb::K
end

"""
    Angletype(th0, cth)

Gromacs angle type.
"""
struct Angletype{D, K}
    th0::D
    cth::K
end

"""
    Torsiontype(f1, f2, f3, f4)

Gromacs torsion type.
"""
struct Torsiontype{T}
    f1::T
    f2::T
    f3::T
    f4::T
end

"""
    placeatoms(n_atoms, box_size, min_dist; dims=3)

Obtain `n_atoms` 3D coordinates in a cube of length `box_size` where no two
points are closer than `min_dist`, accounting for periodic boundary conditions.
"""
function placeatoms(n_atoms::Integer, box_size, min_dist; dims::Integer=3)
    min_dist_sq = min_dist ^ 2
    T = typeof(convert(AbstractFloat, ustrip(box_size)))
    coords = SArray[]
    while length(coords) < n_atoms
        new_coord = SVector{dims}(rand(T, dims)) .* box_size
        okay = true
        for coord in coords
            if sum(abs2, vector(coord, new_coord, box_size)) < min_dist_sq
                okay = false
                break
            end
        end
        if okay
            push!(coords, new_coord)
        end
    end
    return [coords...]
end

"""
    placediatomics(n_molecules, box_size, min_dist, bond_length; dims=3)

Obtain 3D coordinates for `n_molecules` diatomics in a cube of length `box_size`
where no two points are closer than `min_dist` and the bond length is `bond_length`,
accounting for periodic boundary conditions.
"""
function placediatomics(n_molecules::Integer, box_size, min_dist, bond_length; dims::Integer=3)
    min_dist_sq = min_dist ^ 2
    T = typeof(convert(AbstractFloat, ustrip(box_size)))
    coords = SArray[]
    while length(coords) < (n_molecules * 2)
        new_coord_a = SVector{dims}(rand(T, dims)) .* box_size
        shift = SVector{dims}([bond_length, [zero(bond_length) for d in 1:(dims - 1)]...])
        new_coord_b = copy(new_coord_a) + shift
        okay = new_coord_b[1] <= box_size
        for coord in coords
            if sum(abs2, vector(coord, new_coord_a, box_size)) < min_dist_sq ||
                    sum(abs2, vector(coord, new_coord_b, box_size)) < min_dist_sq
                okay = false
                break
            end
        end
        if okay
            push!(coords, new_coord_a)
            push!(coords, new_coord_b)
        end
    end
    return [coords...]
end

"""
    readinputs(topology_file, coordinate_file; units=true)
    readinputs(T, topology_file, coordinate_file; units=true)

Read a Gromacs topology flat file, i.e. all includes collapsed into one file,
and a Gromacs coordinate file.
Returns the atoms, specific interaction lists, general interaction lists,
non-bonded matrix, coordinates and box size.
`units` determines whether the returned values have units.
"""
function readinputs(T::Type,
                    top_file::AbstractString,
                    coord_file::AbstractString;
                    units::Bool=true)
    # Read forcefield and topology file
    atomtypes = Dict{String, Atomtype}()
    bondtypes = Dict{String, Bondtype}()
    angletypes = Dict{String, Angletype}()
    torsiontypes = Dict{String, Torsiontype}()
    atomnames = Dict{String, String}()

    name = "?"
    atoms = Atom[]
    bonds = HarmonicBond[]
    pairs = Tuple{Int, Int}[]
    angles = HarmonicAngle[]
    possible_torsions = Tuple{Int, Int, Int, Int}[]
    torsions = Torsion[]

    if units
        force_unit = u"kJ * mol^-1 * nm^-1"
        energy_unit = u"kJ * mol^-1"
    else
        force_unit = NoUnits
        energy_unit = NoUnits
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
                bondtype = Bondtype(parse(T, c[4])u"nm", parse(T, c[5])u"kJ * mol^-1 * nm^-2")
            else
                bondtype = Bondtype(parse(T, c[4]), parse(T, c[5]))
            end
            bondtypes["$(c[1])/$(c[2])"] = bondtype
            bondtypes["$(c[2])/$(c[1])"] = bondtype
        elseif current_field == "angletypes"
            # Convert th0 to radians
            if units
                angletype = Angletype(deg2rad(parse(T, c[5])), parse(T, c[6])u"kJ * mol^-1")
            else
                angletype = Angletype(deg2rad(parse(T, c[5])), parse(T, c[6]))
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
                torsiontype = Torsiontype((f1)u"kJ * mol^-1", (f2)u"kJ * mol^-1",
                                            (f3)u"kJ * mol^-1", (f4)u"kJ * mol^-1")
            else
                torsiontype = Torsiontype(f1, f2, f3, f4)
            end
            torsiontypes["$(c[1])/$(c[2])/$(c[3])/$(c[4])"] = torsiontype
        elseif current_field == "atomtypes" && length(c) >= 8
            atomname = uppercase(c[2])
            atomnames[c[1]] = atomname
            # Take the first version of each atom type only
            if !haskey(atomtypes, atomname)
                if units
                    atomtypes[atomname] = Atomtype(parse(T, c[5]) * T(1u"q"),
                            parse(T, c[4])u"u", parse(T, c[7])u"nm", parse(T, c[8])u"kJ * mol^-1")
                else
                    atomtypes[atomname] = Atomtype(parse(T, c[5]), parse(T, c[4]),
                            parse(T, c[7]), parse(T, c[8]))
                end
            end
        elseif current_field == "atoms"
            attype = atomnames[c[2]]
            if units
                charge = parse(T, c[7]) * T(1u"q")
                mass = parse(T, c[8])u"u"
            else
                charge = parse(T, c[7])
                mass = parse(T, c[8])
            end
            push!(atoms, Atom(attype=attype, charge=charge, mass=mass,
                    σ=atomtypes[attype].σ, ϵ=atomtypes[attype].ϵ))
        elseif current_field == "bonds"
            i, j = parse.(Int, c[1:2])
            bondtype = bondtypes["$(atoms[i].attype)/$(atoms[j].attype)"]
            push!(bonds, HarmonicBond(i=i, j=j, b0=bondtype.b0, kb=bondtype.kb))
        elseif current_field == "pairs"
            push!(pairs, (parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "angles"
            i, j, k = parse.(Int, c[1:3])
            angletype = angletypes["$(atoms[i].attype)/$(atoms[j].attype)/$(atoms[k].attype)"]
            push!(angles, HarmonicAngle(i=i, j=j, k=k, th0=angletype.th0, cth=angletype.cth))
        elseif current_field == "dihedrals"
            i, j, k, l = parse.(Int, c[1:4])
            push!(possible_torsions, (i, j, k, l))
        elseif current_field == "system"
            name = rstrip(first(split(sl, ";", limit=2)))
        end
    end

    # Add torsions based on wildcard torsion types
    for inds in possible_torsions
        at_types = [atoms[x].attype for x in inds]
        desired_key = join(at_types, "/")
        if haskey(torsiontypes, desired_key)
            d = torsiontypes[desired_key]
            push!(torsions, Torsion(i=inds[1], j=inds[2], k=inds[3], l=inds[4],
                                    f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4))
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
                push!(torsions, Torsion(i=inds[1], j=inds[2], k=inds[3], l=inds[4],
                                        f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4))
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
                if units
                    temp_charge = T(-1u"q")
                else
                    temp_charge = T(-1.0)
                end
            end
            push!(atoms, Atom(attype=attype, charge=temp_charge, mass=atomtypes[attype].mass,
                    σ=atomtypes[attype].σ, ϵ=atomtypes[attype].ϵ))

            # Add O-H bonds and H-O-H angle in water
            if atname == "OW"
                bondtype = bondtypes["OW/HW"]
                push!(bonds, HarmonicBond(i=i, j=(i + 1), b0=bondtype.b0, kb=bondtype.kb))
                push!(bonds, HarmonicBond(i=i, j=(i + 2), b0=bondtype.b0, kb=bondtype.kb))
                angletype = angletypes["HW/OW/HW"]
                push!(angles, HarmonicAngle(i=(i + 1), j=i, k=(i + 2), th0=angletype.th0,
                                            cth=angletype.cth))
            end
        end
    end

    # Calculate matrix of pairs eligible for non-bonded interactions
    n_atoms = length(coords)
    nb_matrix = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        nb_matrix[i, i] = false
    end
    for b in bonds
        nb_matrix[b.i, b.j] = false
        nb_matrix[b.j, b.i] = false
    end
    for a in angles
        # Assume bonding is already specified
        nb_matrix[a.i, a.k] = false
        nb_matrix[a.k, a.i] = false
    end

    # Calculate matrix of pairs eligible for halved non-bonded interactions
    # This applies to specified pairs in the topology file, usually 1-4 bonded
    matrix_14 = falses(n_atoms, n_atoms)
    for (i, j) in pairs
        matrix_14[i, j] = true
        matrix_14[j, i] = true
    end

    lj = LennardJones(cutoff=ShiftedPotentialCutoff(T(1.2)u"nm"), nl_only=true, weight_14=T(0.5),
                        force_unit=force_unit, energy_unit=energy_unit)
    if units
        coulomb = Coulomb(cutoff=ShiftedPotentialCutoff(T(1.2)u"nm"), nl_only=true, weight_14=T(0.5),
                            force_unit=force_unit, energy_unit=energy_unit)
    else
        coulomb = Coulomb(cutoff=ShiftedPotentialCutoff(T(1.2)), nl_only=true, weight_14=T(0.5),
                            force_unit=force_unit, energy_unit=energy_unit)
    end

    # Bounding box for PBCs - box goes 0 to this value in 3 dimensions
    box_size_val = parse(T, first(split(strip(lines[end]), r"\s+")))
    box_size = units ? (box_size_val)u"nm" : box_size_val

    # Ensure array types are concrete
    specific_inter_lists = ([bonds...], [angles...], [torsions...])
    general_inters = (lj, coulomb)

    # Convert atom types to integers so they are bits types
    atoms = [Atom(attype=0, charge=a.charge, mass=a.mass, σ=a.σ, ϵ=a.ϵ) for a in atoms]

    neighbor_finder = TreeNeighborFinder(nb_matrix=nb_matrix, matrix_14=matrix_14, n_steps=10,
                                            dist_cutoff=units ? T(1.5)u"nm" : T(1.5))

    return atoms, specific_inter_lists, general_inters,
            neighbor_finder, [coords...], box_size
end

function readinputs(top_file::AbstractString, coord_file::AbstractString; kwargs...)
    return readinputs(DefaultFloat, top_file, coord_file; kwargs...)
end

#
struct OpenMMAtomtype{M, S, E}
    class::String
    element::String
    mass::M
    σ::S
    ϵ::E
end

#
struct OpenMMResiduetype{C}
    name::String
    types::Dict{String, String}
    charges::Dict{String, C}
end

#
struct OpenMMTorsiontype{T, E}
    proper::Bool
    periodicities::Vector{Int}
    phases::Vector{T}
    ks::Vector{E}
end

#
struct OpenMMForceField{T, M, D, E, C}
    atom_types::Dict{String, OpenMMAtomtype{M, D, E}}
    residue_types::Dict{String, OpenMMResiduetype{C}}
    bond_types::Dict{Tuple{String, String}, Bondtype{D, E}}
    angle_types::Dict{Tuple{String, String, String}, Angletype{T, E}}
    torsion_types::Dict{Tuple{String, String, String, String}, OpenMMTorsiontype{T, E}}
    torsion_order::String
    weight_14_coulomb::T
    weight_14_lj::T
end

#
function readopenmmxml(T::Type, ff_files::AbstractString...)
    atom_types = Dict{String, OpenMMAtomtype}()
    residue_types = Dict{String, OpenMMResiduetype}()
    bond_types = Dict{Tuple{String, String}, Bondtype}()
    angle_types = Dict{Tuple{String, String, String}, Angletype}()
    torsion_types = Dict{Tuple{String, String, String, String}, OpenMMTorsiontype}()
    torsion_order = ""
    weight_14_coulomb = one(T)
    weight_14_lj = one(T)

    for ff_file in ff_files
        ff_xml = parsexml(read(ff_file))
        ff = root(ff_xml)
        for entry in eachelement(ff)
            entry_name = entry.name
            if entry_name == "AtomTypes"
                for atom_type in eachelement(entry)
                    class = atom_type["class"]
                    element = atom_type["element"]
                    mass = parse(T, atom_type["mass"])u"u"
                    σ = T(-1u"nm") # Updated later
                    ϵ = T(-1u"kJ * mol^-1") # Updated later
                    atom_types[class] = OpenMMAtomtype(class, element, mass, σ, ϵ)
                end
            elseif entry_name == "Residues"
                for residue in eachelement(entry)
                    name = residue["name"]
                    types = Dict{String, String}()
                    charges = Dict{String, typeof(T(1u"q"))}()
                    for atom_or_bond in eachelement(residue)
                        # Ignore bonds because they are specified elsewhere
                        if atom_or_bond.name == "Atom"
                            atom_name = atom_or_bond["name"]
                            types[atom_name] = atom_or_bond["type"]
                            charges[atom_name] = parse(T, atom_or_bond["charge"])u"q"
                        end
                    end
                    residue_types[name] = OpenMMResiduetype(name, types, charges)
                end
            elseif entry_name == "HarmonicBondForce"
                for bond in eachelement(entry)
                    atom_type_1 = bond["type1"]
                    atom_type_2 = bond["type2"]
                    b0 = parse(T, bond["length"])u"nm"
                    kb = parse(T, bond["k"])u"kJ * mol^-1"
                    bond_types[(atom_type_1, atom_type_2)] = Bondtype(b0, kb)
                end
            elseif entry_name == "HarmonicAngleForce"
                for angle in eachelement(entry)
                    atom_type_1 = angle["type1"]
                    atom_type_2 = angle["type2"]
                    atom_type_3 = angle["type3"]
                    th0 = parse(T, angle["angle"])
                    k = parse(T, angle["k"])u"kJ * mol^-1"
                    angle_types[(atom_type_1, atom_type_2, atom_type_3)] = Angletype(th0, k)
                end
            elseif entry_name == "PeriodicTorsionForce"
                torsion_order = entry["ordering"]
                for torsion in eachelement(entry)
                    proper = torsion.name == "Proper"
                    atom_type_1 = torsion["type1"]
                    atom_type_2 = torsion["type2"]
                    atom_type_3 = torsion["type3"]
                    atom_type_4 = torsion["type4"]
                    periodicities = Int64[]
                    phases = T[]
                    ks = typeof(T(1u"kJ * mol^-1"))[]
                    phase_i = 1
                    phase_present = true
                    while phase_present
                        push!(periodicities, parse(Int64, torsion["periodicity$phase_i"]))
                        push!(phases, parse(T, torsion["phase$phase_i"]))
                        push!(ks, parse(T, torsion["k$phase_i"])u"kJ * mol^-1")
                        phase_i += 1
                        phase_present = haskey(torsion, "periodicity$phase_i")
                    end
                    torsion_type = OpenMMTorsiontype(proper, periodicities, phases, ks)
                    torsion_types[(atom_type_1, atom_type_2, atom_type_3, atom_type_4)] = torsion_type
                end
            elseif entry_name == "NonbondedForce"
                weight_14_coulomb = parse(T, entry["coulomb14scale"])
                weight_14_lj = parse(T, entry["lj14scale"])
                for atom_or_attr in eachelement(entry)
                    if atom_or_attr.name == "Atom"
                        atom_type = atom_or_attr["type"]
                        # Update previous atom types
                        partial_type = atom_types[atom_type]
                        σ = parse(T, atom_or_attr["sigma"])u"nm"
                        ϵ = parse(T, atom_or_attr["epsilon"])u"kJ * mol^-1"
                        complete_type = OpenMMAtomtype(partial_type.class, partial_type.element,
                                                        partial_type.mass, σ, ϵ)
                        atom_types[atom_type] = complete_type
                    end
                end
            end
        end
    end

    # Check all atoms were updated
    for atom_type in values(atom_types)
        if atom_type.σ < zero(T)u"nm"
            error("Atom of class ", atom_type.class, " has not had σ or ϵ set")
        end
    end

    M = typeof(T(1u"u"))
    D = typeof(T(1u"nm"))
    E = typeof(T(1u"kJ * mol^-1"))
    C = typeof(T(1u"q"))
    return OpenMMForceField{T, M, D, E, C}(atom_types, residue_types, bond_types, angle_types,
                torsion_types, torsion_order, weight_14_coulomb, weight_14_lj)
end

# Return the residue name with N or C added for terminal residues
# Assumes no missing residue numbers
function residuename(res, res_num_to_standard::Dict)
    res_num = id(res)
    res_name = Chemfiles.name(res)
    if res_num_to_standard[res_num]
        if res_num == 1 || !res_num_to_standard[res_num - 1]
            res_name = "N" * res_name
        elseif res_num == length(res_num_to_standard) || !res_num_to_standard[res_num + 1]
            res_name = "C" * res_name
        end
    end
    return res_name
end

function setupsystem(coord_file::AbstractString, force_field)
    T = typeof(force_field.weight_14_coulomb)

    # Chemfiles uses zero-based indexing, be careful
    trajectory = Trajectory(coord_file)
    frame = read(trajectory)
    top = Topology(frame)

    atoms = Atom[]
    bonds = HarmonicBond[]
    pairs = Tuple{Int, Int}[]
    angles = HarmonicAngle[]
    torsions = Torsion[]

    res_num_to_standard = Dict{Int64, Bool}()
    for ri in 1:count_residues(top)
        res = Chemfiles.Residue(top, ri - 1)
        res_num = id(res)
        res_name = Chemfiles.name(res)
        res_num_to_standard[res_num] = res_name in keys(threeletter_to_aa)
    end

    for ai in 1:size(top)
        atom_name = Chemfiles.name(Chemfiles.Atom(top, ai - 1))
        res_name = residuename(residue_for_atom(top, ai - 1), res_num_to_standard)
        type = force_field.residue_types[res_name].types[atom_name]
        charge = force_field.residue_types[res_name].charges[atom_name]
        at = force_field.atom_types[type]
        push!(atoms, Atom(attype=type, charge=charge, mass=at.mass, σ=at.σ, ϵ=at.ϵ))
    end

    for (a1z, a2z) in eachcol(Int64.(Chemfiles.bonds(top)))
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = Chemfiles.name(Chemfiles.Atom(top, a2z))
        res_name_1 = residuename(residue_for_atom(top, a1z), res_num_to_standard)
        res_name_2 = residuename(residue_for_atom(top, a2z), res_num_to_standard)
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        if haskey(force_field.bond_types, (atom_type_1, atom_type_2))
            bond_type = force_field.bond_types[(atom_type_1, atom_type_2)]
        else
            bond_type = force_field.bond_types[(atom_type_2, atom_type_1)]
        end
        push!(bonds, HarmonicBond(i=(a1z + 1), j=(a2z + 1), b0=bond_type.b0, kb=bond_type.kb))
    end

    for (a1z, a2z, a3z) in eachcol(Int64.(Chemfiles.angles(top)))
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = Chemfiles.name(Chemfiles.Atom(top, a2z))
        atom_name_3 = Chemfiles.name(Chemfiles.Atom(top, a3z))
        res_name_1 = residuename(residue_for_atom(top, a1z), res_num_to_standard)
        res_name_2 = residuename(residue_for_atom(top, a2z), res_num_to_standard)
        res_name_3 = residuename(residue_for_atom(top, a3z), res_num_to_standard)
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        atom_type_3 = force_field.residue_types[res_name_3].types[atom_name_3]
        if haskey(force_field.angle_types, (atom_type_1, atom_type_2, atom_type_3))
            angle_type = force_field.angle_types[(atom_type_1, atom_type_2, atom_type_3)]
        else
            angle_type = force_field.angle_types[(atom_type_3, atom_type_2, atom_type_1)]
        end
        push!(angles, HarmonicAngle(i=(a1z + 1), j=(a2z + 1), k=(a3z + 1), th0=angle_type.th0, cth=angle_type.cth))
    end

    # Convert from Å
    coords = [T.(SVector{3}(col)u"nm" / 10.0) for col in eachcol(positions(frame))]

    # Convert from Å
    # Switch this to 3D
    box_size = T(lengths(UnitCell(frame))[1]u"nm" / 10.0)

    specific_inter_lists = ([bonds...], [angles...], [torsions...])

    lj = LennardJones(cutoff=ShiftedPotentialCutoff(T(1.2)u"nm"), nl_only=true,
                        weight_14=force_field.weight_14_lj)
    coulomb = Coulomb(cutoff=ShiftedPotentialCutoff(T(1.2)u"nm"), nl_only=true,
                        weight_14=force_field.weight_14_coulomb)
    general_inters = (lj, coulomb)

    #neighbor_finder = TreeNeighborFinder(nb_matrix=nb_matrix, matrix_14=matrix_14, n_steps=10,
    #                                        dist_cutoff=T(1.5)u"nm")

    # isbits atoms
    return atoms, specific_inter_lists, general_inters,
            coords, box_size#neighbor_finder, coords, box_size
end
