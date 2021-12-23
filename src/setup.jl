# Read files to set up a system
# See http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    place_atoms,
    place_diatomics,
    readinputs,
    OpenMMAtomType,
    OpenMMResidueType,
    PeriodicTorsionType,
    OpenMMForceField,
    setupsystem

"""
    place_atoms(n_atoms, box_size, min_dist; dims=3)

Obtain `n_atoms` 3D coordinates in a box with sides `box_size` where no two
points are closer than `min_dist`, accounting for periodic boundary conditions.
"""
function place_atoms(n_atoms::Integer, box_size, min_dist; dims::Integer=3)
    min_dist_sq = min_dist ^ 2
    T = typeof(convert(AbstractFloat, ustrip(first(box_size))))
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
    place_diatomics(n_molecules, box_size, min_dist, bond_length; dims=3)

Obtain 3D coordinates for `n_molecules` diatomics in a box with sides `box_size`
where no two points are closer than `min_dist` and the bond length is `bond_length`,
accounting for periodic boundary conditions.
"""
function place_diatomics(n_molecules::Integer, box_size, min_dist, bond_length; dims::Integer=3)
    min_dist_sq = min_dist ^ 2
    T = typeof(convert(AbstractFloat, ustrip(first(box_size))))
    coords = SArray[]
    while length(coords) < (n_molecules * 2)
        new_coord_a = SVector{dims}(rand(T, dims)) .* box_size
        shift = SVector{dims}([bond_length, [zero(bond_length) for d in 1:(dims - 1)]...])
        new_coord_b = copy(new_coord_a) + shift
        okay = new_coord_b[1] <= box_size[1]
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
Returns the atoms, atoms data, specific interaction lists, general
interaction lists, neighbor finder, coordinates and box size.
`units` determines whether the returned values have units.
"""
function readinputs(T::Type,
                    top_file::AbstractString,
                    coord_file::AbstractString;
                    units::Bool=true,
                    gpu::Bool=false,
                    gpu_diff_safe::Bool=gpu,
                    dist_cutoff=units ? 1.0u"nm" : 1.0,
                    nl_dist=units ? 1.2u"nm" : 1.2)
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
                bondtype = HarmonicBond(parse(T, c[4])u"nm", parse(T, c[5])u"kJ * mol^-1 * nm^-2")
            else
                bondtype = HarmonicBond(parse(T, c[4]), parse(T, c[5]))
            end
            bondtypes["$(c[1])/$(c[2])"] = bondtype
            bondtypes["$(c[2])/$(c[1])"] = bondtype
        elseif current_field == "angletypes"
            # Convert th0 to radians
            if units
                angletype = HarmonicAngle(deg2rad(parse(T, c[5])), parse(T, c[6])u"kJ * mol^-1")
            else
                angletype = HarmonicAngle(deg2rad(parse(T, c[5])), parse(T, c[6]))
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
                    atomtypes[atomname] = Atom(charge=parse(T, c[5]), mass=parse(T, c[4])u"u",
                            σ=parse(T, c[7])u"nm", ϵ=parse(T, c[8])u"kJ * mol^-1")
                else
                    atomtypes[atomname] = Atom(charge=parse(T, c[5]), mass=parse(T, c[4]),
                            σ=parse(T, c[7]), ϵ=parse(T, c[8]))
                end
            end
        elseif current_field == "atoms"
            attype = atomnames[c[2]]
            charge = parse(T, c[7])
            if units
                mass = parse(T, c[8])u"u"
            else
                mass = parse(T, c[8])
            end
            atom_index = length(atoms) + 1
            push!(atoms, Atom(index=atom_index, charge=charge, mass=mass,
                                σ=atomtypes[attype].σ, ϵ=atomtypes[attype].ϵ))
            push!(atoms_data, AtomData(atom_type=attype, atom_name=c[5], res_number=parse(Int, c[3]),
                                        res_name=c[4]))
        elseif current_field == "bonds"
            i, j = parse.(Int, c[1:2])
            bondtype = bondtypes["$(atoms_data[i].atom_type)/$(atoms_data[j].atom_type)"]
            push!(bonds.is, i)
            push!(bonds.js, j)
            push!(bonds.inters, HarmonicBond(b0=bondtype.b0, kb=bondtype.kb))
        elseif current_field == "pairs"
            push!(pairs, (parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "angles"
            i, j, k = parse.(Int, c[1:3])
            angletype = angletypes["$(atoms_data[i].atom_type)/$(atoms_data[j].atom_type)/$(atoms_data[k].atom_type)"]
            push!(angles.is, i)
            push!(angles.js, j)
            push!(angles.ks, k)
            push!(angles.inters, HarmonicAngle(th0=angletype.th0, cth=angletype.cth))
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
                                σ=atomtypes[attype].σ, ϵ=atomtypes[attype].ϵ))
            push!(atoms_data, AtomData(atom_type=attype, atom_name=atname, res_number=parse(Int, l[1:5]),
                                        res_name=strip(l[6:10])))

            # Add O-H bonds and H-O-H angle in water
            if atname == "OW"
                bondtype = bondtypes["OW/HW"]
                push!(bonds.is, i)
                push!(bonds.js, i + 1)
                push!(bonds.inters, HarmonicBond(b0=bondtype.b0, kb=bondtype.kb))
                push!(bonds.is, i)
                push!(bonds.js, i + 2)
                push!(bonds.inters, HarmonicBond(b0=bondtype.b0, kb=bondtype.kb))
                angletype = angletypes["HW/OW/HW"]
                push!(angles.is, i + 1)
                push!(angles.js, i)
                push!(angles.ks, i + 2)
                push!(angles.inters, HarmonicAngle(th0=angletype.th0, cth=angletype.cth))
            end
        end
    end

    # Calculate matrix of pairs eligible for non-bonded interactions
    n_atoms = length(coords)
    nb_matrix = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        nb_matrix[i, i] = false
    end
    for (i, j) in zip(bonds.is, bonds.js)
        nb_matrix[i, j] = false
        nb_matrix[j, i] = false
    end
    for (i, k) in zip(angles.is, angles.ks)
        # Assume bonding is already specified
        nb_matrix[i, k] = false
        nb_matrix[k, i] = false
    end

    # Calculate matrix of pairs eligible for halved non-bonded interactions
    # This applies to specified pairs in the topology file, usually 1-4 bonded
    matrix_14 = falses(n_atoms, n_atoms)
    for (i, j) in pairs
        matrix_14[i, j] = true
        matrix_14[j, i] = true
    end

    lj = LennardJones(cutoff=DistanceCutoff(T(dist_cutoff)), nl_only=true, weight_14=T(0.5),
                        force_unit=force_unit, energy_unit=energy_unit)
    coulomb_rf = CoulombReactionField(dist_cutoff=T(dist_cutoff), solvent_dielectric=T(solventdielectric),
                                        nl_only=true, weight_14=T(0.5),
                                        coulomb_const=units ? T(coulombconst) : T(ustrip(coulombconst)),
                                        force_unit=force_unit, energy_unit=energy_unit)

    # Bounding box for PBCs - box goes 0 to a value in each of 3 dimensions
    box_size_vals = SVector{3}(parse.(T, split(strip(lines[end]), r"\s+")))
    box_size = units ? (box_size_vals)u"nm" : box_size_vals
    coords = wrapcoordsvec.([coords...], (box_size,))

    general_inters = (lj, coulomb_rf)
    # Ensure array types are concrete
    if gpu
        specific_inter_lists = (InteractionList2Atoms(bonds.is, bonds.js, cu([bonds.inters...])),
                                InteractionList3Atoms(angles.is, angles.js, angles.ks, cu([angles.inters...])),
                                InteractionList4Atoms(torsions.is, torsions.js, torsions.ks, torsions.ls,
                                                        cu([torsions.inters...])))
    else
        specific_inter_lists = (InteractionList2Atoms(bonds.is, bonds.js, [bonds.inters...]),
                                InteractionList3Atoms(angles.is, angles.js, angles.ks, [angles.inters...]),
                                InteractionList4Atoms(torsions.is, torsions.js, torsions.ks, torsions.ls,
                                                        [torsions.inters...]))
    end

    atoms = [Atom(index=a.index, charge=a.charge, mass=a.mass, σ=a.σ, ϵ=a.ϵ) for a in atoms]

    if gpu_diff_safe
        neighbor_finder = DistanceVecNeighborFinder(nb_matrix=gpu ? cu(nb_matrix) : nb_matrix,
                                                    matrix_14=gpu ? cu(matrix_14) : matrix_14, n_steps=10,
                                                    dist_cutoff=T(nl_dist))
    else
        neighbor_finder = CellListMapNeighborFinder(nb_matrix=nb_matrix, matrix_14=matrix_14, n_steps=10,
                                                    x0=coords, unit_cell=box_size, dist_cutoff=T(nl_dist))
    end
    if gpu
        atoms = cu(atoms)
        coords = cu(coords)
    end

    return atoms, atoms_data, specific_inter_lists, general_inters,
            neighbor_finder, coords, box_size
end

function readinputs(top_file::AbstractString, coord_file::AbstractString; kwargs...)
    return readinputs(DefaultFloat, top_file, coord_file; kwargs...)
end

"""
    OpenMMAtomType(class, element, mass, σ, ϵ)

An OpenMM atom type.
"""
struct OpenMMAtomType{M, S, E}
    class::String
    element::String
    mass::M
    σ::S
    ϵ::E
end

"""
    OpenMMResiduetype(name, types, charges, indices)

An OpenMM residue type.
"""
struct OpenMMResiduetype{C}
    name::String
    types::Dict{String, String}
    charges::Dict{String, C}
    indices::Dict{String, Int}
end

"""
    PeriodicTorsionType(proper, periodicities, phases, ks)

A periodic torsion type.
"""
struct PeriodicTorsionType{T, E}
    proper::Bool
    periodicities::Vector{Int}
    phases::Vector{T}
    ks::Vector{E}
end

"""
    OpenMMForceField(ff_files...)
    OpenMMForceField(T, ff_files...)
    OpenMMForceField(atom_types, residue_types, bond_types, angle_types,
                        torsion_types, torsion_order, weight_14_coulomb,
                        weight_14_lj)

An OpenMM force field.
Read one or more OpenMM force field XML files by passing them to the
constructor.
"""
struct OpenMMForceField{T, M, D, E, K}
    atom_types::Dict{String, OpenMMAtomType{M, D, E}}
    residue_types::Dict{String, OpenMMResiduetype{T}}
    bond_types::Dict{Tuple{String, String}, HarmonicBond{D, K}}
    angle_types::Dict{Tuple{String, String, String}, HarmonicAngle{T, E}}
    torsion_types::Dict{Tuple{String, String, String, String}, PeriodicTorsionType{T, E}}
    torsion_order::String
    weight_14_coulomb::T
    weight_14_lj::T
end

function OpenMMForceField(T::Type, ff_files::AbstractString...; units::Bool=true)
    atom_types = Dict{String, OpenMMAtomType}()
    residue_types = Dict{String, OpenMMResiduetype}()
    bond_types = Dict{Tuple{String, String}, HarmonicBond}()
    angle_types = Dict{Tuple{String, String, String}, HarmonicAngle}()
    torsion_types = Dict{Tuple{String, String, String, String}, PeriodicTorsionType}()
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
                    mass = units ? parse(T, atom_type["mass"])u"u" : parse(T, atom_type["mass"])
                    σ = units ? T(-1u"nm") : T(-1) # Updated later
                    ϵ = units ? T(-1u"kJ * mol^-1") : T(-1) # Updated later
                    atom_types[class] = OpenMMAtomType(class, element, mass, σ, ϵ)
                end
            elseif entry_name == "Residues"
                for residue in eachelement(entry)
                    name = residue["name"]
                    types = Dict{String, String}()
                    charges = Dict{String, T}()
                    indices = Dict{String, Int}()
                    index = 1
                    for atom_or_bond in eachelement(residue)
                        # Ignore bonds because they are specified elsewhere
                        if atom_or_bond.name == "Atom"
                            atom_name = atom_or_bond["name"]
                            types[atom_name] = atom_or_bond["type"]
                            charges[atom_name] = parse(T, atom_or_bond["charge"])
                            indices[atom_name] = index
                            index += 1
                        end
                    end
                    residue_types[name] = OpenMMResiduetype(name, types, charges, indices)
                end
            elseif entry_name == "HarmonicBondForce"
                for bond in eachelement(entry)
                    atom_type_1 = bond["type1"]
                    atom_type_2 = bond["type2"]
                    b0 = units ? parse(T, bond["length"])u"nm" : parse(T, bond["length"])
                    kb = units ? parse(T, bond["k"])u"kJ * mol^-1 * nm^-2" : parse(T, bond["k"])
                    bond_types[(atom_type_1, atom_type_2)] = HarmonicBond(b0, kb)
                end
            elseif entry_name == "HarmonicAngleForce"
                for angle in eachelement(entry)
                    atom_type_1 = angle["type1"]
                    atom_type_2 = angle["type2"]
                    atom_type_3 = angle["type3"]
                    th0 = parse(T, angle["angle"])
                    k = units ? parse(T, angle["k"])u"kJ * mol^-1" : parse(T, angle["k"])
                    angle_types[(atom_type_1, atom_type_2, atom_type_3)] = HarmonicAngle(th0, k)
                end
            elseif entry_name == "PeriodicTorsionForce"
                torsion_order = entry["ordering"]
                for torsion in eachelement(entry)
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
                    torsion_type = PeriodicTorsionType(proper, periodicities, phases, ks)
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
                        σ = units ? parse(T, atom_or_attr["sigma"])u"nm" : parse(T, atom_or_attr["sigma"])
                        ϵ = units ? parse(T, atom_or_attr["epsilon"])u"kJ * mol^-1" : parse(T, atom_or_attr["epsilon"])
                        complete_type = OpenMMAtomType(partial_type.class, partial_type.element,
                                                        partial_type.mass, σ, ϵ)
                        atom_types[atom_type] = complete_type
                    end
                end
            end
        end
    end

    # Check all atoms were updated
    for atom_type in values(atom_types)
        if (units && atom_type.σ < zero(T)u"nm") || (!units && atom_type.σ < zero(T))
            error("Atom of class ", atom_type.class, " has not had σ or ϵ set")
        end
    end

    if units
        M = typeof(T(1u"u"))
        D = typeof(T(1u"nm"))
        E = typeof(T(1u"kJ * mol^-1"))
        K = typeof(T(1u"kJ * mol^-1 * nm^-2"))
    else
        M, D, E, K = T, T, T, T
    end
    return OpenMMForceField{T, M, D, E, K}(atom_types, residue_types, bond_types, angle_types,
                torsion_types, torsion_order, weight_14_coulomb, weight_14_lj)
end

OpenMMForceField(ff_files::AbstractString...; kwargs...) = OpenMMForceField(DefaultFloat, ff_files...; kwargs...)

# Return the residue name with N or C added for terminal residues
# Assumes no missing residue numbers, won't work with multiple chains
function residuename(res, res_num_to_standard::Dict, rename_terminal_res::Bool=true)
    res_num = Chemfiles.id(res)
    res_name = Chemfiles.name(res)
    if rename_terminal_res && res_num_to_standard[res_num]
        if res_num == 1 || !res_num_to_standard[res_num - 1]
            res_name = "N" * res_name
        elseif res_num == length(res_num_to_standard) || !res_num_to_standard[res_num + 1]
            res_name = "C" * res_name
        end
    end
    return res_name
end

"""
    setupsystem(coord_file, force_field; dist_cutoff=1.0u"nm")

Read a coordinate file and apply a force field to it.
Any file format readable by Chemfiles can be given.
Returns the atoms, specific interaction lists, general interaction lists,
neighbor finder, coordinates and box size.
"""
function setupsystem(coord_file::AbstractString,
                        force_field;
                        units::Bool=true,
                        gpu::Bool=false,
                        gpu_diff_safe::Bool=gpu,
                        dist_cutoff=units ? 1.0u"nm" : 1.0,
                        nl_dist=units ? 1.2u"nm" : 1.2,
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
    nb_matrix = trues(n_atoms, n_atoms)
    matrix_14 = falses(n_atoms, n_atoms)

    top_bonds     = Vector{Int}[is for is in eachcol(Int.(Chemfiles.bonds(    top)))]
    top_angles    = Vector{Int}[is for is in eachcol(Int.(Chemfiles.angles(   top)))]
    top_torsions  = Vector{Int}[is for is in eachcol(Int.(Chemfiles.dihedrals(top)))]
    top_impropers = Vector{Int}[is for is in eachcol(Int.(Chemfiles.impropers(top)))]

    res_num_to_standard = Dict{Int, Bool}()
    for ri in 1:Chemfiles.count_residues(top)
        res = Chemfiles.Residue(top, ri - 1)
        res_num = Chemfiles.id(res)
        res_name = Chemfiles.name(res)
        standard_res = res_name in keys(threeletter_to_aa)
        res_num_to_standard[res_num] = standard_res

        if standard_res && residuename(res, res_num_to_standard, rename_terminal_res) == "N" * res_name
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

    for ai in 1:n_atoms
        atom_name = Chemfiles.name(Chemfiles.Atom(top, ai - 1))
        res = Chemfiles.residue_for_atom(top, ai - 1)
        res_name = residuename(res, res_num_to_standard, rename_terminal_res)
        type = force_field.residue_types[res_name].types[atom_name]
        charge = force_field.residue_types[res_name].charges[atom_name]
        at = force_field.atom_types[type]
        push!(atoms, Atom(index=ai, charge=charge, mass=at.mass, σ=at.σ, ϵ=at.ϵ))
        push!(atoms_data, AtomData(atom_type=type, atom_name=atom_name, res_number=Chemfiles.id(res),
                                    res_name=Chemfiles.name(res), element=at.element))
        nb_matrix[ai, ai] = false
    end

    for (a1z, a2z) in top_bonds
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = Chemfiles.name(Chemfiles.Atom(top, a2z))
        res_name_1 = residuename(Chemfiles.residue_for_atom(top, a1z), res_num_to_standard, rename_terminal_res)
        res_name_2 = residuename(Chemfiles.residue_for_atom(top, a2z), res_num_to_standard, rename_terminal_res)
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        if haskey(force_field.bond_types, (atom_type_1, atom_type_2))
            bond_type = force_field.bond_types[(atom_type_1, atom_type_2)]
        else
            bond_type = force_field.bond_types[(atom_type_2, atom_type_1)]
        end
        push!(bonds.is, a1z + 1)
        push!(bonds.js, a2z + 1)
        push!(bonds.inters, HarmonicBond(b0=bond_type.b0, kb=bond_type.kb))
        nb_matrix[a1z + 1, a2z + 1] = false
        nb_matrix[a2z + 1, a1z + 1] = false
    end

    for (a1z, a2z, a3z) in top_angles
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = Chemfiles.name(Chemfiles.Atom(top, a2z))
        atom_name_3 = Chemfiles.name(Chemfiles.Atom(top, a3z))
        res_name_1 = residuename(Chemfiles.residue_for_atom(top, a1z), res_num_to_standard, rename_terminal_res)
        res_name_2 = residuename(Chemfiles.residue_for_atom(top, a2z), res_num_to_standard, rename_terminal_res)
        res_name_3 = residuename(Chemfiles.residue_for_atom(top, a3z), res_num_to_standard, rename_terminal_res)
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        atom_type_3 = force_field.residue_types[res_name_3].types[atom_name_3]
        if haskey(force_field.angle_types, (atom_type_1, atom_type_2, atom_type_3))
            angle_type = force_field.angle_types[(atom_type_1, atom_type_2, atom_type_3)]
        else
            angle_type = force_field.angle_types[(atom_type_3, atom_type_2, atom_type_1)]
        end
        push!(angles.is, a1z + 1)
        push!(angles.js, a2z + 1)
        push!(angles.ks, a3z + 1)
        push!(angles.inters, HarmonicAngle(th0=angle_type.th0, cth=angle_type.cth))
        nb_matrix[a1z + 1, a3z + 1] = false
        nb_matrix[a3z + 1, a1z + 1] = false
    end

    for (a1z, a2z, a3z, a4z) in top_torsions
        atom_name_1 = Chemfiles.name(Chemfiles.Atom(top, a1z))
        atom_name_2 = Chemfiles.name(Chemfiles.Atom(top, a2z))
        atom_name_3 = Chemfiles.name(Chemfiles.Atom(top, a3z))
        atom_name_4 = Chemfiles.name(Chemfiles.Atom(top, a4z))
        res_name_1 = residuename(Chemfiles.residue_for_atom(top, a1z), res_num_to_standard, rename_terminal_res)
        res_name_2 = residuename(Chemfiles.residue_for_atom(top, a2z), res_num_to_standard, rename_terminal_res)
        res_name_3 = residuename(Chemfiles.residue_for_atom(top, a3z), res_num_to_standard, rename_terminal_res)
        res_name_4 = residuename(Chemfiles.residue_for_atom(top, a4z), res_num_to_standard, rename_terminal_res)
        atom_type_1 = force_field.residue_types[res_name_1].types[atom_name_1]
        atom_type_2 = force_field.residue_types[res_name_2].types[atom_name_2]
        atom_type_3 = force_field.residue_types[res_name_3].types[atom_name_3]
        atom_type_4 = force_field.residue_types[res_name_4].types[atom_name_4]
        atom_types = (atom_type_1, atom_type_2, atom_type_3, atom_type_4)
        if haskey(force_field.torsion_types, atom_types) && force_field.torsion_types[atom_types].proper
            torsion_type = force_field.torsion_types[atom_types]
        elseif haskey(force_field.torsion_types, reverse(atom_types)) && force_field.torsion_types[reverse(atom_types)].proper
            torsion_type = force_field.torsion_types[reverse(atom_types)]
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
        push!(torsions.is, a1z + 1)
        push!(torsions.js, a2z + 1)
        push!(torsions.ks, a3z + 1)
        push!(torsions.ls, a4z + 1)
        push!(torsions.inters, PeriodicTorsion(periodicities=torsion_type.periodicities,
                                                phases=torsion_type.phases, ks=torsion_type.ks))
        matrix_14[a1z + 1, a4z + 1] = true
        matrix_14[a4z + 1, a1z + 1] = true
    end

    # Note the order here - Chemfiles puts the central atom second
    for (a2z, a1z, a3z, a4z) in top_impropers
        inds_no1 = (a2z, a3z, a4z)
        atom_names = [Chemfiles.name(Chemfiles.Atom(top, a)) for a in inds_no1]
        res_names = [residuename(Chemfiles.residue_for_atom(top, a), res_num_to_standard, rename_terminal_res) for a in inds_no1]
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
        res_name_1 = residuename(Chemfiles.residue_for_atom(top, a1z), res_num_to_standard, rename_terminal_res)
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
            push!(impropers.inters, PeriodicTorsion(periodicities=torsion_type.periodicities,
                                                    phases=torsion_type.phases, ks=torsion_type.ks))
        end
    end

    if units
        force_unit = u"kJ * mol^-1 * nm^-1"
        energy_unit = u"kJ * mol^-1"
    else
        force_unit = NoUnits
        energy_unit = NoUnits
    end

    lj = LennardJones(cutoff=DistanceCutoff(T(dist_cutoff)), nl_only=true, weight_14=force_field.weight_14_lj,
                        force_unit=force_unit, energy_unit=energy_unit)
    coulomb_rf = CoulombReactionField(dist_cutoff=T(dist_cutoff), solvent_dielectric=T(solventdielectric),
                                        nl_only=true, weight_14=force_field.weight_14_coulomb,
                                        coulomb_const=units ? T(coulombconst) : T(ustrip(coulombconst)),
                                        force_unit=force_unit, energy_unit=energy_unit)
    general_inters = (lj, coulomb_rf)

    # All torsions must have the same number of terms on the GPU and for taking gradients
    # For now always pad to 6 terms
    torsion_inters_pad = [PeriodicTorsion(periodicities=t.periodicities, phases=t.phases, ks=t.ks,
                                            n_terms=6) for t in torsions.inters]
    improper_inters_pad = [PeriodicTorsion(periodicities=t.periodicities, phases=t.phases, ks=t.ks,
                                            n_terms=6) for t in impropers.inters]

    # Ensure array types are concrete
    if gpu
        specific_inter_lists = (InteractionList2Atoms(bonds.is, bonds.js, cu([bonds.inters...])),
                                InteractionList3Atoms(angles.is, angles.js, angles.ks, cu([angles.inters...])),
                                InteractionList4Atoms(torsions.is, torsions.js, torsions.ks, torsions.ls,
                                                        cu(torsion_inters_pad)),
                                InteractionList4Atoms(impropers.is, impropers.js, impropers.ks, impropers.ls,
                                                        cu(improper_inters_pad)))
    else
        specific_inter_lists = (InteractionList2Atoms(bonds.is, bonds.js, [bonds.inters...]),
                                InteractionList3Atoms(angles.is, angles.js, angles.ks, [angles.inters...]),
                                InteractionList4Atoms(torsions.is, torsions.js, torsions.ks, torsions.ls,
                                                        torsion_inters_pad),
                                InteractionList4Atoms(impropers.is, impropers.js, impropers.ks, impropers.ls,
                                                        improper_inters_pad))
    end

    # Bounding box for PBCs - box goes 0 to a value in each of 3 dimensions
    # Convert from Å
    if units
        box_size = SVector{3}(T.(Chemfiles.lengths(Chemfiles.UnitCell(frame))u"nm" / 10.0))
    else
        box_size = SVector{3}(T.(Chemfiles.lengths(Chemfiles.UnitCell(frame)) / 10.0))
    end

    # Convert from Å
    if units
        coords = [T.(SVector{3}(col)u"nm" / 10.0) for col in eachcol(Chemfiles.positions(frame))]
    else
        coords = [T.(SVector{3}(col) / 10.0) for col in eachcol(Chemfiles.positions(frame))]
    end
    coords = wrapcoordsvec.(coords, (box_size,))

    atoms = [atoms...]
    if gpu_diff_safe
        neighbor_finder = DistanceVecNeighborFinder(nb_matrix=gpu ? cu(nb_matrix) : nb_matrix,
                                                    matrix_14=gpu ? cu(matrix_14) : matrix_14,
                                                    n_steps=10, dist_cutoff=T(nl_dist))
    else
        neighbor_finder = CellListMapNeighborFinder(nb_matrix=nb_matrix, matrix_14=matrix_14,
                                                    n_steps=10, x0=coords, unit_cell=box_size,
                                                    dist_cutoff=T(nl_dist))
    end
    if gpu
        atoms = cu(atoms)
        coords = cu(coords)
    end

    return atoms, atoms_data, specific_inter_lists, general_inters,
            neighbor_finder, coords, box_size
end
