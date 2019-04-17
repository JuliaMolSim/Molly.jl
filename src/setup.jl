# Read files to set up a simulation
# See http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    Atomtype,
    Bondtype,
    Angletype,
    Dihedraltype,
    readinputs

"Gromacs atom type."
struct Atomtype
    mass::Float64
    charge::Float64
    σ::Float64
    ϵ::Float64
end

"Gromacs bond type."
struct Bondtype
    b0::Float64
    kb::Float64
end

"Gromacs angle type."
struct Angletype
    th0::Float64
    cth::Float64
end

"Gromacs dihedral type."
struct Dihedraltype
    f1::Float64
    f2::Float64
    f3::Float64
    f4::Float64
end

"Read a Gromacs topology flat file, i.e. all includes collapsed into one file."
function readinputs(top_file::AbstractString, coord_file::AbstractString)
    # Read forcefield and topology file
    atomtypes = Dict{String, Atomtype}()
    bondtypes = Dict{String, Bondtype}()
    angletypes = Dict{String, Angletype}()
    dihedraltypes = Dict{String, Dihedraltype}()
    atomnames = Dict{String, String}()

    name = "?"
    atoms = Atom[]
    bonds = Bond[]
    pairs = Tuple{Int, Int}[]
    angles = Angle[]
    possible_dihedrals = Tuple{Int, Int, Int, Int}[]
    dihedrals = Dihedral[]

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
            bondtype = Bondtype(parse(Float64, c[4]), parse(Float64, c[5]))
            bondtypes["$(c[1])/$(c[2])"] = bondtype
            bondtypes["$(c[2])/$(c[1])"] = bondtype
        elseif current_field == "angletypes"
            # Convert th0 to radians
            angletype = Angletype(deg2rad(parse(Float64, c[5])), parse(Float64, c[6]))
            angletypes["$(c[1])/$(c[2])/$(c[3])"] = angletype
            angletypes["$(c[3])/$(c[2])/$(c[1])"] = angletype
        elseif current_field == "dihedraltypes" && c[1] != "#define"
            # Convert back to OPLS types
            f4 = parse(Float64, c[10]) / -4
            f3 = parse(Float64, c[9]) / -2
            f2 = 4 * f4 - parse(Float64, c[8])
            f1 = 3 * f3 - 2 * parse(Float64, c[7])
            dihedraltypes["$(c[1])/$(c[2])/$(c[3])/$(c[4])"] = Dihedraltype(f1, f2, f3, f4)
        elseif current_field == "atomtypes" && length(c) >= 8
            atomname = uppercase(c[2])
            atomnames[c[1]] = atomname
            # Take the first version of each atom type only
            if !haskey(atomtypes, atomname)
                atomtypes[atomname] = Atomtype(parse(Float64, c[4]), parse(Float64, c[5]),
                        parse(Float64, c[7]), parse(Float64, c[8]))
            end
        elseif current_field == "atoms"
            attype = atomnames[c[2]]
            push!(atoms, Atom(attype, c[5], parse(Int, c[3]), c[4],
                parse(Float64, c[7]), parse(Float64, c[8]), atomtypes[attype].σ,
                atomtypes[attype].ϵ))
        elseif current_field == "bonds"
            i, j = parse.(Int, c[1:2])
            bondtype = bondtypes["$(atoms[i].attype)/$(atoms[j].attype)"]
            push!(bonds, Bond(i, j, bondtype.b0, bondtype.kb))
        elseif current_field == "pairs"
            push!(pairs, (parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "angles"
            i, j, k = parse.(Int, c[1:3])
            angletype = angletypes["$(atoms[i].attype)/$(atoms[j].attype)/$(atoms[k].attype)"]
            push!(angles, Angle(i, j, k, angletype.th0, angletype.cth))
        elseif current_field == "dihedrals"
            i, j, k, l = parse.(Int, c[1:4])
            push!(possible_dihedrals, (i, j, k, l))
        elseif current_field == "system"
            name = rstrip(first(split(sl, ";", limit=2)))
        end
    end

    # Add dihedrals based on wildcard dihedral types
    for inds in possible_dihedrals
        at_types = [atoms[x].attype for x in inds]
        desired_key = join(at_types, "/")
        if haskey(dihedraltypes, desired_key)
            d = dihedraltypes[desired_key]
            push!(dihedrals, Dihedral(inds..., d.f1, d.f2, d.f3, d.f4))
        else
            best_score = 0
            best_key = ""
            for k in keys(dihedraltypes)
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
            # If a wildcard match is found, add a new specific dihedral type
            if best_key != ""
                d = dihedraltypes[best_key]
                push!(dihedrals, Dihedral(inds..., d.f1, d.f2, d.f3, d.f4))
            end
        end
    end

    # Read coordinate file and add solvent atoms
    lines = readlines(coord_file)
    coords = Coordinates[]
    for (i, l) in enumerate(lines[3:end-1])
        push!(coords, Coordinates(
            parse(Float64, l[21:28]),
            parse(Float64, l[29:36]),
            parse(Float64, l[37:44])
        ))

        # Some atoms are not specified explicitly in the topology so are added here
        if i > length(atoms)
            atname = strip(l[11:15])
            attype = replace(atname, r"\d+" => "")
            temp_charge = atomtypes[attype].charge
            if attype == "CL" # Temp hack to fix charges
                temp_charge = -1.0
            end
            push!(atoms, Atom(attype, atname, parse(Int, l[1:5]),
                strip(l[6:10]), temp_charge, atomtypes[attype].mass,
                atomtypes[attype].σ, atomtypes[attype].ϵ))

            # Add O-H bonds and H-O-H angle in water
            if atname == "OW"
                bondtype = bondtypes["OW/HW"]
                push!(bonds, Bond(i, i + 1, bondtype.b0, bondtype.kb))
                push!(bonds, Bond(i, i + 2, bondtype.b0, bondtype.kb))
                angletype = angletypes["HW/OW/HW"]
                push!(angles, Angle(i + 1, i, i + 2, angletype.th0, angletype.cth))
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
    #for (i, j) in pairs
    #    nb_matrix[i, j] = 0.5
    #    nb_matrix[j, i] = 0.5
    #end

    lj = LennardJones(true)
    coulomb = Coulomb(true)

    # Bounding box for PBCs - box goes 0 to this value in 3 dimensions
    box_size = parse(Float64, first(split(strip(lines[end]), r"\s+")))

    specific_inter_lists = Dict("Bonds" => bonds,
        "Angles" => angles,
        "Dihedrals" => dihedrals)

    general_inters = Dict("LJ" => lj,
        "Coulomb" => coulomb)

    return atoms, specific_inter_lists, general_inters,
            nb_matrix, coords, box_size
end
