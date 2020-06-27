# Read files to set up a simulation
# See http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    Atomtype,
    Bondtype,
    Angletype,
    Torsiontype,
    readinputs

"Gromacs atom type."
struct Atomtype{T}
    mass::T
    charge::T
    σ::T
    ϵ::T
end

"Gromacs bond type."
struct Bondtype{T}
    b0::T
    kb::T
end

"Gromacs angle type."
struct Angletype{T}
    th0::T
    cth::T
end

"Gromacs torsion type."
struct Torsiontype{T}
    f1::T
    f2::T
    f3::T
    f4::T
end

"""
    readinputs(topology_file, coordinate_file)
    readinputs(T, topology_file, coordinate_file)

Read a Gromacs topology flat file, i.e. all includes collapsed into one file,
and a Gromacs coordinate file.
Returns the atoms, specific interaction lists, general interaction lists,
non-bonded matrix, coordinates and box size.
"""
function readinputs(T::Type,
                    top_file::AbstractString,
                    coord_file::AbstractString)
    # Read forcefield and topology file
    atomtypes = Dict{String, Atomtype{T}}()
    bondtypes = Dict{String, Bondtype{T}}()
    angletypes = Dict{String, Angletype{T}}()
    torsiontypes = Dict{String, Torsiontype{T}}()
    atomnames = Dict{String, String}()

    name = "?"
    atoms = Atom{T}[]
    bonds = HarmonicBond{T}[]
    pairs = Tuple{Int, Int}[]
    angles = HarmonicAngle{T}[]
    possible_torsions = Tuple{Int, Int, Int, Int}[]
    torsions = Torsion{T}[]

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
            bondtype = Bondtype(parse(T, c[4]), parse(T, c[5]))
            bondtypes["$(c[1])/$(c[2])"] = bondtype
            bondtypes["$(c[2])/$(c[1])"] = bondtype
        elseif current_field == "angletypes"
            # Convert th0 to radians
            angletype = Angletype(deg2rad(parse(T, c[5])), parse(T, c[6]))
            angletypes["$(c[1])/$(c[2])/$(c[3])"] = angletype
            angletypes["$(c[3])/$(c[2])/$(c[1])"] = angletype
        elseif current_field == "dihedraltypes" && c[1] != "#define"
            # Convert back to OPLS types
            f4 = parse(T, c[10]) / -4
            f3 = parse(T, c[9]) / -2
            f2 = 4 * f4 - parse(T, c[8])
            f1 = 3 * f3 - 2 * parse(T, c[7])
            torsiontypes["$(c[1])/$(c[2])/$(c[3])/$(c[4])"] = Torsiontype(f1, f2, f3, f4)
        elseif current_field == "atomtypes" && length(c) >= 8
            atomname = uppercase(c[2])
            atomnames[c[1]] = atomname
            # Take the first version of each atom type only
            if !haskey(atomtypes, atomname)
                atomtypes[atomname] = Atomtype(parse(T, c[4]), parse(T, c[5]),
                        parse(T, c[7]), parse(T, c[8]))
            end
        elseif current_field == "atoms"
            attype = atomnames[c[2]]
            push!(atoms, Atom(attype=attype, name=c[5], resnum=parse(Int, c[3]),
                resname=c[4], charge=parse(T, c[7]),
                mass=parse(T, c[8]), σ=atomtypes[attype].σ,
                ϵ=atomtypes[attype].ϵ))
        elseif current_field == "bonds"
            i, j = parse.(Int, c[1:2])
            bondtype = bondtypes["$(atoms[i].attype)/$(atoms[j].attype)"]
            push!(bonds, HarmonicBond(i, j, bondtype.b0, bondtype.kb))
        elseif current_field == "pairs"
            push!(pairs, (parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "angles"
            i, j, k = parse.(Int, c[1:3])
            angletype = angletypes["$(atoms[i].attype)/$(atoms[j].attype)/$(atoms[k].attype)"]
            push!(angles, HarmonicAngle(i, j, k, angletype.th0, angletype.cth))
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
            push!(torsions, Torsion(inds..., d.f1, d.f2, d.f3, d.f4))
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
                push!(torsions, Torsion(inds..., d.f1, d.f2, d.f3, d.f4))
            end
        end
    end

    # Read coordinate file and add solvent atoms
    lines = readlines(coord_file)
    coords = SArray{Tuple{3}, T, 1, 3}[]
    for (i, l) in enumerate(lines[3:end-1])
        push!(coords, SVector(
            parse(T, l[21:28]),
            parse(T, l[29:36]),
            parse(T, l[37:44])
        ))

        # Some atoms are not specified explicitly in the topology so are added here
        if i > length(atoms)
            atname = strip(l[11:15])
            attype = replace(atname, r"\d+" => "")
            temp_charge = atomtypes[attype].charge
            if attype == "CL" # Temp hack to fix charges
                temp_charge = T(-1.0)
            end
            push!(atoms, Atom(attype=attype, name=atname,
                resnum=parse(Int, l[1:5]), resname=strip(l[6:10]),
                charge=temp_charge, mass=atomtypes[attype].mass,
                σ=atomtypes[attype].σ, ϵ=atomtypes[attype].ϵ))

            # Add O-H bonds and H-O-H angle in water
            if atname == "OW"
                bondtype = bondtypes["OW/HW"]
                push!(bonds, HarmonicBond(i, i + 1, bondtype.b0, bondtype.kb))
                push!(bonds, HarmonicBond(i, i + 2, bondtype.b0, bondtype.kb))
                angletype = angletypes["HW/OW/HW"]
                push!(angles, HarmonicAngle(i + 1, i, i + 2, angletype.th0, angletype.cth))
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
    #    nb_matrix[i, j] = T(0.5)
    #    nb_matrix[j, i] = T(0.5)
    #end

    lj = LennardJones(true)
    coulomb = Coulomb(true)

    # Bounding box for PBCs - box goes 0 to this value in 3 dimensions
    box_size = parse(T, first(split(strip(lines[end]), r"\s+")))

    specific_inter_lists = (bonds, angles, torsions)
    general_inters = (lj, coulomb)

    return atoms, specific_inter_lists, general_inters,
            nb_matrix, coords, box_size
end

function readinputs(top_file::AbstractString, coord_file::AbstractString)
    return readinputs(Float64, top_file, coord_file)
end
