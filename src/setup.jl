# Read files and set up simulation
# See http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    readinputs,
    Simulation

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

"Gromacs forcefield information."
struct Forcefield
    name::String
    atomtypes::Dict{String, Atomtype}
    bondtypes::Dict{String, Bondtype}
    angletypes::Dict{String, Angletype}
    dihedraltypes::Dict{String, Dihedraltype}
    atomnames::Dict{String, String}
end

"An atom and its associated information."
struct Atom
    attype::String
    name::String
    resnum::Int
    resname::String
    charge::Float64
    mass::Float64
    σ::Float64
    ϵ::Float64
end

"A bond between two atoms."
struct Bond
    atom_i::Int
    atom_j::Int
end

"A bond angle between three atoms."
struct Angle
    atom_i::Int
    atom_j::Int
    atom_k::Int
end

"A dihedral torsion angle between four atoms."
struct Dihedral
    atom_i::Int
    atom_j::Int
    atom_k::Int
    atom_l::Int
end

"A molecule."
struct Molecule
    name::String
    atoms::Vector{Atom}
    bonds::Vector{Bond}
    angles::Vector{Angle}
    dihedrals::Vector{Dihedral}
    nb_matrix::BitArray{2}
    nb_pairs::BitArray{2}
end

"3D coordinates, e.g. for an atom, in nm."
mutable struct Coordinates <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    z::Float64
end

"3D velocity values, e.g. for an atom, in nm/ps."
mutable struct Velocity <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    z::Float64
end

"A universe of a molecule, its instantaneous data and the surrounding box."
struct Universe
    molecule::Molecule
    coords::Vector{Coordinates}
    velocities::Vector{Velocity}
    temperature::Float64
    box_size::Float64
    neighbour_list::Vector{Tuple{Int, Int, Bool}} # i, j and whether they are 1-4 pairs (halved force)
end

"A simulation of a `Universe`, a `Forcefield` and timing information."
mutable struct Simulation
    forcefield::Forcefield
    universe::Universe
    timestep::Float64
    n_steps::Int
    steps_made::Int
    temperatures::Vector{Float64}
end

function Base.show(io::IO, s::Simulation)
    print("MD simulation with $(s.forcefield.name) forcefield, ",
            "molecule $(s.universe.molecule.name), ",
            "$(length(s.universe.coords)) atoms, ",
            "$(s.steps_made) steps made")
end

# Placeholder if we want to introduce logging later on
report(msg) = println(msg)

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
            bondtypes["$(c[1])/$(c[2])"] = Bondtype(parse(Float64, c[4]), parse(Float64, c[5]))
        elseif current_field == "angletypes"
            # Convert th0 to radians
            angletypes["$(c[1])/$(c[2])/$(c[3])"] = Angletype(deg2rad(
                                parse(Float64, c[5])), parse(Float64, c[6]))
        elseif current_field == "dihedraltypes" && c[1] != "#define"
            # Convert back to OPLS types
            f4 = parse(Float64, c[10]) / -4
            f3 = parse(Float64, c[9]) / -2
            f2 = 4*f4 - parse(Float64, c[8])
            f1 = 3*f3 - 2*parse(Float64, c[7])
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
            push!(atoms, Atom(attype, c[5], parse(Int, c[3]), c[4], parse(Float64, c[7]),
                parse(Float64, c[8]), atomtypes[attype].σ, atomtypes[attype].ϵ))
        elseif current_field == "bonds"
            push!(bonds, Bond(parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "pairs"
            push!(pairs, (parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "angles"
            push!(angles, Angle(parse(Int, c[1]), parse(Int, c[2]), parse(Int, c[3])))
        elseif current_field == "dihedrals"
            push!(dihedrals, Dihedral(parse(Int, c[1]), parse(Int, c[2]),
                parse(Int, c[3]), parse(Int, c[4])))
        elseif current_field == "system"
            name = rstrip(first(split(sl, ";", limit=2)))
        end
    end

    # Add new dihedral types to match wildcards
    retained_dihedrals = Dihedral[]
    for d in dihedrals
        at_types = [atoms[x].attype for x in [d.atom_i, d.atom_j, d.atom_k, d.atom_l]]
        desired_key = join(at_types, "/")
        if haskey(dihedraltypes, desired_key)
            push!(retained_dihedrals, d)
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
                dihedraltypes[desired_key] = dihedraltypes[best_key]
                push!(retained_dihedrals, d)
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

        # This atom was not specified explicitly in the topology and is added here
        if i > length(atoms)
            atname = strip(l[11:15])
            attype = replace(atname, r"\d+" => "")
            temp_charge = atomtypes[attype].charge
            if attype == "CL" # Temp hack to fix charges
                temp_charge = -1.0
            end
            push!(atoms, Atom(attype, atname, parse(Int, l[1:5]), strip(l[6:10]),
                temp_charge, atomtypes[attype].mass,
                atomtypes[attype].σ, atomtypes[attype].ϵ))

            # Add O-H bonds and H-O-H angle in water
            if atname == "OW"
                push!(bonds, Bond(i, i+1))
                push!(bonds, Bond(i, i+2))
                push!(angles, Angle(i+1, i, i+2))
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
        nb_matrix[b.atom_i, b.atom_j] = false
        nb_matrix[b.atom_j, b.atom_i] = false
    end
    for a in angles
        # Assume bonding is already specified
        nb_matrix[a.atom_i, a.atom_k] = false
        nb_matrix[a.atom_k, a.atom_i] = false
    end

    # Calculate matrix of pairs eligible for halved non-bonded interactions
    # This applies to specified pairs in the topology file, usually 1-4 bonded
    nb_pairs = falses(n_atoms, n_atoms)
    for (i, j) in pairs
        nb_pairs[i, j] = true
        nb_pairs[j, i] = true
    end

    # Bounding box for PBCs - box goes 0 to this value in 3 dimensions
    box_size = parse(Float64, first(split(strip(lines[end]), r"\s+")))

    return Forcefield("OPLS", atomtypes, bondtypes, angletypes, dihedraltypes, atomnames),
        Molecule(name, atoms, bonds, angles, retained_dihedrals, nb_matrix, nb_pairs),
        coords,
        box_size
end

# Generate a random velocity from the Maxwell-Boltzmann distribution
function maxwellboltzmann(mass::Real, T::Real)
    norm_dist = sum(rand(12)) - 6
    return abs(norm_dist) * sqrt(T / mass)
end

# Generate a random 3D velocity from the Maxwell-Boltzmann distribution
function Velocity(mass::Real, T::Real)
    return Velocity([maxwellboltzmann(mass, T) for _ in 1:3])
end

function Simulation(forcefield::Forcefield,
                molecule::Molecule,
                coords::Vector{Coordinates},
                box_size::Real,
                temperature::Real,
                timestep::Real,
                n_steps::Real)
    n_atoms = length(coords)
    v = [Velocity(molecule.atoms[i].mass, temperature) for i in 1:n_atoms]
    u = Universe(molecule, coords, v, temperature, box_size, [])
    return Simulation(forcefield, u, timestep, n_steps, 0, [])
end
