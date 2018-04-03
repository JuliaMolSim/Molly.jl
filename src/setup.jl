# Read files and set up simulation. See
#   http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    readinputs,
    Simulation

struct Atomtype
    mass::Float64
    charge::Float64
    σ::Float64
    ϵ::Float64
end

struct Bondtype
    b0::Float64
    kb::Float64
end

struct Angletype
    th0::Float64
    cth::Float64
end

struct Dihedraltype
    f1::Float64
    f2::Float64
    f3::Float64
    f4::Float64
end

struct Forcefield
    name::String
    atomtypes::Dict{String, Atomtype}
    bondtypes::Dict{String, Bondtype}
    angletypes::Dict{String, Angletype}
    dihedraltypes::Dict{String, Dihedraltype}
    atomnames::Dict{String, String}
end

struct Atom
    attype::String
    name::String
    resnum::Int
    resname::String
    charge::Float64
    mass::Float64
end

struct Bond
    atom_i::Int
    atom_j::Int
end

struct Angle
    atom_i::Int
    atom_j::Int
    atom_k::Int
end

struct Dihedral
    atom_i::Int
    atom_j::Int
    atom_k::Int
    atom_l::Int
end

struct Molecule
    name::String
    atoms::Vector{Atom}
    bonds::Vector{Bond}
    angles::Vector{Angle}
    dihedrals::Vector{Dihedral}
    nb_matrix::BitArray{2}
end

mutable struct Coordinates
    x::Float64
    y::Float64
    z::Float64
end

mutable struct Velocity
    x::Float64
    y::Float64
    z::Float64
end

struct Universe
    molecule::Molecule
    coords::Vector{Coordinates}
    velocities::Vector{Velocity}
    box_size::Float64
end

mutable struct Simulation
    forcefield::Forcefield
    universe::Universe
    timestep::Float64
    n_steps::Int
    steps_made::Int
    pes::Vector{Float64}
    kes::Vector{Float64}
    energies::Vector{Float64}
    temps::Vector{Float64}
end

function Base.show(io::IO, s::Simulation)
    print("MD simulation with $(s.forcefield.name) forcefield, molecule $(s.universe.molecule.name), $(length(s.universe.coords)) atoms, $(s.steps_made) steps made")
end

# Read a Gromacs topology flat file
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
            angletypes["$(c[1])/$(c[2])/$(c[3])"] = Angletype(deg2rad(parse(Float64, c[5])), parse(Float64, c[6]))
        elseif current_field == "dihedraltypes" && c[1] != "#define"
            # Convert back to OPLS types
            f4 = parse(Float64, c[10]) / -4
            f3 = parse(Float64, c[9]) / -2
            f2 = 4*f4 - parse(Float64, c[8])
            f1 = 3*f3 - 2*parse(Float64, c[7])
            dihedraltypes["$(c[1])/$(c[2])/$(c[3])/$(c[4])"] = Dihedraltype(f1, f2, f3, f4)
        elseif current_field == "atomtypes" && length(c) >= 8
            atomnames[c[1]] = c[2]
            atomtypes[c[2]] = Atomtype(parse(Float64, c[4]), parse(Float64, c[5]), parse(Float64, c[7]), parse(Float64, c[8]))
        elseif current_field == "atoms"
            push!(atoms, Atom(atomnames[c[2]], c[5], parse(Int, c[3]), c[4], parse(Float64, c[7]), parse(Float64, c[8])))
        elseif current_field == "bonds"
            push!(bonds, Bond(parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "angles"
            push!(angles, Angle(parse(Int, c[1]), parse(Int, c[2]), parse(Int, c[3])))
        elseif current_field == "dihedrals"
            push!(dihedrals, Dihedral(parse(Int, c[1]), parse(Int, c[2]), parse(Int, c[3]), parse(Int, c[4])))
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
            if best_key == ""
                #println("Could not assign proper dihedral for $desired_key")
            else
                #println("Assigned dihedral for $desired_key")
                dihedraltypes[desired_key] = dihedraltypes[best_key]
                push!(retained_dihedrals, d)
            end
        end
    end

    # Calculate matrix of pairs eligible for non-bonded interactions
    n_atoms = length(atoms)
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
    for d in retained_dihedrals
        # Assume bonding and angles are already specified
        nb_matrix[d.atom_i, d.atom_l] = false
        nb_matrix[d.atom_l, d.atom_i] = false
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
        if i > n_atoms
            atname = strip(l[11:15])
            attype = replace(atname, r"\d+", "")
            push!(atoms, Atom(attype, atname, parse(Int, l[1:5]), strip(l[6:10]),
                atomtypes[attype].charge, atomtypes[attype].mass))
        end
    end

    return Forcefield("OPLS", atomtypes, bondtypes, angletypes, dihedraltypes, atomnames),
        Molecule(name, atoms, bonds, angles, retained_dihedrals, nb_matrix),
        coords
end

#=
defaults - ignore for now
bondtypes - done
constrainttypes - ignore for now
angletypes - done
dihedraltypes - done
dihedraltypes - dupe
bondtypes - dupe
angletypes - dupe
dihedraltypes - dupe

atomtypes - done

moleculetype - ignore for now
atoms - done
bonds - done
pairs - TODO, explicit non-bonded pairs
angles - done
dihedrals - done
dihedrals - dupe
system - ignore for now
molecules - ignore for now
=#

function Velocity(starting_velocity::Real)
    θ = rand()*π
    ϕ = rand()*2π
    r = rand()*starting_velocity
    return Velocity(r*sin(θ)*cos(ϕ), r*sin(θ)*sin(ϕ), r*cos(θ))
end

function Simulation(forcefield::Forcefield,
                molecule::Molecule,
                coords::Vector{Coordinates},
                box_size::Real,
                starting_velocity::Real,
                timestep::Real,
                n_steps::Real)
    v = [Velocity(starting_velocity) for _ in coords]
    u = Universe(molecule, coords, v, box_size)
    return Simulation(forcefield, u, timestep, n_steps, 0, [], [], [], [])
end
