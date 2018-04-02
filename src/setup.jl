# Read files and set up simulation. See
#   http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    readtopology,
    readcoordinates,
    Simulation

struct Bondtype
    b0::Float64
    kb::Float64
end

struct Angletype
    th0::Float64
    cth::Float64
end

struct Dihedraltype
    coeffs::Vector{Float64}
end

struct Forcefield
    name::String
    bondtypes::Dict{String, Bondtype}
    angletypes::Dict{String, Angletype}
    dihedraltypes::Dict{String, Dihedraltype}
    atomtypes::Dict{String, String}
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
function readtopology(in_file::AbstractString)
    open(in_file) do f
        bondtypes = Dict{String, Bondtype}()
        angletypes = Dict{String, Angletype}()
        dihedraltypes = Dict{String, Dihedraltype}()
        atomtypes = Dict{String, String}()
        name = "?"
        atoms = Atom[]
        bonds = Bond[]
        angles = Angle[]
        dihedrals = Dihedral[]
        current_field = ""
        for l in eachline(f)
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
                angletypes["$(c[1])/$(c[2])/$(c[3])"] = Angletype(deg2rad(parse(Float64, c[5])), parse(Float64, c[6]))
            elseif current_field == "dihedraltypes" && c[1] != "#define"
                dihedraltypes["$(c[1])/$(c[2])/$(c[3])/$(c[4])"] = Dihedraltype([parse(Float64, i) for i in c[6:11]])
            elseif current_field == "atomtypes"
                atomtypes[c[1]] = c[2]
            elseif current_field == "atoms"
                push!(atoms, Atom(atomtypes[c[2]], c[5], parse(Int, c[3]), c[4], parse(Float64, c[7]), parse(Float64, c[8])))
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
        return Forcefield("OPLS", bondtypes, angletypes, dihedraltypes, atomtypes), Molecule(name, atoms, bonds, angles, dihedrals)
    end
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
pairs - TODO
angles - done
dihedrals - done
dihedrals - dupe
system - ignore for now
molecules - ignore for now
=#

function readcoordinates(in_file::AbstractString)
    lines = readlines(in_file)
    coords = Coordinates[]
    for l in lines[3:end-1]
        push!(coords, Coordinates(
            parse(Float64, l[21:28]),
            parse(Float64, l[29:36]),
            parse(Float64, l[37:44])
        ))
    end
    return coords
end

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
