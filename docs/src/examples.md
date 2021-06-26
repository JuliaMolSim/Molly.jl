# Molly examples

The best examples for learning how the package works are in the [Molly documentation](@ref) section.
Here we give further examples, showing what you can do with the package.
Each is a self-contained block of code.
Made something cool yourself?
Make a PR to add it to this page.

## Making and breaking bonds

There is an example of mutable atom properties in the main docs, but what if you want to make and break bonds during the simulation?
In this case you can use a `GeneralInteraction` to make, break and apply the bonds.
The partners of the atom can be stored in the atom type.
We make a `Logger` to record when the bonds are present, allowing us to visualize them with the `connection_frames` keyword argument to `visualize` (this can take a while to plot).
```julia
using Molly
using Makie
using LinearAlgebra

struct BondableAtom
    mass::Float64
    σ::Float64
    ϵ::Float64
    partners::Set{Int}
end

struct BondableInteraction <: GeneralInteraction
    nl_only::Bool
    prob_formation::Float64
    prob_break::Float64
    dist_formation::Float64
    b0::Float64
    kb::Float64
end

function Molly.force(inter::BondableInteraction,
                        coord_i,
                        coord_j,
                        atom_i,
                        atom_j,
                        box_size)
    # Break bonds randomly
    if j in atom_i.partners
        if rand() < inter.prob_break
            delete!(atom_i.partners, j)
            delete!(atom_j.partners, i)
        end
    # Make bonds between close atoms randomly
    elseif r2 < inter.b0 * inter.dist_formation && rand() < inter.prob_formation
        push!(atom_i.partners, j)
        push!(atom_j.partners, i)
    end
    # Apply the force of a harmonic bond
    if j in atom_i.partners
        dr = vector(coord_i, coord_j, box_size)
        r2 = sum(abs2, dr)
        c = inter.kb * (norm(dr) - inter.b0)
        fdr = -c * normalize(dr)
        return fdr
    else
        return zero(coord_i)
    end
end

struct BondLogger <: Logger
    n_steps::Int
    bonds::Vector{BitVector}
end

function Molly.log_property!(logger::BondLogger, s, step_n)
    if step_n % logger.n_steps == 0
        bonds = BitVector()
        for i in 1:length(s.coords)
            for j in 1:(i - 1)
                push!(bonds, j in s.atoms[i].partners)
            end
        end
        push!(logger.bonds, bonds)
    end
end

temperature = 0.01
timestep = 0.02
box_size = 10.0
n_steps = 2_000
n_atoms = 200

atoms = [BondableAtom(1.0, 0.1, 0.02, Set([])) for i in 1:n_atoms]
coords = [box_size .* rand(SVector{2}) for i in 1:n_atoms]
velocities = [velocity(1.0, temperature; dims=2) for i in 1:n_atoms]
general_inters = (SoftSphere(true), BondableInteraction(true, 0.1, 0.1, 1.1, 0.1, 2.0))

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    general_inters=general_inters,
    coords=coords,
    velocities=velocities,
    temperature=temperature,
    box_size=box_size,
    neighbor_finder=DistanceNeighborFinder(trues(n_atoms, n_atoms), 10, 2.0),
    thermostat=AndersenThermostat(5.0),
    loggers=Dict("coords" => CoordinateLogger(20; dims=2),
                    "bonds" => BondLogger(20, [])),
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s)

connections = Tuple{Int, Int}[]
for i in 1:length(s.coords)
    for j in 1:(i - 1)
        push!(connections, (i, j))
    end
end

visualize(s.loggers["coords"],
            box_size,
            "sim_mutbond.gif",
            connections=connections,
            connection_frames=s.loggers["bonds"].bonds)
```
![Mutable bond simulation](images/sim_mutbond.gif)
