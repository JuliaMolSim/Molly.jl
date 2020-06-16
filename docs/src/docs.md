# Molly documentation

Molly takes a modular approach to molecular simulation.
To run a simulation you create a [`Simulation`](@ref) object and call [`simulate!`](@ref) on it.
The different components of the simulation can be used as defined by the package, or you can define your own versions.
An important principle of the package is that your custom components, particularly force functions, should be easy to define and just as performant as the in-built versions.

This documentation will first introduce the main features of the package with some examples, then will give details on each component of a simulation.
For more information on specific types or functions, see the [Molly API](@ref) section or call `?function_name` in Julia.

## Simulating a gas

Let's look at the simulation of a gas acting under the [Lennard-Jones](https://en.wikipedia.org/wiki/Lennard-Jones_potential) potential to start with.
First, we'll need some atoms with the relevant parameters defined.
```julia
using Molly

n_atoms = 100
mass = 10.0
atoms = [Atom(mass=mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms]
```
Next, we'll need some starting coordinates and velocities.
```julia
box_size = 2.0 # nm
coords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]

temperature = 100 # K
velocities = [velocity(mass, temperature) for i in 1:n_atoms]
```
We store the coordinates and velocities as [static arrays](https://github.com/JuliaArrays/StaticArrays.jl) for performance.
They can be of any number of dimensions and of any number type, e.g. `Float64` or `Float32`.
Now we can define our dictionary of general interactions, i.e. those between most or all atoms.
Because we have defined the relevant parameters for the atoms, we can use the built-in Lennard Jones type.
```julia
general_inters = Dict("LJ" => LennardJones())
```
Finally, we can define and run the simulation.
We use an Andersen thermostat to keep a constant temperature, and we log the temperature and coordinates every 10 steps.
```julia
s = Simulation(
    simulator=VelocityVerlet(), # Use velocity Verlet integration
    atoms=atoms,
    general_inters=general_inters,
    coords=coords,
    velocities=velocities,
    temperature=temperature,
    box_size=box_size,
    thermostat=AndersenThermostat(1.0), # Coupling constant of 1.0
    loggers=Dict("temp" => TemperatureLogger(10),
                    "coords" => CoordinateLogger(10)),
    timestep=0.002, # ps
    n_steps=1_000
)

simulate!(s)
```
By default the simulation is run in parallel on the [number of threads](https://docs.julialang.org/en/v1/manual/parallel-computing/#man-multithreading-1) available to Julia, but this can be turned off by giving the keyword argument `parallel=false` to [`simulate!`](@ref).
An animation of the stored coordinates using can be saved using [`visualize`](@ref), which is available when [Makie.jl](https://github.com/JuliaPlots/Makie.jl) is imported.
```julia
using Makie

visualize(s.loggers["coords"], box_size, "sim_lj.gif")
```
![LJ simulation](images/sim_lj.gif)

## Simulating diatomic molecules

If we want to define specific interactions between atoms, for example bonds, we can do.
Using the same atom definitions as before, let's set up the coordinates so that paired atoms are 1 Å apart.
```julia
coords = [box_size .* rand(SVector{3}) for i in 1:(n_atoms / 2)]
for i in 1:length(coords)
    push!(coords, coords[i] .+ [0.1, 0.0, 0.0])
end

velocities = [velocity(mass, temperature) for i in 1:n_atoms]
```
Now we can use the built-in bond type to place a harmonic constraint between paired atoms.
The arguments are the indices of the two atoms in the bond, the equilibrium distance and the force constant.
```julia
bonds = [HarmonicBond(i, Int(i + n_atoms / 2), 0.1, 300_000.0) for i in 1:Int(n_atoms / 2)]

specific_inter_lists = Dict("Bonds" => bonds)
```
This time, we are also going to use a neighbour list to speed up the Lennard Jones calculation.
We can use the built-in distance neighbour finder.
The arguments are a 2D array of eligible interactions, the number of steps between each update and the cutoff in nm to be classed as a neighbour.
```julia
neighbour_finder = DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 1.2)
```
Now we can simulate as before.
```julia
s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=Dict("LJ" => LennardJones(true)), # true means we are using the neighbour list for this interaction
    coords=coords,
    velocities=velocities,
    temperature=temperature,
    box_size=box_size,
    neighbour_finder=neighbour_finder,
    thermostat=AndersenThermostat(1.0),
    loggers=Dict("temp" => TemperatureLogger(10),
                    "coords" => CoordinateLogger(10)),
    timestep=0.002,
    n_steps=1_000
)

simulate!(s)
```
This time when we view the trajectory we can add lines to show the bonds.
```julia
visualize(s.loggers["coords"], box_size, "sim_diatomic.gif",
            connections=[(i, Int(i + n_atoms / 2)) for i in 1:Int(n_atoms / 2)],
            markersize=0.05, linewidth=5.0)
```
![Diatomic simulation](images/sim_diatomic.gif)

## Simulating a protein

Molly has a rudimentary parser of [Gromacs](http://www.gromacs.org) topology and coordinate files.
Data for a protein can be read into the same data structures as above and simulated in the same way.
Currently, the OPLS-AA forcefield is implemented.
```julia
atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top"),
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"))

temperature = 298

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=coords,
    velocities=[velocity(a.mass, temperature) for a in atoms],
    temperature=temperature,
    box_size=box_size,
    neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),
    thermostat=AndersenThermostat(1.0),
    loggers=Dict("temp" => TemperatureLogger(10),
                    "writer" => StructureWriter(10, "traj_5XER_1ps.pdb")),
    timestep=0.0002,
    n_steps=5_000
)

simulate!(s)
```

## Forces

Forces define how different parts of the system interact.
In Molly they are separated into two types.
[`GeneralInteraction`](@ref)s are present between all or most atoms, and account for example for non-bonded terms.
[`SpecificInteraction`](@ref)s are present between specific atoms, and account for example for bonded terms.

The available general interactions are:
- [`LennardJones`](@ref).
- [`Coulomb`](@ref).

The available specific interactions are:
- [`HarmonicBond`](@ref).
- [`HarmonicAngle`](@ref).
- [`Torsion`](@ref).

To define your own [`GeneralInteraction`](@ref), first define the `struct`:
```julia
struct MyGeneralInter <: GeneralInteraction
    nl_only::Bool
    # Any other properties
end
```
The `nl_only` property is required and determines whether the neighbour list is used to omit distant atoms (`true`) or whether all atom pairs are always considered (`false`).
Next, you need to define the [`force!`](@ref) function acting between a pair of atoms.
For example:
```julia
function force!(forces, inter::MyGeneralInter, s::Simulation, i::Integer, j::Integer)
    dr = vector(s.coords[i], s.coords[j], s.box_size)

    # Replace this with your force calculation
    # A positive force causes the atoms to move together
    f = 0.0

    fdr = f * normalize(dr)
    forces[i] -= fdr
    forces[j] += fdr
    return forces
end
```
If you need to obtain the vector from atom `i` to atom `j`, use the [`vector`](@ref) function.
This gets the vector between the closest images of atoms `i` and `j` accounting for the periodic boundary conditions.
The [`Simulation`](@ref) is available so atom properties or velocities can be accessed, e.g. `s.atoms[i].σ` or `s.velocities[i]`.
This form of the function can also be used to define three-atom interactions by looping a third variable `k` up to `j` in the [`force!`](@ref) function.
To use your custom force, add it to the dictionary of general interactions:
```julia
general_inters = Dict("MyGeneralInter" => MyGeneralInter(true))
```
Then create and run a [`Simulation`](@ref) as above.

To define your own [`SpecificInteraction`](@ref), first define the `struct`:
```julia
struct MySpecificInter <: SpecificInteraction
    # Any number of atoms involved in the interaction
    i::Int
    j::Int
    # Any other properties, e.g. a bond distance corresponding to the energy minimum
end
```
Next, you need to define the [`force!`](@ref) function.
For example:
```julia
function force!(forces, inter::MySpecificInter, s::Simulation)
    dr = vector(s.coords[inter.i], s.coords[inter.j], s.box_size)

    # Replace this with your force calculation
    # A positive force causes the atoms to move together
    f = 0.0

    fdr = f * normalize(dr)
    forces[inter.i] += fdr
    forces[inter.j] -= fdr
    return forces
end
```
The example here is between two atoms but can be adapted for any number of atoms.
To use your custom force, add it to the dictionary of specific interaction lists:
```julia
specific_inter_lists = Dict("MySpecificInter" => [MySpecificInter(1, 2), MySpecificInter(3, 4)])
```

## Simulators

Simulators define what type of simulation is run.
This could be anything from a simple energy minimisation to complicated replica exchange MD.
The available simulators are:
- [`VelocityVerlet`](@ref).
- [`VelocityFreeVerlet`](@ref).

## Thermostats

Thermostats control the temperature over a simulation.
The available thermostats are:
- [`AndersenThermostat`](@ref).

## Neighbour finders

Neighbour finders find close atoms periodically throughout the simulation, saving on computation time by allowing the force calculation between distance atoms to be omitted.
The available neighbour finders are:
- [`DistanceNeighbourFinder`](@ref).

## Loggers

Loggers record properties of the simulation to allow monitoring and analysis.
The available loggers are:
- [`TemperatureLogger`](@ref).
- [`CoordinateLogger`](@ref).
- [`StructureWriter`](@ref).

## Analysis

Molly contains some tools for analysing the results of simulations.
The available analysis functions are:
- [`visualize`](@ref).
- [`rdf`](@ref).
- [`distances`](@ref).
- [`displacements`](@ref).
