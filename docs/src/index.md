# Molly.jl documentation

Molly takes a modular approach to molecular simulation.
To run a simulation you create a `Simulation` object and run `simulate!` on it.
The different components of the simulation can be used as defined by the package, or you can define your own versions.

## Simulating an ideal gas

Let's look at the simulation of an ideal gas to start with.
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
coords = [Coordinates(rand(3) .* box_size) for i in 1:n_atoms]

temperature = 298 # K
velocities = [Velocity(mass, temperature) for i in 1:n_atoms]
```
Now we can define our dictionary of general interactions, i.e. those between most or all atoms.
Because we have defined the relevant parameters, we can use the built-in Lennard Jones type.
```julia
general_inters =  Dict("LJ" => LennardJones())
```
Finally, we can define and run the simulation.
We use an Andersen thermostat to keep a constant temperature, and we log the temperature and coordinates every 100 steps.
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
    loggers=[TemperatureLogger(100), CoordinateLogger(100)],
    timestep=0.002, # ps
    n_steps=100_000
)

simulate!(s)
```
We can get a quick look at the simulation by plotting the coordinate logger.
```julia
using Plots
pyplot(leg=false)

@gif for coords in s.loggers[2]
    plot(s.loggers[2], box_size)
end
```
And can check the temperature by plotting the temperature logger.
```julia
plot(s.loggers[1])
```

## Simulating diatomic molecules

If we want to define specific interactions between atoms, for example bonds, we can do.
Using the same atom definitions as before, lets set up the coordinates so paired atoms are 1 Angstrom apart.
```julia
coords = Coordinates[]
for i in 1:(n_atoms / 2)
    c = rand(3) .* box_size
    push!(coords, Coordinates(c))
    push!(coords, Coordinates(c + [0.1, 0.0, 0.0]))
end

velocities = [Velocity(mass, temperature) for i in 1:n_atoms]
```
Now we can use the built-in bond type to place a harmonic constraint between paired atoms.
The arguments are the indices of the two atoms in the bond, the equilibrium distance and the force constant.
```julia
bonds = [Bond((i * 2) - 1, i * 2, 0.1, 300_000) for i in 1:(n_atoms / 2)]

specific_inter_lists = Dict("Bonds" => bonds)
```
This time, we are also going to use a neighbour list to speed up the Lennard Jones calculation.
We can use the built-in distance neighbour finder.
The arguments are the number of steps between each update and the cutoff in nm to be classed as a neighbour.
```julia
neighbour_finder = DistanceNeighbourFinder(10, 2.0)
```
Now we can simulate as before.
```julia
s = Simulation(
    simulator=VelocityVerlet()
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=Dict("LJ" => LennardJones(true)), # true means we are using the neighbour list for this interaction
    coords=coords,
    velocities=velocities,
    temperature=temperature,
    box_size=box_size,
    neighbour_finder=neighbour_finder,
    thermostat=AndersenThermostat(1.0),
    loggers=[TemperatureLogger(100), CoordinateLogger(100)],
    timestep=0.002,
    n_steps=100_000
)

simulate!(s)
```
This time when we view the trajectory we can add lines to show the bonds.
```julia
connections = [((i * 2) - 1, i * 2) for i in 1:Int(n_atoms / 2)]

@gif for coords in s.loggers[2]
    plot(s.loggers[2], box_size, connections=connections)
end
```

## Simulating a protein in the OPLS-AA forcefield
