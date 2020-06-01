# Molly documentation

*These docs are work in progress*

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
coords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]

temperature = 298 # K
velocities = [velocity(mass, temperature) for i in 1:n_atoms]
```
We store the coordinates and velocities as [static arrays](https://github.com/JuliaArrays/StaticArrays.jl) for performance.
They can be of any number of dimensions.
Now we can define our dictionary of general interactions, i.e. those between most or all atoms.
Because we have defined the relevant parameters for the atoms, we can use the built-in Lennard Jones type.
```julia
general_inters = Dict("LJ" => LennardJones())
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
    loggers=Dict("temp" => TemperatureLogger(100),
                    "coords" => CoordinateLogger(100)),
    timestep=0.002, # ps
    n_steps=100_000
)

simulate!(s)
```
By default the simulation is run in parallel on the [number of threads](https://docs.julialang.org/en/v1/manual/parallel-computing/#man-multithreading-1) available to Julia, but this can be turned off by giving the keyword argument `parallel=false` to `simulate!`.
We can get a quick look at the simulation by plotting the coordinate and temperature loggers (in the future ideally this will be one easy plot command using recipes, and may switch to use Makie.jl).
```julia
using Plots

coords = s.loggers["coords"].coords
temps = s.loggers["temp"].temperatures

splitcoords(coord) = [c[1] for c in coord], [c[2] for c in coord], [c[3] for c in coord]

@gif for (i, coord) in enumerate(coords)
    l = @layout [a b{0.7h}]

    cx, cy, cz = splitcoords(coord)
    p = scatter(cx, cy, cz,
        xlims=(0, box_size),
        ylims=(0, box_size),
        zlims=(0, box_size),
        layout=l,
        legend=false
    )

    plot!(p[2],
        temps[1:i],
        xlabel="Frame",
        ylabel="Temperature / K",
        xlims=(1, i),
        ylims=(0.0, maximum(temps[1:i])),
        legend=false
    )
end
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
bonds = [Bond(i, Int(i + n_atoms / 2), 0.1, 300_000) for i in 1:(n_atoms / 2)]

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
    loggers=Dict("temp" => TemperatureLogger(100),
                    "coords" => CoordinateLogger(100)),
    timestep=0.002,
    n_steps=100_000
)

simulate!(s)
```
This time when we view the trajectory we can add lines to show the bonds.
```julia
using LinearAlgebra

coords = s.loggers["coords"].coords
temps = s.loggers["temp"].temperatures

connections = [(i, Int(i + n_atoms / 2)) for i in 1:Int(n_atoms / 2)]

@gif for (i, coord) in enumerate(coords)
    l = @layout [a b{0.7h}]

    cx, cy, cz = splitcoords(coord)
    p = scatter(cx, cy, cz,
        xlims=(0, box_size),
        ylims=(0, box_size),
        zlims=(0, box_size),
        layout=l,
        legend=false
    )

    for (a1, a2) in connections
        if norm(coord[a1] - coord[a2]) < (box_size / 2)
            plot!(p[1],
                [cx[a1], cx[a2]],
                [cy[a1], cy[a2]],
                [cz[a1], cz[a2]],
                linecolor="lightblue"
            )
        end
    end

    plot!(p[2],
        temps[1:i],
        xlabel="Frame",
        ylabel="Temperature / K",
        xlims=(1, i),
        ylims=(0.0, maximum(temps[1:i])),
        legend=false
    )
end
```
![Diatomic simulation](images/sim_diatomic.gif)

## Simulating a protein in the OPLS-AA forcefield

*In progress*

## Defining your own forces

*In progress*
