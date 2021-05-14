# Molly documentation

This documentation will first introduce the main features of the package with some examples, then will give details on each component of a simulation.
There are further examples in the [Molly examples](@ref) section.
For more information on specific types or functions, see the [Molly API](@ref) section or call `?function_name` in Julia.
The [Differentiable simulation with Molly](@ref) section describes taking gradients through simulations.

Molly takes a modular approach to molecular simulation.
To run a simulation you create a [`Simulation`](@ref) object and call [`simulate!`](@ref) on it.
The different components of the simulation can be used as defined by the package, or you can define your own versions.
An important principle of the package is that your custom components, particularly force functions, should be easy to define and just as performant as the built-in versions.

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

temp = 100 # K
velocities = [velocity(mass, temp) for i in 1:n_atoms]
```
We store the coordinates and velocities as [static arrays](https://github.com/JuliaArrays/StaticArrays.jl) for performance.
They can be of any number of dimensions and of any number type, e.g. `Float64` or `Float32`.
Now we can define our general interactions, i.e. those between most or all atoms.
Because we have defined the relevant parameters for the atoms, we can use the built-in Lennard Jones type.
```julia
general_inters = (LennardJones(),)
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
    temperature=temp,
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

## GPU acceleration

To run simulations on the GPU you will need to have a CUDA-compatible device and to have [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) installed.
Simulation setup is similar to above, but with the coordinates, velocities and atoms moved to the GPU.
Currently, running on the GPU requires using the [`AtomMin`](@ref) type, which is `isbits`, or another `isbits` type for the atoms.
Neighbor lists and thermostats are not currently implemented for the GPU.
This example also shows setting up a simulation to run with `Float32`, which is a good idea for GPUs.
```julia
using Molly
using CUDA

n_atoms = 100
mass = 10.0f0
box_size = 2.0f0
temp = 100.0f0
atoms = cu([AtomMin(mass=mass, σ=0.3f0, ϵ=0.2f0) for i in 1:n_atoms])
coords = cu([box_size .* SVector{3}(rand(Float32, 3)) for i in 1:n_atoms])
velocities = cu([velocity(Float32, mass, temp) for i in 1:n_atoms])

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    general_inters=(LennardJones(),),
    coords=coords,
    velocities=velocities,
    temperature=temp,
    box_size=box_size,
    thermostat=NoThermostat(),
    loggers=Dict("temp" => TemperatureLogger(Float32, 10),
                    "coords" => CoordinateLogger(Float32, 10)),
    timestep=0.002f0,
    n_steps=1_000
)

simulate!(s)
```

## Simulating diatomic molecules

If we want to define specific interactions between atoms, for example bonds, we can do.
Using the same atom definitions as before, let's set up the coordinates so that paired atoms are 1 Å apart.
```julia
coords = [box_size .* rand(SVector{3}) for i in 1:(n_atoms / 2)]
for i in 1:length(coords)
    push!(coords, coords[i] .+ [0.1, 0.0, 0.0])
end

velocities = [velocity(mass, temp) for i in 1:n_atoms]
```
Now we can use the built-in bond type to place a harmonic constraint between paired atoms.
The arguments are the indices of the two atoms in the bond, the equilibrium distance and the force constant.
```julia
bonds = [HarmonicBond(i, Int(i + n_atoms / 2), 0.1, 300_000.0) for i in 1:Int(n_atoms / 2)]

specific_inter_lists = (bonds,)
```
This time, we are also going to use a neighbor list to speed up the Lennard Jones calculation.
We can use the built-in distance neighbor finder.
The arguments are a 2D array of eligible interactions, the number of steps between each update and the cutoff to be classed as a neighbor.
```julia
neighbor_finder = DistanceNeighborFinder(trues(n_atoms, n_atoms), 10, 1.2)
```
Now we can simulate as before.
```julia
s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=(LennardJones(true),), # true means we are using the neighbor list for this interaction
    coords=coords,
    velocities=velocities,
    temperature=temp,
    box_size=box_size,
    neighbor_finder=neighbor_finder,
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

## Simulating gravity

Molly is geared primarily to molecular simulation, but can also be used to simulate other physical systems.
Let's set up a gravitational simulation.
This example also shows the use of `Float32` and a 2D simulation.
```julia
atoms = [Atom(mass=1.0f0), Atom(mass=1.0f0)]
coords = [SVector(0.3f0, 0.5f0), SVector(0.7f0, 0.5f0)]
velocities = [SVector(0.0f0, 1.0f0), SVector(0.0f0, -1.0f0)]
general_inters = (Gravity(false, 1.5f0),)

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    general_inters=general_inters,
    coords=coords,
    velocities=velocities,
    box_size=1.0f0,
    loggers=Dict("coords" => CoordinateLogger(Float32, 10, dims=2)),
    timestep=0.002f0,
    n_steps=2000
)

simulate!(s)
```
When we view the simulation we can use some extra options:
```julia
visualize(s.loggers["coords"], 1.0f0, "sim_gravity.gif",
            trails=4, framerate=15, color=[:orange, :lightgreen],
            markersize=0.05)
```
![Gravity simulation](images/sim_gravity.gif)

## Simulating a protein

Molly has a rudimentary parser of [Gromacs](http://www.gromacs.org) topology and coordinate files.
Data for a protein can be read into the same data structures as above and simulated in the same way.
Currently, the OPLS-AA forcefield is implemented.
Here a [`StructureWriter`](@ref) is used to write the trajectory as a PDB file.
```julia
atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top"),
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"))

temp = 298

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=coords,
    velocities=[velocity(a.mass, temp) for a in atoms],
    temperature=temp,
    box_size=box_size,
    neighbor_finder=DistanceNeighborFinder(nb_matrix, 10),
    thermostat=AndersenThermostat(1.0),
    loggers=Dict("temp" => TemperatureLogger(10),
                    "writer" => StructureWriter(10, "traj_5XER_1ps.pdb")),
    timestep=0.0002,
    n_steps=5_000
)

simulate!(s)
```

## Agent-based modelling

Agent-based modelling (ABM) is conceptually similar to molecular dynamics.
Julia has [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) for ABM, but Molly can also be used to simulate arbitrary agent-based systems in continuous space.
Here we simulate a toy SIR model for disease spread.
This example shows how atom properties can be mutable, i.e. change during the simulation, and includes custom forces and loggers (see below for more).
```julia
@enum Status susceptible infected recovered

# Custom atom type
mutable struct Person
    i::Int64
    status::Status
    mass::Float64
    σ::Float64
    ϵ::Float64
end

# Custom GeneralInteraction
struct SIRInteraction <: GeneralInteraction
    nl_only::Bool
    dist_infection::Float64
    prob_infection::Float64
    prob_recovery::Float64
end

# Custom force function
function Molly.force(inter::SIRInteraction,
                        coord_i,
                        coord_j,
                        atom_i,
                        atom_j,
                        box_size)
    if (atom_i.status == infected && atom_j.status == susceptible) ||
                (atom_i.status == susceptible && atom_j.status == infected)
        # Infect close people randomly
        dr = vector(coord_i, coord_j, box_size)
        r2 = sum(abs2, dr)
        if r2 < inter.dist_infection ^ 2 && rand() < inter.prob_infection
            atom_i.status = infected
            atom_j.status = infected
        end
    end
    # Workaround to obtain a self-interaction
    if atom_i.i == (atom_j.i + 1)
        # Recover randomly
        if atom_i.status == infected && rand() < inter.prob_recovery
            atom_i.status = recovered
        end
    end
    return zero(coord_i)
end

# Custom Logger
struct SIRLogger <: Logger
    n_steps::Int
    fracs_sir::Vector{Vector{Float64}}
end

# Custom logging function
function Molly.log_property!(logger::SIRLogger, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        counts_sir = [
            count(p -> p.status == susceptible, s.atoms),
            count(p -> p.status == infected   , s.atoms),
            count(p -> p.status == recovered  , s.atoms)
        ]
        push!(logger.fracs_sir, counts_sir ./ length(s.atoms))
    end
end

temp = 0.01
timestep = 0.02
box_size = 10.0
n_steps = 1_000
n_people = 500
n_starting = 2
atoms = [Person(i, i <= n_starting ? infected : susceptible, 1.0, 0.1, 0.02) for i in 1:n_people]
coords = [box_size .* rand(SVector{2}) for i in 1:n_people]
velocities = [velocity(1.0, temp, dims=2) for i in 1:n_people]
general_inters = (LennardJones = LennardJones(true), SIR = SIRInteraction(false, 0.5, 0.06, 0.01))

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    general_inters=general_inters,
    coords=coords,
    velocities=velocities,
    temperature=temp,
    box_size=box_size,
    neighbor_finder=DistanceNeighborFinder(trues(n_people, n_people), 10, 2.0),
    thermostat=AndersenThermostat(5.0),
    loggers=Dict("coords" => CoordinateLogger(10, dims=2),
                    "SIR" => SIRLogger(10, [])),
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s)

visualize(s.loggers["coords"], box_size, "sim_agent.gif")
```
![Agent simulation](images/sim_agent.gif)

We can use the logger to plot the fraction of people susceptible (blue), infected (orange) and recovered (green) over the course of the simulation:
![Fraction SIR](images/fraction_sir.png)

## Forces

Forces define how different parts of the system interact. The force on each particle in the system is derived from the potential corresponding to the interaction.
```math
\vec{F}_i = -\sum_j \frac{dV_{ij}(r_{ij})}{dr_{ij}}\frac{\vec{r}_{ij}}{r_{ij}}
```

In Molly there are two types of interactions.
[`GeneralInteraction`](@ref)s are present between all or most atoms, and account for example for non-bonded terms.
[`SpecificInteraction`](@ref)s are present between specific atoms, and account for example for bonded terms.

The available general interactions are:
- [`LennardJones`](@ref).
- [`SoftSphere`](@ref).
- [`Mie`](@ref).
- [`Coulomb`](@ref).
- [`Gravity`](@ref).

The available specific interactions are:
- [`HarmonicBond`](@ref).
- [`HarmonicAngle`](@ref).
- [`Torsion`](@ref).

To define your own [`GeneralInteraction`](@ref), first define the `struct`:
```julia
struct MyGeneralInter <: GeneralInteraction
    nl_only::Bool
    # Any other properties, e.g. constants for the interaction or cutoff parameters
end
```
The `nl_only` property is required and determines whether the neighbor list is used to omit distant atoms (`true`) or whether all atom pairs are always considered (`false`).
Next, you need to define the [`force`](@ref) function acting between a pair of atoms.
This has a set series of arguments.
For example:
```julia
function Molly.force(inter::MyGeneralInter,
                        coord_i,
                        coord_j,
                        atom_i,
                        atom_j,
                        box_size)
    dr = vector(coord_i, coord_j, box_size)

    # Replace this with your force calculation
    # A positive force causes the atoms to move apart
    f = 0.0

    # Obtain a vector for the force
    fdr = f * normalize(dr)
    return fdr
end
```
If you need to obtain the vector from atom `i` to atom `j`, use the [`vector`](@ref) function.
This gets the vector between the closest images of atoms `i` and `j` accounting for the periodic boundary conditions.
Atom properties can be accessed, e.g. `atom_i.σ`.
Typically the force function is where most computation time is spent during the simulation, so consider optimising this function if you want high performance.

To use your custom force, add it to the list of general interactions:
```julia
general_inters = (MyGeneralInter(true),)
```
Then create and run a [`Simulation`](@ref) as above.
Note that you can also use named tuples instead of tuples if you want to access interactions by name:
```julia
general_inters = (MyGeneralInter = MyGeneralInter(true),)
```
For performance reasons it is best to [avoid containers with abstract type parameters](https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-abstract-container-1), such as `Vector{GeneralInteraction}`.

To define your own [`SpecificInteraction`](@ref), first define the `struct`:
```julia
struct MySpecificInter <: SpecificInteraction
    # Any number of atoms involved in the interaction
    i::Int
    j::Int
    # Any other properties, e.g. a bond distance corresponding to the energy minimum
end
```
Next, you need to define the [`force`](@ref) function.
For example:
```julia
function Molly.force(inter::MySpecificInter, coords, s::Simulation)
    dr = vector(coords[inter.i], coords[inter.j], s.box_size)

    # Replace this with your force calculation
    # A positive force causes the atoms to move apart
    f = 0.0

    fdr = f * normalize(dr)
    return [inter.i, inter.j], [-fdr, fdr]
end
```
The return values are a list of the atom indices and a list of the vector forces.
The example here is between two atoms but can be adapted for any number of atoms.
To use your custom force, add it to the specific interaction lists:
```julia
specific_inter_lists = ([MySpecificInter(1, 2), MySpecificInter(3, 4)],)
```
Specific interactions are always run on the CPU (with the results moved to the GPU if required), which is why we can index into `coords` and access `s` without harming performance.

## Cutoffs

The total potential energy of a system is given as a sum of the individual inter-particle potentials
```math
V(\vec{r}_1, \dotsc, \vec{r}_N) = \sum_{i<j}V_{ij}(r_{ij})
```

The forces acting on the particles are given by
```math
\vec{F}_i = -\sum_j \frac{dV_{ij}(r_{ij})}{dr_{ij}}\frac{\vec{r}_{ij}}{r_{ij}}
```

In the case of the Lennard-Jones potential, the inter-particle potential is given by
```math
V_{ij}(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]
```
and the forces are given by
```math
\begin{aligned}
\vec{F}_i &= 24\varepsilon_{ij} \left(2\frac{\sigma_{ij}^{12}}{r_{ij}^{13}} - \frac{\sigma_{ij}^6}{r_{ij}^{7}}\right) \frac{\vec{r}_{ij}}{r_{ij}} \\
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}^{6}}{r_{ij}^{6}}\right)^2 -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \vec{r}_{ij}
\end{aligned}
```

As the potential, and thus also the force decreases rapidly with the distance, in almost every implementation of the Lennard-Jones force calculation there is a cutoff radius beyond which the force is set to 0.

While this sounds like a very sensible approach, it introduces a discontinuity in the force function and it requires us to also modify the potential, as beyond the cutoff radius the force would be 0, but the derivative of the unmodified potential is not. One way to truncate the potential is to shift the potential by its cutoff value.
```math
\begin{aligned}
\vec{F}_{SP}(\vec{r}) &= \begin{cases}
\vec{F}(\vec{r}), r < r_c \\
0, r > r_c
\end{cases} \\
V_{SP}(r) &= \begin{cases}
V(r) - V(r_c), r \le r_c \\
0, r > r_c
\end{cases}
\end{aligned}
```

This way the potential function is continuous and the relation between forces and potentials is satisfied. This truncation method is called shifted potential cutoff.

Another option is to shift the force in order to make it continuous
```math
\begin{aligned}
F_{SF}(r) &= \begin{cases}
F(r) - F(r_c), r \le r_c \\
0, r > r_c
\end{cases} \\
V_{SF}(r) &= \begin{cases}
V(r) - (r-r_c) V'(r_c) - V(r_c), r \le r_c \\
0, r > r_c
\end{cases}
\end{aligned}
```
This requires a more complicated change in the potential in order to satisfy the relation between them. This method is called the shifted force cutoff. The continuity of the force is desirable as it may give better energy conservation properties as shown in [Toxvaerd 2011](http://aip.scitation.org/doi/10.1063/1.3558787).

There are also more complicated truncation methods that interpolate between the original potential and 0, but we will consider those two for the moment.

The truncation approximations that we use can significantly alter the qualitative features of the simulation as shown in many articles in the molecular dynamics literature ([Fitzner 2017](https://aip.scitation.org/doi/full/10.1063/1.4997698), [van der Spoel 2006](https://pubs.acs.org/doi/10.1021/ct0502256) and others).

### Implementation

Since the truncation algorithm is independent of the interaction for which is used, each interaction is defined without including cutoffs.
The corresponding interaction `struct` has a `cutoff` field which is then used via dispatch to apply the chosen cutoff.

## Simulators

Simulators define what type of simulation is run.
This could be anything from a simple energy minimisation to complicated replica exchange MD.
The available simulators are:
- [`VelocityVerlet`](@ref).
- [`VelocityFreeVerlet`](@ref).

To define your own [`Simulator`](@ref), first define the `struct`:
```julia
struct MySimulator <: Simulator
    # Any properties, e.g. an implicit solvent friction constant
end
```
Then, define the function that carries out the simulation.
This example shows some of the helper functions you can use:
```julia
function Molly.simulate!(s::Simulation,
                            simulator::MySimulator,
                            n_steps::Integer;
                            parallel::Bool=true)
    # Find neighbors like this
    neighbors = find_neighbors(s, nothing, s.neighbor_finder, 0,
                                    parallel=parallel)

    # Show a progress bar like this, if you have imported ProgressMeter
    @showprogress for step_n in 1:n_steps
        # Apply the loggers like this
        for logger in values(s.loggers)
            log_property!(logger, s, step_n)
        end

        # Calculate accelerations like this
        accels_t = accelerations(s, neighbors, parallel=parallel)

        # Ensure coordinates stay within the simulation box like this
        for i in 1:length(s.coords)
            s.coords[i] = adjust_bounds.(s.coords[i], s.box_size)
        end

        # Apply the thermostat like this
        apply_thermostat!(s, s.thermostat)

        # Find new neighbors like this
        neighbors = find_neighbors(s, neighbors, s.neighbor_finder, step_n,
                                        parallel=parallel)

        # Increment the step counter like this
        s.n_steps_made[1] += 1
    end
    return s
end
```
To use your custom simulator, give it as the `simulator` argument when creating the [`Simulation`](@ref).

Under the hood there are two implementations of common simulators: an in-place version geared towards CPUs, and an out-of-place version geared towards GPUs and differentiable simulation.
You can define different versions of a simulator for in-place and out-of-place simulations by dispatching on `Simulation{false}` or `Simulation{true}` respectively.
This also applies to thermostats and neighbor lists.
You do not have to define two versions though: you may only intend to use the simulation one way, or the out-of-place version may be performant in all cases.
The above example is more similar to the in-place version; see the source code for an example of the out-of-place version.

The implementation to use is guessed when you call [`Simulation`](@ref) based on whether `coords` is a `CuArray`, but can be given explicitly with the `gpu_diff_safe` argument.

## Thermostats

Thermostats control the temperature over a simulation.
The available thermostats are:
- [`AndersenThermostat`](@ref).

To define your own [`Thermostat`](@ref), first define the `struct`:
```julia
struct MyThermostat <: Thermostat
    # Any properties, e.g. a coupling constant
end
```
Then, define the function that implements the thermostat every timestep:
```julia
function apply_thermostat!(s::Simulation, thermostat::MyThermostat)
    # Do something to the simulation, e.g. scale the velocities
    return s
end
```
The functions [`velocity`](@ref), [`maxwellboltzmann`](@ref) and [`temperature`](@ref) may be useful here.
To use your custom thermostat, give it as the `thermostat` argument when creating the [`Simulation`](@ref).

## Neighbor finders

Neighbor finders find close atoms periodically throughout the simulation, saving on computation time by allowing the force calculation between distant atoms to be omitted.
The available neighbor finders are:
- [`DistanceNeighborFinder`](@ref).
- [`TreeNeighborFinder`](@ref).

To define your own [`NeighborFinder`](@ref), first define the `struct`:
```julia
struct MyNeighborFinder <: NeighborFinder
    nb_matrix::BitArray{2}
    n_steps::Int
    # Any other properties, e.g. a distance cutoff
end
```
Examples of two useful properties are given here: a matrix indicating atom pairs eligible for non-bonded interactions, and a value determining how many timesteps occur between each evaluation of the neighbor finder.
Then, define the neighbor finding function that is called every step by the simulator:
```julia
function find_neighbors(s::Simulation,
                            current_neighbors,
                            nf::MyNeighborFinder,
                            step_n::Integer;
                            parallel::Bool=true)
    if step_n % nf.n_steps == 0
        neighbors = Tuple{Int, Int}[]
        # Add to neighbors
        return neighbors
    else
        return current_neighbors
    end
end
```
To use your custom neighbor finder, give it as the `neighbor_finder` argument when creating the [`Simulation`](@ref).

## Loggers

Loggers record properties of the simulation to allow monitoring and analysis.
The available loggers are:
- [`TemperatureLogger`](@ref).
- [`CoordinateLogger`](@ref).
- [`EnergyLogger`](@ref).
- [`StructureWriter`](@ref).

To define your own [`Logger`](@ref), first define the `struct`:
```julia
struct MyLogger <: Logger
    n_steps::Int
    # Any other properties, e.g. an Array to record values during the trajectory
end
```
Then, define the logging function that is called every step by the simulator:
```julia
function Molly.log_property!(logger::MyLogger, s::Simulation, step_n::Integer)
    if step_n % logger.n_steps == 0
        # Record some property or carry out some action
    end
end
```
The use of `n_steps` is optional and is an example of how to record a property every n steps through the simulation.
To use your custom logger, add it to the dictionary of loggers:
```julia
loggers = Dict("mylogger" => MyLogger(10))
```

## Analysis

Molly contains some tools for analysing the results of simulations.
The available analysis functions are:
- [`visualize`](@ref).
- [`rdf`](@ref).
- [`distances`](@ref).
- [`displacements`](@ref).

Julia is a language well-suited to implementing all kinds of analysis for molecular simulations.
