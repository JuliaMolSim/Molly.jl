# Molly.jl

[![Travis build status](https://travis-ci.org/jgreener64/Molly.jl.svg?branch=master)](https://travis-ci.org/jgreener64/Molly.jl)
[![AppVeyor build status](https://ci.appveyor.com/api/projects/status/8dl6lqavnhqigq4p?svg=true)](https://ci.appveyor.com/project/jgreener64/molly-jl)
[![Coverage Status](https://coveralls.io/repos/github/jgreener64/Molly.jl/badge.svg?branch=master)](https://coveralls.io/github/jgreener64/Molly.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://jgreener64.github.io/Molly.jl/dev)

Much of science can be explained by the movement and interaction of molecules.
Molecular dynamics (MD) is a computational technique used to explore these phenomena, particularly for biological macromolecules.
Molly.jl is a pure Julia implementation of MD.

At the minute the package is a proof of concept for MD in Julia.
**It is not production ready.**
It can simulate a system of atoms with arbitrary interactions as defined by the user.
It can also read in pre-computed Gromacs topology and coordinate files with the OPLS-AA forcefield and run MD on proteins with given parameters.
In theory it can do this for any regular protein, but in practice this in untested.
Implemented features include:
- Interface to allow definition of new forces, thermostats etc.
- Non-bonded interactions - Lennard-Jones Van der Waals/repulsion force, electrostatic Coulomb potential.
- Bonded interactions - covalent bonds, bond angles, dihedral angles.
- Andersen thermostat.
- Velocity Verlet integration.
- Explicit solvent.
- Periodic boundary conditions in a cubic box.
- Neighbour list to speed up calculation of non-bonded forces.

Features not yet implemented include:
- Speed. Seriously, it's not fast yet - ~20x slower than GROMACS by some rough calculations. For reference most of the computational time in MD is spent in the force calculation, and most of that in calculation of non-bonded forces.
- Protein force fields other than OPLS-AA.
- Water models.
- Energy minimisation.
- Other temperature or pressure coupling methods.
- Protein preparation - solvent box, add hydrogens etc.
- Trajectory/topology file format readers/writers.
- Trajectory analysis.
- Parallelisation.
- GPU compatibility.
- Unit tests.

## Installation

Julia v1.0 or later is required.
Install from the Julia REPL.
Enter the package mode by pressing `]` and run `add https://github.com/jgreener64/Molly.jl`.

## Usage

Some examples are given here, see [the documentation](https://jgreener64.github.io/Molly.jl/dev) for more.

Simulation of an ideal gas:
```julia
using Molly

n_atoms = 100
box_size = 2.0 # nm
temperature = 298 # K
mass = 10.0 # Relative atomic mass

atoms = [Atom(mass=mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms]
coords = [Coordinates(rand(3) .* box_size) for i in 1:n_atoms]
velocities = [Velocity(mass, temperature) for i in 1:n_atoms]
general_inters = Dict("LJ" => LennardJones())

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    general_inters=general_inters,
    coords=coords,
    velocities=velocities,
    temperature=temperature,
    box_size=box_size,
    thermostat=AndersenThermostat(1.0),
    loggers=[TemperatureLogger(100)],
    timestep=0.002, # ps
    n_steps=100_000
)

simulate!(s)
```

Simulation of a protein:
```julia
using Molly

timestep = 0.0002 # ps
temperature = 298 # K
n_steps = 5000

atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top"),
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"))

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=coords,
    velocities=[Velocity(a.mass, temperature) for a in atoms],
    temperature=temperature,
    box_size=box_size,
    neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),
    thermostat=AndersenThermostat(1.0),
    loggers=[TemperatureLogger(10), StructureWriter(10, "traj_5XER_1ps.pdb")],
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s)
```

The above 1 ps simulation looks something like this when you output view it in VMD:
![MD simulation](https://github.com/jgreener64/Molly.jl/raw/master/data/5XER/sim_1ps.gif)

## Plans

I plan to work on this in my spare time, but progress will be slow.
MD could provide a nice use case for Julia - I think a reasonably featured and performant MD program could be written in fewer than 1,000 lines of code for example.
Julia is also a well-suited language for trajectory analysis.

Contributions are very welcome - see the [roadmap issue](https://github.com/jgreener64/Molly.jl/issues/2) for more.
