# Molly.jl

[![Travis build status](https://travis-ci.org/JuliaMolSim/Molly.jl.svg?branch=master)](https://travis-ci.org/JuliaMolSim/Molly.jl)
[![AppVeyor build status](https://ci.appveyor.com/api/projects/status/fc9qjhs9pfema614?svg=true)](https://ci.appveyor.com/project/jgreener64/molly-jl-yaoyb)
[![Coverage status](https://coveralls.io/repos/github/JuliaMolSim/Molly.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaMolSim/Molly.jl?branch=master)
[![Latest release](https://img.shields.io/github/release/JuliaMolSim/Molly.jl.svg)](https://github.com/JuliaMolSim/Molly.jl/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/JuliaMolSim/Molly.jl/blob/master/LICENSE.md)
[![Documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMolSim.github.io/Molly.jl/stable)
[![Documentation latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMolSim.github.io/Molly.jl/latest)

Much of science can be explained by the movement and interaction of molecules.
Molecular dynamics (MD) is a computational technique used to explore these phenomena, from noble gases to biological macromolecules.
Molly.jl is a pure Julia package for MD, and for the simulation of physical systems more broadly.

At the minute the package is a proof of concept for MD in Julia.
**It is not production ready.**
It can simulate a system of atoms with arbitrary interactions as defined by the user.
Implemented features include:
- Interface to allow definition of new forces, simulators, thermostats, neighbour finders, loggers etc.
- Read in pre-computed Gromacs topology and coordinate files with the OPLS-AA forcefield and run MD on proteins with given parameters. In theory it can do this for any regular protein, but in practice this is untested.
- Non-bonded interactions - Lennard-Jones Van der Waals/repulsion force, electrostatic Coulomb potential, gravitational potential, soft sphere potential.
- Bonded interactions - covalent bonds, bond angles, torsion angles.
- Andersen thermostat.
- Velocity Verlet and velocity-free Verlet integration.
- Explicit solvent.
- Periodic boundary conditions in a cubic box.
- Neighbour list to speed up calculation of non-bonded forces.
- Automatic multithreading.
- Some analysis functions, e.g. RDF.
- Run with Float64 or Float32.
- Physical agent-based modelling.
- Visualise simulations as animations.
- Differentiable molecular simulation on an experimental branch - see the [relevant docs](https://juliamolsim.github.io/Molly.jl/latest/differentiable.html).

Features not yet implemented include:
- Protein force fields other than OPLS-AA.
- Water models.
- Energy minimisation.
- Other temperature or pressure coupling methods.
- Cell-based neighbour list.
- Protein preparation - solvent box, add hydrogens etc.
- Trajectory/topology file format readers/writers.
- Quantum mechanical modelling.
- GPU compatibility.
- High test coverage.

## Installation

Julia v1.0 or later is required.
Install from the Julia REPL.
Enter the package mode by pressing `]` and run `add Molly`.

## Usage

Some examples are given here, see [the documentation](https://JuliaMolSim.github.io/Molly.jl/stable) for more on how to use the package.

Simulation of a Lennard-Jones gas:
```julia
using Molly

n_atoms = 100
box_size = 2.0 # nm
temp = 298 # K
mass = 10.0 # Relative atomic mass

atoms = [Atom(mass=mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms]
coords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]
velocities = [velocity(mass, temp) for i in 1:n_atoms]
general_inters = (LennardJones(),)

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    general_inters=general_inters,
    coords=coords,
    velocities=velocities,
    temperature=temp,
    box_size=box_size,
    thermostat=AndersenThermostat(1.0),
    loggers=Dict("temp" => TemperatureLogger(100)),
    timestep=0.002, # ps
    n_steps=10_000
)

simulate!(s)
```

Simulation of a protein:
```julia
using Molly

timestep = 0.0002 # ps
temp = 298 # K
n_steps = 5_000

atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top"),
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"))

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=coords,
    velocities=[velocity(a.mass, temp) for a in atoms],
    temperature=temp,
    box_size=box_size,
    neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),
    thermostat=AndersenThermostat(1.0),
    loggers=Dict("temp" => TemperatureLogger(10),
                    "writer" => StructureWriter(10, "traj_5XER_1ps.pdb")),
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s)
```

The above 1 ps simulation looks something like this when you view it in [VMD](https://www.ks.uiuc.edu/Research/vmd):
![MD simulation](https://github.com/JuliaMolSim/Molly.jl/raw/master/data/5XER/sim_1ps.gif)

## Contributing

Contributions are very welcome - see the [roadmap issue](https://github.com/JuliaMolSim/Molly.jl/issues/2) for more.
