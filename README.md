# Molly.jl

[![Build status](https://github.com/JuliaMolSim/Molly.jl/workflows/CI/badge.svg)](https://github.com/JuliaMolSim/Molly.jl/actions)
[![Coverage status](https://codecov.io/gh/JuliaMolSim/Molly.jl/branch/master/graph/badge.svg?token=RD9XF0W90L)](https://codecov.io/gh/JuliaMolSim/Molly.jl)
[![Latest release](https://img.shields.io/github/release/JuliaMolSim/Molly.jl.svg)](https://github.com/JuliaMolSim/Molly.jl/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/JuliaMolSim/Molly.jl/blob/master/LICENSE.md)
[![Documentation stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMolSim.github.io/Molly.jl/stable)
[![Documentation dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaMolSim.github.io/Molly.jl/dev)

Much of science can be explained by the movement and interaction of molecules.
Molecular dynamics (MD) is a computational technique used to explore these phenomena, from noble gases to biological macromolecules.
Molly.jl is a pure Julia package for MD, and for the simulation of physical systems more broadly.

At the minute the package is a proof of concept for MD in Julia.
**It is not production ready.**
It can simulate a system of atoms with arbitrary interactions as defined by the user.
Implemented features include:
- Interface to allow definition of new forces, simulators, thermostats, neighbor finders, loggers etc.
- Read in OpenMM force field files and coordinate files supported by [Chemfiles.jl](https://github.com/chemfiles/Chemfiles.jl). There is also some support for Gromacs files.
- Non-bonded interactions - Lennard-Jones Van der Waals/repulsion force, electrostatic Coulomb potential, gravitational potential, soft sphere potential, Mie potential.
- Bonded interactions - covalent bonds, bond angles, torsion angles.
- Andersen thermostat.
- Velocity Verlet and velocity-free Verlet integration.
- Explicit solvent.
- Periodic boundary conditions in a cubic box.
- Neighbor list to speed up calculation of non-bonded forces.
- Automatic multithreading.
- [Unitful.jl](https://github.com/PainterQubits/Unitful.jl) compatibility so numbers have physical meaning.
- GPU acceleration on CUDA-enabled devices.
- Run with Float64 or Float32.
- Some analysis functions, e.g. RDF.
- Physical agent-based modelling.
- Visualise simulations as animations.
- Differentiable molecular simulation on an experimental branch - see the [relevant docs](https://juliamolsim.github.io/Molly.jl/dev/differentiable).

Features not yet implemented include:
- Energy minimisation.
- Other temperature or pressure coupling methods.
- Protein preparation - solvent box, add hydrogens etc.
- Trajectory/topology file format readers/writers.
- Quantum mechanical modelling.
- High test coverage.

## Installation

[Julia](https://julialang.org/downloads) is required, with Julia v1.6 or later required to get the latest version of Molly.
Install Molly from the Julia REPL.
Enter the package mode by pressing `]` and run `add Molly`.

## Usage

Some examples are given here, see [the documentation](https://juliamolsim.github.io/Molly.jl/stable/docs) for more on how to use the package.

Simulation of a Lennard-Jones gas:
```julia
using Molly

n_atoms = 100
box_size = SVector(2.0, 2.0, 2.0)u"nm"
temp = 298u"K"
atom_mass = 10.0u"u"

atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
coords = placeatoms(n_atoms, box_size, 0.3u"nm")
velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]
general_inters = (LennardJones(),)

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    general_inters=general_inters,
    coords=coords,
    velocities=velocities,
    box_size=box_size,
    thermostat=AndersenThermostat(temp, 1.0u"ps"),
    loggers=Dict("temp" => TemperatureLogger(100)),
    timestep=0.002u"ps",
    n_steps=10_000,
)

simulate!(s)
```

Simulation of a protein:
```julia
using Molly

timestep = 0.0002u"ps"
temp = 298u"K"
n_steps = 5_000

atoms, atoms_data, specific_inter_lists, general_inters, neighbor_finder, coords, box_size = readinputs(
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top"),
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"))

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    atoms_data=atoms_data,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=coords,
    velocities=[velocity(a.mass, temp) for a in atoms],
    box_size=box_size,
    neighbor_finder=neighbor_finder,
    thermostat=AndersenThermostat(temp, 1.0u"ps"),
    loggers=Dict("temp" => TemperatureLogger(10),
                    "writer" => StructureWriter(10, "traj_5XER_1ps.pdb")),
    timestep=timestep,
    n_steps=n_steps,
)

simulate!(s)
```

The above 1 ps simulation looks something like this when you view it in [VMD](https://www.ks.uiuc.edu/Research/vmd):
![MD simulation](https://github.com/JuliaMolSim/Molly.jl/raw/master/data/5XER/sim_1ps.gif)

## Contributing

Contributions are very welcome - see the [roadmap issue](https://github.com/JuliaMolSim/Molly.jl/issues/2) for more.
