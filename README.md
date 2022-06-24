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
**It is not production ready**, though it can do some cool things and is under active development.
Implemented features include:
- Non-bonded interactions - Lennard-Jones Van der Waals/repulsion force, electrostatic Coulomb potential and reaction field, gravitational potential, soft sphere potential, Mie potential.
- Bonded interactions - covalent bonds, bond angles, torsion angles.
- Interface to allow definition of new interactions, simulators, thermostats, neighbor finders, loggers etc.
- Read in OpenMM force field files and coordinate files supported by [Chemfiles.jl](https://github.com/chemfiles/Chemfiles.jl). There is also some support for Gromacs files.
- Andersen, Berendsen and velocity rescaling thermostats.
- Verlet, velocity Verlet, Störmer-Verlet and Langevin integrators.
- Steepest descent energy minimization.
- Periodic boundary conditions in a cubic box.
- Various neighbor list implementations to speed up calculation of non-bonded forces.
- Implicit solvent GBSA methods.
- [Unitful.jl](https://github.com/PainterQubits/Unitful.jl) compatibility so numbers have physical meaning.
- Automatic multithreading.
- GPU acceleration on CUDA-enabled devices.
- Run with Float64 or Float32.
- Some analysis functions, e.g. RDF.
- Visualise simulations as animations.
- Physical agent-based modelling.
- Differentiable molecular simulation. This is a unique feature of the package and the focus of its current development.

Features not yet implemented include:
- Simulators such as REMD and metadynamics.
- Other temperature or pressure coupling methods.
- Protein preparation - solvent box, add hydrogens etc.
- Quantum mechanical modelling.
- Constrained bonds and angles.
- Domain decomposition algorithms.
- Alchemical free energy calculations.
- High test coverage.
- API stability.
- High GPU performance.

## Installation

[Julia](https://julialang.org/downloads) is required, with Julia v1.7 or later required to get the latest version of Molly.
Install Molly from the Julia REPL.
Enter the package mode by pressing `]` and run `add Molly`.

## Usage

Some examples are given here, see [the documentation](https://juliamolsim.github.io/Molly.jl/stable/docs) for more on how to use the package.

Simulation of a Lennard-Jones fluid:
```julia
using Molly

n_atoms = 100
boundary = CubicBoundary(2.0u"nm", 2.0u"nm", 2.0u"nm")
temp = 298.0u"K"
atom_mass = 10.0u"u"

atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
coords = place_atoms(n_atoms, boundary, 0.3u"nm")
velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]
pairwise_inters = (LennardJones(),)
simulator = VelocityVerlet(
    dt=0.002u"ps",
    coupling=AndersenThermostat(temp, 1.0u"ps"),
)

sys = System(
    atoms=atoms,
    pairwise_inters=pairwise_inters,
    coords=coords,
    velocities=velocities,
    boundary=boundary,
    loggers=(temp=TemperatureLogger(100)),
)

simulate!(sys, simulator, 10_000)
```

Simulation of a protein:
```julia
using Molly

sys = System(
    joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"),
    joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top");
    loggers=(
        temp = TemperatureLogger(10),
        writer = StructureWriter(10, "traj_5XER_1ps.pdb"),
    ),
)

temp = 298.0u"K"
random_velocities!(sys, temp)
simulator = VelocityVerlet(
    dt=0.0002u"ps",
    coupling=AndersenThermostat(temp, 1.0u"ps"),
)

simulate!(sys, simulator, 5_000)
```

The above 1 ps simulation looks something like this when you view it in [VMD](https://www.ks.uiuc.edu/Research/vmd):
![MD simulation](https://github.com/JuliaMolSim/Molly.jl/raw/master/data/5XER/sim_1ps.gif)

## Contributing

Contributions are very welcome - see the [roadmap issue](https://github.com/JuliaMolSim/Molly.jl/issues/2) for more.

Join the #molly channel on the [JuliaMolSim Slack](https://join.slack.com/t/juliamolsim/shared_invite/zt-19xb8u1u8-~Nua4TH~kj29NXO7mSTQHg) to discuss the usage and development of Molly.jl.
