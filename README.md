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

The package is described in [a talk](https://youtu.be/trapn-yIv8g?t=1889) at the [JuliaMolSim](https://juliamolsim.github.io) minisymposium at JuliaCon 2022, and also in [a more recent talk](https://www.youtube.com/live/NB7xUHQNox8?feature=share&t=9018) at Enzyme Conference 2023.
Implemented features include:
- Non-bonded interactions - Lennard-Jones Van der Waals/repulsion force, electrostatic Coulomb potential and reaction field, gravitational potential, soft sphere potential, Mie potential, Buckingham potential, soft core variants.
- Bonded interactions - harmonic and Morse bonds, bond angles, torsion angles, harmonic position restraints, FENE bonds.
- Interface to allow definition of new interactions, simulators, thermostats, neighbor finders, loggers etc.
- Read in OpenMM force field files and coordinate files supported by [Chemfiles.jl](https://github.com/chemfiles/Chemfiles.jl). There is also experimental support for Gromacs files.
- Verlet, velocity Verlet, Störmer-Verlet, flexible Langevin and Nosé-Hoover integrators.
- Andersen, Berendsen and velocity rescaling thermostats.
- Monte Carlo barostat.
- Steepest descent energy minimization.
- Replica exchange molecular dynamics.
- Monte Carlo simulation.
- Periodic, triclinic and infinite boundary conditions.
- Flexible loggers to track arbitrary properties throughout simulations.
- Cutoff algorithms for non-bonded interactions.
- Various neighbor list implementations to speed up the calculation of non-bonded forces, including the use of [CellListMap.jl](https://github.com/m3g/CellListMap.jl).
- Implicit solvent GBSA methods.
- [Unitful.jl](https://github.com/PainterQubits/Unitful.jl) compatibility so numbers have physical meaning.
- Set up crystal systems using [SimpleCrystals.jl](https://github.com/ejmeitz/SimpleCrystals.jl).
- Automatic multithreading.
- GPU acceleration on CUDA-enabled devices.
- Run with Float64, Float32 or other float types.
- Some analysis functions, e.g. RDF.
- Visualise simulations as animations with [Makie.jl](https://makie.juliaplots.org/stable).
- Differentiable molecular simulation. This is a unique feature of the package and the focus of its current development.

Features not yet implemented include:
- High GPU performance.
- Ewald or particle mesh Ewald summation.
- Constrained bonds and angles.
- Protein preparation - solvent box, add hydrogens etc.
- Simulators such as metadynamics.
- Quantum mechanical modelling.
- Domain decomposition algorithms.
- Alchemical free energy calculations.
- High test coverage.
- API stability.

## Installation

[Julia](https://julialang.org/downloads) is required, with Julia v1.7 or later required to get the latest version of Molly.
It is recommended to run on the current stable Julia release for the best performance.
Install Molly from the Julia REPL.
Enter the package mode by pressing `]` and run `add Molly`.

## Usage

Some examples are given here, see [the documentation](https://juliamolsim.github.io/Molly.jl/stable/docs) for more on how to use the package.

Simulation of a Lennard-Jones fluid:
```julia
using Molly

n_atoms = 100
boundary = CubicBoundary(2.0u"nm")
temp = 298.0u"K"
atom_mass = 10.0u"u"

atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]
pairwise_inters = (LennardJones(),)
simulator = VelocityVerlet(
    dt=0.002u"ps",
    coupling=AndersenThermostat(temp, 1.0u"ps"),
)

sys = System(
    atoms=atoms,
    coords=coords,
    boundary=boundary,
    velocities=velocities,
    pairwise_inters=pairwise_inters,
    loggers=(temp=TemperatureLogger(100),),
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
        temp=TemperatureLogger(10),
        writer=StructureWriter(10, "traj_5XER_1ps.pdb"),
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

Join the #molly channel on the [JuliaMolSim Zulip](https://juliamolsim.zulipchat.com/join/j5sqhiajbma5hw55hy6wtzv7) to discuss the usage and development of Molly.jl, or post on the [Julia Discourse](https://discourse.julialang.org).
