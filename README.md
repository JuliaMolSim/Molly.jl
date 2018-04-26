# Molly.jl

Much of science can be explained by the movement of molecules. Molecular dynamics (MD) is a computational technique used to explore the motions of molecules, especially biological macromolecules. Molly.jl is a pure Julia implementation of MD.

At the minute the package is a proof of concept for MD of proteins in Julia. It can read in pre-computed Gromacs topology and coordinates files with the OPLS-AA forcefield and run MD with given parameters. Implemented features include:
- Bonded interactions - covalent bonds, bond angles, dihedral angles.
- Non-bonded interactions - Lennard-Jones Van der Waals/repulsion force, electrostatic Coulomb potential.
- Verlet leapfrog integration.
- Explicit solvent.
- Neighbour list to speed up calculation of non-bonded forces.
- (Periodic boundary conditions.)

Features not yet implemented are:
- Speed. Seriously, it's not fast yet.
- Force fields other than OPLS-AA.
- Energy minimisation.
- Protein preparation - solvent box, add hydrogens etc.
- Trajectory/topology file format readers/writers.
- Trajectory analysis.
- Canonical/grand-canonical ensembles etc.
- Parallelisation.
- GPU compatibility.

## Usage

```julia
using Molly

max_starting_velocity = 0.1 # nm/ps
timestep = 0.0002 # ps
n_steps = 800

forcefield, molecule, coords, box_size = readinputs(
            Pkg.dir("Molly", "data", "5XER", "gmx_top_ff.top"),
            Pkg.dir("Molly", "data", "5XER", "gmx_coords.gro"))

s = Simulation(forcefield, molecule, coords, box_size,
            max_starting_velocity, timestep, n_steps)

writepdb("start.pdb", s.universe)
simulate!(s)
writepdb("end.pdb", s.universe)
```

## Video

The above simulation looks something like this when you output more PDB files and view it in VMD:

## Plans

I plan to work on this in my spare time. Contributions are very welcome but bear in mind that I will probably refactor significantly as the package develops.
