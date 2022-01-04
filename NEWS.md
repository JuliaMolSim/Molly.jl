# Molly.jl release notes

## v0.6.0 - Jan 2022

- Differentiable simulation works with Zygote reverse and forward mode on both CPU and GPU. General and specific interactions are supported along with neighbor lists. It does not currently work with units or generic types.
- Significant API changes are made including a number of functions renamed, thermostats renamed to couplers and the removal of some types.
- `Simulation` is renamed to `System` and the time step and coupling are passed to the simulator, which is passed to the `simulate!` function.
- `System` is a sub-type of `AbstractSystem` from AtomsBase.jl and the relevant interface is implemented, allowing interoperability with the wider ecosystem.
- Specific interactions are changed to store indices and parameters as part of types such as `InteractionList2Atoms`. Specific interaction force functions now return types such as `SpecificForce4Atoms`. Specific interactions can now run on the GPU.
- Some abstract types are removed. `NeighborFinder` is renamed to `AbstractNeighborFinder`.
- The `potential_energy` function arguments match the `force` function arguments.
- File reader setup functions are called using `System` and return a `System` directly.
- `find_neighbors!` is renamed to `find_neighbors` and returns the neighbors, which are no longer stored as part of the simulation.
- `VelocityFreeVerlet` is renamed to `StormerVerlet`.
- `RescaleThermostat` and `BerendsenThermostat` are added.
- `random_velocities!` and `velocity_autocorr` are added.
- `VelocityLogger`, `KineticEnergyLogger` and `PotentialEnergyLogger` are added.
- `DistanceVecNeighborFinder` is added for use on the GPU.
- Atomic charges are now dimensionless, i.e 1.0 is an atomic charge of +1.
- `HarmonicAngle` now works in 2D.

## v0.5.0 - Oct 2021

- Readers are added for OpenMM XML force field files and coordinate files supported by Chemfiles.jl. Forces, energies and the results of a short simulation exactly match the OpenMM reference implementation for a standard protein in the a99SB-ILDN force field.
- A cell neighbor list is added using CellListMap.jl.
- `CoulombReactionField` is added to calculate long-range electrostatic interactions.
- The `PeriodicTorsion` interaction is added and the previous Ryckaert-Bellemans `Torsion` is renamed to `RBTorsion`.
- Support for weighting non-bonded interactions between atoms in a 1-4 interaction is added.
- The box size is changed from one value to three, allowing a larger variety of periodic box shapes.
- Support for different mixing rules is added for the Lennard-Jones interaction, with the default being Lorentz-Bertelot mixing.
- A simple `DistanceCutoff` is added.
- Excluded residue names can now be defined for a `StructureWriter`.
- The `placediatomics` and `ustripvec` functions are added.
- The `AtomMin` type is removed, with `Atom` now being a bits type and `AtomData` used to store atom data.
- Visualisation now uses GLMakie.jl.
- `adjust_bounds` is renamed to `wrapcoords`.

## v0.4.0 - Sep 2021

- Unitful.jl support is added and recommended for use, meaning numbers have physical meaning and many errors are caught. More type parameters have been added to various types to allow this. It is still possible to run simulations without units by specifying the `force_unit` and `energy_unit` arguments to `Simulation`.
- Interaction constructors with keyword arguments are added or improved.
- The maximum force for non-bonded interactions is removed.

## v0.3.0 - May 2021

- The spelling of "neighbour" is changed to "neighbor" throughout the package. This affects `NoNeighborFinder`, `DistanceNeighborFinder`, `TreeNeighborFinder`, `find_neighbors!` and the `neighbor_finder` argument to `Simulation`.
- Torsion angle force calculation is improved.
- A bug in force calculation for specific interactions is fixed.
- Support for Julia versions before 1.6 is dropped.

## v0.2.2 - Feb 2021

- The `TreeNeighbourFinder` is added for faster distance neighbour finding using NearestNeighbors.jl.

## v0.2.1 - Dec 2020

- Documentation is added for cutoffs.
- Compatibility bounds are updated for various packages, including requiring CUDA.jl version 2.
- Support for Julia versions before 1.5 is dropped.

## v0.2.0 - Sep 2020

- Shifted potential and shifted force cutoff approaches for non-bonded interactions are introduced.
- An `EnergyLogger` is added and the potential energy for existing interactions defined.
- Simulations can now be run with the `CuArray` type from CUDA.jl, allowing GPU acceleration.
- The in-place `force!` functions are changed to out-of-place `force` functions with different arguments.
- The i-i self-interaction is no longer computed for `force` functions.
- The Mie potential is added.
- The list of neighbours is now stored in the `Simulation` type.
- Documentation is added for experimental differentiable molecular simulation.
- Optimisations are implemented throughout the package.
- Support for Julia versions before 1.3 is dropped.

## v0.1.0 - Jun 2020

Initial release of Molly.jl, a Julia package for molecular simulation.

Features:
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
