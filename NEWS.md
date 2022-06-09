# Molly.jl release notes

## v0.10.3 - Jun 2022

- `place_atoms` now checks for sensible inputs and terminates after a certain number of failed attempts.
- A bug in precompilation is fixed.

## v0.10.2 - May 2022

- `ForceLogger` is added.
- The `parallel` keyword argument is now available in the `log_property!` function.
- More examples are added to the documentation.
- A bug in the `Mie` potential energy is fixed.

## v0.10.1 - May 2022

- A bug in the `Gravity` force and potential energy is fixed.

## v0.10.0 - Apr 2022

- Loggers now also run before the first simulation step, i.e. at step 0, allowing the starting state to be recorded.
- `inject_gradients` now returns general interactions in addition to atoms, pairwise interactions and specific interaction lists.
- Steepest descent energy minimization is added via `SteepestDescentMinimizer`.
- GPU support is added for `potential_energy`.
- The `radius_gyration` function is added.
- A kappa value for ionic screening may be given to `ImplicitSolventOBC` and `ImplicitSolventGBN2`.
- Improvements are made to simulation setup such as allowing multiple macromolecular chains.
- A random number generator can now be passed to `Langevin`, allowing reproducible simulations.
- Gradients through the GB-Neck2 interaction are made to work on the GPU.
- Bugs in `StormerVerlet` are fixed.
- The possibility of a NaN value for the `HarmonicAngle` force when the angle is π is fixed.
- A bug causing `random_velocities` to run slowly is fixed.

## v0.9.0 - Mar 2022

- The arguments to `forces` and `accelerations` are made consistent across implementations.
- Centre of mass motion is removed by default during simulation using `remove_CM_motion!`.
- Coordinates are centred in the simulation box by default during setup.
- The `Langevin` integrator and `Verlet` integrator are added.
- The `MorseBond` potential is added.
- The GB-Neck2 implicit solvent model is added via `ImplicitSolventGBN2`.
- The `CubicSplineCutoff` is added.
- The `rmsd` function is added.
- The AtomsBase.jl interface is made more complete.
- The progress bar is removed from simulations.
- The out-of-place neighbor list type `NeighborListVec` is changed.

## v0.8.0 - Feb 2022

- General interactions are renamed to pairwise interactions throughout to better reflect their nature. The abstract type is now `PairwiseInteraction` and the keyword argument to `System` is now `pairwise_inters`. General interaction now refers to a new type of interaction that takes in the whole system and returns forces for all atoms, allowing interactions such as neural network potentials acting on the whole system. This is available via the keyword argument `general_inters` to `System`.
- Implicit solvent models are added via the `ImplicitSolventOBC` general interaction type and the `implicit_solvent` keyword argument when setting up a `System` from a file. The Onufriev-Bashford-Case GBSA model with parameter sets I and II is provided.
- `charge` is added to access the partial charge of an `Atom`.
- The `box_size` keyword argument may be given when setting up a `System` from a file.
- A bug in `KineticEnergyLogger` is fixed.

## v0.7.0 - Jan 2022

- The `force` and `potential_energy` functions for general interactions now take the vector between atom i and atom j as an argument in order to save on computation.
- Differentiable simulations are made faster and more memory-efficient.
- The AtomsBase.jl interface is updated to v0.2 of AtomsBase.jl.
- `extract_parameters` and `inject_gradients` are added to assist in taking gradients through simulations.
- `bond_angle` and `torsion_angle` are added.
- `random_velocities` is added.
- A `solute` field is added to `Atom` allowing solute-solvent weighting in interactions. This is added to the `LennardJones` interaction.
- A `proper` field is added to `PeriodicTorsion`.
- The float type is added as a type parameter to `System`. `float_type` and `is_gpu_diff_safe` are added to access the type parameters of a `System`.
- A `types` field is added to types such as `InteractionList2Atoms` to record interaction types.
- `find_neighbors` can now be given just the system as an argument.
- Visualisation is updated to use GLMakie.jl v0.5.
- Bugs in velocity generation and temperature calculation with no units are fixed.

## v0.6.0 - Jan 2022

- Differentiable simulation works with Zygote reverse and forward mode AD on both CPU and GPU. General and specific interactions are supported along with neighbor lists. It does not currently work with units, user-defined types and some components of the package.
- Significant API changes are made including a number of functions being renamed, thermostats being renamed to couplers and the removal of some types.
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
- Support for Julia versions before 1.7 is dropped.

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
