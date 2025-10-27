# Related software

There are many mature packages for molecular simulation.
Of particular note here are [OpenMM](https://openmm.org) and [GROMACS](https://www.gromacs.org), both of which influenced the implementation of Molly.
Molly can be thought of as similar to OpenMM in that it exposes simulation internals in a high-level language, though it is written in one language all the way down rather than using multiple device-specific kernels.
It also aims to be differentiable and work just as well with non-molecular physical simulations, though how much this impacts the ability to reach high simulation speeds remains to be seen.

For differentiable simulations there are a number of related packages:
- [Jax, M.D.](https://github.com/google/jax-md)
- [TorchMD](https://github.com/torchmd/torchmd)
- [mdgrad](https://github.com/torchmd/mdgrad)
- [DMFF](https://github.com/deepmodeling/DMFF)
- [Time Machine](https://github.com/proteneer/timemachine)
- [DiffTaichi](https://github.com/taichi-dev/difftaichi)
- [DIMOS](https://github.com/nec-research/DIMOS)

In Julia there are a number of packages related to atomic simulation, some of which are involved with the [JuliaMolSim](https://juliamolsim.github.io) organisation:
- [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl)
- [JuLIP.jl](https://github.com/JuliaMolSim/JuLIP.jl)
- [CellListMap.jl](https://github.com/m3g/CellListMap.jl)
- [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl)
- [ACE.jl](https://github.com/ACEsuit/ACE.jl)
- [AtomicGraphNets.jl](https://github.com/Chemellia/AtomicGraphNets.jl)
- [InteratomicPotentials.jl](https://github.com/cesmix-mit/InteratomicPotentials.jl), [Atomistic.jl](https://github.com/cesmix-mit/Atomistic.jl) and [PotentialLearning.jl](https://github.com/cesmix-mit/PotentialLearning.jl) from the CESMIX project at MIT
- [NBodySimulator.jl](https://github.com/SciML/NBodySimulator.jl), [DiffEqPhysics.jl](https://github.com/SciML/DiffEqPhysics.jl) and the [SciML](https://sciml.ai) ecosystem more broadly
