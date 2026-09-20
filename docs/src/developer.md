# Developer documentation

## Running tests

We use [ParallelTestRunner.jl](https://github.com/JuliaTesting/ParallelTestRunner.jl) to run the tests in parallel across multiple threads.
This allows different test groups to be run, for example `julia test/runtests.jl gradients` for the gradient tests or `julia test/runtests.jl \!gradients \!extra` for the non-gradient tests.
The tests will automatically include multithreading and/or GPU tests if multiple threads and/or a GPU are available.
Warnings appearing at the start of the test run due to unavailable backends is expected.
`test/runtests.jl` does not include all the tests, see the `test/extra` directory for more, though these extra tests do not need to be run for every change.
Various environmental variables can be set to modify the tests:
- `VISTESTS` determines whether to run the [GLMakie.jl](https://github.com/JuliaPlots/Makie.jl) plotting tests which will error on remote systems where a display is not available, default `VISTESTS=1`.
- `GPUTESTS` determines whether to run the GPU tests if a GPU is available, default `GPUTESTS=1`.
- `DEVICE` determines which GPU to run the GPU tests on, default `DEVICE=0`.
The CI run does not carry out all tests - for example the GPU tests are not run - and this is reflected in the code coverage.
Running the test file locally gives higher coverage.

## Periodic boundary conditions

Molly uses the minimum image convention when applying periodic boundary conditions, meaning that of all the periodic copies of an interacting atom, only the closest is considered.
This means that the cutoff distance should not be greater than half of any of the periodic box dimensions, otherwise interactions within the cutoff distance will be missed.

Molly generally keeps all atoms within the "main" periodic box, even when that means splitting molecules over the boundary.
This is different to some other software, where molecules are kept whole.
In practice this doesn't make too much difference since specific interactions like bonds use the nearest periodic image of an atom.
This could lead to issues if a different copy of the atom is the intended interacting atom, but for most molecular systems this is not a problem.
It does lead to some complexity in [`scale_coords!`](@ref), as molecules have to be made whole before scaling.
When writing out files, the default (`correction=:pbc`) is to move atoms such that molecules are whole.
The case where all atoms are in one periodic box can be accessed with `correction=:wrap`.
For more discussion, see the [OpenMM FAQs](https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions).

## Custom neighbor finders

To define your own neighbor finder, first define the `struct`:
```julia
struct MyNeighborFinder
    eligible::BitArray{2}
    special::BitArray{2}
    n_steps::Int
    # Any other properties, e.g. a distance cutoff
end
```
Examples of three useful properties are given here: a matrix indicating atom pairs eligible for pairwise interactions, a matrix indicating atoms in a special arrangement such as 1-4 bonding, and a value determining how many time steps occur between each evaluation of the neighbor finder.
Then, define the neighbor finding function that is called every step by the simulator:
```julia
function Molly.find_neighbors(sys,
                              nf::MyNeighborFinder,
                              current_neighbors=nothing,
                              step_n::Integer=0,
                              force_recompute::Bool=false;
                              n_threads::Integer=Threads.nthreads())
    if force_recompute || step_n % nf.n_steps == 0
        if isnothing(current_neighbors)
            neighbors = NeighborList()
        else
            neighbors = current_neighbors
        end
        empty!(neighbors)
        # Add to neighbors, for example
        push!(neighbors, (1, 2, false)) # atom i, atom j and whether they are in a special interaction
        return neighbors
    else
        return current_neighbors
    end
end
```
To use your custom neighbor finder, give it as the `neighbor_finder` argument when creating the [`System`](@ref).

## Exception types

Molly throws various standard exception types, especially `ArgumentError`s, but also has its own error types:

| Exception type                | Description                                                                                     |
| :---------------------------- | :---------------------------------------------------------------------------------------------- |
| `NaNSimulationError`          | A `NaN` is encountered during a simulation or setup                                             |
| `ForceFieldXMLError`          | An error reading a force field XML file, [see more here](@ref "Simulating a protein")           |
| `MissingResidueTemplateError` | A residue cannot be matched to a residue template, [see more here](@ref "Simulating a protein") |

## Fast math

On CUDA GPUs the pairwise force and energy kernels are compiled with fast math for `Float32` systems, where the loss in precision is not noticeable. `Float64` systems do not use fast math, since there is a noticeable loss in precision and that path is much slower anyway.

## Benchmarks

The `benchmark` directory contains some benchmarks for the package.
