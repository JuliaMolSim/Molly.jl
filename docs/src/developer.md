# Developer documentation

## Running tests

The tests will automatically include multithreading and/or GPU tests if multiple threads and/or a CUDA-enabled GPU are available.
Errors appearing at the start of the test run due to unavailable backends is expected.
`test/runtests.jl` does not include all the tests, see the test directory for more, though these extra tests do not need to be run for every change.
Various environmental variables can be set to modify the tests:
- `VISTESTS` determines whether to run the [GLMakie.jl](https://github.com/JuliaPlots/Makie.jl) plotting tests which will error on remote systems where a display is not available, default `VISTESTS=1`.
- `GPUTESTS` determines whether to run the GPU tests, default `GPUTESTS=1`.
- `DEVICE` determines which GPU to run the GPU tests on, default `DEVICE=0`.
- `GROUP` can be used to run a subset of the tests, options `All`/`Protein`/`Gradients`/`NotGradients`, default `GROUP=All`.
The CI run does not carry out all tests - for example the GPU tests are not run - and this is reflected in the code coverage.

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

## Benchmarks

The `benchmark` directory contains some benchmarks for the package.
