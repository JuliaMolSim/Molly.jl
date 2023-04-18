# Development documentation

## Running tests

The tests will automatically include multithreading and/or GPU tests if multiple threads and/or a CUDA-enabled GPU are available.
`test/runtests.jl` does not include all the tests, see the test directory for more, though these extra tests do not need to be run for every change.
Various environmental variables can be set to modify the tests:
- `VISTESTS` determines whether to run the [GLMakie.jl](https://github.com/JuliaPlots/Makie.jl) plotting tests which will error on remote systems where a display is not available, default `VISTESTS=1`.
- `GPUTESTS` determines whether to run the GPU tests, default `GPUTESTS=1`.
- `DEVICE` determines which GPU to run the GPU tests on, default `DEVICE=0`.
- `GROUP` can be used to run a subset of the tests, options `All`/`Protein`/`Zygote`/`NotZygote`, default `GROUP=All`.
The CI run does not carry out all tests - for example the GPU tests are not run - and this is reflected in the code coverage.

## Benchmarks

The `benchmark` directory contains some benchmarks for the package.
