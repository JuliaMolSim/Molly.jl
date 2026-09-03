# Molly API

The API reference can be found here.

Molly re-exports [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [Unitful.jl](https://github.com/PainterQubits/Unitful.jl), making the likes of `SVector` and `1.0u"nm"` available when you call `using Molly`.

Package extensions are used in order to reduce the number of dependencies:
- To use [`visualize`](@ref), call `using GLMakie`.
- To use [`ASECalculator`](@ref), call `using PythonCall`.
- To use [`rdf`](@ref), call `using KernelDensity`.
- To use [`ANIPotential`](@ref), call `using Lux, HDF5`.
- [`AllegroPotential`](@ref), a native equivariant (Allegro-style) neural network potential, computes energy and analytic forces on the CPU (`using Lux, HDF5`). It is built on O(3)-equivariant primitives (real spherical harmonics, Clebsch-Gordan tensor products, equivariant linear layers) that live in core Molly and are pinned to e3nn's conventions; these and the forces are validated by `test/equivariant.jl` and `test/allegro_potentials.jl`. GPU kernels and a weights artifact are follow-ups.

## Exported names

```@index
Order = [:module, :type, :constant, :function, :macro]
```

## Docstrings

```@autodocs
Modules = [Molly]
Private = false
Order = [:module, :type, :constant, :function, :macro]
```
