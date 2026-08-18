# Molly API

The API reference can be found here.

Molly re-exports [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [Unitful.jl](https://github.com/PainterQubits/Unitful.jl), making the likes of `SVector` and `1.0u"nm"` available when you call `using Molly`.

Package extensions are used in order to reduce the number of dependencies:
- To use [`visualize`](@ref), call `using GLMakie`.
- To use [`ASECalculator`](@ref), call `using PythonCall`.
- To use [`rdf`](@ref), call `using KernelDensity`.
- To use [`ANIPotential`](@ref), call `using Lux, HDF5`.
- [`AllegroPotential`](@ref), a native equivariant (Allegro-style) neural network potential, is under active development. The O(3)-equivariant building blocks it needs (real spherical harmonics, Clebsch-Gordan tensor products, equivariant linear layers) and their analytic gradients live in core Molly and are validated by `test/equivariant.jl`; the model assembly and weight loading (which need `using Lux, HDF5`) land in a follow-up.

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
