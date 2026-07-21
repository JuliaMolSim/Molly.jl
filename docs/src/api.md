# Molly API

The API reference can be found here.

Molly re-exports [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [Unitful.jl](https://github.com/PainterQubits/Unitful.jl), making the likes of `SVector` and `1.0u"nm"` available when you call `using Molly`.

Package extensions are used in order to reduce the number of dependencies:
- To use [`visualize`](@ref), call `using GLMakie`.
- To use [`ASECalculator`](@ref), call `using PythonCall`.
- To use [`rdf`](@ref), call `using KernelDensity`.
- To use [`ANIPotential`](@ref), call `using Lux, HDF5`. Energy and forces both work with just those two packages: forces use an analytic path that runs on CPU or GPU, built on KernelAbstractions (a core Molly dependency, so no extra `using` is needed).

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
