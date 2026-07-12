# Molly API

The API reference can be found here.

Molly re-exports [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [Unitful.jl](https://github.com/PainterQubits/Unitful.jl), making the likes of `SVector` and `1.0u"nm"` available when you call `using Molly`.

Package extensions are used in order to reduce the number of dependencies:
- To use [`visualize`](@ref), call `using GLMakie`.
- To use [`ASECalculator`](@ref), call `using PythonCall`.
- To use [`rdf`](@ref), call `using KernelDensity`.
- To use [`ANIPotential`](@ref), call `using Lux, HDF5` (and `using Enzyme` for AD forces); the on-device GPU functions [`compute_aevs_ka`](@ref) / [`compute_ani_energy_ka`](@ref) / [`compute_ani_forces_ka`](@ref) additionally need `using KernelAbstractions`.

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
