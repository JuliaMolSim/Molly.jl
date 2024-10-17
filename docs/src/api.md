# Molly API

The API reference can be found here.

Molly re-exports [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and [Unitful.jl](https://github.com/PainterQubits/Unitful.jl), making the likes of `SVector` and `1.0u"nm"` available when you call `using Molly`.

Package extensions are used in order to reduce the number of dependencies:
- To use [`visualize`](@ref), call `using GLMakie`.
- To use [`ASECalculator`](@ref), call `using PythonCall`.

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
