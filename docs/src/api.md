# Molly API

The API reference can be found here.

Molly also re-exports [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), making the likes of `SVector` available when you call `using Molly`.

The `visualize` function is only available once you have called `using Makie`.
[Requires.jl](https://github.com/JuliaPackaging/Requires.jl) is used to lazily load this code when [Makie.jl](https://github.com/JuliaPlots/Makie.jl) is available, which reduces the dependencies of the package.

```@index
Order   = [:module, :type, :constant, :function, :macro]
```

```@autodocs
Modules = [Molly]
Private = false
Order   = [:module, :type, :constant, :function, :macro]
```
