export MullerBrown

@doc raw"""
    MullerBrown(; G, nl_only)

The Muller-Brown potential energy surface is given by:
```math
V(x,y) = to-do
```
"""
struct MullerBrown{T} <: SpecificInteraction #wrong type, but I don't see one that fits
    A::Array{T,1}
    a::Array{T,1} #is there a way to specify length should be 4
    b::Array{T,1}
    c::Array{T,1}
    x₀::Array{T,1}
    y₀::Array{T,1}
end

function MullerBrown(; A, a, b, c, x₀, y₀)
    return MullerBrown{typeof(A),typeof(a),typeof(b),typeof(c),typeof(x₀),typeof(y₀)}(
        A, a, b, c, x₀, y₀
    )
end

@inline @inbounds @fastmath function potential_energy(inter::MullerBrown,
                                            x,
                                            y)
    #might be a way to vectorize this
    res = 0.0
    for i in 1:4
        a_part = a[i]*((x - inter.x₀[i])^2)
        b_part = b[i]*(x - inter.x₀[i])*(y - inter.y₀[i])
        c_part = c[i]*((y - inter.y₀[i])^2)
        res += inter.A[i]*exp(a_part + b_part + c_part)
    end
    return res
    
end

