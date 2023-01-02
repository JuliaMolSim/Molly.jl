export MullerBrown

@doc raw"""
    MullerBrown(A, a, b, c, x₀, y₀)

The Muller-Brown potential energy surface, witth 3 minima and 2 saddle points is given by:
```math
V(x,y) = \sum_{n=1}^{4} A_kexp[a_k(x-x_k^0)^2 + b_k(x-x_k^0)(y-y_k^0) + c_k(y-y_k^0)^2]
```
where A, a, b, c, $x_0$, $y_0$ are 4 element arrays
"""


# General interaction:
# https://juliamolsim.github.io/Molly.jl/dev/docs/#General-interactions
struct MullerBrown{T, F, E}
    A::SVector{4,T}
    a::SVector{4,T}
    b::SVector{4,T}
    c::SVector{4,T}
    x₀::SVector{4,T}
    y₀::SVector{4,T}
    force_units::F
    energy_units::E
end

function MullerBrown(A, a, b, c, x₀, y₀; 
                    force_units = u"kJ * mol^-1 * nm^-1",
                    energy_units = u"kJ * mol^-1")

    return MullerBrown{typeof(A),typeof(a),typeof(b),typeof(c),typeof(x₀),typeof(y₀),typeof(energy_units),typeof(force_units)}(
        A, a, b, c, x₀, y₀, force_units, energy_units)
end

#Total potential energy of system
@inline @inbounds function potential_energy(inter::MullerBrown, sys, neighbors=nothing)
    return sum(potential.(Ref(inter),sys.coords))
end

@inline @inbounds @fastmath function potential(inter::MullerBrown, coord)
    x,y = coord

    res = 0.0
    for i in 1:4
        dx = x - inter.x₀[i]
        dy = y - inter.y₀[i]
        a_part = inter.a[i]*(dx^2)
        b_part = inter.b[i]*dx*dy
        c_part = inter.c[i]*(dy^2)
        res += inter.A[i]*exp(a_part + b_part + c_part)
    end
    
    return res
end

#Force acting on each particle
@inline @inbounds function forces(inter::MullerBrown, sys, neighbors=nothing)
    return force.(Ref(inter),sys.coords)
end

@inline @inbounds @fastmath function force(inter::MullerBrown, coord)
    x,y = coord

    res_x = 0.0
    res_y = 0.0
    for i in 1:4
        dx = x - inter.x₀[i]
        dy = y - inter.y₀[i]
        res_x += inter.A[i]*exp(2*inter.a[i]*dx + inter.b[i]*dy)
        res_y += inter.A[i]*exp(inter.b[i]*dx + 2*inter.c[i]*dy)
    end

    return [res_x,res_y]
end