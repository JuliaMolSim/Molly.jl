export MullerBrown

@doc raw"""
    MullerBrown(A, a, b, c, x₀, y₀)

The Muller-Brown potential energy surface, witth 3 minima and 2 saddle points is given by:
```math
V(x,y) = \sum_{n=1}^{4} A_kexp[a_k(x-x_k^0)^2 + b_k(x-x_k^0)(y-y_k^0) + c_k(y-y_k^0)^2]
```
where A, a, b, c, $x_0$, $y_0$ are 4 element arrays
"""
#To-Do
# A should have energy_units, in general are units working
# Test coverage
# Example for documentation
# Should potential return just a number?

# General interaction:
# https://juliamolsim.github.io/Molly.jl/dev/docs/#General-interactions
struct MullerBrown{D,T,L,F,E}
    A::D # Array w/ units of energy
    a::T # Array w/ units of 1/L^2
    b::T # Array w/ units of 1/L^2
    c::T # Array w/ units of 1/L^2
    x₀::L # Array w/ units of L
    y₀::L # Array w/ units of L
    force_units::F
    energy_units::E
end


function MullerBrown(A, a, b, c, x₀, y₀; 
                        force_units = u"kJ * mol^-1 * nm^-1",
                        energy_units = u"kJ * mol^-1")

    if length(A) != 4 || length(a) != 4 || length(b) != 4 || length(c) != 4 || length(x₀) != 4 || length(y₀) != 4
        throw(ArgumentError("The length of each array passed to the Muller-Brown potential should be 4."))
    end

    return MullerBrown{typeof(A), typeof(a), typeof(x₀), typeof(force_units), typeof(energy_units)}(
        A, a, b, c, x₀, y₀, force_units, energy_units)
end

#Total potential energy of system
@inline @inbounds function potential_energy(inter::MullerBrown{D,T,L}, sys, neighbors=nothing) where {D,T,L}
    return sum(potential.(Ref(inter),sys.coords))
end

@inline @inbounds @fastmath function potential(inter::MullerBrown{D,T,L}, coord::SVector{2}) where {D,T,L}
    
    x,y = coord

    res = 0.0 * inter.energy_units #give result units of energy
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
@inline @inbounds function forces(inter::MullerBrown{D,T,L}, sys, neighbors=nothing) where {D,T,L}
    return force.(Ref(inter),sys.coords)
end

@inline @inbounds @fastmath function force(inter::MullerBrown{D,T,L}, coord::SVector{2}) where {D,T,L}

    x,y = coord

    res_x = 0.0 * inter.force_units #give result units of force
    res_y = 0.0 * inter.force_units
    for i in 1:4
        dx = x - inter.x₀[i]
        dy = y - inter.y₀[i]
        exp_part = inter.A[i]*exp(inter.a[i]*(dx^2) + inter.b[i]*dx*dy + inter.c[i]*(dy^2))
        res_x += exp_part*(2*inter.a[i]*dx + inter.b[i]*dy)
        res_y += exp_part*(inter.b[i]*dx + 2*inter.c[i]*dy)
    end
    return SVector{2}([-res_x, -res_y])
end