export MullerBrown

@doc raw"""
    MullerBrown(; A, a, b, c, x0, y0, force_units, energy_units)

The Muller-Brown potential energy surface, witth 3 minima and 2 saddle points is given by:
```math
V(x,y) = \sum_{n=1}^{4} A_kexp[a_k(x-x_k^0)^2 + b_k(x-x_k^0)(y-y_k^0) + c_k(y-y_k^0)^2]
```
where A, a, b, c, x0, y0 are 4 element arrays
"""

struct MullerBrown{D,T,L,F,E}
    A::D # Array w/ units of energy
    a::T # Array w/ units of 1/L^2
    b::T # Array w/ units of 1/L^2
    c::T # Array w/ units of 1/L^2
    x0::L # Array w/ units of L
    y0::L # Array w/ units of L
    force_units::F
    energy_units::E
end

function MullerBrown(; A = SVector(-200.0,-100.0,-170.0,15.0)u"kJ * mol^-1", 
                        a = SVector(-1.0,-1.0,-6.5,0.7)u"nm^-2",
                        b = SVector(0.0,0.0,11.0,0.6)u"nm^-2",
                        c = SVector(-10,-10,-6.5,0.7)u"nm^-2",
                        x0 = SVector(1,0,-0.5,-1)u"nm",
                        y0 = SVector(0.0,0.5,1.5,1.0)u"nm", 
                        force_units = u"kJ * mol^-1 * nm^-1",
                        energy_units = u"kJ * mol^-1")

    if length(A) != 4 || length(a) != 4 || length(b) != 4 || length(c) != 4 || length(x0) != 4 || length(y0) != 4
        throw(ArgumentError("The length of each array passed to the Muller-Brown potential should be 4."))
    end

    return MullerBrown{typeof(A), typeof(a), typeof(x0), typeof(force_units), typeof(energy_units)}(
        A, a, b, c, x0, y0, force_units, energy_units)
end

#Total potential energy of system
@inline @inbounds function potential_energy(inter::MullerBrown, sys, neighbors=nothing;
                                            n_threads::Integer=Threads.nthreads())
    return sum(potential_muller_brown.(Ref(inter),sys.coords))
end

@inline @inbounds @fastmath function potential_muller_brown(inter::MullerBrown, coord::SVector{2}) 
    
    x,y = coord

    res = ustrip(zero(coord[1])) * inter.energy_units #give result units of energy
    for i in 1:4
        dx = x - inter.x0[i]
        dy = y - inter.y0[i]
        a_part = inter.a[i]*(dx^2)
        b_part = inter.b[i]*dx*dy
        c_part = inter.c[i]*(dy^2)
        res += inter.A[i]*exp(a_part + b_part + c_part)
    end
    
    return res
end

#Force acting on each particle
@inline @inbounds function forces(inter::MullerBrown, sys, neighbors=nothing;
                                  n_threads::Integer=Threads.nthreads())
    return force_muller_brown.(Ref(inter),sys.coords)
end

@inline @inbounds @fastmath function force_muller_brown(inter::MullerBrown, coord::SVector{2}) 

    x,y = coord

    res_x = ustrip(zero(coord[1])) * inter.force_units #give result units of force
    res_y = ustrip(zero(coord[1])) * inter.force_units
    for i in 1:4
        dx = x - inter.x0[i]
        dy = y - inter.y0[i]
        exp_part = inter.A[i]*exp(inter.a[i]*(dx^2) + inter.b[i]*dx*dy + inter.c[i]*(dy^2))
        res_x += exp_part*(2*inter.a[i]*dx + inter.b[i]*dy)
        res_y += exp_part*(inter.b[i]*dx + 2*inter.c[i]*dy)
    end
    return SVector(-res_x, -res_y)
end