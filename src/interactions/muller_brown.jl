export MullerBrown

@doc raw"""
    MullerBrown(; A, a, b, c, x0, y0, force_units, energy_units)

The Müller-Brown potential energy surface implemented as an AtomsCalculators.jl calculator.

The potential energy is defined as
```math
V(x,y) = \sum_{n=1}^{4} A_k \exp[a_k(x-x_k^0)^2 + b_k(x-x_k^0)(y-y_k^0) + c_k(y-y_k^0)^2]
```
where `A`, `a`, `b`, `c`, `x0`, `y0` are 4-element `SVector`s with standard defaults.

This potential is only compatible with 2D systems.
It is often used for testing algorithms that find transition states or explore
minimum energy pathways.
There are 3 minima and 2 saddle points with the default parameters.
"""
struct MullerBrown{D, T, L, F, E}
    A::D  # Units of energy
    a::T  # Units of 1/L^2
    b::T  # Units of 1/L^2
    c::T  # Units of 1/L^2
    x0::L # Units of L
    y0::L # Units of L
    force_units::F
    energy_units::E
end

function MullerBrown(; A=SVector(-200.0, -100.0, -170.0, 15.0)u"kJ * mol^-1",
                       a=SVector(  -1.0,   -1.0,   -6.5,  0.7)u"nm^-2",
                       b=SVector(   0.0,    0.0,   11.0,  0.6)u"nm^-2",
                       c=SVector( -10.0,  -10.0,   -6.5,  0.7)u"nm^-2",
                       x0=SVector(  1.0,    0.0,   -0.5, -1.0)u"nm",
                       y0=SVector(  0.0,    0.5,    1.5,  1.0)u"nm",
                       force_units = u"kJ * mol^-1 * nm^-1",
                       energy_units = u"kJ * mol^-1")
    if any(arr -> length(arr) != 4, (A, a, b, c, x0, y0))
        throw(ArgumentError("the length of each SVector for the Müller-Brown potential should be 4"))
    end

    return MullerBrown{typeof(A), typeof(a), typeof(x0), typeof(force_units), typeof(energy_units)}(
        A, a, b, c, x0, y0, force_units, energy_units)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                            inter::MullerBrown; kwargs...)
    return sum(potential_muller_brown.((inter,), sys.coords))
end

@inline function potential_muller_brown(inter::MullerBrown, coord::SVector{2})
    x, y = coord
    res = ustrip(zero(coord[1])) * inter.energy_units

    for i in 1:4
        dx = x - inter.x0[i]
        dy = y - inter.y0[i]
        a_part = inter.a[i] * dx^2
        b_part = inter.b[i] * dx * dy
        c_part = inter.c[i] * dy^2
        res += inter.A[i] * exp(a_part + b_part + c_part)
    end
    return res
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces!(fs, sys,
                                                            inter::MullerBrown; kwargs...)
    fs .+= force_muller_brown.((inter,), sys.coords)
    return fs
end

@inline function force_muller_brown(inter::MullerBrown, coord::SVector{2})
    x, y = coord
    res_x = ustrip(zero(coord[1])) * inter.force_units
    res_y = ustrip(zero(coord[1])) * inter.force_units

    for i in 1:4
        dx = x - inter.x0[i]
        dy = y - inter.y0[i]
        exp_part = inter.A[i] * exp(inter.a[i] * dx^2 + inter.b[i] * dx * dy + inter.c[i] * dy^2)
        res_x += exp_part * (2 * inter.a[i] * dx + inter.b[i] * dy)
        res_y += exp_part * (inter.b[i] * dx + 2 * inter.c[i] * dy)
    end
    return SVector(-res_x, -res_y)
end
