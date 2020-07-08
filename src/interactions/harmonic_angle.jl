"""
    HarmonicAngle(i, j, k, th0, cth)

A bond angle between three atoms.
"""
struct HarmonicAngle{T} <: SpecificInteraction
    i::Int
    j::Int
    k::Int
    th0::T
    cth::T
end

# Sometimes domain error occurs for acos if the value is > 1.0 or < -1.0
acosbound(x::Real) = acos(clamp(x, -1, 1))

function force!(forces,
                a::HarmonicAngle,
                s::Simulation)
    ba = vector(s.coords[a.j], s.coords[a.i], s.box_size)
    bc = vector(s.coords[a.j], s.coords[a.k], s.box_size)
    pa = normalize(ba × (ba × bc))
    pc = normalize(-bc × (ba × bc))
    angle_term = -a.cth * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.th0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    forces[a.i] += fa
    forces[a.j] += fb
    forces[a.k] += fc
    return nothing
end
