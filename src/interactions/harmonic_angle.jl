"""
    HarmonicAngle(; i, j, k, th0, cth)

A bond angle between three atoms.
"""
struct HarmonicAngle{D, K} <: SpecificInteraction
    i::Int
    j::Int
    k::Int
    th0::D
    cth::K
end

HarmonicAngle(; i, j, k, th0, cth) = HarmonicAngle{typeof(th0), typeof(cth)}(i, j, k, th0, cth)

# Sometimes domain error occurs for acos if the value is > 1.0 or < -1.0
acosbound(x::Real) = acos(clamp(x, -1, 1))

@inline @inbounds function force(a::HarmonicAngle,
                                  coords,
                                  s::System)
    ba = vector(coords[a.j], coords[a.i], s.box_size)
    bc = vector(coords[a.j], coords[a.k], s.box_size)
    cross_ba_bc = ba × bc
    if iszero(cross_ba_bc)
        fz = ustrip.(zero(coords[a.i])) * s.force_unit
        return [a.i, a.j, a.k], [fz, fz, fz]
    end
    pa = normalize(ba × cross_ba_bc)
    pc = normalize(-bc × cross_ba_bc)
    angle_term = -a.cth * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.th0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return [a.i, a.j, a.k], [fa, fb, fc]
end

@inline @inbounds function potential_energy(a::HarmonicAngle,
                                            s::System)
    ba = vector(s.coords[a.j], s.coords[a.i], s.box_size)
    bc = vector(s.coords[a.j], s.coords[a.k], s.box_size)
    return (a.cth / 2) * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.th0)^2
end
