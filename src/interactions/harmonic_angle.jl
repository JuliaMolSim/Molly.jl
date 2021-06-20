"""
    HarmonicAngle(; i, j, k, th0, cth, force_units, energy_units)

A bond angle between three atoms.
"""
struct HarmonicAngle{D, K, F, E} <: SpecificInteraction
    i::Int
    j::Int
    k::Int
    th0::D
    cth::K
    force_units::F
    energy_units::E
end

function HarmonicAngle(;
                        i,
                        j,
                        k,
                        th0,
                        cth,
                        force_units=u"kJ * mol^-1 * nm^-1",
                        energy_units=u"kJ * mol^-1")
    return HarmonicAngle{typeof(th0), typeof(cth), typeof(force_units), typeof(energy_units)}(
        i, j, k, th0, cth, force_units, energy_units)
end

# Sometimes domain error occurs for acos if the value is > 1.0 or < -1.0
acosbound(x::Real) = acos(clamp(x, -1, 1))

@inline @inbounds function force(a::HarmonicAngle,
                                  coords,
                                  s::Simulation)
    ba = vector(coords[a.j], coords[a.i], s.box_size)
    bc = vector(coords[a.j], coords[a.k], s.box_size)
    pa = normalize(ba × (ba × bc))
    pc = normalize(-bc × (ba × bc))
    angle_term = -a.cth * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.th0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return [a.i, a.j, a.k], [fa, fb, fc]
end

@inline @inbounds function potential_energy(a::HarmonicAngle,
                                            s::Simulation)
    ba = vector(s.coords[a.j], s.coords[a.i], s.box_size)
    bc = vector(s.coords[a.j], s.coords[a.k], s.box_size)
    return (a.cth / 2) * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - a.th0)^2
end
