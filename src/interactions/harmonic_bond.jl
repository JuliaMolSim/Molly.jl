"""
    HarmonicBond(; i, j, b0, kb, force_units, energy_units)

A harmonic bond between two atoms.
"""
struct HarmonicBond{D, K, F, E} <: SpecificInteraction
    i::Int
    j::Int
    b0::D
    kb::K
    force_units::F
    energy_units::E
end

function HarmonicBond(;
                        i,
                        j,
                        b0,
                        kb,
                        force_units=u"kJ * mol^-1 * nm^-1",
                        energy_units=u"kJ * mol^-1")
    return HarmonicBond{typeof(b0), typeof(kb), typeof(force_units), typeof(energy_units)}(
        i, j, b0, kb, force_units, energy_units)
end

@inline @inbounds function force(b::HarmonicBond,
                coords,
                s::Simulation)
    ab = vector(coords[b.i], coords[b.j], s.box_size)
    c = b.kb * (norm(ab) - b.b0)
    f = c * normalize(ab)
    return [b.i, b.j], [f, -f]
end

@inline @inbounds function potential_energy(b::HarmonicBond,
                                            s::Simulation)
    dr = vector(s.coords[b.i], s.coords[b.j], s.box_size)
    r = norm(dr)
    return (b.kb / 2) * (r - b.b0) ^ 2
end
