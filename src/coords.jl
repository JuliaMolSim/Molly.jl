# Atomic and spatial types and calculations

export
    Atom,
    Coordinates,
    Velocity,
    maxwellboltzmann,
    Acceleration

"An atom and its associated information."
struct Atom
    attype::String
    name::String
    resnum::Int
    resname::String
    charge::Float64
    mass::Float64
    σ::Float64
    ϵ::Float64
end

"3D coordinates, e.g. for an atom, in nm."
mutable struct Coordinates <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    z::Float64
end

"3D velocity values, e.g. for an atom, in nm/ps."
mutable struct Velocity <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    z::Float64
end

# Generate a random 3D velocity from the Maxwell-Boltzmann distribution
function Velocity(mass::Real, T::Real)
    return Velocity([maxwellboltzmann(mass, T) for _ in 1:3])
end

"Generate a random velocity from the Maxwell-Boltzmann distribution."
function maxwellboltzmann(mass::Real, T::Real)
    return rand(Normal(0.0, sqrt(molar_gas_const * T / mass)))
end

"3D acceleration values, e.g. for an atom, in nm/(ps^2)."
mutable struct Acceleration <: FieldVector{3, Float64}
    x::Float64
    y::Float64
    z::Float64
end

"Vector between two coordinate values, accounting for the bounding box."
function vector1D(c1::Real, c2::Real, box_size::Real)
    if c1 < c2
        return (c2 - c1) < (c1 - c2 + box_size) ? (c2 - c1) : (c2 - c1 - box_size)
    else
        return (c1 - c2) < (c2 - c1 + box_size) ? (c2 - c1) : (c2 - c1 + box_size)
    end
end

"3D vector between two `Coordinates`, accounting for the bounding box."
vector(c1::Coordinates, c2::Coordinates, box_size::Real) = [
        vector1D(c1.x, c2.x, box_size),
        vector1D(c1.y, c2.y, box_size),
        vector1D(c1.z, c2.z, box_size)]

"Square distance between two `Coordinates`, accounting for the bounding box."
sqdist(c1::Coordinates, c2::Coordinates, box_size::Real) =
        vector1D(c1.x, c2.x, box_size) ^ 2 +
        vector1D(c1.y, c2.y, box_size) ^ 2 +
        vector1D(c1.z, c2.z, box_size) ^ 2
