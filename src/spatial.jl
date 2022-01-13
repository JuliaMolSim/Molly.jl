# Spatial calculations

export
    vector1D,
    vector,
    wrap_coords,
    wrap_coords_vec,
    maxwell_boltzmann,
    random_velocities!,
    bond_angle,
    torsion_angle

"""
    vector1D(c1, c2, side_length)

Displacement between two 1D coordinate values from c1 to c2, accounting for
the bounding box.
The minimum image convention is used, so the displacement is to the closest
version of the coordinate accounting for the periodic boundaries.
"""
function vector1D(c1, c2, side_length)
    if c1 < c2
        return (c2 - c1) < (c1 - c2 + side_length) ? (c2 - c1) : (c2 - c1 - side_length)
    else
        return (c1 - c2) < (c2 - c1 + side_length) ? (c2 - c1) : (c2 - c1 + side_length)
    end
end

"""
    vector(c1, c2, box_size)

Displacement between two coordinate values from c1 to c2, accounting for
the bounding box.
The minimum image convention is used, so the displacement is to the closest
version of the coordinates accounting for the periodic boundaries.
"""
vector(c1, c2, box_size) = vector1D.(c1, c2, box_size)

@generated function vector(c1::SVector{N}, c2::SVector{N}, box_size) where N
    quote
        Base.Cartesian.@ncall $N SVector{$N} i->vector1D(c1[i], c2[i], box_size[i])
    end
end

square_distance(i, j, coords, box_size) = sum(abs2, vector(coords[i], coords[j], box_size))

# Pad a vector to 3D to allow operations such as the cross product
function vector_pad3D(c1::SVector{2, T}, c2::SVector{2, T}, box_size::SVector{2, T}) where T
    SVector{3, T}(
        vector1D(c1[1], c2[1], box_size[1]),
        vector1D(c1[2], c2[2], box_size[2]),
        zero(T),
    )
end

vector_pad3D(c1::SVector{3}, c2::SVector{3}, box_size::SVector{3}) = vector(c1, c2, box_size)

# Trim a vector back to 2D if required
trim3D(v::SVector{3, T}, box_size::SVector{2}) where T = SVector{2, T}(v[1], v[2])
trim3D(v::SVector{3}, box_size::SVector{3}) = v

"""
    wrap_coords(c, side_length)

Ensure a 1D coordinate is within the simulation box and return the coordinate.
"""
wrap_coords(c, side_length) = c - floor(c / side_length) * side_length

"""
    wrap_coords_vec(c, box_size)

Ensure a coordinate is within the simulation box and return the coordinate.
"""
wrap_coords_vec(v, box_size) = wrap_coords.(v, box_size)

"""
    velocity(mass, temperature; dims=3)

Generate a random velocity from the Maxwell-Boltzmann distribution.
"""
function AtomsBase.velocity(mass, temp; dims::Integer=3)
    return SVector([maxwell_boltzmann(mass, temp) for i in 1:dims]...)
end

"""
    maxwell_boltzmann(mass, temperature)

Generate a random speed along one dimension from the Maxwell-Boltzmann distribution.
"""
function maxwell_boltzmann(mass, temp)
    T = typeof(convert(AbstractFloat, ustrip(temp)))
    k = unit(temp) == NoUnits ? one(T) : uconvert(u"u * nm^2 * ps^-2 * K^-1", T(Unitful.k))
    σ = sqrt(k * temp / mass)
    return rand(Normal(zero(T), T(ustrip(σ)))) * unit(σ)
end

"""
    random_velocities!(sys, temp)

Set the velocities of a `System` to random velocities generated from the
Maxwell-Boltzmann distribution.
"""
function random_velocities!(sys::System, temp)
    sys.velocities = [velocity(a.mass, temp) for a in sys.atoms]
    return sys
end

# Sometimes domain error occurs for acos if the value is > 1.0 or < -1.0
acosbound(x::Real) = acos(clamp(x, -1, 1))

"""
    bond_angle(coord_i, coord_j, coord_k, box_size)
    bond_angle(vec_ba, vec_bc)

Calculate the bond or pseudo-bond angle in radians between three
coordinates or two vectors.
The angle between B→A and B→C is returned in the range 0 to π.
"""
function bond_angle(coords_i, coords_j, coords_k, box_size)
    vec_ba = vector(coords_j, coords_i, box_size)
    vec_bc = vector(coords_j, coords_k, box_size)
    return bond_angle(vec_ba, vec_bc)
end

function bond_angle(vec_ba, vec_bc)
    acosbound(dot(vec_ba, vec_bc) / (norm(vec_ba) * norm(vec_bc)))
end

"""
    torsion_angle(coord_i, coord_j, coord_k, coord_l, box_size)
    torsion_angle(vec_ab, vec_bc, vec_cd)

Calculate the torsion angle in radians defined by four coordinates or
three vectors.
The angle between the planes defined by atoms (i, j, k) and (j, k, l) is
returned in the range -π to π.
"""
function torsion_angle(coords_i, coords_j, coords_k, coords_l, box_size)
    vec_ab = vector(coords_i, coords_j, box_size)
    vec_bc = vector(coords_j, coords_k, box_size)
    vec_cd = vector(coords_k, coords_l, box_size)
    return torsion_angle(vec_ab, vec_bc, vec_cd)
end

function torsion_angle(vec_ab, vec_bc, vec_cd)
    cross_ab_bc = vec_ab × vec_bc
    cross_bc_cd = vec_bc × vec_cd
    θ = atan(
        ustrip(dot(cross_ab_bc × cross_bc_cd, normalize(vec_bc))),
        ustrip(dot(cross_ab_bc, cross_bc_cd)),
    )
    return θ
end
