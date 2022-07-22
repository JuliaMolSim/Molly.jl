# Spatial calculations

export
    CubicBoundary,
    RectangularBoundary,
    TriclinicBoundary,
    box_volume,
    rand_coord,
    vector_1D,
    vector,
    wrap_coord_1D,
    wrap_coords,
    maxwell_boltzmann,
    random_velocities,
    random_velocities!,
    bond_angle,
    torsion_angle,
    remove_CM_motion!

"""
    CubicBoundary(x, y, z)
    CubicBoundary(arr)

Cubic 3D bounding box defined by 3 side lengths.
Setting one or more values to `Inf` gives no boundary in that dimension.
"""
struct CubicBoundary{T}
    side_lengths::SVector{3, T}
end

CubicBoundary(x, y, z) = CubicBoundary(SVector{3}(x, y, z))
CubicBoundary(arr) = CubicBoundary(SVector{3}(arr))

Base.getindex(b::CubicBoundary, i::Integer) = b.side_lengths[i]
Base.firstindex(b::CubicBoundary) = b.side_lengths[1]
Base.lastindex(b::CubicBoundary) = b.side_lengths[3]

"""
    n_dimensions(boundary)

Number of dimensions of a bounding box.
"""
AtomsBase.n_dimensions(::CubicBoundary) = 3

"""
    RectangularBoundary(x, y)
    RectangularBoundary(arr)

Rectangular 2D bounding box defined by 2 side lengths.
Setting one or more values to `Inf` gives no boundary in that dimension.
"""
struct RectangularBoundary{T}
    side_lengths::SVector{2, T}
end

RectangularBoundary(x, y) = RectangularBoundary(SVector{2}(x, y))
RectangularBoundary(arr) = RectangularBoundary(SVector{2}(arr))

Base.getindex(b::RectangularBoundary, i::Integer) = b.side_lengths[i]
Base.firstindex(b::RectangularBoundary) = b.side_lengths[1]
Base.lastindex(b::RectangularBoundary) = b.side_lengths[2]

AtomsBase.n_dimensions(::RectangularBoundary) = 2

Base.broadcastable(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths

"""
    TriclinicBoundary(v1, v2, v3)

Triclinic 3D bounding box defined by 3 basis vectors.
An approximation is used to find the closest periodic image when using the
minimum image convention.
Not currently compatible with infinite boundaries.
"""
struct TriclinicBoundary{T, D, I}
    basis_vectors::SVector{3, SVector{3, D}}
    reciprocal_size::SVector{3, I}
    angles::SVector{3, T}
    cubic_bounds::SVector{3, D}
end

function TriclinicBoundary(basis_vectors::SVector{3})
    reciprocal_size = SVector{3}(
        inv(basis_vectors[1][1]),
        inv(basis_vectors[2][2]),
        inv(basis_vectors[3][3]),
    )
    angles = SVector{3}(
        bond_angle(basis_vectors[1], basis_vectors[2]), # xy angle
        bond_angle(basis_vectors[1], basis_vectors[3]), # xz angle
        bond_angle(basis_vectors[2], basis_vectors[3]), # yz angle
    )
    dx, dy, dz = norm.(basis_vectors)
    cubic_bounds = SVector{3}(
        dx + dy * cos(angles[1]) + dz * cos(angles[2]),
        dy * sin(angles[1]) + dz * sin(angles[3]),
        dz * sin(angles[2]),
    )
    return TriclinicBoundary{eltype(angles), eltype(cubic_bounds), eltype(reciprocal_size)}(
                                basis_vectors, reciprocal_size, angles, cubic_bounds)
end

TriclinicBoundary(v1, v2, v3) = TriclinicBoundary(SVector{3}(v1, v2, v3))
TriclinicBoundary(arr) = TriclinicBoundary(SVector{3}(arr))

Base.getindex(b::TriclinicBoundary, i::Integer) = b.basis_vectors[i]
Base.firstindex(b::TriclinicBoundary) = b.basis_vectors[1]
Base.lastindex(b::TriclinicBoundary) = b.basis_vectors[3]

AtomsBase.n_dimensions(::TriclinicBoundary) = 3

float_type(b::Union{CubicBoundary, RectangularBoundary}) = typeof(ustrip(b[1]))
float_type(b::TriclinicBoundary{T}) where {T} = T

"""
    box_volume(boundary)

Calculate the volume of a bounding box.
"""
box_volume(b::Union{CubicBoundary, RectangularBoundary}) = prod(b.side_lengths)
box_volume(b::TriclinicBoundary) = abs(dot(cross(b[1], b[2]), b[3]))

"""
    rand_coord(boundary)

Generate a random coordinate uniformly distributed within a bounding box.
"""
rand_coord(boundary::CubicBoundary      ) = rand(SVector{3, float_type(boundary)}) .* boundary
rand_coord(boundary::RectangularBoundary) = rand(SVector{2, float_type(boundary)}) .* boundary

function rand_coord(boundary::TriclinicBoundary{T}) where T
    in_bounds = false
    while !in_bounds
        trial_coord = rand(SVector{3, T}) .* boundary.cubic_bounds
        dx = floor(trial_coord[1] * boundary.reciprocal_size[1])
        dy = floor(trial_coord[2] * boundary.reciprocal_size[2])
        dz = floor(trial_coord[3] * boundary.reciprocal_size[3])
        if iszero(dx) && iszero(dy) && iszero(dz)
            in_bounds = true
        end
    end
    return trial_coord
end

"""
    vector_1D(c1, c2, side_length)

Displacement between two 1D coordinate values from c1 to c2, accounting for
the bounding box.
The minimum image convention is used, so the displacement is to the closest
version of the coordinate accounting for the periodic boundaries.
"""
function vector_1D(c1, c2, side_length)
    if c1 < c2
        return (c2 - c1) < (c1 - c2 + side_length) ? (c2 - c1) : (c2 - c1 - side_length)
    else
        return (c1 - c2) < (c2 - c1 + side_length) ? (c2 - c1) : (c2 - c1 + side_length)
    end
end

"""
    vector(c1, c2, boundary)

Displacement between two coordinate values from c1 to c2, accounting for
the bounding box.
The minimum image convention is used, so the displacement is to the closest
version of the coordinates accounting for the periodic boundaries.
For the [`TriclinicBoundary`](@ref) an approximation is used to find the closest
version.
"""
vector(c1, c2, boundary::Union{CubicBoundary, RectangularBoundary}) = vector_1D.(c1, c2, boundary)

@generated function vector(c1::SVector{N}, c2::SVector{N}, boundary::Union{CubicBoundary, RectangularBoundary}) where N
    quote
        Base.Cartesian.@ncall $N SVector{$N} i -> vector_1D(c1[i], c2[i], boundary[i])
    end
end

function vector(c1, c2, boundary::TriclinicBoundary{T}) where T
    dr = c2 - c1
    dx = boundary.basis_vectors[1] * floor(dr[1] * boundary.reciprocal_size[1] + T(0.5))
    dy = boundary.basis_vectors[2] * floor(dr[2] * boundary.reciprocal_size[2] + T(0.5))
    dz = boundary.basis_vectors[3] * floor(dr[3] * boundary.reciprocal_size[3] + T(0.5))
    return dr - dx - dy - dz
end

square_distance(i, j, coords, boundary) = sum(abs2, vector(coords[i], coords[j], boundary))

# Pad a vector to 3D to allow operations such as the cross product
function vector_pad3D(c1::SVector{2, T}, c2::SVector{2, T}, boundary::RectangularBoundary{T}) where T
    SVector{3, T}(
        vector_1D(c1[1], c2[1], boundary[1]),
        vector_1D(c1[2], c2[2], boundary[2]),
        zero(T),
    )
end

vector_pad3D(c1::SVector{3}, c2::SVector{3}, boundary) = vector(c1, c2, boundary)

# Trim a vector back to 2D if required
trim3D(v::SVector{3, T}, boundary::RectangularBoundary{T}) where T = SVector{2, T}(v[1], v[2])
trim3D(v::SVector{3}, boundary) = v

"""
    wrap_coord_1D(c, side_length)

Ensure a 1D coordinate is within the bounding box and return the coordinate.
"""
function wrap_coord_1D(c, side_length)
    if isinf(side_length)
        return c
    else
        return c - floor(c / side_length) * side_length
    end
end

"""
    wrap_coords(c, boundary)

Ensure a coordinate is within the bounding box and return the coordinate.
"""
wrap_coords(v, boundary::Union{CubicBoundary, RectangularBoundary}) = wrap_coord_1D.(v, boundary)

function wrap_coords(v, boundary::TriclinicBoundary)
    dx = boundary.basis_vectors[1] * floor(v[1] * boundary.reciprocal_size[1])
    dy = boundary.basis_vectors[2] * floor(v[2] * boundary.reciprocal_size[2])
    dz = boundary.basis_vectors[3] * floor(v[3] * boundary.reciprocal_size[3])
    return v - dx - dy - dx
end

const mb_conversion_factor = uconvert(u"u * nm^2 * ps^-2 * K^-1", Unitful.k)

"""
    velocity(mass, temperature; dims=3)
    velocity(mass, temperature, k; dims=3)

Generate a random velocity from the Maxwell-Boltzmann distribution, with
optional custom Boltzmann constant.
"""
function AtomsBase.velocity(mass, temp, k=mb_conversion_factor;
                            dims::Integer=3, rng=Random.GLOBAL_RNG)
    k_strip = (unit(mass) == NoUnits) ? ustrip(k) : k
    return SVector([maxwell_boltzmann(mass, temp, k_strip; rng=rng) for i in 1:dims]...)
end

function velocity_3D(mass, temp, k=mb_conversion_factor; rng=Random.GLOBAL_RNG)
    return SVector(
        maxwell_boltzmann(mass, temp, k; rng=rng),
        maxwell_boltzmann(mass, temp, k; rng=rng),
        maxwell_boltzmann(mass, temp, k; rng=rng),
    )
end

function velocity_2D(mass, temp, k=mb_conversion_factor; rng=Random.GLOBAL_RNG)
    return SVector(
        maxwell_boltzmann(mass, temp, k; rng=rng),
        maxwell_boltzmann(mass, temp, k; rng=rng),
    )
end

"""
    maxwell_boltzmann(mass, temperature; rng=Random.GLOBAL_RNG)
    maxwell_boltzmann(mass, temperature, k; rng=Random.GLOBAL_RNG)

Generate a random speed along one dimension from the Maxwell-Boltzmann
distribution, with optional custom Boltzmann constant.
"""
function maxwell_boltzmann(mass, temp, k; rng=Random.GLOBAL_RNG)
    T = typeof(convert(AbstractFloat, ustrip(temp)))
    σ = sqrt(k * temp / mass)
    return rand(rng, Normal(zero(T), T(ustrip(σ)))) * unit(σ)
end

function maxwell_boltzmann(mass, temp; rng=Random.GLOBAL_RNG)
    k = unit(temp) == NoUnits ? ustrip(mb_conversion_factor) : mb_conversion_factor
    return maxwell_boltzmann(mass, temp, k; rng=rng)
end

"""
    random_velocities(sys, temp)

Generate random velocities from the Maxwell-Boltzmann distribution
for a [`System`](@ref).
"""
function random_velocities(sys::AbstractSystem{3}, temp; rng=Random.GLOBAL_RNG)
    vels = velocity_3D.(Array(masses(sys)), temp, sys.k; rng=rng)
    return move_array(vels, sys)
end

function random_velocities(sys::AbstractSystem{2}, temp; rng=Random.GLOBAL_RNG)
    vels = velocity_2D.(Array(masses(sys)), temp, sys.k; rng=rng)
    return move_array(vels, sys)
end

"""
    random_velocities!(sys, temp)

Set the velocities of a [`System`](@ref) to random velocities generated from the
Maxwell-Boltzmann distribution.
"""
function random_velocities!(sys, temp; rng=Random.GLOBAL_RNG)
    sys.velocities = random_velocities(sys, temp; rng=rng)
    return sys
end

# Sometimes domain error occurs for acos if the value is > 1.0 or < -1.0
acosbound(x::Real) = acos(clamp(x, -1, 1))

"""
    bond_angle(coord_i, coord_j, coord_k, boundary)
    bond_angle(vec_ji, vec_jk)

Calculate the bond or pseudo-bond angle in radians between three
coordinates or two vectors.
The angle between j→i and j→k is returned in the range 0 to π.
"""
function bond_angle(coords_i, coords_j, coords_k, boundary)
    vec_ji = vector(coords_j, coords_i, boundary)
    vec_jk = vector(coords_j, coords_k, boundary)
    return bond_angle(vec_ji, vec_jk)
end

function bond_angle(vec_ji, vec_jk)
    acosbound(dot(vec_ji, vec_jk) / (norm(vec_ji) * norm(vec_jk)))
end

"""
    torsion_angle(coord_i, coord_j, coord_k, coord_l, boundary)
    torsion_angle(vec_ij, vec_jk, vec_kl)

Calculate the torsion angle in radians defined by four coordinates or
three vectors.
The angle between the planes defined by atoms (i, j, k) and (j, k, l) is
returned in the range -π to π.
"""
function torsion_angle(coords_i, coords_j, coords_k, coords_l, boundary)
    vec_ij = vector(coords_i, coords_j, boundary)
    vec_jk = vector(coords_j, coords_k, boundary)
    vec_kl = vector(coords_k, coords_l, boundary)
    return torsion_angle(vec_ij, vec_jk, vec_kl)
end

function torsion_angle(vec_ij, vec_jk, vec_kl)
    cross_ij_jk = vec_ij × vec_jk
    cross_jk_kl = vec_jk × vec_kl
    θ = atan(
        ustrip(dot(cross_ij_jk × cross_jk_kl, normalize(vec_jk))),
        ustrip(dot(cross_ij_jk, cross_jk_kl)),
    )
    return θ
end

# Used to write an rrule that can override the Zygote sum adjoint
sum_svec(arr) = sum(arr)

"""
    remove_CM_motion!(system)

Remove the centre of mass motion from a [`System`](@ref).
"""
function remove_CM_motion!(sys)
    atom_masses = masses(sys)
    cm_momentum = sum_svec(sys.velocities .* atom_masses)
    cm_velocity = cm_momentum / sum(atom_masses)
    sys.velocities = sys.velocities .- (cm_velocity,)
    return sys
end
