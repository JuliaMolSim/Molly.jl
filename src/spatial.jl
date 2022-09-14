# Spatial calculations

export
    CubicBoundary,
    RectangularBoundary,
    TriclinicBoundary,
    box_volume,
    box_center,
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

"""
    TriclinicBoundary(v1, v2, v3; approx_images=true)
    TriclinicBoundary(SVector(l1, l2, l3), SVector(α, β, γ); approx_images=true)
    TriclinicBoundary(arr; approx_images=true)

Triclinic 3D bounding box defined by 3 `SVector{3}` basis vectors or basis vector
lengths and angles α/β/γ in radians.
The first basis vector must point along the x-axis and the second must lie in the
xy plane.

An approximation is used to find the closest periodic image when using the
minimum image convention.
The approximation is correct for distances shorter than half the shortest box
height/width.
Setting the keyword argument `approx_images` to `false` means the exact closest
image is found, which is slower.
Not currently compatible with infinite boundaries.
Not currently compatible with automatic differentiation using Zygote.
"""
struct TriclinicBoundary{T, A, D, I}
    basis_vectors::SVector{3, SVector{3, D}}
    reciprocal_size::SVector{3, I}
    α::T
    β::T
    γ::T
    tan_bprojyz_cprojyz::T
    tan_c_cprojxy::T
    cos_a_cprojxy::T
    sin_a_cprojxy::T
    tan_a_b::T
end

ispositive(x) = x > zero(x)

function TriclinicBoundary(bv::SVector{3}; approx_images::Bool=true)
    if !ispositive(bv[1][1]) || !iszero(bv[1][2]) || !iszero(bv[1][3])
        throw(ArgumentError("First basis vector must be along the x-axis (no y or z component) " *
                            "and have a positive x component " * 
                            "when constructing a TriclinicBoundary, got $(bv[1])"))
    end
    if !ispositive(bv[2][2]) || !iszero(bv[2][3])
        throw(ArgumentError("Second basis vector must be in the xy plane (no z component) " *
                            "and have a positive y component " *
                            "when constructing a TriclinicBoundary, got $(bv[2])"))
    end
    if !ispositive(bv[3][3])
        throw(ArgumentError("Third basis vector must have a positive z component " *
                            "when constructing a TriclinicBoundary, got $(bv[3])"))
    end
    reciprocal_size = SVector{3}(inv(bv[1][1]), inv(bv[2][2]), inv(bv[3][3]))
    α = bond_angle(bv[2], bv[3])
    β = bond_angle(bv[1], bv[3])
    γ = bond_angle(bv[1], bv[2])
    # Precompute angles to speed up coordinate wrapping
    tan_bprojyz_cprojyz = tan(bond_angle(
        SVector(zero(bv[2][1]), bv[2][2], bv[2][3]),
        SVector(zero(bv[3][1]), bv[3][2], bv[3][3]),
    ))
    tan_c_cprojxy = tan(bond_angle(SVector(bv[3][1], bv[3][2], zero(bv[3][3])), bv[3]))
    a_cprojxy = bond_angle(bv[1], SVector(bv[3][1], bv[3][2], zero(bv[3][3])))
    tan_a_b = tan(bond_angle(bv[1], bv[2]))
    return TriclinicBoundary{typeof(α), approx_images, eltype(eltype(bv)), eltype(reciprocal_size)}(
                                bv, reciprocal_size, α, β, γ, tan_bprojyz_cprojyz, tan_c_cprojxy,
                                cos(a_cprojxy), sin(a_cprojxy), tan_a_b)
end

function TriclinicBoundary(bv_lengths::SVector{3}, angles::SVector{3}; kwargs...)
    if any(!ispositive, bv_lengths)
        throw(ArgumentError("Basis vector lengths must be positive " *
                            "when constructing a TriclinicBoundary, got $bv_lengths"))
    end
    if !all(a -> 0 < a < π, angles)
        throw(ArgumentError("Basis vector angles must be 0 to π radians " *
                            "when constructing a TriclinicBoundary, got $angles"))
    end
    α, β, γ = angles
    cos_α, cos_β, cos_γ, sin_γ = cos(α), cos(β), cos(γ), sin(γ)
    z = zero(bv_lengths[1])
    v1 = SVector(bv_lengths[1], z, z)
    v2 = SVector(bv_lengths[2] * cos_γ, bv_lengths[2] * sin_γ, z)
    v3x = bv_lengths[3] * cos_β
    v3y = bv_lengths[3] * (cos_α - cos_β * cos_γ) / sin_γ
    v3z = sqrt(bv_lengths[3] ^ 2 - v3x ^ 2 - v3y ^ 2)
    v3 = SVector(v3x, v3y, v3z)
    return TriclinicBoundary(SVector{3}(v1, v2, v3); kwargs...)
end

TriclinicBoundary(v1, v2, v3; kwargs...) = TriclinicBoundary(SVector{3}(v1, v2, v3); kwargs...)
TriclinicBoundary(arr; kwargs...) = TriclinicBoundary(SVector{3}(arr); kwargs...)

Base.getindex(b::TriclinicBoundary, i::Integer) = b.basis_vectors[i]
Base.firstindex(b::TriclinicBoundary) = b.basis_vectors[1]
Base.lastindex(b::TriclinicBoundary) = b.basis_vectors[3]

"""
    n_dimensions(boundary)

Number of dimensions of a [`System`](@ref), [`ReplicaSystem`](@ref) or bounding box.
"""
AtomsBase.n_dimensions(::CubicBoundary) = 3
AtomsBase.n_dimensions(::RectangularBoundary) = 2
AtomsBase.n_dimensions(::TriclinicBoundary) = 3

Base.broadcastable(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths

float_type(b::Union{CubicBoundary, RectangularBoundary}) = typeof(ustrip(b[1]))
float_type(b::TriclinicBoundary{T}) where {T} = T

length_type(b::Union{CubicBoundary{T}, RectangularBoundary{T}}) where {T} = T
length_type(b::TriclinicBoundary{T, A, D}) where {T, A, D} = D

function AtomsBase.bounding_box(b::CubicBoundary)
    z = zero(b[1])
    bb = SVector{3}([
        SVector(b[1], z   , z   ),
        SVector(z   , b[2], z   ),
        SVector(z   , z   , b[3]),
    ])
    return unit(z) == NoUnits ? (bb)u"nm" : bb # Assume nm without other information
end

function AtomsBase.bounding_box(b::RectangularBoundary)
    z = zero(b[1])
    bb = SVector{2}([
        SVector(b[1], z   ),
        SVector(z   , b[2]),
    ])
    return unit(z) == NoUnits ? (bb)u"nm" : bb
end

function AtomsBase.bounding_box(b::TriclinicBoundary)
    return unit(b[1][1]) == NoUnits ? (b.basis_vectors)u"nm" : b.basis_vectors
end

has_infinite_boundary(b::Union{CubicBoundary, RectangularBoundary}) = any(isinf, b.side_lengths)
has_infinite_boundary(b::TriclinicBoundary) = false

"""
    box_volume(boundary)

Calculate the volume of a 3D bounding box or the area of a 2D bounding box.
"""
box_volume(b::Union{CubicBoundary, RectangularBoundary}) = prod(b.side_lengths)
box_volume(b::TriclinicBoundary) = b[1][1] * b[2][2] * b[3][3]

"""
    box_center(boundary)

Calculate the center of a bounding box.
Dimensions with infinite length return zero.
"""
function box_center(b::Union{CubicBoundary, RectangularBoundary})
    return map(x -> isinf(x) ? zero(x) : x / 2, b.side_lengths)
end

box_center(b::TriclinicBoundary) = sum(b.basis_vectors) / 2

# The minimum cubic box surrounding the bounding box, used for visualization
cubic_bounding_box(b::Union{CubicBoundary, RectangularBoundary}) = b.side_lengths
cubic_bounding_box(b::TriclinicBoundary) = sum(b.basis_vectors)

# Coordinates for visualizing bounding box
function bounding_box_lines(boundary::CubicBoundary, dist_unit)
    sl = ustrip.(dist_unit, boundary.side_lengths)
    z = zero(sl[1])
    p1 = SVector(z    , z    , z    )
    p2 = SVector(sl[1], z    , z    )
    p3 = SVector(z    , sl[2], z    )
    p4 = SVector(z    , z    , sl[3])
    p5 = SVector(sl[1], sl[2], z    )
    p6 = SVector(sl[1], z    , sl[3])
    p7 = SVector(z    , sl[2], sl[3])
    p8 = SVector(sl[1], sl[2], sl[3])
    seq = [p1, p4, p7, p3, p1, p2, p5, p3, p7, p8, p6, p4, p6, p2, p5, p8]
    return getindex.(seq, 1), getindex.(seq, 2), getindex.(seq, 3)
end

function bounding_box_lines(boundary::RectangularBoundary, dist_unit)
    sl = ustrip.(dist_unit, boundary.side_lengths)
    xs = [0.0, 0.0, sl[1], sl[1], 0.0]
    ys = [0.0, sl[2], sl[2], 0.0, 0.0]
    return xs, ys
end

function bounding_box_lines(boundary::TriclinicBoundary, dist_unit)
    bv = ustrip_vec.(dist_unit, boundary.basis_vectors)
    p1 = zero(bv[1])
    p2 = bv[1]
    p3 = bv[2]
    p4 = bv[3]
    p5 = bv[1] + bv[2]
    p6 = bv[1] + bv[3]
    p7 = bv[2] + bv[3]
    p8 = bv[1] + bv[2] + bv[3]
    seq = [p1, p4, p7, p3, p1, p2, p5, p3, p7, p8, p6, p4, p6, p2, p5, p8]
    return getindex.(seq, 1), getindex.(seq, 2), getindex.(seq, 3)
end

"""
    rand_coord(boundary)

Generate a random coordinate uniformly distributed within a bounding box.
"""
rand_coord(boundary::CubicBoundary      ) = rand(SVector{3, float_type(boundary)}) .* boundary
rand_coord(boundary::RectangularBoundary) = rand(SVector{2, float_type(boundary)}) .* boundary

function rand_coord(boundary::TriclinicBoundary{T}) where T
    return sum(rand(SVector{3, T}) .* boundary.basis_vectors)
end

"""
    vector_1D(c1, c2, side_length)

Displacement between two 1D coordinate values from c1 to c2, accounting for
periodic boundary conditions in a [`CubicBoundary`](@ref) or [`RectangularBoundary`](@ref).
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
periodic boundary conditions.
The minimum image convention is used, so the displacement is to the closest
version of the coordinates accounting for the periodic boundaries.
For the [`TriclinicBoundary`](@ref) an approximation is used to find the closest
version by default.
"""
vector(c1, c2, boundary::Union{CubicBoundary, RectangularBoundary}) = vector_1D.(c1, c2, boundary)

@generated function vector(c1::SVector{N}, c2::SVector{N}, boundary::Union{CubicBoundary, RectangularBoundary}) where N
    quote
        Base.Cartesian.@ncall $N SVector{$N} i -> vector_1D(c1[i], c2[i], boundary[i])
    end
end

function vector(c1, c2, boundary::TriclinicBoundary{T, true}) where T
    dr = c2 - c1
    dr -= boundary.basis_vectors[3] * floor(dr[3] * boundary.reciprocal_size[3] + T(0.5))
    dr -= boundary.basis_vectors[2] * floor(dr[2] * boundary.reciprocal_size[2] + T(0.5))
    dr -= boundary.basis_vectors[1] * floor(dr[1] * boundary.reciprocal_size[1] + T(0.5))
    return dr
end

function vector(c1, c2, boundary::TriclinicBoundary{T, false}) where T
    bv = boundary.basis_vectors
    offsets = (-1, 0, 1)
    min_sqdist = typemax(c1[1] ^ 2)
    min_dr = zero(c1)
    for ox in offsets, oy in offsets, oz in offsets
        c2_offset = c2 + ox * bv[1] + oy * bv[2] + oz * bv[3]
        dr = c2_offset - c1
        sqdist = dot(dr, dr)
        if sqdist < min_sqdist
            min_dr = dr
            min_sqdist = sqdist
        end
    end
    return min_dr
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
    bv, rs = boundary.basis_vectors, boundary.reciprocal_size
    v_wrap = v
    # Bound in z-axis
    v_wrap -= bv[3] * floor(v_wrap[3] * rs[3])
    # Bound in y-axis
    v_wrap -= bv[2] * floor((v_wrap[2] - v_wrap[3] / boundary.tan_bprojyz_cprojyz) * rs[2])
    dz_projxy = v_wrap[3] / boundary.tan_c_cprojxy
    dx = dz_projxy * boundary.cos_a_cprojxy
    dy = dz_projxy * boundary.sin_a_cprojxy
    # Bound in x-axis
    v_wrap -= bv[1] * floor((v_wrap[1] - dx - (v_wrap[2] - dy) / boundary.tan_a_b) * rs[1])
    return v_wrap
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

function velocity_3D(mass, temp, k=mb_conversion_factor, rng=Random.GLOBAL_RNG)
    return SVector(
        maxwell_boltzmann(mass, temp, k; rng=rng),
        maxwell_boltzmann(mass, temp, k; rng=rng),
        maxwell_boltzmann(mass, temp, k; rng=rng),
    )
end

function velocity_2D(mass, temp, k=mb_conversion_factor, rng=Random.GLOBAL_RNG)
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
    return velocity_3D.(masses(sys), temp, sys.k, rng)
end

function random_velocities(sys::AbstractSystem{2}, temp; rng=Random.GLOBAL_RNG)
    return velocity_2D.(masses(sys), temp, sys.k, rng)
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

Remove the center of mass motion from a [`System`](@ref).
"""
function remove_CM_motion!(sys)
    atom_masses = masses(sys)
    cm_momentum = sum_svec(Array(sys.velocities .* atom_masses))
    cm_velocity = cm_momentum / sum(Array(atom_masses))
    sys.velocities = sys.velocities .- (cm_velocity,)
    return sys
end
